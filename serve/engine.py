import time
from dataclasses import dataclass, field

import torch

from model import KVcache, KvSlab, MiniLLM

# The slab bucket is a multiple of this many columns (D4 / w2-rework B1). The old
# "start at 256, double" rounded 2048 + 128 up to 4096, so every decode step read
# ~2x the columns it needed.
SLAB_GRANULARITY = 64
# A handful of buckets covers the prompt set; keep only the most recently created ones.
MAX_CACHED_SLABS = 3


@dataclass
class EngineOutput:
    """One request's result.

    ``cache_mb`` is the slab's **reserved** bytes (``KvSlab.mb()``: the whole bucket,
    every layer, k + v), not the length the history has grown to. The bucket rounds
    up to 64 columns, so two nearby budgets can report the same reservation, and the
    value is the upper bound of the KV footprint, which is what the memory account wants.
    """

    token_ids: list[int]
    ttft_ms: float
    prefill_ms: float
    decode_ms: float
    tpot_ms: float | None
    cache_mb: float | None = None
    meta: dict = field(default_factory=dict)


class MiniLLMEngine:
    """Single-request engine on the frozen attention path (docs/decisions.md D4).

    prefill (T > 1) uses SDPA over a growing ``KVcache``; decode (T == 1) uses
    ``hand_baddbmm`` over a fixed ``KvSlab``, handed over with ``fill_from``.
    ``decode(token, cache)`` is kept only as the transitional/cat reference the slab
    path is compared against; ``generate`` runs the slab path.
    """

    def __init__(self, model, eos_token_id=None):
        self.model: MiniLLM = model.eval()
        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device
        self.vocab_size = model.config.vocab_size
        self.max_position = model.config.max_position_embeddings
        self._use_cuda_events = self.device.type == "cuda"
        self._slabs: dict[int, KvSlab] = {}

    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]) -> tuple[torch.Tensor, KVcache]:
        """返回最后一个位置的 logits 和新建的 KVcache。"""
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if len(prompt_ids) > self.max_position:
            raise ValueError("prompt longer than max_position_embeddings")
        cache = KVcache(self.model.config.n_layer)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(tokens, past_keys_values=cache, logits_to_keep=1)
        return logits[:, -1, :], cache


    @torch.inference_mode()
    def decode(self, token_id: int, cache: KVcache) -> torch.Tensor:
        """喂一个 token，返回最后一个位置的 logits（过渡版 / 测试参考路径）。"""
        tokens = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        logits, _ = self.model(tokens, past_keys_values=cache, logits_to_keep=1)
        return logits[:, -1, :]

    @torch.inference_mode()
    def decode_slab(self, tokens: list[int], slab: KvSlab) -> torch.Tensor:
        """One forward for every slab row; returns ``(B, vocab)``.

        RoPE positions come from ``slab.pos`` row by row (``MiniLLM._rope_rows``), so a
        single ``model(...)`` covers rows that are at different positions -- no per-row
        loop, no shared RoPE table (P-21).
        """
        if len(tokens) != slab.batch:
            raise ValueError(
                f"decode_slab needs one token per slab row ({slab.batch}), got {len(tokens)}"
            )
        ids = torch.tensor(list(tokens), dtype=torch.long, device=self.device).view(-1, 1)
        logits, _ = self.model(ids, past_keys_values=slab, logits_to_keep=1)
        return logits[:, -1, :]

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> EngineOutput:
        """prefill(sdpa) -> fill_from -> decode_slab loop, until EOS / budget / position limit."""
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if len(prompt_ids) > self.max_position:
            raise ValueError("prompt longer than max_position_embeddings")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        timer = "cuda_event" if self._use_cuda_events else "perf_counter"
        if max_new_tokens == 0:
            return EngineOutput(
                token_ids=[],
                ttft_ms=0.0,
                prefill_ms=0.0,
                decode_ms=0.0,
                tpot_ms=None,
                cache_mb=0.0,
                meta={
                    "impl": "minillm",
                    "prompt_tokens": len(prompt_ids),
                    "out_tokens": 0,
                    "eos_token_id": self.eos_token_id,
                    "timer": timer,
                },
            )

        self._sync()
        start = time.perf_counter()
        prefill_start = self._new_event()
        if prefill_start is not None:
            prefill_start.record()

        logits, cache = self.prefill(prompt_ids)
        # Hand-off is inside the TTFT window: the copy is part of reaching the first token.
        slab = self._slab(len(prompt_ids) + max_new_tokens)
        for layer in range(self.model.config.n_layer):
            slab.fill_from(cache, layer)

        prefill_end = self._new_event()
        if prefill_end is not None:
            prefill_end.record()

        tokens = [int(logits.argmax(dim=-1).item())]
        self._sync()
        ttft_ms = (time.perf_counter() - start) * 1000.0
        if prefill_start is not None and prefill_end is not None:
            prefill_ms = prefill_start.elapsed_time(prefill_end)
        else:
            prefill_ms = ttft_ms

        decode_start = self._new_event()
        if decode_start is not None:
            decode_start.record()
        decode_wall_start = time.perf_counter()

        # ``past_len`` in Python: reading the slab cursor would be a host sync per step, and
        # the row is the only one, so the count is exact (prompt + one write per decode call).
        filled = len(prompt_ids)
        for _ in range(1, max_new_tokens):
            if self.eos_token_id is not None and tokens[-1] == self.eos_token_id:
                break
            if filled + 1 >= self.max_position:
                break
            logits = self.decode_slab([tokens[-1]], slab)
            tokens.append(int(logits.argmax(dim=-1).item()))
            filled += 1

        decode_wall_ms = (time.perf_counter() - decode_wall_start) * 1000.0
        decode_end = self._new_event()
        if decode_end is not None:
            decode_end.record()
        self._sync()

        if decode_start is not None and decode_end is not None:
            decode_ms = decode_start.elapsed_time(decode_end)
        else:
            decode_ms = decode_wall_ms

        return EngineOutput(
            token_ids=tokens,
            ttft_ms=ttft_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tpot_ms=(decode_wall_ms / (len(tokens) - 1)) if len(tokens) > 1 else None,
            cache_mb=slab.mb(),
            meta={
                "impl": "minillm",
                "slab_mb": slab.mb(),
                "prompt_tokens": len(prompt_ids),
                "out_tokens": len(tokens),
                "eos_token_id": self.eos_token_id,
                "timer": timer,
                "decode_wall_ms": decode_wall_ms,
            },
        )

    # ---------------------------------------------------------------------- slab
    def _slab_bucket(self, need: int) -> int:
        """Round the KV budget to 64 columns, capped at ``max_position`` (B1)."""
        need = min(max(int(need), 1), self.max_position)
        bucket = -(-need // SLAB_GRANULARITY) * SLAB_GRANULARITY
        return max(min(bucket, self.max_position), need)

    @torch.inference_mode()
    def _slab(self, need: int) -> KvSlab:
        """Return a reset slab for ``need`` columns, reusing a cached bucket if possible."""
        size = self._slab_bucket(need)
        slab = self._slabs.get(size)
        if slab is None:
            if len(self._slabs) >= MAX_CACHED_SLABS:
                self._slabs.pop(next(iter(self._slabs)))  # FIFO: drop the oldest bucket
            slab = KvSlab(
                self.model.config.n_layer,
                1,
                self.model.config.n_kv_head,
                self.model.head_dim,
                size,
                dtype=self.model.token_embedding.weight.dtype,
                device=self.device,
                n_head=self.model.config.n_head,
            )
            self._slabs[size] = slab
        slab.reset()
        return slab

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _new_event(self) -> torch.cuda.Event | None:
        if not self._use_cuda_events:
            return None
        return torch.cuda.Event(enable_timing=True)
