import time
from dataclasses import dataclass, field

import torch

from model import KVcache, MiniLLM

@dataclass
class EngineOutput:
    token_ids: list[int]
    ttft_ms: float
    prefill_ms: float
    decode_ms: float
    tpot_ms: float | None
    cache_mb: float | None = None
    meta: dict = field(default_factory=dict)


class MiniLLMEngine:
    def __init__(self, model, eos_token_id=None):
        self.model: MiniLLM = model.eval()
        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device
        self.vocab_size = model.config.vocab_size
        self.max_position = model.config.max_position_embeddings
        self._use_cuda_events = self.device.type == "cuda"

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
        """喂一个 token，返回最后一个位置的 logits。"""
        tokens = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        logits, _ = self.model(tokens, past_keys_values=cache, logits_to_keep=1)
        return logits[:, -1, :]

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> EngineOutput:
        """prefill -> argmax -> 循环 decode，直到 EOS / 上限 / 位置上限。"""
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

        for _ in range(1, max_new_tokens):
            if self.eos_token_id is not None and tokens[-1] == self.eos_token_id:
                break
            if cache.past_len(0) + 1 >= self.max_position:
                break
            logits = self.decode(tokens[-1], cache)
            tokens.append(int(logits.argmax(dim=-1).item()))

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
            cache_mb=self._cache_mb(cache),
            meta={
                "impl": "minillm",
                "prompt_tokens": len(prompt_ids),
                "out_tokens": len(tokens),
                "eos_token_id": self.eos_token_id,
                "timer": timer,
                "decode_wall_ms": decode_wall_ms,
            },
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _new_event(self) -> torch.cuda.Event | None:
        if not self._use_cuda_events:
            return None
        return torch.cuda.Event(enable_timing=True)

    @staticmethod
    def _cache_mb(cache: KVcache) -> float:
        total = 0
        for kv in cache.kvs:
            if kv is None:
                continue
            key, value = kv
            total += key.numel() * key.element_size()
            total += value.numel() * value.element_size()
        return total / 2**20
