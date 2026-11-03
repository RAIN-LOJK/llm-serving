import time
from dataclasses import dataclass, field

import torch

from model import KVcache, KvSlab, MiniLLM

# The slab bucket is a multiple of this many columns (D4). The old
# "start at 256, double" rounded 2048 + 128 up to 4096, so every decode step read
# ~2x the columns it needed.
SLAB_GRANULARITY = 64


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


class SlabWriteSink:
    """Prefill container that writes each layer's KV straight into one pool row (W3).

    ``model/layers.py::Attention.forward`` only needs ``past_len`` / ``get_kv`` / ``set_kv``
    from a cache, so this object is accepted there unchanged: it answers the first two as
    "no history" and turns ``set_kv`` into a slice write into ``slab``'s row. Because it never
    gives KV back, the layer's ``torch.cat`` is skipped, and a pool admission never builds a
    second copy of the prompt's KV next to the already-allocated pool row (E0b: 166 MiB of
    prefill ``KVcache`` on top of the 182 MiB pool at ctx 1517).

    It is a **one-shot, ``past_len == 0``** container: ``set_kv`` writes columns ``0..L-1``
    of the row. Chunked prefill (a second forward over the same sink, i.e. a real
    ``past_len > 0``) would silently overwrite those columns, so writing a layer twice
    raises instead of appending.
    """

    def __init__(self, slab: KvSlab, row: int, length: int) -> None:
        self.slab = slab
        self.row = int(row)
        self.length = int(length)
        self._written = [False] * slab.n_layer

    def past_len(self, layer: int | None = None) -> int:
        """Always 0: a full prefill has no history, so SDPA stays ``is_causal=True``."""
        return 0

    def get_kv(self, layer: int) -> tuple[None, None]:
        """Always empty: the layer must not ``cat`` a history that lives in the pool."""
        return (None, None)

    def set_kv(self, layer: int, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Write one layer's ``(k, v)`` into ``slab`` row ``row``, columns ``0..length-1``."""
        if self._written[layer]:
            raise RuntimeError(
                f"layer {layer} was already written: the write-through sink holds exactly one "
                "full prefill (past_len == 0); chunked prefill (past_len > 0) must not reuse it"
            )
        k, v = kv
        expected = (1, self.slab.n_kv_head, self.length, self.slab.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"set_kv needs {expected}, got {tuple(k.shape)} / {tuple(v.shape)}")
        self.slab.ensure_layer(layer)  # no-op on the pool, which is materialised eagerly
        self.slab.keys[layer][self.row, :, : self.length] = k[0, :, : self.length].to(self.slab.dtype)
        self.slab.values[layer][self.row, :, : self.length] = v[0, :, : self.length].to(self.slab.dtype)
        self._written[layer] = True


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
        # Exactly one slab, reallocated when the bucket changes (W2-0005). A bucket cache
        # was measured to be pure cost under this protocol: prompts are grouped, so a new
        # bucket is built once and `warmup=3` already keeps that build out of the timed
        # region, while every retained neighbour stayed in `max_memory_allocated()` as
        # memory the formula account does not know about (W2-0004: ctx 987 ratio 0.737
        # with 3 resident buckets vs 0.870 with one). Keeping one slab makes the
        # allocator peak describe the request being measured, which is what
        # `memory_match_ratio` divides by.
        self._current_slab: KvSlab | None = None
        # W3 pool (docs/decisions.md D5): one slab with ``max_batch`` rows, one seat each,
        # for the lifetime of the scheduler. Separate from ``_current_slab`` so the
        # single-request ``generate`` path keeps the W2 slab semantics bit for bit (E0a).
        self._pool_slab: KvSlab | None = None
        self.last_decode_event_ms: float | None = None

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
        #
        # Interleaved, one layer at a time: allocate layer l, copy it, drop the source layer.
        # ``fill_from`` copies, so the prefill ``KVcache`` is dead the moment its layer has
        # landed -- but the peak is recorded the instant the destinations exist, so
        # allocating all 28 first and copying afterwards puts the same KV in memory twice
        # (348 MB at ctx 1517) at the worst possible moment. Interleaving leaves one layer
        # of overlap instead (~6 MB) and pulls the request peak back down to the prefill
        # phase's own T^2 workspace (W2-0006 notes). It only helps because the slab is also
        # released at the end of each request: a retained slab is already fully materialised
        # when the next prefill starts, and the interleaving never gets a chance to matter.
        slab = self._slab(len(prompt_ids) + max_new_tokens)
        for layer in range(self.model.config.n_layer):
            slab.ensure_layer(layer)
            slab.fill_from(cache, layer)
            cache.kvs[layer] = None
        del cache

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

        # Release the slab before returning. Like the bucket cache this replaced, a slab
        # that outlives its request is memory the *next* request's peak holds but its
        # account cannot explain, and `memory_match_ratio` divides exactly those two.
        # Measured at ctx 1517: retaining it 1532 MB (ratio 0.861), releasing it 1348 MB
        # (ratio 0.978), for ~0.5 ms of extra TTFT spent rebuilding. W3 will keep one slab
        # per concurrent slot and account for it explicitly, which is a different contract.
        slab_mb = slab.mb()
        self._current_slab = None

        return EngineOutput(
            token_ids=tokens,
            ttft_ms=ttft_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tpot_ms=(decode_wall_ms / (len(tokens) - 1)) if len(tokens) > 1 else None,
            cache_mb=slab_mb,
            meta={
                "impl": "minillm",
                "slab_mb": slab_mb,
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
        """Return a reset slab for ``need`` columns, reallocating only on a bucket change.

        One slab, not a cache: see the note in ``__init__``. It is built **lazily** -- layers
        materialise in ``KvSlab.ensure_layer`` as the hand-off reaches them -- so the copy
        loop, not this call, decides when the destination memory exists. The reallocation
        happens synchronously here and is counted in TTFT, which is where the hand-off copy
        already lives (D3), so a new bucket's cost is visible rather than hidden in a
        cached neighbour.
        """
        size = self._slab_bucket(need)
        if self._current_slab is None or self._current_slab.max_len != size:
            self._current_slab = KvSlab(
                self.model.config.n_layer,
                1,
                self.model.config.n_kv_head,
                self.model.head_dim,
                size,
                dtype=self.model.token_embedding.weight.dtype,
                device=self.device,
                n_head=self.model.config.n_head,
                lazy=True,
            )
        self._current_slab.reset()
        return self._current_slab

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    # ------------------------------------------------------------------ W3 batch pool
    # One slab with ``max_batch`` rows and one seat per row, alive for the whole trace
    # (D5). An admission prefills **into** its row (``prefill_into_seat``): the prompt's KV
    # never exists anywhere else, so there is no hand-off copy to account for. Retirement
    # freezes the row instead of freeing it, because a row's cursor is a device tensor and
    # moving KV out of the way would be a copy the protocol does not account for (P-26).
    # Every forward covers all ``max_batch`` rows -- idle ones included, masked out -- so the
    # tensor shapes stay fixed (D4's fused-mask decode is what makes that affordable).
    @torch.inference_mode()
    def open_pool(self, max_batch: int, max_len: int) -> KvSlab:
        """Allocate the pool slab: ``max_batch`` seats, ``max_len`` columns each."""
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        size = self._slab_bucket(max_len)
        if self._pool_slab is not None and (
            self._pool_slab.batch != max_batch or self._pool_slab.max_len != size
        ):
            self._pool_slab = None
            self._release_memory()
        if self._pool_slab is None:
            self._pool_slab = KvSlab(
                self.model.config.n_layer,
                max_batch,
                self.model.config.n_kv_head,
                self.model.head_dim,
                size,
                dtype=self.model.token_embedding.weight.dtype,
                device=self.device,
                n_head=self.model.config.n_head,
            )
        self._pool_slab.reset()
        return self._pool_slab

    @property
    def slab(self) -> KvSlab:
        if self._pool_slab is None:
            raise RuntimeError("open_pool() must be called before the scheduler runs")
        return self._pool_slab

    @property
    def pool_kv_mb(self) -> float | None:
        """Reserved KV of **every** seat: the numerator ``memory_match_ratio`` must use.

        Recording only the request's own share would leave the other seats' reservations
        unaccounted in the denominator, which is the W2-0004 failure mode (P-30).
        """
        return None if self._pool_slab is None else self._pool_slab.mb()

    @torch.inference_mode()
    def prefill_one(self, prompt_ids: list[int]) -> tuple[torch.Tensor, KVcache]:
        """Per-sequence prefill (D4: SDPA, no pad, no mask). Same call as ``prefill``.

        The W2 single-request path still hands the returned ``KVcache`` to a slab with
        ``fill_from``; the W3 pool path does not go through here (``prefill_into_seat``).
        """
        return self.prefill(prompt_ids)

    @torch.inference_mode()
    def prefill_into_seat(self, row: int, prompt_ids: list[int]) -> torch.Tensor:
        """Prefill one prompt straight into pool row ``row``; returns ``(1, vocab)`` logits.

        This is W3's admission: the prompt's K/V is written into the seat's row as each layer
        computes it (``SlabWriteSink``), so no ``KVcache`` is built and no ``fill_from`` copy
        happens -- the pool row is the only copy of this request's prefill KV. The forward
        call itself is unchanged (D4: one per-sequence SDPA prefill, ``past_len == 0``, so
        ``is_causal=True`` and no mask).

        The row is activated only after the whole forward succeeded. A failure can leave a
        **half-written** row (every layer before the failing one already landed its columns),
        so the row is frozen on the way out: cursors back to 0, seat masked, and the next
        admission overwrites the bytes from column 0.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if len(prompt_ids) > self.max_position:
            raise ValueError("prompt longer than max_position_embeddings")
        slab = self.slab
        if not 0 <= int(row) < slab.batch:
            raise IndexError(f"row {row} outside the slab's batch of {slab.batch}")
        length = len(prompt_ids)
        if length > slab.max_len:
            raise IndexError(
                f"prefill length {length} exceeds the pool's max_len {slab.max_len}"
            )

        sink = SlabWriteSink(slab, row, length)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        try:
            logits, _ = self.model(tokens, past_keys_values=sink, logits_to_keep=1)
        except BaseException:
            slab.freeze(row)
            raise
        slab.activate(row, length)
        return logits[:, -1, :]

    @torch.inference_mode()
    def activate_seat(self, slot: int, cache: KVcache, prompt_len: int) -> None:
        """Copy one request's prefill KV into its seat and unfreeze the row's cursor.

        Kept for the W2-style hand-off (``prefill_one`` + ``fill_from``) that the equivalence
        tests and the single-request path compare against; the W3 scheduler admits through
        ``prefill_into_seat`` instead, which needs neither this copy nor a ``KVcache``.
        """
        slab = self.slab
        # Interleaved layer by layer, like the single-request hand-off (W2-0007): allocate
        # layer l, copy it, drop the source layer. Here it bounds the overlap to one layer
        # for this row while the other seats stay resident.
        for layer in range(self.model.config.n_layer):
            slab.ensure_layer(layer)
            slab.fill_from(cache, layer, row=slot)
            cache.kvs[layer] = None
        slab.activate(slot, int(prompt_len))

    @torch.inference_mode()
    def freeze_seat(self, slot: int) -> None:
        """Retire a seat: cursor frozen, row masked out, reservation kept (P-26).

        ``inference_mode`` is load-bearing: the slab's tensors were allocated inside one,
        and PyTorch refuses in-place writes to an inference tensor from outside it.
        """
        self.slab.freeze(slot)

    @torch.inference_mode()
    def decode_batch(self, tokens: list[int], slab: KvSlab | None = None) -> torch.Tensor:
        """One forward over every seat; ``(B, vocab)``. Idle seats get a placeholder token."""
        slab = self.slab if slab is None else slab
        event = self._new_event()
        if event is not None:
            event.record()
        logits = self.decode_slab(list(tokens), slab)
        if event is not None:
            end = self._new_event()
            end.record()
            self._sync()
            self.last_decode_event_ms = float(event.elapsed_time(end))
        else:
            self.last_decode_event_ms = None
        return logits

    def close_pool(self) -> None:
        self._pool_slab = None
        self._release_memory()

    def _release_memory(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _new_event(self) -> torch.cuda.Event | None:
        if not self._use_cuda_events:
            return None
        return torch.cuda.Event(enable_timing=True)
