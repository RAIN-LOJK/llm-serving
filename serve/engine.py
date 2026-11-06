import time
from dataclasses import dataclass, field

import torch

from model import KVcache, KvSlab, MiniLLM, PagedKV, PagedPool
from model.attention import DECODE_BACKEND
from model.packed import PackedPrefill
from model.paged import LPAD_GRANULARITY
from serve.block_manager import BlockManager, FragStats

# The slab bucket is a multiple of this many columns (D4). The old
# "start at 256, double" rounded 2048 + 128 up to 4096, so every decode step read
# ~2x the columns it needed.
SLAB_GRANULARITY = 64


def _empty_gather_stats() -> dict:
    """Zeroed per-step gather account (see ``PagedPool.step_stats``)."""
    return {
        "gather_ms": 0.0,
        "gather_read_mb": 0.0,
        "gather_write_mb": 0.0,
        "gather_calls": 0,
    }


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


class PagedWriteSink:
    """Prefill container that writes each layer's KV straight into physical blocks (W4).

    The paged analogue of :class:`SlabWriteSink`: ``model/layers.py::Attention.forward``
    only needs ``past_len`` / ``get_kv`` / ``set_kv``, so this object is accepted there
    unchanged. It answers the first two as "no history" and turns ``set_kv`` into a
    ``PagedPool.write_prefill`` through the request's block table. Each layer's KV is
    written and dropped inside the forward, so a paged prefill never builds a persistent
    ``KVcache`` next to the pool (the same rule ``SlabWriteSink`` enforces for seats).

    It is one-shot: a second ``set_kv`` for the same layer raises instead of silently
    overwriting columns (chunked prefill would need a real ``past_len > 0`` container).
    """

    def __init__(self, pool: PagedPool, block_table, length: int, *, slots=None) -> None:
        self.pool = pool
        self.block_table = list(block_table)
        self.length = int(length)
        # One slot map for every layer of this request. Without it, ``write_prefill``
        # would rebuild the map per layer (28 rebuilds) and the packed group's sink list
        # would evict every other request's map on each layer.
        self.slots = slots
        self._written = [False] * pool.n_layer

    def past_len(self, layer: int | None = None) -> int:
        return 0

    def get_kv(self, layer: int) -> tuple[None, None]:
        return (None, None)

    def set_kv(self, layer: int, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self._written[layer]:
            raise RuntimeError(
                f"layer {layer} was already written: the paged write-through sink holds "
                "exactly one full prefill (past_len == 0)"
            )
        k, v = kv
        expected = (1, self.pool.n_kv_head, self.length, self.pool.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"set_kv needs {expected}, got {tuple(k.shape)} / {tuple(v.shape)}")
        self.pool.write_prefill_slots(
            layer,
            self.slots if self.slots is not None else self.pool.slot_ids(self.block_table, 0, self.length),
            k[0].permute(1, 0, 2),
            v[0].permute(1, 0, 2),
        )
        self._written[layer] = True


class MiniLLMEngine:
    """Single-request engine on the frozen attention path (docs/decisions.md D4).

    prefill (T > 1) uses SDPA over a growing ``KVcache``; decode (T == 1) uses
    ``hand_baddbmm`` over a fixed ``KvSlab``, handed over with ``fill_from``.
    ``decode(token, cache)`` is kept only as the transitional/cat reference the slab
    path is compared against; ``generate`` runs the slab path.
    """

    def __init__(
        self,
        model,
        eos_token_id=None,
        *,
        prefill: str = "serial",
        max_prefill_tokens: int = 8192,
        decode: str = "auto",
    ):
        self.model: MiniLLM = model.eval()
        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device
        self.vocab_size = model.config.vocab_size
        self.max_position = model.config.max_position_embeddings
        self._use_cuda_events = self.device.type == "cuda"
        if prefill not in {"serial", "packed"}:
            raise ValueError(f"prefill must be 'serial' or 'packed', got {prefill!r}")
        if int(max_prefill_tokens) < 1:
            raise ValueError(f"max_prefill_tokens must be >= 1, got {max_prefill_tokens!r}")
        if decode not in {"auto", "triton", "gather"}:
            raise ValueError(f"decode must be auto/triton/gather, got {decode!r}")
        self.prefill_mode = str(prefill)
        self.max_prefill_tokens = int(max_prefill_tokens)
        self._decode_preference = str(decode)
        self._varlen_backend = None
        self._paged_decode_impl = "gather"
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
        # W4 paged pool (docs/decisions.md D6): physical blocks + block tables, alongside the
        # seat slab and never mixed with it. ``open_paged_pool`` fills these in.
        self._paged_pool: PagedPool | None = None
        self._blocks: BlockManager | None = None
        self._paged_max_batch: int | None = None
        self._paged_max_position: int | None = None
        self._paged_frag = FragStats(1)  # real block size set by ``open_paged_pool``
        # request_id -> {block_table, prompt_len, length, seat, rows}; ``length`` counts
        # written tokens (prompt + decoded), ``rows`` is unused for now (the decode batch is
        # compressed from the live entries' seats each step).
        self._paged_state: dict[int, dict] = {}
        # The per-step ``PagedKV`` for the compressed active batch. Rebuilt when the active
        # seat set or any block table changes; rows keep their block tables, so the pool's
        # flat gather index stays cached across steps.
        self._paged_cache: PagedKV | None = None
        self._paged_cache_key: tuple | None = None
        # Per-lpad mask/bias pair: ONE bucket resident, sized to the active rows (not
        # ``max_batch``), released and rebuilt when the bucket moves or the batch grows (R4).
        self._paged_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._paged_last_lpad = 0
        # The last decode step's gather account (plan §5 / appendix C): host time and
        # shape-derived traffic, produced by ``PagedPool.step_stats`` after ``gather``
        # ran once per layer. Forwarded verbatim by ``paged_stats()``.
        self._paged_gather_stats = _empty_gather_stats()
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
        """Reserved KV of the pool that is open; the numerator ``memory_match_ratio`` uses.

        Recording only the request's own share would leave the other reservations
        unaccounted in the denominator, which is the W2-0004 failure mode (P-30). For the
        paged arm this is the whole block pool (not the blocks in use); its workspace is a
        separate figure (``paged_stats``).
        """
        if self._paged_pool is not None:
            return self._paged_pool.mb()
        return None if self._pool_slab is None else self._pool_slab.mb()

    @torch.inference_mode()
    def prefill_one(self, prompt_ids: list[int]) -> tuple[torch.Tensor, KVcache]:
        """Per-sequence prefill (D4: SDPA, no pad, no mask). Same call as ``prefill``.

        The W2 single-request path still hands the returned ``KVcache`` to a slab with
        ``fill_from``; the W3 pool path does not go through here (``prefill_into_seat``).
        """
        return self.prefill(prompt_ids)

    def _varlen(self):
        """The packed-prefill backend for this device (explicit, never a fallback)."""
        if self._varlen_backend is None:
            from model.attention.varlen import backend_for

            self._varlen_backend = backend_for(self.device)
        return self._varlen_backend

    def attn_backends(self) -> dict[str, str]:
        """Actual attention backends of the open arms, for run provenance."""
        prefill = self._varlen().name
        triton_decode = (
            self._decode_preference == "triton"
            or (self._paged_pool is not None and self._paged_decode_impl == "triton")
        )
        decode = "paged-triton" if triton_decode else DECODE_BACKEND.name
        return {"prefill": prefill, "decode": decode}

    @torch.inference_mode()
    def _packed_prefill(self, sinks, prompts: list[list[int]]) -> torch.Tensor:
        """One varlen forward over ``prompts``; returns ``(n_seqs, vocab)`` last-token logits.

        ``sinks`` are one-shot write sinks, one per prompt, each already pointing at its
        destination. The container writes each layer's K/V through them and selects each
        request's final token before ``lm_head``; the vocabulary projection is never
        materialized for the packed token axis.
        """
        if len(sinks) != len(prompts) or not prompts:
            raise ValueError("packed prefill needs one sink and one prompt per request")
        cu = [0]
        positions: list[int] = []
        tokens: list[int] = []
        for prompt in prompts:
            if not prompt:
                raise ValueError("prompt_ids must not be empty")
            cu.append(cu[-1] + len(prompt))
            positions.extend(range(len(prompt)))
            tokens.extend(prompt)
        total = len(tokens)
        if len(prompts) > 1 and total > self.max_prefill_tokens:
            raise ValueError(
                f"packed prefill group has {total} tokens, above max_prefill_tokens="
                f"{self.max_prefill_tokens}; the scheduler must split the group "
                "(a single over-budget prompt is allowed alone)"
            )
        packed = PackedPrefill(
            cu_seqlens=cu,
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            sinks=sinks,
            backend=self._varlen(),
        )
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).view(1, -1)
        logits, _ = self.model(ids, past_keys_values=packed, logits_to_keep=1)
        return logits

    @torch.inference_mode()
    def prefill_group(self, requests: list[dict]) -> torch.Tensor:
        """Prefill a FIFO group of whole prompts; returns ``(n, vocab)`` logits.

        A request is ``{"id": int, "prompt_ids": list[int], "seat": int, "budget": int}``.
        The seat arm writes each prompt's KV into its own ``KvSlab`` row; the paged arm
        reserves each prompt's blocks and writes through its block table. Both arms run
        the **same** packed varlen computation (only the sink differs), which is what
        makes the serial-vs-packed A/B a scheduling comparison rather than a kernel
        comparison. A failure rolls back every row/block in the group before re-raising;
        the caller aborts the whole group because a single forward cannot attribute the
        error to one request.
        """
        if not requests:
            raise ValueError("prefill_group needs at least one request")
        if self._paged_pool is not None:
            return self._prefill_group_paged(requests)
        return self._prefill_group_seat(requests)

    def _prefill_group_seat(self, requests: list[dict]) -> torch.Tensor:
        slab = self.slab
        rows: list[tuple[int, int]] = []
        seen: set[int] = set()
        for spec in requests:
            prompt = list(spec.get("prompt_ids") or [])
            if not prompt:
                raise ValueError("prompt_ids must not be empty")
            if len(prompt) > self.max_position:
                raise ValueError("prompt longer than max_position_embeddings")
            seat = spec.get("seat")
            if seat is None or not 0 <= int(seat) < slab.batch:
                raise IndexError(f"seat {seat!r} outside the slab's batch of {slab.batch}")
            seat = int(seat)
            if seat in seen:
                raise ValueError(f"two requests in one packed group share seat {seat}")
            if len(prompt) > slab.max_len:
                raise IndexError(
                    f"prefill length {len(prompt)} exceeds the pool's max_len {slab.max_len}"
                )
            seen.add(seat)
            rows.append((seat, len(prompt)))
        sinks = [SlabWriteSink(slab, seat, length) for seat, length in rows]
        try:
            logits = self._packed_prefill(sinks, [list(spec["prompt_ids"]) for spec in requests])
        except BaseException:
            for seat, _ in rows:
                slab.freeze(seat)
            raise
        for seat, length in rows:
            slab.activate(seat, length)
        return logits

    def _prefill_group_paged(self, requests: list[dict]) -> torch.Tensor:
        if self._paged_pool is None or self._blocks is None or self._paged_max_batch is None:
            raise RuntimeError("open_paged_pool() must be called before prefill_group()")
        pool = self._paged_pool
        blocks = self._blocks
        reserved: list[tuple[int, list[int], int, int, int]] = []
        used_seats: set[int] = set()
        try:
            for spec in requests:
                request_id = spec.get("id")
                if request_id is None:
                    request_id = -len(reserved) - 1
                request_id = int(request_id)
                if request_id in self._paged_state:
                    raise ValueError(f"request {request_id} is already in the paged pool")
                prompt = list(spec.get("prompt_ids") or [])
                if not prompt:
                    raise ValueError("prompt_ids must not be empty")
                if len(prompt) > self.max_position:
                    raise ValueError("prompt longer than max_position_embeddings")
                budget = int(spec.get("budget") or 0)
                seat_value = spec.get("seat")
                if seat_value is None:
                    occupied = {state["seat"] for state in self._paged_state.values()}
                    occupied |= used_seats
                    seat = next(
                        (row for row in range(int(self._paged_max_batch)) if row not in occupied),
                        None,
                    )
                    if seat is None:
                        raise RuntimeError("no free seat id for the paged request")
                else:
                    seat = int(seat_value)
                    if not 0 <= seat < int(self._paged_max_batch):
                        raise IndexError(f"seat {seat} outside the paged pool's max_batch")
                if seat in {state["seat"] for state in self._paged_state.values()}:
                    raise ValueError(f"seat {seat} is already occupied by a paged request")
                if seat in used_seats:
                    raise ValueError(f"two requests in one packed group share seat {seat}")
                used_seats.add(seat)
                needed = blocks.blocks_for(len(prompt), budget)
                table = blocks.reserve(needed)
                if table is None:
                    raise RuntimeError(
                        f"cannot reserve {needed} blocks for request {request_id}: "
                        f"{blocks.stats()['free']} free (can_admit should have refused it)"
                    )
                reserved.append((request_id, list(table), len(prompt), seat, budget))
            slots = [pool.slot_ids(table, 0, length) for _, table, length, _, _ in reserved]
            sinks = [
                PagedWriteSink(pool, table, length, slots=slot)
                for (_, table, length, _, _), slot in zip(reserved, slots)
            ]
            logits = self._packed_prefill(
                sinks, [list(spec["prompt_ids"]) for spec in requests]
            )
        except BaseException:
            for _, table, _, _, _ in reserved:
                blocks.free(table)
            raise
        for request_id, table, length, seat, _ in reserved:
            self._paged_state[request_id] = {
                "block_table": list(table),
                "prompt_len": length,
                "length": length,
                "seat": seat,
            }
        self._paged_cache_key = None
        return logits

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
        try:
            logits = self._packed_prefill([sink], [list(prompt_ids)])
        except BaseException:
            slab.freeze(row)
            raise
        slab.activate(row, length)
        return logits

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
        """One forward over every seat; ``(B, vocab)``. Idle seats get a placeholder token.

        When a paged pool is open and no explicit ``slab`` is passed, the call routes to the
        paged decode path instead (the two pools are alternative arms, never mixed).
        """
        if slab is None and self._paged_pool is not None:
            return self._decode_paged(tokens)
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

    # --------------------------------------------------------------- W4 paged pool
    # Physical blocks instead of one row per seat (docs/decisions.md D6). A request reserves
    # ``ceil((prompt + budget) / block_size)`` blocks at admission and gets all of them back
    # on release; there is no eviction and no mid-request allocation. Prefill writes each
    # layer's KV straight into the blocks as it computes it, so -- like the seat path -- no
    # persistent ``KVcache`` is built next to the pool. Decode compresses the active seats
    # into a ``PagedKV`` batch, gathers each layer's block tables into the pool workspace
    # (one ``hand_baddbmm`` per layer, D4) and scatters the logits back to their seat rows.
    @property
    def paged(self) -> bool:
        """True when the paged arm is the one open (the scheduler's duck-typed branch)."""
        return self._paged_pool is not None

    @torch.inference_mode()
    def open_paged_pool(
        self,
        kv_blocks: int,
        block_size: int,
        max_batch: int,
        max_position: int | None = None,
        decode: str | None = None,
    ) -> PagedPool:
        """Allocate the paged KV pool and its block manager.

        ``max_batch`` is the scheduler's concurrency bound (the decode batch is at most that
        wide) and also the gather workspace's row cap (``max_rows``). ``max_position`` is
        the run's position bound used for the RoPE table (``rope_len``); it defaults to the
        model's ``max_position_embeddings`` and is **not** ``n_blocks * block_size`` (P-34).

        ``decode`` selects the paged decode container explicitly: ``"triton"`` (CUDA
        block-table kernel), ``"gather"`` (W4's dense gather + ``hand_baddbmm`` oracle
        path), or ``None`` to resolve the engine's ``decode`` preference. ``"auto"`` means
        triton on CUDA and gather elsewhere; a CUDA-only kernel on CPU raises here rather
        than being swapped silently.
        """
        if int(max_batch) < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        if int(kv_blocks) < 1:
            raise ValueError(f"kv_blocks must be >= 1, got {kv_blocks}")
        if int(block_size) < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        preference = self._decode_preference if decode is None else str(decode)
        if preference == "auto":
            preference = "triton" if self.device.type == "cuda" else "gather"
        if preference not in {"triton", "gather"}:
            raise ValueError(f"paged decode must be 'triton', 'gather' or 'auto', got {preference!r}")
        if preference == "triton" and self.device.type != "cuda":
            raise ValueError(
                "paged decode='triton' is the CUDA block-table kernel; this engine is on "
                f"{self.device}. Select decode='gather' explicitly for CPU/MPS."
            )
        self._paged_decode_impl = preference
        max_position = int(self.max_position if max_position is None else max_position)
        self._paged_pool = PagedPool(
            self.model.config.n_layer,
            int(kv_blocks),
            int(block_size),
            self.model.config.n_kv_head,
            self.model.head_dim,
            dtype=self.model.token_embedding.weight.dtype,
            device=self.device,
            n_head=self.model.config.n_head,
            max_rows=int(max_batch),
        )
        self._blocks = BlockManager(int(kv_blocks), int(block_size))
        self._paged_max_batch = int(max_batch)
        self._paged_max_position = max_position
        self._paged_frag = FragStats(int(block_size))
        self._paged_state = {}
        self._paged_cache = None
        self._paged_cache_key = None
        self._paged_buffers = {}
        self._paged_last_lpad = 0
        self._paged_gather_stats = _empty_gather_stats()
        return self._paged_pool

    @property
    def paged_pool(self) -> PagedPool:
        if self._paged_pool is None:
            raise RuntimeError("open_paged_pool() must be called before the paged scheduler runs")
        return self._paged_pool

    @property
    def block_manager(self) -> BlockManager:
        if self._blocks is None:
            raise RuntimeError("open_paged_pool() must be called before the paged scheduler runs")
        return self._blocks

    @property
    def paged_max_batch(self) -> int | None:
        """The paged pool's row cap (the scheduler must run with the same ``max_batch``)."""
        return self._paged_max_batch

    def can_admit(self, prompt_len: int, budget: int) -> bool:
        """Whether a request of this length/budget fits **now**; never changes state.

        Two conditions, both necessary before a seat is popped (D6): the request's whole
        ``ceil((prompt + budget) / bs)`` reservation is available in the free list, and the
        active request count is below ``max_batch``. A failed check leaves the request in
        the queue (its wait is queue time), and the scheduler does not skip it (FIFO).

        With only the seat pool open there is no block reservation to make (the slab is
        allocated up front), so the answer is ``True``: the hook always describes the arm
        that is actually open.
        """
        if self._paged_pool is None or self._blocks is None:
            return True
        needed = self._blocks.blocks_for(int(prompt_len), int(budget))
        if len(self._paged_state) >= int(self._paged_max_batch):
            return False
        return needed <= self._blocks.stats()["free"]

    def can_admit_group(self, requests: list[dict]) -> bool:
        """Whether a whole packed group fits now; never changes state.

        ``can_admit`` checks one request against the *current* free list. A packed group
        reserves several requests back to back, so checking each head independently lets
        the group over-commit the pool: the first members take blocks, the last one then
        fails inside ``prefill_group`` and aborts requests that were legitimately
        admissible on their own. This hook makes the whole group the admission unit,
        matching the all-or-nothing reservation the engine performs.
        """
        if self._paged_pool is None or self._blocks is None:
            return True
        if len(self._paged_state) + len(requests) > int(self._paged_max_batch):
            return False
        needed = sum(
            self._blocks.blocks_for(len(spec.get("prompt_ids") or []), int(spec.get("budget") or 0))
            for spec in requests
        )
        return needed <= self._blocks.stats()["free"]

    @torch.inference_mode()
    def prefill_into_blocks(
        self,
        request_id: int,
        prompt_ids: list[int],
        *,
        budget: int | None = None,
        seat: int | None = None,
    ) -> torch.Tensor:
        """Prefill one prompt into freshly reserved blocks; returns ``(1, vocab)`` logits.

        ``budget`` is the request's generation budget: the reservation is the upper bound
        ``ceil((prompt + budget) / bs)`` (D6), so decode can never run out of blocks. When it
        is omitted only the prompt's blocks are reserved (a direct call that does not decode
        past the prompt); the scheduler always passes it. ``seat`` is the scheduler's seat
        number, recorded so :meth:`decode_batch` can scatter logits back by seat; when it is
        omitted the next free internal row id is used.

        The reservation is atomic (``BlockManager.reserve`` is all-or-nothing): a failure
        raises before any block is held or any byte is written. If the forward itself dies,
        every block is returned before re-raising, so ``blocks_in_use`` does not creep.
        """
        if self._paged_pool is None or self._blocks is None:
            raise RuntimeError("open_paged_pool() must be called before prefill_into_blocks()")
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if len(prompt_ids) > self.max_position:
            raise ValueError("prompt longer than max_position_embeddings")
        if request_id in self._paged_state:
            raise ValueError(f"request {request_id} is already in the paged pool")
        length = len(prompt_ids)
        needed = self._blocks.blocks_for(length, int(0 if budget is None else budget))
        block_table = self._blocks.reserve(needed)
        if block_table is None:
            raise RuntimeError(
                f"cannot reserve {needed} blocks for request {request_id}: "
                f"{self._blocks.stats()['free']} free (can_admit should have refused it)"
            )
        if seat is None:
            used = {state["seat"] for state in self._paged_state.values()}
            seat = next((row for row in range(self._paged_max_batch) if row not in used), None)
            if seat is None:
                self._blocks.free(block_table)
                raise RuntimeError("no free seat id for the paged request")
        seat = int(seat)
        if not 0 <= seat < int(self._paged_max_batch):
            self._blocks.free(block_table)
            raise IndexError(f"seat {seat} outside the paged pool's max_batch")

        sink = PagedWriteSink(
            self._paged_pool,
            block_table,
            length,
            slots=self._paged_pool.slot_ids(block_table, 0, length),
        )
        try:
            logits = self._packed_prefill([sink], [list(prompt_ids)])
        except BaseException:
            self._blocks.free(block_table)
            raise
        self._paged_state[int(request_id)] = {
            "block_table": list(block_table),
            "prompt_len": length,
            "length": length,
            "seat": seat,
        }
        self._paged_cache_key = None
        return logits

    def release_request(self, request_id: int) -> None:
        """Return every block of ``request_id`` to the free list; idempotent.

        Unlike ``freeze_seat`` (which parks a row and keeps its reservation, P-26), the
        paged path really frees: the blocks leave the ledger and the next request may reuse
        them immediately (the zero-block padding rule is what makes reuse correct, P-33).
        Releasing an unknown or already-released id is a no-op.
        """
        if self._blocks is None:
            return
        state = self._paged_state.pop(int(request_id), None)
        if state is None:
            return
        self._blocks.free(state["block_table"])
        self._paged_cache_key = None

    @torch.inference_mode()
    def _decode_paged(self, tokens: list[int]) -> torch.Tensor:
        """One compressed forward over the active paged rows; ``(max_batch, vocab)``.

        ``tokens`` is indexed by seat (length ``max_batch``), exactly like the seat path.
        Only requests registered in the paged pool are rows of the ``PagedKV`` batch; seats
        with no request are not forwarded and their output rows are zeros (the scheduler
        discards them). With nothing active this does **not** forward: it returns a zero
        ``(max_batch, vocab)`` tensor. The scheduler never calls it in that state.

        One ``hand_baddbmm`` runs per layer (D4, via the container's ``write``/``get``);
        the batch is never expanded to one forward per row. Statistics for the step (row
        count, ``lpad``, block counts, ``FragStats.observe``) are recorded here, from the
        step's own wall clock.
        """
        if self._paged_pool is None or self._blocks is None or self._paged_max_batch is None:
            raise RuntimeError("open_paged_pool() must be called before decode_batch()")
        max_batch = int(self._paged_max_batch)
        vocab = int(self.vocab_size)
        if not self._paged_state:
            self._paged_last_lpad = 0
            self._paged_gather_stats = _empty_gather_stats()
            self.last_decode_event_ms = None
            return torch.zeros(
                (max_batch, vocab),
                dtype=self.model.token_embedding.weight.dtype,
                device=self.device,
            )
        if len(tokens) != max_batch:
            raise ValueError(
                f"decode_batch needs one token per seat (max_batch={max_batch}), got {len(tokens)}"
            )

        active = sorted(self._paged_state.items(), key=lambda item: item[1]["seat"])
        seats = [state["seat"] for _, state in active]
        if len(set(seats)) != len(seats):
            raise RuntimeError(f"two paged requests share a seat: {seats}")
        cache = self._sync_paged_cache(active)

        ids = torch.tensor(
            [int(tokens[seat]) for seat in seats], dtype=torch.long, device=self.device
        ).view(-1, 1)
        # Step boundary for the gather account: every layer's ``PagedKV.get`` below adds
        # one call, its host time and its shape-derived k+v traffic. In triton mode no
        # gather runs, so the account is zero by construction.
        self._paged_pool.begin_step()
        start = time.perf_counter()
        event = self._new_event()
        if event is not None:
            event.record()
        logits, _ = self.model(ids, past_keys_values=cache, logits_to_keep=1)
        logits = logits[:, -1, :]
        if event is not None:
            end = self._new_event()
            end.record()
            self._sync()
            self.last_decode_event_ms = float(event.elapsed_time(end))
        else:
            self.last_decode_event_ms = None
            self._sync()
        dt_s = time.perf_counter() - start
        # Produced by the pool's own accumulator, never recomputed here: the same numbers
        # the step's gather calls just measured (plan §5, appendix C). The triton path
        # never calls gather, so its traffic is exactly zero.
        self._paged_gather_stats = (
            _empty_gather_stats() if self._paged_decode_impl == "triton"
            else self._paged_pool.step_stats()
        )

        for _, state in active:
            state["length"] += 1

        out = logits.new_zeros((max_batch, vocab))
        out[torch.tensor(seats, dtype=torch.long, device=self.device)] = logits

        self._paged_last_lpad = int(cache.lpad or 0)
        written = sum(state["length"] for _, state in active)
        # C2's internal frag charges only blocks that hold bytes: a reserved-but-unwritten
        # block is reservation slack, not internal fragmentation (D6 keeps them apart).
        self._paged_frag.observe(dt_s, written, self._written_blocks(active))
        return out

    def _sync_paged_cache(self, active: list[tuple[int, dict]]) -> PagedKV:
        """Return the decode container for this active set, rebuilding only on change.

        The container's rows are the active requests in seat order. Its struct (block
        tables, seats) is fixed at construction; a request entering or leaving invalidates
        it, while a plain decode step reuses it and only advances its cursors. The per-lpad
        mask/bias buffers are owned here, one pair for the current bucket sized to the
        active rows, so a rebuild reuses the same addresses instead of allocating
        ``(B*H_kv, G, lpad)`` per admission (R4) -- and buckets never accumulate.
        """
        key = tuple(
            (request_id, state["seat"], tuple(state["block_table"])) for request_id, state in active
        )
        if key == self._paged_cache_key and self._paged_cache is not None:
            cache = self._paged_cache
        else:
            cache = PagedKV(
                self._paged_pool,
                [state["block_table"] for _, state in active],
                [state["length"] for _, state in active],
                rope_len=int(self._paged_max_position),
                seats=[state["seat"] for _, state in active],
                n_head=self.model.config.n_head,
                decode_impl=self._paged_decode_impl,
            )
            self._paged_cache = cache
            self._paged_cache_key = key

        if self._paged_decode_impl == "triton":
            # No mask/bias and no gather workspace: the container owns the kernel and
            # reads the pool directly. ``build_mask`` still runs once inside the model
            # forward, but it only refreshes the int32 seqlen vector.
            return cache
        batch = int(cache.batch)
        lpad = self._paged_lpad([state["length"] for _, state in active])
        mask, bias = self._paged_mask_bias(lpad, batch)
        cache.set_buffers(lpad, mask, bias)
        return cache

    def _paged_lpad(self, lengths: list[int]) -> int:
        """This step's bucket, computed exactly like ``PagedKV.build_mask`` (one step ahead)."""
        length = max(int(value) + 1 for value in lengths)
        return max(1, -(-length // int(LPAD_GRANULARITY))) * int(LPAD_GRANULARITY)

    def _paged_mask_bias(self, lpad: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask/bias views for a step, backed by ONE buffer pair for the current ``lpad``.

        The pair is sized to the active row count exactly and rebuilt only when the bucket
        moves or the batch outgrows it. Earlier buckets are released *before* the new
        allocation, so 26 lpad buckets over a run cost one pair, not 26 -- which is what
        keeps the workspace account honest (plan appendix C).
        """
        batch = int(batch)
        if batch < 1:
            raise ValueError("mask/bias need at least one active row")
        entry = self._paged_buffers.get(lpad)
        if entry is not None and entry[0].shape[0] < batch:
            entry = None
        if entry is None:
            self._paged_buffers.clear()  # release first: peak = one bucket
            rows = min(batch, int(self._paged_max_batch or batch))
            head_dim = self.model.config.n_head // self.model.config.n_kv_head
            entry = (
                torch.zeros(
                    (rows, 1, 1, lpad),
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.zeros(
                    (rows * self.model.config.n_kv_head, head_dim, lpad),
                    dtype=self.model.token_embedding.weight.dtype,
                    device=self.device,
                ),
            )
            self._paged_buffers[lpad] = entry
        mask, bias = entry
        return mask[:batch], bias[: batch * self.model.config.n_kv_head]

    def _written_blocks(self, active: list[tuple[int, dict]]) -> int:
        """Blocks holding at least one written token (the last partial block counts)."""
        block_size = self._paged_pool.block_size
        total = 0
        for _, state in active:
            length = int(state["length"])
            total += (length + block_size - 1) // block_size
        return total

    def _paged_buffer_mb(self) -> float:
        """Bytes of the engine-owned mask/bias pair for the current lpad bucket.

        These are the ``(rows, 1, 1, lpad)`` bool mask and the ``(rows*H_kv, G, lpad)`` bias
        tensor for the step's bucket, sized to the active batch (not ``max_batch``), with the
        previous bucket released before the next is allocated. The pool's gather buffers are
        counted separately by ``PagedPool.workspace_mb``. Both are workspace, not KV.
        """
        total = sum(
            tensor.numel() * tensor.element_size()
            for pair in self._paged_buffers.values()
            for tensor in pair
        )
        return total / 2**20

    def paged_stats(self) -> dict:
        """Code-produced memory/occupancy account for the paged arm (P-35).

        ``kv_mb`` is the pool's whole reservation, ``kv_live_mb`` the bytes of written
        tokens, ``kv_alloc_mb`` the capacity of the blocks currently reserved; the three are
        different numbers and all are reported. ``frag_internal`` is the time-weighted C2
        metric; ``reservation_slack`` is the instantaneous upper-bound waste (reported, not
        judged). ``workspace_mb`` is every gather/mask buffer the paged arm allocates on top
        of KV (a cost the seat arm does not pay); in triton mode there is no gather and no
        mask/bias pair, so it is zero. ``gather_ms`` / ``gather_read_mb`` /
        ``gather_write_mb`` / ``gather_calls`` are the **last decode step's** gather
        account: host time summed over the step's per-layer ``gather`` calls and the
        shape-derived k+v traffic (plan §5 / appendix C). Traffic, not occupancy; the
        triton path never gathers, so its traffic is exactly zero.
        """
        if self._paged_pool is None or self._blocks is None:
            raise RuntimeError("open_paged_pool() must be called before paged_stats()")
        block_size = self._paged_pool.block_size
        pool = self._paged_pool
        triton = self._paged_decode_impl == "triton"
        per_token_bytes = 2 * pool.n_layer * pool.n_kv_head * pool.head_dim * pool.dtype.itemsize
        active = sorted(self._paged_state.items(), key=lambda item: item[1]["seat"])
        written_tokens = sum(state["length"] for _, state in active)
        in_use = self._blocks.in_use
        reserved_tokens = in_use * block_size
        stats = self._blocks.stats()
        result = {
            "kv_mb": pool.mb(),
            "kv_live_mb": written_tokens * per_token_bytes / 2**20,
            "kv_alloc_mb": reserved_tokens * per_token_bytes / 2**20,
            "workspace_mb": 0.0 if triton else pool.workspace_mb() + self._paged_buffer_mb(),
            "gather_workspace_mb": 0.0 if triton else pool.workspace_mb(),
            "active_rows": len(active),
            "blocks_in_use": in_use,
            "free_blocks": stats["free"],
            "written_blocks": self._written_blocks(active),
            "frag_internal": self._paged_frag.internal_frag(),
            "reservation_slack": self._paged_frag.reservation_slack(
                reserved_tokens, written_tokens
            ),
            "lpad": self._paged_last_lpad,
            "decode_impl": self._paged_decode_impl,
        }
        # Appended, never replacing an existing key: the step row's gather account.
        result.update(self._paged_gather_stats)
        return result

    def close_pool(self) -> None:
        self._pool_slab = None
        self._paged_pool = None
        self._blocks = None
        self._paged_state = {}
        self._paged_cache = None
        self._paged_cache_key = None
        self._paged_buffers = {}
        self._paged_last_lpad = 0
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
