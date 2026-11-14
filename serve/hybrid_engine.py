"""Hybrid (Qwen3.5) serving engine: pooled linear state + 6-layer full-attention KV.

W7 Step 2's engine half. It implements the ``SchedulerEngine`` protocol
(``serve/engine.py``) for the 24-layer ``LLLF x 6`` Qwen3.5 decoder without touching the
model's math: the 18 linear-attention layers read/write :class:`LinearStatePool` slots and
the 6 full-attention layers write into either

* a :class:`~model.paged.PagedPool` (production, CUDA/Triton block-table decode), or
* a :class:`~model.slab.KvSlab` (W3-style seat baseline, one row per scheduler seat).

Both arms share one :class:`~serve.state_pool.LinearStatePool`; D9-5 keeps the KV pool at
**6 layers** (the compact full-attention indices 0..5), never 24.

Step scope
----------
``prefill_group`` runs **one packed varlen forward per group** (W7 Step 3, D9-6): the
group's prompts are concatenated request-major into one token axis, the 6 full-attention
layers attend with ``cu_seqlens`` (FlashAttention on CUDA, per-segment SDPA on CPU through
``model.attention.varlen.backend_for``) and the 18 GDN layers loop over the segments, each
request starting from empty state. The one-shot write sinks are the same objects
``serve/engine.py`` uses for the 0.6B arm: a request's prompt KV is written into its block
table / seat row as each layer computes it, so no persistent ``KVcache`` is built next to
the pool (the states built here carry ``kv=None``; packed prefill never reads ``state.kv``).

The sinks are indexed by the **compact** full-attention layer (0..5, D9-5), because
``Qwen35Attention``'s packed branch calls ``packed.write(self.layer_index, ...)`` -- the
same index the decode containers use. ``_LayerMapSink`` (which maps the actual 24-layer
index to the compact one) is therefore *not* part of the packed path; it stays as the
bridge for the single-sequence ``Qwen35Model.forward`` reference/control path, which is
what the tests use to compare pool state against a dense prefill.

Decode
------
``decode_batch(tokens, slab=None)`` is indexed by scheduler seat, like
``MiniLLMEngine.decode_batch``. The engine compresses the active handles into the pool's
packed ``[0, B)`` slot order (D9-2), builds one decode container for exactly those rows,
runs ``Qwen35Model.forward_decode_batch`` once, and scatters the ``(B, vocab)`` result back
to ``(max_batch, vocab)`` by seat.

The seat arm needs a compact row adapter (:class:`_SlabRows`) because the slab has one row
per **seat**, not per active batch row. It writes only the batch's seats (advanced-index
write), reads only those rows, narrows K/V and the additive bias to the live width
(``max(valid) + 1`` -- masked columns are identical to a full-width read) and advances only
the selected rows. Using ``KvSlab.advance()`` there would move seats outside this batch.
The narrowed read is a copy for a non-prefix seat set; the seat arm is baseline-only
(docs/w7-plan.md §2.7), so that is reported, not optimised away.

Atomicity
---------
Admission is atomic and pure: ``can_admit_group`` changes nothing. ``prefill_group``
validates every spec, reserves every linear slot and paged block table, builds one
``PackedPrefill`` for the whole group and runs the **single** ``forward_packed``; on any
error it rolls the whole group back (free blocks, freeze touched seats, free slots) before
re-raising. No half-active handle survives, and the linear pool's release clears the
"written" flags of every slot the failed group touched, so the partially written rows can
never be read as a live request. ``release_request`` is idempotent at the engine level
because the scheduler may call it after a failed group already rolled the member back
(§3.3's pool-level "unknown id errors" stays a ``LinearStatePool`` property; the engine is
the scheduler-facing, idempotent boundary; documented on the method).

Like ``MiniLLMEngine``, the pools are allocated **outside** ``torch.inference_mode``:
``LinearStatePool.free`` copies a swapped-in row with ``copy_`` and the scheduler calls
``release_request`` outside a forward. Allocating under inference mode would turn the pool
into inference tensors and make that copy raise.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from model.packed import PackedPrefill
from model.paged import PagedKV, PagedPool
from model.slab import KvSlab
from serve.block_manager import BlockManager, FragStats
from serve.engine import PagedWriteSink, SlabWriteSink
from serve.state_pool import LinearStatePool

__all__ = ["Qwen35Engine"]

ARMS = ("paged", "seat")


class _LayerMapSink:
    """Prefill KV sink addressed by the model's **actual** 24-layer index.

    ``Qwen35Model.forward`` hands each full-attention block its position in the 24-layer
    stack (``3, 7, ..., 23``), while a W7 KV container has only the 6 full-attention layers
    (D9-5). This adapter turns ``set_kv(actual, kv)`` into ``inner.set_kv(compact, kv)``;
    it answers ``past_len``/``get_kv`` as "empty" so the layer takes the ``past_len == 0``
    prefill path and lets the one-shot sink raise if a layer is written twice.

    Reference/control only: production prefill is the packed ``forward_packed`` path
    (Step 3), whose sinks are addressed compactly by ``Qwen35Attention`` itself, so it
    passes ``PagedWriteSink``/``SlabWriteSink`` directly. This adapter is what a test
    needs to run a dense single-sequence forward into the same pool row (the control the
    packed group is checked against). Feeding it to ``PackedPrefill`` would be wrong: the
    packed layer passes the compact index ``0..5``, and compact index 3 is a key of the
    actual-index map.
    """

    def __init__(self, inner, actual_layers: Sequence[int]) -> None:
        actual = [int(layer) for layer in actual_layers]
        mapping = {actual_idx: compact for compact, actual_idx in enumerate(actual)}
        if len(mapping) != len(actual):
            raise ValueError(f"full-attention layer indices must be distinct: {actual}")
        self.inner = inner
        self.mapping = mapping
        self.n_layer = len(mapping)

    def past_len(self, layer: int | None = None) -> int:
        return 0

    def get_kv(self, layer: int):
        return (None, None)

    def set_kv(self, layer: int, kv) -> None:
        try:
            compact = self.mapping[int(layer)]
        except KeyError:
            raise IndexError(
                f"layer {layer} is not a full-attention layer; expected one of "
                f"{sorted(self.mapping)}"
            ) from None
        self.inner.set_kv(compact, kv)


class _SlabRows:
    """Compact row view over one ``KvSlab`` whose rows are scheduler seats.

    The slab is ``max_batch`` seats wide; a decode batch is ``B = active_count`` rows whose
    seat numbers ``seats[i]`` are arbitrary. This adapter is the decode container
    ``Qwen35Model.forward_decode_batch`` / ``Qwen35Attention._attend_decode`` consume:

    * ``build_mask(next_token=True)`` is the slab's own step boundary (one mask/bias build
      for all rows; idle rows stay closed because the scheduler froze them);
    * ``write`` scatters this step's token into the **selected** seat rows at their own
      positions -- one advanced-index write per layer; seats outside the batch are never
      touched or advanced;
    * ``get``/``bias`` return the selected rows narrowed to ``max(valid) + 1`` columns.
      Columns past each row's limit are ``-inf`` in the bias, so the narrower sum is the
      same arithmetic as the full-width read (and matches the dense reference's width);
    * ``pos`` is the selected rows' positions for per-row RoPE, and ``advance`` moves only
      those rows (device cursors + ``KvSlab``'s Python mirrors).

    The seat set is baseline-only (docs/w7-plan.md §2.7): a non-prefix seat set costs one
    gather per layer for ``get``. Prefix sets (``seats == 0..B-1``) are read as views.
    """

    decode_container = True
    block_table_decode = False

    def __init__(self, slab: KvSlab, seats: Sequence[int]) -> None:
        if not isinstance(slab, KvSlab):
            raise TypeError(f"_SlabRows needs a KvSlab, got {type(slab).__name__}")
        rows = [int(seat) for seat in seats]
        if not rows:
            raise ValueError("_SlabRows needs at least one seat")
        if len(set(rows)) != len(rows):
            raise ValueError(f"duplicate seats in one decode batch: {rows}")
        for row in rows:
            if not 0 <= row < slab.batch:
                raise IndexError(f"seat {row} outside the slab's batch of {slab.batch}")
            if not slab._host_active[row]:
                raise RuntimeError(
                    f"seat {row} is frozen; a decode batch may only contain live requests"
                )

        self.slab = slab
        self.seats = rows
        self.batch = len(rows)
        self.n_layer = slab.n_layer
        self.n_kv_head = slab.n_kv_head
        self.n_head = slab.n_head
        self.head_dim = slab.head_dim
        self.dtype = slab.dtype
        self.device = slab.device
        self.max_len = slab.max_len

        self._row_index = torch.tensor(rows, dtype=torch.long, device=slab.device)
        self._prefix = rows == list(range(len(rows)))
        # Positions are constant for the whole forward (``advance`` is the last step), so
        # capture them once: the write and the RoPE gather both need them per layer.
        positions = [slab._host_pos[row] for row in rows]
        if max(positions) >= slab.max_len:
            raise IndexError(
                f"decode position {max(positions)} >= slab max_len {slab.max_len}"
            )
        self._columns = torch.tensor(positions, dtype=torch.long, device=slab.device)
        # One usable column per row is always needed: this step writes at ``pos`` and the
        # mask (next_token=True) allows it.
        valid = [slab._host_valid[row] for row in rows]
        self.width = min(max(valid) + 1, slab.max_len)
        if self.width < 1:  # pragma: no cover - a live request always has a prompt
            raise RuntimeError("a decode batch needs at least one visible column")

        offsets = torch.arange(slab.n_kv_head, dtype=torch.long, device=slab.device)
        self._bias_index = (
            self._row_index.view(-1, 1) * slab.n_kv_head + offsets.view(1, -1)
        ).reshape(-1)

    # ------------------------------------------------------------ container API
    def _check_layer(self, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer must be an int, got {layer!r}")
        if not 0 <= layer < self.n_layer:
            raise IndexError(f"layer {layer} outside the container's {self.n_layer} layers")
        return layer

    def build_mask(self, *, next_token: bool = False):
        """Refresh the slab's shared mask/bias in place (one step boundary)."""
        return self.slab.build_mask(next_token=next_token)

    @property
    def pos(self) -> torch.Tensor:
        """This batch's positions, in compact row order (per-row RoPE source)."""
        return self._columns

    @property
    def valid(self) -> torch.Tensor:
        """This batch's filled lengths (diagnostics; the mask uses the slab's own)."""
        if self._prefix:
            return self.slab.valid[: self.batch]
        return self.slab.valid.index_select(0, self._row_index)

    @property
    def bias(self) -> torch.Tensor:
        """Additive mask for the selected rows, narrowed to ``width`` columns."""
        bias = self.slab.bias[:, :, : self.width]
        if self._prefix:
            return bias[: self.batch * self.n_kv_head]
        return bias.index_select(0, self._bias_index)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        layer = self._check_layer(layer)
        expected = (self.batch, self.n_kv_head, 1, self.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"write needs {expected}, got {tuple(k.shape)} / {tuple(v.shape)}")
        self.slab.ensure_layer(layer)
        keys = self.slab.keys[layer]
        values = self.slab.values[layer]
        # Advanced-index assignment along the seat axis and each row's own column. One
        # kernel per K/V per layer; no full-batch temporary and no unrelated row writes.
        keys[self._row_index, :, self._columns] = k[:, :, 0, :].to(self.dtype)
        values[self._row_index, :, self._columns] = v[:, :, 0, :].to(self.dtype)

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._check_layer(layer)
        keys = self.slab.keys[layer][:, :, : self.width]
        values = self.slab.values[layer][:, :, : self.width]
        if self._prefix:
            return keys[: self.batch], values[: self.batch]
        index = self._row_index
        return keys.index_select(0, index), values.index_select(0, index)

    def advance(self) -> None:
        """One token was written for every batch row: move only those rows' cursors."""
        new_pos = self._columns + 1
        self.slab.pos.index_copy_(0, self._row_index, new_pos)
        current_valid = self.slab.valid.index_select(0, self._row_index)
        self.slab.valid.index_copy_(
            0, self._row_index, torch.maximum(current_valid, new_pos)
        )
        for row in self.seats:
            self.slab._host_pos[row] += 1
            self.slab._host_valid[row] = max(
                self.slab._host_valid[row], self.slab._host_pos[row]
            )
        highest = max(self.slab._host_pos[row] for row in self.seats)
        if highest > self.slab.max_len:
            raise IndexError(
                f"KvSlab overflow: max_len={self.slab.max_len}; a request decoded past "
                "its budget"
            )
        self._columns = new_pos.clone()

    def past_len(self) -> int:
        return max(self.slab._host_valid[row] for row in self.seats)

    def bytes(self) -> int:
        return self.slab.bytes()

    def mb(self) -> float:
        return self.slab.mb()


class Qwen35Engine:
    """``SchedulerEngine`` implementation for the hybrid Qwen3.5 decoder (W7 Step 2).

    One engine owns one open pool arm: the paged block pool (production) or the seat slab
    (E4b baseline). Both share a :class:`LinearStatePool`; ``open_hybrid_pool`` sizes both
    from the run's ``max_batch``/``linear_slots`` and the scheduler must be constructed with
    the same ``max_batch``.
    """

    def __init__(
        self,
        model,
        eos_token_id=None,
        *,
        max_prefill_tokens: int = 8192,
    ) -> None:
        if int(max_prefill_tokens) < 1:
            raise ValueError(f"max_prefill_tokens must be >= 1, got {max_prefill_tokens!r}")
        self.model = model.eval()
        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device
        self.vocab_size = int(model.config.vocab_size)
        self.max_position = int(model.config.max_position_embeddings)
        self.max_prefill_tokens = int(max_prefill_tokens)
        # D9-5: the KV containers cover only the 6 full-attention layers; ``n_layer`` is
        # the compact count, not the model's 24.
        self.full_attention_layers = [int(idx) for idx in model.full_attention_layers]
        self.n_layer = len(self.full_attention_layers)
        self.n_kv_head = int(model.config.n_kv_head)
        self.head_dim = int(model.head_dim)
        self.n_head = int(model.config.n_head)
        self.dtype = model.token_embedding.weight.dtype

        self.last_decode_event_ms: float | None = None
        self._use_cuda_events = self.device.type == "cuda"
        # Packed-prefill arm, built lazily by ``_varlen()`` (one per engine).
        self._varlen_backend = None

        # The arm is fixed at open time and kept after ``close_pool`` for provenance.
        # A fresh hybrid engine reports the production default (paged) before it is opened;
        # every tensor-touching method still requires an open pool.
        self._pool_kind = "paged"
        self._pool: PagedPool | KvSlab | None = None
        self._blocks: BlockManager | None = None
        self._state_pool: LinearStatePool | None = None
        self._frag: FragStats | None = None
        self._meta: dict[int, dict] = {}
        self._seats: dict[int, int] = {}
        self.max_batch: int | None = None
        self.max_len: int | None = None
        self._block_size = 16

    # ------------------------------------------------------------- properties
    @property
    def paged(self) -> bool:
        """Whether the open arm is the paged one (the scheduler's only mode switch)."""
        return self._pool_kind == "paged"

    @property
    def paged_max_batch(self) -> int | None:
        """The scheduler's row cap; ``None`` before a pool is opened."""
        return self.max_batch

    @property
    def pool_kv_mb(self) -> float | None:
        """Reserved KV of the open arm (whole block pool / whole seat slab)."""
        if self._pool is None:
            return None
        return float(self._pool.mb())

    @property
    def kv_pool_mb(self) -> float | None:
        """Alias of :attr:`pool_kv_mb` (the bench summary reads both names)."""
        return self.pool_kv_mb

    @property
    def slab(self) -> KvSlab:
        """The seat slab; the paged arm has none and raises instead of faking one."""
        if self._pool is None:
            raise RuntimeError("open_hybrid_pool() must be called before the scheduler runs")
        if self._pool_kind != "seat":
            raise RuntimeError(
                "the paged hybrid arm has no seat slab; decode goes through PagedKV "
                "(use pool='seat' for the E4b baseline)"
            )
        return self._pool

    @property
    def state_pool(self) -> LinearStatePool:
        """The linear-state pool (read-only access for tests/accounting)."""
        if self._state_pool is None:
            raise RuntimeError("open_hybrid_pool() must be called before reading the pool")
        return self._state_pool

    @property
    def block_manager(self) -> BlockManager:
        """The paged block manager; only the paged arm has one."""
        if self._blocks is None:
            raise RuntimeError("the paged arm is not open; no block manager exists")
        return self._blocks

    @property
    def linear_state_mb(self) -> float:
        """Used (active rows) linear-state MiB, from the pool's real tensors."""
        return float(self.state_pool.stats().used_mb)

    @property
    def linear_slots(self) -> int:
        """The linear-state slot capacity (the fill-mode budget knob)."""
        return int(self.state_pool.capacity)

    # ---------------------------------------------------------------- pool
    @torch.no_grad()
    def open_hybrid_pool(
        self,
        *,
        max_batch: int,
        max_len: int,
        linear_slots: int,
        kv_blocks: int | None = None,
        block_size: int = 16,
        pool: str = "paged",
    ):
        """Allocate one arm's KV pool plus the shared linear-state pool.

        ``pool="paged"`` builds a 6-layer :class:`PagedPool` + :class:`BlockManager`; it is
        CUDA/Triton-only, exactly like the 0.6B paged arm (there is no dense paged
        fallback). ``pool="seat"`` builds a 6-layer :class:`KvSlab` with ``max_batch`` rows
        and ``max_len`` columns (the E4b baseline).

        ``linear_slots`` may be smaller than ``max_batch`` (fill-only capacity runs); it
        must be at least 1. ``max_len`` is the seat row width / trace position bound and
        must not exceed the model's ``max_position``. ``kv_blocks`` and ``block_size`` are
        required/used only by the paged arm; the seat arm allocates its rows up front.

        The pools are allocated **outside** ``torch.inference_mode`` on purpose: the state
        pool's release path copies rows with ``copy_`` from the scheduler thread, and an
        inference tensor rejects that (see the module docstring). Returns the KV container.
        """
        if pool not in ARMS:
            raise ValueError(f"pool must be one of {ARMS}, got {pool!r}")
        for name, value in (
            ("max_batch", max_batch),
            ("max_len", max_len),
            ("linear_slots", linear_slots),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        if linear_slots < 1:
            raise ValueError(
                f"linear_slots must be >= 1 for the hybrid engine, got {linear_slots}"
            )
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if max_len > self.max_position:
            raise ValueError(
                f"max_len={max_len} exceeds the model's max_position={self.max_position}"
            )
        if self._meta:
            raise RuntimeError(
                f"{len(self._meta)} request(s) are still active; release them before "
                "re-opening the hybrid pool"
            )
        if self._pool is not None:
            self.close_pool()

        if pool == "paged":
            if self.device.type != "cuda":
                raise ValueError(
                    "paged hybrid decode is the CUDA block-table Triton kernel; the "
                    f"engine is on {self.device}. Use pool='seat' on CPU/MPS."
                )
            from model.triton_decode import TRITON_AVAILABLE

            if not TRITON_AVAILABLE:
                raise RuntimeError(
                    "paged hybrid decode requires Triton, but Triton could not be "
                    "imported; use pool='seat' (the dense oracle lives in tests)."
                )
            if kv_blocks is None:
                raise ValueError("pool='paged' needs kv_blocks")
            if isinstance(kv_blocks, bool) or not isinstance(kv_blocks, int):
                raise TypeError(f"kv_blocks must be an int, got {kv_blocks!r}")
            if kv_blocks < 1:
                raise ValueError(f"kv_blocks must be >= 1, got {kv_blocks}")
            if isinstance(block_size, bool) or not isinstance(block_size, int):
                raise TypeError(f"block_size must be an int, got {block_size!r}")
            if block_size < 1:
                raise ValueError(f"block_size must be >= 1, got {block_size}")
            kv_pool = PagedPool(
                self.n_layer,
                kv_blocks,
                block_size,
                self.n_kv_head,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
                n_head=self.n_head,
            )
            blocks = BlockManager(kv_blocks, block_size)
            frag = FragStats(block_size)
            self._block_size = int(block_size)
        else:
            kv_pool = KvSlab(
                self.n_layer,
                max_batch,
                self.n_kv_head,
                self.head_dim,
                int(max_len),
                dtype=self.dtype,
                device=self.device,
                n_head=self.n_head,
            )
            blocks = None
            frag = None
            self._block_size = int(block_size)

        state_pool = LinearStatePool(
            self.model.config,
            int(linear_slots),
            device=self.device,
            dtype=self.dtype,
        )
        self._pool_kind = pool
        self._pool = kv_pool
        self._blocks = blocks
        self._state_pool = state_pool
        self._frag = frag
        self._meta = {}
        self._seats = {}
        self.max_batch = int(max_batch)
        self.max_len = int(max_len)
        self.last_decode_event_ms = None
        return kv_pool

    # ------------------------------------------------------------- admission
    def can_admit_group(self, requests: list[dict]) -> bool:
        """Whether the whole candidate group fits **now**; never changes state.

        The group is the admission unit because ``prefill_group`` reserves every member
        back to back: two conditions apply to all members together (D9-1/§3.7):

        * rows: ``active + len(group) <= max_batch`` **and** ``<= linear_slots``;
        * paged: the sum of every member's ``blocks_for(prompt, budget)`` must fit the
          current free list (the same all-or-nothing test ``prefill_group`` makes);
          seat: every prompt plus budget must fit the row width.

        No prompt/seat semantics are validated here (``prefill_group`` does that); this is
        a capacity answer only, so an impossible spec refuses later rather than silently
        changing the meaning of admission.
        """
        self._require_pool()
        specs = list(requests)
        active = self._state_pool.active_count
        if active + len(specs) > min(int(self.max_batch), self._state_pool.capacity):
            return False
        if self.paged:
            needed = 0
            for spec in specs:
                if not isinstance(spec, dict):
                    return False
                try:
                    prompt = list(spec.get("prompt_ids") or [])
                    budget = int(spec.get("budget") or 0)
                except (TypeError, ValueError):
                    return False
                if budget < 0:
                    return False
                needed += self._blocks.blocks_for(len(prompt), budget)
            return needed <= self._blocks.stats()["free"]
        for spec in specs:
            if not isinstance(spec, dict):
                return False
            try:
                prompt = list(spec.get("prompt_ids") or [])
                budget = int(spec.get("budget") or 0)
            except (TypeError, ValueError):
                return False
            if budget < 0:
                return False
            if not prompt or len(prompt) > int(self.max_len):
                return False
            if len(prompt) + budget > int(self.max_len):
                return False
        return True

    # -------------------------------------------------------------- prefill
    @torch.no_grad()
    def prefill_group(self, requests: list[dict]) -> torch.Tensor:
        """Prefill a FIFO group with one packed varlen forward; ``(n, vocab)`` logits.

        The scheduler's group is the atomic admission/reservation unit **and** one
        forward (W7 Step 3, D9-6): every member's prompt is concatenated request-major into
        a single token axis and ``Qwen35Model.forward_packed`` runs once. The 6
        full-attention layers attend with ``cu_seqlens`` and write each request's KV
        straight into its paged block table or seat row through the one-shot sinks; the 18
        GDN layers loop over the segments, each request prefilling from empty state into
        its own pool slot. There is no per-request forward and no serial fallback (G2).

        Validation and reservation happen for the whole group before the forward. Any
        failure -- during reservation, inside the forward, or during activation -- frees
        every block and slot of the group and freezes every seat row it touched, then
        re-raises; no half-active handle survives and the scheduler can safely abort the
        group.

        The order of the returned rows is the request order of ``requests`` (and of
        ``cu_seqlens``); each row is that request's final prompt token (P-13).
        """
        self._require_pool()
        if not requests:
            raise ValueError("prefill_group needs at least one request")

        specs = self._validate_specs(requests)
        total_tokens = sum(len(spec["prompt"]) for spec in specs)
        if len(specs) > 1 and total_tokens > self.max_prefill_tokens:
            raise ValueError(
                f"prefill_group has {total_tokens} prompt tokens, above "
                f"max_prefill_tokens={self.max_prefill_tokens}; the scheduler must split "
                "the group (a single over-budget prompt is allowed alone)"
            )

        reserved = self._reserve_specs(specs)
        try:
            # Fresh single-row states: packed prefill never reads state.kv (the KV goes
            # through the sinks), and the GDN layers require conv/recurrent to be empty.
            states = [self._state_pool.state(entry["handle"].slot) for entry in reserved]
            sinks = [self._packed_sink(entry) for entry in reserved]
            tokens, cu_seqlens, positions = self._packed_layout(specs)
            packed = PackedPrefill(
                cu_seqlens=cu_seqlens,
                positions=positions,
                sinks=sinks,
                backend=self._varlen(),
            )
            logits, _ = self.model.forward_packed(tokens, packed, states)
            if not self.paged:
                for entry in reserved:
                    self._pool.activate(entry["spec"]["seat"], entry["spec"]["length"])
            for entry in reserved:
                self._state_pool.activate(entry["handle"])
        except BaseException:
            self._rollback(reserved)
            raise

        for entry in reserved:
            spec = entry["spec"]
            handle = entry["handle"]
            handle.prompt_len = spec["length"]
            handle.length = spec["length"]
            handle.block_table = list(entry["table"]) if entry["table"] is not None else None
            self._meta[spec["id"]] = {
                "handle": handle,
                "seat": spec["seat"],
                "prompt_len": spec["length"],
                "length": spec["length"],
                "block_table": list(entry["table"]) if entry["table"] is not None else None,
            }
            self._seats[spec["seat"]] = spec["id"]
        return logits

    def _varlen(self):
        """The packed-prefill backend for this device (explicit, never a fallback).

        Same selection as ``MiniLLMEngine._varlen``: CUDA -> ``FlashVarlen`` (which raises
        if the kernel is missing), CPU/MPS -> ``SdpaVarlen``. Both arms are recorded in
        ``attn_backends()`` so a run that silently executed the reference arm is visible.
        """
        if self._varlen_backend is None:
            from model.attention.varlen import backend_for

            self._varlen_backend = backend_for(self.device)
        return self._varlen_backend

    def _packed_layout(self, specs: list[dict]):
        """Request-major packed ``(tokens, cu_seqlens, positions)`` for one group.

        Tokens are concatenated in request order; every request's positions restart at 0
        (the packed layout has one token axis, not one shared position axis), which is what
        ``PackedPrefill`` validates and what the per-token RoPE table is gathered with.
        """
        cu_seqlens = [0]
        positions: list[int] = []
        tokens: list[int] = []
        for spec in specs:
            prompt = spec["prompt"]
            cu_seqlens.append(cu_seqlens[-1] + len(prompt))
            positions.extend(range(len(prompt)))
            tokens.extend(prompt)
        return (
            torch.tensor([tokens], dtype=torch.long, device=self.device),
            cu_seqlens,
            torch.tensor(positions, dtype=torch.long, device=self.device),
        )

    def _packed_sink(self, entry: dict):
        """One-shot write sink for this member, indexed by the **compact** full-attn layer.

        No ``_LayerMapSink`` here: ``Qwen35Attention``'s packed branch writes
        ``packed.write(self.layer_index, ...)`` with the compact index (0..5, D9-5), which
        is exactly the index ``PagedWriteSink``/``SlabWriteSink`` take.
        """
        spec = entry["spec"]
        if self.paged:
            return PagedWriteSink(
                self._pool,
                entry["table"],
                spec["length"],
                slots=self._pool.slot_ids(entry["table"], 0, spec["length"]),
            )
        return SlabWriteSink(self._pool, spec["seat"], spec["length"])

    def _validate_specs(self, requests: list[dict]) -> list[dict]:
        """Validate every member before anything is allocated (no state change)."""
        specs: list[dict] = []
        seen_ids: set[int] = set()
        seen_seats: set[int] = set()
        for index, raw in enumerate(requests):
            if not isinstance(raw, dict):
                raise TypeError(f"request {index} must be a dict, got {type(raw).__name__}")
            request_id = raw.get("id")
            if request_id is None:
                raise ValueError(f"request {index} has no 'id'")
            try:
                request_id = int(request_id)
            except (TypeError, ValueError):
                raise ValueError(f"request id {request_id!r} is not an integer") from None
            if request_id in seen_ids:
                raise ValueError(f"duplicate request id {request_id} in one group")
            if request_id in self._meta:
                raise ValueError(f"request {request_id} is already active")
            seen_ids.add(request_id)

            prompt = raw.get("prompt_ids")
            if prompt is None:
                raise ValueError(f"request {request_id} has no 'prompt_ids'")
            prompt = list(prompt)
            if not prompt:
                raise ValueError(f"request {request_id} has an empty prompt")
            for token in prompt:
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError(
                        f"request {request_id} prompt has a non-integer token {token!r}"
                    )
                if not 0 <= token < self.vocab_size:
                    raise ValueError(
                        f"request {request_id} token {token} outside [0, {self.vocab_size})"
                    )
            length = len(prompt)
            if length > self.max_position:
                raise ValueError(
                    f"request {request_id} prompt ({length}) exceeds max_position "
                    f"{self.max_position}"
                )
            if length > int(self.max_len):
                raise ValueError(
                    f"request {request_id} prompt ({length}) exceeds the pool's max_len "
                    f"{self.max_len}"
                )

            seat_value = raw.get("seat")
            if seat_value is None:
                raise ValueError(f"request {request_id} has no 'seat'")
            try:
                seat = int(seat_value)
            except (TypeError, ValueError):
                raise ValueError(f"request {request_id} seat {seat_value!r} is not an int") from None
            if not 0 <= seat < int(self.max_batch):
                raise IndexError(
                    f"request {request_id} seat {seat} outside the pool's batch of "
                    f"{self.max_batch}"
                )
            if seat in seen_seats:
                raise ValueError(f"two requests in one group share seat {seat}")
            if seat in self._seats:
                raise ValueError(f"seat {seat} is already occupied")
            seen_seats.add(seat)

            try:
                budget = int(raw.get("budget") or 0)
            except (TypeError, ValueError):
                raise ValueError(
                    f"request {request_id} budget {raw.get('budget')!r} is not an int"
                ) from None
            if budget < 0:
                raise ValueError(f"request {request_id} has a negative budget ({budget})")
            if not self.paged and length + budget > int(self.max_len):
                raise ValueError(
                    f"request {request_id} prompt+budget {length + budget} exceeds the "
                    f"seat pool's max_len {self.max_len}"
                )

            needed = (
                self._blocks.blocks_for(length, budget) if self.paged else 0
            )
            specs.append(
                {
                    "id": request_id,
                    "prompt": prompt,
                    "length": length,
                    "seat": seat,
                    "budget": budget,
                    "needed": needed,
                }
            )

        active = self._state_pool.active_count
        if active + len(specs) > int(self.max_batch):
            raise RuntimeError(
                f"cannot admit {len(specs)} request(s): {active} active, max_batch="
                f"{self.max_batch}"
            )
        if active + len(specs) > self._state_pool.capacity:
            raise RuntimeError(
                f"cannot admit {len(specs)} request(s): {active} active, linear_slots="
                f"{self._state_pool.capacity}"
            )
        if self.paged:
            needed = sum(spec["needed"] for spec in specs)
            free = self._blocks.stats()["free"]
            if needed > free:
                raise RuntimeError(
                    f"cannot reserve {needed} block(s) for {len(specs)} request(s): "
                    f"{free} free (can_admit_group should have refused the group)"
                )
        return specs

    def _reserve_specs(self, specs: list[dict]) -> list[dict]:
        """Allocate every slot and block table, or none (rollback on failure)."""
        reserved: list[dict] = []
        try:
            for spec in specs:
                handle = self._state_pool.alloc(spec["id"])
                table = None
                if self.paged:
                    table = self._blocks.reserve(spec["needed"])
                    if table is None:  # pragma: no cover - validation checked the sum
                        raise RuntimeError(
                            f"cannot reserve {spec['needed']} block(s) for request "
                            f"{spec['id']}: {self._blocks.stats()['free']} free"
                        )
                handle.prompt_len = spec["length"]
                handle.length = spec["length"]
                handle.block_table = list(table) if table is not None else None
                reserved.append({"spec": spec, "handle": handle, "table": table})
        except BaseException:
            self._rollback(reserved)
            raise
        return reserved

    def _rollback(self, reserved: list[dict]) -> None:
        """Free every reserved/prefilled member of a failed group (best effort, ordered)."""
        for entry in reversed(reserved):
            spec = entry["spec"]
            if not self.paged:
                self._pool.freeze(spec["seat"])
            if entry["table"] is not None:
                self._blocks.free(entry["table"])
            self._state_pool.free(entry["handle"])
            self._meta.pop(spec["id"], None)
            if self._seats.get(spec["seat"]) == spec["id"]:
                del self._seats[spec["seat"]]

    # --------------------------------------------------------------- decode
    @torch.inference_mode()
    def decode_batch(self, tokens: list[int], slab: KvSlab | None = None) -> torch.Tensor:
        """One forward over every active request; returns ``(max_batch, vocab)``.

        ``tokens`` is indexed by scheduler seat and must be exactly ``max_batch`` long
        (idle seats carry the scheduler's placeholder and are not forwarded). Active
        handles are compressed to the pool's packed ``[0, B)`` slots; logits are scattered
        back to their seat rows, so ``Scheduler._decode`` can index ``ids[req.seat]``.
        With no active request this returns zeros without a forward.

        Paged: a :class:`PagedKV` over the active block tables (``rope_len=max_position``,
        the run's RoPE bound, not the block capacity). Seat: :class:`_SlabRows`, which maps
        the compact rows onto their seat rows and advances only those. ``slab`` is accepted
        for the scheduler's seat call; on the paged arm an explicit slab is refused rather
        than silently ignored (the two arms are never mixed).
        """
        self._require_pool()
        if len(tokens) != int(self.max_batch):
            raise ValueError(
                f"decode_batch needs one token per seat (max_batch={self.max_batch}), got "
                f"{len(tokens)}"
            )
        if self.paged:
            if slab is not None:
                raise ValueError(
                    "the paged hybrid arm assembles its own PagedKV batch; do not pass a slab"
                )
        elif slab is not None and slab is not self._pool:
            raise ValueError("decode_batch got a slab other than this engine's seat pool")

        active = sorted(self._meta.items(), key=lambda item: item[1]["handle"].slot)
        if not active:
            self.last_decode_event_ms = None
            return torch.zeros(
                (int(self.max_batch), self.vocab_size),
                dtype=self.dtype,
                device=self.device,
            )

        batch = len(active)
        slots = [meta["handle"].slot for _, meta in active]
        if slots != list(range(batch)):
            raise RuntimeError(
                f"active handles must occupy the packed slots [0, {batch}), got {slots}"
            )
        if self._state_pool.active_count != batch:
            raise RuntimeError(
                f"{self._state_pool.active_count} active pool slots for {batch} active "
                "handles; the pool and the engine's request table disagree"
            )
        seats = [meta["seat"] for _, meta in active]
        if len(set(seats)) != batch:
            raise RuntimeError(f"two active requests share a seat: {seats}")
        lengths = [int(meta["length"]) for _, meta in active]
        block_tables = [meta["block_table"] for _, meta in active]

        ids = torch.tensor(
            [[int(tokens[seat])] for seat in seats],
            dtype=torch.long,
            device=self.device,
        )
        if self.paged:
            container = PagedKV(
                self._pool,
                block_tables,
                lengths,
                # Run position bound (prompt + budget + 1), not the model's 262144-token
                # maximum: PagedKV only needs a RoPE table wide enough for this trace, and
                # building the full model bound would allocate ~33 MiB of bf16 cos/sin on
                # the first decode for no benefit (W4's PagedEngine passes `need` too).
                rope_len=self.max_len,
                seats=seats,
                n_head=self.n_head,
            )
        else:
            container = _SlabRows(self._pool, seats)
        state = self._state_pool.state(range(batch), kv=container)

        start = time.perf_counter()
        event = self._new_event()
        if event is not None:
            event.record()
        logits, _ = self.model.forward_decode_batch(ids, state)
        logits = logits[:, -1, :]
        if event is not None:
            end = self._new_event()
            end.record()
            self._sync()
            self.last_decode_event_ms = float(event.elapsed_time(end))
        else:
            self.last_decode_event_ms = None
        dt_s = time.perf_counter() - start

        for _, meta in active:
            meta["length"] = int(meta["length"]) + 1
            meta["handle"].length = meta["length"]

        out = logits.new_zeros((int(self.max_batch), self.vocab_size))
        out.index_copy_(
            0,
            torch.tensor(seats, dtype=torch.long, device=self.device),
            logits,
        )

        if self.paged and self._frag is not None:
            written = sum(int(meta["length"]) for _, meta in active)
            written_blocks = sum(
                (int(meta["length"]) + self._block_size - 1) // self._block_size
                for _, meta in active
            )
            self._frag.observe(dt_s, written, written_blocks)
        return out

    # -------------------------------------------------------------- release
    def release_request(self, request_id: int) -> None:
        """Release one request's KV and linear slot; **idempotent**.

        The plan (§3.3) wants an unknown id to error in the *pool*, and
        :meth:`LinearStatePool.free` still does that. At the engine boundary the scheduler
        legitimately calls this twice: a failed group is rolled back inside
        ``prefill_group`` and then ``Scheduler._run_group`` aborts every member, which
        calls this again. So an unknown/already-released id is a no-op here; a half-active
        request cannot survive and a double release cannot double-free.
        """
        if self._state_pool is None:
            return
        try:
            request_id = int(request_id)
        except (TypeError, ValueError):
            raise TypeError(f"request_id must be an int, got {request_id!r}") from None
        meta = self._meta.get(request_id)
        if meta is None:
            return
        seat = meta["seat"]
        table = meta.get("block_table")
        if table is not None and self._blocks is not None:
            self._blocks.free(table)
        if not self.paged and self._pool is not None:
            self._pool.freeze(seat)
        self._state_pool.free(meta["handle"])
        self._meta.pop(request_id, None)
        if self._seats.get(seat) == request_id:
            del self._seats[seat]

    def freeze_seat(self, seat: int) -> None:
        """Seat hook: retire the request on ``seat``, or park a never-used row.

        The scheduler calls this for every virgin seat before the first decode and for a
        seat whose request retired. An unassigned seat is just ``KvSlab.freeze`` (safe for
        a row that never activated); an owned seat goes through :meth:`release_request` so
        its linear slot is returned too. The paged arm has no seats and raises instead of
        pretending it does.
        """
        self._require_pool()
        if self.paged:
            raise RuntimeError(
                "freeze_seat() is the seat arm's hook; the paged arm retires requests "
                "through release_request()"
            )
        try:
            seat = int(seat)
        except (TypeError, ValueError):
            raise TypeError(f"seat must be an int, got {seat!r}") from None
        if not 0 <= seat < int(self.max_batch):
            raise IndexError(f"seat {seat} outside the pool's batch of {self.max_batch}")
        request_id = self._seats.get(seat)
        if request_id is not None:
            self.release_request(request_id)
        else:
            self._pool.freeze(seat)

    # ------------------------------------------------------------- accounting
    def paged_stats(self) -> dict:
        """Code-produced pool/state account, with the W4 field names preserved.

        ``kv_mb`` is the whole reservation, ``kv_live_mb`` the bytes of written tokens and
        ``kv_alloc_mb`` the capacity of the blocks (or seat rows) currently reserved --
        three different numbers, all reported. The paged arm's ``frag_internal`` is the
        time-weighted C2 metric and ``reservation_slack`` the instantaneous upper bound;
        the seat arm has no blocks, so both are ``None`` and the block counts are zero.
        ``workspace_mb`` is a structural zero for both arms (§3.8), not a missing estimate.

        W7 adds only ``linear_state_mb`` / ``linear_slots`` / ``active_rows`` /
        ``peak_active`` / ``leak`` (D9-8). ``leak`` is True only when the engine's request
        table and the pools disagree (a slot or block with no owner, or an owner with no
        state); it is False while work is in flight and False after a clean release/close.
        """
        self._require_pool()
        active = sorted(self._meta.items(), key=lambda item: item[1]["handle"].slot)
        written = sum(int(meta["length"]) for _, meta in active)
        state_stats = self._state_pool.stats()
        if self.paged:
            pool = self._pool
            blocks = self._blocks
            per_token = (
                2 * pool.n_layer * pool.n_kv_head * pool.head_dim * pool.dtype.itemsize
            )
            in_use = blocks.in_use
            reserved_tokens = in_use * pool.block_size
            block_stats = blocks.stats()
            written_blocks = sum(
                (int(meta["length"]) + pool.block_size - 1) // pool.block_size
                for _, meta in active
            )
            fields = {
                "kv_mb": pool.mb(),
                "kv_live_mb": written * per_token / 2**20,
                "kv_alloc_mb": reserved_tokens * per_token / 2**20,
                "workspace_mb": 0.0,
                "blocks_in_use": in_use,
                "free_blocks": block_stats["free"],
                "written_blocks": written_blocks,
                "frag_internal": (
                    None if self._frag is None else self._frag.internal_frag()
                ),
                "reservation_slack": (
                    None
                    if self._frag is None
                    else self._frag.reservation_slack(reserved_tokens, written)
                ),
                "decode_impl": "triton",
            }
        else:
            slab = self._pool
            per_token = (
                2 * slab.n_layer * slab.n_kv_head * slab.head_dim * slab.dtype.itemsize
            )
            fields = {
                "kv_mb": slab.mb(),
                "kv_live_mb": written * per_token / 2**20,
                "kv_alloc_mb": slab.mb(),
                "workspace_mb": 0.0,
                # The seat arm really has no blocks; never pretend it is paged.
                "blocks_in_use": 0,
                "free_blocks": 0,
                "written_blocks": 0,
                "frag_internal": None,
                "reservation_slack": None,
                "decode_impl": "hand_baddbmm",
            }
        fields.update(
            {
                "linear_state_mb": state_stats.used_mb,
                "linear_slots": state_stats.slot_count,
                "active_rows": len(active),
                "peak_active": state_stats.peak_active,
                "leak": self._leaked(),
            }
        )
        return fields

    def _leaked(self) -> bool:
        """Whether the request table and the pool ledgers disagree right now."""
        if self._state_pool is None:
            return False
        if self._state_pool.stats().active != len(self._meta):
            return True
        if self._blocks is not None:
            expected = sum(
                len(meta["block_table"] or []) for meta in self._meta.values()
            )
            if self._blocks.in_use != expected:
                return True
        if not self._meta:
            try:
                self._state_pool.assert_no_leaks()
            except AssertionError:
                return True
        return False

    def attn_backends(self) -> dict[str, str]:
        """Backends actually used, for run provenance.

        ``prefill`` is the packed varlen arm (``flash-attn-varlen`` on CUDA,
        ``sdpa-varlen`` on CPU/MPS), selected once by ``backend_for``; ``decode`` is the
        Triton block-table kernel or the seat arm's ``hand_baddbmm``.
        """
        return {
            "prefill": self._varlen().name,
            "decode": "paged-triton" if self.paged else "hand_baddbmm",
        }

    # --------------------------------------------------------------- lifetime
    def close_pool(self) -> None:
        """Release every request and drop the KV pool; safe to call repeatedly.

        After the release loop the linear pool must be empty (``assert_no_leaks``) and the
        paged block ledger must have no held blocks. The closed ``LinearStatePool`` object
        is kept for diagnostics (``engine.state_pool.assert_no_leaks()``), but every
        tensor-touching method requires a freshly opened pool.
        """
        if self._pool is None and self._state_pool is None:
            return
        for request_id in list(self._meta):
            self.release_request(request_id)
        if self._state_pool is not None:
            self._state_pool.close()
            self._state_pool.assert_no_leaks()
        if self._blocks is not None and self._blocks.in_use:
            raise AssertionError(
                f"BlockManager leaked {self._blocks.in_use} block(s) after close_pool()"
            )
        self._pool = None
        self._blocks = None
        self._frag = None
        self._meta = {}
        self._seats = {}
        self.last_decode_event_ms = None
        self._release_memory()

    # ---------------------------------------------------------------- helpers
    def _require_pool(self) -> None:
        if self._pool is None or self._state_pool is None:
            raise RuntimeError(
                "open_hybrid_pool() must be called before the scheduler runs"
            )

    def _release_memory(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _new_event(self) -> torch.cuda.Event | None:
        if not self._use_cuda_events:
            return None
        return torch.cuda.Event(enable_timing=True)
