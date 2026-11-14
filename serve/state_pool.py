"""Per-request linear-attention state pool and pooled ``Qwen35State`` (W7 Step 1).

``Qwen35State`` (``model/qwen35_model.py``) owns one request's 18 Gated DeltaNet states
and allocates them as the forward runs. W7 needs the opposite: a fixed pool of slots,
allocated once, with the *active* slots packed into ``[0, active_count)`` so a future
batch-decode forward can read a contiguous ``[0:B)`` view without an 18 MiB gather every
step. This module is that storage and lifetime layer, and nothing else:

* :class:`LinearStatePool` preallocates, per linear-attention layer, ``conv [capacity,
  conv_dim, k - 1]`` in the model dtype and ``recurrent [capacity, num_v_heads, K, V]``
  in **fp32** (D8's delta-rule contract). Allocation is O(1); release copies the last
  active slot's rows into the hole (``copy_``, never reassignment) so the active region
  stays packed.
* :class:`RequestHandle` is the request's one handle: slot, phase, and placeholders for
  ``prompt_len``/``length``/``block_table``. KV blocks are **not** allocated here (D9-1
  keeps KV in ``BlockManager``/``PagedPool``; Step 2 wires the two together).
* :class:`PoolStats` reports ``used``/``reserved``/``slack`` from the real tensors. The
  analytic ``18.633 MiB`` is a cross-check in the tests, never an input to accounting.
* :class:`PooledQwen35State` subclasses ``Qwen35State`` so ``Qwen35Model.forward``'s
  ``isinstance`` check still accepts it, and replaces storage/lifetime: the getters
  return views of the pool rows and the setters ``copy_`` back into those rows.

Scope (W7 Step 1 only): no batch decode, no scheduler, no paged/seat KV container, no
``forward_packed``. The pooled state is exercised one request at a time against the
single-sequence reference; the batch row view is tested at the tensor level.

Three deliberate properties of this implementation
--------------------------------------------------
1. **Pool rows are uninitialized storage.** ``free`` does not zero the row it returns;
   a slot's bytes are only meaningful once a setter has written them, and
   ``PooledQwen35State`` gates every read on per-(slot, layer) "written" flags. A failed
   (reserved-only) request therefore cannot leak stale state into the next request: the
   next prefill reads ``None`` for every layer and overwrites every row it later reads.
2. **A ``PooledQwen35State`` is bound to physical slot rows, not to handles.** Releasing
   a lower slot relocates the last slot's rows and updates *that handle's* ``slot``.
   Each slot carries an epoch bumped on allocation, swap-in and free; a state object
   captures its rows' epochs and raises on any later read that would see a released,
   moved or reused row. Step 2 must rebuild the batch state after a release (or hold
   handles) rather than reuse a stale object.
3. **``cull`` rebinds which rows this state presents; it never frees pool slots.** A
   contiguous row run is a zero-copy view; an arbitrary surviving order falls back to
   ``index_select`` (a gather) on read and ``index_copy_`` on write. That fallback is
   correct but not zero-copy, which is why W7's scheduler releases handles instead of
   relying on ``cull`` (§3.4).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch

# The two underscore names are the base state's own geometry/kind contract. Re-deriving
# them here would let the pool and the layer disagree about ``k - 1`` or a layer kind.
from model.qwen35_model import Qwen35State, _LINEAR_ATTENTION, _linear_dims

__all__ = [
    "LinearStatePool",
    "PooledQwen35State",
    "PoolExhausted",
    "PoolStats",
    "RequestHandle",
    "ACTIVE",
    "FREE",
    "RELEASED",
    "RELEASING",
    "RESERVED",
]

#: Slot / handle phases. A slot is ``free`` or mirrors the phase of its owner.
FREE = "free"
RESERVED = "reserved"
ACTIVE = "active"
RELEASING = "releasing"
RELEASED = "released"

_MIB = 1024 * 1024


def _mb(nbytes: int) -> float:
    return nbytes / _MIB


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass
class RequestHandle:
    """One request's linear-state slot and lifecycle phase.

    ``slot`` is an index into :class:`LinearStatePool` while the request is live and
    ``None`` after release. ``prompt_len``/``length``/``block_table`` are placeholders
    Step 2 fills in without changing this class's shape; this module never allocates a
    KV block (D9-1).
    """

    request_id: Hashable
    slot: int | None
    phase: str = RESERVED
    prompt_len: int | None = None
    length: int | None = None
    block_table: list[int] | None = None


@dataclass(frozen=True)
class PoolStats:
    """Pool accounting in MiB plus the exact integer bytes it was computed from.

    ``used`` counts the rows of active slots (``element_size * numel`` of the real
    tensor slices), ``reserved`` counts every preallocated row, and ``slack`` is their
    difference. Free slots are reserved but not used.
    """

    used_mb: float
    reserved_mb: float
    slack_mb: float
    slot_count: int
    active: int
    free: int
    used_bytes: int
    reserved_bytes: int
    slack_bytes: int
    per_slot_bytes: int
    peak_active: int


class PoolExhausted(RuntimeError):
    """``LinearStatePool.alloc`` had no free slot; the pool is unchanged."""


# --------------------------------------------------------------------- the pool


class LinearStatePool:
    """Fixed-capacity storage for every linear-attention layer's per-request state.

    Shapes follow the config, not a constant: ``conv_dim``, the value-head count and the
    head geometry come from the same ``_linear_dims`` helper the layer and
    ``Qwen35State`` use, and ``k - 1`` is the minimal sufficient conv state (P-49). The
    pool allocates once in ``__init__``; neither ``alloc`` nor ``free`` allocates state
    storage.

    Active slots are exactly ``[0, active_count)``. ``free`` swaps the last active slot
    into the released hole, so there is never a hole inside the active region and the
    per-step decode view is a slice, not a gather. Slot bytes are uninitialized until a
    setter writes them (see the module docstring).
    """

    def __init__(
        self,
        config,
        capacity: int,
        *,
        device,
        dtype: torch.dtype,
    ):
        conv_dim, num_v_heads, head_k_dim, head_v_dim, kernel = _linear_dims(config)
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity must be an int, got {capacity!r}")
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {dtype!r}")

        self.config = config
        self.capacity = capacity
        self.device = torch.device(device)
        self.dtype = dtype
        self.conv_dim = conv_dim
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.kernel_size = kernel
        self.linear_layers = [
            idx
            for idx, kind in enumerate(config.layer_types)
            if kind == _LINEAR_ATTENTION
        ]

        # One tensor per linear-attention layer. A single shared tensor would alias every
        # layer's conv/recurrent state; the model reads all 18 in one forward.
        self.conv: dict[int, torch.Tensor] = {
            layer: torch.empty(
                capacity, conv_dim, kernel - 1, dtype=dtype, device=self.device
            )
            for layer in self.linear_layers
        }
        self.recurrent: dict[int, torch.Tensor] = {
            layer: torch.empty(
                capacity,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                dtype=torch.float32,
                device=self.device,
            )
            for layer in self.linear_layers
        }
        # Per-(slot, layer) "a setter has written this row" flags. A read is only valid
        # after its layer's prefill wrote it; before that the getter returns None exactly
        # like a fresh ``Qwen35State``. Per layer, not per slot: within one prefill the
        # model reads layer L before writing it.
        self._conv_ready: dict[int, list[bool]] = {
            layer: [False] * capacity for layer in self.linear_layers
        }
        self._recurrent_ready: dict[int, list[bool]] = {
            layer: [False] * capacity for layer in self.linear_layers
        }

        self._handles: dict[Hashable, RequestHandle] = {}
        self._slot_owner: list[RequestHandle | None] = [None] * capacity
        self._slot_phase: list[str] = [FREE] * capacity
        # Bumped whenever a slot's contents change identity (allocation, swap-in, free).
        # A PooledQwen35State captures the epochs of its rows so a state object held
        # across a release raises instead of reading another request's state.
        self._slot_epoch: list[int] = [0] * capacity
        self._active_count = 0
        self._peak_active = 0
        self._closed = False

    # ------------------------------------------------------------- inspection

    @property
    def active_count(self) -> int:
        """Number of occupied slots; the active region is exactly ``[0, this)``."""
        return self._active_count

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def peak_active(self) -> int:
        return self._peak_active

    def slot_phase(self, slot: int) -> str:
        """``free``/``reserved``/``active``/``releasing``/``released`` for one slot."""
        self._check_slot(slot)
        return self._slot_phase[slot]

    def handle(self, request_id: Hashable) -> RequestHandle:
        """The handle for ``request_id`` (including already-released handles)."""
        try:
            return self._handles[request_id]
        except KeyError:
            raise KeyError(f"unknown request id {request_id!r}") from None

    def _check_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError(f"slot must be an int, got {slot!r}")
        if not 0 <= slot < self.capacity:
            raise IndexError(f"slot={slot} is out of range for capacity={self.capacity}")
        return slot

    # ------------------------------------------------------------------ bytes

    def conv_row_bytes(self, layer: int) -> int:
        """Bytes of one slot's conv row for ``layer`` (from the real tensor)."""
        tensor = self.conv[layer]
        return tensor.shape[1] * tensor.shape[2] * tensor.element_size()

    def recurrent_row_bytes(self, layer: int) -> int:
        """Bytes of one slot's recurrent row for ``layer`` (fp32, from the tensor)."""
        tensor = self.recurrent[layer]
        return (
            tensor.shape[1]
            * tensor.shape[2]
            * tensor.shape[3]
            * tensor.element_size()
        )

    def per_slot_bytes(self) -> int:
        """Bytes of every linear-attention layer's state for one slot."""
        return sum(
            self.conv_row_bytes(layer) + self.recurrent_row_bytes(layer)
            for layer in self.linear_layers
        )

    def linear_state_bytes(self) -> int:
        """Bytes actually occupied by the active rows (``used``)."""
        total = 0
        for layer in self.linear_layers:
            total += _tensor_nbytes(self.conv[layer][: self._active_count])
            total += _tensor_nbytes(self.recurrent[layer][: self._active_count])
        return total

    def reserved_bytes(self) -> int:
        """Bytes preallocated for ``capacity`` slots (``reserved``)."""
        total = 0
        for layer in self.linear_layers:
            total += _tensor_nbytes(self.conv[layer])
            total += _tensor_nbytes(self.recurrent[layer])
        return total

    def stats(self) -> PoolStats:
        used = self.linear_state_bytes()
        reserved = self.reserved_bytes()
        return PoolStats(
            used_mb=_mb(used),
            reserved_mb=_mb(reserved),
            slack_mb=_mb(reserved - used),
            slot_count=self.capacity,
            active=self._active_count,
            free=self.capacity - self._active_count,
            used_bytes=used,
            reserved_bytes=reserved,
            slack_bytes=reserved - used,
            per_slot_bytes=self.per_slot_bytes(),
            peak_active=self._peak_active,
        )

    # ------------------------------------------------------------ allocation

    def alloc(self, request_id: Hashable) -> RequestHandle:
        """Reserve the slot at ``active_count`` for ``request_id`` and return its handle.

        O(1); raises :class:`PoolExhausted` (pool unchanged) when full. Reusing the id of
        an already-released handle is allowed and returns a fresh handle.
        """
        if self._closed:
            raise RuntimeError("LinearStatePool is closed; alloc is not allowed")
        try:
            hash(request_id)
        except TypeError:
            raise TypeError(f"request_id must be hashable, got {request_id!r}") from None
        existing = self._handles.get(request_id)
        if existing is not None and existing.phase != RELEASED:
            raise ValueError(
                f"request id {request_id!r} already holds slot {existing.slot} "
                f"({existing.phase}); release it first"
            )
        if self._active_count >= self.capacity:
            raise PoolExhausted(
                f"LinearStatePool is full: {self._active_count}/{self.capacity} slots "
                "are active"
            )

        slot = self._active_count
        handle = RequestHandle(request_id=request_id, slot=slot)
        self._handles[request_id] = handle
        self._slot_owner[slot] = handle
        self._slot_phase[slot] = RESERVED
        self._slot_epoch[slot] += 1
        for layer in self.linear_layers:
            self._conv_ready[layer][slot] = False
            self._recurrent_ready[layer][slot] = False
        self._active_count += 1
        if self._active_count > self._peak_active:
            self._peak_active = self._active_count
        return handle

    def activate(self, handle_or_id) -> RequestHandle:
        """``reserved -> active``. Idempotent for an already-active handle."""
        handle = self._resolve(handle_or_id)
        if handle is None:
            raise RuntimeError(
                f"request {handle_or_id!r} is already released; it cannot be activated"
            )
        if handle.phase == ACTIVE:
            return handle
        if handle.phase != RESERVED:
            raise RuntimeError(
                f"request {handle.request_id!r} is {handle.phase!r}; only 'reserved' "
                "can be activated"
            )
        handle.phase = ACTIVE
        self._slot_phase[handle.slot] = ACTIVE
        return handle

    def free(self, handle_or_id) -> None:
        """Release a slot, packing the active region by swapping the last row in.

        Releasing an already-released handle (or its id) is a no-op. An unknown id
        raises ``KeyError``; a live handle that belongs to another pool raises
        ``ValueError``. Releasing a merely *reserved* handle is the W7 rollback path
        (§3.3): the slot returns to the free region and no half-allocated state survives
        in the accounting.
        """
        handle = self._resolve(handle_or_id)
        if handle is None:
            return

        slot = handle.slot
        if slot is None:  # pragma: no cover - invariant: live handles have a slot
            raise RuntimeError(f"live handle {handle.request_id!r} has no slot")
        handle.phase = RELEASING
        self._slot_phase[slot] = RELEASING

        last = self._active_count - 1
        if slot != last:
            moved = self._slot_owner[last]
            if moved is None:  # pragma: no cover - invariant: [0, active) is occupied
                raise RuntimeError(f"active slot {last} has no owner")
            for layer in self.linear_layers:
                self.conv[layer][slot].copy_(self.conv[layer][last])
                self.recurrent[layer][slot].copy_(self.recurrent[layer][last])
                self._conv_ready[layer][slot] = self._conv_ready[layer][last]
                self._recurrent_ready[layer][slot] = self._recurrent_ready[layer][last]
            moved.slot = slot
            self._slot_owner[slot] = moved
            self._slot_phase[slot] = moved.phase
            self._slot_epoch[slot] += 1

        # The (possibly moved-out) tail row is now free. Its ready flags must not survive
        # into the next allocation; the bytes are left as-is (see module docstring).
        self._slot_owner[last] = None
        self._slot_phase[last] = FREE
        self._slot_epoch[last] += 1
        for layer in self.linear_layers:
            self._conv_ready[layer][last] = False
            self._recurrent_ready[layer][last] = False
        self._active_count = last
        handle.slot = None
        handle.phase = RELEASED

    def _resolve(self, handle_or_id) -> RequestHandle | None:
        """The live handle for ``handle_or_id``, or ``None`` when already released."""
        if isinstance(handle_or_id, RequestHandle):
            handle = handle_or_id
            if handle.phase == RELEASED:
                return None
            if self._handles.get(handle.request_id) is not handle:
                raise ValueError(
                    f"handle for request {handle.request_id!r} does not belong to this "
                    "pool (or was replaced by a newer handle)"
                )
            return handle
        try:
            handle = self._handles[handle_or_id]
        except (KeyError, TypeError):
            raise KeyError(f"unknown request id {handle_or_id!r}") from None
        return None if handle.phase == RELEASED else handle

    # ----------------------------------------------------------------- states

    def state(self, rows, *, kv=None) -> "PooledQwen35State":
        """A :class:`PooledQwen35State` over ``rows`` (an int or a sequence of slots)."""
        if isinstance(rows, int) and not isinstance(rows, bool):
            rows = [rows]
        return PooledQwen35State(self, rows, kv=kv)

    # --------------------------------------------------------------- lifetime

    def close(self) -> None:
        """Release every slot. Safe to call repeatedly; handles are marked released."""
        if self._closed:
            return
        for handle in self._handles.values():
            if handle.phase != RELEASED:
                handle.phase = RELEASED
                handle.slot = None
        for slot in range(self.capacity):
            self._slot_owner[slot] = None
            self._slot_phase[slot] = FREE
        for layer in self.linear_layers:
            self._conv_ready[layer] = [False] * self.capacity
            self._recurrent_ready[layer] = [False] * self.capacity
        self._active_count = 0
        self._closed = True

    def assert_no_leaks(self) -> None:
        """Raise ``AssertionError`` unless every slot is free and every handle released."""
        problems: list[str] = []
        if self._active_count != 0:
            problems.append(f"active_count={self._active_count} (expected 0)")
        live = [
            f"{handle.request_id!r}:{handle.phase}"
            for handle in self._handles.values()
            if handle.phase != RELEASED
        ]
        if live:
            problems.append(f"{len(live)} unreleased handle(s): {', '.join(live)}")
        owned = [
            f"{slot}:{owner.request_id!r}"
            for slot, owner in enumerate(self._slot_owner)
            if owner is not None
        ]
        if owned:
            problems.append(f"{len(owned)} slot(s) still owned: {', '.join(owned)}")
        if problems:
            raise AssertionError("LinearStatePool leaked: " + "; ".join(problems))

    # ------------------------------------------------------------- row access

    @staticmethod
    def _is_contiguous_run(rows: Sequence[int]) -> bool:
        return all(high == low + 1 for low, high in zip(rows, rows[1:]))

    @classmethod
    def _rows_view(cls, storage: torch.Tensor, rows: Sequence[int]) -> torch.Tensor:
        """Rows of ``storage`` as a view when contiguous, else an exact gather copy."""
        if cls._is_contiguous_run(rows):
            start = rows[0]
            return storage[start : start + len(rows)]
        index = torch.tensor(list(rows), dtype=torch.long, device=storage.device)
        return storage.index_select(0, index)

    @classmethod
    def _write_rows(cls, storage: torch.Tensor, rows: Sequence[int], value: torch.Tensor):
        """``copy_`` ``value`` into ``rows`` without ever rebinding ``storage``."""
        if cls._is_contiguous_run(rows):
            start = rows[0]
            storage[start : start + len(rows)].copy_(value)
        else:
            index = torch.tensor(list(rows), dtype=torch.long, device=storage.device)
            storage.index_copy_(0, index, value)


# --------------------------------------------------------------- pooled state


class PooledQwen35State(Qwen35State):
    """``Qwen35State`` whose linear-attention storage lives in a :class:`LinearStatePool`.

    ``rows`` is the list of pool slots this state presents, in order. One request uses
    ``[slot]``; a batch-decode view uses ``range(B)`` (the packed active region).
    ``batch`` is ``len(rows)`` and the inherited shape checks use it. The optional
    ``kv`` container is injected into ``Qwen35State.__init__`` unchanged: Step 1's
    numeric reference passes a plain ``KVcache``; Step 2 will pass a paged/seat view.

    The getters return pool views (dtype/shape as the config says) and the setters
    validate exactly like the base class, then ``copy_`` into the pool rows. There is no
    per-request tensor: the pool is the storage, and a setter never rebinds it.

    Freshness: before a layer's state has been written by a setter, ``conv_state`` /
    ``recurrent_state`` return ``None`` for that layer, matching a fresh
    ``Qwen35State`` (and what a ``past_len == 0`` prefill requires). The flag lives in
    the pool per (slot, layer), so a batch state created after per-request prefills sees
    the states that were written, while a slot reused after release starts ``None``.

    ``cull`` rebinds ``rows`` to the surviving slots and culls the injected KV; it does
    not release pool slots or rewind ``position``. See the module docstring for the
    gather fallback on a non-contiguous surviving order.
    """

    def __init__(self, pool: LinearStatePool, rows: Sequence[int], kv=None):
        if not isinstance(pool, LinearStatePool):
            raise TypeError(
                f"pool must be a LinearStatePool, got {type(pool).__name__}"
            )
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError(f"rows must be a sequence of slot indices, got {rows!r}")
        row_list = list(rows)
        if not row_list:
            raise ValueError("rows must contain at least one slot")
        for row in row_list:
            if isinstance(row, bool) or not isinstance(row, int):
                raise TypeError(f"row indices must be ints, got {row!r}")
            if not 0 <= row < pool.active_count:
                raise ValueError(
                    f"row {row} is not in the active region [0, {pool.active_count}); "
                    "allocate the slot before wrapping it in a pooled state"
                )
        if len(set(row_list)) != len(row_list):
            raise ValueError(f"rows must be distinct, got {row_list}")

        super().__init__(
            pool.config,
            batch=len(row_list),
            device=pool.device,
            dtype=pool.dtype,
            kv=kv,
        )
        self.pool = pool
        self._rows = row_list
        self._row_epochs = [pool._slot_epoch[row] for row in row_list]

    # -------------------------------------------------------------- internals

    @property
    def rows(self) -> tuple[int, ...]:
        """The pool slots this state reads and writes, in row order."""
        return tuple(self._rows)

    def _check_live(self) -> None:
        if self.pool.closed:
            raise RuntimeError("the backing LinearStatePool is closed")
        for row, epoch in zip(self._rows, self._row_epochs):
            if row >= self.pool.active_count or self.pool._slot_epoch[row] != epoch:
                raise RuntimeError(
                    f"slot {row} is not the row this state was built on "
                    f"(active_count={self.pool.active_count}); the pool released, "
                    "reused or moved it after the state was created -- rebuild the "
                    "pooled state for the surviving rows"
                )

    def _layer_ready(self, ready: dict[int, list[bool]], layer: int, name: str) -> bool:
        flags = [ready[layer][row] for row in self._rows]
        if all(flags):
            return True
        if not any(flags):
            return False
        raise RuntimeError(
            f"{name} for layer {layer} is written for only some of rows {self._rows}; "
            "a pooled state never mixes an initialized and an uninitialized row"
        )

    # --------------------------------------------------- linear-attention state

    def conv_state(self, layer: int) -> torch.Tensor | None:
        self._check_layer(layer)
        self._check_linear_kind(layer, "conv_state")
        self._check_live()
        if not self._layer_ready(self.pool._conv_ready, layer, "conv_state"):
            return None
        return self.pool._rows_view(self.pool.conv[layer], self._rows)

    def set_conv_state(self, layer: int, state) -> None:
        """Validate like ``Qwen35State`` and ``copy_`` into the pool rows.

        The conv state follows the model dtype, and the pool has exactly one dtype, so a
        tensor of another dtype is rejected rather than silently cast (the base setter
        records the dtype as given because it stores the tensor itself; here the storage
        already exists).
        """
        self._check_layer(layer)
        self._check_linear_kind(layer, "conv_state")
        conv_dim, _, _, _, kernel = _linear_dims(self.config)
        expected = (self.batch, conv_dim, kernel - 1)
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"conv_state for layer {layer} must be a Tensor, got "
                f"{type(state).__name__}"
            )
        if tuple(state.shape) != expected:
            raise ValueError(
                f"conv_state for layer {layer} must be (batch, conv_dim, k - 1) = "
                f"{expected}, got {tuple(state.shape)}"
            )
        if state.dtype != self.pool.dtype:
            raise ValueError(
                f"conv_state for layer {layer} must use the pool/model dtype "
                f"{self.pool.dtype}, got {state.dtype}"
            )
        self._check_live()
        self.pool._write_rows(self.pool.conv[layer], self._rows, state)
        for row in self._rows:
            self.pool._conv_ready[layer][row] = True

    def recurrent_state(self, layer: int) -> torch.Tensor | None:
        self._check_layer(layer)
        self._check_linear_kind(layer, "recurrent_state")
        self._check_live()
        if not self._layer_ready(self.pool._recurrent_ready, layer, "recurrent_state"):
            return None
        return self.pool._rows_view(self.pool.recurrent[layer], self._rows)

    def set_recurrent_state(self, layer: int, state) -> None:
        """Validate the fp32 delta-rule contract and ``copy_`` into the pool rows."""
        self._check_layer(layer)
        self._check_linear_kind(layer, "recurrent_state")
        _, num_v_heads, head_k_dim, head_v_dim, _ = _linear_dims(self.config)
        expected = (self.batch, num_v_heads, head_k_dim, head_v_dim)
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"recurrent_state for layer {layer} must be a Tensor, got "
                f"{type(state).__name__}"
            )
        if tuple(state.shape) != expected:
            raise ValueError(
                f"recurrent_state for layer {layer} must be "
                f"(batch, num_v_heads, K, V) = {expected}, got {tuple(state.shape)}"
            )
        if state.dtype != torch.float32:
            raise ValueError(
                f"recurrent_state for layer {layer} must be float32 (the delta rule's "
                f"contract, HF behaviour), got {state.dtype}"
            )
        self._check_live()
        self.pool._write_rows(self.pool.recurrent[layer], self._rows, state)
        for row in self._rows:
            self.pool._recurrent_ready[layer][row] = True

    # -------------------------------------------------------------- bookkeeping

    def cull(self, keep: torch.Tensor) -> None:
        """Shrink to the batch rows in ``keep`` (same indices/order as the base class).

        ``position`` is not rewound and no pool slot is released: this only changes
        which rows this state object presents. The injected KV cache is culled in place
        exactly like the base class.
        """
        keep = self._checked_keep(keep)
        self._check_live()
        keep_list = [int(value) for value in keep.tolist()]
        self._kv.cull(keep.to(self.pool.device))
        self._rows = [self._rows[index] for index in keep_list]
        self._row_epochs = [self.pool._slot_epoch[row] for row in self._rows]
        self.batch = len(self._rows)

    def linear_state_bytes(self) -> int:
        """Bytes this state's rows occupy, counted from the pool's real tensors.

        A layer's rows count only once a setter has written them, matching
        ``Qwen35State.linear_state_bytes`` (which skips unset layers).
        """
        self._check_live()
        total = 0
        for layer in self.pool.linear_layers:
            if self._layer_ready(self.pool._conv_ready, layer, "conv_state"):
                total += self.batch * self.pool.conv_row_bytes(layer)
            if self._layer_ready(
                self.pool._recurrent_ready, layer, "recurrent_state"
            ):
                total += self.batch * self.pool.recurrent_row_bytes(layer)
        return total
