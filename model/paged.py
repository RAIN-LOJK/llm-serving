"""W4 paged KV: a physical block pool plus the W4.5 block-table decode container.

The W3 slab (``model/slab.py``) reserves ``max_len`` columns per row, so one long request
pins the whole row width. Paged KV reserves ``ceil(tokens / block_size)`` physical blocks
per request instead and maps logical columns to physical ones through a **block table** --
a plain Python list ``block_table[j // bs]`` -> physical block id, column ``j % bs``
(docs/decisions.md D6). That mapping is the whole point: the same bytes, allocated per
request's own bound, and no KV is ever moved after allocation.

What lives here
---------------
``PagedPool``
    The tensor side. One ``(n_blocks, block_size, n_kv_head, head_dim)`` K and V tensor per
    layer, the flat views a slot id indexes, and the slot-map builders prefill uses. It
    stores bytes; it does not decode.

``PagedKV``
    The decode container ``model/layers.py`` / ``model/minillm.py`` consume. It owns the
    per-row block tables, ``pos``/``valid``/``active``, the int32 block table and seqlen the
    Triton kernel reads, and the seat -> batch-row map. It is *not* a drop-in for growing
    ``KVcache``: it is decode-only, exactly like ``KvSlab``.

Decode is Triton-only
---------------------
``build_mask`` refreshes the int32 ``seqlen`` vector; ``Attention`` calls ``attend`` once
per layer and ``model/triton_decode.py`` reads K/V directly from the physical blocks. There
is no dense gather, no ``lpad`` bucket, no zero block and no mask/bias pair in the
production path: W4's gather control arm was deleted in W4.5 step 2 (docs/decisions.md D7).
The slow dense oracle now lives in ``tests/reference/paged.py`` and is imported by tests
only -- production keeps the kernel.

A paged container requires CUDA and Triton. The gather arm used to be the CPU/MPS path; it
was deleted rather than kept as a fallback, so constructing a container on CPU raises
immediately instead of silently gathering. The seat arm (``KvSlab`` + ``hand_baddbmm``)
remains the CPU/MPS decode path.

Decode step order (the contract ``Attention`` and ``MiniLLM`` rely on)
----------------------------------------------------------------------
``MiniLLM.forward`` calls, once per step, in this order::

    paged.build_mask(next_token=True)     # seqlen := pos + 1 (int32), refreshed in place
    for layer in layers:                  # Attention.forward, once per layer
        paged.write(layer, k, v)          # scatter this step's token at pos[row]
        y = paged.attend(layer, q)        # Triton block-table kernel reads the pool
    paged.advance()                       # pos += 1, valid := max(valid, pos)

The block mapping is layer independent, so every layer reads the same int32 block table;
only the physical pool differs per layer. Correctness never relies on freshly allocated
blocks being zero: a block that a shorter request reuses still holds the previous request's
bytes, and the kernel masks the last page with ``column < seqlen[row]``, so bytes past a
row's length never reach the softmax (P-42).
"""

from __future__ import annotations

import torch

from model.triton_decode import TRITON_AVAILABLE

__all__ = ["PagedPool", "PagedKV"]


def _check_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Whether two device handles name the same physical device.

    ``torch.device("cuda")`` (the pool's handle when it is built with ``device="cuda"``)
    and the ``cuda:0`` a tensor reports are the same device -- the first means "the
    current one" and the tensors were allocated on it. Comparing the handles directly
    would reject a pool's own slot map, so the index is compared only when both sides
    name one explicitly. Type mismatches (cpu vs cuda) and different GPU indices still
    fail, so this cannot hide a cross-device mistake.
    """
    return a.type == b.type and (a.index is None or b.index is None or a.index == b.index)


class PagedPool:
    """Physical KV blocks: ``(n_blocks, block_size, n_kv_head, head_dim)`` K and V per layer.

    The tensors are allocated once and never replaced, so addresses are stable for the run.
    A slot id is a row of the flattened ``(n_blocks * block_size, n_kv_head, head_dim)``
    view; prefill and decode both write through :meth:`write_slots`.
    """

    def __init__(
        self,
        n_layer: int,
        n_blocks: int,
        block_size: int,
        n_kv_head: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        n_head: int | None = None,
    ) -> None:
        self.n_layer = _check_positive("n_layer", n_layer)
        self.n_blocks = _check_positive("n_blocks", n_blocks)
        self.block_size = _check_positive("block_size", block_size)
        self.n_kv_head = _check_positive("n_kv_head", n_kv_head)
        self.head_dim = _check_positive("head_dim", head_dim)
        n_head = self.n_kv_head if n_head is None else int(n_head)
        if n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be a multiple of n_kv_head ({self.n_kv_head})")
        self.n_head = n_head
        self.n_group = n_head // self.n_kv_head
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (self.n_blocks, self.block_size, self.n_kv_head, self.head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.n_layer)]
        self.values = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.n_layer)]
        # Flat ``(n_blocks * block_size, n_kv_head, head_dim)`` views of the pools, built
        # once: a slot id is a row of this view. The pool tensors are allocated here and
        # never replaced (only written in place), so the views cannot go stale; rebuilding
        # them per layer cost ~4 us of the 36 us per-layer decode write (W45 profile).
        self._keys_flat = [t.view(-1, self.n_kv_head, self.head_dim) for t in self.keys]
        self._values_flat = [t.view(-1, self.n_kv_head, self.head_dim) for t in self.values]

        # One cached 1-D slot tensor per (block table, start, length), so the ``n_layer``
        # ``write_prefill_slots`` calls of one prompt reuse it instead of rebuilding it per
        # layer. The key contains the table, so a new request (or a reused pool) rebuilds
        # it on the first layer and can never read a stale mapping; the cache is one
        # entry, not one per table.
        self._prefill_key: tuple | None = None
        self._prefill_slots: torch.Tensor | None = None

    # ------------------------------------------------------------------ io
    def _check_layer(self, layer: int) -> int:
        if not 0 <= int(layer) < self.n_layer:
            raise IndexError(f"layer {layer} outside the pool's {self.n_layer} layers")
        return int(layer)

    def _check_block_ids(self, block_ids, needed: int) -> None:
        for block_id in block_ids:
            if not isinstance(block_id, int) or isinstance(block_id, bool):
                raise ValueError(f"block ids must be ints, got {block_id!r}")
            if not 0 <= block_id < self.n_blocks:
                raise IndexError(f"block id {block_id} outside the pool's {self.n_blocks} blocks")
        if len(block_ids) < needed:
            raise IndexError(
                f"block table has {len(block_ids)} blocks, needs {needed} to address this write"
            )

    def _slot_ids(self, block_table, start: int, length: int) -> torch.Tensor:
        """1-D ``(length,)`` flat slot ids for logical tokens ``[start, start + length)``.

        Token ``t`` lives at physical slot ``block_table[t // block_size] * block_size +
        (t % block_size)`` -- the row of the pool's flattened
        ``(n_blocks * block_size, n_kv_head, head_dim)`` view. Built with tensor ops (one
        small table tensor plus a gather, no per-token Python loop) and cached per
        ``(block table, start, length)``, so the layer loop of one prefill builds it once.
        """
        key = (tuple(block_table), start, length)
        if key == self._prefill_key and self._prefill_slots is not None:
            return self._prefill_slots
        table = torch.tensor(list(block_table), dtype=torch.long, device=self.device)
        positions = torch.arange(start, start + length, dtype=torch.long, device=self.device)
        block_index = positions // self.block_size
        slots = table[block_index] * self.block_size + (positions % self.block_size)
        self._prefill_key = key
        self._prefill_slots = slots
        return slots

    def slot_ids(self, block_table, start: int, length: int) -> torch.Tensor:
        """Public slot-map builder for a caller that writes the same map every layer.

        A packed prefill writes one request per sink; building the map inside every
        layer would rebuild it 28 times per request and thrash the one-entry cache.
        The engine builds it once per request and hands it to ``write_prefill_slots``.
        """
        start = int(start)
        length = int(length)
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        self._check_block_ids(
            block_table, (start + length + self.block_size - 1) // self.block_size
        )
        return self._slot_ids(block_table, start, length)

    def write_slots(
        self, layer: int, slot_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write precomputed flat slots: one device-indexed copy per K/V per layer.

        The primitive behind both hot paths -- packed prefill (:meth:`write_prefill`,
        once per prompt) and decode (:meth:`PagedKV.write`, once per layer per step).
        ``slot_ids`` is a 1-D device map whose entry ``i`` is the flat pool row
        ``block_id * block_size + offset`` of the ``i``-th token; the caller owns the map
        and must rebuild it whenever its block table or position changes. Every entry
        must be distinct (a repeated slot makes the winner undefined) and inside the
        pool; :meth:`slot_ids` and :class:`PagedKV` are the two callers that guarantee it.

        ``index_copy_`` rather than ``keys[slot_ids] = ...``: both scatter the same rows,
        but the decode write of one step costs 83 us of host time per layer in the
        list-indexed form against 17 us for the two copies here, and the prefill form is
        already per-prompt. No ``int(slot_ids.min())`` host scan: that would be a
        device-to-host sync per layer (P-46).
        """
        layer = self._check_layer(layer)
        if k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"write_slots needs (length, n_kv_head, head_dim), got "
                f"{tuple(k.shape)} / {tuple(v.shape)}"
            )
        if tuple(k.shape) != tuple(v.shape):
            raise ValueError(f"k/v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.shape[1:] != (self.n_kv_head, self.head_dim):
            raise ValueError(
                f"write_slots needs head shape {(self.n_kv_head, self.head_dim)}, "
                f"got {tuple(k.shape[1:])}"
            )
        if slot_ids.dim() != 1 or slot_ids.numel() != k.shape[0]:
            raise ValueError(
                f"slot_ids must be 1-D with one entry per token ({k.shape[0]}), got "
                f"{tuple(slot_ids.shape)}"
            )
        if slot_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"slot_ids must be an integer tensor, got {slot_ids.dtype}")
        if not _same_device(slot_ids.device, self.device):
            raise ValueError(
                f"slot_ids on {slot_ids.device} but the pool is on {self.device}"
            )
        # ``index_copy_`` requires a long index even though the public contract accepts
        # int32 (the prefill builder returns long; a hand-built map may be int32). Cast
        # only when needed: the decode slot map is already long.
        if slot_ids.dtype != torch.int64:
            slot_ids = slot_ids.to(torch.int64)
        keys = self._keys_flat[layer]
        values = self._values_flat[layer]
        keys.index_copy_(0, slot_ids, k if k.dtype == self.dtype else k.to(self.dtype))
        values.index_copy_(0, slot_ids, v if v.dtype == self.dtype else v.to(self.dtype))

    def write_prefill_slots(
        self, layer: int, slot_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write precomputed flat slots (one copy per K/V per layer).

        Kept as the named prefill entry point (``PagedWriteSink`` calls it) on top of
        :meth:`write_slots`; ``slot_ids`` is the map returned by :meth:`slot_ids`, reused
        by every layer of one prompt.
        """
        self.write_slots(layer, slot_ids, k, v)

    def write_prefill(
        self, layer: int, block_table, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write ``L`` logical tokens ``[start, start + L)`` through ``block_table``.

        ``k``/``v`` are ``(L, n_kv_head, head_dim)`` and token ``t`` maps to physical slot
        ``block_table[t // block_size] * block_size + (t % block_size)`` (see
        :meth:`_slot_ids`). The pool tensors are contiguous
        ``(n_blocks, block_size, n_kv_head, head_dim)``, so one advanced-index write per
        layer per K/V places every token.

        That replaces the previous per-block slice loop (``ceil(L/bs)`` slice copies per
        layer per tensor -- ~95 blocks x 28 layers x 2 for a 1500-token prompt, thousands
        of small CUDA copies). The slot ids are shared by every layer of one prefill, and
        the write is a pure copy, so the bytes are identical to the per-block version.

        ``start`` is the logical index of the first token (0 for a fresh request; ``> 0``
        appends a chunk after bytes already written through the same table). All tokens
        must map to *distinct* physical blocks: with a table that repeats a block id the
        advanced-index winner is undefined (the allocator never issues duplicates).
        """
        layer = self._check_layer(layer)
        if k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"write_prefill needs (length, n_kv_head, head_dim), got "
                f"{tuple(k.shape)} / {tuple(v.shape)}"
            )
        if tuple(k.shape) != tuple(v.shape):
            raise ValueError(f"k/v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.shape[1:] != (self.n_kv_head, self.head_dim):
            raise ValueError(
                f"write_prefill needs head shape {(self.n_kv_head, self.head_dim)}, "
                f"got {tuple(k.shape[1:])}"
            )
        start = int(start)
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        length = k.shape[0]
        if length == 0:
            return
        self._check_block_ids(block_table, (start + length + self.block_size - 1) // self.block_size)

        slot_ids = self._slot_ids(block_table, start, length)
        self.write_prefill_slots(layer, slot_ids, k, v)

    # ------------------------------------------------------------------ accounting
    def bytes(self) -> int:
        """Physical KV bytes of the pool (k + v, all layers)."""
        return (
            2 * self.n_layer * self.n_blocks * self.block_size * self.n_kv_head
            * self.head_dim * self.dtype.itemsize
        )

    def mb(self) -> float:
        return self.bytes() / 2**20


class PagedKV:
    """Decode-time container over a :class:`PagedPool` (the ``KvSlab`` protocol).

    Rows are the *active* requests only (the engine compresses its seats into this batch;
    ``seats[row]`` is the seat id, and ``row_of_seat`` is the reverse map). There is no
    reserved-but-idle row: an idle request simply is not in ``block_tables``.

    Per-row state (``pos``, ``valid``, ``active``) is a device tensor updated in place, with
    Python mirrors for the eager path -- the same design as ``KvSlab``. ``max_len`` is the
    **run's position bound** (``rope_len``: prompt + budget, or the model's position limit),
    used by the model to size the RoPE table once; it is *not* ``lpad`` (which no longer
    exists) and not ``n_blocks * block_size`` (P-34).

    Decode is the Triton block-table kernel. There is deliberately no ``get()``,
    ``set_buffers()``, mask/bias or ``decode_impl`` switch: W4's gather control arm was
    deleted, and a second implementation in production would be a fallback target.
    """

    #: Explicit protocol marker: ``model/layers.py`` and ``model/minillm.py`` route containers
    #: carrying this attribute down the decode path instead of the growing-``KVcache`` path.
    decode_container = True
    #: Container-owned block-table decode: ``model/layers.py`` calls :meth:`attend` instead
    #: of ``KvSlab.get`` + a dense backend. It is a constant protocol flag, not a switch.
    block_table_decode = True

    def __init__(
        self,
        pool: PagedPool,
        block_tables,
        lengths,
        *,
        rope_len: int,
        seats=None,
        n_head: int | None = None,
        check_bounds: bool = True,
    ) -> None:
        if not isinstance(pool, PagedPool):
            raise TypeError(f"PagedKV needs a PagedPool, got {type(pool).__name__}")
        if pool.device.type != "cuda":
            raise ValueError(
                "paged decode is the CUDA block-table Triton kernel; the gather control "
                "arm was deleted in W4.5 step 2. "
                f"The pool is on {pool.device}. Run the paged arm on CUDA, or use the "
                "seat arm (KvSlab + hand_baddbmm) on CPU/MPS. The dense numeric oracle "
                "lives in tests/reference/paged.py."
            )
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "paged decode requires Triton, but Triton could not be imported; the "
                "gather control arm was deleted, so there is no eager fallback. The "
                "dense numeric oracle lives in tests/reference/paged.py."
            )
        n_rows = len(block_tables)
        if n_rows < 1:
            raise ValueError("PagedKV needs at least one row")
        if len(lengths) != n_rows:
            raise ValueError(
                f"PagedKV needs one length per block table, got {len(lengths)} for {n_rows}"
            )
        rope_len = int(rope_len)
        if rope_len < 1:
            raise ValueError(f"rope_len (the run's position bound) must be >= 1, got {rope_len}")

        self.pool = pool
        self.n_layer = pool.n_layer
        self.n_kv_head = pool.n_kv_head
        self.head_dim = pool.head_dim
        self.n_head = pool.n_kv_head if n_head is None else int(n_head)
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be a multiple of n_kv_head ({self.n_kv_head})"
            )
        self.n_group = self.n_head // self.n_kv_head
        self.dtype = pool.dtype
        self.device = pool.device
        self.check_bounds = bool(check_bounds)

        self.block_tables = [list(bt) for bt in block_tables]
        self.seats = list(range(n_rows)) if seats is None else list(seats)
        if len(self.seats) != n_rows:
            raise ValueError(f"PagedKV needs one seat per row, got {len(self.seats)} for {n_rows}")
        self.row_of_seat = {seat: row for row, seat in enumerate(self.seats)}
        self.batch = n_rows
        self.max_len = rope_len

        lengths = [int(n) for n in lengths]
        for row, (block_table, length) in enumerate(zip(self.block_tables, lengths)):
            if length < 0:
                raise ValueError(f"row {row}: length must be >= 0, got {length}")
            if length > (len(block_table) * pool.block_size):
                raise IndexError(
                    f"row {row}: length {length} exceeds its block table's capacity "
                    f"({len(block_table)} blocks x {pool.block_size})"
                )
            if length > rope_len:
                raise IndexError(
                    f"row {row}: length {length} exceeds the run's position bound {rope_len}"
                )

        # In-place device state plus host mirrors, exactly like ``KvSlab``.
        self.pos = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.valid = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.active = torch.ones(n_rows, dtype=torch.bool, device=self.device)
        self._host_pos = list(lengths)
        self._host_valid = list(lengths)
        self._host_active = [True] * n_rows

        # One padded int32 block table per container for the whole batch (one row per
        # request, zero-padded past the request's own blocks). The Triton kernel reads it
        # directly, and the decode write path derives its flat slot ids from it, so both
        # use one copy. It is built once per container and never mutated: the block tables
        # are fixed at construction (D6 has no mid-request allocation), so a container
        # rebuild -- not a step -- is what refreshes it.
        max_blocks = max((len(table) for table in self.block_tables), default=1)
        table = torch.zeros((n_rows, max_blocks), dtype=torch.int32, device=self.device)
        for row, block_table in enumerate(self.block_tables):
            if not block_table:
                continue
            # Validate every id once here, not per layer per step: the write path indexes
            # the pool through these ids.
            pool._check_block_ids(
                block_table, (max(lengths[row], 1) + pool.block_size - 1) // pool.block_size
            )
            table[row, : len(block_table)] = torch.tensor(
                block_table, dtype=torch.int32, device=self.device
            )
        self._block_table_tensor = table
        self._block_table_i64 = table.to(torch.int64)
        # The int32 seqlen vector the kernel masks against; ``build_mask`` refreshes it in
        # place every step, and the launch plan holds this same tensor by reference.
        self._seqlen = torch.zeros(n_rows, dtype=torch.int32, device=self.device)
        # Flat pool slot each row writes next, derived from ``pos`` and the table above.
        # It is rebuilt whenever ``pos`` moves (``advance``), so the per-layer write is one
        # device-indexed copy with no Python list building and no host->device index copy.
        self._slot_ids: torch.Tensor | None = None
        self._slot_error: tuple[int, int, int, int] | None = None
        self._refresh_slots()
        # Lazily built launch-metadata cache for ``attend`` (see model/triton_decode.py).
        # It is per container, so a rebuilt container -- the only way the block table can
        # change -- can never reuse the previous one's mapping.
        self._decode_plan = None
        self._step_written = False
        self._seqlen_ready = False

    # ---------------------------------------------------------------- step hooks
    def build_mask(self, *, next_token: bool = False) -> None:
        """Step boundary: refresh the int32 seqlen vector the kernel masks against.

        ``seqlen[row] = pos[row] + next_token``; ``next_token=True`` is the decode
        convention (the column about to be written is already allowed). The name is kept
        because ``MiniLLM.forward`` calls it for every decode container, but a paged
        container has no mask or bias to build: the kernel masks the last page by seqlen.
        """
        counts = [length + int(next_token) for length in self._host_valid]
        longest = max(counts)
        if self.check_bounds and longest > self.max_len:
            raise IndexError(
                f"PagedKV overflow: needs {longest} positions, run position bound is "
                f"{self.max_len}"
            )
        self._seqlen.copy_((self.pos + int(next_token)).to(torch.int32))
        self._step_written = False
        self._seqlen_ready = True
        return None

    def _refresh_slots(self) -> None:
        """Rebuild the per-row flat write slot from ``pos`` and the padded block table.

        Slot ``row = block_table[row][pos // bs] * bs + pos % bs`` -- exactly the mapping
        the old per-row write computed in Python. Here it is one host bounds loop plus
        three small device ops at the step boundary, shared by all 28 layers.

        Invariant: the block table tensor and ``pos`` are the only inputs, and both are
        container-owned (the table is fixed at construction, ``pos`` moves only through
        :meth:`advance`). A container is rebuilt -- never mutated -- when the active set
        changes, so a slot map can never outlive its block table.
        """
        bs = self.pool.block_size
        self._slot_error = None
        for row, position in enumerate(self._host_pos):
            block_index = position // bs
            if block_index >= len(self.block_tables[row]):
                self._slot_error = (row, position, block_index, len(self.block_tables[row]))
                break
        if self._slot_error is not None:
            # No slot map for an out-of-range position: ``write`` raises before touching
            # the pool. A request that ends *on* its last block therefore parks here
            # without an error until a write is actually attempted.
            self._slot_ids = None
            return
        block = torch.div(self.pos, bs, rounding_mode="floor")
        block_id = self._block_table_i64.gather(1, block.view(-1, 1)).squeeze(1)
        self._slot_ids = block_id * bs + (self.pos % bs)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter this step's token into each row at its own logical position."""
        expected = (self.batch, self.n_kv_head, 1, self.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"write needs {expected}, got {tuple(k.shape)} / {tuple(v.shape)}")
        if self.check_bounds and max(self._host_pos) >= self.max_len:
            raise IndexError(
                f"PagedKV overflow: write position {max(self._host_pos)} >= max_len "
                f"{self.max_len}"
            )
        if self._slot_ids is None:
            row, position, block_index, have = self._slot_error or (-1, -1, -1, -1)
            raise IndexError(
                f"row {row}: position {position} needs block {block_index}, but its block "
                f"table has {have} blocks"
            )
        self.pool.write_slots(layer, self._slot_ids, k[:, :, 0, :], v[:, :, 0, :])
        self._step_written = True

    def attend(
        self, layer: int, q: torch.Tensor, *, dropout_p: float = 0.0
    ) -> torch.Tensor:
        """Block-table decode for one layer; ``q`` is ``(B, H, 1, D)``, output the same.

        The Triton kernel reads K/V from the physical pool, one page per logical page of
        the request's block table, and masks the tail of the last page by ``seqlen``.
        """
        if not self._seqlen_ready:
            raise RuntimeError("PagedKV.attend() before build_mask(): seqlen is stale")
        if not self._step_written:
            raise RuntimeError("PagedKV.attend() before write(): this step has no token yet")
        if dropout_p:
            raise ValueError("paged decode is inference-only; dropout_p must be 0")
        expected = (self.batch, self.n_head, 1, self.head_dim)
        if tuple(q.shape) != expected:
            raise ValueError(f"attend needs q {expected}, got {tuple(q.shape)}")
        from model.triton_decode import paged_decode_plan

        q3 = q[:, :, 0, :].contiguous()
        # One plan per container: every layer's pool tensor has the same shape/layout, and
        # the block table and seqlen are container-owned, so the only thing that changes
        # per layer -- the query pointer -- is checked by ``run``. The plan validates the
        # full paged_decode contract for all layers on first use; a container rebuild (new
        # block tables) builds a new plan, which is what keeps this from going stale.
        if self._decode_plan is None:
            self._decode_plan = paged_decode_plan(
                q3,
                self.pool.keys,
                self.pool.values,
                self._block_table_tensor,
                self._seqlen,
                scale=1.0 / (self.head_dim ** 0.5),
            )
        out = self._decode_plan.run(layer, q3)
        return out.unsqueeze(2)

    def advance(self) -> None:
        """Move every active row's cursors one token (in place, then host mirrors)."""
        self.pos.add_(self.active)
        self.valid.copy_(torch.maximum(self.valid, self.pos))
        for row in range(self.batch):
            if not self._host_active[row]:
                continue
            self._host_pos[row] += 1
            self._host_valid[row] = max(self._host_valid[row], self._host_pos[row])
        self._step_written = False
        self._seqlen_ready = False
        if self.check_bounds and max(self._host_pos) > self.max_len:
            raise IndexError(
                f"PagedKV overflow: max_len={self.max_len}; a request decoded past its budget"
            )
        self._refresh_slots()

    # ---------------------------------------------------------------- helpers
    def past_len(self) -> int:
        """Longest row's filled length (``KVcache.past_len`` compatibility)."""
        return max(self._host_valid) if self._host_valid else 0

    def bytes(self) -> int:
        """KV bytes held by this container's backing pool."""
        return self.pool.bytes()

    def mb(self) -> float:
        return self.bytes() / 2**20
