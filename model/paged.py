"""W4 paged KV: a physical block pool and the decode container built on it.

The W3 slab (``model/slab.py``) reserves ``max_len`` columns per row, so one long request
pins the whole row width. Paged KV reserves ``ceil(tokens / block_size)`` physical blocks
per request instead and maps logical columns to physical ones through a **block table** --
a plain Python list ``block_table[j // bs]`` -> physical block id, column ``j % bs``
(docs/w4-plan.md §2, D6). That mapping is the whole point: the same bytes, allocated per
request's own bound, and no KV is ever moved after allocation.

What lives here
---------------
``PagedPool``
    The tensor side. One ``(n_blocks, block_size, n_kv_head, head_dim)`` K and V tensor per
    layer, a dedicated read-only **zero block** for padding, and ONE gather workspace pair
    for the current ``lpad`` bucket, sized to the high-water active-row count and reused
    every step.

``PagedKV``
    The decode container ``model/layers.py`` / ``model/minillm.py`` consume. It owns the
    per-row block tables, ``pos``/``valid``/``active``, the per-step bool mask and additive
    bias, and the seat -> batch-row map. It is *not* a drop-in for growing ``KVcache``: it is
    decode-only, exactly like ``KvSlab``.

Decode step order (the contract ``Attention`` and ``MiniLLM`` rely on)
----------------------------------------------------------------------
``MiniLLM.forward`` calls, once per step, in this order::

    paged.build_mask(next_token=True)     # picks lpad := bucket(max(valid) + 1, 64),
                                          # refreshes mask[:lpad] and bias[:lpad] in place
    for layer in layers:                  # Attention.forward, once per layer
        paged.write(layer, k, v)          # scatter this step's token at pos[row]
        k_ws, v_ws = paged.get(layer)     # gather block tables -> (B, H_kv, lpad, head_dim)
        backend_decode.decode(..., bias=paged.bias)   # one hand_baddbmm per layer (D4)
    paged.advance()                       # pos += 1, valid := max(valid, pos)

``get`` may only be called after ``write`` for the current step (it includes the token just
written). Every layer's gather uses the *same* ``lpad`` and the same logical index, because
the block mapping is layer independent; only the physical pool differs per layer. The
workspace pair is reused across layers inside a step (attention consumes it before the next
layer's gather) and across steps that keep the same ``lpad``, so ``workspace_mb()`` is the
**current** bucket's pair, not a sum over buckets: changing ``lpad`` releases the old pair
before the new one is allocated (the run's 26 buckets cost one pair, never 26). Rows are
sized to the high-water active row count; ``max_rows`` is only a hard cap on the batch, never
the allocation size (plan appendix C).

Nothing here slices, pads along the sequence axis, or changes the attention math: gather
produces a dense ``(B, H_kv, lpad, D)`` tensor and the existing ``hand_baddbmm`` mask/bias
contract applies unchanged (plan appendix B). There is deliberately no paged-attention
kernel and no prefix cache (plan §9).

Correctness never relies on freshly allocated blocks being zero. A block that a shorter
request reuses still holds the previous request's bytes; every position past a row's length
is overwritten from the **dedicated zero block**, and the bool mask closes those columns.
"""

from __future__ import annotations

import time

import torch

from model.attention.hand_baddbmm import additive_bias

__all__ = ["PagedPool", "PagedKV", "LPAD_GRANULARITY"]

#: ``lpad`` is rounded up to this many columns, matching the slab's column bucket (P-25).
LPAD_GRANULARITY = 64


def _bucket(value: int, granularity: int = LPAD_GRANULARITY) -> int:
    """Round ``value`` up to a bucket, never below one column."""
    value = max(int(value), 1)
    return -(-value // granularity) * granularity


def _check_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


class PagedPool:
    """Physical KV blocks: ``(n_blocks, block_size, n_kv_head, head_dim)`` K and V per layer.

    The tensors are allocated once and never replaced, so addresses are stable for the run.
    ``zero_block`` is a dedicated, shared, read-only ``(block_size, n_kv_head, head_dim)``
    tile used only as a *source* when padding a gathered row. It is not one of the pools
    blocks, is not reachable through the block allocator, and is excluded from ``bytes()``.
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
        max_rows: int | None = None,
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
        self.values = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.n_layer)]        # Dedicated zero tile. Shared by every padding column of every row and layer.
        self.zero_block = torch.zeros(
            (self.block_size, self.n_kv_head, self.head_dim), dtype=dtype, device=self.device
        )

        if max_rows is not None:
            _check_positive("max_rows", max_rows)
        self.max_rows = max_rows

        # Gather workspace: at most ONE ``(k, v)`` pair for the current ``lpad`` bucket,
        # shaped ``(rows_cap, n_kv_head, lpad, head_dim)``. Switching buckets (or growing
        # the batch inside a bucket) releases the old pair before allocating the new one;
        # every step of one bucket reuses it, so addresses are stable between rebuilds.
        # ``max_rows`` is only a hard cap on ``n_rows`` -- never the allocation size.
        self._workspace: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._rows_cap = 0
        self._current_lpad: int | None = None
        self._workspace_peak_bytes = 0
        # Diagnostic: how many times the pair was (re)allocated (bucket move or row growth).
        self.workspace_rebuilds = 0
        # The block mapping is layer independent, so the flat index is built once per
        # (block tables, lpad) and reused by every layer (and every step that keeps the
        # same tables) -- the layer axis of ``gather`` never rebuilds it.
        self._index_key: tuple | None = None
        self._index_flat: torch.Tensor | None = None

        # Per-step gather accumulators (plan §5 / appendix C). ``begin_step`` clears them;
        # ``gather`` adds one host-clock sample and one byte count per layer.
        self._step_gather_s = 0.0
        self._step_gather_read_bytes = 0
        self._step_gather_write_bytes = 0
        self._step_gather_calls = 0

    # ------------------------------------------------------------ step account
    def begin_step(self) -> None:
        """Start a new decode step: zero the gather accumulator (call once per step)."""
        self._step_gather_s = 0.0
        self._step_gather_read_bytes = 0
        self._step_gather_write_bytes = 0
        self._step_gather_calls = 0

    def step_stats(self) -> dict:
        """This step's gather account, produced by :meth:`gather` itself.

        ``gather_ms`` is **host wall-clock time** summed over every layer's ``gather``
        call (``time.perf_counter``), not GPU kernel time. On CUDA it includes kernel
        dispatch and any implicit synchronization in the copies, so it is a diagnostic
        upper bound -- the plan's rule is to read ``step_ms`` / ``decode_event_ms`` for
        cost and use this number only to split the step.

        ``gather_read_mb`` / ``gather_write_mb`` are **traffic**, computed from shapes:
        per layer ``n_rows * lpad * n_kv_head * head_dim * itemsize`` bytes read and the
        same written (plan appendix C), summed over every gather call of the step (so
        ``n_layer`` times for a normal step). Traffic is not occupancy: the workspace
        that holds those bytes is reported by :meth:`workspace_mb` and never mixed in.
        """
        return {
            "gather_ms": self._step_gather_s * 1000.0,
            "gather_read_mb": self._step_gather_read_bytes / 2**20,
            "gather_write_mb": self._step_gather_write_bytes / 2**20,
            "gather_calls": self._step_gather_calls,
        }

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

    def write_prefill(
        self, layer: int, block_table, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write ``L`` logical tokens ``[start, start + L)`` through ``block_table``.

        ``k``/``v`` are ``(L, n_kv_head, head_dim)``. The write is segmented per physical
        block (one copy per block, not one per token), so a prefill wipe costs ``ceil(L/bs)``
        copies. ``start`` is the logical index of the first token (0 for a fresh request).
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

        k = k.to(self.dtype)
        v = v.to(self.dtype)
        written = 0
        while written < length:
            assert_pos = start + written
            block_id = block_table[assert_pos // self.block_size]
            offset = assert_pos % self.block_size
            n = min(self.block_size - offset, length - written)
            self.keys[layer][block_id, offset : offset + n] = k[written : written + n]
            self.values[layer][block_id, offset : offset + n] = v[written : written + n]
            written += n

    def write_token(
        self,
        layer: int,
        rows,
        block_tables,
        positions,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter one token per row at its logical ``position``.

        ``rows``/``positions`` are Python lists (one entry per row), ``block_tables`` is the
        per-row logical -> physical mapping, and ``k``/``v`` are ``(n_rows, n_kv_head,
        head_dim)``. All rows are written with one advanced-index assignment.
        """
        layer = self._check_layer(layer)
        n_rows = len(rows)
        if not (len(block_tables) == len(positions) == n_rows):
            raise ValueError(
                "write_token needs one entry per row: "
                f"rows={n_rows}, block_tables={len(block_tables)}, positions={len(positions)}"
            )
        if k.dim() != 3 or tuple(k.shape) != tuple(v.shape):
            raise ValueError(
                f"write_token needs matching (n_rows, n_kv_head, head_dim) k/v, got "
                f"{tuple(k.shape)} / {tuple(v.shape)}"
            )
        if tuple(k.shape[1:]) != (self.n_kv_head, self.head_dim):
            raise ValueError(
                f"write_token needs head shape {(self.n_kv_head, self.head_dim)}, "
                f"got {tuple(k.shape[1:])}"
            )
        if k.shape[0] != n_rows:
            raise ValueError(f"write_token got {k.shape[0]} k rows for {n_rows} rows")

        block_ids: list[int] = []
        offsets: list[int] = []
        for row, (block_table, position) in enumerate(zip(block_tables, positions)):
            position = int(position)
            if position < 0:
                raise ValueError(f"row {row}: position must be >= 0, got {position}")
            block_index = position // self.block_size
            if block_index >= len(block_table):
                raise IndexError(
                    f"row {row}: position {position} needs block {block_index}, but its "
                    f"block table has {len(block_table)} blocks"
                )
            block_id = block_table[block_index]
            self._check_block_ids([block_id], 1)
            block_ids.append(block_id)
            offsets.append(position % self.block_size)

        self.keys[layer][block_ids, offsets] = k.to(self.dtype)
        self.values[layer][block_ids, offsets] = v.to(self.dtype)

    # ------------------------------------------------------------------ gather
    def _workspace_for(self, lpad: int, n_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the one workspace pair for ``lpad``, allocating only when it must grow.

        Lifecycle (this is what keeps the workspace inside the budget):

        * at most one ``lpad`` bucket exists -- moving to a new bucket drops the old pair
          *before* the new one is allocated, so the peak is one pair, not the sum over
          every bucket the run has visited;
        * rows are sized to the high-water active row count **exactly**, so
          ``workspace_mb()`` equals the appendix-C formula (``2 * n_active * lpad *
          n_kv_head * head_dim * itemsize``); the pair is rebuilt only when ``n_rows``
          exceeds that high-water mark, and a step at the same (bucket, row cap) reuses
          the same addresses;
        * ``max_rows`` is a hard cap, not the allocation size: exceeding it means the
          caller's batch is wider than the run's concurrency bound, so it raises.
        """
        n_rows = int(n_rows)
        if n_rows < 1:
            raise ValueError("workspace needs at least one row")
        if self.max_rows is not None and n_rows > self.max_rows:
            raise ValueError(
                f"gather needs {n_rows} rows, but the pool's max_rows cap is {self.max_rows}"
            )
        if self._current_lpad != lpad or not self._workspace or n_rows > self._rows_cap:
            # Release the outgoing pair first: the peak is one bucket, not old + new.
            self._workspace.clear()
            self._current_lpad = lpad
            shape = (n_rows, self.n_kv_head, lpad, self.head_dim)
            self._workspace[lpad] = (
                torch.zeros(shape, dtype=self.dtype, device=self.device),
                torch.zeros(shape, dtype=self.dtype, device=self.device),
            )
            self._rows_cap = n_rows
            self.workspace_rebuilds += 1
            self._workspace_peak_bytes = max(
                self._workspace_peak_bytes, self._workspace_bytes()
            )
        return self._workspace[lpad]

    def _flat_index(self, block_tables, lpad: int) -> torch.Tensor:
        """``(n_rows, lpad)`` flat slot ids ``block_id * block_size + offset``.

        Built with tensor ops (no per-column Python loop) and cached per (block tables,
        lpad), so all layers -- and every step that does not change the block tables --
        reuse it.
        """
        key = (lpad, tuple(tuple(bt) for bt in block_tables))
        if key == self._index_key and self._index_flat is not None:
            return self._index_flat
        n_rows = len(block_tables)
        width = max((len(bt) for bt in block_tables), default=1)
        table = torch.zeros((n_rows, max(width, 1)), dtype=torch.long, device=self.device)
        for row, block_table in enumerate(block_tables):
            if block_table:
                self._check_block_ids(block_table, 1)
                table[row, : len(block_table)] = torch.tensor(block_table, dtype=torch.long)
        columns = torch.arange(lpad, device=self.device).view(1, lpad)
        block_index = torch.clamp(columns // self.block_size, max=table.shape[1] - 1)
        block_ids = torch.gather(table, 1, block_index.expand(n_rows, lpad))
        flat = block_ids * self.block_size + (columns % self.block_size)
        # Padded columns (>= the row's length) still index the pool here; ``gather``
        # overwrites them from the zero block, so their value never reaches a live column.
        self._index_key = key
        self._index_flat = flat
        return flat

    def gather(self, layer: int, block_tables, lengths, lpad: int):
        """Materialize ``(n_rows, n_kv_head, lpad, head_dim)`` K/V views of the pool.

        ``lengths[row]`` is the number of live logical tokens of that row; every column
        ``>= lengths[row]`` is filled from the dedicated zero block (never from the pool,
        and never by "clearing" anything). ``lpad`` must cover the longest row and must
        match the caller's mask width -- the caller (``PagedKV``) owns that invariant.

        The returned tensors are views of the workspace pair for the current ``lpad``
        bucket, so repeated steps at the same bucket and row count do not allocate; a
        bucket change or a wider batch rebuilds the pair (see :meth:`_workspace_for`).

        Each call also feeds the per-step accumulator read by :meth:`step_stats`: one
        ``perf_counter`` sample around the materialization (host time, no
        ``cuda.synchronize`` -- the dispatch cost on CUDA is part of the number and that is
        intended) and the shape-derived traffic of this layer's k and v.
        """
        layer = self._check_layer(layer)
        n_rows = len(block_tables)
        if n_rows == 0:
            raise ValueError("gather needs at least one row")
        if len(lengths) != n_rows:
            raise ValueError(f"gather got {len(lengths)} lengths for {n_rows} block tables")
        lengths = [int(n) for n in lengths]
        if lpad < max(lengths):
            raise ValueError(
                f"lpad {lpad} does not cover the longest row ({max(lengths)}); gather columns "
                "must be >= every row's length"
            )
        for row, (block_table, length) in enumerate(zip(block_tables, lengths)):
            if length < 0 or length > lpad:
                raise ValueError(f"row {row}: length {length} outside [0, {lpad}]")
            needed = (max(length, 1) + self.block_size - 1) // self.block_size
            if len(block_table) < needed:
                raise IndexError(
                    f"row {row}: {length} tokens need {needed} blocks, block table has "
                    f"{len(block_table)}"
                )

        started = time.perf_counter()
        flat = self._flat_index(block_tables, lpad)
        workspace_k, workspace_v = self._workspace_for(lpad, n_rows)
        slot_shape = (n_rows, lpad, self.n_kv_head, self.head_dim)
        for pool_tensor, workspace in ((self.keys[layer], workspace_k),
                                       (self.values[layer], workspace_v)):
            pooled = pool_tensor.reshape(self.n_blocks * self.block_size,
                                         self.n_kv_head, self.head_dim)
            selected = pooled.index_select(0, flat.reshape(-1)).view(slot_shape)
            workspace[:n_rows].copy_(selected.permute(0, 2, 1, 3))
            for row, length in enumerate(lengths):
                if length < lpad:
                    # Dedicated zero block as the *source*: a reused block's stale bytes
                    # never reach a padded column, and this is not an allocation-time clear.
                    workspace[row, :, length:lpad, :] = self.zero_block[:1].permute(1, 0, 2)

        # Traffic (plan appendix C): k + v, every row and every lpad column, once per
        # layer. A gather reads the pool and writes the workspace -- equal byte counts.
        layer_bytes = n_rows * lpad * self.n_kv_head * self.head_dim * self.dtype.itemsize
        self._step_gather_s += time.perf_counter() - started
        self._step_gather_read_bytes += 2 * layer_bytes
        self._step_gather_write_bytes += 2 * layer_bytes
        self._step_gather_calls += 1

        return workspace_k[:n_rows], workspace_v[:n_rows]

    # ------------------------------------------------------------------ accounting
    def bytes(self) -> int:
        """Physical KV bytes of the pool (k + v, all layers); excludes the zero block."""
        return (
            2 * self.n_layer * self.n_blocks * self.block_size * self.n_kv_head
            * self.head_dim * self.dtype.itemsize
        )

    def mb(self) -> float:
        return self.bytes() / 2**20

    def workspace_mb(self) -> float:
        """Bytes of the gather workspace **currently resident** (at most one lpad bucket)."""
        return self._workspace_bytes() / 2**20

    def _workspace_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for pair in self._workspace.values()
            for tensor in pair
        )

    def workspace_peak_mb(self) -> float:
        """High-water mark of ``workspace_mb`` (diagnostic only; not part of the budget)."""
        return max(self._workspace_peak_bytes, self._workspace_bytes()) / 2**20

    def reset_workspace(self) -> None:
        """Drop the cached workspace/index; the next gather allocates afresh."""
        self._workspace.clear()
        self._current_lpad = None
        self._rows_cap = 0
        self._index_key = None
        self._index_flat = None


class PagedKV:
    """Decode-time container over a :class:`PagedPool` (the ``KvSlab`` protocol).

    Rows are the *active* requests only (the engine compresses its seats into this batch;
    ``seats[row]`` is the seat id, and ``row_of_seat`` is the reverse map). There is no
    reserved-but-idle row: an idle request simply is not in ``block_tables``.

    Per-row state (``pos``, ``valid``, ``active``) is a device tensor updated in place, with
    Python mirrors for the eager path -- the same design as ``KvSlab``. ``max_len`` is the
    **run's position bound** (``rope_len``: prompt + budget, or the model's position limit),
    used by the model to size the RoPE table once; it is *not* ``lpad`` and not
    ``n_blocks * block_size`` (P-34).
    """

    #: Explicit protocol marker: ``model/layers.py`` and ``model/minillm.py`` route containers
    #: carrying this attribute down the decode path instead of the growing-``KVcache`` path.
    decode_container = True

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
        self.bucket = LPAD_GRANULARITY

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

        # Per-step mask/bias buffers: ONE pair for the current ``lpad`` bucket, refreshed in
        # place. Switching buckets releases the old pair (the engine may also install its
        # own pair through ``set_buffers``), so buckets never accumulate.
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.lpad: int | None = None
        self.mask: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self._step_written = False

    # ---------------------------------------------------------------- buffered state
    def _buffers_for(self, lpad: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self._buffers.get(lpad)
        if entry is not None and (
            entry[0].shape[0] < self.batch
            or entry[0].shape[-1] != lpad
            or entry[1].shape[0] < self.batch * self.n_kv_head
        ):
            entry = None
        if entry is None:
            # Release the outgoing bucket *before* allocating: peak = one pair, not two.
            self._buffers.clear()
            row_cap = self.batch
            mask = torch.zeros(row_cap, 1, 1, lpad, dtype=torch.bool, device=self.device)
            bias = torch.zeros(
                row_cap * self.n_kv_head, self.n_group, lpad,
                dtype=self.dtype, device=self.device,
            )
            entry = (mask, bias)
            self._buffers[lpad] = entry
        return entry

    def set_buffers(self, lpad: int, mask: torch.Tensor, bias: torch.Tensor) -> None:
        """Install externally owned ``(mask, bias)`` for ``lpad``; drops every other bucket.

        ``serve/engine.py`` allocates one pair per run and hands the current step's views in
        here, so the container does not pay a second pair of buffers.
        """
        if tuple(mask.shape) != (self.batch, 1, 1, lpad):
            raise ValueError(f"mask must be {(self.batch, 1, 1, lpad)}, got {tuple(mask.shape)}")
        if bias.shape[-1] != lpad or bias.shape[0] < self.batch * self.n_kv_head:
            raise ValueError(
                f"bias must cover {self.batch * self.n_kv_head} rows x {lpad} columns, "
                f"got {tuple(bias.shape)}"
            )
        self._buffers.clear()
        self._buffers[lpad] = (mask, bias)

    # ---------------------------------------------------------------- step hooks
    def build_mask(self, *, next_token: bool = False) -> torch.Tensor:
        """Step boundary: choose ``lpad`` and refresh the mask/bias for it.

        ``lpad = bucket(max(valid) + next_token, 64)``; it is the gather column count, the
        mask width and the bias width for this step, and is always >= the longest row.
        ``next_token=True`` is the decode convention: the column about to be written is
        already allowed.
        """
        counts = [length + int(next_token) for length in self._host_valid]
        longest = max(counts)
        if self.check_bounds and longest > self.max_len:
            raise IndexError(
                f"PagedKV overflow: needs {longest} positions, run position bound is "
                f"{self.max_len}"
            )
        # ``lpad`` may exceed ``max_len`` (it is a bucket of what the rows need, and the
        # padded columns are masked); it is not the RoPE table length (P-34).
        lpad = _bucket(longest, self.bucket)
        buffer_mask, buffer_bias = self._buffers_for(lpad)
        # Views of the batch's rows only: the backing pair may be padded up to the row
        # granularity, but mask/bias must describe exactly this batch.
        mask = buffer_mask[: self.batch]
        bias = buffer_bias[: self.batch * self.n_kv_head]

        positions = torch.arange(lpad, device=self.device).view(1, 1, 1, lpad)
        limit = (self.valid + int(next_token)) * self.active
        allowed = positions < limit.view(-1, 1, 1, 1)
        mask.copy_(allowed)
        bias.copy_(additive_bias(mask, self.dtype, self.batch, self.n_kv_head, self.n_group))

        self.lpad = lpad
        self.mask = mask
        self.bias = bias
        self._step_written = False
        return mask

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
        self.pool.write_token(
            layer,
            list(range(self.batch)),
            self.block_tables,
            list(self._host_pos),
            k[:, :, 0, :],
            v[:, :, 0, :],
        )
        self._step_written = True

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this layer's ``(batch, n_kv_head, lpad, head_dim)`` K/V workspace view.

        Uses ``pos + 1`` (host mirror) columns per row -- the token just written -- so
        ``write`` must have run for this step already. Valid columns come from the pool;
        padding comes from the zero block; the mask built in ``build_mask`` closes the
        padding columns.
        """
        if self.lpad is None:
            raise RuntimeError("PagedKV.get() before build_mask(): the step has no lpad yet")
        if not self._step_written:
            raise RuntimeError("PagedKV.get() before write(): this step has no token yet")
        lengths = [position + 1 for position in self._host_pos]
        return self.pool.gather(layer, self.block_tables, lengths, self.lpad)

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
        if self.check_bounds and max(self._host_pos) > self.max_len:
            raise IndexError(
                f"PagedKV overflow: max_len={self.max_len}; a request decoded past its budget"
            )

    # ---------------------------------------------------------------- helpers
    def past_len(self) -> int:
        """Longest row's filled length (``KVcache.past_len`` compatibility)."""
        return max(self._host_valid) if self._host_valid else 0

    def bytes(self) -> int:
        """KV bytes held by this container's backing pool."""
        return self.pool.bytes()

    def mb(self) -> float:
        return self.bytes() / 2**20
