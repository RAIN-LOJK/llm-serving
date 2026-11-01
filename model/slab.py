"""Fixed-size KV buffers for batched decode (the "slab" layout).

Why this exists
---------------
``model.KVcache`` grows with ``torch.cat``: every decode step reallocates the whole
history for every layer. That is (a) an O(context) copy per step and (b) a *new address*
per step, which makes ``torch.cuda.graph`` impossible. A slab fixes both: one allocation
per layer, a write cursor, stable addresses.

This lives in ``model/`` (next to ``KVcache``) because ``serve/`` may depend on ``model/``
but never the other way round; the scheduler-side block manager that will page over it is
W4's job and will keep this write/read contract.

What the caller must guarantee
------------------------------
* **One step boundary before writing.** ``build_mask`` (called once per forward by
  ``MiniLLM``) is the step hook: it refreshes the mask and the additive bias from the
  cursors. ``write`` then runs once per layer against a broadcast view of ``pos`` -- the
  column is the same for every layer, so recomputing it 28 times per step was pure
  bookkeeping.
* ``write`` happens exactly once per layer per step, with tensors shaped
  ``(batch, n_kv_head, 1, head_dim)``, and ``advance`` runs once after all layers.
* ``pos``/``valid`` advance together in the synchronized case (every row one token per
  step, which is what W2/W3's decode loop does). Per-row positions are supported: the
  write is a ``scatter_`` over a per-row column index, so rows may sit at different
  columns (W3's scheduler admits and retires rows at different times).
* Reading is a *view* over the first ``valid`` columns plus the bool mask and the additive
  bias (the fused-mask form ``hand_baddbmm`` consumes); all are preallocated buffers
  updated in place so a captured graph only has to read them.

Host bookkeeping
----------------
Every bound check used to be ``bool((pos > max_len).any())`` -- a device-to-host sync, and
``write`` paid one per layer (28 syncs/step on Qwen3-0.6B, each draining the pipeline).
The positions are now mirrored in Python lists (``_host_pos`` / ``_host_valid``) and the
default guard compares ints: no ``.item()``, no ``bool(tensor)``, no sync on the normal
path. Overflow is still reported. ``check_bounds="strict"`` reads the tensors back for a
caller that poked ``pos``/``valid`` directly or wants the old exact check;
``check_bounds=False`` is the graph-capture mode (replay never runs this Python anyway).
"""

from __future__ import annotations

import torch

from model.attention.hand_baddbmm import additive_bias

__all__ = ["KvSlab"]


class KvSlab:
    """One fixed KV buffer per layer, written by column instead of concatenated."""

    def __init__(
        self,
        n_layer: int,
        batch: int,
        n_kv_head: int,
        head_dim: int,
        max_len: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        n_head: int | None = None,
        check_bounds: bool | str = True,
        lazy: bool = False,
    ) -> None:
        if min(n_layer, batch, n_kv_head, head_dim, max_len) < 1:
            raise ValueError("KvSlab needs positive n_layer/batch/n_kv_head/head_dim/max_len")
        n_head = n_kv_head if n_head is None else int(n_head)
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be a multiple of n_kv_head ({n_kv_head})")
        self.n_layer = n_layer
        self.batch = batch
        self.n_kv_head = n_kv_head
        self.n_head = n_head
        self.n_group = n_head // n_kv_head  # GQA groups the decode bias broadcasts over
        self.head_dim = head_dim
        self.max_len = max_len
        self.dtype = dtype
        self.device = torch.device(device)
        if check_bounds not in (True, False, "strict"):
            raise ValueError(f"check_bounds must be True, False or 'strict', got {check_bounds!r}")
        self.check_bounds = check_bounds

        # ``lazy=True`` leaves the per-layer buffers unmaterialised until ``ensure_layer``
        # is called. The hand-off is then able to allocate layer l, copy it, and drop the
        # source before layer l+1 exists, so the "same KV twice" overlap is one layer
        # instead of all of them (serve/engine.py; W2-0006 notes).
        self._shape = (batch, n_kv_head, max_len, head_dim)
        if lazy:
            self.keys = [None] * n_layer
            self.values = [None] * n_layer
        else:
            self.keys = [torch.zeros(self._shape, dtype=dtype, device=self.device) for _ in range(n_layer)]
            self.values = [torch.zeros(self._shape, dtype=dtype, device=self.device) for _ in range(n_layer)]

        # Positions are tensors so the same objects can be graph inputs; they are updated
        # in place, never replaced. The Python mirrors below are the fast-path guard.
        self.pos = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.valid = torch.zeros(batch, dtype=torch.long, device=self.device)
        self._host_pos = [0] * batch
        self._host_valid = [0] * batch
        # Bool allow-mask (True = may attend), (batch, 1, 1, max_len). In-place updates only.
        self.mask = torch.zeros(batch, 1, 1, max_len, dtype=torch.bool, device=self.device)
        # Additive bias for the decode kernel, (batch*n_kv_head, n_group, max_len), the
        # fused form of the mask. Identical for every layer, so it is built once per step
        # here instead of once per layer in ``hand_baddbmm``.
        self.bias = torch.zeros(
            batch * n_kv_head, self.n_group, max_len, dtype=dtype, device=self.device
        )
        # Column index for the per-layer scatter: a *broadcast view* of ``pos`` reshaped to
        # (batch, n_kv_head, 1, head_dim). It aliases ``pos``, so it never needs refreshing
        # per layer (or even per step) and never allocates. ``scatter_`` accepts it.
        self._columns = self.pos.view(-1, 1, 1, 1).expand(batch, n_kv_head, 1, head_dim)
        self._positions = torch.arange(max_len, device=self.device).view(1, 1, 1, max_len)

    # ---------------------------------------------------------------- lifecycle
    def ensure_layer(self, layer: int) -> None:
        """Materialise one layer's buffers if ``lazy=True`` left them absent.

        Idempotent and allocation-free once the layer exists, so the eager path can call it
        per layer and pay nothing. Called by ``serve/engine.py`` during the hand-off, in the
        order ``ensure_layer(l) -> fill_from(l) -> drop source layer l``.
        """
        if self.keys[layer] is None:
            self.keys[layer] = torch.zeros(self._shape, dtype=self.dtype, device=self.device)
            self.values[layer] = torch.zeros(self._shape, dtype=self.dtype, device=self.device)

    def reset(self) -> None:
        """Forget every row (buffers are reused, not freed)."""
        self.pos.zero_()
        self.valid.zero_()
        self.mask.zero_()
        self.bias.zero_()
        self._host_pos = [0] * self.batch
        self._host_valid = [0] * self.batch

    def advance(self) -> None:
        """One token was written for every row: move both cursors (in place)."""
        self.pos += 1
        self.valid.copy_(torch.maximum(self.valid, self.pos))
        self.advance_host()
        if self.check_bounds and self._overflowed_advance_guard():
            raise IndexError(
                f"KvSlab overflow: max_len={self.max_len}; a request decoded past its budget"
            )

    # ---------------------------------------------------------------- io
    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter one token per row into column ``pos[row]`` for one layer.

        The index is a broadcast view of ``pos`` (see ``_columns``): the column is the same
        for every layer within a step, and aliasing ``pos`` means no per-layer (or per-step)
        copy of the index at all.
        """
        expected = (self.batch, self.n_kv_head, 1, self.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"write needs {expected}, got {tuple(k.shape)} / {tuple(v.shape)}")
        if self.check_bounds and self._overflowed_write_guard():
            raise IndexError(
                f"KvSlab overflow: write column {self._guard_pos()} >= max_len={self.max_len}"
            )
        self.keys[layer].scatter_(2, self._columns, k.to(self.dtype))
        self.values[layer].scatter_(2, self._columns, v.to(self.dtype))

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The full buffers (a view, no copy). Callers apply ``mask``/``bias`` themselves."""
        return self.keys[layer], self.values[layer]

    def past_len(self, layer: int | None = None) -> int:
        """Longest row's filled length; ``KVcache.past_len`` compatibility."""
        return int(self.valid.max().item())

    def sync_host(self) -> None:
        """Re-read ``pos``/``valid`` into the Python mirrors (one host sync).

        Only needed by a caller that moved the cursors behind this class's back -- in
        practice ``serve/graph.py``, where a captured step advances the tensors on the
        device without running this Python. The eager path never calls it.
        """
        self._host_pos = [int(v) for v in self.pos.tolist()]
        self._host_valid = [int(v) for v in self.valid.tolist()]

    def advance_host(self) -> None:
        """Mirror one already-performed step in Python (a graph replay ran ``advance``).

        The device tensors moved inside the captured graph, so their Python mirrors have to
        follow. One token per row per replay makes this exact and synchronisation-free.
        """
        self._host_pos = [p + 1 for p in self._host_pos]
        self._host_valid = [max(v, p) for v, p in zip(self._host_valid, self._host_pos)]

    # ---------------------------------------------------------------- mask
    def build_mask(self, *, next_token: bool = False) -> torch.Tensor:
        """Step boundary: refresh columns, ``mask`` and ``bias`` in place; return the mask.

        ``next_token=True`` is the decode-step convention: the mask must already allow the
        column this step is about to write (``pos``), because the write happens inside the
        layer loop while the mask is read by the same forward. Called once per forward, so
        everything that depends only on the batch (not the layer) is paid once.
        """
        limit = self.valid + int(next_token) if next_token else self.valid
        if self.check_bounds and self._overflowed_limit_guard(next_token):
            raise IndexError(
                f"KvSlab overflow: needs {self._guard_limit(next_token)} columns, "
                f"max_len={self.max_len}"
            )
        allowed = (self._positions < limit.view(-1, 1, 1, 1)).expand_as(self.mask)
        self.mask.copy_(allowed)
        # Same construction as the backend's mask path, once per step instead of per layer.
        self.bias.copy_(
            additive_bias(self.mask, self.dtype, self.batch, self.n_kv_head, self.n_group)
        )
        return self.mask

    # ---------------------------------------------------------------- bounds
    def _overflowed_write_guard(self) -> bool:
        if self.check_bounds == "strict":
            return bool((self.pos >= self.max_len).any())
        return max(self._host_pos) >= self.max_len

    def _overflowed_advance_guard(self) -> bool:
        # ``advance`` is where the cursor first goes *past* the last column it wrote, so it
        # reports a budget that was spent exactly (``pos > max_len``), like the old check.
        if self.check_bounds == "strict":
            return bool((self.pos > self.max_len).any())
        return max(self._host_pos) > self.max_len

    def _overflowed_limit_guard(self, next_token: bool) -> bool:
        if self.check_bounds == "strict":
            limit = self.valid + int(next_token) if next_token else self.valid
            return bool((limit > self.max_len).any())
        return max(self._host_valid) + int(next_token) > self.max_len

    def _guard_pos(self) -> int:
        """Value for the overflow message; only strict mode reads the tensor back."""
        if self.check_bounds == "strict":
            return int(self.pos.max())
        return max(self._host_pos)

    def _guard_limit(self, next_token: bool) -> int:
        if self.check_bounds == "strict":
            limit = self.valid + int(next_token) if next_token else self.valid
            return int(limit.max())
        return max(self._host_valid) + int(next_token)

    def fill_from(self, cache, layer: int, row: int | None = None, length: int | None = None) -> None:
        """Seed the slab from a prefill that used ``model.KVcache`` (one-time copy).

        Prefill keeps its flash path; the slab only takes over the decode loop, so the
        hand-off is an explicit copy instead of a second prefill implementation.

        ``row=None`` copies every row and sets the whole cursor, which is the single-request
        hand-off. ``row=int`` copies only that row from a batch-1 cache and updates
        ``pos[row]`` / ``valid[row]``, so rows admitted with different prompt lengths do not
        overwrite each other's cursor (P-26: one seat per row).
        """
        past_k, past_v = cache.get_kv(layer)
        if past_k is None:
            raise ValueError(f"layer {layer}: the cache is empty, nothing to copy")
        length = int(past_k.shape[2]) if length is None else int(length)
        if length > self.max_len:
            raise IndexError(f"prefill length {length} exceeds slab max_len {self.max_len}")
        if row is None:
            self.keys[layer][:, :, :length] = past_k[:, :, :length].to(self.dtype)
            self.values[layer][:, :, :length] = past_v[:, :, :length].to(self.dtype)
            self.pos.fill_(length)
            self.valid.fill_(length)
            self._host_pos = [length] * self.batch
            self._host_valid = [length] * self.batch
            return
        if not 0 <= row < self.batch:
            raise IndexError(f"row {row} outside the slab's batch of {self.batch}")
        if past_k.shape[0] != 1:
            raise ValueError(
                f"fill_from(row={row}) needs a batch-1 cache (one prefill per row), got "
                f"batch {past_k.shape[0]}"
            )
        self.keys[layer][row, :, :length] = past_k[0, :, :length].to(self.dtype)
        self.values[layer][row, :, :length] = past_v[0, :, :length].to(self.dtype)
        self.pos[row] = length
        self.valid[row] = length
        self._host_pos[row] = length
        self._host_valid[row] = length

    # ---------------------------------------------------------------- accounting
    def bytes(self) -> int:
        """KV bytes held, matching the W0/W2 formula account (k + v, all layers)."""
        return 2 * self.n_layer * self.batch * self.n_kv_head * self.max_len * self.head_dim * self.dtype.itemsize

    def mb(self) -> float:
        return self.bytes() / 2**20
