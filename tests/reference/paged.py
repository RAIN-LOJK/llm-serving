"""``paged``: the dense gather/decode oracle for the paged arm, frozen outside production.

W4 shipped the paged decode control arm as a dense gather of the block table followed by
``hand_baddbmm``. W4.5 replaced it with the Triton block-table kernel
(``model/triton_decode.py::paged_decode``); the gather arm was deleted from production in
W4.5 step 2. This module is what is left of it: the slow, obvious implementation that
numeric tests compare the kernel against. Nothing here is imported by ``model/`` or
``serve/`` -- it exists so a kernel bug cannot agree with a single formulated oracle.

Three pieces:

* :func:`gather_pool` -- materialize ``(n_rows, n_kv_head, lpad, head_dim)`` K/V from a
  ``PagedPool`` with plain Python indexing, zero-filling every column past a row's length.
  This is the W4 gather workspace as a test reader: it pins the pool's slot mapping and
  block writes without needing a GPU.
* :func:`write_token_reference` -- the pre-slot-map per-row decode write, one physical
  ``(block, offset)`` at a time. ``PagedKV.write`` uses one device slot map for the whole
  batch; this is the independent mapping it is compared against.
* :func:`reference_paged_decode` -- the dense PyTorch decode oracle moved out of
  ``model/triton_decode.py``. It accepts CPU or CUDA tensors, gathers the row's logical
  K/V with explicit indexing, softmaxes in fp32 and multiplies V. W4 docs call this the
  dense oracle (D7 G2).

Provenance: ``gather_pool``/``write_token_reference`` are the W4 implementations as they
stood before deletion (git ``a4d8443`` for ``PagedPool.gather``; ``PagedPool.write_token``).
``reference_paged_decode`` is the same function that lived in ``model/triton_decode.py``
through W4.5, including its layout validation. Do not optimize or "fix" them: a change here
silently moves the oracle and makes every kernel comparison meaningless.
"""

from __future__ import annotations

import math

import torch

__all__ = ["gather_pool", "write_token_reference", "reference_paged_decode"]


def gather_pool(pool, layer: int, block_tables, lengths, lpad: int):
    """Dense ``(n_rows, n_kv_head, lpad, head_dim)`` K/V view of ``pool`` for one layer.

    ``lengths[row]`` is the number of live logical tokens of that row; every column
    ``>= lengths[row]`` is zero. The implementation is the W4 one, minus the workspace
    reuse: for each row it builds the logical token -> physical slot list with Python
    indexing, selects those rows from the layer's flattened pool and copies them into a
    fresh zero tensor. Rows are independent, so a wrong block table or a stale block never
    agrees with the kernel by accident.
    """
    if not 0 <= int(layer) < pool.n_layer:
        raise IndexError(f"layer {layer} outside the pool's {pool.n_layer} layers")
    n_rows = len(block_tables)
    if n_rows == 0:
        raise ValueError("gather needs at least one row")
    if len(lengths) != n_rows:
        raise ValueError(f"gather got {len(lengths)} lengths for {n_rows} block tables")
    lengths = [int(n) for n in lengths]
    lpad = int(lpad)
    if lpad < max(lengths):
        raise ValueError(
            f"lpad {lpad} does not cover the longest row ({max(lengths)}); gather columns "
            "must be >= every row's length"
        )

    block_size = pool.block_size
    flat_k = pool.keys[layer].reshape(-1, pool.n_kv_head, pool.head_dim)
    flat_v = pool.values[layer].reshape(-1, pool.n_kv_head, pool.head_dim)
    out_k = torch.zeros(
        (n_rows, pool.n_kv_head, lpad, pool.head_dim), dtype=pool.dtype, device=pool.device
    )
    out_v = torch.zeros_like(out_k)
    for row, (block_table, length) in enumerate(zip(block_tables, lengths)):
        if length < 0 or length > lpad:
            raise ValueError(f"row {row}: length {length} outside [0, {lpad}]")
        needed = (max(length, 1) + block_size - 1) // block_size
        if len(block_table) < needed:
            raise IndexError(
                f"row {row}: {length} tokens need {needed} blocks, block table has "
                f"{len(block_table)}"
            )
        if length == 0:
            continue
        slots = [
            block_table[token // block_size] * block_size + (token % block_size)
            for token in range(length)
        ]
        slot_ids = torch.tensor(slots, dtype=torch.long, device=pool.device)
        out_k[row, :, :length, :] = flat_k.index_select(0, slot_ids).permute(1, 0, 2)
        out_v[row, :, :length, :] = flat_v.index_select(0, slot_ids).permute(1, 0, 2)
    return out_k, out_v


def write_token_reference(pool, layer: int, rows, block_tables, positions, k, v) -> None:
    """Per-row decode write with plain indexing (the pre-slot-map implementation).

    ``rows``/``positions`` are Python lists; ``k``/``v`` are
    ``(n_rows, n_kv_head, head_dim)``. Token ``row`` goes to physical
    ``block_table[pos // bs]``, column ``pos % bs``. This is the slow oracle for
    ``PagedPool.write_slots``/``PagedKV.write``; it validates the same mapping but with no
    device gather and no cached slot map.
    """
    block_size = pool.block_size
    if not 0 <= int(layer) < pool.n_layer:
        raise IndexError(f"layer {layer} outside the pool's {pool.n_layer} layers")
    n_rows = len(rows)
    if not (len(block_tables) == len(positions) == n_rows):
        raise ValueError(
            "write_token_reference needs one entry per row: "
            f"rows={n_rows}, block_tables={len(block_tables)}, positions={len(positions)}"
        )
    for row, (block_table, position) in enumerate(zip(block_tables, positions)):
        position = int(position)
        block_index = position // block_size
        if block_index >= len(block_table):
            raise IndexError(
                f"row {row}: position {position} needs block {block_index}, but its "
                f"block table has {len(block_table)} blocks"
            )
        block_id = block_table[block_index]
        if not 0 <= block_id < pool.n_blocks:
            raise IndexError(f"block id {block_id} outside the pool's {pool.n_blocks} blocks")
        offset = position % block_size
        pool.keys[layer][block_id, offset] = k[row].to(pool.dtype)
        pool.values[layer][block_id, offset] = v[row].to(pool.dtype)


def _validate_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
) -> None:
    """Layout validation shared with the W4.5 oracle (structural errors only).

    This is the same contract ``model/triton_decode.py::_validate`` enforces on the
    kernel, minus the CUDA/int32 requirements that only the launch has: the oracle accepts
    int64 block tables/seqlen and CPU tensors because it is a pure PyTorch computation.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be (B, H, D), got shape {tuple(q.shape)}")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError(
            "k_cache/v_cache must be (n_blocks, block_size, H_kv, D), got "
            f"{tuple(k_cache.shape)} / {tuple(v_cache.shape)}"
        )
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        raise ValueError(
            f"k_cache/v_cache shapes differ: {tuple(k_cache.shape)} vs {tuple(v_cache.shape)}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"q dtype must be fp16, bf16 or fp32, got {q.dtype}")
    if not (k_cache.dtype == v_cache.dtype == q.dtype):
        raise ValueError(
            "q/k_cache/v_cache must share one dtype, got "
            f"{q.dtype}, {k_cache.dtype}, {v_cache.dtype}"
        )

    B, H, D = q.shape
    _, block_size, H_kv, cache_d = k_cache.shape
    if D < 1:
        raise ValueError(f"head_dim must be positive, got {D}")
    if D != cache_d:
        raise ValueError(f"q head_dim {D} != cache head_dim {cache_d}")
    if H_kv < 1 or H % H_kv != 0:
        raise ValueError(f"n_head ({H}) must be a multiple of n_kv_head ({H_kv})")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size & (block_size - 1):
        raise ValueError(
            f"block_size must be a power of two (16 and 32 are the supported sizes), "
            f"got {block_size}"
        )
    if D & (D - 1):
        raise ValueError(f"head_dim must be a power of two for the kernel, got {D}")

    if block_table.dim() != 2:
        raise ValueError(
            f"block_table must be (B, max_blocks), got shape {tuple(block_table.shape)}"
        )
    B_bt, max_blocks = block_table.shape
    if B_bt != B:
        raise ValueError(f"block_table batch {B_bt} does not match q batch {B}")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"block_table must be int32 (the kernel contract) or int64, got "
            f"{block_table.dtype}"
        )

    if seqlen.dim() != 1:
        raise ValueError(f"seqlen must be (B,), got shape {tuple(seqlen.shape)}")
    if seqlen.shape[0] != B:
        raise ValueError(f"seqlen has {seqlen.shape[0]} entries for a batch of {B}")
    if seqlen.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"seqlen must be int32 or int64, got {seqlen.dtype}")

    device = q.device
    for name, tensor in (
        ("q", q),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("block_table", block_table),
        ("seqlen", seqlen),
    ):
        if tensor.device != device:
            raise ValueError(
                f"{name} is on {tensor.device} but q is on {device}; all paged_decode "
                "tensors must share one device"
            )

    if B == 0:
        return
    capacity = max_blocks * block_size
    low = int(seqlen.min())
    high = int(seqlen.max())
    if low < 0:
        raise ValueError(f"seqlen values must be >= 0, got {low}")
    if high > capacity:
        raise ValueError(
            f"seqlen {high} exceeds the block table capacity {capacity} "
            f"({max_blocks} blocks x {block_size} tokens)"
        )


def reference_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense PyTorch oracle for ``paged_decode``; CPU or CUDA, never in production.

    The implementation is deliberately the slow, obvious one: for each row it gathers the
    row's logical K/V with explicit tensor indexing, computes one ``(H_kv, G, n)`` score
    block in fp32, softmaxes it and multiplies by V. Duplicated or non-contiguous block
    ids therefore behave exactly like the kernel's page loop; a zero-length row is zeroed
    explicitly rather than asking ``softmax`` to normalize an empty page.
    """
    _validate_reference(q, k_cache, v_cache, block_table, seqlen)
    B, H, D = q.shape
    block_size = k_cache.shape[1]
    H_kv = k_cache.shape[2]
    G = H // H_kv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scale = float(scale)

    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if B == 0:
        return out
    table = block_table.to(torch.long)
    lengths = seqlen.tolist()  # oracle only: one host read is fine here
    qf = q.to(torch.float32).reshape(B, H_kv, G, D)

    for b in range(B):
        n = int(lengths[b])
        if n <= 0:
            out[b].zero_()
            continue
        pages = (n + block_size - 1) // block_size
        ids = table[b, :pages]
        # (pages, block_size, H_kv, D) -> (n, H_kv, D); the slice drops the masked tail.
        k = k_cache.index_select(0, ids).reshape(pages * block_size, H_kv, D)[:n]
        v = v_cache.index_select(0, ids).reshape(pages * block_size, H_kv, D)[:n]
        kf = k.to(torch.float32).permute(1, 0, 2)  # (H_kv, n, D)
        vf = v.to(torch.float32).permute(1, 0, 2)
        scores = torch.einsum("hgd,hnd->hgn", qf[b], kf) * scale
        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum("hgn,hnd->hgd", probs, vf)
        out[b] = context.reshape(H, D).to(q.dtype)
    return out
