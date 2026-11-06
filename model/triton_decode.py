"""W4.5 block-table decode: one Triton kernel reading the paged KV pool directly.

W4's paged arm materialized a dense ``(B, H_kv, lpad, D)`` copy of the pool through
``PagedPool.gather`` and then ran ``hand_baddbmm`` on it. The copy is the cost: it reads
every live KV byte, writes it again, and pays an ``lpad`` bucket that is wider than any
request in the batch. ``decode`` has exactly one query token, so the gather buys nothing:
the attention math can read K/V page by page from the physical pool and never materialize
a dense sequence. This module is that path.

Contract
--------
``paged_decode(q, k_cache, v_cache, block_table, seqlen, *, scale=None)``

* ``q``: ``(B, H, D)``; ``k_cache``/``v_cache``: ``(n_blocks, block_size, H_kv, D)``;
  K/V and q share a dtype (fp16, bf16 or fp32, and q may be non-contiguous -- it is
  copied once, because a ``(B, H, D)`` query is tiny);
* ``block_table``: ``(B, max_blocks)`` **int32 on the same CUDA device**. Row ``b``, entry
  ``j`` is the physical block holding logical page ``j``. Only
  ``ceil(seqlen[b] / block_size)`` entries are read; the rest may be garbage;
* ``seqlen``: ``(B,)`` int32, or int64 (cast here); values in
  ``[0, max_blocks * block_size]``. The last page is partial and is masked by
  ``column < seqlen[b]``;
* output ``(B, H, D)`` in ``q.dtype``; ``scale=None`` means ``1 / sqrt(D)``.

There is no gather, no ``lpad`` width, no zero block and no causal mask. A decode query
may attend every key it has, so causal masking would be wrong; the only mask is the tail
of the last page. This is why the kernel does not need the W4 ``mask``/``bias`` pair and
why its output is independent of any batch neighbour's length.

``block_size`` is inferred from ``k_cache.shape[1]`` (16 and 32 are the tested sizes,
both fixed by W4.5); ``D`` and ``block_size`` are ``tl.constexpr`` so each shape gets its
own generated kernel. GQA is not expanded: ``h_kv = h // (H // H_kv)``, so the group's
query heads read the same K/V bytes (and hit the same cache lines).

Kernel shape
------------
Grid ``(B, H)``: one program per query head. The page loop is a device-side dynamic loop
(``ceil(seqlen[b] / block_size)``), one page per iteration, with online softmax in fp32
(``m_i``, ``l_i``, ``acc``). That is deliberately the simple version: it keeps the
arithmetic identical to the dense reference and needs no partial-result merge. Grid size is
``B * H``, so B=1 is latency-bound by the 16 programs of the 0.6B carrier; the benchmark's
B=8/32 rows are the throughput shape. Split-K (partial ``m``/``l``/``acc`` plus a merge
kernel) is the documented next step and is not enabled here: the simple kernel stays the
default until an integration measurement asks for the switch.

Availability
------------
Triton and CUDA are hard requirements for :func:`paged_decode`. It raises ``RuntimeError``
when either is missing or when a tensor is not on CUDA; it never falls back to
:func:`reference_paged_decode`. That function is the numeric oracle (pure PyTorch, CPU or
CUDA) and the only implementation used by tests as ground truth.

Integration caveat
------------------
The seqlen range check is host-side and therefore the only synchronizing part of the
wrapper. Its result is cached per ``(seqlen tensor, tensor version, capacity)`` so a
persistent ``seqlen`` buffer -- the engine's natural layout -- pays one sync, not one per
layer. A caller that allocates a new ``seqlen`` tensor every step will pay one sync per
step. Inference tensors carry no version counter, so their first successful check is
reused until a different tensor object arrives; the kernel clamps the page walk to
``max_blocks`` as the safety net for a later in-place mutation. The kernel itself does no
bounds checking on block ids; the block table is trusted, exactly like the allocator
contract in ``model/paged.py``.
"""

from __future__ import annotations

import math

import torch

try:  # Triton is optional on CPU/MPS hosts; the module must still import there.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised by the CPU-only test gate
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

__all__ = ["TRITON_AVAILABLE", "paged_decode", "paged_decode_plan", "reference_paged_decode"]

#: ``log2(e)``; folding it into the score scale lets the softmax use ``tl.exp2``.
_LOG2E = 1.4426950408889634

#: Last seqlen range validation: ``(seqlen, version, capacity)``. Holding the tensor keeps
#: its data pointer stable, so the identity check cannot alias a freed tensor.
_SEQ_CHECK: tuple[torch.Tensor, int, int] | None = None


def _require_backend() -> None:
    """Fail loudly instead of falling back; the paged arm owns a CUDA-only fast path."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "paged_decode requires Triton, but Triton could not be imported; the paged "
            "decode arm has no eager fallback. Use reference_paged_decode only as a test "
            "oracle."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "paged_decode requires CUDA, but torch.cuda is unavailable; the paged decode "
            "arm has no CPU fallback. Use reference_paged_decode only as a test oracle."
        )


def _require_cuda_tensors(tensors: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Every input must be a CUDA tensor; there is deliberately no host path here."""
    for name, tensor in tensors:
        if not tensor.is_cuda:
            raise RuntimeError(
                f"paged_decode requires CUDA tensors, got {name} on device {tensor.device}; "
                "there is no silent fallback to reference_paged_decode."
            )


def _validate(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    for_kernel: bool,
) -> None:
    """Validate the layout contract shared by the kernel and its oracle.

    Structural errors (shape, dtype, device) are ``ValueError``; only "no Triton/CUDA"
    is a ``RuntimeError``. The distinction matters because a wrong block table is a caller
    bug, while a missing backend is an environment capability.
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
    if for_kernel and block_table.dtype != torch.int32:
        raise ValueError(
            "paged_decode requires an int32 block_table (the kernel's indexing contract); "
            f"got {block_table.dtype}. Cast at the allocator boundary, not per step."
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
    # One range check per tensor version (see _SEQ_CHECK). A new or mutated tensor is
    # revalidated; an unchanged one costs no host synchronization. Inference tensors have
    # no version counter (the engine's PagedKV builds ``seqlen`` under inference_mode), so
    # their first successful check is reused; the kernel also clamps the page walk to the
    # table width, which keeps a later in-place mutation from reading past the table.
    global _SEQ_CHECK
    version = -1 if seqlen.is_inference() else int(seqlen._version)
    cached = _SEQ_CHECK
    if (
        cached is not None
        and cached[0] is seqlen
        and cached[1] == version
        and cached[2] == capacity
    ):
        return
    low = int(seqlen.min())
    high = int(seqlen.max())
    if low < 0:
        raise ValueError(f"seqlen values must be >= 0, got {low}")
    if high > capacity:
        raise ValueError(
            f"seqlen {high} exceeds the block table capacity {capacity} "
            f"({max_blocks} blocks x {block_size} tokens)"
        )
    _SEQ_CHECK = (seqlen, version, capacity)


def _launch_config(block_dim: int) -> tuple[int, int]:
    """``(num_warps, num_stages)`` for one decode kernel with this head dim.

    Measured on a 4090: one warp per program is fastest for D 64/128 because the page
    reductions stay inside a warp (no shared-memory barrier per page); D=256 needs two
    warps to keep enough lanes busy. ``num_stages=1`` avoids pipelining a loop whose
    carried softmax state is the dependency. The sweep behind these numbers is
    ``scripts/bench_decode_kernel.py``; note it re-reads one layer's KV, so it measures a
    warm-L2 kernel. On the cold, 28-layer E2 pattern a second warp is ~24% faster
    (112.1 -> 84.9 us/layer), but the end-to-end step is CPU-dispatch-bound and did not
    move (W45-0005: 23.37 vs 23.43 ms, inside run-to-run noise), so the shipped shape is
    unchanged.
    """
    return (1 if block_dim <= 128 else 2), 1


def paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one query token per row against a paged KV pool.

    See the module docstring for the contract. This is the production path: it launches
    :func:`_paged_decode_kernel` and never copies the KV pool to a dense sequence.
    """
    _require_backend()
    _require_cuda_tensors(
        (
            ("q", q),
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("block_table", block_table),
            ("seqlen", seqlen),
        )
    )
    _validate(q, k_cache, v_cache, block_table, seqlen, for_kernel=True)

    B, H, D = q.shape
    block_size = k_cache.shape[1]
    if B == 0:
        return torch.empty_like(q)

    # ``q`` is (B, H, D) and tiny; contiguous is what the kernel's stride arguments
    # describe, and the copy is cheaper than carrying four strides through the page loop.
    q = q.contiguous()
    if block_table.stride(1) != 1:
        block_table = block_table.contiguous()
    seqlen = seqlen.contiguous()
    seqlen32 = seqlen.to(torch.int32) if seqlen.dtype != torch.int32 else seqlen
    out = torch.empty_like(q)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    # Fold log2(e) into the score once, so the kernel's max/exp updates can use tl.exp2.
    scale_log2 = float(scale) * _LOG2E

    block_dim = D
    num_warps, num_stages = _launch_config(block_dim)
    grid = (B, H)
    _paged_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        seqlen32,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        block_table.shape[1],
        scale_log2,
        H=H,
        H_KV=k_cache.shape[2],
        BLOCK_D=block_dim,
        BLOCK_N=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


class PagedDecodePlan:
    """Per-container launch metadata for repeated :func:`paged_decode` calls.

    Why this exists
    ---------------
    ``paged_decode`` re-checks the whole contract on every call. That is right for a
    public entry point, but a 28-layer decode step calls it 28 times with tensors whose
    *structure* is identical: the pool is allocated once, every layer has the same shape
    and strides, and the block table / seqlen are container-owned. Measured on the E2
    workload (W45-0005, B=8, 28 layers), the wrapper cost 64.9 us of host time per layer
    against 37.7 us for the same launch with precomputed metadata; switching the
    container to this plan moved its decode step from 23.60 ms to 23.08 ms (single-run
    harness numbers, +-0.3 ms run-to-run). The launch is only part of the step -- it is
    partly hidden behind the kernel -- which is why the win is smaller than 0.76 ms.

    A plan validates every ``(k_cache, v_cache)`` pair **once** (the full
    :func:`paged_decode` contract, including the seqlen range check) and caches the launch
    metadata that follows from those shapes. ``run`` then re-checks only the one input
    that changes per layer -- the query -- and launches.

    Invariants (and invalidation)
    -----------------------------
    * The validated ``k``/``v`` tensors are held by reference. A caller that *replaces*
      an entry of the pool's layer list must build a new plan; the container rebuilds
      itself (and its plan) whenever the active set changes, and a pool's tensors are
      allocated once and never replaced.
    * The block table and int32 seqlen are held by reference and may be mutated in place
      (that is the decode contract: ``seqlen`` advances every step). The seqlen range
      check is therefore the same cached one ``paged_decode`` already uses, and the kernel
      clamps the page walk to the table width. An int64 seqlen is rejected rather than
      copied, because a copy would silently stop tracking those in-place updates.
    * ``run`` refuses a query whose shape, dtype, device or strides differ from the
      validated one, so the cached strides can never describe a different layout.
    """

    __slots__ = (
        "n_layer",
        "grid",
        "_tail",
        "_meta",
        "q_shape",
        "q_dtype",
        "q_device",
        "q_strides",
        "_k",
        "_v",
        "_block_table",
        "_seqlen",
    )

    def __init__(
        self,
        q: torch.Tensor,
        k_caches,
        v_caches,
        block_table: torch.Tensor,
        seqlen: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> None:
        _require_backend()
        _require_cuda_tensors(
            (("q", q), ("block_table", block_table), ("seqlen", seqlen))
        )
        if q.dim() != 3:
            raise ValueError(f"q must be (B, H, D), got shape {tuple(q.shape)}")
        if not q.is_contiguous():
            raise ValueError(
                "paged_decode_plan needs a contiguous (B, H, D) query, got strides "
                f"{q.stride()}; call .contiguous() once, not per layer"
            )
        b, h, d = q.shape
        if b == 0:
            raise ValueError("paged_decode_plan needs at least one row")
        # The plan holds the seqlen tensor by reference so in-place step updates are seen.
        # An int64 input would be copied to int32 once and then go stale; make the caller
        # cast at the boundary (the container already stores int32).
        if seqlen.dtype != torch.int32:
            raise ValueError(
                "paged_decode_plan requires an int32 seqlen so in-place updates are "
                f"visible; got {seqlen.dtype}. Cast once at the container boundary."
            )
        if seqlen.stride(0) != 1:
            raise ValueError(
                "paged_decode_plan requires a contiguous 1-D seqlen; got strides "
                f"{seqlen.stride()}"
            )
        if block_table.stride(1) != 1:
            raise ValueError(
                "paged_decode_plan requires a column-contiguous block table; got strides "
                f"{block_table.stride()}"
            )
        k_caches = list(k_caches)
        v_caches = list(v_caches)
        if len(k_caches) != len(v_caches) or not k_caches:
            raise ValueError(
                "paged_decode_plan needs one k_cache/v_cache per layer, got "
                f"{len(k_caches)} / {len(v_caches)}"
            )
        for layer, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
            # Full contract once per layer: shape/dtype/device cross-checks, the int32
            # block-table rule, and the (cached) seqlen range validation.
            _require_cuda_tensors(
                (
                    (f"k_cache[{layer}]", k_cache),
                    (f"v_cache[{layer}]", v_cache),
                )
            )
            _validate(q, k_cache, v_cache, block_table, seqlen, for_kernel=True)
            if tuple(k_cache.shape) != tuple(k_caches[0].shape) or tuple(
                v_cache.shape
            ) != tuple(v_caches[0].shape):
                raise ValueError("paged_decode_plan needs one pool shape for every layer")
            if k_cache.stride() != k_caches[0].stride():
                raise ValueError("paged_decode_plan needs one pool layout for every layer")
            if v_cache.stride() != v_caches[0].stride():
                raise ValueError("paged_decode_plan needs one v-pool layout for every layer")

        self.n_layer = len(k_caches)
        block_size = int(k_caches[0].shape[1])
        self._k = k_caches
        self._v = v_caches
        self._block_table = block_table
        self._seqlen = seqlen

        self.q_shape = (int(b), int(h), int(d))
        self.q_dtype = q.dtype
        self.q_device = q.device
        self.q_strides = (int(q.stride(0)), int(q.stride(1)), int(q.stride(2)))

        if scale is None:
            scale = 1.0 / math.sqrt(d)
        scale_log2 = float(scale) * _LOG2E
        # One heuristic for both entry points, so the plan and paged_decode can never
        # launch the same kernel with different block shapes.
        num_warps, _num_stages = _launch_config(int(d))
        self.grid = (int(b), int(h))
        self._meta = {
            "H": int(h),
            "H_KV": int(k_caches[0].shape[2]),
            "BLOCK_D": int(d),
            "BLOCK_N": int(block_size),
            "num_warps": num_warps,
            "num_stages": _num_stages,
        }
        # Every trailing scalar is fixed by the validated layouts: q/out share the (B,H,D)
        # strides, and every layer's pool shares one stride tuple.
        k0 = k_caches[0]
        v0 = v_caches[0]
        self._tail = (
            self.q_strides[0],
            self.q_strides[1],
            self.q_strides[2],
            int(k0.stride(0)),
            int(k0.stride(1)),
            int(k0.stride(2)),
            int(k0.stride(3)),
            int(v0.stride(0)),
            int(v0.stride(1)),
            int(v0.stride(2)),
            int(v0.stride(3)),
            self.q_strides[0],
            self.q_strides[1],
            self.q_strides[2],
            int(block_table.stride(0)),
            int(block_table.shape[1]),
            scale_log2,
        )

    def run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Launch the kernel for one layer; ``q`` must match the validated layout."""
        if not 0 <= int(layer) < self.n_layer:
            raise IndexError(f"layer {layer} outside the plan's {self.n_layer} layers")
        if (
            tuple(q.shape) != self.q_shape
            or q.dtype != self.q_dtype
            or q.device != self.q_device
            or q.stride() != self.q_strides
        ):
            raise ValueError(
                "paged_decode_plan.run got a query that does not match the validated "
                f"layout {(self.q_shape, self.q_dtype, self.q_strides)}; got "
                f"{(tuple(q.shape), q.dtype, q.stride())}. Rebuild the plan for a new layout."
            )
        out = torch.empty_like(q)
        _paged_decode_kernel[self.grid](
            q, self._k[layer], self._v[layer], self._block_table, self._seqlen, out,
            *self._tail, **self._meta,
        )
        return out


def paged_decode_plan(
    q: torch.Tensor,
    k_caches,
    v_caches,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    scale: float | None = None,
) -> PagedDecodePlan:
    """Validate one container's paged-decode tensors and cache their launch metadata.

    ``k_caches`` / ``v_caches`` are one tensor per layer (the pool's ``keys`` /
    ``values``). The returned plan's :meth:`PagedDecodePlan.run` is the per-layer hot
    path; :func:`paged_decode` remains the fully-revalidating public entry point.
    """
    return PagedDecodePlan(q, k_caches, v_caches, block_table, seqlen, scale=scale)


def reference_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense PyTorch oracle for :func:`paged_decode`; CPU or CUDA, never in production.

    The implementation is deliberately the slow, obvious one: for each row it gathers the
    row's logical K/V with explicit tensor indexing, computes one ``(H_kv, G, n)`` score
    block in fp32, softmaxes it and multiplies by V. Duplicated or non-contiguous block
    ids therefore behave exactly like the kernel's page loop; a zero-length row is zeroed
    explicitly rather than asking ``softmax`` to normalize an empty page.
    """
    _validate(q, k_cache, v_cache, block_table, seqlen, for_kernel=False)
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


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        seqlen_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kn,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vn,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_od,
        stride_btb,
        max_blocks,
        scale_log2,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One query head, one page per iteration, online softmax in fp32.

        ``qk`` is accumulated in log2 space: the caller passes ``scale * log2(e)``, so
        both the max update and the exponentials use ``tl.exp2``. Masking the tail of the
        last page with ``-inf`` before the max keeps the page's invalid columns out of
        ``m_i``/``l_i``; the loop bound already guarantees the last page has one valid
        column, and a zero-length row runs zero iterations and divides by 1.
        """
        row = tl.program_id(0)
        head = tl.program_id(1)
        group = H // H_KV
        head_kv = head // group

        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            q_ptr + row * stride_qb + head * stride_qh + offs_d * stride_qd
        ).to(tl.float32)

        length = tl.load(seqlen_ptr + row)
        n_pages = tl.minimum(tl.cdiv(length, BLOCK_N), max_blocks)
        offs_n = tl.arange(0, BLOCK_N)

        # fp32 running state: max, normalizer, weighted value accumulator.
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for page in tl.range(0, n_pages):
            block_id = tl.load(block_table_ptr + row * stride_btb + page).to(tl.int64)
            block_base_k = block_id * stride_kb
            block_base_v = block_id * stride_vb
            columns = page * BLOCK_N + offs_n
            valid = columns < length

            k = tl.load(
                k_ptr
                + block_base_k
                + offs_n[:, None] * stride_kn
                + head_kv * stride_kh
                + offs_d[None, :] * stride_kd,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            # One query token against BLOCK_N keys: a rowwise dot, not a GEMM.
            scores = tl.sum(k * q[None, :], axis=1) * scale_log2
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(scores - m_new)

            v = tl.load(
                v_ptr
                + block_base_v
                + offs_n[:, None] * stride_vn
                + head_kv * stride_vh
                + offs_d[None, :] * stride_vd,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        result = acc / tl.maximum(l_i, 1.0)
        tl.store(
            out_ptr + row * stride_ob + head * stride_oh + offs_d * stride_od,
            result.to(out_ptr.dtype.element_ty),
        )
