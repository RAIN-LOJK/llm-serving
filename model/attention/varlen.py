"""Packed-prefill attention: one varlen call for a whole arrival group.

W4.5 replaces "one SDPA prefill per request" with "one packed call per group of requests".
The layout that makes that possible has no batch dimension and no padding::

    q: (total_tokens, n_head,    head_dim)
    k: (total_tokens, n_kv_head, head_dim)
    v: (total_tokens, n_kv_head, head_dim)
    cu_seqlens: [0, len_0, len_0 + len_1, ..., total_tokens]   # n_seqs + 1 entries

Each segment is one request, and ``causal=True`` means causal *inside* each segment. That
is the reason for the extra ``cu_seqlens`` argument: a plain ``is_causal=True`` over the
concatenated tokens would let request 0 read the suffix of request 1, or read packed tokens
that belong to nobody, and would silently produce a plausible but wrong first token on
every serving shape. There is no error to catch when that happens, so the segment
boundaries are validated here and the kernel never sees the raw prefix list.

**Two arms, picked by device, never by availability** (``backend_for``):

* ``FlashVarlen`` -- the CUDA kernel, ``flash_attn.flash_attn_varlen_func`` (FA2 2.8.3).
* ``SdpaVarlen`` -- one ``F.scaled_dot_product_attention`` per segment. CPU/MPS arm, and
  the correctness reference the CUDA arm is checked against in ``tests/``. It is not a
  production fallback: a request whose device resolves to CPU has no fused kernel to run.

The caller records ``backend.name`` in the run provenance. That is what makes the choice
auditable -- a packed-prefill run that silently executed the reference arm would be visible
as ``sdpa-varlen`` instead of ``flash-attn-varlen``. For the same reason ``FlashVarlen``
raises ``RuntimeError`` when flash-attn is missing or when the tensors are not CUDA instead
of degrading to SDPA: a fallback would turn every "packed is faster" measurement into a
measurement of the baseline, and the E3 comparison would look flat rather than broken.

**No padding.** The segments stay ragged; only ``max_seqlen`` (a scalar FA2 needs for its
launch configuration) is passed alongside. Padding the time axis would materialize a
``(B, T_max)`` slab and knock SDPA off its fused backends -- PITFALLS P-22, which is why the
W2 prefill path never pads either. Packing is over tokens instead.

Out of scope here: KV-cache writes, slot maps and paged block tables. The kernel takes dense
packed K/V and returns packed hidden states; where the K/V land (a seat row or a paged
block) is the caller's decision. That is also why this module does not implement
``AttentionBackend`` (``base.py``): that protocol's ``(B, H, T, D)`` shape with an optional
bool mask is the W2 contract, and the packed call has neither a batch dim nor a mask.

**The FA2 call actually used** (flash-attn 2.8.3, RTX 4090)::

    flash_attn_varlen_func(
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        dropout_p=0.0, softmax_scale=scale, causal=causal,
    )

with ``cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=q.device)``. Five details
of that signature are load-bearing and are normalized here rather than at every caller:

* ``cu_seqlens`` must be ``int32``. FA2 2.8.3 rejects an ``int64`` tensor with
  ``RuntimeError: cu_seqlens_q must have dtype int32``; the wrapper converts, so a Python
  list or an ``int64`` tensor both work.
* ``max_seqlen_*`` is a Python ``int`` at the call site (FA2 accepts a 0-dim tensor too, but
  a tensor would be read at every layer and every call).
* ``cu_seqlens`` is passed twice (q and k) because packed prefill does not chunk yet:
  ``total_q == total_k``.
* q/k/v are passed through ``.contiguous()``. A packed group is frequently a slice of a
  larger buffer and FA2's layout assumption is a dense ``(total, H, D)``; the copy is a
  no-op when the caller already packed into a fresh tensor.
* FA2 computes fp16/bf16 only. fp32 is rejected with ``RuntimeError`` here instead of
  surfacing as a CUDA-side error, and fp32 correctness runs belong on ``SdpaVarlen``.

``softmax_scale=None`` resolves to ``1 / sqrt(head_dim)`` once, in the shared validator, so
both arms use the same number; the resolved value is forwarded to FA2/F.sdpa as a keyword
rather than left to each library's default. SDPA has a ``scale=`` argument, so a custom
scale is honoured there too instead of being silently ignored.
"""

from __future__ import annotations

import math
import operator
from typing import Sequence

import torch
import torch.nn.functional as F

__all__ = ["SdpaVarlen", "FlashVarlen", "backend_for"]


# ---------------------------------------------------------------------------
# shared validation: both arms see the same contract
# ---------------------------------------------------------------------------


def _as_int(value, name: str) -> int:
    """Coerce an integer-like scalar (Python int, numpy int, 0-dim tensor) or raise."""
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(
            f"{name} must be an int, got {type(value).__name__} ({value!r})"
        ) from None


def _segment_bounds(cu_seqlens, total_tokens: int) -> list[int]:
    """Validate ``cu_seqlens`` and return it as a plain list of segment bounds.

    The invariant is the packed-prefill contract from the module docstring: entry 0 is 0,
    the last entry is ``total_tokens``, and the entries are strictly increasing (one
    non-empty segment per request). A zero-length segment is rejected rather than skipped:
    it is not a request, and both arms would have to special-case an empty attention row
    that nothing can attend to.
    """
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.dim() != 1:
            raise ValueError(
                "cu_seqlens must be a 1-D int tensor of n_seqs + 1 offsets, got shape "
                f"{tuple(cu_seqlens.shape)}; the packed layout has no batch dimension"
            )
        if cu_seqlens.dtype.is_floating_point or cu_seqlens.dtype.is_complex:
            raise ValueError(
                "cu_seqlens must hold integers (token offsets), got dtype "
                f"{cu_seqlens.dtype}"
            )
        entries = cu_seqlens.tolist()
    else:
        if isinstance(cu_seqlens, (str, bytes)):
            raise ValueError(
                f"cu_seqlens must be a sequence of ints or a 1-D int tensor, got "
                f"{type(cu_seqlens).__name__}"
            )
        try:
            entries = list(cu_seqlens)
        except TypeError:
            raise ValueError(
                f"cu_seqlens must be a sequence of ints or a 1-D int tensor, got "
                f"{type(cu_seqlens).__name__}"
            ) from None
    bounds = [_as_int(entry, "every cu_seqlens entry") for entry in entries]

    if len(bounds) < 2:
        raise ValueError(
            "cu_seqlens needs n_seqs + 1 entries (one start offset per segment plus the "
            f"final end offset); got {len(bounds)}"
        )
    if bounds[0] != 0:
        raise ValueError(
            f"cu_seqlens must start at 0, got {bounds[0]} (cu_seqlens={bounds})"
        )
    if bounds[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens must end at total_tokens={total_tokens} (q.size(0)), got "
            f"{bounds[-1]} (cu_seqlens={bounds})"
        )
    for index, (previous, current) in enumerate(zip(bounds, bounds[1:])):
        if current <= previous:
            raise ValueError(
                "cu_seqlens must be strictly increasing (one non-empty segment per "
                f"request); entry {index + 1} is {current} after {previous} "
                f"(cu_seqlens={bounds})"
            )
    return bounds


def _validate_packed(q, k, v, cu_seqlens, max_seqlen) -> list[int]:
    """Check the packed layout and return the segment bounds.

    Shape and offset errors raise ``ValueError``: they are caller bugs that no kernel can
    fix. Device *availability* (CPU tensors handed to ``FlashVarlen``) is the one failure
    that is not a shape error and is raised by the CUDA arm as ``RuntimeError`` instead.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != 3:
            raise ValueError(
                "packed prefill takes 3-D (total_tokens, n_head, head_dim) tensors with "
                f"no batch dimension, got {name}.shape={tuple(tensor.shape)}"
            )
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            "q, k, v must be on the same device, got q={}, k={}, v={}".format(
                q.device, k.device, v.device
            )
        )
    if not q.dtype.is_floating_point:
        raise ValueError(
            f"packed prefill needs floating-point q/k/v, got dtype {q.dtype}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, v must share one dtype, got q={}, k={}, v={}".format(
                q.dtype, k.dtype, v.dtype
            )
        )

    total_tokens, n_head, head_dim = q.shape
    if k.size(0) != total_tokens or v.size(0) != total_tokens:
        raise ValueError(
            "q, k, v must cover the same packed tokens: got q={}, k={}, v={} rows".format(
                total_tokens, k.size(0), v.size(0)
            )
        )
    n_kv_head = k.size(1)
    if k.size(1) != v.size(1):
        raise ValueError(
            "k and v must have the same number of heads, got k={}, v={}".format(
                k.size(1), v.size(1)
            )
        )
    if n_head <= 0 or n_kv_head <= 0:
        raise ValueError(
            f"n_head and n_kv_head must be positive, got {n_head} and {n_kv_head}"
        )
    if n_head % n_kv_head:
        raise ValueError(
            "n_head must be a multiple of n_kv_head (GQA/MQA); got n_head={}, "
            "n_kv_head={}".format(n_head, n_kv_head)
        )
    if k.size(2) != head_dim or v.size(2) != head_dim:
        raise ValueError(
            "q, k, v must share head_dim; got q={}, k={}, v={}".format(
                head_dim, k.size(2), v.size(2)
            )
        )

    bounds = _segment_bounds(cu_seqlens, total_tokens)
    max_seqlen = _as_int(max_seqlen, "max_seqlen")
    lengths = [end - start for start, end in zip(bounds, bounds[1:])]
    longest = max(lengths)
    if max_seqlen != longest:
        raise ValueError(
            f"max_seqlen={max_seqlen} does not match the longest segment ({longest}); "
            f"segment lengths={lengths}. The value is not clamped: it sizes FA2's launch "
            "tiles and a value smaller than the real maximum would index past a segment."
        )
    return bounds


def _resolve_scale(softmax_scale, head_dim: int) -> float:
    """``None`` means the library default, spelled out so both arms agree on it."""
    if softmax_scale is None:
        return 1.0 / math.sqrt(head_dim)
    return float(softmax_scale)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


class SdpaVarlen:
    """Per-segment ``F.scaled_dot_product_attention``: the reference arm.

    It exists for CPU/MPS and for tests, not as a production fallback. The segment loop is
    the point: each slice is a separate attention problem, so no query can see another
    request's keys even though the tokens share one contiguous buffer. A fused varlen
    kernel gets that from ``cu_seqlens``; here it comes from slicing.
    """

    name = "sdpa-varlen"

    def prefill_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: Sequence[int] | torch.Tensor,
        max_seqlen: int,
        softmax_scale: float | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """Attend within each packed segment; return ``(total_tokens, n_head, head_dim)``."""
        bounds = _validate_packed(q, k, v, cu_seqlens, max_seqlen)
        scale = _resolve_scale(softmax_scale, q.size(-1))
        enable_gqa = q.size(1) != k.size(1)
        segments = [
            F.scaled_dot_product_attention(
                # SDPA's batch dim is the leading one, so a packed slice becomes B=1.
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            .squeeze(0)
            .transpose(0, 1)
            for start, end in zip(bounds, bounds[1:])
        ]
        return torch.cat(segments, dim=0)


class FlashVarlen:
    """FA2 ``flash_attn_varlen_func``: the CUDA arm.

    The package is imported in ``__init__``, not at module import: importing
    ``model.attention.varlen`` on a CPU-only host must stay possible (the SDPA arm is
    importable there), while *constructing* the CUDA arm must fail loudly if the kernel is
    not installed. A missing wheel, a CPU-only build or a broken extension all become one
    ``RuntimeError`` that names the problem; none of them becomes SDPA.
    """

    name = "flash-attn-varlen"

    def __init__(self) -> None:
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception as exc:  # ImportError, OSError, or a CUDA-side RuntimeError
            raise RuntimeError(
                "the flash-attn-varlen backend requires the `flash-attn` package with CUDA "
                f"support, but importing it failed: {type(exc).__name__}: {exc}. Install "
                "the matching flash-attn wheel, or run on CPU/MPS through SdpaVarlen via "
                "backend_for(); there is no silent fallback to SDPA."
            ) from exc
        self._varlen_func = flash_attn_varlen_func

    def prefill_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: Sequence[int] | torch.Tensor,
        max_seqlen: int,
        softmax_scale: float | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """Attend within each packed segment; return ``(total_tokens, n_head, head_dim)``."""
        bounds = _validate_packed(q, k, v, cu_seqlens, max_seqlen)
        scale = _resolve_scale(softmax_scale, q.size(-1))
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError(
                "FlashVarlen is the CUDA kernel path, but q/k/v are on "
                f"{q.device}/{k.device}/{v.device}. Use SdpaVarlen on CPU/MPS and let "
                "backend_for() pick the arm; falling back here would report SDPA timings "
                "under the flash-attn name."
            )
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "flash-attn varlen computes fp16/bf16 only, got q.dtype="
                f"{q.dtype}. Cast the packed batch to bf16/fp16 (the carrier runs bf16), "
                "or run the fp32 correctness check against SdpaVarlen."
            )
        packed_cu = torch.tensor(bounds, dtype=torch.int32, device=q.device)
        max_seqlen = int(max_seqlen)
        return self._varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q=packed_cu,
            cu_seqlens_k=packed_cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=bool(causal),
        )


def backend_for(device) -> SdpaVarlen | FlashVarlen:
    """Select the packed-prefill arm for a device.

    This is device selection, not an availability check: it does not ask whether CUDA is
    usable, and it never returns a different arm because the preferred one is missing. A
    CUDA device constructs ``FlashVarlen``, whose ``__init__`` raises if the kernel is
    absent. The caller stores ``backend_for(device).name`` in provenance, so the selection
    is part of the run record rather than a hidden branch.

    The arm is constructed per call rather than cached in a module-level constant: a global
    would be mutable state shared across engines and tests, and constructing it is an
    import lookup plus one attribute assignment.
    """
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda":
        return FlashVarlen()
    if resolved.type in ("cpu", "mps"):
        return SdpaVarlen()
    raise ValueError(
        f"no packed-prefill backend for device {resolved!s} (type {resolved.type!r}); "
        "the arms are CUDA (flash-attn) and CPU/MPS (sdpa)."
    )
