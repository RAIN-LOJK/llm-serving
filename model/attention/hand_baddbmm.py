"""``hand_baddbmm``: the decode-only attention path, one call for all B sequences.

Decode (T == 1) has a shape of its own -- ``(B, H, 1, D)`` against ``(B, H_kv, L, D)`` --
and the cheap formulation is *one* batched call over every sequence, not one dispatch per
sequence. The oracle (``tests/reference/``) must stay slow, and ``sdpa`` is a
dispatcher whose kernel choice is not ours, so the decode shape gets this module.

Four decisions below are load-bearing, and they were measured (4090, bf16, B=8, L=512,
D=128, 28 layers: ~0.7 ms/step here against ~7.4 ms/step for the batched SDPA fallback,
which materializes GQA-expanded K/V and a ``(B, H, 1, L)`` additive mask):

* **GQA is not expanded.** ``q`` is viewed as ``(B, H_kv, G, 1, D)`` with ``G = H // H_kv``
  and the score batch is ``B * H_kv``; K/V keep their ``(B, H_kv, L, D)`` layout and are
  flattened only to ``(B * H_kv, ...)``. ``repeat_interleave`` to ``H`` heads would
  materialize ``(B, H, L, D)`` K/V -- exactly the allocation that makes the oracle slow.
* **The scale is folded into q**, never into the scores: decode has ``L >> D``, so q is
  ``(B, H, 1, D)`` against scores of ``(B, H, 1, L)``.
* **The mask is fused into the score GEMM.** The bool allow-mask becomes an additive bias
  of shape ``(B*H_kv, G, L)`` and ``torch.baddbmm(bias, q, k^T, beta=1, alpha=1)`` computes
  scores + bias in one call; a separate ``scores + bias`` would read and write the whole
  ``(B, H, 1, L)`` tensor a second time. Without a mask the bias is all zeros, so that case
  goes straight to ``torch.bmm`` instead of materializing zeros.
* **A caller that owns the mask can hand in the bias.** Building the bias is per-step work
  that does not depend on the layer: the ``KvSlab`` refreshes one persistent
  ``(B*H_kv, G, L)`` buffer in ``build_mask`` and passes it as ``bias=``, so a 28-layer
  decode materializes it once instead of 28 times. ``bias`` is exactly what
  ``additive_bias(mask, ...)`` would have returned; the numbers are unchanged.
* **softmax runs in the input dtype** (bf16 for the bf16 runs). Upcasting to fp32 and back
  costs a full-size round trip and buys nothing at ``atol=2e-2``; it is the same softmax
  the oracle has always done.

Mask convention is the contract's bool allow-mask (``base.py``, PITFALLS P-24):
``(B, 1, 1, L)`` or ``(B, 1, T, L)`` with ``T == 1``. A float mask is rejected with
``ValueError`` rather than reinterpreted: a bf16 ``-inf`` mask silently drifted up to
1.5e-1 elsewhere, so accepting one here would hide the caller bug. A row with no allowed
key at all (left-padded queries, ragged decode) gets its bias row zeroed before the
softmax, exactly like the oracle: ``exp(-inf) / 0`` is NaN, and a NaN in a padded row
leaks into the residual stream.

Decode only, on purpose: ``prefill()`` raises ``NotImplementedError``, so a
multi-token query fails loudly at the backend instead of silently getting decode math for a
prefill. Also out of scope: paged KV (the call takes a dense ``(B, H_kv, L, D)`` slab -- the
intended W3/W4 caller shape), fp8, and any kernel whose numerics differ from eager PyTorch
GEMM. The slab layout (``model/slab.py``) is what makes the shapes and the mask static across
steps, which is what a captured decode step over a fixed slab requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["HandBaddbmmBackend", "BACKEND", "additive_bias", "NEG_INF"]

NEG_INF = float("-inf")


class HandBaddbmmBackend:
    """Batched-GEMV decode: one ``baddbmm`` + one ``bmm`` for the whole batch."""

    name = "hand_baddbmm"

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = False,
        mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "hand_baddbmm is a decode-only backend (T == 1); a multi-token query is prefill. "
            "model/layers.py dispatches on T and does not fall back to another backend."
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if q.dim() != 4:
            raise ValueError(f"q must be (B, H, T, D), got shape {tuple(q.shape)}")
        if q.shape[2] != 1:
            raise ValueError(
                f"decode expects T == 1, got T={q.shape[2]}; hand_baddbmm is decode-only. "
                "Use prefill for T > 1 (a non-causal mask here would silently leak future "
                "tokens, P-19)."
            )
        if mask is not None and bias is not None:
            raise ValueError(
                "decode takes mask= (bool allow-mask) or bias= (the same mask already turned "
                "into the additive form), not both; passing both would apply it twice"
            )
        _check_shapes(q, k, v, mask)
        B, H, _, D = q.shape
        H_kv, L = k.shape[1], k.shape[2]
        G = H // H_kv  # GQA groups; never expanded into H real heads

        # scale on q (tiny), not on the scores (B, H, 1, L)
        qg = (q * (1.0 / math.sqrt(D))).reshape(B * H_kv, G, D)
        # (B, H_kv, L, D) -> (B*H_kv, D, L); a strided view for a contiguous k, no GQA copy
        kT = k.transpose(-2, -1).reshape(B * H_kv, D, L)
        vg = v.reshape(B * H_kv, L, D)

        if bias is not None:
            bias = _check_bias(bias, q.dtype, B, H_kv, G, L)
        elif mask is not None:
            bias = additive_bias(mask, q.dtype, B, H_kv, G)
        if bias is None:
            scores = torch.bmm(qg, kT)
        else:
            scores = torch.baddbmm(bias, qg, kT, beta=1.0, alpha=1.0)

        # bf16 (or fp32) softmax in place of an upcast round trip; see the module docstring
        probs = F.dropout(torch.softmax(scores, dim=-1), p=dropout_p, training=dropout_p > 0)
        out = torch.bmm(probs, vg)  # (B*H_kv, G, D)
        return out.reshape(B, H, 1, D)


def additive_bias(
    mask: torch.Tensor, dtype: torch.dtype, B: int, H_kv: int, G: int
) -> torch.Tensor:
    """Bool allow-mask -> ``(B*H_kv, G, L)`` additive bias, dead rows zeroed.

    The mask is shared by every query head and every group, so the bias is materialized
    once per (batch, kv head) and broadcast over ``G``. The materialization matches the
    score tensor's size; it is what lets the ``+ bias`` ride along inside ``baddbmm``.

    Public because ``KvSlab`` caches the result: it is identical for all layers, so the
    slab builds it once per decode step and passes it back as ``bias=``.
    """
    # A left-padded query may have no allowed key at all; zero that row instead of letting
    # exp(-inf)/0 produce NaN (the row's output is discarded). Same guard as the oracle.
    # Every step here runs on the small (B, 1, 1, L) tensor; only the final broadcast copy
    # below touches score-sized memory.
    alive = mask.any(dim=-1, keepdim=True)
    bias = torch.where(mask, 0.0, NEG_INF)  # (B_m, 1, 1, L)
    bias = torch.where(alive, bias, 0.0)
    L = bias.shape[-1]
    return bias.to(dtype).expand(B, H_kv, G, L).reshape(B * H_kv, G, L)


def _check_bias(
    bias: torch.Tensor, dtype: torch.dtype, B: int, H_kv: int, G: int, L: int
) -> torch.Tensor:
    """Validate an already-materialized additive bias (the ``KvSlab.bias`` fast path).

    ``(B*H_kv, G, L)`` is the fully materialized form; ``(B*H_kv, 1, L)`` is the same tensor
    with the GQA axis left as a broadcast dim (``baddbmm`` broadcasts it, so the values and
    the kernel are identical). Anything else is a caller bug, not something to reinterpret.
    """
    if bias.dtype is torch.bool:
        raise ValueError(
            "bias must be the additive form produced by additive_bias(), got a bool tensor; "
            "a bool allow-mask belongs in mask="
        )
    if tuple(bias.shape) not in ((B * H_kv, G, L), (B * H_kv, 1, L)):
        raise ValueError(
            "bias must be the additive form produced by additive_bias(), shape "
            f"{(B * H_kv, G, L)} or broadcastable {(B * H_kv, 1, L)}, got {tuple(bias.shape)}; "
            "a bool mask belongs in mask="
        )
    if bias.dtype != dtype:
        raise ValueError(f"bias dtype must match q ({dtype}), got {bias.dtype}")
    return bias


def _check_shapes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> None:
    """Fail loudly on shapes the math would otherwise broadcast into nonsense.

    The backend contract has no silent fallback (base.py); a mask that is one column short
    must not be allowed to broadcast, which is the P-05 failure mode.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dim() != 4:
            raise ValueError(f"q/k/v must be (B, H, T, D), got {name} with shape {tuple(t.shape)}")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"k/v sequence lengths differ: {k.shape[2]} vs {v.shape[2]}")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"k/v head counts differ: {k.shape[1]} vs {v.shape[1]}")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"q/k/v head_dim must match, got {q.shape[-1]}, {k.shape[-1]}, {v.shape[-1]}"
        )
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
        raise ValueError(
            "q/k/v batch sizes must match: "
            f"got q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"n_head ({q.shape[1]}) must be a multiple of n_kv_head ({k.shape[1]})")

    if mask is None:
        return
    if mask.dtype is not torch.bool:
        raise ValueError(
            "the hand_baddbmm backend takes only a bool allow-mask (True = may attend), got "
            f"{mask.dtype}; float additive masks are not part of the contract. With bf16 "
            "inputs a -inf float mask has silently drifted by up to 1.5e-1 (P-24), so it is "
            "rejected rather than reinterpreted. Build a bool (B, 1, 1, L) mask in "
            "model/layers.py instead."
        )
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(
            "the hand_baddbmm backend takes a bool allow-mask of shape (B, 1, 1, L) or "
            f"(B, 1, T, L), got {tuple(mask.shape)}"
        )
    if mask.size(0) not in (1, q.size(0)):
        raise ValueError(
            f"mask batch must be 1 or {q.size(0)}, got {mask.size(0)} (shape {tuple(mask.shape)})"
        )
    if mask.size(2) != 1:
        raise ValueError(
            f"decode is T == 1, so the mask query axis must be 1, got {tuple(mask.shape)}"
        )
    if mask.size(3) != k.size(2):
        raise ValueError(
            "mask must cover every key column: "
            f"expected last dim {k.size(2)} (past + current), got {mask.size(3)}"
        )


BACKEND = HandBaddbmmBackend()
