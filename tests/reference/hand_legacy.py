"""``hand_legacy``: the hand-written attention math, frozen as the numerical oracle.

This lives under ``tests/reference/`` on purpose (D4): it is *not* on the run path anymore,
so ``model/`` no longer contains the old implementation, but it stays an independent oracle
for the two shipped backends. A golden fixture only pins one fixed path; if both paths
drift the same way, golden cannot see it -- an oracle can.

Provenance: this is steps 4-8 of ``Attention.forward`` as it stood before the SDPA switch
(git ``26b268d^``, i.e. ``fa7bd54``). Other backends (``sdpa``, ``hand_baddbmm``, a future
fused kernel) are judged *against* this file, so it is deliberately not optimized:

* GQA expands with ``k.repeat_interleave(n_reps, dim=1)`` and materializes ``(B, H, L, D)``
  K/V. That allocation is what makes this backend slow, and it is exactly what makes it a
  usable reference: every head multiplies a real (not strided) K.
* Scores are ``q @ k.transpose(-2, -1) / math.sqrt(head_dim)`` -- a materialized ``T x L``
  tensor, divided rather than multiplied by a folded scale.
* The additive mask is built in ``q.dtype``. The historical comment says why: a fp32 mask
  promotes bf16 scores to fp32, and the following ``attn @ v`` is then fp32 x bf16, which
  raises. The W2 bf16 runs hit exactly that.
* A fully masked softmax row (left-padded queries, ragged decode) is zeroed before the
  softmax, mirroring the historical ``mask.isinf().all(-1, True)`` guard. Without it the
  row is ``exp(-inf) / 0`` -> NaN, and NaN in a padded row leaks into later layers.
* ``F.dropout(..., p=dropout_p, training=dropout_p > 0)``: dropout follows the argument,
  not ``module.training``, because a backend is a plain object with no mode of its own.

Do not "fix" any of the above. A change here silently moves the oracle and makes every
cross-backend comparison meaningless; if a faster formulation is wanted, add a new backend.

Mask convention is the contract's bool allow-mask ``(B, 1, T, L)`` (``True`` = may attend).
The historical ``(B, L)`` 0/1 convention is converted at this boundary, and ``is_causal``
builds the historical ``torch.triu(..., diagonal=past_len + 1)`` mask internally with
``past_len = L - T``; both are algebraically the same mask, so callers never pass a float
``-inf`` mask (see base.py, PITFALLS P-24). With ``mask is None and not is_causal`` no mask
is built at all: every query attends every key (decode against an equal-length cache).

Not covered by this oracle (see ``tests/unit/test_attention_legacy_backend.py``):
contiguous KV slabs / paging, CUDA graphs, fp8 or other low-precision paths, and any kernel
whose numerics differ from eager PyTorch matmul. It also says nothing about speed -- it is
the slow path on purpose.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["HandLegacyBackend", "BACKEND"]


class HandLegacyBackend:
    """The pre-SDPA manual attention, kept bit-for-bit as the numerical reference."""

    name = "hand_legacy"

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
        return self._attend(q, k, v, is_causal=is_causal, mask=mask, dropout_p=dropout_p)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        # T == 1: the single query may attend every cached key, so there is no causal mask
        # to build (the historical path also left `mask` untouched when T == 1 and no pad).
        if q.dim() == 4 and q.shape[2] != 1:
            raise ValueError(
                f"decode expects T == 1, got T={q.shape[2]}; use prefill for T > 1 "
                "(a non-causal mask here would silently leak future tokens, P-19)"
            )
        return self._attend(q, k, v, is_causal=False, mask=mask, dropout_p=dropout_p)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        mask: torch.Tensor | None,
        dropout_p: float,
    ) -> torch.Tensor:
        _check_shapes(q, k, v, is_causal=is_causal, mask=mask)
        B, H, T, head_dim = q.shape
        L = k.shape[2]

        # (historical step 4) GQA: repeat K/V heads. Materializes on purpose.
        if k.shape[1] != H:
            n_reps = H // k.shape[1]
            k = k.repeat_interleave(n_reps, dim=1)
            v = v.repeat_interleave(n_reps, dim=1)

        # (historical step 5) scores: Q @ K^T / sqrt(head_dim)
        attn = q @ k.transpose(-2, -1) / math.sqrt(head_dim)

        # (historical step 6) causal mask, then the padding mask, both in q.dtype.
        additive = None
        if is_causal:
            past_len = L - T
            additive = torch.triu(
                torch.full((T, L), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=past_len + 1,
            )
        if mask is not None:
            pad = torch.where(mask, 0.0, float("-inf")).to(q.dtype)
            additive = pad if additive is None else additive + pad
        if additive is not None:
            # A left-padded query can have no allowed key at all; exp(-inf)/0 would be NaN.
            # The historical fallback was to zero that row (the row's output is discarded).
            additive = torch.where(additive.isinf().all(-1, keepdim=True), 0.0, additive)
            attn = attn + additive

        # (historical step 7) softmax + dropout
        attn = F.dropout(F.softmax(attn, dim=-1), p=dropout_p, training=dropout_p > 0)

        # (historical step 8) y = attn @ V. The transpose/reshape back to (B, T, C) and the
        # output projection stay in model/layers.py, outside the backend contract.
        return attn @ v


def _check_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    mask: torch.Tensor | None,
) -> None:
    """Fail loudly on a shape the math would otherwise broadcast into nonsense.

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
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"n_head ({q.shape[1]}) must be a multiple of n_kv_head ({k.shape[1]})"
        )
    if is_causal and k.shape[2] < q.shape[2]:
        raise ValueError(
            f"causal prefill needs L >= T, got L={k.shape[2]} and T={q.shape[2]}"
        )
    if mask is not None:
        if mask.dtype is not torch.bool:
            raise TypeError(
                "mask must be a bool allow-mask (True = may attend), "
                f"got dtype {mask.dtype} (float -inf masks are not part of the contract, P-24)"
            )
        if mask.shape[-1] != k.shape[2]:
            raise ValueError(
                "mask must cover every key column: "
                f"expected last dim {k.shape[2]} (past + current), got {mask.shape[-1]}"
            )
        if mask.shape[-2] not in (1, q.shape[2]):
            raise ValueError(
                f"mask must be (B, 1, T, L) or (B, 1, 1, L); got {tuple(mask.shape)}"
            )


BACKEND = HandLegacyBackend()
