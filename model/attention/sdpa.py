"""The ``sdpa`` backend: PyTorch's fused ``F.scaled_dot_product_attention``.

This is the W2 baseline kernel and the reference the hand-written backends are
checked against. It owns the attention math only -- projection, QK-norm, RoPE,
KV-cache growth and the "which mask do I need" decision all stay in
``model/layers.py`` (see ``base.py``).

Two things are deliberately explicit here:

* ``F.scaled_dot_product_attention`` is looked up at call time through the
  ``torch.nn.functional`` module, never imported by name. ``tests/unit/
  test_sdpa_attention.py`` monkeypatches that attribute to assert the mask /
  ``is_causal`` / ``enable_gqa`` kwargs; a from-import would freeze the
  original function and turn those six contracts into always-green tests.
* A float (additive) mask is rejected instead of being forwarded. With bf16
  inputs and ``enable_gqa``, a ``-inf`` float mask put SDPA up to 1.5e-1 away
  from the oracle while the equivalent bool mask matched to 9e-4 (PITFALLS
  P-24). The contract is a bool allow-mask, so a float mask is a caller bug,
  not something to silently reinterpret.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["SdpaBackend", "BACKEND"]


def _check_mask(mask: torch.Tensor | None, q: torch.Tensor, k: torch.Tensor) -> None:
    """Validate the bool allow-mask contract before the kernel sees it."""
    if mask is None:
        return
    if mask.dtype is not torch.bool:
        raise ValueError(
            "the sdpa backend takes only a bool allow-mask (True = may attend), got "
            f"{mask.dtype}; float additive masks are not part of the contract. With bf16 "
            "inputs and enable_gqa, a -inf float mask put SDPA up to 1.5e-1 away from the "
            "oracle while the equivalent bool mask matched to 9e-4 (PITFALLS P-24). "
            "Build a bool (B, 1, T, L) mask in model/layers.py instead."
        )
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(
            "the sdpa backend takes a bool allow-mask of shape (B, 1, T, L), got "
            f"{tuple(mask.shape)}"
        )
    if mask.size(0) not in (1, q.size(0)) or mask.size(2) != q.size(2) or mask.size(3) != k.size(2):
        raise ValueError(
            "attention mask does not match the attention shapes: expected "
            f"(B={q.size(0)}, 1, T={q.size(2)}, L={k.size(2)}), got {tuple(mask.shape)}"
        )


class SdpaBackend:
    """``F.scaled_dot_product_attention`` with the project's mask convention."""

    name = "sdpa"

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
        _check_mask(mask, q, k)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=(q.size(1) != k.size(1)),
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        _check_mask(mask, q, k)
        # T == 1 needs neither is_causal nor a mask unless the batch is left-padded;
        # the mask, when present, says which of the L cached keys may be attended.
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
            enable_gqa=(q.size(1) != k.size(1)),
        )


BACKEND = SdpaBackend()
