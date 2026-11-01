"""Attention backend contract.

The policy is frozen (``docs/decisions.md`` D4): prefill (T > 1) runs on
``sdpa``, decode (T == 1) runs on ``hand_baddbmm``. Those are the two
constants in ``model/attention/__init__.py``; there is no registry and no
configuration value to select anything else.

**Two entry points, not one.** Prefill is compute-bound and wants a fused
kernel; decode is KV-bandwidth / dispatch bound and wants a batched GEMV
shape. One ``attention(q, k, v)`` signature forces the wrong tool on one of
the two — measured: the same PyTorch math is 2.2x slower than flash on
prefill-2048 but 3x *faster* than the SDPA math fallback on decode. So
``prefill`` and ``decode`` are separate methods.

**No silent fallback.** A backend that cannot serve a shape raises.
``model/layers.py`` dispatches on ``T`` directly to one constant backend and
never catches an error to try the other: there is no second implementation on
the run path to fall back to. ``hand_baddbmm.prefill`` raises for a
multi-token query, and that error is a bug in the caller, not something to
retry elsewhere.

Mask convention: a **bool allow-mask** of shape ``(B, 1, T, L)``, ``True`` means
"may attend". Float additive masks containing ``-inf`` are deliberately not part
of the contract: with bf16 inputs and ``enable_gqa``, SDPA produced outputs up
to 1.5e-1 away from the oracle with a ``-inf`` float mask, while the identical
call with the equivalent bool mask matched to 9e-4 (PITFALLS P-24).

Backends own the attention math only. KV-cache growth, RoPE, positions and the
"which mask do I need" decision stay in ``model/layers.py`` so the two
implementations cannot drift in different directions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

__all__ = ["AttentionBackend"]


@runtime_checkable
class AttentionBackend(Protocol):
    """Implemented by ``model/attention/<name>.py``.

    Shapes (already projected, RoPE applied, GQA *not* expanded):
        q: (B, H,  T, D)   k, v: (B, H_kv, L, D)
    where ``L = past_len + T``. Return: (B, H, T, D).
    """

    name: str

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = False,
        mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor: ...

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor: ...
