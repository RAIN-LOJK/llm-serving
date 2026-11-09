"""W5 Step 2: the two Qwen3.5 norms (HF ``Qwen3_5RMSNorm`` / ``Qwen3_5RMSNormGated``).

Both are **not** ``model/layers.py::RMSNorm`` and the difference is silent if it slips:

* ``Qwen3_5RMSNorm`` multiplies by ``(1 + weight)`` and initializes ``weight`` to zeros, so
  a zero weight *is* plain RMSNorm and a one weight doubles the output. Its whole reduction
  runs in fp32 and the result is cast back to the input dtype.
* ``Qwen3_5RMSNormGated`` multiplies by ``weight`` directly (ones), casts the normed value
  back to the input dtype *before* the weight multiply, applies ``silu`` to the **gate** in
  fp32, and multiplies last. W6's linear-attention layers use it; W5 only pins it here.

The fp32 test uses fp16 inputs large enough that ``x**2`` overflows fp16 (65504) but not
fp32 -- a norm that computed in the input dtype would produce ``inf``/``0`` instead.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model.qwen35 import Qwen35RMSNorm, Qwen35RMSNormGated

DIM = 8
EPS = 1e-6


def _plain_rms(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Reference for ``(1 + w)`` with ``w = 0``: x * rsqrt(mean(x^2) + eps), fp32."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# ------------------------------------------------------------------ Qwen35RMSNorm


def test_qwen35_norm_weight_starts_at_zero():
    """The checkpoint's weight is a delta from a plain RMSNorm, not a scale."""
    norm = Qwen35RMSNorm(DIM, EPS)
    assert torch.equal(norm.weight, torch.zeros(DIM))
    assert norm.eps == EPS


def test_zero_weight_is_plain_rms():
    norm = Qwen35RMSNorm(DIM, EPS)
    x = torch.randn(3, 5, DIM)
    assert torch.allclose(norm(x), _plain_rms(x), atol=0.0, rtol=0.0)


def test_one_weight_doubles_the_output():
    """``(1 + w)``, not ``w``: a ones weight is a 2x scale, not an identity."""
    norm = Qwen35RMSNorm(DIM, EPS)
    norm.weight.data.fill_(1.0)
    x = torch.randn(2, 4, DIM)
    base = _plain_rms(x)
    assert torch.allclose(norm(x), 2.0 * base, atol=1e-6)
    assert not torch.allclose(norm(x), base, atol=1e-3)  # the wrong convention is visible


def test_norm_reduces_in_fp32_even_for_fp16_input():
    """fp16 ``x**2`` overflows here; a half-precision reduction would return zeros."""
    norm = Qwen35RMSNorm(DIM, EPS)
    x = torch.full((2, 3, DIM), 300.0, dtype=torch.float16)  # 300**2 = 90000 > 65504
    out = norm(x)

    assert out.dtype is torch.float16
    assert torch.isfinite(out).all()
    expected = _plain_rms(x).to(torch.float16)
    assert torch.equal(out, expected)
    # Teeth: the naive fp16 reduction would see inf variance and collapse to zero.
    naive = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))
    assert not torch.isfinite(x.pow(2).mean(-1, keepdim=True)).all()
    assert not torch.allclose(out, naive)


def test_norm_is_per_row_and_returns_the_input_dtype():
    norm = Qwen35RMSNorm(DIM, EPS)
    x = torch.randn(3, 4, DIM)
    x[1] *= 7.0  # a different scale per (batch, position) row
    out = norm(x)

    assert out.dtype is x.dtype
    for b in range(3):
        for t in range(4):
            assert torch.allclose(out[b, t], _plain_rms(x[b, t]), atol=0.0, rtol=0.0)


# ------------------------------------------------------------ Qwen35RMSNormGated


def test_gated_norm_weight_starts_at_one():
    norm = Qwen35RMSNormGated(DIM, EPS)
    assert torch.equal(norm.weight, torch.ones(DIM))
    assert norm.variance_epsilon == EPS


def test_gated_norm_multiplies_by_silu_of_the_gate():
    norm = Qwen35RMSNormGated(DIM, EPS)
    hidden = torch.randn(2, 3, DIM)
    gate = torch.randn(2, 3, DIM)

    out = norm(hidden, gate)
    expected = _plain_rms(hidden) * torch.ones(DIM) * F.silu(gate.float())

    assert out.dtype is hidden.dtype
    assert torch.allclose(out, expected, atol=1e-6)

    # The silu is on the gate, not on the normed hidden state: gate=0 zeroes the output
    # even though the normed hidden state is non-zero.
    zero_gate = norm(hidden, torch.zeros_like(gate))
    assert torch.equal(zero_gate, torch.zeros_like(zero_gate))
    assert _plain_rms(hidden).abs().min() > 0


def test_gated_norm_is_not_zero_centered():
    """``weight`` multiplies directly: a ones weight is the identity, not a 2x scale."""
    norm = Qwen35RMSNormGated(DIM, EPS)
    hidden = torch.randn(2, 3, DIM)
    gate = torch.ones(2, 3, DIM)

    out = norm(hidden, gate)
    plain = _plain_rms(hidden)
    assert torch.allclose(out, plain * F.silu(gate.float()), atol=1e-6)
    assert not torch.allclose(out, 2.0 * plain * F.silu(gate.float()), atol=1e-3)


def test_gated_norm_matches_hf_op_order_in_bf16():
    """norm -> to(input_dtype) -> * weight -> * silu(gate.float()) -> to(input_dtype)."""
    norm = Qwen35RMSNormGated(DIM, EPS)
    hidden = torch.randn(2, 3, DIM).to(torch.bfloat16)
    gate = torch.randn(2, 3, DIM) * 0.1

    out = norm(hidden, gate)

    hidden32 = hidden.float()
    normed = hidden32 * torch.rsqrt(hidden32.pow(2).mean(-1, keepdim=True) + EPS)
    expected = norm.weight * normed.to(torch.bfloat16)
    expected = expected * F.silu(gate.float())
    expected = expected.to(torch.bfloat16)
    assert torch.equal(out, expected)

    # Teeth: applying silu in bf16 first would round differently.
    coarse = norm.weight * normed.to(torch.bfloat16) * F.silu(gate.to(torch.bfloat16))
    assert not torch.equal(out, coarse.to(torch.bfloat16))


@pytest.mark.parametrize("cls", [Qwen35RMSNorm, Qwen35RMSNormGated])
def test_norm_eps_is_used_not_a_default(cls):
    loose = cls(DIM, 1.0)  # eps = 1 flattens the normalization noticeably
    tight = cls(DIM, 1e-6)
    x = torch.randn(1, 2, DIM) * 0.01
    if cls is Qwen35RMSNormGated:
        gate = torch.ones(1, 2, DIM)
        assert not torch.allclose(loose(x, gate), tight(x, gate))
    else:
        assert not torch.allclose(loose(x), tight(x))
