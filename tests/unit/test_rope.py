"""RoPE contract: GPT-NeoX pairing, position zero identity, table shapes."""

from __future__ import annotations

import torch

from model.layers import apply_rotary_emb, precompute_rope_freqs


def test_freq_table_shapes_and_origin():
    sin, cos = precompute_rope_freqs(dim=8, max_seq_len=4, theta=10000.0)
    assert sin.shape == (4, 4)
    assert cos.shape == (4, 4)
    assert torch.allclose(sin[0], torch.zeros(4))
    assert torch.allclose(cos[0], torch.ones(4))


def test_position_zero_is_identity():
    sin, cos = precompute_rope_freqs(dim=8, max_seq_len=4)
    x = torch.randn(2, 1, 2, 8)
    y = apply_rotary_emb(x, cos[:1], sin[:1])
    assert torch.allclose(y, x, atol=1e-6)


def test_gpt_neox_convention():
    """First half pairs with second half; this is NOT the interleaved convention."""
    sin, cos = precompute_rope_freqs(dim=8, max_seq_len=4)
    x = torch.zeros(1, 1, 1, 8)
    x[..., 0] = 1.0
    y = apply_rotary_emb(x, cos[1:2], sin[1:2])
    expected = torch.zeros(8)
    expected[0] = cos[1, 0]
    expected[4] = sin[1, 0]
    assert torch.allclose(y[0, 0, 0], expected, atol=1e-6)


def test_rotate_half_formula():
    sin, cos = precompute_rope_freqs(dim=8, max_seq_len=4)
    x = torch.randn(1, 1, 1, 8)
    y = apply_rotary_emb(x, cos[2:3], sin[2:3])
    x1, x2 = x[..., :4], x[..., 4:]
    expected = torch.cat(
        [x1 * cos[2, :4] - x2 * sin[2, :4], x1 * sin[2, :4] + x2 * cos[2, :4]], dim=-1
    )
    assert torch.allclose(y, expected, atol=1e-6)
