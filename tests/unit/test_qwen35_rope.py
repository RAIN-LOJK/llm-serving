"""W5 Step 1: partial-RoPE against HF's text rotary embedding.

Qwen3.5-0.8B has ``head_dim=256`` but rotates only ``rotary_dim = int(256 * 0.25) = 64``
dimensions, so the cos/sin table has 32 frequencies (not 128) and the remaining 192
dimensions pass through. For text-only input the three M-RoPE position streams are equal,
which makes HF's table exactly the 1-D 64-wide table duplicated in both halves.

No weights and no network: the HF side is the rotary *module*, constructed from a config.
"""

from __future__ import annotations

import pytest
import torch

configuration_qwen3_5 = pytest.importorskip(
    "transformers.models.qwen3_5.configuration_qwen3_5"
)
modeling_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from model.layers import (  # noqa: E402  (import after the skip guard)
    apply_rotary_emb,
    apply_rotary_emb_partial,
    rope_cos_sin,
)

HEAD_DIM = 256
ROTARY_DIM = 64
ROPE_PARAMS = {
    "rope_type": "default",
    "rope_theta": 10_000_000,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
}


def _hf_cos_sin(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """HF text rotary table, (length, rotary_dim), duplicated halves."""
    config = configuration_qwen3_5.Qwen3_5TextConfig(
        head_dim=HEAD_DIM,
        hidden_size=1024,
        rope_parameters=dict(ROPE_PARAMS),
    )
    rotary = modeling_qwen3_5.Qwen3_5TextRotaryEmbedding(config)
    position_ids = torch.arange(length)[None, None, :].expand(3, 1, length)
    cos, sin = rotary(torch.zeros(1, length, HEAD_DIM), position_ids)
    return cos[0], sin[0]


# --------------------------------------------------------------------- table values


def test_partial_table_shape_and_values_match_hf():
    length = 11
    cos, sin = rope_cos_sin(
        HEAD_DIM, 1e7, 0, length, rotary_dim=ROTARY_DIM
    )
    assert cos.shape == (length, ROTARY_DIM // 2) == (length, 32)
    assert sin.shape == (length, ROTARY_DIM // 2)

    hf_cos, hf_sin = _hf_cos_sin(length)
    assert hf_cos.shape == (length, ROTARY_DIM)
    # HF duplicates the half-table (``cat(freqs_thw, freqs_thw)``); ours is that half.
    assert torch.equal(cos, hf_cos[:, : ROTARY_DIM // 2])
    assert torch.equal(sin, hf_sin[:, : ROTARY_DIM // 2])


def test_text_mrope_degenerates_to_1d_rope():
    """G3: the three position streams are equal, so HF cos/sin == cat([1D(64), 1D(64)])."""
    length = 9
    cos, sin = rope_cos_sin(HEAD_DIM, 1e7, 0, length, rotary_dim=ROTARY_DIM)
    hf_cos, hf_sin = _hf_cos_sin(length)

    full_cos = torch.cat([cos, cos], dim=-1)
    full_sin = torch.cat([sin, sin], dim=-1)
    assert full_cos.shape == hf_cos.shape
    assert (full_cos - hf_cos).abs().max().item() == 0.0
    assert (full_sin - hf_sin).abs().max().item() == 0.0


def test_full_width_table_matches_the_explicit_rotary_dim():
    """``rotary_dim=None`` keeps the old arithmetic: same bits as rotary_dim=head_dim."""
    length = 5
    old_cos, old_sin = rope_cos_sin(8, 10_000.0, 2, length)
    new_cos, new_sin = rope_cos_sin(8, 10_000.0, 2, length, rotary_dim=8)
    assert old_cos.shape == (length, 4)
    assert torch.equal(old_cos, new_cos)
    assert torch.equal(old_sin, new_sin)


# ------------------------------------------------------------------- applying it


def test_tail_dimensions_are_untouched():
    length = 6
    cos, sin = rope_cos_sin(HEAD_DIM, 1e7, 0, length, rotary_dim=ROTARY_DIM)
    x = torch.randn(2, length, 3, HEAD_DIM)

    out = apply_rotary_emb_partial(x, cos, sin, ROTARY_DIM)

    assert out.shape == x.shape
    assert torch.equal(out[..., ROTARY_DIM:], x[..., ROTARY_DIM:])
    # The rotated slice is exactly the existing GPT-NeoX rotation over 64 dims.
    assert torch.equal(
        out[..., :ROTARY_DIM], apply_rotary_emb(x[..., :ROTARY_DIM], cos, sin)
    )


def test_rotary_dim_covering_the_head_equals_apply_rotary_emb():
    length = 4
    cos, sin = rope_cos_sin(64, 10_000.0, 0, length, rotary_dim=64)
    x = torch.randn(2, length, 3, 64)
    assert torch.equal(
        apply_rotary_emb_partial(x, cos, sin, 64), apply_rotary_emb(x, cos, sin)
    )


def test_per_row_tables_match_row_by_row_computation():
    """W6 decode hands cos/sin shaped (B, T, D/2); each row sits at its own position."""
    batch, length = 2, 4
    table_len = length + 3
    cos_all, sin_all = rope_cos_sin(
        HEAD_DIM, 1e7, 0, table_len, rotary_dim=ROTARY_DIM
    )
    positions = torch.stack(
        [torch.arange(length), torch.arange(3, 3 + length)]
    )  # (B, T), different offset per row
    cos_rows = cos_all[positions]  # (B, T, 32)
    sin_rows = sin_all[positions]
    assert cos_rows.shape == (batch, length, ROTARY_DIM // 2)

    x = torch.randn(batch, length, 2, HEAD_DIM)
    out = apply_rotary_emb_partial(x, cos_rows, sin_rows, ROTARY_DIM)

    for row in range(batch):
        expected = apply_rotary_emb_partial(
            x[row : row + 1], cos_all[positions[row]], sin_all[positions[row]], ROTARY_DIM
        )
        assert torch.equal(out[row : row + 1], expected)


def test_partial_rotation_rejects_a_table_of_the_wrong_width():
    """A (T, head_dim/2) table is the full-width one; slicing it as partial is a bug."""
    length = 3
    cos, sin = rope_cos_sin(16, 10_000.0, 0, length)  # (T, 8) = full width of 16
    x = torch.randn(1, length, 1, 8)
    with pytest.raises(ValueError, match=r"rotary_dim=4"):
        apply_rotary_emb_partial(x, cos, sin, 4)
