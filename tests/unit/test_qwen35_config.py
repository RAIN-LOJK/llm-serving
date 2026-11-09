"""W5 Step 0: the Qwen3.5-0.8B config facts and the fields they force into ``MiniLLMConfig``.

The first test reads the local snapshot's ``config.json`` (metadata only, no weights, no
network) and pins the numbers §2 of ``docs/w5-plan.md`` calls the fact baseline. It skips
when the snapshot is absent: the fast gate must stay usable on a machine without weights.

The rest are weight-free guards on defaults and on the two derived-invariant checks
(``layer_types`` length, ``sum(mrope_section) == rotary_dim // 2``). Defaults matter: the
Qwen3-0.6B path must keep 28 full-attention layers and a full-width RoPE table.
"""

from __future__ import annotations

import json

import pytest

from model.config import MiniLLMConfig
from model.loading import cached_snapshot

MODEL = "Qwen/Qwen3.5-0.8B"
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23]


def _text_config() -> dict:
    snapshot = cached_snapshot(MODEL)
    if snapshot is None or not (snapshot / "config.json").exists():
        pytest.skip(f"no local snapshot for {MODEL}; run the HF download first")
    return json.loads((snapshot / "config.json").read_text())["text_config"]


# ------------------------------------------------------------------ the snapshot facts


def test_snapshot_layer_types_are_lllf_times_six():
    text = _text_config()
    assert text["num_hidden_layers"] == 24
    assert text["layer_types"] == (["linear_attention"] * 3 + ["full_attention"]) * 6
    assert [i for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"] == (
        FULL_ATTENTION_LAYERS
    )


def test_snapshot_head_and_mlp_dimensions():
    text = _text_config()
    assert text["head_dim"] == 256  # not hidden_size // num_attention_heads == 128
    assert text["hidden_size"] == 1024
    assert text["num_attention_heads"] == 8
    assert text["num_key_value_heads"] == 2
    assert text["intermediate_size"] == 3584


def test_snapshot_norm_dtype_and_vocab_facts():
    text = _text_config()
    assert text["rms_norm_eps"] == 1e-6
    assert text["attention_bias"] is False
    assert text["tie_word_embeddings"] is True
    assert text["vocab_size"] == 248320
    assert text["attn_output_gate"] is True


def test_snapshot_rope_is_partial_and_text_mrope_degenerates():
    text = _text_config()
    rope = text["rope_parameters"]
    assert rope["rope_type"] == "default"
    assert rope["rope_theta"] == 10_000_000
    assert rope["partial_rotary_factor"] == 0.25
    assert rope["mrope_interleaved"] is True
    assert rope["mrope_section"] == [11, 11, 10]

    rotary_dim = int(text["head_dim"] * rope["partial_rotary_factor"])
    assert rotary_dim == 64  # 256 * 0.25; the table has 32 frequencies
    assert sum(rope["mrope_section"]) == rotary_dim // 2 == 32


def test_snapshot_linear_attention_fields_are_carried_for_w6():
    text = _text_config()
    assert text["linear_num_key_heads"] == 16
    assert text["linear_num_value_heads"] == 16
    assert text["linear_key_head_dim"] == 128
    assert text["linear_value_head_dim"] == 128
    assert text["linear_conv_kernel_dim"] == 4


def test_snapshot_values_build_a_valid_config():
    """The facts are internally consistent: MiniLLMConfig accepts them and derives 64."""
    text = _text_config()
    rope = text["rope_parameters"]
    cfg = MiniLLMConfig(
        vocab_size=text["vocab_size"],
        max_position_embeddings=text["max_position_embeddings"],
        n_embd=text["hidden_size"],
        n_head=text["num_attention_heads"],
        n_kv_head=text["num_key_value_heads"],
        n_layer=text["num_hidden_layers"],
        n_inner=text["intermediate_size"],
        norm_eps=text["rms_norm_eps"],
        bias=text["attention_bias"],
        tie_word_embeddings=text["tie_word_embeddings"],
        head_dim=text["head_dim"],
        rope_theta=rope["rope_theta"],
        partial_rotary_factor=rope["partial_rotary_factor"],
        mrope_section=rope["mrope_section"],
        layer_types=text["layer_types"],
        attn_output_gate=text["attn_output_gate"],
        linear_num_key_heads=text["linear_num_key_heads"],
        linear_num_value_heads=text["linear_num_value_heads"],
        linear_key_head_dim=text["linear_key_head_dim"],
        linear_value_head_dim=text["linear_value_head_dim"],
        linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
    )
    assert cfg.rotary_dim == 64
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[0] == "linear_attention"


# ----------------------------------------------------------------------- defaults


def test_defaults_keep_qwen3_full_attention_behaviour():
    """Qwen3-0.6B shape: 28 layers, head_dim 128, no partial RoPE, no M-RoPE."""
    cfg = MiniLLMConfig(
        vocab_size=151936,
        max_position_embeddings=40960,
        n_embd=1024,
        n_head=16,
        n_kv_head=8,
        n_layer=28,
        head_dim=128,
        rope_theta=1_000_000,
        qk_norm=True,
    )
    assert cfg.layer_types == ["full_attention"] * 28
    assert cfg.attn_output_gate is False
    assert cfg.partial_rotary_factor == 1.0
    assert cfg.mrope_section is None
    assert cfg.rotary_dim == 128  # == head_dim, so the table is unchanged
    assert (
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
        cfg.linear_conv_kernel_dim,
    ) == (None, None, None, None, None)


def test_toy_default_is_also_full_attention():
    cfg = MiniLLMConfig()
    assert cfg.n_layer == 4
    assert cfg.layer_types == ["full_attention"] * 4
    assert cfg.rotary_dim == 128 // 4  # head_dim None -> n_embd // n_head


def test_mixed_layer_types_are_accepted():
    cfg = MiniLLMConfig(
        n_layer=4, layer_types=["linear_attention"] * 3 + ["full_attention"]
    )
    assert cfg.layer_types == ["linear_attention"] * 3 + ["full_attention"]


# ----------------------------------------------------------------------- validation


def test_layer_types_length_must_equal_n_layer():
    with pytest.raises(ValueError, match=r"layer_types must have exactly n_layer=3"):
        MiniLLMConfig(n_layer=3, layer_types=["full_attention"] * 2)


def test_unknown_layer_type_is_rejected():
    with pytest.raises(ValueError, match=r"layer_types\[1\].*sliding_attention"):
        MiniLLMConfig(
            n_layer=2, layer_types=["full_attention", "sliding_attention"]
        )


def test_mrope_section_must_cover_half_the_rotary_dim():
    common = dict(n_embd=1024, n_head=8, head_dim=256, partial_rotary_factor=0.25)
    with pytest.raises(ValueError, match=r"mrope_section=.*rotary_dim=64"):
        MiniLLMConfig(n_layer=1, mrope_section=[11, 11, 11], **common)

    cfg = MiniLLMConfig(n_layer=1, mrope_section=[11, 11, 10], **common)
    assert cfg.rotary_dim == 64


def test_rotary_dim_falls_back_to_hidden_over_heads():
    cfg = MiniLLMConfig(n_embd=128, n_head=4, head_dim=None, partial_rotary_factor=0.25)
    assert cfg.rotary_dim == int((128 // 4) * 0.25) == 8

    full = MiniLLMConfig(n_embd=128, n_head=4, head_dim=None)
    assert full.rotary_dim == 32
