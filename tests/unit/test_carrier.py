"""The analytic account is arithmetic, so it gets arithmetic tests.

If these drift, ``docs/carrier_estimates.json`` and the W0 crossover figure drift with
them, and nobody notices until the probe run disagrees by 8x.
"""

from __future__ import annotations

import pytest

from bench import carrier


def pure_config() -> dict:
    return {
        "num_hidden_layers": 4,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "max_position_embeddings": 64,
    }


def hybrid_config() -> dict:
    return {
        "num_hidden_layers": 4,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "vocab_size": 128,
        "max_position_embeddings": 64,
    }


def test_pure_attention_kv_per_token():
    est = carrier.estimate(pure_config(), dtype="bfloat16")
    # 2 (K,V) * 2 kv heads * 8 dim * 2 bytes * 4 layers
    assert est["kv_bytes_per_token"] == 256
    assert est["n_full_layers"] == 4
    assert est["n_linear_layers"] == 0
    assert est["fixed_bytes_per_sequence"] == 0


def test_hybrid_state_and_conv():
    est = carrier.estimate(hybrid_config(), dtype="bfloat16")
    # 3 linear layers * 2 v heads * 4 * 4 * 4 bytes (fp32 state)
    assert est["recurrent_bytes_per_sequence"] == 3 * 2 * 4 * 4 * 4
    # channels = 2 * (2*4) + (2*4) = 24; 3 layers * 4 taps * 24 * 2 bytes
    assert est["conv_bytes_per_sequence"] == 3 * 4 * 24 * 2
    assert est["kv_bytes_per_token"] == 2 * 2 * 8 * 2 * 1
    assert est["fixed_bytes_per_sequence"] == (
        est["recurrent_bytes_per_sequence"] + est["conv_bytes_per_sequence"]
    )


def test_nested_text_config_is_unwrapped():
    est = carrier.estimate({"text_config": hybrid_config()}, dtype="bfloat16")
    assert est["n_linear_layers"] == 3


def test_crossover_and_ratio():
    cand = carrier.estimate(hybrid_config(), dtype="bfloat16")
    ref = carrier.estimate(pure_config(), dtype="bfloat16")
    expected = cand["fixed_bytes_per_sequence"] / (ref["kv_bytes_per_token"] - cand["kv_bytes_per_token"])
    assert carrier.crossover_tokens(cand, ref) == pytest.approx(expected)
    assert carrier.asymptotic_ratio(cand, ref) == pytest.approx(ref["kv_bytes_per_token"] / cand["kv_bytes_per_token"])


def test_no_crossover_when_kv_is_not_cheaper():
    cand = carrier.estimate(pure_config(), dtype="bfloat16")
    assert carrier.crossover_tokens(cand, cand) is None


def test_layer_type_length_must_match():
    cfg = hybrid_config()
    cfg["layer_types"] = ["linear_attention"]
    with pytest.raises(ValueError):
        carrier.estimate(cfg)


def test_human_bytes_is_binary():
    assert carrier.human_bytes(1024) == "1.00 KiB"
    assert carrier.human_bytes(2**20) == "1.00 MiB"
    assert carrier.human_bytes(None) == "—"
