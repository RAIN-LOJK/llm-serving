"""Cache byte accounting has to be exact, otherwise W0's ratio is fiction."""

from __future__ import annotations

import types

import pytest
import torch

from bench.cache_usage import cache_usage, cache_usage_mb


def test_legacy_tuple_cache_counts_kv():
    k = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
    v = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
    usage = cache_usage([(k, v)])
    assert usage["kv"] == 2 * k.numel() * 2
    assert usage["state"] == 0
    assert usage["total"] == usage["kv"]
    assert usage["n_tensors"] == 2


def test_named_layers_split_kv_state_conv():
    kv = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    state = torch.zeros(2, 4, 4, dtype=torch.float32)
    conv = torch.zeros(1, 24, 4, dtype=torch.float32)
    full = types.SimpleNamespace(keys=kv, values=kv.clone(), dtype=torch.bfloat16, device="cpu")
    linear = types.SimpleNamespace(
        conv_states={0: conv},
        recurrent_states={0: state},
        dtype=torch.float32,
        device="cpu",
    )
    usage = cache_usage(types.SimpleNamespace(layers=[full, linear]))
    assert usage["kv"] == 2 * kv.numel() * 2
    assert usage["state"] == state.numel() * 4
    assert usage["conv"] == conv.numel() * 4
    assert usage["total"] == usage["kv"] + usage["state"] + usage["conv"]


def test_shared_tensor_is_counted_once():
    shared = torch.zeros(1, 2, 4, 8)
    layer = types.SimpleNamespace(keys=shared, values=shared)
    assert cache_usage(types.SimpleNamespace(layers=[layer]))["kv"] == shared.numel() * 4


def test_empty_cache_is_zero():
    assert cache_usage(None)["total"] == 0
    assert cache_usage_mb(None)["cache_mb"] == 0.0


@pytest.mark.parametrize("budget", [1])
def test_tiny_qwen35_hybrid_cache_matches_config(budget):
    """The real HF hybrid cache must land exactly on the analytic numbers."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 32, (1, 6)), use_cache=True)

    usage = cache_usage(out.past_key_values)
    # 1 full layer, 2 kv heads, head_dim 8, 6 tokens, fp32
    assert usage["kv"] == 2 * 2 * 8 * 6 * 4
    # 3 linear layers, 2 value heads, 4x4 state, fp32
    assert usage["state"] == 3 * 2 * 4 * 4 * 4
    # 3 linear layers, conv window over 2*d_k + d_v = 24 channels, kernel 4, fp32
    assert usage["conv"] == 3 * 24 * 4 * 4
    assert transformers.__version__  # imported for the skip marker side effect
