"""W5 Step 4: the loader's Qwen3.5 naming contract and its guards.

Three layers of coverage, cheapest first:

1. **Synthetic keys** — no weights, no cache: both prefix conventions map, and the
   out-of-scope heads (``mtp.``, ``visual.``) are dropped rather than half-mapped.
2. **Synthetic tiny snapshot** — a real ``config.json`` plus a fake layer in
   ``model.safetensors``: ``load_layer_weights`` returns repo-relative keys, and missing
   or misshaped keys fail loudly instead of producing a partial block.
3. **The local 0.8B snapshot** (marked ``hf``, skipped without it) — real layer shapes,
   and the whole-model load still fails with the W6 ``linear_attn.*`` gap and no partial
   ``load_state_dict``.

``load_layer_weights`` returns keys *relative to the block* (``self_attn.q_proj.weight``),
so a W5 block can do ``block.load_state_dict(load_layer_weights(...), strict=True)``.
"""

from __future__ import annotations

import pytest
import torch

from model.loading import cached_snapshot, config_from_hf, load_checkpoint, load_layer_weights, map_hf_key
from model.minillm import MiniLLM

MODEL = "Qwen/Qwen3.5-0.8B"
QWEN3_MODEL = "Qwen/Qwen3-0.6B"


# --------------------------------------------------------------------- key mapping


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "layers.3.self_attn.q_proj.weight",
        ),
        ("model.layers.3.self_attn.q_proj.weight", "layers.3.self_attn.q_proj.weight"),
        ("model.language_model.embed_tokens.weight", "token_embedding.weight"),
        ("model.embed_tokens.weight", "token_embedding.weight"),
        ("model.language_model.norm.weight", "norm.weight"),
        ("model.norm.weight", "norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
        # The per-layer RoPE cache is not a parameter; dropping it is already the contract.
        ("model.language_model.layers.0.self_attn.rotary_emb.inv_freq", None),
        ("model.layers.0.self_attn.rotary_emb.inv_freq", None),
    ],
)
def test_map_hf_key_handles_both_prefixes(key, expected):
    assert map_hf_key(key) == expected


@pytest.mark.parametrize(
    "key",
    [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.merger.linear_fc1.weight",
    ],
)
def test_mtp_and_vision_keys_are_dropped(key):
    """D2: the MTP head and the vision tower have no module in model/; do not map them."""
    assert map_hf_key(key) is None


# ------------------------------------------------------ synthetic snapshot fixture


TINY_TEXT = dict(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    attn_output_gate=True,  # q_proj doubles: 2 heads * 8 dims * 2 (query + gate)
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
    },
)

# Shapes a W5 full-attention block must have for TINY_TEXT.
EXPECTED_TINY = {
    "input_layernorm.weight": (16,),
    "post_attention_layernorm.weight": (16,),
    "self_attn.q_proj.weight": (32, 16),
    "self_attn.k_proj.weight": (8, 16),
    "self_attn.v_proj.weight": (8, 16),
    "self_attn.o_proj.weight": (16, 16),
    "self_attn.q_norm.weight": (8,),
    "self_attn.k_norm.weight": (8,),
    "mlp.gate_proj.weight": (32, 16),
    "mlp.up_proj.weight": (32, 16),
    "mlp.down_proj.weight": (16, 32),
}


def _write_tiny_snapshot(
    directory,
    *,
    prefix="model.language_model.layers.3.",
    text_overrides=None,
    **overrides,
):
    """config.json + one synthetic layer in model.safetensors; returns nothing."""
    config_class = pytest.importorskip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    ).Qwen3_5TextConfig
    from safetensors.torch import save_file

    text = {**TINY_TEXT, **(text_overrides or {})}
    config_class(**text).save_pretrained(directory)
    tensors = {
        prefix + name: torch.zeros(shape) for name, shape in EXPECTED_TINY.items()
    }
    for name, shape in overrides.items():
        key = prefix + name
        if shape is None:
            tensors.pop(key)
        else:
            tensors[key] = torch.zeros(shape)
    save_file(tensors, str(directory / "model.safetensors"))


def test_load_layer_weights_returns_block_relative_keys(tmp_path):
    _write_tiny_snapshot(tmp_path)
    weights = load_layer_weights(tmp_path, 3, dtype=torch.float16)

    assert set(weights) == set(EXPECTED_TINY)
    assert weights["self_attn.q_proj.weight"].shape == (32, 16)
    assert weights["self_attn.q_norm.weight"].shape == (8,)
    assert all(tensor.dtype == torch.float16 for tensor in weights.values())


def test_load_layer_weights_accepts_the_plain_model_prefix(tmp_path):
    _write_tiny_snapshot(tmp_path, prefix="model.layers.3.")
    assert set(load_layer_weights(tmp_path, 3)) == set(EXPECTED_TINY)


def test_layer_shapes_follow_attn_output_gate(tmp_path):
    """Qwen3-style layers have no output gate, so q_proj is not doubled."""
    _write_tiny_snapshot(
        tmp_path,
        text_overrides={"attn_output_gate": False},
        **{"self_attn.q_proj.weight": (16, 16)},
    )
    weights = load_layer_weights(tmp_path, 3)
    assert weights["self_attn.q_proj.weight"].shape == (16, 16)


def test_missing_layer_key_is_reported(tmp_path):
    _write_tiny_snapshot(tmp_path, **{"mlp.down_proj.weight": None})
    with pytest.raises(RuntimeError, match=r"mlp\.down_proj\.weight"):
        load_layer_weights(tmp_path, 3)


def test_wrong_layer_key_shape_is_reported(tmp_path):
    _write_tiny_snapshot(tmp_path, **{"self_attn.q_proj.weight": (16, 16)})
    with pytest.raises(
        RuntimeError, match=r"self_attn\.q_proj\.weight.*\(32, 16\).*\(16, 16\)"
    ):
        load_layer_weights(tmp_path, 3)


def test_unexpected_layer_key_is_reported(tmp_path):
    _write_tiny_snapshot(tmp_path, **{"linear_attn.out_proj.weight": (16, 16)})
    with pytest.raises(RuntimeError, match=r"linear_attn\.out_proj\.weight"):
        load_layer_weights(tmp_path, 3)


def test_linear_attention_layer_is_refused(tmp_path):
    _write_tiny_snapshot(tmp_path)
    with pytest.raises(ValueError, match=r"linear_attention.*W6"):
        load_layer_weights(tmp_path, 0)


def test_layer_index_out_of_range_is_refused(tmp_path):
    _write_tiny_snapshot(tmp_path)
    with pytest.raises(ValueError, match=r"layer_idx=4.*0\.\.3"):
        load_layer_weights(tmp_path, 4)


# ---------------------------------------------------------------- real 0.8B facts


def _snapshot_or_skip(repo=MODEL, *, weights: bool = True):
    snapshot = cached_snapshot(repo)
    if snapshot is None or not (snapshot / "config.json").exists():
        pytest.skip(f"no local snapshot for {repo}; run the HF download first")
    if weights and not list(snapshot.glob("*.safetensors")):
        pytest.skip(f"no safetensors in {snapshot}")
    return snapshot


def test_config_from_hf_reads_all_qwen35_fields():
    snapshot = _snapshot_or_skip(weights=False)
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    cfg = config_from_hf(hf)

    assert cfg.n_layer == 24
    assert cfg.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 6
    assert cfg.n_embd == 1024
    assert cfg.n_head == 8
    assert cfg.n_kv_head == 2
    assert cfg.n_inner == 3584
    assert cfg.head_dim == 256
    assert cfg.rotary_dim == 64
    assert cfg.norm_eps == 1e-6
    assert cfg.bias is False
    assert cfg.tie_word_embeddings is True
    assert cfg.qk_norm is True
    assert cfg.attn_output_gate is True
    assert cfg.partial_rotary_factor == 0.25
    assert cfg.mrope_section == [11, 11, 10]
    assert cfg.linear_num_key_heads == 16
    assert cfg.linear_num_value_heads == 16
    assert cfg.linear_key_head_dim == 128
    assert cfg.linear_value_head_dim == 128
    assert cfg.linear_conv_kernel_dim == 4


def test_config_from_hf_keeps_qwen3_full_attention_defaults():
    snapshot = _snapshot_or_skip(QWEN3_MODEL, weights=False)
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    cfg = config_from_hf(hf)

    assert cfg.n_layer == 28
    assert cfg.layer_types == ["full_attention"] * 28
    assert cfg.head_dim == 128
    assert cfg.rotary_dim == 128
    assert cfg.partial_rotary_factor == 1.0
    assert cfg.mrope_section is None
    assert cfg.attn_output_gate is False
    assert cfg.linear_num_key_heads is None


@pytest.mark.hf
def test_real_layer_weights_have_qwen35_shapes():
    snapshot = _snapshot_or_skip()
    weights = load_layer_weights(snapshot, 3)  # LLLF*6 -> layer 3 is full attention

    assert weights["self_attn.q_proj.weight"].shape == (4096, 1024)  # 2 * 8 * 256
    assert weights["self_attn.k_proj.weight"].shape == (512, 1024)  # 2 * 256
    assert weights["self_attn.v_proj.weight"].shape == (512, 1024)
    assert weights["self_attn.o_proj.weight"].shape == (1024, 2048)
    assert weights["self_attn.q_norm.weight"].shape == (256,)
    assert weights["self_attn.k_norm.weight"].shape == (256,)
    assert weights["input_layernorm.weight"].shape == (1024,)
    assert weights["post_attention_layernorm.weight"].shape == (1024,)
    assert weights["mlp.gate_proj.weight"].shape == (3584, 1024)
    assert weights["mlp.up_proj.weight"].shape == (3584, 1024)
    assert weights["mlp.down_proj.weight"].shape == (1024, 3584)
    assert all(tensor.dtype == torch.float32 for tensor in weights.values())
    assert not any(name.startswith("linear_attn.") for name in weights)


@pytest.mark.hf
@pytest.mark.slow
def test_qwen35_full_checkpoint_reports_the_w6_gap_without_partial_load(monkeypatch):
    snapshot = _snapshot_or_skip()

    loads: list[str] = []
    original = MiniLLM.load_state_dict

    def spy(self, *args, **kwargs):
        loads.append("called")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MiniLLM, "load_state_dict", spy)

    with pytest.raises(RuntimeError) as excinfo:
        load_checkpoint(snapshot)

    message = str(excinfo.value)
    assert "linear_attn" in message
    assert "W6" in message
    # The snapshot omits lm_head because it is tied; that must not be blamed on model/.
    assert "lm_head" not in message
    assert loads == []  # the mismatch must raise before any weight is copied in
