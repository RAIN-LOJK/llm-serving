"""W5 Step 4 / W6 Step 4: the loader's Qwen3.5 naming contract and its guards.

Three layers of coverage, cheapest first:

1. **Synthetic keys** — no weights, no cache: both prefix conventions map, and the
   out-of-scope heads (``mtp.``, ``visual.``) are dropped rather than half-mapped.
2. **Synthetic tiny snapshot** — a real ``config.json`` plus a fake layer in
   ``model.safetensors``: ``load_layer_weights`` returns repo-relative keys, and missing
   or misshaped keys fail loudly instead of producing a partial block. Both layer kinds
   are covered: W5's full attention (``self_attn.*``, layer 3) and W6's linear attention
   (``linear_attn.*``, layer 0).
3. **The local 0.8B snapshot** (marked ``hf``, skipped without it) — real layer shapes
   for both kinds, the hybrid whole-model load (``load_qwen35_model``) with its state
   byte accounting, and ``load_checkpoint``'s explicit refusal of the hybrid config
   (P-54: no ``linear_attn.*`` gap wall, no partial load).

``load_layer_weights`` returns keys *relative to the block* (``self_attn.q_proj.weight``
or ``linear_attn.in_proj_qkv.weight``), so a block can do
``block.load_state_dict(load_layer_weights(...), strict=True)``. Whole-model loading has
one entry point per model shape: ``load_qwen35_model`` for the hybrid 0.8B,
``load_checkpoint`` for the all-full-attention 0.6B.
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.loading import (
    _expected_linear_attention_shapes,
    cached_snapshot,
    config_from_hf,
    load_checkpoint,
    load_layer_weights,
    load_qwen35_model,
    map_hf_key,
)
from model.minillm import MiniLLM
from model.qwen35 import Qwen35FullAttentionBlock
from model.qwen35_gdn import Qwen35GDNBlock
from model.qwen35_model import Qwen35State

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

# Shapes a W6 linear-attention block must have for TINY_TEXT: key_dim = 2*4 = 8,
# value_dim = 2*4 = 8, so conv_dim = 2*8 + 8 = 24.
EXPECTED_LINEAR_TINY = {
    "input_layernorm.weight": (16,),
    "linear_attn.in_proj_qkv.weight": (24, 16),
    "linear_attn.in_proj_z.weight": (8, 16),
    "linear_attn.in_proj_b.weight": (2, 16),
    "linear_attn.in_proj_a.weight": (2, 16),
    "linear_attn.conv1d.weight": (24, 1, 4),
    "linear_attn.dt_bias": (2,),
    "linear_attn.A_log": (2,),
    "linear_attn.norm.weight": (4,),
    "linear_attn.out_proj.weight": (16, 8),
    "post_attention_layernorm.weight": (16,),
    "mlp.gate_proj.weight": (32, 16),
    "mlp.up_proj.weight": (32, 16),
    "mlp.down_proj.weight": (16, 32),
}


def _write_tiny_snapshot(
    directory,
    *,
    prefix="model.language_model.layers.3.",
    text_overrides=None,
    layer_kind="full_attention",
    **overrides,
):
    """config.json + one synthetic layer in model.safetensors; returns nothing.

    ``layer_kind`` picks which block's key/shape table to write: ``full_attention``
    (W5, layer 3 of ``TINY_TEXT``) or ``linear_attention`` (W6, layer 0). ``overrides``
    replaces a shape or, with ``None``, drops the key -- that is how the missing-key
    guards are built.
    """
    config_class = pytest.importorskip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    ).Qwen3_5TextConfig
    from safetensors.torch import save_file

    if layer_kind not in ("full_attention", "linear_attention"):
        raise ValueError(f"unknown layer_kind={layer_kind!r}")
    expected = EXPECTED_LINEAR_TINY if layer_kind == "linear_attention" else EXPECTED_TINY
    text = {**TINY_TEXT, **(text_overrides or {})}
    config_class(**text).save_pretrained(directory)
    tensors = {prefix + name: torch.zeros(shape) for name, shape in expected.items()}
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


# ------------------------------------------------- W6: one linear-attention layer


def _write_tiny_linear_snapshot(directory, **overrides):
    """config.json + a synthetic linear layer 0 (TINY_TEXT's LLLF schedule)."""
    _write_tiny_snapshot(
        directory,
        prefix="model.language_model.layers.0.",
        layer_kind="linear_attention",
        **overrides,
    )


def test_linear_attention_layer_returns_block_relative_weights(tmp_path):
    """W6 replaces W5's refusal: layer 0 now loads the gated-delta-net key set."""
    from transformers import AutoConfig

    _write_tiny_linear_snapshot(tmp_path)
    weights = load_layer_weights(tmp_path, 0, dtype=torch.float16)

    assert set(weights) == set(EXPECTED_LINEAR_TINY)
    assert {name: tuple(tensor.shape) for name, tensor in weights.items()} == (
        EXPECTED_LINEAR_TINY
    )
    # Same dtype contract as the full-attention test: the cast happens in the reader.
    assert all(tensor.dtype == torch.float16 for tensor in weights.values())

    # The loader's table and the module's parameters are the same names, and a half
    # tensor is cast into the fp32 parameter by load_state_dict itself; `strict=True` is
    # what makes a drifting key set a test failure instead of a silent partial load.
    config = config_from_hf(AutoConfig.from_pretrained(tmp_path, local_files_only=True))
    block = Qwen35GDNBlock(config)
    assert set(block.state_dict()) == set(EXPECTED_LINEAR_TINY)
    block.load_state_dict(weights, strict=True)
    assert block.linear_attn.conv1d.weight.dtype == torch.float32
    assert block.linear_attn.conv1d.weight.shape == (24, 1, 4)


def test_missing_linear_attention_key_is_reported(tmp_path):
    _write_tiny_linear_snapshot(tmp_path, **{"linear_attn.A_log": None})
    with pytest.raises(
        RuntimeError, match=r"(?s)linear-attention block:.*linear_attn\.A_log"
    ):
        load_layer_weights(tmp_path, 0)


def test_wrong_linear_attention_shape_is_reported(tmp_path):
    """A non-depthwise conv weight must fail on shape, not load and compute garbage."""
    _write_tiny_linear_snapshot(
        tmp_path, **{"linear_attn.conv1d.weight": (24, 4, 4)}
    )
    with pytest.raises(
        RuntimeError, match=r"conv1d\.weight.*\(24, 1, 4\).*\(24, 4, 4\)"
    ):
        load_layer_weights(tmp_path, 0)


def test_linear_shapes_require_the_linear_config_fields():
    """A hand-built config can schedule a linear layer without the five linear_* fields.

    No HF snapshot can express this (``Qwen3_5TextConfig`` rejects ``None``), but
    ``MiniLLMConfig`` can, and the loader must name the missing field instead of
    falling back to 0.8B/Qwen3-Next defaults.
    """
    config = MiniLLMConfig(
        n_embd=16,
        n_head=2,
        n_layer=1,
        layer_types=["linear_attention"],
    )
    with pytest.raises(ValueError, match=r"linear_key_head_dim"):
        _expected_linear_attention_shapes(config)


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
def test_real_linear_layer_weights_have_qwen35_shapes():
    """Layer 0 of 0.8B is linear_attention: the gated-delta-net geometry, from the file."""
    snapshot = _snapshot_or_skip()
    weights = load_layer_weights(snapshot, 0)

    # key_dim = 16 * 128 = 2048, value_dim = 16 * 128 = 2048, conv_dim = 6144.
    assert weights["linear_attn.in_proj_qkv.weight"].shape == (6144, 1024)
    assert weights["linear_attn.in_proj_z.weight"].shape == (2048, 1024)
    assert weights["linear_attn.in_proj_b.weight"].shape == (16, 1024)
    assert weights["linear_attn.in_proj_a.weight"].shape == (16, 1024)
    assert weights["linear_attn.conv1d.weight"].shape == (6144, 1, 4)  # depthwise
    assert weights["linear_attn.A_log"].shape == (16,)
    assert weights["linear_attn.dt_bias"].shape == (16,)
    assert weights["linear_attn.norm.weight"].shape == (128,)  # head_v_dim, not hidden
    assert weights["linear_attn.out_proj.weight"].shape == (1024, 2048)
    assert weights["input_layernorm.weight"].shape == (1024,)
    assert weights["post_attention_layernorm.weight"].shape == (1024,)
    assert weights["mlp.gate_proj.weight"].shape == (3584, 1024)
    assert weights["mlp.up_proj.weight"].shape == (3584, 1024)
    assert weights["mlp.down_proj.weight"].shape == (1024, 3584)
    assert all(tensor.dtype == torch.float32 for tensor in weights.values())
    assert not any(name.startswith("self_attn.") for name in weights)


@pytest.mark.hf
def test_qwen35_hybrid_checkpoint_is_rejected_by_load_checkpoint(monkeypatch):
    """P-54: ``load_checkpoint`` is the all-full path and refuses a hybrid config.

    Before Step 4 this read the whole checkpoint and then raised the ``linear_attn.*``
    gap wall. Now the config names the right entry point first, before ``MiniLLM`` is
    built; the spy proves no weight is copied into a half-built model.
    """
    snapshot = _snapshot_or_skip()

    loads: list[str] = []
    original = MiniLLM.load_state_dict

    def spy(self, *args, **kwargs):
        loads.append("called")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MiniLLM, "load_state_dict", spy)

    with pytest.raises(ValueError) as excinfo:
        load_checkpoint(snapshot)

    message = str(excinfo.value)
    assert "load_qwen35_model" in message
    assert "linear_attention" in message
    assert loads == []  # refused before any weight is copied


@pytest.mark.hf
@pytest.mark.slow
def test_load_qwen35_model_from_real_checkpoint():
    """The hybrid 0.8B checkpoint loads, runs a 7-token prefill, and sizes its state.

    This is the whole-model acceptance for Step 4 (P-54): 24 layers on the real
    ``LLLF x 6`` schedule, a finite ``(1, 1, 248320)`` logits row from
    ``logits_to_keep=1``, and the exact §5.6 state bytes for a 7-token fp32 request.
    """
    snapshot = _snapshot_or_skip()
    model = load_qwen35_model(snapshot)

    assert len(model.layers) == 24
    assert sum(isinstance(layer, Qwen35GDNBlock) for layer in model.layers) == 18
    assert sum(
        isinstance(layer, Qwen35FullAttentionBlock) for layer in model.layers
    ) == 6
    # ``to()`` must not have split the shared embedding/lm_head Parameter.
    assert model.lm_head.weight is model.token_embedding.weight

    state = Qwen35State(model.config)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    logits, loss = model(ids, state, logits_to_keep=1)

    assert loss is None
    assert logits.shape == (1, 1, 248320)
    assert torch.isfinite(logits).all()
    assert state.position == 7

    # conv: 6144 channels * (k - 1 = 3) * 4 B; recurrent: 16 * 128 * 128 * 4 B fp32.
    assert state.linear_state_bytes() == 18 * (6144 * 3 * 4 + 16 * 128 * 128 * 4)
    # K/V: 2 kv heads * 2 (K,V) * 256 head_dim * 7 tokens * 4 B, on 6 full layers.
    assert state.kv_bytes() == 6 * (2 * 2 * 256 * 7 * 4)
    assert state.nbytes() == state.linear_state_bytes() + state.kv_bytes()
