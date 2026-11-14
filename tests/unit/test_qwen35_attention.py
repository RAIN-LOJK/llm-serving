"""W5 Step 2: ``Qwen35Attention`` / ``Qwen35FullAttentionBlock`` on a tiny config.

The tiny config mirrors the 0.8B *shape rules* at a size CPU can run in milliseconds:
``head_dim`` (32) is independent of ``n_embd // n_head`` (32 here, 256 vs 128 there),
``n_kv_head < n_head`` (2 vs 4), ``attn_output_gate`` doubles ``q_proj``, and
``partial_rotary_factor=0.25`` rotates 8 of 32 dimensions.

What each test pins (``docs/w5-plan.md`` §4 Step 2):

* projection shapes and the **per-head** ``chunk`` of query|gate (a global
  ``[all q | all gate]`` reading is a different model -- the counterexample below shows
  the output moves);
* ``gate = -inf`` silences the layer exactly, which only holds if the sigmoid is applied
  after attention and before ``o_proj``;
* q/k norm is a per-head, zero-centered RMSNorm (``(1 + w)``, fp32);
* partial RoPE leaves the tail of every head untouched;
* causality: replacing token ``t`` leaves positions ``< t`` bit-identical (fp32 CPU);
* incremental decode through ``KVcache`` reproduces the full prefill (atol 1e-5);
* the unsupported containers reject instead of silently taking another path.
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import KVcache, apply_rotary_emb, rope_cos_sin
from model.qwen35 import Qwen35Attention, Qwen35FullAttentionBlock
from model.slab import KvSlab

TINY = dict(
    vocab_size=128,
    max_position_embeddings=64,
    n_embd=128,
    n_head=4,
    n_kv_head=2,
    n_layer=1,
    n_inner=256,
    dropout=0.0,
    bias=False,
    norm_eps=1e-6,
    head_dim=32,
    rope_theta=1e7,
    qk_norm=True,
    layer_types=["full_attention"],
    attn_output_gate=True,
    partial_rotary_factor=0.25,
)


@pytest.fixture
def cfg() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


@pytest.fixture
def attn(cfg) -> Qwen35Attention:
    module = Qwen35Attention(cfg)
    # The real checkpoint's norm weights are learned; randomizing them makes the
    # ``(1 + w)`` factor visible instead of silently 1.0.
    torch.nn.init.normal_(module.q_norm.weight, std=0.5)
    torch.nn.init.normal_(module.k_norm.weight, std=0.5)
    return module.eval()


def _tables(cfg, start: int, length: int):
    return rope_cos_sin(
        cfg.head_dim, cfg.rope_theta, start, length, rotary_dim=cfg.rotary_dim
    )


class _FakePacked:
    """Minimal object that claims the packed-prefill protocol, for the layout guards.

    W7 Step 3 implements the packed path, so these guards are now about *when* a layer
    may take it: the container's token axis, the per-token cos/sin and this layer's
    compact layer index. A rejected container must never reach ``write``/``attend``.
    """

    packed_prefill = True
    cu_seqlens = [0, 1, 2]
    positions = torch.arange(2)
    last_token_index = torch.tensor([1])
    total_tokens = 2
    n_seqs = 1

    def past_len(self, layer=None):
        return 0

    def write(self, layer, k, v):
        raise AssertionError("a rejected packed container must not reach write()")

    def attend(self, q, k, v):
        raise AssertionError("a rejected packed container must not reach attend()")


# ------------------------------------------------------------------- shapes / layout


def test_projection_shapes(cfg, attn):
    assert attn.q_proj.out_features == cfg.n_head * cfg.head_dim * 2 == 256
    assert attn.k_proj.out_features == cfg.n_kv_head * cfg.head_dim == 64
    assert attn.v_proj.out_features == 64
    assert attn.o_proj.in_features == cfg.n_head * cfg.head_dim == 128
    assert attn.q_proj.bias is None and attn.o_proj.bias is None  # attention_bias=false
    assert attn.rotary_dim == 8 and attn.head_dim == 32

    x = torch.randn(2, 5, cfg.n_embd)
    q, gate, k, v = attn._project(x)
    assert q.shape == gate.shape == (2, 5, 4, 32)
    assert k.shape == v.shape == (2, 5, 2, 32)
    out = attn(x)
    assert out.shape == x.shape and out.dtype is x.dtype


def test_q_proj_output_is_query_and_gate_interleaved_per_head(cfg, attn):
    """``view(B, T, H, 2D).chunk(2, -1)``: head h owns rows ``[2hD, 2hD+2D)``."""
    x = torch.randn(1, 3, cfg.n_embd)
    q, gate, k, v = attn._project(x)
    flat = attn.q_proj(x)

    heads, dim = cfg.n_head, cfg.head_dim
    for h in range(heads):
        assert torch.equal(q[0, :, h], flat[0, :, h * 2 * dim : h * 2 * dim + dim])
        assert torch.equal(gate[0, :, h], flat[0, :, h * 2 * dim + dim : (h + 1) * 2 * dim])
    # The layout inverse used by the W5 golden test: cat(q, gate) rebuilds q_proj's output.
    assert torch.equal(torch.cat([q, gate], dim=-1).reshape(1, 3, -1), flat)


def test_global_query_gate_split_is_a_different_model(cfg, attn):
    """A ``[all q | all gate]`` reading relabels every head; the output must move."""
    x = torch.randn(1, 5, cfg.n_embd)
    q, gate, k, v = attn._project(x)
    flat = attn.q_proj(x)
    width = cfg.n_head * cfg.head_dim
    q_global = flat[..., :width].reshape(1, 5, cfg.n_head, cfg.head_dim)
    gate_global = flat[..., width:].reshape(1, 5, cfg.n_head, cfg.head_dim)
    assert not torch.equal(q, q_global)
    assert not torch.equal(gate, gate_global)

    cos, sin = _tables(cfg, 0, 5)
    q_rope, k_rope = attn._rope(attn.q_norm(q), attn.k_norm(k), cos, sin)
    q_rope_global, k_rope_global = attn._rope(
        attn.q_norm(q_global), attn.k_norm(k), cos, sin
    )
    y = attn._merge(attn._attend(q_rope, k_rope, v), gate)
    y_global = attn._merge(attn._attend(q_rope_global, k_rope_global, v), gate_global)
    assert (y - y_global).abs().max().item() > 1e-3


def test_attention_output_matches_block_attention_path(cfg):
    """The block's ``x + attn(norm(x))`` is just the attention helper plus residuals."""
    block = Qwen35FullAttentionBlock(cfg).eval()
    x = torch.randn(1, 4, cfg.n_embd)
    expected = x + block.self_attn(block.input_layernorm(x))
    expected = expected + block.mlp(block.post_attention_layernorm(expected))
    assert torch.equal(block(x), expected)


# ------------------------------------------------------------------------- gating


def test_gate_minus_inf_zeroes_the_layer(cfg, attn):
    """``sigmoid(-inf) == 0`` after attention; ``o_proj`` has no bias, so the output is 0."""
    x = torch.randn(2, 5, cfg.n_embd)
    q, gate, k, v = attn._project(x)
    cos, sin = _tables(cfg, 0, 5)
    q, k = attn._rope(attn.q_norm(q), attn.k_norm(k), cos, sin)
    y = attn._attend(q, k, v)

    out = attn._merge(y, torch.full_like(gate, float("-inf")))
    assert torch.equal(out, torch.zeros_like(out))
    # Teeth: a finite gate does not zero it, so the assertion above is about the gate.
    assert not torch.equal(attn._merge(y, gate), torch.zeros_like(out))


def test_gate_multiplies_before_o_proj(cfg, attn):
    """``_merge`` is exactly ``o_proj(y * sigmoid(gate))`` with the head-major layout."""
    x = torch.randn(1, 3, cfg.n_embd)
    q, gate, k, v = attn._project(x)
    cos, sin = _tables(cfg, 0, 3)
    q, k = attn._rope(attn.q_norm(q), attn.k_norm(k), cos, sin)
    y = attn._attend(q, k, v)

    width = cfg.n_head * cfg.head_dim
    expected = torch.nn.functional.linear(
        y.reshape(1, 3, width) * torch.sigmoid(gate.reshape(1, 3, width)),
        attn.o_proj.weight,
    )
    assert torch.equal(attn._merge(y, gate), expected)


# --------------------------------------------------------------------- norms


def test_qk_norm_is_per_head_zero_centered_rms(cfg, attn):
    x = torch.randn(1, 4, cfg.n_embd)
    q, gate, k, v = attn._project(x)
    qn, kn = attn.q_norm(q), attn.k_norm(k)

    def reference(t, weight):
        t = t.float()
        return (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + cfg.norm_eps)) * (
            1.0 + weight.float()
        )

    q_ref = reference(q, attn.q_norm.weight)
    k_ref = reference(k, attn.k_norm.weight)
    assert torch.allclose(qn, q_ref, atol=1e-6)
    assert torch.allclose(kn, k_ref, atol=1e-6)
    assert attn.q_norm.eps == cfg.norm_eps == 1e-6  # P-03: eps is wired, not defaulted

    # Each head is normalized on its own, and k uses k_norm's weight (not q_norm's).
    for h in range(cfg.n_head):
        assert torch.allclose(
            qn[:, :, h], reference(q[:, :, h], attn.q_norm.weight), atol=1e-6
        )
    for h in range(cfg.n_kv_head):
        assert torch.allclose(
            kn[:, :, h], reference(k[:, :, h], attn.k_norm.weight), atol=1e-6
        )
    # A cross-head reduction would give a different number: pin it.
    assert not torch.allclose(
        qn[0, 0, 0], reference(q[0, 0].reshape(1, -1), attn.q_norm.weight[0]).reshape(-1)[:32],
        atol=1e-4,
    )


def test_attention_rejects_a_config_without_output_gate():
    with pytest.raises(ValueError, match="attn_output_gate"):
        Qwen35Attention(MiniLLMConfig(**{**TINY, "attn_output_gate": False}))


# ------------------------------------------------------------------------ RoPE


def test_rope_rotates_only_the_first_rotary_dim(cfg, attn):
    batch, seq = 2, 5
    q = torch.randn(batch, seq, cfg.n_head, cfg.head_dim)
    k = torch.randn(batch, seq, cfg.n_kv_head, cfg.head_dim)
    cos, sin = _tables(cfg, 0, seq)

    q_rope, k_rope = attn._rope(q, k, cos, sin)
    rotary = cfg.rotary_dim
    assert torch.equal(q_rope[..., rotary:], q[..., rotary:])
    assert torch.equal(k_rope[..., rotary:], k[..., rotary:])
    # The rotated slice is the existing GPT-NeoX rotation over rotary_dim.
    assert torch.equal(q_rope[..., :rotary], apply_rotary_emb(q[..., :rotary], cos, sin))
    assert not torch.equal(q_rope[..., :rotary], q[..., :rotary])


def test_rope_rejects_a_full_width_table(cfg, attn):
    """A ``(T, head_dim // 2)`` table is the full-width one; partial RoPE needs ``(T, 4)``."""
    q = torch.randn(1, 3, cfg.n_head, cfg.head_dim)
    k = torch.randn(1, 3, cfg.n_kv_head, cfg.head_dim)
    cos, sin = rope_cos_sin(cfg.head_dim, cfg.rope_theta, 0, 3)  # (3, 16)
    with pytest.raises(ValueError, match=r"rotary_dim=8"):
        attn._rope(q, k, cos, sin)


def test_self_computed_tables_use_past_len(cfg, attn):
    """``cos=None`` computes the table from ``past_len``; passing it must change nothing."""
    prefix = torch.randn(1, 3, cfg.n_embd)
    x = torch.randn(1, 1, cfg.n_embd)
    explicit_cache, implicit_cache = KVcache(1), KVcache(1)
    attn(prefix, past_keys_values=explicit_cache)
    attn(prefix, past_keys_values=implicit_cache)
    assert explicit_cache.past_len(0) == implicit_cache.past_len(0) == 3

    cos, sin = _tables(cfg, 3, 1)
    explicit = attn(x, past_keys_values=explicit_cache, cos=cos, sin=sin)
    implicit = attn(x, past_keys_values=implicit_cache)
    assert torch.equal(explicit, implicit)


# ----------------------------------------------------------------- causality


def test_causality_holds_bit_for_bit(cfg, attn):
    x = torch.randn(1, 7, cfg.n_embd)
    changed = x.clone()
    t = 4
    changed[0, t] += 1.0

    before, after = attn(x), attn(changed)
    assert torch.equal(before[:, :t], after[:, :t]), "a later token leaked into position < t"
    assert not torch.equal(before[:, t], after[:, t]), "the perturbation was a no-op"


def test_block_causality_holds_bit_for_bit(cfg):
    block = Qwen35FullAttentionBlock(cfg).eval()
    x = torch.randn(1, 6, cfg.n_embd)
    changed = x.clone()
    changed[0, 3] -= 2.0

    before, after = block(x), block(changed)
    assert torch.equal(before[:, :3], after[:, :3])
    assert not torch.equal(before[:, 3], after[:, 3])


# ------------------------------------------------------------- incremental decode


def test_incremental_decode_matches_full_prefill(cfg, attn):
    x = torch.randn(1, 6, cfg.n_embd)

    prefill_cache = KVcache(1)
    full = attn(x, past_keys_values=prefill_cache)
    assert prefill_cache.past_len(0) == 6
    k, v = prefill_cache.get_kv(0)
    assert k.shape == (1, cfg.n_kv_head, 6, cfg.head_dim)

    decode_cache = KVcache(1)
    steps = torch.cat(
        [attn(x[:, t : t + 1], past_keys_values=decode_cache) for t in range(x.shape[1])],
        dim=1,
    )
    assert steps.shape == full.shape
    assert torch.allclose(steps, full, atol=1e-5), (
        (steps - full).abs().max().item()
    )


def test_decode_with_a_nonzero_past_matches_the_prefill_tail(cfg, attn):
    """Decode positions start at ``past_len``; a wrong RoPE start shows up here."""
    x = torch.randn(1, 5, cfg.n_embd)
    cache = KVcache(1)
    prefill = attn(x[:, :3], past_keys_values=cache)
    step = attn(x[:, 3:4], past_keys_values=cache)
    assert torch.allclose(step, prefill[:, 3:4], atol=1e-6)


# -------------------------------------------------------------------- guards


def test_attention_rejects_masks_and_unsupported_containers(cfg, attn):
    x = torch.randn(1, 3, cfg.n_embd)
    with pytest.raises(NotImplementedError, match="attention_mask"):
        attn(x, attention_mask=torch.ones(1, 3, dtype=torch.bool))

    cache = KVcache(1)
    attn(x, past_keys_values=cache)  # 3 past tokens
    with pytest.raises(NotImplementedError, match="chunked-prefill"):
        attn(x, past_keys_values=cache)

    # Packed prefill (W7 Step 3) needs the packed token axis, per-token cos/sin and this
    # layer's compact full-attention index; each is rejected before any projection.
    with pytest.raises(ValueError, match="one token axis"):
        attn(x, past_keys_values=_FakePacked())
    cos, sin = _tables(cfg, 0, 2)
    with pytest.raises(ValueError, match="compact full-attention index"):
        attn(torch.randn(1, 2, cfg.n_embd), past_keys_values=_FakePacked(), cos=cos, sin=sin)
    indexed = Qwen35Attention(cfg, layer_index=0)
    with pytest.raises(ValueError, match="per-token cos/sin"):
        indexed(torch.randn(1, 2, cfg.n_embd), past_keys_values=_FakePacked())

    slab = KvSlab(n_layer=1, batch=1, n_kv_head=cfg.n_kv_head, head_dim=cfg.head_dim, max_len=8)
    with pytest.raises(NotImplementedError, match="slab/paged"):
        attn(x, past_keys_values=slab)
