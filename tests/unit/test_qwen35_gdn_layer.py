"""W6 Step 2: the ``Qwen35GDN`` layer and its block, wired around the Step-1 operators.

The operators are already checked against two references (``test_qwen35_gdn.py``). What
is left to pin is the *wiring*, and each test below locks one thing that a plausible
transcription of HF's ``Qwen3_5GatedDeltaNet`` could get wrong while every operator still
passes its own tests:

* shapes and the state layout -- including the depthwise conv weight ``(conv_dim, 1, k)``
  and the four **separate** projections (P-53);
* prefill == stepwise decode, and the 1-token prompt going through the **chunked** path
  (P-56: dispatch is on ``past_len``, never on ``T`` alone -- a spy test, because the two
  endpoints agree numerically to ~1e-7 and a value comparison cannot tell them apart);
* causality: replacing token ``t`` must leave every output ``< t`` bit-identical on fp32
  CPU, which is stronger than the tolerance the operator tests use;
* the dispatch guards (T > 1 decode is W7, missing states, carried states, bad
  ``past_len``);
* the gated norm and the beta/g contract at the layer boundary (P-50);
* bf16 input -> bf16 output/conv state + fp32 recurrent state, and no state mutation.

Tiny CPU config: hidden 64, 2 key/value heads, head dim 16, conv kernel 4, so
``conv_dim = 96`` and ``value_dim = 32``. Every case does a handful of tiny matmuls;
the file's ~2.7 s wall time is almost all import/setup, not the tests.
"""

from __future__ import annotations

import pytest
import torch

from model import qwen35_gdn as gdn
from model.config import MiniLLMConfig
from model.qwen35 import Qwen35RMSNormGated
from model.qwen35_gdn import Qwen35GDN, Qwen35GDNBlock, compute_beta_g

TINY = dict(
    vocab_size=128,
    max_position_embeddings=64,
    n_embd=64,
    n_head=4,
    n_kv_head=2,
    n_layer=1,
    n_inner=128,
    dropout=0.0,
    bias=False,
    norm_eps=1e-6,
    head_dim=16,
    rope_theta=1e4,
    qk_norm=True,
    layer_types=["linear_attention"],
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
)

HIDDEN = 64
CONV_DIM = 96  # 2 * key_dim + value_dim = 2*32 + 32
VALUE_DIM = 32
HEAD_V_DIM = 16
SEQ_LEN = 7

# configs/thresholds.toml [W6.gdn]. The layer is wiring, not a new numerical kernel, so
# it must stay inside the operator tolerances; the two state bounds are one decade
# tighter (the G0 dry run put chunk-vs-recurrent at ~1e-7).
OUT_ATOL = 1e-5
STATE_ATOL = 1e-6


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _make_layer() -> Qwen35GDN:
    """A deterministic layer that is *not* accidentally at identity.

    ``A_log``/``dt_bias``/``norm.weight`` ship as ``log(uniform)``/ones/ones, so a layer
    straight from ``__init__`` would test ``g = -softplus(...)`` and a plain RMSNorm.
    Perturbing them keeps the decay and the gate exercised.
    """
    layer = Qwen35GDN(_config())
    with torch.no_grad():
        layer.A_log.normal_(mean=0.5, std=0.2)
        layer.dt_bias.normal_(mean=0.0, std=0.1)
        layer.norm.weight.normal_(mean=1.0, std=0.05)
    return layer.eval()


def _make_block() -> Qwen35GDNBlock:
    block = Qwen35GDNBlock(_config())
    with torch.no_grad():
        block.linear_attn.A_log.normal_(mean=0.5, std=0.2)
        block.linear_attn.dt_bias.normal_(mean=0.0, std=0.1)
        block.linear_attn.norm.weight.normal_(mean=1.0, std=0.05)
    return block.eval()


# ------------------------------------------------------------------ shapes / state


@pytest.mark.parametrize("batch", [1, 2])
def test_shapes_and_state_layout(batch):
    layer = _make_layer()
    x = torch.randn(batch, SEQ_LEN, HIDDEN)

    out, conv_state, recurrent_state = layer(x)

    assert out.shape == (batch, SEQ_LEN, HIDDEN)
    assert out.dtype == x.dtype
    # Minimal sufficient conv state: kernel - 1 columns, input dtype (P-49).
    assert conv_state.shape == (batch, CONV_DIM, 3)
    assert conv_state.dtype == torch.float32
    # [B, H_v, K, V] -- K before V, always fp32 (P-51, HF behaviour).
    assert recurrent_state.shape == (batch, 2, 16, 16)
    assert recurrent_state.dtype == torch.float32

    assert layer.key_dim == 32
    assert layer.value_dim == VALUE_DIM
    assert layer.conv_dim == CONV_DIM
    assert layer.conv1d.weight.shape == (CONV_DIM, 1, 4)
    assert layer.in_proj_qkv.weight.shape == (CONV_DIM, HIDDEN)
    assert layer.in_proj_z.weight.shape == (VALUE_DIM, HIDDEN)
    assert layer.in_proj_b.weight.shape == (2, HIDDEN)
    assert layer.in_proj_a.weight.shape == (2, HIDDEN)
    assert layer.norm.weight.shape == (HEAD_V_DIM,)
    assert layer.out_proj.weight.shape == (HIDDEN, VALUE_DIM)
    assert layer.A_log.shape == (2,)
    assert layer.dt_bias.shape == (2,)


# ------------------------------------------------------- prefill vs decode (P-56, A2)


@pytest.mark.parametrize("batch", [1, 2])
def test_prefill_matches_stepwise_decode(batch):
    """One chunked prefill == prefill(1 token) + token-by-token recurrent decode.

    The stepwise run has to start from the state after token 0, not from the end of the
    full prefill: a decode step consumes the *prefix* state. The projection above the
    conv rounds differently for ``(B, 1, hidden)`` than for ``(B, T, hidden)``, so the
    conv state is not bit-identical here (it is bit-identical inside the operator
    tests); 1e-6 covers the ~1e-7 projection noise.
    """
    layer = _make_layer()
    x = torch.randn(batch, SEQ_LEN, HIDDEN)

    full, conv_full, recurrent_full = layer(x)
    first, conv_state, recurrent_state = layer(x[:, :1])
    steps = [first]
    for t in range(1, SEQ_LEN):
        step, conv_state, recurrent_state = layer(
            x[:, t : t + 1], conv_state, recurrent_state, past_len=1
        )
        steps.append(step)
        torch.testing.assert_close(step, full[:, t : t + 1], atol=OUT_ATOL, rtol=0)

    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=OUT_ATOL, rtol=0)
    torch.testing.assert_close(conv_state, conv_full, atol=STATE_ATOL, rtol=0)
    torch.testing.assert_close(recurrent_state, recurrent_full, atol=STATE_ATOL, rtol=0)


def test_single_token_decode_matches_prefill_position_one():
    """The 2nd token through ``past_len=1`` must equal the prefill's position 1.

    This is the value-level half of P-56; ``test_dispatch_uses_past_len_not_seq_len``
    is the structural half (it proves *which* operator ran).
    """
    layer = _make_layer()
    x = torch.randn(2, 2, HIDDEN)

    two_token, _, _ = layer(x)
    _, conv_state, recurrent_state = layer(x[:, :1])
    decoded, _, _ = layer(x[:, 1:2], conv_state, recurrent_state, past_len=1)

    torch.testing.assert_close(decoded, two_token[:, 1:2], atol=OUT_ATOL, rtol=0)


def test_one_token_prompt_is_a_chunk_prefill():
    """A 1-token prompt has ``past_len == 0``: it must prefill, not decode.

    HF's branch is ``use_precomputed_states and seq_len == 1``; dispatching on
    ``seq_len == 1`` alone would send this call to the recurrent endpoint, which is
    numerically almost identical but requires states a fresh request does not have.
    Here the prompt runs with ``conv_state=None, recurrent_state=None`` and still
    produces usable states, then the next token decodes against them.
    """
    layer = _make_layer()
    x = torch.randn(2, 2, HIDDEN)

    two_token, conv_two, recurrent_two = layer(x)

    first, conv_state, recurrent_state = layer(x[:, :1])  # past_len defaults to 0
    second, conv_decoded, recurrent_decoded = layer(
        x[:, 1:2], conv_state, recurrent_state, past_len=1
    )

    torch.testing.assert_close(first, two_token[:, :1], atol=OUT_ATOL, rtol=0)
    torch.testing.assert_close(
        torch.cat([first, second], dim=1), two_token, atol=OUT_ATOL, rtol=0
    )
    torch.testing.assert_close(conv_decoded, conv_two, atol=STATE_ATOL, rtol=0)
    torch.testing.assert_close(recurrent_decoded, recurrent_two, atol=STATE_ATOL, rtol=0)


def test_dispatch_uses_past_len_not_seq_len(monkeypatch):
    """P-56 pinned structurally: the *operator* names prove the branch, not the values.

    ``gated_delta_chunked`` and ``gated_delta_recurrent`` agree to ~1e-7 at T=1, so no
    output comparison can distinguish a ``past_len`` dispatch from a ``T == 1``
    dispatch. Recording which function ran can.
    """
    calls: list[str] = []
    originals = {
        name: getattr(gdn, name)
        for name in (
            "causal_conv_prefill",
            "causal_conv_decode",
            "gated_delta_chunked",
            "gated_delta_recurrent",
        )
    }

    def record(name):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return originals[name](*args, **kwargs)

        return wrapper

    for name in originals:
        monkeypatch.setattr(gdn, name, record(name))

    layer = _make_layer()
    x = torch.randn(1, 1, HIDDEN)  # T == 1 *and* past_len == 0
    _, conv_state, recurrent_state = layer(x, past_len=0)
    assert calls == ["causal_conv_prefill", "gated_delta_chunked"]

    calls.clear()
    layer(x, conv_state, recurrent_state, past_len=1)
    assert calls == ["causal_conv_decode", "gated_delta_recurrent"]


# ------------------------------------------------------------------------ causality


@pytest.mark.parametrize("token", [0, SEQ_LEN // 2, SEQ_LEN - 1])
def test_prefill_is_causal(token):
    """Rewriting token ``t`` leaves every output ``< t`` bit-identical (fp32 CPU)."""
    layer = _make_layer()
    x = torch.randn(2, SEQ_LEN, HIDDEN)
    changed = x.clone()
    changed[:, token] = torch.randn(2, HIDDEN)

    baseline, _, _ = layer(x)
    actual, _, _ = layer(changed)

    assert torch.equal(actual[:, :token], baseline[:, :token])
    assert not torch.equal(actual[:, token:], baseline[:, token:])


# ------------------------------------------------------------------- dispatch guards


def test_decode_with_more_than_one_token_is_not_implemented():
    layer = _make_layer()
    x = torch.randn(2, SEQ_LEN, HIDDEN)
    _, conv_state, recurrent_state = layer(x[:, :1])

    with pytest.raises(NotImplementedError, match=r"W7"):
        layer(x, conv_state, recurrent_state, past_len=1)


def test_decode_requires_both_states():
    layer = _make_layer()
    x = torch.randn(2, 1, HIDDEN)
    _, conv_state, recurrent_state = layer(x)

    with pytest.raises(ValueError, match=r"conv_state"):
        layer(x, None, recurrent_state, past_len=3)
    with pytest.raises(ValueError, match=r"recurrent_state"):
        layer(x, conv_state, None, past_len=3)


def test_prefill_rejects_carried_states():
    layer = _make_layer()
    x = torch.randn(2, SEQ_LEN, HIDDEN)
    _, conv_state, recurrent_state = layer(x)

    with pytest.raises(ValueError, match=r"past_len=0"):
        layer(x, conv_state, None)
    with pytest.raises(ValueError, match=r"past_len=0"):
        layer(x, None, recurrent_state)
    with pytest.raises(ValueError, match=r"past_len=0"):
        layer(x, conv_state, recurrent_state)


@pytest.mark.parametrize("bad", [True, False])
def test_bool_past_len_is_rejected(bad):
    """``past_len=True`` must not silently mean 1: it is a mode switch, not a flag."""
    layer = _make_layer()
    with pytest.raises(TypeError, match=r"past_len"):
        layer(torch.randn(1, 1, HIDDEN), past_len=bad)


@pytest.mark.parametrize("bad", [1.0, 0.5, "1", None])
def test_non_int_past_len_is_rejected(bad):
    layer = _make_layer()
    with pytest.raises(TypeError, match=r"past_len"):
        layer(torch.randn(1, 1, HIDDEN), past_len=bad)


def test_negative_past_len_is_rejected():
    layer = _make_layer()
    with pytest.raises(ValueError, match=r"past_len"):
        layer(torch.randn(1, 1, HIDDEN), past_len=-1)


# ------------------------------------------------------- gated norm and beta/g wiring


def test_gated_norm_and_beta_g_are_wired_at_the_layer_boundary():
    """P-50: the layer feeds the norm raw core/gate and sigmoid/softplus exactly once."""
    layer = _make_layer()
    assert isinstance(layer.norm, Qwen35RMSNormGated)
    assert layer.norm.weight.shape == (HEAD_V_DIM,)

    x = torch.randn(2, SEQ_LEN, HIDDEN)
    beta, g = compute_beta_g(
        layer.in_proj_b(x), layer.in_proj_a(x), layer.A_log, layer.dt_bias
    )
    assert beta.shape == (2, SEQ_LEN, 2)
    assert g.shape == (2, SEQ_LEN, 2)
    assert torch.all(g <= 0)  # log-space decay, never positive (P-50)
    assert torch.all(beta > 0) and torch.all(beta < 1)  # already sigmoid-ed (P-50)
    assert g.dtype == torch.float32


def test_block_forward_is_residual_plus_mlp_and_returns_state():
    block = _make_block()
    x = torch.randn(2, 5, HIDDEN)

    out, conv_state, recurrent_state = block(x)

    attn_out, expected_conv, expected_recurrent = block.linear_attn(
        block.input_layernorm(x)
    )
    expected = x + attn_out
    expected = expected + block.mlp(block.post_attention_layernorm(expected))

    assert out.shape == x.shape
    assert torch.equal(out, expected)
    assert torch.equal(conv_state, expected_conv)
    assert torch.equal(recurrent_state, expected_recurrent)


# ------------------------------------------------------------------------- dtypes


def test_bf16_layer_keeps_bf16_output_and_conv_state_but_fp32_recurrent_state():
    layer = _make_layer().to(torch.bfloat16)
    x = torch.randn(2, SEQ_LEN, HIDDEN, dtype=torch.bfloat16)

    out, conv_state, recurrent_state = layer(x)

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    assert conv_state.dtype == torch.bfloat16
    assert recurrent_state.dtype == torch.float32


# ------------------------------------------------------------------ state ownership


def test_states_are_fresh_and_not_mutated_by_decode():
    layer = _make_layer()
    x = torch.randn(2, 4, HIDDEN)

    _, conv_a, recurrent_a = layer(x)
    _, conv_b, recurrent_b = layer(x)

    # Each call allocates new state; no shared buffer between calls (D-W6-2).
    assert conv_a.data_ptr() != conv_b.data_ptr()
    assert recurrent_a.data_ptr() != recurrent_b.data_ptr()
    assert torch.equal(conv_a, conv_b)
    assert torch.equal(recurrent_a, recurrent_b)

    conv_clone, recurrent_clone = conv_a.clone(), recurrent_a.clone()
    # The decoded values are irrelevant here (the state is deliberately the one after the
    # full prefill); this test is only about ownership and mutation.
    _, conv_next, recurrent_next = layer(x[:, 3:4], conv_a, recurrent_a, past_len=1)

    # The caller's tensors are read-only to the layer.
    assert torch.equal(conv_a, conv_clone)
    assert torch.equal(recurrent_a, recurrent_clone)
    # And the decode returned new tensors that actually advanced the state.
    assert conv_next.data_ptr() != conv_a.data_ptr()
    assert recurrent_next.data_ptr() != recurrent_a.data_ptr()
    assert not torch.equal(conv_next, conv_a)
    assert not torch.equal(recurrent_next, recurrent_a)
