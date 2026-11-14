"""W6 Step 1: the Gated DeltaNet operators, checked against two independent references.

``model/qwen35_gdn.py`` implements three operators; each is pinned twice, because a
single reference cannot tell a shared bug from a correct implementation:

* short conv -- causality (replacing token ``t`` must leave ``< t`` bit-identical on
  fp32 CPU) and prefill-vs-stepwise-decode equivalence, which is what makes the
  ``k - 1`` token state *sufficient*;
* recurrent delta rule -- vs fla's ``naive_recurrent_gated_delta_rule`` **and** vs HF's
  ``torch_recurrent_gated_delta_rule`` torch fallback;
* chunked (WY/UT) rule -- vs the recurrent form, vs HF's ``torch_chunk_gated_delta_rule``
  and vs fla's ``naive_chunk_gated_delta_rule``, plus the ``T = C - 1 / C / C + 1``
  off-by-one set that chunked kernels habitually get wrong.

The two HF functions are reached through the *module* (``hf.torch_...``), never imported
by name: ``tests/conftest.py`` swaps those module attributes for the torch fallbacks at
fixture time, and a module-level ``from ... import ...`` would capture the pre-pin fla
Triton wrapper and fail on CPU (P-48).

Every operator comparison uses the pre-registered ``configs/thresholds.toml [W6.gdn]``
values -- ``atol = rtol = 1e-5`` for outputs, ``1e-5`` for the final state -- **not**
tuned numbers: the G0 dry run put chunk-vs-recurrent at ~1e-7, i.e. ~100x headroom.
Tests that pass raw (unnormalized) q/k are not comparing unnormalized k across endpoints:
the operator normalizes internally (D8 decision 3, P-58), and the references get the
same normalization applied on the test side.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model.qwen35_gdn import (
    causal_conv_decode,
    causal_conv_prefill,
    compute_beta_g,
    gated_delta_chunked,
    gated_delta_recurrent,
    l2norm,
)

# configs/thresholds.toml [W6.gdn], pre-registered 2026-11-14 (W6-0000). Do not relax.
ATOL = 1e-5  # operator_atol
RTOL = 1e-5  # operator_rtol
STATE_ATOL = 1e-5  # state_atol
CONV_ATOL = 1e-6  # conv_atol
CAUSAL_ATOL = 0.0  # causal_atol: fp32 CPU, bit-identical

CHUNK = 64  # production chunk_size

# T = C - 1, C, C + 1 and the two next boundaries: the padding/slicing off-by-one set.
BOUNDARY_SEQ_LENS = [1, 63, 64, 65, 127, 128, 129]


# ------------------------------------------------------------------ helpers


def _randn(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * scale


def _gdn_inputs(
    batch: int = 1,
    seq_len: int = 65,
    num_k_heads: int = 2,
    num_v_heads: int = 2,
    head_k_dim: int = 32,
    head_v_dim: int = 32,
    seed: int = 0,
):
    """Raw operator inputs: unnormalized q/k, log-space g <= 0, beta already sigmoid.

    q/k are deliberately *not* pre-normalized -- the operator owns that (D8 decision 3),
    and feeding raw vectors is the layer's real call pattern.
    """
    return (
        _randn((batch, seq_len, num_k_heads, head_k_dim), seed),
        _randn((batch, seq_len, num_k_heads, head_k_dim), seed + 1),
        _randn((batch, seq_len, num_v_heads, head_v_dim), seed + 2),
        -F.softplus(_randn((batch, seq_len, num_v_heads), seed + 3)),
        torch.sigmoid(_randn((batch, seq_len, num_v_heads), seed + 4)),
    )


def _state(
    batch: int,
    num_v_heads: int = 2,
    head_k_dim: int = 32,
    head_v_dim: int = 32,
    seed: int = 900,
    scale: float = 0.1,
) -> torch.Tensor:
    return _randn((batch, num_v_heads, head_k_dim, head_v_dim), seed, scale)


def _normalized_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly the operator's normalization, for references that take raw q/k.

    fla's naive functions do not normalize, so the test side does it; HF's do it itself
    behind ``use_qk_l2norm_in_kernel=True``. Either way only the ``K**-0.5`` scale on q
    is applied here (``scale=K**-0.5`` is passed separately to fla), never twice.
    """
    return l2norm(q, dim=-1, eps=1e-6), l2norm(k, dim=-1, eps=1e-6)


def _hf_module():
    """The HF module **object**: conftest pins its attributes to the torch fallbacks."""
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    for name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
        resolved = getattr(hf, name)
        if getattr(resolved, "__wrapped__", None) is not None:
            raise AssertionError(
                f"{name} is still the hub/fla kernel wrapper: tests/conftest.py's "
                "_torch_kernel_fallbacks fixture did not run, and the Triton kernel "
                "cannot run on CPU (P-48)"
            )
    return hf


def _assert_operator_close(actual, expected) -> None:
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


# ------------------------------------------------------------------ l2norm


def test_l2norm_matches_the_hf_formula():
    x = _randn((2, 3, 5), 0)
    expected = x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(l2norm(x), expected, atol=0.0, rtol=0.0)
    # and it really is the unit l2 norm when the vector is not in the eps regime
    torch.testing.assert_close(
        l2norm(x).norm(2, dim=-1), torch.ones(2, 3), atol=1e-6, rtol=0.0
    )


def test_l2norm_zero_vector_is_zero_not_nan():
    """``eps`` sits inside the ``rsqrt``: a zero vector maps to zero, not to inf/NaN."""
    out = l2norm(torch.zeros(2, 4))
    assert torch.equal(out, torch.zeros(2, 4))
    assert torch.isfinite(out).all()


def test_l2norm_eps_floors_the_denominator():
    """Below ``sqrt(eps)`` the eps term dominates, so a tiny vector is *not* blown up."""
    tiny = torch.full((3, 4), 1e-4)
    out = l2norm(tiny)
    # Without the eps term a length-4 vector would come out with unit norm.
    assert (out.norm(2, dim=-1) < 0.5).all()
    assert torch.allclose(out, tiny * torch.rsqrt(tiny.pow(2).sum(-1, keepdim=True) + 1e-6))


def test_l2norm_stays_fp32_and_rejects_non_float():
    x = _randn((2, 3), 2)
    assert l2norm(x).dtype == torch.float32
    with pytest.raises(TypeError):
        l2norm(torch.arange(4))
    with pytest.raises(TypeError):
        l2norm([1.0, 2.0])


# ------------------------------------------------------------------ beta / g


def test_compute_beta_g_matches_the_explicit_formula():
    b = _randn((2, 5, 3), 10)
    a = _randn((2, 5, 3), 11)
    A_log = _randn((3,), 12)
    dt_bias = _randn((3,), 13)
    beta, g = compute_beta_g(b, a, A_log, dt_bias)
    torch.testing.assert_close(beta, torch.sigmoid(b), atol=0.0, rtol=0.0)
    # Same expression up to one ulp of softplus's internal beta*x/beta round trip.
    torch.testing.assert_close(
        g, -A_log.exp() * torch.log1p(torch.exp(a + dt_bias)), atol=1e-6, rtol=1e-6
    )


def test_compute_beta_g_ranges_and_dtypes():
    b = _randn((4, 7, 5), 14)
    a = _randn((4, 7, 5), 15)
    beta, g = compute_beta_g(b, a, _randn((5,), 16), _randn((5,), 17))
    assert ((beta > 0) & (beta < 1)).all()
    assert (g <= 0).all()
    assert beta.dtype == b.dtype and g.dtype == torch.float32

    # bf16 model tensors: beta keeps the model dtype, g is promoted (HF's A_log.float()).
    b16, a16 = b.to(torch.bfloat16), a.to(torch.bfloat16)
    A16, dt16 = _randn((5,), 16).to(torch.bfloat16), _randn((5,), 17).to(torch.bfloat16)
    beta16, g16 = compute_beta_g(b16, a16, A16, dt16)
    assert beta16.dtype == torch.bfloat16 and g16.dtype == torch.float32


def test_compute_beta_g_is_not_double_activated():
    """P-50 regression: a second sigmoid/softplus must be visible here."""
    b = torch.full((1, 1, 2), 2.0)
    a = torch.full((1, 1, 2), 1.0)
    beta, g = compute_beta_g(b, a, torch.zeros(2), torch.zeros(2))

    torch.testing.assert_close(beta, torch.sigmoid(b), atol=0.0, rtol=0.0)
    torch.testing.assert_close(g, -F.softplus(a), atol=0.0, rtol=0.0)
    assert not torch.allclose(beta, beta.sigmoid())
    assert not torch.allclose(g, -F.softplus(F.softplus(a)))
    assert not torch.allclose(g, -F.softplus(a).sigmoid())


def test_compute_beta_g_validates_shapes():
    good = _randn((2, 5, 3), 18)
    with pytest.raises(ValueError):
        compute_beta_g(_randn((2, 5), 18), good, _randn((3,), 19), _randn((3,), 20))
    with pytest.raises(ValueError):
        compute_beta_g(good, _randn((2, 4, 3), 18), _randn((3,), 19), _randn((3,), 20))
    with pytest.raises(ValueError):
        compute_beta_g(good, good, _randn((4,), 19), _randn((3,), 20))
    with pytest.raises(TypeError):
        compute_beta_g(good.to(torch.int64), good, _randn((3,), 19), _randn((3,), 20))


# ------------------------------------------------------------------ short conv


@pytest.mark.parametrize("token", [0, 9, 16])
def test_conv_prefill_is_causal(token):
    """Replacing token ``t`` leaves ``< t`` bit-identical (fp32 CPU) and moves ``t``."""
    x = _randn((2, 6, 17), 20)
    weight = _randn((6, 1, 4), 21)
    out_a, _ = causal_conv_prefill(x, weight)

    x_b = x.clone()
    x_b[:, :, token] = _randn((2, 6), 22)
    out_b, _ = causal_conv_prefill(x_b, weight)

    torch.testing.assert_close(
        out_a[:, :, :token], out_b[:, :, :token], atol=CAUSAL_ATOL, rtol=CAUSAL_ATOL
    )
    assert not torch.equal(out_a[:, :, token], out_b[:, :, token])


def test_conv_prefill_matches_stepwise_decode():
    x = _randn((2, 6, 17), 30)
    weight = _randn((6, 1, 4), 31)
    out_prefill, state_prefill = causal_conv_prefill(x, weight)

    state = x.new_zeros((x.shape[0], x.shape[1], 3))
    steps = []
    for token in range(x.shape[-1]):
        out_t, state = causal_conv_decode(x[:, :, token : token + 1], state, weight)
        steps.append(out_t)
    out_stepwise = torch.cat(steps, dim=-1)

    torch.testing.assert_close(out_stepwise, out_prefill, atol=CONV_ATOL, rtol=CONV_ATOL)
    torch.testing.assert_close(state, state_prefill, atol=CONV_ATOL, rtol=CONV_ATOL)

    # The state is the pre-SiLU *input* window, not the activation of it: its last
    # column is the last input token, one column per token, oldest first.
    torch.testing.assert_close(state_prefill[..., -1], x[..., -1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(state_prefill[..., 0], x[..., -3], atol=0.0, rtol=0.0)
    assert not torch.allclose(state_prefill[..., -1], F.silu(x[..., -1]))


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5])
def test_conv_shapes_and_left_padded_state(seq_len):
    x = _randn((2, 6, seq_len), 40 + seq_len)
    weight = _randn((6, 1, 4), 41)
    out, state = causal_conv_prefill(x, weight)

    assert out.shape == (2, 6, seq_len)
    assert state.shape == (2, 6, 3)
    if seq_len <= 3:
        zeros = 3 - seq_len
        torch.testing.assert_close(
            state[..., :zeros], torch.zeros(2, 6, zeros), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(state[..., zeros:], x, atol=0.0, rtol=0.0)
    else:
        torch.testing.assert_close(state, x[..., -3:], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5])
def test_conv_decode_shapes_and_state_window(seq_len):
    x = _randn((2, 6, seq_len), 50 + seq_len)
    conv_state = _randn((2, 6, 3), 51)
    weight = _randn((6, 1, 4), 52)
    before = conv_state.clone()

    out, new_state = causal_conv_decode(x, conv_state, weight)

    assert out.shape == (2, 6, seq_len)
    assert new_state.shape == (2, 6, 3)
    torch.testing.assert_close(conv_state, before, atol=0.0, rtol=0.0)  # not mutated
    torch.testing.assert_close(
        new_state, torch.cat([before, x], dim=-1)[..., -3:], atol=0.0, rtol=0.0
    )


def test_conv_weight_layouts_agree():
    """``(conv_dim, 1, k)`` (HF ``nn.Conv1d``) and ``(conv_dim, k)`` are the same conv."""
    x = _randn((2, 6, 9), 60)
    weight = _randn((6, 1, 4), 61)
    flat_out, flat_state = causal_conv_prefill(x, weight[:, 0, :])
    out, state = causal_conv_prefill(x, weight)
    torch.testing.assert_close(flat_out, out, atol=0.0, rtol=0.0)
    torch.testing.assert_close(flat_state, state, atol=0.0, rtol=0.0)


def test_conv_rejects_unsupported_arguments():
    x = _randn((2, 6, 5), 62)
    conv_state = _randn((2, 6, 3), 63)
    weight = _randn((6, 1, 4), 64)
    with pytest.raises(ValueError):
        causal_conv_prefill(x, weight, activation="gelu")
    with pytest.raises(ValueError):
        causal_conv_decode(x[:, :, :1], conv_state, weight, activation="relu")
    with pytest.raises(ValueError):  # HF cache length (kernel_size=4), not kernel-1
        causal_conv_decode(x[:, :, :1], _randn((2, 6, 4), 65), weight)
    with pytest.raises(ValueError):  # wrong channel count
        causal_conv_prefill(x, _randn((7, 1, 4), 66))
    with pytest.raises(ValueError):  # not depthwise: (conv_dim, 2, k)
        causal_conv_decode(x[:, :, :1], conv_state, _randn((6, 2, 4), 67))
    with pytest.raises(ValueError):  # x must already be (B, conv_dim, T)
        causal_conv_prefill(x[:, :, 0], weight)
    with pytest.raises(ValueError):  # T >= 1
        causal_conv_prefill(x[:, :, :0], weight)
    with pytest.raises(TypeError):
        causal_conv_prefill(x.to(torch.int32), weight)


# ------------------------------------------------------------------ recurrent


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 63, 64, 65, 127, 128, 129, 257])
def test_recurrent_matches_fla_naive(batch, seq_len):
    naive = pytest.importorskip("fla.ops.gated_delta_rule.naive")
    q, k, v, g, beta = _gdn_inputs(batch, seq_len, seed=seq_len)
    initial_state = _state(batch)

    out, state = gated_delta_recurrent(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    q_norm, k_norm = _normalized_qk(q, k)
    ref_out, ref_state = naive.naive_recurrent_gated_delta_rule(
        q_norm,
        k_norm,
        v,
        beta,
        g,
        scale=32**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )
    _assert_operator_close(out, ref_out)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=RTOL)


@pytest.mark.parametrize("batch, seq_len", [(1, 1), (2, 1), (1, 65), (2, 64), (2, 129)])
def test_recurrent_matches_hf_recurrent(batch, seq_len):
    hf = _hf_module()
    q, k, v, g, beta = _gdn_inputs(batch, seq_len, seed=100 + seq_len)
    initial_state = _state(batch, seed=101 + seq_len)

    out, state = gated_delta_recurrent(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    # Raw q/k: HF normalizes inside the kernel, exactly like our operator (D8, P-58).
    ref_out, ref_state = hf.torch_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    _assert_operator_close(out, ref_out)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=RTOL)


def test_recurrent_output_state_dtype_and_optional_state():
    q, k, v, g, beta = _gdn_inputs(1, 65, seed=120)
    out, state = gated_delta_recurrent(q, k, v, g, beta)
    assert out.dtype == torch.float32  # q.dtype
    assert out.shape == (1, 65, 2, 32)
    assert state is None

    _, state_bf16_input = gated_delta_recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial_state=_state(1).to(torch.bfloat16),
        output_final_state=True,
    )
    assert state_bf16_input.dtype == torch.float32


# ------------------------------------------------------------------ chunked


@pytest.mark.parametrize(
    "batch, seq_len", [(1, 1), (1, 63), (1, 64), (1, 65), (2, 127), (2, 129), (2, 257)]
)
def test_chunked_matches_hf_chunk(batch, seq_len):
    hf = _hf_module()
    q, k, v, g, beta = _gdn_inputs(batch, seq_len, seed=200 + seq_len)
    initial_state = _state(batch, seed=201 + seq_len)

    out, state = gated_delta_chunked(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=CHUNK,
        initial_state=initial_state,
        output_final_state=True,
    )
    ref_out, ref_state = hf.torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=CHUNK,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    _assert_operator_close(out, ref_out)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=RTOL)


@pytest.mark.parametrize("seq_len", [1, 64, 65, 129])
def test_chunked_matches_fla_naive_chunk(seq_len):
    naive = pytest.importorskip("fla.ops.gated_delta_rule.naive")
    q, k, v, g, beta = _gdn_inputs(1, seq_len, seed=300 + seq_len)
    initial_state = _state(1, seed=301 + seq_len)

    out, state = gated_delta_chunked(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=CHUNK,
        initial_state=initial_state,
        output_final_state=True,
    )
    q_norm, k_norm = _normalized_qk(q, k)
    # fla's chunk signature is (q, k, v, g, beta) -- g before beta, unlike the recurrent one.
    ref_out, ref_state = naive.naive_chunk_gated_delta_rule(
        q_norm,
        k_norm,
        v,
        g,
        beta,
        chunk_size=CHUNK,
        scale=32**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )
    _assert_operator_close(out, ref_out)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=RTOL)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seq_len", BOUNDARY_SEQ_LENS)
def test_chunked_matches_recurrent(batch, seq_len):
    q, k, v, g, beta = _gdn_inputs(batch, seq_len, seed=400 + seq_len)
    initial_state = _state(batch, seed=401 + seq_len)

    out_rec, state_rec = gated_delta_recurrent(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    out_chunk, state_chunk = gated_delta_chunked(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    _assert_operator_close(out_chunk, out_rec)
    torch.testing.assert_close(state_chunk, state_rec, atol=STATE_ATOL, rtol=RTOL)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 65, 129])
def test_chunked_matches_recurrent_with_gqa(batch, seq_len):
    """ratio = H_v // H_k = 2: q/k are repeated per value head inside the operator."""
    q, k, v, g, beta = _gdn_inputs(
        batch, seq_len, num_k_heads=2, num_v_heads=4, seed=500 + seq_len
    )
    assert q.shape[2] == 2 and v.shape[2] == 4 and beta.shape[2] == 4
    initial_state = _state(batch, num_v_heads=4, seed=501 + seq_len)

    out_rec, state_rec = gated_delta_recurrent(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    out_chunk, state_chunk = gated_delta_chunked(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    assert out_chunk.shape == (batch, seq_len, 4, 32)
    assert state_chunk.shape == (batch, 4, 32, 32)
    _assert_operator_close(out_chunk, out_rec)
    torch.testing.assert_close(state_chunk, state_rec, atol=STATE_ATOL, rtol=RTOL)


def test_gqa_repeat_matches_manual_repeat_interleave():
    """The internal repeat must be ``repeat_interleave`` (h0, h0, h1, h1), not tiled."""
    q, k, v, g, beta = _gdn_inputs(2, 65, num_k_heads=2, num_v_heads=4, seed=600)
    initial_state = _state(2, num_v_heads=4, seed=601)

    out, state = gated_delta_chunked(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    out_manual, state_manual = gated_delta_chunked(
        q.repeat_interleave(2, dim=2),
        k.repeat_interleave(2, dim=2),
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.testing.assert_close(out, out_manual, atol=0.0, rtol=0.0)
    torch.testing.assert_close(state, state_manual, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("seq_len", BOUNDARY_SEQ_LENS)
def test_chunked_boundary_lengths_and_padded_rows_are_discarded(seq_len):
    q, k, v, g, beta = _gdn_inputs(1, seq_len, seed=700 + seq_len)
    out, state = gated_delta_chunked(q, k, v, g, beta, output_final_state=True)
    assert out.shape == (1, seq_len, 2, 32)
    assert out.dtype == torch.float32
    assert state.shape == (1, 2, 32, 32)

    out_rec, state_rec = gated_delta_recurrent(q, k, v, g, beta, output_final_state=True)
    _assert_operator_close(out, out_rec)
    torch.testing.assert_close(state, state_rec, atol=STATE_ATOL, rtol=RTOL)


@pytest.mark.parametrize("token", [0, 64, 100, 128])
def test_chunked_is_causal(token):
    """T = 129 = 2C + 1; token 64 is a chunk boundary, 100 sits inside chunk 1."""
    q, k, v, g, beta = _gdn_inputs(1, 129, seed=800)
    out_a, _ = gated_delta_chunked(q, k, v, g, beta, output_final_state=True)

    q_b, k_b, v_b, g_b, beta_b = (t.clone() for t in (q, k, v, g, beta))
    q_b[:, token] = _randn((1, 2, 32), 801)
    k_b[:, token] = _randn((1, 2, 32), 802)
    v_b[:, token] = _randn((1, 2, 32), 803)
    g_b[:, token] = -F.softplus(_randn((1, 2), 804))
    beta_b[:, token] = torch.sigmoid(_randn((1, 2), 805))
    out_b, _ = gated_delta_chunked(q_b, k_b, v_b, g_b, beta_b, output_final_state=True)

    torch.testing.assert_close(
        out_a[:, :token], out_b[:, :token], atol=CAUSAL_ATOL, rtol=CAUSAL_ATOL
    )
    assert not torch.equal(out_a[:, token], out_b[:, token])


def test_chunked_output_state_dtype_and_optional_state():
    q, k, v, g, beta = _gdn_inputs(1, 65, seed=810)
    out, state = gated_delta_chunked(q, k, v, g, beta)
    assert out.dtype == torch.float32 and state is None

    _, state_bf16_input = gated_delta_chunked(
        q,
        k,
        v,
        g,
        beta,
        initial_state=_state(1).to(torch.bfloat16),
        output_final_state=True,
    )
    assert state_bf16_input.dtype == torch.float32


# ------------------------------------------------------------------ shapes / dtype / contract


@pytest.mark.parametrize("operator", [gated_delta_recurrent, gated_delta_chunked])
def test_bf16_inputs_give_bf16_output_and_fp32_state(operator):
    q, k, v, g, beta = _gdn_inputs(1, 65, seed=820)
    inputs_bf16 = [t.to(torch.bfloat16) for t in (q, k, v, g, beta)]
    initial_state = _state(1, seed=821)
    initial_state_bf16 = initial_state.to(torch.bfloat16)

    out_bf16, state_bf16 = operator(
        *inputs_bf16, initial_state=initial_state_bf16, output_final_state=True
    )
    assert out_bf16.dtype == torch.bfloat16
    assert state_bf16.dtype == torch.float32

    # fp32 reference on the *same* (bf16-rounded) values: the only difference is the
    # bf16 rounding of the output, and the state must not even move.
    out_ref, state_ref = operator(
        *(t.float() for t in inputs_bf16),
        initial_state=initial_state_bf16.float(),
        output_final_state=True,
    )
    torch.testing.assert_close(out_bf16.float(), out_ref, atol=2e-2, rtol=0.0)
    torch.testing.assert_close(state_bf16, state_ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("operator", [gated_delta_recurrent, gated_delta_chunked])
def test_operators_do_not_mutate_their_inputs(operator):
    q, k, v, g, beta = _gdn_inputs(2, 65, seed=830)
    initial_state = _state(2, seed=831)
    copies = [t.clone() for t in (q, k, v, g, beta, initial_state)]

    operator(q, k, v, g, beta, initial_state=initial_state, output_final_state=True)

    for original, saved in zip((q, k, v, g, beta, initial_state), copies):
        torch.testing.assert_close(original, saved, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("operator", [gated_delta_recurrent, gated_delta_chunked])
def test_state_is_k_before_v_with_rectangular_heads(operator):
    """K != V: a transposed state would still compare equal when K == V."""
    q, k, v, g, beta = _gdn_inputs(
        1, 65, head_k_dim=16, head_v_dim=32, seed=840
    )
    out, state = operator(q, k, v, g, beta, output_final_state=True)
    assert out.shape == (1, 65, 2, 32)
    assert state.shape == (1, 2, 16, 32)
    ref_out, ref_state = gated_delta_recurrent(q, k, v, g, beta, output_final_state=True)
    _assert_operator_close(out, ref_out)
    torch.testing.assert_close(state, ref_state, atol=STATE_ATOL, rtol=RTOL)


@pytest.mark.parametrize("operator", [gated_delta_recurrent, gated_delta_chunked])
def test_qk_normalization_is_inside_the_operator(operator):
    """P-58: raw q/k of any magnitude must give the same answer, with no NaN blow-up.

    The operator l2-normalizes both and scales q by ``K**-0.5``; if that were skipped
    (or applied twice), the WY triangular solve's conditioning would move the result by
    orders of magnitude -- the G0 dry run saw 6e19 and NaN for unnormalized k.
    """
    q, k, v, g, beta = _gdn_inputs(1, 65, seed=850)
    out_a, state_a = operator(q, k, v, g, beta, output_final_state=True)

    # l2norm is scale-invariant up to the eps term, so a 2x q / 0.5x k must not matter.
    out_b, state_b = operator(q * 2.0, k * 0.5, v, g, beta, output_final_state=True)
    torch.testing.assert_close(out_b, out_a, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(state_b, state_a, atol=STATE_ATOL, rtol=RTOL)
    assert torch.isfinite(out_b).all()


@pytest.mark.parametrize("operator", [gated_delta_recurrent, gated_delta_chunked])
def test_operator_shape_and_dtype_errors(operator):
    q, k, v, g, beta = _gdn_inputs(2, 65, seed=860)

    with pytest.raises(ValueError):  # rank 3 q
        operator(q[:, :, 0], k, v, g, beta)
    with pytest.raises(ValueError):  # T = 0 is not a path
        operator(q[:, :0], k[:, :0], v[:, :0], g[:, :0], beta[:, :0])
    with pytest.raises(ValueError):  # g is per value head, no broadcasting
        operator(q, k, v, g[..., :1], beta)
    # H_v must be an integer multiple of H_k (3 value heads, 2 key heads).
    q_odd, k_odd, v_odd, g_odd, beta_odd = _gdn_inputs(
        2, 5, num_k_heads=2, num_v_heads=3, seed=862
    )
    with pytest.raises(ValueError):
        operator(q_odd, k_odd, v_odd, g_odd, beta_odd)
    # H_v < H_k (1 value head, 4 key heads).
    q_few, k_few, v_few, g_few, beta_few = _gdn_inputs(
        2, 5, num_k_heads=4, num_v_heads=1, seed=863
    )
    with pytest.raises(ValueError):
        operator(q_few, k_few, v_few, g_few, beta_few)
    # Transposed state layout (V before K); K != V so the shape check can see it.
    q_rect, k_rect, v_rect, g_rect, beta_rect = _gdn_inputs(
        1, 5, head_k_dim=16, head_v_dim=32, seed=861
    )
    with pytest.raises(ValueError):
        operator(
            q_rect,
            k_rect,
            v_rect,
            g_rect,
            beta_rect,
            initial_state=torch.zeros(1, 2, 32, 16),
        )
    with pytest.raises(TypeError):  # integer tensors are rejected, not upcast
        operator(q.to(torch.int64), k, v, g, beta)
    with pytest.raises(TypeError):
        operator(q, k, v, g.to(torch.int64), beta)


def test_chunked_validates_chunk_size():
    q, k, v, g, beta = _gdn_inputs(1, 65, seed=870)
    with pytest.raises(ValueError):
        gated_delta_chunked(q, k, v, g, beta, chunk_size=0)
    with pytest.raises(ValueError):
        gated_delta_chunked(q, k, v, g, beta, chunk_size=-64)
    with pytest.raises(TypeError):
        gated_delta_chunked(q, k, v, g, beta, chunk_size=63.5)
    with pytest.raises(TypeError):
        gated_delta_chunked(q, k, v, g, beta, chunk_size=True)


# ------------------------------------------------------------------ slow gate


@pytest.mark.slow
def test_chunked_matches_recurrent_at_08b_head_geometry():
    """0.8B's per-layer geometry (H=16, K=V=128) over 16 chunks, with a nonzero state."""
    q, k, v, g, beta = _gdn_inputs(
        2, 1024, num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128, seed=880
    )
    initial_state = _state(2, num_v_heads=16, head_k_dim=128, head_v_dim=128, seed=881)

    out_rec, state_rec = gated_delta_recurrent(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    out_chunk, state_chunk = gated_delta_chunked(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    torch.testing.assert_close(out_chunk, out_rec, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(state_chunk, state_rec, atol=STATE_ATOL, rtol=RTOL)
