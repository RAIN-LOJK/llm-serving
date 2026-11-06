"""Packed-prefill kernel contracts: per-segment causality, GQA, dtypes, device choice.

W4.5 adds a second prefill path: a whole arrival group is packed into one
``(total_tokens, H, D)`` call instead of one padded ``(B, T_max)`` slab. The failure this
file is written to catch is cross-request leakage: a kernel that lets request 1's queries
attend over request 0's tokens returns plausible numbers and a wrong first token, with no
exception anywhere. So the oracle below is written out longhand (scores -> causal mask ->
softmax -> ``@ v``) instead of calling SDPA, and the leakage case perturbs one segment and
demands the other segments come back bitwise identical.

The device tests pin the other half of the contract: ``backend_for`` is selection, not
fallback -- CUDA means ``FlashVarlen`` or a raised error, never quietly SDPA. The CUDA arm
is compared against the SDPA arm on the same packed input (bf16, ``max|d| < 5e-2``); token
argmax equivalence belongs to the later E0 gate, not here.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from model.attention.varlen import FlashVarlen, SdpaVarlen, backend_for

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the flash-attn varlen arm needs a CUDA device"
)

HEAD_DIM = 128

# 1, 2, 4 and 8 segments; lengths cover 1, 17, 128 and 1517, including a group whose
# segments are all different (the shape the E3 packing step actually produces).
LENGTH_SETS = [
    pytest.param([1], id="1seg-len1"),
    pytest.param([17], id="1seg-len17"),
    pytest.param([128], id="1seg-len128"),
    pytest.param([1517], id="1seg-len1517"),
    pytest.param([1, 17], id="2seg-1-17"),
    pytest.param([128, 1517], id="2seg-128-1517"),
    pytest.param([1, 17, 128, 1517], id="4seg-1-17-128-1517"),
    pytest.param([1, 1, 17, 128, 1517, 3, 64, 200], id="8seg-mixed"),
]


def _packed(lengths, n_head, n_kv_head, head_dim, *, dtype=torch.float32, seed=0, device="cpu"):
    """Build a packed group: q/k/v are (total_tokens, heads, head_dim), cu is n_seqs + 1.

    Values are drawn in fp32 and cast, so bf16/fp16 runs quantize the same inputs the fp32
    oracle sees; drawing directly in bf16 would hide a dtype bug behind a different draw.
    """
    generator = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    q = torch.randn(total, n_head, head_dim, generator=generator)
    k = torch.randn(total, n_kv_head, head_dim, generator=generator)
    v = torch.randn(total, n_kv_head, head_dim, generator=generator)
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return (
        q.to(device=device, dtype=dtype),
        k.to(device=device, dtype=dtype),
        v.to(device=device, dtype=dtype),
        cu_seqlens,
        max(lengths),
    )


def _oracle(q, k, v, *, cu_seqlens, scale, causal):
    """Explicit softmax attention per segment, accumulated in fp32.

    Longhand on purpose: it must not share a call with ``SdpaVarlen``, otherwise a wrong
    mask or scale in the module would be mirrored by the oracle. GQA is expanded with
    ``repeat_interleave`` so q head ``h`` reads kv head ``h // group``, the mapping both
    SDPA's ``enable_gqa`` and FA2's varlen kernel use.
    """
    group = q.size(1) // k.size(1)
    outputs = []
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        q_seg = q[start:end].float().transpose(0, 1)  # (H, T, D)
        k_seg = k[start:end].float().transpose(0, 1).repeat_interleave(group, dim=0)
        v_seg = v[start:end].float().transpose(0, 1).repeat_interleave(group, dim=0)
        scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale
        if causal:
            length = scores.size(-1)
            allowed = torch.ones(length, length, dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_seg).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _sdpa_one(q_seg, k_seg, v_seg, *, causal):
    """One segment through SDPA with an explicit bool mask (not ``is_causal``).

    Used as the "individually computed sequences" arm: it is a different mask path from the
    module's, so a bug in how ``SdpaVarlen`` slices or forwards ``is_causal`` cannot cancel
    out against itself.
    """
    length = q_seg.size(0)
    mask = (
        torch.ones(length, length, dtype=torch.bool).tril() if causal else None
    )
    out = F.scaled_dot_product_attention(
        q_seg.transpose(0, 1).unsqueeze(0),
        k_seg.transpose(0, 1).unsqueeze(0),
        v_seg.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        is_causal=False,
        enable_gqa=(q_seg.size(1) != k_seg.size(1)),
    )
    return out.squeeze(0).transpose(0, 1)


# ---------------------------------------------------------------------------
# 1. numeric contract: per-segment oracle, 1/2/4/8 segments, GQA 2:1 and 8:1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,atol",
    [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)],
    ids=["fp32", "bf16"],
)
@pytest.mark.parametrize(
    "n_head,n_kv_head",
    [(8, 4), (8, 1)],
    ids=["gqa-2to1", "gqa-8to1"],
)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_sdpa_varlen_matches_the_per_segment_oracle(lengths, n_head, n_kv_head, dtype, atol):
    q, k, v, cu_seqlens, max_seqlen = _packed(
        lengths, n_head, n_kv_head, HEAD_DIM, dtype=dtype
    )

    out = SdpaVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    reference = _oracle(
        q, k, v, cu_seqlens=cu_seqlens, scale=1.0 / math.sqrt(HEAD_DIM), causal=True
    )
    assert out.shape == (sum(lengths), n_head, HEAD_DIM)
    assert out.dtype == dtype, "the packed output must stay in q.dtype"
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), reference, atol=atol, rtol=0.0)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "dtype,atol",
    [(torch.float32, 1e-5), (torch.bfloat16, 2e-2), (torch.float16, 2e-2)],
    ids=["fp32", "bf16", "fp16"],
)
def test_sdpa_varlen_head_dims_and_dtypes(head_dim, dtype, atol):
    """The three head dims and three dtypes the carrier family needs (W4.5 scope)."""
    lengths = [1, 33, 128]
    q, k, v, cu_seqlens, max_seqlen = _packed(lengths, 8, 4, head_dim, dtype=dtype)

    out = SdpaVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    reference = _oracle(
        q, k, v, cu_seqlens=cu_seqlens, scale=1.0 / math.sqrt(head_dim), causal=True
    )
    assert out.shape == (sum(lengths), 8, head_dim)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), reference, atol=atol, rtol=0.0)


def test_sdpa_varlen_non_causal_matches_the_oracle():
    """``causal=False`` must reach SDPA, not be hard-coded to the prefill case."""
    q, k, v, cu_seqlens, max_seqlen = _packed([17, 64], 8, 1, HEAD_DIM)

    out = SdpaVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, causal=False
    )

    reference = _oracle(
        q, k, v, cu_seqlens=cu_seqlens, scale=1.0 / math.sqrt(HEAD_DIM), causal=False
    )
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=0.0)


def test_softmax_scale_is_forwarded_not_defaulted():
    """A caller-provided scale must reach the kernel; ``None`` is 1/sqrt(head_dim)."""
    q, k, v, cu_seqlens, max_seqlen = _packed([9, 64], 8, 4, HEAD_DIM)
    backend = SdpaVarlen()

    scaled = backend.prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=0.5
    )
    default = backend.prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    reference = _oracle(q, k, v, cu_seqlens=cu_seqlens, scale=0.5, causal=True)
    torch.testing.assert_close(scaled, reference, atol=1e-5, rtol=0.0)
    assert not torch.allclose(scaled, default), "softmax_scale=0.5 had no effect"


def test_sdpa_varlen_dispatches_one_sdpa_call_per_segment(monkeypatch):
    """The reference arm's contract is a slice per request, not one call over the pack.

    The spy also pins ``is_causal``, ``enable_gqa`` and the forwarded scale: those three
    kwargs are what make the per-segment loop equivalent to the varlen kernel.
    """
    calls = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, **kwargs):
        calls.append({"shape": tuple(q.shape), **kwargs})
        return real(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)

    lengths = [17, 128, 1]
    q, k, v, cu_seqlens, max_seqlen = _packed(lengths, 8, 4, HEAD_DIM)
    SdpaVarlen().prefill_varlen(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    assert len(calls) == len(lengths)
    for call, length in zip(calls, lengths):
        assert call["shape"] == (1, 8, length, HEAD_DIM)
        assert call["is_causal"] is True
        assert call["enable_gqa"] is True
        assert call["attn_mask"] is None
        assert call["scale"] == pytest.approx(1.0 / math.sqrt(HEAD_DIM))

    # Equal head counts are MHA: no GQA expansion is requested.
    q, k, v, cu_seqlens, max_seqlen = _packed([17], 4, 4, HEAD_DIM)
    SdpaVarlen().prefill_varlen(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    assert calls[-1]["enable_gqa"] is False


def test_cu_seqlens_accepts_an_int_tensor_as_well_as_a_list():
    q, k, v, cu_seqlens, max_seqlen = _packed([17, 64, 3], 8, 4, HEAD_DIM)
    backend = SdpaVarlen()

    from_list = backend.prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )
    from_int32 = backend.prefill_varlen(
        q,
        k,
        v,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
        max_seqlen=max_seqlen,
    )
    from_int64 = backend.prefill_varlen(
        q,
        k,
        v,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64),
        max_seqlen=max_seqlen,
    )

    # Same inputs, same code path: the container of the offsets cannot change the numerics.
    assert torch.equal(from_list, from_int32)
    assert torch.equal(from_list, from_int64)


# ---------------------------------------------------------------------------
# 2. no cross-request leakage
# ---------------------------------------------------------------------------


def test_packed_call_equals_individually_computed_sequences():
    """Packing must be a scheduling detail: per-request outputs are unchanged."""
    lengths = [17, 1, 128]
    q, k, v, cu_seqlens, max_seqlen = _packed(lengths, 8, 4, HEAD_DIM, seed=3)
    backend = SdpaVarlen()

    packed = backend.prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    individual = torch.cat(
        [
            _sdpa_one(q[start:end], k[start:end], v[start:end], causal=True)
            for start, end in zip(cu_seqlens, cu_seqlens[1:])
        ]
    )
    torch.testing.assert_close(packed, individual, atol=1e-6, rtol=0.0)

    # The same requests as three one-segment packed calls: identical by construction, so a
    # slice/offset bug shows up as a mismatch instead of a tolerance-level wobble.
    one_at_a_time = torch.cat(
        [
            backend.prefill_varlen(
                q[start:end],
                k[start:end],
                v[start:end],
                cu_seqlens=[0, end - start],
                max_seqlen=end - start,
            )
            for start, end in zip(cu_seqlens, cu_seqlens[1:])
        ]
    )
    assert torch.equal(packed, one_at_a_time)


def test_changing_segment_zero_does_not_change_the_other_segments():
    """The regression that matters: request 0's tokens must not influence request 1.

    Segment 0 is overwritten with new values (lengths and ``cu_seqlens`` unchanged), so the
    packed rows of segments 1 and 2 are untouched. Their outputs must be bitwise identical;
    any leakage would move them.
    """
    lengths = [17, 1, 128]
    q, k, v, cu_seqlens, max_seqlen = _packed(lengths, 8, 4, HEAD_DIM, seed=5)
    backend = SdpaVarlen()
    first_end = cu_seqlens[1]

    before = backend.prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    generator = torch.Generator().manual_seed(99)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:first_end] = torch.randn(first_end, 8, HEAD_DIM, generator=generator)
    k2[:first_end] = torch.randn(first_end, 4, HEAD_DIM, generator=generator)
    v2[:first_end] = torch.randn(first_end, 4, HEAD_DIM, generator=generator)

    after = backend.prefill_varlen(
        q2, k2, v2, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    assert not torch.equal(after[:first_end], before[:first_end]), (
        "the perturbation was not visible in segment 0, so the isolation assertion below "
        "would be vacuous"
    )
    assert torch.equal(after[first_end:], before[first_end:])


# ---------------------------------------------------------------------------
# 3. validation: shape and cu_seqlens errors are ValueError, never silent repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cu_seqlens,max_seqlen,match",
    [
        pytest.param([1, 3, 8], 5, "must start at 0", id="non-zero-start"),
        pytest.param([0, 5, 3, 8], 5, "strictly increasing", id="descending"),
        pytest.param([0, 4, 7], 4, "must end at", id="wrong-end"),
        pytest.param([0, 4, 8], 5, "max_seqlen", id="max-seqlen-mismatch"),
        pytest.param([0, 4, 8], 3, "max_seqlen", id="max-seqlen-too-small"),
        pytest.param([0, 4, 4, 8], 4, "strictly increasing", id="empty-segment"),
        pytest.param([0], 0, "needs n_seqs", id="no-final-offset"),
    ],
)
def test_cu_seqlens_validation(cu_seqlens, max_seqlen, match):
    q, k, v, _, _ = _packed([8], 8, 4, HEAD_DIM)  # total_tokens = 8
    with pytest.raises(ValueError, match=match):
        SdpaVarlen().prefill_varlen(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )


def test_cu_seqlens_tensor_shape_and_dtype_are_validated():
    q, k, v, cu_seqlens, max_seqlen = _packed([4, 4], 8, 4, HEAD_DIM)
    backend = SdpaVarlen()

    with pytest.raises(ValueError, match="1-D"):
        backend.prefill_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor([cu_seqlens], dtype=torch.int32),
            max_seqlen=max_seqlen,
        )
    with pytest.raises(ValueError, match="integers"):
        backend.prefill_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.float32),
            max_seqlen=max_seqlen,
        )


def test_packed_shape_and_dtype_validation():
    q, k, v, cu_seqlens, max_seqlen = _packed([4, 4], 8, 4, HEAD_DIM)
    backend = SdpaVarlen()

    with pytest.raises(ValueError, match="3-D"):
        backend.prefill_varlen(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
    with pytest.raises(ValueError, match="same packed tokens"):
        backend.prefill_varlen(
            q, k[:-1], v[:-1], cu_seqlens=[0, 4], max_seqlen=4
        )
    with pytest.raises(ValueError, match="multiple of n_kv_head"):
        backend.prefill_varlen(
            q, k[:, :3], v[:, :3], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
    with pytest.raises(ValueError, match="share head_dim"):
        backend.prefill_varlen(
            q,
            k[:, :, :64],
            v[:, :, :64],
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
    with pytest.raises(ValueError, match="share one dtype"):
        backend.prefill_varlen(
            q,
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
    with pytest.raises(ValueError, match="floating-point"):
        backend.prefill_varlen(
            q.to(torch.int32),
            k.to(torch.int32),
            v.to(torch.int32),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )


# ---------------------------------------------------------------------------
# 4. device selection: explicit arms, loud failure, no availability fallback
# ---------------------------------------------------------------------------


def test_backend_for_cpu_is_sdpa_varlen():
    for device in ("cpu", torch.device("cpu")):
        backend = backend_for(device)
        assert isinstance(backend, SdpaVarlen)
        assert backend.name == "sdpa-varlen"


def test_backend_for_mps_selects_sdpa_varlen():
    """Device *type* selection only; no MPS tensor is created, so this runs on Linux too."""
    assert isinstance(backend_for(torch.device("mps")), SdpaVarlen)


def test_backend_for_rejects_unknown_device_types():
    with pytest.raises(ValueError, match="no packed-prefill backend"):
        backend_for("meta")


def test_flash_varlen_fails_loudly_when_flash_attn_is_absent(monkeypatch):
    """A CUDA device with no wheel is a raised error, not a silent SDPA run."""
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    with pytest.raises(RuntimeError, match="flash-attn"):
        FlashVarlen()


@requires_cuda
def test_backend_for_cuda_is_flash_varlen():
    for device in ("cuda", torch.device("cuda:0")):
        backend = backend_for(device)
        assert isinstance(backend, FlashVarlen)
        assert backend.name == "flash-attn-varlen"


@requires_cuda
def test_flash_varlen_rejects_non_cuda_inputs():
    q, k, v, cu_seqlens, max_seqlen = _packed(
        [4, 4], 8, 4, HEAD_DIM, dtype=torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="CUDA kernel path"):
        FlashVarlen().prefill_varlen(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )


@requires_cuda
def test_flash_varlen_rejects_fp32():
    """FA2 is fp16/bf16 only; the wrapper says so instead of surfacing a CUDA error."""
    q, k, v, cu_seqlens, max_seqlen = _packed(
        [4, 4], 8, 4, HEAD_DIM, dtype=torch.float32, device="cuda"
    )
    with pytest.raises(RuntimeError, match="fp16/bf16"):
        FlashVarlen().prefill_varlen(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )


# ---------------------------------------------------------------------------
# 5. CUDA arm vs the SDPA arm on the same packed input (E0's kernel-level gate)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([128, 1517], id="2seg-128-1517"),
        pytest.param([128, 512, 1517, 512], id="4seg"),
        pytest.param([128, 512, 1517, 128, 512, 128, 512, 1517], id="8seg"),
    ],
)
def test_flash_varlen_matches_sdpa_on_the_same_packed_input(lengths):
    """bf16 FA2 vs the per-segment SDPA reference, GQA 2:1.

    ``max|d| < 5e-2`` is the kernel-level bar. Token-argmax equivalence is deliberately
    not asserted here: that is the later E0 gate over the full 0.6B stack, where a near-tie
    can be judged against an fp32 oracle instead of being pinned to one kernel's rounding.
    """
    q, k, v, cu_seqlens, max_seqlen = _packed(
        lengths, 8, 4, HEAD_DIM, dtype=torch.bfloat16, seed=7, device="cuda"
    )

    reference = SdpaVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )
    out = FlashVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    assert out.shape == reference.shape == (sum(lengths), 8, HEAD_DIM)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    delta = (out.float() - reference.float()).abs().max().item()
    assert delta < 5e-2, f"bf16 FA2 vs SDPA max|d|={delta:.4e} for lengths {lengths}"


@requires_cuda
def test_flash_varlen_matches_sdpa_with_a_custom_softmax_scale():
    q, k, v, cu_seqlens, max_seqlen = _packed(
        [17, 64], 8, 4, HEAD_DIM, dtype=torch.bfloat16, seed=11, device="cuda"
    )

    reference = SdpaVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=0.37
    )
    out = FlashVarlen().prefill_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=0.37
    )

    delta = (out.float() - reference.float()).abs().max().item()
    assert delta < 5e-2, f"softmax_scale was not forwarded consistently: max|d|={delta:.4e}"
