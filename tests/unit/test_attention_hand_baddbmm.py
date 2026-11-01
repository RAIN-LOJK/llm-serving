"""``hand_baddbmm`` is the decode-only *performance* path, so it is pinned on two axes.

**Numerics.** Every shape decode actually sees is compared against ``hand_legacy`` (the
oracle, now under ``tests/reference/``): ragged left pad with per-row valid lengths, all-valid, ``n_kv_head == n_head``
(GQA with a single group) and B == 1, in fp32 and bf16. The backend's promise is that the
math is unchanged while the per-sequence dispatch disappears, so the oracle is the right
reference and the tolerances are the same ones the oracle matrix uses.

**Cost.** One ``decode`` call must issue a batch-independent handful of matmul-family ops.
The regression this blocks is "one dispatch per sequence": measured on a 4090 (B=8, L=512,
D=128, 28 layers) the batched form is ~0.7 ms/step while the SDPA math fallback is
~7.4 ms/step, and a loop over B is slower still. A per-sequence loop would show up here as
matmul counts that scale with B.

Only the paths the backend declares are exercised: prefill raises, ``T > 1`` raises, a float
mask is a ``ValueError``, and fully masked rows stay finite. Not covered: paged/contiguous
KV slabs, CUDA-graph capture itself, fp8 -- see the backend module docstring.
"""

from __future__ import annotations

import pytest
import torch

from model.attention.base import AttentionBackend
from model.attention.hand_baddbmm import BACKEND, HandBaddbmmBackend
from tests.reference.hand_legacy import BACKEND as ORACLE

DTYPES = [torch.float32, torch.bfloat16]


def _atol(dtype: torch.dtype) -> float:
    """bf16 keeps ~3 decimal digits; softmax over L keys amplifies that (same as the oracle)."""
    return 1e-5 if dtype == torch.float32 else 2e-2


def _qkv(B, H, Hkv, L, D, dtype, seed=0):
    """fp32 draws cast down, so the fp32 and bf16 cases describe the same tensors."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, 1, D, generator=g).to(dtype)
    k = torch.randn(B, Hkv, L, D, generator=g).to(dtype)
    v = torch.randn(B, Hkv, L, D, generator=g).to(dtype)
    return q, k, v


def _ragged_mask(B, L, valid):
    """Bool allow-mask ``(B, 1, 1, L)`` where row b may attend ``valid[b]`` key columns."""
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    for b, n in enumerate(valid):
        mask[b, :, :, n:] = False
    return mask


# (id, B, H, H_kv, L, per-row valid key lengths or None for no mask)
SHAPES = [
    pytest.param("ragged", 3, 4, 2, 17, [17, 12, 7], id="ragged"),
    pytest.param("all-valid", 2, 4, 2, 9, [9, 9], id="all-valid"),
    pytest.param("mha", 2, 4, 4, 11, None, id="n_kv_head-eq-n_head"),
    pytest.param("b1", 1, 4, 2, 5, None, id="B-1"),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,B,H,Hkv,L,valid", SHAPES)
def test_decode_matches_the_legacy_oracle(name, B, H, Hkv, L, valid, dtype):
    q, k, v = _qkv(B, H, Hkv, L, 8, dtype, seed=sum(map(ord, name)))
    mask = None if valid is None else _ragged_mask(B, L, valid)

    out = BACKEND.decode(q, k, v, mask=mask)
    ref = ORACLE.decode(q, k, v, mask=mask)

    assert out.shape == (B, H, 1, 8)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=_atol(dtype))


def test_float_mask_is_rejected():
    q, k, v = _qkv(2, 4, 2, 9, 16, torch.float32)

    additive = torch.zeros(2, 1, 1, 9)
    additive[..., 5:] = float("-inf")  # the P-24 shape: -inf in bf16 drifts up to 1.5e-1
    with pytest.raises(ValueError, match="bool allow-mask"):
        BACKEND.decode(q, k, v, mask=additive)

    with pytest.raises(ValueError, match="bool allow-mask"):
        BACKEND.decode(q, k, v, mask=torch.ones(2, 1, 1, 9))  # 0/1 float, also not the contract


def test_mask_must_cover_every_key_column():
    q, k, v = _qkv(2, 4, 2, 9, 16, torch.float32)

    with pytest.raises(ValueError, match="cover every key column"):
        BACKEND.decode(q, k, v, mask=torch.ones(2, 1, 1, 8, dtype=torch.bool))  # P-05

    with pytest.raises(ValueError, match="mask batch must be 1 or"):
        BACKEND.decode(q, k, v, mask=torch.ones(3, 1, 1, 9, dtype=torch.bool))


def test_prefill_is_not_implemented():
    q, k, v = _qkv(2, 4, 2, 5, 16, torch.float32)

    with pytest.raises(NotImplementedError, match="decode-only"):
        BACKEND.prefill(q, k, v)
    with pytest.raises(NotImplementedError, match="decode-only"):
        BACKEND.prefill(q, k, v, is_causal=True)


def test_decode_rejects_more_than_one_query():
    q, k, v = _qkv(2, 4, 2, 6, 16, torch.float32)

    # a multi-token query is prefill, and this backend has no non-causal branch for it
    with pytest.raises(ValueError, match="decode expects T == 1"):
        BACKEND.decode(torch.randn(2, 4, 2, 16), k, v)
    with pytest.raises(ValueError, match="decode expects T == 1"):
        BACKEND.decode(torch.randn(2, 4, 3, 16), k, v)


def test_fully_masked_rows_are_finite_and_do_not_poison_other_rows():
    for dtype in DTYPES:
        B, H, Hkv, L, D = 2, 4, 2, 9, 16
        q, k, v = _qkv(B, H, Hkv, L, D, dtype)

        mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
        mask[0] = False  # left-padded query with no allowed key at all
        out = BACKEND.decode(q, k, v, mask=mask)
        assert torch.isfinite(out).all()

        # the guard zeroes that row's bias, so it degrades to plain full attention (finite,
        # and discarded by the caller) rather than exp(-inf)/0 -> NaN
        torch.testing.assert_close(out[0], BACKEND.decode(q, k, v)[0], rtol=0, atol=0.0)
        # and zeroing it must leave the live rows bit-for-bit alone
        wide = torch.ones_like(mask)
        torch.testing.assert_close(out[1], BACKEND.decode(q, k, v, mask=wide)[1], rtol=0, atol=0.0)


def test_dropout_parameter_drives_the_training_flag():
    q, k, v = _qkv(1, 4, 2, 7, 16, torch.float32)

    clean = BACKEND.decode(q, k, v)
    torch.manual_seed(0)
    dropped = BACKEND.decode(q, k, v, dropout_p=0.5)
    assert torch.isfinite(dropped).all()
    assert not torch.equal(clean, dropped)  # dropout is on because p > 0, not module.training

    torch.manual_seed(0)
    assert torch.equal(dropped, BACKEND.decode(q, k, v, dropout_p=0.5))


def test_one_decode_call_is_batch_independent(monkeypatch):
    """The cost model this backend exists for: dispatch count must not track B.

    ``monkeypatch`` replaces the module attributes, so the backend must call
    ``torch.baddbmm`` / ``torch.bmm`` through ``torch`` (never a from-import) -- which is
    also what makes the count meaningful.
    """
    counts = {"matmul": 0, "baddbmm": 0, "bmm": 0}
    real = {name: getattr(torch, name) for name in counts}

    for name in counts:
        def counted(*args, _real=real[name], _name=name, **kwargs):
            counts[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, counted)

    per_batch = {}
    for B in (1, 8):
        q, k, v = _qkv(B, 4, 2, 64, 16, torch.bfloat16)
        mask = _ragged_mask(B, 64, [64] * B)
        counts.update({name: 0 for name in counts})

        out = BACKEND.decode(q, k, v, mask=mask)

        assert out.shape == (B, 4, 1, 16)
        per_batch[B] = dict(counts)
        assert sum(counts.values()) <= 3, f"B={B}: {counts}"

    assert per_batch[1] == per_batch[8], (
        "the op count changed with batch size -- something is dispatching per sequence: "
        f"{per_batch}"
    )
    # and the handful is the expected one: mask fused into baddbmm, then probs @ V
    assert per_batch[8] == {"matmul": 0, "baddbmm": 1, "bmm": 1}

    # the unmasked path is the same shape of call, only without the bias
    counts.update({name: 0 for name in counts})
    q, k, v = _qkv(8, 4, 2, 64, 16, torch.bfloat16)
    BACKEND.decode(q, k, v)
    assert sum(counts.values()) <= 3, counts
    assert counts["baddbmm"] == 0 and counts["bmm"] == 2


def test_backend_identity():
    assert isinstance(BACKEND, HandBaddbmmBackend)
    assert isinstance(BACKEND, AttentionBackend)  # the Protocol, runtime-checkable
    assert BACKEND.name == "hand_baddbmm"
    assert not hasattr(BACKEND, "caps")


# ------------------------------------------------------------------ precomputed bias

def test_precomputed_bias_matches_the_mask_path():
    """``bias=`` is the layer-hoisted mask: identical numbers, one fewer materialization."""
    from model.attention.hand_baddbmm import additive_bias

    for dtype in DTYPES:
        q, k, v = _qkv(3, 4, 2, 17, 8, dtype, seed=1)
        mask = _ragged_mask(3, 17, [17, 12, 7])
        bias = additive_bias(mask, dtype, 3, 2, 2)

        via_mask = BACKEND.decode(q, k, v, mask=mask)
        via_bias = BACKEND.decode(q, k, v, bias=bias)
        torch.testing.assert_close(via_bias, via_mask, rtol=0, atol=0.0)


def test_precomputed_bias_accepts_the_broadcast_group_axis():
    """The slab keeps ``(B*H_kv, 1, L)`` and lets ``baddbmm`` broadcast over G."""
    from model.attention.hand_baddbmm import additive_bias

    q, k, v = _qkv(2, 4, 2, 9, 8, torch.float32)
    mask = _ragged_mask(2, 9, [9, 4])
    wide = additive_bias(mask, torch.float32, 2, 2, 2)
    narrow = wide[:, :1, :]
    torch.testing.assert_close(
        BACKEND.decode(q, k, v, bias=narrow), BACKEND.decode(q, k, v, bias=wide), rtol=0, atol=0.0
    )


def test_mask_and_bias_together_are_rejected():
    q, k, v = _qkv(1, 4, 2, 5, 8, torch.float32)
    mask = torch.ones(1, 1, 1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="not both"):
        BACKEND.decode(q, k, v, mask=mask, bias=torch.zeros(2, 2, 5))


def test_a_bool_tensor_is_not_accepted_as_bias():
    q, k, v = _qkv(1, 4, 2, 5, 8, torch.float32)
    with pytest.raises(ValueError, match="bias must be the additive form"):
        BACKEND.decode(q, k, v, bias=torch.ones(2, 2, 5, dtype=torch.bool))

