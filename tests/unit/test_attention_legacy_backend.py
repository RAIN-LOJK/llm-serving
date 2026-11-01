"""``hand_legacy`` is the numerical oracle, so it gets pinned against an *independent*
matmul reference written here from the formulas.

The reference below is not a copy of the backend: it works in fp32, builds causality from
``key_pos <= query_pos`` (not from ``triu``), and masks with a boolean ``masked_fill``
instead of an additive ``-inf`` tensor. If the two disagree, the backend moved.

Shapes covered: full prefill with ``is_causal``, decode (T == 1), chunked prefill
(``past_len > 0`` and ``T > 1``), left pad with per-row padding, an unpadded prefill with no
mask, and fully masked rows. Not covered on purpose: paged/contiguous slabs, CUDA graphs,
fp8 -- see the module docstring. The oracle lives under ``tests/reference/`` (D4); it is not
on the run path and ``model/`` must not import it.
"""

from __future__ import annotations

import math

import pytest
import torch

from model.attention.base import AttentionBackend
from tests.reference.hand_legacy import BACKEND, HandLegacyBackend

DTYPES = [torch.float32, torch.bfloat16]


def _atol(dtype: torch.dtype) -> float:
    """bf16 keeps ~3 decimal digits; a materialized T x L softmax amplifies that."""
    return 1e-5 if dtype == torch.float32 else 2e-2


def _qkv(B, H, H_kv, T, L, D, dtype):
    """Same fp32 draws cast down, so fp32 and bf16 cases describe the same tensors."""
    q = torch.randn(B, H, T, D).to(dtype)
    k = torch.randn(B, H_kv, L, D).to(dtype)
    v = torch.randn(B, H_kv, L, D).to(dtype)
    return q, k, v


def _leftpad_mask(B, H, T, L, pads):
    """Bool allow-mask with ``pads[b]`` dead key columns on the left of row b."""
    mask = torch.ones(B, 1, T, L, dtype=torch.bool)
    for b, pad in enumerate(pads):
        mask[b, :, :, :pad] = False
    return mask


def _reference(q, k, v, *, causal=False, mask=None):
    """fp32 matmul reference: expand GQA, scale, mask, softmax, @ V."""
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    kf = kf.repeat_interleave(qf.shape[1] // kf.shape[1], dim=1)
    vf = vf.repeat_interleave(qf.shape[1] // vf.shape[1], dim=1)

    T, L = qf.shape[-2], kf.shape[-2]
    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(qf.shape[-1])

    allowed = torch.ones(T, L, dtype=torch.bool)
    if causal:
        past_len = L - T
        query_pos = torch.arange(past_len, past_len + T).view(T, 1)
        key_pos = torch.arange(L).view(1, -1)
        allowed = key_pos <= query_pos
    allowed = allowed.expand(qf.shape[0], 1, T, L)
    if mask is not None:
        allowed = allowed & mask.to(torch.bool).expand(qf.shape[0], 1, T, L)

    # a row with no allowed key has no softmax; the historical guard drops its mask row, so
    # the row attends everything (finite, and its output is discarded by the caller)
    allowed = allowed | ~allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vf)


def test_prefill_causal_matches_reference():
    for dtype in DTYPES:
        B, H, H_kv, T, D = 2, 4, 2, 7, 16
        q, k, v = _qkv(B, H, H_kv, T, T, D, dtype)

        out = BACKEND.prefill(q, k, v, is_causal=True)
        ref = _reference(q, k, v, causal=True)

        assert out.shape == (B, H, T, D)
        assert out.dtype == dtype
        torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=_atol(dtype))


def test_prefill_without_mask_attends_everywhere():
    for dtype in DTYPES:
        B, H, H_kv, T, D = 1, 4, 1, 6, 16
        q, k, v = _qkv(B, H, H_kv, T, T, D, dtype)

        out = BACKEND.prefill(q, k, v)  # mask is None and not causal -> no mask at all
        ref = _reference(q, k, v)

        torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=_atol(dtype))


def test_decode_matches_reference():
    for dtype in DTYPES:
        B, H, H_kv, T, L, D = 2, 4, 2, 1, 6, 16
        q, k, v = _qkv(B, H, H_kv, T, L, D, dtype)

        out = BACKEND.decode(q, k, v)
        ref = _reference(q, k, v)

        assert out.shape == (B, H, T, D)
        torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=_atol(dtype))


def test_chunked_prefill_matches_reference():
    for dtype in DTYPES:
        B, H, H_kv, T, L, D = 2, 4, 2, 3, 7, 16  # past_len = 4
        q, k, v = _qkv(B, H, H_kv, T, L, D, dtype)

        out = BACKEND.prefill(q, k, v, is_causal=True)
        ref = _reference(q, k, v, causal=True)

        assert out.shape == (B, H, T, D)
        torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=_atol(dtype))


def test_leftpad_matches_reference():
    for dtype in DTYPES:
        B, H, H_kv, L, D = 2, 4, 2, 8, 16
        T = L
        pads = [3, 1]
        q, k, v = _qkv(B, H, H_kv, T, L, D, dtype)
        mask = _leftpad_mask(B, 1, T, L, pads)

        out = BACKEND.prefill(q, k, v, is_causal=True, mask=mask)
        ref = _reference(q, k, v, causal=True, mask=mask)

        assert torch.isfinite(out).all()
        for b, pad in enumerate(pads):
            # only the non-pad query rows are real; those must match the fp32 reference
            torch.testing.assert_close(
                out[b, :, pad:], ref[b, :, pad:].to(dtype), rtol=0, atol=_atol(dtype)
            )


def test_leftpad_pad_keys_do_not_change_outputs():
    """Changing the pad K/V columns must not move any real query row, even in bf16."""
    for dtype in DTYPES:
        B, H, H_kv, L, D = 2, 4, 2, 9, 16
        T = L
        pads = [4, 2]
        q, k, v = _qkv(B, H, H_kv, T, L, D, dtype)
        mask = _leftpad_mask(B, 1, T, L, pads)

        k_bad, v_bad = k.clone(), v.clone()
        for b, pad in enumerate(pads):
            k_bad[b, :, :pad] = torch.randn(H_kv, pad, D).to(dtype) * 10.0
            v_bad[b, :, :pad] = torch.randn(H_kv, pad, D).to(dtype) * 10.0

        out = BACKEND.prefill(q, k, v, is_causal=True, mask=mask)
        out_bad = BACKEND.prefill(q, k_bad, v_bad, is_causal=True, mask=mask)

        for b, pad in enumerate(pads):
            torch.testing.assert_close(out[b, :, pad:], out_bad[b, :, pad:], rtol=0, atol=0.0)
        # the dead rows are still finite; their values are discarded by the caller
        assert torch.isfinite(out_bad).all()


def test_fully_masked_row_is_finite_and_does_not_poison_other_rows():
    B, H, H_kv, T, L, D = 2, 4, 2, 4, 6, 16
    q, k, v = _qkv(B, H, H_kv, T, L, D, torch.float32)

    mask = torch.ones(B, 1, T, L, dtype=torch.bool)
    mask[0, :, 1, :] = False  # batch 0, query row 1 has no key it may attend
    out = BACKEND.prefill(q, k, v, mask=mask)
    assert torch.isfinite(out).all()

    # zeroing that row (the historical guard) must not perturb the rows around it
    wide = torch.ones(B, 1, T, L, dtype=torch.bool)
    torch.testing.assert_close(out[0, :, 2:], BACKEND.prefill(q, k, v, mask=wide)[0, :, 2:])
    torch.testing.assert_close(out[1], BACKEND.prefill(q, k, v, mask=wide)[1])


def test_fully_masked_rows_with_causal_mask_are_finite():
    B, H, H_kv, T, L, D = 1, 4, 2, 5, 5, 16
    q, k, v = _qkv(B, H, H_kv, T, L, D, torch.bfloat16)

    mask = torch.zeros(B, 1, T, L, dtype=torch.bool)  # nothing allowed, causal or not
    out = BACKEND.prefill(q, k, v, is_causal=True, mask=mask)

    assert torch.isfinite(out).all()
    # the guard drops the mask row entirely, so a dead row is plain full attention
    expected = BACKEND.prefill(q, k, v)  # mask is None -> no mask at all
    torch.testing.assert_close(out, expected)


def test_chunked_then_decode_matches_full_prefill():
    B, H, H_kv, L, D = 2, 4, 2, 8, 16
    q, k, v = _qkv(B, H, H_kv, L, L, D, torch.float32)

    full = BACKEND.prefill(q, k, v, is_causal=True)
    head = BACKEND.prefill(q[:, :, :4], k[:, :, :4], v[:, :, :4], is_causal=True)
    tail = BACKEND.prefill(q[:, :, 4:], k, v, is_causal=True)  # past_len = 4, T = 4
    last = BACKEND.decode(q[:, :, 7:], k, v)

    torch.testing.assert_close(head, full[:, :, :4], rtol=0, atol=1e-6)
    torch.testing.assert_close(tail, full[:, :, 4:], rtol=0, atol=1e-6)
    torch.testing.assert_close(last, full[:, :, 7:], rtol=0, atol=1e-6)


def test_dropout_parameter_drives_the_training_flag():
    B, H, H_kv, T, D = 1, 4, 2, 4, 16
    q, k, v = _qkv(B, H, H_kv, T, T, D, torch.float32)

    clean = BACKEND.prefill(q, k, v, is_causal=True)
    torch.manual_seed(0)
    dropped = BACKEND.prefill(q, k, v, is_causal=True, dropout_p=0.5)
    assert torch.isfinite(dropped).all()
    assert not torch.equal(clean, dropped)  # dropout is on because p > 0, not module.training

    torch.manual_seed(0)
    repeated = BACKEND.prefill(q, k, v, is_causal=True, dropout_p=0.5)
    assert torch.equal(dropped, repeated)


def test_rejects_a_mask_that_does_not_cover_past_keys():
    B, H, H_kv, T, L, D = 1, 4, 2, 2, 7, 16  # chunked: 5 cached + 2 current
    q, k, v = _qkv(B, H, H_kv, T, L, D, torch.float32)

    short = torch.ones(B, 1, T, T, dtype=torch.bool)  # only the new columns (P-05)
    with pytest.raises(ValueError, match="cover every key column"):
        BACKEND.prefill(q, k, v, is_causal=True, mask=short)

    with pytest.raises(TypeError, match="bool allow-mask"):
        BACKEND.prefill(q, k, v, mask=torch.zeros(B, 1, T, L))  # float 0/-inf style

    with pytest.raises(ValueError, match="decode expects T == 1"):
        BACKEND.decode(q, k, v)


def test_backend_identity():
    assert isinstance(BACKEND, HandLegacyBackend)
    assert isinstance(BACKEND, AttentionBackend)  # the Protocol, runtime-checkable
    assert BACKEND.name == "hand_legacy"
    assert not hasattr(BACKEND, "caps")


def test_oracle_is_not_importable_from_model():
    """D4/C2: the old implementation left the run path; ``model/`` must not re-export it."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("model.attention.hand_legacy")
