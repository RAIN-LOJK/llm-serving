"""SDPA attention contracts: causal fast path, padding, chunked prefill, GQA.

The engine has one attention path now. These tests pin the three shapes that are easy to
get silently wrong:

* full prefill -> ``is_causal=True`` and no materialized mask;
* decode (one query) -> no mask at all;
* padded / chunked prefill -> an explicit boolean mask, because ``is_causal`` only aligns
  when the query and key blocks start at the same position.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model import KVcache


def _capture_sdpa(monkeypatch):
    calls = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, **kwargs):
        calls.append({"q": q.shape, "k": k.shape, **kwargs})
        return real(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    return calls


def test_full_prefill_uses_causal_sdpa_without_mask(tiny_model, monkeypatch):
    calls = _capture_sdpa(monkeypatch)
    tokens = torch.randint(0, tiny_model.config.vocab_size, (2, 9))

    with torch.no_grad():
        tiny_model(tokens)

    assert len(calls) == tiny_model.config.n_layer
    for call in calls:
        assert call["attn_mask"] is None
        assert call["is_causal"] is True
        assert call["enable_gqa"] is True


def test_decode_uses_no_mask(tiny_model, monkeypatch):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 7))
    cache = KVcache(tiny_model.config.n_layer)

    with torch.no_grad():
        tiny_model(tokens[:, :6], past_keys_values=cache)

    calls = _capture_sdpa(monkeypatch)
    with torch.no_grad():
        tiny_model(tokens[:, 6:7], past_keys_values=cache)

    assert calls
    for call in calls:
        assert call["attn_mask"] is None
        assert call["is_causal"] is False


def test_leftpad_uses_explicit_mask(tiny_model, monkeypatch):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0

    calls = _capture_sdpa(monkeypatch)
    with torch.no_grad():
        tiny_model(tokens, attention_mask=mask)

    assert calls
    for call in calls:
        assert call["attn_mask"] is not None
        assert call["is_causal"] is False


def test_chunked_prefill_matches_full_recompute(tiny_model):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
    cache = KVcache(tiny_model.config.n_layer)

    with torch.no_grad():
        full, _ = tiny_model(tokens)
        tiny_model(tokens[:, :5], past_keys_values=cache)
        chunk, _ = tiny_model(tokens[:, 5:], past_keys_values=cache)

    torch.testing.assert_close(chunk, full[:, 5:], atol=1e-5, rtol=1e-4)


def test_attention_mask_must_cover_all_keys(tiny_model):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
    short_mask = torch.ones((1, 7), dtype=torch.long)

    with pytest.raises(ValueError, match="attention_mask must cover"):
        tiny_model(tokens, attention_mask=short_mask)


def test_all_masked_rows_stay_finite(tiny_model):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0

    with torch.no_grad():
        logits, _ = tiny_model(tokens, attention_mask=mask)

    assert torch.isfinite(logits).all()
