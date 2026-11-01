"""The ``sdpa`` backend, and the fixed dispatch in ``Attention.forward`` around it.

``tests/unit/test_sdpa_attention.py`` already pins *which* mask each serving shape
gets by monkeypatching ``F.scaled_dot_product_attention``. This file pins the other
half of the extraction:

* the backend's output equals a direct ``F.scaled_dot_product_attention`` call
  element for element (the extraction must not change numbers);
* the bool allow-mask is honoured, and a float mask is rejected rather than
  reinterpreted (PITFALLS P-24);
* ``Attention`` holds the two constant backends (``PREFILL_BACKEND`` / ``DECODE_BACKEND``),
  dispatches on ``T``, and lets backend errors escape;
* the model config has no attention knob left (D4).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model import KVcache, MiniLLM, MiniLLMConfig
from model.attention import DECODE_BACKEND, PREFILL_BACKEND
from model.attention.sdpa import BACKEND, SdpaBackend


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tensors(*, batch=2, heads=4, kv_heads=2, queries=5, keys=5, dim=8, seed=0):
    """Random q/k/v already in backend shape: q (B, H, T, D), k/v (B, H_kv, L, D)."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, queries, dim, generator=generator)
    k = torch.randn(batch, kv_heads, keys, dim, generator=generator)
    v = torch.randn(batch, kv_heads, keys, dim, generator=generator)
    return q, k, v


def _reference(q, k, v, *, is_causal=False, mask=None, dropout_p=0.0):
    """The inlined call the backend is supposed to be."""
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=(q.size(1) != k.size(1)),
    )


def _causal_allow_mask(batch, queries, keys, *, invalid_keys=()):
    """Causal bool allow-mask; ``invalid_keys`` additionally hide those key columns.

    Keys are chosen so every query still has at least one allowed key (otherwise SDPA
    would produce an all-masked softmax row and the comparison would be NaN != NaN).
    """
    query_pos = torch.arange(queries).view(queries, 1)
    key_pos = torch.arange(keys).view(1, -1)
    out = (key_pos <= query_pos).view(1, 1, queries, keys).expand(batch, 1, queries, keys).clone()
    for key in invalid_keys:
        out[..., key] = False
    return out


class _RecordingBackend:
    """Delegating spy installed as ``backend_prefill`` / ``backend_decode``."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.calls = []

    def prefill(self, q, k, v, **kwargs):
        self.calls.append(("prefill", kwargs))
        return self.inner.prefill(q, k, v, **kwargs)

    def decode(self, q, k, v, **kwargs):
        self.calls.append(("decode", kwargs))
        return self.inner.decode(q, k, v, **kwargs)


def _record_backends(model):
    prefill, decode = {}, {}
    for index, layer in enumerate(model.layers):
        attn = layer.self_attn
        prefill[index] = _RecordingBackend(attn.backend_prefill)
        decode[index] = _RecordingBackend(attn.backend_decode)
        attn.backend_prefill = prefill[index]
        attn.backend_decode = decode[index]
    return prefill, decode


# ---------------------------------------------------------------------------
# backend identity
# ---------------------------------------------------------------------------

def test_sdpa_backend_is_the_prefill_constant():
    assert isinstance(BACKEND, SdpaBackend)
    assert BACKEND is PREFILL_BACKEND
    assert DECODE_BACKEND is not BACKEND
    assert BACKEND.name == "sdpa"
    assert not hasattr(BACKEND, "caps")


# ---------------------------------------------------------------------------
# kernel equivalence with the inlined call
# ---------------------------------------------------------------------------

def test_prefill_is_causal_matches_inline_sdpa():
    q, k, v = _tensors()
    got = BACKEND.prefill(q, k, v, is_causal=True)

    assert torch.equal(got, _reference(q, k, v, is_causal=True))


def test_prefill_bool_mask_matches_inline_sdpa():
    q, k, v = _tensors(keys=6)
    mask = _causal_allow_mask(2, 5, 6, invalid_keys=(1,))

    got = BACKEND.prefill(q, k, v, mask=mask)

    assert torch.equal(got, _reference(q, k, v, mask=mask))


def test_decode_matches_inline_sdpa():
    q, k, v = _tensors(queries=1, keys=7)

    assert torch.equal(BACKEND.decode(q, k, v), _reference(q, k, v))


def test_decode_bool_mask_matches_inline_sdpa():
    q, k, v = _tensors(queries=1, keys=7)
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    mask[:, :, :, :3] = False

    got = BACKEND.decode(q, k, v, mask=mask)

    assert torch.equal(got, _reference(q, k, v, mask=mask))


def test_bool_mask_hides_masked_keys():
    q, k, v = _tensors(queries=4, keys=5)
    # Bidirectional allow-mask with the first key hidden for every query. Key 0 would
    # otherwise be attended, so its hiding is what the assertion below tests; every row
    # still sees keys 1..4, so no softmax row is fully masked.
    mask = torch.ones(2, 1, 4, 5, dtype=torch.bool)
    mask[..., 0] = False

    baseline = BACKEND.prefill(q, k, v, mask=mask)
    dirty_k = k.clone()
    dirty_v = v.clone()
    dirty_k[..., 0, :] = 1e4
    dirty_v[..., 0, :] = -1e4

    assert torch.equal(BACKEND.prefill(q, dirty_k, dirty_v, mask=mask), baseline)

    # Sanity: the same perturbation of an *allowed* key must change the output.
    clean_k = k.clone()
    clean_v = v.clone()
    clean_k[..., 1, :] = 1e4
    clean_v[..., 1, :] = -1e4
    assert not torch.equal(BACKEND.prefill(q, clean_k, clean_v, mask=mask), baseline)


# ---------------------------------------------------------------------------
# mask contract: bool only, shape (B, 1, T, L)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_float_mask_is_rejected(dtype):
    q, k, v = _tensors()
    mask = torch.zeros(2, 1, 5, 5, dtype=dtype)
    mask[..., 1:] = float("-inf")

    with pytest.raises(ValueError, match="P-24"):
        BACKEND.prefill(q, k, v, mask=mask)


def test_float_mask_is_rejected_on_decode():
    q, k, v = _tensors(queries=1, keys=7)
    mask = torch.zeros(2, 1, 1, 7, dtype=torch.float32)
    mask[:, :, :, :3] = float("-inf")

    with pytest.raises(ValueError, match="bool allow-mask"):
        BACKEND.decode(q, k, v, mask=mask)


def test_mask_shape_must_be_b1tl():
    q, k, v = _tensors()
    mask = torch.ones(2, 5, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"shape \(B, 1, T, L\)"):
        BACKEND.prefill(q, k, v, mask=mask)


# ---------------------------------------------------------------------------
# fixed wiring through Attention
# ---------------------------------------------------------------------------

def test_config_has_no_attention_knob():
    """D4: the attention backend is not a configuration value anymore."""
    config = MiniLLMConfig()
    assert not hasattr(config, "attn_prefill")
    assert not hasattr(config, "attn_decode")


def test_attention_holds_the_two_constants(tiny_model: MiniLLM):
    attn = tiny_model.layers[0].self_attn

    assert attn.backend_prefill is PREFILL_BACKEND
    assert attn.backend_decode is DECODE_BACKEND


# ---------------------------------------------------------------------------
# phase dispatch and error propagation
# ---------------------------------------------------------------------------

def test_multi_token_forward_uses_the_prefill_backend(tiny_model: MiniLLM):
    prefill, decode = _record_backends(tiny_model)
    tokens = torch.randint(0, tiny_model.config.vocab_size, (2, 6))

    with torch.no_grad():
        tiny_model(tokens)

    for index in prefill:
        assert len(prefill[index].calls) == 1, "one dispatch per layer"
        assert not decode[index].calls
        phase, kwargs = prefill[index].calls[0]
        assert phase == "prefill"
        assert kwargs["is_causal"] is True
        assert kwargs["mask"] is None


def test_single_token_without_cache_uses_the_decode_backend(tiny_model: MiniLLM):
    prefill, decode = _record_backends(tiny_model)
    tokens = torch.randint(0, tiny_model.config.vocab_size, (2, 1))

    with torch.no_grad():
        tiny_model(tokens)

    for index in decode:
        assert len(decode[index].calls) == 1
        assert not prefill[index].calls
        phase, kwargs = decode[index].calls[0]
        assert phase == "decode"
        assert kwargs["mask"] is None


def test_chunked_prefill_uses_the_prefill_backend_with_a_bool_mask(tiny_model: MiniLLM):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
    cache = KVcache(tiny_model.config.n_layer)
    with torch.no_grad():
        tiny_model(tokens[:, :5], past_keys_values=cache)

    prefill, decode = _record_backends(tiny_model)
    with torch.no_grad():
        tiny_model(tokens[:, 5:], past_keys_values=cache)

    for index in prefill:
        assert len(prefill[index].calls) == 1
        assert not decode[index].calls
        phase, kwargs = prefill[index].calls[0]
        assert phase == "prefill"
        assert kwargs["is_causal"] is False
        assert kwargs["mask"].dtype is torch.bool
        assert tuple(kwargs["mask"].shape) == (1, 1, 3, 8)


def test_backend_errors_are_not_swallowed(tiny_model: MiniLLM):
    class Exploding:
        name = "exploding"

        def prefill(self, *args, **kwargs):
            raise NotImplementedError("prefill kernel is not implemented")

        def decode(self, *args, **kwargs):
            raise NotImplementedError("decode kernel is not implemented")

    tiny_model.layers[0].self_attn.backend_decode = Exploding()
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 1))

    with pytest.raises(NotImplementedError, match="decode kernel is not implemented"):
        with torch.no_grad():
            tiny_model(tokens)
