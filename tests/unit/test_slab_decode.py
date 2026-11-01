"""The slab decode path must equal the ``cat``-grown path, token for token.

The slab exists to remove ``torch.cat`` (O(context) copy per step, new address per step).
It changes *where the bytes live*, never the numbers, so the cat path is the reference.
The decode kernel is fixed now (``hand_baddbmm``, D4), so there is no backend switch to
parametrize: this file is "slab container == cat container" on the one shipped kernel.

* same logits after every step as ``MiniLLMEngine.decode(token, cache)``;
* the mask must already allow the column being written this step — if it lagged one step
  behind, the freshly written token would be invisible and the outputs would drift.
"""

from __future__ import annotations

import pytest
import torch

from model import KVcache, KvSlab, MiniLLM, MiniLLMConfig
from serve.engine import MiniLLMEngine

PROMPT = [3, 5, 7, 11, 13, 17, 19]


def _config(**overrides) -> MiniLLMConfig:
    base = dict(vocab_size=64, max_position_embeddings=64, n_embd=32, n_head=4, n_kv_head=2,
                n_layer=2, dropout=0.0, bias=False)
    base.update(overrides)
    return MiniLLMConfig(**base)


def _model(**overrides) -> MiniLLM:
    torch.manual_seed(0)
    return MiniLLM(_config(**overrides)).eval()


def _slab_for(model: MiniLLM, cache: KVcache, batch: int = 1, max_len: int = 32) -> KvSlab:
    slab = KvSlab(
        model.config.n_layer, batch, model.config.n_kv_head, model.head_dim, max_len,
        dtype=model.token_embedding.weight.dtype, device=model.token_embedding.weight.device,
    )
    for layer in range(model.config.n_layer):
        slab.fill_from(cache, layer)
    return slab


def test_slab_decode_matches_cat_decode():
    model = _model()
    engine = MiniLLMEngine(model)

    logits, cache = engine.prefill(PROMPT)
    slab = _slab_for(model, cache)
    assert slab.past_len() == len(PROMPT)

    token = int(logits.argmax(-1))
    for step in range(5):
        # ``MiniLLMEngine.decode`` already returns the last position: (B, V), not (B, T, V).
        reference = engine.decode(token, cache)
        got = engine.decode_slab([token], slab)
        drift = (got - reference).abs().max().item()
        assert drift <= 1e-5, f"step {step}: slab vs cat drifted {drift:.2e}"
        assert int(got.argmax(-1)) == int(reference.argmax(-1))
        token = int(reference.argmax(-1))

    assert slab.past_len() == len(PROMPT) + 5


def test_slab_and_cat_are_bitwise_equal_in_bfloat16():
    """bf16 is the measured dtype: same kernels + same math must give the same bits.

    The slab path passes the precomputed bias instead of the mask, so this is the guard
    that "hoisting the mask" changed no numerics at all (fp32 covers the other direction
    in the test above, where the two orderings differ only by ~6e-8).
    """
    model = _model().to(torch.bfloat16)
    engine = MiniLLMEngine(model)

    logits, cache = engine.prefill(PROMPT)
    slab = _slab_for(model, cache)
    token = int(logits.argmax(-1))
    for step in range(5):
        reference = engine.decode(token, cache)
        got = engine.decode_slab([token], slab)
        assert got.dtype is torch.bfloat16
        assert torch.equal(got, reference), (
            f"step {step}: slab vs cat differ by "
            f"{(got.float() - reference.float()).abs().max().item():.2e} in bf16"
        )
        token = int(reference.argmax(-1))


def test_mask_already_allows_the_column_being_written():
    """Off-by-one guard: the step's own token must be visible to its own attention."""
    model = _model()
    engine = MiniLLMEngine(model)
    logits, cache = engine.prefill(PROMPT)
    slab = _slab_for(model, cache)
    prefix = len(PROMPT)

    token = int(logits.argmax(-1))
    for step in range(3):
        engine.decode_slab([token], slab)
        visible = int(slab.mask.sum())
        assert visible == prefix + step + 1, (
            f"step {step}: mask allows {visible} columns, expected {prefix + step + 1}"
        )
        assert bool(slab.mask[0, 0, 0, : prefix + step + 1].all())
        assert not bool(slab.mask[0, 0, 0, prefix + step + 1 :].any())
        token = 1


def test_slab_path_refuses_prefill():
    model = _model()
    engine = MiniLLMEngine(model)
    _, cache = engine.prefill(PROMPT)
    slab = _slab_for(model, cache)
    with pytest.raises(ValueError, match="decode-only"):
        with torch.no_grad():
            model(torch.tensor([PROMPT]), past_keys_values=slab)


def test_slab_overflow_is_reported():
    model = _model()
    engine = MiniLLMEngine(model)
    logits, cache = engine.prefill(PROMPT)
    slab = _slab_for(model, cache, max_len=len(PROMPT) + 1)

    token = int(logits.argmax(-1))
    engine.decode_slab([token], slab)
    with pytest.raises(IndexError, match="overflow|max_len"):
        engine.decode_slab([token], slab)


def test_fill_from_copies_the_prefill_history():
    model = _model()
    engine = MiniLLMEngine(model)
    _, cache = engine.prefill(PROMPT)

    slab = _slab_for(model, cache)
    for layer in range(model.config.n_layer):
        past_k, past_v = cache.get_kv(layer)
        got_k, got_v = slab.get(layer)
        length = past_k.shape[2]
        assert torch.equal(got_k[:, :, :length], past_k)
        assert torch.equal(got_v[:, :, :length], past_v)


def test_fill_from_per_row_copies_two_different_length_caches():
    """B3: rows admitted with different prompt lengths keep their own seat and cursor."""
    model = _model()
    engine = MiniLLMEngine(model)
    short = PROMPT[:3]
    long = PROMPT[:6]
    _, short_cache = engine.prefill(short)
    _, long_cache = engine.prefill(long)

    slab = KvSlab(
        model.config.n_layer, 2, model.config.n_kv_head, model.head_dim, 16,
        dtype=model.token_embedding.weight.dtype,
    )
    for layer in range(model.config.n_layer):
        slab.fill_from(short_cache, layer, row=0)
        slab.fill_from(long_cache, layer, row=1)

    assert slab.pos.tolist() == [len(short), len(long)]
    assert slab.valid.tolist() == [len(short), len(long)]
    assert slab.past_len() == len(long)

    for layer in range(model.config.n_layer):
        got_k, got_v = slab.get(layer)
        short_k, short_v = short_cache.get_kv(layer)
        long_k, long_v = long_cache.get_kv(layer)
        assert torch.equal(got_k[0, :, : len(short)], short_k[0])
        assert torch.equal(got_v[0, :, : len(short)], short_v[0])
        assert torch.equal(got_k[1, :, : len(long)], long_k[0])
        assert torch.equal(got_v[1, :, : len(long)], long_v[0])

    # the mask follows the per-row cursors
    mask = slab.build_mask(next_token=True)
    assert int(mask[0].sum()) == len(short) + 1
    assert int(mask[1].sum()) == len(long) + 1
