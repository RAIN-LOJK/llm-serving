"""S2 guardrail: ``serve/engine.py`` must reproduce the S1 path token for token.

The roadmap (``ROADMAP.md`` §4) makes this the acceptance for S2: extracting the Engine
interface may change structure, never numbers. These tests run on the tiny fixture model,
so they stay in the fast gate and do not need Qwen3 weights.

The Qwen3 end-to-end check is the W1 golden (``test_align_qwen3.py``); this file guards the
prefill/decode split that the golden path does not exercise (it calls ``model.forward``
directly). ``cache_mb`` is the slab's reserved bytes (bucket-rounded), not the grown length.
"""

from __future__ import annotations

import pytest
import torch

from llm_utils.inference import generate
from model import MiniLLM, MiniLLMConfig
from serve.engine import MiniLLMEngine

SLAB_GRANULARITY = 64


def prompt(tiny_model: MiniLLM, length: int = 12) -> torch.Tensor:
    return torch.randint(0, tiny_model.config.vocab_size, (length,))


def _wide_model() -> MiniLLM:
    """Tiny model with room for more than one slab bucket (max_position 256 > 64)."""
    return MiniLLM(
        MiniLLMConfig(
            vocab_size=64, max_position_embeddings=256, n_embd=32, n_head=4, n_kv_head=2,
            n_layer=2, dropout=0.0, bias=False,
        )
    ).eval()


def test_greedy_ids_match_transitional_generate(tiny_model: MiniLLM):
    tokens = prompt(tiny_model)
    engine = MiniLLMEngine(tiny_model)

    got = engine.generate(tokens.tolist(), 8)
    expected = generate(tiny_model, tokens.unsqueeze(0), 8, temperature=0.0)[0]

    assert got.token_ids == expected.tolist()


def test_prefill_decode_equals_one_forward(tiny_model: MiniLLM):
    tokens = prompt(tiny_model, 10)
    engine = MiniLLMEngine(tiny_model)

    logits_prefill, cache = engine.prefill(tokens.tolist())
    with torch.no_grad():
        full, _ = tiny_model(tokens.unsqueeze(0))
    assert torch.allclose(logits_prefill, full[:, -1], atol=1e-5), "prefill logits drifted"

    nxt = int(logits_prefill.argmax(-1))
    logits_decode = engine.decode(nxt, cache)
    with torch.no_grad():
        extended, _ = tiny_model(torch.cat([tokens, torch.tensor([nxt])]).unsqueeze(0))
    assert torch.allclose(logits_decode, extended[:, -1], atol=1e-5), "decode logits drifted"


def test_zero_budget_returns_no_tokens(tiny_model: MiniLLM):
    out = MiniLLMEngine(tiny_model).generate(prompt(tiny_model).tolist(), 0)
    assert out.token_ids == []
    assert out.tpot_ms is None


def test_eos_stops_before_the_budget(tiny_model: MiniLLM):
    tokens = prompt(tiny_model)
    first = MiniLLMEngine(tiny_model).generate(tokens.tolist(), 1).token_ids[0]

    out = MiniLLMEngine(tiny_model, eos_token_id=first).generate(tokens.tolist(), 8)

    assert out.token_ids == [first], "EOS token must not be followed by more decoding"


def test_cache_accounting_reports_the_reserved_bucket():
    """``cache_mb`` is the slab's reserved bytes, rounded up to a 64-column bucket."""
    model = _wide_model()
    engine = MiniLLMEngine(model)
    tokens = prompt(model, 12).tolist()

    small = engine.generate(tokens, 1)
    large = engine.generate(tokens, 60)

    assert small.cache_mb > 0
    assert large.cache_mb > small.cache_mb, "a bigger budget must land in a bigger bucket"
    itemsize = model.token_embedding.weight.element_size()
    for out, budget in ((small, 1), (large, 60)):
        need = len(tokens) + budget
        bucket = min(-(-need // SLAB_GRANULARITY) * SLAB_GRANULARITY, model.config.max_position_embeddings)
        expected = (
            model.config.n_layer * 2 * 1 * model.config.n_kv_head * bucket
            * model.head_dim * itemsize
        ) / 2**20
        assert out.cache_mb == pytest.approx(expected), (
            f"budget {budget}: cache_mb {out.cache_mb} != reserved bucket {bucket} ({expected})"
        )


def test_no_slab_outlives_its_request():
    """W2-0006: the slab is released when ``generate`` returns, like the bucket cache.

    ``memory_match_ratio`` divides the single request's account by the whole process's
    allocator peak. A slab that survives its request is memory on the denominator side
    that the numerator cannot explain -- the same defect that sank W2-0004 with three
    resident buckets. Measured at ctx 1517: retaining it 1532 MB (ratio 0.861),
    releasing it 1348 MB (ratio 0.978). W3 keeps one slab per concurrent slot instead,
    and accounts for it explicitly.
    """
    model = _wide_model()
    engine = MiniLLMEngine(model)
    tokens = prompt(model, 12).tolist()

    small = engine.generate(tokens, 1)  # 13 -> bucket 64
    assert small.cache_mb > 0, "the reservation is still reported"
    assert engine._current_slab is None, "the slab must not outlive the request"

    engine.generate(tokens, 60)  # 72 -> bucket 128
    assert engine._current_slab is None


def test_a_bucket_change_is_a_rebuild_not_an_accumulation():
    """Every request builds its own slab: buckets never stack up in the allocator."""
    model = _wide_model()
    engine = MiniLLMEngine(model)
    tokens = prompt(model, 12).tolist()

    small = engine.generate(tokens, 1)  # 13 -> 64 columns
    large = engine.generate(tokens, 60)  # 72 -> 128 columns

    itemsize = model.token_embedding.weight.element_size()
    per_column = model.config.n_layer * 2 * model.config.n_kv_head * model.head_dim * itemsize / 2**20
    assert large.cache_mb == pytest.approx(2 * SLAB_GRANULARITY * per_column)
    assert small.cache_mb == pytest.approx(SLAB_GRANULARITY * per_column)


def test_lazy_slab_materialises_every_layer_by_the_end():
    """``lazy=True`` defers allocation to ``ensure_layer``; the hand-off must fill all of it."""
    model = _wide_model()
    engine = MiniLLMEngine(model)
    tokens = prompt(model, 12).tolist()

    # The slab is released after every request, so drive the pieces directly.
    with torch.inference_mode():  # fill_from updates tensors in place
        logits, cache = engine.prefill(tokens)
        slab = engine._slab(len(tokens) + 8)
        assert all(k is None for k in slab.keys), "lazy slab must not allocate up front"
        for layer in range(model.config.n_layer):
            slab.ensure_layer(layer)
            slab.fill_from(cache, layer)
            cache.kvs[layer] = None
        past_len = slab.past_len()

    assert all(k is not None for k in slab.keys)
    assert all(v is not None for v in slab.values)
    assert past_len == len(tokens)

def test_prefill_rejects_oversized_prompt(tiny_model: MiniLLM):
    engine = MiniLLMEngine(tiny_model)
    too_long = [0] * (tiny_model.config.max_position_embeddings + 1)
    try:
        engine.prefill(too_long)
    except ValueError as exc:
        assert "max_position" in str(exc)
    else:  # pragma: no cover - the assertion below explains the failure
        raise AssertionError("prompt longer than max_position_embeddings must be rejected")


def test_logits_to_keep_slices_before_the_head(tiny_model: MiniLLM):
    tokens = prompt(tiny_model, 9).unsqueeze(0)
    with torch.no_grad():
        full, _ = tiny_model(tokens)
        kept, _ = tiny_model(tokens, logits_to_keep=1)

    assert kept.shape == (1, 1, tiny_model.config.vocab_size)
    assert torch.allclose(kept[:, -1], full[:, -1], atol=1e-6)


def test_logits_to_keep_refuses_to_feed_a_loss(tiny_model: MiniLLM):
    tokens = prompt(tiny_model, 9).unsqueeze(0)
    with pytest.raises(ValueError, match="inference-only"):
        tiny_model(tokens, targets=tokens, logits_to_keep=1)
