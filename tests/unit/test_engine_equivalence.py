"""S2 guardrail: ``serve/engine.py`` must reproduce the S1 path token for token.

The roadmap (``ROADMAP.md`` §4) makes this the acceptance for S2: extracting the Engine
interface may change structure, never numbers. These tests run on the tiny fixture model,
so they stay in the fast gate and do not need Qwen3 weights.

The Qwen3 end-to-end check is the W1 golden (``test_align_qwen3.py``); this file guards the
prefill/decode split that the golden path does not exercise (it calls ``model.forward``
directly).
"""

from __future__ import annotations

import pytest
import torch

from llm_utils.inference import generate
from model import MiniLLM
from serve.engine import MiniLLMEngine


def prompt(tiny_model: MiniLLM, length: int = 12) -> torch.Tensor:
    return torch.randint(0, tiny_model.config.vocab_size, (length,))


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


def test_cache_accounting_grows_with_decoding(tiny_model: MiniLLM):
    tokens = prompt(tiny_model).tolist()
    engine = MiniLLMEngine(tiny_model)

    short = engine.generate(tokens, 1)
    long = engine.generate(tokens, 6)

    assert long.cache_mb > short.cache_mb > 0


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
