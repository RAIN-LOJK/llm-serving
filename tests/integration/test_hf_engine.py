"""Optional HF integration smoke.

Excluded from the fast gate (``-m "not slow and not hf"``). Run explicitly after the
W0/W1 download step::

    MINILLM_RUN_HF=1 .venv/bin/python -m pytest tests/integration/test_hf_engine.py -q
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.hf, pytest.mark.slow]


@pytest.mark.skipif(
    os.environ.get("MINILLM_RUN_HF") != "1",
    reason="set MINILLM_RUN_HF=1 to run HF integration tests (downloads a tiny model)",
)
def test_hf_engine_smoke():
    from bench.engines.hf import HFEngine

    engine = HFEngine(
        name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        revision=None,
        dtype="float32",
        device="cpu",
    )
    engine.load()
    try:
        prompt = engine.encode("hello")
        result = engine.generate(prompt, max_new_tokens=2)
        assert 1 <= len(result.token_ids) <= 2
        assert result.ttft_ms is not None
    finally:
        engine.close()


@pytest.mark.skipif(
    os.environ.get("MINILLM_RUN_HF") != "1",
    reason="set MINILLM_RUN_HF=1 to load the real Qwen3.5 text model",
)
def test_qwen35_text_model_extraction():
    """The published Qwen3.5 checkpoint is multimodal; only its text decoder may load."""
    from bench.engines.hf import HFEngine

    engine = HFEngine(
        name="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        dtype="bfloat16",
        device="cpu",
    )
    engine.load()
    try:
        assert engine.arch == "Qwen3_5ForCausalLM"
        assert [n for n, _ in engine.model.named_parameters() if "visual" in n] == []
        assert engine.vocab_size == 248320
        assert engine.weights_mb and engine.weights_mb < 1700
        result = engine.generate(engine.encode("hello"), max_new_tokens=1)
        assert result.state_mb and result.state_mb > 0
        assert result.meta["logits_to_keep"] == 1
    finally:
        engine.close()
