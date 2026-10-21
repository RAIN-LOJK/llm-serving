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
