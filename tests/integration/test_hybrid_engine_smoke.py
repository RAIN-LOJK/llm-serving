"""W7 Step 2 real-model smoke: 0.8B bf16/CUDA through ``Qwen35Engine`` (paged).

**This is a smoke, not E0.** It loads the pinned 0.8B text decoder, opens the paged
hybrid pool with two linear slots, prefills two short prompts, decodes two steps at
``B=2``, releases both requests and closes with no leak. It asserts finiteness and the
lifecycle ledger only -- no golden, no HF comparison, no benchmark and no formal run.
The correctness gates live in Step 0/1/2 unit tests and the separate E0 dry run.

Run explicitly (never part of the fast gate)::

    MINILLM_RUN_HF=1 python -m pytest tests/integration/test_hybrid_engine_smoke.py -q -s

Skipped unless ``MINILLM_RUN_HF=1``; also requires CUDA, Triton and the pinned snapshot in
the local HF cache (no network).
"""

from __future__ import annotations

import os

import pytest
import torch

from serve.hybrid_engine import Qwen35Engine

REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PROMPTS = [
    [100, 200, 300, 400],
    [11, 22, 33, 44, 55, 66],
]
BUDGETS = [2, 2]
MAX_BATCH = 2
MAX_LEN = 32
KV_BLOCKS = 16
BLOCK_SIZE = 16
DECODE_STEPS = 2

pytestmark = [
    pytest.mark.hf,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("MINILLM_RUN_HF") != "1",
        reason="set MINILLM_RUN_HF=1 to load the pinned 0.8B checkpoint",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the hybrid smoke is a bf16/CUDA paged run; no CUDA device",
    ),
]


def _snapshot():
    from model.loading import cached_snapshot

    snapshot = cached_snapshot(REPO, REVISION)
    if snapshot is None:
        pytest.skip(f"{REPO}@{REVISION} is not in the local HF cache")
    return snapshot


def test_qwen35_08b_paged_engine_two_requests_two_steps_no_leak():
    from model.loading import load_qwen35_model

    model = load_qwen35_model(_snapshot(), dtype=torch.bfloat16, device="cuda").eval()
    engine = Qwen35Engine(model, max_prefill_tokens=8192)
    pool = None
    try:
        engine.open_hybrid_pool(
            max_batch=MAX_BATCH,
            max_len=MAX_LEN,
            linear_slots=2,
            kv_blocks=KV_BLOCKS,
            block_size=BLOCK_SIZE,
            pool="paged",
        )
        pool = engine.state_pool
        assert engine.paged is True
        assert engine.paged_max_batch == MAX_BATCH
        assert engine.n_layer == 6 and engine.n_kv_head == 2 and engine.head_dim == 256
        assert engine.pool_kv_mb is not None and engine.pool_kv_mb > 0.0

        logits = engine.prefill_group(
            [
                {
                    "id": index,
                    "prompt_ids": list(prompt),
                    "seat": index,
                    "budget": budget,
                }
                for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS))
            ]
        )
        assert tuple(logits.shape) == (MAX_BATCH, engine.vocab_size)
        assert torch.isfinite(logits).all()
        tokens = [int(logits[row].argmax()) for row in range(MAX_BATCH)]
        assert all(0 <= token < engine.vocab_size for token in tokens)

        for step in range(DECODE_STEPS):
            out = engine.decode_batch(tokens)
            assert tuple(out.shape) == (MAX_BATCH, engine.vocab_size)
            assert torch.isfinite(out).all(), f"step {step}: non-finite logits"
            tokens = [int(out[row].argmax()) for row in range(MAX_BATCH)]
            stats = engine.paged_stats()
            assert stats["active_rows"] == MAX_BATCH
            assert stats["decode_impl"] == "triton"
            assert stats["leak"] is False

        assert engine.last_decode_event_ms is not None
        stats = engine.paged_stats()
        assert stats["blocks_in_use"] == 2  # one 16-token block per short prompt
        assert stats["kv_live_mb"] > 0.0
        assert stats["linear_state_mb"] > 0.0

        engine.release_request(0)
        engine.release_request(1)
        engine.release_request(0)  # engine-level release is idempotent
        stats = engine.paged_stats()
        assert stats["active_rows"] == 0
        assert stats["blocks_in_use"] == 0
        assert stats["free_blocks"] == KV_BLOCKS
        assert stats["linear_state_mb"] == 0.0
        assert stats["leak"] is False
    finally:
        engine.close_pool()
        engine.close_pool()
        if pool is not None:
            pool.assert_no_leaks()
        del model
        torch.cuda.empty_cache()
