"""Regression: re-opening the paged pool must free the previous one first.

W8 found that ``MiniLLMEngine.open_paged_pool`` built the new pool while the old one was
still referenced, so every warmup pass peaked at two pools. Pools above ~11 GiB then
OOM'd on the second pass even though a single pool fits (D10, W8 closeout).

The discriminating signal is the **peak** during the second open, not the steady state:
reference counting frees the old pool once the new one is built, but construction itself
transiently holds both.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.gpu]


@pytest.mark.skipif(
    os.environ.get("MINILLM_RUN_HF") != "1",
    reason="set MINILLM_RUN_HF=1 to run HF integration tests",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="paged pool is CUDA-only")
def test_reopen_does_not_keep_two_pools_alive():
    from bench.engines.minillm import ServeEngine

    wrapper = ServeEngine(
        name="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        dtype="bfloat16",
        device="cuda",
    )
    wrapper.load()
    engine = wrapper.engine

    blocks, block_size, batch, pos = 256, 16, 4, 1024
    engine.open_paged_pool(blocks, block_size, batch, max_position=pos)
    one_pool = torch.cuda.memory_allocated()

    cfg = engine.model.config
    pool_bytes = (
        2
        * blocks
        * block_size
        * cfg.n_kv_head
        * engine.model.head_dim
        * 2
        * cfg.n_layer
    )

    torch.cuda.reset_peak_memory_stats()
    second = engine.open_paged_pool(blocks, block_size, batch, max_position=pos)
    peak = torch.cuda.max_memory_allocated()

    assert second is engine._paged_pool
    assert peak - one_pool < pool_bytes // 2, (
        "re-opening the pool peaked at two pools "
        f"({(peak - one_pool) / 2**20:.0f} MiB extra, "
        f"one pool is {pool_bytes / 2**20:.0f} MiB)"
    )
