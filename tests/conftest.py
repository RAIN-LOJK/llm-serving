"""Shared fixtures.

Fast tests must not depend on ``weights/`` or a HuggingFace download. They build a tiny
deterministic model on CPU; integration tests that do need weights are marked ``hf`` and
excluded from the fast gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MiniLLM, MiniLLMConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True, scope="session")
def _torch_kernel_fallbacks():
    """Pin transformers' Qwen3.5 kernels to their torch fallbacks.

    ``causal-conv1d`` and ``fla`` are installed on the 4090 for speed. transformers resolves
    them at import time and calls them unconditionally, but the CUDA wrappers assert
    ``x.is_cuda()`` even when the tiny fixture model runs on CPU -- so the fast gate would
    fail on an otherwise healthy environment. The unit tests are the reference, not the
    kernel speed path; keep them deterministic with and without those packages.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    except Exception:  # transformers missing or model renamed: tests skip themselves
        yield
        return

    restored = []
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule"):
        resolved = getattr(module, name, None)
        fallback = getattr(resolved, "__wrapped__", None)
        if fallback is not None and fallback is not resolved:
            restored.append((name, resolved))
            setattr(module, name, fallback)
    try:
        yield
    finally:
        for name, resolved in restored:
            setattr(module, name, resolved)


@pytest.fixture
def tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=128,
        max_position_embeddings=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        n_inner=None,
        dropout=0.0,
        bias=False,
        norm_eps=1e-5,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_model(tiny_config: MiniLLMConfig) -> MiniLLM:
    model = MiniLLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def device() -> str:
    return "cpu"
