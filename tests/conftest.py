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
