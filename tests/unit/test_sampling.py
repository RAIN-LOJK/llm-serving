"""Sampling contracts: greedy shortcut, top-k, top-p shape/range."""

from __future__ import annotations

import torch

from llm_utils.inference import sample


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert sample(logits, temperature=0.0).item() == 3


def test_top_k_one_is_argmax():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert sample(logits, temperature=1.0, top_k=1).item() == 3


def test_top_p_keeps_top_token():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    # The top token is always inside the nucleus; top_p only truncates the tail.
    assert sample(logits, temperature=1.0, top_p=0.05).item() == 3


def test_sampled_shape():
    logits = torch.randn(3, 16)
    out = sample(logits, temperature=1.0, top_k=4, top_p=0.9)
    assert out.shape == (3, 1)
    assert out.min() >= 0 and out.max() < 16
