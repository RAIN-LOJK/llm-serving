"""RoPE table guard and shared generation budget for left-padded batches."""

from __future__ import annotations

import torch

from llm_utils.inference import generate
from model import MiniLLM, MiniLLMConfig


def _tiny(max_position: int) -> MiniLLM:
    model = MiniLLM(
        MiniLLMConfig(
            vocab_size=128,
            max_position_embeddings=max_position,
            n_embd=32,
            n_head=4,
            n_kv_head=2,
            n_layer=2,
            dropout=0.0,
            norm_eps=1e-5,
        )
    )
    model.eval()
    return model


def test_generate_stops_before_rope_overflow():
    model = _tiny(8)
    prompt = torch.randint(0, model.config.vocab_size, (1, 6))
    out = generate(model, prompt, 10, temperature=0.0)[0]
    # prompt_len 6 + generated 2 == max_position 8; the next decode would overflow.
    assert len(out) == 2


def test_leftpad_uses_shared_budget():
    model = _tiny(8)
    seqs = [
        torch.randint(0, model.config.vocab_size, (2,)),
        torch.randint(0, model.config.vocab_size, (6,)),
    ]
    width = max(len(s) for s in seqs)
    tokens = torch.zeros((2, width), dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for i, seq in enumerate(seqs):
        tokens[i, width - len(seq) :] = seq
        mask[i, width - len(seq) :] = 1

    outs = generate(model, tokens, 10, temperature=0.0, attention_mask=mask)
    # Budget is max_position - T_max = 2 for every row, not per-row length.
    assert [len(o) for o in outs] == [2, 2]
