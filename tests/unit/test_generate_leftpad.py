"""Left-pad batch generation must equal per-sequence generation."""

from __future__ import annotations

import torch

from llm_utils.inference import generate
from model import MiniLLM


def test_leftpad_batch_matches_single(tiny_model: MiniLLM):
    n = 6
    seqs = [torch.randint(0, tiny_model.config.vocab_size, (length,)) for length in (5, 7, 9)]
    width = max(len(s) for s in seqs)
    tokens = torch.full((len(seqs), width), 0, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, seq in enumerate(seqs):
        tokens[i, width - len(seq) :] = seq
        mask[i, width - len(seq) :] = 1

    got = generate(tiny_model, tokens, n, temperature=0.0, attention_mask=mask)
    ref = [
        generate(tiny_model, seq.unsqueeze(0), n, temperature=0.0)[0]
        for seq in seqs
    ]

    for i, (a, b) in enumerate(zip(got, ref)):
        assert torch.equal(a, b), f"seq {i}: batch={a.tolist()} single={b.tolist()}"
