"""Attention contracts that do not need trained weights:
causality, left-pad equivalence, and config wiring.
"""

from __future__ import annotations

import torch

from model import MiniLLM, MiniLLMConfig
from model.layers import TransformerBlock


def test_causal_mask_hides_future(tiny_model: MiniLLM):
    tokens = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    changed = tokens.clone()
    changed[0, -1] = (tokens[0, -1] + 1) % tiny_model.config.vocab_size

    with torch.no_grad():
        logits_a, _ = tiny_model(tokens)
        logits_b, _ = tiny_model(changed)

    # The last position may change; all earlier positions must not see it.
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)


def test_leftpad_mask_matches_unpadded(tiny_model: MiniLLM):
    seq = torch.randint(0, tiny_model.config.vocab_size, (7,))
    pad = 4
    tokens = torch.full((1, 7 + pad), 0, dtype=torch.long)
    tokens[0, pad:] = seq
    mask = torch.zeros_like(tokens)
    mask[0, pad:] = 1

    with torch.no_grad():
        padded_logits, _ = tiny_model(tokens, attention_mask=mask)
        single_logits, _ = tiny_model(seq.unsqueeze(0))

    # RoPE is relative: a common left-pad shift preserves all valid-token dot products.
    assert torch.allclose(padded_logits[0, pad:], single_logits[0], atol=1e-5)


def test_norm_eps_wired():
    config = MiniLLMConfig(
        vocab_size=64,
        max_position_embeddings=32,
        n_embd=16,
        n_head=2,
        n_kv_head=2,
        n_layer=1,
        norm_eps=1e-5,
    )
    block = TransformerBlock(config)
    assert block.input_layernorm.eps == config.norm_eps
    assert block.post_attention_layernorm.eps == config.norm_eps


def test_gqa_head_counts(tiny_model: MiniLLM):
    attn = tiny_model.layers[0].self_attn
    assert attn.n_head == tiny_model.config.n_head
    assert attn.n_kv_head == tiny_model.config.n_kv_head
    tokens = torch.randint(0, tiny_model.config.vocab_size, (2, 9))
    with torch.no_grad():
        logits, _ = tiny_model(tokens)
    assert logits.shape == (2, 9, tiny_model.config.vocab_size)
