"""KV cache equivalence and cache bookkeeping."""

from __future__ import annotations

import torch

from llm_utils.inference import generate
from model import KVcache, MiniLLM


def _ref_greedy(model: MiniLLM, prompt: torch.Tensor, n: int) -> torch.Tensor:
    out = prompt.clone()
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(out)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    return out[0, prompt.shape[1] :]


def test_cache_matches_full_recompute(tiny_model: MiniLLM):
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
    n = 6

    ref = _ref_greedy(tiny_model, prompt, n)
    got = generate(tiny_model, prompt, n, temperature=0.0)[0]

    assert torch.equal(ref, got), f"cache mismatch: ref={ref.tolist()} got={got.tolist()}"


def test_cache_cull_and_past_len():
    cache = KVcache(n_layer=2)
    k = torch.zeros(3, 2, 5, 4)
    v = torch.ones(3, 2, 5, 4)
    cache.set_kv(0, (k, v))
    cache.set_kv(1, (k, v))
    assert cache.past_len(0) == 5
    assert cache.past_len(1) == 5

    cache.cull(torch.tensor([True, False, True]))
    k2, v2 = cache.get_kv(0)
    assert k2.shape == (2, 2, 5, 4)
    assert v2.shape == (2, 2, 5, 4)
