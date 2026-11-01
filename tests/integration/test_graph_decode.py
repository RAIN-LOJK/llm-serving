"""A captured decode step must be bit-identical to the eager one.

This is the parked path's own test (``serve/graph.py`` is not wired into any measurement;
W4 will connect it). It imports nothing that the frozen run path deleted: the eager
reference is the shipped ``MiniLLMEngine.prefill`` + ``decode_slab``.

Marked ``gpu``: a CUDA graph needs a real device. It checks the two things that can silently
break a capture:

* numbers: replay vs eager, token by token (same kernels, same data -> equal);
* addresses: the KV/mask buffers must not move, otherwise the capture would replay a stale
  graph and produce plausible-looking garbage.
"""

from __future__ import annotations

import pytest
import torch

from model import KVcache, KvSlab, MiniLLM, MiniLLMConfig
from serve.engine import MiniLLMEngine
from serve.graph import GraphDecoder

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PROMPT = [3, 5, 7, 11, 13, 17, 19]
STEPS = 4


def _model(device="cuda") -> MiniLLM:
    torch.manual_seed(0)
    config = MiniLLMConfig(
        vocab_size=64, max_position_embeddings=64, n_embd=32, n_head=4, n_kv_head=2,
        n_layer=2, dropout=0.0, bias=False,
    )
    return MiniLLM(config).to(device).eval()


def _prefill(device="cuda"):
    model = _model(device)
    logits, cache = MiniLLMEngine(model).prefill(PROMPT)
    return model, logits, cache


def _slab(model: MiniLLM, cache: KVcache, max_len: int) -> KvSlab:
    slab = KvSlab(
        model.config.n_layer, 1, model.config.n_kv_head, model.head_dim, max_len,
        dtype=model.token_embedding.weight.dtype, device="cuda",
    )
    for layer in range(model.config.n_layer):
        slab.fill_from(cache, layer)
    return slab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_graph_replay_matches_eager_token_for_token():
    model, logits, cache = _prefill()
    max_len = len(PROMPT) + 8

    eager_slab = _slab(model, cache, max_len)
    graph_slab = _slab(model, cache, max_len)

    token = int(logits.argmax(-1))
    expected = []
    for _ in range(STEPS):
        with torch.no_grad():
            got = MiniLLMEngine(model).decode_slab([token], eager_slab)
        expected.append(got[0].clone())
        token = int(got[0].argmax(-1))

    decoder = GraphDecoder(model, graph_slab, max_steps=STEPS).capture()
    assert int(graph_slab.valid.max().item()) == len(PROMPT), "capture did not restore the slab"

    token = int(logits.argmax(-1))
    for step, reference in enumerate(expected):
        replayed = decoder.step([token])
        # ``decode_slab`` returns (B, vocab) and the reference above stored one row, so
        # compare the matching rows rather than letting the shapes broadcast a pass.
        assert torch.equal(replayed[0], reference), (
            f"step {step}: graph replay differs from eager by "
            f"{(replayed[0] - reference).abs().max().item():.2e}"
        )
        token = int(replayed.argmax(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_capture_does_not_move_the_buffers():
    model, _, cache = _prefill()
    slab = _slab(model, cache, len(PROMPT) + 8)
    before = [(k.data_ptr(), v.data_ptr()) for k, v in zip(slab.keys, slab.values)]
    mask_ptr = slab.mask.data_ptr()

    GraphDecoder(model, slab, max_steps=4).capture()
    decoder = GraphDecoder(model, slab, max_steps=4).capture()
    decoder.step([1])

    after = [(k.data_ptr(), v.data_ptr()) for k, v in zip(slab.keys, slab.values)]
    assert after == before, "the capture reallocated KV buffers; the graph would read stale memory"
    assert slab.mask.data_ptr() == mask_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_budget_larger_than_the_slab_is_refused():
    model, _, cache = _prefill()
    slab = _slab(model, cache, len(PROMPT) + 2)
    with pytest.raises(ValueError, match="budget does not fit"):
        GraphDecoder(model, slab, max_steps=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_step_requires_one_token_per_row():
    model, _, cache = _prefill()
    decoder = GraphDecoder(model, _slab(model, cache, len(PROMPT) + 8), max_steps=4).capture()
    with pytest.raises(ValueError, match="one per row"):
        decoder.step([1, 2])
