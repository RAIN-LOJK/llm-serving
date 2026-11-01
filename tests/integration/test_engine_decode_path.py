"""The shipped engine path (prefill SDPA -> slab -> hand_baddbmm) == the transitional path.

There is no decode-mode switch anymore (D4): ``MiniLLMEngine.generate`` always runs the slab
path. This file pins that it is the same model as the transitional ``llm_utils.inference``
generator (``KVcache`` grown with ``torch.cat``): identical token ids on a tiny model, and
the path works on CPU where CUDA is absent. Both stay in the fast gate.
"""

from __future__ import annotations

import torch

from llm_utils.inference import generate
from model import MiniLLM, MiniLLMConfig, KVcache, KvSlab
from serve.engine import MiniLLMEngine

PROMPT = [3, 5, 7, 11, 13, 17, 19, 23]
BUDGET = 6


def _model() -> MiniLLM:
    torch.manual_seed(0)
    return MiniLLM(
        MiniLLMConfig(
            vocab_size=64, max_position_embeddings=64, n_embd=32, n_head=4, n_kv_head=2,
            n_layer=2, dropout=0.0, bias=False,
        )
    ).eval()


def test_engine_slab_path_matches_transitional_generate_token_for_token():
    model = _model()
    engine = MiniLLMEngine(model)

    got = engine.generate(PROMPT, BUDGET)
    expected = generate(model, torch.tensor([PROMPT]), BUDGET, temperature=0.0)[0]

    assert got.token_ids == expected.tolist(), (
        "the frozen attention path (sdpa prefill + hand_baddbmm slab decode) must be the "
        "same model as the transitional cat path, not merely a similar one"
    )
    assert got.meta["impl"] == "minillm"


def test_engine_path_runs_without_cuda():
    """The shipped path is device-agnostic; the parked graph path is the CUDA-only one."""
    out = MiniLLMEngine(_model()).generate(PROMPT, BUDGET)

    assert len(out.token_ids) == BUDGET
    assert out.cache_mb and out.cache_mb > 0


def test_decode_slab_covers_every_row_in_one_forward():
    """B4: a two-row slab decodes both rows in one ``model(...)`` call."""
    model = _model()
    engine = MiniLLMEngine(model)

    short, long = PROMPT[:3], PROMPT[:6]
    _, short_cache = engine.prefill(short)
    _, long_cache = engine.prefill(long)
    slab = KvSlab(
        model.config.n_layer, 2, model.config.n_kv_head, model.head_dim, 16,
        dtype=model.token_embedding.weight.dtype,
    )
    for layer in range(model.config.n_layer):
        slab.fill_from(short_cache, layer, row=0)
        slab.fill_from(long_cache, layer, row=1)

    batched = engine.decode_slab([1, 2], slab)

    # Row 1 starts out equal to the long-prefix reference; row 0 has a different history.
    assert batched.shape == (2, model.config.vocab_size)
    assert not torch.equal(batched[0], batched[1])

    fresh = KvSlab(
        model.config.n_layer, 1, model.config.n_kv_head, model.head_dim, 16,
        dtype=model.token_embedding.weight.dtype,
    )
    for layer in range(model.config.n_layer):
        fresh.fill_from(long_cache, layer)
    single = engine.decode_slab([2], fresh)
    # Same model, same kernel; batching changes the bmm shape, so a tiny fp drift is fine.
    torch.testing.assert_close(batched[1], single[0], atol=1e-5, rtol=1e-4)
