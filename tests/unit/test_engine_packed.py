"""Engine-side packed prefill: seat and paged sinks, group rollback, kernel reporting.

These tests use the tiny CPU model except for the Triton case, which runs on CUDA when
available. They compare a packed group against the per-sequence ``KVcache`` reference the
whole W3/W4 paged arm is already checked against, so a wrong slot map or a missing sink
fails here rather than in a benchmark.
"""

from __future__ import annotations

import pytest
import torch

from model import MiniLLM, MiniLLMConfig
from model.triton_decode import TRITON_AVAILABLE
from serve.engine import MiniLLMEngine

N_LAYER = 2
N_KV_HEAD = 2
HEAD_DIM = 8
N_HEAD = 4
BS = 16
CUDA_READY = TRITON_AVAILABLE and torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(
    not CUDA_READY, reason="paged decode is the CUDA Triton block-table kernel"
)
PROMPTS = [
    [3, 5, 7, 11, 13],
    [3, 5, 7, 11, 13, 17, 19, 23, 29],
    [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
]


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=64,
        max_position_embeddings=128,
        n_embd=32,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        n_layer=N_LAYER,
        dropout=0.0,
        bias=False,
    )


def _engine() -> MiniLLMEngine:
    torch.manual_seed(0)
    model = MiniLLM(_config()).eval()
    return MiniLLMEngine(model, max_prefill_tokens=64)


def _cuda_engine() -> MiniLLMEngine:
    """The paged arm's decode kernel is CUDA-only, so its engine lives on CUDA too."""
    torch.manual_seed(0)
    model = MiniLLM(_config()).to("cuda", dtype=torch.bfloat16).eval()
    return MiniLLMEngine(model, max_prefill_tokens=64)


def _group_specs(prompts) -> list[dict]:
    return [
        {"id": index, "prompt_ids": prompt, "seat": index, "budget": 12}
        for index, prompt in enumerate(prompts)
    ]


# ------------------------------------------------------------------ seat arm
def test_seat_packed_group_matches_per_sequence_prefill():
    engine = _engine()
    engine.open_pool(4, 128)
    reference = torch.cat([engine.prefill_one(prompt)[0] for prompt in PROMPTS])

    logits = engine.prefill_group(_group_specs(PROMPTS))
    assert tuple(logits.shape) == (len(PROMPTS), engine.vocab_size)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)
    assert engine.attn_backends()["prefill"] == "sdpa-varlen"


def test_seat_packed_group_decodes_like_per_sequence():
    engine = _engine()
    engine.open_pool(4, 128)
    refs = [engine.prefill_one(prompt) for prompt in PROMPTS]
    logits = engine.prefill_group(_group_specs(PROMPTS))
    tokens = [int(row.argmax()) for row in logits]
    assert tokens == [int(row[0].argmax()) for row in refs]

    for step in range(5):
        got = engine.decode_batch(tokens + [0])
        expected = torch.cat([engine.decode(token, cache) for (_, cache), token in zip(refs, tokens)])
        torch.testing.assert_close(got[: len(tokens)], expected, atol=1e-5, rtol=1e-5)
        tokens = [int(row.argmax()) for row in got[: len(tokens)]]
        refs = [(expected[index : index + 1], refs[index][1]) for index in range(len(tokens))]


# ------------------------------------------------------------------ paged arm
@needs_cuda
def test_paged_packed_group_writes_every_block_and_matches_prefill():
    engine = _cuda_engine()
    engine.open_paged_pool(32, BS, 4, max_position=128)
    reference = torch.cat([engine.prefill_one(prompt)[0] for prompt in PROMPTS])

    logits = engine.prefill_group(_group_specs(PROMPTS))
    torch.testing.assert_close(logits, reference, atol=2e-2, rtol=2e-2)
    assert [int(row.argmax()) for row in logits] == [
        int(row.argmax()) for row in reference
    ]
    # Decode then uses the Triton kernel; there is no gather workspace to allocate.
    engine.decode_batch([int(row.argmax()) for row in logits] + [0])
    stats = engine.paged_stats()
    assert stats["active_rows"] == len(PROMPTS)
    assert stats["decode_impl"] == "triton"
    assert stats["workspace_mb"] == 0.0
    assert "gather_calls" not in stats and "lpad" not in stats


@needs_cuda
def test_paged_group_failure_returns_every_reservation():
    engine = _cuda_engine()
    engine.open_paged_pool(32, BS, 4, max_position=128)
    before = engine.block_manager.stats()["free"]

    class Boom(Exception):
        pass

    original = engine._packed_prefill

    def explode(*args, **kwargs):
        raise Boom("forward died")

    engine._packed_prefill = explode  # type: ignore[method-assign]
    try:
        with pytest.raises(Boom):
            engine.prefill_group(_group_specs(PROMPTS))
    finally:
        engine._packed_prefill = original  # type: ignore[method-assign]

    assert engine.block_manager.stats()["free"] == before
    assert engine.paged_stats()["active_rows"] == 0


@needs_cuda
def test_paged_group_rejects_two_requests_on_one_seat():
    engine = _cuda_engine()
    engine.open_paged_pool(32, BS, 4, max_position=128)
    specs = _group_specs(PROMPTS[:2])
    specs[1]["seat"] = specs[0]["seat"]
    with pytest.raises(ValueError, match="share seat"):
        engine.prefill_group(specs)
    assert engine.block_manager.stats()["free"] == 32


def test_packed_group_rejects_an_over_budget_group():
    engine = _engine()
    engine.max_prefill_tokens = 5
    engine.open_pool(4, 128)
    with pytest.raises(ValueError, match="max_prefill_tokens"):
        engine.prefill_group(_group_specs(PROMPTS[:2]))
    # A single over-budget prompt is legal: it simply forms its own group (P-44).
    logits = engine.prefill_group(_group_specs(PROMPTS[2:3]))
    assert tuple(logits.shape) == (1, engine.vocab_size)


# ------------------------------------------------------------------ CUDA triton
@needs_cuda
def test_paged_engine_rejects_cpu_with_a_clear_error():
    """The gather arm was the CPU/MPS paged path; it must fail loudly, never fall back."""
    engine = _engine()  # CPU
    with pytest.raises(ValueError, match="CUDA") as excinfo:
        engine.open_paged_pool(32, BS, 4, max_position=128)
    message = str(excinfo.value)
    assert "gather control arm was deleted" in message
    assert "tests/reference/paged.py" in message
    assert "impl='continuous'" in message


@needs_cuda
def test_paged_triton_packed_group_decodes_like_per_sequence():
    engine = _cuda_engine()
    engine.open_paged_pool(32, BS, 4, max_position=128)
    refs = [engine.prefill_one(prompt) for prompt in PROMPTS]

    logits = engine.prefill_group(_group_specs(PROMPTS))
    tokens = [int(row.argmax()) for row in logits]
    assert tokens == [int(row[0].argmax()) for row in refs]
    assert engine.attn_backends()["decode"] == "paged-triton"

    for step in range(5):
        got = engine.decode_batch(tokens + [0])
        expected = torch.cat([engine.decode(token, cache) for (_, cache), token in zip(refs, tokens)])
        assert [int(row.argmax()) for row in got[: len(tokens)]] == [
            int(row.argmax()) for row in expected
        ]
        torch.testing.assert_close(
            got[: len(tokens)].float(), expected.float(), atol=2e-2, rtol=2e-2
        )
        tokens = [int(row.argmax()) for row in got[: len(tokens)]]
        refs = [(expected[index : index + 1], refs[index][1]) for index in range(len(tokens))]

    stats = engine.paged_stats()
    assert stats["decode_impl"] == "triton"
    assert stats["workspace_mb"] == 0.0
    assert "gather_calls" not in stats and "gather_read_mb" not in stats
    assert "lpad" not in stats
