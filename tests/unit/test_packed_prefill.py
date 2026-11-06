"""Packed whole-prompt prefill: one varlen forward, no cross-request attention.

The tests use the tiny CPU model and ``SdpaVarlen``: on CPU the varlen backend is a
per-sequence SDPA loop, so the packed result must match one plain forward per prompt
to float32 noise. They also pin the two properties a padded batch would hide:

* every request's positions start at 0 (request-major packing, not ``arange(total)``);
* changing request 0 cannot move request 1's logits (``cu_seqlens`` is a real boundary,
  not a mask over a shared axis).

No weights, no golden files, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from model import MiniLLM, MiniLLMConfig
from model.packed import PackedPrefill, as_packed_prefill

PROMPTS = [
    [3, 5, 7, 11, 13],
    [3, 5, 7, 11, 13, 17, 19, 23, 29],
    [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
]


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=64,
        max_position_embeddings=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        dropout=0.0,
        bias=False,
    )


def _model() -> MiniLLM:
    torch.manual_seed(0)
    return MiniLLM(_config()).eval()


class RecordingSink:
    """One-shot sink: records each layer's ``(1, H, L, D)`` K/V, writes nothing."""

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        self.calls = 0

    def set_kv(self, layer: int, kv) -> None:
        k, v = kv
        expected = (1, 2, self.length, 8)  # tiny model: n_kv_head=2, head_dim=8
        assert tuple(k.shape) == expected, f"layer {layer}: {tuple(k.shape)} != {expected}"
        assert tuple(v.shape) == tuple(k.shape)
        self.k.append(k)
        self.v.append(v)
        self.calls += 1


def _packed(model: MiniLLM, prompts: list[list[int]]):
    cu = [0]
    positions: list[int] = []
    sinks = []
    tokens: list[int] = []
    for prompt in prompts:
        cu.append(cu[-1] + len(prompt))
        positions.extend(range(len(prompt)))
        sinks.append(RecordingSink(len(prompt)))
        tokens.extend(prompt)
    container = PackedPrefill(
        cu_seqlens=cu,
        positions=torch.tensor(positions, dtype=torch.long),
        sinks=sinks,
    )
    ids = torch.tensor(tokens, dtype=torch.long).view(1, -1)
    logits, _ = model(ids, past_keys_values=container, logits_to_keep=1)
    return logits, container, sinks


def test_packed_matches_per_sequence_forward():
    model = _model()
    packed, container, sinks = _packed(model, PROMPTS)
    assert tuple(packed.shape) == (len(PROMPTS), model.config.vocab_size)

    reference = []
    with torch.no_grad():
        for prompt in PROMPTS:
            logits, _ = model(torch.tensor([prompt]), logits_to_keep=1)
            reference.append(logits[0, -1])
    reference = torch.stack(reference)
    torch.testing.assert_close(packed, reference, atol=1e-5, rtol=1e-5)

    assert container.total_tokens == sum(len(p) for p in PROMPTS)
    assert container.max_seqlen == max(len(p) for p in PROMPTS)
    assert container.last_token_index.tolist() == [
        sum(len(p) for p in PROMPTS[: i + 1]) - 1 for i in range(len(PROMPTS))
    ]
    assert all(sink.calls == model.config.n_layer for sink in sinks)


def test_request_positions_restart_at_zero():
    """Two requests of different lengths: the second must not inherit the first's RoPE."""
    model = _model()
    packed, container, _ = _packed(model, [[3, 5, 7], [11, 13, 17, 19, 23]])
    assert container.positions.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    with torch.no_grad():
        second, _ = model(torch.tensor([[11, 13, 17, 19, 23]]), logits_to_keep=1)
    torch.testing.assert_close(packed[1], second[0, -1], atol=1e-5, rtol=1e-5)


def test_no_cross_request_attention_leak():
    model = _model()
    baseline, _, _ = _packed(model, PROMPTS)

    changed = [list(prompt) for prompt in PROMPTS]
    changed[0][-1] = (changed[0][-1] + 7) % model.config.vocab_size
    after, _, _ = _packed(model, changed)

    assert not torch.allclose(baseline[0], after[0]), "request 0 should have changed"
    torch.testing.assert_close(baseline[1:], after[1:], atol=0.0, rtol=0.0)


def test_model_without_logits_to_keep_returns_all_tokens():
    model = _model()
    cu = [0, 3, 5]
    container = PackedPrefill(
        cu_seqlens=cu,
        positions=torch.tensor([0, 1, 2, 0, 1]),
        sinks=[RecordingSink(3), RecordingSink(2)],
    )
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    with torch.no_grad():
        logits, _ = model(ids, past_keys_values=container)
    assert tuple(logits.shape) == (1, 5, model.config.vocab_size)


def test_packed_prefill_rejects_bad_layouts():
    with pytest.raises(ValueError, match="start at 0"):
        PackedPrefill(cu_seqlens=[1, 3], positions=torch.tensor([0, 1]), sinks=[RecordingSink(2)])
    with pytest.raises(ValueError, match="strictly increasing"):
        PackedPrefill(cu_seqlens=[0, 2, 2], positions=torch.tensor([0, 1]), sinks=[RecordingSink(2)])
    with pytest.raises(ValueError, match="one sink per request"):
        PackedPrefill(cu_seqlens=[0, 2, 4], positions=torch.tensor([0, 1, 0, 1]), sinks=[RecordingSink(2)])
    with pytest.raises(ValueError, match="request-local"):
        PackedPrefill(cu_seqlens=[0, 2, 4], positions=torch.tensor([0, 1, 2, 3]), sinks=[RecordingSink(2), RecordingSink(2)])


def test_as_packed_prefill_refuses_incomplete_container():
    class Incomplete:
        packed_prefill = True

    with pytest.raises(TypeError, match="missing packed-prefill hook"):
        as_packed_prefill(Incomplete())
    assert as_packed_prefill(None) is None
    assert as_packed_prefill(object()) is None
