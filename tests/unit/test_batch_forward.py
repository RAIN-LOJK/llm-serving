"""W3 first layer: seats in one ``KvSlab`` and the batched decode forward (docs/decisions.md D5).

The concurrency unit of W3 is **one slab, one row per seat** (D5, §2.1). Every step
forwards the whole block -- idle seats included -- and each row keeps its own cursors,
its own RoPE position and its own mask row. These tests pin that:

* a seated batch decodes to the same numbers as one ``KVcache`` per sequence;
* a frozen seat stops moving and is fully masked, without disturbing the other rows;
* the slab path stays decode-only (T == 1) while the ``KVcache`` path keeps its
  left-padded batch prefill -- that is W1's acceptance, not something W3 may break;
* RoPE is gathered per row (P-21: a shared table silently produces wrong tokens);
* a single seat is bit-identical to today's single-request path;
* attention is dispatched once per layer per step, not once per sequence (D3/P-23);
* the pool's KV account counts **every** seat's reservation (P-30), not one request's share.

All of it runs on CPU with the tiny model; no golden files, no HF weights.
"""

from __future__ import annotations

import pytest
import torch

from model import KVcache, KvSlab, MiniLLM, MiniLLMConfig
from model.attention import DECODE_BACKEND
from model.attention.hand_baddbmm import BACKEND as DECODE_BACKEND_IMPL
from model.layers import MLP, rope_cos_sin
from serve.engine import MiniLLMEngine

# Three prompts of different lengths (5 / 9 / 13), like the E0 case.
PROMPTS = [
    [3, 5, 7, 11, 13],
    [3, 5, 7, 11, 13, 17, 19, 23, 29],
    [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
]
STEPS = 6


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(vocab_size=64, max_position_embeddings=64, n_embd=32, n_head=4,
                         n_kv_head=2, n_layer=2, dropout=0.0, bias=False)


def _model(dtype: torch.dtype = torch.float32) -> MiniLLM:
    torch.manual_seed(0)
    return MiniLLM(_config()).to(dtype).eval()


def _pool(model: MiniLLM, prompts: list[list[int]], max_len: int = 64):
    """One B-row slab, one prompt per seat: ``(slab, [(cache, next_token), ...])``."""
    batch = len(prompts)
    dtype = model.token_embedding.weight.dtype
    slab = KvSlab(model.config.n_layer, batch, model.config.n_kv_head, model.head_dim, max_len,
                  dtype=dtype, n_head=model.config.n_head)
    engine = MiniLLMEngine(model)
    seats = []
    for row, prompt in enumerate(prompts):
        logits, cache = engine.prefill(prompt)
        # fill_from copies the bytes; activate() only seats the row (no KV move).
        for layer in range(model.config.n_layer):
            slab.fill_from(cache, layer, row=row)
        slab.activate(row, len(prompt))
        seats.append((cache, int(logits.argmax(-1))))
    return slab, seats


def _tokens(seats) -> list[int]:
    return [token for _, token in seats]


# ------------------------------------------------------------------ the six W3 cases

def test_seated_decode_matches_per_sequence():
    """A B-row slab with different prompt lengths == one KVcache per sequence."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab, seats = _pool(model, PROMPTS)

    for step in range(STEPS):
        got = engine.decode_slab(_tokens(seats), slab)
        ref = torch.cat([engine.decode(token, cache) for cache, token in seats], dim=0)
        seats = [(cache, int(row.argmax(-1))) for (cache, _), row in zip(seats, ref)]

        drift = (got - ref).abs().max().item()
        assert drift <= 1e-5, f"step {step}: batched slab vs per-sequence drifted {drift:.2e}"
        assert [int(row.argmax(-1)) for row in got] == [int(row.argmax(-1)) for row in ref], (
            f"step {step}: argmax ids differ between the seated batch and per-sequence decode"
        )

    # seats are independent: each row's cursor moved by exactly STEPS tokens
    assert slab.pos.tolist() == [len(p) + STEPS for p in PROMPTS]
    assert slab.valid.tolist() == [len(p) + STEPS for p in PROMPTS]


def test_frozen_seat_is_masked_out():
    """A frozen seat stops advancing and is closed; the other rows are untouched (P-26)."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab, seats = _pool(model, PROMPTS)

    # Warm the seat once, so freezing it leaves stale KV resident in its buffer.
    engine.decode_slab(_tokens(seats), slab)
    slab.freeze(1)
    assert slab.active.tolist() == [True, False, True]
    assert slab.pos.tolist()[1] == 0 and slab.valid.tolist()[1] == 0

    mask = slab.build_mask(next_token=True)
    assert not bool(mask[1].any()), "a frozen seat must be fully masked"
    assert bool(mask[0, 0, 0, : len(PROMPTS[0]) + 2].all())
    assert bool(mask[2, 0, 0, : len(PROMPTS[2]) + 2].all())

    before = (list(slab.pos.tolist()), list(slab.valid.tolist()))
    tokens = [int(seats[0][1]), 1, int(seats[2][1])]
    warmed = engine.decode_slab(tokens, slab)
    after = (list(slab.pos.tolist()), list(slab.valid.tolist()))
    assert after[0][1] == before[0][1], "advance() moved a frozen seat's pos"
    assert after[1][1] == before[1][1], "advance() moved a frozen seat's valid"
    assert after[0][0] == before[0][0] + 1 and after[0][2] == before[0][2] + 1
    assert not bool(slab.build_mask(next_token=True)[1].any())
    assert torch.isfinite(warmed).all(), "an all-masked seat must be finite, not NaN"

    # Rows 0 and 2 must match a pool where the middle seat was frozen from the start.
    # (Both pools take the same number of writes for the live rows, so the cursors agree.)
    fresh, fresh_seats = _pool(model, PROMPTS)
    fresh.freeze(1)
    engine.decode_slab(_tokens(fresh_seats), fresh)  # same single warm step for rows 0 and 2
    never = engine.decode_slab(_tokens(fresh_seats), fresh)

    assert torch.equal(warmed[0], never[0]), "a retired seat leaked into row 0"
    assert torch.equal(warmed[2], never[2]), "a retired seat leaked into row 2"


def test_slab_rejects_multi_token_forward():
    """D4: the slab path is T == 1 only; the KVcache path keeps its batched prefill (P-22)."""
    model = _model()
    slab, _ = _pool(model, PROMPTS[:2])
    with pytest.raises(ValueError, match="decode-only"):
        model(torch.tensor([[1, 2], [3, 4]]), past_keys_values=slab)

    # ... and the same shape on the KVcache path is W1's left-padded batch prefill.
    assert DECODE_BACKEND is DECODE_BACKEND_IMPL  # layers.py dispatches on T, no swap
    width = len(PROMPTS[2])
    ids = torch.zeros(2, width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(PROMPTS[:2]):
        ids[row, width - len(prompt):] = torch.tensor(prompt)
        mask[row, width - len(prompt):] = 1
    batched, _ = model(ids, attention_mask=mask, past_keys_values=KVcache(model.config.n_layer))
    assert batched.shape == (2, width, model.config.vocab_size)
    assert torch.isfinite(batched).all()

    single, _ = model(torch.tensor([PROMPTS[0]]), past_keys_values=KVcache(model.config.n_layer))
    torch.testing.assert_close(batched[0, -1], single[0, -1], atol=1e-5, rtol=1e-4)


def test_rope_uses_per_row_positions(monkeypatch):
    """P-21: rows at past_len 5 and 9 need their own RoPE positions, not one shared table."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab, seats = _pool(model, PROMPTS[:2])
    assert slab.pos.tolist() == [len(PROMPTS[0]), len(PROMPTS[1])]

    # The cached table is per row, and each row's row equals that row's own position.
    cos, sin = model._rope_rows(slab.pos, slab.max_len)
    assert cos.shape[0] == 2 and not torch.equal(cos[0], cos[1])
    for row, prompt in enumerate(PROMPTS[:2]):
        ref_cos, ref_sin = rope_cos_sin(model.head_dim, model.rope_theta, len(prompt), 1,
                                        torch.device("cpu"), torch.float32)
        assert torch.equal(cos[row, 0], ref_cos[0]) and torch.equal(sin[row, 0], ref_sin[0])

    got = engine.decode_slab(_tokens(seats), slab)
    ref = torch.cat([engine.decode(token, cache) for cache, token in seats], dim=0)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)

    # Deliberately feed the batch one table keyed on a single position (the old bug).
    def shared_table(positions, length):
        one_cos, one_sin = rope_cos_sin(model.head_dim, model.rope_theta,
                                        int(positions.max()), 1, positions.device,
                                        model.token_embedding.weight.dtype)
        batch = positions.numel()
        return one_cos.expand(batch, -1).unsqueeze(1), one_sin.expand(batch, -1).unsqueeze(1)

    slab2, seats2 = _pool(model, PROMPTS[:2])
    monkeypatch.setattr(model, "_rope_rows", shared_table)
    wrong = engine.decode_slab(_tokens(seats2), slab2)
    assert not torch.allclose(wrong[0], got[0], atol=1e-5), (
        "a shared RoPE table must change row 0's logits -- otherwise this test proves nothing"
    )


def test_single_seat_is_bit_identical():
    """One seat of a slab == today's single-request KVcache path, bit for bit (E0a unit form).

    bf16 is the measured dtype. In fp32 the two container paths already differ in the last
    ulp of the bias broadcast (*before* W3: ``test_slab_decode.py`` pins fp32 at 1e-5 and
    bf16 at exact equality), so the bit-exact claim is made where it holds and fp32 is
    checked at the same 1e-5 the existing test uses.
    """
    model = _model(torch.bfloat16)
    engine = MiniLLMEngine(model)
    logits, cache = engine.prefill(PROMPTS[0])

    slab = KvSlab(model.config.n_layer, 1, model.config.n_kv_head, model.head_dim, 32,
                  dtype=model.token_embedding.weight.dtype, n_head=model.config.n_head)
    for layer in range(model.config.n_layer):
        slab.fill_from(cache, layer)
    slab.activate(0, len(PROMPTS[0]))
    assert slab.active.tolist() == [True]

    token = int(logits.argmax(-1))
    for step in range(STEPS):
        reference = engine.decode(token, cache)
        got = engine.decode_slab([token], slab)
        assert torch.equal(got, reference), (
            f"step {step}: single seated row differs from the KVcache path by "
            f"{(got.float() - reference.float()).abs().max().item():.2e}"
        )
        token = int(reference.argmax(-1))

    fp32 = _model(torch.float32)
    fp32_engine = MiniLLMEngine(fp32)
    logits32, cache32 = fp32_engine.prefill(PROMPTS[0])
    slab32 = KvSlab(fp32.config.n_layer, 1, fp32.config.n_kv_head, fp32.head_dim, 32,
                    dtype=torch.float32, n_head=fp32.config.n_head)
    for layer in range(fp32.config.n_layer):
        slab32.fill_from(cache32, layer)
    slab32.activate(0, len(PROMPTS[0]))
    token32 = int(logits32.argmax(-1))
    for step in range(STEPS):
        reference = fp32_engine.decode(token32, cache32)
        got = fp32_engine.decode_slab([token32], slab32)
        torch.testing.assert_close(got, reference, atol=1e-5, rtol=1e-4)
        token32 = int(reference.argmax(-1))


def test_attention_dispatched_once_per_batch(monkeypatch):
    """D3/P-23: one ``hand_baddbmm.decode`` per layer per step, not one per sequence."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab, seats = _pool(model, PROMPTS)
    batch, n_layer = slab.batch, model.config.n_layer
    assert batch > 1

    decode_calls: list[int] = []
    original_decode = DECODE_BACKEND.decode

    def counting_decode(*args, **kwargs):
        decode_calls.append(1)
        return original_decode(*args, **kwargs)

    mlp_calls: list[int] = []
    original_mlp = MLP.forward

    def counting_mlp(self, *args, **kwargs):
        mlp_calls.append(1)
        return original_mlp(self, *args, **kwargs)

    monkeypatch.setattr(DECODE_BACKEND, "decode", counting_decode)
    monkeypatch.setattr(MLP, "forward", counting_mlp)

    engine.decode_slab(_tokens(seats), slab)
    assert len(decode_calls) == n_layer, (
        f"attention dispatched {len(decode_calls)} times for B={batch}, expected {n_layer} "
        f"(one per layer, not {n_layer} x {batch} = {n_layer * batch})"
    )
    assert len(mlp_calls) == n_layer

    decode_calls.clear()
    mlp_calls.clear()
    engine.decode_slab(_tokens(seats), slab)
    assert len(decode_calls) == n_layer and len(mlp_calls) == n_layer


# ------------------------------------------------------------ the seat contract itself

def test_slab_forward_needs_every_row():
    """The block is forwarded whole, idle seats included: a subset is a caller bug."""
    model = _model()
    slab, seats = _pool(model, PROMPTS)
    with pytest.raises(ValueError, match="slab.batch"):
        model(torch.tensor([[int(seats[0][1])], [int(seats[1][1])]]), past_keys_values=slab)


def test_seats_are_active_after_construction_and_reset():
    slab = KvSlab(1, 3, 1, 4, 8)
    assert slab.active.tolist() == [True, True, True]
    assert slab.active.dtype is torch.bool and slab.active.device == slab.pos.device

    slab.freeze(1)
    slab.reset()
    assert slab.active.tolist() == [True, True, True]
    assert slab.pos.tolist() == [0, 0, 0] and slab.valid.tolist() == [0, 0, 0]


def test_all_rows_advance_keeps_the_w2_behaviour():
    """``advance(all_rows=True)`` still moves frozen rows -- the graph/park path's contract."""
    slab = KvSlab(1, 2, 1, 4, 8)
    slab.activate(0, 3)
    slab.activate(1, 3)
    slab.freeze(1)

    slab.advance()
    assert slab.pos.tolist() == [4, 0] and slab.valid.tolist() == [4, 0]
    assert slab._host_pos == [4, 0] and slab._host_valid == [4, 0]

    slab.advance(all_rows=True)
    assert slab.pos.tolist() == [5, 1] and slab.valid.tolist() == [5, 1]
    assert slab._host_pos == [5, 1] and slab._host_valid == [5, 1]


# --------------------------------------------------------------- pool accounting (P-30)

def test_pool_reservation_covers_every_seat():
    """P-30: ``pool_kv_mb`` is the whole pool's reservation, not one request's share.

    The pool is allocated once for ``max_batch`` seats, so the numerator of
    ``memory_match_ratio`` must count every seat -- otherwise the other seats' KV stays in
    the allocator peak unaccounted (the W2-0004 mismatch, in pool form). Seats coming and
    going only move cursors: the reservation never changes.
    """
    model = _model()
    engine = MiniLLMEngine(model)
    cfg = model.config
    batch = 4
    need = 50  # inside the 64-column bucket (== max_position here), so max_len != need

    assert engine.pool_kv_mb is None, "no pool is accounted before open_pool()"
    slab = engine.open_pool(batch, need)
    assert slab.batch == batch
    assert slab.max_len == 64 and slab.max_len > need, "the bucket rounds max_len, not need"

    itemsize = slab.dtype.itemsize
    per_seat_mb = (2 * cfg.n_layer * cfg.n_kv_head * slab.max_len * model.head_dim
                   * itemsize / 2**20)
    total_mb = (2 * cfg.n_layer * batch * cfg.n_kv_head * slab.max_len * model.head_dim
                * itemsize / 2**20)
    assert total_mb == pytest.approx(batch * per_seat_mb)
    assert engine.pool_kv_mb == slab.mb() == pytest.approx(total_mb, abs=1e-9)

    # A seat in service and one decode step move cursors only: one allocation, no growth.
    # Admission is the write-through prefill (no KVcache, no fill_from copy on this path).
    reserved = engine.pool_kv_mb
    with torch.inference_mode():
        engine.prefill_into_seat(1, PROMPTS[0])
        engine.decode_batch([0] * batch)
    assert engine.pool_kv_mb == reserved, "the pool grew while a seat was decoding"

    # The same bucket is reused (same object, same reservation), not rebuilt per call.
    again = engine.open_pool(batch, 33)
    assert again is slab and engine.pool_kv_mb == reserved
