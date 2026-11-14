"""W7 Step 2 engine: hybrid prefill/decode, atomic admission, and pool lifetime.

The model is the existing 24-layer ``LLLF x 6`` tiny config (hidden 32, 4/2 heads, conv
kernel 4) on CPU fp32. Nothing here touches the model: the engine under test writes the 18
linear-attention layers into a :class:`LinearStatePool` and the 6 full-attention layers into
a seat ``KvSlab`` (always) or a paged ``PagedPool`` (CUDA/Triton only).

What is pinned, and why:

* **Prefill is the dense prefill.** The per-request engine prefill writes KV through the
  compact full-attention sink while the GDN layers write the pool slot; its logits must
  match a plain ``Qwen35State`` prefill exactly enough for greedy to agree (fp32; measured
  0.0 here, compared at ``atol=1e-5``).
* **Batched decode row == separate decode.** Three steps over ``B=2`` seats are compared
  per row against two independent dense decodes. The prompts have different lengths, so a
  shared scalar position would be wrong; the seat adapter must use per-row ``pos``.
* **Seats are not compact rows.** A second test places two requests on seats 0 and 2 of a
  three-seat slab, freezes the virgin seat, decodes a non-prefix batch, then releases the
  lower request (the state pool swaps the last slot into the hole) and decodes again.
  Seat 1 must stay untouched and the surviving row must still match its dense reference.
* **Atomicity.** ``can_admit_group`` refuses a group that only fits one linear slot and
  changes nothing; ``prefill_group`` refuses the same group without allocating. A forward
  failure after the first member succeeded rolls the whole group back: no slot, block or
  active handle survives, and the seat rows are frozen for reuse.
* **Lifetime.** Release is idempotent at the engine boundary, stats report the pool's real
  bytes, ``leak`` is false after a clean release, and repeated ``close_pool`` is safe.

The paged arm (last test) runs only with CUDA + Triton and compares the same tiny model's
paged logits/greedy against the seat arm for three steps. On this host Triton 3.4 accepts
the tiny geometry (``block_size=4``, ``head_dim=8``); CPU/no-Triton hosts skip cleanly.
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.qwen35_model import Qwen35Model, Qwen35State
from model.triton_decode import TRITON_AVAILABLE
from serve.engine import SchedulerEngine
from serve.hybrid_engine import Qwen35Engine, _LayerMapSink

TINY = dict(
    vocab_size=64,
    max_position_embeddings=64,
    n_embd=32,
    n_head=4,
    n_kv_head=2,
    n_layer=24,
    n_inner=64,
    dropout=0.0,
    bias=False,
    norm_eps=1e-6,
    head_dim=8,
    rope_theta=1e4,
    qk_norm=True,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    tie_word_embeddings=True,
)

VOCAB = 64
PROMPTS = [
    [3, 5, 7, 11, 13],
    [2, 4, 6, 8, 10, 12, 14, 16, 18],
]
DECODE_STEPS = 3
ATOL = 1e-5
#: bf16 decode through two different KV kernels (Triton block table vs hand_baddbmm).
#: Same weights/dtype, different accumulation order: measured ~2e-3, compared at 2e-2.
ATOL_BF16 = 2e-2


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model(
    device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Qwen35Model:
    torch.manual_seed(0)
    return Qwen35Model(_config()).to(device=device, dtype=dtype).eval()


def _ids(prompt: list[int], device: str = "cpu") -> torch.Tensor:
    return torch.tensor([prompt], dtype=torch.long, device=device)


def _dense_prefill(model: Qwen35Model, prompt: list[int]) -> tuple[Qwen35State, torch.Tensor]:
    state = Qwen35State(model.config, device=model.token_embedding.weight.device)
    with torch.no_grad():
        logits, _ = model(_ids(prompt, str(state.device)), state, logits_to_keep=1)
    return state, logits[0, -1]


def _dense_step(model: Qwen35Model, state: Qwen35State, token: int) -> torch.Tensor:
    device = str(state.device)
    with torch.no_grad():
        logits, _ = model(
            torch.tensor([[int(token)]], dtype=torch.long, device=device),
            state,
            logits_to_keep=1,
        )
    return logits[0, -1]


def _assert_row(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    expected = expected.reshape(-1)
    actual = actual.reshape(-1)
    torch.testing.assert_close(
        actual, expected, atol=ATOL, rtol=0, msg=f"{what}: seat engine drifted"
    )
    assert int(actual.argmax()) == int(expected.argmax()), f"{what}: greedy id differs"


def _specs(prompts, seats, budgets) -> list[dict]:
    return [
        {"id": index, "prompt_ids": list(prompt), "seat": seat, "budget": budget}
        for index, (prompt, seat, budget) in enumerate(zip(prompts, seats, budgets))
    ]


# ---------------------------------------------------------------- sink adapter


def test_layer_map_sink_maps_actual_to_compact_and_rejects_non_full_layers():
    calls = []

    class Inner:
        def set_kv(self, layer, kv):
            calls.append((layer, kv))

    sink = _LayerMapSink(Inner(), [3, 7, 11, 15, 19, 23])
    assert sink.past_len(7) == 0
    assert sink.get_kv(7) == (None, None)
    sink.set_kv(3, ("k0", "v0"))
    sink.set_kv(23, ("k5", "v5"))
    assert calls == [(0, ("k0", "v0")), (5, ("k5", "v5"))]
    with pytest.raises(IndexError, match="not a full-attention layer"):
        sink.set_kv(4, ("x", "y"))


# --------------------------------------------------------------- seat arm


def test_seat_engine_prefill_decode_matches_dense_and_closes_cleanly():
    model = _model()
    engine = Qwen35Engine(model, eos_token_id=None, max_prefill_tokens=64)
    assert isinstance(engine, SchedulerEngine)
    engine.open_hybrid_pool(max_batch=2, max_len=32, linear_slots=2, pool="seat")

    assert engine.paged is False
    assert engine.paged_max_batch == 2 and engine.max_batch == 2 and engine.max_len == 32
    assert engine.vocab_size == VOCAB and engine.max_position == 64
    assert engine.n_layer == 6 and engine.n_kv_head == 2 and engine.head_dim == 8
    assert engine.n_head == 4
    assert engine.linear_slots == 2 and engine.linear_state_mb == 0.0
    assert engine.slab.batch == 2
    assert engine.pool_kv_mb == pytest.approx(engine.slab.mb())
    assert engine.kv_pool_mb == engine.pool_kv_mb
    assert engine.paged_stats()["active_rows"] == 0

    specs = _specs(PROMPTS, [0, 1], [4, 3])
    assert engine.can_admit_group(specs) is True
    logits = engine.prefill_group(specs)
    assert tuple(logits.shape) == (2, VOCAB)

    references = []
    for row, prompt in enumerate(PROMPTS):
        state, dense = _dense_prefill(model, prompt)
        references.append((state, dense))
        _assert_row(logits[row], dense, f"prefill row {row}")

    stats = engine.paged_stats()
    assert stats["active_rows"] == 2
    assert stats["linear_slots"] == 2
    assert stats["linear_state_mb"] > 0.0
    assert stats["blocks_in_use"] == 0 and stats["free_blocks"] == 0
    assert stats["written_blocks"] == 0
    assert stats["frag_internal"] is None and stats["reservation_slack"] is None
    assert stats["decode_impl"] == "hand_baddbmm"
    assert stats["workspace_mb"] == 0.0
    assert stats["leak"] is False
    assert engine.slab.pos.tolist() == [5, 9]
    assert engine.slab.valid.tolist() == [5, 9]

    tokens = [int(logits[0].argmax()), int(logits[1].argmax())]
    for step in range(DECODE_STEPS):
        out = engine.decode_batch(tokens)
        assert tuple(out.shape) == (2, VOCAB)
        assert engine.last_decode_event_ms is None  # CPU: scheduler falls back to wall
        for row, (state, _) in enumerate(references):
            dense = _dense_step(model, state, tokens[row])
            _assert_row(out[row], dense, f"decode step {step} row {row}")
        assert engine.slab.pos.tolist() == [5 + step + 1, 9 + step + 1]
        assert engine.slab.valid.tolist() == [5 + step + 1, 9 + step + 1]
        tokens = [int(out[0].argmax()), int(out[1].argmax())]

    per_token = 2 * 6 * 2 * 8 * 4  # 6 compact layers, K+V, 2 kv heads, D=8, fp32
    expected_live = (5 + 9 + 2 * DECODE_STEPS) * per_token / 2**20
    assert engine.paged_stats()["kv_live_mb"] == pytest.approx(expected_live)
    assert engine.state_pool.handle(0).length == 5 + DECODE_STEPS
    assert engine.state_pool.handle(1).length == 9 + DECODE_STEPS

    # Releasing the upper request leaves the lower one live; the decode batch shrinks and
    # the idle seat's row must be zeros (the scheduler discards it).
    engine.release_request(1)
    engine.release_request(1)  # idempotent at the engine boundary
    assert engine.paged_stats()["active_rows"] == 1
    idle = engine.decode_batch([tokens[0], 0])
    dense = _dense_step(model, references[0][0], tokens[0])
    _assert_row(idle[0], dense, "partial batch row 0")
    assert float(idle[1].abs().sum()) == 0.0
    assert engine.slab.pos.tolist() == [5 + DECODE_STEPS + 1, 0]
    assert engine.slab.active.tolist() == [True, False]

    engine.release_request(0)
    stats = engine.paged_stats()
    assert stats["active_rows"] == 0
    assert stats["linear_state_mb"] == 0.0
    assert stats["kv_live_mb"] == 0.0
    assert stats["leak"] is False

    pool = engine.state_pool
    engine.close_pool()
    engine.close_pool()
    pool.assert_no_leaks()
    with pytest.raises(RuntimeError, match="open_hybrid_pool"):
        engine.paged_stats()


def test_seat_engine_non_prefix_batch_and_slot_swap_match_dense():
    """Seats 0 and 2, seat 1 frozen: only the batch rows move, then the slot swaps."""
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=3, max_len=32, linear_slots=3, pool="seat")
    engine.freeze_seat(1)  # the scheduler parks virgin seats before the first decode
    assert engine.slab.active.tolist() == [True, False, True]

    prompts = [[3, 5, 7], [2, 4, 6, 8, 10, 12]]
    logits = engine.prefill_group(_specs(prompts, [0, 2], [4, 3]))
    references = []
    for row, prompt in enumerate(prompts):
        state, dense = _dense_prefill(model, prompt)
        references.append((state, dense))
        _assert_row(logits[row], dense, f"non-prefix prefill row {row}")
    assert engine.slab.pos.tolist() == [3, 0, 6]
    assert engine.slab.active.tolist() == [True, False, True]

    tokens = [int(logits[0].argmax()), 0, int(logits[1].argmax())]
    out = engine.decode_batch(tokens)
    for row, seat in enumerate((0, 2)):
        dense = _dense_step(model, references[row][0], tokens[seat])
        _assert_row(out[seat], dense, f"non-prefix decode seat {seat}")
    # Seat 1 was never written or advanced.
    assert engine.slab.pos.tolist() == [4, 0, 7]
    assert engine.slab.valid.tolist() == [4, 0, 7]
    assert engine.slab.active.tolist() == [True, False, True]

    # Release the lower request: the pool swaps the last active slot into slot 0 and the
    # surviving handle now decodes from slot 0 while still landing on seat 2.
    engine.release_request(0)
    assert engine.state_pool.stats().active == 1
    tokens = [0, 0, int(out[2].argmax())]
    out = engine.decode_batch(tokens)
    dense = _dense_step(model, references[1][0], tokens[2])
    _assert_row(out[2], dense, "post-swap decode")
    assert float(out[0].abs().sum()) == 0.0

    engine.release_request(1)
    assert engine.paged_stats()["leak"] is False
    engine.close_pool()


# ------------------------------------------------------------- atomicity


def test_linear_slots_one_refuses_a_two_request_group_without_state_change():
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=2, max_len=32, linear_slots=1, pool="seat")
    specs = _specs(PROMPTS, [0, 1], [2, 2])

    assert engine.can_admit_group(specs) is False
    stats = engine.state_pool.stats()
    assert stats.active == 0 and stats.peak_active == 0
    assert engine.paged_stats()["active_rows"] == 0
    assert engine.paged_stats()["linear_state_mb"] == 0.0
    assert engine.paged_stats()["blocks_in_use"] == 0

    assert engine.can_admit_group(specs[:1]) is True
    with pytest.raises(RuntimeError, match="linear_slots=1"):
        engine.prefill_group(specs)
    assert engine.state_pool.stats().active == 0
    assert engine.paged_stats()["active_rows"] == 0

    engine.close_pool()


def test_failed_prefill_rolls_the_whole_group_back(monkeypatch):
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=2, max_len=32, linear_slots=2, pool="seat")
    specs = _specs(PROMPTS, [0, 1], [2, 2])

    real_forward = model.forward_packed

    def flaky(*args, **kwargs):
        # W7 Step 3: the group is one packed forward, so a failure anywhere inside it
        # happens after every slot/block of the group was reserved.
        raise RuntimeError("prefill blew up")

    monkeypatch.setattr(model, "forward_packed", flaky)
    with pytest.raises(RuntimeError, match="prefill blew up"):
        engine.prefill_group(specs)
    monkeypatch.setattr(model, "forward_packed", real_forward)

    # Both members' pool slots were allocated and no seat was activated; all rolled back.
    stats = engine.state_pool.stats()
    assert stats.active == 0
    assert stats.peak_active == 2
    assert engine.paged_stats()["active_rows"] == 0
    assert engine.paged_stats()["linear_state_mb"] == 0.0
    assert engine.paged_stats()["leak"] is False
    assert engine.slab.active.tolist() == [False, False]
    assert engine.slab.pos.tolist() == [0, 0]
    assert engine.slab.valid.tolist() == [0, 0]

    # The same ids and seats are reusable, and the rows are not stale.
    logits = engine.prefill_group(specs[:1])
    state, dense = _dense_prefill(model, PROMPTS[0])
    _assert_row(logits[0], dense, "retry after rollback")
    out = engine.decode_batch([int(logits[0].argmax()), 0])
    _assert_row(out[0], _dense_step(model, state, int(logits[0].argmax())), "retry decode")
    engine.release_request(0)
    assert engine.paged_stats()["leak"] is False
    engine.close_pool()


# ---------------------------------------------------------------- paged arm

_PAGED_SKIP = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="paged hybrid decode is CUDA + Triton only (there is no dense paged fallback)",
)


@_PAGED_SKIP
def test_paged_engine_matches_seat_for_prefill_and_three_decode_steps():
    # bf16 on CUDA: the packed-prefill flash arm computes fp16/bf16 only and D9-7 makes
    # bf16 the formal dtype; the two arms still share one model, so this stays an
    # arm-vs-arm comparison (Triton block-table vs hand_baddbmm), not a kernel A/B.
    model = _model("cuda", torch.bfloat16)
    seat = Qwen35Engine(model, max_prefill_tokens=64)
    paged = Qwen35Engine(model, max_prefill_tokens=64)
    seat.open_hybrid_pool(max_batch=2, max_len=32, linear_slots=2, pool="seat")
    paged.open_hybrid_pool(
        max_batch=2, max_len=32, linear_slots=2, kv_blocks=16, block_size=4, pool="paged"
    )
    assert seat.paged is False and paged.paged is True
    assert paged.paged_max_batch == 2
    with pytest.raises(RuntimeError, match="no seat slab"):
        paged.slab

    specs = _specs(PROMPTS, [0, 1], [4, 3])
    seat_logits = seat.prefill_group(specs)
    paged_logits = paged.prefill_group(specs)
    # The two arms run the *same* packed varlen prefill (only the sink differs), so the
    # prefill logits are bit-identical even in bf16.
    torch.testing.assert_close(paged_logits, seat_logits, atol=0.0, rtol=0)
    assert paged_logits.argmax(-1).tolist() == seat_logits.argmax(-1).tolist()

    seat_tokens = [int(seat_logits[0].argmax()), int(seat_logits[1].argmax())]
    paged_tokens = list(seat_tokens)
    for step in range(DECODE_STEPS):
        seat_out = seat.decode_batch(seat_tokens)
        paged_out = paged.decode_batch(paged_tokens)
        torch.testing.assert_close(paged_out, seat_out, atol=ATOL_BF16, rtol=0)
        assert paged_out.argmax(-1).tolist() == seat_out.argmax(-1).tolist(), (
            f"step {step}: paged greedy differs from seat"
        )
        seat_tokens = [int(seat_out[0].argmax()), int(seat_out[1].argmax())]
        paged_tokens = [int(paged_out[0].argmax()), int(paged_out[1].argmax())]
    assert paged.last_decode_event_ms is not None

    paged_stats = paged.paged_stats()
    assert paged_stats["decode_impl"] == "triton"
    assert paged_stats["blocks_in_use"] > 0
    assert paged_stats["written_blocks"] > 0
    assert paged_stats["kv_live_mb"] <= paged_stats["kv_alloc_mb"] <= paged_stats["kv_mb"]
    assert paged_stats["leak"] is False

    for engine in (seat, paged):
        engine.release_request(0)
        engine.release_request(1)
        stats = engine.paged_stats()
        assert stats["active_rows"] == 0
        assert stats["linear_state_mb"] == 0.0
        assert stats["leak"] is False
        engine.close_pool()
        engine.close_pool()
