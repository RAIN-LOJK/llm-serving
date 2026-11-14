"""W7 Step 3: hybrid fill-only capacity with cross-step residency (plan §5.4).

``FillScheduler`` used to release every request at the end of its step, so
``peak_concurrency`` was one group's size. W7's fill probe instead keeps every admitted
request resident (prefill + reserve, no decode) until a pass cannot admit the FIFO head;
that resident set is the pool's capacity. These tests pin the lifecycle on the 24-layer
tiny model:

* **residency**: step ``active`` accumulates and stays put on the zero-admit pass, and
  ``retired`` stays 0 until the final drain;
* **no decode**: ``out`` is empty, ``out_tokens == 0`` and ``decode_batch`` is never
  called (a raising stub proves it);
* **accounting**: ``state_used/reserved/slack`` come from the pool's real tensors, the
  trace/step/request rows carry ``state_total_mb`` and ``fill_only``, and a later request
  stays queued (no slot, ``t_finish=None``) while every resident gets one ``t_finish`` at
  the drain;
* **no leak**: the pool has no active slot or block after the run and after ``close``.

The paged arm runs the same tiny model in bf16 on CUDA + Triton (the packed-prefill Flash
arm computes bf16/fp16 only; CUDA fp32 packed prefill fails by design) with a block pool
small enough to cap at two requests.
"""

from __future__ import annotations

import pytest
import torch

from bench.engines.fill import FillScheduler
from bench.engines.hybrid import HybridEngine
from bench.loadgen import closed_trace
from model.config import MiniLLMConfig
from model.qwen35_model import Qwen35Model
from model.triton_decode import TRITON_AVAILABLE
from serve.hybrid_engine import Qwen35Engine
from serve.scheduler import Request

# The W6 tiny config (24 layers, LLLF x 6), same fixture the engine tests use.
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
#: Six requests, two linear slots: the first pass admits two, the next is refused.
PROMPTS = [
    [3, 5, 7, 11, 13],
    [2, 4, 6, 8, 10],
    [1, 3, 5, 7, 9],
    [4, 8, 12, 16, 20],
    [6, 12, 18, 24, 30],
    [7, 14, 21, 28, 35],
]

_PAGED_SKIP = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="paged hybrid fill needs CUDA + Triton (there is no dense paged fallback)",
)


class _FakeFillEngine:
    """Capacity-only stand-in: two residents fit, ``prefill_group`` counts forwards."""

    paged = True
    max_position = 64
    max_prefill_tokens = 8192
    eos_token_id = None

    def __init__(self, capacity: int = 2) -> None:
        self.capacity = int(capacity)
        self.active = 0
        self.preflights = 0
        self.releases = 0

    def can_admit_group(self, requests) -> bool:
        return self.active + len(requests) <= self.capacity

    def prefill_group(self, requests):
        self.preflights += 1
        self.active += len(requests)
        return [[0.0, 1.0] for _ in requests]

    def release_request(self, request_id) -> None:
        self.releases += 1
        self.active -= 1

    def paged_stats(self) -> dict:
        return {
            "linear_state_mb": float(self.active),
            "active_rows": self.active,
            "leak": False,
        }


def _request(index: int) -> Request:
    return Request(id=index, prompt_ids=[1 + index, 2], budget=1, arrival_ts=0.0)


def test_fill_scheduler_drains_every_resident_at_one_timestamp():
    """Residents are released only after the loop; the refused tail stays queued."""
    engine = _FakeFillEngine(capacity=2)
    scheduler = FillScheduler(engine, max_batch=4)
    requests = [_request(index) for index in range(6)]
    for request in requests:
        scheduler.submit(request)
    scheduler.close()

    out = scheduler.run()

    # One packed prefill for the two residents; the refused pass runs no forward.
    assert engine.preflights == 1
    assert [row["active"] for row in scheduler.step_rows] == [2, 2]
    assert [row["admitted"] for row in scheduler.step_rows] == [2, 0]
    assert all(row["retired"] == 0 for row in scheduler.step_rows)
    # The two residents share the drain timestamp and are the only finished requests.
    residents, queued = out[:2], out[2:]
    assert all(request.t_finish is not None for request in residents)
    assert len({request.t_finish for request in residents}) == 1
    assert all(request.state == "finished" and request.out == [] for request in residents)
    assert all(request.t_finish is None for request in queued)
    assert all(request.state == "queued" for request in queued)
    assert engine.active == 0


def _model(device: str = "cpu", dtype: torch.dtype = torch.float32) -> Qwen35Model:
    torch.manual_seed(0)
    return Qwen35Model(MiniLLMConfig(**TINY)).to(device=device, dtype=dtype).eval()


def _fill_adapter(
    model: Qwen35Model,
    *,
    engine: Qwen35Engine | None = None,
    device: str = "cpu",
    max_batch: int = 4,
    linear_slots: int = 2,
    pool: str = "seat",
    kv_blocks: int = 1,
    block_size: int = 4,
) -> HybridEngine:
    """A loaded adapter around an in-memory tiny model (no HF download)."""
    adapter = HybridEngine(
        "tiny-qwen35",
        dtype="float32" if model.token_embedding.weight.dtype == torch.float32 else "bfloat16",
        device=device,
        max_batch=max_batch,
        policy="continuous",
        vocab_size=VOCAB,
        seed=0,
        fill=True,
        max_prefill_tokens=8192,
        linear_slots=linear_slots,
        kv_blocks=kv_blocks,
        block_size=block_size,
        pool=pool,
    )
    adapter._engine = Qwen35Engine(model) if engine is None else engine
    adapter._eos_token_id = None
    return adapter


def _trace(prompts=PROMPTS, budgets=1):
    entries = [(f"p{index}", list(prompt)) for index, prompt in enumerate(prompts)]
    return closed_trace(entries, n=len(entries), budgets=[budgets], seed=0)


def _forbid_decode(engine: Qwen35Engine) -> None:
    """Make any decode call fail loudly: a fill probe must not decode."""

    def boom(*_args, **_kwargs):
        raise AssertionError("fill-only run called decode_batch()")

    engine.decode_batch = boom  # type: ignore[method-assign]


def test_seat_fill_keeps_residents_and_accounts_state_without_decode():
    model = _model()
    engine = Qwen35Engine(model)
    adapter = _fill_adapter(model, engine=engine, linear_slots=2)
    _forbid_decode(engine)
    try:
        result = adapter.run_trace(_trace())
        pool_stats = engine.state_pool.stats()
        capacity = result.summary["capacity"]

        # ---- residency: one two-request pass, then the same two residents are held
        # while the next pass admits nothing (that refusal *is* the capacity answer).
        assert [row["active"] for row in result.step_rows] == [2, 2]
        assert [row["admitted"] for row in result.step_rows] == [2, 0]
        assert all(row["retired"] == 0 for row in result.step_rows)
        assert all(row["fill_only"] is True for row in result.step_rows)
        assert all(row["decode_event_ms"] == 0.0 for row in result.step_rows)

        # ---- no decode anywhere
        assert capacity["peak_concurrency"] == 2
        assert result.trace_row["out_tokens_total"] == 0
        assert result.trace_row["fill_only"] is True
        assert all(row["out_tokens"] == 0 for row in result.request_rows)

        # ---- the third request never got a slot and stays queued
        finished = [row for row in result.request_rows if row["state"] == "finished"]
        queued = [row for row in result.request_rows if row["state"] == "queued"]
        assert len(finished) == 2 and len(queued) == 4
        assert all(row["admitted"] is True for row in finished)
        assert all(row["admitted"] is False for row in queued)
        assert all(row["linear_mb"] is None for row in queued)
        assert all(row["state_total_mb"] is None for row in queued)
        # One drain timestamp finishes every resident together (t_finish > arrival).
        assert len({row["e2e_ms"] for row in finished}) == 1
        assert all(row["e2e_ms"] > 0.0 for row in finished)
        assert all(row["e2e_ms"] == 0.0 for row in queued)
        row = finished[0]
        assert row["fill_only"] is True
        assert row["linear_mb"] == pytest.approx(pool_stats.per_slot_bytes / 2**20)
        assert row["state_total_mb"] == pytest.approx(
            row["linear_mb"] + row["kv_written_mb"]
        )

        # ---- the state ledger is measured from the pool, not the analytic 18.633 MiB.
        # W7 §6.3 basis: used = linear used + written KV at the same step; reserved =
        # every linear slot + the whole KV container.
        per_slot_mb = pool_stats.per_slot_bytes / 2**20
        assert capacity["per_slot_state_mb"] == pytest.approx(per_slot_mb)
        assert capacity["linear_slots"] == 2
        assert capacity["linear_reserved_mb"] == pytest.approx(pool_stats.reserved_mb)
        assert capacity["kv_reserved_mb"] == pytest.approx(capacity["kv_mb"])
        assert capacity["state_reserved_mb"] == pytest.approx(
            pool_stats.reserved_mb + capacity["kv_mb"]
        )
        assert capacity["state_total_mb"] == pytest.approx(capacity["state_reserved_mb"])
        assert capacity["linear_used_mb"] == pytest.approx(2 * per_slot_mb)
        assert capacity["kv_used_mb"] == pytest.approx(
            result.step_rows[-1]["kv_live_mb"]
        )
        assert capacity["state_used_mb"] == pytest.approx(
            capacity["linear_used_mb"] + capacity["kv_used_mb"]
        )
        assert capacity["state_slack_mb"] == pytest.approx(
            capacity["state_reserved_mb"] - capacity["state_used_mb"]
        )
        assert capacity["workspace_mb"] == 0.0
        assert capacity["kv_total_mb"] == pytest.approx(capacity["kv_mb"])
        assert capacity["over_budget"] is False
        assert capacity["leak"] is False
        assert capacity["workload_sha256"] == _trace().sha256
        assert result.trace_row["state_total_mb"] == pytest.approx(
            capacity["state_total_mb"]
        )

        # ---- the drain released everyone; nothing is left for close_pool to find
        assert engine.state_pool.stats().active == 0
        assert engine.paged_stats()["active_rows"] == 0
        assert engine.paged_stats()["leak"] is False
    finally:
        adapter.close()
    assert engine.state_pool.stats().active == 0
    engine.state_pool.assert_no_leaks()


@_PAGED_SKIP
def test_paged_fill_capacity_is_bounded_by_blocks_not_slots():
    # bf16: the packed-prefill Flash arm computes bf16/fp16 only. Four blocks of four
    # tokens; a four-token prompt plus a one-token budget reserves two blocks, so the
    # pool admits exactly two residents and the third is refused on the free list.
    model = _model("cuda", torch.bfloat16)
    engine = Qwen35Engine(model)
    adapter = _fill_adapter(
        model,
        engine=engine,
        device="cuda",
        max_batch=4,
        linear_slots=4,
        pool="paged",
        kv_blocks=4,
        block_size=4,
    )
    _forbid_decode(engine)
    try:
        result = adapter.run_trace(_trace(prompts=[[1, 2, 3, 4]] * 6))
        capacity = result.summary["capacity"]

        assert capacity["peak_concurrency"] == 2
        assert capacity["peak_concurrency"] == capacity["kv_blocks"] // 2
        assert capacity["kv_blocks"] == 4
        assert capacity["block_size"] == 4
        # §6.3 total basis: two resident slots' linear state plus their written KV.
        assert capacity["linear_used_mb"] == pytest.approx(
            2 * capacity["per_slot_state_mb"]
        )
        assert capacity["linear_reserved_mb"] == pytest.approx(
            4 * capacity["per_slot_state_mb"]
        )
        assert capacity["kv_used_mb"] == pytest.approx(
            result.step_rows[-1]["kv_live_mb"]
        )
        assert capacity["state_used_mb"] == pytest.approx(
            capacity["linear_used_mb"] + capacity["kv_used_mb"]
        )
        assert capacity["state_reserved_mb"] == pytest.approx(
            capacity["linear_reserved_mb"] + capacity["kv_mb"]
        )
        assert capacity["state_slack_mb"] == pytest.approx(
            capacity["state_reserved_mb"] - capacity["state_used_mb"]
        )
        assert capacity["state_total_mb"] == pytest.approx(capacity["state_reserved_mb"])
        assert result.step_rows[-1]["blocks_in_use"] == 4
        assert result.step_rows[-1]["free_blocks"] == 0
        assert result.step_rows[-1]["admitted"] == 0
        assert all(row["out_tokens"] == 0 for row in result.request_rows)

        # The drain returns every block and slot to the pool.
        assert engine.paged_stats()["active_rows"] == 0
        assert engine.paged_stats()["blocks_in_use"] == 0
        assert engine.paged_stats()["leak"] is False
    finally:
        adapter.close()
    engine.state_pool.assert_no_leaks()
