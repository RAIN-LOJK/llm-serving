"""``scripts/sim_paged_capacity.py``: the W4 capacity simulation against hand arithmetic.

The simulator is the deterministic half of ``docs/w4-plan.md`` §4.1: no model, no
weights, no arrivals. These tests pin the pieces the report leans on -- the seat row
model, the paged fill-only capacity, the Appendix C workspace formula, and the D6 rule
that workspace is paid out of the *same* total budget as KV (P-36).
"""

from __future__ import annotations

import json

import pytest

from serve.block_manager import BlockManager
from scripts.sim_paged_capacity import (
    DISTRIBUTIONS,
    MIB,
    ModelShape,
    build_requests,
    bucket_width,
    capacity_ratio,
    concurrency_gate,
    effective_seat_max_batch,
    load_trace,
    paged_capacity,
    run_sweep,
    seat_capacity,
    seat_row_width,
    select_distributions,
    workspace_mb,
)

# A tiny shape that makes every number hand-checkable:
# KV = 2 * n_kv_head * head_dim * dtype_bytes * n_layer = 2*2*4*2*2 = 64 B/token.
TINY = ModelShape(n_layer=2, n_kv_head=2, head_dim=4, n_head=2, vocab=8, dtype_bytes=2)


# ---------------------------------------------------------------- geometry

def test_kv_bytes_per_token_matches_qwen3_0_6b():
    assert TINY.kv_bytes_per_token == 64
    # carrier.md: 112 KiB/token for the 0.6B full-attention skeleton.
    assert ModelShape().kv_bytes_per_token == 112 * 1024


def test_bucket_width_rounds_up_to_64():
    assert bucket_width(1) == 64
    assert bucket_width(64) == 64
    assert bucket_width(65) == 128
    assert bucket_width(2176) == 2176
    assert bucket_width(2177) == 2240


def test_seat_row_width_matches_plan_anchors():
    # docs/w4-plan.md §4.1: W3-like max-2048 -> row width 2240 (2048+128+1 bucketed).
    assert seat_row_width(build_requests("w3-like-max-2048")) == 2240
    # §4.2: 8 longest frozen prompts (1517) + budget 128 -> 1664.
    assert seat_row_width([(1517, 128)] * 8) == 1664


# ---------------------------------------------------------------- workspace (Appendix C)

def test_workspace_matches_appendix_c_anchors():
    shape = ModelShape()
    anchors = [
        # (n_active, lpad, max_batch, quoted MiB)
        (8, 1664, 8, 55),
        (20, 2240, 64, 195),
        (23, 1664, 64, 170),
    ]
    for n_active, lpad, max_batch, quoted in anchors:
        value = workspace_mb(
            n_active=n_active,
            lpad=lpad,
            max_batch=max_batch,
            shape=shape,
        )
        assert value == pytest.approx(quoted, abs=4.0), (n_active, lpad, max_batch, value)


def test_workspace_terms_are_separable():
    shape = ModelShape()
    total = workspace_mb(n_active=10, lpad=1024, max_batch=16, shape=shape)
    gather = 10 * 1024 * (2 * shape.n_kv_head * shape.head_dim * shape.dtype_bytes) / MIB
    bias = 10 * shape.n_kv_head * (shape.n_head // shape.n_kv_head) * 1024 * shape.dtype_bytes / MIB
    logits = 16 * shape.vocab * shape.dtype_bytes / MIB
    assert total == pytest.approx(gather + bias + logits)


def test_workspace_does_not_multiply_by_layer_count():
    # Appendix C: the gather buffer is one layer's worth, reused layer by layer.
    shallow = ModelShape(n_layer=2)
    deep = ModelShape(n_layer=28)
    assert workspace_mb(n_active=8, lpad=1664, max_batch=8, shape=shallow) == pytest.approx(
        workspace_mb(n_active=8, lpad=1664, max_batch=8, shape=deep)
    )


# ---------------------------------------------------------------- hand example

def test_hand_computed_small_example():
    # 4 requests, prompt 10 + budget 6 = 16 tokens.
    requests = [(10, 6)] * 4
    # Seat: row width bucket(17) = 64 tokens -> 64 * 64 B = 4096 B per row.
    # Total budget 8192 B = 2 rows, but max_batch caps at 8 -> capacity 2, nothing unused.
    budget_mb = 8192 / MIB
    seat = seat_capacity(budget_mb, requests, shape=TINY, max_batch=8)
    assert seat.row_width == 64
    assert seat.row_mb == pytest.approx(4096 / MIB)
    assert seat.capacity == 2
    assert seat.kv_mb == pytest.approx(8192 / MIB)
    assert seat.seat_unused_mb == pytest.approx(0.0)

    # Paged: bs16 -> 1024 B/block, 8 blocks. Each request needs exactly 1 block
    # (16 tokens), so all four fit, and the fill-only capacity is 4.
    paged = paged_capacity(
        budget_mb, requests, block_size=16, shape=TINY, workspace_override_mb=0.0
    )
    assert paged.n_blocks == 8
    assert paged.reserved_blocks == 4
    assert paged.capacity == 4
    # Internal frag is measured once prompt + budget is written: 4*16 tokens in
    # 4*16 block slots -> no tail waste. Fill-only writes only the 40 prompt tokens,
    # so the reservation slack (not internal frag) carries the unwritten budget.
    assert paged.internal_frag == pytest.approx(0.0)
    assert paged.reservation_slack == pytest.approx(1 - 40 / 64)
    assert capacity_ratio(seat, paged) == pytest.approx(2.0)


def test_seat_unused_is_budget_left_by_row_granularity():
    requests = [(10, 6)] * 4  # row = 64 tokens = 4096 B
    budget_mb = 3 * 4096 / MIB  # room for 3 rows
    seat = seat_capacity(budget_mb, requests, shape=TINY, max_batch=8)
    assert seat.capacity == 3
    assert seat.seat_unused_mb == pytest.approx(0.0)

    # 2.5 rows' worth: two rows fit, half a row is unusable.
    half = seat_capacity(2.5 * 4096 / MIB, requests, shape=TINY, max_batch=8)
    assert half.capacity == 2
    assert half.seat_unused_mb == pytest.approx(0.5 * 4096 / MIB)


# ---------------------------------------------------------------- BlockManager contract

def test_reserved_blocks_follow_the_real_block_manager():
    requests = [(16, 16), (1, 16), (0, 0), (31, 1), (32, 0)]
    for block_size in (16, 32):
        bm = BlockManager(1, block_size)  # only blocks_for() is being pinned
        expected = sum(bm.blocks_for(prompt, budget) for prompt, budget in requests)
        # A budget large enough that every request is admitted.
        result = paged_capacity(
            1e6, requests, block_size=block_size, shape=TINY, workspace_override_mb=0.0
        )
        assert result.capacity == len(requests)
        assert result.reserved_blocks == expected


def test_block_boundaries_one_block_per_exact_fit():
    assert BlockManager(1, 16).blocks_for(16, 0) == 1
    assert BlockManager(1, 16).blocks_for(17, 0) == 2
    assert BlockManager(1, 16).blocks_for(0, 0) == 1  # empty request still one block
    # 17 written tokens in 2 blocks -> 1 - 17/32 internal frag at the tail.
    result = paged_capacity(1e6, [(17, 0)], block_size=16, shape=TINY, workspace_override_mb=0.0)
    assert result.reserved_blocks == 2
    assert result.internal_frag == pytest.approx(1 - 17 / 32)


# ---------------------------------------------------------------- D6: same total budget

def test_budget_sweep_is_monotone_for_both_arms():
    budgets = [900.0, 1200.0, 1456.0, 1640.0, 2400.0, 3200.0]
    for name in DISTRIBUTIONS:
        requests = build_requests(name)
        previous_seat = 0
        for budget in budgets:
            seat = seat_capacity(budget, requests, shape=ModelShape())
            assert seat.capacity >= previous_seat
            previous_seat = seat.capacity
        for block_size in (16, 32):
            previous = -1
            for budget in budgets:
                paged = paged_capacity(
                    budget, requests, block_size=block_size, shape=ModelShape()
                )
                assert paged.capacity >= previous, (name, block_size, budget)
                previous = paged.capacity


def test_workspace_comes_out_of_the_same_budget():
    requests = build_requests("w3-like-max-2048")
    capacities = []
    for workspace in (0.0, 55.0, 120.0, 240.0, 400.0):
        paged = paged_capacity(
            1640.0,
            requests,
            block_size=16,
            shape=ModelShape(),
            workspace_override_mb=workspace,
        )
        capacities.append(paged.capacity)
        assert paged.workspace_mb == pytest.approx(workspace)
        assert paged.kv_total_mb <= 1640.0 + 1e-9
    assert capacities == sorted(capacities, reverse=True)
    assert capacities[-1] < capacities[0]


def test_capacity_per_gib_formula():
    requests = build_requests("w3-like-max-2048")
    seat = seat_capacity(1640.0, requests, shape=ModelShape())
    paged = paged_capacity(1640.0, requests, block_size=16, shape=ModelShape())
    assert seat.capacity_per_gib == pytest.approx(seat.capacity / (seat.kv_mb / 1024))
    assert paged.capacity_per_gib == pytest.approx(paged.capacity / (paged.kv_total_mb / 1024))


def test_ratio_uses_integer_admissions():
    requests = build_requests("mild-mixed")
    seat = seat_capacity(1640.0, requests, shape=ModelShape())
    paged = paged_capacity(1640.0, requests, block_size=16, shape=ModelShape())
    assert capacity_ratio(seat, paged) == pytest.approx(paged.capacity / seat.capacity)


def test_admission_is_fcfs_with_head_of_line_blocking():
    # 4 blocks. Long first (3 blocks), then a short (1): a second short would still
    # fit in the raw multiset, but admission stops at 2 because fill-only never
    # releases and the queue head is refused.
    requests = [(3 * 16, 0), (16, 0), (16, 0), (16, 0)]
    paged = paged_capacity(
        4 * 16 * TINY.kv_bytes_per_token / MIB,
        requests,
        block_size=16,
        shape=TINY,
        workspace_override_mb=0.0,
    )
    assert paged.n_blocks == 4
    assert paged.capacity == 2
    # Order is part of the input: shorts first admit three of them, not one.
    reordered = requests[1:] + requests[:1]
    paged_reordered = paged_capacity(
        4 * 16 * TINY.kv_bytes_per_token / MIB,
        reordered,
        block_size=16,
        shape=TINY,
        workspace_override_mb=0.0,
    )
    assert paged_reordered.capacity == 3


def test_distributions_have_the_planned_sizes():
    assert len(build_requests("uniform-2048")) == 32
    assert len(build_requests("mild-mixed")) == 32
    w3 = build_requests("w3-like-max-2048")
    assert len(w3) == 32
    lengths = [prompt for prompt, _budget in w3]
    assert sum(1 for n in lengths if n <= 64) == 12
    assert sum(1 for n in lengths if 256 <= n <= 512) == 12
    assert sum(1 for n in lengths if 1536 <= n <= 2048) == 8
    assert max(lengths) == 2048
    # No budget token is silently dropped: every request pays prompt + budget.
    assert all(budget == 128 for _prompt, budget in build_requests("uniform-2048"))


# ---------------------------------------------------------------- trace input


def _metrics_row(
    *, repeat: int, arrival_idx: int, context: int, out_tokens: int, budget: int
) -> dict:
    return {
        "row_kind": "request",
        "repeat": repeat,
        "arrival_idx": arrival_idx,
        "context": context,
        "out_tokens": out_tokens,
        "budget": budget,
    }


def _write_metrics(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_trace_metrics_dedups_repeats_and_orders_by_arrival_idx(tmp_path):
    # Two repeats of the same 4-request trace, plus step / trace rows that must be
    # ignored, written out of arrival order inside each repeat.
    repeat_0 = [
        _metrics_row(repeat=0, arrival_idx=3, context=300, out_tokens=100, budget=128),
        _metrics_row(repeat=0, arrival_idx=1, context=40, out_tokens=16, budget=16),
        _metrics_row(repeat=0, arrival_idx=0, context=20, out_tokens=4, budget=32),
        _metrics_row(repeat=0, arrival_idx=2, context=200, out_tokens=32, budget=64),
    ]
    repeat_1 = [
        _metrics_row(repeat=1, arrival_idx=idx, context=999, out_tokens=999, budget=128)
        for idx in range(4)
    ]
    noise = [
        {"row_kind": "step", "repeat": 0, "step": 0, "active": 1},
        {"row_kind": "trace", "repeat": 0, "peak_concurrency": 4},
    ]
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, noise + repeat_1 + repeat_0)

    trace = load_trace(path)
    assert trace.source == "metrics-jsonl"
    assert trace.label == "metrics.jsonl"
    # Only repeat 0 survives; the repeat-1 sentinel lengths (999) never appear.
    assert trace.n_requests == 4
    # prompt_len = context - out_tokens, ordered by arrival_idx.
    assert trace.requests == [(16, 32), (24, 16), (168, 64), (200, 128)]


def test_trace_metrics_label_override(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, [_metrics_row(repeat=0, arrival_idx=0, context=10, out_tokens=0, budget=8)])
    assert load_trace(path, label="0011 trace").label == "0011 trace"


def test_trace_json_list_pair_and_dict_formats(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([[39, 128], [1517, 64], [0, 16]]), encoding="utf-8")
    trace = load_trace(pairs)
    assert trace.source == "json-list"
    assert trace.requests == [(39, 128), (1517, 64), (0, 16)]

    dicts = tmp_path / "dicts.json"
    dicts.write_text(
        json.dumps(
            [
                {"prompt_len": 1517, "budget": 64},
                {"prompt_len": 39, "budget": 128},
            ]
        ),
        encoding="utf-8",
    )
    trace = load_trace(dicts)
    assert trace.source == "json-list"
    # Given order is preserved, not sorted.
    assert trace.requests == [(1517, 64), (39, 128)]


def test_trace_seat_capacity_matches_hand_arithmetic(tmp_path):
    path = tmp_path / "trace.json"
    # Longest reservation = 1517 + 128 = 1645 -> +1 -> bucket -> 1664 columns.
    path.write_text(json.dumps([[1517, 128], [64, 16], [512, 32]]), encoding="utf-8")
    trace = load_trace(path)
    assert seat_row_width(trace.requests) == 1664

    # Row = 1664 * 64 B = 106496 B; budget 4 rows' worth holds exactly 4 rows, and the
    # trace length (3) is the binding cap only because no --seat-max-batch is implied.
    row_mb = 1664 * TINY.kv_bytes_per_token / MIB
    seat = seat_capacity(4 * row_mb, trace.requests, shape=TINY, max_batch=len(trace.requests))
    assert seat.row_width == 1664
    assert seat.capacity == 3
    assert seat.seat_unused_mb == pytest.approx(row_mb)

    # 2.5 rows' worth: two rows fit, and the trace has room for them.
    seat2 = seat_capacity(2.5 * row_mb, trace.requests, shape=TINY, max_batch=len(trace.requests))
    assert seat2.capacity == 2


def test_trace_paged_capacity_is_a_fcfs_prefix(tmp_path):
    path = tmp_path / "trace.json"
    # 3 blocks of 16 tokens each; a 3-block request followed by 1-block requests.
    path.write_text(json.dumps([[48, 0], [16, 0], [16, 0]]), encoding="utf-8")
    trace = load_trace(path)
    budget_mb = 4 * 16 * TINY.kv_bytes_per_token / MIB
    paged = paged_capacity(
        budget_mb,
        trace.requests,
        block_size=16,
        shape=TINY,
        workspace_override_mb=0.0,
        max_batch=trace.n_requests,
    )
    assert paged.capacity == 2  # 3 + 1 blocks; the third does not fit at the head


def test_trace_wins_over_distributions():
    trace = load_trace_stub()
    names, ignored = select_distributions("uniform-2048", trace=trace)
    assert names == [trace.label]
    assert ignored is True

    names, ignored = select_distributions(None, trace=trace)
    assert names == [trace.label]
    assert ignored is False

    names, ignored = select_distributions(None, trace=None)
    assert names == list(DISTRIBUTIONS)
    assert ignored is False

    names, ignored = select_distributions("mild-mixed", trace=None)
    assert names == ["mild-mixed"]
    assert ignored is False


def load_trace_stub():
    from scripts.sim_paged_capacity import Trace

    return Trace(requests=[(10, 16)], label="stub", source="json-list")


def test_seat_max_batch_is_uncapped_on_a_trace():
    requests = [(10, 6)] * 4
    # Synthetic keeps the historical default of 64.
    assert effective_seat_max_batch(None, requests, trace_mode=False) == 64
    # A trace is capped only by its own length; an explicit value still wins.
    assert effective_seat_max_batch(None, requests, trace_mode=True) == 4
    assert effective_seat_max_batch(2, requests, trace_mode=True) == 2


def test_trace_sweep_reports_source_and_request_count(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([[10, 6]] * 4), encoding="utf-8")
    trace = load_trace(path)
    rows = run_sweep(
        [2.0],
        [trace.label],
        [16],
        shape=TINY,
        workspace_override_mb=0.0,
        seat_max_batch=None,
        paged_max_batch=64,
        trace=trace,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "trace"
    assert row["trace_source"] == "json-list"
    assert row["n_requests"] == 4
    assert row["distribution"] == "trace.json"


def test_concurrency_gate_pass_and_fail():
    assert concurrency_gate(29, 29) == {
        "sim": 29,
        "expected": 29,
        "delta": 0,
        "tolerance": 1,
        "gate": "PASS",
    }
    assert concurrency_gate(28, 29)["gate"] == "PASS"
    assert concurrency_gate(30, 29)["gate"] == "PASS"
    fail = concurrency_gate(26, 29)
    assert fail["gate"] == "FAIL"
    assert fail["delta"] == -3
    assert concurrency_gate(26, 29, tolerance=3)["gate"] == "PASS"


def test_cli_trace_gate_exit_code(tmp_path):
    from scripts.sim_paged_capacity import main

    path = tmp_path / "trace.json"
    path.write_text(json.dumps([[10, 6]] * 4), encoding="utf-8")
    profile = [
        "--trace",
        str(path),
        "--budgets",
        "0.01171875",  # 12288 B: the TINY shape's paged fixpoint admits all four
        "--expect-budget",
        "0.01171875",
        "--block-sizes",
        "16",
        "--n-layer",
        "2",
        "--n-kv-head",
        "2",
        "--n-head",
        "2",
        "--head-dim",
        "4",
        "--vocab",
        "8",
        "--workspace-mb",
        "0",
    ]
    assert main(profile + ["--expect-concurrency", "4"]) == 0
    assert main(profile + ["--expect-concurrency", "9"]) == 1


# ---------------------------------------------------------------- --paged-blocks (E1)


def test_paged_blocks_uses_exact_fifo_fill_not_a_budget_fixpoint():
    # 6 blocks fit 3 requests of two blocks each. The budget-derived arm has room for
    # ~12 blocks, so it would solve a different (larger) capacity; the pinned pool must
    # answer with exactly the FIFO fill of 6 blocks, as the real fixed-`kv_blocks` pool.
    requests = [(32, 0)] * 5  # two 16-token blocks each
    block_mb = 16 * TINY.kv_bytes_per_token / MIB
    fixed = paged_capacity(
        100 * block_mb, requests, block_size=16, shape=TINY,
        workspace_override_mb=0.0, paged_blocks=6,
    )
    assert fixed.capacity == 3
    assert fixed.n_blocks == 6
    assert fixed.blocks_fixed is True

    derived = paged_capacity(
        100 * block_mb, requests, block_size=16, shape=TINY, workspace_override_mb=0.0
    )
    assert derived.blocks_fixed is False
    assert derived.capacity == 5
    assert derived.n_blocks > 6


def test_paged_blocks_total_is_blocks_times_block_mb_plus_workspace():
    requests = [(10, 6)] * 4
    block_mb = 16 * TINY.kv_bytes_per_token / MIB
    result = paged_capacity(
        2.5 * block_mb, requests, block_size=16, shape=TINY,
        workspace_override_mb=0.0, paged_blocks=3,
    )
    assert result.kv_mb == pytest.approx(3 * block_mb)
    assert result.kv_total_mb == pytest.approx(3 * block_mb)
    # 3 blocks of 16 tokens: one request (prompt 10 + budget 6 = 16) per block.
    assert result.capacity == 3
    assert result.over_budget is True
    assert result.over_by_mb == pytest.approx(0.5 * block_mb)

    # With an explicit workspace the total is N*block_mb + workspace, and the capacity
    # is still the fill on N blocks (the budget is irrelevant).
    with_ws = paged_capacity(
        0.0, requests, block_size=16, shape=TINY,
        workspace_override_mb=0.25, paged_blocks=3,
    )
    assert with_ws.kv_total_mb == pytest.approx(3 * block_mb + 0.25)
    assert with_ws.capacity == 3
    assert with_ws.over_budget is True
    assert with_ws.capacity_per_gib == pytest.approx(3 / (with_ws.kv_total_mb / 1024))


def test_paged_blocks_over_budget_still_reports_capacity():
    # A pool far above the budget: still a capacity, flagged OVER (plan §1.2 C1).
    requests = [(16, 0)] * 4
    block_mb = 16 * TINY.kv_bytes_per_token / MIB
    result = paged_capacity(
        block_mb, requests, block_size=16, shape=TINY,
        workspace_override_mb=0.0, paged_blocks=64,
    )
    assert result.over_budget is True
    assert result.over_by_mb == pytest.approx(63 * block_mb)
    assert result.capacity == 4
    # capacity_per_gib divides by the real kv_total_mb, not the budget.
    assert result.capacity_per_gib == pytest.approx(4 / (64 * block_mb / 1024))

    # At or under the budget the flag stays clear.
    fit = paged_capacity(
        64 * block_mb, requests, block_size=16, shape=TINY,
        workspace_override_mb=0.0, paged_blocks=64,
    )
    assert fit.over_budget is False
    assert fit.over_by_mb == 0.0


def test_paged_blocks_rejects_a_non_positive_pool():
    with pytest.raises(ValueError, match="paged_blocks"):
        paged_capacity(
            1.0, [(10, 6)], block_size=16, shape=TINY, workspace_override_mb=0.0,
            paged_blocks=0,
        )


def test_paged_blocks_sweep_row_carries_kv_blocks():
    from scripts.sim_paged_capacity import Trace

    block_mb = 16 * TINY.kv_bytes_per_token / MIB
    trace = Trace(requests=[(10, 6)] * 4, label="tiny", source="json-list")
    # Four 1-block requests; a 2-block pool fits the first two.
    rows = run_sweep(
        [4 * block_mb],
        [trace.label],
        [16],
        shape=TINY,
        workspace_override_mb=0.0,
        seat_max_batch=None,
        paged_max_batch=64,
        trace=trace,
        paged_blocks=2,
    )
    row = rows[0]
    assert row["kv_blocks"] == 2
    assert row["blocks_fixed"] is True
    assert row["paged"]["capacity"] == 2
    assert row["over_budget"] is False  # 2 blocks <= the 4-block budget
    # The derived row at the same budget reports a different pool size.
    derived = run_sweep(
        [4 * block_mb],
        [trace.label],
        [16],
        shape=TINY,
        workspace_override_mb=0.0,
        seat_max_batch=None,
        paged_max_batch=64,
        trace=trace,
    )[0]
    assert derived["kv_blocks"] == 4
    assert derived["blocks_fixed"] is False
    assert derived["paged"]["capacity"] == 4


def test_derived_pool_is_unchanged_when_paged_blocks_is_omitted():
    # Zero-regression pin: the default path must not depend on paged_blocks.
    requests = build_requests("w3-like-max-2048")
    before = paged_capacity(1640.0, requests, block_size=16, shape=ModelShape())
    after = paged_capacity(
        1640.0, requests, block_size=16, shape=ModelShape(), paged_blocks=None
    )
    assert before == after
    assert before.n_blocks == after.n_blocks
    assert before.capacity == after.capacity
    assert before.blocks_fixed is False and before.over_budget is False
    # The budget-derived pool never overruns the budget it was solved from.
    assert before.kv_total_mb <= 1640.0 + 1e-9


def test_cli_paged_blocks_gate_uses_the_fixed_pool(tmp_path):
    from scripts.sim_paged_capacity import main

    path = tmp_path / "trace.json"
    path.write_text(json.dumps([[10, 6]] * 4), encoding="utf-8")
    block_mb = 16 * TINY.kv_bytes_per_token / MIB
    profile = [
        "--trace",
        str(path),
        "--budgets",
        repr(64 * block_mb),
        "--expect-budget",
        repr(64 * block_mb),
        "--block-sizes",
        "16",
        "--paged-blocks",
        "2",
        "--n-layer",
        "2",
        "--n-kv-head",
        "2",
        "--n-head",
        "2",
        "--head-dim",
        "4",
        "--vocab",
        "8",
        "--workspace-mb",
        "0",
    ]
    # Exactly the fixed pool's FIFO fill, not the budget-derived 4.
    assert main(profile + ["--expect-concurrency", "2"]) == 0
    assert main(profile + ["--expect-concurrency", "4"]) == 1

