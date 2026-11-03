"""W3 §5.5: trace-level aggregation is pure, so it is pinned with hand-built rows.

The rows below use the field names frozen in ``docs/w3-plan.md`` §6.1. Nothing here touches
a model, a GPU, or a clock: ``aggregate`` must be checkable from numbers alone, otherwise
the C2 decomposition check cannot be trusted.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from bench.metrics import percentiles, summarize
from bench.scenario import aggregate, analytic_straggler_ratio


# --- fixtures ---------------------------------------------------------------
#
# Six request rows over two repeats, six steps over two repeats. TTFT decomposes exactly
# (queue + prefill == ttft) so ``decomposed`` must be true; the per-trace throughput values
# are chosen so every reported percentile/median has a single obvious answer.

def _request_rows() -> list[dict]:
    rows = []
    for idx, (ttft, queue) in enumerate(
        [(10.0, 9.0), (20.0, 18.0), (30.0, 27.0), (40.0, 36.0), (50.0, 45.0), (60.0, 54.0)]
    ):
        repeat = 0 if idx < 3 else 1
        rows.append(
            {
                "row_kind": "request",
                "repeat": repeat,
                "arrival_idx": idx,
                "out_tokens": 10 if repeat == 0 else 20,
                "ttft_ms": ttft,
                "queue_ms": queue,
                "prefill_ms": ttft - queue,
            }
        )
    return rows


def _step_rows() -> list[dict]:
    return [
        # repeat 0: host_gap_ms given on the first step, derived on the second.
        {"row_kind": "step", "repeat": 0, "step": 0, "active": 2, "step_ms": 40.0,
         "prefill_ms": 10.0, "decode_event_ms": 30.0, "host_gap_ms": 0.0},
        {"row_kind": "step", "repeat": 0, "step": 1, "active": 2, "step_ms": 40.0,
         "prefill_ms": 10.0, "decode_event_ms": 30.0},
        {"row_kind": "step", "repeat": 1, "step": 0, "active": 4, "step_ms": 50.0,
         "prefill_ms": 5.0, "decode_event_ms": 30.0, "host_gap_ms": 15.0},
        {"row_kind": "step", "repeat": 1, "step": 1, "active": 4, "step_ms": 55.0,
         "prefill_ms": 5.0, "decode_event_ms": 30.0, "host_gap_ms": 20.0},
    ]


def _trace_rows() -> list[dict]:
    return [
        {"row_kind": "trace", "repeat": 0, "makespan_s": 10.0, "out_tokens_total": 1000,
         "tok_s": 100.0, "tok_s_active": 110.0, "req_s": 3.0, "offered_rate": 20.0,
         "achieved_rate": 3.0, "queue_depth_max": 20, "queue_depth_at_last_arrival": 12,
         "saturated": True, "drops": 0},
        {"row_kind": "trace", "repeat": 1, "makespan_s": 12.0, "out_tokens_total": 1440,
         "tok_s": 120.0, "tok_s_active": 130.0, "req_s": 4.0, "offered_rate": 20.0,
         "achieved_rate": 4.0, "queue_depth_max": 18, "queue_depth_at_last_arrival": 5,
         "saturated": True, "drops": 0},
    ]


def _summary(**kwargs):
    return aggregate(
        _request_rows(),
        _step_rows(),
        _trace_rows(),
        **kwargs,
    )


# --- percentiles ------------------------------------------------------------

def test_percentiles_empty_is_all_none():
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None}


def test_percentiles_single_value():
    assert percentiles([7.0]) == {"p50": 7.0, "p95": 7.0, "p99": 7.0}


def test_percentiles_known_rank():
    # Nearest rank, same rule as summarize: rank = ceil(q * n), 1-based.
    stats = percentiles(range(1, 101))
    assert stats == {"p50": 50.0, "p95": 95.0, "p99": 99.0}


def test_percentiles_ignores_missing_and_non_finite():
    stats = percentiles([1.0, None, float("nan"), 3.0])
    assert stats == {"p50": 1.0, "p95": 3.0, "p99": 3.0}


def test_percentiles_agrees_with_summarize_p95():
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0]
    assert percentiles(values)["p95"] == summarize(values)["p95"]


# --- throughput / latency ---------------------------------------------------

def test_throughput_block_comes_from_trace_rows():
    summary = _summary()
    throughput = summary["throughput"]
    assert throughput["n_traces"] == 2
    assert throughput["tok_s"]["n"] == 2
    assert throughput["tok_s"]["median"] == 110.0
    assert throughput["tok_s"]["p95"] == 120.0
    assert throughput["tok_s_active"]["median"] == 120.0
    assert throughput["req_s"]["median"] == 3.5
    assert throughput["makespan_s"]["median"] == 11.0
    assert throughput["makespan_s"]["p95"] == 12.0


def test_achieved_rate_and_offered_rate():
    throughput = _summary()["throughput"]
    assert throughput["offered_rate"] == 20.0
    assert throughput["achieved_rate"]["median"] == 3.5
    assert throughput["achieved_rate"]["n"] == 2


def test_achieved_rate_falls_back_to_req_s():
    trace_rows = [dict(_trace_rows()[0])]
    trace_rows[0].pop("achieved_rate")
    throughput = aggregate([], [], trace_rows)["throughput"]
    assert throughput["achieved_rate"]["median"] == 3.0


def test_latency_percentiles_use_every_request_row():
    latency = _summary()["latency"]
    assert latency["ttft_ms"] == {"p50": 30.0, "p95": 60.0, "p99": 60.0}
    assert latency["queue_ms"] == {"p50": 27.0, "p95": 54.0, "p99": 54.0}
    assert latency["prefill_ms"] == {"p50": 3.0, "p95": 6.0, "p99": 6.0}
    assert latency["queue_share_p99"] == pytest.approx(54.0 / 60.0)


def test_queue_share_p99_is_none_without_denominator():
    rows = [dict(row, ttft_ms=0.0, queue_ms=0.0, prefill_ms=0.0) for row in _request_rows()]
    latency = aggregate(rows, [], [])["latency"]
    assert latency["queue_share_p99"] is None


# --- decomposition (C2) -----------------------------------------------------

def test_decomposed_when_ttft_equals_queue_plus_prefill():
    summary = _summary()
    assert summary["latency"]["decomposed"] is True
    assert summary["consistency"]["decomposed_rate"] == 1.0
    assert summary["consistency"]["ttft_minus_queue_prefill_max_ms"] == 0.0


def test_decomposed_is_false_when_a_request_is_perturbed_by_5ms():
    rows = _request_rows()
    rows[2]["ttft_ms"] = rows[2]["queue_ms"] + rows[2]["prefill_ms"] + 5.0
    summary = aggregate(rows, _step_rows(), _trace_rows())
    assert summary["latency"]["decomposed"] is False
    assert summary["consistency"]["decomposed_rate"] == pytest.approx(5.0 / 6.0)
    assert summary["consistency"]["ttft_minus_queue_prefill_max_ms"] == pytest.approx(5.0)


def test_decomposed_is_false_when_a_component_is_missing():
    rows = _request_rows()
    for row in rows:
        row.pop("prefill_ms")
    summary = aggregate(rows, [], [])
    latency = summary["latency"]
    assert latency["decomposed"] is False
    assert latency["prefill_ms"]["p99"] is None
    assert summary["consistency"]["decomposed_rate"] == 0.0


def test_decomposed_boundary_is_exactly_99_percent():
    # 100 clean requests and one off by 5 ms: the rate is exactly 0.99, which still counts.
    clean = [
        {"ttft_ms": 30.0, "queue_ms": 28.0, "prefill_ms": 2.0} for _ in range(99)
    ]
    perturbed = [{"ttft_ms": 35.0, "queue_ms": 28.0, "prefill_ms": 2.0}]
    summary = aggregate(clean + perturbed, [], [])
    assert summary["consistency"]["decomposed_rate"] == pytest.approx(0.99)
    assert summary["latency"]["decomposed"] is True

    summary = aggregate(clean + [perturbed[0], dict(perturbed[0], queue_ms=20.0)], [], [])
    assert summary["consistency"]["decomposed_rate"] < 0.99
    assert summary["latency"]["decomposed"] is False


# --- steps ------------------------------------------------------------------

def test_steps_group_by_active_and_average():
    by_active = _summary()["steps"]["by_active"]
    assert [entry["active"] for entry in by_active] == [2, 4]
    two, four = by_active
    assert two == {"active": 2, "n": 2, "step_ms": 40.0, "prefill_ms": 10.0,
                   "decode_event_ms": 30.0, "host_gap_ms": 0.0}
    assert four == {"active": 4, "n": 2, "step_ms": 52.5, "prefill_ms": 5.0,
                    "decode_event_ms": 30.0, "host_gap_ms": 17.5}


def test_host_gap_is_derived_only_when_all_terms_exist():
    step_rows = [
        {"row_kind": "step", "active": 1, "step_ms": 10.0, "prefill_ms": 2.0,
         "decode_event_ms": 3.0},
        {"row_kind": "step", "active": 1, "step_ms": 10.0, "decode_event_ms": 3.0},
    ]
    entry = aggregate([], step_rows, [])["steps"]["by_active"][0]
    # First step derives 5.0; the second has no prefill_ms, so it cannot be guessed.
    assert entry["host_gap_ms"] == 5.0
    assert entry["n"] == 2


def test_decode_only_tok_s_is_sum_tokens_over_decode_time():
    throughput = _summary()["throughput"]
    decode = throughput["decode_only_tok_s"]
    # repeat 0: 30 tokens / 60 ms -> 500 tok/s; repeat 1: 60 / 60 -> 1000 tok/s.
    assert decode["n"] == 2
    assert decode["median"] == 750.0
    assert decode["p95"] == 1000.0


def test_decode_only_tok_s_is_none_without_decode_timing():
    step_rows = [{key: value for key, value in row.items() if key != "decode_event_ms"}
                 for row in _step_rows()]
    throughput = aggregate(_request_rows(), step_rows, _trace_rows())["throughput"]
    assert throughput["decode_only_tok_s"] == {
        "n": 0, "median": None, "p95": None, "min": None, "max": None, "mean": None, "cv": None,
    }


# --- saturation (P-29) ------------------------------------------------------

def test_p29_saturated_even_when_achieved_rate_is_below_any_capacity_guess():
    # A weak arm: 0.5 req/s achieved is below every plausible capacity estimate, yet the
    # trace is fully submitted and 7 requests are still queued. The old "compare against a
    # capacity constant" definition would call this not_saturated and invalidate C1.
    trace_rows = [
        {"row_kind": "trace", "repeat": 0, "makespan_s": 256.0, "out_tokens_total": 128,
         "tok_s": 0.5, "tok_s_active": 0.5, "req_s": 0.5, "offered_rate": 20.0,
         "achieved_rate": 0.5, "queue_depth_max": 64, "queue_depth_at_last_arrival": 7,
         "saturated": True, "drops": 0}
    ]
    throughput = aggregate([], [], trace_rows)["throughput"]
    assert throughput["saturated"] is True


def test_not_saturated_when_queue_is_empty_at_trace_end():
    trace_rows = [dict(_trace_rows()[0], queue_depth_at_last_arrival=0)]
    throughput = aggregate([], [], trace_rows)["throughput"]
    assert throughput["saturated"] is False


def test_saturation_requires_every_repeat_to_be_backed_up():
    trace_rows = [
        dict(_trace_rows()[0], queue_depth_at_last_arrival=12),
        dict(_trace_rows()[1], queue_depth_at_last_arrival=0),
    ]
    assert aggregate([], [], trace_rows)["throughput"]["saturated"] is False


# --- saturation end to end: dummy arm, no weights ----------------------------
#
# The bug this pins: the old field read the *last* step row's ``queued``, and a finished
# run has always drained its queue -- so ``saturated`` was False by construction and every
# E2 pair was ``invalid`` (P-29). The depth is now taken when the last request *arrives*.

def _run_e2_shaped_trace(rate: float | None, n: int = 32, max_batch: int = 4):
    """A trace of the E2 shape (``bench/prompts.json`` prefix) through the dummy arm.

    ``rate=None`` builds the closed trace (every arrival at t=0); a float builds the same
    arrival count as a Poisson trace. Flat budgets keep the run short on a CPU box.
    """
    from bench.engines.continuous import ContinuousEngine
    from bench.loadgen import PromptSet, closed_trace, poisson_trace

    prompt_set = PromptSet.load(Path(__file__).resolve().parents[2] / "bench" / "prompts.json")
    engine = ContinuousEngine("dummy", max_batch=max_batch, vocab_size=1024, seed=0)
    engine.load()
    try:
        entries = [(prompt.id, engine.encode(prompt.text)) for prompt in prompt_set.prompts[:n]]
        budgets = [8] * n
        trace = (
            closed_trace(entries, n=n, budgets=budgets, seed=0)
            if rate is None
            else poisson_trace(entries, rate=rate, n=n, budgets=budgets, seed=0)
        )
        return engine.run_trace(trace)
    finally:
        engine.close()


def test_saturated_is_true_when_backlog_remains_at_the_last_arrival():
    # Arrivals burst in far faster than the dummy arm can drain them. The run still
    # finishes (its last step has queued=0) -- that must not hide the backlog.
    result = _run_e2_shaped_trace(rate=5000.0)
    assert result.step_rows[-1]["queued"] == 0
    assert result.summary["throughput"]["saturated"] is True
    assert result.summary["throughput"]["queue_depth_at_last_arrival"]["median"] > 0


def test_not_saturated_when_requests_arrive_one_at_a_time():
    # Same trace shape, spread out: each request is served long before the next lands.
    result = _run_e2_shaped_trace(rate=50.0)
    assert result.summary["throughput"]["saturated"] is False
    assert result.summary["throughput"]["queue_depth_at_last_arrival"]["median"] == 0


def test_not_saturated_for_a_closed_trace_that_fits_the_pool():
    # n == max_batch and everything arrives at t=0: every request gets a seat at that
    # same instant, so nothing is queued at the last arrival (the E1 shape).
    result = _run_e2_shaped_trace(rate=None, n=4, max_batch=4)
    assert result.summary["throughput"]["saturated"] is False
    assert result.summary["throughput"]["queue_depth_at_last_arrival"]["median"] == 0


# --- analytic straggler ratio ----------------------------------------------

def test_analytic_straggler_ratio_matches_monte_carlo():
    budgets = [16, 32, 64, 128]
    max_batch = 8
    ratio = analytic_straggler_ratio(budgets, max_batch)
    assert ratio is not None

    # Fixed seed: the test may be approximate, the implementation may not be.
    rng = random.Random(1234)
    draws = 100_000
    maxima = 0.0
    means = 0.0
    for _ in range(draws):
        sample = [rng.choice(budgets) for _ in range(max_batch)]
        maxima += max(sample)
        means += sum(sample) / max_batch
    assert ratio == pytest.approx(maxima / means, rel=0.02)


def test_analytic_straggler_ratio_uniform_budgets_is_one():
    assert analytic_straggler_ratio([64, 64, 64, 64], 8) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("budgets", "max_batch"),
    [(None, 8), ([], 8), ([16, 32], 1), ([16, 32], 0)],
)
def test_analytic_straggler_ratio_edges(budgets, max_batch):
    assert analytic_straggler_ratio(budgets, max_batch) is None


def test_aggregate_threads_budgets_into_straggler_ratio():
    summary = _summary(budgets=[16, 32, 64, 128], max_batch=8)
    assert summary["throughput"]["analytic_straggler_ratio"] == pytest.approx(
        analytic_straggler_ratio([16, 32, 64, 128], 8)
    )
    assert _summary()["throughput"]["analytic_straggler_ratio"] is None


# --- shape ------------------------------------------------------------------

def test_aggregate_returns_the_four_spec_blocks():
    summary = _summary()
    assert set(summary) == {"throughput", "latency", "steps", "consistency"}
    assert set(summary["throughput"]) == {
        "n_traces", "tok_s", "tok_s_active", "decode_only_tok_s", "req_s", "makespan_s",
        "offered_rate", "achieved_rate", "saturated", "queue_depth_at_last_arrival",
        "analytic_straggler_ratio",
    }
    assert set(summary["latency"]) == {
        "ttft_ms", "queue_ms", "prefill_ms", "queue_share_p99", "decomposed",
    }
    assert set(summary["consistency"]) == {
        "decomposed_rate", "ttft_minus_queue_prefill_max_ms",
    }
