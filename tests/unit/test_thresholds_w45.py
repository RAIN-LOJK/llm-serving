"""W4.5 threshold evaluators: every new line is paired, directional, and never defaults to pass.

The four ratio criteria compare two arms of this repo that ran the same trace on the
same pool width, so the attention backend / prefill mode is the variable under test and
must not invalidate the pair. ``golden_token_match_min`` needs no baseline (it reads the
runner's golden block) and ``paged_vs_seat_token_match_min`` compares the two arms'
per-prompt token digests. Missing measurements are ``pending``, a broken pair is
``invalid``, and neither is a pass.
"""

from __future__ import annotations

import pytest

from bench import thresholds

#: The real `[W45.paged]` section, copied so this file also fails if the table drifts.
W45 = {
    "decode_step_ratio_max": 1.15,
    "packed_prefill_ttft_p99_speedup_min": 1.15,
    "packed_prefill_ttft_p50_ratio_max": 1.10,
    "capacity_speedup_min": 1.50,
    "fragmentation_max": 0.05,
    "golden_token_match_min": 1.0,
    "paged_vs_seat_token_match_min": 0.999,
}
TABLE = {"W45": {"paged": dict(W45)}}

PROTOCOL = {
    "max_new_tokens": 128,
    "prompt_sha256": "abc123",
    "dtype": "bfloat16",
    "device_class": "measure-cuda",
    "trace_sha256": "trace-abc",
    "max_batch": 8,
    "prefill_mode": "packed",
    "max_prefill_tokens": 8192,
    "attn_prefill": "flash-attn-varlen",
    "attn_decode": "triton",
}

_MISSING = object()


def scenario_summary(
    *,
    steps=None,
    ttft_p50=None,
    ttft_p99=None,
    tokens=None,
    golden=_MISSING,
    protocol=None,
):
    summary = {
        "protocol": dict(PROTOCOL if protocol is None else protocol),
        "latency": {
            "ttft_ms": {"p50": ttft_p50, "p95": ttft_p99, "p99": ttft_p99},
        },
        "steps": {"by_active": list(steps or [])},
    }
    if tokens is not None:
        summary["tokens"] = tokens
    if golden is not _MISSING:
        summary["golden"] = golden
    return summary


def step_rows(*pairs):
    """``(active, step_ms)`` -> the ``steps.by_active`` shape ``bench.scenario`` emits."""
    return [{"active": active, "step_ms": step_ms, "n": 1} for active, step_ms in pairs]

def _check(name, ours, baseline=None):
    return thresholds.evaluate([name], ours, TABLE, baseline=baseline)[0]


# --------------------------------------------------------------------------- config


def test_real_w45_section_is_provisional_and_complete():
    table = thresholds.load()
    section = table["W45"]["paged"]
    assert section["status"] == "target"
    assert section["basis_run"] == ""
    for name, value in W45.items():
        assert section[name] == value, name
    # W4's own section is untouched by the W4.5 addition.
    assert table["W4"]["paged"]["max_concurrency_speedup_min"] == 1.50
    assert table["W4"]["paged"]["fragmentation_max"] == 0.05


# ------------------------------------------------------------------ decode step ratio


def test_decode_step_ratio_passes_below_the_ceiling():
    ours = scenario_summary(steps=step_rows((8, 10.0), (4, 20.0)))
    baseline = scenario_summary(steps=step_rows((8, 12.5), (4, 25.0)))
    # The decode backend is the variable: different attn_decode must not invalidate.
    baseline["protocol"]["attn_decode"] = "hand_baddbmm"
    entry = _check("decode_step_ratio_max", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(0.8)
    assert entry["active_bucket"] == 8


def test_decode_step_ratio_fails_above_the_ceiling():
    ours = scenario_summary(steps=step_rows((8, 10.0), (4, 20.0)))
    baseline = scenario_summary(steps=step_rows((8, 8.0), (4, 16.0)))
    entry = _check("decode_step_ratio_max", ours, baseline)
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(1.25)


def test_decode_step_ratio_uses_the_largest_common_bucket_only():
    """C4 is the steady-state B=8 decode cost, not an average over admissions."""
    ours = scenario_summary(steps=step_rows((8, 10.0), (4, 20.0)))
    baseline = scenario_summary(steps=step_rows((8, 20.0)))
    entry = _check("decode_step_ratio_max", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["active_bucket"] == 8
    assert entry["value"] == pytest.approx(0.5)


def test_decode_step_ratio_is_pending_without_a_baseline_or_steps():
    assert _check("decode_step_ratio_max", scenario_summary())["status"] == "pending"
    baseline = scenario_summary(steps=step_rows((8, 10.0)))
    entry = _check("decode_step_ratio_max", scenario_summary(), baseline)
    assert entry["status"] == "pending", entry
    assert "active bucket" in entry["reason"]


def test_decode_step_ratio_is_invalid_on_a_different_trace():
    ours = scenario_summary(steps=step_rows((8, 10.0)))
    baseline = scenario_summary(steps=step_rows((8, 10.0)))
    baseline["protocol"]["trace_sha256"] = "trace-other"
    entry = _check("decode_step_ratio_max", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("trace_sha256" in problem for problem in entry["problems"])


# ---------------------------------------------------------------- packed prefill TTFT


def test_packed_p99_speedup_passes_and_fails_directionally():
    ours = scenario_summary(ttft_p99=100.0)
    faster = scenario_summary(ttft_p99=120.0)
    same = scenario_summary(ttft_p99=100.0)
    assert _check("packed_prefill_ttft_p99_speedup_min", ours, faster)["status"] == "pass"
    failed = _check("packed_prefill_ttft_p99_speedup_min", ours, same)
    assert failed["status"] == "fail", failed
    assert failed["value"] == pytest.approx(1.0)


def test_packed_p99_speedup_is_pending_without_ttft():
    baseline = scenario_summary(ttft_p99=100.0)
    entry = _check("packed_prefill_ttft_p99_speedup_min", scenario_summary(), baseline)
    assert entry["status"] == "pending", entry
    assert "p99" in entry["reason"]


def test_packed_p99_speedup_is_invalid_on_a_different_pool_width():
    ours = scenario_summary(ttft_p99=100.0)
    baseline = scenario_summary(ttft_p99=120.0)
    baseline["protocol"]["max_batch"] = 32
    entry = _check("packed_prefill_ttft_p99_speedup_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("max_batch" in problem for problem in entry["problems"])


def test_packed_p50_ratio_passes_when_the_median_does_not_regress():
    ours = scenario_summary(ttft_p50=50.0)
    baseline = scenario_summary(ttft_p50=50.0)
    entry = _check("packed_prefill_ttft_p50_ratio_max", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(1.0)


def test_packed_p50_ratio_fails_when_the_median_regresses():
    entry = _check(
        "packed_prefill_ttft_p50_ratio_max",
        scenario_summary(ttft_p50=60.0),
        scenario_summary(ttft_p50=50.0),
    )
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(1.2)


def test_packed_p50_ratio_is_pending_without_a_baseline():
    entry = _check("packed_prefill_ttft_p50_ratio_max", scenario_summary(ttft_p50=50.0))
    assert entry["status"] == "pending", entry
    assert "baseline" in entry["reason"]


# ------------------------------------------------------------------------------ golden


def test_golden_match_passes_at_full_agreement():
    entry = _check(
        "golden_token_match_min",
        scenario_summary(
            golden={
                "path": "tests/golden/qwen3-0-6b/0001",
                "match_rate": 1.0,
                "matched_ids": 64,
                "total_ids": 64,
                "matched_prompts": 1,
                "total_prompts": 1,
                "details": [{"prompt_id": "s01", "status": "pass"}],
            }
        ),
    )
    assert entry["status"] == "pass", entry
    assert entry["matched_ids"] == 64


def test_golden_match_fails_on_a_single_different_id():
    entry = _check(
        "golden_token_match_min",
        scenario_summary(golden={"match_rate": 63 / 64, "matched_prompts": 0, "total_prompts": 1}),
    )
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(0.9844, abs=1e-4)


def test_golden_match_is_pending_without_a_block_or_on_an_error():
    assert _check("golden_token_match_min", scenario_summary())["status"] == "pending"
    error = _check(
        "golden_token_match_min",
        scenario_summary(golden={"path": "missing", "error": "FileNotFoundError: nope"}),
    )
    assert error["status"] == "pending", error
    assert "unavailable" in error["reason"]
    assert (
        _check("golden_token_match_min", scenario_summary(golden={"match_rate": None}))["status"]
        == "pending"
    )


# ------------------------------------------------------------------ paged vs seat C3


def test_paged_vs_seat_token_match_passes_on_identical_digests():
    entry = _check(
        "paged_vs_seat_token_match_min",
        scenario_summary(tokens={"s02": "b", "s01": "a"}),
        scenario_summary(tokens={"s01": "a", "s03": "c", "s02": "b"}),
    )
    assert entry["status"] == "pass", entry
    assert entry["n_prompts"] == 2
    assert entry["value"] == 1.0


def test_paged_vs_seat_token_match_fails_on_one_digest():
    entry = _check(
        "paged_vs_seat_token_match_min",
        scenario_summary(tokens={"s01": "a", "s02": "b"}),
        scenario_summary(tokens={"s01": "a", "s02": "different"}),
    )
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(0.5)
    assert entry["mismatched_prompt_ids"] == ["s02"]


def test_paged_vs_seat_token_match_is_pending_without_data():
    assert _check("paged_vs_seat_token_match_min", scenario_summary(tokens={"s01": "a"}))[
        "status"
    ] == "pending"
    empty = _check(
        "paged_vs_seat_token_match_min",
        scenario_summary(tokens={}),
        scenario_summary(tokens={"s01": "a"}),
    )
    assert empty["status"] == "pending", empty
    no_common = _check(
        "paged_vs_seat_token_match_min",
        scenario_summary(tokens={"s01": "a"}),
        scenario_summary(tokens={"s02": "b"}),
    )
    assert no_common["status"] == "pending", no_common


# -------------------------------------------------------------------------- capacity


def capacity_summary(peak, kv_total_mb=1600.0, *, max_batch=16):
    return {
        "protocol": {
            "dtype": "bfloat16",
            "device_class": "measure-cuda",
            "max_batch": max_batch,
            "attn_prefill": "flash-attn-varlen",
            "attn_decode": "triton",
        },
        "capacity": {"peak_concurrency": peak, "kv_total_mb": kv_total_mb},
    }


def test_capacity_speedup_min_is_an_alias_for_the_existing_check():
    entry = _check(
        "capacity_speedup_min",
        capacity_summary(28),
        capacity_summary(9),
    )
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(28 / 9, abs=1e-3)
    assert entry["section"] == "W45.paged"
    # The old name still has its own evaluator and W4 section.
    old = thresholds.evaluate(
        ["max_concurrency_speedup_min"],
        capacity_summary(28),
        {"W4": {"paged": {"max_concurrency_speedup_min": 1.50}}},
        baseline=capacity_summary(9),
    )[0]
    assert old["status"] == "pass", old


def test_capacity_speedup_min_fails_below_target_and_pends_without_data():
    failed = _check("capacity_speedup_min", capacity_summary(10), capacity_summary(9))
    assert failed["status"] == "fail", failed
    pending = _check(
        "capacity_speedup_min",
        {"protocol": capacity_summary(10)["protocol"]},
        capacity_summary(9),
    )
    assert pending["status"] == "pending", pending
    assert "peak_concurrency" in pending["reason"]
