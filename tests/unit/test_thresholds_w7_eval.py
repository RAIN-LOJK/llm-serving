"""W7 threshold evaluators: paired, explicit, and never a default pass.

The W7 pair is a hybrid 0.8B arm against a pure 0.6B arm under the same 1640 MiB state
budget. These tests pin the five behaviours the evaluators must keep:

* a capacity/P1 ratio is read only when both fill arms report the same context,
  device/dtype/batch and a budget inside 1640 MiB;
* TPOT growth is TPOT(8k)/TPOT(512) from the per-context ``groups``, not from an
  overall median that mixes contexts;
* the 512 latency criterion reports TTFT and TPOT separately and returns ``口径依赖``
  when they disagree instead of forcing a pass;
* a missing memory-crossover measurement stays pending (no extrapolation);
* a broken pair is ``invalid``, never a quiet ratio.

The summaries here are synthetic; no W7 run exists.
"""

from __future__ import annotations

import copy

import pytest

from bench import thresholds

#: The real `[W7.hybrid]` values, copied so this file also fails if the table drifts.
W7 = {
    "p1_state_ratio_8k_min": 5.0,
    "capacity_speedup_512_min": 2.0,
    "capacity_speedup_2k_min": 4.0,
    "capacity_speedup_8k_min": 6.0,
    "memory_crossover_ctx_max": 256,
    "p2_latency_material_ratio": 1.10,
    "hybrid_tpot_growth_max": 1.10,
    "pure_tpot_growth_min": 1.15,
}
TABLE = {"W7": {"hybrid": dict(W7)}}

PROTOCOL = {
    "max_new_tokens": 128,
    "prompt_sha256": None,
    "dtype": "bfloat16",
    "device_class": "measure-cuda",
    "max_batch": 64,
    "trace_sha256": "trace-w7",
}


def _check(name, ours, baseline=None):
    return thresholds.evaluate([name], ours, TABLE, baseline=baseline)[0]


def fill_summary(
    *,
    peak,
    state_total,
    context=512,
    state_used="same",
    protocol=None,
    capacity_context=True,
    groups=True,
    capacity_extra=None,
):
    """One fill summary with the fields the W7 evaluators consume."""
    capacity = {
        "peak_concurrency": peak,
        "state_total_mb": state_total,
        "budget_mb": 1640.0,
    }
    if state_used == "same":
        capacity["state_used_mb"] = state_total
    elif state_used is not None:
        capacity["state_used_mb"] = state_used
    if capacity_context and context is not None:
        capacity["context_tokens"] = context
    if capacity_extra:
        capacity.update(capacity_extra)
    summary = {
        "protocol": dict(PROTOCOL) if protocol is None else dict(protocol),
        "capacity": capacity,
    }
    if groups and context is not None:
        summary["groups"] = [
            {
                "context": context,
                "out_tokens": {"median": 0.0},
                "batch": 1,
                "n": max(1, int(peak)),
            }
        ]
    else:
        summary["groups"] = []
    return summary


def scenario_summary(*, tpot_512, tpot_8k, batch=8, ttft_512=20.0, protocol=None):
    """E3 summary with the three per-context groups and their nominal prompt context."""
    summary = {
        "protocol": dict(PROTOCOL) if protocol is None else dict(protocol),
        "groups": [
            {
                "context": 512 + 128,
                "out_tokens": {"median": 128.0},
                "batch": 1,
                "n": 1,
                "ttft_ms": {"median": ttft_512},
                "tpot_ms": {"median": tpot_512},
            },
            {
                "context": 2048 + 128,
                "out_tokens": {"median": 128.0},
                "batch": 1,
                "n": 1,
                "ttft_ms": {"median": ttft_512 * 2.0},
                "tpot_ms": {"median": (tpot_512 + tpot_8k) / 2.0},
            },
            {
                "context": 8192 + 128,
                "out_tokens": {"median": 128.0},
                "batch": 1,
                "n": 1,
                "ttft_ms": {"median": ttft_512 * 4.0},
                "tpot_ms": {"median": tpot_8k},
            },
        ],
    }
    summary["protocol"]["max_batch"] = batch
    return summary


# --------------------------------------------------------------------------- capacity


def test_capacity_speedup_512_passes_below_the_line():
    ours = fill_summary(peak=62, state_total=1620.0, context=512)
    baseline = fill_summary(peak=23, state_total=1638.0, context=512)
    entry = _check("capacity_speedup_512_min", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(62 / 23, rel=1e-3)
    assert entry["context_tokens"] == 512
    assert entry["state_total_mb"] == 1620.0


def test_capacity_speedup_2k_and_8k_use_their_own_contexts():
    for name, context, ours_peak, base_peak, value in (
        ("capacity_speedup_2k_min", 2048, 37, 6, 37 / 6),
        ("capacity_speedup_8k_min", 8192, 14, 1, 14.0),
    ):
        ours = fill_summary(peak=ours_peak, state_total=1630.0, context=context)
        baseline = fill_summary(peak=base_peak, state_total=1638.0, context=context)
        entry = _check(name, ours, baseline)
        assert entry["status"] == "pass", entry
        assert entry["value"] == pytest.approx(value, rel=1e-3)


def test_capacity_speedup_fails_below_the_line():
    ours = fill_summary(peak=20, state_total=1620.0, context=512)
    baseline = fill_summary(peak=23, state_total=1638.0, context=512)
    entry = _check("capacity_speedup_512_min", ours, baseline)
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(20 / 23, rel=1e-3)


def test_capacity_speedup_pending_when_context_is_not_recorded():
    ours = fill_summary(peak=62, state_total=1620.0, context=None)
    baseline = fill_summary(peak=23, state_total=1638.0, context=None)
    entry = _check("capacity_speedup_512_min", ours, baseline)
    assert entry["status"] == "pending", entry
    assert "context" in entry["reason"]


def test_capacity_speedup_pending_when_state_total_is_missing():
    ours = fill_summary(peak=62, state_total=1620.0, context=512)
    ours["capacity"].pop("state_total_mb")
    baseline = fill_summary(peak=23, state_total=1638.0, context=512)
    entry = _check("capacity_speedup_512_min", ours, baseline)
    assert entry["status"] == "pending", entry
    assert "state_total_mb" in entry["reason"]


def test_capacity_speedup_invalid_when_over_budget():
    ours = fill_summary(peak=62, state_total=1700.0, context=512)
    baseline = fill_summary(peak=23, state_total=1638.0, context=512)
    entry = _check("capacity_speedup_512_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("1640" in problem for problem in entry["problems"])


def test_capacity_speedup_invalid_on_context_mismatch():
    ours = fill_summary(peak=62, state_total=1620.0, context=1024)
    baseline = fill_summary(peak=23, state_total=1638.0, context=1024)
    entry = _check("capacity_speedup_512_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("context_tokens" in problem for problem in entry["problems"])


def test_capacity_speedup_invalid_on_trace_mismatch():
    ours = fill_summary(peak=62, state_total=1620.0, context=512)
    baseline = fill_summary(peak=23, state_total=1638.0, context=512)
    mismatch = copy.deepcopy(baseline)
    mismatch["protocol"]["trace_sha256"] = "other-trace"
    entry = _check("capacity_speedup_512_min", ours, mismatch)
    assert entry["status"] == "invalid", entry
    assert any("trace_sha256" in problem for problem in entry["problems"])


def test_capacity_speedup_invalid_on_device_or_dtype_mismatch():
    ours = fill_summary(peak=62, state_total=1620.0, context=512)
    baseline = fill_summary(peak=23, state_total=1638.0, context=512)
    mismatch = copy.deepcopy(baseline)
    mismatch["protocol"]["dtype"] = "float32"
    entry = _check("capacity_speedup_512_min", ours, mismatch)
    assert entry["status"] == "invalid", entry
    assert any("dtype" in problem for problem in entry["problems"])


# --------------------------------------------------------------------------------- P1


def test_p1_state_ratio_passes_when_hybrid_is_five_times_smaller():
    ours = fill_summary(peak=14, state_total=1625.9, state_used=1604.9, context=8192)
    baseline = fill_summary(peak=1, state_total=1638.0, state_used=896.0, context=8192)
    entry = _check("p1_state_ratio_8k_min", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx((896.0 / 1) / (1604.9 / 14), rel=1e-3)
    assert "state_used_mb" in entry["basis"]


def test_p1_state_ratio_fails_when_hybrid_is_not_small_enough():
    ours = fill_summary(peak=1, state_total=500.0, state_used=400.0, context=8192)
    baseline = fill_summary(peak=1, state_total=1638.0, state_used=896.0, context=8192)
    entry = _check("p1_state_ratio_8k_min", ours, baseline)
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(896.0 / 400.0, rel=1e-3)


def test_p1_falls_back_to_reserved_state_and_says_so():
    ours = fill_summary(peak=14, state_total=1625.9, state_used=None, context=8192)
    baseline = fill_summary(peak=1, state_total=1638.0, state_used=None, context=8192)
    entry = _check("p1_state_ratio_8k_min", ours, baseline)
    assert entry["status"] == "pass", entry
    assert "state_total_mb" in entry["basis"]
    assert "reserved" in entry["basis"]


def test_p1_pending_when_state_is_missing():
    ours = fill_summary(peak=14, state_total=1625.9, state_used=None, context=8192)
    ours["capacity"].pop("state_total_mb")
    baseline = fill_summary(peak=1, state_total=1638.0, state_used=None, context=8192)
    baseline["capacity"].pop("state_total_mb")
    entry = _check("p1_state_ratio_8k_min", ours, baseline)
    assert entry["status"] == "pending", entry
    assert "state" in entry["reason"]


def test_p1_invalid_when_the_pair_is_not_at_8k():
    ours = fill_summary(peak=14, state_total=1625.9, context=2048)
    baseline = fill_summary(peak=6, state_total=1638.0, context=2048)
    entry = _check("p1_state_ratio_8k_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("8192" in problem for problem in entry["problems"])


# ------------------------------------------------------------------------- TPOT growth


def test_hybrid_tpot_growth_passes_at_or_below_the_ceiling():
    entry = _check("hybrid_tpot_growth_max", scenario_summary(tpot_512=10.0, tpot_8k=10.5))
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(1.05)
    assert entry["batch"] == 8


def test_hybrid_tpot_growth_fails_above_the_ceiling():
    entry = _check("hybrid_tpot_growth_max", scenario_summary(tpot_512=10.0, tpot_8k=12.0))
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(1.2)


def test_pure_tpot_growth_requires_the_minimum():
    passing = _check("pure_tpot_growth_min", scenario_summary(tpot_512=10.0, tpot_8k=12.0))
    failing = _check("pure_tpot_growth_min", scenario_summary(tpot_512=10.0, tpot_8k=11.0))
    assert passing["status"] == "pass", passing
    assert failing["status"] == "fail", failing


def test_tpot_growth_pending_without_the_8k_group():
    summary = scenario_summary(tpot_512=10.0, tpot_8k=10.5)
    summary["groups"] = [group for group in summary["groups"] if group["context"] != 8192 + 128]
    entry = _check("hybrid_tpot_growth_max", summary)
    assert entry["status"] == "pending", entry
    assert "8192" in entry["reason"]


# ---------------------------------------------------------------------------- P2 latency


def test_p2_passes_when_both_ttft_and_tpot_are_materially_worse():
    ours = scenario_summary(tpot_512=10.0, tpot_8k=10.5, ttft_512=20.0)
    baseline = scenario_summary(tpot_512=9.0, tpot_8k=9.0, ttft_512=18.0)
    entry = _check("p2_latency_material_ratio", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["ttft_ratio"] == pytest.approx(20 / 18, rel=1e-3)
    assert entry["tpot_ratio"] == pytest.approx(10 / 9, rel=1e-3)


def test_p2_fails_when_neither_metric_is_materially_worse():
    ours = scenario_summary(tpot_512=9.5, tpot_8k=10.0, ttft_512=19.0)
    baseline = scenario_summary(tpot_512=9.0, tpot_8k=9.0, ttft_512=18.0)
    entry = _check("p2_latency_material_ratio", ours, baseline)
    assert entry["status"] == "fail", entry


def test_p2_reports_kou_jing_yi_lai_when_ttft_and_tpot_disagree():
    ours = scenario_summary(tpot_512=9.5, tpot_8k=10.0, ttft_512=20.0)
    baseline = scenario_summary(tpot_512=9.0, tpot_8k=9.0, ttft_512=18.0)
    entry = _check("p2_latency_material_ratio", ours, baseline)
    assert entry["status"] == "口径依赖", entry
    assert entry["ttft_ratio"] >= 1.10 > entry["tpot_ratio"]


def test_p2_invalid_on_batch_mismatch():
    ours = scenario_summary(tpot_512=10.0, tpot_8k=10.5, batch=1, ttft_512=20.0)
    baseline = scenario_summary(tpot_512=9.0, tpot_8k=9.0, batch=8, ttft_512=18.0)
    entry = _check("p2_latency_material_ratio", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("max_batch" in problem or "batch" in problem for problem in entry["problems"])


def test_p2_pending_without_a_paired_baseline():
    ours = scenario_summary(tpot_512=10.0, tpot_8k=10.5)
    entry = _check("p2_latency_material_ratio", ours, None)
    assert entry["status"] == "pending", entry


# ---------------------------------------------------------------------- memory crossover


def test_memory_crossover_passes_from_an_explicit_field():
    summary = {"memory_crossover": {"ctx_tokens": 193.0}}
    entry = _check("memory_crossover_ctx_max", summary)
    assert entry["status"] == "pass", entry
    assert entry["value"] == 193.0
    assert entry["source"] == "memory_crossover.ctx_tokens"


def test_memory_crossover_fails_above_the_ceiling():
    entry = _check("memory_crossover_ctx_max", {"memory_crossover": {"ctx_tokens": 300.0}})
    assert entry["status"] == "fail", entry


def test_memory_crossover_is_pending_without_an_explicit_measurement():
    """An 8k capacity ratio must not be reverse-engineered into a crossover token."""
    entry = _check("memory_crossover_ctx_max", {"capacity": {"peak_concurrency": 14}})
    assert entry["status"] == "pending", entry
    assert "does not record" in entry["reason"]


# ------------------------------------------------------------------------------ lookup


def test_real_w7_threshold_section_resolves_every_w7_criterion():
    real = thresholds.load()
    for name in W7:
        found = thresholds.lookup(name, real)
        assert found is not None, name
        section, value = found
        assert section == "W7.hybrid", (name, section)
        assert value == W7[name], (name, value)
