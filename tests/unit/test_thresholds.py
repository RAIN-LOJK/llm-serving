"""Threshold evaluation: criteria must mean something, and must not be fooled by mixing.

The regression this file pins: W2's prompt set spans 16..2048 tokens, so ``overall`` TTFT
CV is huge even on a perfectly steady machine. CV must be judged per length bucket.
"""

from __future__ import annotations

import pytest

from bench import thresholds


PROTOCOL = {
    "max_new_tokens": 128,
    "prompt_sha256": "abc123",
    "dtype": "bfloat16",
    "device_class": "measure-cuda",
}


def summary_with(cv_ttft=None, cv_tpot=None, bucket="short", overall_cv=3.0, protocol=None):
    """``overall_cv`` is deliberately awful: that is what mixing lengths looks like."""
    return {
        "overall": {
            "ttft_ms": {"cv": overall_cv, "median": 100.0},
            "tpot_ms": {"cv": overall_cv, "median": 10.0},
            "weights_mb": {"median": None},
            "kv_mb": {"median": None},
            "peak_mem_mb": {"median": None},
        },
        "stability": {
            bucket: {
                "ttft_ms_cv_median": cv_ttft,
                "ttft_ms_cv_max": cv_ttft,
                "tpot_ms_cv_median": cv_tpot,
                "tpot_ms_cv_max": cv_tpot,
            }
        },
        "by_prompt": [],
        "groups": [],
        "protocol": PROTOCOL if protocol is None else protocol,
    }


@pytest.fixture(scope="module")
def table():
    return thresholds.load()


def test_real_thresholds_file_is_readable(table):
    assert table["schema"] == 1
    assert table["W1"]["align"]["status"] == "calibrated"


def test_lookup_finds_a_key_in_any_section(table):
    section, value = thresholds.lookup("ttft_cv_max", table)
    assert section == "W2.bench"
    assert value == pytest.approx(0.10)


def test_cv_is_judged_inside_buckets_not_on_overall(table):
    # overall CV = 3.0 (mixing lengths); every bucket is steady -> must pass
    result = thresholds.evaluate(["ttft_cv_max"], summary_with(cv_ttft=0.03), table)
    assert result[0]["status"] == "pass"
    assert result[0]["bucket"] == "short"


def test_cv_fails_when_a_bucket_is_actually_noisy(table):
    result = thresholds.evaluate(["tpot_cv_max"], summary_with(cv_tpot=0.25, bucket="long"), table)
    assert result[0]["status"] == "fail"
    assert result[0]["bucket"] == "long"


def test_cv_without_repeats_is_pending_not_pass(table):
    result = thresholds.evaluate(["ttft_cv_max"], summary_with(), table)
    assert result[0]["status"] == "pending"
    assert "repeats" in result[0]["reason"]


def test_decode_ratio_is_matched_on_prompts(table):
    summary = summary_with()
    summary["by_prompt"] = [
        {"model": "m", "prompt": "s01", "tpot_ms": {"median": 10.0}, "out_tokens": {"median": 40}},
        {"model": "m", "prompt": "s02", "tpot_ms": {"median": 20.0}, "out_tokens": {"median": 40}},
    ]
    baseline = {
        "protocol": PROTOCOL,
        "by_prompt": [
            {"model": "m", "prompt": "s01", "tpot_ms": {"median": 12.0}, "out_tokens": {"median": 40}},
            {"model": "m", "prompt": "s02", "tpot_ms": {"median": 22.0}, "out_tokens": {"median": 40}},
            {"model": "m", "prompt": "never-ran", "tpot_ms": {"median": 99.0}, "out_tokens": {"median": 1}},
        ],
    }
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "pass"
    assert result[0]["value"] == pytest.approx(1.15, abs=0.01)  # median(1.2, 1.1)
    assert result[0]["n_prompts"] == 2


def test_decode_ratio_fails_when_we_are_slower(table):
    summary = summary_with()
    summary["by_prompt"] = [
        {"model": "m", "prompt": "s01", "tpot_ms": {"median": 20.0}, "out_tokens": {"median": 40}}
    ]
    baseline = {
        "protocol": PROTOCOL,
        "by_prompt": [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 10.0}, "out_tokens": {"median": 40}}],
    }
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "fail"
    assert result[0]["value"] == pytest.approx(0.5)


def test_decode_ratio_pending_without_baseline(table):
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary_with(), table)
    assert result[0]["status"] == "pending"
    assert "baseline" in result[0]["reason"]


def test_memory_ratio_judges_each_context_not_one_mixed_number(table):
    summary = summary_with()
    summary["overall"].update({"peak_mem_mb": {"median": 1200.0}, "driver_mem_mb": {"median": 1500.0}})
    summary["groups"] = [
        {"context": 64, "weights_mb": {"median": 1000.0}, "kv_mb": {"median": 50.0},
         "peak_mem_mb": {"median": 1050.0}},
        {"context": 512, "weights_mb": {"median": 1000.0}, "kv_mb": {"median": 200.0},
         "peak_mem_mb": {"median": 1300.0}},
    ]
    result = thresholds.evaluate(["memory_match_ratio"], summary, table)
    assert result[0]["status"] == "pass"          # median(1.0, 0.923) = 0.96
    assert result[0]["value"] == pytest.approx(0.96, abs=0.01)
    assert [item["context"] for item in result[0]["by_context"]] == [64, 512]
    assert result[0]["peak_over_driver"] == pytest.approx(0.8)


def test_memory_ratio_pending_without_groups(table):
    result = thresholds.evaluate(["memory_match_ratio"], summary_with(), table)
    assert result[0]["status"] == "pending"


def test_unknown_criterion_does_not_silently_pass(table):
    result = thresholds.evaluate(["made_up_metric"], summary_with(cv_ttft=0.0), table)
    assert result[0]["status"] == "pending"
    assert "no such threshold key" in result[0]["reason"]


def test_decode_ratio_rejects_a_dtype_mismatch(table):
    summary = summary_with()
    summary["by_prompt"] = [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 10.0},
                             "out_tokens": {"median": 40}}]
    baseline = {
        "protocol": {**PROTOCOL, "dtype": "float32"},
        "by_prompt": [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 12.0},
                       "out_tokens": {"median": 40}}],
    }
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "invalid"
    assert any("dtype" in problem for problem in result[0]["problems"])


def test_decode_ratio_rejects_a_different_attention_backend(table):
    """两个自实现臂换后端 = 换了一个引擎；比值不可归因，必须 invalid 而不是给数字。

    机制层键（``attn_prefill`` / ``attn_decode``）只在 baseline 报告自实现后端时生效
    （``_reports_mechanism``），HF 臂不在这一层。
    """
    summary = summary_with()
    summary["by_prompt"] = [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 10.0},
                             "out_tokens": {"median": 40}}]
    baseline = {
        "protocol": {**PROTOCOL, "attn_prefill": "sdpa", "attn_decode": "hand_baddbmm"},
        "by_prompt": [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 12.0},
                       "out_tokens": {"median": 40}}],
    }
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "invalid"
    assert any("attn_prefill" in problem for problem in result[0]["problems"])


def test_decode_ratio_accepts_the_hf_arm_as_baseline(table):
    """``decode_speed_vs_hf_min`` is a cross-implementation ratio: HF reports no in-house
    backend (``hf:...``), so only the common keys may gate it."""
    summary = summary_with()
    summary["by_prompt"] = [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 10.0},
                             "out_tokens": {"median": 40}}]
    baseline = {
        "protocol": {
            **PROTOCOL,
            "attn_prefill": "hf:hf_default",
            "attn_decode": "hf:hf_default",
        },
        "by_prompt": [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 12.0},
                       "out_tokens": {"median": 40}}],
    }
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "pass"
    assert result[0]["value"] == pytest.approx(1.2)
    # Explicitly asking for the mechanism layer still refuses it (different engines).
    assert thresholds.mechanism_mismatches(summary, baseline)


def test_mechanism_layer_refuses_two_inhouse_arms_on_different_backends(table):
    ours = {"protocol": {**PROTOCOL, "attn_prefill": "sdpa", "attn_decode": "hand_baddbmm"}}
    same = {"protocol": dict(ours["protocol"])}
    other = {"protocol": {**PROTOCOL, "attn_prefill": "sdpa", "attn_decode": "sdpa"}}

    assert thresholds.mechanism_mismatches(ours, same) == []
    problems = thresholds.mechanism_mismatches(ours, other)
    assert any("attn_decode" in problem for problem in problems)


def test_common_keys_ignore_backend_provenance(table):
    """The common layer must not invalidate a cross-implementation pairing."""
    ours = {"protocol": {**PROTOCOL, "attn_prefill": "sdpa", "attn_decode": "hand_baddbmm"}}
    baseline = {"protocol": {**PROTOCOL, "attn_prefill": "hf:hf_default"}}
    assert thresholds._protocol_mismatches(ours, baseline) == []


def test_decode_ratio_rejects_a_different_prompt_set(table):
    summary = summary_with()
    baseline = {"protocol": {**PROTOCOL, "prompt_sha256": "other"}, "by_prompt": []}
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "invalid"
    assert any("prompt_sha256" in problem for problem in result[0]["problems"])


def test_decode_ratio_rejects_different_output_lengths(table):
    summary = summary_with()
    summary["by_prompt"] = [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 10.0},
                             "out_tokens": {"median": 40}}]
    baseline = {
        "protocol": PROTOCOL,
        "by_prompt": [{"model": "m", "prompt": "s01", "tpot_ms": {"median": 12.0},
                       "out_tokens": {"median": 60}}],
    }
    result = thresholds.evaluate(["decode_speed_vs_hf_min"], summary, table, baseline=baseline)
    assert result[0]["status"] == "invalid"
    assert any("out_tokens" in problem for problem in result[0]["problems"])


# ---------------------------------------------------------------- W3 scenario criteria
SCENARIO_PROTOCOL = dict(
    PROTOCOL,
    attn_prefill="sdpa",
    attn_decode="hand_baddbmm",
    trace_sha256="trace-abc",
    max_batch=8,
)


def scenario_summary(tok_s=200.0, saturated=True, queue_at_last_arrival=5, decomposed=True):
    return {
        "protocol": dict(SCENARIO_PROTOCOL),
        "throughput": {
            "tok_s": {"median": tok_s},
            "saturated": saturated,
            "queue_depth_at_last_arrival": {"median": queue_at_last_arrival},
        },
        "latency": {"decomposed": decomposed},
    }


def _evaluate(name, ours, baseline):
    table = {"W3": {"continuous": {name: 1.30 if name == "throughput_speedup_min" else True}}}
    return thresholds.evaluate([name], ours, table, baseline=baseline)[0]


def test_throughput_speedup_passes_above_target():
    ours = scenario_summary(tok_s=280.0)
    base = scenario_summary(tok_s=200.0)
    entry = _evaluate("throughput_speedup_min", ours, base)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(1.4, abs=1e-3)


def test_throughput_speedup_fails_below_target():
    entry = _evaluate("throughput_speedup_min", scenario_summary(tok_s=210.0), scenario_summary(tok_s=200.0))
    assert entry["status"] == "fail", entry


def test_throughput_speedup_pending_without_baseline():
    entry = _evaluate("throughput_speedup_min", scenario_summary(), None)
    assert entry["status"] == "pending", entry


def test_throughput_speedup_invalid_on_a_different_trace():
    ours = scenario_summary()
    base = scenario_summary()
    base["protocol"]["trace_sha256"] = "trace-other"
    entry = _evaluate("throughput_speedup_min", ours, base)
    assert entry["status"] == "invalid", entry
    assert any("trace_sha256" in problem for problem in entry["problems"])


def test_throughput_speedup_invalid_when_a_protocol_key_is_missing():
    """A dropped key is not a pass: comparability is unproven, so no ratio (D5)."""
    base = scenario_summary()
    del base["protocol"]["trace_sha256"]
    entry = _evaluate("throughput_speedup_min", scenario_summary(), base)
    assert entry["status"] == "invalid", entry
    assert any("trace_sha256" in problem and "missing" in problem for problem in entry["problems"])


def test_throughput_speedup_invalid_on_a_different_pool_width():
    base = scenario_summary()
    base["protocol"]["max_batch"] = 32
    entry = _evaluate("throughput_speedup_min", scenario_summary(), base)
    assert entry["status"] == "invalid", entry
    assert any("max_batch" in problem for problem in entry["problems"])


def test_throughput_speedup_invalid_when_either_arm_is_not_saturated():
    """P-29: a non-saturated pair cannot show a scheduling speedup, whatever the ratio."""
    entry = _evaluate(
        "throughput_speedup_min",
        scenario_summary(saturated=False),
        scenario_summary(saturated=True),
    )
    assert entry["status"] == "invalid", entry


def test_ttft_decomposed_reads_the_structural_flag():
    passing = _evaluate("ttft_p99_decomposed", scenario_summary(decomposed=True), None)
    failing = _evaluate("ttft_p99_decomposed", scenario_summary(decomposed=False), None)
    assert passing["status"] == "pass", passing
    assert failing["status"] == "fail", failing


def test_ttft_decomposed_is_pending_without_the_block():
    entry = _evaluate("ttft_p99_decomposed", {"protocol": dict(SCENARIO_PROTOCOL)}, None)
    assert entry["status"] == "pending", entry


# ------------------------------------------------------------- W4 paged criteria
#
# C1 compares *allocator capacity* (fill-only max simultaneous admissions) under one total
# budget, so the pair is checked on device/dtype/workload/budget only -- ``max_batch`` is
# the variable under test and may differ. C2 judges internal frag only; reservation and
# seat-row slack are reported beside the value and must never flip the verdict.

W4_TABLE = {"W4": {"paged": {"max_concurrency_speedup_min": 1.50, "fragmentation_max": 0.05}}}
W4_PROTOCOL = {"dtype": "bfloat16", "device_class": "measure-cuda"}


def capacity_summary(
    peak=20,
    kv_total_mb=1606.0,
    *,
    max_batch=64,
    block_size=16,
    frag=None,
    reservation_slack=None,
    seat_row_slack=None,
    workload_sha256=None,
    protocol=None,
):
    summary = {
        "protocol": dict(W4_PROTOCOL if protocol is None else protocol, max_batch=max_batch),
        "capacity": {
            "peak_concurrency": peak,
            "kv_total_mb": kv_total_mb,
            "block_size": block_size,
            "max_batch": max_batch,
        },
    }
    if workload_sha256 is not None:
        summary["protocol"]["workload_sha256"] = workload_sha256
    for key, value in (
        ("frag_internal", frag),
        ("reservation_slack", reservation_slack),
        ("seat_row_slack", seat_row_slack),
    ):
        if value is not None:
            summary["capacity"][key] = value
    return summary


def _w4(name, ours, baseline, limit=None):
    if limit is None:
        limit = 1.50 if name == "max_concurrency_speedup_min" else 0.05
    table = {"W4": {"paged": {name: limit}}}
    return thresholds.evaluate([name], ours, table, baseline=baseline)[0]


def test_capacity_speedup_passes_exactly_at_the_target():
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0, max_batch=6, block_size=None)
    entry = _w4("max_concurrency_speedup_min", capacity_summary(peak=9), baseline)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(1.5)
    assert entry["ours_peak_concurrency"] == 9
    assert entry["baseline_peak_concurrency"] == 6


def test_capacity_speedup_fails_below_the_target():
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0)
    entry = _w4("max_concurrency_speedup_min", capacity_summary(peak=8), baseline)
    assert entry["status"] == "fail", entry
    assert entry["value"] == pytest.approx(8 / 6, abs=1e-3)


def test_capacity_speedup_allows_a_different_max_batch():
    """D6: the paged arm's pool is wider than the seat arm's by construction."""
    ours = capacity_summary(peak=20, max_batch=64)
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0, max_batch=6)
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(3.333, abs=1e-3)


def test_capacity_speedup_invalid_when_ours_exceeds_the_budget():
    ours = capacity_summary(peak=20, kv_total_mb=1700.0)
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0)
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("kv_total_mb" in problem for problem in entry["problems"])


def test_capacity_speedup_allows_ours_below_the_shared_budget():
    ours = capacity_summary(peak=20, kv_total_mb=1456.0)
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0)
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "pass", entry


def test_capacity_speedup_invalid_on_a_different_workload():
    ours = capacity_summary(peak=20, workload_sha256="w3-like-max2048")
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0, workload_sha256="uniform-2048")
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("workload_sha256" in problem for problem in entry["problems"])


def test_capacity_speedup_invalid_on_a_dtype_mismatch():
    ours = capacity_summary(peak=20)
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0, protocol={**W4_PROTOCOL, "dtype": "float32"})
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "invalid", entry
    assert any("dtype" in problem for problem in entry["problems"])


def test_capacity_speedup_pending_without_the_capacity_block():
    ours = capacity_summary(peak=None)
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0)
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "pending", entry
    assert "peak_concurrency" in entry["reason"]


def test_capacity_speedup_never_reads_the_dynamic_peak():
    """``throughput.peak_concurrency`` is E3's observed occupancy, not allocator capacity."""
    ours = {
        "protocol": dict(W4_PROTOCOL, max_batch=64),
        "throughput": {"peak_concurrency": 20},
    }
    baseline = capacity_summary(peak=6, kv_total_mb=1638.0)
    entry = _w4("max_concurrency_speedup_min", ours, baseline)
    assert entry["status"] == "pending", entry


def test_capacity_speedup_pending_without_a_baseline():
    entry = _w4("max_concurrency_speedup_min", capacity_summary(peak=20), None)
    assert entry["status"] == "pending", entry
    assert "baseline" in entry["reason"]


def test_fragmentation_passes_at_4_9_percent():
    entry = _w4("fragmentation_max", capacity_summary(frag=0.049), None)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(0.049)


def test_fragmentation_passes_exactly_at_5_percent():
    entry = _w4("fragmentation_max", capacity_summary(frag=0.05), None)
    assert entry["status"] == "pass", entry


def test_fragmentation_fails_at_5_1_percent():
    entry = _w4("fragmentation_max", capacity_summary(frag=0.051), None)
    assert entry["status"] == "fail", entry


def test_fragmentation_ignores_a_large_reservation_slack():
    """C2 is internal frag only: a 90% reservation slack must not fail a 1.4% frag run."""
    summary = capacity_summary(frag=0.014, reservation_slack=0.90, seat_row_slack=0.26)
    entry = _w4("fragmentation_max", summary, None)
    assert entry["status"] == "pass", entry
    assert entry["value"] == pytest.approx(0.014)
    assert entry["reservation_slack"] == pytest.approx(0.90)
    assert entry["seat_row_slack"] == pytest.approx(0.26)
    assert "not judged" in entry["reason"]


def test_fragmentation_falls_back_to_the_memory_block():
    summary = {"memory": {"frag_internal": 0.03, "reservation_slack": 0.2}}
    entry = _w4("fragmentation_max", summary, None)
    assert entry["status"] == "pass", entry
    assert entry["reservation_slack"] == pytest.approx(0.2)


def test_fragmentation_pending_without_a_value():
    entry = _w4("fragmentation_max", capacity_summary(), None)
    assert entry["status"] == "pending", entry
    assert "frag_internal" in entry["reason"]


def test_step_cost_ratio_is_record_only_never_a_pass_line():
    """C4 has no threshold key: the ratio may be echoed, but the status stays pending."""
    summary = {"steps": {"step_cost_ratio": {"median": 1.08}}}
    entry = thresholds.evaluate(["step_cost_ratio"], summary, W4_TABLE)[0]
    assert entry["status"] == "pending", entry
    assert entry["value"] == pytest.approx(1.08)
    assert "record-only" in entry["reason"]
    assert "1.5" in entry["reason"]
