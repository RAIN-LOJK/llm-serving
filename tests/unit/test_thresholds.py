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
