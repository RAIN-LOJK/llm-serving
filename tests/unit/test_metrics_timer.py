"""Timer selection and aggregation: the parts of W2 that can be checked without a GPU."""

from __future__ import annotations

import pytest
import torch

from bench.metrics import (
    CudaEventTimer,
    Timer,
    make_timer,
    per_prompt_summary,
    repeat_stability,
    summarize,
)


def test_auto_timer_resolves_to_wall_clock_off_cuda():
    timer = make_timer("cpu", "auto")
    assert isinstance(timer, Timer)
    assert timer.kind == "perf_counter"


def test_wall_timer_measures_something_positive():
    timer = make_timer("cpu", "perf_counter")
    timer.start()
    elapsed = timer.stop()
    assert elapsed >= 0.0


def test_cuda_event_timer_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        make_timer("cpu", "cuda_event")
    with pytest.raises(RuntimeError, match="CUDA"):
        CudaEventTimer()


def test_auto_on_cuda_requests_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # 'auto' resolves from the device string, so this must ask for events and then refuse
    # loudly -- never quietly fall back to a wall clock.
    with pytest.raises(RuntimeError, match="CUDA"):
        make_timer("cuda", "auto")


def test_unknown_timer_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown timer"):
        make_timer("cpu", "wallclock")


def test_summarize_reports_n_median_p95_and_cv():
    stats = summarize([10.0, 12.0, 11.0, 13.0, 12.0])
    assert stats["n"] == 5
    assert stats["median"] == 12.0
    assert stats["p95"] == 13.0
    assert stats["cv"] is not None and stats["cv"] < 0.10


def test_summarize_ignores_missing_and_non_finite():
    stats = summarize([1.0, None, float("nan"), 3.0])
    assert stats["n"] == 2
    assert stats["median"] == 2.0


def test_summarize_empty_is_explicit():
    stats = summarize([])
    assert stats["n"] == 0
    assert stats["median"] is None


def _row(prompt, bucket, ttft, tpot=5.0, model="m", context=16):
    return {"model": model, "prompt_id": prompt, "bucket": bucket, "context": context,
            "ttft_ms": ttft, "tpot_ms": tpot}


def test_repeat_stability_does_not_mix_length_buckets():
    rows = [
        _row("s01", "short", 10.0), _row("s01", "short", 10.4),
        _row("l01", "long", 400.0, context=2048), _row("l01", "long", 404.0, context=2048),
    ]
    stability = repeat_stability(rows)

    # mixing them would give a ~1.9 CV; per bucket it is ~2%
    assert stability["short"]["ttft_ms_cv_median"] < 0.05
    assert stability["long"]["ttft_ms_cv_median"] < 0.05


def test_repeat_stability_flags_a_noisy_prompt():
    rows = [_row("s01", "short", 10.0), _row("s01", "short", 20.0)]
    stability = repeat_stability(rows)
    assert stability["short"]["ttft_ms_cv_median"] > 0.10
    assert stability["short"]["ttft_ms_cv_max"] > 0.10


def test_repeat_stability_needs_two_repeats():
    stability = repeat_stability([_row("s01", "short", 10.0)])
    assert stability["short"]["ttft_ms_cv_median"] is None


def test_per_prompt_summary_groups_by_prompt():
    rows = [_row("s01", "short", 10.0), _row("s01", "short", 12.0), _row("s02", "short", 20.0)]
    entries = per_prompt_summary(rows)

    assert [entry["prompt"] for entry in entries] == ["s01", "s02"]
    assert entries[0]["n"] == 2
    assert entries[0]["ttft_ms"]["median"] == 11.0
