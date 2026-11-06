"""Resolve ``[accept].criteria`` against ``configs/thresholds.toml``.

Three rules this module exists to enforce:

* **CV is measured inside a bucket.** ``overall`` mixes 16-token and 2048-token prompts;
  its CV is dominated by the length spread, not by machine jitter, so a steady run would
  fail a CV criterion for the wrong reason.
* **Ratio criteria need the paired baseline.** ``decode_speed_vs_hf_min`` is
  ``median(tpot_baseline / tpot_ours)`` over the prompts both arms ran — matched on prompt
  id, not on a mixed median.
* **No data means pending, with the reason.** A criterion never passes by default.

A criterion name must exist as a key under some ``[Wx.y]`` section of the thresholds file.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or ROOT / "configs" / "thresholds.toml").open("rb") as handle:
        return tomllib.load(handle)


def _flatten(table: Mapping[str, Any], prefix: str = ""):
    """Yield (dotted section, value); ``[W2.bench]`` becomes section "W2.bench"."""
    for key, value in table.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _flatten(value, name)
        else:
            yield name, value


def lookup(name: str, thresholds: Mapping[str, Any]) -> tuple[str, Any] | None:
    """Find ``name`` at any depth; returns (section, value)."""
    for dotted, value in _flatten(thresholds):
        if dotted.rsplit(".", 1)[-1] == name:
            return dotted.rsplit(".", 1)[0], value
    return None


# --------------------------------------------------------------------------- measurements

def _median_of(summary: Mapping[str, Any], block: str, name: str) -> float | None:
    entry = (summary.get(block) or {}).get(name) or {}
    value = entry.get("median")
    return float(value) if value is not None else None


def _worst_bucket_cv(
    summary: Mapping[str, Any], metric: str
) -> tuple[float | None, str | None, float | None]:
    """Worst bucket's median per-prompt CV, plus that bucket's worst single prompt."""
    stability = summary.get("stability") or {}
    candidates = [
        (float(entry[f"{metric}_cv_median"]), bucket, entry.get(f"{metric}_cv_max"))
        for bucket, entry in stability.items()
        if entry.get(f"{metric}_cv_median") is not None
    ]
    if not candidates:
        return None, None, None
    value, bucket, worst_prompt = max(candidates, key=lambda item: item[0])
    return value, bucket, float(worst_prompt) if worst_prompt is not None else None


def _check_cv(summary: Mapping[str, Any], limit: float, metric: str) -> dict[str, Any]:
    value, bucket, worst_prompt = _worst_bucket_cv(summary, metric)
    if value is None:
        return {"status": "pending", "reason": f"no bucket has >=2 repeats for {metric}"}
    return {
        "status": "pass" if value <= limit else "fail",
        "value": round(value, 4),
        "target": limit,
        "bucket": bucket,
        "worst_prompt_cv": None if worst_prompt is None else round(worst_prompt, 4),
        "reason": f"worst bucket's median per-prompt {metric} CV over repeats",
    }


# Protocol keys that make any two arms comparable: same workload, same protocol,
# same kind of device. A cross-implementation ratio (our engine vs the HF arm) needs
# only these -- the HF arm has no in-house attention backend to record, so demanding
# ``attn_prefill`` equality there would mark every honest pairing invalid.
_COMMON_KEYS = (
    "max_new_tokens",
    "prompt_sha256",
    "dtype",
    "device_class",
)

# Protocol keys that make two *in-house* arms comparable: same engine, same kernels.
# Two self-implemented runs on different backends measure different engines; the ratio
# would be unattributable (the same reason dtype/device are checked).
_MECHANISM_KEYS = (
    "attn_prefill",
    "attn_decode",
)

# Scenario arms (W3): a throughput ratio is only a scheduling statement if both arms ran
# the same arrivals on the same pool width. ``policy`` is deliberately absent -- it is the
# variable under test.
_SCENARIO_KEYS = (
    "trace_sha256",
    "max_batch",
)


def _protocol_mismatches(
    summary: Mapping[str, Any], baseline: Mapping[str, Any], *, mechanism: bool = False
) -> list[str]:
    """A pair is only comparable if it ran the same workload on the same kind of device.

    ``mechanism=True`` additionally demands the same attention backend pair. Use it when
    both arms are this repo's engine (e.g. a refactor-invariance check); leave it off for
    a cross-implementation ratio against the HF arm, which reports no backend.
    """
    ours = summary.get("protocol") or {}
    theirs = baseline.get("protocol") or {}
    keys = _COMMON_KEYS + (_MECHANISM_KEYS if mechanism else ())
    problems = []
    for key in keys:
        if ours.get(key) != theirs.get(key):
            problems.append(f"{key}: {ours.get(key)!r} vs baseline {theirs.get(key)!r}")
    return problems


def _scenario_mismatches(
    summary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    """Extra pairing check for two scenario arms: same trace, same pool width (D5).

    A key missing from either arm is itself a mismatch: comparability is unproven, so the
    pair must be refused rather than read as a ratio.
    """
    ours = summary.get("protocol") or {}
    theirs = baseline.get("protocol") or {}
    problems = []
    for key in _SCENARIO_KEYS:
        if ours.get(key) is None or theirs.get(key) is None:
            problems.append(
                f"{key}: missing from an arm "
                f"({ours.get(key)!r} vs baseline {theirs.get(key)!r})"
            )
        elif ours.get(key) != theirs.get(key):
            problems.append(f"{key}: {ours.get(key)!r} vs baseline {theirs.get(key)!r}")
    return problems


def _is_inhouse(value: Any) -> bool:
    """True for a backend name this repo ran itself (``sdpa``, ``hand_baddbmm``...).

    The HF arm records ``hf:<transformers impl>``: a different implementation class, not a
    kernel this repo can swap, so a cross-implementation pairing must not demand equality.
    """
    return bool(value) and not str(value).startswith("hf:")


def _reports_mechanism(protocol: Mapping[str, Any] | None) -> bool:
    """True when the arm's provenance describes an in-house attention backend pair."""
    protocol = protocol or {}
    return _is_inhouse(protocol.get("attn_prefill")) or _is_inhouse(protocol.get("attn_decode"))


def mechanism_mismatches(
    summary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    """Mechanism-layer pairing check: common keys **and** the same in-house backends.

    This is the stricter layer. ``decode_speed_vs_hf_min`` needs only the common keys
    against the HF arm, but two self-implemented runs on different kernels measure
    different engines and must not be paired. Call this (or pass ``mechanism=True`` to
    ``_decode_ratio``) before trusting a ratio between two of this repo's runs.
    """
    return _protocol_mismatches(summary, baseline, mechanism=True)


def _decode_ratio(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    mechanism: bool | None = None,
) -> tuple[float | None, int, list[str]]:
    """median(tpot_baseline / tpot_ours) over the prompts both arms ran.

    Any prompt whose output length differs makes the pair incomparable: TPOT over 40 tokens
    and TPOT over 60 are different quantities.

    ``mechanism=None`` (the default) applies the mechanism keys exactly when the baseline
    reports an in-house backend: an HF-arm baseline records ``hf:...`` and is checked on the
    common keys only, while two self-implemented runs must agree on
    ``attn_prefill``/``attn_decode``.
    """
    if not baseline:
        return None, 0, []
    if mechanism is None:
        mechanism = _reports_mechanism(baseline.get("protocol"))
    problems = _protocol_mismatches(summary, baseline, mechanism=mechanism)
    ours = {(entry.get("model"), entry.get("prompt")): entry for entry in summary.get("by_prompt") or []}
    theirs = {(entry.get("model"), entry.get("prompt")): entry for entry in baseline.get("by_prompt") or []}
    ratios = []
    for key, entry in ours.items():
        other = theirs.get(key)
        if other is None:
            continue
        mine = (entry.get("tpot_ms") or {}).get("median")
        reference = (other.get("tpot_ms") or {}).get("median")
        mine_out = (entry.get("out_tokens") or {}).get("median")
        ref_out = (other.get("out_tokens") or {}).get("median")
        if mine_out is not None and ref_out is not None and int(mine_out) != int(ref_out):
            problems.append(f"prompt {key[1]}: out_tokens {int(mine_out)} vs {int(ref_out)}")
        if mine and reference and mine > 0:
            ratios.append(float(reference) / float(mine))
    return (statistics.median(ratios) if ratios else None), len(ratios), problems


# ------------------------------------------------------------------------ W4 capacity (D6)

#: A capacity pairing fixes the device class and dtype. ``max_batch`` is deliberately
#: absent: D6 lets the paged arm run a much wider pool than the seat arm, and the two
#: arms are compared on allocator capacity, not on pool width. ``kv_total_mb`` is the
#: shared total budget (KV + workspace); the baseline reports it as the upper bound and
#: our arm must fit inside it.
_CAPACITY_KEYS = ("dtype", "device_class")


def _block(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return summary.get(name) or {}


def _ledger_value(summary: Mapping[str, Any], name: str) -> Any:
    """``capacity.<name>`` with a ``memory.<name>`` fallback (both blocks are planned)."""
    value = _block(summary, "capacity").get(name)
    if value is None:
        value = _block(summary, "memory").get(name)
    return value


def _peak_concurrency(summary: Mapping[str, Any]) -> Any:
    """Allocator capacity: the max simultaneous admissions of a fill-only run.

    Never the dynamic peak in ``throughput.peak_concurrency`` -- that one is E3's
    observed occupancy and answers a different question (D6).
    """
    return _block(summary, "capacity").get("peak_concurrency")


def _workload_id(summary: Mapping[str, Any]) -> Any:
    """Length-distribution identity; optional, so both arms may legitimately omit it."""
    protocol = summary.get("protocol") or {}
    value = protocol.get("workload_sha256")
    return value if value is not None else _block(summary, "capacity").get("workload_sha256")


def _capacity_mismatches(summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Pairing check for C1: same device/dtype, same workload, same total budget.

    A missing ``dtype``/``device_class`` proves nothing, so it is itself a mismatch. The
    budget check only compares the two ``kv_total_mb`` values when both arms report one:
    ours must not exceed the baseline's shared upper bound (equal is fine). ``max_batch``
    is never compared here -- D6 allows the paged arm a wider pool.
    """
    ours = summary.get("protocol") or {}
    theirs = baseline.get("protocol") or {}
    problems: list[str] = []
    for key in _CAPACITY_KEYS:
        mine, reference = ours.get(key), theirs.get(key)
        if mine is None or reference is None:
            problems.append(f"{key}: missing from an arm ({mine!r} vs baseline {reference!r})")
        elif mine != reference:
            problems.append(f"{key}: {mine!r} vs baseline {reference!r}")

    mine_id, reference_id = _workload_id(summary), _workload_id(baseline)
    if mine_id is not None or reference_id is not None:
        # Reported by one arm only is still a mismatch: comparability is unproven.
        if mine_id != reference_id:
            problems.append(f"workload_sha256: {mine_id!r} vs baseline {reference_id!r}")

    mine_total = _ledger_value(summary, "kv_total_mb")
    reference_total = _ledger_value(baseline, "kv_total_mb")
    if mine_total is not None and reference_total is not None:
        try:
            if float(mine_total) > float(reference_total):
                problems.append(
                    f"kv_total_mb: {mine_total!r} exceeds baseline {reference_total!r} "
                    "(both arms must fit the same total budget)"
                )
        except (TypeError, ValueError):
            problems.append(f"kv_total_mb: {mine_total!r} vs baseline {reference_total!r}")
    return problems


def _check_capacity_speedup(
    summary: Mapping[str, Any], baseline: Mapping[str, Any] | None, limit: float
) -> dict[str, Any]:
    """C1: fill-only allocator capacity, ours / baseline, under one total budget."""
    if not baseline:
        return {"status": "pending", "reason": "no paired baseline summary"}
    mine, reference = _peak_concurrency(summary), _peak_concurrency(baseline)
    if mine is None or reference is None:
        return {"status": "pending", "reason": "capacity.peak_concurrency missing from an arm"}

    problems = _capacity_mismatches(summary, baseline)
    if problems:
        return {
            "status": "invalid",
            "problems": problems[:6],
            "reason": "arms are not comparable; fix the pair before reading a capacity ratio",
        }
    try:
        mine_value, reference_value = float(mine), float(reference)
    except (TypeError, ValueError):
        return {"status": "pending", "reason": "capacity.peak_concurrency is not numeric"}
    if reference_value <= 0.0:
        return {"status": "pending", "reason": "baseline capacity.peak_concurrency is not positive"}

    ratio = mine_value / reference_value
    entry: dict[str, Any] = {
        "status": "pass" if ratio >= limit else "fail",
        "value": round(ratio, 3),
        "target": limit,
        "ours_peak_concurrency": mine_value,
        "baseline_peak_concurrency": reference_value,
        "reason": "fill-only allocator capacity (peak_concurrency) ours / baseline "
        "under the same total budget; max_batch may differ",
    }
    for key, value in (
        ("kv_total_mb", _ledger_value(summary, "kv_total_mb")),
        ("baseline_kv_total_mb", _ledger_value(baseline, "kv_total_mb")),
        ("block_size", _ledger_value(summary, "block_size")),
    ):
        if value is not None:
            entry[key] = value
    return entry


def _check_fragmentation(summary: Mapping[str, Any], limit: float) -> dict[str, Any]:
    """C2: judge **internal** frag only. The two slack measures are reported, not judged."""
    raw = _ledger_value(summary, "frag_internal")
    if raw is None:
        return {
            "status": "pending",
            "reason": "summary has no capacity.frag_internal (or memory.frag_internal)",
        }
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return {"status": "pending", "reason": "frag_internal is not numeric"}

    entry: dict[str, Any] = {
        "status": "pass" if value <= limit else "fail",
        "value": round(value, 4),
        "target": limit,
        "reason": "internal frag only (unwritten tail of the last block); "
        "reservation slack / seat row slack reported, not judged",
    }
    for key in ("reservation_slack", "seat_row_slack"):
        slack = _ledger_value(summary, key)
        if slack is None:
            continue
        try:
            entry[key] = round(float(slack), 4)
        except (TypeError, ValueError):
            entry[key] = slack
    return entry


def _step_cost_ratio(summary: Mapping[str, Any]) -> float | None:
    """C4's B=8 matched-``lpad`` step ratio, when a summary happens to carry it."""
    for container in (summary, _block(summary, "steps")):
        raw = container.get("step_cost_ratio") if isinstance(container, Mapping) else None
        if isinstance(raw, Mapping):
            raw = raw.get("median", raw.get("value"))
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


def _memory_ratio(summary: Mapping[str, Any]) -> tuple[float | None, float | None, list[dict[str, Any]]]:
    """Share of the observed peak that the formula account explains, per context.

    ``peak`` is the allocator peak (torch). The account is ``weights + KV (+ state)``; what
    it misses is activation workspace (the T^2 attention matrix) plus allocator slack. That
    miss is a real property of the implementation, so it is reported per bucket instead of
    being averaged into one number that hides the long-context rows.
    """
    per_bucket: list[dict[str, Any]] = []
    for group in summary.get("groups") or []:
        median = lambda name: ((group.get(name) or {}).get("median"))  # noqa: E731
        weights, kv, peak = median("weights_mb"), median("kv_mb"), median("peak_mem_mb")
        if weights is None or kv is None or not peak:
            continue
        state = median("state_mb") or 0.0
        conv = median("conv_mb") or 0.0
        per_bucket.append(
            {
                "context": group.get("context"),
                "ratio": (weights + kv + state + conv) / peak,
                "peak_mb": peak,
            }
        )
    if not per_bucket:
        return None, None, []

    ratio = statistics.median(item["ratio"] for item in per_bucket)
    peak = _median_of(summary, "overall", "peak_mem_mb")
    driver = _median_of(summary, "overall", "driver_mem_mb")
    peak_over_driver = peak / driver if peak and driver else None
    return ratio, peak_over_driver, per_bucket


# ------------------------------------------------------------------------------ evaluate

def evaluate(
    criteria: Sequence[str],
    summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One entry per criterion; ``status`` is pass / fail / pending."""
    results: list[dict[str, Any]] = []
    for raw in criteria:
        name = str(raw)
        if name == "step_cost_ratio":
            # W4 C4 is record-only and has no threshold key: report the ratio if the
            # summary carries it, but never turn it into a pass/fail line.
            entry: dict[str, Any] = {
                "name": name,
                "status": "pending",
                "reason": "C4 is record-only (no pass line); >1.5x retracts the throughput claim",
            }
            ratio = _step_cost_ratio(summary)
            if ratio is not None:
                entry["value"] = round(ratio, 3)
            results.append(entry)
            continue
        found = lookup(name, thresholds)
        if found is None:
            results.append({"name": name, "status": "pending", "reason": "no such threshold key"})
            continue
        section, limit = found
        entry: dict[str, Any] = {"name": name, "section": section}

        if name.endswith("_cv_max"):
            metric = name[: -len("_cv_max")] + "_ms"
            entry.update(_check_cv(summary, float(limit), metric))
        elif name == "memory_match_ratio":
            ratio, peak_over_driver, per_bucket = _memory_ratio(summary)
            if ratio is None:
                entry.update({"status": "pending", "reason": "peak/weights/KV missing from the summary"})
            else:
                entry.update(
                    {
                        "status": "pass" if ratio >= float(limit) else "fail",
                        "value": round(ratio, 3),
                        "target": float(limit),
                        "peak_over_driver": None if peak_over_driver is None else round(peak_over_driver, 3),
                        "by_context": [
                            {"context": item["context"], "ratio": round(item["ratio"], 3)}
                            for item in per_bucket
                        ],
                        "reason": "weights+KV share of allocator peak, judged on the median context "
                        "(driver ratio reported, not judged)",
                    }
                )
        elif name == "decode_speed_vs_hf_min":
            ratio, n, problems = _decode_ratio(summary, baseline)
            if problems:
                entry.update(
                    {
                        "status": "invalid",
                        "problems": problems[:6],
                        "reason": "arms are not comparable; fix the pair before reading a ratio",
                    }
                )
            elif ratio is None:
                entry.update({"status": "pending", "reason": "no paired baseline summary with matching prompts"})
            else:
                entry.update(
                    {
                        "status": "pass" if ratio >= float(limit) else "fail",
                        "value": round(ratio, 3),
                        "target": float(limit),
                        "n_prompts": n,
                        "reason": "median per-prompt baseline TPOT / ours",
                    }
                )
        elif name == "throughput_speedup_min":
            # Scenario arms: tok/s on the wall-clock window (D5). The pair must share a
            # trace and a pool width, otherwise the ratio is a workload difference.
            if not baseline:
                entry.update({"status": "pending", "reason": "no paired baseline summary"})
            else:
                problems = _protocol_mismatches(summary, baseline, mechanism=True)
                problems += _scenario_mismatches(summary, baseline)
                ours_bps = summary.get("throughput") or {}
                theirs_bps = baseline.get("throughput") or {}
                if problems:
                    entry.update(
                        {
                            "status": "invalid",
                            "problems": problems[:6],
                            "reason": "arms are not comparable; fix the pair before reading a ratio",
                        }
                    )
                elif not ours_bps.get("saturated") or not theirs_bps.get("saturated"):
                    entry.update(
                        {
                            "status": "invalid",
                            "reason": "both arms must be saturated "
                            "(queue_depth_at_last_arrival > 0: queued when the last request arrived); "
                            "a non-saturated pair cannot "
                            "show a throughput speedup",
                        }
                    )
                else:
                    mine = (ours_bps.get("tok_s") or {}).get("median")
                    reference = (theirs_bps.get("tok_s") or {}).get("median")
                    if not mine or not reference:
                        entry.update({"status": "pending", "reason": "tok_s missing from a summary"})
                    else:
                        speedup = float(mine) / float(reference)
                        entry.update(
                            {
                                "status": "pass" if speedup >= float(limit) else "fail",
                                "value": round(speedup, 3),
                                "target": float(limit),
                                "ours_tok_s": round(float(mine), 3),
                                "baseline_tok_s": round(float(reference), 3),
                                "reason": "continuous tok/s / static tok/s on the same trace",
                            }
                        )
        elif name == "ttft_p99_decomposed":
            latency = summary.get("latency") or {}
            decomposed = latency.get("decomposed")
            if decomposed is None:
                entry.update({"status": "pending", "reason": "summary has no latency.decomposed"})
            else:
                entry.update(
                    {
                        "status": "pass" if bool(decomposed) == bool(limit) else "fail",
                        "value": bool(decomposed),
                        "target": bool(limit),
                        "reason": "ttft == queue + prefill within 1 ms on >= 99% of requests",
                    }
                )
        elif name == "max_concurrency_speedup_min":
            # W4 C1: fill-only allocator capacity under one total budget (D6). Paired on
            # device/dtype/workload/budget, deliberately *not* on ``max_batch``.
            entry.update(_check_capacity_speedup(summary, baseline, float(limit)))
        elif name == "fragmentation_max":
            # W4 C2: internal frag only; the direction is ``<= limit``.
            entry.update(_check_fragmentation(summary, float(limit)))
        else:
            entry.update({"status": "pending", "reason": "no evaluator for this criterion yet"})
        results.append(entry)
    return results
