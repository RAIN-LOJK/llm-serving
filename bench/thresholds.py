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

import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _number(value: Any) -> float | None:
    """Finite float, or ``None`` -- a missing/malformed measurement is never a zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


# ------------------------------------------------------------------------ W4.5 (D7 draft)
#
# The packed-prefill and decode-kernel criteria are *paired*: they compare two arms of
# this repo that ran the same trace on the same pool width. The attention backend is the
# variable under test, so these checks deliberately use the common + scenario keys only;
# ``mechanism=True`` would mark every honest packed-vs-control-arm pair invalid. The
# serial control arm is deleted from production (D7); its historical summaries remain
# valid as the E3 baseline.


def _scenario_pair_problems(
    summary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    return _protocol_mismatches(summary, baseline, mechanism=False) + _scenario_mismatches(
        summary, baseline
    )


def _latency_percentile(
    summary: Mapping[str, Any], metric: str, percentile: str
) -> float | None:
    block = (summary.get("latency") or {}).get(metric) or {}
    return _number(block.get(percentile))


def _check_decode_step_ratio(
    summary: Mapping[str, Any], baseline: Mapping[str, Any] | None, limit: float
) -> dict[str, Any]:
    """C4: step_ms ratio at the largest common decode-only ``active`` bucket.

    D6 defines C4 as the steady-state decode cost at the run's full batch. Averaging every
    common ``active`` bucket would charge the kernel for admission/prefill steps (W4.5's
    ``prefill_groups`` marks them) and for ``active=0`` idle steps, so the largest common
    bucket whose prefill fraction is negligible (<= 0.1 groups/step; E2's active=8 bucket
    has 5 prefill steps among 625 decode steps) is selected. If neither arm reports
    ``prefill_groups`` (historical rows) the largest common bucket is used as-is.

    The decode backend is the variable under test, so the pair is checked on the common
    workload/device keys and the scenario keys only (mechanism=False). Buckets are matched
    on ``active``, never on position: a run with a different step count would otherwise
    compare unrelated steps.
    """
    if not baseline:
        return {"status": "pending", "reason": "no paired baseline summary"}
    problems = _scenario_pair_problems(summary, baseline)
    if problems:
        return {
            "status": "invalid",
            "problems": problems[:6],
            "reason": "arms are not comparable; fix the pair before reading a decode ratio",
        }

    ours = (summary.get("steps") or {}).get("by_active") or []
    theirs = (baseline.get("steps") or {}).get("by_active") or []
    mine_by_active = {
        entry.get("active"): entry for entry in ours if isinstance(entry, Mapping)
    }
    ref_by_active = {
        entry.get("active"): entry for entry in theirs if isinstance(entry, Mapping)
    }

    def pure_decode(entry: Mapping[str, Any]) -> bool:
        # ``prefill_groups`` is additive (W4.5); a historical row without it is a
        # decode-only bucket by construction. A steady-state bucket may still contain a
        # handful of prefill steps (one per repeat), so the test is "prefill is not a
        # material fraction", not "exactly zero": E2's active=8 bucket has 5 prefill
        # steps among 625 decode steps (mean 0.008).
        groups = _number(entry.get("prefill_groups"))
        return groups is None or groups <= 0.1

    candidates = [
        active
        for active, entry in mine_by_active.items()
        if active in ref_by_active
        and pure_decode(entry)
        and pure_decode(ref_by_active[active])
        and _number(entry.get("step_ms")) is not None
        and _number(ref_by_active[active].get("step_ms")) is not None
    ]
    if not candidates:
        candidates = [
            active
            for active, entry in mine_by_active.items()
            if active in ref_by_active
            and _number(entry.get("step_ms")) is not None
            and _number(ref_by_active[active].get("step_ms")) is not None
        ]
    if not candidates:
        return {
            "status": "pending",
            "reason": "no active bucket has step_ms on both arms",
        }
    # C4 is the steady-state decode cost at the run's full batch. The earlier buckets
    # include admissions/prefill steps (W4.5's ``prefill_groups`` marks them), and the
    # active=0 bucket is not a decode step at all; averaging them in would charge the
    # kernel for the scheduler's prefill.
    active = max(candidates, key=lambda value: float(value))
    mine = _number(mine_by_active[active].get("step_ms"))
    reference = _number(ref_by_active[active].get("step_ms"))
    if mine is None or reference is None or reference <= 0.0:
        return {"status": "pending", "reason": "selected active bucket has no positive step_ms"}
    ratio = mine / reference
    return {
        "status": "pass" if ratio <= limit else "fail",
        "value": round(ratio, 3),
        "target": limit,
        "active_bucket": active,
        "ours_step_ms": round(mine, 3),
        "baseline_step_ms": round(reference, 3),
        "reason": "steady-state decode step_ms at the largest common active bucket "
        "(prefill steps excluded), ours / baseline; decode kernel is the variable",
    }


def _check_ttft_ratio(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    limit: float,
    *,
    percentile: str,
    mode: str,
) -> dict[str, Any]:
    """E3: paired TTFT p99 speedup / p50 ratio, packed vs the historical serial arm.

    ``mode="speedup"`` is ``baseline / ours >= limit`` (the packed arm must be faster);
    ``mode="ratio"`` is ``ours / baseline <= limit`` (the packed arm must not regress the
    median). Both arms must share the trace and pool width; the prefill mode is the
    variable and is therefore not part of the pairing keys. The serial control arm was
    deleted from production after W4.5 closed (D7); this evaluator now only reads the
    historical baseline summaries under ``results/W45/``.
    """
    if not baseline:
        return {"status": "pending", "reason": "no paired baseline summary"}
    problems = _scenario_pair_problems(summary, baseline)
    if problems:
        return {
            "status": "invalid",
            "problems": problems[:6],
            "reason": "arms are not comparable; fix the pair before reading a TTFT ratio",
        }

    metric = "ttft_ms"
    mine = _latency_percentile(summary, metric, percentile)
    reference = _latency_percentile(baseline, metric, percentile)
    if mine is None or reference is None:
        return {
            "status": "pending",
            "reason": f"latency.{metric}.{percentile} missing from an arm",
        }
    if reference <= 0.0 or mine <= 0.0:
        return {
            "status": "pending",
            "reason": f"latency.{metric}.{percentile} is not positive",
        }

    if mode == "speedup":
        value = reference / mine
        passed = value >= limit
        reason = "baseline / ours ttft p99 (packed prefill must be faster)"
    else:
        value = mine / reference
        passed = value <= limit
        reason = "ours / baseline ttft p50 (packed prefill must not regress the median)"
    return {
        "status": "pass" if passed else "fail",
        "value": round(value, 3),
        "target": limit,
        f"ours_{percentile}": mine,
        f"baseline_{percentile}": reference,
        "reason": reason,
    }


def _check_golden_token_match(
    summary: Mapping[str, Any], limit: float
) -> dict[str, Any]:
    """E0: the runner's golden comparison, judged by token-id match rate."""
    block = summary.get("golden")
    if not isinstance(block, Mapping):
        return {"status": "pending", "reason": "summary has no golden comparison block"}
    if block.get("error"):
        return {
            "status": "pending",
            "reason": f"golden comparison unavailable: {block.get('error')}",
        }
    rate = _number(block.get("match_rate"))
    if rate is None:
        return {"status": "pending", "reason": "golden.match_rate missing"}
    entry: dict[str, Any] = {
        "status": "pass" if rate >= limit else "fail",
        "value": round(rate, 4),
        "target": limit,
        "reason": "exact token-id prefix vs tests/golden (E0; never regenerated here)",
    }
    for key in ("matched_ids", "total_ids", "matched_prompts", "total_prompts", "path"):
        if block.get(key) is not None:
            entry[key] = block[key]
    return entry


def _check_paged_vs_seat_token_match(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    limit: float,
) -> dict[str, Any]:
    """C3: per-prompt ``token_sha256`` agreement on the ids both arms actually ran."""
    if not baseline:
        return {"status": "pending", "reason": "no paired baseline summary"}
    ours = summary.get("tokens")
    theirs = baseline.get("tokens")
    if not isinstance(ours, Mapping) or not isinstance(theirs, Mapping):
        return {"status": "pending", "reason": "summary.tokens missing from an arm"}
    if not ours or not theirs:
        return {"status": "pending", "reason": "summary.tokens is empty on an arm"}
    common = sorted(set(ours) & set(theirs))
    if not common:
        return {
            "status": "pending",
            "reason": "the two token maps share no prompt ids",
        }

    mismatched = [
        key
        for key in common
        if ours.get(key) is None
        or theirs.get(key) is None
        or ours.get(key) != theirs.get(key)
    ]
    rate = (len(common) - len(mismatched)) / len(common)
    entry: dict[str, Any] = {
        "status": "pass" if rate >= limit else "fail",
        "value": round(rate, 4),
        "target": limit,
        "n_prompts": len(common),
        "matched_prompts": len(common) - len(mismatched),
        "reason": "per-prompt token_sha256 agreement on the shared prompt ids",
    }
    if mismatched:
        entry["mismatched_prompt_ids"] = mismatched[:20]
    return entry


def _step_cost_ratio(summary: Mapping[str, Any]) -> float | None:
    """C4's steady-state decode step ratio, when a summary happens to carry it."""
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


# ------------------------------------------------------------------------ W7 (D9)
#
# W7 pairs a hybrid arm (Qwen3.5-0.8B, six full-attention layers + 18 linear-state
# layers) with a pure 0.6B arm under the same 1640 MiB state budget. The pairing is
# *not* the W4.5 mechanism pairing: the two arms run different models and different
# attention backends, so only the protocol keys (dtype/device/max_new_tokens) and the
# workload geometry (synthetic context + fill/capacity basis) may be compared.
#
# Cross-model synthetic traces share ``trace_sha256`` after W7's cap: ``bench/run.py``
# draws synthetic ids below ``min(model.vocab_size, 151936)``, so the 0.6B (151936) and
# 0.8B (248320) arms see identical ids for the same contexts/seed. The evaluators still
# pair on the recorded context/budget *and* require the hash to match; a mismatch is a
# workload difference, not a capacity result.

_W7_TOTAL_BUDGET_MB = 1640.0
_W7_CAPACITY_CONTEXTS = {
    "capacity_speedup_512_min": 512,
    "capacity_speedup_2k_min": 2048,
    "capacity_speedup_8k_min": 8192,
}


def _w7_capacity(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return _block(summary, "capacity")


def _w7_nominal_context(group: Mapping[str, Any]) -> int | None:
    """The prompt context a group belongs to: ``context - out_tokens``.

    ``bench.scenario`` keys a group by ``prompt + generated`` (D5/P-30), not by the
    prompt alone. For a fill probe ``out_tokens`` is 0; for a scenario run the group's
    own median output length recovers the prompt context without guessing a bucket.
    """
    context = _number(group.get("context"))
    out = _number((group.get("out_tokens") or {}).get("median"))
    if context is None or out is None:
        return None
    return int(round(context - out))


def _w7_single_context(summary: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """The one context a fill probe ran at, or a reason it cannot be established."""
    raw = _w7_capacity(summary).get("context_tokens")
    if raw is not None:
        value = _number(raw)
        if value is None or value <= 0 or not float(value).is_integer():
            return None, "capacity.context_tokens is not a positive integer"
        return int(value), None
    groups = summary.get("groups") or []
    contexts = sorted(
        {
            context
            for context in (_w7_nominal_context(group) for group in groups)
            if context is not None
        }
    )
    if not contexts:
        return None, (
            "summary records neither capacity.context_tokens nor a groups[] entry "
            "with context - out_tokens"
        )
    if len(contexts) > 1:
        return None, (
            f"summary mixes prompt contexts {contexts}; a single-context fill probe is "
            "required for a capacity/P1 pairing"
        )
    return contexts[0], None


def _w7_weighted_median(values: Sequence[tuple[float, int]]) -> float:
    ordered = sorted(values)
    total = sum(weight for _, weight in ordered)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative * 2 >= total:
            return value
    return ordered[-1][0]


def _w7_group_series(
    summary: Mapping[str, Any], context: int, metric: str
) -> tuple[float | None, int | None, int, str | None]:
    """``(weighted median, batch, n_groups, reason)`` for one prompt context."""
    selected: list[tuple[float, int, int]] = []
    for group in summary.get("groups") or []:
        if _w7_nominal_context(group) != int(context):
            continue
        value = _number((group.get(metric) or {}).get("median"))
        if value is None:
            continue
        try:
            batch = int(group.get("batch") or 1)
        except (TypeError, ValueError):
            batch = 1
        selected.append((value, max(1, int(group.get("n") or 1)), batch))
    if not selected:
        return None, None, 0, f"summary has no groups[] entry for prompt context {context}"
    # Scenario request rows do not carry a per-request ``batch`` (the run's width is
    # ``[engine].max_batch``), so ``bench.scenario`` keys a scenario group with batch=1.
    # The protocol's ``max_batch`` is the authoritative width; only fall back to the
    # group field when the summary has no protocol block.
    protocol_batch: int | None = None
    raw_batch = (summary.get("protocol") or {}).get("max_batch")
    if isinstance(raw_batch, int) and not isinstance(raw_batch, bool) and raw_batch >= 1:
        protocol_batch = raw_batch
    batches = sorted({batch for _, _, batch in selected})
    if protocol_batch is None and len(batches) != 1:
        return None, None, len(selected), (
            f"prompt context {context} is spread over batches {batches}; one run must "
            "measure one batch"
        )
    value = _w7_weighted_median([(value, weight) for value, weight, _ in selected])
    return value, protocol_batch if protocol_batch is not None else batches[0], len(selected), None


def _w7_pair_problems(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    require_batch: bool,
) -> list[str]:
    """Protocol-level pairing for W7: same dtype/device/protocol, same pool width."""
    problems = _protocol_mismatches(summary, baseline, mechanism=False)
    ours_hash = (summary.get("protocol") or {}).get("trace_sha256")
    theirs_hash = (baseline.get("protocol") or {}).get("trace_sha256")
    if ours_hash is None or theirs_hash is None:
        problems.append(
            f"trace_sha256: missing from an arm ({ours_hash!r} vs baseline {theirs_hash!r})"
        )
    elif ours_hash != theirs_hash:
        problems.append(f"trace_sha256: {ours_hash!r} vs baseline {theirs_hash!r}")
    if require_batch:
        ours = (summary.get("protocol") or {}).get("max_batch")
        theirs = (baseline.get("protocol") or {}).get("max_batch")
        if ours is None or theirs is None:
            problems.append(f"max_batch: missing from an arm ({ours!r} vs baseline {theirs!r})")
        elif ours != theirs:
            problems.append(f"max_batch: {ours!r} vs baseline {theirs!r}")
    return problems


def _w7_state_budget(
    summary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """``(missing, over_budget)`` for both fill arms against the 1640 MiB state budget.

    A missing ledger is a data gap (pending); an explicit value above the budget is an
    invalid run (D9-8 / plan §6.4), never silently clamped.
    """
    missing: list[str] = []
    over: list[str] = []
    for label, arm in (("ours", summary), ("baseline", baseline)):
        value = _number(_ledger_value(arm, "state_total_mb"))
        if value is None:
            missing.append(
                f"state_total_mb: missing from {label} "
                "(capacity.state_total_mb is required for the W7 budget check)"
            )
        elif value > _W7_TOTAL_BUDGET_MB:
            over.append(
                f"state_total_mb: {label} {value:.3f} MiB > {_W7_TOTAL_BUDGET_MB:.0f} MiB"
            )
    return missing, over


def _check_w7_capacity_speedup(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    limit: float,
    context: int,
) -> dict[str, Any]:
    """E1/E2: hybrid/pure ``peak_concurrency`` at one pre-registered fill context."""
    if not baseline:
        return {"status": "pending", "reason": "no paired pure 0.6B fill summary"}
    ours_ctx, ours_error = _w7_single_context(summary)
    ref_ctx, ref_error = _w7_single_context(baseline)
    if ours_error or ref_error:
        return {
            "status": "pending",
            "reason": "; ".join(
                part
                for part in (
                    f"ours: {ours_error}" if ours_error else "",
                    f"baseline: {ref_error}" if ref_error else "",
                )
                if part
            ),
        }
    problems = _w7_pair_problems(summary, baseline, require_batch=True)
    if ours_ctx != int(context) or ref_ctx != int(context):
        problems.append(
            f"context_tokens: ours {ours_ctx!r}, baseline {ref_ctx!r}, criterion {context}"
        )
    if problems:
        return {
            "status": "invalid",
            "problems": problems[:6],
            "reason": "fill arms are not the same context/device/dtype pair",
        }
    missing_budget, over_budget = _w7_state_budget(summary, baseline)
    if missing_budget:
        return {"status": "pending", "reason": "; ".join(missing_budget)}
    if over_budget:
        return {
            "status": "invalid",
            "problems": over_budget[:6],
            "reason": "a fill arm is over the 1640 MiB state budget; do not judge a ratio",
        }

    mine, reference = _peak_concurrency(summary), _peak_concurrency(baseline)
    if mine is None or reference is None:
        return {"status": "pending", "reason": "capacity.peak_concurrency missing from an arm"}
    try:
        mine_value, reference_value = float(mine), float(reference)
    except (TypeError, ValueError):
        return {"status": "pending", "reason": "capacity.peak_concurrency is not numeric"}
    if reference_value <= 0.0:
        return {"status": "pending", "reason": "baseline peak_concurrency is not positive"}

    ratio = mine_value / reference_value
    entry: dict[str, Any] = {
        "status": "pass" if ratio >= limit else "fail",
        "value": round(ratio, 3),
        "target": limit,
        "context_tokens": int(context),
        "ours_peak_concurrency": mine_value,
        "baseline_peak_concurrency": reference_value,
        "state_total_mb": _ledger_value(summary, "state_total_mb"),
        "baseline_state_total_mb": _ledger_value(baseline, "state_total_mb"),
        "budget_mb": _W7_TOTAL_BUDGET_MB,
        "reason": "fill-only allocator capacity at the named context; hybrid / pure, "
        "both arms inside the same 1640 MiB state budget",
    }
    return entry


def _check_w7_p1_state_ratio(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    limit: float,
) -> dict[str, Any]:
    """P1: per-sequence state at 8k, hybrid vs pure. Ratio is pure / hybrid."""
    if not baseline:
        return {"status": "pending", "reason": "no paired pure 0.6B fill summary"}
    ours_ctx, ours_error = _w7_single_context(summary)
    ref_ctx, ref_error = _w7_single_context(baseline)
    if ours_error or ref_error:
        return {
            "status": "pending",
            "reason": "; ".join(
                part
                for part in (
                    f"ours: {ours_error}" if ours_error else "",
                    f"baseline: {ref_error}" if ref_error else "",
                )
                if part
            ),
        }
    problems = _w7_pair_problems(summary, baseline, require_batch=True)
    if ours_ctx != 8192 or ref_ctx != 8192:
        problems.append(
            f"context_tokens: ours {ours_ctx!r}, baseline {ref_ctx!r}, P1 endpoint is 8192"
        )
    if problems:
        return {
            "status": "invalid",
            "problems": problems[:6],
            "reason": "P1 requires the 8k fill pair on the same device/dtype",
        }

    ours_peak, ref_peak = _peak_concurrency(summary), _peak_concurrency(baseline)
    if ours_peak is None or ref_peak is None:
        return {"status": "pending", "reason": "capacity.peak_concurrency missing from an arm"}
    try:
        ours_peak_value, ref_peak_value = float(ours_peak), float(ref_peak)
    except (TypeError, ValueError):
        return {"status": "pending", "reason": "capacity.peak_concurrency is not numeric"}
    if ours_peak_value <= 0.0 or ref_peak_value <= 0.0:
        return {"status": "pending", "reason": "capacity.peak_concurrency is not positive"}

    used_ours = _number(_w7_capacity(summary).get("state_used_mb"))
    used_ref = _number(_w7_capacity(baseline).get("state_used_mb"))
    if used_ours is not None and used_ref is not None:
        basis = "capacity.state_used_mb / peak_concurrency (written KV + linear slot)"
        ours_per = used_ours / ours_peak_value
        ref_per = used_ref / ref_peak_value
        used_total_ours, used_total_ref = used_ours, used_ref
    else:
        total_ours = _number(_w7_capacity(summary).get("state_total_mb"))
        total_ref = _number(_w7_capacity(baseline).get("state_total_mb"))
        if total_ours is None or total_ref is None:
            return {
                "status": "pending",
                "reason": "capacity.state_used_mb (or state_total_mb) missing from an arm; "
                "request rows are not in summary.json, so the ratio cannot be computed",
            }
        basis = (
            "capacity.state_total_mb / peak_concurrency (reserved; integer floor can "
            "amplify the 8k ratio)"
        )
        ours_per = total_ours / ours_peak_value
        ref_per = total_ref / ref_peak_value
        used_total_ours, used_total_ref = total_ours, total_ref

    if ours_per <= 0.0:
        return {"status": "pending", "reason": "hybrid per-sequence state is not positive"}
    ratio = ref_per / ours_per
    return {
        "status": "pass" if ratio >= limit else "fail",
        "value": round(ratio, 3),
        "target": limit,
        "context_tokens": 8192,
        "basis": basis,
        "ours_state_mb_per_seq": round(ours_per, 3),
        "baseline_state_mb_per_seq": round(ref_per, 3),
        "ours_state_mb": used_total_ours,
        "baseline_state_mb": used_total_ref,
        "ours_peak_concurrency": ours_peak_value,
        "baseline_peak_concurrency": ref_peak_value,
        "reason": "per-sequence state at 8k: pure / hybrid; P1 passes when the hybrid "
        "sequence costs <= 1/limit of the pure one",
    }


def _check_w7_tpot_growth(
    summary: Mapping[str, Any], limit: float, *, mode: str
) -> dict[str, Any]:
    """P3: TPOT(8k) / TPOT(512) from the per-context groups of one scenario arm."""
    tpot_512, batch_512, n_512, error_512 = _w7_group_series(summary, 512, "tpot_ms")
    tpot_8k, batch_8k, n_8k, error_8k = _w7_group_series(summary, 8192, "tpot_ms")
    if error_512 or error_8k:
        return {
            "status": "pending",
            "reason": "; ".join(part for part in (error_512, error_8k) if part),
        }
    if batch_512 != batch_8k:
        return {
            "status": "pending",
            "reason": f"512 is at batch {batch_512}, 8k at batch {batch_8k}; one growth "
            "ratio needs one batch",
        }
    if tpot_512 is None or tpot_8k is None or tpot_512 <= 0.0:
        return {"status": "pending", "reason": "TPOT at 512 or 8k is missing/non-positive"}
    growth = tpot_8k / tpot_512
    if mode == "max":
        status = "pass" if growth <= limit else "fail"
        direction = "hybrid TPOT growth must stay <= target"
    elif mode == "min":
        status = "pass" if growth >= limit else "fail"
        direction = "pure TPOT growth must reach >= target"
    else:  # pragma: no cover - internal misuse
        raise ValueError(f"unknown W7 growth mode {mode!r}")
    return {
        "status": status,
        "value": round(growth, 3),
        "target": limit,
        "batch": batch_512,
        "tpot_512_ms": round(tpot_512, 3),
        "tpot_8k_ms": round(tpot_8k, 3),
        "n_groups_512": n_512,
        "n_groups_8k": n_8k,
        "reason": f"TPOT(8k)/TPOT(512) from per-context groups; {direction}",
    }


def _check_w7_p2_latency(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    limit: float,
) -> dict[str, Any]:
    """P2-latency: hybrid/pure TTFT and TPOT at 512. Split results are 口径依赖."""
    if not baseline:
        return {"status": "pending", "reason": "no paired pure 0.6B scenario summary"}
    problems = _w7_pair_problems(summary, baseline, require_batch=True)
    if problems:
        return {
            "status": "invalid",
            "problems": problems[:6],
            "reason": "512 latency pair is not the same device/dtype/batch",
        }
    ours_ttft, ours_batch, _, ours_ttft_error = _w7_group_series(summary, 512, "ttft_ms")
    ours_tpot, _, _, ours_tpot_error = _w7_group_series(summary, 512, "tpot_ms")
    ref_ttft, ref_batch, _, ref_ttft_error = _w7_group_series(baseline, 512, "ttft_ms")
    ref_tpot, _, _, ref_tpot_error = _w7_group_series(baseline, 512, "tpot_ms")
    errors = [
        part
        for part in (
            f"ours TTFT: {ours_ttft_error}" if ours_ttft_error else "",
            f"ours TPOT: {ours_tpot_error}" if ours_tpot_error else "",
            f"baseline TTFT: {ref_ttft_error}" if ref_ttft_error else "",
            f"baseline TPOT: {ref_tpot_error}" if ref_tpot_error else "",
        )
        if part
    ]
    if errors:
        return {"status": "pending", "reason": "; ".join(errors)}
    if ours_batch != ref_batch:
        return {
            "status": "invalid",
            "problems": [f"batch: {ours_batch!r} vs baseline {ref_batch!r}"],
            "reason": "the 512 latency pair must run the same batch",
        }
    if not ours_ttft or not ours_tpot or not ref_ttft or not ref_tpot:
        return {"status": "pending", "reason": "TTFT/TPOT at 512 is missing or non-positive"}
    ttft_ratio = ours_ttft / ref_ttft
    tpot_ratio = ours_tpot / ref_tpot
    if ttft_ratio >= limit and tpot_ratio >= limit:
        status = "pass"
        reason = "both TTFT and TPOT show hybrid >= target slower than pure at 512"
    elif ttft_ratio < limit and tpot_ratio < limit:
        status = "fail"
        reason = "neither TTFT nor TPOT is materially worse at 512"
    else:
        status = "口径依赖"
        reason = "TTFT and TPOT disagree on materiality at 512; report both, do not force a pass"
    return {
        "status": status,
        "target": limit,
        "batch": ours_batch,
        "ttft_ratio": round(ttft_ratio, 3),
        "tpot_ratio": round(tpot_ratio, 3),
        "ours_ttft_ms": round(ours_ttft, 3),
        "baseline_ttft_ms": round(ref_ttft, 3),
        "ours_tpot_ms": round(ours_tpot, 3),
        "baseline_tpot_ms": round(ref_tpot, 3),
        "reason": reason,
    }


def _w7_memory_crossover(summary: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:
    """An explicitly recorded memory-crossover measurement, and where it came from."""
    paths = (
        ("memory_crossover", "ctx_tokens"),
        ("memory_crossover", "context_tokens"),
        ("memory_crossover", "prompt_tokens"),
        ("crossover", "memory_ctx_tokens"),
        ("crossover", "memory_context_tokens"),
        ("crossover", "memory", "ctx_tokens"),
        ("capacity", "memory_crossover_ctx"),
        ("memory_crossover_ctx",),
    )
    for path in paths:
        node: Any = summary
        for part in path:
            if not isinstance(node, Mapping):
                node = None
                break
            node = node.get(part)
        if node is None:
            continue
        value = _number(node)
        if value is None:
            return None, None, f"{'.'.join(path)} is present but not numeric"
        return value, ".".join(path), None
    return None, None, None


def _check_w7_memory_crossover(
    summary: Mapping[str, Any], limit: float
) -> dict[str, Any]:
    value, path, error = _w7_memory_crossover(summary)
    if error:
        return {"status": "pending", "reason": error}
    if value is None:
        return {
            "status": "pending",
            "reason": "summary does not record a measured memory crossover "
            "(e.g. memory_crossover.ctx_tokens); refusing to infer one from capacity ratios",
        }
    return {
        "status": "pass" if value <= limit else "fail",
        "value": round(value, 3),
        "target": limit,
        "source": path,
        "reason": "measured memory crossover in prompt tokens; explicit summary field only",
    }


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
        elif name == "decode_step_ratio_max":
            # W4.5 C4: same active buckets on both arms; the decode kernel is the
            # variable, so a different attn_decode is expected, not a mismatch.
            entry.update(_check_decode_step_ratio(summary, baseline, float(limit)))
        elif name == "p1_state_ratio_8k_min":
            entry.update(_check_w7_p1_state_ratio(summary, baseline, float(limit)))
        elif name in _W7_CAPACITY_CONTEXTS:
            entry.update(
                _check_w7_capacity_speedup(
                    summary,
                    baseline,
                    float(limit),
                    _W7_CAPACITY_CONTEXTS[name],
                )
            )
        elif name == "hybrid_tpot_growth_max":
            entry.update(_check_w7_tpot_growth(summary, float(limit), mode="max"))
        elif name == "pure_tpot_growth_min":
            entry.update(_check_w7_tpot_growth(summary, float(limit), mode="min"))
        elif name == "p2_latency_material_ratio":
            entry.update(_check_w7_p2_latency(summary, baseline, float(limit)))
        elif name == "memory_crossover_ctx_max":
            entry.update(_check_w7_memory_crossover(summary, float(limit)))
        elif name == "packed_prefill_ttft_p99_speedup_min":
            entry.update(
                _check_ttft_ratio(
                    summary,
                    baseline,
                    float(limit),
                    percentile="p99",
                    mode="speedup",
                )
            )
        elif name == "packed_prefill_ttft_p50_ratio_max":
            entry.update(
                _check_ttft_ratio(
                    summary,
                    baseline,
                    float(limit),
                    percentile="p50",
                    mode="ratio",
                )
            )
        elif name == "golden_token_match_min":
            entry.update(_check_golden_token_match(summary, float(limit)))
        elif name == "paged_vs_seat_token_match_min":
            entry.update(
                _check_paged_vs_seat_token_match(summary, baseline, float(limit))
            )
        elif name in {"max_concurrency_speedup_min", "capacity_speedup_min"}:
            # W4 C1 / W4.5 alias: fill-only allocator capacity under one total budget
            # (D6). Paired on device/dtype/workload/budget, deliberately *not* on
            # ``max_batch``.
            entry.update(_check_capacity_speedup(summary, baseline, float(limit)))
        elif name == "fragmentation_max":
            # W4 C2: internal frag only; the direction is ``<= limit``.
            entry.update(_check_fragmentation(summary, float(limit)))
        else:
            entry.update({"status": "pending", "reason": "no evaluator for this criterion yet"})
        results.append(entry)
    return results
