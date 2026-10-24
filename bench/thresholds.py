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


def _protocol_mismatches(summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """A pair is only comparable if it ran the same workload on the same kind of device."""
    ours = summary.get("protocol") or {}
    theirs = baseline.get("protocol") or {}
    problems = []
    for key in ("max_new_tokens", "prompt_sha256", "dtype", "device_class"):
        if ours.get(key) != theirs.get(key):
            problems.append(f"{key}: {ours.get(key)!r} vs baseline {theirs.get(key)!r}")
    return problems


def _decode_ratio(
    summary: Mapping[str, Any], baseline: Mapping[str, Any] | None
) -> tuple[float | None, int, list[str]]:
    """median(tpot_baseline / tpot_ours) over the prompts both arms ran.

    Any prompt whose output length differs makes the pair incomparable: TPOT over 40 tokens
    and TPOT over 60 are different quantities.
    """
    if not baseline:
        return None, 0, []
    problems = _protocol_mismatches(summary, baseline)
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
        elif name == "max_concurrency_speedup_min":
            entry.update({"status": "pending", "reason": "needs the W4 block manager run"})
        else:
            entry.update({"status": "pending", "reason": "no evaluator for this criterion yet"})
        results.append(entry)
    return results
