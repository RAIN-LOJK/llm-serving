"""Trace-level aggregation for W3 scenario runs (D5 semantics, ``docs/decisions.md``).

Pure functions: rows in, summary blocks out. No file or accelerator access; everything here
is derived from the three ``metrics.jsonl`` row kinds of D5:

- ``request`` rows: one per request per repeat (latency percentiles, decode-only tokens);
- ``step`` rows: one per scheduler step (per-``active`` means);
- ``trace`` rows: one per repeat, the authoritative throughput row.

The row fields are read by the names frozen in D5 — the runner and the thresholds
readers key off those names, so do not rename anything in the returned dicts.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from bench.metrics import percentiles, summarize

#: A request decomposes iff ``|ttft - (queue + prefill)|`` stays under this (C2, §2.1).
DECOMPOSE_TOLERANCE_MS = 1.0
#: Fraction of requests that must decompose before the trace block claims ``decomposed``.
DECOMPOSE_MIN_RATE = 0.99


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float | None]:
    return [_finite(row.get(key)) for row in rows]


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [value for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def _decode_only_by_repeat(
    request_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
) -> dict[Any, float]:
    """``Σ out_tokens / Σ decode_event_ms * 1000`` per repeat.

    Output tokens live on request rows, ``decode_event_ms`` on step rows, so the two are
    joined on ``repeat``. Repeats with no decode timing (or a zero total) are dropped and
    surface as ``None`` in the summary.
    """
    tokens: dict[Any, float] = {}
    decode_ms: dict[Any, float] = {}
    for row in request_rows:
        value = _finite(row.get("out_tokens"))
        if value is None:
            continue
        key = row.get("repeat")
        tokens[key] = tokens.get(key, 0.0) + value
    for row in step_rows:
        value = _finite(row.get("decode_event_ms"))
        if value is None:
            continue
        key = row.get("repeat")
        decode_ms[key] = decode_ms.get(key, 0.0) + value

    out: dict[Any, float] = {}
    for key, total in tokens.items():
        elapsed = decode_ms.get(key)
        if elapsed is None or elapsed <= 0.0:
            continue
        out[key] = total / elapsed * 1000.0
    return out


def analytic_straggler_ratio(
    budgets: Sequence[int] | None,
    max_batch: int,
) -> float | None:
    """``E[max of max_batch sampled budgets] / E[mean of max_batch sampled budgets]``.

    Budgets are drawn uniformly with replacement (D5). With at most a handful of distinct
    budget values the expectation is exact, so no Monte Carlo seed is needed:

        E[max] = Σ_v v · (F(v)^B − F(v^-)^B)

    ``max_batch <= 1`` has no wave-length straggle, and an empty/zero-mean budget list
    cannot define a ratio — both return ``None``.
    """
    if budgets is None or max_batch is None or int(max_batch) <= 1:
        return None
    vals = [value for value in (_finite(budget) for budget in budgets) if value is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if mean == 0.0:
        return None

    counts = Counter(vals)
    n = len(vals)
    batch = int(max_batch)
    cdf = 0.0
    previous = 0.0
    expected_max = 0.0
    for value in sorted(counts):
        cdf += counts[value] / n
        expected_max += value * (cdf**batch - previous**batch)
        previous = cdf
    return expected_max / mean


def _throughput(
    request_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int] | None,
    max_batch: int,
) -> dict[str, Any]:
    tok_s = summarize(_values(trace_rows, "tok_s"))
    tok_s_active = summarize(_values(trace_rows, "tok_s_active"))
    req_s = summarize(_values(trace_rows, "req_s"))
    makespan_s = summarize(_values(trace_rows, "makespan_s"))
    queue_depth = summarize(_values(trace_rows, "queue_depth_at_last_arrival"))

    achieved_values = []
    for row in trace_rows:
        value = _finite(row.get("achieved_rate"))
        if value is None:
            # D5: fall back to req_s when the row does not carry its own achieved_rate.
            value = _finite(row.get("req_s"))
        achieved_values.append(value)
    achieved_rate = summarize(achieved_values)

    depth_values = _values(trace_rows, "queue_depth_at_last_arrival")
    # P-29: saturation is "the trace has been fully submitted and requests are still
    # waiting for capacity", read at the instant the last request arrived (never at the
    # end of the run, which has necessarily drained). It is never compared against a
    # capacity estimate, and a row whose depth is missing cannot prove saturation.
    saturated = bool(trace_rows) and all(
        value is not None and value > 0.0 for value in depth_values
    )

    decode_by_repeat = _decode_only_by_repeat(request_rows, step_rows)
    decode_values: list[float | None] = []
    for row in trace_rows:
        value = decode_by_repeat.get(row.get("repeat"))
        if value is None and len(trace_rows) == 1 and len(decode_by_repeat) == 1:
            # Single-repeat rows may omit ``repeat``; there is only one group to attach.
            value = next(iter(decode_by_repeat.values()))
        decode_values.append(value)

    return {
        "n_traces": len(trace_rows),
        "tok_s": tok_s,
        "tok_s_active": tok_s_active,
        "decode_only_tok_s": summarize(decode_values),
        "req_s": req_s,
        "makespan_s": makespan_s,
        "offered_rate": summarize(_values(trace_rows, "offered_rate"))["median"],
        "achieved_rate": achieved_rate,
        "saturated": saturated,
        "queue_depth_at_last_arrival": queue_depth,
        "analytic_straggler_ratio": analytic_straggler_ratio(budgets, max_batch),
    }


def _latency(
    request_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ttft = percentiles(_values(request_rows, "ttft_ms"))
    queue = percentiles(_values(request_rows, "queue_ms"))
    prefill = percentiles(_values(request_rows, "prefill_ms"))

    queue_share_p99 = None
    if queue["p99"] is not None and ttft["p99"] not in (None, 0.0):
        queue_share_p99 = queue["p99"] / ttft["p99"]

    triples = [
        (t, q, p)
        for t, q, p in zip(
            _values(request_rows, "ttft_ms"),
            _values(request_rows, "queue_ms"),
            _values(request_rows, "prefill_ms"),
        )
        if t is not None and q is not None and p is not None
    ]
    decomposed_rate = (
        sum(abs(t - (q + p)) < DECOMPOSE_TOLERANCE_MS for t, q, p in triples) / len(triples)
        if triples
        else 0.0
    )
    max_error = max((abs(t - (q + p)) for t, q, p in triples), default=None)
    all_p99_present = all(block["p99"] is not None for block in (ttft, queue, prefill))
    decomposed = all_p99_present and decomposed_rate >= DECOMPOSE_MIN_RATE

    latency = {
        "ttft_ms": ttft,
        "queue_ms": queue,
        "prefill_ms": prefill,
        "queue_share_p99": queue_share_p99,
        "decomposed": decomposed,
    }
    consistency = {
        "decomposed_rate": decomposed_rate,
        "ttft_minus_queue_prefill_max_ms": max_error,
    }
    return latency, consistency


def _steps(step_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[float, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        active = _finite(row.get("active"))
        if active is None:
            continue
        groups.setdefault(active, []).append(row)

    by_active = []
    for active in sorted(groups):
        rows = groups[active]
        step_ms = _values(rows, "step_ms")
        prefill_ms = _values(rows, "prefill_ms")
        decode_ms = _values(rows, "decode_event_ms")

        host_gap_ms = []
        for row in rows:
            gap = _finite(row.get("host_gap_ms"))
            if gap is None:
                # D5: host_gap_ms = step_ms - decode_event_ms - prefill_ms. Any missing
                # term means the gap is unknowable for this step, not zero.
                s = _finite(row.get("step_ms"))
                d = _finite(row.get("decode_event_ms"))
                p = _finite(row.get("prefill_ms"))
                gap = None if s is None or d is None or p is None else s - d - p
            host_gap_ms.append(gap)

        by_active.append(
            {
                "active": int(active) if float(active).is_integer() else active,
                "n": len(rows),
                "step_ms": _mean(step_ms),
                "prefill_ms": _mean(prefill_ms),
                "decode_event_ms": _mean(decode_ms),
                "host_gap_ms": _mean(host_gap_ms),
            }
        )
    return {"by_active": by_active}


def aggregate(
    request_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int] | None = None,
    max_batch: int = 1,
) -> dict[str, Any]:
    """Build the D5 ``throughput``/``latency``/``steps``/``consistency`` blocks.

    ``budgets`` is the trace's budget distribution (used only for
    ``analytic_straggler_ratio``) and ``max_batch`` the seat count. Latency percentiles are
    taken over every request row, never over per-trace pre-aggregates, so a trace that
    contributes more requests weighs more. ``saturated`` follows P-29: every trace row's
    ``queue_depth_at_last_arrival`` must be positive; capacity is never consulted.
    """
    latency, consistency = _latency(request_rows)
    return {
        "throughput": _throughput(request_rows, step_rows, trace_rows, budgets, max_batch),
        "latency": latency,
        "steps": _steps(step_rows),
        "consistency": consistency,
    }
