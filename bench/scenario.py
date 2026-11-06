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
    # W4 (D6): the observed occupancy of a run, not the allocator capacity of a fill
    # probe -- ``capacity.peak_concurrency`` answers C1, this one is E3's dynamic peak.
    peak_concurrency = summarize(_values(trace_rows, "peak_concurrency"))["median"]

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
        "peak_concurrency": peak_concurrency,
        "analytic_straggler_ratio": analytic_straggler_ratio(budgets, max_batch),
    }


def _latency(
    request_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ttft = percentiles(_values(request_rows, "ttft_ms"))
    queue = percentiles(_values(request_rows, "queue_ms"))
    prefill = percentiles(_values(request_rows, "prefill_ms"))
    # W4.5 (D7 TTFT row): a serial row has no pack_wait field, and its wait is
    # structurally zero; anything else is used as written. ``prefill_shared_ms`` is the
    # per-group shared prefill -- identical to ``prefill_ms`` for a serial request, but
    # reported separately because a packed group pays it once for every member.
    pack_wait_values: list[float | None] = []
    for row in request_rows:
        wait = _finite(row.get("pack_wait_ms"))
        pack_wait_values.append(0.0 if wait is None else wait)
    pack_wait = percentiles(pack_wait_values)
    shared = percentiles(_values(request_rows, "prefill_shared_ms"))

    queue_share_p99 = None
    if queue["p99"] is not None and ttft["p99"] not in (None, 0.0):
        queue_share_p99 = queue["p99"] / ttft["p99"]

    # D5 froze ``ttft == queue + prefill``; W4.5 extends the identity by the packed
    # arm's admission wait: ``ttft == queue + pack_wait + prefill``. A missing
    # pack_wait is 0, so every W3/W4 row decomposes exactly as it did before.
    rows = zip(
        _values(request_rows, "ttft_ms"),
        _values(request_rows, "queue_ms"),
        pack_wait_values,
        _values(request_rows, "prefill_ms"),
    )
    quadruples = [
        (t, q, w, p) for t, q, w, p in rows if t is not None and q is not None and p is not None
    ]
    decomposed_rate = (
        sum(abs(t - (q + w + p)) < DECOMPOSE_TOLERANCE_MS for t, q, w, p in quadruples)
        / len(quadruples)
        if quadruples
        else 0.0
    )
    max_error = max((abs(t - (q + w + p)) for t, q, w, p in quadruples), default=None)
    all_p99_present = all(block["p99"] is not None for block in (ttft, queue, prefill))
    decomposed = all_p99_present and decomposed_rate >= DECOMPOSE_MIN_RATE

    latency = {
        "ttft_ms": ttft,
        "queue_ms": queue,
        "prefill_ms": prefill,
        "pack_wait_ms": pack_wait,
        "prefill_shared_ms": shared,
        "queue_share_p99": queue_share_p99,
        "decomposed": decomposed,
    }
    consistency = {
        "decomposed_rate": decomposed_rate,
        "ttft_minus_queue_prefill_max_ms": max_error,
    }
    return latency, consistency


def _memory(
    step_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """W4 (D6) memory ledger, derived from the rows the arm already writes.

    Field meanings are frozen in ``docs/decisions.md`` D6:

    * ``kv_mb`` -- the KV pool's reservation (``kv_pool_mb`` when the row carries the
      total-budget split, otherwise the W3 ``kv_mb``);
    * ``workspace_mb`` -- the gather workspace, a paged-only cost; the seat arm has 0;
    * ``kv_total_mb`` -- ``kv_mb + workspace_mb``, the number capacity is judged against
      (P-36: never KV alone);
    * ``seat_unused_mb`` -- budget a seat arm cannot use because of row granularity;
    * ``frag_internal`` / ``reservation_slack`` / ``seat_row_slack`` -- the paged ledger
      and its seat analogue (only internal frag is judged, C2);
    * ``capacity_per_gib`` -- ``peak_concurrency / (kv_total_mb / 1024)``;
    * ``bytes_per_active_request`` -- ``kv_total_mb * MiB / peak_concurrency``.

    Every value is ``None`` when no row carries the term, and a paged arm's unmeasured
    workspace stays ``None`` rather than reading as "free".
    """
    def last(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        for row in reversed(rows):
            value = _finite(row.get(key))
            if value is not None:
                return value
        return None

    pool = last(trace_rows, "kv_pool_mb")
    if pool is None:
        pool = last(trace_rows, "kv_mb")
    workspace = last(trace_rows, "workspace_mb")
    if workspace is None:
        workspace = last(step_rows, "workspace_mb")
    if workspace is None and pool is not None and not any(
        "blocks_in_use" in row for row in step_rows
    ):
        # Structural, not inferred from data: a seat arm gathers nothing, so its gather
        # workspace is exactly zero. A paged arm that failed to report one stays None
        # (an unmeasured workspace must not read as "free").
        workspace = 0.0
    total = last(trace_rows, "kv_total_mb")
    if total is None:
        total = last(step_rows, "kv_total_mb")
    if total is None and pool is not None and workspace is not None:
        total = pool + workspace

    peak = _finite01([row.get("peak_concurrency") for row in trace_rows])
    if peak is None:
        peak = max(
            (int(row.get("active") or 0) for row in step_rows),
            default=0,
        ) or None

    capacity_per_gib = (
        peak / (total / 1024.0) if peak is not None and total is not None and total > 0 else None
    )
    bytes_per_active = (
        total * 2**20 / peak if peak is not None and total is not None and peak > 0 else None
    )

    values = {
        "kv_mb": pool,
        "workspace_mb": workspace,
        "kv_total_mb": total,
        "seat_unused_mb": last(trace_rows, "seat_unused_mb"),
        "frag_internal": last(step_rows, "frag_internal"),
        "reservation_slack": last(step_rows, "reservation_slack"),
        # The seat arm has no per-step block ledger; if a row ever carries one, use it.
        "seat_row_slack": last(trace_rows, "seat_row_slack"),
        "capacity_per_gib": capacity_per_gib,
        "bytes_per_active_request": bytes_per_active,
    }
    return values


def _finite01(values: Sequence[Any]) -> float | None:
    """Median of the finite values (a tiny local helper; ``summarize`` is overkill)."""
    vals = sorted(value for value in (_finite(v) for v in values) if value is not None)
    if not vals:
        return None
    middle = len(vals) // 2
    if len(vals) % 2:
        return vals[middle]
    return (vals[middle - 1] + vals[middle]) / 2.0


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
        # W4.5 step attribution. A W3/W4 row has none of these fields; the mean must stay
        # ``None`` there rather than inventing a zero that would read as "no pack wait".
        prefill_groups = _values(rows, "prefill_groups")
        pack_wait_ms = _values(rows, "pack_wait_ms")

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
                "prefill_groups": _mean(prefill_groups),
                "pack_wait_ms": _mean(pack_wait_ms),
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
        "memory": _memory(step_rows, trace_rows, request_rows),
    }
