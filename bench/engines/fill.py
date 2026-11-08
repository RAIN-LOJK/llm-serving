"""Fill-only execution for W4 capacity probes (``docs/decisions.md`` D6).

A fill run answers one question: **how many requests fit at the same time** under a
memory budget. It must therefore prefill and reserve, but never decode -- decoding
would retire requests and recycle their capacity, which is exactly the quantity the
run is trying to hold fixed.

Two pieces live here, both selected only by ``[exp].type = "fill"``:

* :class:`FillScheduler` -- a :class:`serve.scheduler.Scheduler` subclass that admits
  (through the engine's optional ``can_admit_group`` hook, so a paged pool that cannot
  reserve the packed group refuses the FIFO head) and then finishes every resident
  request without a decode call. It reuses the base admission loop, timestamps and
  release path; only :meth:`~serve.scheduler.Scheduler.step` / ``run`` are replaced.
* :func:`run_fill_trace` -- the seat-arm runner (``impl="continuous"``), mirroring
  :meth:`bench.engines.continuous.ContinuousEngine.run_trace` but stopping after
  admission. The paged arm builds its own trace row inside
  :class:`bench.engines.paged.PagedEngine`; both share :func:`capacity_block`.

The fill step row's ``active`` means "admitted in this admission pass", not
"still running at the end of the step" (a fill step retires everything it admits,
so the next pass measures the same budget from an empty pool). The base scheduler's
end-of-step occupancy is untouched for non-fill runs.

``bench/engines/paged.py`` imports :data:`W4_TOTAL_BUDGET_MB`. D6 freezes the W4
total budget (KV + workspace) at 1640 MiB; the seat arm's reservation is whatever
``max_batch`` rows cost, so its unused remainder is reported as ``seat_unused_mb``
and its ``kv_total_mb`` is the shared upper bound -- the number
``bench/thresholds.py`` compares the paged arm against.
"""

from __future__ import annotations

import time
from typing import Any

from serve.scheduler import Request, Scheduler

__all__ = [
    "FillScheduler",
    "W4_TOTAL_BUDGET_MB",
    "bucket_tokens",
    "capacity_block",
    "run_fill_trace",
]

#: D6 total budget (KV + workspace) for W4, in MiB (``docs/decisions.md`` D6).
W4_TOTAL_BUDGET_MB = 1640.0

#: The seat arm buckets each row to 64 columns (``serve/engine.py::SLAB_GRANULARITY``).
ROW_GRANULARITY = 64


def bucket_tokens(tokens: int) -> int:
    """Round ``tokens`` up to the row bucket (never below one column)."""
    return -(-max(int(tokens), 1) // ROW_GRANULARITY) * ROW_GRANULARITY


def _prefill_timings(
    admitted: list[Request], step_groups: Any = None
) -> tuple[float, int, float, float]:
    """``(prefill_ms, groups, pack_wait_ms, prefill_shared_ms)`` for one admission pass.

    A packed group pays its shared prefill **once**; summing per request would multiply
    that time by the group size. The base scheduler keeps a per-group ledger
    (``_step_groups``: one entry per forward, including a failed one), and that ledger is
    the authoritative accounting. A fill step that admitted nothing reports 0 groups.

    ``pack_wait_ms`` is per request (each request that waited for the shared prefill
    contributes its own wait), matching the base scheduler's step row; only the shared
    prefill is a group-level cost and is counted once.
    """
    groups = [
        group
        for group in (step_groups or [])
        if isinstance(group, dict)
        and group.get("start") is not None
        and group.get("end") is not None
    ]
    prefill_ms = sum(
        (float(group["end"]) - float(group["start"])) * 1000.0 for group in groups
    )
    group_count = len(groups)
    pack_wait_ms = sum(
        (m.t_prefill_start - m.t_admitted) * 1000.0
        for m in admitted
        if m.t_prefill_start is not None and m.t_admitted is not None
    )
    return prefill_ms, group_count, pack_wait_ms, prefill_ms


class FillScheduler(Scheduler):
    """Admission + reservation only; no decode, no KV growth after prefill."""

    def step(self) -> dict:
        self._admitted_this_step = 0
        self._retired_this_step = 0
        self._step_groups = []
        self._step_origin = self.now()

        admitted = self._admit()
        prefill_ms, prefill_groups, pack_wait_ms, prefill_shared_ms = _prefill_timings(
            admitted, getattr(self, "_step_groups", None)
        )

        # The resident set is the capacity answer, so the pool's ledger must be read
        # *before* anything is released (after release, every block is free again).
        resident = len(self._active)
        stats: dict[str, Any] | None = None
        if resident:
            provider = getattr(self.engine, "paged_stats", None)
            if callable(provider):
                try:
                    stats = dict(provider() or {})
                except Exception:  # noqa: BLE001 - instrumentation must not fail a fill run
                    stats = None

        # No decode: finish every resident request now. ``_release`` returns its seat
        # (and, on the paged arm, its blocks) to the free list. Fill-only means
        # "prefill + reserve", so the first-token argmax the prefill returned is not a
        # generated token and is not counted.
        for req in list(self._active):
            req.out = []
            req.t_finish = self.now()
            req.state = "finished"
            self._release(req)

        step_ms = (self.now() - self._step_origin) * 1000.0
        now = self.now()
        queued = sum(1 for item in self._queue if item[0] <= now)
        row: dict[str, Any] = {
            "step": self._step_no,
            "active": resident,
            "queued": queued,
            "pending": len(self._queue) - queued,
            "admitted": self._admitted_this_step,
            "retired": self._retired_this_step,
            "step_ms": step_ms,
            "prefill_ms": prefill_ms,
            # W4.5 step attribution (D7 TTFT row). A group is one packed prefill
            # forward; an idle fill pass reports zero groups.
            "prefill_groups": prefill_groups,
            "pack_wait_ms": pack_wait_ms,
            "prefill_shared_ms": prefill_shared_ms,
            # No decode happened; the key stays so ``bench/scenario.py``'s step
            # aggregation keeps the same shape as a scenario row.
            "decode_event_ms": 0.0,
            "host_gap_ms": step_ms - prefill_ms,
            "fill_only": True,
        }
        # Paged engines append block counts / frag / memory to the row. The snapshot
        # above is the resident-set account; the base hook would now report the
        # post-release (empty) pool, which is not what a capacity probe wants.
        if stats is not None:
            row.update(stats)
        else:
            self._attach_paged_stats(row)
        self._step_no += 1
        self._rows.append(row)
        return row

    def run(self) -> list[Request]:
        """Drain the queue; stop when a pass admits nothing (capacity exhausted)."""
        self._closed = True
        while self._queue or self._active:
            row = self.step()
            if not self._active and not row["admitted"] and not row["retired"]:
                # Every seat is free and the FIFO head still cannot be admitted: the
                # pool is out of capacity for it (``can_admit_group`` refused). Leaving
                # it queued is the honest outcome of a fill probe, not an error.
                break
        return sorted(self._all, key=lambda r: r.id)


def capacity_block(
    *,
    peak_concurrency: int,
    kv_mb: float | None,
    workspace_mb: float | None,
    kv_total_mb: float | None,
    block_size: int | None,
    kv_blocks: int | None,
    max_batch: int,
    frag_internal: float | None = None,
    reservation_slack: float | None = None,
    seat_unused_mb: float | None = None,
    workload_sha256: str | None = None,
) -> dict[str, Any]:
    """The ``summary.capacity`` block consumed by the W4 C1/C2 evaluators.

    ``peak_concurrency`` is the allocator capacity on a fill run (max simultaneous
    admissions) and the dynamic peak on a scenario run; the two are never mixed in
    one block (D6). ``capacity_per_gib`` divides that count by the *total* budget
    ``kv_mb + workspace_mb``, never by KV alone (P-36). W4.5's paged Triton arm has no
    gather workspace, so it reports ``workspace_mb = 0.0``; the W4 ``workspace_estimated``
    provenance is gone with the gather control arm (historical summaries keep the key).
    """
    total = None if kv_total_mb is None else float(kv_total_mb)
    per_gib = (
        float(peak_concurrency) / (total / 1024.0)
        if total is not None and total > 0.0
        else None
    )
    return {
        "peak_concurrency": int(peak_concurrency),
        "kv_mb": None if kv_mb is None else float(kv_mb),
        "workspace_mb": None if workspace_mb is None else float(workspace_mb),
        "kv_total_mb": total,
        "block_size": None if block_size is None else int(block_size),
        "kv_blocks": None if kv_blocks is None else int(kv_blocks),
        "max_batch": int(max_batch),
        "frag_internal": frag_internal,
        "reservation_slack": reservation_slack,
        "seat_unused_mb": seat_unused_mb,
        "capacity_per_gib": per_gib,
        "workload_sha256": workload_sha256,
    }


def run_fill_trace(adapter, trace, *, repeat: int = 0):
    """Fill-only trace for the seat arm (``impl="continuous"``).

    Returns a :class:`~bench.engines.continuous.TraceResult` with the same row kinds
    as a scenario run, plus ``summary["capacity"]``. The engine, its pool and the
    scheduler are the seat path's; only the step loop is :class:`FillScheduler`.
    Row fields shared with :meth:`ContinuousEngine.run_trace` keep their names and
    meanings; the trace row gains the W4 total-budget fields.
    """
    from bench.engines.continuous import TraceResult, _request_row, _trace_clock, _trace_row
    from bench.scenario import aggregate

    if adapter._engine is None:
        raise RuntimeError("call load() before run_trace()")
    engine = adapter._engine

    longest_prompt = max((len(a.prompt_ids) for a in trace.arrivals), default=0)
    longest_budget = max((a.budget for a in trace.arrivals), default=0)
    need = longest_prompt + longest_budget + 1
    slab = engine.open_pool(adapter.max_batch, need)
    t0 = time.perf_counter()
    adapter.pool_kv_mb = engine.pool_kv_mb
    adapter.max_len = int(getattr(slab, "max_len", need))

    scheduler = FillScheduler(
        engine,
        max_batch=adapter.max_batch,
        policy=adapter.policy,
        # Same clock convention as the scenario path: arrivals are relative to the
        # trace's origin, timestamps are wall clock.
        now=_trace_clock(t0),
        eos_token_id=adapter.eos_token_id,
        max_position=int(engine.max_position),
    )
    prompt_ids_by_idx = {a.idx: a.prompt_id for a in trace.arrivals}
    for arrival in trace.arrivals:
        scheduler.submit(
            Request(
                id=arrival.idx,
                prompt_ids=list(arrival.prompt_ids),
                budget=int(arrival.budget),
                arrival_ts=float(arrival.at_s),
            )
        )
    scheduler.close()
    requests = scheduler.run()

    t_arrival = min((r.t_arrival for r in requests), default=0.0)
    finishes = [r.t_finish for r in requests if r.t_finish is not None]
    t_finish = max(finishes, default=t_arrival)
    firsts = [r.t_first for r in requests if r.t_first is not None]
    t_first_token = min(firsts, default=t_arrival)
    out_total = sum(len(r.out) for r in requests)

    request_rows = [
        _request_row(r, prompt_ids_by_idx.get(r.id), adapter.pool_kv_mb, adapter.max_batch, repeat)
        for r in requests
    ]
    step_rows = [dict(row, row_kind="step", repeat=repeat) for row in scheduler.step_rows]
    trace_row = _trace_row(
        requests=requests,
        step_rows=step_rows,
        repeat=repeat,
        policy=adapter.policy,
        max_batch=adapter.max_batch,
        offered_rate=trace.rate,
        pool_kv_mb=adapter.pool_kv_mb,
        trace_sha256=trace.sha256,
        t_arrival=t_arrival,
        t_first_token=t_first_token,
        t_finish=t_finish,
        out_total=out_total,
    )

    # D6: the seat arm has no workspace; its reservation is the shared upper bound and
    # the row-granularity remainder is reported separately (plan §1.2 C1).
    kv = float(adapter.pool_kv_mb or 0.0)
    total = max(kv, W4_TOTAL_BUDGET_MB)
    unused = max(0.0, total - kv)
    row_width = int(adapter.max_len or bucket_tokens(need))
    live = sum(len(r.prompt_ids) for r in requests if r.t_finish is not None)
    # Fill-only requests never decode, so each seat row's occupied columns are its
    # prompt; the unused remainder is what the worst-case row width wastes. Reported
    # for the D6 comparison table, never judged.
    seat_row_slack = (
        1.0 - live / (len(requests) * row_width) if requests and row_width > 0 else None
    )
    trace_row.update(
        {
            "kv_pool_mb": kv,
            "workspace_mb": 0.0,
            "kv_total_mb": total,
            "seat_unused_mb": unused,
            "seat_row_slack": seat_row_slack,
            "fill_only": True,
        }
    )

    summary = aggregate(
        request_rows,
        step_rows,
        [trace_row],
        budgets=trace.budgets,
        max_batch=adapter.max_batch,
    )
    latency = summary.get("latency") or {}
    trace_row.update(
        {
            "ttft_p99_ms": (latency.get("ttft_ms") or {}).get("p99"),
            "queue_p99_ms": (latency.get("queue_ms") or {}).get("p99"),
            "prefill_p99_ms": (latency.get("prefill_ms") or {}).get("p99"),
        }
    )
    summary["capacity"] = capacity_block(
        peak_concurrency=max((int(row.get("active") or 0) for row in step_rows), default=0),
        kv_mb=kv,
        workspace_mb=0.0,
        kv_total_mb=total,
        block_size=None,
        kv_blocks=None,
        max_batch=adapter.max_batch,
        seat_unused_mb=unused,
        workload_sha256=trace.sha256,
    )
    return TraceResult(
        request_rows=request_rows,
        step_rows=step_rows,
        trace_row=trace_row,
        summary=summary,
        repeat=repeat,
        impl=getattr(adapter, "impl", "continuous"),
    )
