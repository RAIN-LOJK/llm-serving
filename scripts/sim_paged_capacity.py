#!/usr/bin/env python3
"""Deterministic W4 capacity simulation: seat rows vs paged blocks (``simulated``).

This is the A · capacity half of ``docs/w4-plan.md`` §4.1. It runs **no model, no
weights, no arrivals and no Poisson trace**: given a total budget (KV + workspace,
D6), a request-length distribution and a block size, it asks two questions.

* **seat** -- one ``KvSlab`` of ``max_batch`` rows, every row reserved at the width
  ``bucket(max(prompt + budget + 1), 64)`` of the whole trace (one row per request,
  the D6 seat unit). Capacity is how many rows the budget can hold.
* **paged** -- the real :class:`serve.block_manager.BlockManager`, filled FCFS in the
  trace's request order; each admitted request reserves ``blocks_for(prompt,
  budget)`` blocks up front, nothing is released, and the first request that does not
  fit stops admission (head-of-line blocking, D6). Capacity is how many requests are
  admitted before that stop, so the distribution's **order matters** and is part of
  the input.

The length distribution can be synthetic (:func:`build_requests`, the capacity-curve
half of §4.1) or -- with ``--trace`` -- a **real trace** replayed from a run directory:
a ``metrics.jsonl`` (``row_kind="request"`` rows, smallest ``repeat`` only, ordered by
``arrival_idx``) or a JSON list of ``[prompt_len, budget]`` pairs. Feeding the same
trace the real engine saw is what makes the E1 "allocator sim ±1" gate meaningful;
with a synthetic distribution it only measures workload modelling, not the allocator.
``--paged-blocks`` pins the pool to the run's fixed ``kv_blocks`` so the simulation
replays **the same allocator**, not an idealised budget-maximised one.

The workspace is paid **out of the same budget** (D6, Appendix C): the paged arm has
``kv_total_mb = kv_mb + workspace_mb``, and ``workspace_mb`` grows with the number of
admitted rows, so capacity and workspace are solved as a fixed point (binary search
over the admitted count). ``capacity_ratio = paged_capacity / seat_capacity`` on
integer admission counts.

Every number this script prints or writes is a **simulation result**, not a
measurement; the report must label it ``simulated``. Real fill runs
(``W4-0002/0003``) are the measurement half.

Usage::

    python scripts/sim_paged_capacity.py
    python scripts/sim_paged_capacity.py --out docs/fig_w4_capacity.png
    python scripts/sim_paged_capacity.py --json /tmp/capacity.json --block-sizes 16,32
    python scripts/sim_paged_capacity.py --budgets 1200,1456,1640,2400 --workspace-mb 240
    python scripts/sim_paged_capacity.py --trace results/W3/0011-continuous-b8-mixed/metrics.jsonl
    python scripts/sim_paged_capacity.py --trace trace.json --expect-concurrency 29
    python scripts/sim_paged_capacity.py --trace trace.json --budgets 1640 --paged-blocks 800
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.block_manager import BlockManager, FragStats  # noqa: E402  (needs ROOT on the path)

MIB = 1024 * 1024
GRANULARITY = 64  # KvSlab row buckets (serve/engine.py SLAB_GRANULARITY)


# ---------------------------------------------------------------- model shape


@dataclass(frozen=True)
class ModelShape:
    """Qwen3-0.6B defaults: every number below is a shape, not a measurement."""

    n_layer: int = 28
    n_kv_head: int = 8
    head_dim: int = 128
    n_head: int = 16
    vocab: int = 151936
    dtype_bytes: int = 2  # bf16

    @property
    def kv_bytes_per_token_per_layer(self) -> int:
        """K and V in one layer for one token: 4 KiB at the 0.6B shape."""
        return 2 * self.n_kv_head * self.head_dim * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        """All layers: 112 KiB at the 0.6B shape (docs/carrier.md)."""
        return self.kv_bytes_per_token_per_layer * self.n_layer


def bucket_width(tokens: int) -> int:
    """Round ``tokens`` up to the next 64 columns (the seat / gather bucket)."""
    return -(-max(int(tokens), 1) // GRANULARITY) * GRANULARITY


def workspace_mb(
    *,
    n_active: int,
    lpad: int,
    max_batch: int,
    shape: ModelShape,
) -> float:
    """Appendix C workspace for one layer's gather, plus bias and logits.

    ::

        K+V:    n_active * lpad * kv_bytes_per_token_per_layer
        bias:   n_active * n_head * lpad * dtype_bytes      (H_kv * G == n_head)
        logits: max_batch * vocab * dtype_bytes

    The gather buffer is one layer wide and reused layer by layer, so ``n_layer``
    deliberately does not appear here. This is occupancy, not gather traffic.
    """
    if n_active < 0 or lpad < 0 or max_batch < 0:
        raise ValueError("n_active, lpad and max_batch must be non-negative")
    gather = n_active * lpad * shape.kv_bytes_per_token_per_layer
    bias = n_active * shape.n_head * lpad * shape.dtype_bytes
    logits = max_batch * shape.vocab * shape.dtype_bytes
    return (gather + bias + logits) / MIB


# ---------------------------------------------------------------- distributions

BUDGET_TOKENS = 128
DISTRIBUTIONS = ("uniform-2048", "mild-mixed", "w3-like-max-2048")


def _spread(low: int, high: int, n: int) -> list[int]:
    """``n`` deterministic lengths spanning ``[low, high]`` inclusive."""
    if n == 1:
        return [high]
    step = (high - low) / (n - 1)
    return [int(round(low + index * step)) for index in range(n)]


def build_requests(name: str, budget: int = BUDGET_TOKENS) -> list[tuple[int, int]]:
    """``[(prompt_tokens, budget_tokens)]`` for one named distribution.

    * ``uniform-2048``: 32 x 2048 -- the negative control (seat's rows fit it exactly).
    * ``mild-mixed``: 16 x 512 then 16 x 2048.
    * ``w3-like-max-2048``: 12 short / 12 medium / 8 long (long bucket 1536-2048),
      mirroring ``bench/prompts.json``'s 12/12/8 buckets in file order (short bucket
      first); the longest prompt is 2048.

    The bucket order is part of the input on purpose: admission is greedy and
    non-preempting, so a fill-only run that meets the short bucket first admits more
    requests than the same multiset in long-first order.
    """
    if name == "uniform-2048":
        return [(2048, budget)] * 32
    if name == "mild-mixed":
        return [(512, budget)] * 16 + [(2048, budget)] * 16
    if name == "w3-like-max-2048":
        short = _spread(19, 64, 12)
        medium = _spread(256, 512, 12)
        long = _spread(1536, 2048, 8)
        return [(n, budget) for n in short + medium + long]
    raise ValueError(f"unknown distribution {name!r}; expected one of {DISTRIBUTIONS}")


# ---------------------------------------------------------------- trace input

TRACE_SOURCES = ("metrics-jsonl", "json-list")

#: D6 primary budget: the gate compares the paged capacity at this budget.
PRIMARY_BUDGET_MB = 1640.0
#: The E1 "allocator sim ±1" tolerance.
EXPECT_TOLERANCE = 1


@dataclass(frozen=True)
class Trace:
    """``[(prompt_len, budget)]`` in arrival order, plus where it came from."""

    requests: list[tuple[int, int]]
    label: str
    source: str  # one of TRACE_SOURCES

    @property
    def n_requests(self) -> int:
        return len(self.requests)


def _non_negative_int(value: object, field: str, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"trace {label!r}: {field} must be an integer, got {value!r}")
    number = int(value)
    if number != value or number < 0:
        raise ValueError(f"trace {label!r}: {field} must be a non-negative integer, got {value!r}")
    return number


def _looks_like_metrics_row(item: dict) -> bool:
    return "row_kind" in item or ("context" in item and "out_tokens" in item)


def _requests_from_metrics_rows(rows: list[dict], *, label: str) -> list[tuple[int, int]]:
    """``metrics.jsonl`` request rows -> ``[(context - out_tokens, budget)]``.

    Only ``row_kind == "request"`` rows count; a row without ``row_kind`` (the pre-W3
    metrics files) is treated as a request row when it carries ``context`` /
    ``out_tokens``. A run repeats the trace several times, so only the **smallest
    ``repeat``** is kept: repeats of the same trace are not extra requests. Rows are
    then ordered by ``arrival_idx`` when the field is present.
    """
    request_rows = [
        row
        for row in rows
        if row.get("row_kind") == "request"
        or (row.get("row_kind") is None and "context" in row and "out_tokens" in row)
    ]
    if not request_rows:
        raise ValueError(f"trace {label!r}: no row_kind='request' rows with context/out_tokens")
    missing = [
        index
        for index, row in enumerate(request_rows)
        if "context" not in row or "out_tokens" not in row or "budget" not in row
    ]
    if missing:
        raise ValueError(
            f"trace {label!r}: request row {missing[0]} lacks context/out_tokens/budget; "
            "the trace needs prompt_len = context - out_tokens and the budget"
        )
    first_repeat = min(int(row.get("repeat") or 0) for row in request_rows)
    chosen = [row for row in request_rows if int(row.get("repeat") or 0) == first_repeat]
    if all("arrival_idx" in row for row in chosen):
        chosen = sorted(chosen, key=lambda row: int(row["arrival_idx"]))

    requests: list[tuple[int, int]] = []
    for row in chosen:
        context = _non_negative_int(row["context"], "context", label=label)
        out_tokens = _non_negative_int(row.get("out_tokens") or 0, "out_tokens", label=label)
        prompt = context - out_tokens
        if prompt < 0:
            raise ValueError(
                f"trace {label!r}: out_tokens {out_tokens} exceeds context {context}"
            )
        budget = _non_negative_int(row["budget"], "budget", label=label)
        requests.append((prompt, budget))
    return requests


def _parse_trace_items(items: list, *, label: str) -> list[tuple[int, int]]:
    if items and any(isinstance(item, dict) and _looks_like_metrics_row(item) for item in items):
        return _requests_from_metrics_rows(
            [item for item in items if isinstance(item, dict)], label=label
        )
    requests: list[tuple[int, int]] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            prompt_raw, budget_raw = item.get("prompt_len"), item.get("budget")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            prompt_raw, budget_raw = item
        else:
            raise ValueError(
                f"trace {label!r}: item {index} is not [prompt_len, budget] or "
                f"{{'prompt_len': .., 'budget': ..}}: {item!r}"
            )
        requests.append(
            (
                _non_negative_int(prompt_raw, f"item {index} prompt_len", label=label),
                _non_negative_int(budget_raw, f"item {index} budget", label=label),
            )
        )
    return requests


def parse_trace_text(text: str, *, label: str = "trace") -> Trace:
    """Auto-detect ``metrics.jsonl`` vs a JSON list and return the trace.

    A file whose first non-space character is ``[`` is parsed as one JSON list
    (either ``[[prompt_len, budget], ...]`` or ``[{"prompt_len": .., "budget": ..}, ...]``,
    in the given order); anything else is parsed line by line as ``metrics.jsonl``
    (``row_kind == "request"`` rows, smallest ``repeat`` only, ordered by
    ``arrival_idx``). A JSON list of full metrics rows is also accepted.
    """
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"trace {label!r} is empty")
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trace {label!r}: invalid JSON list: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"trace {label!r}: a JSON trace must be a list")
        requests = _parse_trace_items(payload, label=label)
        source = "json-list"
    else:
        rows: list[dict] = []
        for lineno, line in enumerate(stripped.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trace {label!r}: line {lineno} is not JSON: {exc}") from exc
            if not isinstance(row, dict):
                continue
            rows.append(row)
        requests = _requests_from_metrics_rows(rows, label=label)
        source = "metrics-jsonl"
    if not requests:
        raise ValueError(f"trace {label!r} has no requests")
    return Trace(requests=requests, label=label, source=source)


def load_trace(path: str | Path, *, label: str | None = None) -> Trace:
    """Read a trace file; ``label`` defaults to the file's basename."""
    trace_path = Path(path)
    text = trace_path.read_text(encoding="utf-8")
    return parse_trace_text(text, label=label or trace_path.name)


def effective_seat_max_batch(
    seat_max_batch: int | None, requests: list[tuple[int, int]], *, trace_mode: bool
) -> int:
    """``--seat-max-batch`` when given; otherwise unbounded on a trace, 64 on synthetic.

    A trace replay must not cap the seat arm below what the budget holds (the request
    asks: ``floor(budget_mb / row_mb)`` unless the user explicitly passes a smaller
    ``--seat-max-batch``); the synthetic distributions keep their historical 64.
    """
    if seat_max_batch is not None:
        return int(seat_max_batch)
    return len(requests) if trace_mode else 64


def select_distributions(
    distributions: str | None, *, trace: Trace | None
) -> tuple[list[str], bool]:
    """Resolve ``--trace`` vs ``--distributions``: **the trace wins**.

    Returns ``(names, distributions_ignored)``. ``distributions=None`` means the flag
    was not passed and the default synthetic set is used (unless a trace is given).
    """
    if trace is not None:
        return [trace.label], distributions is not None
    text = distributions if distributions is not None else ",".join(DISTRIBUTIONS)
    return [name.strip() for name in text.split(",") if name.strip()], False


def seat_row_width(requests: list[tuple[int, int]]) -> int:
    """``bucket(max(prompt + budget + 1), 64)`` -- the KvSlab row width of the trace.

    The ``+ 1`` mirrors ``ContinuousEngine.run_trace`` (``need = prompt + budget + 1``)
    and the engine's 64-column bucket, so 2048 + 128 becomes 2240 and 1517 + 128
    becomes 1664 (``docs/w4-plan.md`` §4.1 / §4.2).
    """
    if not requests:
        raise ValueError("a distribution needs at least one request")
    widest = max(prompt + budget for prompt, budget in requests)
    return bucket_width(widest + 1)


# ---------------------------------------------------------------- seat model


@dataclass
class SeatResult:
    row_width: int
    row_mb: float
    capacity: int
    kv_mb: float  # rows actually reserved
    kv_total_mb: float  # seat has no workspace
    workspace_mb: float
    seat_unused_mb: float  # budget left over by row granularity (and max_batch)
    max_batch: int
    capacity_per_gib: float


def seat_capacity(
    budget_mb: float,
    requests: list[tuple[int, int]],
    *,
    shape: ModelShape | None = None,
    max_batch: int = 64,
) -> SeatResult:
    """How many KvSlab rows the budget holds, each reserved at the worst width.

    ``seat_unused_mb`` is the budget the rows cannot reach: what is left after
    ``capacity * row_mb``. It mixes both causes D6 lists -- a whole row that does not
    fit (row granularity) and rows beyond ``max_batch`` -- and is reported, never
    handed to the paged arm (C1 compares upper bounds, not actual use).
    """
    shape = shape or ModelShape()
    row_width = seat_row_width(requests)
    row_mb = row_width * shape.kv_bytes_per_token / MIB
    by_budget = int(math.floor((budget_mb + 1e-12) / row_mb)) if row_mb > 0 else 0
    capacity = max(0, min(by_budget, max_batch, len(requests)))
    kv_mb = capacity * row_mb
    return SeatResult(
        row_width=row_width,
        row_mb=row_mb,
        capacity=capacity,
        kv_mb=kv_mb,
        kv_total_mb=kv_mb,
        workspace_mb=0.0,
        seat_unused_mb=max(0.0, budget_mb - kv_mb),
        max_batch=max_batch,
        capacity_per_gib=(capacity / (kv_mb / 1024.0)) if kv_mb > 0 else 0.0,
    )


# ---------------------------------------------------------------- paged model


@dataclass
class PagedResult:
    block_size: int
    block_mb: float
    n_blocks: int
    capacity: int
    reserved_blocks: int
    written_tokens: int
    kv_mb: float  # pool reservation (n_blocks)
    kv_live_mb: float  # prompt tokens written (fill-only)
    workspace_mb: float
    kv_total_mb: float
    internal_frag: float
    reservation_slack: float
    lpad: int
    max_batch: int
    capacity_per_gib: float
    workspace_auto: bool
    blocks_fixed: bool = False  # True when n_blocks came from --paged-blocks
    over_budget: bool = False  # kv_total_mb > budget_mb
    over_by_mb: float = 0.0


def _fill(
    requests: list[tuple[int, int]], n_blocks: int, block_size: int, limit: int | None = None
) -> tuple[list[tuple[int, int]], BlockManager, int]:
    """Fill-only FCFS admission; returns ``(admitted, manager, written)``.

    Requests arrive in trace order and each reserves ``blocks_for(prompt, budget)``
    atomically. The first request that does not fit stops admission: with D6's
    ``can_admit`` the request stays at the queue head, and a fill-only run decodes
    nothing, so nothing is ever released to let a later, shorter request in. The
    capacity is therefore a prefix length, and the input order is part of the input.
    ``limit`` caps the count at the solved capacity: admitting one more row would
    widen the workspace, and the pool cannot grow after the fact (D6/R4).

    The manager is built with ``min(n_blocks, demand)``: a fill-only run can never
    reserve more than the trace's own demand, so the extra ids are unobservable and
    materialising them (a 1 TiB test budget is ~1e9 ids) is pure cost. ``n_blocks``
    reported by :func:`paged_capacity` is still the budget-derived pool size.
    """
    demand = sum(
        max(1, -(-(prompt + budget) // block_size)) for prompt, budget in requests
    )
    manager = BlockManager(max(1, min(n_blocks, demand)), block_size)
    if n_blocks < 1 or limit == 0:
        return [], manager, 0
    admitted: list[tuple[int, int]] = []
    written = 0
    for prompt, budget in requests:
        if limit is not None and len(admitted) >= limit:
            break
        blocks = manager.blocks_for(prompt, budget)
        if manager.reserve(blocks) is None:
            break  # head-of-line blocking: no preemption, no release during fill
        admitted.append((prompt, budget))
        written += prompt  # fill-only: prefill writes the prompt, decode never runs
    return admitted, manager, written


def paged_capacity(
    budget_mb: float,
    requests: list[tuple[int, int]],
    *,
    block_size: int = 16,
    shape: ModelShape | None = None,
    workspace_override_mb: float | None = None,
    max_batch: int = 64,
    lpad: int | None = None,
    paged_blocks: int | None = None,
) -> PagedResult:
    """Fill-only paged capacity with the workspace paid out of the same budget.

    ``workspace_override_mb=None`` solves the fixed point: more admitted rows need a
    wider workspace, which leaves fewer blocks, which can only lower the admission
    count. ``workspace_mb`` increases monotonically with the admitted count and
    ``fill(n_blocks)`` is monotone in ``n_blocks``, so a binary search over the
    count is exact.

    ``paged_blocks`` pins the pool size instead of deriving it from the budget -- the
    E1 case, where the real engine runs with a config's fixed ``kv_blocks``. No
    fixed-point search is needed then: the FIFO fill runs on exactly those blocks and
    the workspace is charged for the admitted count. ``kv_total_mb`` is reported as
    ``paged_blocks * block_mb + workspace`` even when that exceeds ``budget_mb``
    (flagged ``over_budget`` and still giving a capacity: the point of the probe is
    to see whether the fixed pool overruns the budget).
    """
    shape = shape or ModelShape()
    if not requests:
        raise ValueError("a distribution needs at least one request")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if paged_blocks is not None and paged_blocks < 1:
        raise ValueError(f"paged_blocks must be >= 1, got {paged_blocks}")
    block_bytes = block_size * shape.kv_bytes_per_token
    block_mb = block_bytes / MIB
    lpad = lpad or seat_row_width(requests)
    auto = workspace_override_mb is None

    def workspace_for(n_active: int) -> float:
        if workspace_override_mb is not None:
            return float(workspace_override_mb)
        return workspace_mb(n_active=n_active, lpad=lpad, max_batch=max_batch, shape=shape)

    def blocks_for_workspace(workspace: float) -> int:
        free_mb = max(0.0, budget_mb - workspace)
        return max(0, int(math.floor(free_mb * MIB / block_bytes)))

    if paged_blocks is not None:
        # Fixed pool (E1): no binary search -- the fill on these blocks *is* the answer.
        admitted, manager, written = _fill(requests, paged_blocks, block_size)
        capacity = len(admitted)
        n_blocks = int(paged_blocks)
    else:
        upper = min(len(requests), max_batch)

        def admitted_with(n_active: int) -> int:
            n_blocks = blocks_for_workspace(workspace_for(n_active))
            admitted, _manager, _written = _fill(requests, n_blocks, block_size)
            return len(admitted)

        # Largest c with fill(blocks(c)) >= c; the predicate is monotone in c.
        low, high = 0, upper
        while low < high:
            middle = (low + high + 1) // 2
            if admitted_with(middle) >= middle:
                low = middle
            else:
                high = middle - 1
        capacity = low

        workspace = workspace_for(capacity)
        n_blocks = blocks_for_workspace(workspace)
        admitted, manager, written = _fill(requests, n_blocks, block_size, limit=capacity)
        if len(admitted) != capacity:  # defensive: the fixed point above should guarantee it
            raise RuntimeError(
                f"paged fixed point drifted: solved {capacity}, filled {len(admitted)}"
            )

    workspace = workspace_for(capacity)
    reserved_blocks = manager.stats()["in_use"]

    # Two different instants, both from D6 (the report must not mix them):
    # * internal frag (C2) is evaluated at the state where every admitted request has
    #   written the whole prompt + budget it reserved, so only the last block's tail is
    #   wasted -- that is exactly "written tokens vs allocated block capacity";
    # * reservation slack is evaluated at the fill-only instant, where the prompt is
    #   written and the decode budget is still an unwritten reservation.
    reserved_tokens = reserved_blocks * block_size
    written_full = sum(prompt + budget for prompt, budget in admitted)
    frag = FragStats(block_size)
    frag.observe(1.0, written_full, reserved_blocks)
    internal = frag.internal_frag() or 0.0
    slack = frag.reservation_slack(reserved_tokens, written) or 0.0

    kv_mb = n_blocks * block_mb
    kv_live_mb = written * shape.kv_bytes_per_token / MIB
    kv_total_mb = kv_mb + workspace
    over_by = max(0.0, kv_total_mb - budget_mb)
    return PagedResult(
        block_size=block_size,
        block_mb=block_mb,
        n_blocks=n_blocks,
        capacity=capacity,
        reserved_blocks=reserved_blocks,
        written_tokens=written,
        kv_mb=kv_mb,
        kv_live_mb=kv_live_mb,
        workspace_mb=workspace,
        kv_total_mb=kv_total_mb,
        internal_frag=internal,
        reservation_slack=slack,
        lpad=lpad,
        max_batch=max_batch,
        capacity_per_gib=(capacity / (kv_total_mb / 1024.0)) if kv_total_mb > 0 else 0.0,
        workspace_auto=auto,
        blocks_fixed=paged_blocks is not None,
        over_budget=over_by > 1e-9,
        over_by_mb=over_by,
    )


def capacity_ratio(seat: SeatResult, paged: PagedResult) -> float | None:
    """``paged_capacity / seat_capacity`` on integer admission counts."""
    if seat.capacity <= 0:
        return None
    return paged.capacity / seat.capacity


# ---------------------------------------------------------------- sweep


def run_sweep(
    budgets: list[float],
    distributions: list[str],
    block_sizes: list[int],
    *,
    shape: ModelShape,
    workspace_override_mb: float | None,
    seat_max_batch: int | None,
    paged_max_batch: int,
    budget_tokens: int = BUDGET_TOKENS,
    trace: Trace | None = None,
    paged_blocks: int | None = None,
) -> list[dict]:
    """Sweep budgets x (trace | named distributions) x block sizes.

    ``trace`` replaces the synthetic generators: its request list (and only it) is
    replayed, the distribution name is the trace label, ``seat_max_batch`` defaults to
    the trace length (no cap), and every row carries ``source`` + ``n_requests``.
    ``paged_blocks`` pins the paged pool to that many blocks (E1: the config's fixed
    ``kv_blocks``) instead of deriving it from the budget; the row then also carries
    ``kv_blocks``, ``blocks_fixed`` and the over-budget flag.
    """
    rows: list[dict] = []
    for budget in budgets:
        for name in distributions:
            if trace is not None:
                requests = list(trace.requests)
            else:
                requests = build_requests(name, budget=budget_tokens)
            seat_cap = effective_seat_max_batch(
                seat_max_batch, requests, trace_mode=trace is not None
            )
            seat = seat_capacity(
                budget, requests, shape=shape, max_batch=seat_cap
            )
            for block_size in block_sizes:
                paged = paged_capacity(
                    budget,
                    requests,
                    block_size=block_size,
                    shape=shape,
                    workspace_override_mb=workspace_override_mb,
                    max_batch=paged_max_batch,
                    paged_blocks=paged_blocks,
                )
                rows.append(
                    {
                        "budget_mb": float(budget),
                        "distribution": name,
                        "source": "trace" if trace is not None else "distribution",
                        "trace_source": trace.source if trace is not None else None,
                        "n_requests": len(requests),
                        "kv_blocks": paged.n_blocks,
                        "blocks_fixed": paged.blocks_fixed,
                        "over_budget": paged.over_budget,
                        "seat": asdict(seat),
                        "paged": asdict(paged),
                        "ratio": capacity_ratio(seat, paged),
                    }
                )
    return rows


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'budget':>7} {'distribution':<18} {'bs':>3} "
        f"{'seat':>5} {'paged':>6} {'ratio':>6} {'frag%':>7} "
        f"{'slack%':>7} {'cap/GiB':>8} {'ws MiB':>8} {'blocks':>7} {'source':>12} {'n':>4}"
    )
    lines = [
        "simulated (analytic; no model, no arrivals) -- kv_total_mb = kv_mb + workspace_mb (D6)",
        "frag% = written(prompt+budget) / allocated blocks; slack% = 1 - prompt / allocated "
        "blocks at the fill-only instant; cap/GiB = paged capacity / (kv_total_mb / 1024)",
        "source = 'distribution' (synthetic) or 'trace' (real prompt/budget sequence replayed)",
        "blocks = paged pool size (derived from the budget unless --paged-blocks is given; "
        "'*' marks a fixed pool, OVER marks kv_total_mb > budget, still with a capacity)",
        header,
        "-" * len(header),
    ]
    for row in rows:
        seat, paged = row["seat"], row["paged"]
        ratio = row["ratio"]
        source = row.get("source") or "distribution"
        blocks = f"{paged['n_blocks']}{'*' if paged.get('blocks_fixed') else ''}"
        if paged.get("over_budget"):
            blocks += " OVER"
        lines.append(
            f"{row['budget_mb']:>7.0f} {row['distribution']:<18} {paged['block_size']:>3} "
            f"{seat['capacity']:>5} {paged['capacity']:>6} "
            f"{('%.2f' % ratio) if ratio is not None else 'n/a':>6} "
            f"{100 * paged['internal_frag']:>7.2f} "
            f"{100 * paged['reservation_slack']:>7.2f} "
            f"{paged['capacity_per_gib']:>8.2f} {paged['workspace_mb']:>8.1f} "
            f"{blocks:>7} {source:>12} {row['n_requests']:>4}"
        )
    return "\n".join(lines)


def concurrency_gate(
    simulated: int, expected: int, *, tolerance: int = EXPECT_TOLERANCE
) -> dict:
    """The E1 gate: does the simulated paged capacity sit within ``±tolerance`` of N?

    ``simulated`` is the paged capacity **of the current mode**: with
    ``--paged-blocks`` it is the capacity on that fixed block count (the real run's
    ``kv_blocks``); without it, the budget-derived capacity. Mixing the two would
    compare an allocator the run never used against the run.
    """
    delta = int(simulated) - int(expected)
    return {
        "sim": int(simulated),
        "expected": int(expected),
        "delta": delta,
        "tolerance": int(tolerance),
        "gate": "PASS" if abs(delta) <= tolerance else "FAIL",
    }


def format_gate(result: dict) -> str:
    return (
        f"sim={result['sim']} expected={result['expected']} "
        f"delta={result['delta']} gate={result['gate']}"
        f"(±{result['tolerance']})"
    )


# ---------------------------------------------------------------- figure


def plot_capacity(rows: list[dict], out: Path, budgets: list[float]) -> None:
    """Ratio vs budget per distribution; bs16 solid, bs32 dashed; 1640 and 1.5x marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_2 = "#52514e"
    grid = "#e1e0d9"
    axis = "#c3c2b7"
    colors = {
        "uniform-2048": "#2a78d6",
        "mild-mixed": "#eb6834",
        "w3-like-max-2048": "#2f9e58",
    }
    primary_bs = 16
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # Synthetic distributions keep their fixed colours/order; a trace replay (whose label
    # is not in that list) is drawn last in the neutral ink.
    names_in_rows = {row["distribution"] for row in rows}
    ordered_names = [name for name in DISTRIBUTIONS if name in names_in_rows]
    ordered_names += sorted(names_in_rows - set(ordered_names))
    for name in ordered_names:
        for block_size in sorted({row["paged"]["block_size"] for row in rows}):
            points = [
                (row["budget_mb"], row["ratio"])
                for row in rows
                if row["distribution"] == name and row["paged"]["block_size"] == block_size
            ]
            points = [(x, y) for x, y in points if y is not None]
            if not points:
                continue
            style = "-" if block_size == primary_bs else "--"
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                style,
                color=colors.get(name, ink_2),
                linewidth=1.6 if block_size == primary_bs else 1.1,
                marker="o" if block_size == primary_bs else "s",
                markersize=4.0 if block_size == primary_bs else 3.2,
                markerfacecolor=colors.get(name, ink_2),
                markeredgecolor=surface,
                markeredgewidth=0.6,
                zorder=3,
                label=f"{name} · bs{block_size}",
            )

    ax.axvline(1640, color=ink_2, linestyle=":", linewidth=1.0, zorder=2)
    ax.text(1640, 0.02, " 1640 MiB", color=ink_2, fontsize=8.5, ha="left", va="bottom")
    ax.axhline(1.5, color=ink, linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        max(budgets),
        1.5,
        " 1.5× C1",
        color=ink,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )

    ax.set_xlim(min(budgets) * 0.97, max(budgets) * 1.03)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color=grid, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=ink_2, labelsize=8.8, length=2.5, width=0.7)
    ax.set_xlabel("total budget MiB (KV + workspace, D6)", fontsize=9.5, color=ink_2)
    ax.set_ylabel("capacity ratio · paged / seat", fontsize=9.5, color=ink_2)
    ax.set_title(
        "W4 capacity · simulated, not measured",
        fontsize=13.5,
        color=ink,
        loc="left",
        pad=10,
    )
    fig.text(
        0.012,
        0.955,
        "deterministic allocator simulation · Qwen3-0.6B shapes · fill-only · "
        "solid = bs16, dashed = bs32",
        fontsize=8.8,
        color=ink_2,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, fontsize=8.6, loc="upper left", handlelength=2.2)

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    try:
        fig.savefig(temporary, dpi=150, format="png", facecolor=surface, bbox_inches="tight")
        temporary.replace(out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")


# ---------------------------------------------------------------- CLI


def _parse_int_list(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_float_list(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulated W4 paged-vs-seat capacity (no model, no arrivals).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default=None, help="optional capacity-curve PNG path")
    parser.add_argument("--json", dest="json_path", default=None, help="optional JSON output")
    parser.add_argument("--block-sizes", default="16,32", help="comma-separated block sizes")
    parser.add_argument(
        "--budgets",
        default="1200,1456,1640,2400",
        help="comma-separated total budgets in MiB (KV + workspace)",
    )
    parser.add_argument(
        "--workspace-mb",
        type=float,
        default=None,
        help="override the Appendix C workspace estimate (single value, all rows)",
    )
    parser.add_argument(
        "--distributions",
        default=None,
        help=(
            "comma-separated distribution names (default: all synthetic ones; "
            "ignored when --trace is given)"
        ),
    )
    parser.add_argument(
        "--trace",
        default=None,
        metavar="PATH",
        help=(
            "replay a real trace instead of synthetic distributions: a run's "
            "metrics.jsonl (row_kind='request', smallest repeat, by arrival_idx) or a "
            "JSON list [[prompt_len, budget], ...] / [{'prompt_len':..,'budget':..}, ...]"
        ),
    )
    parser.add_argument(
        "--trace-label",
        default=None,
        help="display name for --trace (default: the trace file's basename)",
    )
    parser.add_argument(
        "--seat-max-batch",
        type=int,
        default=None,
        help=(
            "cap on the seat row count (default: 64 for synthetic distributions; "
            "uncapped -- floor(budget_mb / row_mb) -- for --trace)"
        ),
    )
    parser.add_argument("--paged-max-batch", type=int, default=64)
    parser.add_argument(
        "--paged-blocks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "pin the paged pool to N blocks instead of deriving it from the budget "
            "(the E1 case: the run's fixed kv_blocks). kv_total_mb = N*block_mb + "
            "workspace is reported as-is and marked OVER when it exceeds the budget, "
            "but a capacity is still given"
        ),
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=BUDGET_TOKENS,
        help="decode budget reserved per request (D6 reserves prompt + budget)",
    )
    parser.add_argument(
        "--expect-concurrency",
        type=int,
        default=None,
        metavar="N",
        help=(
            "compare the paged capacity at --expect-budget (default 1640 MiB, bs16) with N; "
            "prints 'sim=<x> expected=<N> delta=<x-N> gate=<PASS|FAIL>(±1)' and exits 1 "
            "when the delta exceeds ±1"
        ),
    )
    parser.add_argument(
        "--expect-budget",
        type=float,
        default=PRIMARY_BUDGET_MB,
        help="ignored unless --expect-concurrency is given",
    )
    parser.add_argument("--n-layer", type=int, default=28)
    parser.add_argument("--n-kv-head", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    args = parser.parse_args(argv)

    shape = ModelShape(
        n_layer=args.n_layer,
        n_kv_head=args.n_kv_head,
        head_dim=args.head_dim,
        n_head=args.n_head,
        vocab=args.vocab,
        dtype_bytes=args.dtype_bytes,
    )
    budgets = _parse_float_list(args.budgets)
    block_sizes = _parse_int_list(args.block_sizes)
    trace = load_trace(args.trace, label=args.trace_label) if args.trace else None
    distributions, ignored = select_distributions(args.distributions, trace=trace)
    if ignored:
        print(
            f"note: --distributions ignored, replaying trace {trace.label!r} "
            f"({trace.n_requests} requests, source={trace.source})",
            file=sys.stderr,
        )
    if trace is None:
        for name in distributions:
            if name not in DISTRIBUTIONS:
                parser.error(f"unknown distribution {name!r}")
    if trace is not None and args.budget_tokens != BUDGET_TOKENS:
        print(
            "note: --budget-tokens ignored on a trace (each request carries its own budget)",
            file=sys.stderr,
        )

    rows = run_sweep(
        budgets,
        distributions,
        block_sizes,
        shape=shape,
        workspace_override_mb=args.workspace_mb,
        seat_max_batch=args.seat_max_batch,
        paged_max_batch=args.paged_max_batch,
        budget_tokens=args.budget_tokens,
        trace=trace,
        paged_blocks=args.paged_blocks,
    )
    print(format_table(rows))

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "simulated",
            "note": (
                "deterministic allocator simulation (scripts/sim_paged_capacity.py); "
                "no model, no arrivals, not a measurement"
            ),
            "rule": "kv_total_mb = kv_mb + workspace_mb (D6)",
            "admission": (
                "fill-only FCFS prefix: the first request that does not fit stops admission "
                "(head-of-line blocking, no preemption); capacity is order-dependent"
            ),
            "internal_frag_instant": "prompt + budget written per admitted request",
            "reservation_slack_instant": "prompt written, budget still reserved (fill-only)",
            "model": asdict(shape),
            "kv_bytes_per_token": shape.kv_bytes_per_token,
            "budget_tokens": args.budget_tokens,
            "workspace_override_mb": args.workspace_mb,
            "seat_max_batch": args.seat_max_batch,
            "paged_max_batch": args.paged_max_batch,
            "paged_blocks": args.paged_blocks,
            "trace": (
                {
                    "path": str(args.trace),
                    "label": trace.label,
                    "source": trace.source,
                    "n_requests": trace.n_requests,
                    "requests": [list(item) for item in trace.requests],
                }
                if trace is not None
                else None
            ),
            "rows": rows,
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {output}")

    if args.out:
        plot_capacity(rows, Path(args.out), budgets)

    if args.expect_concurrency is not None:
        primary_bs = 16 if 16 in block_sizes else min(block_sizes)
        candidates = [
            row
            for row in rows
            if row["paged"]["block_size"] == primary_bs
            and abs(row["budget_mb"] - args.expect_budget) < 1e-6
        ]
        if not candidates:
            parser.error(
                f"--expect-concurrency needs a row at budget {args.expect_budget:g} MiB "
                f"with block size {primary_bs}; pass --budgets/--block-sizes accordingly"
            )
        result = concurrency_gate(
            candidates[0]["paged"]["capacity"], args.expect_concurrency
        )
        result.update(
            budget_mb=args.expect_budget,
            block_size=primary_bs,
            distribution=candidates[0]["distribution"],
            source=candidates[0]["source"],
            kv_blocks=candidates[0]["kv_blocks"],
            blocks_fixed=bool(candidates[0]["blocks_fixed"]),
        )
        mode = (
            f"fixed pool: {candidates[0]['kv_blocks']} blocks (--paged-blocks)"
            if candidates[0]["blocks_fixed"]
            else f"budget-derived pool: {candidates[0]['kv_blocks']} blocks"
        )
        print(
            f"gate subject: {mode}, budget {args.expect_budget:g} MiB, bs{primary_bs}, "
            f"trace/distribution {candidates[0]['distribution']!r}"
        )
        print(format_gate(result))
        if result["gate"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
