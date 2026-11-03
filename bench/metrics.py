"""Timing, memory, and aggregation helpers.

Rules encoded here (see docs/experiments.md):
- never report a single number without n / median / p95 / CV;
- synchronize the accelerator before reading a wall clock;
- keep device-specific memory sources separate instead of pretending they are equal.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

import torch


def now() -> float:
    """Monotonic wall clock in seconds."""
    return time.perf_counter()


def sync(device: str | None = None) -> None:
    """Synchronize the accelerator that actually owns ``device``."""
    dev = (device or "").lower()
    try:
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        # Timing must not fail because a backend refuses to synchronize.
        pass


def elapsed_ms(start: float, end: float | None = None) -> float:
    return ((now() if end is None else end) - start) * 1000.0


class Timer:
    """Small helper: ``stop`` synchronizes before reading the clock."""

    def __init__(self, device: str | None = None, kind: str = "auto") -> None:
        self.device = device
        self.kind = "perf_counter" if kind == "auto" else kind
        self._start: float | None = None

    def start(self) -> float:
        sync(self.device)
        self._start = now()
        return self._start

    def stop(self, start: float | None = None) -> float:
        sync(self.device)
        return elapsed_ms(self._start if start is None else start)

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._last = self.stop()


class CudaEventTimer:
    """Timing on the GPU timeline; ``stop`` synchronizes before reading the delta.

    Same ``start()/stop()`` interface as :class:`Timer`. Refuses to construct without CUDA
    instead of silently falling back to a wall clock — the whole point of ``cuda_event`` is
    that a wall clock would include host-side launch gaps.
    """

    kind = "cuda_event"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("timer='cuda_event' needs CUDA, but torch.cuda is unavailable")
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def start(self) -> "CudaEventTimer":
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def stop(self) -> float:
        if self._start is None or self._end is None:
            raise RuntimeError("stop() called before start()")
        self._end.record()
        torch.cuda.synchronize()
        return self._start.elapsed_time(self._end)


def make_timer(device: str | None = None, kind: str = "auto") -> Timer | CudaEventTimer:
    """Pick the timer the measure protocol asks for.

    ``auto`` resolves from the device actually in use: CUDA gets events, everything else
    gets ``perf_counter`` + an explicit backend sync (MPS/CPU).
    """
    resolved = kind
    if resolved == "auto":
        resolved = "cuda_event" if str(device or "").lower().startswith("cuda") else "perf_counter"
    if resolved == "perf_counter":
        return Timer(device, kind="perf_counter")
    if resolved == "cuda_event":
        return CudaEventTimer()
    raise ValueError(f"unknown timer kind {kind!r}; choose auto/perf_counter/cuda_event")


def current_memory_mb() -> float | None:
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 2**20
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / 2**20
    except Exception:
        return None
    return rss_mb()


def peak_memory_mb() -> float | None:
    """Best available *process* peak memory for the active backend."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / 2**20
    except Exception:
        return None
    return rss_mb()


def rss_mb() -> float | None:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KiB.
        return value / 2**20 if sys.platform == "darwin" else value / 1024.0
    except Exception:
        return None


def reset_peak() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def driver_memory_mb() -> float | None:
    """Driver-level memory, kept separate from the torch allocator number.

    On CUDA this is ``cudaMemGetInfo`` through ``torch.cuda.mem_get_info``: used = total -
    free, i.e. the device-level number that includes the CUDA context (~0.5 GB here). It is
    the same axis the old ``nvidia-smi`` subprocess query measured, at ~8 us instead of a
    50-90 ms fork+exec that landed *inside* the measured window (P-27). The trade is that
    it is device-wide, not per-process: a run sharing the GPU inflates it, so official
    runs stay exclusive. Only ever a cross-check, never the memory denominator
    (configs/thresholds.toml).

    On MPS it uses the driver allocation counter; on CPU it falls back to RSS.
    """
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return (total_bytes - free_bytes) / 2**20
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / 2**20
    except Exception:
        return None
    return rss_mb()


def summarize(values: Iterable[float | None]) -> dict[str, float | int]:
    """Median + p95 + min/max + CV. ``n`` is the count of finite values."""
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    n = len(vals)
    if n == 0:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None, "mean": None, "cv": None}
    mean = sum(vals) / n
    if n == 1:
        median = vals[0]
        p95 = vals[0]
        cv = 0.0
    else:
        median = statistics.median(vals)
        p95 = vals[min(n - 1, max(0, math.ceil(0.95 * n) - 1))]
        cv = statistics.pstdev(vals) / mean if mean else 0.0
    return {
        "n": n,
        "median": median,
        "p95": p95,
        "min": vals[0],
        "max": vals[-1],
        "mean": mean,
        "cv": cv,
    }


def percentiles(
    values: Iterable[float | None],
    qs: Sequence[float] = (0.50, 0.95, 0.99),
) -> dict[str, float | None]:
    """Nearest-rank percentiles keyed ``p50``/``p95``/``p99``.

    Same rank rule as :func:`summarize` (``ceil(q*n)``, 1-based, clamped), so a request-level
    p95 and the p95 inside a summary mean the same thing. ``None``/non-finite entries are
    ignored; an empty sample returns ``None`` for every key instead of raising.
    """
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    n = len(vals)
    out: dict[str, float | None] = {}
    for q in qs:
        qf = float(q)
        if not 0.0 <= qf <= 1.0:
            raise ValueError(f"percentile q must be within [0, 1], got {q!r}")
        key = f"p{round(qf * 100)}"
        out[key] = None if n == 0 else vals[min(n - 1, max(0, math.ceil(qf * n) - 1))]
    return out


def repeat_stability(
    rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str] = ("ttft_ms", "tpot_ms"),
) -> dict[str, dict[str, Any]]:
    """Repeat-to-repeat CV, computed *inside* a length bucket.

    ``overall`` CV is meaningless for W2: the prompt set spans 16..2048 tokens, so TTFT
    varies by an order of magnitude across buckets no matter how steady the machine is.
    A 4% jitter plus a 40x length spread still gives a scary-looking overall CV.

    Values are grouped per bucket, then per prompt (or context, for synthetic workloads),
    and each prompt contributes one CV over its repeats. Buckets report the median and the
    worst prompt so one pathological prompt stays visible without deciding the verdict.
    """
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        bucket = str(row.get("bucket") or f"ctx{row.get('context')}")
        group = str(row.get("prompt_id") or bucket)
        buckets.setdefault(bucket, {}).setdefault(group, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for bucket, groups in buckets.items():
        entry: dict[str, Any] = {"n_prompts": len(groups)}
        for metric in metric_names:
            cvs = []
            for group in groups.values():
                values = [row.get(metric) for row in group]
                if sum(value is not None and math.isfinite(float(value)) for value in values) < 2:
                    continue
                cv = summarize(values)["cv"]
                if cv is not None:
                    cvs.append(float(cv))
            entry[f"{metric}_cv_median"] = statistics.median(cvs) if cvs else None
            entry[f"{metric}_cv_max"] = max(cvs) if cvs else None
            entry[f"{metric}_n"] = len(cvs)
        out[bucket] = entry
    return out


def per_prompt_summary(
    rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str] = ("ttft_ms", "tpot_ms"),
) -> list[dict[str, Any]]:
    """One entry per (model, prompt): the unit a paired baseline can be matched on."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("model")), str(row.get("prompt_id") or f"ctx{row.get('context')}"))
        groups.setdefault(key, []).append(row)

    out = []
    for (model, prompt), subset in sorted(groups.items()):
        out.append(
            {
                "model": model,
                "prompt": prompt,
                "bucket": subset[0].get("bucket"),
                "n": len(subset),
                **{name: summarize([row.get(name) for row in subset]) for name in metric_names},
            }
        )
    return out


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def device_class(snapshot: dict[str, Any]) -> str:
    torch_info = snapshot.get("torch", {})
    if torch_info.get("cuda_available"):
        return "measure-cuda"
    if torch_info.get("mps_available"):
        return "dev-mps"
    return "cpu"


def key_metric(summary: dict[str, Any]) -> tuple[str, float | None]:
    """Pick a compact number for the index table."""
    overall = summary.get("overall", {})
    for name in ("ttft_ms", "e2e_ms", "peak_mem_mb"):
        agg = overall.get(name) or {}
        if agg.get("median") is not None:
            return name, float(agg["median"])
    return "—", None


def as_list(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]
