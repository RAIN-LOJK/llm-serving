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

    On CUDA this asks nvidia-smi for the current process; on MPS it uses the driver
    allocation counter; on CPU it falls back to RSS.
    """
    try:
        if torch.cuda.is_available():
            import os
            import subprocess

            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                total = 0.0
                for line in proc.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[0] == str(os.getpid()):
                        total += float(parts[1])
                return total or None
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
