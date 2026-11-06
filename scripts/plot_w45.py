#!/usr/bin/env python3
"""Generate the two W4.5 evidence figures from committed run summaries.

Inputs (read-only; every measured value comes from these files):

* ``results/W45/0004-seat-b8-cost/summary.json`` -- E2 seat baseline;
* ``results/W45/0005-paged-b8-cost/summary.json`` -- E2 paged pre-fix;
* ``results/W45/0012-paged-b8-cost-fixed/summary.json`` -- E2 paged post-fix,
  which supersedes ``W45-0005`` but is shown alongside it as before/after evidence;
* ``results/W45/0006-static-serial-prefill/summary.json`` -- E3 serial prefill;
* ``results/W45/0007-static-packed-prefill/summary.json`` -- E3 packed prefill.

Outputs (``figures/`` by default):

* ``figures/fig_w4_5_decode.png``  -- steady-state decode step at ``active == 8``
  with the 1.15x acceptance line and ratios against the seat baseline;
* ``figures/fig_w4_5_prefill.png`` -- TTFT p50/p99 for the serial and packed arms,
  with the p99 speedup and p50 ratio computed from the summaries.

All inputs are validated and both figures are rendered to memory before anything is
written, so a missing field or file never leaves a partial figure behind. Text is
English because the machine has no CJK font in matplotlib.

Usage::

    python scripts/plot_w45.py
    python scripts/plot_w45.py --results results --out-dir /tmp/w45-figs-check
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

# --- style (same conventions as scripts/plot_w4.py) -------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#2f9e58"
RED = "#c23b22"
NEUTRAL = "#8a8984"

ACTIVE = 8
DECODE_ACCEPTANCE = "decode_step_ratio_max"

DECODE_RUNS = {
    "seat": ("W45/0004-seat-b8-cost", "W45-0004"),
    "pre": ("W45/0005-paged-b8-cost", "W45-0005"),
    "post": ("W45/0012-paged-b8-cost-fixed", "W45-0012"),
}
PREFILL_RUNS = {
    "serial": ("W45/0006-static-serial-prefill", "W45-0006"),
    "packed": ("W45/0007-static-packed-prefill", "W45-0007"),
}

DECODE_NAME = "fig_w4_5_decode.png"
PREFILL_NAME = "fig_w4_5_prefill.png"


def fail(message: str) -> NoReturn:
    """Abort with one clear, prefixed message; no figure is written."""
    raise SystemExit(f"plot_w45: {message}")


def _read_summary(results: Path, relative: str, expected_id: str) -> dict[str, Any]:
    path = results / relative / "summary.json"
    if not path.is_file():
        fail(f"missing input: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    run_id = summary.get("run_id")
    if run_id != expected_id:
        fail(f"{path}: run_id is {run_id!r}, expected {expected_id!r}")
    return summary


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"missing or non-numeric value: {where}")
    number = float(value)
    if not math.isfinite(number):
        fail(f"non-finite value: {where}")
    return number


def _active_step_ms(summary: dict[str, Any], run_id: str, active: int = ACTIVE) -> float:
    buckets = (summary.get("steps") or {}).get("by_active")
    if not isinstance(buckets, list):
        fail(f"{run_id}: summary.json has no steps.by_active list")
    matches = [
        bucket
        for bucket in buckets
        if isinstance(bucket, dict) and bucket.get("active") == active
    ]
    if len(matches) != 1:
        fail(
            f"{run_id}: expected exactly one steps.by_active bucket with "
            f"active == {active}, found {len(matches)}"
        )
    return _number(matches[0].get("step_ms"), f"{run_id} steps.by_active[active={active}].step_ms")


def _decode_acceptance(summary: dict[str, Any], run_id: str) -> dict[str, Any]:
    for entry in summary.get("acceptance") or []:
        if isinstance(entry, dict) and entry.get("name") == DECODE_ACCEPTANCE:
            return entry
    fail(f"{run_id}: summary.json has no {DECODE_ACCEPTANCE!r} acceptance entry")


def _ttft(summary: dict[str, Any], run_id: str, percentile: str) -> float:
    latency = summary.get("latency") or {}
    ttft = latency.get("ttft_ms") or {}
    return _number(ttft.get(percentile), f"{run_id} latency.ttft_ms.{percentile}")


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=INK_2, labelsize=9.0, length=2.5, width=0.7)


def _header(fig, title: str, subtitle: str, note: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.012, 0.985, title, fontsize=13.5, color=INK, ha="left", va="top")
    fig.text(0.012, 0.930, subtitle, fontsize=8.6, color=INK_2, ha="left", va="top")
    fig.text(0.012, 0.028, note, fontsize=8.3, color=INK_2, ha="left", va="bottom")


def _decode_figure(
    seat_ms: float,
    pre_ms: float,
    post_ms: float,
    target: float,
    pre_status: str,
    post_status: str,
):
    pre_ratio = pre_ms / seat_ms
    post_ratio = post_ms / seat_ms
    acceptance_ms = target * seat_ms

    fig, ax = plt.subplots(figsize=(10.6, 5.9), dpi=150)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.790, bottom=0.205)
    _style_axis(ax)

    labels = [
        "seat baseline\nW45-0004",
        "paged pre-fix\nW45-0005",
        "paged post-fix\nW45-0012",
    ]
    values = [seat_ms, pre_ms, post_ms]
    colors = [NEUTRAL, RED, GREEN]
    inside = ["baseline", f"{pre_ratio:.3f}× · {pre_status.upper()}", f"{post_ratio:.3f}× · {post_status.upper()}"]

    ax.bar(range(3), values, width=0.56, color=colors, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, max(max(values), acceptance_ms) * 1.18)
    ax.set_ylabel("decode step (ms)", fontsize=9.2, color=INK_2)

    for x, value, text in zip(range(3), values, inside):
        ax.annotate(
            f"{value:.3f} ms",
            xy=(x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.3,
            color=INK,
            zorder=6,
        )
        ax.text(
            x,
            value * 0.55,
            text,
            ha="center",
            va="center",
            fontsize=8.8,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    ax.axhline(acceptance_ms, color=INK_2, linestyle="--", linewidth=1.0, zorder=5)
    ax.text(
        2.52,
        acceptance_ms + 0.28,
        f"{target:.2f}× acceptance = {acceptance_ms:.3f} ms",
        ha="right",
        va="bottom",
        fontsize=8.6,
        color=INK_2,
        zorder=6,
    )

    _header(
        fig,
        "W4.5 E2 · decode step at active = 8 (ms)",
        "Steady-state step_ms, prefill steps excluded · "
        "W45-0012 supersedes W45-0005; both are shown as before/after evidence",
        f"Post-fix {post_ratio:.3f}× is acceptance-pass, not target-pass: "
        f"{post_ms:.3f} ms remains above the 1.00× target ({seat_ms:.3f} ms).",
    )
    return fig


def _prefill_figure(
    serial_p50: float,
    serial_p99: float,
    packed_p50: float,
    packed_p99: float,
):
    p99_speedup = serial_p99 / packed_p99
    p50_ratio = packed_p50 / serial_p50

    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=150)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.790, bottom=0.205)
    _style_axis(ax)

    positions = [0.0, 1.0]
    width = 0.32
    p50_values = [serial_p50, packed_p50]
    p99_values = [serial_p99, packed_p99]
    ax.bar(
        [position - width / 2 for position in positions],
        p50_values,
        width=width,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.0,
        label="TTFT p50",
        zorder=3,
    )
    ax.bar(
        [position + width / 2 for position in positions],
        p99_values,
        width=width,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=1.0,
        label="TTFT p99",
        zorder=3,
    )

    for x, value in zip(
        [position - width / 2 for position in positions] + [position + width / 2 for position in positions],
        p50_values + p99_values,
    ):
        ax.annotate(
            f"{value:.1f} ms",
            xy=(x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.1,
            color=INK,
            zorder=5,
        )

    ax.set_xticks(positions, ["serial\nW45-0006", "packed\nW45-0007"])
    ax.set_xlim(-0.60, 1.60)
    ax.set_ylim(0, max(serial_p99, packed_p99) * 1.16)
    ax.set_ylabel("TTFT (ms)", fontsize=9.2, color=INK_2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=8.8,
        labelcolor=INK_2,
    )

    ax.text(
        0.985,
        0.965,
        f"p99 speedup  {p99_speedup:.2f}×  (serial / packed)\n"
        f"p50 ratio    {p50_ratio:.3f}×  (packed / serial)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color=INK_2,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": SURFACE, "edgecolor": GRID, "alpha": 0.95},
        zorder=6,
    )

    _header(
        fig,
        "W4.5 E3 · TTFT serial vs packed prefill (ms)",
        "Static B=8 mixed workload · packed wins both the median and the tail · "
        "p50/p99 from summary.json",
        "No error bars: each arm has one committed summary.json; bars are TTFT p50 / p99 only.",
    )
    return fig


def _render(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        facecolor=SURFACE,
        bbox_inches="tight",
        metadata={"Software": "minillm/scripts/plot_w45.py"},
    )
    plt.close(fig)
    return buffer.getvalue()


def _write_all(out_dir: Path, payloads: list[tuple[str, bytes]]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    written: list[Path] = []
    try:
        for name, payload in payloads:
            handle = tempfile.NamedTemporaryFile(
                dir=out_dir, prefix=f".{name}.", suffix=".tmp", delete=False
            )
            try:
                handle.write(payload)
            finally:
                handle.close()
            staged.append((Path(handle.name), out_dir / name))
        for temporary, final in staged:
            os.chmod(temporary, 0o644)
            os.replace(temporary, final)
            written.append(final)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="results root (default: results)")
    parser.add_argument("--out-dir", default="figures", help="figure directory (default: figures)")
    args = parser.parse_args(argv)

    results = Path(args.results)
    out_dir = Path(args.out_dir)

    # Load and validate every input before rendering anything.
    seat = _read_summary(results, *DECODE_RUNS["seat"])
    pre = _read_summary(results, *DECODE_RUNS["pre"])
    post = _read_summary(results, *DECODE_RUNS["post"])
    serial = _read_summary(results, *PREFILL_RUNS["serial"])
    packed = _read_summary(results, *PREFILL_RUNS["packed"])

    seat_ms = _active_step_ms(seat, DECODE_RUNS["seat"][1])
    pre_ms = _active_step_ms(pre, DECODE_RUNS["pre"][1])
    post_ms = _active_step_ms(post, DECODE_RUNS["post"][1])

    pre_entry = _decode_acceptance(pre, DECODE_RUNS["pre"][1])
    post_entry = _decode_acceptance(post, DECODE_RUNS["post"][1])
    target = _number(post_entry.get("target"), f"{DECODE_RUNS['post'][1]} acceptance target")
    pre_target = pre_entry.get("target")
    if pre_target is not None and not math.isclose(
        _number(pre_target, f"{DECODE_RUNS['pre'][1]} acceptance target"), target, rel_tol=0.0, abs_tol=1e-12
    ):
        fail("the pre-fix and post-fix acceptance targets differ")

    pre_ratio = pre_ms / seat_ms
    post_ratio = post_ms / seat_ms
    pre_status = "pass" if pre_ratio <= target else "fail"
    post_status = "pass" if post_ratio <= target else "fail"
    for entry, status, run_id in ((pre_entry, pre_status, DECODE_RUNS["pre"][1]),
                                  (post_entry, post_status, DECODE_RUNS["post"][1])):
        reported = str(entry.get("status") or "").lower()
        if reported and reported != status:
            fail(
                f"{run_id}: acceptance status {reported!r} disagrees with the "
                f"step ratio ({status})"
            )

    serial_p50 = _ttft(serial, PREFILL_RUNS["serial"][1], "p50")
    serial_p99 = _ttft(serial, PREFILL_RUNS["serial"][1], "p99")
    packed_p50 = _ttft(packed, PREFILL_RUNS["packed"][1], "p50")
    packed_p99 = _ttft(packed, PREFILL_RUNS["packed"][1], "p99")

    decode_fig = _decode_figure(seat_ms, pre_ms, post_ms, target, pre_status, post_status)
    prefill_fig = _prefill_figure(serial_p50, serial_p99, packed_p50, packed_p99)

    decode_png = _render(decode_fig)
    prefill_png = _render(prefill_fig)
    written = _write_all(out_dir, [(DECODE_NAME, decode_png), (PREFILL_NAME, prefill_png)])

    print(
        f"wrote {written[0]} "
        f"(pre-fix {pre_ratio:.3f}× {pre_status}, post-fix {post_ratio:.3f}× {post_status}, "
        f"acceptance {target:.2f}×)"
    )
    print(
        f"wrote {written[1]} "
        f"(p99 speedup {serial_p99 / packed_p99:.2f}×, p50 ratio {packed_p50 / serial_p50:.3f}×)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
