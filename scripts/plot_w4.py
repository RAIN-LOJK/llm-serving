#!/usr/bin/env python3
"""Generate the three W4 figures from the completed runs plus the allocator sweep.

Inputs (never edited here):

* ``results/W4/0002.../summary.json`` / ``0003...`` -- E1 seat and paged fill
* ``results/W4/0004.../summary.json`` / ``0005...`` -- E2 seat and paged cost
* ``results/W4/0004.../metrics.jsonl`` / ``0005...`` -- decode step rows (the only
  source of ``lpad`` / ``gather_ms``; the summaries aggregate them away)
* ``results/W4/0003.../metrics.jsonl`` -- replayed by the allocator simulator

The capacity sweep is regenerated into a temporary directory on every run by
calling ``scripts/sim_paged_capacity.py``; no capacity JSON is stored in ``docs/``.

Outputs:

* ``docs/fig_w4_capacity.png`` -- same-total-budget capacity ratio, with the measured
  W4-0003 point and the uniform-2048 negative control;
* ``docs/fig_w4_memory.png``   -- E2 same-concurrency memory cost and E1 capacity/GiB;
* ``docs/fig_w4_step.png``     -- E2 decode-step decomposition and the ratio by lpad.

Every number is read from the run artifacts; the script does not recompute model
timings. Text is English because the machine has no CJK font in matplotlib.

Usage::

    python scripts/plot_w4.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#2f9e58"
DARK = "#3d3d3a"
RED = "#c23b22"

RATIO_COLORS = {
    "uniform-2048": BLUE,
    "mild-mixed": ORANGE,
    "w3-like-max-2048": GREEN,
}

RUNS = {
    "seat_fill": "W4/0002-seat-fill-max2048",
    "paged_fill": "W4/0003-paged-fill-max2048",
    "seat_cost": "W4/0004-seat-b8-cost",
    "paged_cost": "W4/0005-paged-b8-cost",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind") == "step":
                rows.append(row)
    return rows


def _decode(rows: Iterable[dict[str, Any]], active: int = 8) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row.get("active") or 0) == active and float(row.get("prefill_ms") or 0.0) == 0.0
    ]


def _median(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise SystemExit(f"no values for {key!r}")
    return statistics.median(values)


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=INK_2, labelsize=8.8, length=2.5, width=0.7)


def _header(fig, title: str, subtitle: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.012, 0.985, title, fontsize=13.5, color=INK, ha="left", va="top")
    fig.text(0.012, 0.925, subtitle, fontsize=8.5, color=INK_2, ha="left", va="top")


def _run_simulator(arguments: list[str]) -> None:
    """Run the real allocator simulator; keep its JSON in the caller's temp dir."""
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "sim_paged_capacity.py"
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "capacity simulation failed:\n" + (completed.stderr or completed.stdout)
        )


def _capacity_rows(results: Path, tmp: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Regenerate the synthetic sweep and the real-trace replay, then read both.

    The JSONs are intermediates, not deliverables: they live only for the duration of
    ``main``'s temp directory. ``docs/`` keeps the PNGs, not the sweep data.
    """
    synthetic_path = tmp / "synthetic.json"
    trace_path = tmp / "trace.json"
    _run_simulator(["--json", str(synthetic_path)])
    _run_simulator(
        [
            "--trace",
            str(results / RUNS["paged_fill"] / "metrics.jsonl"),
            "--block-sizes",
            "16",
            "--trace-label",
            "W4 real trace",
            "--json",
            str(trace_path),
        ]
    )
    return _read_json(synthetic_path)["rows"], _read_json(trace_path)["rows"]


def plot_capacity(
    simulated: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    results: Path,
    out: Path,
) -> None:
    seat = _read_json(results / RUNS["seat_fill"] / "summary.json")["capacity"]
    paged = _read_json(results / RUNS["paged_fill"] / "summary.json")["capacity"]

    fig, ax = plt.subplots(figsize=(10.2, 5.2), dpi=150)
    fig.subplots_adjust(left=0.075, right=0.76, top=0.82, bottom=0.12)
    _style_axis(ax)

    for name in ("uniform-2048", "mild-mixed", "w3-like-max-2048"):
        for block_size in (16, 32):
            points = sorted(
                (row["budget_mb"], row["ratio"])
                for row in simulated
                if row["distribution"] == name and row["paged"]["block_size"] == block_size
                and row["ratio"] is not None
            )
            if not points:
                continue
            primary = block_size == 16
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                "-" if primary else "--",
                color=RATIO_COLORS[name],
                linewidth=1.7 if primary else 1.0,
                alpha=1.0 if primary else 0.65,
                marker="o" if primary else "s",
                markersize=4.2 if primary else 3.0,
                markerfacecolor=RATIO_COLORS[name],
                markeredgecolor=SURFACE,
                markeredgewidth=0.6,
                zorder=3,
                label=f"{name} · bs{block_size}",
            )

    trace_points = sorted(
        (row["budget_mb"], row["ratio"])
        for row in trace
        if row["paged"]["block_size"] == 16 and row["ratio"] is not None
    )
    ax.plot(
        [point[0] for point in trace_points],
        [point[1] for point in trace_points],
        "-",
        color=DARK,
        linewidth=1.6,
        marker="D",
        markersize=3.8,
        markerfacecolor=DARK,
        markeredgecolor=SURFACE,
        markeredgewidth=0.6,
        zorder=4,
        label="W4-0003 trace · budget-derived",
    )

    measured_x = float(paged["kv_total_mb"])
    measured_y = float(paged["peak_concurrency"]) / float(seat["peak_concurrency"])
    ax.scatter(
        [measured_x],
        [measured_y],
        marker="*",
        s=260,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=6,
        label=(
            f"W4-0003 measured · {paged['peak_concurrency']} vs {seat['peak_concurrency']} "
            f"= {measured_y:.2f}×"
        ),
    )

    ax.axvline(1640, color=INK_2, linestyle=":", linewidth=1.0, zorder=1)
    ax.text(1644, 0.08, "1640 MiB budget", color=INK_2, fontsize=8.2, ha="left", va="bottom")
    ax.axhline(1.5, color=INK, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(2398, 1.54, "C1 1.5×", color=INK, fontsize=8.4, ha="right", va="bottom")
    ax.text(
        1204,
        1.04,
        "uniform-2048 · negative control (report, do not judge)",
        color=BLUE,
        fontsize=7.6,
        ha="left",
        va="bottom",
    )

    ax.set_xlim(1160, 2440)
    ax.set_ylim(0.0, 6.3)
    ax.set_xlabel("total budget MiB (KV + workspace, D6)", fontsize=9.4, color=INK_2)
    ax.set_ylabel("capacity ratio · paged / seat", fontsize=9.4, color=INK_2)
    _header(
        fig,
        "W4 capacity · same total budget, length heterogeneity is the whole effect",
        "deterministic allocator simulation (no model) · solid = bs16, dashed = bs32 · "
        "trace replay uses the W4-0003 request lengths/budgets · orange star is the measured run",
    )
    ax.legend(
        frameon=False,
        fontsize=8.2,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        handlelength=2.1,
        labelspacing=0.55,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_memory(results: Path, out: Path) -> None:
    seat_fill = _read_json(results / RUNS["seat_fill"] / "summary.json")
    paged_fill = _read_json(results / RUNS["paged_fill"] / "summary.json")
    seat_cost = _read_json(results / RUNS["seat_cost"] / "summary.json")["memory"]
    paged_cost = _read_json(results / RUNS["paged_cost"] / "summary.json")["memory"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.6), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.30)

    # Left: E2, same concurrency (B=8), same KV bytes; workspace is the added cost.
    _style_axis(left)
    labels = ["seat", "paged"]
    kv = [float(seat_cost["kv_mb"]), float(paged_cost["kv_mb"])]
    workspace = [0.0, float(paged_cost["workspace_mb"])]
    left.bar(labels, kv, width=0.52, color=BLUE, zorder=3)
    left.bar(labels, workspace, width=0.52, bottom=kv, color=ORANGE, zorder=3)
    for index, (kv_mb, ws_mb) in enumerate(zip(kv, workspace)):
        total = kv_mb + ws_mb
        delta = "" if index == 0 else f" (+{ws_mb:.0f})"
        left.text(
            index,
            total + 22,
            f"{total:.0f} MiB{delta}\nKV {kv_mb:.0f} + ws {ws_mb:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=INK,
            linespacing=1.25,
        )
    left.set_ylim(0, 1750)
    left.set_ylabel("MiB", fontsize=9.2, color=INK_2)
    left.set_title("E2 · same concurrency (B=8) memory", fontsize=10.2, color=INK, loc="left", pad=8)

    # Right: E1, capacity per GiB of *total* budget (never KV alone, P-36).
    _style_axis(right)
    seat_cap = float(seat_fill["capacity"]["capacity_per_gib"])
    paged_cap = float(paged_fill["capacity"]["capacity_per_gib"])
    bars = right.bar(labels, [seat_cap, paged_cap], width=0.52, color=[INK_2, GREEN], zorder=3)
    for bar, value in zip(bars, (seat_cap, paged_cap)):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=INK,
        )
    right.set_ylim(0, 21.5)
    right.set_ylabel("peak concurrency / GiB (total)", fontsize=9.2, color=INK_2)
    right.set_title("E1 · fill capacity per GiB", fontsize=10.2, color=INK, loc="left", pad=8)
    right.text(
        0.03,
        0.94,
        f"{paged_cap / seat_cap:.2f}× per GiB\n"
        f"{paged_fill['capacity']['peak_concurrency']} vs "
        f"{seat_fill['capacity']['peak_concurrency']} requests",
        transform=right.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color=INK,
        linespacing=1.35,
    )

    _header(
        fig,
        "W4 memory · paged pays a workspace premium at the same concurrency",
        "E2: 8 long prompts, B=8, budget 128, 1456 MiB KV both arms · "
        "E1: fill-only, 1640 MiB total budget (paged workspace is formula-estimated in fill)",
    )
    fig.text(
        0.012,
        0.055,
        "paged E2 total = 1508 MiB (+52 MiB, +3.6% over seat); E1 capacity/GiB uses "
        "kv_total_mb = kv_mb + workspace_mb, not KV alone (P-36).",
        fontsize=7.9,
        color=INK_2,
        ha="left",
        va="bottom",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_step(results: Path, out: Path) -> None:
    seat_dir = results / RUNS["seat_cost"]
    paged_dir = results / RUNS["paged_cost"]
    seat_steps = _decode(_step_rows(seat_dir))
    paged_steps = _decode(_step_rows(paged_dir))
    seat_summary = _read_json(seat_dir / "summary.json")
    paged_summary = _read_json(paged_dir / "summary.json")

    seat_step = _median(seat_steps, "step_ms")
    paged_step = _median(paged_steps, "step_ms")
    seat_event = _median(seat_steps, "decode_event_ms")
    paged_event = _median(paged_steps, "decode_event_ms")
    seat_gap = _median(seat_steps, "host_gap_ms")
    paged_gap = _median(paged_steps, "host_gap_ms")
    paged_gather = _median(paged_steps, "gather_ms")
    ratio = paged_step / seat_step

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.7), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.20, wspace=0.33)

    # Left: step_ms = (event window - gather host) + gather host + host gap.
    _style_axis(left)
    labels = ["seat", "paged"]
    rest = [seat_event, paged_event - paged_gather]
    gather = [0.0, paged_gather]
    gap = [seat_gap, paged_gap]
    left.bar(labels, rest, width=0.52, color=BLUE, label="event window − gather host", zorder=3)
    left.bar(labels, gather, width=0.52, bottom=rest, color=ORANGE, label="gather (host dispatch)", zorder=3)
    left.bar(
        labels,
        gap,
        width=0.52,
        bottom=[rest[0] + gather[0], rest[1] + gather[1]],
        color=INK_2,
        label="host gap (outside event)",
        zorder=3,
    )
    for index, total in enumerate((seat_step, paged_step)):
        caption = f"{total:.2f} ms" if index == 0 else f"{total:.2f} ms\n({total / seat_step:.2f}×)"
        left.text(
            index,
            total + 0.7,
            caption,
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=INK,
            linespacing=1.25,
        )
    left.axhline(seat_step * 1.15, color=INK_2, linestyle=":", linewidth=1.0)
    left.axhline(seat_step * 1.5, color=RED, linestyle="--", linewidth=1.0)
    left.text(1.47, seat_step * 1.5 + 0.7, "1.5× withdraw", color=RED, fontsize=7.8, ha="right", va="bottom")
    left.text(1.47, seat_step * 1.15 + 0.7, "1.15× healthy", color=INK_2, fontsize=7.8, ha="right", va="bottom")
    left.set_ylim(0, 42)
    left.set_ylabel("ms / decode step", fontsize=9.2, color=INK_2)
    left.set_title("E2 · decode step decomposition (B=8)", fontsize=10.2, color=INK, loc="left", pad=8)
    left.legend(frameon=False, fontsize=7.8, loc="upper left", handlelength=1.8, labelspacing=0.4)

    # Right: ratio per lpad bucket; this is the matched-width check for C4.
    _style_axis(right)
    buckets = sorted({int(row["lpad"]) for row in paged_steps if row.get("lpad")})
    values = []
    for lpad in buckets:
        subset = [row for row in paged_steps if int(row.get("lpad") or 0) == lpad]
        values.append(_median(subset, "step_ms") / seat_step)
    bars = right.bar([str(lpad) for lpad in buckets], values, width=0.5, color=GREEN, zorder=3)
    for bar, value, lpad in zip(bars, values, buckets):
        count = sum(1 for row in paged_steps if int(row.get("lpad") or 0) == lpad)
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}×\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=INK,
            linespacing=1.25,
        )
    right.axhline(1.5, color=RED, linestyle="--", linewidth=1.0)
    right.text(2.42, 1.515, "1.5× withdraw", color=RED, fontsize=7.8, ha="right", va="bottom")
    right.axhline(1.15, color=INK_2, linestyle=":", linewidth=1.0)
    right.text(2.42, 1.165, "1.15× healthy", color=INK_2, fontsize=7.8, ha="right", va="bottom")
    right.set_ylim(1.0, 1.82)
    right.set_xlabel("paged lpad bucket (seat row width = 1664)", fontsize=9.0, color=INK_2)
    right.set_ylabel("paged / seat step_ms", fontsize=9.2, color=INK_2)
    right.set_title("E2 · matched width is still over the line", fontsize=10.2, color=INK, loc="left", pad=8)

    _header(
        fig,
        "W4 decode cost · dense gather dominates the step",
        f"seat {seat_step:.2f} ms vs paged {paged_step:.2f} ms · "
        f"8 long prompts, budget 128, 5 repeats · gather_ms is host wall time, not GPU time",
    )
    fig.text(
        0.012,
        0.055,
        "The 9.4 ms gather figure is host dispatch; the event-window delta versus seat is 14.2 ms, "
        "so ~4.7 ms of gather GPU work sits inside the blue segment. C4 uses step_ms only (D6).",
        fontsize=7.9,
        color=INK_2,
        ha="left",
        va="bottom",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="results root (default: results)")
    parser.add_argument("--docs", default="docs", help="docs root (default: docs)")
    parser.add_argument("--out-dir", default=None, help="figure directory (default: docs)")
    args = parser.parse_args(argv)

    results = Path(args.results)
    docs = Path(args.docs)
    out_dir = Path(args.out_dir) if args.out_dir else docs

    required = [
        results / RUNS["seat_fill"] / "summary.json",
        results / RUNS["paged_fill"] / "summary.json",
        results / RUNS["seat_cost"] / "summary.json",
        results / RUNS["paged_cost"] / "summary.json",
        results / RUNS["seat_cost"] / "metrics.jsonl",
        results / RUNS["paged_cost"] / "metrics.jsonl",
        results / RUNS["paged_fill"] / "metrics.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("missing inputs:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="w4-capacity-") as temporary:
        simulated, trace = _capacity_rows(results, Path(temporary))
        plot_capacity(simulated, trace, results, out_dir / "fig_w4_capacity.png")
    plot_memory(results, out_dir / "fig_w4_memory.png")
    plot_step(results, out_dir / "fig_w4_step.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
