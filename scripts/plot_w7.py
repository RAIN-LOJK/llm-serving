#!/usr/bin/env python3
"""Generate the two W7 evidence figures from committed run summaries.

Inputs (read-only; every plotted number comes from these files):

* ``results/W7/0004-hybrid-fill-512`` + ``0005-pure-fill-512``
* ``results/W7/0006-hybrid-fill-2k``  + ``0007-pure-fill-2k``
* ``results/W7/0008-hybrid-fill-8k``  + ``0009-pure-fill-8k``
* ``results/W7/0010-e3-hybrid-b1`` / ``0011-e3-pure-b1``
* ``results/W7/0028-e3-hybrid-b8-tpot`` / ``0029-e3-pure-b8-tpot``
* ``results/W7/0020…0025`` -- 128/192/256 crossover scan (hybrid/pure pairs)

Outputs (``figures/`` by default):

* ``figures/fig_w7_capacity.png``  -- fill-only peak concurrency vs prompt context,
  hybrid vs pure, with the hybrid/pure ratio at each context;
* ``figures/fig_w7_crossover.png`` -- per-sequence state vs context (fill probes) and
  TPOT vs context (E3, B=1/B=8), the two crossover views.

The script never invents a point: every required summary and field is validated before
rendering, both figures are rendered to memory first, and a missing run/field aborts with
one clear message instead of leaving a partial figure. Text is English because the
machine has no CJK font in matplotlib.

Usage::

    python scripts/plot_w7.py
    python scripts/plot_w7.py --check
    python scripts/plot_w7.py --results /tmp/w7-synth --out /tmp/w7-figs
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

# --- style (same conventions as scripts/plot_w4.py / plot_w45.py) ------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#2f9e58"
NEUTRAL = "#8a8984"

CAPACITY_NAME = "fig_w7_capacity.png"
CROSSOVER_NAME = "fig_w7_crossover.png"

#: The W7 pre-registered contexts (docs/hybrid_attention.md §4).
CONTEXTS = (512, 2048, 8192)
CAPACITY_RUNS = {
    512: ("W7-0004", "W7-0005"),
    2048: ("W7-0006", "W7-0007"),
    8192: ("W7-0008", "W7-0009"),
}
E3_RUNS = {
    (1, "hybrid"): "W7-0026",
    (1, "pure"): "W7-0027",
    (8, "hybrid"): "W7-0028",
    (8, "pure"): "W7-0029",
}
SCAN_RUNS = {
    128: ("W7-0020", "W7-0021"),
    192: ("W7-0022", "W7-0023"),
    256: ("W7-0024", "W7-0025"),
}


def fail(message: str) -> NoReturn:
    """Abort with one clear, prefixed message; no figure is written."""
    raise SystemExit(f"plot_w7: {message}")


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"missing or non-numeric value: {where}")
    number = float(value)
    if not math.isfinite(number):
        fail(f"non-finite value: {where}")
    return number


def _summary_index(results: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every W7 summary keyed by run id; duplicate ids are a hard error."""
    w7 = results / "W7"
    if not w7.is_dir():
        fail(f"no W7 results under {results}/W7; formal runs have not been executed")
    paths = sorted(w7.glob("*/summary.json"))
    if not paths:
        fail(f"{w7} contains no */summary.json; nothing to plot")
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot read {path}: {exc}")
        if not isinstance(summary, dict):
            fail(f"{path}: summary.json must contain an object")
        run_id = summary.get("run_id")
        if not isinstance(run_id, str) or not run_id.startswith("W7-"):
            fail(f"{path}: run_id is {run_id!r}, expected W7-####")
        if run_id in index:
            fail(f"duplicate run_id {run_id}: {index[run_id][0]} and {path}")
        index[run_id] = (path, summary)
    return index


def _require(
    index: dict[str, tuple[Path, dict[str, Any]]], run_id: str, results: Path
):
    if run_id not in index:
        fail(f"missing required W7 run {run_id} under {results}/W7")
    return index[run_id]


def _nominal_context(group: dict[str, Any]) -> int | None:
    """Prompt context of a group: ``context - out_tokens`` (D5/P-30 definition)."""
    try:
        context = float(group.get("context"))
        out = float((group.get("out_tokens") or {}).get("median"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(context) or not math.isfinite(out):
        return None
    return int(round(context - out))


def _single_context(summary: dict[str, Any], run_id: str) -> int:
    raw = (summary.get("capacity") or {}).get("context_tokens")
    if raw is not None:
        return int(_number(raw, f"{run_id} capacity.context_tokens"))
    contexts = sorted(
        {
            context
            for context in (_nominal_context(group) for group in summary.get("groups") or [])
            if context is not None
        }
    )
    if len(contexts) != 1:
        fail(
            f"{run_id}: cannot establish one prompt context "
            f"(capacity.context_tokens absent; groups give {contexts or 'nothing'})"
        )
    return contexts[0]


def _capacity(summary: dict[str, Any], run_id: str) -> dict[str, float]:
    block = summary.get("capacity") or {}
    peak = _number(block.get("peak_concurrency"), f"{run_id} capacity.peak_concurrency")
    if peak <= 0:
        fail(f"{run_id}: capacity.peak_concurrency is not positive")
    used_raw = block.get("state_used_mb")
    if used_raw is None:
        # Reserved state is a legitimate fallback, but label the figure accordingly.
        used = _number(block.get("state_total_mb"), f"{run_id} capacity.state_total_mb")
        basis = "reserved"
    else:
        used = _number(used_raw, f"{run_id} capacity.state_used_mb")
        basis = "used"
    if used <= 0:
        fail(f"{run_id}: capacity state ({basis}) is not positive")
    return {"peak": peak, "state": used, "basis": basis}


def _weighted_median(values: list[tuple[float, int]]) -> float:
    ordered = sorted(values)
    total = sum(weight for _, weight in ordered)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative * 2 >= total:
            return value
    return ordered[-1][0]


def _groups_at(
    summary: dict[str, Any], run_id: str, context: int, metric: str
) -> tuple[float, int]:
    """Weighted median and batch for one prompt context from ``groups``."""
    selected: list[tuple[float, int, int]] = []
    for group in summary.get("groups") or []:
        if _nominal_context(group) != int(context):
            continue
        raw = (group.get(metric) or {}).get("median")
        if raw is None:
            continue
        value = _number(raw, f"{run_id} groups[context={context}].{metric}.median")
        try:
            batch = int(group.get("batch") or 1)
        except (TypeError, ValueError):
            batch = 1
        selected.append((value, max(1, int(group.get("n") or 1)), batch))
    if not selected:
        fail(f"{run_id}: no groups[] entry for prompt context {context} ({metric})")
    # Scenario request rows have no per-request `batch`; bench.scenario therefore keys
    # their groups with batch=1. The protocol's max_batch is the actual running width.
    raw_batch = (summary.get("protocol") or {}).get("max_batch")
    protocol_batch: int | None = None
    if isinstance(raw_batch, int) and not isinstance(raw_batch, bool) and raw_batch >= 1:
        protocol_batch = raw_batch
    batches = sorted({batch for _, _, batch in selected})
    if protocol_batch is None and len(batches) != 1:
        fail(f"{run_id}: prompt context {context} spans batches {batches}")
    return (
        _weighted_median([(value, weight) for value, weight, _ in selected]),
        protocol_batch if protocol_batch is not None else batches[0],
    )


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.5)
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
    fig.text(0.012, 0.932, subtitle, fontsize=8.6, color=INK_2, ha="left", va="top")
    fig.text(0.012, 0.025, note, fontsize=8.0, color=INK_2, ha="left", va="bottom")


def _capacity_figure(capacity: dict[int, dict[str, dict[str, float]]]):
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=150)
    fig.subplots_adjust(left=0.095, right=0.975, top=0.795, bottom=0.205)
    _style_axis(ax)

    for arm, color, marker, label in (
        ("hybrid", BLUE, "o", "hybrid Qwen3.5-0.8B"),
        ("pure", ORANGE, "s", "pure Qwen3-0.6B"),
    ):
        xs = list(CONTEXTS)
        ys = [capacity[ctx][arm]["peak"] for ctx in CONTEXTS]
        ax.plot(xs, ys, color=color, marker=marker, markersize=6.5, linewidth=1.9, label=label, zorder=3)
        for ctx, peak in zip(xs, ys):
            ax.annotate(
                f"{peak:.0f}",
                (ctx, peak),
                textcoords="offset points",
                xytext=(0, 9 if arm == "hybrid" else -15),
                ha="center",
                fontsize=8.4,
                color=color,
            )

    for ctx in CONTEXTS:
        hybrid = capacity[ctx]["hybrid"]["peak"]
        pure = capacity[ctx]["pure"]["peak"]
        ax.annotate(
            f"{hybrid / pure:.2f}×",
            (ctx, max(hybrid, pure)),
            textcoords="offset points",
            xytext=(0, 22),
            ha="center",
            fontsize=8.2,
            color=INK_2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(CONTEXTS), [str(ctx) for ctx in CONTEXTS])
    ax.minorticks_off()
    ax.set_xlabel("prompt context (tokens)", fontsize=9.4, color=INK_2)
    ax.set_ylabel("peak concurrent requests (fill-only)", fontsize=9.4, color=INK_2)
    ax.set_ylim(0, max(capacity[ctx]["hybrid"]["peak"] for ctx in CONTEXTS) * 1.28)
    ax.legend(frameon=False, fontsize=9.0, loc="upper right")

    _header(
        fig,
        "W7 capacity: same 1640 MiB state budget, hybrid vs pure",
        "fill-only probes (prefill + reserve, no decode), budgets=[128]; annotated ratios are hybrid / pure",
        "source: results/W7/0004-0009 summary.json · capacity.peak_concurrency · no value is hardcoded",
    )
    return fig


def _crossover_figure(
    capacity: dict[int, dict[str, dict[str, float]]],
    scan: dict[int, dict[str, dict[str, float]]],
    tpot: dict[tuple[int, str], dict[int, float]],
):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.9), dpi=150)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.790, bottom=0.205, wspace=0.26)
    for ax in (left, right):
        _style_axis(ax)

    # -- left: per-sequence state (used when the arm recorded it, else reserved) ----
    state = dict(capacity)
    state.update(scan)
    state_contexts = sorted(state)
    for arm, color, marker, label in (
        ("hybrid", BLUE, "o", "hybrid 0.8B"),
        ("pure", ORANGE, "s", "pure 0.6B"),
    ):
        xs = list(state_contexts)
        per_seq = [state[ctx][arm]["state"] / state[ctx][arm]["peak"] for ctx in state_contexts]
        left.plot(
            xs,
            per_seq,
            color=color,
            marker=marker,
            markersize=6.0,
            linewidth=1.8,
            label=label,
            zorder=3,
        )
        for ctx, value in zip(xs, per_seq):
            left.annotate(
                f"{value:.0f}",
                (ctx, value),
                textcoords="offset points",
                xytext=(0, 9 if arm == "hybrid" else -14),
                ha="center",
                fontsize=8.2,
                color=color,
            )
    for ctx in (512, 2048, 8192):
        hybrid = state[ctx]["hybrid"]["state"] / state[ctx]["hybrid"]["peak"]
        pure = state[ctx]["pure"]["state"] / state[ctx]["pure"]["peak"]
        left.annotate(
            f"{pure / hybrid:.2f}×",
            (ctx, max(hybrid, pure)),
            textcoords="offset points",
            xytext=(0, 20),
            ha="center",
            fontsize=8.0,
            color=INK_2,
        )
    # Crossover is read off the plotted points, never hardcoded: find the adjacent
    # scan contexts where pure/hybrid flips sign and linearly interpolate.
    crossover = None
    for left_ctx, right_ctx in zip(state_contexts, state_contexts[1:]):
        diff_left = (
            state[left_ctx]["pure"]["state"] / state[left_ctx]["pure"]["peak"]
            - state[left_ctx]["hybrid"]["state"] / state[left_ctx]["hybrid"]["peak"]
        )
        diff_right = (
            state[right_ctx]["pure"]["state"] / state[right_ctx]["pure"]["peak"]
            - state[right_ctx]["hybrid"]["state"] / state[right_ctx]["hybrid"]["peak"]
        )
        if diff_left * diff_right < 0:
            crossover = left_ctx + (right_ctx - left_ctx) * diff_left / (diff_left - diff_right)
            break
    if crossover is not None:
        left.axvline(crossover, color=GREEN, linewidth=1.1, linestyle=":", zorder=1)
        left.annotate(
            f"crossover ≈ {crossover:.0f}",
            (crossover, left.get_ylim()[1] * 0.60),
            textcoords="offset points",
            xytext=(5, 0),
            ha="left",
            fontsize=8.0,
            color=GREEN,
        )
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xticks(list(state_contexts), [str(ctx) for ctx in state_contexts])
    left.minorticks_off()
    left.set_xlabel("prompt context (tokens)", fontsize=9.2, color=INK_2)
    left.set_ylabel("state per active sequence (MiB)", fontsize=9.2, color=INK_2)
    left.set_ylim(
        top=max(
            state[ctx][arm]["state"] / state[ctx][arm]["peak"]
            for ctx in state_contexts
            for arm in ("hybrid", "pure")
        )
        * 2.4
    )
    left.legend(frameon=False, fontsize=8.8, loc="upper left")
    left.set_title("per-sequence state (fill; used where recorded)", fontsize=10.0, color=INK, loc="left")

    # -- right: TPOT vs context for E3, B=1 and B=8 --------------------------------
    styles = {
        (1, "hybrid"): (BLUE, "o", "-", "hybrid B=1"),
        (1, "pure"): (ORANGE, "s", "-", "pure B=1"),
        (8, "hybrid"): (BLUE, "o", "--", "hybrid B=8"),
        (8, "pure"): (ORANGE, "s", "--", "pure B=8"),
    }
    for key in ((1, "hybrid"), (1, "pure"), (8, "hybrid"), (8, "pure")):
        color, marker, linestyle, label = styles[key]
        values = tpot[key]
        right.plot(
            list(CONTEXTS),
            [values[ctx] for ctx in CONTEXTS],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=6.0,
            linewidth=1.7,
            label=label,
            zorder=3,
        )
    right.set_xscale("log", base=2)
    right.set_xticks(list(CONTEXTS), [str(ctx) for ctx in CONTEXTS])
    right.minorticks_off()
    right.set_xlabel("prompt context (tokens)", fontsize=9.2, color=INK_2)
    right.set_ylabel("TPOT (ms/token, median)", fontsize=9.2, color=INK_2)
    right.set_ylim(0, max(tpot[key][ctx] for key in tpot for ctx in CONTEXTS) * 1.38)
    right.legend(frameon=False, fontsize=8.4, loc="upper left")
    right.set_title("TPOT vs context (E3 scenario, budgets=[128])", fontsize=10.0, color=INK, loc="left")

    _header(
        fig,
        "W7 crossover: per-sequence state and TPOT vs context",
        "left: fill probes W7-0004…0009 + scan W7-0020…0025; right: E3 W7-0026/0027 + W7-0028/0029; ratios are pure / hybrid",
        "source: results/W7/*/summary.json · capacity.state_used_mb|state_total_mb and groups[].tpot_ms · no value is hardcoded",
    )
    return fig


def _render(fig, software: str) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        facecolor=SURFACE,
        bbox_inches="tight",
        metadata={"Software": software},
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
    parser.add_argument("--out", default="figures", help="figure directory (default: figures)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and summarize inputs; render nothing",
    )
    args = parser.parse_args(argv)

    results = Path(args.results)
    out_dir = Path(args.out)
    index = _summary_index(results)

    capacity: dict[int, dict[str, dict[str, float]]] = {}
    for context in CONTEXTS:
        hybrid_id, pure_id = CAPACITY_RUNS[context]
        hybrid_path, hybrid_summary = _require(index, hybrid_id, results)
        pure_path, pure_summary = _require(index, pure_id, results)
        for path, summary, run_id in (
            (hybrid_path, hybrid_summary, hybrid_id),
            (pure_path, pure_summary, pure_id),
        ):
            recorded = _single_context(summary, run_id)
            if recorded != context:
                fail(f"{run_id} ({path}): context is {recorded}, expected {context}")
        capacity[context] = {
            "hybrid": _capacity(hybrid_summary, hybrid_id),
            "pure": _capacity(pure_summary, pure_id),
        }

    scan: dict[int, dict[str, dict[str, float]]] = {}
    for context, (hybrid_id, pure_id) in SCAN_RUNS.items():
        for arm, run_id in (("hybrid", hybrid_id), ("pure", pure_id)):
            path, summary = _require(index, run_id, results)
            recorded = _single_context(summary, run_id)
            if recorded != context:
                fail(f"{run_id} ({path}): context is {recorded}, expected {context}")
            scan.setdefault(context, {})[arm] = _capacity(summary, run_id)

    tpot: dict[tuple[int, str], dict[int, float]] = {}
    for key, run_id in E3_RUNS.items():
        path, summary = _require(index, run_id, results)
        values: dict[int, float] = {}
        batches: set[int] = set()
        for context in CONTEXTS:
            value, batch = _groups_at(summary, run_id, context, "tpot_ms")
            values[context] = value
            batches.add(batch)
        if len(batches) != 1:
            fail(f"{run_id} ({path}): E3 groups span batches {sorted(batches)}")
        if batches != {key[0]}:
            fail(f"{run_id} ({path}): groups report batch {sorted(batches)[0]}, expected {key[0]}")
        tpot[key] = values

    if args.check:
        print(
            f"check ok: capacity {len(CAPACITY_RUNS) * 2} fill summaries at "
            f"{', '.join(str(ctx) for ctx in CONTEXTS)}; "
            f"scan {len(SCAN_RUNS) * 2} summaries at 128/192/256; "
            f"E3 {len(E3_RUNS)} summaries at B=1/B=8"
        )
        for context in CONTEXTS:
            hybrid = capacity[context]["hybrid"]
            pure = capacity[context]["pure"]
            print(
                f"  {context:>5}: hybrid peak {hybrid['peak']:.0f} "
                f"({hybrid['basis']} state {hybrid['state'] / hybrid['peak']:.1f} MiB/seq), "
                f"pure peak {pure['peak']:.0f} "
                f"({pure['basis']} state {pure['state'] / pure['peak']:.1f} MiB/seq), "
                f"capacity ratio {hybrid['peak'] / pure['peak']:.2f}x"
            )
        for context in sorted(scan):
            hybrid = scan[context]["hybrid"]
            pure = scan[context]["pure"]
            print(
                f"  {context:>5}: hybrid {hybrid['state'] / hybrid['peak']:.2f} MiB/seq, "
                f"pure {pure['state'] / pure['peak']:.2f} MiB/seq"
            )
        for key in ((1, "hybrid"), (1, "pure"), (8, "hybrid"), (8, "pure")):
            print(
                f"  TPOT {key[1]:<6} B={key[0]}: "
                + ", ".join(f"{ctx}={tpot[key][ctx]:.2f} ms" for ctx in CONTEXTS)
            )
        return 0

    capacity_png = _render(_capacity_figure(capacity), "minillm/scripts/plot_w7.py")
    crossover_png = _render(_crossover_figure(capacity, scan, tpot), "minillm/scripts/plot_w7.py")
    written = _write_all(
        out_dir,
        [(CAPACITY_NAME, capacity_png), (CROSSOVER_NAME, crossover_png)],
    )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
