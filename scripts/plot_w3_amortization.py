#!/usr/bin/env python3
"""Plot W3 E1: batching amortizes the per-step host cost across context lengths.

Reads **only** each E1 run's ``summary.json`` (never ``metrics.jsonl``) from
``results/W3/0017-...`` through ``results/W3/0019-...`` plus ``W3-0006`` /
``W3-0007`` and writes ``docs/fig_w3_amortization.png``. The original ctx512
cells (``0003-0005``) emitted EOS after 3 tokens and are superseded by
``0017-0019`` at ctx511 (see PITFALLS P-31); 511 is 1 token below 512.

What is drawn (three panels, shared discrete x axis = batch size B):

* left  -- per-token cost in ms/token, both wall-clock (``tok_s``) and
           decode-only (``decode_only_tok_s``);
* right -- throughput in tokens/s for the same two scopes;
* third -- the per-step split that explains the curves: mean ``decode_event_ms``
           and ``host_gap_ms`` at ``active == max_batch``.

Why facets instead of a dual y axis: ms/token and tok/s are the same quantity
inverted (tok/s ~= 1000 / (ms/token) within one scope), so two y scales would
only invent an alignment. Small multiples share the x axis but never share two
independent y scales. Colour follows the context length (an entity), not the
rank; line style follows the measurement scope/metric.

Missing data: a missing ``summary.json`` / metric makes that (ctx, B) point a
gap (the line is broken, nothing is interpolated). If *no* E1 point can be
loaded, the script prints an explanation and exits non-zero without touching
the output file. A bad figure is never written: the PNG is saved to a hidden
temporary file first and only then atomically moved into place.

Direct labels live in a dedicated gutter to the right of every panel, never
over the data ink. Their vertical order follows their points, the stack is
pushed apart until every pair keeps ``LABEL_SEP_PX`` of white space, and a
hairline leader joins each label back to its last measured point. The layout is
verified numerically (matplotlib renderer + ``Text.get_window_extent``) by
``audit_label_layout``: it reports intersecting label pairs and the worst
overlap in pixels, and the PNG is not written if that audit fails. There is no
dual y axis, only facets (see the paragraph above).

Usage::

    python scripts/plot_w3_amortization.py
    python scripts/plot_w3_amortization.py --results results --out docs/fig_w3_amortization.png
    python scripts/plot_w3_amortization.py --debug        # per-label boxes + audit dump
    python scripts/plot_w3_amortization.py --show
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

# The backend must be selected before pyplot is imported. File output is always
# headless; --show is an explicit opt-in for a local interactive window.
_SHOW_REQUESTED = "--show" in sys.argv
if not _SHOW_REQUESTED:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# --- style constants -------------------------------------------------------
# Fixed categorical order from the dataviz reference palette (light mode),
# slots 1-2: blue then orange. Colour is keyed by ctx, never by rank, and is
# never cycled. The two-colour subset passed
#   validate_palette.js "#2a78d6,#eb6834" --mode light
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text / direct labels
MUTED = "#898781"        # tick labels
GRID = "#e1e0d9"         # hairline grid
AXIS = "#c3c2b7"         # baseline / spine
CTX_COLORS = {511: "#2a78d6", 2048: "#eb6834"}
CTX_ORDER = (511, 2048)  # fixed assignment: 511 -> slot 1, 2048 -> slot 2
B_VALUES = (1, 8, 32)    # discrete x positions, not a log axis
B_X = {batch: index for index, batch in enumerate(B_VALUES)}
LINE_WIDTH = 2.0
MARKER_SIZE = 8.5

# --- label gutter geometry (figure fractions unless stated in px) -----------
# Every panel is a manually positioned plot area plus a gutter to its right. The
# gutter is real estate inside the figure, so a label can never land on another
# panel's plot area (which is what data-coordinate labels used to do) and can
# never run past the figure edge. Pixel constants are measured with the actual
# renderer, not with "looks about right".
FIG_SIZE = (16.5, 5.2)   # inches
FIG_DPI = 150
PANEL_LEFT = 0.055
PANEL_BOTTOM = 0.115
PANEL_HEIGHT = 0.70
PANEL_WIDTH = 0.200
GUTTER_WIDTH = 0.075
PLOT_GUTTER_GAP = 0.008  # blank space between the right spine and the gutter
PANEL_GAP = 0.040        # blank space between a gutter and the next plot area
GUTTER_PAD_PX = 10.0     # text inset from the gutter's left edge
LABEL_SEP_PX = 3.0       # minimum blank space between two label boxes
LABEL_FONT_SIZE = 8.5
LEADER_STUB = 0.045      # data units of leader just outside the right spine
LINK = "#b6b4ab"         # hairline leader: neutral, never a series colour
XLIM = (-0.35, 2.02)     # the plot area stops just right of the B=32 tick

# E1 matrix (docs/w3-plan.md §3). B is re-read from protocol.max_batch when
# present; these values are the fallback and define what "complete" means.
# W3-0003..0005 are the invalid ctx512 cells (EOS after 3 tokens); they stay in
# results/ as evidence but are not plotted. W3-0017..0019 rerun the same B
# matrix at 511, the nearest context whose synthetic prompt reaches 128 tokens.
E1_RUNS: dict[str, tuple[int, int]] = {
    "W3-0017": (511, 1),
    "W3-0018": (511, 8),
    "W3-0019": (511, 32),
    "W3-0006": (2048, 1),
    "W3-0007": (2048, 8),
}

WALL_STYLE = {"linestyle": "-", "marker": "o", "markerfacecolor": "fill"}
DECODE_STYLE = {"linestyle": (0, (4, 2)), "marker": "o", "markerfacecolor": "white"}
EVENT_STYLE = {"linestyle": "-", "marker": "o", "markerfacecolor": "fill"}
GAP_STYLE = {"linestyle": (0, (4, 2)), "marker": "s", "markerfacecolor": "white"}


@dataclass(frozen=True)
class Point:
    """One (ctx, B) cell, all values read from that run's summary.json."""

    ctx: int
    batch: int
    run_id: str
    tok_s: float | None
    tok_s_active: float | None
    decode_only_tok_s: float | None
    decode_event_ms: float | None
    host_gap_ms: float | None
    step_ms: float | None
    n_steps: int | None


# --- reading ---------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """A JSON number as a finite float, else None (None values are gaps)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(summary: dict[str, Any], block: str, key: str) -> float | None:
    node = (summary.get(block) or {}).get(key)
    if not isinstance(node, dict):
        return None
    return _finite(node.get("median"))


def _stage_dir(root: Path) -> Path:
    """``--results results`` points at the stage root; accept the W3 dir too."""
    stage = root / "W3"
    if stage.is_dir():
        return stage
    if any(root.glob("*/summary.json")):
        return root
    return stage


def _ctx_for(run_id: str, run_dir: Path) -> int | None:
    """Context length is not stored in summary.json; the run slug names it."""
    match = re.search(r"ctx(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    expected = E1_RUNS.get(run_id)
    return expected[0] if expected else None


def _batch_for(run_id: str, summary: dict[str, Any]) -> int | None:
    protocol_batch = _finite((summary.get("protocol") or {}).get("max_batch"))
    if protocol_batch is not None and protocol_batch >= 1:
        return int(protocol_batch)
    expected = E1_RUNS.get(run_id)
    return expected[1] if expected else None


def _step_means(summary: dict[str, Any], batch: int) -> tuple[float | None, float | None, float | None, int | None]:
    """Mean step split at the full batch, else at the largest active value seen."""
    entries = (summary.get("steps") or {}).get("by_active") or []
    candidates = [entry for entry in entries if isinstance(entry, dict) and _finite(entry.get("active")) is not None]
    if not candidates:
        return None, None, None, None
    exact = [entry for entry in candidates if int(float(entry["active"])) == batch]
    picked = exact[0] if exact else max(candidates, key=lambda entry: float(entry["active"]))
    n = picked.get("n")
    return (
        _finite(picked.get("decode_event_ms")),
        _finite(picked.get("host_gap_ms")),
        _finite(picked.get("step_ms")),
        int(n) if _finite(n) is not None else None,
    )


def _point_from(run_id: str, run_dir: Path, summary: dict[str, Any]) -> Point | None:
    ctx = _ctx_for(run_id, run_dir)
    batch = _batch_for(run_id, summary)
    if ctx is None or batch is None:
        return None
    if ctx not in CTX_COLORS or batch not in B_X:
        return None
    decode_event, host_gap, step_ms, n_steps = _step_means(summary, batch)
    return Point(
        ctx=ctx,
        batch=batch,
        run_id=run_id,
        tok_s=_median(summary, "throughput", "tok_s"),
        tok_s_active=_median(summary, "throughput", "tok_s_active"),
        decode_only_tok_s=_median(summary, "throughput", "decode_only_tok_s"),
        decode_event_ms=decode_event,
        host_gap_ms=host_gap,
        step_ms=step_ms,
        n_steps=n_steps,
    )


def load_points(stage: Path) -> tuple[dict[tuple[int, int], Point], list[str], list[str]]:
    """Return ``{(ctx, B): Point}`` plus load problems and malformed summaries."""
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    problems: list[str] = []

    for summary_path in sorted(stage.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{summary_path}: {exc}")
            continue
        if not isinstance(summary, dict):
            problems.append(f"{summary_path}: top-level JSON is not an object")
            continue
        run_id = str(summary.get("run_id") or "")
        if run_id in E1_RUNS:
            found[run_id] = (summary_path, summary)

    points: dict[tuple[int, int], Point] = {}
    for run_id in E1_RUNS:
        if run_id not in found:
            continue
        summary_path, summary = found[run_id]
        point = _point_from(run_id, summary_path.parent, summary)
        if point is None:
            ctx, batch = E1_RUNS[run_id]
            problems.append(f"{run_id}: cannot resolve ctx/B (expected ctx{batch} B{batch}); skipped")
            continue
        points[(point.ctx, point.batch)] = point
    return points, problems, [p for p in problems if "cannot resolve" in p]


# --- plotting --------------------------------------------------------------


def _apply_chrome(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _series(
    points: dict[tuple[int, int], Point], ctx: int, field: str, *, inverse: bool = False
) -> tuple[list[int], list[float | None]]:
    xs: list[int] = []
    ys: list[float | None] = []
    for batch in B_VALUES:
        point = points.get((ctx, batch))
        xs.append(B_X[batch])
        value = getattr(point, field) if point is not None else None
        if value is not None and inverse:
            value = 1000.0 / value if value > 0 else None
        ys.append(value)
    return xs, ys


def _plot_series(
    ax: plt.Axes,
    xs: list[int],
    ys: list[float | None],
    *,
    color: str,
    style: dict[str, Any],
    label: str | None = None,
) -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=1.6,
        markeredgecolor=color,
        label=label,
        zorder=3,
        **{k: v for k, v in style.items() if k != "markerfacecolor"},
        markerfacecolor=color if style.get("markerfacecolor") == "fill" else "white",
    )


@dataclass
class EndLabel:
    """One direct label: the point it belongs to and where its text ended up.

    ``y_px`` starts at the point's display y and is then relaxed into the gutter;
    ``artist`` exists only once the renderer does, so the audit measures the real
    text box instead of guessing a width.
    """

    text: str
    x: float
    y: float
    linestyle: Any
    target_px: float = 0.0
    y_px: float = 0.0
    height_px: float = 0.0
    artist: Any = None


@dataclass
class LabelPanel:
    """A plot area plus the gutter to its right and the labels that live there."""

    index: int
    ax: plt.Axes
    gutter: tuple[float, float]  # figure-fraction x0, x1
    items: list[EndLabel]


def _renderer(fig: plt.Figure) -> Any:
    """Agg exposes ``get_renderer``; other canvases (``--show``) expose ``renderer``."""
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig.canvas.renderer


def _panel_rects(count: int) -> list[tuple[float, float, float, float]]:
    block = PANEL_WIDTH + PLOT_GUTTER_GAP + GUTTER_WIDTH + PANEL_GAP
    return [
        (PANEL_LEFT + index * block, PANEL_BOTTOM, PANEL_WIDTH, PANEL_HEIGHT)
        for index in range(count)
    ]


def _layout_end_labels(fig: plt.Figure, panels: list[LabelPanel]) -> None:
    """Place every direct label in its panel's gutter and draw its leader.

    Vertical placement runs in display pixels with the real renderer: a label
    starts at its point's y, the stack is pushed apart until neighbouring boxes
    keep ``LABEL_SEP_PX`` of white space, and the result is clamped inside the
    axes' vertical span. Clamping is applied *after* spreading and re-checked, so
    a shift can never push two labels back into each other. Labels are plain
    figure text in INK_2 (never a series colour); leaders are neutral hairlines
    at ``zorder`` 2, below the data lines, so they cannot cover a series.
    """
    fig.canvas.draw()
    renderer = _renderer(fig)
    width_px = fig.bbox.width
    height_px = fig.bbox.height

    for panel in panels:
        items = panel.items
        if not items:
            continue
        ax = panel.ax
        axes_box = ax.get_window_extent(renderer)
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if span <= 0 or axes_box.height <= 0:
            continue

        items.sort(key=lambda item: item.y)
        for item in items:
            item.target_px = axes_box.y0 + (item.y - ymin) / span * axes_box.height
            item.artist = fig.text(
                0.0,
                0.0,
                item.text,
                ha="left",
                va="center",
                fontsize=LABEL_FONT_SIZE,
                color=INK_2,
                zorder=5,
            )
            item.height_px = item.artist.get_window_extent(renderer).height

        gaps = [
            (items[i].height_px + items[i + 1].height_px) / 2.0 + LABEL_SEP_PX
            for i in range(len(items) - 1)
        ]
        low = axes_box.y0 + items[0].height_px / 2.0 + 0.5
        high = axes_box.y1 - items[-1].height_px / 2.0 - 0.5
        ys = [item.target_px for item in items]
        for i in range(1, len(items)):
            ys[i] = max(ys[i], ys[i - 1] + gaps[i - 1])
        if ys[-1] > high:
            ys[-1] = high
            for i in range(len(items) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gaps[i])
        if ys[0] < low:
            ys[0] = low
            for i in range(1, len(items)):
                ys[i] = max(ys[i], ys[i - 1] + gaps[i - 1])
        if ys[-1] > high:  # more labels than vertical room: shrink the stack to fit
            room = max(high - low, 0.0)
            centre = (ys[0] + ys[-1]) / 2.0
            scale = room / (ys[-1] - ys[0]) if ys[-1] > ys[0] else 1.0
            ys = [min(high, max(low, centre + (y - centre) * scale)) for y in ys]

        text_x = (panel.gutter[0] * width_px + GUTTER_PAD_PX) / width_px
        for item, y_px in zip(items, ys):
            item.y_px = y_px
            item.artist.set_position((text_x, y_px / height_px))

        # Leaders: point -> just outside the right spine -> gutter edge. Both
        # ends are computed through the same inverse transform, so the leader
        # stays glued to the point (and to the label) at any dpi.
        inverse = ax.transData.inverted()
        stub = ax.get_xlim()[1] + LEADER_STUB
        tip = inverse.transform((panel.gutter[0] * width_px - 2.5, 0.0))[0]
        for item in items:
            label_y = inverse.transform((0.0, item.y_px))[1]
            ax.plot(
                [item.x, stub, tip],
                [item.y, item.y, label_y],
                color=LINK,
                linewidth=0.7,
                linestyle=item.linestyle,
                alpha=0.75,
                zorder=2,
                clip_on=False,
                solid_capstyle="butt",
                dash_capstyle="butt",
            )

    fig.canvas.draw()


def audit_label_layout(
    fig: plt.Figure, panels: list[LabelPanel], *, verbose: bool = False
) -> int:
    """Numeric self-check of the direct labels; returns the violation count.

    Nothing here looks at the PNG. Every text box comes from the live renderer
    (``Text.get_window_extent``), so the numbers are the numbers a reader would
    measure on the raster. Reported per run:

    * how many label pairs intersect and the largest overlap in pixels;
    * whether every box stays inside its own gutter and inside the figure;
    * whether every box is clear of its panel's plot area (lines cannot hide).

    ``verbose`` (``--debug``) also dumps every box and its gutter bounds.
    """
    fig.canvas.draw()
    renderer = _renderer(fig)
    fig_box = fig.bbox
    entries: list[tuple[LabelPanel, EndLabel, Any, Any, tuple[float, float]]] = []
    for panel in panels:
        gutter_px = (panel.gutter[0] * fig_box.width, panel.gutter[1] * fig_box.width)
        axes_box = panel.ax.get_window_extent(renderer)
        for item in panel.items:
            if item.artist is None:
                continue
            box = item.artist.get_window_extent(renderer)
            entries.append((panel, item, box, axes_box, gutter_px))
            if verbose:
                print(
                    f"  [label] panel {panel.index + 1} {item.text!r:20s} "
                    f"y={item.y_px:6.1f}px box=({box.x0:7.1f},{box.y0:6.1f})-"
                    f"({box.x1:7.1f},{box.y1:6.1f}) "
                    f"gutter_x=({gutter_px[0]:.1f},{gutter_px[1]:.1f}) "
                    f"axes_x1={axes_box.x1:.1f}"
                )

    pairs = 0
    worst = 0.0
    worst_pair = ""
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            first = entries[i][2]
            second = entries[j][2]
            overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
            overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
            if overlap_x > 0 and overlap_y > 0:
                pairs += 1
                if min(overlap_x, overlap_y) > worst:
                    worst = min(overlap_x, overlap_y)
                    worst_pair = f"{entries[i][1].text!r} x {entries[j][1].text!r}"

    inside_gutter = 0
    inside_figure = 0
    clear_of_plot = 0
    offenders: list[str] = []
    for panel, item, box, axes_box, gutter_px in entries:
        tol = 0.5
        ok_gutter = (
            box.x0 >= gutter_px[0] - tol
            and box.x1 <= gutter_px[1] + tol
            and box.y0 >= axes_box.y0 - tol
            and box.y1 <= axes_box.y1 + tol
        )
        ok_figure = (
            box.x0 >= -tol
            and box.y0 >= -tol
            and box.x1 <= fig_box.width + tol
            and box.y1 <= fig_box.height + tol
        )
        ok_clear = box.x0 >= axes_box.x1 - tol
        inside_gutter += int(ok_gutter)
        inside_figure += int(ok_figure)
        clear_of_plot += int(ok_clear)
        if not (ok_gutter and ok_figure and ok_clear):
            offenders.append(
                f"panel {panel.index + 1} {item.text!r}: box=({box.x0:.1f},{box.y0:.1f})-"
                f"({box.x1:.1f},{box.y1:.1f}) gutter_x=({gutter_px[0]:.1f},{gutter_px[1]:.1f}) "
                f"axes=({axes_box.x0:.1f},{axes_box.y0:.1f})-({axes_box.x1:.1f},{axes_box.y1:.1f}) "
                f"gutter_ok={ok_gutter} figure_ok={ok_figure} clear_ok={ok_clear}"
            )

    total = len(entries)
    print(
        f"[audit] labels={total} intersecting_pairs={pairs} max_overlap_px={worst:.1f}"
        + (f"  ({worst_pair})" if worst_pair else "")
    )
    print(
        f"[audit] inside_own_gutter={inside_gutter}/{total} "
        f"inside_figure={inside_figure}/{total} clear_of_plot_area={clear_of_plot}/{total}"
    )
    for line in offenders:
        print(f"plot_w3_amortization: label outside its gutter/figure or over the plot: {line}",
              file=sys.stderr)
    return pairs + len(offenders)


def report_geometry(fig: plt.Figure, panels: list[LabelPanel], titles: list[str]) -> None:
    """Print the final figure size and every axes box as numbers (no image needed)."""
    width_in, height_in = fig.get_size_inches()
    print(
        f"figure: {width_in:.2f} x {height_in:.2f} in @ {fig.dpi:.0f} dpi "
        f"= {width_in * fig.dpi:.0f} x {height_in * fig.dpi:.0f} px"
    )
    renderer = _renderer(fig)
    for panel, title in zip(panels, titles):
        box = panel.ax.get_window_extent(renderer)
        position = panel.ax.get_position()
        print(
            f"  axes[{panel.index}] {title!r}: bbox=({box.x0:.1f}, {box.y0:.1f})-"
            f"({box.x1:.1f}, {box.y1:.1f}) px = {box.width:.1f} x {box.height:.1f} px; "
            f"fig_frac=({position.x0:.3f}, {position.y0:.3f}, "
            f"{position.width:.3f}, {position.height:.3f}); "
            f"gutter_x=({panel.gutter[0]:.3f}, {panel.gutter[1]:.3f})"
        )


def _set_headroom(ax: plt.Axes, values: list[float | None], *, floor_zero: bool = True) -> None:
    finite = [value for value in values if value is not None]
    top = max(finite, default=1.0)
    bottom = min(finite, default=0.0)
    ax.set_ylim(min(0.0, bottom) if floor_zero else bottom - 0.08 * (top - bottom or 1.0),
                top + 0.10 * (top if floor_zero else (top - bottom)) if top > 0 else 1.0)


def _scope_legend(ax: plt.Axes, *, solid_name: str, dashed_name: str) -> None:
    """Colour = ctx, line style = scope. Text stays neutral."""
    handles = [
        Line2D([], [], color=CTX_COLORS[ctx], linewidth=LINE_WIDTH, marker="o",
               markersize=6, label=f"ctx{ctx}")
        for ctx in CTX_ORDER
    ] + [
        Line2D([], [], color=INK_2, linewidth=LINE_WIDTH, linestyle="-", marker="o",
               markersize=6, label=solid_name),
        Line2D([], [], color=INK_2, linewidth=LINE_WIDTH, linestyle=(0, (4, 2)),
               marker="o", markersize=6, markerfacecolor="white", label=dashed_name),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        labelspacing=0.35,
        borderaxespad=0.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)


def _step_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([], [], color=CTX_COLORS[ctx], linewidth=LINE_WIDTH, marker="o",
               markersize=6, label=f"ctx{ctx}")
        for ctx in CTX_ORDER
    ] + [
        Line2D([], [], color=INK_2, linewidth=LINE_WIDTH, linestyle="-", marker="o",
               markersize=6, label="decode event"),
        Line2D([], [], color=INK_2, linewidth=LINE_WIDTH, linestyle=(0, (4, 2)),
               marker="s", markersize=6, markerfacecolor="white", label="host gap"),
    ]
    # Lower right: the step split rises towards the top right, so the empty
    # corner sits below the curves. The direct labels now live in the gutter
    # outside the axes, so the legend no longer has to dodge them.
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        labelspacing=0.35,
        borderaxespad=0.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)


PANEL_TITLES = ("Per-token cost", "Throughput", "Per-step split (mean at active = B)")


def build_figure(
    points: dict[tuple[int, int], Point], n_runs: int
) -> tuple[plt.Figure, list[LabelPanel]]:
    """Three facets, side by side, each with its own label gutter on the right."""
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.patch.set_facecolor(SURFACE)
    rects = _panel_rects(len(PANEL_TITLES))
    axes = [fig.add_axes(rect) for rect in rects]
    series: list[list[tuple[list[int], list[float | None], str, Any]]] = []

    # --- panel 1: per-token cost (ms/token) --------------------------------
    ax = axes[0]
    entries: list[tuple[list[int], list[float | None], str, Any]] = []
    values: list[float | None] = []
    for ctx in CTX_ORDER:
        color = CTX_COLORS[ctx]
        xs, wall = _series(points, ctx, "tok_s", inverse=True)
        _, decode = _series(points, ctx, "decode_only_tok_s", inverse=True)
        _plot_series(ax, xs, wall, color=color, style=WALL_STYLE)
        _plot_series(ax, xs, decode, color=color, style=DECODE_STYLE)
        entries += [
            (xs, wall, f"ctx{ctx} wall", WALL_STYLE["linestyle"]),
            (xs, decode, f"ctx{ctx} decode", DECODE_STYLE["linestyle"]),
        ]
        values += wall + decode
    _set_headroom(ax, values)
    ax.set_title(PANEL_TITLES[0], fontsize=11, loc="left", pad=8)
    ax.set_ylabel("ms / token (lower is better)")
    ax.set_xlabel("batch size B")
    _scope_legend(ax, solid_name="wall clock", dashed_name="decode only")
    series.append(entries)

    # --- panel 2: throughput (tokens/s) ------------------------------------
    ax = axes[1]
    entries = []
    values = []
    for ctx in CTX_ORDER:
        color = CTX_COLORS[ctx]
        xs, wall = _series(points, ctx, "tok_s")
        _, decode = _series(points, ctx, "decode_only_tok_s")
        _plot_series(ax, xs, wall, color=color, style=WALL_STYLE)
        _plot_series(ax, xs, decode, color=color, style=DECODE_STYLE)
        entries += [
            (xs, wall, f"ctx{ctx} wall", WALL_STYLE["linestyle"]),
            (xs, decode, f"ctx{ctx} decode", DECODE_STYLE["linestyle"]),
        ]
        values += wall + decode
    _set_headroom(ax, values)
    ax.set_title(PANEL_TITLES[1], fontsize=11, loc="left", pad=8)
    ax.set_ylabel("tokens / s (higher is better)")
    ax.set_xlabel("batch size B")
    series.append(entries)

    # --- panel 3: per-step split -------------------------------------------
    ax = axes[2]
    entries = []
    values = []
    for ctx in CTX_ORDER:
        color = CTX_COLORS[ctx]
        xs = [B_X[batch] for batch in B_VALUES]
        decode = [points[(ctx, batch)].decode_event_ms if (ctx, batch) in points else None
                  for batch in B_VALUES]
        gap = [points[(ctx, batch)].host_gap_ms if (ctx, batch) in points else None
               for batch in B_VALUES]
        _plot_series(ax, xs, decode, color=color, style=EVENT_STYLE)
        _plot_series(ax, xs, gap, color=color, style=GAP_STYLE)
        entries += [
            (xs, decode, f"ctx{ctx} decode", EVENT_STYLE["linestyle"]),
            (xs, gap, f"ctx{ctx} host gap", GAP_STYLE["linestyle"]),
        ]
        values += decode + gap
    _set_headroom(ax, values)
    ax.set_title(PANEL_TITLES[2], fontsize=11, loc="left", pad=8)
    ax.set_ylabel("ms / step")
    ax.set_xlabel("batch size B")
    _step_legend(ax)
    series.append(entries)

    # --- shared x axis: discrete B, never a log axis with equidistant ticks -
    for ax in axes:
        _apply_chrome(ax)
        ax.set_xticks([B_X[batch] for batch in B_VALUES])
        ax.set_xticklabels([str(batch) for batch in B_VALUES])
        ax.set_xlim(*XLIM)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"W3 E1 · {n_runs} runs", fontsize=14, color=INK, y=0.975)
    fig.text(
        0.5,
        0.925,
        "Batch amortization by context length · numbers from each run's summary.json · "
        f"generated {generated}",
        ha="center",
        va="top",
        fontsize=9,
        color=INK_2,
    )

    panels: list[LabelPanel] = []
    for index, ax in enumerate(axes):
        gutter_x0 = rects[index][0] + PANEL_WIDTH + PLOT_GUTTER_GAP
        items: list[EndLabel] = []
        for xs, ys, text, linestyle in series[index]:
            measured = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not measured:
                continue
            x, y = measured[-1]
            items.append(
                EndLabel(text=text, x=float(x), y=float(y), linestyle=linestyle)
            )
        panels.append(
            LabelPanel(
                index=index,
                ax=ax,
                gutter=(gutter_x0, gutter_x0 + GUTTER_WIDTH),
                items=items,
            )
        )

    _layout_end_labels(fig, panels)
    return fig, panels


# --- CLI -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="results root (the W3 stage is joined)")
    parser.add_argument("--out", default="docs/fig_w3_amortization.png", help="output PNG")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="dump every label box and the numeric label-layout audit",
    )
    parser.add_argument("--show", action="store_true", help="also call plt.show() for a local window")
    args = parser.parse_args()

    stage = _stage_dir(Path(args.results))
    if not stage.is_dir():
        print(f"plot_w3_amortization: no W3 results directory at {stage}", file=sys.stderr)
        print("  E1 has not been run yet; nothing to plot.", file=sys.stderr)
        print("  expected runs: " + ", ".join(f"{rid} (ctx{ctx}/B{batch})"
                                               for rid, (ctx, batch) in E1_RUNS.items()),
              file=sys.stderr)
        print("  run E1 (docs/w3-plan.md §3), then re-run this script.", file=sys.stderr)
        return 2

    points, problems, _ = load_points(stage)
    for problem in problems:
        print(f"plot_w3_amortization: warning: {problem}", file=sys.stderr)

    if not points:
        print(f"plot_w3_amortization: no usable E1 summaries under {stage}", file=sys.stderr)
        print("  expected run directories (summary.json with scenario throughput/steps blocks):",
              file=sys.stderr)
        for run_id, (ctx, batch) in E1_RUNS.items():
            print(f"    {run_id}: ctx={ctx} B={batch}", file=sys.stderr)
        print("  E1 has not been run yet, or the summaries predate the scenario blocks.",
              file=sys.stderr)
        return 2

    missing = [
        f"{run_id} (ctx{ctx}/B{batch})"
        for run_id, (ctx, batch) in E1_RUNS.items()
        if (ctx, batch) not in points
    ]
    if missing:
        print(
            "plot_w3_amortization: warning: missing E1 cells drawn as line gaps (not interpolated): "
            + ", ".join(missing),
            file=sys.stderr,
        )

    n_runs = len({point.run_id for point in points.values()})
    fig, panels = build_figure(points, n_runs)

    # Numeric layout check instead of looking at the image: every label box is
    # measured with the live renderer. A figure that fails is never written.
    report_geometry(fig, panels, list(PANEL_TITLES))
    violations = audit_label_layout(fig, panels, verbose=args.debug)
    if violations:
        print(
            f"plot_w3_amortization: refusing to write {args.out}: "
            f"{violations} label-layout violation(s) (see [audit] above)",
            file=sys.stderr,
        )
        plt.close(fig)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    try:
        fig.savefig(temporary, dpi=FIG_DPI, format="png", facecolor=SURFACE)
        temporary.replace(out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"wrote {out} ({len(points)} points from {n_runs} runs, "
        f"{out.stat().st_size / 1024:.1f} KiB)"
    )

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
