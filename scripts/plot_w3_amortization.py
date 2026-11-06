#!/usr/bin/env python3
"""Plot W3 E1: batching amortizes the per-step host cost across context lengths.

Reads **only** each E1 run's ``summary.json`` (never ``metrics.jsonl``) from
``results/W3/0017-...`` through ``results/W3/0019-...`` plus ``W3-0006`` /
``W3-0007`` and writes ``figures/fig_w3_amortization.png``. The original ctx512
cells (``0003-0005``) emitted EOS after 3 tokens and are superseded by
``0017-0019`` at ctx511 (see PITFALLS P-31); 511 is 1 token below 512.

Layout: a 2x2 GridSpec, four views of the same five measurements.

* top-left  -- per-token cost, ms/token;
* top-right -- throughput, tokens/s;
* bottom-left  -- the per-step split as stacked bars, ``decode_event_ms`` +
  ``prefill_ms`` + ``host_gap_ms`` at ``active == max_batch`` (they sum to
  ``step_ms``, printed at the end of every bar);
* bottom-right -- ``host_gap_ms`` alone on a 0-0.16 ms axis, because 0.1 ms is
  0.5 % of a 21-27 ms step and is invisible at the left panel's scale.

In the two top panels the decode-only scope is a solid line with small dots;
the wall-clock scope is a hollow marker shifted ``WALL_OFFSET`` category units
to the right and drawn without a line. The two scopes therefore never
overprint, and one figure-level legend explains colour (ctx) and style (scope).

Why facets instead of a dual y axis: ms/token and tok/s are the same quantity
inverted, so two y scales would only invent an alignment. The host-gap zoom is
a facet too: its own axis, never a second y axis stacked on the bars. Colour
follows the context length (an entity), not the rank; line style follows the
measurement scope, and the bar shade (solid / light / dark sliver) follows the
step segment.

Missing data: a missing ``summary.json`` / metric makes that (ctx, B) point a
gap (the line is broken, nothing is interpolated). If *no* E1 point can be
loaded, the script prints an explanation and exits non-zero without touching
the output file. A bad figure is never written: the PNG is saved to a hidden
temporary file first and only then atomically moved into place.

Text is kept to the five measurements: the only free text inside the axes is
two ratio notes, the bar totals, the host-gap values and the two ctx group
heads. Every one of those boxes is measured with the live renderer by
``audit_label_layout``; the PNG is not written if a box overlaps another label,
a marker, a data line or its panel edge.

Usage::

    python scripts/plot_w3_amortization.py
    python scripts/plot_w3_amortization.py --results results --out figures/fig_w3_amortization.png
    python scripts/plot_w3_amortization.py --debug        # per-label boxes + audit dump
    python scripts/plot_w3_amortization.py --show
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
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
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# --- style constants -------------------------------------------------------
# Fixed categorical order from the dataviz reference palette (light mode),
# slots 1-2: blue then orange. Colour is keyed by ctx, never by rank, and is
# never cycled. The two-colour subset passed
#   validate_palette.js "#2a78d6,#eb6834" --mode light
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text / ticks / labels
MUTED = "#7c7a74"        # ratio annotations
GRID = "#e1e0d9"         # hairline grid
AXIS = "#c3c2b7"         # baseline / spine
GAP_DARK = "#3c3b37"     # host-gap segment: neutral, never a ctx colour
CTX_COLORS = {511: "#2a78d6", 2048: "#eb6834"}
CTX_ORDER = (511, 2048)  # fixed assignment: 511 -> slot 1, 2048 -> slot 2
B_VALUES = (1, 8, 32)    # discrete x positions, not a log axis
B_X = {batch: index for index, batch in enumerate(B_VALUES)}
LINE_WIDTH = 1.3
DECODE_MARKER = 3.6      # solid dot on the decode line
WALL_MARKER = 4.3        # hollow ring beside the line
WALL_OFFSET = 0.07       # category units to the right of the decode point

# --- typography (points) ---------------------------------------------------
TITLE_FS = 14.0
SUBTITLE_FS = 9.5
PANEL_TITLE_FS = 11.0
AXIS_LABEL_FS = 9.5
TICK_FS = 8.8
LEGEND_FS = 9.0
LABEL_FS = 8.8           # bar totals, host-gap values, ratio notes

# --- layout (figure fractions) ---------------------------------------------
# A 2x2 grid with one header band. Panel width and height are set by the grid;
# every axis then gets the same margins and the same title/legend treatment.
FIG_SIZE = (11.8, 8.4)   # inches
FIG_DPI = 150
GRID_LEFT = 0.075
GRID_RIGHT = 0.982
GRID_BOTTOM = 0.085
GRID_TOP = 0.855
GRID_WSPACE = 0.20
GRID_HSPACE = 0.38
XLIM = (-0.30, 2.30)     # room for the offset wall markers at B=32
MARKER_HALO_PX = 6.0     # audit clearance around a drawn marker
LABEL_PAD_PX = 2.0       # audit tolerance around the axes box

# Stacked-bar rows, top to bottom: ctx511 B=1/8/32, a gap, ctx2048 B=1/8.
# ``y`` is the data coordinate of the row centre; the blank 1.4-0.9 gap keeps
# the two contexts visually grouped without a second legend.
ROWS: tuple[tuple[int, int, float, str], ...] = (
    (511, 1, 4.52, "B=1"),
    (511, 8, 3.68, "B=8"),
    (511, 32, 2.84, "B=32"),
    (2048, 1, 1.34, "B=1"),
    (2048, 8, 0.50, "B=8"),
)
GROUP_HEADERS = {511: 5.02, 2048: 1.86}
ROW_YLIM = (0.0, 5.42)
BAR_HEIGHT = 0.60

# E1 matrix (docs/decisions.md D5). B is re-read from protocol.max_batch when
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
    prefill_ms: float | None
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


def _step_means(
    summary: dict[str, Any], batch: int
) -> tuple[float | None, float | None, float | None, float | None, int | None]:
    """Mean step split at the full batch, else at the largest active value seen."""
    entries = (summary.get("steps") or {}).get("by_active") or []
    candidates = [entry for entry in entries if isinstance(entry, dict) and _finite(entry.get("active")) is not None]
    if not candidates:
        return None, None, None, None, None
    exact = [entry for entry in candidates if int(float(entry["active"])) == batch]
    picked = exact[0] if exact else max(candidates, key=lambda entry: float(entry["active"]))
    n = picked.get("n")
    return (
        _finite(picked.get("decode_event_ms")),
        _finite(picked.get("prefill_ms")),
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
    decode_event, prefill, host_gap, step_ms, n_steps = _step_means(summary, batch)
    return Point(
        ctx=ctx,
        batch=batch,
        run_id=run_id,
        tok_s=_median(summary, "throughput", "tok_s"),
        tok_s_active=_median(summary, "throughput", "tok_s_active"),
        decode_only_tok_s=_median(summary, "throughput", "decode_only_tok_s"),
        decode_event_ms=decode_event,
        prefill_ms=prefill,
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
            problems.append(f"{run_id}: cannot resolve ctx/B (expected ctx{ctx} B{batch}); skipped")
            continue
        points[(point.ctx, point.batch)] = point
    return points, problems, [p for p in problems if "cannot resolve" in p]


# --- plotting --------------------------------------------------------------


def _apply_chrome(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=INK_2, labelsize=TICK_FS, length=2.5, width=0.7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


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


def _measured(xs: list[int], ys: list[float | None]) -> tuple[list[float], list[float]]:
    """Drop gaps, keeping x/y paired (a gap breaks the line, never shifts it)."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _set_ylim(ax: plt.Axes, values: list[float | None], *, headroom: float = 0.10) -> None:
    finite = [value for value in values if value is not None]
    top = max(finite, default=1.0)
    ax.set_ylim(0.0, top * (1.0 + headroom))


def _mix(color: str, base: str, weight: float) -> str:
    """Blend ``color`` over ``base`` in sRGB; weight is the share of ``color``."""
    first = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    second = tuple(int(base[i : i + 2], 16) for i in (1, 3, 5))
    blended = (round(first[i] * weight + second[i] * (1.0 - weight)) for i in range(3))
    return "#" + "".join(f"{channel:02x}" for channel in blended)


def _ratio(points: dict[tuple[int, int], Point], ctx: int, low: int, high: int) -> float | None:
    first = points.get((ctx, low))
    second = points.get((ctx, high))
    if first is None or second is None:
        return None
    if not first.decode_only_tok_s or not second.decode_only_tok_s:
        return None
    return second.decode_only_tok_s / first.decode_only_tok_s


@dataclass
class EndLabel:
    """One free text box inside an axes: where it goes and its text."""

    text: str
    x: float
    y: float
    dx_pt: float = 0.0
    dy_pt: float = 0.0
    ha: str = "left"
    va: str = "center"
    artist: Any = None


@dataclass
class LabelPanel:
    """An axes, its free text and the data ink those boxes must miss."""

    index: int
    ax: plt.Axes
    items: list[EndLabel] = field(default_factory=list)
    series: list[tuple[list[float], list[float]]] = field(default_factory=list)
    obstacles: list[Any] = field(default_factory=list)


def _renderer(fig: plt.Figure) -> Any:
    """Agg exposes ``get_renderer``; other canvases (``--show``) expose ``renderer``."""
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig.canvas.renderer


def place_labels(fig: plt.Figure, panels: list[LabelPanel]) -> None:
    """Draw every free text box at its anchor (data coords) or point offset.

    Text that belongs to a bar end or a lollipop is placed directly in data
    coordinates; the two ratio notes use a point offset so they keep a fixed
    gap above the line at any dpi. No arrows: these are value labels, not
    annotations with leaders.
    """
    for panel in panels:
        for item in panel.items:
            if item.artist is not None:
                continue
            if item.dx_pt or item.dy_pt:
                item.artist = panel.ax.annotate(
                    item.text,
                    xy=(item.x, item.y),
                    xytext=(item.dx_pt, item.dy_pt),
                    textcoords="offset points",
                    ha=item.ha,
                    va=item.va,
                    fontsize=LABEL_FS,
                    color=MUTED,
                    annotation_clip=False,
                    zorder=6,
                )
            else:
                item.artist = panel.ax.text(
                    item.x,
                    item.y,
                    item.text,
                    ha=item.ha,
                    va=item.va,
                    fontsize=LABEL_FS,
                    color=INK_2,
                    zorder=6,
                )
    fig.canvas.draw()


def _inside(inner: Any, outer: Any, pad: float = 0.0) -> bool:
    return (
        inner.x0 >= outer.x0 - pad
        and inner.y0 >= outer.y0 - pad
        and inner.x1 <= outer.x1 + pad
        and inner.y1 <= outer.y1 + pad
    )


def _point_in_box(point: tuple[float, float], box: Any, pad: float = 0.0) -> bool:
    return (
        box.x0 - pad <= point[0] <= box.x1 + pad
        and box.y0 - pad <= point[1] <= box.y1 + pad
    )


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a: Any, b: Any, c: Any, d: Any) -> bool:
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
        return False
    if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    return first * second <= 0 and third * fourth <= 0


def _segment_hits_box(
    start: tuple[float, float], end: tuple[float, float], box: Any, pad: float = 0.0
) -> bool:
    """True when the segment enters the (optionally inflated) text box."""
    x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    if _point_in_box(start, box, pad) or _point_in_box(end, box, pad):
        return True
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    for index in range(4):
        if _segments_cross(start, end, corners[index], corners[(index + 1) % 4]):
            return True
    return False


def audit_label_layout(
    fig: plt.Figure, panels: list[LabelPanel], *, verbose: bool = False
) -> int:
    """Numeric self-check of the free text; returns the violation count.

    Nothing here looks at the PNG. Every text box comes from the live renderer
    (``get_window_extent``), so the numbers are the numbers a reader would
    measure on the raster. Checked per label:

    * no overlap with any other label box;
    * no overlap with a drawn marker or a segment of any series in the panel;
    * no overlap with the panel's legend or its plot edge;
    * the whole box stays inside the figure.

    ``verbose`` (``--debug``) also dumps every box and its anchor.
    """
    fig.canvas.draw()
    renderer = _renderer(fig)
    fig_box = fig.bbox
    entries: list[tuple[LabelPanel, EndLabel, Any, Any]] = []
    for panel in panels:
        axes_box = panel.ax.get_window_extent(renderer)
        for item in panel.items:
            if item.artist is None:
                continue
            box = item.artist.get_window_extent(renderer)
            entries.append((panel, item, box, axes_box))
            if verbose:
                print(
                    f"  [label] panel {panel.index + 1} {item.text!r:10s} "
                    f"anchor=({item.x:.3f},{item.y:.3f}) "
                    f"box=({box.x0:7.1f},{box.y0:6.1f})-({box.x1:7.1f},{box.y1:6.1f}) "
                    f"axes=({axes_box.x0:.1f},{axes_box.y0:.1f})-"
                    f"({axes_box.x1:.1f},{axes_box.y1:.1f})"
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

    outside_panel = 0
    outside_figure = 0
    over_data = 0
    over_obstacle = 0
    offenders: list[str] = []
    for panel, item, box, axes_box in entries:
        ok_panel = _inside(box, axes_box, pad=LABEL_PAD_PX)
        ok_figure = _inside(box, fig_box)
        hits: list[str] = []

        transform = panel.ax.transData
        for xs, ys in panel.series:
            points = [tuple(point) for point in transform.transform(list(zip(xs, ys)))]
            for index, point in enumerate(points):
                if _point_in_box(point, box, pad=MARKER_HALO_PX):
                    hits.append(f"marker@({xs[index]:.3g},{ys[index]:.3g})")
                    break
            for index in range(len(points) - 1):
                if _segment_hits_box(points[index], points[index + 1], box, pad=1.0):
                    hits.append(
                        f"segment@({xs[index]:.3g}->{xs[index + 1]:.3g})"
                    )
                    break

        obstacle_hits: list[str] = []
        for obstacle in panel.obstacles:
            if obstacle.get_window_extent(renderer).overlaps(box):
                obstacle_hits.append("legend")

        outside_panel += int(not ok_panel)
        outside_figure += int(not ok_figure)
        over_data += int(bool(hits))
        over_obstacle += int(bool(obstacle_hits))
        if not (ok_panel and ok_figure) or hits or obstacle_hits:
            offenders.append(
                f"panel {panel.index + 1} {item.text!r}: box=({box.x0:.1f},{box.y0:.1f})-"
                f"({box.x1:.1f},{box.y1:.1f}) axes=({axes_box.x0:.1f},{axes_box.y0:.1f})-"
                f"({axes_box.x1:.1f},{axes_box.y1:.1f}) panel_ok={ok_panel} "
                f"figure_ok={ok_figure} data_hits={hits or 'none'} "
                f"obstacle_hits={obstacle_hits or 'none'}"
            )

    total = len(entries)
    print(
        f"[audit] labels={total} intersecting_pairs={pairs} max_overlap_px={worst:.1f}"
        + (f"  ({worst_pair})" if worst_pair else "")
    )
    print(
        f"[audit] inside_panel={total - outside_panel}/{total} "
        f"inside_figure={total - outside_figure}/{total} "
        f"clear_of_data={total - over_data}/{total} "
        f"clear_of_legend={total - over_obstacle}/{total}"
    )
    for line in offenders:
        print(f"plot_w3_amortization: label layout violation: {line}", file=sys.stderr)
    return pairs + len(offenders)


def report_geometry(fig: plt.Figure, titles: list[str]) -> None:
    """Print the final figure size and every axes box as numbers (no image needed)."""
    width_in, height_in = fig.get_size_inches()
    print(
        f"figure: {width_in:.2f} x {height_in:.2f} in @ {fig.dpi:.0f} dpi "
        f"= {width_in * fig.dpi:.0f} x {height_in * fig.dpi:.0f} px"
    )
    renderer = _renderer(fig)
    for index, (ax, title) in enumerate(zip(fig.axes, titles)):
        box = ax.get_window_extent(renderer)
        position = ax.get_position()
        print(
            f"  axes[{index}] {title!r}: bbox=({box.x0:.1f}, {box.y0:.1f})-"
            f"({box.x1:.1f}, {box.y1:.1f}) px = {box.width:.1f} x {box.height:.1f} px; "
            f"fig_frac=({position.x0:.3f}, {position.y0:.3f}, "
            f"{position.width:.3f}, {position.height:.3f})"
        )


def _draw_metric(
    ax: plt.Axes,
    points: dict[tuple[int, int], Point],
    panel: LabelPanel,
    *,
    invert: bool,
    title: str,
    ylabel: str,
) -> list[float | None]:
    """One metric: decode-only solid line + offset hollow wall-clock markers."""
    values: list[float | None] = []
    for ctx in CTX_ORDER:
        color = CTX_COLORS[ctx]
        xs, decode = _series(points, ctx, "decode_only_tok_s", inverse=invert)
        _, wall = _series(points, ctx, "tok_s", inverse=invert)
        ax.plot(
            xs,
            decode,
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=DECODE_MARKER,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.7,
            zorder=3,
        )
        wall_x = [x + WALL_OFFSET for x in xs]
        ax.plot(
            wall_x,
            wall,
            linestyle="none",
            marker="o",
            markersize=WALL_MARKER,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=4,
        )
        panel.series.append(_measured(xs, decode))
        for wx, wy in zip(wall_x, wall):
            if wy is not None:
                panel.series.append(([float(wx)], [float(wy)]))
        values += decode + wall

    _set_ylim(ax, values, headroom=0.10)
    _apply_chrome(ax, grid_axis="y")
    ax.set_xticks([B_X[batch] for batch in B_VALUES])
    ax.set_xticklabels([str(batch) for batch in B_VALUES])
    ax.set_xlim(*XLIM)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.set_title(title, fontsize=PANEL_TITLE_FS, loc="left", pad=6, color=INK)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS, color=INK_2)
    ax.set_xlabel("batch size B (categorical)", fontsize=AXIS_LABEL_FS, color=INK_2)
    return values


def _draw_split(
    ax: plt.Axes,
    points: dict[tuple[int, int], Point],
    panel: LabelPanel,
) -> None:
    """Stacked per-step split; every bar is labelled with its total step_ms."""
    for ctx, batch, y, _label in ROWS:
        point = points.get((ctx, batch))
        if point is None:
            continue
        color = CTX_COLORS[ctx]
        segments = (
            (point.decode_event_ms, color),
            (point.prefill_ms, _mix(color, SURFACE, 0.40)),
            (point.host_gap_ms, GAP_DARK),
        )
        left = 0.0
        for width, fill in segments:
            if width is None or width <= 0:
                continue
            ax.barh(
                y,
                width,
                left=left,
                height=BAR_HEIGHT,
                color=fill,
                edgecolor=SURFACE,
                linewidth=0.9,
                zorder=3,
            )
            left += width
        if point.step_ms is not None:
            panel.items.append(
                EndLabel(
                    f"{point.step_ms:.1f} ms",
                    x=max(left, point.step_ms) + 0.55,
                    y=y,
                    ha="left",
                    va="center",
                )
            )
            panel.series.append(([float(max(left, point.step_ms))], [float(y)]))

    max_step = max((point.step_ms or 0.0) for point in points.values())
    ax.set_xlim(0.0, max_step * 1.16)
    ax.set_ylim(*ROW_YLIM)
    ax.set_yticks([row[2] for row in ROWS])
    ax.set_yticklabels([row[3] for row in ROWS])
    _apply_chrome(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_title(
        "Step split · decode + prefill + host gap",
        fontsize=PANEL_TITLE_FS,
        loc="left",
        pad=6,
        color=INK,
    )
    ax.set_xlabel("ms / step", fontsize=AXIS_LABEL_FS, color=INK_2)
    for ctx, y in GROUP_HEADERS.items():
        header = ax.text(
            0.02,
            y,
            f"ctx{ctx}",
            ha="left",
            va="bottom",
            fontsize=TICK_FS + 0.4,
            fontweight="semibold",
            color=CTX_COLORS[ctx],
            zorder=5,
        )
        panel.items.append(EndLabel(f"ctx{ctx}", x=0.02, y=y, artist=header))


def _draw_gap(
    ax: plt.Axes,
    points: dict[tuple[int, int], Point],
    panel: LabelPanel,
) -> None:
    """Host-gap zoom: one lollipop per cell, own 0-0.16 ms axis."""
    for ctx, batch, y, _label in ROWS:
        point = points.get((ctx, batch))
        if point is None or point.host_gap_ms is None:
            continue
        color = CTX_COLORS[ctx]
        ax.hlines(y, 0.0, point.host_gap_ms, color=color, linewidth=1.3, zorder=3)
        ax.plot(
            [point.host_gap_ms],
            [y],
            marker="o",
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )
        panel.items.append(
            EndLabel(
                f"{point.host_gap_ms:.3f}",
                x=point.host_gap_ms + 0.0045,
                y=y,
                ha="left",
                va="center",
            )
        )
        panel.series.append(([float(point.host_gap_ms)], [float(y)]))

    ax.set_xlim(0.0, 0.16)
    ax.set_ylim(*ROW_YLIM)
    ax.set_yticks([row[2] for row in ROWS])
    ax.set_yticklabels([row[3] for row in ROWS])
    ax.set_xticks([0.0, 0.04, 0.08, 0.12, 0.16])
    _apply_chrome(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_title("Host gap · zoom", fontsize=PANEL_TITLE_FS, loc="left", pad=6, color=INK)
    ax.set_xlabel("ms / step", fontsize=AXIS_LABEL_FS, color=INK_2)


def _figure_legend(fig: plt.Figure) -> None:
    """One shared key: colour = ctx, style = scope."""
    handles = [
        Patch(facecolor=CTX_COLORS[511], edgecolor="none", label="ctx511"),
        Patch(facecolor=CTX_COLORS[2048], edgecolor="none", label="ctx2048"),
        Line2D([], [], color=INK_2, linewidth=LINE_WIDTH, marker="o",
               markersize=DECODE_MARKER, label="decode only"),
        Line2D([], [], color=INK_2, linewidth=0, marker="o",
               markersize=WALL_MARKER, markerfacecolor="none",
               markeredgecolor=INK_2, markeredgewidth=1.0, label="wall clock"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(GRID_RIGHT, 0.966),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.5,
        borderaxespad=0.0,
    )


def build_figure(
    points: dict[tuple[int, int], Point], n_runs: int
) -> tuple[plt.Figure, list[LabelPanel], list[str]]:
    """A 2x2 GridSpec: cost | throughput above, step split | gap zoom below."""
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(
        2,
        2,
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )
    ax_cost = fig.add_subplot(grid[0, 0])
    ax_rate = fig.add_subplot(grid[0, 1])
    ax_split = fig.add_subplot(grid[1, 0])
    ax_gap = fig.add_subplot(grid[1, 1])

    titles = [
        "Per-token cost",
        "Throughput",
        "Step split · decode + prefill + host gap",
        "Host gap · zoom",
    ]
    panels = [
        LabelPanel(index=0, ax=ax_cost),
        LabelPanel(index=1, ax=ax_rate),
        LabelPanel(index=2, ax=ax_split),
        LabelPanel(index=3, ax=ax_gap),
    ]

    _draw_metric(
        ax_cost,
        points,
        panels[0],
        invert=True,
        title=titles[0],
        ylabel="ms / token",
    )
    _draw_metric(
        ax_rate,
        points,
        panels[1],
        invert=False,
        title=titles[1],
        ylabel="tokens / s",
    )

    # Two ratio notes are the only conclusion text in the figure; they sit a
    # fixed point offset above the decode line, halfway along each segment.
    first = points.get((511, 1))
    second = points.get((511, 8))
    third = points.get((511, 32))
    if first and second and first.decode_only_tok_s and second.decode_only_tok_s:
        ratio = _ratio(points, 511, 1, 8)
        if ratio is not None:
            panels[1].items.append(
                EndLabel(
                    f"×{ratio:.1f}",
                    x=0.5,
                    y=(first.decode_only_tok_s + second.decode_only_tok_s) / 2.0,
                    dy_pt=13.0,
                    ha="center",
                    va="bottom",
                )
            )
    if second and third and second.decode_only_tok_s and third.decode_only_tok_s:
        ratio = _ratio(points, 511, 8, 32)
        if ratio is not None:
            panels[1].items.append(
                EndLabel(
                    f"×{ratio:.1f}",
                    x=1.5,
                    y=(second.decode_only_tok_s + third.decode_only_tok_s) / 2.0,
                    dy_pt=13.0,
                    ha="center",
                    va="bottom",
                )
            )

    _draw_split(ax_split, points, panels[2])
    _draw_gap(ax_gap, points, panels[3])

    # --- shared header: one short title, one provenance line, one key ------
    fig.text(
        GRID_LEFT,
        0.962,
        "Batching amortizes the decode step",
        ha="left",
        va="top",
        fontsize=TITLE_FS,
        color=INK,
    )
    fig.text(
        GRID_LEFT,
        0.925,
        "5 runs × 5 repeats · medians from summary.json",
        ha="left",
        va="top",
        fontsize=SUBTITLE_FS,
        color=INK_2,
    )
    _figure_legend(fig)

    place_labels(fig, panels)
    return fig, panels, titles


# --- CLI -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="results root (the W3 stage is joined)")
    parser.add_argument("--out", default="figures/fig_w3_amortization.png", help="output PNG")
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
        print("  run E1 (docs/decisions.md D5), then re-run this script.", file=sys.stderr)
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
    fig, panels, titles = build_figure(points, n_runs)

    # Numeric layout check instead of looking at the image: every label box is
    # measured with the live renderer. A figure that fails is never written.
    report_geometry(fig, titles)
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
