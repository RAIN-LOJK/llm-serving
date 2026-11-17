#!/usr/bin/env python3
"""Generate the W8 criterion matrix (markdown + figure) from committed summaries.

Every number comes from a committed ``summary.json`` / ``verdict.json``; the script
validates the run's protocol before rendering and aborts on a dirty cell instead of
leaving a partial table.

Cells (2 arms x 3 contexts x 3 batch sizes = 18):

* pure  Qwen3-0.6B: short row is **511** (P-31; 512 EOS), 2k, 8k;
* hybrid Qwen3.5-0.8B: 512, 2k, 8k.

Batch 1/8 cells for 2k/8k come from the W7 E3 runs (one run covers all three contexts);
batch 32 cells come from the W8 single-context runs. ``8k x B=32`` for pure is a
structural N/A (28.1 GiB of KV alone); ``W8-0008`` measures the B=20 bound.

Usage::

    python scripts/w8_matrix.py
    python scripts/w8_matrix.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "reports"

PURE = "pure"
HYBRID = "hybrid"
CONTEXTS = ("short", "2k", "8k")
BATCHES = (1, 8, 32)

# analytic per-sequence state (bytes): pure = 2*8*128*2*28 per token; hybrid = 2*2*256*2*6
# per token plus the 18.633 MiB linear slots (D9).
PURE_TOKEN_BYTES = 114_688
HYBRID_TOKEN_BYTES = 12_288
HYBRID_SLOT_MB = 18.6328125

# arm -> ctx -> batch -> (run id, by_prompt row, static note)
CELLS: dict[str, dict[str, dict[int, tuple[str, str, str]]]] = {
    PURE: {
        "short": {1: ("W8-0004", "ctx511", ""), 8: ("W8-0005", "ctx511", ""), 32: ("W8-0006", "ctx511", "")},
        "2k": {1: ("W7-0027", "ctx2048", ""), 8: ("W7-0029", "ctx2048", ""), 32: ("W8-0007", "ctx2048", "")},
        "8k": {
            1: ("W7-0027", "ctx8192", "decode N/A (EOS at 2 tokens)"),
            8: ("W7-0029", "ctx8192", "decode N/A (EOS at 2 tokens)"),
            32: ("", "", "N/A: B=20 bound (W8-0008)"),
        },
    },
    HYBRID: {
        "short": {1: ("W7-0026", "ctx512", ""), 8: ("W7-0028", "ctx512", ""), 32: ("W8-0001", "ctx512", "")},
        "2k": {1: ("W7-0026", "ctx2048", ""), 8: ("W7-0028", "ctx2048", ""), 32: ("W8-0002", "ctx2048", "")},
        "8k": {1: ("W7-0026", "ctx8192", ""), 8: ("W7-0028", "ctx8192", ""), 32: ("W8-0003", "ctx8192", "")},
    },
}

CTX_TOKENS = {PURE: {"short": 511, "2k": 2048, "8k": 8192}, HYBRID: {"short": 512, "2k": 2048, "8k": 8192}}
CTX_LABEL = {PURE: "511", HYBRID: "512"}


def _summary(run: str) -> dict:
    run_dir = next((RESULTS / "W8").glob(run.lower().replace("w8-", "") + "-*")) if run.startswith("W8") else None
    if run.startswith("W7"):
        run_dir = next((RESULTS / "W7").glob(run.lower().replace("w7-", "") + "-*"))
    if run_dir is None or not (run_dir / "summary.json").exists():
        raise SystemExit(f"missing summary for {run}")
    return json.loads((run_dir / "summary.json").read_text())


def _verdict(run: str) -> str:
    if run.startswith("W8"):
        run_dir = next((RESULTS / "W8").glob(run.lower().replace("w8-", "") + "-*"))
    else:
        run_dir = next((RESULTS / "W7").glob(run.lower().replace("w7-", "") + "-*"))
    return json.loads((run_dir / "verdict.json").read_text()).get("verdict", "unreviewed")


def _row(summary: dict, prompt: str) -> dict:
    for row in summary.get("by_prompt") or []:
        if row.get("prompt") == prompt:
            return row
    raise SystemExit(f"no by_prompt row {prompt!r} (have {[r.get('prompt') for r in summary.get('by_prompt') or []]})")


def _validate(run: str, summary: dict) -> None:
    protocol = summary.get("protocol") or {}
    if summary.get("device_class") != "measure-cuda":
        raise SystemExit(f"{run}: device_class={summary.get('device_class')!r}, expected measure-cuda")
    if protocol.get("max_new_tokens") != 128:
        raise SystemExit(f"{run}: max_new_tokens={protocol.get('max_new_tokens')!r}, expected 128")
    if _verdict(run) == "invalid":
        raise SystemExit(f"{run}: verdict=invalid, refusing to render")


def collect() -> list[dict]:
    cache: dict[str, dict] = {}
    cells: list[dict] = []
    for arm in (PURE, HYBRID):
        for ctx in CONTEXTS:
            for batch in BATCHES:
                run, prompt, note = CELLS[arm][ctx][batch]
                if not run:
                    cells.append({"arm": arm, "ctx": ctx, "batch": batch, "run": "", "note": note})
                    continue
                if run not in cache:
                    cache[run] = _summary(run)
                    _validate(run, cache[run])
                summary = cache[run]
                row = _row(summary, prompt)
                out = (row.get("out_tokens") or {}).get("median")
                tpot = (row.get("tpot_ms") or {}).get("median")
                ttft = (row.get("ttft_ms") or {}).get("median")
                if out is not None and out <= 4:
                    tpot = None
                    note = note or f"decode N/A (EOS at {int(out)} tokens)"
                tokens = CTX_TOKENS[arm][ctx] + 128
                state_mb = (
                    (PURE_TOKEN_BYTES * tokens) / 2**20
                    if arm == PURE
                    else (HYBRID_TOKEN_BYTES * tokens) / 2**20 + HYBRID_SLOT_MB
                )
                cells.append(
                    {
                        "arm": arm,
                        "ctx": ctx,
                        "batch": batch,
                        "run": run,
                        "ttft_ms": ttft,
                        "tpot_ms": tpot,
                        "out_tokens": out,
                        "state_mb": state_mb,
                        "peak_concurrency": (summary.get("capacity") or {}).get("peak_concurrency"),
                        "queue_share_p99": (summary.get("latency") or {}).get("queue_share_p99"),
                        "note": note,
                    }
                )
    return cells


def _fmt(value, digits=1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def table(cells: list[dict]) -> str:
    lines = [
        "# W8 判据矩阵（2 臂 × 3 ctx × 3 B）",
        "",
        "> 由 `scripts/w8_matrix.py` 从 `summary.json` 生成，请勿手改。",
        "> 列：TTFT p50 / decode TPOT p50（每请求 `decode_event` 累加）/ out tokens /",
        "> 每序列状态（解析式）；`—` 表示该格不可测，原因见备注列。",
        "",
        "| 臂 | ctx | B | TTFT p50 (ms) | queue p99 | TPOT p50 (ms) | out tok | 状态/序列 (MiB) | 峰值并发 | run | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        ctx = CTX_LABEL[cell["arm"]] if cell["ctx"] == "short" else cell["ctx"]
        lines.append(
            "| {arm} | {ctx} | {batch} | {ttft} | {qs} | {tpot} | {out} | {state} | {conc} | {run} | {note} |".format(
                arm=cell["arm"],
                ctx=ctx,
                batch=cell["batch"],
                ttft=_fmt(cell.get("ttft_ms")),
                qs=_fmt(cell.get("queue_share_p99"), 2),
                tpot=_fmt(cell.get("tpot_ms"), 2),
                out=_fmt(cell.get("out_tokens"), 0),
                state=_fmt(cell.get("state_mb")),
                conc=cell.get("peak_concurrency") if cell.get("peak_concurrency") is not None else "—",
                run=cell.get("run") or "—",
                note=cell.get("note") or "",
            )
        )
    lines += [
        "",
        "注：",
        "- pure 的 short 行是 **511**（512 合成 prompt 第 3 个 token 即 EOS，P-31）；hybrid 是 512。",
        "- 每序列状态是解析式（pure 112 KiB/token；hybrid 12 KiB/token + 18.63 MiB 线性槽），不是实测列。",
        "- B=1/8 的 2k/8k 来自 W7 E3（一个 run 覆盖三个 ctx）；峰值并发与 queue p99 是 run 级，不按 ctx 拆。",
        "- TTFT 是端到端（含排队）：B=1 的行受同批前序请求的 128-token decode 串行影响，**不可跨格比较**；",
        "  架构成本看 TPOT。",
        "- pure 8k 的 decode 为 N/A（EOS 早退），8k×B=32 为结构性 N/A（28.1 GiB KV > 24.5 GiB）。",
        "",
    ]
    return "\n".join(lines)


def figure(cells: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.full((2, 3, 3), np.nan)
    for cell in cells:
        arm_i = 0 if cell["arm"] == PURE else 1
        ctx_i = CONTEXTS.index(cell["ctx"])
        b_i = BATCHES.index(cell["batch"])
        if cell.get("tpot_ms") is not None:
            values[arm_i, b_i, ctx_i] = cell["tpot_ms"]

    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="#fcfcfb")
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e8e7e1")
    for b_i, batch in enumerate(BATCHES):
        ax = axes[b_i]
        ax.set_facecolor("#fcfcfb")
        ax.imshow(np.ma.masked_invalid(values[:, b_i, :]), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3), ["511/512", "2k", "8k"])
        ax.set_yticks(range(2), ["pure 0.6B", "hybrid 0.8B"])
        for arm_i in range(2):
            for ctx_i in range(3):
                value = values[arm_i, b_i, ctx_i]
                text = "N/A" if not np.isfinite(value) else f"{value:.1f}"
                ax.text(ctx_i, arm_i, text, ha="center", va="center", fontsize=9,
                        color="#0b0b0b" if (not np.isfinite(value) or value < 0.6 * vmax) else "white")
        ax.set_title(f"B={batch}", fontsize=11, color="#0b0b0b")
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, 2), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=2)
        ax.tick_params(which="both", length=0)
    fig.suptitle("W8 decode TPOT (ms): pure 0.6B vs hybrid 0.8B, by context and batch", fontsize=12)
    fig.text(0.5, 0.01, "N/A: pure 8k decode hits EOS after 2 tokens; pure 8k x B=32 needs 28.1 GiB of KV.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate runs only")
    parser.add_argument("--report", default=str(REPORT / "matrix.md"))
    parser.add_argument("--figure", default=str(FIGURES / "fig_w8_matrix.png"))
    args = parser.parse_args()

    cells = collect()
    text = table(cells)
    if args.check:
        print(f"{len(cells)} cells ok")
        return 0
    Path(args.report).write_text(text, encoding="utf-8")
    figure(cells, Path(args.figure))
    print(f"wrote {args.report} and {args.figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
