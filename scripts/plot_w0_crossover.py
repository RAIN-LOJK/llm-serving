#!/usr/bin/env python3
"""Plot the W0 carrier probe: per-sequence state, prefill, and decode vs context.

Reads one run's ``summary.json`` (default: ``results/W0/0001-carrier-probe``) and writes
``figures/fig_w0_crossover.png``. Numbers come from the run directory, never from this file;
the analytic account lives in ``docs/carrier.md``.

Usage::

    python scripts/plot_w0_crossover.py
    python scripts/plot_w0_crossover.py --run results/W0/0001-carrier-probe --out /tmp/w0.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
COLORS = {"Qwen/Qwen3-0.6B": "#1f77b4", "Qwen/Qwen3.5-0.8B": "#d62728"}
LABELS = {"Qwen/Qwen3-0.6B": "Qwen3-0.6B (full attention)", "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B (3:1 hybrid)"}

# State at exactly N tokens. ``cache_mb`` is post-decode and adds (out_tokens-1) tokens.
KEY = "cache_prefill_mb"


def _series(groups: dict[tuple[str, int], dict], model: str, key: str) -> tuple[list[int], list[float]]:
    ctxs = sorted(c for (m, c) in groups if m == model)
    return ctxs, [groups[(model, c)][key]["median"] for c in ctxs]


def _crossover(groups: dict[tuple[str, int], dict]) -> float | None:
    """Where the hybrid's fixed state stops costing more than the attention KV."""
    a_model, b_model = MODELS[0], MODELS[1]
    a_ctx, a_cache = _series(groups, a_model, KEY)
    b_ctx, b_cache = _series(groups, b_model, KEY)
    if len(a_ctx) < 2 or len(b_ctx) < 2:
        return None
    a_slope = (a_cache[-1] - a_cache[0]) / (a_ctx[-1] - a_ctx[0])
    b_slope = (b_cache[-1] - b_cache[0]) / (b_ctx[-1] - b_ctx[0])
    if a_slope <= b_slope:
        return None
    a_intercept = a_cache[0] - a_slope * a_ctx[0]
    b_intercept = b_cache[0] - b_slope * b_ctx[0]
    return (b_intercept - a_intercept) / (a_slope - b_slope)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/W0/0001-carrier-probe", help="run directory")
    parser.add_argument("--out", default="figures/fig_w0_crossover.png", help="output PNG")
    args = parser.parse_args()

    run_dir = Path(args.run)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    groups = {(g["model"], int(g["context"])): g for g in summary["groups"]}
    run_id = summary.get("run_id", run_dir.name)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for model in MODELS:
        ctx, cache = _series(groups, model, KEY)
        axes[0].plot(ctx, cache, "o-", color=COLORS[model], label=LABELS[model])
        ctx, prefill = _series(groups, model, "prefill_ms")
        axes[1].plot(ctx, prefill, "o-", color=COLORS[model], label=LABELS[model])
        ctx, tpot = _series(groups, model, "tpot_ms")
        axes[2].plot(ctx, tpot, "o-", color=COLORS[model], label=LABELS[model])

    axes[0].set_title("Per-sequence state at N tokens (prefill)")
    axes[0].set_ylabel("MiB")
    axes[0].set_yscale("log")
    axes[0].set_ylim(8, 6000)

    # Both carriers are linear in context, so the crossing sits below the smallest
    # measured bucket; draw the fitted segment instead of pretending it was measured.
    a_ctx, a_cache = _series(groups, MODELS[0], KEY)
    b_ctx, b_cache = _series(groups, MODELS[1], KEY)
    a_slope = (a_cache[-1] - a_cache[0]) / (a_ctx[-1] - a_ctx[0])
    b_slope = (b_cache[-1] - b_cache[0]) / (b_ctx[-1] - b_ctx[0])
    a_intercept = a_cache[0] - a_slope * a_ctx[0]
    b_intercept = b_cache[0] - b_slope * b_ctx[0]
    x_fit = [128, a_ctx[0]]
    axes[0].plot(x_fit, [a_intercept + a_slope * x for x in x_fit], ":", lw=1, color=COLORS[MODELS[0]])
    axes[0].plot(x_fit, [b_intercept + b_slope * x for x in x_fit], ":", lw=1, color=COLORS[MODELS[1]])

    cross = _crossover(groups)
    if cross is not None:
        axes[0].axvline(cross, ls="--", lw=1, color="0.4")
        axes[0].annotate(f"crossover ≈ {cross:.0f} tok\n(analytic; below 512 bucket)",
                         xy=(cross, 8), xytext=(150, 9), fontsize=8, color="0.3", va="bottom")

    for ctx in (8192, 32768):
        a = groups[(MODELS[0], ctx)][KEY]["median"]
        b = groups[(MODELS[1], ctx)][KEY]["median"]
        axes[0].annotate(f"{a / b:.1f}×", xy=(ctx, (a * b) ** 0.5), xytext=(-6, 0),
                         textcoords="offset points", fontsize=9, color="0.25", ha="right", va="center")

    axes[1].set_title("Prefill latency")
    axes[1].set_ylabel("ms")
    axes[1].set_yscale("log")

    axes[2].set_title("Decode latency (per token)")
    axes[2].set_ylabel("ms/token")
    axes[2].set_ylim(0, max(30, axes[2].get_ylim()[1]))

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([512, 2048, 8192, 32768])
        ax.set_xticklabels(["512", "2k", "8k", "32k"])
        ax.set_xlabel("context tokens")
        ax.grid(alpha=0.25, which="both")

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{run_id} · RTX 4090 · bf16 · batch=1 · greedy 128 tokens", fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (crossover ≈ {cross:.0f} tokens)" if cross else f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
