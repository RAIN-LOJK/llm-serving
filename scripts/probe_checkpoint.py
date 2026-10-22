#!/usr/bin/env python3
"""W0 step 1: read the two carrier configs and write the analytic state account.

No weights, no GPU: this only needs ``config.json`` at the pinned revision. It answers
"how many bytes per token / per sequence does each carrier keep" before we spend a run on
the real allocator.

Usage::

    python scripts/probe_checkpoint.py                  # print
    python scripts/probe_checkpoint.py --write          # docs/carrier_estimates.json
    python scripts/probe_checkpoint.py --offline        # cache only, no network
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from bench import carrier  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "W0" / "0001-carrier-probe.toml"
DEFAULT_OUT = ROOT / "docs" / "carrier_estimates.json"


def load_models(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open("rb") as handle:
        cfg = tomllib.load(handle)
    models = cfg.get("models") or []
    if not models:
        raise SystemExit(f"no [[models]] in {config_path}")
    return models


def fetch_config(name: str, revision: str | None, offline: bool) -> dict[str, Any]:
    """Prefer the local HF cache; only hit the network when explicitly allowed."""
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {"repo_id": name, "filename": "config.json"}
    if revision:
        kwargs["revision"] = revision
    try:
        path = hf_hub_download(local_files_only=offline, **kwargs)
    except Exception as exc:
        if offline:
            raise SystemExit(
                f"{name}@{revision} not in the local HF cache ({type(exc).__name__}: {exc}).\n"
                f"Download it once with:\n"
                f"  hf download {name} --revision {revision} config.json"
            ) from exc
        raise
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment TOML with [[models]]")
    parser.add_argument("--write", action="store_true", help="write the JSON account next to docs/")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON path for --write")
    parser.add_argument("--offline", action="store_true", help="use the HF cache only")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    models = load_models(config_path)
    raw: dict[str, Any] = {}
    estimates: dict[str, Any] = {}
    for model in models:
        name = str(model["name"])
        revision = model.get("revision")
        dtype = str(model.get("dtype", "bfloat16"))
        raw[name] = fetch_config(name, revision, args.offline)
        estimates[name] = carrier.estimate(raw[name], dtype=dtype)
        estimates[name]["revision"] = revision

    print(f"config: {config_path}")
    print()
    print(carrier.markdown_table(estimates))
    print()

    names = list(estimates)
    if len(names) >= 2:
        candidate, reference = names[1], names[0]
        cmp = carrier.compare(estimates, candidate, reference)
        print(f"# {candidate} vs {reference}")
        cross = cmp["crossover_tokens"]
        print(f"crossover: {cross:.0f} tokens" if cross is not None else "crossover: none")
        print(f"asymptotic KV ratio: {cmp['asymptotic_ratio']:.2f}x")
        print()
        print("| context | candidate | reference | reference/candidate |")
        print("|---|---|---|---|")
        for ctx, row in cmp["by_context"].items():
            ratio = f"{row['ratio']:.2f}x" if row["ratio"] else "—"
            print(
                f"| {ctx} | {carrier.human_bytes(row['candidate'])} | "
                f"{carrier.human_bytes(row['reference'])} | {ratio} |"
            )
        print()
        print("预测对照（ROADMAP §3.2，待 W0-0001 实测判定）：")
        for ctx in ("512", "2048", "8192", "32768"):
            ratio = cmp["by_context"][ctx]["ratio"]
            verdict = "candidate 更省" if ratio and ratio > 1 else "candidate 更贵"
            print(f"  - ctx={ctx}: {verdict}（{ratio:.2f}x）" if ratio else f"  - ctx={ctx}: {verdict}")

    if args.write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "configs": {name: carrier.text_config(raw[name]) for name in raw},
            "estimates": estimates,
            "comparison": carrier.compare(estimates, names[1], names[0]) if len(names) >= 2 else None,
            "note": "analytic account from config.json only; not a measurement",
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
