"""Check a W4 E0 run's greedy output against ``tests/golden/`` (C3's hard gate).

    python scripts/check_paged_golden.py results/W4/0001-paged-numeric-gate

The runner records token ids but never compares them to the golden fixture: E0 is
``record_only`` on purpose, because the fixture holds four prompts and a run picks one.
This script closes that gap so the gate is a command, not a manual diff of 64 ids.

Exit codes: 0 = every compared prompt matches, 1 = mismatch or missing data,
2 = the run/file cannot be read.

The comparison is **exact ids**, not a tolerance: both arms run the same greedy decode
over the same tokenizer, so a single different id means the paged path (block table,
mask width, gather columns or logits scatter) drifted. Do not loosen this to a metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "tests" / "golden" / "qwen3-0-6b" / "0001"


def load_golden(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(meta, ids_by_prompt)``; ``ids_by_prompt`` maps prompt id -> greedy ids."""
    meta_path = path / "meta.json"
    tensors_path = path / "tensors.npz"
    if not meta_path.exists() or not tensors_path.exists():
        raise FileNotFoundError(f"{path} does not hold meta.json + tensors.npz")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(tensors_path) as arrays:
        greedy = np.asarray(arrays["greedy_ids"])
    prompts = meta.get("prompts") or []
    if len(prompts) != len(greedy):
        raise ValueError(
            f"golden has {len(prompts)} prompt entries but {len(greedy)} greedy rows"
        )
    return meta, {str(p["id"]): greedy[i] for i, p in enumerate(prompts)}


def load_run_rows(path: Path) -> list[dict]:
    """Request rows of a run directory's ``metrics.jsonl`` (first repeat only)."""
    metrics = path if path.is_file() else path / "metrics.jsonl"
    if not metrics.exists():
        raise FileNotFoundError(f"no metrics.jsonl under {path}")
    rows: list[dict] = []
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind", "request") != "request":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"{metrics} has no request rows")
    repeats = [row.get("repeat") for row in rows if row.get("repeat") is not None]
    if repeats:
        first = min(repeats)
        rows = [row for row in rows if row.get("repeat") in (None, first)]
    return rows


def compare(
    run_rows: list[dict], golden: dict[str, np.ndarray]
) -> list[dict]:
    """One result per request row that names a golden prompt."""
    results: list[dict] = []
    for row in run_rows:
        prompt_id = row.get("prompt_id")
        if prompt_id not in golden:
            results.append(
                {"prompt_id": prompt_id, "status": "skipped", "reason": "not in golden"}
            )
            continue
        expected = [int(x) for x in golden[prompt_id]]
        actual = [int(x) for x in (row.get("token_ids") or [])]
        if not actual:
            results.append(
                {"prompt_id": prompt_id, "status": "missing", "reason": "no token_ids"}
            )
            continue
        # A run may stop early (budget < golden length); compare the common prefix and say so.
        width = min(len(expected), len(actual))
        mismatch = next(
            (i for i in range(width) if expected[i] != actual[i]), None
        )
        entry = {
            "prompt_id": prompt_id,
            "status": "pass" if mismatch is None else "fail",
            "compared": width,
            "expected_n": len(expected),
            "actual_n": len(actual),
        }
        if mismatch is not None:
            entry["first_mismatch"] = mismatch
            entry["expected_id"] = expected[mismatch]
            entry["actual_id"] = actual[mismatch]
        results.append(entry)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory (or its metrics.jsonl)")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        meta, golden = load_golden(args.golden)
        rows = load_run_rows(args.run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = compare(rows, golden)
    compared = [r for r in results if r["status"] in {"pass", "fail"}]
    if args.json:
        print(json.dumps({"golden": str(args.golden), "results": results}, indent=2))
    else:
        print(f"golden: {args.golden} ({meta.get('spec', '?')})")
        for entry in results:
            detail = entry.get("reason") or ""
            if entry["status"] in {"pass", "fail"}:
                detail = f"{entry['compared']}/{entry['expected_n']} ids"
                if entry["status"] == "fail":
                    detail += (
                        f", first mismatch at {entry['first_mismatch']}: "
                        f"expected {entry['expected_id']}, got {entry['actual_id']}"
                    )
            print(f"  {entry['prompt_id']:>6}  {entry['status']:<8} {detail}")

    if not compared:
        print("error: no request row matched a golden prompt", file=sys.stderr)
        return 2
    failed = [r for r in compared if r["status"] == "fail"]
    print(
        f"\n{len(compared) - len(failed)}/{len(compared)} prompt(s) match the golden greedy ids"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
