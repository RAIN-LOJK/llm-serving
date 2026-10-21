#!/usr/bin/env python3
"""Regenerate or validate the results index.

Examples::

    .venv/bin/python scripts/exp_index.py
    .venv/bin/python scripts/exp_index.py --check
    .venv/bin/python scripts/exp_index.py --root /tmp/minillm-smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.index import build_index, check_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results", help="results root (default: results)")
    parser.add_argument("--check", action="store_true", help="validate run contract; do not write")
    args = parser.parse_args()

    if args.check:
        errors = check_root(args.root)
        if errors:
            print(f"✗ {len(errors)} contract error(s):")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("✓ run contract OK")
        return 0

    path = build_index(args.root)
    print(f"✓ index written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
