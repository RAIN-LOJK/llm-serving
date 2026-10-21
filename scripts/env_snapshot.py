#!/usr/bin/env python3
"""Print or write the environment snapshot used by every run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.env import snapshot, write_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write JSON to this path instead of stdout")
    args = parser.parse_args()

    if args.out:
        path = write_snapshot(args.out, ROOT)
        print(path)
    else:
        print(json.dumps(snapshot(ROOT), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
