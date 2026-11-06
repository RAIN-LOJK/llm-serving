#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$RUN_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT=/root/workstation/minillm
fi
cd "$REPO_ROOT"
EXPECTED=b8fa5c1972327db5431ca62b231b8285811855b2
ACTUAL="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "$EXPECTED" != "unknown" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "warning: commit mismatch ($ACTUAL != $EXPECTED)" >&2
fi
exec python -m bench.run \
  --config "$RUN_DIR/config.toml" \
  --out "${OUT_ROOT:-results-repro}" \
  --allow-dirty
