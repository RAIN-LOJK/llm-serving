#!/usr/bin/env python3
"""Build the fixed 32-prompt serving workload.

Design:
- store *text*, not token ids: carrier models use different tokenizers;
- 32 prompts across the ROADMAP §6 length buckets (short / medium / long);
- a version string and a hash over the canonical prompt payload freeze the set;
- ``--check`` fails if the committed JSON no longer matches this generator.

This is a v0 set seeded from the legacy eval prompts. Replacing it is allowed only with
a version bump, because changing prompts invalidates cross-run comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).with_name("prompt_sources") / "seed.txt"
DEFAULT_OUT = Path(__file__).with_name("prompts.json")

VERSION = "v0-2026-10"
CREATED_AT = "2026-10-21"
PROVENANCE = (
    "Seed text recovered from the legacy eval prompts (tag legacy-learning-2026-08); "
    "32 prompts across ROADMAP §6 length buckets. Token lengths are measured per model "
    "at run time. Replace only with a version bump."
)

# bucket, target token range, count, target character range (Chinese ~1 token / 1.3 chars)
BUCKETS = [
    ("short", (16, 64), 12, (28, 72)),
    ("medium", (256, 512), 12, (380, 700)),
    ("long", (1024, 2048), 8, (1600, 2800)),
]


def _paragraphs() -> list[str]:
    text = SOURCE.read_text(encoding="utf-8").lstrip("\ufeff")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _make_text(paragraphs: list[str], index: int, target_chars: int) -> str:
    rng = random.Random(20261021 + index)
    start = index % len(paragraphs)
    rotated = paragraphs[start:] + paragraphs[:start]
    base = "\n\n".join(rotated)
    while len(base) < target_chars + 8:
        base = base + "\n\n" + "\n\n".join(rotated)
    offset = rng.randrange(0, max(1, len(base) - target_chars)) if len(base) > target_chars else 0
    text = base[offset : offset + target_chars].strip()
    return text if len(text) >= 20 else text + " " + base[:20]


def build() -> dict:
    paragraphs = _paragraphs()
    if len(paragraphs) < 3:
        raise SystemExit(f"seed file needs at least 3 paragraphs, got {len(paragraphs)}")

    prompts = []
    index = 0
    for bucket, token_range, count, char_range in BUCKETS:
        for i in range(count):
            rng = random.Random(20261021 + index * 17 + i)
            target_chars = rng.randint(*char_range)
            text = _make_text(paragraphs, index, target_chars)
            prompts.append(
                {
                    "id": f"{bucket[0]}{i + 1:02d}",
                    "bucket": bucket,
                    "target_tokens": list(token_range),
                    "text": text,
                }
            )
            index += 1

    payload = {
        "version": VERSION,
        "created_at": CREATED_AT,
        "provenance": PROVENANCE,
        "buckets": {
            bucket: {"target_tokens": list(token_range), "n": count}
            for bucket, token_range, count, _ in BUCKETS
        },
        "prompts": prompts,
    }
    raw = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return payload


def _canonical(payload: dict) -> str:
    prompts = payload["prompts"]
    raw = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true", help="verify the committed file matches")
    args = parser.parse_args()

    payload = build()
    path = Path(args.out)
    if args.check:
        if not path.exists():
            print(f"✗ missing {path}", file=sys.stderr)
            return 1
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("version") != payload["version"] or _canonical(existing) != _canonical(payload):
            print(f"✗ {path} is stale; rerun `python bench/make_prompts.py`", file=sys.stderr)
            return 1
        print(f"✓ {path}: {len(payload['prompts'])} prompts, sha256={payload['sha256'][:12]}…")
        return 0

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"✓ wrote {path}: {len(payload['prompts'])} prompts, sha256={payload['sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
