"""The fixed prompt set must be 32 prompts and match its own frozen hash."""

from __future__ import annotations

import json
from pathlib import Path

from bench.loadgen import PromptSet
from bench.make_prompts import build

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "bench" / "prompts.json"


def test_prompt_set_is_frozen_and_bucketed():
    data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompt_set = PromptSet.load(PROMPTS)

    assert len(prompt_set.prompts) == 32
    buckets = prompt_set.by_bucket()
    assert len(buckets["short"]) == 12
    assert len(buckets["medium"]) == 12
    assert len(buckets["long"]) == 8
    assert prompt_set.version == data["version"]
    assert prompt_set.sha256 == data["sha256"]


def test_generator_matches_committed_file():
    payload = build()
    assert payload["sha256"] == PromptSet.load(PROMPTS).sha256
    assert len(payload["prompts"]) == 32
