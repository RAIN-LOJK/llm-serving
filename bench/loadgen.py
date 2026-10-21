"""Workload generation: fixed prompt sets and synthetic token sequences.

The prompt set stores *text* rather than token ids because tokenizers differ across
carrier models. Each run records the actual tokenized length, and cross-model
comparisons use that length rather than the bucket label.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

BUCKETS: dict[str, tuple[int, int]] = {
    "short": (16, 64),
    "medium": (256, 512),
    "long": (1024, 2048),
}


@dataclass(frozen=True)
class Prompt:
    id: str
    bucket: str
    target_tokens: tuple[int, int]
    text: str


@dataclass
class PromptSet:
    version: str
    prompts: list[Prompt]
    provenance: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        payload = [
            {
                "id": p.id,
                "bucket": p.bucket,
                "target_tokens": list(p.target_tokens),
                "text": p.text,
            }
            for p in self.prompts
        ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "PromptSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        prompts = [
            Prompt(
                id=str(item["id"]),
                bucket=str(item["bucket"]),
                target_tokens=tuple(int(x) for x in item["target_tokens"]),  # type: ignore[arg-type]
                text=str(item["text"]),
            )
            for item in data["prompts"]
        ]
        return cls(
            version=str(data.get("version", "unknown")),
            prompts=prompts,
            provenance=str(data.get("provenance", "")),
            created_at=str(data.get("created_at", "")),
            extra={k: v for k, v in data.items() if k not in {"version", "prompts", "provenance", "created_at"}},
        )

    def by_bucket(self) -> dict[str, list[Prompt]]:
        grouped: dict[str, list[Prompt]] = {}
        for prompt in self.prompts:
            grouped.setdefault(prompt.bucket, []).append(prompt)
        return grouped


def synthetic_ids(length: int, vocab_size: int, seed: int = 0) -> list[int]:
    """Deterministic random token ids; content is irrelevant for memory/latency probes."""
    if length <= 0:
        return []
    rng = random.Random(seed)
    return [rng.randrange(max(1, vocab_size)) for _ in range(length)]


def closed_loop(
    prompt_set: PromptSet,
    repeats: int = 1,
    limit: int | None = None,
) -> Iterator[tuple[int, Prompt]]:
    """Yield ``(repeat_index, prompt)`` in a fixed order."""
    count = 0
    for repeat in range(repeats):
        for prompt in prompt_set.prompts:
            if limit is not None and count >= limit:
                return
            yield repeat, prompt
            count += 1


def poisson_arrivals(*args: object, **kwargs: object) -> Iterator[tuple[float, Prompt]]:
    """Reserved for W3. Open-loop arrival is intentionally not faked yet."""
    raise NotImplementedError("poisson arrival is a W3 feature; use arrival='closed' for now")
