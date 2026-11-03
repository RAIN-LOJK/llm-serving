"""Workload generation: fixed prompt sets, synthetic token sequences and arrival traces.

The prompt set stores *text* rather than token ids because tokenizers differ across
carrier models. Each run records the actual tokenized length, and cross-model
comparisons use that length rather than the bucket label.

Arrival traces (W3) are the open-loop workload contract: a :class:`Trace` is fully
determined by ``(seed, rate, n, budgets, prompt set, burstiness, shuffle)`` and its
``sha256`` is the first check that two paired runs saw the same workload. This module
never tokenizes: a trace built from a :class:`PromptSet` carries ``prompt_ids=()`` and
the runner fills the ids in afterwards (see :class:`Trace`).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

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


@dataclass(frozen=True)
class Arrival:
    """One request in a trace.

    ``prompt_ids`` is what the request actually runs on. Both workload kinds
    (synthetic and prompts) go through this field so the scheduler never has to
    know where the tokens came from.
    """

    idx: int
    at_s: float
    prompt_id: str
    prompt_ids: tuple[int, ...]
    budget: int


@dataclass(frozen=True)
class Trace:
    """A fully specified open-loop workload.

    The trace is determined by ``(seed, rate, n, budgets, prompt set, burstiness,
    shuffle)``; :attr:`sha256` is the first check that two paired runs (e.g. static
    vs continuous at the same ``max_batch``) saw the same arrivals.

    **This module never tokenizes.** When ``prompts`` is a :class:`PromptSet` the
    token ids are unknown here, so every ``prompt_ids`` is ``()`` and only the
    ``prompt_id`` is filled in; the caller (the runner) is then responsible for
    tokenizing and filling them. In other words, ``needs_tokenization`` means
    "built from a ``PromptSet``" — equivalently ``prompt_sha256 is not None`` —
    and such a trace's :attr:`sha256` is a pre-tokenization hash. The main path
    (E1 synthetic, E2 prompts) passes an already tokenized
    ``Sequence[(prompt_id, prompt_ids)]``, which yields a final hash directly.

    ``rate`` is ``None`` for closed traces (all arrivals at ``t=0``).
    """

    arrivals: list[Arrival]
    rate: float | None
    budgets: tuple[int, ...]
    seed: int
    burstiness: float
    prompt_sha256: str | None
    shuffle: bool

    @property
    def sha256(self) -> str:
        """Stable hash of the schedule.

        Covers every ``arrival`` as ``(idx, at_s, prompt_id, prompt_ids, budget)``
        plus ``rate``, ``budgets``, ``seed``, ``burstiness``, ``prompt_sha256`` and
        ``shuffle``. Floats are formatted to 6 decimals before hashing so the
        digest does not drift with platform float formatting; ``at_s`` below
        1e-6 is therefore invisible to the hash by design.
        """
        payload = {
            "arrivals": [
                {
                    "idx": a.idx,
                    "at_s": f"{a.at_s:.6f}",
                    "prompt_id": a.prompt_id,
                    "prompt_ids": list(a.prompt_ids),
                    "budget": a.budget,
                }
                for a in self.arrivals
            ],
            "rate": None if self.rate is None else f"{self.rate:.6f}",
            "budgets": list(self.budgets),
            "seed": self.seed,
            "burstiness": f"{self.burstiness:.6f}",
            "prompt_sha256": self.prompt_sha256,
            "shuffle": self.shuffle,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prompt_entries(
    prompts: PromptSet | Sequence[tuple[str, Sequence[int]]],
) -> tuple[list[tuple[str, tuple[int, ...]]], str | None]:
    """Normalize the two accepted prompt inputs to ``[(prompt_id, prompt_ids)]``."""
    if isinstance(prompts, PromptSet):
        entries = [(p.id, ()) for p in prompts.prompts]
        prompt_sha256: str | None = prompts.sha256
    else:
        entries = [(str(pid), tuple(int(t) for t in ids)) for pid, ids in prompts]
        prompt_sha256 = None
    if not entries:
        raise ValueError("a trace needs at least one prompt")
    return entries, prompt_sha256


def _arrival_times(n: int, rate: float, seed: int, burstiness: float) -> list[float]:
    """Gamma-spaced arrival times normalized to a total duration of ``n / rate``.

    Intervals are ``gamma(shape=burstiness, scale=1/(rate*burstiness))`` (vLLM
    ``benchmarks/serve.py::get_request``); ``burstiness=1`` is Poisson. Plain
    cumulative summation is 1-2% off the nominal duration (vLLM's own comment),
    which makes different seeds incomparable, so every time is scaled by
    ``(n/rate) / sum(intervals)``. The first arrival is at ``t=0``:
    ``at_s[i] = scale * cumsum(intervals[:i])``.
    """
    rng = random.Random(seed)
    intervals = [rng.gammavariate(burstiness, 1.0 / (rate * burstiness)) for _ in range(n)]
    scale = (n / rate) / sum(intervals) if n else 0.0
    times: list[float] = []
    cumulative = 0.0
    for i in range(n):
        times.append(cumulative * scale)
        cumulative += intervals[i]
    return times


def _build(
    prompts: PromptSet | Sequence[tuple[str, Sequence[int]]],
    *,
    times: list[float],
    rate: float | None,
    budgets: Sequence[int],
    seed: int,
    burstiness: float,
    shuffle: bool,
) -> Trace:
    entries, prompt_sha256 = _prompt_entries(prompts)
    budget_list = tuple(int(b) for b in budgets)
    if not budget_list:
        raise ValueError("budgets must not be empty")

    order = list(entries)
    if shuffle:
        random.Random(seed).shuffle(order)

    rng = random.Random(seed)
    arrivals = [
        Arrival(
            idx=i,
            at_s=times[i],
            prompt_id=order[i % len(order)][0],
            prompt_ids=order[i % len(order)][1],
            budget=rng.choice(budget_list),
        )
        for i in range(len(times))
    ]
    return Trace(
        arrivals=arrivals,
        rate=rate,
        budgets=budget_list,
        seed=seed,
        burstiness=burstiness,
        prompt_sha256=prompt_sha256,
        shuffle=shuffle,
    )


def poisson_trace(
    prompts: PromptSet | Sequence[tuple[str, Sequence[int]]],
    *,
    rate: float,
    n: int,
    budgets: Sequence[int],
    seed: int,
    shuffle: bool = False,
    burstiness: float = 1.0,
) -> Trace:
    """Open-loop trace: ``n`` requests over ``n/rate`` seconds.

    ``burstiness`` is the gamma shape parameter (1.0 = Poisson, >1 = smoother than
    Poisson, <1 = burstier). ``budgets`` are sampled uniformly with replacement
    per request using ``seed``. Requests rotate through ``prompts`` in order;
    ``shuffle=True`` permutes that order once, reproducibly, from ``seed``.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if burstiness <= 0:
        raise ValueError("burstiness must be > 0")
    times = _arrival_times(n, rate, seed, burstiness)
    return _build(
        prompts,
        times=times,
        rate=float(rate),
        budgets=budgets,
        seed=seed,
        burstiness=float(burstiness),
        shuffle=shuffle,
    )


def closed_trace(
    prompts: PromptSet | Sequence[tuple[str, Sequence[int]]],
    *,
    n: int,
    budgets: Sequence[int],
    seed: int,
    shuffle: bool = False,
) -> Trace:
    """Closed-loop trace: all ``n`` requests arrive at ``t=0`` (``rate=None``)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return _build(
        prompts,
        times=[0.0] * n,
        rate=None,
        budgets=budgets,
        seed=seed,
        burstiness=1.0,
        shuffle=shuffle,
    )
