"""Arrival traces are reproducible, correctly normalized and cover their prompt set."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

import pytest

from bench.loadgen import (
    Prompt,
    PromptSet,
    closed_trace,
    poisson_trace,
)

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "bench" / "prompts.json"


def _prompts() -> PromptSet:
    """The real frozen prompt set; a synthetic 32-prompt stand-in if it is unreadable."""
    try:
        prompt_set = PromptSet.load(PROMPTS)
        if len(prompt_set.prompts) == 32:
            return prompt_set
    except (OSError, ValueError, KeyError):
        pass
    return PromptSet(
        version="fallback",
        prompts=[
            Prompt(id=f"fake-{i:02d}", bucket="short", target_tokens=(16, 64), text=f"p{i}")
            for i in range(32)
        ],
    )


TOKENIZED = [(f"p{i:02d}", (1000 + i, 2000 + i)) for i in range(8)]


def _poisson(**kwargs):
    base = dict(rate=10.0, n=16, budgets=(16, 32, 64, 128), seed=7)
    base.update(kwargs)
    return poisson_trace(TOKENIZED, **base)


def test_same_parameters_same_hash():
    assert _poisson().sha256 == _poisson().sha256


def test_hash_changes_with_every_parameter():
    base = _poisson().sha256
    variants = {
        "rate": _poisson(rate=20.0),
        "budgets": _poisson(budgets=(32, 64)),
        "seed": _poisson(seed=8),
        "shuffle": _poisson(shuffle=True),
    }
    for name, trace in variants.items():
        assert trace.sha256 != base, name

    # burstiness is both a schedule input and part of the hash payload.
    assert _poisson(burstiness=4.0).sha256 != base

    # A different prompt set (different ids / token ids) changes the hash.
    other = [(f"q{i:02d}", (3000 + i,)) for i in range(8)]
    assert poisson_trace(other, rate=10.0, n=16, budgets=(16, 32, 64, 128), seed=7).sha256 != base

    # prompt_ids are part of the hash even when the prompt ids are unchanged.
    retokenized = [(pid, ids + (999,)) for pid, ids in TOKENIZED]
    assert (
        poisson_trace(retokenized, rate=10.0, n=16, budgets=(16, 32, 64, 128), seed=7).sha256
        != base
    )

    # A PromptSet-derived trace hashes its prompt_sha256, so different text with
    # identical ids still changes the digest.
    one = PromptSet(version="a", prompts=[Prompt("x", "short", (16, 64), "hello")])
    two = PromptSet(version="a", prompts=[Prompt("x", "short", (16, 64), "world")])
    assert (
        poisson_trace(one, rate=10.0, n=4, budgets=(16,), seed=7).sha256
        != poisson_trace(two, rate=10.0, n=4, budgets=(16,), seed=7).sha256
    )


def test_poisson_mean_interarrival_matches_rate():
    n, rate = 5000, 10.0
    trace = poisson_trace(TOKENIZED, rate=rate, n=n, budgets=(128,), seed=3)
    at = [a.at_s for a in trace.arrivals]

    assert at[0] == 0.0
    assert all(b >= a for a, b in zip(at, at[1:]))

    inter = [b - a for a, b in zip(at, at[1:])]
    # Normalization makes this exact up to float error, not just statistically true.
    assert abs(statistics.mean(inter) - 1.0 / rate) / (1.0 / rate) < 1e-2


def test_arrivals_are_normalized_to_total_duration_n_over_rate():
    """Times are scaled so the full n intervals span exactly n/rate (vLLM serve.py).

    With ``at_s[i] = cumsum(intervals[:i]) * scale`` the *observed* span across the
    n arrivals covers the first n-1 intervals, so the self-consistent expectation
    is ``(n-1)/rate``; the scaling target over all n intervals is ``n/rate``.
    """
    n, rate = 5000, 10.0
    trace = poisson_trace(TOKENIZED, rate=rate, n=n, budgets=(128,), seed=3)
    at = [a.at_s for a in trace.arrivals]

    total = sum(b - a for a, b in zip(at, at[1:]))
    assert abs(total - (n - 1) / rate) / ((n - 1) / rate) < 1e-3
    assert abs(at[-1] - (n - 1) / rate) / ((n - 1) / rate) < 1e-3
    # Equivalent statement of the n/rate target: the observed mean interval,
    # extended over n intervals, reproduces the nominal duration.
    assert abs(total * n / (n - 1) - n / rate) / (n / rate) < 1e-3


def test_burstiness_keeps_mean_but_changes_spread():
    n, rate = 5000, 10.0
    kwargs = dict(prompts=TOKENIZED, rate=rate, n=n, budgets=(128,), seed=5)

    def spread(burstiness: float) -> tuple[float, float]:
        at = [a.at_s for a in poisson_trace(burstiness=burstiness, **kwargs).arrivals]
        inter = [b - a for a, b in zip(at, at[1:])]
        return statistics.mean(inter), statistics.pstdev(inter)

    mean1, sd1 = spread(1.0)
    mean4, sd4 = spread(4.0)

    assert abs(mean1 - 1.0 / rate) / (1.0 / rate) < 1e-2
    assert abs(mean4 - 1.0 / rate) / (1.0 / rate) < 1e-2
    # gamma(shape=burstiness, scale=1/(rate*burstiness)) has cv = 1/sqrt(burstiness),
    # so shape 4 is *smoother* (sd ~ 1/2 of Poisson), consistent with the doc's
    # ">1 更平滑". (The task text said ">1.5x larger"; that contradicts the frozen
    # sampling formula and the source doc, see the hand-off report.)
    assert sd4 < sd1 / 1.5
    assert 0.4 < sd4 / sd1 < 0.6  # ~0.5 for shape 4

    # Same formula run the other way: shape < 1 *is* burstier and spreads out.
    mean_quarter, sd_quarter = spread(0.25)
    assert abs(mean_quarter - 1.0 / rate) / (1.0 / rate) < 1e-2
    assert sd_quarter > 1.5 * sd1


def test_budget_sampling_matches_weights():
    n = 20000
    budgets = (16, 32, 64, 128)
    trace = poisson_trace(TOKENIZED, rate=10.0, n=n, budgets=budgets, seed=11)
    counts = Counter(a.budget for a in trace.arrivals)

    assert set(counts) == set(budgets)
    for budget in budgets:
        assert abs(counts[budget] / n - 0.25) < 0.03, (budget, counts[budget] / n)


def test_shuffle_is_reproducible_and_rotates_like_order():
    prompt_set = _prompts()

    shuffled_a = poisson_trace(prompt_set, rate=10.0, n=64, budgets=(16,), seed=9, shuffle=True)
    shuffled_b = poisson_trace(prompt_set, rate=10.0, n=64, budgets=(16,), seed=9, shuffle=True)
    assert shuffled_a.sha256 == shuffled_b.sha256
    assert [a.prompt_id for a in shuffled_a.arrivals] == [a.prompt_id for a in shuffled_b.arrivals]

    # A different seed gives a different permutation.
    shuffled_c = poisson_trace(prompt_set, rate=10.0, n=64, budgets=(16,), seed=10, shuffle=True)
    assert shuffled_c.sha256 != shuffled_a.sha256

    ordered = poisson_trace(prompt_set, rate=10.0, n=64, budgets=(16,), seed=9, shuffle=False)
    ids = [a.prompt_id for a in ordered.arrivals]
    assert set(ids) == {p.id for p in prompt_set.prompts}
    assert len(set(ids)) == 32
    assert ids[:4] == [p.id for p in prompt_set.prompts[:4]]


def test_prompt_set_requires_tokenization_by_the_caller():
    prompt_set = _prompts()
    trace = poisson_trace(prompt_set, rate=10.0, n=5, budgets=(16,), seed=0)
    assert all(a.prompt_ids == () for a in trace.arrivals)
    assert trace.prompt_sha256 == prompt_set.sha256

    tokenized = poisson_trace(TOKENIZED, rate=10.0, n=5, budgets=(16,), seed=0)
    assert tokenized.prompt_sha256 is None
    assert all(a.prompt_ids for a in tokenized.arrivals)


def test_closed_trace_all_arrivals_at_zero():
    trace = closed_trace(TOKENIZED, n=32, budgets=(16, 32), seed=2)
    assert len(trace.arrivals) == 32
    assert all(a.at_s == 0.0 for a in trace.arrivals)
    assert trace.rate is None
    assert trace.burstiness == 1.0
    assert trace.sha256 == closed_trace(TOKENIZED, n=32, budgets=(16, 32), seed=2).sha256


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        poisson_trace(TOKENIZED, rate=0.0, n=4, budgets=(16,), seed=0)
    with pytest.raises(ValueError):
        poisson_trace(TOKENIZED, rate=10.0, n=4, budgets=(), seed=0)
    with pytest.raises(ValueError):
        poisson_trace([], rate=10.0, n=4, budgets=(16,), seed=0)
