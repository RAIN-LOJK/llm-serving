"""W7 cross-carrier synthetic traces must hash identically.

``bench/run.py::_build_trace`` draws synthetic ids with the model's vocab size. The two
W7 carriers have different vocabularies (0.6B 151936, 0.8B 248320), so before the W7 cap
the same synthetic workload produced different ids -- and therefore a different
``trace_sha256`` -- on the pure and hybrid arms. W7 G4 requires the paired hash to match;
the cap uses the smaller vocabulary, under which both embeddings are valid. Prompt
workloads are still tokenized per model and are deliberately not covered here.
"""

from __future__ import annotations

from bench.engines.dummy import DummyEngine
from bench.run import SYNTHETIC_VOCAB_CAP, _build_trace


def _cfg(seed: int = 0, contexts=(64, 512, 1024)) -> dict:
    return {
        "exp": {"seed": seed},
        "workload": {
            "kind": "synthetic",
            "context_lengths": list(contexts),
            "arrival": "closed",
            "shuffle": False,
        },
    }


def _trace(vocab_size: int, *, seed: int = 0, requests: int = 6):
    engine = DummyEngine(vocab_size=vocab_size, seed=0)
    return _build_trace(
        _cfg(seed=seed),
        engine,
        None,
        requests=requests,
        budgets=(4,),
        max_new=4,
    )


def test_cross_carrier_synthetic_traces_share_ids_and_sha256():
    small = _trace(151936)
    large = _trace(248320)

    assert [len(a.prompt_ids) for a in small.arrivals] == [
        len(a.prompt_ids) for a in large.arrivals
    ]
    assert [a.prompt_ids for a in small.arrivals] == [a.prompt_ids for a in large.arrivals]
    assert small.sha256 == large.sha256

    # The shared ids are valid on both carriers: below the smaller vocabulary.
    assert all(
        token < SYNTHETIC_VOCAB_CAP
        for arrival in small.arrivals
        for token in arrival.prompt_ids
    )


def test_cap_only_changes_the_vocabulary_domain_not_the_schedule():
    """A smaller engine's own vocab is still respected; a larger one is capped."""
    small = _trace(1024)
    assert all(
        token < 1024 for arrival in small.arrivals for token in arrival.prompt_ids
    )
    # A 1024-vocab trace differs from the capped 151936 trace (different ids), which is
    # expected: the cap is a cross-carrier bound, not a global fixed seed space.
    assert small.sha256 != _trace(SYNTHETIC_VOCAB_CAP).sha256


def test_seed_still_changes_the_synthetic_trace():
    assert _trace(248320, seed=0).sha256 != _trace(248320, seed=1).sha256
