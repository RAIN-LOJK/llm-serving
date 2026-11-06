"""The ``minillm`` engine impl: factory wiring and fail-loudly behaviour."""

from __future__ import annotations

import pytest

from bench.engines import build_engine
from bench.engines.minillm import ServeEngine


def test_factory_wires_minillm():
    engine = build_engine({"name": "Qwen/Qwen3-0.6B"}, {"impl": "minillm", "device": "cpu"})
    assert isinstance(engine, ServeEngine)
    assert engine.impl == "minillm"
    assert engine.device == "cpu"


def test_factory_keeps_unknown_impls_loud():
    # `continuous` is wired (W3) and `paged` is wired (W4); the arm that is still unwired
    # must keep failing with the stage that owns it rather than silently degrading.
    with pytest.raises(NotImplementedError, match="W7"):
        build_engine({"name": "x"}, {"impl": "hybrid"})
    with pytest.raises(ValueError, match="unknown engine impl"):
        build_engine({"name": "x"}, {"impl": "bogus"})


def test_factory_wires_continuous():
    """The scenario arm exists and reports its own impl; it refuses per-request generate."""
    from bench.engines.continuous import ContinuousEngine

    engine = build_engine({"name": "dummy"}, {"impl": "continuous", "max_batch": 4})
    assert isinstance(engine, ContinuousEngine)
    assert engine.impl == "continuous"
    assert engine.max_batch == 4
    with pytest.raises(NotImplementedError, match="run_trace"):
        engine.generate([1, 2, 3], 4)


def test_missing_checkpoint_fails_loudly():
    engine = ServeEngine("definitely/not-a-cached-model", device="cpu")
    with pytest.raises(FileNotFoundError, match="not in the HF cache"):
        engine.load()


def test_encode_requires_load():
    engine = ServeEngine("Qwen/Qwen3-0.6B", device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        engine.encode("hello")


def test_unknown_dtype_is_rejected():
    engine = ServeEngine("Qwen/Qwen3-0.6B", dtype="float8", device="cpu")
    with pytest.raises(ValueError, match="unsupported dtype"):
        engine.load()
