"""W7 Step 2: the ``hybrid`` runner arm -- config contract, factory, manifest.

These tests never load the 0.8B checkpoint: the adapter, the validator and the manifest
are all readable before ``load()``. The real-model path is covered by the Step-2b
integration smoke, and the engine itself by ``tests/unit/test_hybrid_engine.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.engines import build_engine
from bench.engines.hybrid import HybridEngine
from bench.run import (
    ConfigError,
    _execute,
    _scheduler_manifest,
    _summary,
    load_config,
    run_config,
    validate_config,
)

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

CONFIG = f"""
schema = 1

[exp]
id = "W7-9001"
stage = "W7"
type = "scenario"
slug = "hybrid-contract"
title = "hybrid contract smoke"
hypothesis = []
seed = 0

[[models]]
name = "Qwen/Qwen3.5-0.8B"
revision = "{REVISION}"
dtype = "bfloat16"

[workload]
kind = "prompts"
prompts = "bench/prompts.json"
max_new_tokens = 8
arrival = "closed"
requests = 4
budgets = [4, 8]
batch_sizes = [1]

[scheduler]
policy = "continuous"

[engine]
impl = "hybrid"
device = "cpu"
max_batch = 2
linear_slots = 2
kv_blocks = 64
block_size = 16
pool = "paged"
vocab_size = 1024
seed = 0

[measure]
warmup = 0
repeats = 1
timer = "perf_counter"
memory = ["torch_peak"]

[accept]
mode = "record_only"
criteria = []
"""


def _config(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    text = CONFIG
    for old, new in (replacements or {}).items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "hybrid.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _engine(tmp_path: Path | None = None, replacements: dict[str, str] | None = None):
    """Build an unloaded adapter from the contract config (or a variant of it)."""
    engine_cfg: dict[str, object] = {
        "impl": "hybrid",
        "device": "cpu",
        "max_batch": 2,
        "policy": "continuous",
        "linear_slots": 2,
        "kv_blocks": 64,
        "block_size": 16,
        "pool": "paged",
        "vocab_size": 1024,
        "seed": 0,
        "_policy": "continuous",
    }
    if replacements:
        engine_cfg.update(replacements)
    return build_engine(
        {"name": "Qwen/Qwen3.5-0.8B", "revision": REVISION, "dtype": "bfloat16"},
        engine_cfg,
    )


# --------------------------------------------------------------------------- contract
def test_minimal_hybrid_config_is_accepted(tmp_path: Path):
    assert validate_config(load_config(_config(tmp_path))) == []


@pytest.mark.parametrize(
    "replacements",
    [
        {"linear_slots = 2\n": ""},                  # missing slot budget
        {"linear_slots = 2": "linear_slots = 0"},    # empty slot pool
        {"linear_slots = 2": "linear_slots = -1"},   # negative slot budget
        {"linear_slots = 2": "linear_slots = 1.5"},  # not an integer slot count
        {'pool = "paged"': 'pool = "bogus"'},         # unknown KV container
        {"block_size = 16": "block_size = 8"},        # outside {16, 32}
        {"block_size = 16\n": ""},                    # paged pool needs block_size
        {"kv_blocks = 64\n": ""},                     # paged pool needs kv_blocks
        {"kv_blocks = 64": "kv_blocks = 0"},          # empty block pool
        {"kv_blocks = 64": "kv_blocks = 1.5"},        # not an integer block count
        {'policy = "continuous"': 'policy = "bogus"'},
        {"batch_sizes = [1]": "batch_sizes = [2]"},   # concurrency is max_batch (P-20)
        {"requests = 4\n": ""},                        # hybrid rides the scenario path
    ],
)
def test_hybrid_config_violations_are_rejected(tmp_path: Path, replacements):
    config = _config(tmp_path, replacements)
    errors = validate_config(load_config(config))
    assert errors, "expected at least one validation error"
    with pytest.raises(ConfigError):
        run_config(config, out_root=tmp_path / "results", allow_dirty=True)


def test_hybrid_seat_pool_does_not_require_block_keys(tmp_path: Path):
    """The seat baseline allocates rows up front; block geometry is paged-only."""
    config = _config(
        tmp_path,
        {
            'pool = "paged"': 'pool = "seat"',
            "kv_blocks = 64\n": "",
            "block_size = 16\n": "",
        },
    )
    assert validate_config(load_config(config)) == []


def test_hybrid_seat_pool_still_validates_block_size_when_written(tmp_path: Path):
    config = _config(tmp_path, {'pool = "paged"': 'pool = "seat"', "block_size = 16": "block_size = 8"})
    errors = validate_config(load_config(config))
    assert any("block_size" in err for err in errors), errors


def test_hybrid_closed_trace_may_be_shorter_than_max_batch(tmp_path: Path):
    """Paged/hybrid max_batch is a scheduling limit, not a seat-allocation bound."""
    config = _config(tmp_path, {"max_batch = 2": "max_batch = 8", "requests = 4": "requests = 2"})
    assert validate_config(load_config(config)) == []


def test_hybrid_fill_config_validates(tmp_path: Path):
    """E4b's fill arm is a Step-3 capability, but the config contract knows the arm."""
    config = _config(
        tmp_path,
        {
            'type = "scenario"': 'type = "fill"',
            'pool = "paged"': 'pool = "seat"',
            "kv_blocks = 64\n": "",
        },
    )
    assert validate_config(load_config(config)) == []


def test_hybrid_fill_adapter_accepts_the_arm(tmp_path: Path):
    """W7 Step 3: fill rides FillScheduler; the Step-2 refusal is gone.

    The unloaded adapter must now fail on load order instead of raising the old
    ``NotImplementedError``; the real fill lifecycle is covered by
    ``tests/unit/test_hybrid_fill.py``.
    """
    engine = _engine(replacements={"_fill": True})
    assert engine.fill is True
    with pytest.raises(RuntimeError, match="call load"):
        engine.run_trace(None)  # the load-order guard fires before the trace is touched


# --------------------------------------------------------------------------- prefill/decode
def test_hybrid_prefill_key_is_provenance_not_a_switch(tmp_path: Path):
    for value in ("per_request", "packed"):
        config = _config(
            tmp_path,
            {"seed = 0\n\n[measure]": f'seed = 0\nprefill = "{value}"\n\n[measure]'},
        )
        assert validate_config(load_config(config)) == [], value


def test_hybrid_rejects_the_deleted_serial_prefill_arm(tmp_path: Path):
    config = _config(
        tmp_path,
        {"seed = 0\n\n[measure]": 'seed = 0\nprefill = "serial"\n\n[measure]'},
    )
    errors = validate_config(load_config(config))
    assert any("prefill" in err and "serial" in err for err in errors), errors


def test_hybrid_decode_triton_is_tolerated_on_the_paged_arm(tmp_path: Path):
    config = _config(tmp_path, {"seed = 0\n\n[measure]": 'seed = 0\ndecode = "triton"\n\n[measure]'})
    assert validate_config(load_config(config)) == []


def test_hybrid_decode_triton_is_refused_on_the_seat_arm(tmp_path: Path):
    config = _config(
        tmp_path,
        {
            'pool = "paged"': 'pool = "seat"',
            "seed = 0\n\n[measure]": 'seed = 0\ndecode = "triton"\n\n[measure]',
        },
    )
    errors = validate_config(load_config(config))
    assert any("decode" in err and "triton" in err for err in errors), errors


# --------------------------------------------------------------------------- factory
def test_build_engine_returns_hybrid_without_loading_weights(tmp_path: Path):
    engine = _engine()
    assert isinstance(engine, HybridEngine)
    assert engine.impl == "hybrid"
    assert (engine.linear_slots, engine.kv_blocks, engine.block_size, engine.pool) == (
        2,
        64,
        16,
        "paged",
    )
    assert engine.device == "cpu"
    assert engine.vocab_size == 1024  # config default before load; read real at load()
    with pytest.raises(NotImplementedError, match="run_trace"):
        engine.generate([1, 2, 3], 4)


def test_factory_does_not_touch_the_model_for_a_dry_run(tmp_path: Path):
    plan = run_config(_config(tmp_path), out_root=tmp_path / "results", dry_run=True)
    assert plan["engine"]["impl"] == "hybrid"
    assert not (tmp_path / "results").exists()


# --------------------------------------------------------------------------- manifest
def test_manifest_records_w7_paged_fields_and_packed_prefill():
    engine = _engine(replacements={"block_size": 32, "_policy": "static"})
    entry = _scheduler_manifest(engine, fill_only=False)
    assert entry["seats"] == "linear_slots_plus_block_tables"
    assert entry["pool"] == "paged"
    assert entry["linear_slots"] == 2
    assert entry["kv_blocks"] == 64
    assert entry["block_size"] == 32
    assert entry["decode"] == "paged-triton"
    assert entry["policy"] == "static"
    assert entry["max_batch"] == 2
    # Step 3: the engine runs one packed varlen forward per FIFO group.
    assert entry["prefill"] == "packed"


def test_manifest_records_the_seat_baseline():
    engine = _engine(replacements={"pool": "seat"})
    entry = _scheduler_manifest(engine, fill_only=False)
    assert entry["seats"] == "one_kvslab_one_row_each"
    assert entry["pool"] == "seat"
    assert entry["kv_blocks"] is None
    assert entry["decode"] == "hand_baddbmm"
    assert entry["prefill"] == "packed"


def test_summary_protocol_records_hybrid_packed_prefill(tmp_path: Path):
    cfg = load_config(_config(tmp_path))
    engine = _engine()
    manifest = {
        "run_id": "W7-9001",
        "type": "scenario",
        "device_class": "cpu",
        "git": {"dirty": True},
        "engine": {
            "impl": "hybrid",
            "dtype": "bfloat16",
            "attn": {"prefill": "sdpa", "decode": "paged-triton"},
            "max_prefill_tokens": 8192,
        },
        "scheduler": _scheduler_manifest(engine, fill_only=False),
        "trace": {},
    }
    summary = _summary(cfg, manifest, rows=[])
    assert summary["protocol"]["prefill_mode"] == "packed"
    assert summary["protocol"]["decode"] == "paged-triton"


# --------------------------------------------------------------------------- routing
def test_closed_hybrid_trace_uses_the_scenario_path(tmp_path: Path, monkeypatch):
    """Closed arrival must still take the scheduler route, like continuous/paged."""
    cfg = load_config(_config(tmp_path))
    calls: list[tuple] = []

    def fake_scenario(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"row_kind": "trace", "ok": True}]

    monkeypatch.setattr("bench.run._scenario_execute", fake_scenario)
    rows = _execute(cfg, tmp_path, {}, {}, None, None, None)
    assert len(calls) == 1
    assert rows == [{"row_kind": "trace", "ok": True}]
