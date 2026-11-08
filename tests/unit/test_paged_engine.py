"""W4 paged arm: factory wiring, fill-only capacity, and the summary blocks it feeds.

Everything here runs on the dummy stand-ins (no weights, no GPU): the point is that the
runner/scenario/threshold plumbing sees ``capacity`` and ``memory`` blocks with the field
names D6 froze, and that the seat arm's W3 output is unchanged. Numbers from these tests
are meaningless by construction and are never asserted as benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import scenario
from bench.engines import build_engine
from bench.engines.continuous import ContinuousEngine
from bench.engines.fill import FillScheduler, W4_TOTAL_BUDGET_MB, capacity_block
from bench.engines.paged import PagedEngine
from bench.loadgen import PromptSet, closed_trace
from bench.run import _flatten_scenario, load_config, run_config, validate_config

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "bench" / "prompts.json"


# --------------------------------------------------------------------------- factory
def test_factory_wires_paged_and_keeps_hybrid_loud():
    engine = build_engine(
        {"name": "dummy"},
        {"impl": "paged", "max_batch": 4, "block_size": 16, "kv_blocks": 64},
    )
    assert isinstance(engine, PagedEngine)
    assert engine.impl == "paged"
    assert (engine.block_size, engine.kv_blocks, engine.max_batch) == (16, 64, 4)
    with pytest.raises(NotImplementedError, match="W7"):
        build_engine({"name": "x"}, {"impl": "hybrid"})


def test_paged_arm_refuses_per_request_generate():
    engine = build_engine(
        {"name": "dummy"}, {"impl": "paged", "max_batch": 1, "block_size": 16, "kv_blocks": 8}
    )
    with pytest.raises(NotImplementedError, match="run_trace"):
        engine.generate([1, 2, 3], 4)


# --------------------------------------------------------------------------- fill run
def _paged_trace(n: int = 8, budgets=(8,), seed: int = 0):
    engine = build_engine(
        {"name": "dummy",
         "revision": "0" * 40},
        {"impl": "paged", "max_batch": 4, "block_size": 16, "kv_blocks": 8,
         "vocab_size": 1024, "seed": 0, "_fill": True},
    )
    engine.load()
    prompt_set = PromptSet.load(PROMPTS)
    entries = [(prompt.id, engine.encode(prompt.text)) for prompt in prompt_set.prompts[:n]]
    trace = closed_trace(entries, n=n, budgets=list(budgets), seed=seed)
    return engine, trace


def test_fill_run_stops_after_admission_and_reports_capacity():
    engine, trace = _paged_trace()
    try:
        result = engine.run_trace(trace)
    finally:
        engine.close()

    assert result.impl == "paged"
    assert result.step_rows, "a fill run still writes step rows"
    for row in result.step_rows:
        # `_steps` groups on `active`; fill rows must carry it and never decode.
        assert "active" in row and row["active"] >= 0
        assert row["fill_only"] is True
        assert row["decode_event_ms"] == 0.0
        assert "paged_stats" not in row
        assert "blocks_in_use" in row and "workspace_mb" in row
    # Every request was admitted eventually; fill generates no tokens.
    assert all(req["state"] == "finished" for req in result.request_rows)
    assert all(req["out_tokens"] == 0 for req in result.request_rows)
    capacity = result.summary["capacity"]
    assert capacity["peak_concurrency"] == max(row["active"] for row in result.step_rows)
    assert capacity["block_size"] == 16 and capacity["kv_blocks"] == 8
    # Triton paged decode has no gather workspace: the total budget is the pool itself.
    assert capacity["workspace_mb"] == 0.0
    assert capacity["kv_total_mb"] == capacity["kv_mb"]
    assert capacity["workload_sha256"] == trace.sha256


def test_fill_capacity_is_bounded_by_blocks_not_seats():
    # 8 blocks x 16 tokens: the dummy prompts are ~20-35 tokens + budget 8, so only a
    # handful fit; the cap must come from blocks, not from max_batch=64.
    engine = build_engine(
        {"name": "dummy"}, {"impl": "paged", "max_batch": 64, "block_size": 16,
                            "kv_blocks": 8, "vocab_size": 1024, "_fill": True},
    )
    engine.load()
    try:
        prompt_set = PromptSet.load(PROMPTS)
        entries = [(p.id, engine.encode(p.text)) for p in prompt_set.prompts[:8]]
        trace = closed_trace(entries, n=8, budgets=[8], seed=0)
        result = engine.run_trace(trace)
    finally:
        engine.close()
    assert result.summary["capacity"]["peak_concurrency"] < 8
    assert result.summary["capacity"]["peak_concurrency"] <= 64


def test_fill_request_rows_carry_the_block_account():
    engine, trace = _paged_trace(n=4)
    try:
        result = engine.run_trace(trace)
    finally:
        engine.close()
    for row in result.request_rows:
        assert row["blocks_reserved"] >= row["blocks_written"] >= 1
        # kv_own_mb is this request's reservation, not pool / max_batch.
        assert row["kv_own_mb"] is not None
        assert row["kv_own_mb"] > 0.0
        assert row["block_size"] == 16


def test_fill_scheduler_is_a_scheduler_subclass():
    from serve.scheduler import Scheduler

    assert issubclass(FillScheduler, Scheduler)


def test_seat_fill_run_reaches_the_capacity_block(tmp_path: Path):
    """`type="fill"` on the continuous arm is the E1 baseline: it must run and report."""
    text = PAGED_FILL.replace(
        'impl = "paged"', 'impl = "continuous"'
    ).replace(
        "block_size = 16\nkv_blocks = 8\n", ""
    ).replace("W4-9002", "W4-9003").replace("paged-fill-test", "seat-fill-test")
    config = tmp_path / "seat-fill.toml"
    config.write_text(text, encoding="utf-8")
    assert validate_config(load_config(config)) == []
    run_dir = run_config(config, out_root=tmp_path / "results", allow_dirty=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["scheduler"]["fill_only"] is True
    assert manifest["scheduler"]["seats"] == "one_kvslab_one_row_each"
    assert summary["capacity"]["peak_concurrency"] == 4  # = max_batch, no block limit
    assert summary["capacity"]["seat_unused_mb"] is not None
    assert summary["capacity"]["kv_total_mb"] == pytest.approx(W4_TOTAL_BUDGET_MB)
    assert summary["memory"]["workspace_mb"] == 0.0
    assert summary["key_metric"]["name"] == "peak_concurrency"


def test_fill_requires_a_scheduler_arm(tmp_path: Path):
    """A per-request arm has no scheduler to run fill on; the config must be refused."""
    config = tmp_path / "bad-fill.toml"
    config.write_text(
        PAGED_FILL.replace('impl = "paged"', 'impl = "minillm"')
        .replace("[scheduler]\npolicy = \"continuous\"\n\n", "")
        .replace("W4-9002", "W4-9004"),
        encoding="utf-8",
    )
    errors = validate_config(load_config(config))
    assert any("fill" in error for error in errors), errors


# --------------------------------------------------------------------------- summary
def _rows_for(config_text: str, tmp_path: Path):
    config = tmp_path / "w4.toml"
    config.write_text(config_text, encoding="utf-8")
    assert validate_config(load_config(config)) == []
    run_dir = run_config(config, out_root=tmp_path / "results", allow_dirty=True)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return rows, manifest, summary


PAGED_FILL = """
schema = 1

[exp]
id = "W4-9002"
stage = "W4"
type = "fill"
slug = "paged-fill-test"
title = "paged fill on dummy"
hypothesis = []
seed = 0

[[models]]
name = "dummy"
revision = "0000000000000000000000000000000000000000"
dtype = "float32"

[workload]
kind = "prompts"
prompts = "bench/prompts.json"
max_new_tokens = 8
arrival = "closed"
requests = 8
budgets = [8]
batch_sizes = [1]

[scheduler]
policy = "continuous"

[engine]
impl = "paged"
device = "cpu"
max_batch = 4
block_size = 16
kv_blocks = 8
vocab_size = 1024
seed = 0

[measure]
warmup = 0
repeats = 2
timer = "perf_counter"
memory = ["torch_peak"]

[accept]
mode = "record_only"
criteria = []
"""


def test_decode_gather_control_arm_is_rejected(tmp_path: Path):
    """(a) W4.5 step 2 deleted the gather arm; a config asking for it must fail loudly."""
    text = PAGED_FILL.replace(
        "seed = 0\n\n[measure]", 'seed = 0\ndecode = "gather"\n\n[measure]'
    )
    config = tmp_path / "decode-gather.toml"
    config.write_text(text, encoding="utf-8")

    errors = validate_config(load_config(config))
    assert errors, "decode='gather' must be rejected"
    joined = " ".join(errors)
    assert "deleted" in joined and "gather" in joined
    assert "tests/reference/paged.py" in joined


def test_decode_triton_is_a_legacy_noop(tmp_path: Path):
    """(b) Historical packed W45 configs wrote decode='triton'; it still validates."""
    text = PAGED_FILL.replace(
        "seed = 0\n\n[measure]", 'seed = 0\ndecode = "triton"\n\n[measure]'
    )
    config = tmp_path / "decode-triton.toml"
    config.write_text(text, encoding="utf-8")

    assert validate_config(load_config(config)) == []


def test_decode_triton_requires_the_paged_arm(tmp_path: Path):
    """'triton' only names the block-table pool; other arms reject the legacy key."""
    text = (
        PAGED_FILL.replace('impl = "paged"', 'impl = "continuous"')
        .replace("block_size = 16\nkv_blocks = 8\n", "")
        .replace('[scheduler]\npolicy = "continuous"\n\n', "")
        .replace("seed = 0\n\n[measure]", 'seed = 0\ndecode = "triton"\n\n[measure]')
        .replace("W4-9002", "W4-9005")
    )
    config = tmp_path / "decode-continuous.toml"
    config.write_text(text, encoding="utf-8")

    errors = validate_config(load_config(config))
    assert any("requires [engine].impl='paged'" in error for error in errors), errors


def test_paged_fill_summary_and_rows_expose_capacity_and_memory(tmp_path: Path):
    rows, manifest, summary = _rows_for(PAGED_FILL, tmp_path)

    assert manifest["engine"]["impl"] == "paged"
    assert manifest["scheduler"]["seats"] == "paged_blocks"
    assert manifest["scheduler"]["block_size"] == 16
    assert manifest["scheduler"]["kv_blocks"] == 8
    assert manifest["scheduler"]["fill_only"] is True
    assert manifest["scheduler"]["decode"] == "triton"

    assert summary["protocol"]["block_size"] == 16
    assert summary["protocol"]["kv_blocks"] == 8
    assert summary["protocol"]["workload_sha256"] == manifest["trace"]["sha256"]
    assert summary["key_metric"]["name"] == "peak_concurrency"

    capacity = summary["capacity"]
    assert capacity["peak_concurrency"] >= 1
    assert capacity["max_batch"] == 4
    assert capacity["workspace_mb"] == 0.0
    assert "workspace_estimated" not in capacity
    assert capacity["kv_total_mb"] == pytest.approx(
        capacity["kv_mb"] + capacity["workspace_mb"]
    )
    memory = summary["memory"]
    for key in (
        "kv_mb", "workspace_mb", "kv_total_mb", "seat_unused_mb", "frag_internal",
        "reservation_slack", "seat_row_slack", "capacity_per_gib", "bytes_per_active_request",
    ):
        assert key in memory, key
    assert summary["throughput"]["peak_concurrency"] is not None

    step_rows = [row for row in rows if row["row_kind"] == "step"]
    assert step_rows and all(row["active"] >= 0 for row in step_rows)
    assert all(row["impl"] == "paged" for row in step_rows)
    request_rows = [row for row in rows if row["row_kind"] == "request"]
    assert request_rows and all(row["impl"] == "paged" for row in request_rows)


def test_flatten_scenario_uses_the_result_impl(tmp_path: Path):
    engine, trace = _paged_trace(n=4)
    try:
        result = engine.run_trace(trace)
    finally:
        engine.close()
    manifest = {
        "run_id": "W4-9002",
        "engine": {"resolved_device": "cpu", "impl": "paged"},
        "device_class": "cpu",
        "scheduler": {"policy": "continuous", "max_batch": 4, "seats": "paged_blocks",
                      "block_size": 16, "kv_blocks": 8},
    }
    request_rows, step_rows = _flatten_scenario(
        result, {"name": "dummy"}, manifest, weights_mb=0.0, peak_mem_mb=0.0,
        driver_mem_mb=0.0,
    )
    assert {row["impl"] for row in request_rows + step_rows} == {"paged"}
    assert all(row["block_size"] == 16 and row["kv_blocks"] == 8 for row in request_rows + step_rows)


def test_continuous_summary_is_unchanged_by_the_new_blocks(tmp_path: Path):
    """The seat arm keeps every W3 key and value; only additive blocks appear."""
    from bench.engines.continuous import ContinuousEngine

    engine = ContinuousEngine("dummy", max_batch=4, vocab_size=1024, seed=0)
    engine.load()
    try:
        prompt_set = PromptSet.load(PROMPTS)
        entries = [(p.id, engine.encode(p.text)) for p in prompt_set.prompts[:8]]
        trace = closed_trace(entries, n=8, budgets=[8], seed=0)
        result = engine.run_trace(trace)
    finally:
        engine.close()

    assert result.impl == "continuous"
    assert "capacity" not in result.summary
    assert set(result.summary["throughput"]) == {
        "n_traces", "tok_s", "tok_s_active", "decode_only_tok_s", "req_s", "makespan_s",
        "offered_rate", "achieved_rate", "saturated", "queue_depth_at_last_arrival",
        "analytic_straggler_ratio", "peak_concurrency",
    }
    assert set(result.summary) == {"throughput", "latency", "steps", "consistency", "memory"}
    # The seat arm has no block ledger: those fields are None, never invented.
    memory = result.summary["memory"]
    assert memory["frag_internal"] is None
    assert memory["seat_unused_mb"] is None


def test_aggregate_memory_block_without_any_ledger_is_all_none():
    memory = scenario.aggregate([], [], [])["memory"]
    assert set(memory) == {
        "kv_mb", "workspace_mb", "kv_total_mb", "seat_unused_mb", "frag_internal",
        "reservation_slack", "seat_row_slack", "capacity_per_gib",
        "bytes_per_active_request",
    }
    assert all(value is None for value in memory.values())


def test_capacity_block_never_divides_by_kv_alone():
    block = capacity_block(
        peak_concurrency=20,
        kv_mb=1400.0,
        workspace_mb=200.0,
        kv_total_mb=1600.0,
        block_size=16,
        kv_blocks=800,
        max_batch=64,
        workload_sha256="abc",
    )
    assert block["capacity_per_gib"] == pytest.approx(20 / (1600 / 1024))
    assert block["kv_total_mb"] == pytest.approx(W4_TOTAL_BUDGET_MB - 40.0)


# --------------------------------------------------------------------------- thresholds
def test_capacity_criterion_is_no_longer_pending_on_a_paged_summary(tmp_path: Path):
    """C1 must be evaluable: `_summary` hands `capacity` to the evaluator (with a baseline)."""
    from bench import thresholds

    ours, manifest, summary = _rows_for(PAGED_FILL, tmp_path)
    baseline = {
        "protocol": {
            "dtype": summary["protocol"]["dtype"],
            "device_class": summary["protocol"]["device_class"],
            "workload_sha256": summary["protocol"]["workload_sha256"],
        },
        "capacity": {
            "peak_concurrency": 1,
            "kv_total_mb": summary["capacity"]["kv_total_mb"],
            "block_size": None,
        },
    }
    entry = thresholds.evaluate(
        ["max_concurrency_speedup_min"], summary, {"W4": {"paged": {"max_concurrency_speedup_min": 1.5}}},
        baseline=baseline,
    )[0]
    assert entry["status"] in {"pass", "fail"}, entry
    assert entry["status"] != "pending"
