"""The run contract itself is testable: dry-run, immutable dirs, index validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.index import check_root
from bench.run import ConfigError, RunExistsError, load_config, run_config, validate_config

CONFIG = """
schema = 1

[exp]
id = "W0-9001"
stage = "W0"
type = "smoke"
slug = "runner-smoke"
title = "contract smoke"
hypothesis = []
seed = 0

[[models]]
name = "dummy"

[workload]
kind = "synthetic"
context_lengths = [8, 16]
max_new_tokens = 2
batch_sizes = [1]
arrival = "closed"

[engine]
impl = "dummy"
device = "cpu"
vocab_size = 1024
seed = 0

[measure]
warmup = 1
repeats = 2
timer = "perf_counter"
memory = ["torch_peak"]

[accept]
mode = "record_only"
criteria = []
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "smoke.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_dry_run_writes_nothing(tmp_path: Path):
    plan = run_config(_write_config(tmp_path), out_root=tmp_path / "results", dry_run=True)
    assert plan["run_id"] == "W0-9001"
    assert "0001" not in plan["run_dir"]  # 9001-runner-smoke
    assert not (tmp_path / "results").exists()


def test_run_contract_and_index(tmp_path: Path):
    root = tmp_path / "results"
    run_dir = run_config(
        _write_config(tmp_path),
        out_root=root,
        allow_dirty=True,
    )
    assert run_dir == root / "W0" / "9001-runner-smoke"
    for name in (
        "config.toml",
        "manifest.json",
        "env.json",
        "metrics.jsonl",
        "summary.json",
        "notes.md",
        "verdict.json",
        "reproduce.sh",
        "stdout.log",
    ):
        assert (run_dir / name).exists(), name

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    verdict = json.loads((run_dir / "verdict.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["n_rows"] == 4  # 2 contexts x 2 repeats
    # Which attention backend ran is provenance, not a note: a paired comparison across
    # two different backends must come out "invalid" rather than produce a ratio.
    assert manifest["engine"]["attn"] == {"prefill": "dummy", "decode": "dummy"}
    assert summary["protocol"]["attn_prefill"] == "dummy"
    assert summary["protocol"]["attn_decode"] == "dummy"
    assert summary["n_rows"] == 4
    assert summary["overall"]["ttft_ms"]["n"] == 4
    assert verdict["verdict"] == "unreviewed"

    errors = check_root(root)
    assert errors == [], errors
    assert "W0-9001" in (root / "INDEX.md").read_text(encoding="utf-8")


def test_duplicate_run_id_is_refused(tmp_path: Path):
    root = tmp_path / "results"
    config = _write_config(tmp_path)
    run_config(config, out_root=root, allow_dirty=True)
    with pytest.raises(RunExistsError):
        run_config(config, out_root=root, allow_dirty=True)


def test_invalid_config_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text("schema = 1\n[exp]\nstage = 'W0'\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        run_config(path, out_root=tmp_path / "results", allow_dirty=True)


def test_check_catches_missing_verdict(tmp_path: Path):
    root = tmp_path / "results"
    run_dir = run_config(_write_config(tmp_path), out_root=root, allow_dirty=True)
    (run_dir / "verdict.json").unlink()
    errors = check_root(root)
    assert any("verdict.json" in error for error in errors), errors


# ---------------------------------------------------------------- W3 scenario contract
SCENARIO_CONFIG = """
schema = 1

[exp]
id = "W3-9009"
stage = "W3"
type = "scenario"
slug = "scenario-smoke"
title = "scenario contract smoke"
hypothesis = []
seed = 0

[[models]]
name = "dummy"
revision = "0000000000000000000000000000000000000000"

[workload]
kind = "prompts"
prompts = "bench/prompts.json"
max_new_tokens = 8
arrival = "poisson"
rate = 40.0
requests = 16
budgets = [4, 8]
batch_sizes = [1]

[scheduler]
policy = "continuous"

[engine]
impl = "continuous"
device = "cpu"
max_batch = 4
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


def _scenario_config(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    text = SCENARIO_CONFIG
    for old, new in (replacements or {}).items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "scenario.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_historical_closed_config_without_requests_still_validates(tmp_path: Path):
    """P-28: `requests` is not a global requirement -- W0/W2 configs never wrote it."""
    path = _write_config(tmp_path)
    plan = run_config(path, out_root=tmp_path / "results", dry_run=True)
    assert "requests" not in plan["workload"]


def test_scenario_config_is_accepted(tmp_path: Path):
    config = _scenario_config(tmp_path)
    assert validate_config(load_config(config)) == []


def test_scenario_run_writes_all_three_row_kinds(tmp_path: Path):
    root = tmp_path / "results"
    run_dir = run_config(_scenario_config(tmp_path), out_root=root, allow_dirty=True)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    kinds = {row.get("row_kind") for row in rows}
    assert kinds == {"request", "trace", "step"}, kinds

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert manifest["scheduler"]["policy"] == "continuous"
    assert manifest["scheduler"]["max_batch"] == 4
    assert manifest["trace"]["sha256"]
    assert summary["protocol"]["trace_sha256"] == manifest["trace"]["sha256"]
    assert summary["protocol"]["max_batch"] == 4
    assert "throughput" in summary and "latency" in summary and "steps" in summary
    # `overall` keeps the W2 meaning: request rows only, never trace/step rows.
    assert summary["overall"]["ttft_ms"]["n"] == 16
    # The memory account (bench/thresholds.py::_memory_ratio) reads groups, so the request
    # rows must carry the fields it divides: context, weights, KV, and the observed peak.
    request_rows = [row for row in rows if row.get("row_kind") == "request"]
    for row in request_rows:
        for name in ("context", "weights_mb", "peak_mem_mb", "kv_mb"):
            assert row.get(name) is not None, name
        assert row["context"] > 0
    # A bucket at context=0 only would leave the memory criterion permanently pending.
    assert any(group["context"] > 0 for group in summary["groups"]), summary["groups"]
    assert check_root(root) == []


@pytest.mark.parametrize(
    "replacements",
    [
        {"rate = 40.0": "rate = 0.0"},                       # poisson without a rate
        {"requests = 16\n": ""},                              # requests missing on scenario
        {"batch_sizes = [1]": "batch_sizes = [2]"},          # concurrency as a row label
        {"budgets = [4, 8]": "budgets = [4, 128]"},          # budget above max_new_tokens
        {'policy = "continuous"': 'policy = "bogus"'},       # unknown policy
        {'type = "scenario"': 'type = "microbench"'},        # poisson outside scenario
    ],
)
def test_scenario_config_violations_are_rejected(tmp_path: Path, replacements):
    config = _scenario_config(tmp_path, replacements)
    errors = validate_config(load_config(config))
    assert errors, "expected at least one validation error"
    with pytest.raises(ConfigError):
        run_config(config, out_root=tmp_path / "results", allow_dirty=True)


def test_scheduler_table_without_continuous_impl_is_rejected(tmp_path: Path):
    config = _scenario_config(
        tmp_path,
        {
            'impl = "continuous"': 'impl = "dummy"',
            'arrival = "poisson"': 'arrival = "closed"',
            "requests = 16\n": "",
        },
    )
    errors = validate_config(load_config(config))
    assert any("[scheduler]" in err for err in errors), errors


def test_closed_batch_requires_at_least_one_request_per_seat(tmp_path: Path):
    """Fewer requests than seats would leave the pool partly idle and measure neither batch
    nor capacity (the E1 shape). More is allowed: a longer closed trace is the capacity probe."""
    config = _scenario_config(
        tmp_path,
        {'arrival = "poisson"': 'arrival = "closed"', "requests = 16": "requests = 2"},
    )
    errors = validate_config(load_config(config))
    assert any("requests must be >=" in err for err in errors), errors


def test_closed_batch_allows_a_longer_trace(tmp_path: Path):
    """Capacity probe: a closed trace longer than the pool keeps every seat busy until the
    trace drains, which is what ``req_s`` needs to be a sustained-throughput estimate."""
    config = _scenario_config(
        tmp_path,
        {'arrival = "poisson"': 'arrival = "closed"', "requests = 16": "requests = 64"},
    )
    assert validate_config(load_config(config)) == []

