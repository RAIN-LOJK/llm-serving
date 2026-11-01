"""The run contract itself is testable: dry-run, immutable dirs, index validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.index import check_root
from bench.run import ConfigError, RunExistsError, run_config

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
