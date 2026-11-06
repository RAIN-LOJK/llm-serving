"""The run contract itself is testable: dry-run, immutable dirs, index validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.index import check_root
from bench.loadgen import PromptSet
from bench.run import (
    ConfigError,
    RunExistsError,
    _build_trace,
    _cases_for_model,
    _effective_prompt_sha256,
    _prompt_info,
    load_config,
    run_config,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "bench" / "prompts.json"
LONG_IDS = ["l01", "l02", "l03", "l04", "l05", "l06", "l07", "l08"]

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


# ---------------------------------------------------------------- W4 paged / fill contract
PAGED_CONFIG = """
schema = 1

[exp]
id = "W4-9009"
stage = "W4"
type = "microbench"
slug = "paged-contract"
title = "paged contract smoke"
hypothesis = []
seed = 0

[[models]]
name = "dummy"
revision = "0000000000000000000000000000000000000000"

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
impl = "paged"
device = "cpu"
max_batch = 4
block_size = 16
kv_blocks = 64
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


def _paged_config(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    text = PAGED_CONFIG
    for old, new in (replacements or {}).items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "paged.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_paged_config_is_accepted(tmp_path: Path):
    """D6: a paged arm needs block_size/kv_blocks and may carry ``[scheduler]``."""
    assert validate_config(load_config(_paged_config(tmp_path))) == []


def test_paged_accepts_continuous_and_static_policies(tmp_path: Path):
    for policy in ("static", "continuous"):
        config = _paged_config(tmp_path, {'policy = "continuous"': f'policy = "{policy}"'})
        assert validate_config(load_config(config)) == [], policy


@pytest.mark.parametrize(
    "replacements",
    [
        {"block_size = 16\n": ""},                 # missing block_size
        {"kv_blocks = 64\n": ""},                   # missing kv_blocks
        {"block_size = 16": "block_size = 8"},      # outside {16, 32}
        {"kv_blocks = 64": "kv_blocks = 0"},        # empty pool
        {"kv_blocks = 64": "kv_blocks = -4"},       # negative pool
        {"kv_blocks = 64": "kv_blocks = 1.5"},      # not an integer block count
        {'policy = "continuous"': 'policy = "bogus"'},
        {"batch_sizes = [1]": "batch_sizes = [2]"},  # concurrency is max_batch (P-20)
        {"requests = 4\n": ""},                       # paged is on the scenario path
    ],
)
def test_paged_config_violations_are_rejected(tmp_path: Path, replacements):
    config = _paged_config(tmp_path, replacements)
    errors = validate_config(load_config(config))
    assert errors, "expected at least one validation error"
    with pytest.raises(ConfigError):
        run_config(config, out_root=tmp_path / "results", allow_dirty=True)


def test_paged_block_size_32_is_also_allowed(tmp_path: Path):
    assert validate_config(load_config(_paged_config(tmp_path, {"block_size = 16": "block_size = 32"}))) == []


def test_paged_block_size_must_be_16_or_32(tmp_path: Path):
    errors = validate_config(load_config(_paged_config(tmp_path, {"block_size = 16": "block_size = 8"})))
    assert any("block_size" in err for err in errors), errors


def test_paged_without_block_size_reports_it(tmp_path: Path):
    errors = validate_config(load_config(_paged_config(tmp_path, {"block_size = 16\n": ""})))
    assert any("block_size" in err for err in errors), errors


def test_paged_without_kv_blocks_reports_it(tmp_path: Path):
    errors = validate_config(load_config(_paged_config(tmp_path, {"kv_blocks = 64\n": ""})))
    assert any("kv_blocks" in err for err in errors), errors


def test_minillm_with_scheduler_is_still_rejected(tmp_path: Path):
    """Only continuous/paged have a scheduler to steer; minillm must keep refusing the knob."""
    config = _paged_config(
        tmp_path,
        {
            'impl = "paged"': 'impl = "minillm"',
            "block_size = 16\n": "",
            "kv_blocks = 64\n": "",
            "requests = 4\n": "",
        },
    )
    errors = validate_config(load_config(config))
    assert any("[scheduler]" in err for err in errors), errors


FILL_CONFIG = """
schema = 1

[exp]
id = "W4-9008"
stage = "W4"
type = "fill"
slug = "fill-contract"
title = "fill contract smoke"
hypothesis = []
seed = 0

[[models]]
name = "dummy"
revision = "0000000000000000000000000000000000000000"

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


def _fill_config(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    text = FILL_CONFIG
    for old, new in (replacements or {}).items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "fill.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_fill_config_is_accepted_on_both_workload_kinds(tmp_path: Path):
    assert validate_config(load_config(_fill_config(tmp_path))) == []
    synthetic = _fill_config(
        tmp_path,
        {
            'kind = "prompts"\nprompts = "bench/prompts.json"': 'kind = "synthetic"\ncontext_lengths = [64]',
        },
    )
    assert validate_config(load_config(synthetic)) == []


def test_fill_requires_requests(tmp_path: Path):
    config = _fill_config(tmp_path, {"requests = 8\n": ""})
    errors = validate_config(load_config(config))
    assert any("requests" in err and "fill" in err for err in errors), errors
    with pytest.raises(ConfigError):
        run_config(config, out_root=tmp_path / "results", allow_dirty=True)


def test_fill_does_not_require_requests_on_a_closed_minillm_arm(tmp_path: Path):
    """The new 'fill' rule must not leak into other types, and a plain closed arm is unchanged."""
    config = _fill_config(
        tmp_path,
        {
            'type = "fill"': 'type = "microbench"',
            "requests = 8\n": "",
            'impl = "continuous"': 'impl = "minillm"',
            '[scheduler]\npolicy = "continuous"\n\n': "",
        },
    )
    assert validate_config(load_config(config)) == []


def test_w3_configs_still_validate():
    """P-28 regression: the W4 additions must not change any existing W3 verdict."""
    repo_root = Path(__file__).resolve().parents[2]
    for name in (
        "W3/0001-single-request-recheck.toml",
        "W3/0011-continuous-b8-mixed.toml",
        "W3/9001-scheduler-smoke.toml",
        "W3/9003-capacity-b8-mixed.toml",
    ):
        path = repo_root / "configs" / "experiments" / name
        assert validate_config(load_config(path)) == [], name


# ---------------------------------------------------------------- optional prompt subset
def _toml_ids(ids: list[str]) -> str:
    return "prompt_ids = [" + ", ".join(f'"{pid}"' for pid in ids) + "]\n"


def _subset_config(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    """SCENARIO_CONFIG with an explicit `prompt_ids` subset."""
    base = {"budgets = [4, 8]\n": "budgets = [4, 8]\n" + _toml_ids(LONG_IDS)}
    return _scenario_config(tmp_path, {**base, **(replacements or {})})


def test_prompt_ids_subset_is_accepted(tmp_path: Path):
    assert validate_config(load_config(_subset_config(tmp_path))) == []


def test_prompt_ids_must_exist_in_the_prompt_set(tmp_path: Path):
    config = _scenario_config(
        tmp_path,
        {"budgets = [4, 8]\n": 'budgets = [4, 8]\nprompt_ids = ["l01", "nope"]\n'},
    )
    errors = validate_config(load_config(config))
    assert any("nope" in err for err in errors), errors


@pytest.mark.parametrize(
    "line",
    [
        "prompt_ids = []\n",                    # empty list
        "prompt_ids = [1, 2]\n",                # not strings
        "prompt_ids = 'l01'\n",                 # not a list
    ],
)
def test_invalid_prompt_ids_are_rejected(tmp_path: Path, line: str):
    config = _scenario_config(tmp_path, {"budgets = [4, 8]\n": "budgets = [4, 8]\n" + line})
    errors = validate_config(load_config(config))
    assert any("prompt_ids" in err for err in errors), errors


def test_prompt_ids_on_synthetic_workload_is_rejected(tmp_path: Path):
    """A key that changes nothing must not be silently accepted on the synthetic path."""
    config = _scenario_config(
        tmp_path,
        {
            'kind = "prompts"\nprompts = "bench/prompts.json"': 'kind = "synthetic"\ncontext_lengths = [64]',
            "budgets = [4, 8]\n": 'budgets = [4, 8]\nprompt_ids = ["l01"]\n',
            'arrival = "poisson"': 'arrival = "closed"',
            "requests = 16": "requests = 4",
        },
    )
    errors = validate_config(load_config(config))
    assert any("prompt_ids" in err and "synthetic" in err for err in errors), errors


def test_prompt_ids_filter_builds_a_deterministic_trace(tmp_path: Path):
    from bench.engines.dummy import DummyEngine

    cfg = load_config(_subset_config(tmp_path))
    prompt_set = PromptSet.load(PROMPTS)
    engine = DummyEngine(vocab_size=1024, seed=0)

    traced = _build_trace(cfg, engine, prompt_set, requests=8, budgets=(4,), max_new=8)
    assert [a.prompt_id for a in traced.arrivals] == LONG_IDS
    # The runner feeds tokenized entries, so the trace's own `prompt_sha256` stays None
    # (unchanged W3 behaviour); workload identity lives in the manifest's `[prompt]` block.
    assert traced.prompt_sha256 is None
    assert _prompt_info(cfg)["sha256"] == _effective_prompt_sha256(prompt_set, LONG_IDS)

    # Rebuilding is deterministic, and only the named prompts appear.
    again = _build_trace(cfg, engine, prompt_set, requests=8, budgets=(4,), max_new=8)
    assert traced.sha256 == again.sha256
    assert {a.prompt_id for a in again.arrivals} == set(LONG_IDS)

    # Reversing the ids reverses the trace order: the order is the configured order.
    reversed_cfg = load_config(
        _subset_config(
            tmp_path,
            {"budgets = [4, 8]\n": "budgets = [4, 8]\n" + _toml_ids(LONG_IDS[::-1])},
        )
    )
    reversed_trace = _build_trace(reversed_cfg, engine, prompt_set, requests=8, budgets=(4,), max_new=8)
    assert [a.prompt_id for a in reversed_trace.arrivals] == LONG_IDS[::-1]
    assert reversed_trace.sha256 != traced.sha256


def test_prompt_ids_filter_limits_per_request_cases(tmp_path: Path):
    from bench.engines.dummy import DummyEngine

    cfg = load_config(_subset_config(tmp_path))
    cases = list(_cases_for_model(cfg, DummyEngine(vocab_size=1024), PromptSet.load(PROMPTS)))
    assert [case["prompt_id"] for case in cases] == LONG_IDS
    assert all(case["bucket"] == "long" for case in cases)


def test_prompt_sha256_distinguishes_subsets_and_the_full_set(tmp_path: Path):
    prompt_set = PromptSet.load(PROMPTS)
    full = _effective_prompt_sha256(prompt_set, None)
    assert full == prompt_set.sha256
    assert _prompt_info(load_config(_subset_config(tmp_path)))["sha256"] != full

    other = ["l01", "l02"]
    assert _effective_prompt_sha256(prompt_set, LONG_IDS) != _effective_prompt_sha256(prompt_set, other)
    # Order is a trace concern, not a workload-identity one.
    assert _effective_prompt_sha256(prompt_set, LONG_IDS) == _effective_prompt_sha256(
        prompt_set, LONG_IDS[::-1]
    )


def test_subset_manifest_records_both_hashes_and_the_ids(tmp_path: Path):
    cfg = load_config(_subset_config(tmp_path))
    info = _prompt_info(cfg)
    assert info["prompt_ids"] == LONG_IDS
    assert info["n_prompts"] == len(LONG_IDS)
    assert info["prompt_set_sha256"] == PromptSet.load(PROMPTS).sha256
    assert info["n_prompts_in_set"] == 32
    assert info["sha256"] != info["prompt_set_sha256"]


def test_historical_prompt_configs_keep_the_full_prompt_sha256():
    """P-28: configs without `prompt_ids` must be byte-for-byte unchanged."""
    repo_root = Path(__file__).resolve().parents[2]
    full = PromptSet.load(PROMPTS).sha256
    for name in ("W3/0011-continuous-b8-mixed.toml", "W3/9001-scheduler-smoke.toml"):
        cfg = load_config(repo_root / "configs" / "experiments" / name)
        assert "prompt_ids" not in cfg["workload"], name
        info = _prompt_info(cfg)
        assert info["sha256"] == full, name
        assert "prompt_ids" not in info, name


def test_w4_configs_validate_and_paired_arms_share_a_workload():
    """Every committed W4 config must pass the contract, and each pair must differ only in
    [engine] -- same seed and same [workload] -- or the trace_sha256 pairing gate (D5)
    would refuse the comparison."""
    repo_root = Path(__file__).resolve().parents[2]
    w4 = repo_root / "configs" / "experiments" / "W4"
    loaded = {}
    for path in sorted(w4.glob("*.toml")):
        cfg = load_config(path)
        assert validate_config(cfg) == [], path.name
        loaded[path.name] = cfg
    for left, right in (
        ("0002-seat-fill-max2048.toml", "0003-paged-fill-max2048.toml"),
        ("0004-seat-b8-cost.toml", "0005-paged-b8-cost.toml"),
        ("0006-seat-saturate.toml", "0007-paged-saturate.toml"),
    ):
        a, b = loaded[left], loaded[right]
        assert a["exp"]["seed"] == b["exp"]["seed"], (left, right)
        assert a["workload"] == b["workload"], (left, right)
        assert a["engine"]["impl"] != b["engine"]["impl"], (left, right)

