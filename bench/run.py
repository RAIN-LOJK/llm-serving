"""Execute one experiment run and write a self-contained run directory.

Typical use::

    python -m bench.run --config configs/experiments/W0/0001-carrier-probe.toml

The runner is intentionally conservative: it refuses to overwrite an existing run,
records environment/provenance, drops warmup rows, aggregates median/p95/CV, and leaves
the human judgement in ``notes.md`` / ``verdict.json`` instead of inventing one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from bench import env as envmod
from bench import index as indexmod
from bench import metrics as metricsmod
from bench import thresholds as thresholdmod
from bench.index import find_summary
from bench.engines import build_engine
from bench import loadgen
from bench.loadgen import PromptSet, synthetic_ids

SCHEMA_VERSION = 1
TYPES = {"probe", "microbench", "scenario", "smoke", "fill"}
IMPLS = {"dummy", "hf", "minillm", "continuous", "paged", "hybrid"}
TIMERS = {"auto", "perf_counter", "cuda_event"}
RUN_ID_RE = re.compile(r"^(?P<stage>[A-Z]\d+)-(?P<seq>\d{4})$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class ConfigError(ValueError):
    """The TOML is missing a required field or has an invalid value."""


class RunExistsError(FileExistsError):
    """Never overwrite a run; choose a new id or a different --out root."""


class DirtyTreeError(RuntimeError):
    """Official runs require a clean git tree."""


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def validate_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if cfg.get("schema") != SCHEMA_VERSION:
        errors.append(f"schema must be {SCHEMA_VERSION}, got {cfg.get('schema')!r}")

    exp = cfg.get("exp") or {}
    if not isinstance(exp, dict):
        return ["[exp] must be a table"]
    for key in ("id", "stage", "type", "slug", "title"):
        if not exp.get(key):
            errors.append(f"[exp].{key} is required")
    run_id = str(exp.get("id", ""))
    match = RUN_ID_RE.match(run_id)
    if not match:
        errors.append(f"[exp].id {run_id!r} must look like W0-0001")
    elif exp.get("stage") and match.group("stage") != exp["stage"]:
        errors.append(f"[exp].id prefix {match.group('stage')!r} != stage {exp['stage']!r}")
    if exp.get("type") not in TYPES:
        errors.append(f"[exp].type must be one of {sorted(TYPES)}")
    if exp.get("slug") and not SLUG_RE.match(str(exp["slug"])):
        errors.append("[exp].slug must be lower-case letters/numbers/hyphens")

    models = cfg.get("models") or []
    if not isinstance(models, list) or not models:
        errors.append("at least one [[models]] entry is required")
        models = []
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"models[{i}] must be a table")
            continue
        if not model.get("name"):
            errors.append(f"models[{i}].name is required")
        if exp.get("type") != "smoke":
            revision = str(model.get("revision", ""))
            if not revision:
                errors.append(f"models[{i}].revision is required (pin a commit hash)")
            elif revision.lower() == "main" or "REPLACE" in revision.upper() or revision.startswith("<"):
                errors.append(f"models[{i}].revision must be a commit hash, not {revision!r}")

    engine = cfg.get("engine") or {}
    if not isinstance(engine, dict):
        errors.append("[engine] must be a table")
        engine = {}
    if engine.get("impl") not in IMPLS:
        errors.append(f"[engine].impl must be one of {sorted(IMPLS)}")
    device = str(engine.get("device", "auto"))
    if device not in {"auto", "cuda", "mps", "cpu"} and not device.startswith("cuda:"):
        errors.append("[engine].device must be auto/cuda/mps/cpu (or cuda:N)")
    if int(engine.get("max_batch", 1)) < 1:
        errors.append("[engine].max_batch must be >= 1")
    # ---- W4 paged contract (docs/decisions.md D6) --------------------
    # Only the new `impl="paged"` path gains required fields; the closed W0/W2/W3 paths
    # are untouched (P-28: never add a mandatory key to a closed path).
    if engine.get("impl") == "paged":
        block_size = engine.get("block_size")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size not in {16, 32}:
            errors.append(
                "[engine].block_size is required and must be 16 or 32 when [engine].impl='paged' "
                f"(got {block_size!r})"
            )
        kv_blocks = engine.get("kv_blocks")
        if not isinstance(kv_blocks, int) or isinstance(kv_blocks, bool) or kv_blocks < 1:
            errors.append(
                "[engine].kv_blocks is required and must be an integer >= 1 when "
                f"[engine].impl='paged' (got {kv_blocks!r})"
            )

    workload = cfg.get("workload") or {}
    if not isinstance(workload, dict):
        errors.append("[workload] must be a table")
        workload = {}
    kind = workload.get("kind")
    if kind not in {"synthetic", "prompts"}:
        errors.append("[workload].kind must be 'synthetic' or 'prompts'")
    if int(workload.get("max_new_tokens", -1)) < 0:
        errors.append("[workload].max_new_tokens must be >= 0")
    batches = workload.get("batch_sizes", [1])
    if not isinstance(batches, list) or not batches or any(int(b) < 1 for b in batches):
        errors.append("[workload].batch_sizes must be a non-empty list of positive ints")
    if kind == "synthetic":
        contexts = workload.get("context_lengths")
        if not isinstance(contexts, list) or not contexts or any(int(c) <= 0 for c in contexts):
            errors.append("[workload].context_lengths must be a non-empty list of positive ints")
    elif kind == "prompts":
        prompts = workload.get("prompts")
        if not prompts:
            errors.append("[workload].prompts is required for kind='prompts'")
        elif not Path(str(prompts)).exists():
            errors.append(f"[workload].prompts not found: {prompts}")
    # ---- optional prompt subset (W4 E2) -----------------------------
    # `prompt_ids` narrows a frozen PromptSet to a named subset (e.g. the `long` bucket).
    # It is only meaningful for kind='prompts'; on a synthetic workload it would steer
    # nothing, so writing it there is a config error rather than a silent no-op (P-28:
    # closed paths are unchanged because they never write the key).
    prompt_ids = workload.get("prompt_ids")
    if kind == "synthetic" and "prompt_ids" in workload:
        errors.append(
            "[workload].prompt_ids is only valid when kind='prompts' "
            "(it would be silently ignored for kind='synthetic')"
        )
    if kind == "prompts" and prompt_ids is not None:
        if not isinstance(prompt_ids, list) or not prompt_ids:
            errors.append("[workload].prompt_ids must be a non-empty list of prompt ids")
        elif any(not isinstance(pid, str) or not pid for pid in prompt_ids):
            errors.append("[workload].prompt_ids entries must be non-empty strings")
        else:
            prompts = workload.get("prompts")
            if prompts and Path(str(prompts)).exists():
                try:
                    known = {p.id for p in PromptSet.load(prompts).prompts}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(
                        f"[workload].prompts could not be parsed: {prompts} "
                        f"({type(exc).__name__}: {exc})"
                    )
                else:
                    missing = [pid for pid in prompt_ids if pid not in known]
                    if missing:
                        errors.append(f"[workload].prompt_ids not found in {prompts}: {missing}")
    if workload.get("arrival", "closed") not in {"closed", "poisson"}:
        errors.append("[workload].arrival must be 'closed' or 'poisson'")

    # ---- W3 scenario contract (docs/decisions.md D5) ----------------
    # W4 (D6): `impl="paged"` is a scheduler arm too -- the paged pool replaces the seat
    # slabs, but the trace path and the `[scheduler]` knob are the same, so it joins the
    # scenario predicate. Closed W0/W2/W3 configs never write `paged`, so their results
    # are unchanged.
    policy = str((cfg.get("scheduler") or {}).get("policy", "")) if "scheduler" in cfg else ""
    scenario = workload.get("arrival") == "poisson" or engine.get("impl") in {"continuous", "paged"}
    scheduler_impls = {"continuous", "paged"}
    if "scheduler" in cfg and not isinstance(cfg.get("scheduler"), dict):
        errors.append("[scheduler] must be a table")
    if "scheduler" in cfg and engine.get("impl") not in scheduler_impls:
        # A knob that changes nothing is worse than no knob: it would be recorded in the
        # manifest as if it had steered the run.
        errors.append("[scheduler] is only valid with [engine].impl='continuous' or 'paged'")
    if engine.get("impl") in scheduler_impls and policy not in {"static", "continuous"}:
        errors.append("[scheduler].policy must be 'static' or 'continuous'")
    if workload.get("arrival") == "poisson":
        if wk_type := cfg.get("exp", {}).get("type"):
            if wk_type != "scenario":
                errors.append("[workload].arrival='poisson' requires [exp].type='scenario'")
        rate = workload.get("rate")
        if not isinstance(rate, (int, float)) or float(rate) <= 0:
            errors.append("[workload].rate is required and must be > 0 when arrival='poisson'")
        burstiness = float(workload.get("burstiness", 1.0))
        if burstiness <= 0:
            errors.append("[workload].burstiness must be > 0")
    # `requests` is only mandatory on the scenario path. Historical closed configs (all of
    # W0/W2) do not write it and must keep validating byte for byte (P-28).
    requests = workload.get("requests")
    if scenario:
        if not isinstance(requests, int) or requests < 1:
            errors.append(
                "[workload].requests must be >= 1 on the scenario path "
                "(poisson or impl='continuous'/'paged')"
            )
        elif engine.get("impl") == "continuous" and workload.get("arrival") == "closed":
            # One request per seat is the E1 shape (a batch probe, not a queue), but a closed
            # trace longer than the pool is a legitimate *capacity* probe: every request is
            # already waiting at t=0, so the seats stay full until the trace drains and
            # ``req_s`` measures sustained throughput rather than one wave's tail. Fewer
            # requests than seats would leave the pool partly idle and measure neither.
            # (Paged arms are exempt: there ``max_batch`` is a scheduling limit, not a
            # memory bound -- a fill probe with ``requests < max_batch`` is legal.)
            if int(requests) < int(engine.get("max_batch", 1)):
                errors.append(
                    "[workload].requests must be >= [engine].max_batch for a closed batch "
                    "(fewer requests than seats leaves the pool partly idle)"
                )
    if requests is not None and (not isinstance(requests, int) or requests < 1):
        errors.append("[workload].requests must be >= 1")
    # W4 `type="fill"` (D6): a capacity probe that only prefills and reserves blocks, so it
    # needs a request count even when it rides the closed path. `kind` stays whatever the
    # global rule above allows (prompts or synthetic); nothing else changes for other types.
    if cfg.get("exp", {}).get("type") == "fill":
        if not isinstance(requests, int) or requests < 1:
            errors.append("[workload].requests must be >= 1 when [exp].type='fill'")
        # Fill-only is a scheduler capability (admission + reservation): a per-request
        # arm has no trace to run it on, so the config would silently fail at run time.
        if engine.get("impl") not in scheduler_impls:
            errors.append(
                "[exp].type='fill' requires [engine].impl='continuous' or 'paged'"
            )
    if scenario and [int(b) for b in workload.get("batch_sizes", [1])] != [1]:
        errors.append(
            "[workload].batch_sizes must be [1] on the scenario path: concurrency is "
            "[engine].max_batch, not a row label (P-20)"
        )
    budgets = workload.get("budgets")
    if budgets is not None:
        if not isinstance(budgets, list) or not budgets:
            errors.append("[workload].budgets must be a non-empty list of positive ints")
        else:
            max_new = int(workload.get("max_new_tokens", 0))
            for b in budgets:
                if not isinstance(b, int) or b < 1 or b > max_new:
                    errors.append(f"[workload].budgets entries must satisfy 1 <= b <= max_new_tokens, got {b!r}")

    measure = cfg.get("measure") or {}
    if int(measure.get("warmup", -1)) < 0:
        errors.append("[measure].warmup must be >= 0")
    if int(measure.get("repeats", 0)) < 1:
        errors.append("[measure].repeats must be >= 1")
    if measure.get("timer", "auto") not in TIMERS:
        errors.append(f"[measure].timer must be one of {sorted(TIMERS)}")
    if measure.get("timer") == "cuda_event" and not device.startswith("cuda"):
        errors.append(
            f"timer='cuda_event' requires [engine].device = 'cuda' (got {device!r}); "
            "use 'auto' to resolve it from the device at run time"
        )

    accept = cfg.get("accept") or {}
    if accept.get("mode", "record_only") not in {"record_only", "threshold"}:
        errors.append("[accept].mode must be 'record_only' or 'threshold'")
    if accept.get("mode") == "threshold" and not accept.get("criteria"):
        errors.append("[accept].criteria is required when mode='threshold'")

    return errors


def resolve_run_dir(cfg: dict[str, Any], out_root: str | Path) -> Path:
    exp = cfg["exp"]
    match = RUN_ID_RE.match(str(exp["id"]))
    if not match:
        raise ConfigError(f"invalid run id {exp['id']!r}")
    return Path(out_root) / str(exp["stage"]) / f"{int(match.group('seq')):04d}-{exp['slug']}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# provenance / templates
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _select_prompts(prompt_set: PromptSet, prompt_ids: list[str] | None) -> list[Any]:
    """Narrow a frozen PromptSet to ``prompt_ids``, preserving the given order.

    Without ``prompt_ids`` the full set is returned untouched, so historical configs
    (and every consumer's byte output) are unchanged. Validation already guarantees the
    ids exist; this re-check keeps a programmatic caller from silently getting a subset
    that does not match its config.
    """
    if not prompt_ids:
        return list(prompt_set.prompts)
    by_id = {prompt.id: prompt for prompt in prompt_set.prompts}
    missing = [pid for pid in prompt_ids if pid not in by_id]
    if missing:
        raise ConfigError(f"[workload].prompt_ids not found in the prompt set: {missing}")
    return [by_id[pid] for pid in prompt_ids]


def _effective_prompt_sha256(prompt_set: PromptSet, prompt_ids: list[str] | None) -> str:
    """The workload identity a paired run must match.

    Full set -> the frozen ``PromptSet.sha256``. A subset -> a hash over the *frozen
    set's* digest plus the sorted ids, so two different subsets can never share a hash
    (nor can a subset collide with the full set). Order is deliberately excluded here:
    the trace hash already covers arrival order.
    """
    if not prompt_ids:
        return prompt_set.sha256
    payload = prompt_set.sha256 + "\n" + json.dumps(
        sorted(prompt_ids), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_info(cfg: dict[str, Any]) -> dict[str, Any] | None:
    workload = cfg.get("workload") or {}
    if workload.get("kind") != "prompts":
        return None
    prompt_set = PromptSet.load(workload["prompts"])
    prompt_ids = workload.get("prompt_ids")
    info = {
        "path": str(workload["prompts"]),
        "version": prompt_set.version,
        "sha256": prompt_set.sha256,
        "n_prompts": len(prompt_set.prompts),
    }
    if prompt_ids:
        # Keep both identities side by side: `sha256` is the workload the run actually
        # executed (the subset), `prompt_set_sha256` is the frozen file it came from.
        info["sha256"] = _effective_prompt_sha256(prompt_set, prompt_ids)
        info["n_prompts"] = len(prompt_ids)
        info["prompt_set_sha256"] = prompt_set.sha256
        info["n_prompts_in_set"] = len(prompt_set.prompts)
        info["prompt_ids"] = list(prompt_ids)
    return info


def _device_class_for(device: str, snapshot: dict[str, Any]) -> str:
    dev = (device or "").lower()
    if dev.startswith("cuda"):
        return "measure-cuda"
    if dev == "mps":
        return "dev-mps"
    if dev == "cpu":
        return "cpu"
    return metricsmod.device_class(snapshot)


def _base_manifest(
    cfg: dict[str, Any],
    config_path: Path,
    config_sha: str,
    snapshot: dict[str, Any],
    prompt_info: dict[str, Any] | None,
    command: str,
) -> dict[str, Any]:
    exp = cfg["exp"]
    engine = cfg.get("engine") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(exp["id"]),
        "stage": str(exp["stage"]),
        "type": str(exp["type"]),
        "slug": str(exp["slug"]),
        "title": str(exp["title"]),
        "hypothesis": list(exp.get("hypothesis", [])),
        "supersedes": exp.get("supersedes"),
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "duration_s": None,
        "operator": os.environ.get("MINILLM_OPERATOR") or os.environ.get("USER") or "unknown",
        "git": snapshot.get("git", {}),
        "device_class": _device_class_for(str(engine.get("device", "auto")), snapshot),
        "packages": snapshot.get("packages", {}),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "prompt": prompt_info,
        "models": copy.deepcopy(cfg.get("models", [])),
        "engine": copy.deepcopy(engine),
        "workload": copy.deepcopy(cfg.get("workload", {})),
        "measure": copy.deepcopy(cfg.get("measure", {})),
        "accept": copy.deepcopy(cfg.get("accept", {})),
        "command": command,
        "artifacts": [
            "config.toml",
            "env.json",
            "metrics.jsonl",
            "summary.json",
            "notes.md",
            "verdict.json",
            "reproduce.sh",
            "stdout.log",
        ],
    }


def _notes_template(cfg: dict[str, Any], run_dir: Path) -> str:
    exp = cfg["exp"]
    workload = cfg.get("workload") or {}
    return f"""# {exp['id']} · {exp['title']}

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `{run_dir}`）
- 类型：`{exp['type']}`；workload：`{workload.get('kind', '—')}`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

{chr(10).join(f'- `{h}`' for h in exp.get('hypothesis', [])) or '- （待填）'}

## 观察

- 关键数字（引用 summary.json）：
- 与预测的偏差：

## 结论（含“什么时候不划算”）

（待填）

## 限制与偏差

- 设备 / dtype / 样本量 / 已知噪声：

## 下一步

（待填）
"""


def _verdict_template(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "verdict": "unreviewed",
        "reviewer": None,
        "reviewed_at": None,
        "decision_id": None,
        "notes": "",
    }


def _reproduce_script(cfg: dict[str, Any], run_dir: Path, repo_root: Path) -> str:
    commit = str(cfg.get("_commit") or "unknown")
    root = shlex.quote(str(repo_root))
    return f"""#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$RUN_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT={root}
fi
cd "$REPO_ROOT"
EXPECTED={shlex.quote(commit)}
ACTUAL="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "$EXPECTED" != "unknown" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "warning: commit mismatch ($ACTUAL != $EXPECTED)" >&2
fi
exec python -m bench.run \\
  --config "$RUN_DIR/config.toml" \\
  --out "${{OUT_ROOT:-results-repro}}" \\
  --allow-dirty
"""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def _warmup_count(cfg: dict[str, Any]) -> int:
    return int((cfg.get("measure") or {}).get("warmup", 3))


def _repeat_count(cfg: dict[str, Any]) -> int:
    return int((cfg.get("measure") or {}).get("repeats", 5))


def _cases_for_model(
    cfg: dict[str, Any],
    engine: Any,
    prompt_set: PromptSet | None,
):
    workload = cfg.get("workload") or {}
    exp = cfg.get("exp") or {}
    seed = int(exp.get("seed", 0))
    batches = [int(b) for b in workload.get("batch_sizes", [1])]

    if workload.get("kind") == "synthetic":
        for ctx in workload.get("context_lengths", []):
            ctx = int(ctx)
            for batch in batches:
                for row in range(batch):
                    yield {
                        "context": ctx,
                        "bucket": None,
                        "batch": batch,
                        "request_in_batch": row,
                        "prompt_id": None,
                        "prompt_ids": synthetic_ids(ctx, engine.vocab_size, seed + ctx * 1000 + row),
                    }
    else:
        if prompt_set is None:
            raise ConfigError("prompt workload requires a loaded prompt set")
        selected = _select_prompts(prompt_set, workload.get("prompt_ids"))
        for prompt in selected:
            ids = engine.encode(prompt.text)
            for batch in batches:
                for row in range(batch):
                    yield {
                        "context": len(ids),
                        "bucket": prompt.bucket,
                        "batch": batch,
                        "request_in_batch": row,
                        "prompt_id": prompt.id,
                        "prompt_ids": ids,
                    }


def _scenario_execute(
    cfg: dict[str, Any],
    run_dir: Path,
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    engine_override: str | None,
    device_override: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """W3/W4 execution path: one scheduler over a trace per repeat (D5 / D6).

    Rows are appended in the same ``metrics.jsonl`` stream as the per-request path, but
    they carry ``row_kind`` so old consumers (``overall``/``groups``/``stability``) can go
    on reading only ``row_kind="request"``.

    ``[exp].type = "fill"`` (W4 E1) rides the same path: the adapter's ``fill`` flag
    swaps the scheduler for the fill-only loop (prefill + reserve, no decode), so the
    capacity probe still produces request/step/trace rows and a ``capacity`` block.
    """
    workload = cfg.get("workload") or {}
    prompt_set = PromptSet.load(workload["prompts"]) if workload.get("kind") == "prompts" else None
    max_new = int(workload.get("max_new_tokens", 0))
    warmup = _warmup_count(cfg)
    repeats = _repeat_count(cfg)
    requests = int(workload["requests"])
    budgets = tuple(int(b) for b in (workload.get("budgets") or [max_new]))
    fill_only = str((cfg.get("exp") or {}).get("type")) == "fill"
    rows: list[dict[str, Any]] = []
    metrics_path = run_dir / "metrics.jsonl"
    trace_summaries: list[dict[str, Any]] = []
    trace_meta: dict[str, Any] = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for model_cfg in cfg.get("models", []):
            engine_cfg = dict(cfg.get("engine") or {})
            if engine_override:
                engine_cfg["impl"] = engine_override
            if device_override:
                engine_cfg["device"] = device_override
            if "scheduler" in cfg:
                engine_cfg["_policy"] = str((cfg.get("scheduler") or {}).get("policy", "continuous"))
            # W4 E1: only the scheduler arms can run a fill probe; the runner has already
            # refused `type="fill"` on an arm without a trace path.
            if fill_only:
                engine_cfg["_fill"] = True

            engine = build_engine(model_cfg, engine_cfg)
            engine.load()
            manifest["device_class"] = _device_class_for(engine.device, snapshot)
            manifest["engine"]["resolved_device"] = engine.device
            manifest["engine"]["vocab_size"] = int(getattr(engine, "vocab_size", 0))
            manifest["engine"]["dtype"] = model_cfg.get("dtype")
            attn_backends = getattr(engine, "attn_backends", None)
            if callable(attn_backends):
                manifest["engine"]["attn"] = dict(attn_backends())

            try:
                trace = _build_trace(cfg, engine, prompt_set, requests, budgets, max_new)
                trace_meta = {
                    "arrival": workload.get("arrival", "closed"),
                    "rate": trace.rate,
                    "burstiness": trace.burstiness,
                    "requests": requests,
                    "budgets": list(budgets),
                    "seed": int((cfg.get("exp") or {}).get("seed", 0)),
                    "sha256": trace.sha256,
                    "shuffle": trace.shuffle,
                }
                manifest["scheduler"] = _scheduler_manifest(engine, fill_only=fill_only)
                manifest["trace"] = trace_meta

                for _ in range(warmup):
                    engine.run_trace(trace, repeat=-1)
                for repeat in range(repeats):
                    metricsmod.reset_peak()
                    try:
                        result = engine.run_trace(trace, repeat=repeat)
                        # Read the peak *after* the trace, before the next reset: it is the
                        # repeat's whole-run maximum, the same quantity the per-request arm
                        # writes on each row (bench/thresholds.py::_memory_ratio).
                        request_rows, step_rows = _flatten_scenario(
                            result,
                            model_cfg,
                            manifest,
                            weights_mb=engine.weights_mb,
                            peak_mem_mb=metricsmod.peak_memory_mb(),
                            driver_mem_mb=metricsmod.driver_memory_mb(),
                        )
                        trace_summaries.append(result.summary)
                    except Exception as exc:  # keep a failed repeat as rows, not a lost run
                        trace_meta.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
                        request_rows = [
                            {
                                "row_kind": "trace",
                                "repeat": repeat,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "tok_s": None,
                                "saturated": False,
                                "queue_depth_at_last_arrival": 0,
                            }
                        ]
                        step_rows = []
                    for row in request_rows + step_rows:
                        metrics_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    metrics_file.flush()
                    rows.extend(request_rows + step_rows)
            finally:
                engine.close()

    manifest["trace"] = trace_meta
    manifest["_scenario_summaries"] = trace_summaries
    return rows


def _scheduler_manifest(engine: Any, *, fill_only: bool) -> dict[str, Any]:
    """``manifest["scheduler"]``: the arm's concurrency unit, policy and pool shape.

    The paged arm's unit is a block table, not a seat row; it also records the block
    geometry so a reader can turn ``kv_blocks``/``block_size`` into a budget. The seat
    arm keeps exactly the W3 keys and values (P-28) except for the additive ``fill_only``.
    """
    impl = str(getattr(engine, "impl", ""))
    entry: dict[str, Any] = {
        "policy": engine.policy,
        "max_batch": int(engine.max_batch),
    }
    if impl == "paged":
        entry.update(
            {
                "seats": "paged_blocks",
                "prefill": "per_sequence",
                "block_size": int(getattr(engine, "block_size", 0)),
                "kv_blocks": int(getattr(engine, "kv_blocks", 0)),
                "free_rows": "not_applicable",
            }
        )
    else:
        # Same keys, same order, same values as W3.
        entry.update(
            {
                "seats": "one_kvslab_one_row_each",
                "prefill": "per_sequence",
                "free_rows": "masked_in_forward",
            }
        )
    if fill_only:
        entry["fill_only"] = True
    return entry


def _build_trace(
    cfg: dict[str, Any],
    engine: Any,
    prompt_set: PromptSet | None,
    requests: int,
    budgets: tuple[int, ...],
    max_new: int,
):
    """Tokenize the workload once, then hand already-tokenized entries to ``loadgen``."""
    workload = cfg.get("workload") or {}
    seed = int((cfg.get("exp") or {}).get("seed", 0))
    arrival = workload.get("arrival", "closed")

    if workload.get("kind") == "synthetic":
        prompt_ids_by_ctx = {
            int(ctx): synthetic_ids(int(ctx), engine.vocab_size, seed + int(ctx) * 1000)
            for ctx in workload.get("context_lengths", [])
        }
        entries = [(f"ctx{ctx}", ids) for ctx, ids in prompt_ids_by_ctx.items()]
        if not entries:
            raise ConfigError("[workload].context_lengths is empty")
        prompt_sha = None
    else:
        if prompt_set is None:
            raise ConfigError("prompt workload requires a loaded prompt set")
        selected = _select_prompts(prompt_set, workload.get("prompt_ids"))
        entries = [(prompt.id, engine.encode(prompt.text)) for prompt in selected]
        prompt_sha = _effective_prompt_sha256(prompt_set, workload.get("prompt_ids"))

    if arrival == "poisson":
        return loadgen.poisson_trace(
            entries,
            rate=float(workload["rate"]),
            n=requests,
            budgets=budgets,
            seed=seed,
            shuffle=bool(workload.get("shuffle", False)),
            burstiness=float(workload.get("burstiness", 1.0)),
        )
    return loadgen.closed_trace(
        entries,
        n=requests,
        budgets=budgets,
        seed=seed,
        shuffle=bool(workload.get("shuffle", False)),
    )


def _flatten_scenario(
    result: Any,
    model_cfg: dict[str, Any],
    manifest: dict[str, Any],
    *,
    weights_mb: float | None,
    peak_mem_mb: float | None,
    driver_mem_mb: float | None,
) -> tuple[list[dict], list[dict]]:
    """Stamp run/model provenance onto the arm's rows so one metrics file stays queryable."""
    common = {
        "run_id": manifest["run_id"],
        "model": str(model_cfg.get("name", "—")),
        # The arm that actually produced these rows (W4 adds `paged`); the manifest's
        # `[engine].impl` is the same value, but reading it off the result means an
        # `--engine` override or a future third arm cannot be mislabelled as W3's.
        "impl": str(getattr(result, "impl", None) or manifest.get("engine", {}).get("impl") or "continuous"),
        "device": manifest["engine"].get("resolved_device"),
        "device_class": manifest["device_class"],
        "dtype": model_cfg.get("dtype"),
        "policy": str(manifest.get("scheduler", {}).get("policy")),
        "max_batch": int(manifest.get("scheduler", {}).get("max_batch", 1)),
    }
    if manifest.get("scheduler", {}).get("seats") == "paged_blocks":
        common["block_size"] = int(manifest["scheduler"].get("block_size", 0))
        common["kv_blocks"] = int(manifest["scheduler"].get("kv_blocks", 0))
    # Memory fields belong to measured request outcomes only: the memory account in
    # bench/thresholds.py groups on request rows, and a step row's lifetime is not a run
    # peak (the per-request arm likewise writes these on its measured row alone).
    memory = {
        "weights_mb": weights_mb,
        "peak_mem_mb": peak_mem_mb,
        "driver_mem_mb": driver_mem_mb,
    }
    # Merged in one mapping so a row that ever grows its own memory field wins instead of
    # raising "multiple values for keyword argument".
    request_rows = [dict(common, **{**memory, **row}) for row in result.request_rows]
    # Step rows are instrumentation, not outcomes: they are "ok" by construction (a
    # failing step aborts the repeat and is recorded on the trace row instead).
    step_rows = [dict(common, ok=True, **row) for row in result.step_rows]
    trace_rows = [dict(common, **result.trace_row)]
    return request_rows + trace_rows, step_rows


def _execute(
    cfg: dict[str, Any],
    run_dir: Path,
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    engine_override: str | None,
    device_override: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    workload = cfg.get("workload") or {}
    engine_impl = str((cfg.get("engine") or {}).get("impl") or "")
    exp_type = str((cfg.get("exp") or {}).get("type") or "")
    # W4: `type="fill"` is a scheduler capacity probe (prefill + reserve, no decode), so
    # it rides the scenario path even on a closed seat arm. `paged` joins the W3 predicate;
    # both are additive -- closed W0/W2/W3 configs never write either.
    if (
        workload.get("arrival") == "poisson"
        or (engine_override or engine_impl) in {"continuous", "paged"}
        or exp_type == "fill"
    ):
        return _scenario_execute(
            cfg, run_dir, manifest, snapshot, engine_override, device_override, limit
        )
    if workload.get("arrival", "closed") != "closed":
        raise NotImplementedError(f"arrival={workload.get('arrival')!r} is not implemented")

    prompt_set = PromptSet.load(workload["prompts"]) if workload.get("kind") == "prompts" else None
    max_new_tokens = int(workload.get("max_new_tokens", 0))
    warmup = _warmup_count(cfg)
    repeats = _repeat_count(cfg)
    rows: list[dict[str, Any]] = []
    metrics_path = run_dir / "metrics.jsonl"

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for model_cfg in cfg.get("models", []):
            engine_cfg = dict(cfg.get("engine") or {})
            if engine_override:
                engine_cfg["impl"] = engine_override
            if device_override:
                engine_cfg["device"] = device_override
            # `[scheduler]` is its own table (it is not an engine property), so hand the
            # one field the engine factory needs through a private key rather than
            # flattening a second config section into `[engine]`.
            if "scheduler" in cfg:
                engine_cfg["_policy"] = str((cfg.get("scheduler") or {}).get("policy", "continuous"))

            engine = build_engine(model_cfg, engine_cfg)
            engine.load()
            manifest["device_class"] = _device_class_for(engine.device, snapshot)
            manifest["engine"]["resolved_device"] = engine.device
            manifest["engine"]["vocab_size"] = int(getattr(engine, "vocab_size", 0))
            manifest["engine"]["dtype"] = model_cfg.get("dtype")
            # Only the HF arm exposes an attention-backend field; the in-house decoder is
            # SDPA-only and deliberately has no backend field to record.
            attn_impl = getattr(engine, "attn_implementation", None)
            if attn_impl is not None:
                manifest["engine"]["attn_implementation"] = attn_impl
            # Which attention backends actually ran. The value is a constant now (D4), but
            # it is still provenance: a future kernel swap must show up in the run, not in
            # a comment, or two runs would be compared across different engines.
            attn_backends = getattr(engine, "attn_backends", None)
            if callable(attn_backends):
                manifest["engine"]["attn"] = dict(attn_backends())

            timer = metricsmod.make_timer(engine.device, str((cfg.get("measure") or {}).get("timer", "auto")))
            cases = list(_cases_for_model(cfg, engine, prompt_set))
            if limit is not None:
                cases = cases[:limit]

            try:
                for case in cases:
                    for _ in range(warmup):
                        engine.generate(case["prompt_ids"], max_new_tokens)
                    for repeat in range(repeats):
                        metricsmod.reset_peak()
                        timer.start()
                        try:
                            result = engine.generate(case["prompt_ids"], max_new_tokens)
                            e2e_ms = timer.stop()
                            row = {
                                "run_id": manifest["run_id"],
                                "model": str(model_cfg.get("name", "—")),
                                "impl": engine.impl,
                                "device": engine.device,
                                "device_class": manifest["device_class"],
                                "dtype": model_cfg.get("dtype"),
                                "phase": "measured",
                                "repeat": repeat,
                                "context": case["context"],
                                "bucket": case.get("bucket"),
                                "batch": case["batch"],
                                "request_in_batch": case["request_in_batch"],
                                "prompt_id": case["prompt_id"],
                                "prompt_tokens": len(case["prompt_ids"]),
                                "max_new_tokens": max_new_tokens,
                                "out_tokens": len(result.token_ids),
                                "ttft_ms": result.ttft_ms,
                                "tpot_ms": result.tpot_ms,
                                "prefill_ms": result.prefill_ms,
                                "decode_ms": result.decode_ms,
                                "e2e_ms": e2e_ms,
                                "peak_mem_mb": result.peak_mem_mb
                                if result.peak_mem_mb is not None
                                else metricsmod.peak_memory_mb(),
                                "driver_mem_mb": metricsmod.driver_memory_mb(),
                                "weights_mb": result.weights_mb,
                                "kv_mb": result.kv_mb,
                                "state_mb": result.state_mb,
                                "conv_mb": result.conv_mb,
                                "cache_mb": result.cache_mb,
                                "cache_prefill_mb": result.cache_prefill_mb,
                                "mem_delta_mb": result.mem_delta_mb,
                                "ok": True,
                                "error": None,
                            }
                        except Exception as exc:  # keep OOM/failure as a row, not a lost run
                            row = {
                                "run_id": manifest["run_id"],
                                "model": str(model_cfg.get("name", "—")),
                                "impl": getattr(engine, "impl", "—"),
                                "device": getattr(engine, "device", "—"),
                                "device_class": manifest["device_class"],
                                "dtype": model_cfg.get("dtype"),
                                "phase": "measured",
                                "repeat": repeat,
                                "context": case["context"],
                                "bucket": case.get("bucket"),
                                "batch": case["batch"],
                                "request_in_batch": case["request_in_batch"],
                                "prompt_id": case["prompt_id"],
                                "prompt_tokens": len(case["prompt_ids"]),
                                "max_new_tokens": max_new_tokens,
                                "out_tokens": 0,
                                "ttft_ms": None,
                                "tpot_ms": None,
                                "prefill_ms": None,
                                "decode_ms": None,
                                "e2e_ms": timer.stop(),
                                "peak_mem_mb": metricsmod.peak_memory_mb(),
                                "driver_mem_mb": metricsmod.driver_memory_mb(),
                                "weights_mb": None,
                                "kv_mb": None,
                                "state_mb": None,
                                "conv_mb": None,
                                "cache_mb": None,
                                "cache_prefill_mb": None,
                                "mem_delta_mb": None,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            try:
                                if __import__("torch").cuda.is_available():
                                    __import__("torch").cuda.empty_cache()
                            except Exception:
                                pass
                        metrics_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        metrics_file.flush()
                        rows.append(row)
            finally:
                engine.close()

    return rows


def _summary(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    out_root: str | Path = "results",
) -> dict[str, Any]:
    metric_names = (
        "ttft_ms",
        "tpot_ms",
        "prefill_ms",
        "decode_ms",
        "e2e_ms",
        "peak_mem_mb",
        "driver_mem_mb",
        "weights_mb",
        "kv_mb",
        "state_mb",
        "conv_mb",
        "cache_mb",
        "cache_prefill_mb",
        "mem_delta_mb",
    )
    # Scenario runs write three row kinds into one stream. `overall`/`groups`/`stability`
    # keep the W2 meaning exactly by reading request rows only; the trace-level numbers go
    # into their own blocks (docs/decisions.md D5), and step rows feed `steps.by_active`.
    request_rows = [r for r in rows if r.get("row_kind", "request") == "request"]
    trace_rows = [r for r in rows if r.get("row_kind") == "trace"]
    step_rows = [r for r in rows if r.get("row_kind") == "step"]
    rows_for_stats = request_rows or rows

    overall = {name: metricsmod.summarize([r.get(name) for r in rows_for_stats]) for name in metric_names}
    stability = metricsmod.repeat_stability(rows_for_stats, ("ttft_ms", "tpot_ms", "e2e_ms"))
    by_prompt = metricsmod.per_prompt_summary(rows_for_stats, ("ttft_ms", "tpot_ms", "out_tokens"))

    grouped = request_rows or [r for r in rows if r.get("row_kind") is None]
    group_keys = sorted(
        {(r.get("model"), int(r.get("context") or 0), int(r.get("batch") or 1)) for r in grouped},
        key=lambda key: (str(key[0]), key[1], key[2]),
    )
    groups = []
    for model, context, batch in group_keys:
        subset = [
            r
            for r in grouped
            if (r.get("model"), int(r.get("context") or 0), int(r.get("batch") or 1))
            == (model, context, batch)
        ]
        groups.append(
            {
                "model": model,
                "context": context,
                "batch": batch,
                "n": len(subset),
                "n_errors": sum(1 for r in subset if not r.get("ok", True)),
                **{name: metricsmod.summarize([r.get(name) for r in subset]) for name in metric_names},
            }
        )

    warnings: list[str] = []
    errors = [r for r in rows if not r.get("ok", True)]
    if errors:
        warnings.append(f"{len(errors)}/{len(rows)} rows failed (see metrics.jsonl)")
    for bucket, entry in stability.items():
        for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
            cv = entry.get(f"{name}_cv_median")
            if cv is not None and cv > 0.10:
                warnings.append(
                    f"{name} CV={cv:.3f} > 0.10 in bucket {bucket}; rerun the whole config before citing"
                )
    if manifest.get("git", {}).get("dirty"):
        warnings.append("git tree was dirty; this run is exploratory and must not be cited")
    if manifest.get("device_class") == "dev-mps" and manifest.get("type") != "smoke":
        warnings.append("dev-mps numbers are development-only; official numbers belong on measure-cuda")
    measure = cfg.get("measure") or {}
    if int(measure.get("warmup", 3)) != 3 or int(measure.get("repeats", 5)) != 5:
        warnings.append("measure protocol deviates from warmup=3/repeats=5; explain in notes.md")
    if manifest.get("type") in {"microbench", "scenario"} and not cfg.get("baseline", {}).get("run_id"):
        warnings.append("no paired baseline run_id; mechanism conclusions are incomplete")

    attn = (manifest.get("engine") or {}).get("attn") or {}
    protocol = {
        "warmup": _warmup_count(cfg),
        "repeats": _repeat_count(cfg),
        "timer": (cfg.get("measure") or {}).get("timer", "auto"),
        "max_new_tokens": int((cfg.get("workload") or {}).get("max_new_tokens", 0)),
        "prompt_sha256": (manifest.get("prompt") or {}).get("sha256"),
        "dtype": (manifest.get("engine") or {}).get("dtype"),
        "device_class": manifest.get("device_class"),
        # Constant provenance of the frozen attention policy (D4): prefill "sdpa",
        # decode "hand_baddbmm" for this engine. Absent for the HF/dummy arms, which
        # is why the pairing check treats "both None" as equal (bench/thresholds.py).
        "attn_prefill": attn.get("prefill"),
        "attn_decode": attn.get("decode"),
    }
    scenario_blocks = _scenario_blocks(rows, manifest, trace_rows, request_rows, step_rows)
    if scenario_blocks is not None:
        protocol.update(scenario_blocks["protocol"])
        # W4 C1 pairs on the workload identity. The trace hash already covers it, but
        # it is recorded under `capacity.workload_sha256` as well so a summary whose
        # `protocol` block is trimmed still carries the pairing key (D6).
        capacity = scenario_blocks.get("capacity") or {}
        if capacity and capacity.get("workload_sha256") is not None:
            protocol.setdefault("workload_sha256", capacity["workload_sha256"])

    accept = cfg.get("accept") or {}
    baseline_id = str((cfg.get("baseline") or {}).get("run_id") or "")
    baseline_summary = find_summary(baseline_id, out_root) if baseline_id else None
    if baseline_id and baseline_summary is None:
        warnings.append(f"baseline {baseline_id} has no summary.json under {out_root}; ratio criteria stay pending")
    baseline_info = {"run_id": baseline_id or None, "found": baseline_summary is not None}

    if accept.get("mode", "record_only") == "record_only":
        acceptance = [{"name": "record_only", "status": "recorded"}]
    else:
        evaluated = {
            "overall": overall,
            "groups": groups,
            "stability": stability,
            "by_prompt": by_prompt,
            "protocol": protocol,
        }
        if scenario_blocks is not None:
            evaluated.update(
                {
                    "throughput": scenario_blocks["throughput"],
                    "latency": scenario_blocks["latency"],
                    "steps": scenario_blocks["steps"],
                    "consistency": scenario_blocks["consistency"],
                    "memory": scenario_blocks["memory"],
                    # W4 C1/C2 read `capacity`; absent -> those criteria stay pending
                    # (the evaluator's own behaviour, not an error).
                    "capacity": scenario_blocks.get("capacity"),
                }
            )
        acceptance = thresholdmod.evaluate(
            accept.get("criteria", []),
            evaluated,
            thresholdmod.load(cfg.get("thresholds")),
            baseline=baseline_summary,
        )

    key_name, key_value = metricsmod.key_metric({"overall": overall})
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "generated_at": _utc_now(),
        "type": manifest.get("type"),
        "device_class": manifest.get("device_class"),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "overall": overall,
        "groups": groups,
        "stability": stability,
        "by_prompt": by_prompt,
        "baseline": baseline_info,
        "warnings": warnings,
        "acceptance": acceptance,
        "key_metric": {"name": key_name, "value": key_value},
        "protocol": protocol,
    }
    if scenario_blocks is not None:
        # D5 blocks, plus the tok/s key metric a scenario index row should show.
        result.update(
            {
                "throughput": scenario_blocks["throughput"],
                "latency": scenario_blocks["latency"],
                "steps": scenario_blocks["steps"],
                "consistency": scenario_blocks["consistency"],
                "memory": scenario_blocks["memory"],
            }
        )
        capacity = scenario_blocks.get("capacity")
        if capacity is not None:
            result["capacity"] = capacity
        fill_only = bool((manifest.get("scheduler") or {}).get("fill_only"))
        if fill_only and capacity and capacity.get("peak_concurrency") is not None:
            # A fill run generates nothing, so tok/s is 0 or meaningless; the E1 index
            # row should show the capacity number instead.
            result["key_metric"] = {
                "name": "peak_concurrency",
                "value": float(capacity["peak_concurrency"]),
            }
        else:
            tok_s = (scenario_blocks["throughput"].get("tok_s") or {}).get("median")
            if tok_s is not None:
                result["key_metric"] = {"name": "tok_s", "value": float(tok_s)}
    return result


def _scenario_blocks(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate the three row kinds with ``bench.scenario``; ``None`` for W2-style runs."""
    if not trace_rows:
        return None
    from bench import scenario as scenariomod

    trace_meta = manifest.get("trace") or {}
    blocks = scenariomod.aggregate(
        request_rows,
        step_rows,
        trace_rows,
        budgets=trace_meta.get("budgets"),
        max_batch=int((manifest.get("scheduler") or {}).get("max_batch", 1)),
    )
    # The first trace's hash is what a paired arm must match (D5): repeats of one run
    # build the same trace, so a spread here would mean the run itself was inconsistent.
    hashes = sorted({str(r.get("trace_sha256")) for r in trace_rows if r.get("trace_sha256")})
    blocks["protocol"] = {
        "trace_sha256": hashes[0] if len(hashes) == 1 else None,
        "max_batch": int((manifest.get("scheduler") or {}).get("max_batch", 1)),
    }
    # W4 (D6): the paged arm's scheduler unit and block geometry are pairing provenance.
    # The seat arm never writes these keys, so its protocol block is unchanged.
    scheduler = manifest.get("scheduler") or {}
    if scheduler.get("seats") == "paged_blocks":
        blocks["protocol"].update(
            {
                "block_size": scheduler.get("block_size"),
                "kv_blocks": scheduler.get("kv_blocks"),
            }
        )
    if scheduler.get("fill_only"):
        blocks["protocol"]["fill_only"] = True
    blocks["consistency"].update(
        {
            "trace_sha256_consistent": len(hashes) <= 1,
            "n_trace_hashes": len(hashes),
        }
    )
    # The adapter's summary is the only place a fill/closed run's allocator capacity is
    # known: rows alone cannot tell "6 admitted" from "6 seats that happened to be busy".
    # Repeats of one run are deterministic for a fill probe; if they ever disagree, the
    # capacity criterion must not silently pick one, so it is left pending (None).
    capacities = [
        summary.get("capacity")
        for summary in manifest.get("_scenario_summaries") or []
        if isinstance(summary, Mapping) and summary.get("capacity") is not None
    ]
    blocks["capacity"] = _merged_capacity(capacities) if capacities else None
    return blocks


def _merged_capacity(summaries: list[Any]) -> dict[str, Any] | None:
    """One ``capacity`` block from the repeats' summaries.

    Numbers are averaged over the repeats with the median (a fill probe is deterministic,
    so this equals the single value; a scenario run's dynamic peak genuinely varies and
    a median is the honest summary). Identity/provenance fields must agree exactly; a
    field that disagrees is reported as ``None`` with ``capacity_inconsistent=True``
    rather than an arbitrary pick.
    """
    import statistics

    merged: dict[str, Any] = {}
    keys = {key for summary in summaries for key in summary}
    for key in sorted(keys):
        values = [summary.get(key) for summary in summaries]
        first = values[0]
        if all(value == first for value in values[1:]):
            # Identity/provenance and deterministic fill numbers: keep the exact value
            # (an int stays an int so `block_size`/`kv_blocks` read as block counts).
            merged[key] = first
            continue
        numbers = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(numbers) == len(values):
            merged[key] = statistics.median(numbers)
        else:
            merged[key] = None
            merged["capacity_inconsistent"] = True
    return merged


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_config(
    config_path: str | Path,
    *,
    out_root: str | Path = "results",
    allow_dirty: bool = False,
    dry_run: bool = False,
    engine_override: str | None = None,
    device_override: str | None = None,
    limit: int | None = None,
    write_index: bool = True,
) -> Path | dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    cfg = load_config(config_path)
    errors = validate_config(cfg)
    if errors:
        raise ConfigError("invalid config:\n" + "\n".join(f"- {err}" for err in errors))

    snapshot = envmod.snapshot()
    prompt_info = _prompt_info(cfg)
    run_dir = resolve_run_dir(cfg, out_root)
    command_parts = [sys.executable, "-m", "bench.run", "--config", str(config_path)]
    if allow_dirty:
        command_parts.append("--allow-dirty")
    if engine_override:
        command_parts.extend(["--engine", engine_override])
    if device_override:
        command_parts.extend(["--device", device_override])
    if limit is not None:
        command_parts.extend(["--limit", str(limit)])
    if str(out_root) != "results":
        command_parts.extend(["--out", str(out_root)])
    command = shlex.join(command_parts)

    if dry_run:
        return {
            "run_id": cfg["exp"]["id"],
            "run_dir": str(run_dir),
            "command": command,
            "models": cfg.get("models", []),
            "workload": cfg.get("workload", {}),
            "engine": cfg.get("engine", {}),
            "measure": cfg.get("measure", {}),
            "prompt": prompt_info,
            "git": snapshot.get("git", {}),
            "device_class": _device_class_for(str((cfg.get("engine") or {}).get("device", "auto")), snapshot),
        }

    if run_dir.exists():
        raise RunExistsError(f"{run_dir} already exists; runs are immutable. Choose a new id/out root.")
    if not allow_dirty and cfg["exp"]["type"] != "smoke":
        dirty = snapshot.get("git", {}).get("dirty")
        if dirty is not False:
            raise DirtyTreeError(
                "official runs require a clean git tree (dirty=%r). Commit first, or use "
                "--allow-dirty for an exploratory run." % (dirty,)
            )

    run_dir.mkdir(parents=True, exist_ok=False)
    config_sha = _sha256(config_path)
    shutil.copy2(config_path, run_dir / "config.toml")
    write_json(run_dir / "env.json", snapshot)

    repo_root = Path(snapshot.get("cwd") or Path.cwd())
    cfg_with_commit = dict(cfg)
    cfg_with_commit["_commit"] = snapshot.get("git", {}).get("commit")
    manifest = _base_manifest(cfg, config_path, config_sha, snapshot, prompt_info, command)
    manifest["device_class"] = _device_class_for(
        device_override or str((cfg.get("engine") or {}).get("device", "auto")), snapshot
    )
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "notes.md").write_text(_notes_template(cfg, run_dir), encoding="utf-8")
    write_json(run_dir / "verdict.json", _verdict_template(str(cfg["exp"]["id"])))
    reproduce = run_dir / "reproduce.sh"
    reproduce.write_text(_reproduce_script(cfg_with_commit, run_dir, repo_root), encoding="utf-8")
    reproduce.chmod(0o755)
    (run_dir / "stdout.log").write_text("", encoding="utf-8")

    started = metricsmod.now()
    try:
        rows = _execute(cfg, run_dir, manifest, snapshot, engine_override, device_override, limit)
        summary = _summary(cfg, manifest, rows, out_root)
        write_json(run_dir / "summary.json", summary)

        duration = metricsmod.elapsed_ms(started) / 1000.0
        manifest["status"] = "completed" if rows and any(r.get("ok", True) for r in rows) else "failed"
        manifest["finished_at"] = _utc_now()
        manifest["duration_s"] = round(duration, 3)
        manifest["n_rows"] = len(rows)
        manifest["n_errors"] = sum(1 for r in rows if not r.get("ok", True))
        manifest["summary"] = "summary.json"
        write_json(run_dir / "manifest.json", manifest)

        log_lines = [
            f"run_id: {manifest['run_id']}",
            f"status: {manifest['status']}",
            f"device_class: {manifest['device_class']}",
            f"rows: {len(rows)} (errors: {manifest['n_errors']})",
            f"summary: {run_dir / 'summary.json'}",
        ]
        (run_dir / "stdout.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if write_index:
            try:
                indexmod.build_index(out_root)
            except Exception as exc:  # index is derived; don't mask a completed run
                print(f"warning: index update failed: {exc}", file=sys.stderr)
        return run_dir
    except KeyboardInterrupt:
        manifest["status"] = "aborted"
        manifest["finished_at"] = _utc_now()
        manifest["duration_s"] = round(metricsmod.elapsed_ms(started) / 1000.0, 3)
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "error.txt").write_text("aborted by KeyboardInterrupt\n", encoding="utf-8")
        raise
    except Exception:
        manifest["status"] = "failed"
        manifest["finished_at"] = _utc_now()
        manifest["duration_s"] = round(metricsmod.elapsed_ms(started) / 1000.0, 3)
        manifest["error"] = traceback.format_exc()
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment TOML")
    parser.add_argument("--out", default="results", help="results root (default: results)")
    parser.add_argument("--allow-dirty", action="store_true", help="allow exploratory runs on a dirty tree")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan; write nothing")
    parser.add_argument("--engine", help="override [engine].impl (e.g. dummy for smoke)")
    parser.add_argument("--device", help="override [engine].device")
    parser.add_argument("--limit", type=int, help="limit measured cases per model (smoke only)")
    parser.add_argument("--no-index", action="store_true", help="do not regenerate results/INDEX.md")
    args = parser.parse_args(argv)

    try:
        result = run_config(
            args.config,
            out_root=args.out,
            allow_dirty=args.allow_dirty,
            dry_run=args.dry_run,
            engine_override=args.engine,
            device_override=args.device,
            limit=args.limit,
            write_index=not args.no_index,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"run written: {result}")
        print(f"summary: {result / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
