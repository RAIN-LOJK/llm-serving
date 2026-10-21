"""Execute one experiment run and write a self-contained run directory.

Typical use::

    .venv/bin/python -m bench.run --config configs/experiments/W0/0001-carrier-probe.toml

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
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from bench import env as envmod
from bench import index as indexmod
from bench import metrics as metricsmod
from bench.engines import build_engine
from bench.loadgen import PromptSet, synthetic_ids

SCHEMA_VERSION = 1
TYPES = {"probe", "microbench", "scenario", "smoke"}
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
    if workload.get("arrival", "closed") not in {"closed", "poisson"}:
        errors.append("[workload].arrival must be 'closed' or 'poisson'")
    if workload.get("arrival") == "poisson":
        errors.append("arrival='poisson' is not implemented until W3; use 'closed' for now")

    measure = cfg.get("measure") or {}
    if int(measure.get("warmup", -1)) < 0:
        errors.append("[measure].warmup must be >= 0")
    if int(measure.get("repeats", 0)) < 1:
        errors.append("[measure].repeats must be >= 1")
    if measure.get("timer", "auto") not in TIMERS:
        errors.append(f"[measure].timer must be one of {sorted(TIMERS)}")
    if measure.get("timer") == "cuda_event":
        errors.append("timer='cuda_event' is not wired yet; use 'auto' or 'perf_counter' until W2")

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


def _prompt_info(cfg: dict[str, Any]) -> dict[str, Any] | None:
    workload = cfg.get("workload") or {}
    if workload.get("kind") != "prompts":
        return None
    prompt_set = PromptSet.load(workload["prompts"])
    return {
        "path": str(workload["prompts"]),
        "version": prompt_set.version,
        "sha256": prompt_set.sha256,
        "n_prompts": len(prompt_set.prompts),
    }


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
exec .venv/bin/python -m bench.run \\
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
                        "batch": batch,
                        "request_in_batch": row,
                        "prompt_id": None,
                        "prompt_ids": synthetic_ids(ctx, engine.vocab_size, seed + ctx * 1000 + row),
                    }
    else:
        if prompt_set is None:
            raise ConfigError("prompt workload requires a loaded prompt set")
        for prompt in prompt_set.prompts:
            ids = engine.encode(prompt.text)
            for batch in batches:
                for row in range(batch):
                    yield {
                        "context": len(ids),
                        "batch": batch,
                        "request_in_batch": row,
                        "prompt_id": prompt.id,
                        "prompt_ids": ids,
                    }


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
    if workload.get("arrival", "closed") != "closed":
        raise NotImplementedError("open-loop arrival is a W3 feature; use arrival='closed'")

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

            engine = build_engine(model_cfg, engine_cfg)
            engine.load()
            manifest["device_class"] = _device_class_for(engine.device, snapshot)
            manifest["engine"]["resolved_device"] = engine.device
            manifest["engine"]["vocab_size"] = int(getattr(engine, "vocab_size", 0))

            cases = list(_cases_for_model(cfg, engine, prompt_set))
            if limit is not None:
                cases = cases[:limit]

            try:
                for case in cases:
                    for _ in range(warmup):
                        engine.generate(case["prompt_ids"], max_new_tokens)
                    for repeat in range(repeats):
                        metricsmod.reset_peak()
                        start = metricsmod.now()
                        try:
                            result = engine.generate(case["prompt_ids"], max_new_tokens)
                            e2e_ms = metricsmod.elapsed_ms(start)
                            row = {
                                "run_id": manifest["run_id"],
                                "model": str(model_cfg.get("name", "—")),
                                "impl": engine.impl,
                                "device": engine.device,
                                "device_class": manifest["device_class"],
                                "phase": "measured",
                                "repeat": repeat,
                                "context": case["context"],
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
                                "phase": "measured",
                                "repeat": repeat,
                                "context": case["context"],
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
                                "e2e_ms": metricsmod.elapsed_ms(start),
                                "peak_mem_mb": metricsmod.peak_memory_mb(),
                                "driver_mem_mb": metricsmod.driver_memory_mb(),
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
) -> dict[str, Any]:
    metric_names = ("ttft_ms", "tpot_ms", "prefill_ms", "decode_ms", "e2e_ms", "peak_mem_mb", "driver_mem_mb")
    overall = {name: metricsmod.summarize([r.get(name) for r in rows]) for name in metric_names}

    group_keys = sorted(
        {(r["model"], int(r["context"]), int(r["batch"])) for r in rows},
        key=lambda key: (key[0], key[1], key[2]),
    )
    groups = []
    for model, context, batch in group_keys:
        subset = [r for r in rows if (r["model"], int(r["context"]), int(r["batch"])) == (model, context, batch)]
        groups.append(
            {
                "model": model,
                "context": context,
                "batch": batch,
                "n": len(subset),
                "n_errors": sum(1 for r in subset if not r["ok"]),
                **{name: metricsmod.summarize([r.get(name) for r in subset]) for name in metric_names},
            }
        )

    warnings: list[str] = []
    errors = [r for r in rows if not r["ok"]]
    if errors:
        warnings.append(f"{len(errors)}/{len(rows)} rows failed (see metrics.jsonl)")
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        agg = overall[name]
        if agg["n"] >= 3 and agg["cv"] is not None and agg["cv"] > 0.10:
            warnings.append(f"{name} CV={agg['cv']:.3f} > 0.10; rerun the whole config before citing")
    if manifest.get("git", {}).get("dirty"):
        warnings.append("git tree was dirty; this run is exploratory and must not be cited")
    if manifest.get("device_class") == "dev-mps" and manifest.get("type") != "smoke":
        warnings.append("dev-mps numbers are development-only; official numbers belong on measure-cuda")
    measure = cfg.get("measure") or {}
    if int(measure.get("warmup", 3)) != 3 or int(measure.get("repeats", 5)) != 5:
        warnings.append("measure protocol deviates from warmup=3/repeats=5; explain in notes.md")
    if manifest.get("type") in {"microbench", "scenario"} and not cfg.get("baseline", {}).get("run_id"):
        warnings.append("no paired baseline run_id; mechanism conclusions are incomplete")

    accept = cfg.get("accept") or {}
    if accept.get("mode", "record_only") == "record_only":
        acceptance = [{"name": "record_only", "status": "recorded"}]
    else:
        acceptance = [
            {"name": str(name), "status": "pending", "reason": "threshold evaluation lands in W2"}
            for name in accept.get("criteria", [])
        ]

    key_name, key_value = metricsmod.key_metric({"overall": overall})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "generated_at": _utc_now(),
        "type": manifest.get("type"),
        "device_class": manifest.get("device_class"),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "overall": overall,
        "groups": groups,
        "warnings": warnings,
        "acceptance": acceptance,
        "key_metric": {"name": key_name, "value": key_value},
        "protocol": {
            "warmup": _warmup_count(cfg),
            "repeats": _repeat_count(cfg),
            "timer": (cfg.get("measure") or {}).get("timer", "auto"),
            "max_new_tokens": int((cfg.get("workload") or {}).get("max_new_tokens", 0)),
        },
    }


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
        summary = _summary(cfg, manifest, rows)
        write_json(run_dir / "summary.json", summary)

        duration = metricsmod.elapsed_ms(started) / 1000.0
        manifest["status"] = "completed" if rows and any(r["ok"] for r in rows) else "failed"
        manifest["finished_at"] = _utc_now()
        manifest["duration_s"] = round(duration, 3)
        manifest["n_rows"] = len(rows)
        manifest["n_errors"] = sum(1 for r in rows if not r["ok"])
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
