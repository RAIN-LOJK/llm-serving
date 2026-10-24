"""Run discovery, validation, and the generated results index.

``INDEX.md`` is derived data. Never edit it by hand; run ``scripts/exp_index.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

RUN_DIR_RE = re.compile(r"^(?P<seq>\d{4})-(?P<slug>[a-z0-9][a-z0-9-]*)$")
RUN_ID_RE = re.compile(r"^(?P<stage>[A-Z]\d+)-(?P<seq>\d{4})$")
VERDICTS = {"unreviewed", "pass", "fail", "inconclusive", "invalid"}
STATUSES = {"running", "completed", "failed", "aborted"}

REQUIRED = (
    "config.toml",
    "manifest.json",
    "env.json",
    "notes.md",
    "verdict.json",
    "reproduce.sh",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_run_dirs(root: str | Path = "results"):
    root = Path(root)
    if not root.exists():
        return
    for stage_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for run_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            if (run_dir / "manifest.json").exists():
                yield run_dir


def find_summary(run_id: str, root: str | Path = "results") -> dict[str, Any] | None:
    """Locate another run's ``summary.json`` by run id (used for paired baselines)."""
    for run_dir in iter_run_dirs(root):
        path = run_dir / "summary.json"
        if not path.exists():
            continue
        try:
            summary = _read_json(path)
        except json.JSONDecodeError:
            continue
        if summary.get("run_id") == run_id:
            return summary
    return None


def load_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    manifest: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    verdict: dict[str, Any] = {}
    errors: list[str] = []

    for name, target in (("manifest.json", "manifest"), ("summary.json", "summary"), ("verdict.json", "verdict")):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            value = _read_json(path)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: invalid JSON ({exc})")
            continue
        if target == "manifest":
            manifest = value
        elif target == "summary":
            summary = value
        else:
            verdict = value

    return {
        "dir": run_dir,
        "manifest": manifest,
        "summary": summary,
        "verdict": verdict,
        "missing": missing,
        "errors": errors,
    }


def check_root(root: str | Path = "results") -> list[str]:
    """Return a list of contract violations. Empty list means the tree is valid."""
    root = Path(root)
    errors: list[str] = []
    if not root.exists():
        return [f"{root}: results root does not exist"]

    for run_dir in iter_run_dirs(root):
        rel = run_dir.relative_to(root)
        loaded = load_run(run_dir)
        errors.extend(f"{rel}: {err}" for err in loaded["errors"])
        errors.extend(f"{rel}: missing {name}" for name in loaded["missing"])
        manifest = loaded["manifest"]
        if not manifest:
            continue

        run_id = str(manifest.get("run_id", ""))
        match = RUN_ID_RE.match(run_id)
        if not match:
            errors.append(f"{rel}: invalid run_id {run_id!r} (expected e.g. W0-0001)")
        else:
            stage = str(manifest.get("stage", ""))
            if match.group("stage") != stage:
                errors.append(f"{rel}: run_id stage {match.group('stage')!r} != manifest stage {stage!r}")
            if run_dir.parent.name != stage:
                errors.append(f"{rel}: directory stage {run_dir.parent.name!r} != manifest stage {stage!r}")
            if run_dir.name[:4] != match.group("seq"):
                errors.append(f"{rel}: directory sequence {run_dir.name[:4]!r} != run_id sequence {match.group('seq')!r}")

        if manifest.get("status") not in STATUSES:
            errors.append(f"{rel}: invalid status {manifest.get('status')!r}")
        if manifest.get("status") == "completed":
            for name in ("summary.json", "metrics.jsonl"):
                if not (run_dir / name).exists():
                    errors.append(f"{rel}: completed run is missing {name}")

        verdict = loaded["verdict"].get("verdict")
        if verdict is not None and verdict not in VERDICTS:
            errors.append(f"{rel}: invalid verdict {verdict!r}")

        config_path = run_dir / "config.toml"
        expected = manifest.get("config_sha256")
        if config_path.exists() and expected and _sha256(config_path) != expected:
            errors.append(f"{rel}: config.toml changed after the run (sha256 mismatch)")

    return errors


def _first_model(manifest: dict[str, Any]) -> str:
    models = manifest.get("models") or []
    if not models:
        return "—"
    names = [str(m.get("name", "—")) for m in models]
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


def _key_metric(summary: dict[str, Any]) -> str:
    key = summary.get("key_metric") or {}
    name = key.get("name")
    value = key.get("value")
    if name is None or value is None:
        return "—"
    try:
        return f"{name} {float(value):.3g}"
    except (TypeError, ValueError):
        return f"{name} {value}"


def _row(loaded: dict[str, Any]) -> dict[str, str]:
    run_dir: Path = loaded["dir"]
    manifest = loaded["manifest"]
    summary = loaded["summary"]
    verdict = loaded["verdict"]
    git = manifest.get("git") or {}
    commit = str(git.get("commit") or "—")[:7]
    if git.get("dirty"):
        commit += "*"
    started = str(manifest.get("started_at") or "")
    engine = manifest.get("engine") or {}
    return {
        "run": str(manifest.get("run_id") or run_dir.name),
        "stage": str(manifest.get("stage") or run_dir.parent.name),
        "type": str(manifest.get("type") or "—"),
        "title": str(manifest.get("title") or "—"),
        "model": _first_model(manifest),
        "engine": str(engine.get("impl") or "—"),
        "key_metric": _key_metric(summary),
        "verdict": str(verdict.get("verdict") or "unreviewed"),
        "status": str(manifest.get("status") or "—"),
        "date": started[:10] if started else "—",
        "commit": commit,
        "path": str(run_dir),
    }


def build_index(root: str | Path = "results") -> Path:
    """Regenerate ``INDEX.md`` and ``index.csv`` under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [_row(load_run(run_dir)) for run_dir in iter_run_dirs(root)]

    columns = ["run", "stage", "type", "title", "model", "engine", "key_metric", "verdict", "status", "date", "commit", "path"]
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 实验索引",
        "",
        "> 本文件由 `scripts/exp_index.py` 生成，请勿手改。",
        "> 新 run 请放到 `results/<stage>/<4位序号>-<slug>/`，契约见 [docs/experiments.md](../docs/experiments.md)。",
        "",
        "| run | stage | type | title | model | engine | key metric | verdict | status | date | commit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| {run} | {stage} | {type} | {title} | {model} | {engine} | {key_metric} | {verdict} | {status} | {date} | {commit} |".format(
                    **row
                )
            )
    else:
        lines.append("| _暂无 run_ |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "_下一步与阶段进度见 [ROADMAP](../ROADMAP.md) §12。_",
            "",
        ]
    )
    index_path = root / "INDEX.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path
