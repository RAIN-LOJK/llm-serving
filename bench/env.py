"""Environment snapshot used by every run.

The snapshot is deliberately explicit: a number without device / dtype / commit is not
reproducible. ``bench.run`` writes this to ``env.json`` and copies the git fields into
``manifest.json``.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_info(root: str | Path | None = None) -> dict[str, Any]:
    """Return commit / branch / dirty. ``dirty`` is None when git is unavailable."""
    cwd = str(root or Path.cwd())
    commit = _run(["git", "rev-parse", "HEAD"], cwd)
    branch = _run(["git", "branch", "--show-current"], cwd)
    status = _run(["git", "status", "--porcelain"], cwd)
    dirty = None if status is None else bool(status)
    described = _run(["git", "describe", "--tags", "--always", "--dirty"], cwd)
    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": dirty,
        "describe": described,
    }


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency in practice
        return {"version": None, "error": repr(exc)}

    try:
        mps = getattr(torch.backends, "mps", None)
        mps_available = bool(mps is not None and mps.is_available())
        cuda_available = bool(torch.cuda.is_available())
        info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "mps_available": mps_available,
        }
        if cuda_available:
            info["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                    "total_memory_mb": round(
                        torch.cuda.get_device_properties(i).total_memory / 2**20, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        return info
    except Exception as exc:  # pragma: no cover
        return {"version": getattr(torch, "__version__", None), "error": repr(exc)}


def _nvidia_smi() -> dict[str, Any] | None:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    return {"name": parts[0], "memory_total_mb": parts[1], "driver_version": parts[2]}


def hardware_info() -> dict[str, Any]:
    mem_bytes = None
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        mem_bytes = None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory_gb": round(mem_bytes / 2**30, 2) if mem_bytes else None,
        "nvidia_smi": _nvidia_smi(),
    }


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
        "matplotlib",
        "pytest",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot(root: str | Path | None = None) -> dict[str, Any]:
    """Capture everything that can change a measured number."""
    env_keys = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "TOKENIZERS_PARALLELISM",
    )
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "git": git_info(root),
        "torch": torch_info(),
        "hardware": hardware_info(),
        "packages": package_versions(),
        "env": {k: os.environ[k] for k in env_keys if k in os.environ},
    }


def write_snapshot(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot(root), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path
