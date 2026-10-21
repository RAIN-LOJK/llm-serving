"""Engine factory.

``impl`` values that are not implemented yet fail loudly with the stage that owns them.
This keeps the runner contract stable while ``serve/`` is being built.
"""

from __future__ import annotations

from typing import Any

from bench.engines.base import Engine, GenerationResult
from bench.engines.dummy import DummyEngine
from bench.engines.hf import HFEngine

__all__ = ["Engine", "GenerationResult", "DummyEngine", "HFEngine", "build_engine"]

_NOT_YET = {
    "minillm": "the toy MiniLLM engine adapter is not wired yet (S2/W2)",
    "continuous": "continuous batching lives in serve/scheduler.py (W3)",
    "paged": "paged KV lives in serve/block_manager.py (W4)",
    "hybrid": "hybrid attention lives in serve/state_pool.py (W7)",
}


def build_engine(model_cfg: dict[str, Any], engine_cfg: dict[str, Any] | None = None) -> Engine:
    """Merge ``[[models]]`` and ``[engine]`` sections into one engine instance."""
    engine_cfg = dict(engine_cfg or {})
    impl = str(engine_cfg.get("impl", "dummy"))
    device = str(engine_cfg.get("device", "auto"))
    name = str(model_cfg.get("name", "dummy"))

    if impl == "dummy":
        return DummyEngine(
            name=name,
            device=device if device != "auto" else "cpu",
            vocab_size=int(engine_cfg.get("vocab_size", 1024)),
            seed=int(engine_cfg.get("seed", 0)),
        )
    if impl == "hf":
        return HFEngine(
            name=name,
            revision=model_cfg.get("revision"),
            dtype=str(model_cfg.get("dtype", engine_cfg.get("dtype", "bfloat16"))),
            device=device,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
    if impl in _NOT_YET:
        raise NotImplementedError(f"engine impl={impl!r}: {_NOT_YET[impl]}")
    raise ValueError(f"unknown engine impl={impl!r}")
