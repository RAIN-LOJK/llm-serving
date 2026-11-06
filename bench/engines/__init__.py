"""Engine factory.

``impl`` values that are not implemented yet fail loudly with the stage that owns them.
This keeps the runner contract stable while ``serve/`` is being built.
"""

from __future__ import annotations

from typing import Any

from bench.engines.base import Engine, GenerationResult
from bench.engines.continuous import ContinuousEngine
from bench.engines.dummy import DummyEngine
from bench.engines.hf import HFEngine
from bench.engines.minillm import ServeEngine
from bench.engines.paged import PagedEngine

__all__ = [
    "Engine",
    "GenerationResult",
    "ContinuousEngine",
    "DummyEngine",
    "HFEngine",
    "PagedEngine",
    "ServeEngine",
    "build_engine",
]

_NOT_YET = {
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
    if impl == "minillm":
        return ServeEngine(
            name=name,
            revision=model_cfg.get("revision"),
            dtype=str(model_cfg.get("dtype", engine_cfg.get("dtype", "float32"))),
            device=device,
            model_path=model_cfg.get("path"),
        )
    if impl == "continuous":
        # W3 (D5): one scheduler over a trace. Same weights/tokenizer path as `minillm`;
        # `name="dummy"` selects the deterministic stand-in used by the smoke config.
        # `policy` lives in `[scheduler]` (a top-level table), not `[engine]`, so the
        # runner passes it in through `engine_cfg["_policy"]` after config validation.
        return ContinuousEngine(
            name=name,
            revision=model_cfg.get("revision"),
            dtype=str(model_cfg.get("dtype", engine_cfg.get("dtype", "float32"))),
            device=device,
            model_path=model_cfg.get("path"),
            max_batch=int(engine_cfg.get("max_batch", 1)),
            policy=str(engine_cfg.get("_policy", "continuous")),
            vocab_size=int(engine_cfg.get("vocab_size", 1024)),
            seed=int(engine_cfg.get("seed", 0)),
            fill=bool(engine_cfg.get("_fill", False)),
        )
    if impl == "paged":
        # W4 (D6): the same trace/scheduler path as `continuous`, with physical KV blocks
        # instead of one slab row per seat. `block_size` / `kv_blocks` come from `[engine]`
        # and are validated by the runner; `fill` selects the fill-only step loop
        # (`bench/engines/fill.py`) for `[exp].type = "fill"` capacity probes.
        return PagedEngine(
            name=name,
            revision=model_cfg.get("revision"),
            dtype=str(model_cfg.get("dtype", engine_cfg.get("dtype", "float32"))),
            device=device,
            model_path=model_cfg.get("path"),
            max_batch=int(engine_cfg.get("max_batch", 1)),
            policy=str(engine_cfg.get("_policy", "continuous")),
            vocab_size=int(engine_cfg.get("vocab_size", 1024)),
            seed=int(engine_cfg.get("seed", 0)),
            block_size=int(engine_cfg.get("block_size", 16)),
            kv_blocks=int(engine_cfg.get("kv_blocks", 1)),
            fill=bool(engine_cfg.get("_fill", False)),
        )
    if impl in _NOT_YET:
        raise NotImplementedError(f"engine impl={impl!r}: {_NOT_YET[impl]}")
    raise ValueError(f"unknown engine impl={impl!r}")
