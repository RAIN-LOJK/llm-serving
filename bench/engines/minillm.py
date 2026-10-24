"""The in-house decoder (``serve/engine.py``) behind the runner's Engine protocol.

Loads straight from the local HF cache through ``model.loading`` — no ``from_pretrained``,
so what the benchmark measures is the hand-written decoder end to end. Attention runs
through PyTorch SDPA (``F.scaled_dot_product_attention``); the surrounding cache, RoPE and
decode loop stay in this repo. Timing comes from the engine itself (CUDA events when it
runs on a GPU), not from the runner's wall clock.

``encode`` uses ``add_special_tokens=False`` to match ``bench/engines/hf.py``: if the two
arms tokenized differently, the comparison would silently measure different prompts.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from bench.engines.base import Engine, GenerationResult
from bench.metrics import driver_memory_mb, peak_memory_mb
from model.loading import cached_snapshot, load_checkpoint
from serve.engine import MiniLLMEngine

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
LOGITS_TO_KEEP = 1


class ServeEngine:
    """Adapter: ``miniLLM.Engine`` -> ``bench.engines.base.Engine``."""

    impl = "minillm"

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        dtype: str = "float32",
        device: str = "cpu",
        model_path: str | None = None,
    ) -> None:
        self.name = name
        self.revision = revision
        self.dtype_name = dtype
        self.device = "cpu" if device in {"", "auto"} else device
        self.model_path = model_path
        self.vocab_size = 0
        # The engine has one attention path now; keep the field for run provenance.
        self.attn_implementation = "sdpa"
        self.weights_mb: float | None = None
        self.eos_token_id: int | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.engine: MiniLLMEngine | None = None
        self._baseline_mem_mb: float | None = None

    def load(self) -> None:
        from transformers import AutoTokenizer

        if self.dtype_name not in _DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype_name!r}; choose {sorted(_DTYPES)}")

        path = self.model_path or cached_snapshot(self.name, self.revision)
        if path is None:
            raise FileNotFoundError(
                f"{self.name}@{self.revision} is not in the HF cache; "
                "fetch it first (HF_ENDPOINT=https://hf-mirror.com) or pass model_path"
            )

        self.model = load_checkpoint(
            path,
            dtype=_DTYPES[self.dtype_name],
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.engine = MiniLLMEngine(self.model, eos_token_id=self.eos_token_id)
        self.vocab_size = int(self.model.config.vocab_size)
        self.weights_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 2**20
        # Baseline for "how much does one sequence add on top of the weights".
        self._baseline_mem_mb = driver_memory_mb()

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("call load() before encode()")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        if self.engine is None:
            raise RuntimeError("call load() before generate()")
        out = self.engine.generate(list(prompt_ids), max_new_tokens)

        driver_now = driver_memory_mb()
        mem_delta = (
            driver_now - self._baseline_mem_mb
            if driver_now is not None and self._baseline_mem_mb is not None
            else None
        )
        return GenerationResult(
            token_ids=out.token_ids,
            ttft_ms=out.ttft_ms,
            tpot_ms=out.tpot_ms,
            prefill_ms=out.prefill_ms,
            decode_ms=out.decode_ms,
            peak_mem_mb=peak_memory_mb(),
            weights_mb=self.weights_mb,
            # Qwen3 is a pure-attention carrier: the whole per-sequence state is KV.
            # A hybrid carrier will fill state_mb/conv_mb through serve/state_pool.py (W7).
            kv_mb=out.cache_mb,
            state_mb=0.0,
            conv_mb=0.0,
            cache_mb=out.cache_mb,
            mem_delta_mb=mem_delta,
            meta={
                "impl": self.impl,
                "model": self.name,
                "revision": self.revision,
                "dtype": self.dtype_name,
                "device": self.device,
                "timer": out.meta.get("timer"),
                "prompt_tokens": out.meta.get("prompt_tokens"),
                "out_tokens": out.meta.get("out_tokens"),
                "decode_wall_ms": out.meta.get("decode_wall_ms"),
                "logits_to_keep": LOGITS_TO_KEEP,
            },
        )

    def close(self) -> None:
        self.engine = None
        self.model = None
        self.tokenizer = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
