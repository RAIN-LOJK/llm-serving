"""The in-house decoder (``serve/engine.py``) behind the runner's Engine protocol.

Loads straight from the local HF cache through ``model.loading`` — no ``from_pretrained``,
so what the benchmark measures is the hand-written decoder end to end. The attention path is
``docs/decisions.md`` D7: prefill runs as one packed varlen group (FA2 on CUDA,
SDPA-per-segment elsewhere) and seat decode still runs ``hand_baddbmm`` over a ``KvSlab``.
The surrounding cache, RoPE and decode loop stay in this repo. Timing comes from
the engine itself (CUDA events when it runs on a GPU), not from the runner's wall clock.

``encode`` uses ``add_special_tokens=False`` to match ``bench/engines/hf.py``: if the two
arms tokenized differently, the comparison would silently measure different prompts.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from bench.engines.base import Engine, GenerationResult
from model.attention import provenance as attention_provenance
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
        *,
        max_prefill_tokens: int = 8192,
        decode: str = "auto",
    ) -> None:
        self.name = name
        self.revision = revision
        self.dtype_name = dtype
        self.device = "cpu" if device in {"", "auto"} else device
        self.model_path = model_path
        self._max_prefill_tokens = int(max_prefill_tokens)
        self._decode = str(decode)
        self.vocab_size = 0
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
        self.engine = MiniLLMEngine(
            self.model,
            eos_token_id=self.eos_token_id,
            # W4.5 (D7): the scheduler always prefills whole groups; only the group
            # budget and the paged decode kernel remain configurable.
            max_prefill_tokens=self._max_prefill_tokens,
            decode=self._decode,
        )
        self.vocab_size = int(self.model.config.vocab_size)
        self.weights_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 2**20
        # Baseline for "how much does one sequence add on top of the weights".
        self._baseline_mem_mb = driver_memory_mb()

    def attn_backends(self) -> dict[str, str]:
        """The attention path the loaded decoder actually runs.

        W4.5 makes the pair non-constant: packed prefill can resolve to
        ``flash-attn-varlen`` (CUDA) or ``sdpa-varlen`` (CPU/tests), and the paged decode
        kernel can be ``paged-triton`` or ``hand_baddbmm``. The engine owns that answer;
        this adapter only forwards it, falling back to the frozen D4 pair for an engine
        that predates the hook (the W2/W3 arms).
        """
        engine = getattr(self, "engine", None)
        report = getattr(engine, "attn_backends", None)
        if callable(report):
            return dict(report())
        return dict(attention_provenance())

    @property
    def decode_impl(self) -> str:
        """Resolved decode kernel, when the decoder reports one.

        W4.5's paged arm may swap ``hand_baddbmm`` for a Triton block-table kernel; the
        adapter must record which one actually ran, not the config's ``"auto"``. The
        decoder owns that choice, so this is a pass-through to its ``decode_impl`` (or
        ``decode_backend``) attribute. An engine that predates the knob resolves to
        ``"gather"`` -- the path this repo actually has -- rather than inventing a name.
        """
        for name in ("decode_impl", "decode_backend", "_paged_decode_impl"):
            value = getattr(self.engine, name, None)
            if value and str(value) != "auto":
                return str(value)
        return "gather"

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
