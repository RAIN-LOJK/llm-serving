"""Minimal HuggingFace reference engine.

This is the W0/W2 baseline arm: one request at a time, manual prefill + greedy decode
so TTFT/TPOT are actually measurable (``model.generate`` hides them). Continuous
batching and paged KV are deliberately *not* implemented here; they belong in ``serve/``.

Only text-only causal-LM checkpoints are handled. Qwen3.5's multimodal wrapper may need
an explicit text-model extraction in W0; that is intentionally left as a clear failure
rather than a silent fallback.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from bench.cache_usage import cache_usage_mb, model_bytes_mb
from bench.engines.base import GenerationResult
from bench.metrics import driver_memory_mb, elapsed_ms, now, peak_memory_mb

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_device(requested: str) -> str:
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("device='mps' requested but MPS is unavailable")
    return req


class HFEngine:
    impl = "hf"

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.name = name
        self.revision = revision
        self.dtype_name = dtype
        self.device = _resolve_device(device)
        self.trust_remote_code = trust_remote_code
        self.vocab_size = 0
        self.model: Any = None
        self.tokenizer: Any = None
        self.eos_token_id: int | None = None
        self.weights_mb: float | None = None
        self._baseline_mem_mb: float | None = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.dtype_name not in _DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype_name!r}; choose {sorted(_DTYPES)}")
        dtype = _DTYPES[self.dtype_name]

        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype, **kwargs)
        except TypeError:
            # Older transformers uses torch_dtype.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name, torch_dtype=dtype, **kwargs
            )

        self.model.to(self.device)
        self.model.eval()
        config = getattr(self.model, "config", None)
        text_config = getattr(config, "text_config", None)
        self.vocab_size = int(
            getattr(text_config or config, "vocab_size", self.tokenizer.vocab_size)
        )
        self.eos_token_id = getattr(config, "eos_token_id", None) or self.tokenizer.eos_token_id
        self.weights_mb = model_bytes_mb(self.model)
        # Baseline for "how much does one sequence add on top of the weights".
        self._baseline_mem_mb = driver_memory_mb()

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("call load() before encode()")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.inference_mode()
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("call load() first")
        if max_new_tokens <= 0:
            return GenerationResult(token_ids=[], ttft_ms=0.0, prefill_ms=0.0, decode_ms=0.0)

        device = torch.device(self.device)
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)

        start = now()
        out = self.model(input_ids, use_cache=True)
        past = out.past_key_values
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        prefill_end = now()
        prefill_cache = cache_usage_mb(past)
        tokens = [next_id]
        decode_times: list[float] = []
        decode_start = now()

        for _ in range(1, max_new_tokens):
            if self.eos_token_id is not None and tokens[-1] == self.eos_token_id:
                break
            t0 = now()
            out = self.model(
                torch.tensor([[tokens[-1]]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            tokens.append(int(torch.argmax(out.logits[:, -1, :], dim=-1).item()))
            decode_times.append(elapsed_ms(t0))

        decode_end = now()
        final_cache = cache_usage_mb(past)
        driver_now = driver_memory_mb()
        mem_delta = (
            driver_now - self._baseline_mem_mb
            if driver_now is not None and self._baseline_mem_mb is not None
            else None
        )
        return GenerationResult(
            token_ids=tokens,
            ttft_ms=elapsed_ms(start, prefill_end),
            tpot_ms=(sum(decode_times) / len(decode_times)) if decode_times else None,
            prefill_ms=elapsed_ms(start, prefill_end),
            decode_ms=elapsed_ms(decode_start, decode_end),
            peak_mem_mb=peak_memory_mb(),
            weights_mb=self.weights_mb,
            kv_mb=final_cache["kv_mb"],
            state_mb=final_cache["state_mb"],
            conv_mb=final_cache["conv_mb"],
            cache_mb=final_cache["cache_mb"],
            cache_prefill_mb=prefill_cache["cache_mb"],
            mem_delta_mb=mem_delta,
            meta={
                "impl": self.impl,
                "model": self.name,
                "revision": self.revision,
                "dtype": self.dtype_name,
                "device": self.device,
                "transformers": __import__("transformers").__version__,
                "cache_kv_mb": prefill_cache["kv_mb"],
                "cache_state_mb": prefill_cache["state_mb"],
            },
        )

    def close(self) -> None:
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
