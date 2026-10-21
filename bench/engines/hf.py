"""Minimal HuggingFace reference engine.

This is the W0/W2 baseline arm: one request at a time, manual prefill + greedy decode
so TTFT/TPOT are actually measurable (``model.generate`` hides them). Continuous
batching and paged KV are deliberately *not* implemented here; they belong in ``serve/``.

Only text-only causal LMs are handled. Multimodal checkpoints (Qwen3.5) are loaded through
their declared ``text_config`` so the vision tower and the MTP head never enter device
memory; a guard fails loudly if a wrapper class or multimodal parameter leaks through.

Prefill asks for ``logits_to_keep=1``: the default would materialize ``[seq, vocab]``
(~9.3 GiB at 32k for a 152k vocab, ~15.2 GiB for 248k) even though greedy decoding only
needs the last position.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from bench.cache_usage import cache_usage_mb, model_bytes_mb
from bench.engines.base import GenerationResult
from bench.metrics import driver_memory_mb, elapsed_ms, now, peak_memory_mb

LOGITS_TO_KEEP = 1
MULTIMODAL_PARAM_PREFIXES = ("model.visual.", "visual.", "mtp.")

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _assert_text_only(model: Any, text_config: Any) -> None:
    """Fail loudly if a multimodal checkpoint loaded its vision tower or MTP head."""
    if text_config is None:
        return
    class_name = type(model).__name__
    if "ConditionalGeneration" in class_name:
        raise RuntimeError(
            f"loaded {class_name}, expected a text-only causal LM; "
            "pass config=config.get_text_config() when loading a multimodal checkpoint"
        )
    leaked = [
        name
        for name, _ in model.named_parameters()
        if name.startswith(MULTIMODAL_PARAM_PREFIXES)
    ]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} multimodal parameters are resident (e.g. {leaked[0]}); "
            "the vision tower/MTP head must not be loaded for text-only probing"
        )
    expected = int(getattr(text_config, "vocab_size", 0))
    actual = int(getattr(getattr(model, "config", None), "vocab_size", 0))
    if expected and actual != expected:
        raise RuntimeError(f"text vocab mismatch: loaded {actual}, expected {expected}")


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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if self.dtype_name not in _DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype_name!r}; choose {sorted(_DTYPES)}")
        dtype = _DTYPES[self.dtype_name]

        config = AutoConfig.from_pretrained(
            self.name, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        text_config = getattr(config, "text_config", None)

        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }
        if text_config is not None:
            # Qwen3.5 ships as `...ForConditionalGeneration`; loading the text config
            # explicitly selects the decoder and skips vision + MTP weights.
            kwargs["config"] = config.get_text_config()

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

        _assert_text_only(self.model, text_config)
        self.model.to(self.device)
        self.model.eval()
        model_config = getattr(self.model, "config", None)
        self.vocab_size = int(
            getattr(text_config or model_config, "vocab_size", self.tokenizer.vocab_size)
        )
        self.eos_token_id = (
            getattr(text_config, "eos_token_id", None)
            or getattr(model_config, "eos_token_id", None)
            or getattr(config, "eos_token_id", None)
            or self.tokenizer.eos_token_id
        )
        self.weights_mb = model_bytes_mb(self.model)
        # Baseline for "how much does one sequence add on top of the weights".
        self._baseline_mem_mb = driver_memory_mb()
        self.arch = type(self.model).__name__
        self.attn_implementation = getattr(model_config, "_attn_implementation", None)

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("call load() before encode()")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _forward(self, input_ids: torch.Tensor, past_key_values: Any = None):
        """One forward pass that keeps exactly one position of logits."""
        kwargs: dict[str, Any] = {"use_cache": True, "logits_to_keep": LOGITS_TO_KEEP}
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        try:
            out = self.model(input_ids, **kwargs)
        except TypeError as exc:
            raise RuntimeError(
                f"{type(self.model).__name__}.forward() rejected logits_to_keep; "
                "update the engine instead of falling back to [seq, vocab] logits"
            ) from exc
        if out.logits.shape[1] != LOGITS_TO_KEEP:
            raise RuntimeError(
                f"model returned {out.logits.shape[1]} logit positions for "
                f"{input_ids.shape[1]} input tokens; logits_to_keep was ignored, which "
                "materializes [seq, vocab]"
            )
        return out

    @torch.inference_mode()
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("call load() first")
        if max_new_tokens <= 0:
            return GenerationResult(token_ids=[], ttft_ms=0.0, prefill_ms=0.0, decode_ms=0.0)

        device = torch.device(self.device)
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)

        start = now()
        out = self._forward(input_ids)
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
            out = self._forward(
                torch.tensor([[tokens[-1]]], dtype=torch.long, device=device),
                past_key_values=past,
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
                "arch": getattr(self, "arch", type(self.model).__name__),
                "logits_to_keep": LOGITS_TO_KEEP,
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
