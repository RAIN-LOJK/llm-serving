"""Deterministic no-model engine used to validate the run contract.

It intentionally produces meaningless text: its only job is to exercise timing,
metrics, summary, and index plumbing without downloading weights.
"""

from __future__ import annotations

import zlib
from typing import Sequence

from bench.engines.base import GenerationResult
from bench.metrics import elapsed_ms, now, peak_memory_mb


class DummyEngine:
    impl = "dummy"

    def __init__(
        self,
        name: str = "dummy",
        device: str = "cpu",
        vocab_size: int = 1024,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.device = device
        self.vocab_size = vocab_size
        self.seed = seed

    def load(self) -> None:
        return None

    def encode(self, text: str) -> list[int]:
        # Roughly one token per two characters; deterministic across runs.
        length = max(1, min(len(text) // 2, 2048))
        base = zlib.crc32(text.encode("utf-8")) + self.seed
        return [(base + i) % self.vocab_size for i in range(length)]

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        start = now()
        tokens: list[int] = []
        decode_times: list[float] = []
        prefill_ms = None
        ttft_ms = None
        peak = peak_memory_mb()

        if max_new_tokens <= 0:
            return GenerationResult(
                token_ids=[],
                ttft_ms=0.0,
                tpot_ms=None,
                prefill_ms=0.0,
                decode_ms=0.0,
                peak_mem_mb=peak,
                meta={"impl": self.impl, "note": "no generation"},
            )

        # First token: measure TTFT.
        first = (self.seed + (prompt_ids[-1] if prompt_ids else 0)) % self.vocab_size
        tokens.append(first)
        prefill_end = now()
        prefill_ms = elapsed_ms(start, prefill_end)
        ttft_ms = prefill_ms

        for step in range(1, max_new_tokens):
            t0 = now()
            tokens.append((first + step) % self.vocab_size)
            decode_times.append(elapsed_ms(t0))

        decode_ms = sum(decode_times) if decode_times else 0.0
        tpot = (sum(decode_times) / len(decode_times)) if decode_times else None
        return GenerationResult(
            token_ids=tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            peak_mem_mb=peak,
            meta={"impl": self.impl, "vocab_size": self.vocab_size},
        )

    def close(self) -> None:
        return None
