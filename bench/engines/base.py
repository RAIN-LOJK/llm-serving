"""Engine protocol shared by the HF baseline, the toy engine, and future serve/ engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class GenerationResult:
    token_ids: list[int]
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    peak_mem_mb: float | None = None
    # Per-sequence state, counted from the cache tensors themselves (W0 carriers).
    weights_mb: float | None = None
    kv_mb: float | None = None
    state_mb: float | None = None
    conv_mb: float | None = None
    cache_mb: float | None = None
    cache_prefill_mb: float | None = None
    mem_delta_mb: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    name: str
    impl: str
    device: str
    vocab_size: int

    def load(self) -> None: ...

    def encode(self, text: str) -> list[int]: ...

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> GenerationResult: ...

    def close(self) -> None: ...
