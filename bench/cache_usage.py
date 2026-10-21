"""Byte accounting for a live ``past_key_values`` object.

W0 compares a pure-attention carrier with a hybrid one, and the difference lives entirely
in per-sequence state: KV grows with context, the Gated DeltaNet state does not. Peak
process memory cannot show that (weights differ by ~0.3 GiB), so we count the cache tensors
themselves.

The walker is deliberately structural rather than version-specific: transformers 5 keeps
KV in ``DynamicLayer.keys/values`` and hybrid state in ``LinearAttentionLayer.conv_states``
/ ``recurrent_states`` dicts, while older versions used a list of ``(k, v)`` tuples. Both
shape families are handled, and unknown tensor attributes fall into the bucket of their
parent so a future rename under-counts nothing silently.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

KV_ATTRS = {"keys", "values", "key_cache", "value_cache", "key", "value"}
STATE_ATTRS = {"recurrent_states", "recurrent_state", "ssm_states", "ssm_state"}
CONV_ATTRS = {"conv_states", "conv_state"}

BUCKETS = ("kv", "state", "conv")


def _bucket_for(attr: str, default: str) -> str:
    name = attr.lower()
    if name in KV_ATTRS:
        return "kv"
    if name in STATE_ATTRS:
        return "state"
    if name in CONV_ATTRS:
        return "conv"
    return default


def _walk(obj: Any, bucket: str, out: dict[str, int], seen: set[int]) -> None:
    if obj is None:
        return
    if torch.is_tensor(obj):
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        out[bucket] += obj.numel() * obj.element_size()
        return
    if isinstance(obj, Mapping):
        for value in obj.values():
            _walk(value, bucket, out, seen)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, bucket, out, seen)
        return
    if hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            if attr.startswith("_"):
                continue
            _walk(value, _bucket_for(attr, bucket), out, seen)


def cache_usage(cache: Any) -> dict[str, float | int]:
    """Bytes held by ``cache``, split into KV / recurrent / conv.

    Values are raw bytes; callers convert to MiB. ``total`` is the sum of the three
    buckets, not a device counter.
    """
    if cache is None:
        return {"kv": 0, "state": 0, "conv": 0, "total": 0, "n_tensors": 0}
    out = {name: 0 for name in BUCKETS}
    seen: set[int] = set()
    # Legacy `list[tuple[K, V]]` has no attribute names, so default to the KV bucket.
    _walk(cache, "kv", out, seen)
    return {
        **out,
        "total": sum(out.values()),
        "n_tensors": len(seen),
    }


def cache_usage_mb(cache: Any) -> dict[str, float]:
    usage = cache_usage(cache)
    return {
        "kv_mb": usage["kv"] / 2**20,
        "state_mb": usage["state"] / 2**20,
        "conv_mb": usage["conv"] / 2**20,
        "cache_mb": usage["total"] / 2**20,
    }


def model_bytes_mb(model: Any) -> float:
    """Bytes actually resident for loaded parameters (vision tower included if loaded)."""
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    return total / 2**20
