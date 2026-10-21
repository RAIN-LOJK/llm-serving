"""Analytic carrier accounting: how many bytes per token does each model keep?

W0 asks one question that does not need a GPU: given the published config, how much
per-sequence state does a 3:1 hybrid keep relative to a pure-attention model of the same
class? The answer is exact arithmetic from ``config.json``; the probe run later checks it
against real allocator numbers.

Two quantities matter and they scale differently:

* KV cache: ``2 * n_kv_head * head_dim * dtype_bytes`` per full-attention *layer*, per token.
  Linear in context, zero when the layer is linear.
* Recurrent state (Gated DeltaNet): ``n_v_head * d_k * d_v`` floats per linear *layer*,
  plus a short-conv window. Constant in context, so it only pays off past a crossover.

This module is deliberately torch-free so the fast test gate can import it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "uint8": 1,
}

FULL = "full_attention"
LINEAR = "linear_attention"


def _dtype_bytes(name: Any, default: int = 2) -> int:
    return DTYPE_BYTES.get(str(name).lower(), default)


def text_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Qwen3.5 checkpoints nest the language model under ``text_config``."""
    nested = raw.get("text_config")
    return nested if isinstance(nested, Mapping) else raw


def layer_types(cfg: Mapping[str, Any]) -> list[str]:
    n_layers = int(cfg["num_hidden_layers"])
    declared = cfg.get("layer_types")
    if declared:
        types = [str(t) for t in declared]
        if len(types) != n_layers:
            raise ValueError(f"layer_types has {len(types)} entries, num_hidden_layers={n_layers}")
        return types
    return [FULL] * n_layers


def attention_spec(cfg: Mapping[str, Any], dtype: str = "bfloat16") -> dict[str, Any]:
    """Per-layer attention geometry, with the Qwen defaults made explicit."""
    n_layers = int(cfg["num_hidden_layers"])
    types = layer_types(cfg)
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // n_heads)
    return {
        "n_layers": n_layers,
        "layer_types": types,
        "n_full_layers": sum(1 for t in types if t == FULL),
        "n_linear_layers": sum(1 for t in types if t == LINEAR),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": _dtype_bytes(dtype),
        "hidden_size": int(cfg["hidden_size"]),
        "vocab_size": int(cfg.get("vocab_size", 0)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings", 0)),
        "attn_output_gate": bool(cfg.get("attn_output_gate", False)),
        "partial_rotary_factor": float(
            (cfg.get("rope_parameters") or {}).get("partial_rotary_factor", 1.0)
        ),
        "mrope": bool((cfg.get("rope_parameters") or {}).get("mrope_interleaved", False)),
    }


def kv_bytes_per_token(spec: Mapping[str, Any]) -> int:
    """K and V, every full-attention layer, one token."""
    return (
        2
        * int(spec["n_kv_heads"])
        * int(spec["head_dim"])
        * int(spec["dtype_bytes"])
        * int(spec["n_full_layers"])
    )


def recurrent_bytes_per_sequence(cfg: Mapping[str, Any], spec: Mapping[str, Any]) -> int:
    """Gated DeltaNet state: S[d_k, d_v] per value head per linear layer, in fp32.

    ``mamba_ssm_dtype`` is declared by the checkpoint (fp32 for Qwen3.5); the accumulation
    dtype is a property of the operator, not of the weight file.
    """
    n_linear = int(spec["n_linear_layers"])
    if n_linear == 0:
        return 0
    n_v_heads = int(cfg.get("linear_num_value_heads", 0))
    d_k = int(cfg.get("linear_key_head_dim", 0))
    d_v = int(cfg.get("linear_value_head_dim", 0))
    if not (n_v_heads and d_k and d_v):
        raise ValueError("linear layers present but linear_* head geometry is missing")
    state_bytes = _dtype_bytes(cfg.get("mamba_ssm_dtype", "float32"), default=4)
    return n_linear * n_v_heads * d_k * d_v * state_bytes


def conv_bytes_per_sequence(cfg: Mapping[str, Any], spec: Mapping[str, Any]) -> int:
    """Short causal conv window: kernel taps over the q/k/v channels, in weight dtype.

    Estimated, not load-bearing: it is < 1 MB here and is reported only so the fixed cost
    is not silently understated.
    """
    n_linear = int(spec["n_linear_layers"])
    if n_linear == 0:
        return 0
    kernel = int(cfg.get("linear_conv_kernel_dim", 0))
    if kernel == 0:
        return 0
    d_k = int(cfg.get("linear_num_key_heads", 0)) * int(cfg.get("linear_key_head_dim", 0))
    d_v = int(cfg.get("linear_num_value_heads", 0)) * int(cfg.get("linear_value_head_dim", 0))
    channels = 2 * d_k + d_v  # q, k, v are concatenated before the conv
    return n_linear * kernel * channels * int(spec["dtype_bytes"])


def estimate(raw: Mapping[str, Any], dtype: str = "bfloat16") -> dict[str, Any]:
    """Full analytic account for one checkpoint config."""
    cfg = text_config(raw)
    spec = attention_spec(cfg, dtype)
    kv = kv_bytes_per_token(spec)
    state = recurrent_bytes_per_sequence(cfg, spec)
    conv = conv_bytes_per_sequence(cfg, spec)
    fixed = state + conv
    return {
        **spec,
        "kv_bytes_per_token": kv,
        "recurrent_bytes_per_sequence": state,
        "conv_bytes_per_sequence": conv,
        "fixed_bytes_per_sequence": fixed,
        "kv_bytes_at": {
            str(ctx): kv * int(ctx) for ctx in (512, 2048, 8192, 32768)
        },
        "total_cache_bytes_at": {
            str(ctx): kv * int(ctx) + fixed for ctx in (512, 2048, 8192, 32768)
        },
    }


def crossover_tokens(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> float | None:
    """Context length past which ``candidate`` holds less per-sequence state.

    Solves ``kv_ref * N == kv_cand * N + fixed_cand``. ``None`` when the candidate saves
    nothing per token (no crossover, it is never cheaper).
    """
    saving = int(reference["kv_bytes_per_token"]) - int(candidate["kv_bytes_per_token"])
    if saving <= 0:
        return None
    return int(candidate["fixed_bytes_per_sequence"]) / saving


def asymptotic_ratio(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> float | None:
    """``reference / candidate`` cache bytes as context grows; 1.0 when they are equal."""
    kv_c = int(candidate["kv_bytes_per_token"])
    return (int(reference["kv_bytes_per_token"]) / kv_c) if kv_c else None


def human_bytes(value: float | None) -> str:
    if value is None:
        return "—"
    units: Sequence[tuple[str, int]] = (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10))
    for unit, size in units:
        if abs(value) >= size:
            return f"{value / size:.2f} {unit}"
    return f"{value:.0f} B"


def markdown_table(estimates: Mapping[str, Mapping[str, Any]]) -> str:
    """Compact table for ``docs/carrier.md``; byte counts are binary units."""
    header = (
        "| 模型 | 层型 | 全注意力层 | KV / token | 线性状态 / 序列 | 常数项 / 序列 | 512 | 2k | 8k | 32k |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, est in estimates.items():
        types = est["layer_types"]
        pattern = "".join("L" if t == LINEAR else "F" for t in types)
        total = est["total_cache_bytes_at"]
        rows.append(
            "| {name} | `{pattern}` | {full} | {kv} | {state} | {fixed} | {c512} | {c2k} | {c8k} | {c32k} |".format(
                name=name,
                pattern=pattern,
                full=est["n_full_layers"],
                kv=human_bytes(est["kv_bytes_per_token"]),
                state=human_bytes(est["recurrent_bytes_per_sequence"]),
                fixed=human_bytes(est["fixed_bytes_per_sequence"]),
                c512=human_bytes(total["512"]),
                c2k=human_bytes(total["2048"]),
                c8k=human_bytes(total["8192"]),
                c32k=human_bytes(total["32768"]),
            )
        )
    return header + "\n".join(rows)


def compare(
    estimates: Mapping[str, Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    cand, ref = estimates[candidate], estimates[reference]
    table = {}
    for ctx in ("512", "2048", "8192", "32768"):
        c = cand["total_cache_bytes_at"][ctx]
        r = ref["total_cache_bytes_at"][ctx]
        table[ctx] = {"candidate": c, "reference": r, "ratio": (r / c) if c else None}
    return {
        "candidate": candidate,
        "reference": reference,
        "crossover_tokens": crossover_tokens(cand, ref),
        "asymptotic_ratio": asymptotic_ratio(cand, ref),
        "by_context": table,
    }
