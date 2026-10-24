"""Load a HuggingFace checkpoint into the hand-written decoder.

This is the only place that knows how HF names things. It is separate from ``model/layers.py``
so that the forward path stays free of checkpoint plumbing, and it is shared by the W1
alignment harness (``scripts/align_hf.py``) and the benchmark adapter
(``bench/engines/minillm.py``) — one loader, one naming contract.

Checkpoint keys map onto the repo's modules like this::

    model.embed_tokens.weight                    -> token_embedding.weight
    model.layers.N.self_attn.{q,k,v,o}_proj      -> layers.N.self_attn.{q,k,v,o}_proj
    model.layers.N.self_attn.{q,k}_norm.weight   -> layers.N.self_attn.{q,k}_norm
    model.layers.N.{input,post_attention}_layernorm -> same names
    model.layers.N.mlp.{gate,up,down}_proj       -> layers.N.mlp.{gate,up,down}_proj
    model.norm.weight                            -> norm.weight
    lm_head.weight                               -> lm_head.weight (tied to the embedding)

If a key does not map, the error lists exactly which parameters ``model/`` still has to grow.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

from model.config import MiniLLMConfig
from model.minillm import MiniLLM


def cached_snapshot(repo: str, revision: str | None = None) -> Path | None:
    """Local HF cache lookup; returns None instead of raising (callers decide how to fail)."""
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
    if revision:
        path = snapshots / revision
        return path if path.exists() else None
    dirs = sorted(p for p in snapshots.glob("*") if p.is_dir()) if snapshots.exists() else []
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def rope_parameters(hf) -> dict:
    """RoPE settings, normalised across transformers versions.

    transformers 5.x moved ``rope_theta`` (and ``partial_rotary_factor``, ``mrope_*``) into
    ``rope_parameters``; 4.x exposed them as flat attributes. Reading only the flat one
    silently falls back to 1e4 — the Qwen3 default is 1e6, so every layer would diff for
    the wrong reason (same failure mode as PITFALLS P-03: a default that looks wired).
    """
    params = dict(getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {})
    for name in ("rope_theta", "partial_rotary_factor", "rope_type"):
        if name not in params and getattr(hf, name, None) is not None:
            params[name] = getattr(hf, name)
    return params


def config_from_hf(hf) -> MiniLLMConfig:
    """Map a (possibly multimodal) HF config onto the hand-written config."""
    text = hf.get_text_config() if hasattr(hf, "get_text_config") else hf
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads

    rope = rope_parameters(text)
    theta = rope.get("rope_theta")
    if theta is None:
        raise ValueError(
            f"{type(text).__name__} exposes no rope_theta (rope_parameters={rope!r}); "
            "refusing to guess 1e4 for a checkpoint that may use something else"
        )
    rope_type = str(rope.get("rope_type", "default"))
    if rope_type != "default":
        raise ValueError(
            f"rope_type={rope_type!r}: model/ implements unscaled RoPE only; "
            "scaled variants (linear/dynamic/yarn) need their own alignment"
        )

    return MiniLLMConfig(
        vocab_size=text.vocab_size,
        max_position_embeddings=text.max_position_embeddings,
        n_embd=text.hidden_size,
        n_head=text.num_attention_heads,
        n_kv_head=text.num_key_value_heads,
        n_layer=text.num_hidden_layers,
        n_inner=text.intermediate_size,
        dropout=0.0,
        bias=bool(getattr(text, "attention_bias", False)),
        norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(text.tie_word_embeddings),
        head_dim=head_dim,
        rope_theta=float(theta),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
        qk_norm=True,
    )


def map_hf_key(key: str) -> str | None:
    """Qwen3 checkpoint key -> this repo's module path. None means "not ours"."""
    if key.endswith("rotary_emb.inv_freq") or key.endswith("rotary_emb.cos_cached"):
        return None
    if key.startswith("model."):
        key = key[len("model."):]
    if key.startswith("embed_tokens."):
        key = "token_embedding." + key[len("embed_tokens."):]
    return key


def read_checkpoint(model_path: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {model_path}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard), device="cpu"))
    return {k: v.to(dtype) for k, v in state.items()}


def _describe_mismatch(own: dict, incoming: dict) -> str | None:
    """Compact, grouped report of what model/ still has to grow (None when it fits)."""

    def like(key: str) -> str:
        return re.sub(r"layers\.\d+", "layers.*", key)

    missing = sorted({like(k) for k in set(own) - set(incoming)})
    extra = sorted({like(k) for k in set(incoming) - set(own)})
    shapes = {
        (like(k), tuple(own[k].shape), tuple(incoming[k].shape))
        for k in set(own) & set(incoming)
        if own[k].shape != incoming[k].shape
    }
    if not (missing or extra or shapes):
        return None

    lines = ["the checkpoint does not fit model/ yet — that gap is the W1 to-do:"]
    if extra:
        lines.append(f"  checkpoint has {len(extra)} parameter kind(s) the model lacks:")
        lines += [f"    {k}" for k in extra[:8]]
    if missing:
        lines.append("  model has parameters the checkpoint cannot fill:")
        lines += [f"    {k}" for k in missing[:8]]
    if shapes:
        lines.append("  shape mismatch (model vs checkpoint):")
        lines += [
            f"    {k}: {own_shape} vs {ckpt_shape}" for k, own_shape, ckpt_shape in sorted(shapes)[:8]
        ]
    lines.append("  W1 expects: head_dim=128 (not hidden/n_head=64) and q_norm/k_norm over head_dim.")
    return "\n".join(lines)


def load_checkpoint(
    model_path: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> MiniLLM:
    """Build a ``MiniLLM`` from an HF snapshot directory, without ``from_pretrained``.

    Deliberately does not go through ``AutoModelForCausalLM``: the measured path must be the
    hand-written decoder end to end, and a silent partial load is exactly the failure mode
    that costs days.
    """
    from transformers import AutoConfig

    model_path = Path(model_path)
    hf = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = MiniLLM(config_from_hf(hf))

    mapped = {}
    for key, value in read_checkpoint(model_path, dtype).items():
        local = map_hf_key(key)
        if local is not None:
            mapped[local] = value

    problem = _describe_mismatch(model.state_dict(), mapped)
    if problem:
        raise RuntimeError(problem)
    model.load_state_dict(mapped, strict=True)

    return model.to(device=device, dtype=dtype).eval()
