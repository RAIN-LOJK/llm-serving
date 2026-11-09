"""Load a HuggingFace checkpoint into the hand-written decoder.

This is the only place that knows how HF names things. It is separate from ``model/layers.py``
so that the forward path stays free of checkpoint plumbing, and it is shared by the W1
alignment harness (``scripts/align_hf.py``) and the benchmark adapter
(``bench/engines/minillm.py``) — one loader, one naming contract.

Checkpoint keys map onto the repo's modules like this (Qwen3.5 nests the language model
under ``model.language_model.``; Qwen3 uses ``model.``)::

    model.language_model.embed_tokens.weight     -> token_embedding.weight
    model.language_model.layers.N.self_attn...   -> layers.N.self_attn...
    model.layers.N.self_attn...                  -> layers.N.self_attn...
    model.language_model.norm.weight             -> norm.weight
    lm_head.weight                               -> lm_head.weight (tied to the embedding)

``mtp.*`` and the visual tower are out of scope (docs/decisions.md D2): ``map_hf_key``
returns None for them so a whole-model load reports the one real gap (W6's
``linear_attn.*``) instead of a wall of MTP/vision keys.

``load_layer_weights`` is the W5 entry point: it reads a single full-attention layer and
returns keys relative to the block (``self_attn.q_proj.weight``), so a block can load it
with ``strict=True``. Whole-model loading of 0.8B stays a failure until W6 implements the
18 linear-attention layers.

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

    layer_types = getattr(text, "layer_types", None)
    mrope_section = rope.get("mrope_section")

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
        layer_types=None if layer_types is None else list(layer_types),
        attn_output_gate=bool(getattr(text, "attn_output_gate", False)),
        mrope_section=None if mrope_section is None else list(mrope_section),
        linear_num_key_heads=getattr(text, "linear_num_key_heads", None),
        linear_num_value_heads=getattr(text, "linear_num_value_heads", None),
        linear_key_head_dim=getattr(text, "linear_key_head_dim", None),
        linear_value_head_dim=getattr(text, "linear_value_head_dim", None),
        linear_conv_kernel_dim=getattr(text, "linear_conv_kernel_dim", None),
    )


def map_hf_key(key: str) -> str | None:
    """Qwen3/Qwen3.5 checkpoint key -> this repo's module path. None means "not ours"."""
    if key.endswith("rotary_emb.inv_freq") or key.endswith("rotary_emb.cos_cached"):
        return None
    # Qwen3.5's decoder lives under ``model.language_model.``; Qwen3's directly under
    # ``model.``. Strip the longer prefix first so ``language_model.`` never leaks into a
    # local key (it would look like an unknown module instead of a mapped one).
    if key.startswith("model.language_model."):
        key = key[len("model.language_model."):]
    elif key.startswith("model."):
        key = key[len("model."):]
    # D2: MTP and the vision tower have no module in model/. Drop them explicitly; letting
    # them through would bury the real gap (linear_attn.*) under 190 MTP/vision kinds.
    if key.startswith(("mtp.", "visual.")):
        return None
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

    lines = ["the checkpoint does not fit model/ yet — model/ still has to grow the kinds below:"]
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
    if any(kind.startswith("layers.*.linear_attn.") for kind in extra):
        lines.append(
            "  W6 to-do: layers.*.linear_attn.* (gated delta-net) has no module in "
            "model/ yet; W5 loads one full-attention layer at a time with "
            "load_layer_weights()."
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ one layer (W5)

_FULL_ATTENTION = "full_attention"


def _layer_prefixes(layer_idx: int) -> tuple[str, str]:
    """Both checkpoint layouts, Qwen3.5 (``language_model``) and Qwen3 (plain ``model``)."""
    return (
        f"model.language_model.layers.{layer_idx}.",
        f"model.layers.{layer_idx}.",
    )


def _expected_layer_shapes(config: MiniLLMConfig) -> dict[str, tuple[int, ...]]:
    """Parameter shapes of one full-attention block, derived from the HF config."""
    head_dim = config.head_dim or config.n_embd // config.n_head
    q_out = config.n_head * head_dim * (2 if config.attn_output_gate else 1)
    kv_out = config.n_kv_head * head_dim
    n_inner = config.n_inner if config.n_inner is not None else config.n_embd * 4
    shapes = {
        "input_layernorm.weight": (config.n_embd,),
        "self_attn.q_proj.weight": (q_out, config.n_embd),
        "self_attn.k_proj.weight": (kv_out, config.n_embd),
        "self_attn.v_proj.weight": (kv_out, config.n_embd),
        "self_attn.o_proj.weight": (config.n_embd, config.n_head * head_dim),
        "post_attention_layernorm.weight": (config.n_embd,),
        "mlp.gate_proj.weight": (n_inner, config.n_embd),
        "mlp.up_proj.weight": (n_inner, config.n_embd),
        "mlp.down_proj.weight": (config.n_embd, n_inner),
    }
    if config.qk_norm:
        shapes["self_attn.q_norm.weight"] = (head_dim,)
        shapes["self_attn.k_norm.weight"] = (head_dim,)
    return shapes


def _read_layer_tensors(snapshot: Path, layer_idx: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Read only one layer's tensors out of the shards, keyed by the tail after ``layers.N.``."""
    from safetensors import safe_open

    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {snapshot}")

    prefixes = _layer_prefixes(layer_idx)
    found: dict[str, torch.Tensor] = {}
    source: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                for prefix in prefixes:
                    if not key.startswith(prefix):
                        continue
                    tail = key[len(prefix):]
                    if tail in found:
                        raise RuntimeError(
                            f"layer {layer_idx} appears under both {source[tail]!r} and "
                            f"{key!r}; refusing to guess which one is the model"
                        )
                    found[tail] = handle.get_tensor(key).to(dtype)
                    source[tail] = key
                    break
    return found


def _describe_layer_mismatch(
    snapshot: Path,
    layer_idx: int,
    expected: dict[str, tuple[int, ...]],
    got: dict[str, torch.Tensor],
) -> str | None:
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    shapes = [
        (name, expected[name], tuple(got[name].shape))
        for name in sorted(set(expected) & set(got))
        if tuple(got[name].shape) != expected[name]
    ]
    if not (missing or unexpected or shapes):
        return None

    lines = [f"layer {layer_idx} in {snapshot} does not fit the W5 full-attention block:"]
    if missing:
        lines.append("  missing from the checkpoint:")
        lines += [f"    {name}" for name in missing]
    if unexpected:
        lines.append("  checkpoint keys the W5 block does not define:")
        lines += [f"    {name}" for name in unexpected]
    if shapes:
        lines.append("  shape mismatch (expected vs checkpoint):")
        lines += [f"    {name}: expected {want}, got {have}" for name, want, have in shapes]
    return "\n".join(lines)


def load_layer_weights(
    snapshot: str | Path,
    layer_idx: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load exactly one full-attention layer, keyed relative to the block.

    W5 loads one layer at a time (D-W5-2): the 18 ``linear_attn`` layers do not exist in
    ``model/`` yet, so a whole-model load cannot succeed. Requesting a linear-attention
    layer raises instead of returning ``self_attn`` keys that would silently fit nothing.

    The returned keys are what a W5 block's ``state_dict`` uses
    (``self_attn.q_proj.weight``, ``input_layernorm.weight``, ``mlp.*``, ...), so the
    block can be filled with ``block.load_state_dict(..., strict=True)``. Tensors come back
    on CPU; the caller owns the device move.
    """
    from transformers import AutoConfig

    snapshot = Path(snapshot)
    hf = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config = config_from_hf(hf)

    if not 0 <= layer_idx < config.n_layer:
        raise ValueError(
            f"layer_idx={layer_idx} is out of range for n_layer={config.n_layer} "
            f"(expected 0..{config.n_layer - 1})"
        )
    layer_type = config.layer_types[layer_idx]
    if layer_type != _FULL_ATTENTION:
        raise ValueError(
            f"layer {layer_idx} is {layer_type!r}: load_layer_weights() loads one "
            "full_attention layer; linear_attention (gated delta-net) is the W6 to-do"
        )

    expected = _expected_layer_shapes(config)
    mapped: dict[str, torch.Tensor] = {}
    for tail, tensor in _read_layer_tensors(snapshot, layer_idx, dtype).items():
        local = map_hf_key(tail)  # drops the per-layer rotary cache, if present
        if local is not None:
            mapped[local] = tensor

    problem = _describe_layer_mismatch(snapshot, layer_idx, expected, mapped)
    if problem:
        raise RuntimeError(problem)

    return {name: mapped[name] for name in sorted(mapped)}


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

    # A tied checkpoint may omit ``lm_head.weight`` (Qwen3.5-0.8B does; Qwen3-0.6B ships
    # it). That is not a gap in model/: fill it from the embedding, which is what HF's
    # ``tie_weights`` does, so the mismatch report stays about parameters model/ genuinely
    # lacks instead of blaming a tied weight.
    if (
        "lm_head.weight" not in mapped
        and model.config.tie_word_embeddings
        and "token_embedding.weight" in mapped
    ):
        mapped["lm_head.weight"] = mapped["token_embedding.weight"]

    problem = _describe_mismatch(model.state_dict(), mapped)
    if problem:
        raise RuntimeError(problem)
    model.load_state_dict(mapped, strict=True)

    return model.to(device=device, dtype=dtype).eval()
