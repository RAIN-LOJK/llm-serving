"""W1 alignment harness: HF checkpoint -> ``model/``, then diff against golden.

Two jobs, deliberately in one file:

1. **The naming contract.** ``map_hf_key`` says which attribute names ``model/`` must grow
   to accept a Qwen3 checkpoint. If a key does not map, the error lists exactly what is
   still missing — that list is the W1 to-do, not a loader bug.
2. **The microscope.** ``report`` prints a per-layer ``max|Δ|`` table, so a red test points
   at a layer instead of saying "it differs".

    python scripts/align_hf.py --golden tests/golden/qwen3-0-6b/0001

The model side is expected to provide, beyond what the toy decoder already has:

* ``config.head_dim`` — Qwen3-0.6B uses 128 while ``hidden/n_head`` is 64
* ``config.rope_theta`` — 1e6, not the 1e4 default
* ``config.qk_norm`` — RMSNorm over ``head_dim`` applied to q and k (``q_norm``/``k_norm``)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import KVcache, MiniLLM, MiniLLMConfig  # noqa: E402

LAYER_TOL = 1e-3
LOGITS_TOL = 0.05


# --------------------------------------------------------------------------- HF -> local

def cached_snapshot(repo: str, revision: str | None) -> Path | None:
    """Local HF cache lookup; returns None instead of raising (tests skip on it)."""
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
        raise SystemExit(f"no *.safetensors in {model_path}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard), device="cpu"))
    return {k: v.to(dtype) for k, v in state.items()}


def _describe_mismatch(own: dict, incoming: dict) -> str | None:
    """Compact, grouped report of what model/ still has to grow (None when it fits)."""
    import re

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


def load_qwen3(model_path: Path, dtype: torch.dtype = torch.float32, device: str = "cpu") -> MiniLLM:
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = MiniLLM(config_from_hf(hf))

    state, mapped = read_checkpoint(model_path, dtype), {}
    for key, value in state.items():
        local = map_hf_key(key)
        if local is not None:
            mapped[local] = value

    problem = _describe_mismatch(model.state_dict(), mapped)
    if problem:
        raise RuntimeError(problem)
    model.load_state_dict(mapped, strict=True)

    model.to(device=device, dtype=dtype).eval()
    return model


# ------------------------------------------------------------------------------- forward

def layer_hidden(model: MiniLLM, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Block outputs for every layer: (n_layers, batch, seq, hidden)."""
    captured: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out)
        )
        for layer in model.layers
    ]
    try:
        with torch.no_grad():
            logits, _ = model(tokens, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(captured), logits


def greedy_ids(
    model: MiniLLM,
    tokens: torch.Tensor,
    n_new: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy decode that keeps the pad mask at full key width (PITFALLS P-05)."""
    kv = KVcache(model.config.n_layer)
    mask = attention_mask
    ones = torch.ones(tokens.shape[0], 1, dtype=mask.dtype) if mask is not None else None
    out: list[torch.Tensor] = []
    with torch.no_grad():
        logits = model(tokens, attention_mask=mask, past_keys_values=kv)[0][:, -1]
        for _ in range(n_new):
            nxt = logits.argmax(-1, keepdim=True)
            out.append(nxt)
            if mask is not None:
                mask = torch.cat([mask, ones], dim=1)
            logits = model(nxt, attention_mask=mask, past_keys_values=kv)[0][:, -1]
    return torch.cat(out, dim=1)


# ------------------------------------------------------------------------------ compare

def load_golden(golden_dir: Path) -> dict:
    meta = json.loads((golden_dir / "meta.json").read_text())
    arrays = dict(np.load(golden_dir / "tensors.npz"))
    return {"meta": meta, "arrays": arrays, "dir": golden_dir}


def compare(
    model: MiniLLM,
    golden: dict,
    layer_tol: float = LAYER_TOL,
    logits_tol: float = LOGITS_TOL,
    check_greedy: bool = True,
) -> dict:
    """Numeric comparison only; ``report`` adds the printing. Tests import this."""
    arrays = golden["arrays"]
    hidden, positions = arrays["hidden"], arrays["positions"]
    input_ids, lengths = arrays["input_ids"], arrays["lengths"]
    greedy_ref, logits_ref = arrays["greedy_ids"], arrays["logits_last"]

    n_prompts, n_layers = hidden.shape[0], hidden.shape[1]
    per_layer = torch.zeros(n_layers, dtype=torch.float64)
    logits_delta = torch.zeros(n_prompts, dtype=torch.float64)
    greedy_ok = 0

    for i in range(n_prompts):
        length = int(lengths[i])
        ids = torch.from_numpy(input_ids[i : i + 1, -length:]).long()
        blocks, logits = layer_hidden(model, ids)
        pick = torch.from_numpy(positions[i]).long()
        ref = torch.from_numpy(hidden[i]).float()  # (layers, positions, hidden)
        got = blocks[:, 0].index_select(1, pick)  # (layers, positions, hidden)
        per_layer = torch.maximum(per_layer, (got - ref).abs().amax(dim=(1, 2)).double())

        logits_delta[i] = (logits[0, -1].double() - torch.from_numpy(logits_ref[i]).double()).abs().max()
        if check_greedy:
            out = greedy_ids(model, ids, greedy_ref.shape[1])
            ref_ids = torch.from_numpy(greedy_ref[i]).long()
            valid = ref_ids >= 0
            greedy_ok += int(torch.equal(out[0][valid], ref_ids[valid]))

    ok_layers = bool(per_layer.max() <= layer_tol)
    ok_logits = bool(logits_delta.max() <= logits_tol)
    ok_greedy = (not check_greedy) or greedy_ok == n_prompts
    return {
        "per_layer": per_layer,
        "logits": logits_delta,
        "greedy_ok": greedy_ok,
        "n_prompts": n_prompts,
        "ok": ok_layers and ok_logits and ok_greedy,
    }


def report(model: MiniLLM, golden: dict, layer_tol: float = LAYER_TOL, logits_tol: float = LOGITS_TOL) -> dict:
    result = compare(model, golden, layer_tol, logits_tol)
    meta = golden["meta"]
    per_layer, logits_delta, greedy_ok = result["per_layer"], result["logits"], result["greedy_ok"]
    n_prompts, n_ids = result["n_prompts"], golden["arrays"]["greedy_ids"].shape[1]

    print(f"golden   {golden['dir'].relative_to(ROOT)}")
    print(f"model    {meta['model']['repo']}@{meta['model']['revision']}  {meta['model']['dtype']}/{meta['model']['device']}")
    print(f"tolerance layer {layer_tol:g}, logits {logits_tol:g}")
    print(f"{'layer':>5}  {'max|Δ|':>10}")
    for layer, delta in enumerate(per_layer.tolist()):
        flag = "  <= FAIL" if delta > layer_tol else ""
        print(f"{layer:>5}  {delta:>10.3e}{flag}")
    print(f"{'logits':>5}  {logits_delta.max().item():>10.3e}   (per prompt {np.array2string(logits_delta.numpy(), precision=2)})")
    print(f"greedy   {greedy_ok}/{n_prompts} prompts, {n_ids} ids each")
    print(f"result   {'ok' if result['ok'] else 'FAIL'}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff model/ against a W1 golden fixture")
    ap.add_argument("--golden", type=Path, default=ROOT / "tests/golden/qwen3-0-6b/0001")
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--layer-tol", type=float, default=LAYER_TOL)
    ap.add_argument("--logits-tol", type=float, default=LOGITS_TOL)
    args = ap.parse_args()

    golden = load_golden(args.golden)
    model_path = args.model_path or cached_snapshot(
        golden["meta"]["model"]["repo"], golden["meta"]["model"]["revision"]
    )
    if model_path is None:
        raise SystemExit("checkpoint is not in the HF cache; pass --model-path")

    model = load_qwen3(model_path)
    result = report(model, golden, args.layer_tol, args.logits_tol)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
