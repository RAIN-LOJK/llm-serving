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

from model import KVcache, MiniLLM  # noqa: E402
from model.loading import (  # noqa: E402  (re-exported: tests import them from here)
    cached_snapshot,
    config_from_hf,
    load_checkpoint,
    map_hf_key,
    read_checkpoint,
    rope_parameters,
)

load_qwen3 = load_checkpoint

LAYER_TOL = 1e-3
LOGITS_TOL = 0.05


# --------------------------------------------------------------------------- HF -> local

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
