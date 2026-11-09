"""W5 Step 3 acceptance: one full-attention layer vs the HF golden, stage by stage.

A whole-model diff ("layer 23 is off by 3.1e-2") cannot say *which* operator is wrong. This
test compares every intermediate stage of ``Qwen35FullAttentionBlock`` against the
reference captured by ``scripts/make_golden_qwen35_layer.py``, in the order the data flows:

    input_layernorm -> q_proj -> q_norm/k_norm -> q_rope/k_rope -> gate
                    -> attn_gated (o_proj input) -> layer_out

The stages are produced by the production helpers (``_project``/``_rope``/``_attend`` via
the block, ``q_norm``/``k_norm``/``input_layernorm`` modules) -- this file never
re-implements the arithmetic. ``attn_gated`` is captured by a forward pre-hook on
``self_attn.o_proj``, the same measurement the golden script makes on HF.

Tolerances are read from ``configs/thresholds.toml [W5.layer]`` (pre-registered in
docs/w5-plan.md §5); the golden path can point at a draft with ``MINILLM_W5_GOLDEN``:

    MINILLM_RUN_HF=1 MINILLM_W5_GOLDEN=/tmp/w5-golden-dry \\
        python -m pytest tests/integration/test_align_qwen35_layer.py -q -s

Rows are ``rel = max|Δ| / max|ref|`` over the stored rows of each stage.

    layer  prompt  stage             max_abs      rel        target   verdict
    3      s01     input_layernorm   0.000e+00    0.000e+00  1.0e-05  pass
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from model.layers import rope_cos_sin
from model.loading import config_from_hf, load_layer_weights
from model.qwen35 import Qwen35FullAttentionBlock
from scripts.make_golden import resolve_snapshot

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(
    os.environ.get("MINILLM_W5_GOLDEN") or ROOT / "tests" / "golden" / "qwen35-0-8b" / "0001"
)

pytestmark = [pytest.mark.slow, pytest.mark.hf]

#: Stage -> threshold key in ``[W5.layer]``.
STAGE_THRESHOLDS = {
    "input_layernorm": "norm_rel_max",
    "q_proj": "proj_rel_max",
    "q_norm": "norm_rel_max",
    "k_norm": "norm_rel_max",
    "gate": "proj_rel_max",
    "q_rope": "rope_rel_max",
    "k_rope": "rope_rel_max",
    "attn_gated": "attn_gated_rel_max",
    "layer_out": "layer_out_rel_max",
}
STAGE_ORDER = tuple(STAGE_THRESHOLDS)
#: Stages stored at every position; the rest are stored at 0 / T//2 / T-1.
FULL_STAGES = frozenset({"layer_in", "q_rope", "k_rope"})


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tolerances() -> dict:
    section = tomllib.loads((ROOT / "configs" / "thresholds.toml").read_text())["W5"]["layer"]
    missing = {"basis_golden", *STAGE_THRESHOLDS.values(), "causal_atol"} - set(section)
    assert not missing, f"[W5.layer] is missing {sorted(missing)}"
    assert section["status"] in ("target", "calibrated"), section["status"]
    return section


@pytest.fixture(scope="module")
def golden() -> dict:
    if os.environ.get("MINILLM_RUN_HF") != "1":
        pytest.skip("set MINILLM_RUN_HF=1 to run the HF golden alignment")
    if not (GOLDEN / "meta.json").exists():
        pytest.skip(f"no W5 golden at {GOLDEN}; generate it with scripts/make_golden_qwen35_layer.py")
    meta = json.loads((GOLDEN / "meta.json").read_text())
    if meta.get("spec") != "w5-golden-v1":
        pytest.skip(f"{GOLDEN} has spec={meta.get('spec')!r}, expected 'w5-golden-v1'")
    arrays = dict(np.load(GOLDEN / "tensors.npz"))
    return {"meta": meta, "arrays": arrays, "dir": GOLDEN}


@pytest.fixture(scope="module")
def aligned(golden):
    """``(config, {layer: block})`` built from the snapshot recorded in the golden meta."""
    meta = golden["meta"]
    try:
        snapshot, _ = resolve_snapshot(meta["model"]["repo"], meta["model"]["revision"])
    except SystemExit as exc:  # resolve_snapshot raises SystemExit when the cache is cold
        pytest.skip(str(exc))
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    cfg = config_from_hf(hf)

    blocks = {}
    for layer_idx in meta["layers"]:
        assert cfg.layer_types[layer_idx] == "full_attention", (
            f"golden layer {layer_idx} is not a full-attention layer in the snapshot"
        )
        block = Qwen35FullAttentionBlock(cfg).eval()
        block.load_state_dict(load_layer_weights(snapshot, layer_idx), strict=True)
        blocks[layer_idx] = block
    return cfg, blocks


# ------------------------------------------------------------------- stage capture


def compute_stages(cfg, block, layer_in: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run one block and return every stage, using the production helpers only."""
    attn = block.self_attn
    batch, seq, _ = layer_in.shape

    normed = block.input_layernorm(layer_in)
    q, gate, k, v = attn._project(normed)
    q_norm = attn.q_norm(q)
    k_norm = attn.k_norm(k)
    cos, sin = rope_cos_sin(
        cfg.head_dim, cfg.rope_theta, 0, seq, layer_in.device, layer_in.dtype,
        rotary_dim=cfg.rotary_dim,
    )
    q_rope, k_rope = attn._rope(q_norm, k_norm, cos, sin)

    captured: dict[str, torch.Tensor] = {}
    handle = attn.o_proj.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("attn_gated", inputs[0].detach())
    )
    try:
        layer_out = block(layer_in)
    finally:
        handle.remove()

    return {
        "input_layernorm": normed,
        # The layout inverse of the per-head chunk; pinned by
        # tests/unit/test_qwen35_attention.py::test_q_proj_output_is_query_and_gate_interleaved_per_head
        "q_proj": torch.cat([q, gate], dim=-1).reshape(batch, seq, -1),
        "q_norm": q_norm,
        "k_norm": k_norm,
        "gate": gate,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "attn_gated": captured["attn_gated"],
        "layer_out": layer_out,
    }


def compare(name: str, ours: torch.Tensor, ref: torch.Tensor, sampled: torch.Tensor) -> dict:
    """``max|Δ|`` and ``rel`` over the rows the golden stored for this stage."""
    ours = ours[0].float()
    if name not in FULL_STAGES:
        ours = ours.index_select(0, sampled)
    ref = ref.float()
    assert ours.shape == ref.shape, f"{name}: ours {tuple(ours.shape)} vs golden {tuple(ref.shape)}"
    max_abs = (ours - ref).abs().max().item()
    scale = ref.abs().max().item()
    rel = max_abs / scale if scale > 0 else max_abs
    return {"stage": name, "max_abs": max_abs, "rel": rel, "scale": scale}


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'layer':>5} {'prompt':>7} {'stage':<17} {'max_abs':>11} {'rel':>11} "
          f"{'target':>10}  verdict")
    for row in rows:
        print(
            f"{row['layer']:>5} {row['prompt']:>7} {row['stage']:<17} "
            f"{row['max_abs']:>11.3e} {row['rel']:>11.3e} {row['target']:>10.1e}  "
            f"{'pass' if row['ok'] else 'FAIL'}"
        )


# ------------------------------------------------------------------------- tests


def test_layer_stages_match_golden(aligned, golden, tolerances):
    cfg, blocks = aligned
    meta, arrays = golden["meta"], golden["arrays"]
    print(
        f"\nW5 golden {golden['dir']} (spec {meta['spec']}, device {meta['model']['device']}, "
        f"{meta['model']['dtype']}, {meta['model']['attn_implementation']}); "
        f"tolerance basis {tolerances['basis_golden']}"
    )

    # Guard against a fixture that stored one layer under two keys: the first and last
    # full-attention layers must not see identical residual streams.
    first, last = meta["layers"][0], meta["layers"][-1]
    probe = meta["prompts"][0]["id"]
    if first != last:
        assert not np.array_equal(
            arrays[f"L{first}/{probe}/layer_out"], arrays[f"L{last}/{probe}/layer_out"]
        ), f"golden layer {first} and {last} stored identical outputs -- wrong keys?"

    rows: list[dict] = []
    failures: list[str] = []
    with torch.no_grad():
        for layer_idx in meta["layers"]:
            block = blocks[layer_idx]
            for prompt in meta["prompts"]:
                prompt_id = prompt["id"]
                layer_in = torch.from_numpy(arrays[f"L{layer_idx}/{prompt_id}/layer_in"]).float()
                layer_in = layer_in.unsqueeze(0)
                assert layer_in.shape[1] == prompt["n_tokens"]
                sampled = torch.from_numpy(arrays[f"{prompt_id}/positions"]).long()
                stages = compute_stages(cfg, block, layer_in)

                for stage in STAGE_ORDER:
                    ref = torch.from_numpy(arrays[f"L{layer_idx}/{prompt_id}/{stage}"])
                    result = compare(stage, stages[stage], ref, sampled)
                    target = float(tolerances[STAGE_THRESHOLDS[stage]])
                    ok = result["rel"] <= target
                    rows.append(
                        {**result, "layer": layer_idx, "prompt": prompt_id, "target": target, "ok": ok}
                    )
                    if not ok:
                        failures.append(
                            f"layer {layer_idx}/{prompt_id}/{stage}: rel={result['rel']:.3e} "
                            f"> {target:.1e} (max|Δ|={result['max_abs']:.3e}, "
                            f"max|ref|={result['scale']:.3e})"
                        )

    _print_table(rows)
    assert not failures, "W5 stage alignment failed:\n" + "\n".join(failures)


def test_causality_on_golden_input_is_exact(aligned, golden, tolerances):
    """G6: replacing layer input token t must not move any position < t (fp32 CPU)."""
    cfg, blocks = aligned
    meta, arrays = golden["meta"], golden["arrays"]
    layer_idx = meta["layers"][0]
    block = blocks[layer_idx]
    prompt_id = meta["prompts"][0]["id"]

    clean = torch.from_numpy(arrays[f"L{layer_idx}/{prompt_id}/layer_in"]).float().unsqueeze(0)
    t = clean.shape[1] // 2
    perturbed = clean.clone()
    perturbed[0, t] += 1.0

    with torch.no_grad():
        before, after = block(clean), block(perturbed)

    atol = float(tolerances["causal_atol"])
    leaked = (before[:, :t] - after[:, :t]).abs().max().item()
    assert leaked <= atol, (
        f"a later token leaked into positions < {t}: max|Δ|={leaked:.3e} > causal_atol={atol}"
    )
    moved = (before[:, t] - after[:, t]).abs().max().item()
    assert moved > 0.0, "the perturbation was a no-op -- the causality test proves nothing"
