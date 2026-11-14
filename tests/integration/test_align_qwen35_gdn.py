"""W6 Step 3 acceptance: one Gated DeltaNet layer vs the HF golden, stage by stage.

``Qwen35GDNBlock`` is six operations deep (norm -> qkv projection -> short conv -> delta
rule -> gated norm -> MLP); a whole-block diff ("layer 0 is off by 4.0e-2") cannot say
which one is wrong. This test compares every intermediate stage against the reference
captured by ``scripts/make_golden_qwen35_w6.py``, in data-flow order:

    input_layernorm -> in_proj_qkv -> in_proj_z/b/a -> conv_out
                    -> q_pre/k_pre, beta/g -> core_attn_out -> norm_out -> layer_out

and, for the long prompt, the two carried states (``final_state``, ``final_conv_state``).
Every stage is produced by the **production** helpers (``block.input_layernorm``,
``la.in_proj_*``, ``causal_conv_prefill``, ``compute_beta_g``, ``gated_delta_chunked``,
``la.norm``, ``la.out_proj``, ``block.mlp``) -- this file never re-implements the
arithmetic. ``layer_out`` is additionally asserted bit-identical to
``block(layer_in)``, i.e. the composite below must be the block.

``conv_out`` is the **post-SiLU** conv activation; ``final_conv_state`` is the pre-SiLU
input columns, sliced ``[..., -3:]`` from HF's 4-column cache (P-49). ``q_pre``/``k_pre``
are the raw q/k at the rule entry, before the in-kernel l2norm and the q scale. Those two
conventions are what make the stage table diagnostic instead of decorative.

Tolerances are read from ``configs/thresholds.toml [W6.gdn]`` (pre-registered in
docs/w6-plan.md §6.3); relative stages use ``max|Δ| / max|ref|`` over the rows the golden
stored, the two states use absolute atol. The golden path can point at a Step-3 draft:

    MINILLM_RUN_HF=1 MINILLM_W6_GOLDEN=/tmp/w6-golden-dry \\
        python -m pytest tests/integration/test_align_qwen35_gdn.py -q -s

The formal gate (after the user commits ``tests/golden/qwen35-0-8b/0002``) must not set
``MINILLM_W6_GOLDEN``. The committed fixture is additionally required to have
``meta["source"]["dirty"] is False`` (P-57); a ``/tmp`` draft is exempt, because the
generator script itself is untracked while Step 3 is in progress.

    layer  prompt  stage             max_abs      rel        target   verdict
    0      s01     input_layernorm   0.000e+00    0.000e+00  1.0e-05  pass
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from model.loading import config_from_hf, load_layer_weights
from model.qwen35_gdn import (
    Qwen35GDNBlock,
    causal_conv_prefill,
    compute_beta_g,
    gated_delta_chunked,
)
from scripts.make_golden import resolve_snapshot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN = ROOT / "tests" / "golden" / "qwen35-0-8b" / "0002"
GOLDEN = Path(os.environ.get("MINILLM_W6_GOLDEN") or DEFAULT_GOLDEN)

pytestmark = [pytest.mark.slow, pytest.mark.hf]

#: Stage -> threshold key in ``[W6.gdn]`` (pre-registered rel line).
STAGE_THRESHOLDS = {
    "input_layernorm": "norm_rel_max",
    "in_proj_qkv": "proj_rel_max",
    "in_proj_z": "proj_rel_max",
    "in_proj_b": "proj_rel_max",
    "in_proj_a": "proj_rel_max",
    "conv_out": "conv_rel_max",
    "q_pre": "proj_rel_max",
    "k_pre": "proj_rel_max",
    "beta": "proj_rel_max",
    "g": "proj_rel_max",
    "core_attn_out": "gdn_out_rel_max",
    "norm_out": "norm_rel_max",
    "layer_out": "layer_out_rel_max",
}
STAGE_ORDER = tuple(STAGE_THRESHOLDS)
#: Stages compared with an absolute atol instead of a relative one.
ABSOLUTE_STAGES = {"final_state": "state_atol", "final_conv_state": "conv_atol"}


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tolerances() -> dict:
    section = tomllib.loads((ROOT / "configs" / "thresholds.toml").read_text())["W6"]["gdn"]
    required = {
        "basis_golden",
        "causal_atol",
        "state_atol",
        "conv_atol",
        *STAGE_THRESHOLDS.values(),
    }
    missing = required - set(section)
    assert not missing, f"[W6.gdn] is missing {sorted(missing)}"
    assert section["status"] in ("target", "calibrated"), section["status"]
    return section


@pytest.fixture(scope="module")
def golden() -> dict:
    if os.environ.get("MINILLM_RUN_HF") != "1":
        pytest.skip("set MINILLM_RUN_HF=1 to run the HF golden alignment")
    if not (GOLDEN / "meta.json").exists() or not (GOLDEN / "tensors.npz").exists():
        pytest.skip(
            f"no W6 golden at {GOLDEN}; generate it with scripts/make_golden_qwen35_w6.py"
        )
    meta = json.loads((GOLDEN / "meta.json").read_text())
    if meta.get("spec") != "w6-gdn-v1":
        pytest.skip(f"{GOLDEN} has spec={meta.get('spec')!r}, expected 'w6-gdn-v1'")
    # P-57: only the committed fixture is required to be clean. A /tmp draft is generated
    # from an untracked script (this one), so dirty=true is expected there.
    if GOLDEN.resolve() == DEFAULT_GOLDEN.resolve():
        assert meta["source"]["dirty"] is False, (
            f"{GOLDEN} was generated from a dirty tree (P-57): {meta['source']}"
        )
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
        assert cfg.layer_types[layer_idx] == "linear_attention", (
            f"golden layer {layer_idx} is not a linear-attention layer in the snapshot"
        )
        block = Qwen35GDNBlock(cfg).eval()
        block.load_state_dict(load_layer_weights(snapshot, layer_idx), strict=True)
        blocks[layer_idx] = block
    return cfg, blocks


# ------------------------------------------------------------------- stage capture


def compute_stages(cfg, block, layer_in: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run one GDN block and return every stage, using the production helpers only."""
    la = block.linear_attn
    batch, seq, _ = layer_in.shape

    normed = block.input_layernorm(layer_in)

    mixed = la.in_proj_qkv(normed)  # (1, T, conv_dim), raw projection
    conv_out, conv_state = causal_conv_prefill(mixed.transpose(1, 2), la.conv1d.weight)
    conv_out = conv_out.transpose(1, 2)  # post-SiLU, (1, T, conv_dim)

    query, key, value = torch.split(
        conv_out, [la.key_dim, la.key_dim, la.value_dim], dim=-1
    )
    query = query.reshape(batch, seq, la.num_k_heads, la.head_k_dim)
    key = key.reshape(batch, seq, la.num_k_heads, la.head_k_dim)
    value = value.reshape(batch, seq, la.num_v_heads, la.head_v_dim)

    b = la.in_proj_b(normed)
    a = la.in_proj_a(normed)
    beta, g = compute_beta_g(b, a, la.A_log, la.dt_bias)

    core, final_state = gated_delta_chunked(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=True,
    )

    z_proj = la.in_proj_z(normed)
    z = z_proj.reshape(batch, seq, la.num_v_heads, la.head_v_dim)
    norm_out = la.norm(
        core.reshape(-1, la.head_v_dim), z.reshape(-1, la.head_v_dim)
    ).reshape(batch, seq, la.value_dim)

    attn_out = la.out_proj(norm_out)
    x = layer_in + attn_out
    layer_out = x + block.mlp(block.post_attention_layernorm(x))

    return {
        "input_layernorm": normed,
        "in_proj_qkv": mixed,
        "in_proj_z": z_proj,
        "in_proj_b": b,
        "in_proj_a": a,
        "conv_out": conv_out,
        "q_pre": query,
        "k_pre": key,
        "beta": beta,
        "g": g,
        "core_attn_out": core,
        "norm_out": norm_out,
        "layer_out": layer_out,
        # Carried state, not a stage; compared against the fixture's final_conv_state.
        "conv_state": conv_state,
        "final_state": final_state,
    }


def compare(stage: str, ours: torch.Tensor, ref: torch.Tensor, sampled: torch.Tensor) -> dict:
    """``max|Δ|`` and ``rel`` over the rows the golden stored for this stage."""
    ours = ours[0].float()
    ours = ours.index_select(0, sampled)
    ref = ref.float()
    assert ours.shape == ref.shape, (
        f"{stage}: ours {tuple(ours.shape)} vs golden {tuple(ref.shape)}"
    )
    max_abs = (ours - ref).abs().max().item()
    scale = ref.abs().max().item()
    rel = max_abs / scale if scale > 0 else max_abs
    return {"stage": stage, "max_abs": max_abs, "rel": rel, "scale": scale}


def compare_absolute(stage: str, ours: torch.Tensor, ref: torch.Tensor) -> dict:
    """``max|Δ|`` for a full tensor compared with an absolute atol."""
    ours = ours.to(torch.float32)
    ref = ref.float()
    assert ours.shape == ref.shape, (
        f"{stage}: ours {tuple(ours.shape)} vs golden {tuple(ref.shape)}"
    )
    return {"stage": stage, "max_abs": (ours - ref).abs().max().item()}


def _print_stage_table(rows: list[dict]) -> None:
    print(
        f"\n{'layer':>5} {'prompt':>7} {'stage':<17} {'max_abs':>11} {'rel':>11} "
        f"{'target':>10}  verdict"
    )
    for row in rows:
        print(
            f"{row['layer']:>5} {row['prompt']:>7} {row['stage']:<17} "
            f"{row['max_abs']:>11.3e} {row['rel']:>11.3e} {row['target']:>10.1e}  "
            f"{'pass' if row['ok'] else 'FAIL'}"
        )


def _print_state_table(rows: list[dict]) -> None:
    print(
        f"\n{'layer':>5} {'prompt':>7} {'stage':<17} {'max_abs':>11} {'target':>10}  "
        "verdict"
    )
    for row in rows:
        print(
            f"{row['layer']:>5} {row['prompt']:>7} {row['stage']:<17} "
            f"{row['max_abs']:>11.3e} {row['target']:>10.1e}  "
            f"{'pass' if row['ok'] else 'FAIL'}"
        )


# ------------------------------------------------------------------------- tests


def test_layer_stages_match_golden(aligned, golden, tolerances):
    cfg, blocks = aligned
    meta, arrays = golden["meta"], golden["arrays"]
    final_prompts = set(meta["stages"].get("final_prompts", ()))
    print(
        f"\nW6 golden {golden['dir']} (spec {meta['spec']}, device {meta['model']['device']}, "
        f"{meta['model']['dtype']}, {meta['model']['attn_implementation']}); "
        f"tolerance basis {tolerances['basis_golden']}"
    )

    # Guard against a fixture that stored one layer under two keys: L0 and L2 must not see
    # identical residual streams for the same prompt (they are different weights).
    first, last = meta["layers"][0], meta["layers"][-1]
    if first != last:
        for prompt in meta["prompts"]:
            prompt_id = prompt["id"]
            assert not np.array_equal(
                arrays[f"L{first}/{prompt_id}/layer_out"],
                arrays[f"L{last}/{prompt_id}/layer_out"],
            ), (
                f"golden layers {first} and {last} stored identical layer_out for "
                f"{prompt_id} -- wrong keys?"
            )

    rows: list[dict] = []
    state_rows: list[dict] = []
    failures: list[str] = []
    with torch.no_grad():
        for layer_idx in meta["layers"]:
            block = blocks[layer_idx]
            for prompt in meta["prompts"]:
                prompt_id = prompt["id"]
                layer_in = torch.from_numpy(
                    arrays[f"L{layer_idx}/{prompt_id}/layer_in"]
                ).float()
                layer_in = layer_in.unsqueeze(0)
                assert layer_in.shape[1] == prompt["n_tokens"]
                sampled = torch.from_numpy(arrays[f"{prompt_id}/positions"]).long()

                stages = compute_stages(cfg, block, layer_in)

                # The composite above must be the block, not a look-alike. Both go through
                # the same helpers in the same order, so this is bit-exact on CPU fp32.
                block_out, block_conv_state, block_final_state = block(layer_in)
                assert torch.equal(stages["layer_out"], block_out), (
                    f"layer {layer_idx}/{prompt_id}: compute_stages' layer_out differs "
                    f"from block(layer_in) by "
                    f"{(stages['layer_out'] - block_out).abs().max().item():.3e}; the "
                    "test's composite is not the production path"
                )
                assert torch.equal(stages["conv_state"], block_conv_state)
                assert torch.equal(stages["final_state"], block_final_state)

                for stage in STAGE_ORDER:
                    ref = torch.from_numpy(arrays[f"L{layer_idx}/{prompt_id}/{stage}"])
                    result = compare(stage, stages[stage], ref, sampled)
                    target = float(tolerances[STAGE_THRESHOLDS[stage]])
                    ok = result["rel"] <= target
                    rows.append(
                        {
                            **result,
                            "layer": layer_idx,
                            "prompt": prompt_id,
                            "target": target,
                            "ok": ok,
                        }
                    )
                    if not ok:
                        failures.append(
                            f"layer {layer_idx}/{prompt_id}/{stage}: "
                            f"rel={result['rel']:.3e} > {target:.1e} "
                            f"(max|Δ|={result['max_abs']:.3e}, "
                            f"max|ref|={result['scale']:.3e})"
                        )

                final_keys = {
                    stage: f"L{layer_idx}/{prompt_id}/{stage}" for stage in ABSOLUTE_STAGES
                }
                expected_final = prompt_id in final_prompts
                present = [stage for stage, key in final_keys.items() if key in arrays]
                if not present and not expected_final:
                    continue
                # A fixture that stores only one of the two states, or claims m02 stores
                # them and does not, is broken -- not "all present rows pass".
                assert set(present) == set(ABSOLUTE_STAGES), (
                    f"L{layer_idx}/{prompt_id}: expected both "
                    f"{sorted(ABSOLUTE_STAGES)} in the golden, found {present}"
                )
                for stage, key in ABSOLUTE_STAGES.items():
                    ref = torch.from_numpy(arrays[final_keys[stage]])
                    ours = (
                        stages["final_state"][0]
                        if stage == "final_state"
                        else stages["conv_state"][0]
                    )
                    result = compare_absolute(stage, ours, ref)
                    target = float(tolerances[key])
                    ok = result["max_abs"] <= target
                    state_rows.append(
                        {
                            **result,
                            "layer": layer_idx,
                            "prompt": prompt_id,
                            "target": target,
                            "ok": ok,
                        }
                    )
                    if not ok:
                        failures.append(
                            f"layer {layer_idx}/{prompt_id}/{stage}: "
                            f"max|Δ|={result['max_abs']:.3e} > {target:.1e} (absolute)"
                        )

    _print_stage_table(rows)
    if state_rows:
        _print_state_table(state_rows)
    assert not failures, (
        f"W6 GDN stage alignment failed on {len(failures)} row(s); first: "
        f"{failures[0]}\n" + "\n".join(failures)
    )


def test_causality_on_golden_input_is_exact(aligned, golden, tolerances):
    """G1: replacing layer input token t must not move any position < t (fp32 CPU)."""
    cfg, blocks = aligned
    meta, arrays = golden["meta"], golden["arrays"]
    layer_idx = meta["layers"][0]
    block = blocks[layer_idx]
    prompt_id = meta["prompts"][0]["id"]

    clean = (
        torch.from_numpy(arrays[f"L{layer_idx}/{prompt_id}/layer_in"]).float().unsqueeze(0)
    )
    t = clean.shape[1] // 2
    perturbed = clean.clone()
    perturbed[0, t] += 1.0

    with torch.no_grad():
        before = block(clean)[0]
        after = block(perturbed)[0]

    atol = float(tolerances["causal_atol"])
    leaked = (before[:, :t] - after[:, :t]).abs().max().item()
    assert leaked <= atol, (
        f"a later token leaked into positions < {t}: max|Δ|={leaked:.3e} > "
        f"causal_atol={atol}"
    )
    moved = (before[:, t] - after[:, t]).abs().max().item()
    assert moved > 0.0, "the perturbation was a no-op -- the causality test proves nothing"
