"""W6 Step 5 acceptance: the whole 24-layer Qwen3.5-0.8B decoder vs the HF golden.

``Qwen35GDN`` was aligned one stage at a time in ``test_align_qwen35_gdn.py``; that says
nothing about the *assembly* -- the ``LLLF`` schedule, the shared ``Qwen35State``, the
zero-centered final norm, the tied ``lm_head`` and the cached decode path. This test closes
that gap against ``tests/golden/qwen35-0-8b/0002`` (docs/w6-plan.md §6.1, §7.2), using the
full-model arrays ``scripts/make_golden_qwen35_w6.py`` captures in the same prefill as the
layer stages:

* ``layer_hidden``: **every** decoder block's output (residual included -- the block
  output, not ``hidden_states[-1]``, which transformers 5.x post-final-norms; P-16) at the
  rows ``sorted({0, T // 2, T - 1})``, compared per layer with
  ``[W6.gdn].layer_hidden_max_abs``;
* ``logits_last``: the last prefill position's logits, compared with
  ``full_logits_max_abs``;
* ``greedy_ids``: 64 ids decoded through one ``Qwen35State`` (prefill + 63 one-token
  decodes), compared **exactly** to ``greedy_tokens_exact``. EOS is deliberately not
  special-cased (P-55): this is a numerical gate, not a chat demo. The fixture also stores
  each step's top-2 margin, so the first mismatch reports both margins -- a near-tie
  changes the *attribution*, never the gate;
* for ``s01`` only (36 tokens), a full prefill vs token-by-token decode on fresh states
  within ``decode_prefill_atol``. Running this for ``m02`` (240 tokens) would add minutes;
  the property is exercised at every position of the short prompt instead.

The production model is compared as-is (``load_qwen35_model``, fp32, CPU, eval); this file
never re-implements the forward. Point it at a Step-5 draft with::

    MINILLM_RUN_HF=1 MINILLM_W6_GOLDEN=/tmp/w6-golden-dry \\
        python -m pytest tests/integration/test_align_qwen35_model.py -q -s

The formal gate (after the user commits ``0002``) must not set ``MINILLM_W6_GOLDEN``; the
committed fixture is additionally required to have ``meta["source"]["dirty"] is False``
(P-57), while a ``/tmp`` draft is exempt. The fixture must advertise ``"full"`` in
``meta["scope"]`` -- a layer-only draft has no full-model arrays and is skipped here
instead of failing on a missing key.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from model.loading import load_qwen35_model
from model.qwen35_model import Qwen35State
from scripts.make_golden import resolve_snapshot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN = ROOT / "tests" / "golden" / "qwen35-0-8b" / "0002"
GOLDEN = Path(os.environ.get("MINILLM_W6_GOLDEN") or DEFAULT_GOLDEN)

pytestmark = [pytest.mark.slow, pytest.mark.hf]

#: The short prompt the prefill-vs-decode check uses (36 tokens). ``m02`` is excluded:
#: 240 one-token decodes of the full vocabulary would make the gate minutes long.
DECODE_PROMPT = "s01"
#: Only a prompt this short is worth decoding token by token (s01 = 36, m02 = 240).
MAX_DECODE_TOKENS = 64


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tolerances() -> dict:
    section = tomllib.loads((ROOT / "configs" / "thresholds.toml").read_text())["W6"]["gdn"]
    required = {
        "basis_golden",
        "layer_hidden_max_abs",
        "full_logits_max_abs",
        "greedy_tokens_exact",
        "decode_prefill_atol",
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
    if "full" not in meta.get("scope", []):
        pytest.skip(
            f"{GOLDEN} has scope={meta.get('scope')!r}; the full-model test needs 'full' "
            "(regenerate with a current scripts/make_golden_qwen35_w6.py)"
        )
    # P-57: only the committed fixture is required to be clean. A /tmp draft is generated
    # from an untracked script/test pair, so dirty=true is expected there.
    if GOLDEN.resolve() == DEFAULT_GOLDEN.resolve():
        assert meta["source"]["dirty"] is False, (
            f"{GOLDEN} was generated from a dirty tree (P-57): {meta['source']}"
        )
    arrays = dict(np.load(GOLDEN / "tensors.npz"))
    return {"meta": meta, "arrays": arrays, "dir": GOLDEN}


@pytest.fixture(scope="module")
def model(golden) -> torch.nn.Module:
    """The production hybrid model, loaded fp32/CPU/eval from the snapshot in meta."""
    meta = golden["meta"]
    try:
        snapshot, _ = resolve_snapshot(meta["model"]["repo"], meta["model"]["revision"])
    except SystemExit as exc:  # resolve_snapshot raises SystemExit when the cache is cold
        pytest.skip(str(exc))
    assert meta["model"]["dtype"] == "float32", (
        f"{GOLDEN} stores {meta['model']['dtype']} references; the absolute tolerances "
        "assume CPU fp32"
    )
    assert meta["model"]["attn_implementation"] == "eager", meta["model"]
    return load_qwen35_model(snapshot, dtype=torch.float32, device="cpu").eval()


# ------------------------------------------------------------------- model capture


def _prefill_with_hidden(model, ids: torch.Tensor, state: Qwen35State):
    """One prefill: ``(hidden (n_layer, T, hidden), logits (1, 1, vocab))``.

    The hooks capture the block output (residual included), which is what the golden
    stores; a GDN block returns a ``(x, conv_state, recurrent_state)`` tuple and a
    full-attention block a bare tensor, so both are normalised here.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [layer.register_forward_hook(hook) for layer in model.layers]
    try:
        with torch.no_grad():
            logits, _ = model(ids, state, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == len(model.layers), (
        f"captured {len(captured)} block outputs, model has {len(model.layers)} layers"
    )
    return torch.stack([row[0] for row in captured]), logits


def _top2_margin(logits: torch.Tensor) -> float:
    """Top-1 minus top-2 for one step (P-55: recorded, never used to relax the gate)."""
    top2 = logits.topk(2).values
    return float(top2[0] - top2[1])


def _greedy_stepwise(model, state: Qwen35State, prompt_logits: torch.Tensor, steps: int):
    """First id from the prefill logits, then ``steps - 1`` one-token decodes.

    EOS is not special-cased: exactly ``steps`` ids come back, matching the fixture's
    fixed-length gate (P-55).
    """
    ids = [int(prompt_logits.argmax())]
    margins = [_top2_margin(prompt_logits)]
    for _ in range(steps - 1):
        token = torch.tensor([[ids[-1]]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(token, state, logits_to_keep=1)
        step_logits = logits[0, -1]
        ids.append(int(step_logits.argmax()))
        margins.append(_top2_margin(step_logits))
    return ids, margins


def _prefill_vs_decode(model, ids: torch.Tensor) -> tuple[float, int]:
    """``(max|Δ|, worst position)`` between a full prefill and T one-token decodes.

    Both runs start from a fresh ``Qwen35State`` and go through ``Qwen35Model.forward``,
    so the only difference is the GDN/attention dispatch (chunked prefill vs recurrent
    decode, P-56). ``logits_to_keep=None`` keeps every position of the prefill.
    """
    with torch.no_grad():
        prefill, _ = model(ids, Qwen35State(model.config), logits_to_keep=None)
        state = Qwen35State(model.config)
        step_logits = []
        for position in range(ids.shape[1]):
            logits, _ = model(ids[:, position : position + 1], state, logits_to_keep=1)
            step_logits.append(logits[0, 0])
    stepwise = torch.stack(step_logits)  # (T, vocab)
    assert prefill.shape == (1, ids.shape[1], stepwise.shape[-1]), (
        f"prefill logits {tuple(prefill.shape)} vs stepwise {tuple(stepwise.shape)}"
    )
    per_position = (prefill[0] - stepwise).abs().amax(dim=-1)
    return per_position.max().item(), int(per_position.argmax())


# ------------------------------------------------------------------------ printing


def _print_hidden_table(prompt_id: str, per_layer: list[float], target: float) -> None:
    print(
        f"\n  {prompt_id}: layer_hidden vs golden "
        f"(block output, residual included; target {target:.1e})"
    )
    print(f"{'layer':>5} {'max_abs':>11} {'target':>10}  verdict")
    for layer_idx, value in enumerate(per_layer):
        print(
            f"{layer_idx:>5} {value:>11.3e} {target:>10.1e}  "
            f"{'pass' if value <= target else 'FAIL'}"
        )


def _print_summary(rows: list[dict]) -> None:
    print(
        f"\n{'prompt':>7} {'hidden worst':>13} {'layer':>6} {'logits':>11} "
        f"{'greedy':>9} {'prefill-decode':>14}  verdict"
    )
    for row in rows:
        decode = "-" if row["decode"] is None else f"{row['decode']:.3e}"
        print(
            f"{row['prompt']:>7} {row['hidden']:>13.3e} {row['hidden_layer']:>6} "
            f"{row['logits']:>11.3e} {row['greedy']:>5}/{row['steps']:<3} "
            f"{decode:>14}  {'pass' if row['ok'] else 'FAIL'}"
        )


# -------------------------------------------------------------------------- tests


def test_full_model_matches_golden(model, golden, tolerances):
    meta, arrays = golden["meta"], golden["arrays"]
    full = meta["full_model"]
    hidden_tol = float(tolerances["layer_hidden_max_abs"])
    logits_tol = float(tolerances["full_logits_max_abs"])
    steps = int(tolerances["greedy_tokens_exact"])
    decode_tol = float(tolerances["decode_prefill_atol"])

    assert full["n_layer"] == len(model.layers) == model.config.n_layer, (
        f"golden n_layer={full['n_layer']}, src model n_layer={len(model.layers)}, "
        f"config n_layer={model.config.n_layer}"
    )
    assert list(full["layer_types"]) == list(model.config.layer_types), (
        "golden layer_types disagree with the snapshot's LLLF schedule"
    )
    assert int(full["greedy_steps"]) == steps, (
        f"golden captured {full['greedy_steps']} greedy steps, [W6.gdn] requires {steps}"
    )
    assert full["hidden_rule"] == (
        "decoder layer output, residual included, hook on model.layers[i]"
    ), f"golden hidden_rule={full['hidden_rule']!r}; this test compares block outputs"

    # Prefer s01 (36 tokens) for the O(T) full-vocabulary decode check; never m02.
    candidates = [
        prompt
        for prompt in meta["prompts"]
        if prompt["id"] != "m02" and int(prompt["n_tokens"]) <= MAX_DECODE_TOKENS
    ]
    decode_prompt = next(
        (prompt for prompt in candidates if prompt["id"] == DECODE_PROMPT),
        candidates[0] if candidates else None,
    )
    if decode_prompt is None:
        print(
            f"\n  prefill-vs-decode: skipped, no prompt with <= {MAX_DECODE_TOKENS} tokens "
            f"(the fixture has {[p['id'] for p in meta['prompts']]})"
        )
    else:
        print(
            f"\n  prefill-vs-decode: {decode_prompt['id']} "
            f"({decode_prompt['n_tokens']} tokens, target {decode_tol:.1e})"
        )

    print(
        f"\nW6 golden {golden['dir']} (spec {meta['spec']}, scope {meta['scope']}, "
        f"{meta['model']['dtype']}/{meta['model']['device']}); tolerance basis "
        f"{tolerances['basis_golden']}"
    )

    rows: list[dict] = []
    failures: list[str] = []
    for prompt in meta["prompts"]:
        prompt_id = prompt["id"]
        ids = torch.from_numpy(arrays[f"{prompt_id}/input_ids"]).long().unsqueeze(0)
        assert ids.shape[1] == prompt["n_tokens"], (
            f"{prompt_id}: input_ids has {ids.shape[1]} tokens, meta says "
            f"{prompt['n_tokens']}"
        )
        sampled = torch.from_numpy(arrays[f"{prompt_id}/positions"]).long()
        expected_positions = sorted({0, prompt["n_tokens"] // 2, prompt["n_tokens"] - 1})
        assert (
            sampled.tolist() == expected_positions == list(prompt["sampled_positions"])
        ), (
            f"{prompt_id}: positions {sampled.tolist()} (npz) / "
            f"{prompt['sampled_positions']} (meta) disagree with the rule "
            f"{expected_positions} (sorted({{0, T // 2, T - 1}}))"
        )
        ref_hidden = torch.from_numpy(arrays[f"{prompt_id}/layer_hidden"]).float()
        ref_logits = torch.from_numpy(arrays[f"{prompt_id}/logits_last"]).float()
        assert ref_logits.shape == (model.config.vocab_size,), (
            f"{prompt_id}: golden logits_last is {tuple(ref_logits.shape)}, expected "
            f"({model.config.vocab_size},)"
        )
        ref_ids = np.asarray(arrays[f"{prompt_id}/greedy_ids"], dtype=np.int64)
        ref_margins = arrays[f"{prompt_id}/greedy_margins"].astype(np.float64)
        assert ref_ids.shape == (steps,), (
            f"{prompt_id}: golden greedy_ids is {ref_ids.shape}, expected ({steps},)"
        )
        assert ref_margins.shape == (steps,), (
            f"{prompt_id}: golden greedy_margins is {ref_margins.shape}, expected "
            f"({steps},)"
        )
        assert ref_hidden.shape == (len(model.layers), sampled.numel(), model.config.n_embd), (
            f"{prompt_id}: golden layer_hidden is {tuple(ref_hidden.shape)}, expected "
            f"{(len(model.layers), sampled.numel(), model.config.n_embd)}"
        )

        state = Qwen35State(model.config)
        hidden_all, logits = _prefill_with_hidden(model, ids, state)
        assert logits.shape == (1, 1, model.config.vocab_size), (
            f"{prompt_id}: prefill logits are {tuple(logits.shape)}, expected "
            f"(1, 1, {model.config.vocab_size})"
        )
        hidden = hidden_all.index_select(1, sampled)  # (n_layer, 3, hidden)

        per_layer = (hidden - ref_hidden).abs().amax(dim=(1, 2)).tolist()
        _print_hidden_table(prompt_id, per_layer, hidden_tol)
        worst_layer = int(np.argmax(per_layer))
        worst_hidden = per_layer[worst_layer]
        if worst_hidden > hidden_tol:
            failures.append(
                f"{prompt_id}: layer_hidden layer {worst_layer} max|Δ|={worst_hidden:.3e} "
                f"> {hidden_tol:.1e}; per-layer "
                f"{[f'{value:.1e}' for value in per_layer]}"
            )

        logits_max_abs = (logits[0, -1] - ref_logits).abs().max().item()
        if logits_max_abs > logits_tol:
            failures.append(
                f"{prompt_id}: last-position logits max|Δ|={logits_max_abs:.3e} > "
                f"{logits_tol:.1e} (max|ref|={ref_logits.abs().max().item():.3e})"
            )

        greedy_ids, greedy_margins = _greedy_stepwise(model, state, logits[0, -1], steps)
        got_ids = np.asarray(greedy_ids, dtype=np.int64)
        greedy_match = int((got_ids == ref_ids).sum())
        if greedy_match != steps:
            step = int(np.nonzero(got_ids != ref_ids)[0][0])
            failures.append(
                f"{prompt_id}: greedy diverges at step {step}: golden "
                f"{int(ref_ids[step])}, got {int(got_ids[step])}; our top-2 margin "
                f"{greedy_margins[step]:.3e}, golden margin {ref_margins[step]:.3e} "
                f"({greedy_match}/{steps} ids match)"
            )

        decode_max_abs = None
        decode_position = None
        if decode_prompt is not None and prompt_id == decode_prompt["id"]:
            decode_max_abs, decode_position = _prefill_vs_decode(model, ids)
            if decode_max_abs > decode_tol:
                failures.append(
                    f"{prompt_id}: prefill vs stepwise decode max|Δ|="
                    f"{decode_max_abs:.3e} at position {decode_position} > "
                    f"{decode_tol:.1e}"
                )

        row_ok = (
            worst_hidden <= hidden_tol
            and logits_max_abs <= logits_tol
            and greedy_match == steps
            and (decode_max_abs is None or decode_max_abs <= decode_tol)
        )
        rows.append(
            {
                "prompt": prompt_id,
                "hidden": worst_hidden,
                "hidden_layer": worst_layer,
                "logits": logits_max_abs,
                "greedy": greedy_match,
                "steps": steps,
                "decode": decode_max_abs,
                "decode_position": decode_position,
                "min_margin": min(greedy_margins),
                "golden_min_margin": float(ref_margins.min()),
                "ok": row_ok,
            }
        )

    _print_summary(rows)
    print(
        f"\n{'prompt':>7} {'min margin (ours)':>18} {'min margin (golden)':>20}"
    )
    for row in rows:
        print(
            f"{row['prompt']:>7} {row['min_margin']:>18.3e} "
            f"{row['golden_min_margin']:>20.3e}"
        )

    assert not failures, (
        f"W6 full-model alignment failed on {len(failures)} check(s); first: "
        f"{failures[0]}\n" + "\n".join(failures)
    )
