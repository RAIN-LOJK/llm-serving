"""W7 Step 0 / G0: bf16/CUDA greedy 64/64 against HF in the same dtype and device.

The formal serving path is bf16 on the 4090, so the numerical gate is **not** the W6
fp32/CPU golden: it is the same checkpoint in the same dtype on the same device
(D9-7, ``docs/w7-plan.md`` §2.4). This file compares, for ``s01`` and ``m02``:

* **Qwen3.5-0.8B**: our ``load_qwen35_model(..., dtype=bfloat16, device="cuda")``
  against the HF *text decoder* loaded from the same snapshot via
  ``config.get_text_config()`` (vision tower and MTP excluded, P-14), bf16/CUDA/eager;
* **Qwen3-0.6B**: our ``load_checkpoint(...)`` against the HF causal LM, bf16/CUDA/eager,
  on the **same** ``s01``/``m02`` token ids. They come from the committed W6 fixture
  (max id 147117 < the 0.6B vocab 151936), so both models see one fixed input instead of
  two tokenizations.

Greedy is a fixed 64 steps: step 0 from the prefill logits, then 63 one-token decodes
through an explicit cache (``Qwen35State`` / ``KVcache`` on our side, ``past_key_values``
on HF's). EOS is never special-cased (P-55). Every step's top1-top2 margin is recorded;
on a mismatch the first differing step and both margins are reported. The gate is strict
equality **64/64**: margins explain a near-tie, they never relax it. For 0.8B the W6
fp32 golden greedy ids are printed as a diagnostic only -- they are not the pass
condition.

    MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen35_bf16.py -q -s

Requires CUDA and both pinned snapshots in the local HF cache; no network is used. The
module-scoped fixtures keep the runtime bounded (two prompts x 64 steps x two models,
each checkpoint loaded once per test). ``tests/conftest.py`` pins transformers' Qwen3.5
kernels to their torch fallbacks for the whole session (P-15/P-48), so the HF reference
is the pinned torch path, the same contract the W6 golden used -- not the optional
``fla``/``causal_conv1d`` CUDA fast path.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch

from model import KVcache
from model.loading import cached_snapshot, load_checkpoint, load_qwen35_model
from model.qwen35_model import Qwen35State

ROOT = Path(__file__).resolve().parents[2]
W6_GOLDEN = ROOT / "tests" / "golden" / "qwen35-0-8b" / "0002"

Q35_REPO = "Qwen/Qwen3.5-0.8B"
Q35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
Q3_REPO = "Qwen/Qwen3-0.6B"
Q3_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

PROMPTS = ("s01", "m02")
STEPS = 64
#: Reported next to the min margin; not a gate and not a relaxation.
NEAR_TIE = 1e-3

pytestmark = [
    pytest.mark.hf,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("MINILLM_RUN_HF") != "1",
        reason="set MINILLM_RUN_HF=1 to load the pinned checkpoints",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="G0 is a bf16/CUDA gate; no CUDA device",
    ),
]


class Greedy(NamedTuple):
    ids: list[int]
    margins: list[float]


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def w6_fixture() -> dict:
    """The committed W6 s01/m02 ids (and fp32 greedy ids, for diagnostics only)."""
    meta_path = W6_GOLDEN / "meta.json"
    tensors_path = W6_GOLDEN / "tensors.npz"
    if not meta_path.is_file() or not tensors_path.is_file():
        pytest.skip(
            f"no W6 golden at {W6_GOLDEN}; G0 reuses its s01/m02 token ids"
        )
    meta = json.loads(meta_path.read_text())
    arrays = dict(np.load(tensors_path))
    n_tokens = {prompt["id"]: int(prompt["n_tokens"]) for prompt in meta["prompts"]}
    return {"meta": meta, "arrays": arrays, "n_tokens": n_tokens}


@pytest.fixture(scope="module")
def prompt_inputs(w6_fixture) -> dict[str, torch.Tensor]:
    arrays, n_tokens = w6_fixture["arrays"], w6_fixture["n_tokens"]
    inputs = {}
    for prompt in PROMPTS:
        ids = torch.from_numpy(arrays[f"{prompt}/input_ids"]).long().unsqueeze(0)
        assert ids.shape[1] == n_tokens[prompt], (
            f"{prompt}: npz has {ids.shape[1]} ids, meta says {n_tokens[prompt]}"
        )
        inputs[prompt] = ids
    return inputs


@pytest.fixture(scope="module")
def w6_greedy(w6_fixture) -> dict[str, dict]:
    arrays = w6_fixture["arrays"]
    out = {}
    for prompt in PROMPTS:
        ids = np.asarray(arrays[f"{prompt}/greedy_ids"], dtype=np.int64)
        margins = np.asarray(arrays[f"{prompt}/greedy_margins"], dtype=np.float64)
        assert ids.shape == (STEPS,) and margins.shape == (STEPS,), (
            f"{prompt}: W6 greedy fixture is {ids.shape}/{margins.shape}, expected "
            f"({STEPS},)"
        )
        out[prompt] = {"ids": ids, "margins": margins}
    return out


# ------------------------------------------------------------------------ loaders


def _snapshot(repo: str, revision: str) -> Path:
    """Resolve the pinned revision from the local HF cache; never fall back to ``main``."""
    snapshot = cached_snapshot(repo, revision)
    if snapshot is None:
        pytest.skip(f"{repo}@{revision} is not in the local HF cache")
    assert snapshot.name == revision, f"{snapshot} does not end in revision {revision}"
    return snapshot


def _load_hf_text_decoder(snapshot: Path):
    """P-14: the *text* decoder of the multimodal checkpoint, bf16/CUDA/eager."""
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    text_config = hf_config.get_text_config()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        config=text_config,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.to("cuda").eval()
    leaked = sorted(
        name
        for name, _ in model.named_parameters()
        if name.startswith(("visual.", "mtp.", "model.visual.", "model.mtp."))
    )
    assert not leaked, f"text-only load leaked vision/MTP parameters: {leaked[:5]}"
    return model


def _load_hf_causal_lm(snapshot: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    )
    return model.to("cuda").eval()


def _release() -> None:
    """Free the previous checkpoint's blocks before loading the next one."""
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------- greedy


def _top2_margin(logits: torch.Tensor) -> float:
    """Top-1 minus top-2 (P-55: recorded, never used to relax the gate)."""
    top2 = logits.float().topk(2).values
    return float(top2[0] - top2[1])


def _greedy(step_logits, first_logits: torch.Tensor) -> Greedy:
    """First id from the prefill, then ``STEPS - 1`` one-token cache-backed decodes."""
    ids = [int(first_logits.argmax())]
    margins = [_top2_margin(first_logits)]
    for _ in range(STEPS - 1):
        token = torch.tensor([[ids[-1]]], dtype=torch.long, device="cuda")
        logits = step_logits(token)
        ids.append(int(logits.argmax()))
        margins.append(_top2_margin(logits))
    assert len(ids) == len(margins) == STEPS
    return Greedy(ids, margins)


@torch.no_grad()
def _hf_greedy(model, ids: torch.Tensor) -> Greedy:
    """HF greedy through one explicit ``DynamicCache`` (mutated in place)."""
    out = model(ids.to("cuda"), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values

    def step(token: torch.Tensor) -> torch.Tensor:
        out = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1]

    return _greedy(step, out.logits[0, -1])


@torch.no_grad()
def _our_qwen35_greedy(model, ids: torch.Tensor) -> Greedy:
    state = Qwen35State(model.config, device="cuda", dtype=torch.bfloat16)
    logits, _ = model(ids.to("cuda"), state, logits_to_keep=1)

    def step(token: torch.Tensor) -> torch.Tensor:
        logits, _ = model(token, state, logits_to_keep=1)
        return logits[0, -1]

    return _greedy(step, logits[0, -1])


@torch.no_grad()
def _our_qwen3_greedy(model, ids: torch.Tensor) -> Greedy:
    kv = KVcache(model.config.n_layer)
    logits, _ = model(ids.to("cuda"), past_keys_values=kv)

    def step(token: torch.Tensor) -> torch.Tensor:
        logits, _ = model(token, past_keys_values=kv)
        return logits[0, -1]

    return _greedy(step, logits[0, -1])


# --------------------------------------------------------------------- reporting


def _mismatch(prompt: str, got: Greedy, ref: Greedy) -> str | None:
    """First differing step with both margins; ``None`` means strict 64/64."""
    for step, (ours, hf) in enumerate(zip(got.ids, ref.ids)):
        if ours != hf:
            matched = sum(int(a == b) for a, b in zip(got.ids, ref.ids))
            return (
                f"{prompt}: greedy diverges at step {step}: ours {ours} "
                f"(top1-top2 {got.margins[step]:.3e}), HF {hf} "
                f"(top1-top2 {ref.margins[step]:.3e}); {matched}/{STEPS} ids match"
            )
    return None


def _min_margin(margins: list[float]) -> tuple[float, int]:
    step = min(range(len(margins)), key=margins.__getitem__)
    return margins[step], step


def _tie_steps(margins: list[float]) -> list[int]:
    """Steps whose top-1 and top-2 are exactly equal in bf16 (reported, not a gate)."""
    return [step for step, value in enumerate(margins) if value == 0.0]


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(
        f"{'prompt':>7} {'tok':>4} {'match':>7} {'min margin ours':>17} {'step':>5} "
        f"{'min margin HF':>15} {'step':>5} {'exact ties o/h':>15} {'near ties o/h':>14}"
    )
    for row in rows:
        print(
            f"{row['prompt']:>7} {row['tokens']:>4} {row['match']:>7} "
            f"{row['min_margin_ours']:>17.3e} {row['min_step_ours']:>5} "
            f"{row['min_margin_hf']:>15.3e} {row['min_step_hf']:>5} "
            f"{row['exact_ours']:>7}/{row['exact_hf']:<7} "
            f"{row['near_ours']:>6}/{row['near_hf']:<7}"
        )
        if row["ties_ours"] or row["ties_hf"]:
            print(
                f"  {row['prompt']}: exact-tie steps ours {row['ties_ours']} / "
                f"HF {row['ties_hf']}"
            )


def _print_w6_diagnostic(prompt: str, got: Greedy, golden: dict) -> None:
    """fp32 golden vs bf16 ids: diagnostic only, never the pass condition."""
    ref_ids, ref_margins = golden["ids"], golden["margins"]
    matched = sum(int(a == int(b)) for a, b in zip(got.ids, ref_ids))
    first = next(
        (step for step, (a, b) in enumerate(zip(got.ids, ref_ids)) if a != int(b)), None
    )
    suffix = ""
    if first is not None:
        suffix = (
            f"; first diff step {first}: bf16 {got.ids[first]} "
            f"(margin {got.margins[first]:.3e}) vs fp32 {int(ref_ids[first])} "
            f"(margin {float(ref_margins[first]):.3e})"
        )
    print(f"  {prompt}: W6 fp32 golden diagnostic {matched}/{STEPS}{suffix}")


def _row(prompt: str, tokens: int, got: Greedy, ref: Greedy) -> dict:
    min_ours, step_ours = _min_margin(got.margins)
    min_hf, step_hf = _min_margin(ref.margins)
    matched = sum(int(a == b) for a, b in zip(got.ids, ref.ids))
    return {
        "prompt": prompt,
        "tokens": tokens,
        "match": f"{matched}/{STEPS}",
        "matched": matched,
        "min_margin_ours": min_ours,
        "min_step_ours": step_ours,
        "min_margin_hf": min_hf,
        "min_step_hf": step_hf,
        "exact_ours": sum(1 for value in got.margins if value == 0.0),
        "exact_hf": sum(1 for value in ref.margins if value == 0.0),
        "ties_ours": _tie_steps(got.margins),
        "ties_hf": _tie_steps(ref.margins),
        "near_ours": sum(1 for value in got.margins if value < NEAR_TIE),
        "near_hf": sum(1 for value in ref.margins if value < NEAR_TIE),
    }


# -------------------------------------------------------------------------- tests


def test_qwen35_08b_bf16_greedy_matches_hf(prompt_inputs, w6_greedy):
    """G0 for the hybrid model: strict 64/64 on s01 and m02, bf16/CUDA."""
    snapshot = _snapshot(Q35_REPO, Q35_REVISION)
    print(
        f"\nG0 Qwen3.5-0.8B: ours bfloat16/cuda vs HF bfloat16/cuda/eager, "
        f"{Q35_REPO}@{Q35_REVISION}"
    )

    hf = _load_hf_text_decoder(snapshot)
    try:
        ref = {prompt: _hf_greedy(hf, ids) for prompt, ids in prompt_inputs.items()}
    finally:
        del hf
        _release()

    ours = load_qwen35_model(snapshot, dtype=torch.bfloat16, device="cuda").eval()
    try:
        rows, failures = [], []
        for prompt, ids in prompt_inputs.items():
            got = _our_qwen35_greedy(ours, ids)
            mismatch = _mismatch(prompt, got, ref[prompt])
            if mismatch is not None:
                failures.append(mismatch)
            rows.append(_row(prompt, ids.shape[1], got, ref[prompt]))
            _print_w6_diagnostic(prompt, got, w6_greedy[prompt])
    finally:
        del ours
        _release()

    _print_table("Qwen3.5-0.8B bf16/CUDA: ours vs HF (greedy 64, no EOS)", rows)
    assert not failures, (
        "G0 failed for Qwen3.5-0.8B: strict 64/64 is the gate (D9-7); margins explain "
        "a near-tie but do not relax it. First failure: "
        f"{failures[0]}\n" + "\n".join(failures)
    )


def test_qwen3_06b_bf16_greedy_matches_hf(prompt_inputs):
    """G0 for the pure-attention model: strict 64/64 on the same s01/m02 ids."""
    snapshot = _snapshot(Q3_REPO, Q3_REVISION)
    print(
        f"\nG0 Qwen3-0.6B: ours bfloat16/cuda vs HF bfloat16/cuda/eager, "
        f"{Q3_REPO}@{Q3_REVISION}"
    )

    hf = _load_hf_causal_lm(snapshot)
    try:
        ref = {prompt: _hf_greedy(hf, ids) for prompt, ids in prompt_inputs.items()}
    finally:
        del hf
        _release()

    ours = load_checkpoint(snapshot, dtype=torch.bfloat16, device="cuda").eval()
    try:
        vocab = int(ours.config.vocab_size)
        rows, failures = [], []
        for prompt, ids in prompt_inputs.items():
            assert int(ids.max()) < vocab, (
                f"{prompt}: token id {int(ids.max())} >= 0.6B vocab {vocab}; the "
                "shared W6 ids are not valid for this model"
            )
            got = _our_qwen3_greedy(ours, ids)
            mismatch = _mismatch(prompt, got, ref[prompt])
            if mismatch is not None:
                failures.append(mismatch)
            rows.append(_row(prompt, ids.shape[1], got, ref[prompt]))
    finally:
        del ours
        _release()

    _print_table("Qwen3-0.6B bf16/CUDA: ours vs HF (greedy 64, no EOS)", rows)
    assert not failures, (
        "G0 failed for Qwen3-0.6B: strict 64/64 is the gate (D9-7); margins explain "
        "a near-tie but do not relax it. First failure: "
        f"{failures[0]}\n" + "\n".join(failures)
    )
