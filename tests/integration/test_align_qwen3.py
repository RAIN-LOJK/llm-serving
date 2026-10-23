"""W1 acceptance: the hand-written decoder must reproduce the HF golden (ROADMAP §5).

Red until ``model/`` grows QK-Norm, an explicit ``head_dim`` and ``rope_theta``::

    MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py -q

While porting, ``python scripts/align_hf.py`` prints the per-layer ``max|Δ|`` table.
The fixture is committed under ``tests/golden/``; this test never regenerates it.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
import torch

from scripts.align_hf import (
    cached_snapshot,
    compare,
    greedy_ids,
    layer_hidden,
    load_golden,
    load_qwen3,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen3-0-6b" / "0001"

pytestmark = [pytest.mark.hf, pytest.mark.slow]


@pytest.fixture(scope="module")
def tolerances() -> dict:
    return tomllib.loads((ROOT / "configs" / "thresholds.toml").read_text())["W1"]["align"]


@pytest.fixture(scope="module")
def aligned(tolerances):
    """(model, golden) loaded once per module -- the checkpoint is 1.2 GB in fp32."""
    if os.environ.get("MINILLM_RUN_HF") != "1":
        pytest.skip("set MINILLM_RUN_HF=1 to load the 0.6B checkpoint")
    if not (GOLDEN / "meta.json").exists():
        pytest.skip(f"no golden at {GOLDEN}; generate it with scripts/make_golden.py")
    golden = load_golden(GOLDEN)
    model_path = cached_snapshot(
        golden["meta"]["model"]["repo"], golden["meta"]["model"]["revision"]
    )
    if model_path is None:
        pytest.skip("the checkpoint revision recorded in meta.json is not in the HF cache")
    return load_qwen3(model_path), golden


def prompt_ids(golden: dict, index: int) -> torch.Tensor:
    arrays = golden["arrays"]
    length = int(arrays["lengths"][index])
    return torch.from_numpy(arrays["input_ids"][index : index + 1, -length:]).long()


def left_padded_batch(golden: dict, count: int):
    """First ``count`` prompts, left-padded to a common width (PITFALLS P-05)."""
    arrays = golden["arrays"]
    lengths = arrays["lengths"][:count]
    width = int(lengths.max())
    ids = torch.zeros((count, width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row in range(count):
        length = int(lengths[row])
        ids[row, width - length :] = torch.from_numpy(arrays["input_ids"][row, -length:])
        mask[row, width - length :] = 1
    return ids, mask


def valid_prefix(golden: dict, index: int) -> torch.Tensor:
    """Golden greedy ids up to EOS (``-1`` marks steps the reference never reached)."""
    reference = torch.from_numpy(golden["arrays"]["greedy_ids"][index]).long()
    assert (reference >= 0).any()
    return reference[reference >= 0]


def test_layer_and_logits_match_golden(aligned, tolerances):
    model, golden = aligned
    layer_tol, logits_tol = tolerances["layer_max_abs_diff"], tolerances["logits_max_abs_diff"]
    result = compare(model, golden, layer_tol, logits_tol, check_greedy=False)

    per_layer = result["per_layer"]
    worst_layer = int(per_layer.argmax())
    assert per_layer[worst_layer] <= layer_tol, (
        f"layer {worst_layer}: max|Δ| = {per_layer[worst_layer]:.3e} > {layer_tol:g}\n"
        f"per-layer: {[f'{v:.1e}' for v in per_layer.tolist()]}\n"
        "first mismatching layer is the operator to fix; run `python scripts/align_hf.py`"
    )
    worst_logits = result["logits"].max().item()
    assert worst_logits <= logits_tol, (
        f"last-position logits: max|Δ| = {worst_logits:.3e} > {logits_tol:g}"
    )


def test_greedy_tokens_match(aligned):
    model, golden = aligned
    n_ids = golden["arrays"]["greedy_ids"].shape[1]
    for index in range(len(golden["meta"]["prompts"])):
        expected = valid_prefix(golden, index)
        got = greedy_ids(model, prompt_ids(golden, index), n_ids)[0]
        if torch.equal(got[: expected.numel()], expected):
            continue
        step = int((got[: expected.numel()] != expected).nonzero()[0])
        prompt = golden["meta"]["prompts"][index]
        raise AssertionError(
            f"prompt {prompt['id']}: diverges at step {step} "
            f"(golden {expected[step].item()}, got {got[step].item()})"
        )


def test_batch_sizes_agree(aligned):
    """Batch 1/2/4 must produce the same ids as the single-sequence reference."""
    model, golden = aligned
    n_ids = golden["arrays"]["greedy_ids"].shape[1]
    baseline = greedy_ids(model, prompt_ids(golden, 0), n_ids)

    for count in (2, 4):
        ids, mask = left_padded_batch(golden, count)
        batched = greedy_ids(model, ids, n_ids, attention_mask=mask)
        assert torch.equal(batched[0], baseline[0]), (
            f"batch {count} row 0 differs from batch 1 -- left pad or mask handling"
        )
        for row in range(count):
            expected = valid_prefix(golden, row)
            assert torch.equal(batched[row][: expected.numel()], expected), (
                f"batch {count} row {row} diverges from the golden"
            )


def test_causality(aligned):
    """Changing token t must not move any position < t, for every layer."""
    model, golden = aligned
    ids = prompt_ids(golden, 0)
    t = ids.shape[1] // 2
    perturbed = ids.clone()
    perturbed[0, t] = (ids[0, t] + 1) % model.config.vocab_size

    before, _ = layer_hidden(model, ids)
    after, _ = layer_hidden(model, perturbed)

    assert torch.allclose(before[:, :, :t], after[:, :, :t], atol=1e-6), (
        "a later token leaked into an earlier position"
    )
    assert not torch.allclose(before[:, :, t], after[:, :, t], atol=1e-6), (
        "perturbation was a no-op -- the causality test proves nothing"
    )
