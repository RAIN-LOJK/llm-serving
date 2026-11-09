"""W5's threshold table is a pre-registration, not a result to be re-fitted.

W5 is a numerical gate (single layer vs the HF golden), so it never goes through
``bench.thresholds.evaluate``. The table is enforced here instead: the values below are
the ones written in ``docs/w5-plan.md`` §5 **before** the fixture existed, and the formal
golden came in at ``0.0`` on every stage without changing them. A future edit that
loosens a line turns this file red, which is the point: a wrong gate order, norm
convention or RoPE width is an O(1) relative error, so none of these lines needs to move.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: docs/w5-plan.md §5, pre-registered 2026-11-09.
PREREGISTERED = {
    "proj_rel_max": 1e-5,
    "norm_rel_max": 1e-5,
    "rope_rel_max": 1e-5,
    "attn_gated_rel_max": 1e-5,
    "layer_out_rel_max": 1e-4,
    "causal_atol": 0.0,
}
BASIS_GOLDEN = "tests/golden/qwen35-0-8b/0001"


@pytest.fixture(scope="module")
def section() -> dict:
    table = tomllib.loads((ROOT / "configs" / "thresholds.toml").read_text())
    assert "W5" in table and "layer" in table["W5"], "configs/thresholds.toml lost [W5.layer]"
    return table["W5"]["layer"]


def test_preregistered_values_are_unchanged(section):
    for name, expected in PREREGISTERED.items():
        assert name in section, f"[W5.layer] is missing {name}"
        assert section[name] == pytest.approx(expected), (
            f"[W5.layer].{name} = {section[name]!r}, pre-registered {expected!r}; "
            "loosening this line needs a new plan/ADR, not an edit here"
        )


def test_status_points_at_the_committed_golden(section):
    assert section["status"] == "calibrated", (
        "W5 is closed; status must be 'calibrated' (the basis is the committed golden, "
        "the values were not fitted)"
    )
    assert section["basis_golden"] == BASIS_GOLDEN
    assert section.get("calibrated") == "2026-11-09"


def test_basis_golden_exists(section):
    golden = ROOT / section["basis_golden"]
    assert (golden / "meta.json").exists(), f"missing W5 golden meta at {golden}"
    assert (golden / "tensors.npz").exists(), f"missing W5 golden tensors at {golden}"


def test_lines_are_tight_enough_to_catch_wiring_errors(section):
    """A wrong norm convention / gate order / rotary width is O(1) relative.

    These ceilings are deliberately far below that: a line of 1e-2 would pass a
    plausible-looking implementation that is wrong in the third decimal.
    """
    for name in ("proj_rel_max", "norm_rel_max", "rope_rel_max", "attn_gated_rel_max"):
        assert section[name] <= 1e-4
    assert section["layer_out_rel_max"] <= 1e-3
    assert section["causal_atol"] == 0.0, "the causal test runs fp32 CPU; it must be exact"
