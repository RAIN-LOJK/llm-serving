"""W6 pre-registration: the GDN thresholds are frozen before the fixture exists.

The plan (``docs/w6-plan.md`` §6.3) registers these numbers *before* the first real
measurement. This file is the guard against a later "fit the line to the result" edit:
if a value changes, the diff has to explain why in D8 and in ``ROADMAP.md`` §13.

The existence check is deliberately conditional on ``status == "calibrated"``: Steps 0-4
run before ``0002`` is generated and committed, so the fast gate must stay green without
the fixture.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "configs" / "thresholds.toml"

NUMERIC = {
    "operator_atol": 1e-5,
    "operator_rtol": 1e-5,
    "state_atol": 1e-5,
    "conv_atol": 1e-6,
    "causal_atol": 0.0,
    "proj_rel_max": 1e-5,
    "conv_rel_max": 1e-5,
    "norm_rel_max": 1e-5,
    "gdn_out_rel_max": 1e-5,
    "layer_out_rel_max": 1e-4,
    "layer_hidden_max_abs": 1e-2,
    "full_logits_max_abs": 0.05,
    "greedy_tokens_exact": 64,
    "decode_prefill_atol": 1e-4,
}


@pytest.fixture(scope="module")
def gdn() -> dict:
    table = tomllib.loads(THRESHOLDS.read_text())
    assert "W6" in table and "gdn" in table["W6"], "configs/thresholds.toml has no [W6.gdn]"
    return table["W6"]["gdn"]


def test_gdn_section_is_preregistered(gdn):
    assert gdn["status"] in {"target", "calibrated"}
    assert gdn["basis_golden"] == "tests/golden/qwen35-0-8b/0002"


@pytest.mark.parametrize("name, expected", NUMERIC.items())
def test_gdn_values_are_not_fitted(gdn, name, expected):
    assert name in gdn, f"[W6.gdn] is missing {name}"
    assert math.isclose(float(gdn[name]), float(expected), rel_tol=0.0, abs_tol=0.0), (
        f"[W6.gdn].{name} changed from the pre-registered {expected!r} to {gdn[name]!r}; "
        "relaxing it needs a D8 revision and a ROADMAP §13 entry, before the formal golden"
    )


def test_calibration_cannot_precede_the_fixture(gdn):
    """A `calibrated` status must point at a committed fixture, not at a /tmp draft."""
    if gdn["status"] != "calibrated":
        assert "calibrated" not in gdn or not gdn["calibrated"]
        return
    assert gdn.get("calibrated"), "calibrated status needs a `calibrated` date"
    basis = ROOT / gdn["basis_golden"]
    assert (basis / "meta.json").is_file(), f"{basis} has no meta.json"
    assert (basis / "tensors.npz").is_file(), f"{basis} has no tensors.npz"
