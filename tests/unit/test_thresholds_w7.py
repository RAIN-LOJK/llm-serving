"""W7 pre-registration: lock ``[W7.state]`` and ``[W7.hybrid]`` before any W7 run.

``docs/decisions.md`` D9 and ``docs/w7-plan.md`` §5.7 freeze these numbers in Step 0.
No W7 result exists yet, so both sections must stay ``target`` with an empty
``basis_run``; a ``calibrated`` table without a ``W7-####`` basis is the failure mode
this file guards against. The numeric values are never fitted: changing one requires a
D9 revision (and a re-registration before the next formal run), not a threshold edit.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "configs" / "thresholds.toml"

#: The complete Step-0 tables. Key sets are compared exactly, so an accidental extra
#: key (for example a `calibrated` date) fails before it can be read as evidence.
EXPECTED = {
    "state": {
        "status": "target",
        "basis_run": "",
        "state_estimate_error_max": 0.20,
    },
    "hybrid": {
        "status": "target",
        "basis_run": "",
        "greedy_tokens_exact": 64,
        "p1_state_ratio_8k_min": 5.0,
        "capacity_speedup_512_min": 2.0,
        "capacity_speedup_2k_min": 4.0,
        "capacity_speedup_8k_min": 6.0,
        "memory_crossover_ctx_max": 256,
        "p2_latency_material_ratio": 1.10,
        "hybrid_tpot_growth_max": 1.10,
        "pure_tpot_growth_min": 1.15,
    },
}
LEGAL_STATUS = {"target", "calibrated"}
NUMERIC = [
    (section, name, expected)
    for section, values in EXPECTED.items()
    for name, expected in values.items()
    if name not in {"status", "basis_run"}
]


@pytest.fixture(scope="module")
def w7() -> dict:
    table = tomllib.loads(THRESHOLDS.read_text())
    assert "W7" in table, "configs/thresholds.toml has no [W7.*] tables"
    return table["W7"]


@pytest.mark.parametrize("section", sorted(EXPECTED))
def test_section_has_exactly_the_preregistered_keys(w7, section):
    assert section in w7, f"configs/thresholds.toml has no [W7.{section}]"
    actual = set(w7[section])
    expected = set(EXPECTED[section])
    assert actual == expected, (
        f"[W7.{section}] keys changed: missing {sorted(expected - actual)}, "
        f"extra {sorted(actual - expected)}; this table is frozen by D9 and "
        "docs/w7-plan.md §5.7"
    )


@pytest.mark.parametrize("section", sorted(EXPECTED))
def test_status_is_legal(w7, section):
    status = w7[section]["status"]
    assert status in LEGAL_STATUS, (
        f"[W7.{section}].status={status!r}; legal values are {sorted(LEGAL_STATUS)}"
    )


@pytest.mark.parametrize("section,name,expected", NUMERIC)
def test_values_are_the_preregistered_numbers(w7, section, name, expected):
    actual = w7[section][name]
    assert type(actual) is type(expected), (
        f"[W7.{section}].{name} is {type(actual).__name__} ({actual!r}), expected "
        f"{type(expected).__name__} ({expected!r}); a changed TOML type is still a "
        "changed pre-registration"
    )
    assert math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=0.0), (
        f"[W7.{section}].{name} changed from the pre-registered {expected!r} to "
        f"{actual!r}; W7 values may only be calibrated at closeout (status/basis_run), "
        "never fitted or lowered"
    )


def test_state_table_is_unchanged(w7):
    """Step 0 must not touch ``[W7.state]``; no W7 evidence exists to calibrate it."""
    state = w7["state"]
    assert state["status"] == "target", (
        f"[W7.state].status is {state['status']!r}; Step 0 keeps the existing target. "
        "Only the W7 closeout may set 'calibrated' with a W7-#### basis_run."
    )
    assert state["basis_run"] == "", (
        f"[W7.state].basis_run is {state['basis_run']!r}; no W7 run exists at Step 0"
    )


def test_hybrid_is_preregistered_not_calibrated(w7):
    """A calibrated hybrid table at Step 0 would be a fit without a run."""
    hybrid = w7["hybrid"]
    assert hybrid["status"] == "target", (
        f"[W7.hybrid].status is {hybrid['status']!r}; Step 0 pre-registers 'target'. "
        "A 'calibrated' status requires a W7-#### basis_run and a D9 closeout revision."
    )
    assert hybrid["basis_run"] == "", (
        f"[W7.hybrid].basis_run is {hybrid['basis_run']!r}; Step 0 has no run yet"
    )
