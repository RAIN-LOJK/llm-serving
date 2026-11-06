"""``scripts/check_paged_golden.py``: the W4 E0 gate (docs/decisions.md D6 / C3).

E0 is ``record_only`` in the runner (a run picks one prompt, the fixture holds four), so
the "paged output == golden greedy ids" claim needs an out-of-band check. These tests pin
the checker itself -- the comparison must be exact-id, must not silently pass an empty or
short run, and must not compare a prompt the golden does not contain.

The fixtures here are synthetic (a tiny npz + metrics.jsonl under ``tmp_path``); the real
golden is never regenerated or modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.check_paged_golden import compare, load_golden, load_run_rows, main


def _write_golden(root: Path, ids: dict[str, list[int]]) -> Path:
    golden = root / "golden"
    golden.mkdir()
    meta = {
        "spec": "test-golden",
        "prompts": [{"id": key, "sha256": "x", "n_tokens": 3} for key in ids],
    }
    (golden / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    width = max(len(value) for value in ids.values())
    # Pad rows to a rectangle; the checker compares only the common prefix.
    rows = [list(value) + [0] * (width - len(value)) for value in ids.values()]
    np.savez_compressed(golden / "tensors.npz", greedy_ids=np.array(rows, dtype=np.int32))
    return golden


def _write_run(root: Path, rows: list[dict]) -> Path:
    run = root / "run"
    run.mkdir()
    with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return run


def _request(prompt_id: str, tokens: list[int], repeat: int = 0) -> dict:
    return {
        "row_kind": "request",
        "prompt_id": prompt_id,
        "token_ids": tokens,
        "repeat": repeat,
        "impl": "paged",
    }


def test_matching_ids_pass(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3, 4]})
    run = _write_run(tmp_path, [_request("s01", [1, 2, 3, 4])])
    _, ids = load_golden(golden)
    results = compare(load_run_rows(run), ids)
    assert [entry["status"] for entry in results] == ["pass"]
    assert results[0]["compared"] == 4
    assert main([str(run), "--golden", str(golden)]) == 0


def test_one_wrong_id_fails_and_is_located(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3, 4]})
    run = _write_run(tmp_path, [_request("s01", [1, 2, 9, 4])])
    _, ids = load_golden(golden)
    results = compare(load_run_rows(run), ids)
    assert results[0]["status"] == "fail"
    assert results[0]["first_mismatch"] == 2
    assert results[0]["expected_id"] == 3
    assert results[0]["actual_id"] == 9
    assert main([str(run), "--golden", str(golden)]) == 1


def test_short_run_compares_the_common_prefix(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3, 4]})
    run = _write_run(tmp_path, [_request("s01", [1, 2])])
    _, ids = load_golden(golden)
    results = compare(load_run_rows(run), ids)
    assert results[0]["status"] == "pass"
    assert results[0]["compared"] == 2


def test_prompt_absent_from_golden_is_skipped_not_failed(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3]})
    run = _write_run(tmp_path, [_request("m07", [1, 2, 3])])
    _, ids = load_golden(golden)
    assert compare(load_run_rows(run), ids)[0]["status"] == "skipped"


def test_no_comparable_row_is_an_error(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3]})
    run = _write_run(tmp_path, [_request("m07", [1, 2, 3])])
    # A skipped row proves nothing: the gate must not exit 0 on it.
    assert main([str(run), "--golden", str(golden)]) == 2


def test_only_the_first_repeat_is_read(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3]})
    run = _write_run(
        tmp_path,
        [
            _request("s01", [1, 2, 3], repeat=0),
            _request("s01", [9, 9, 9], repeat=1),
        ],
    )
    _, ids = load_golden(golden)
    results = compare(load_run_rows(run), ids)
    assert len(results) == 1
    assert results[0]["status"] == "pass"


def test_missing_token_ids_is_reported_not_passed(tmp_path):
    golden = _write_golden(tmp_path, {"s01": [1, 2, 3]})
    row = _request("s01", [])
    row["token_ids"] = []
    run = _write_run(tmp_path, [row])
    _, ids = load_golden(golden)
    assert compare(load_run_rows(run), ids)[0]["status"] == "missing"
    assert main([str(run), "--golden", str(golden)]) == 2


def test_real_golden_loads_and_has_s01():
    """Guard against the fixture's shape drifting out from under the checker."""
    root = Path(__file__).resolve().parents[2] / "tests" / "golden" / "qwen3-0-6b" / "0001"
    if not (root / "meta.json").exists():
        pytest.skip("golden fixture not present")
    _, ids = load_golden(root)
    assert "s01" in ids
    assert len(ids["s01"]) == 64
