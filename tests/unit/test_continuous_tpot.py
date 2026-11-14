"""Keep both TPOT views: service inter-token latency and isolated decode cost.

W7 E3 B=8 exposes the difference: a 512 request can be active while another request's
8k prefill runs. ``t_finish - t_first`` charging that prefill is the *service* view
(what the caller observes); the per-request decode sum is the *decode* view (what the
architecture costs). Neither replaces the other.
"""

from __future__ import annotations

import pytest

from bench.engines.continuous import _request_row
from serve.scheduler import Request


def _finished_request(*, decoded_ms: float, steps: int) -> Request:
    req = Request(id=7, prompt_ids=[1, 2], budget=4, arrival_ts=0.0)
    req.out = [10, 11, 12, 13]
    req.state = "finished"
    req.t_first = 0.1
    req.t_finish = 0.4
    req.decode_ms = decoded_ms
    req.decode_steps = steps
    return req


def test_request_row_tpot_uses_decode_step_sum():
    row = _request_row(_finished_request(decoded_ms=6.0, steps=3), None, None, 2, 0)
    assert row["tpot_ms"] == pytest.approx(2.0)
    assert row["tpot_basis"] == "decode_steps"
    assert row["tpot_decode_ms"] == pytest.approx(2.0)
    assert row["tpot_service_ms"] == pytest.approx(100.0)  # 0.3 s / 3 tokens
    assert row["decode_ms_total"] == pytest.approx(6.0)
    assert row["decode_steps"] == 3
    assert row["e2e_ms"] == pytest.approx(400.0)


def test_request_row_marks_historical_wall_fallback():
    """Rows without an accumulator keep the old formula, explicitly labelled."""
    row = _request_row(_finished_request(decoded_ms=0.0, steps=0), None, None, 2, 0)
    assert row["tpot_ms"] == pytest.approx(100.0)  # 0.3 s / 3 tokens
    assert row["tpot_basis"] == "wall"
    assert row["tpot_service_ms"] == pytest.approx(100.0)
    assert row["tpot_decode_ms"] is None
    assert row["decode_steps"] == 0


def test_request_row_has_no_tpot_for_one_token():
    req = _finished_request(decoded_ms=0.0, steps=0)
    req.out = [10]
    row = _request_row(req, None, None, 2, 0)
    assert row["tpot_ms"] is None
    assert row["tpot_service_ms"] is None
    assert row["tpot_decode_ms"] is None
    assert row["tpot_basis"] == "none"
