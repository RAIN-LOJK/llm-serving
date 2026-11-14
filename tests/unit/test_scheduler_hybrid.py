"""W7 Step 2: the **same** ``Scheduler`` drives the hybrid engine, seat and paged.

Two halves:

* CPU/seat: the tiny 24-layer model runs through ``Qwen35Engine(pool="seat")`` under
  ``Scheduler(max_batch=2, policy="continuous")``. The scheduler's output stream must equal
  a manual per-request greedy loop on a dense ``Qwen35State``. This is the "one scheduler,
  no model branch" check: ``serve/scheduler.py`` must not mention the model at all, and the
  seat arm must retire through ``freeze_seat`` (rows end frozen, state pool empty).
* CUDA/Triton: the same trace through ``pool="paged"`` must produce the same tokens and
  must retire through ``release_request`` -- the scheduler's ``engine.paged`` flag is what
  selects that hook. ``freeze_seat`` is installed as a trap and must never run. Blocks and
  slots must be fully returned and ``leak`` must be false.

The tiny model is CPU fp32 for the seat test and CUDA **bf16** for the paged one: since
W7 Step 3 the group prefill runs through the flash-attn varlen arm, which computes
fp16/bf16 only (D9-7's formal dtype), and the dense reference is the same bf16 model, so
the comparison is like-for-like; no weights are loaded, and the paged test skips cleanly
without CUDA/Triton.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.config import MiniLLMConfig
from model.qwen35_model import Qwen35Model, Qwen35State
from model.triton_decode import TRITON_AVAILABLE
from serve.hybrid_engine import Qwen35Engine
from serve.scheduler import Request, Scheduler

TINY = dict(
    vocab_size=64,
    max_position_embeddings=64,
    n_embd=32,
    n_head=4,
    n_kv_head=2,
    n_layer=24,
    n_inner=64,
    dropout=0.0,
    bias=False,
    norm_eps=1e-6,
    head_dim=8,
    rope_theta=1e4,
    qk_norm=True,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    tie_word_embeddings=True,
)

REQUESTS = [
    (0, [3, 5, 7, 11, 13], 4),
    (1, [2, 4, 6, 8, 10, 12, 14, 16, 18], 3),
]
SCHEDULER_SOURCE = Path(__file__).resolve().parents[2] / "serve" / "scheduler.py"


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model(
    device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Qwen35Model:
    torch.manual_seed(0)
    return Qwen35Model(_config()).to(device=device, dtype=dtype).eval()


def _dense_greedy(model: Qwen35Model, prompt: list[int], budget: int) -> list[int]:
    """The scheduler's per-request reference: first token at prefill, then decode."""
    device = str(next(model.parameters()).device)
    state = Qwen35State(model.config, device=device)
    with torch.no_grad():
        logits, _ = model(
            torch.tensor([prompt], dtype=torch.long, device=device),
            state,
            logits_to_keep=1,
        )
    token = int(logits[0, -1].argmax())
    out = [token]
    while len(out) < budget:
        with torch.no_grad():
            logits, _ = model(
                torch.tensor([[token]], dtype=torch.long, device=device),
                state,
                logits_to_keep=1,
            )
        token = int(logits[0, -1].argmax())
        out.append(token)
    return out


def _submit_all(scheduler: Scheduler) -> None:
    for request_id, prompt, budget in REQUESTS:
        scheduler.submit(
            Request(
                id=request_id,
                prompt_ids=list(prompt),
                budget=budget,
                arrival_ts=0.0,
            )
        )
    scheduler.close()


def _assert_manual_outputs(model, finished) -> None:
    by_id = {request.id: request for request in finished}
    for request_id, prompt, budget in REQUESTS:
        request = by_id[request_id]
        assert request.state == "finished", request.error
        assert request.error is None
        assert request.out == _dense_greedy(model, prompt, budget), (
            f"request {request_id}: scheduler tokens differ from the dense reference"
        )


# ------------------------------------------------------------------ seat arm


def test_seat_hybrid_runs_under_the_same_scheduler_to_completion():
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=2, max_len=32, linear_slots=2, pool="seat")
    assert engine.paged is False

    scheduler = Scheduler(engine, max_batch=2, policy="continuous")
    _submit_all(scheduler)
    finished = scheduler.run()

    _assert_manual_outputs(model, finished)
    assert scheduler.active == [] and scheduler.queued == []
    # The seat arm retires by freezing rows; both seats are parked and the pool is empty.
    assert engine.slab.active.tolist() == [False, False]
    stats = engine.paged_stats()
    assert stats["active_rows"] == 0
    assert stats["linear_state_mb"] == 0.0
    assert stats["decode_impl"] == "hand_baddbmm"
    assert stats["blocks_in_use"] == 0
    assert stats["leak"] is False

    pool = engine.state_pool
    engine.close_pool()
    engine.close_pool()
    pool.assert_no_leaks()


def test_scheduler_has_no_model_branch_and_reads_engine_paged():
    source = SCHEDULER_SOURCE.read_text(encoding="utf-8")
    assert "Qwen35" not in source
    assert "qwen35" not in source
    assert "isinstance(engine" not in source
    # The one model-shaped decision is the engine's own ``paged`` flag.
    assert 'getattr(self.engine, "paged", None)' in source
    assert "def _paged_mode" in source
    # Retirement dispatches on that flag, never on a model class.
    assert "release_request" in source and "freeze_seat" in source


# ------------------------------------------------------------------ paged arm


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="paged hybrid decode is CUDA + Triton only (there is no dense paged fallback)",
)
def test_paged_hybrid_scheduler_matches_dense_and_releases_blocks():
    model = _model("cuda", torch.bfloat16)
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(
        max_batch=2, max_len=32, linear_slots=2, kv_blocks=16, block_size=4, pool="paged"
    )
    assert engine.paged is True

    released: list[int] = []
    real_release = engine.release_request

    def counting_release(request_id):
        released.append(int(request_id))
        return real_release(request_id)

    def forbidden_freeze(seat):  # pragma: no cover - must never run on the paged arm
        raise AssertionError(f"paged scheduler called freeze_seat({seat})")

    engine.release_request = counting_release
    engine.freeze_seat = forbidden_freeze

    scheduler = Scheduler(engine, max_batch=2, policy="continuous")
    _submit_all(scheduler)
    finished = scheduler.run()

    _assert_manual_outputs(model, finished)
    assert sorted(released) == [0, 1], (
        "the paged arm must retire through release_request, driven by engine.paged"
    )
    stats = engine.paged_stats()
    assert stats["active_rows"] == 0
    assert stats["linear_state_mb"] == 0.0
    assert stats["blocks_in_use"] == 0
    assert stats["free_blocks"] == 16
    assert stats["leak"] is False
    # Every decode step row carried the paged account (the scheduler attaches it).
    step_rows = [row for row in scheduler.step_rows if row["active"]]
    assert step_rows and all(row["decode_impl"] == "triton" for row in step_rows)
    assert all("blocks_in_use" in row for row in step_rows)

    pool = engine.state_pool
    engine.close_pool()
    engine.close_pool()
    pool.assert_no_leaks()
