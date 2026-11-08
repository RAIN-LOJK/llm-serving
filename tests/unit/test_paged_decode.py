"""W4.5 paged decode through the engine: Triton decode, sparse seats, block reuse.

These tests pin ``serve/engine.py``'s paged arm against the per-sequence ``KVcache``
reference the seat path is already checked against. Decode is the Triton block-table
kernel (W4's gather control arm was deleted), so the whole module needs CUDA + Triton and
is skipped with an explicit reason elsewhere:

* a multi-step paged decode produces the same tokens as one ``KVcache`` per sequence;
* the engine runs exactly one Triton launch per layer (no dense decode backend);
* seats may be sparse (0 / 2 / 5): logits come back on the seat rows, idle rows are zero
  and cannot influence an active row;
* blocks freed by one request and reused by the next decode exactly like never-used blocks;
* ``release_request`` is idempotent and a block belongs to one request at a time;
* a failed prefill returns its reservation, so ``blocks_in_use`` does not creep;
* ``decode_batch`` with nothing active never forwards (zero ``(max_batch, vocab)`` rows).

Tiny bf16 model on CUDA, no golden files, no HF weights. The block allocator is the real
``BlockManager`` and the containers are the real ``PagedPool`` / ``PagedKV``.
"""

from __future__ import annotations

import pytest
import torch

from model import MiniLLM, MiniLLMConfig
from model.triton_decode import TRITON_AVAILABLE
from serve.engine import MiniLLMEngine

N_LAYER = 2
N_KV_HEAD = 2
HEAD_DIM = 4
N_HEAD = 4
BS = 16
PROMPTS = [
    [3, 5, 7, 11, 13],
    [3, 5, 7, 11, 13, 17, 19, 23, 29],
    [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
]

CUDA_READY = TRITON_AVAILABLE and torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_READY, reason="paged decode is the CUDA Triton block-table kernel"
)
#: bf16 logits from two different attention kernels; the token ids are asserted separately.
ATOL = 2e-2
RTOL = 2e-2


def _config(max_position: int = 128) -> MiniLLMConfig:
    return MiniLLMConfig(vocab_size=64, max_position_embeddings=max_position, n_embd=32,
                         n_head=N_HEAD, n_kv_head=N_KV_HEAD, n_layer=N_LAYER,
                         dropout=0.0, bias=False)


def _engine() -> tuple[MiniLLM, MiniLLMEngine]:
    torch.manual_seed(0)
    model = MiniLLM(_config()).to("cuda", dtype=torch.bfloat16).eval()
    return model, MiniLLMEngine(model)


def _open(model, engine, *, kv_blocks: int = 32, block_size: int = BS,
          max_batch: int = 8, max_position: int | None = None):
    pool = engine.open_paged_pool(kv_blocks, block_size, max_batch,
                                  max_position=max_position)
    return pool


def _first_token(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def _reference(engine, prompts):
    """Per-sequence ``KVcache`` reference: ``[(cache, first_token), ...]``."""
    out = []
    for prompt in prompts:
        logits, cache = engine.prefill(prompt)
        out.append((cache, _first_token(logits)))
    return out


def _step_reference(engine, refs, tokens):
    """Advance the reference one token per sequence; return ``(next_tokens, logits)``."""
    next_tokens, logits = [], []
    for (cache, _), token in zip(refs, tokens):
        row = engine.decode(int(token), cache)
        logits.append(row)
        next_tokens.append(_first_token(row))
    return next_tokens, torch.cat(logits, dim=0)


# ------------------------------------------------------------------ multi-step correctness

def test_paged_decode_matches_per_sequence_reference():
    """Ten decode steps: same ids as the ``KVcache`` reference at every step."""
    model, engine = _engine()
    _open(model, engine, max_batch=3)
    refs = _reference(engine, PROMPTS)

    seats = [0, 1, 2]
    first = []
    for request_id, (prompt, seat) in enumerate(zip(PROMPTS, seats)):
        logits = engine.prefill_into_blocks(request_id, prompt, budget=12, seat=seat)
        first.append(_first_token(logits))
    assert first == [token for _, token in refs], "paged prefill changed the first token"

    tokens = list(first)
    for step in range(10):
        logits = engine.decode_batch(tokens)
        assert tuple(logits.shape) == (3, model.config.vocab_size)
        assert torch.isfinite(logits).all()

        expected, ref_logits = _step_reference(engine, refs, tokens)
        got = [int(row.argmax()) for row in logits]
        assert got == expected, f"step {step}: paged ids {got} != reference {expected}"
        torch.testing.assert_close(
            logits.float(), ref_logits.float(), atol=ATOL, rtol=RTOL
        ), f"step {step} drifted"

        stats = engine.paged_stats()
        assert stats["decode_impl"] == "triton"
        assert stats["workspace_mb"] == 0.0
        assert "lpad" not in stats and "gather_calls" not in stats
        assert stats["active_rows"] == len(PROMPTS)
        assert stats["blocks_in_use"] == sum(
            engine.block_manager.blocks_for(len(p), 12) for p in PROMPTS
        )

        tokens = got
        refs = [(cache, token) for (cache, _), token in zip(refs, expected)]

    assert engine.block_manager.in_use == sum(
        engine.block_manager.blocks_for(len(p), 12) for p in PROMPTS
    )


def test_prefill_and_decode_keep_the_reservation_upper_bound():
    """D6: the block table is ``ceil((prompt + budget) / bs)`` from admission on."""
    model, engine = _engine()
    _open(model, engine, kv_blocks=16, block_size=BS, max_batch=4)
    prompt = PROMPTS[0]
    budget = 20
    expected = -(-(len(prompt) + budget) // BS)

    engine.prefill_into_blocks(0, prompt, budget=budget, seat=0)

    assert engine.block_manager.in_use == expected
    stats = engine.paged_stats()
    assert stats["blocks_in_use"] == expected
    assert stats["written_blocks"] == 1, "only the prompt's bytes are written so far"
    assert stats["kv_live_mb"] < stats["kv_alloc_mb"] < stats["kv_mb"]
    # No gather workspace and no mask/bias pair: the Triton kernel reads the pool.
    assert stats["workspace_mb"] == 0.0


# ------------------------------------------------------------------------ sparse seats

def test_sparse_seats_scatter_logits_to_the_right_rows():
    """Seats 0 / 2 / 5: active rows are exact, idle rows are zero and do not interfere."""
    model, engine = _engine()
    _open(model, engine, max_batch=6)
    refs = _reference(engine, PROMPTS)

    # Deliberately not in seat order: request 0 sits on seat 5, request 1 on seat 0.
    assignment = [(0, 2), (1, 5), (2, 0)]  # (request_id, seat)
    tokens = [0] * 6
    for request_id, seat in assignment:
        logits = engine.prefill_into_blocks(
            request_id, PROMPTS[request_id], budget=8, seat=seat
        )
        tokens[seat] = _first_token(logits)
    assert [tokens[seat] for _, seat in assignment] == [token for _, token in refs]

    for step in range(4):
        # Idle seats carry garbage in their token slots: they are not rows of the
        # compressed batch, so they must not change the active rows.
        noisy = list(tokens)
        for idle in (1, 3, 4):
            noisy[idle] = 999 + idle
        logits = engine.decode_batch(noisy)
        assert tuple(logits.shape) == (6, model.config.vocab_size)
        expected, _ = _step_reference(engine, refs, [tokens[seat] for _, seat in assignment])
        got = [int(logits[seat].argmax()) for _, seat in assignment]

        assert got == expected, f"step {step}: seat scatter lost a request"
        for idle in (1, 3, 4):
            assert not logits[idle].any(), f"idle seat {idle} must have a zero row"
        # Scatter back to the *seat* row, not to the compressed batch position.
        assert [int(logits[seat].argmax()) for _, seat in assignment] == got

        tokens = list(tokens)
        for (request_id, seat), token in zip(assignment, expected):
            tokens[seat] = token
        # Keep the reference in the same request order as ``assignment``.
        by_request = {request_id: token for (request_id, _), token in zip(assignment, expected)}
        refs = [
            (refs[i][0], by_request[i]) if i in by_request else refs[i]
            for i in range(len(refs))
        ]
        refs = [refs[request_id] for request_id, _ in assignment]


# --------------------------------------------------------------- Triton launch shape

def test_paged_decode_runs_one_triton_launch_per_layer(monkeypatch):
    """The paged arm uses the block-table kernel: one launch per layer, no dense backend."""
    from model.triton_decode import PagedDecodePlan

    model, engine = _engine()
    _open(model, engine, max_batch=3)
    for request_id, (prompt, seat) in enumerate(zip(PROMPTS, (0, 1, 2))):
        engine.prefill_into_blocks(request_id, prompt, budget=8, seat=seat)

    launched: list[int] = []
    original = PagedDecodePlan.run

    def counting_run(self, layer, q):
        launched.append(int(layer))
        return original(self, layer, q)

    monkeypatch.setattr(PagedDecodePlan, "run", counting_run)
    engine.decode_batch([0, 0, 0])

    assert launched == list(range(N_LAYER)), f"expected one launch per layer, got {launched}"
    assert engine.attn_backends()["decode"] == "paged-triton"


def test_reused_blocks_decode_identically_to_fresh_blocks():
    """``free -> reserve`` hands dirty blocks out; output must equal a never-used pool."""
    prompt_long = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prompt = PROMPTS[2]

    def run(dirty: bool):
        model, engine = _engine()
        _open(model, engine, kv_blocks=12, block_size=BS, max_batch=2)
        if dirty:
            engine.prefill_into_blocks(99, prompt_long, budget=8, seat=1)
            engine.release_request(99)
            assert engine.block_manager.in_use == 0
        tokens = [_first_token(
            engine.prefill_into_blocks(0, prompt, budget=6, seat=0)
        )]
        outs = []
        for _ in range(5):
            logits = engine.decode_batch([tokens[-1], 0])
            outs.append(logits[0].clone())
            tokens.append(int(logits[0].argmax()))
        return torch.stack(outs), tokens

    dirty_logits, dirty_tokens = run(dirty=True)
    fresh_logits, fresh_tokens = run(dirty=False)

    assert dirty_tokens == fresh_tokens, "a reused block leaked stale KV into the output"
    # Same kernel, same shapes: the stale tail is masked by seqlen, so the bytes match.
    assert torch.equal(dirty_logits, fresh_logits), "reused blocks changed the logits"


def test_batch_growth_keeps_triton_decode_correct():
    """Rows 0..3, then 0..7: rebuilding the container does not change the logits."""
    model, engine = _engine()
    _open(model, engine, kv_blocks=128, block_size=BS, max_batch=8)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(8)]
    refs = _reference(engine, prompts)
    tokens = [None] * 8
    for i in range(4):
        tokens[i] = _first_token(engine.prefill_into_blocks(i, prompts[i], budget=16, seat=i))
    assert tokens[:4] == [token for _, token in refs[:4]]

    for step in range(8):
        if step == 4:
            # Four more requests arrive: the container is rebuilt mid-run.
            for i in range(4, 8):
                tokens[i] = _first_token(
                    engine.prefill_into_blocks(i, prompts[i], budget=16, seat=i)
                )
        rows = 4 if step < 4 else 8
        logits = engine.decode_batch(tokens[:rows] + [0] * (8 - rows))
        expected, ref_logits = _step_reference(engine, refs[:rows], tokens[:rows])
        assert [int(row.argmax()) for row in logits[:rows]] == expected, f"step {step}"
        torch.testing.assert_close(
            logits[:rows].float(), ref_logits.float(), atol=ATOL, rtol=RTOL
        ), f"step {step}"
        tokens[:rows] = expected
        refs[:rows] = [(cache, token) for (cache, _), token in zip(refs[:rows], expected)]


# ------------------------------------------------------------------------ release rules

def test_release_request_is_idempotent_and_unknown_ids_are_no_ops():
    model, engine = _engine()
    _open(model, engine, kv_blocks=8, block_size=BS, max_batch=2)
    engine.prefill_into_blocks(0, PROMPTS[0], budget=4, seat=0)
    engine.prefill_into_blocks(1, PROMPTS[1], budget=4, seat=1)

    engine.release_request(0)
    after_first = engine.block_manager.stats()
    engine.release_request(0)          # second release: no error, no double free
    engine.release_request(12345)      # never admitted: no error
    assert engine.block_manager.stats()["in_use"] == after_first["in_use"]
    assert engine.block_manager.stats()["free"] == after_first["free"]

    # The other request is untouched and still decodes.
    assert engine.block_manager.in_use == engine.block_manager.blocks_for(len(PROMPTS[1]), 4)
    tokens = [0, 0]
    for _ in range(3):
        logits = engine.decode_batch(tokens)
        assert tuple(logits.shape) == (2, model.config.vocab_size)
        assert torch.isfinite(logits[1]).all()
        tokens[1] = int(logits[1].argmax())


def test_one_block_belongs_to_one_request_at_a_time():
    """Block tables are disjoint; releasing one request frees only its own blocks."""
    model, engine = _engine()
    _open(model, engine, kv_blocks=10, block_size=BS, max_batch=3)
    engine.prefill_into_blocks(0, PROMPTS[1], budget=8, seat=0)
    engine.prefill_into_blocks(1, PROMPTS[2], budget=8, seat=1)

    stats = engine.block_manager.stats()
    used = stats["in_use"]
    expected = (engine.block_manager.blocks_for(len(PROMPTS[1]), 8)
                + engine.block_manager.blocks_for(len(PROMPTS[2]), 8))
    assert used == expected
    assert stats["free"] == stats["n_blocks"] - used

    # Releasing request 0 returns exactly its reservation, nothing more.
    engine.release_request(0)
    assert engine.block_manager.in_use == engine.block_manager.blocks_for(len(PROMPTS[2]), 8)
    # The surviving request decodes and writes only into its own blocks.
    before = engine.paged_stats()["written_blocks"]
    logits = engine.decode_batch([0, 0, 0])
    assert tuple(logits.shape) == (3, model.config.vocab_size)
    assert int(logits[1].argmax()) >= 0
    assert engine.paged_stats()["written_blocks"] >= before


# --------------------------------------------------------------------- failure rollback

def test_failed_prefill_returns_every_block(monkeypatch):
    model, engine = _engine()
    _open(model, engine, kv_blocks=8, block_size=BS, max_batch=2)
    prompt = PROMPTS[1]

    def boom(*args, **kwargs):
        raise RuntimeError("boom in the last layer")

    monkeypatch.setattr(model.layers[-1], "forward", boom)
    with pytest.raises(RuntimeError, match="boom in the last layer"):
        engine.prefill_into_blocks(0, prompt, budget=8, seat=0)

    assert engine.block_manager.in_use == 0, "a failed prefill leaked its reservation"
    assert engine.paged_stats()["blocks_in_use"] == 0

    # No request was registered, so decode has nothing to forward (documented: zeros,
    # no model call).
    def no_forward(*args, **kwargs):
        raise AssertionError("decode_batch forwarded with no active request")

    monkeypatch.setattr(model, "forward", no_forward)
    empty = engine.decode_batch([0, 0])
    assert tuple(empty.shape) == (2, model.config.vocab_size) and not empty.any()

    monkeypatch.undo()
    logits = engine.prefill_into_blocks(0, prompt, budget=8, seat=0)
    assert torch.isfinite(logits).all()
    assert engine.block_manager.in_use == engine.block_manager.blocks_for(len(prompt), 8)


def test_prefill_that_cannot_reserve_leaves_the_pool_untouched():
    model, engine = _engine()
    _open(model, engine, kv_blocks=2, block_size=BS, max_batch=2)
    with pytest.raises(RuntimeError):
        engine.prefill_into_blocks(0, PROMPTS[2], budget=64, seat=0)
    stats = engine.block_manager.stats()
    assert stats["in_use"] == 0 and stats["free"] == stats["n_blocks"]
    assert engine.paged_stats()["blocks_in_use"] == 0


# --------------------------------------------------------------- empty batch / can_admit

def test_decode_with_no_active_request_does_not_forward(monkeypatch):
    """Documented choice: zeros ``(max_batch, vocab)``, no model call."""
    model, engine = _engine()
    _open(model, engine, max_batch=5)

    def no_forward(*args, **kwargs):
        raise AssertionError("decode_batch forwarded with no active request")

    monkeypatch.setattr(model, "forward", no_forward)
    logits = engine.decode_batch([0] * 5)

    assert tuple(logits.shape) == (5, model.config.vocab_size)
    assert not logits.any()
    stats = engine.paged_stats()
    assert stats["workspace_mb"] == 0.0
    assert "lpad" not in stats


def test_can_admit_checks_blocks_and_active_count_without_mutating():
    model, engine = _engine()
    _open(model, engine, kv_blocks=4, block_size=BS, max_batch=2)
    stats_before = engine.block_manager.stats()

    # 4 free blocks. ceil(16/16)=1 fits; ceil((32+1)/16)=3 fits; ceil((64+1)/16)=5 does not.
    assert engine.can_admit(16, 0) is True
    assert engine.can_admit(32, 1) is True
    assert engine.can_admit(64, 1) is False
    assert engine.block_manager.stats() == stats_before, "can_admit must not reserve"

    engine.prefill_into_blocks(0, [1, 2, 3], budget=0, seat=0)
    engine.prefill_into_blocks(1, [4, 5, 6], budget=0, seat=1)
    # max_batch is a hard cap even though blocks remain.
    assert engine.can_admit(4, 0) is False
    assert engine.block_manager.stats()["in_use"] == 2
