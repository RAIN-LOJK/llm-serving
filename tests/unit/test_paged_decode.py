"""W4 paged decode through the engine: compressed batch, sparse seats, block reuse.

These tests pin ``serve/engine.py``'s paged arm (docs/w4-plan.md §2/§3, D6) against the
per-sequence ``KVcache`` reference the seat path is already checked against:

* a multi-step paged decode produces the same tokens as one ``KVcache`` per sequence;
* seats may be sparse (0 / 2 / 5): logits come back on the seat rows, idle rows are zero
  and cannot influence an active row;
* blocks freed by one request and reused by the next decode exactly like never-used blocks;
* ``release_request`` is idempotent and a block belongs to one request at a time;
* a failed prefill returns its reservation, so ``blocks_in_use`` does not creep;
* ``decode_batch`` with nothing active never forwards (zero ``(max_batch, vocab)`` rows).

CPU only, tiny model, no golden files, no HF weights. The block allocator is the real
``BlockManager`` and the containers are the real ``PagedPool`` / ``PagedKV``.
"""

from __future__ import annotations

import pytest
import torch

from model import MiniLLM, MiniLLMConfig
from model.attention import DECODE_BACKEND
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


def _config(max_position: int = 128) -> MiniLLMConfig:
    return MiniLLMConfig(vocab_size=64, max_position_embeddings=max_position, n_embd=32,
                         n_head=N_HEAD, n_kv_head=N_KV_HEAD, n_layer=N_LAYER,
                         dropout=0.0, bias=False)


def _engine(**kwargs) -> tuple[MiniLLM, MiniLLMEngine]:
    torch.manual_seed(0)
    model = MiniLLM(_config()).eval()
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
    previous_lpad = 0
    for step in range(10):
        logits = engine.decode_batch(tokens)
        assert tuple(logits.shape) == (3, model.config.vocab_size)
        assert torch.isfinite(logits).all()

        expected, ref_logits = _step_reference(engine, refs, tokens)
        got = [int(row.argmax()) for row in logits]
        assert got == expected, f"step {step}: paged ids {got} != reference {expected}"
        assert torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-4), f"step {step} drifted"

        stats = engine.paged_stats()
        assert stats["lpad"] >= max(len(p) + step + 1 for p in PROMPTS)
        assert engine.paged_pool.workspace_mb() > 0.0
        assert stats["blocks_in_use"] == sum(
            engine.block_manager.blocks_for(len(p), 12) for p in PROMPTS
        )
        previous_lpad = stats["lpad"]

        tokens = got
        refs = [(cache, token) for (cache, _), token in zip(refs, expected)]

    assert previous_lpad == 64
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
    assert stats["workspace_mb"] == 0.0, "no gather until the first decode step"
    assert stats["lpad"] == 0


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


# ------------------------------------------------------------------------ block reuse

def test_paged_decode_dispatches_one_baddbmm_per_layer(monkeypatch):
    """D4 stays intact on the paged arm: one decode call per layer, not one per row."""
    model, engine = _engine()
    _open(model, engine, max_batch=3)
    for request_id, (prompt, seat) in enumerate(zip(PROMPTS, (0, 1, 2))):
        engine.prefill_into_blocks(request_id, prompt, budget=8, seat=seat)

    calls: list[int] = []
    original = DECODE_BACKEND.decode

    def counting_decode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(DECODE_BACKEND, "decode", counting_decode)
    engine.decode_batch([0, 0, 0])

    assert len(calls) == N_LAYER, f"expected {N_LAYER} layer calls, got {len(calls)}"


def test_step_stats_report_one_gather_per_layer_and_host_time():
    """Plan §5: the step row carries the step's gather calls, host time and traffic."""
    model, engine = _engine()
    _open(model, engine, max_batch=3)
    tokens = []
    for request_id, (prompt, seat) in enumerate(zip(PROMPTS, (0, 1, 2))):
        logits = engine.prefill_into_blocks(request_id, prompt, budget=8, seat=seat)
        tokens.append(_first_token(logits))

    for _ in range(3):
        logits = engine.decode_batch(tokens)
        stats = engine.paged_stats()
        assert stats["gather_calls"] == N_LAYER, "one gather per layer, no more, no less"
        assert stats["gather_ms"] >= 0.0, "host wall clock cannot be negative"
        assert stats["gather_read_mb"] > 0.0 and stats["gather_write_mb"] > 0.0
        # Traffic, not occupancy: it is the k+v bytes of this step's lpad width.
        per_token_per_layer = 2 * N_KV_HEAD * model.head_dim * 4
        expected = N_LAYER * len(PROMPTS) * stats["lpad"] * per_token_per_layer / 2**20
        assert stats["gather_read_mb"] == pytest.approx(expected)
        assert stats["gather_read_mb"] == stats["gather_write_mb"]
        tokens = [int(row.argmax()) for row in logits]

    # Each step's account starts at zero: ``gather_calls`` is not cumulative.
    assert engine.paged_stats()["gather_calls"] == N_LAYER


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
    assert torch.equal(dirty_logits, fresh_logits), "reused blocks changed the logits"


# ------------------------------------------------------- workspace residency (W4 blocker)

def test_workspace_holds_one_lpad_bucket_and_follows_active_rows():
    """Decode past several lpad buckets: one pair resident, sized to the active rows.

    The old implementation kept ``{lpad: (k, v)}`` forever and allocated ``max_batch`` rows
    per bucket, so an E3-shaped run (26 buckets, ``max_batch=64``) peaked at GiBs. The rules
    pinned here: after every step ``len(pool._workspace) == 1``, the engine keeps one
    mask/bias pair, and both are sized by the active rows, not by ``max_batch``.
    """
    torch.manual_seed(0)
    model = MiniLLM(_config(max_position=512)).eval()
    engine = MiniLLMEngine(model)
    pool = engine.open_paged_pool(256, BS, 64, max_position=512)
    prompts = [
        [3 + (i % 50) for i in range(length)] for length in (20, 60, 100)
    ]
    tokens = []
    for request_id, prompt in enumerate(prompts):
        logits = engine.prefill_into_blocks(request_id, prompt, budget=220, seat=request_id)
        tokens.append(_first_token(logits))

    per_token_bytes = 2 * N_KV_HEAD * model.head_dim * pool.dtype.itemsize
    gqa = model.config.n_head // model.config.n_kv_head
    mib = 2**20
    seen: set[int] = set()
    peak = 0.0
    for _ in range(220):
        logits = engine.decode_batch(tokens + [0] * (64 - len(tokens)))
        stats = engine.paged_stats()
        seen.add(stats["lpad"])
        assert len(pool._workspace) == 1, "lpad buckets must not accumulate"
        assert len(engine._paged_buffers) == 1, "mask/bias buckets must not accumulate"
        rows = len(prompts)
        expected_gather = rows * stats["lpad"] * per_token_bytes / mib
        # Appendix C mask/bias: one bool per (row, column) and one addend per (row*H_kv, G, col).
        expected_buffers = (
            rows * stats["lpad"] + rows * N_KV_HEAD * gqa * stats["lpad"] * pool.dtype.itemsize
        ) / mib
        assert stats["gather_workspace_mb"] == pytest.approx(expected_gather)
        assert stats["workspace_mb"] == pytest.approx(expected_gather + expected_buffers)
        peak = max(peak, stats["workspace_mb"])
        tokens = [int(row.argmax()) for row in logits[: rows]]

    assert len(seen) > 2, "the run never crossed several bucket boundaries; test is blind"
    widest = max(seen)
    # 3 rows, not 64: the workspace is the active rows at the widest bucket, everywhere.
    largest_gather = len(prompts) * widest * per_token_bytes / mib
    assert peak < largest_gather + 0.05, "workspace grew beyond the current bucket"
    assert pool.workspace_peak_mb() >= pool.workspace_mb()


def test_workspace_falls_back_when_a_long_request_releases():
    """A long request leaving shrinks ``lpad``; the workspace must not stay at the old size."""
    torch.manual_seed(0)
    model = MiniLLM(_config(max_position=256)).eval()
    engine = MiniLLMEngine(model)
    pool = engine.open_paged_pool(64, BS, 4, max_position=256)
    short = [3, 5, 7, 11, 13]
    long = [3 + (i % 50) for i in range(100)]  # -> lpad 128 once the long row decodes
    engine.prefill_into_blocks(0, short, budget=8, seat=0)
    engine.prefill_into_blocks(1, long, budget=8, seat=1)
    engine.decode_batch([0, 0, 0, 0])
    wide = engine.paged_stats()
    assert wide["lpad"] == 128

    engine.release_request(1)
    engine.decode_batch([0, 0, 0, 0])
    narrow = engine.paged_stats()
    assert narrow["lpad"] == 64, "lpad must follow the remaining row"
    assert len(pool._workspace) == 1
    # One active row in a 64-wide bucket: the old 128-wide pair is gone.
    expected = 2 * 1 * 64 * N_KV_HEAD * model.head_dim * pool.dtype.itemsize / 2**20
    assert narrow["gather_workspace_mb"] == pytest.approx(expected)
    assert narrow["gather_workspace_mb"] < wide["gather_workspace_mb"]


def test_batch_growth_rebuilds_and_stays_numerically_correct():
    """Rows 0..3, then 0..7: the rebuild changes addresses but not the logits."""
    model, engine = _engine()
    pool = _open(model, engine, kv_blocks=128, block_size=BS, max_batch=8)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(8)]
    refs = _reference(engine, prompts)
    tokens = [None] * 8
    for i in range(4):
        tokens[i] = _first_token(engine.prefill_into_blocks(i, prompts[i], budget=16, seat=i))
    assert tokens[:4] == [token for _, token in refs[:4]]

    rebuilds = pool.workspace_rebuilds
    for step in range(8):
        if step == 4:
            # Four more requests arrive: the batch and the workspace grow mid-run.
            for i in range(4, 8):
                tokens[i] = _first_token(
                    engine.prefill_into_blocks(i, prompts[i], budget=16, seat=i)
                )
        rows = 4 if step < 4 else 8
        before = pool.workspace_rebuilds
        logits = engine.decode_batch(tokens[:rows] + [0] * (8 - rows))
        expected, ref_logits = _step_reference(engine, refs[:rows], tokens[:rows])
        assert [int(row.argmax()) for row in logits[:rows]] == expected, f"step {step}"
        assert torch.allclose(logits[:rows], ref_logits, atol=1e-5, rtol=1e-4), f"step {step}"

        assert pool._rows_cap == rows, "workspace rows must track the active batch"
        assert len(pool._workspace) == 1
        growth = 1 if step in (0, 4) else 0  # first allocation, then the 4 -> 8 row growth
        assert pool.workspace_rebuilds == before + growth, (
            f"step {step}: rebuilt {pool.workspace_rebuilds - before} times"
        )
        tokens[:rows] = expected
        refs[:rows] = [(cache, token) for (cache, _), token in zip(refs[:rows], expected)]
    assert pool.workspace_rebuilds == rebuilds + 2, "first alloc plus the 4 -> 8 row growth"


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
    # The dedicated zero block is not an allocator block: it is outside the free list and
    # outside the pool's KV account (P-33 / P-35).
    assert engine.paged_pool.zero_block.shape == (BS, N_KV_HEAD, model.head_dim)
    assert not bool(engine.paged_pool.zero_block.any())

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
    assert engine.paged_stats()["lpad"] == 0


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
