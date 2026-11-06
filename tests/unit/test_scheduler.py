"""Scheduler contract tests (docs/decisions.md D5) on a fake engine and clock.

No torch, no weights: the engine is duck-typed exactly like the contract the real
engine will implement (``prefill_into_seat`` / ``decode_batch`` / ``freeze_seat`` /
``slab``). The fake clock advances only when the engine "runs"
or when the scheduler sleeps, so "step k perturbs step k+1" is observable to the
instruction.

The fake engine is deterministic by construction: a request's token stream is a
function of *its own* prompt and of how many decode writes its seat has taken
since activation, never of the seat index or of another request. That is what
makes ``static`` and ``continuous`` produce the same tokens (P-23) while still
letting the seat be recycled.
"""

from __future__ import annotations

from collections import Counter

import pytest

from serve.scheduler import IDLE_TOKEN, Request, Scheduler


# ---------------------------------------------------------------- fakes
class FakeClock:
    """A clock that only moves when someone advances or sleeps on it."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)
        self.sleeps: list[float] = []
        self.now_calls = 0

    def now(self) -> float:
        self.now_calls += 1
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0, f"scheduler slept a negative delay: {seconds}"
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeSlab:
    def __init__(self, batch: int) -> None:
        self.batch = batch


class FakeEngine:
    """Deterministic stand-in for ``serve/engine.py``'s W3 interface."""

    impl = "continuous"

    def __init__(
        self,
        clock: FakeClock,
        max_batch: int,
        *,
        vocab_size: int = 64,
        eos_token_id: int | None = None,
        max_position: int = 512,
        prefill_s: float = 0.001,
        decode_s: float = 0.002,
        first_token=None,
        next_token=None,
    ) -> None:
        self.clock = clock
        self.slab = FakeSlab(max_batch)
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.max_position = max_position
        self.device = "fake"
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self._first_token = first_token or self._default_first
        self._next_token = next_token or self._default_next

        self.occupied: set[int] = set()
        self.seat_prompt: dict[int, tuple[int, ...]] = {}
        self.writes: dict[int, int] = {}
        self.prefill_calls: list[tuple[int, ...]] = []
        self.activations: list[tuple[int, tuple[int, ...]]] = []
        self.freezes: list[int] = []
        self.release_freezes: list[int] = []
        self.decode_calls = 0
        self.decode_tokens: list[list[int]] = []
        self.occupied_seen = 0
        self.occupied_at_decode: list[set[int]] = []
        self.last_decode_event_ms: float | None = None

    # token rules: request-local only (prompt + write index + previous token)
    def _default_first(self, prompt_ids: list[int]) -> int:
        return (prompt_ids[0] * 31 + len(prompt_ids)) % self.vocab_size

    def _default_next(self, prev: int, writes: int, prompt_ids: tuple[int, ...]) -> int:
        return (prev * 3 + writes + 1 + len(prompt_ids)) % self.vocab_size

    def _one_hot(self, token: int) -> list[float]:
        row = [0.0] * self.vocab_size
        row[token] = 1.0
        return row

    # -------------------------------------------------- engine interface
    def prefill_one(self, prompt_ids: list[int]):
        assert prompt_ids, "empty prompt reached the engine"
        assert len(prompt_ids) <= self.max_position
        self.clock.advance(self.prefill_s)
        prompt = tuple(prompt_ids)
        self.prefill_calls.append(prompt)
        return [self._one_hot(self._first_token(list(prompt)))], {
            "prompt": prompt,
            "length": len(prompt),
        }

    def activate_seat(self, slot: int, cache, prompt_len: int) -> None:
        assert 0 <= slot < self.slab.batch, f"seat {slot} outside the slab"
        assert slot not in self.occupied, f"seat {slot} activated while still occupied"
        assert cache["length"] == prompt_len, "activate_seat got the wrong length"
        self.occupied.add(slot)
        self.seat_prompt[slot] = cache["prompt"]
        self.writes[slot] = 0
        self.activations.append((slot, cache["prompt"]))
        self.occupied_seen = max(self.occupied_seen, len(self.occupied))

    def prefill_into_seat(self, slot: int, prompt_ids: list[int]):
        """W3 admission: prefill + seat write in one call, as ``serve/engine.py`` does.

        Deliberately self-contained: the scheduler must not be able to fall back to the W2
        two-step hand-off (``prefill_one`` + ``activate_seat``), and a test poisons both to
        pin that down.
        """
        assert prompt_ids, "empty prompt reached the engine"
        assert len(prompt_ids) <= self.max_position
        assert 0 <= slot < self.slab.batch, f"seat {slot} outside the slab"
        assert slot not in self.occupied, f"seat {slot} activated while still occupied"
        self.clock.advance(self.prefill_s)
        prompt = tuple(prompt_ids)
        self.prefill_calls.append(prompt)
        self.occupied.add(slot)
        self.seat_prompt[slot] = prompt
        self.writes[slot] = 0
        self.activations.append((slot, prompt))
        self.occupied_seen = max(self.occupied_seen, len(self.occupied))
        return [self._one_hot(self._first_token(list(prompt)))]

    def freeze_seat(self, slot: int) -> None:
        # ``KvSlab.freeze`` is idempotent and safe on a row that never activated
        # (pos/valid reset to 0, active=False), so this does not assert occupancy:
        # the scheduler parks never-used seats once at the start, and a failed admission
        # can leave a half-written row (the engine freezes it before re-raising).
        assert 0 <= slot < self.slab.batch, f"seat {slot} outside the slab"
        self.freezes.append(slot)
        if slot in self.occupied:
            self.occupied.discard(slot)
            self.release_freezes.append(slot)

    def decode_batch(self, tokens, slab) -> list[list[float]]:
        assert slab is self.slab, "decode_batch got a slab other than engine.slab"
        assert len(tokens) == slab.batch, f"{len(tokens)} tokens for {slab.batch} rows"
        self.clock.advance(self.decode_s)
        self.decode_calls += 1
        self.decode_tokens.append(list(tokens))
        self.occupied_at_decode.append(set(self.occupied))
        self.occupied_seen = max(self.occupied_seen, len(self.occupied))

        rows = []
        for slot in range(slab.batch):
            if slot in self.occupied:
                prev = int(tokens[slot])
                token = self._next_token(prev, self.writes[slot], self.seat_prompt[slot])
                self.writes[slot] += 1
                rows.append(self._one_hot(token))
            else:
                # Idle rows are still forwarded; the scheduler must use the documented
                # placeholder so an idle row can never look like real work.
                assert tokens[slot] == IDLE_TOKEN, f"idle row {slot} got {tokens[slot]}"
                rows.append(self._one_hot(IDLE_TOKEN))
        return rows


# ---------------------------------------------------------------- helpers
def _scheduler(engine, max_batch, policy="continuous", clock=None):
    clock = clock or engine.clock
    return Scheduler(
        engine,
        max_batch=max_batch,
        policy=policy,
        now=clock.now,
        sleep=clock.sleep,
    )


def _request(idx: int, budget: int, arrival: float = 0.0, prompt_len: int = 3) -> Request:
    # The first prompt id is a unique marker so the fake engine can tell requests
    # apart when it logs prefills.
    return Request(
        id=idx,
        prompt_ids=[100 + idx] + [(idx + k) % 7 for k in range(prompt_len - 1)],
        budget=budget,
        arrival_ts=arrival,
    )


def _drain(sched: Scheduler) -> None:
    while sched.queued or sched.active:
        sched.step()


def _by_id(requests: list[Request]) -> dict[int, Request]:
    return {req.id: req for req in requests}


# ---------------------------------------------------------------- scheduler contract
def test_capacity_never_exceeded():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=3)
    sched = _scheduler(engine, max_batch=3)
    for i in range(10):
        sched.submit(_request(i, budget=4, arrival=i * 0.01))

    steps = 0
    while sched.queued or sched.active:
        row = sched.step()
        steps += 1
        assert row["active"] <= 3, row
        assert len(sched.active) <= 3
        assert len(engine.occupied) <= 3
        assert engine.occupied_seen <= 3
    assert steps > 0
    assert len(sched.step_rows) == steps
    assert [r["step"] for r in sched.step_rows] == list(range(steps))


def test_each_request_finishes_once():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2)
    requests = [_request(i, budget=3 + (i % 3), arrival=i * 0.005) for i in range(7)]
    for req in requests:
        sched.submit(req)

    result = sched.run()

    assert [req.id for req in result] == sorted(req.id for req in requests)
    assert all(req.state == "finished" for req in result)
    assert all(req.t_finish is not None for req in result)
    # Exactly one first token plus budget-1 decodes: no duplicate, no lost token.
    assert all(len(req.out) == req.budget for req in result)
    # Each request prefilled exactly once and froze its seat exactly once.
    assert sorted(call[0] for call in engine.prefill_calls) == sorted(100 + r.id for r in requests)
    assert len(engine.prefill_calls) == len(requests)
    # Every request freezes its own seat exactly once; a never-used seat may be parked
    # once as well (it starts live in a freshly reset slab).
    assert len(engine.release_freezes) == len(requests)
    assert set(engine.freezes) == set(engine.release_freezes)
    assert not engine.occupied


def test_each_request_holds_one_seat():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2)
    for i in range(6):
        sched.submit(_request(i, budget=3, arrival=i * 0.005))

    while sched.queued or sched.active:
        sched.step()
        seats = [req.seat for req in sched.active]
        assert None not in seats
        assert len(set(seats)) == len(seats) == len(sched.active)
        assert all(0 <= seat < 2 for seat in seats)
        # A request that is decoding must hold a seat the engine considers occupied.
        assert set(seats) <= engine.occupied

    uses = Counter(seat for seat, _ in engine.activations)
    assert len(uses) <= 2, uses
    assert sum(uses.values()) == 6
    assert max(uses.values()) >= 2, "seats must be recycled by later requests"


def test_static_runs_waves():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=3)
    sched = _scheduler(engine, max_batch=3, policy="static")
    budgets = [5, 6, 7, 2, 3, 4, 1]
    requests = [_request(i, budget=budgets[i]) for i in range(len(budgets))]
    for req in requests:
        sched.submit(req)

    result = sched.run()
    rows = sched.step_rows

    assert all(req.state == "finished" for req in result)
    assert max(row["active"] for row in rows) == 3
    assert engine.occupied_seen == 3
    # A wave must end (active back to 0) while work is still queued.
    assert any(row["active"] == 0 and row["queued"] > 0 for row in rows)
    # Full waves first, a partial (closed-trace) last wave.
    assert [row["admitted"] for row in rows if row["admitted"]] == [3, 3, 1]
    # No admission while a wave is running.
    for prev, row in zip(rows, rows[1:]):
        if row["admitted"]:
            assert prev["active"] == 0, (prev, row)


def test_continuous_refills_next_step():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2, policy="continuous")
    short = _request(0, budget=2)
    long_ = _request(1, budget=8)
    third = _request(2, budget=3)
    for req in (short, long_, third):
        sched.submit(req)

    first_row = sched.step()
    assert first_row["admitted"] == 2
    assert short.state == "finished"
    assert third.state == "queued" and third.t_admitted is None

    t_step2 = clock.t
    second_row = sched.step()

    assert second_row["admitted"] == 1
    assert third.state == "decoding"
    assert third.t_admitted == pytest.approx(t_step2, abs=1e-12)
    assert third.t_admitted >= short.t_finish
    assert [req.id for req in sched.active] == [long_.id, third.id]


def test_lifecycle_timestamps_monotone():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2)
    for i in range(6):
        # Four requests at t=0 overload max_batch=2 (real queueing); two arrive later
        # (the run must sleep for them rather than drop them).
        sched.submit(_request(i, budget=2 + (i % 4), arrival=0.0 if i < 4 else 0.05))

    result = sched.run()
    assert all(req.state == "finished" for req in result)

    queued_positive = False
    prefill_total_ms = 0.0
    for req in result:
        assert req.t_arrival <= req.t_admitted
        assert req.t_admitted <= req.t_prefill_start
        assert req.t_prefill_start <= req.t_first
        assert req.t_first <= req.t_finish

        queue_ms = (req.t_prefill_start - req.t_arrival) * 1000.0
        prefill_ms = (req.t_first - req.t_prefill_start) * 1000.0
        ttft_ms = (req.t_first - req.t_arrival) * 1000.0
        assert queue_ms >= 0.0 and prefill_ms >= 0.0
        # C2: the identity is structural, not a coincidence.
        assert ttft_ms == pytest.approx(queue_ms + prefill_ms, abs=1e-9)
        assert prefill_ms == pytest.approx(engine.prefill_s * 1000.0, abs=1e-9)
        prefill_total_ms += prefill_ms
        queued_positive = queued_positive or queue_ms > 0.0

    assert queued_positive, "staggered arrivals must queue somewhere"
    # Every step row decomposes the same way; host_gap may be anything (even negative).
    for row in sched.step_rows:
        assert row["host_gap_ms"] == pytest.approx(
            row["step_ms"] - row["decode_event_ms"] - row["prefill_ms"], abs=1e-9
        )
    assert sum(row["prefill_ms"] for row in sched.step_rows) == pytest.approx(
        prefill_total_ms, abs=1e-9
    )


def test_budget_eos_and_max_position_terminate():
    # --- budget
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=1)
    sched = _scheduler(engine, max_batch=1)
    req = _request(0, budget=4)
    sched.submit(req)
    finished = sched.run()[0]
    assert finished.state == "finished"
    assert len(finished.out) == 4
    assert finished.t_finish is not None

    # --- EOS: the EOS token terminates and no token follows it
    eos = 5
    clock = FakeClock()
    engine = FakeEngine(
        clock,
        max_batch=1,
        eos_token_id=eos,
        next_token=lambda prev, writes, prompt: eos,
    )
    sched = _scheduler(engine, max_batch=1)
    req = _request(0, budget=10)
    sched.submit(req)
    finished = sched.run()[0]
    assert finished.state == "finished"
    assert finished.out[-1] == eos
    assert len(finished.out) == 2, "EOS must stop sampling, not just be logged"
    assert finished.out[0] != eos

    # --- max_position: prompt_len + out_tokens hits the position table's end
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=1, max_position=6)
    sched = _scheduler(engine, max_batch=1)
    req = _request(0, budget=100, prompt_len=4)
    sched.submit(req)
    finished = sched.run()[0]
    assert finished.state == "finished"
    assert len(finished.out) == 2, "4 prompt + 2 generated == max_position 6"
    assert len(finished.out) <= finished.budget


def test_arrival_order_invariance():
    budgets = {i: 2 + (i % 3) for i in range(6)}
    arrivals = {i: i * 0.01 for i in range(6)}

    def run(order, times) -> dict[int, Request]:
        clock = FakeClock()
        engine = FakeEngine(clock, max_batch=2)
        sched = _scheduler(engine, max_batch=2)
        for i in order:
            sched.submit(
                Request(
                    id=i,
                    prompt_ids=[50 + i, i, 3],
                    budget=budgets[i],
                    arrival_ts=times[i],
                )
            )
        return _by_id(sched.run())

    base = run(range(6), arrivals)
    reversed_submission = run(list(reversed(range(6))), arrivals)
    # Same requests, different *arrival order* (the times travel with the ids).
    permuted = {0: 0.05, 1: 0.04, 2: 0.03, 3: 0.02, 4: 0.01, 5: 0.0}
    permuted_arrivals = run(range(6), permuted)

    for i in range(6):
        assert base[i].state == "finished"
        assert base[i].out == reversed_submission[i].out, f"id {i}: submission order changed tokens"
        assert base[i].out == permuted_arrivals[i].out, f"id {i}: arrival order changed tokens"


def test_static_and_continuous_ids_equal():
    budgets = [2, 5, 3, 4, 2, 6, 3, 4]
    arrivals = [i * 0.01 for i in range(len(budgets))]

    def run(policy: str) -> dict[int, list[int]]:
        clock = FakeClock()
        engine = FakeEngine(clock, max_batch=3)
        sched = _scheduler(engine, max_batch=3, policy=policy)
        for i, (budget, arrival) in enumerate(zip(budgets, arrivals)):
            sched.submit(_request(i, budget=budget, arrival=arrival))
        result = sched.run()
        assert all(req.state == "finished" for req in result)
        return {req.id: req.out for req in result}

    assert run("static") == run("continuous")


# ---------------------------------------------------------------- extra self-checks
def test_poisson_arrival_is_waited_for():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2)
    for i in range(4):
        sched.submit(_request(i, budget=2, arrival=i * 0.25))

    result = sched.run()

    assert all(req.state == "finished" for req in result)
    assert clock.sleeps, "future arrivals must be slept on, not spun on"
    assert all(seconds > 0.0 for seconds in clock.sleeps)
    # 0.75 s of arrivals, minus the engine time already spent when each one came due.
    assert sum(clock.sleeps) == pytest.approx(0.75, abs=0.02)
    assert clock.now_calls < 200, "too many now() calls for three sleeps: busy-waiting?"
    assert all(req.t_arrival >= 0.0 for req in result)
    assert all(req.t_admitted >= req.t_arrival for req in result)


def test_static_opens_last_wave_when_trace_ends():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=4)
    sched = _scheduler(engine, max_batch=4, policy="static")
    requests = [_request(i, budget=3) for i in range(3)]
    for req in requests:
        sched.submit(req)

    # Not closed and short of max_batch: the wave must not open yet.
    idle = sched.step()
    assert idle["admitted"] == 0
    assert not sched.active

    sched.close()
    wave = sched.step()
    assert wave["admitted"] == 3, "a partial last wave must open once the trace ends"
    assert len(sched.active) == 3

    _drain(sched)
    assert all(req.state == "finished" for req in requests)
    assert not clock.sleeps, "nothing was in the future; no sleep is needed"


def test_idle_rows_get_the_placeholder_token():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=3)
    sched = _scheduler(engine, max_batch=3, policy="static")
    # Two requests, max_batch 3: the wave keeps one idle row for its whole life.
    requests = [_request(i, budget=4) for i in range(2)]
    for req in requests:
        sched.submit(req)
    sched.close()

    rows = []
    while sched.queued or sched.active:
        rows.append(sched.step())

    # One forward per step over the whole slab, never a per-active-row batch.
    assert all(len(tokens) == 3 for tokens in engine.decode_tokens)
    idle_seats = {0, 1, 2} - engine.occupied_at_decode[0]
    assert len(idle_seats) == 1
    idle = idle_seats.pop()
    for tokens in engine.decode_tokens:
        assert tokens[idle] == IDLE_TOKEN
    # The idle row never gets a token appended anywhere.
    assert all(len(req.out) == 4 for req in requests)


def test_policy_is_validated():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=1)
    with pytest.raises(ValueError):
        Scheduler(engine, max_batch=1, policy="batched")
    with pytest.raises(ValueError):
        Scheduler(engine, max_batch=0)


def test_decode_event_ms_prefers_engine_event():
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    engine.last_decode_event_ms = 0.5  # ms, as a CUDA event would report
    sched = _scheduler(engine, max_batch=2)
    for i in range(4):
        sched.submit(_request(i, budget=3))

    rows = []
    while sched.queued or sched.active:
        rows.append(sched.step())

    decode_rows = [row for row in rows if row["decode_event_ms"]]
    assert decode_rows
    assert all(row["decode_event_ms"] == pytest.approx(0.5, abs=1e-12) for row in decode_rows)
    # host_gap is step - event - prefill, recorded as-is; the engine event is smaller
    # than the wall clock, so the gap is a real host number, not silently zeroed.
    for row in rows:
        assert row["host_gap_ms"] == pytest.approx(
            row["step_ms"] - row["decode_event_ms"] - row["prefill_ms"], abs=1e-9
        )
    assert sum(row["decode_event_ms"] for row in rows) > 0.0


def test_engine_failure_aborts_and_run_continues():
    clock = FakeClock()

    class ExplodingPrefill(FakeEngine):
        def prefill_into_seat(self, slot, prompt_ids):
            if prompt_ids[0] == 101:  # request id 1 only
                self.clock.advance(0.001)
                raise RuntimeError("prefill blew up")
            return super().prefill_into_seat(slot, prompt_ids)

    engine = ExplodingPrefill(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2)
    for i in range(4):
        sched.submit(_request(i, budget=3))

    result = _by_id(sched.run())

    assert result[1].state == "aborted"
    assert "RuntimeError" in result[1].error
    assert result[1].seat is None
    assert result[1].t_finish is not None
    assert all(result[i].state == "finished" for i in (0, 2, 3))
    assert all(len(result[i].out) == 3 for i in (0, 2, 3))
    assert not engine.occupied, "every seat must be frozen, including the aborted one"


def test_admission_goes_through_prefill_into_seat():
    """Admission is one call into the seat, never the W2 ``prefill_one`` + ``activate_seat``.

    The two-step hand-off copies a prefill ``KVcache`` into the row; on the pool path that
    would put the same KV in memory twice, so the scheduler must not go near it (W3 E0b).
    """
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=1)

    def legacy(*args, **kwargs):
        raise AssertionError("the scheduler must admit through prefill_into_seat")

    engine.prefill_one = legacy
    engine.activate_seat = legacy
    sched = _scheduler(engine, max_batch=1)
    req = _request(0, budget=2)
    sched.submit(req)

    finished = sched.run()[0]

    assert finished.state == "finished"
    assert len(finished.out) == 2
    assert engine.activations == [(0, tuple(req.prompt_ids))]


def test_queued_counts_only_arrived_requests():
    """``pending`` is the not-yet-arrived remainder, not backlog.

    A slow Poisson trace submits every request at t=0 with a future ``arrival_ts``.
    Counting those in ``queued`` would report a backlog on a trace that is
    *underloaded* -- the opposite of what the number means, and it would make
    ``queue_depth_max`` useless as an overload signal.
    """
    clock = FakeClock()
    engine = FakeEngine(clock, max_batch=2)
    sched = _scheduler(engine, max_batch=2)
    # 4 requests, one every 0.5 s; each needs a few steps of ~3 ms, so no backlog can
    # form. Only the arrival bookkeeping is under test.
    for i in range(4):
        sched.submit(_request(i, budget=5, arrival=i * 0.5))
    sched.run()
    rows = sched.step_rows

    # Step 0 has one arrived request (admitted), three still in the future.
    assert rows[0]["pending"] == 3, rows[0]
    assert rows[0]["queued"] == 0, rows[0]

    # Underloaded: the arrived backlog never exceeds one waiting request.
    assert max(row["queued"] for row in rows) <= 1, rows
    # Without the fix every row would report the full future remainder (4, 3, 2, 1).
    assert max(row["queued"] + row["pending"] for row in rows) <= 4, rows


# ---------------------------------------------------------------- W4 paged admission
class PagedFakeEngine(FakeEngine):
    """Fake engine with the W4 hooks: block admission, block prefill, block release.

    Pure Python (like ``FakeEngine``): blocks are integers, a request holds
    ``ceil((prompt + budget) / block_size)`` of them, and the bookkeeping is the same
    all-or-nothing ledger the real ``BlockManager`` implements. Seats are still handed to
    ``prefill_into_blocks`` so the decode path can scatter logits back by seat number.
    """

    impl = "paged"
    paged = True

    def __init__(self, clock, max_batch, *, n_blocks, block_size=4, **kwargs):
        super().__init__(clock, max_batch, **kwargs)
        self.n_blocks = int(n_blocks)
        self.block_size = int(block_size)
        self.free_blocks = list(range(self.n_blocks))
        self.held: dict[int, list[int]] = {}
        self.seats: dict[int, int] = {}
        self.released: list[int] = []
        self.can_admit_calls: list[tuple[int, int]] = []
        self.blocks_prefills: list[tuple[int, tuple[int, ...], int]] = []

    def _needed(self, prompt_len, budget):
        return max(1, -(-(int(prompt_len) + int(budget)) // self.block_size))

    @property
    def paged_max_batch(self):
        return self.slab.batch

    def can_admit(self, prompt_len, budget):
        self.can_admit_calls.append((int(prompt_len), int(budget)))
        if len(self.held) >= self.slab.batch:
            return False
        return self._needed(prompt_len, budget) <= len(self.free_blocks)

    def prefill_into_blocks(self, request_id, prompt_ids, *, seat=None, budget=None):
        needed = self._needed(len(prompt_ids), 0 if budget is None else budget)
        if needed > len(self.free_blocks):
            raise RuntimeError("cannot reserve blocks")
        if seat is None or not 0 <= int(seat) < self.slab.batch:
            raise IndexError(f"bad seat {seat} for prefill_into_blocks")
        if request_id in self.held:
            raise ValueError(f"request {request_id} already admitted")
        self.clock.advance(self.prefill_s)
        ids = self.free_blocks[:needed]
        del self.free_blocks[:needed]
        self.held[request_id] = ids
        self.seats[request_id] = int(seat)
        prompt = tuple(prompt_ids)
        self.prefill_calls.append(prompt)
        self.blocks_prefills.append((request_id, prompt, int(seat)))
        self.occupied.add(int(seat))
        self.seat_prompt[int(seat)] = prompt
        self.writes[int(seat)] = 0
        self.activations.append((int(seat), prompt))
        self.occupied_seen = max(self.occupied_seen, len(self.occupied))
        return [self._one_hot(self._first_token(list(prompt)))]

    def release_request(self, request_id):
        ids = self.held.pop(request_id, None)
        if ids is None:
            return
        self.free_blocks.extend(ids)
        self.free_blocks.sort()
        seat = self.seats.pop(request_id, None)
        self.released.append(request_id)
        if seat is not None:
            self.occupied.discard(seat)

    def freeze_seat(self, slot):  # must never be used by the paged arm
        raise AssertionError(f"the paged arm must release blocks, not freeze seat {slot}")

    def decode_batch(self, tokens, slab=None):
        # The real paged engine compresses seats itself; accept the seat-indexed list.
        assert slab is None, "the paged arm has no seat slab to hand over"
        return super().decode_batch(tokens, self.slab)


def _paged_scheduler(engine, max_batch, policy="continuous", clock=None):
    return _scheduler(engine, max_batch, policy=policy, clock=clock)


def test_paged_waits_for_blocks_and_keeps_ttft_identity():
    """A request that cannot reserve blocks stays queued; its wait is queue time."""
    clock = FakeClock()
    # 4 blocks x 4 tokens. Each request needs ceil((4 + 4) / 4) = 2 blocks, so the first two
    # take the whole pool and the third must wait for a release.
    engine = PagedFakeEngine(clock, max_batch=3, n_blocks=4, block_size=4)
    sched = _paged_scheduler(engine, max_batch=3)
    first = Request(id=0, prompt_ids=[10, 1, 2, 3], budget=4, arrival_ts=0.0)
    second = Request(id=1, prompt_ids=[11, 4, 5, 6], budget=4, arrival_ts=0.0)
    third = Request(id=2, prompt_ids=[12, 7, 8, 9], budget=4, arrival_ts=0.0)
    for req in (first, second, third):
        sched.submit(req)

    # Step 0: req 0 takes 2 of 4 blocks, req 1 takes 2 -> 0 free; req 2 is refused.
    row = sched.step()
    assert row["admitted"] == 2 and row["queued"] == 1
    assert third.state == "queued" and third.t_admitted is None
    assert engine.can_admit_calls[:3] == [(4, 4), (4, 4), (4, 4)]

    # The block ledger is never oversold: admitted requests hold <= n_blocks blocks.
    assert sum(len(ids) for ids in engine.held.values()) <= engine.n_blocks

    result = _by_id(sched.run())
    assert all(req.state == "finished" for req in result.values())
    for req in result.values():
        queue_ms = (req.t_prefill_start - req.t_arrival) * 1000.0
        prefill_ms = (req.t_first - req.t_prefill_start) * 1000.0
        ttft_ms = (req.t_first - req.t_arrival) * 1000.0
        assert ttft_ms == pytest.approx(queue_ms + prefill_ms, abs=1e-9)
    # Waiting for blocks is queue time, not prefill time: only the head request queued.
    assert result[2].t_prefill_start > result[2].t_arrival
    assert (result[2].t_first - result[2].t_prefill_start) == pytest.approx(
        engine.prefill_s, abs=1e-12
    )
    # Every request was released exactly once and every block came back.
    assert sorted(engine.released) == [0, 1, 2]
    assert engine.free_blocks == list(range(engine.n_blocks))


def test_paged_admission_never_oversells_within_one_step():
    """Two admissions in one step must not both pass ``can_admit`` on stale capacity."""
    clock = FakeClock()
    # 3 blocks x 4: each request needs 2 blocks, so only one fits, even with max_batch 3.
    engine = PagedFakeEngine(clock, max_batch=3, n_blocks=3, block_size=4)
    sched = _paged_scheduler(engine, max_batch=3)
    requests = [_request(i, budget=4, prompt_len=4) for i in range(3)]
    for req in requests:
        sched.submit(req)

    row = sched.step()

    assert row["admitted"] == 1, "block capacity (not max_batch) must cap the step"
    assert len(engine.held) == 1
    assert len(engine.released) + len(engine.held) == 1
    assert sum(len(ids) for ids in engine.held.values()) <= engine.n_blocks
    # A refused head is not popped: it is still first in line next step.
    assert [req.id for req in sched.queued] == [1, 2]
    assert row["queued"] == 2
    assert row["active"] <= 3

    rows = [row]
    while sched.queued or sched.active:
        rows.append(sched.step())
        assert rows[-1]["active"] <= 3
    assert all(req.state == "finished" for req in requests)
    assert engine.free_blocks == list(range(engine.n_blocks))


def test_paged_fifo_head_blocks_smaller_requests():
    """FIFO: when the head cannot fit, a smaller request behind it is not jumped ahead."""
    clock = FakeClock()
    # 3 blocks x 4. The head needs 3 blocks (12 slots), the follower needs 1 (4 slots).
    engine = PagedFakeEngine(clock, max_batch=2, n_blocks=3, block_size=4)
    sched = _paged_scheduler(engine, max_batch=2)
    big = Request(id=0, prompt_ids=list(range(20, 24)), budget=8, arrival_ts=0.0)
    small = Request(id=1, prompt_ids=[30, 31, 32, 33], budget=0, arrival_ts=0.0)
    sched.submit(big)
    sched.submit(small)

    # Hold one block out of the pool so the 3-block head cannot fit, while the 1-block
    # follower could: the scheduler must stop at the head, not skip it.
    engine.free_blocks.remove(0)
    row = sched.step()

    assert row["admitted"] == 0, "the follower must not be admitted past an unfitting head"
    assert big.state == "queued" and small.state == "queued"
    assert engine.held == {}
    assert [req.id for req in sched.queued] == [0, 1]

    # Put the block back; now the head fits and is admitted first.
    engine.free_blocks.insert(0, 0)
    row = sched.step()
    assert row["admitted"] >= 1
    assert engine.blocks_prefills[0][0] == 0, "the FIFO head must be admitted first"


def test_paged_release_uses_release_request_not_freeze():
    """Retirement calls ``release_request`` and still recycles the scheduler's seat."""
    clock = FakeClock()
    engine = PagedFakeEngine(clock, max_batch=2, n_blocks=8, block_size=4)
    sched = _paged_scheduler(engine, max_batch=2)
    for i in range(4):
        sched.submit(_request(i, budget=2, arrival=i * 0.01))

    result = sched.run()

    assert all(req.state == "finished" for req in result)
    assert sorted(engine.released) == [0, 1, 2, 3]
    assert engine.free_blocks == list(range(8))
    # Blocks were prefilled through the paged hook, never the seat one.
    assert [call[0] for call in engine.blocks_prefills] == [0, 1, 2, 3]
    assert all(call[2] in (0, 1) for call in engine.blocks_prefills)


def test_paged_step_rows_carry_pool_stats_without_renaming_seat_fields():
    """The paged hook appends its account; the seat row keeps every field and name."""
    clock = FakeClock()
    engine = PagedFakeEngine(clock, max_batch=2, n_blocks=8, block_size=4)

    recorded: list[dict] = []

    class WithStats(PagedFakeEngine):
        def paged_stats(self):
            stats = {
                "lpad": 64,
                "blocks_in_use": sum(len(v) for v in self.held.values()),
                "written_blocks": 1,
                "frag_internal": 0.25,
                "reservation_slack": 0.5,
                "kv_alloc_mb": 0.125,
                "kv_live_mb": 0.0625,
                "workspace_mb": 0.5,
            }
            recorded.append(stats)
            return stats

    engine.__class__ = WithStats
    sched = _paged_scheduler(engine, max_batch=2)
    sched.submit(_request(0, budget=2))
    row = sched.step()

    seat_fields = {
        "step", "active", "queued", "pending", "admitted", "retired",
        "step_ms", "prefill_ms", "decode_event_ms", "host_gap_ms",
    }
    assert seat_fields <= set(row)
    for key in ("lpad", "blocks_in_use", "written_blocks", "frag_internal",
                "reservation_slack", "kv_alloc_mb", "kv_live_mb", "workspace_mb"):
        assert key in row, f"missing paged stats field {key}"
    assert row["host_gap_ms"] == pytest.approx(
        row["step_ms"] - row["decode_event_ms"] - row["prefill_ms"], abs=1e-9
    )


def test_paged_pool_shape_is_checked_once():
    """The paged pool's row cap must equal the scheduler's ``max_batch``."""
    clock = FakeClock()
    engine = PagedFakeEngine(clock, max_batch=2, n_blocks=8, block_size=4)

    class WrongCap(PagedFakeEngine):
        @property
        def paged_max_batch(self):
            return 3

    engine.__class__ = WrongCap
    sched = _paged_scheduler(engine, max_batch=2)
    sched.submit(_request(0, budget=2))

    with pytest.raises(ValueError, match="max_batch"):
        sched.run()
