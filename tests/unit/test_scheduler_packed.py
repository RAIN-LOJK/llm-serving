"""W4.5 packed-prefill admission: FIFO groups, one shared forward, one TTFT identity.

The fake engine has exactly the real one's grouped interface (``max_prefill_tokens`` /
``prefill_group``), so these tests pin the scheduler rather than any kernel. The scheduler
has no serial path any more (D7): ``test_scheduler.py`` runs the same grouped interface
with a richer per-seat fake.
"""

from __future__ import annotations

import pytest

from serve.scheduler import Request, Scheduler


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeSlab:
    def __init__(self, batch: int) -> None:
        self.batch = batch


class PackedFakeEngine:
    impl = "continuous"

    def __init__(
        self,
        clock: FakeClock,
        *,
        max_batch: int,
        max_prefill_tokens: int,
        vocab_size: int = 64,
        prefill_s: float = 0.010,
        decode_s: float = 0.001,
        fail: bool = False,
    ) -> None:
        self.clock = clock
        self.slab = FakeSlab(max_batch)
        self.max_prefill_tokens = max_prefill_tokens
        self.vocab_size = vocab_size
        self.max_position = 512
        self.eos_token_id = None
        self.device = "fake"
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.fail = fail
        self.groups: list[list[tuple[int, ...]]] = []
        self.decode_calls = 0
        self.freezes: list[int] = []

    def _row(self, token: int) -> list[float]:
        row = [0.0] * self.vocab_size
        row[int(token) % self.vocab_size] = 1.0
        return row

    def prefill_group(self, requests: list[dict]):
        if self.fail:
            raise RuntimeError("packed forward blew up")
        assert requests, "empty group reached the engine"
        total = sum(len(spec["prompt_ids"]) for spec in requests)
        assert len(requests) == 1 or total <= self.max_prefill_tokens, (
            f"engine received a {total}-token group above max_prefill_tokens="
            f"{self.max_prefill_tokens}"
        )
        self.clock.advance(self.prefill_s)
        self.groups.append([tuple(spec["prompt_ids"]) for spec in requests])
        return [
            self._row((spec["prompt_ids"][0] + len(spec["prompt_ids"])) % self.vocab_size)
            for spec in requests
        ]

    def decode_batch(self, tokens, slab=None):
        assert len(tokens) == self.slab.batch
        self.clock.advance(self.decode_s)
        self.decode_calls += 1
        return [self._row(int(token) + 1) for token in tokens]

    def freeze_seat(self, slot: int) -> None:
        self.freezes.append(int(slot))


def _request(idx: int, length: int, *, budget: int = 2, arrival: float = 0.0) -> Request:
    return Request(
        id=idx,
        prompt_ids=[idx + 1] * length,
        budget=budget,
        arrival_ts=arrival,
    )


def _run(engine, requests, *, max_batch: int, policy: str = "continuous"):
    scheduler = Scheduler(
        engine,
        max_batch=max_batch,
        policy=policy,
        now=engine.clock.now,
        sleep=engine.clock.sleep,
    )
    for request in requests:
        scheduler.submit(request)
    scheduler.close()
    return scheduler, scheduler.run()


def test_continuous_group_is_fifo_and_stops_at_token_budget():
    clock = FakeClock()
    engine = PackedFakeEngine(clock, max_batch=4, max_prefill_tokens=10)
    _, result = _run(engine, [_request(0, 4), _request(1, 3), _request(2, 5)], max_batch=4)

    # 4 + 3 fits; the 5-token head would make 12 > 10, so it opens the next group.
    assert engine.groups[0] == [tuple([1] * 4), tuple([2] * 3)]
    assert engine.groups[1] == [tuple([3] * 5)]
    assert [req.state for req in result] == ["finished"] * 3

    by_id = {req.id: req for req in result}
    # Same group -> same shared prefill start/end; different groups differ.
    assert by_id[0].prefill_group == by_id[1].prefill_group
    assert by_id[2].prefill_group != by_id[0].prefill_group
    assert by_id[0].t_prefill_start == by_id[1].t_prefill_start
    assert by_id[0].t_first == by_id[1].t_first
    assert by_id[2].t_prefill_start > by_id[0].t_first

    for req in result:
        queue = req.t_admitted - req.t_arrival
        pack_wait = req.t_prefill_start - req.t_admitted
        shared = req.t_first - req.t_prefill_start
        assert req.t_admitted <= req.t_prefill_start <= req.t_first
        assert queue >= 0.0 and pack_wait >= 0.0 and shared > 0.0
        assert (req.t_first - req.t_arrival) == pytest.approx(queue + pack_wait + shared)


def test_oversized_head_is_admitted_alone_not_skipped():
    clock = FakeClock()
    engine = PackedFakeEngine(clock, max_batch=2, max_prefill_tokens=5)
    _, result = _run(engine, [_request(0, 6), _request(1, 2)], max_batch=2)

    assert engine.groups[0] == [tuple([1] * 6)], "an over-budget head must go alone"
    assert engine.groups[1] == [tuple([2] * 2)]
    assert [req.id for req in engine.groups and result] == [0, 1]


def test_static_wave_runs_all_token_groups_in_one_step():
    clock = FakeClock()
    engine = PackedFakeEngine(clock, max_batch=4, max_prefill_tokens=10)
    _, result = _run(
        engine,
        [_request(0, 4), _request(1, 4), _request(2, 4), _request(3, 4)],
        max_batch=4,
        policy="static",
    )

    assert engine.groups == [
        [tuple([1] * 4), tuple([2] * 4)],
        [tuple([3] * 4), tuple([4] * 4)],
    ]
    # Both groups belong to the same wave/step: the second starts in the same step.
    assert len({req.t_admitted for req in result}) == 1
    assert len({req.prefill_group for req in result}) == 2


def test_group_failure_aborts_every_member_and_frees_seats():
    clock = FakeClock()
    engine = PackedFakeEngine(clock, max_batch=2, max_prefill_tokens=16, fail=True)
    scheduler, result = _run(engine, [_request(0, 3), _request(1, 3)], max_batch=2)

    assert [req.state for req in result] == ["aborted", "aborted"]
    assert all("packed forward blew up" in (req.error or "") for req in result)
    assert sorted(engine.freezes) == [0, 1]
    assert scheduler.active == []
    assert len(scheduler.queued) == 0


def test_group_admission_checks_the_whole_candidate_group():
    """A paged engine reserves every member; the scheduler must ask before filling."""
    clock = FakeClock()
    engine = PackedFakeEngine(clock, max_batch=4, max_prefill_tokens=16)

    def can_admit_group(requests):
        return len(requests) <= 1

    engine.can_admit_group = can_admit_group  # type: ignore[method-assign]
    _, result = _run(engine, [_request(0, 3), _request(1, 3)], max_batch=4)

    assert engine.groups == [[tuple([1] * 3)], [tuple([2] * 3)]]
    assert [req.state for req in result] == ["finished", "finished"]
    assert all(req.t_prefill_start >= req.t_admitted for req in result[1:])


def test_step_row_counts_shared_prefill_once():
    clock = FakeClock()
    engine = PackedFakeEngine(clock, max_batch=4, max_prefill_tokens=16)
    scheduler, _ = _run(engine, [_request(0, 3), _request(1, 3)], max_batch=4)

    rows = [row for row in scheduler.step_rows if row["admitted"]]
    first = rows[0]
    assert first["prefill_groups"] == 1
    assert first["prefill_shared_ms"] == pytest.approx(engine.prefill_s * 1000.0)
    assert first["prefill_ms"] == pytest.approx(first["prefill_shared_ms"])
    assert first["pack_wait_ms"] >= 0.0
    assert first["prefill_ids_per_group"] == [2]
