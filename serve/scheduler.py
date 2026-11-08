"""Static-wave vs continuous-refill scheduler (docs/decisions.md D5, packed prefill D7).

One :meth:`Scheduler.step` is exactly the sequence frozen in the plan:

1. **Admission** (policy-dependent, only for requests that have arrived). Both
   policies admit through **one packed FIFO group per forward** (D7); the serial
   per-request control arm is gone, so there is no switch and no fallback.

   * ``continuous`` -- build one group from the FIFO head, adding members while
     seats and ``max_prefill_tokens`` allow. A seat freed during step *k* is
     refilled at step *k+1*, because admission runs before retirement inside a step.
   * ``static`` -- only when ``active == 0`` open one wave: wait until
     ``max_batch`` requests have arrived, or, once the trace is known to be
     exhausted (:meth:`Scheduler.close`, or :meth:`Scheduler.run`, which cannot
     receive submissions while it blocks), take whatever is left. A partial wave
     is never waited for forever. The wave is then split into
     ``max_prefill_tokens`` groups, all prefilled in the same step.

   A request is stamped ``t_admitted`` when it is selected, ``t_prefill_start``
   when its group's forward starts and ``t_first`` when it returns. Every member
   of a group shares the last two, which is what makes
   ``ttft == queue + pack_wait + prefill_shared`` an identity.

2. **Batch decode** -- one ``decode_batch`` over the whole slab. ``tokens`` is
   always length ``max_batch``: active seats carry their last token, **idle
   seats carry the placeholder** :data:`IDLE_TOKEN`, and their row is frozen
   (mask closed, cursor parked; see :meth:`Scheduler._park_virgin_seats`). Their
   logits are discarded. One argmax over ``(B, vocab)`` does the device-to-host
   read.

3. **Retirement** -- ``budget`` / EOS / ``max_position`` all terminate. The
   seat is frozen (not reclaimed, P-26) and returned to the free list,
   ``t_finish`` is stamped. An engine exception during a group's prefill aborts
   every member of that group (one forward cannot attribute the failure); an
   exception in the batch decode aborts every active request (it cannot be
   attributed to one row) and the run continues with whatever is still queued.

Timing
------
``prefill_ms`` is the sum of this step's **group** durations
``t_first - t_prefill_start``: a shared forward is counted once, not once per
request (an aborted group contributes the time it actually spent).
``decode_event_ms`` is ``engine.last_decode_event_ms`` when the engine provides a
float, otherwise the decode wall clock. ``step_ms`` is measured from the last
wake-up, so time spent sleeping for a future arrival is not counted as compute.
``host_gap_ms = step_ms - decode_event_ms - prefill_ms`` is recorded as-is,
including when it is negative.

The step/request attribution follows D7: ``queue_ms = t_admitted - t_arrival``,
``pack_wait_ms = t_prefill_start - t_admitted`` and
``prefill_shared_ms = t_first - t_prefill_start``.

The module depends on the standard library only; the engine is duck-typed
(``prefill_group`` / ``decode_batch`` / ``freeze_seat`` / ``slab`` /
``vocab_size`` / ``device`` / ``eos_token_id`` / ``max_position`` /
``max_prefill_tokens``), so the unit tests can use a fake engine without torch.

W4 paged arm (``docs/decisions.md`` D6)
-------------------------------------
The same step applies to a paged engine, through optional hooks only -- nothing is
imported and no branch is taken when they are absent:

* **Admission asks before popping.** If the engine has
  ``can_admit_group(requests)``, the whole candidate group is offered to it
  *before* any member is popped. A refusal leaves the head queued (so its wait is
  queue time) and **stops** the admission pass -- a head that does not fit is
  never skipped (FIFO). The group, not the request, is the admission unit: it
  matches the all-or-nothing block reservation ``prefill_group`` performs, which
  is what keeps the pool from being oversold inside one step.
* **Prefill goes into blocks.** ``prefill_group`` receives one spec per member,
  ``{id, prompt_ids, seat, budget}``; the engine owns the seat sink / block sink
  choice (D7) and the reservation is the upper bound over ``prompt + budget``.
* **Release frees blocks.** If the engine has ``release_request``, retirement calls it
  instead of ``freeze_seat`` (D6: blocks go back to the free list; the seat number still
  returns to the scheduler's free list).
* **Seat-only bookkeeping is skipped.** ``_park_virgin_seats`` and ``_check_pool`` read
  ``engine.slab``, which a paged engine does not have; they run only on the seat path.
* **The step row gains the pool's account** when the engine has ``paged_stats()``; every
  existing field keeps its name.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["IDLE_TOKEN", "POLICIES", "Request", "Scheduler"]

POLICIES = ("static", "continuous")

#: Token written into idle rows of the batch decode. The row is masked out and its
#: logits are discarded, so the value is arbitrary; 0 keeps the batch shape fixed.
IDLE_TOKEN = 0


@dataclass
class Request:
    """One request and its lifecycle timestamps (wall clock, seconds)."""

    id: int
    prompt_ids: list[int]
    budget: int
    arrival_ts: float
    state: str = "queued"  # queued | prefilling | decoding | finished | aborted
    seat: int | None = None
    out: list[int] = field(default_factory=list)
    t_arrival: float = 0.0
    t_admitted: float | None = None
    t_prefill_start: float | None = None
    t_first: float | None = None
    t_finish: float | None = None
    error: str | None = None
    #: W4.5 packed prefill: the group this request was prefilled in. Every request of a
    #: group shares ``t_prefill_start`` and ``t_first``, which is what makes
    #: ``ttft == queue + pack_wait + prefill_shared`` an identity.
    prefill_group: int | None = None


def _row_argmax(logits: Any) -> list[int]:
    """Greedy ids for every row of a ``(B, vocab)`` logits object.

    Works for a torch tensor (``argmax(dim=-1).tolist()``) and for a fake engine
    returning plain lists, so this module never imports torch.
    """
    argmax = getattr(logits, "argmax", None)
    if argmax is not None:
        ids = argmax(dim=-1).tolist()
        return [int(value) for value in (ids if isinstance(ids, list) else [ids])]
    rows = list(logits)
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]  # a single flat row
    return [max(range(len(row)), key=row.__getitem__) for row in rows]


# ---------------------------------------------------------------- scheduler
class Scheduler:
    """Admit / decode / retire over one ``KvSlab`` whose rows are the seats."""

    def __init__(
        self,
        engine,
        *,
        max_batch: int,
        policy: str = "continuous",
        now: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        eos_token_id: int | None = None,
        max_position: int | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        if int(max_batch) < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch!r}")
        if eos_token_id is None:
            eos_token_id = getattr(engine, "eos_token_id", None)
        if max_position is None:
            max_position = getattr(engine, "max_position", None)

        self.engine = engine
        self.max_batch = int(max_batch)
        self.policy = policy
        self.now = now
        self.sleep = sleep
        self.eos_token_id = eos_token_id
        self.max_position = max_position

        # Admission order is (arrival_ts, id): the heap key makes the admissible
        # sequence independent of the order ``submit`` was called in (C3).
        self._queue: list[tuple[float, int, Request]] = []
        self._active: list[Request] = []
        self._all: list[Request] = []
        self._ids: set[int] = set()
        self._free: list[int] = list(range(self.max_batch))
        heapq.heapify(self._free)
        self._rows: list[dict] = []
        self._step_no = 0
        self._closed = False
        self._admitted_this_step = 0
        self._retired_this_step = 0
        self._step_origin: float | None = None
        self._virgin_seats: set[int] | None = set(range(self.max_batch))
        self._pool_checked = False
        # W4.5 packed groups: one id per distinct prefill forward, plus this step's
        # (group_id, n_requests, n_tokens, start, end) durations for the step row.
        self._group_no = 0
        self._step_groups: list[dict] = []
        # True when the last admission pass stopped because the FIFO head could not reserve
        # its blocks (a paged refresh that fails for any other reason is not this).
        self._blocked_this_step = False

    # ------------------------------------------------------------ public
    def submit(self, req: Request) -> None:
        """Enqueue a request; ``arrival_ts`` is when it becomes admissible."""
        if req.state != "queued":
            raise ValueError(f"request {req.id} is {req.state!r}, only 'queued' can be submitted")
        if not req.prompt_ids:
            raise ValueError(f"request {req.id} has an empty prompt")
        if req.budget < 0:
            raise ValueError(f"request {req.id} has a negative budget ({req.budget})")
        if req.id in self._ids:
            raise ValueError(f"duplicate request id {req.id}")
        req.t_arrival = float(req.arrival_ts)
        self._ids.add(req.id)
        self._all.append(req)
        heapq.heappush(self._queue, (req.t_arrival, req.id, req))

    def close(self) -> None:
        """Mark the trace fully submitted: static may now open a partial wave."""
        self._closed = True

    def run(self) -> list[Request]:
        """Run until every submitted request is finished or aborted.

        ``run`` blocks, so no further ``submit`` can arrive while it is in flight:
        for the static arm the moment it starts is the end of the trace, so a
        partial last wave opens instead of waiting forever for a full one.
        """
        self._closed = True
        while self._queue or self._active:
            row = self.step()
            if not self._active and not row["admitted"] and not row["retired"]:
                # Nothing ran and nothing can run. Either the queue head has not arrived and
                # the clock is not moving (an injected sleep that does not advance time), or
                # the head can never fit its block reservation (a pool too small for it).
                # Failing is better than spinning forever.
                if self._blocked_this_step:
                    raise RuntimeError(
                        "scheduler cannot make progress: the FIFO head cannot reserve its "
                        "blocks and nothing is active to release any"
                    )
                raise RuntimeError(
                    "scheduler cannot make progress: "
                    f"{len(self._queue)} queued, none arrived, and sleep did not advance time"
                )
        return sorted(self._all, key=lambda r: r.id)

    @property
    def active(self) -> list[Request]:
        return list(self._active)

    @property
    def queued(self) -> list[Request]:
        return sorted((item[2] for item in self._queue), key=lambda r: (r.arrival_ts, r.id))

    @property
    def step_rows(self) -> list[dict]:
        return list(self._rows)

    # ------------------------------------------------------------ seats
    def _paged_mode(self) -> bool:
        """Whether this engine is the W4 paged arm (explicit, never inferred from a try).

        A paged engine advertises ``paged`` (bool or property) or ``impl = "paged"``. An
        adapter that wraps the engine may expose neither, so the two hooks that only a
        paged engine has (``prefill_into_blocks`` + ``paged_stats``) are the last, still
        explicit, signal. The real engine's ``slab`` property raises when no seat pool is
        open, so probing it with ``hasattr`` would either raise or misclassify -- the flag
        is the contract.
        """
        flag = getattr(self.engine, "paged", None)
        if callable(flag):
            flag = flag()
        if isinstance(flag, bool):
            return flag
        if getattr(self.engine, "impl", None) == "paged":
            return True
        return callable(getattr(self.engine, "prefill_into_blocks", None)) and callable(
            getattr(self.engine, "paged_stats", None)
        )

    def _group_available(self, requests: list[Request]) -> bool:
        """Whether a whole candidate group fits (W4.5 packed admission, D7).

        The paged engine reserves every member's blocks back to back; checking only the
        head against the pre-group free list over-commits the pool and aborts the last
        member inside ``prefill_group``. The engine's optional ``can_admit_group`` makes
        the group the admission unit. A seat engine has no such hook (its slab is
        allocated up front), so the answer is True.
        """
        if not requests:
            return True
        group = getattr(self.engine, "can_admit_group", None)
        if not callable(group):
            return True
        return bool(
            group(
                [
                    {
                        "id": req.id,
                        "prompt_ids": list(req.prompt_ids),
                        "budget": req.budget,
                    }
                    for req in requests
                ]
            )
        )

    def _park_virgin_seats(self) -> None:
        """Freeze every seat that has never held a request, once, after the first step's
        admissions (seats claimed by that step are activated again anyway).

        A fresh pool slab starts with **every** row live: ``KvSlab.reset()`` leaves
        ``active`` all True because W2's single-request path depends on that. A row that
        carries no request is still forwarded every step (fixed block shape, D4), but it
        is not a request row: an untouched live row would attend to column 0, write its
        placeholder into it and advance its cursor one column per step, i.e. it would
        grow like a request whose output nobody reads. §2.1's idle seat is the frozen
        row instead (``valid=0``, mask closed, cursor parked), so the scheduler parks
        every never-used seat once, using only the engine's documented ``freeze_seat``.
        Row-level output is discarded either way; the point is that the slab state an
        idle seat reports matches the contract, and no stray cursor growth has to be
        explained when the bucket is tight.
        """
        if self._virgin_seats is None:
            return
        for seat in sorted(self._virgin_seats):
            self.engine.freeze_seat(seat)
        self._virgin_seats = None

    # ------------------------------------------------------------ one step
    def step(self) -> dict:
        self._admitted_this_step = 0
        self._retired_this_step = 0
        self._blocked_this_step = False
        self._step_groups = []
        self._step_origin = self.now()

        admitted = self._admit()
        # A group shares one forward, so its cost is counted once, not once per
        # request. Summing per-request ``t_first - t_prefill_start`` here would
        # multiply the shared prefill by the group size and make host_gap negative.
        prefill_ms = sum(
            (group["end"] - group["start"]) * 1000.0 for group in self._step_groups
        )

        # A request can terminate on its very first token (budget 1 / EOS /
        # max_position). Retire those before the batch decode so they never get a
        # token past their limit and their row is idle for the whole decode.
        self._terminate()

        if self._active:
            # Idle seats must be parked before the first forward that covers them, and the
            # pool shape is checked once. Parking reads ``engine.slab`` and exists only on
            # the seat path; the paged pool has no idle rows, but its row cap is still
            # checked (``_check_pool`` knows which arm it is on).
            if not self._paged_mode():
                self._park_virgin_seats()
            self._check_pool()
        decode_event_ms = self._decode()
        self._terminate()

        step_ms = (self.now() - self._step_origin) * 1000.0
        # ``queued`` counts requests the step clock has already reached: a request that
        # has been submitted but whose arrival is still in the future is not waiting for
        # capacity yet, and counting it would inflate ``queue_depth_max`` on exactly the
        # traces that are *not* overloaded (a slow Poisson trace has every request
        # submitted from t=0). ``pending`` keeps that invisible remainder visible.
        now = self.now()
        queued = sum(1 for item in self._queue if item[0] <= now)
        row = {
            "step": self._step_no,
            "active": len(self._active),
            "queued": queued,
            "pending": len(self._queue) - queued,
            "admitted": self._admitted_this_step,
            "retired": self._retired_this_step,
            "step_ms": step_ms,
            "prefill_ms": prefill_ms,
            "decode_event_ms": decode_event_ms,
            "host_gap_ms": step_ms - decode_event_ms - prefill_ms,
        }
        # W4.5 TTFT decomposition (D7): ``queue + pack_wait + prefill_shared``. These
        # keys are additive; historical W3/W4 rows without them are still readable by
        # ``bench/scenario.py`` (a missing ``pack_wait_ms`` means 0).
        pack_wait_ms = sum(
            max(0.0, (req.t_prefill_start or req.t_admitted or 0.0) - (req.t_admitted or 0.0))
            for req in admitted
            if req.t_prefill_start is not None and req.t_admitted is not None
        ) * 1000.0
        row.update(
            {
                "prefill_groups": len(self._step_groups),
                "pack_wait_ms": pack_wait_ms,
                "prefill_shared_ms": prefill_ms,
                "prefill_ids_per_group": [group["n"] for group in self._step_groups],
            }
        )
        self._attach_paged_stats(row)
        self._step_no += 1
        self._rows.append(row)
        return row

    def _attach_paged_stats(self, row: dict) -> None:
        """Add the paged pool's account to a step row, when the engine reports one (W4).

        The seat path appends nothing: no existing field is renamed or dropped. On the
        paged arm the row carries the block counts, the two fragmentation numbers and the
        memory figures; the Triton arm has no ``lpad`` or gather account.
        """
        stats = getattr(self.engine, "paged_stats", None)
        if not self._paged_mode() or not callable(stats):
            return
        for key, value in stats().items():
            row.setdefault(key, value)

    # ------------------------------------------------------------ admission
    def _admit(self) -> list[Request]:
        if self.policy == "continuous":
            return self._admit_packed_continuous()
        return self._admit_packed_static()

    def _wave_target(self) -> float:
        """Arrival of the last member of the wave that would open next."""
        size = min(self.max_batch, len(self._queue))
        return max(item[0] for item in heapq.nsmallest(size, self._queue))

    # ------------------------------------------------------ W4.5 packed groups
    def _max_prefill_tokens(self) -> int:
        return int(getattr(self.engine, "max_prefill_tokens", 8192))

    def _take_seat(self) -> int:
        seat = heapq.heappop(self._free)
        if self._virgin_seats:
            self._virgin_seats.discard(seat)
        return seat

    def _finish_zero_budget(self, req: Request) -> None:
        """A request with no generation budget never takes a seat or a prefill."""
        req.t_admitted = req.t_prefill_start = req.t_first = req.t_finish = self.now()
        req.state = "finished"
        self._admitted_this_step += 1
        self._retired_this_step += 1

    def _admit_packed_continuous(self) -> list[Request]:
        """One FIFO group per step: no waiting to fill, no skipping the head."""
        if len(self._active) >= self.max_batch:
            return []
        specs: list[tuple[Request, int]] = []
        tokens = 0
        max_tokens = self._max_prefill_tokens()
        while self._queue and len(self._active) + len(specs) < self.max_batch:
            head_arrival = self._queue[0][0]
            if head_arrival > self.now():
                if self._active or specs:
                    break
                self._wait_until(head_arrival)
                if self._queue and self._queue[0][0] > self.now():
                    break
                continue
            head = self._queue[0][2]
            if not self._group_available([req for req, _ in specs] + [head]):
                # A head that cannot fit *with the group already selected* stays queued;
                # it is never skipped for a later, smaller request (FIFO).
                self._blocked_this_step = True
                break
            length = len(head.prompt_ids)
            if specs and tokens + length > max_tokens:
                # A head that does not fit the current group opens the next one; it is
                # never skipped for a later, smaller request (P-44).
                break
            req = heapq.heappop(self._queue)[2]
            if req.budget <= 0:
                self._finish_zero_budget(req)
                continue
            req.t_admitted = self.now()
            specs.append((req, self._take_seat()))
            tokens += length
        if not specs:
            return []
        return self._run_group(specs)

    def _admit_packed_static(self) -> list[Request]:
        """Open one wave, then prefill it in as many token-budget groups as it needs.

        A wave is still admitted as a unit (every member's ``t_admitted`` is stamped when
        the wave opens), but the forward is split by ``max_prefill_tokens``. All groups of
        the wave run in this step, so a group boundary never delays a member to the next
        wave.
        """
        if self._active or not self._queue:
            return []
        while True:
            target = self._wave_target()
            if target <= self.now():
                break
            self._wait_until(target)
            if self.now() < target:
                if len(self._queue) < self.max_batch and not self._closed:
                    return []
                break
        if len(self._queue) < self.max_batch and not self._closed:
            return []

        specs: list[tuple[Request, int]] = []
        now = self.now()
        while len(self._active) + len(specs) < self.max_batch and self._queue:
            if self._queue[0][0] > now:
                break
            head = self._queue[0][2]
            if not self._group_available([req for req, _ in specs] + [head]):
                # FIFO: the wave stops at a member that cannot reserve its blocks with the
                # members already selected; it opens later once blocks have been released.
                self._blocked_this_step = True
                break
            req = heapq.heappop(self._queue)[2]
            if req.budget <= 0:
                self._finish_zero_budget(req)
                continue
            req.t_admitted = now
            specs.append((req, self._take_seat()))

        max_tokens = self._max_prefill_tokens()
        groups: list[list[tuple[Request, int]]] = []
        current: list[tuple[Request, int]] = []
        tokens = 0
        for spec in specs:
            length = len(spec[0].prompt_ids)
            if current and tokens + length > max_tokens:
                groups.append(current)
                current = []
                tokens = 0
            current.append(spec)
            tokens += length
        if current:
            groups.append(current)

        admitted: list[Request] = []
        for group in groups:
            admitted.extend(self._run_group(group))
        return admitted

    def _run_group(self, specs: list[tuple[Request, int]]) -> list[Request]:
        """One engine call for the whole group; one argmax; one shared ``t_first``."""
        group_id = self._group_no
        self._group_no += 1
        self._admitted_this_step += len(specs)
        started = self.now()
        for req, seat in specs:
            req.seat = seat
            req.state = "prefilling"
            req.prefill_group = group_id
            req.t_admitted = started if req.t_admitted is None else req.t_admitted
            req.t_prefill_start = started
        payload = [
            {
                "id": req.id,
                "prompt_ids": list(req.prompt_ids),
                "seat": seat,
                "budget": req.budget,
            }
            for req, seat in specs
        ]
        try:
            logits = self.engine.prefill_group(payload)
            ids = _row_argmax(logits)
            if len(ids) != len(specs):
                raise ValueError(
                    f"prefill_group returned {len(ids)} rows for {len(specs)} requests"
                )
        except Exception as exc:  # noqa: BLE001 - one forward cannot attribute a failure
            ended = self.now()
            for req, _ in specs:
                req.t_first = req.t_finish = ended
                self._abort(req, exc)
            self._step_groups.append(
                {
                    "id": group_id,
                    "n": len(specs),
                    "tokens": sum(len(req.prompt_ids) for req, _ in specs),
                    "start": started,
                    "end": ended,
                    "ok": False,
                }
            )
            return []
        ended = self.now()
        admitted: list[Request] = []
        for (req, _), first in zip(specs, ids):
            req.out = [int(first)]
            req.t_first = ended
            req.state = "decoding"
            self._active.append(req)
            admitted.append(req)
        self._step_groups.append(
            {
                "id": group_id,
                "n": len(specs),
                "tokens": sum(len(req.prompt_ids) for req, _ in specs),
                "start": started,
                "end": ended,
                "ok": True,
            }
        )
        return admitted

    # ------------------------------------------------------------ step plumbing
    def _check_pool(self) -> None:
        """One-shot shape check: the pool must have exactly ``max_batch`` seats.

        The engine indexes rows by seat number and ``decode_batch`` must cover the whole
        block, so a pool of the wrong width would silently drop or overrun a seat.
        Checked once, on the first decode, not per step.
        """
        if self._pool_checked:
            return
        self._pool_checked = True
        if self._paged_mode():
            # The paged pool has no slab; its shape contract is the row cap ``max_batch``.
            # Check it once too, so a mismatch fails before any decode (not on the first
            # request that happens to land on the missing seat).
            batch = getattr(self.engine, "paged_max_batch", None)
            if batch is not None and int(batch) != self.max_batch:
                raise ValueError(
                    f"engine's paged pool was opened with max_batch={batch} but the scheduler "
                    f"runs max_batch={self.max_batch}; open the pool with the same max_batch"
                )
            return
        slab = self.engine.slab
        batch = getattr(slab, "batch", self.max_batch)
        if batch != self.max_batch:
            raise ValueError(
                f"engine.slab has {batch} rows but the scheduler runs max_batch="
                f"{self.max_batch}; open the pool with the same max_batch"
            )

    def _wait_until(self, target: float) -> None:
        delay = target - self.now()
        if delay > 0:
            self.sleep(delay)
            # Waiting for an arrival is not compute: the step row starts at the wake-up.
            self._step_origin = self.now()

    # ------------------------------------------------------------ decode
    def _decode(self) -> float:
        if not self._active:
            return 0.0
        tokens = [IDLE_TOKEN] * self.max_batch
        for req in self._active:
            tokens[req.seat] = req.out[-1]
        t0 = self.now()
        try:
            if self._paged_mode():
                # The paged engine compresses the active seats into its own batch; there is
                # no seat slab to hand over.
                logits = self.engine.decode_batch(tokens)
            else:
                logits = self.engine.decode_batch(tokens, self.engine.slab)
            ids = _row_argmax(logits)
            if len(ids) != self.max_batch:
                raise ValueError(
                    f"decode_batch returned {len(ids)} rows, the slab has {self.max_batch}"
                )
        except Exception as exc:  # noqa: BLE001 - a batch failure hits every active row
            t1 = self.now()
            for req in list(self._active):
                req.t_finish = t1
                self._abort(req, exc)
            return (t1 - t0) * 1000.0
        t1 = self.now()

        for req in self._active:
            req.out.append(ids[req.seat])

        event_ms = getattr(self.engine, "last_decode_event_ms", None)
        if isinstance(event_ms, (int, float)):
            return float(event_ms)
        return (t1 - t0) * 1000.0

    # ------------------------------------------------------------ retirement
    def _terminate(self) -> None:
        """Retire every active request that reached budget / EOS / max_position."""
        for req in list(self._active):
            if self._termination(req) is not None:
                req.t_finish = self.now()
                req.state = "finished"
                self._release(req)

    def _termination(self, req: Request) -> str | None:
        if len(req.out) >= req.budget:
            return "budget"
        if self.eos_token_id is not None and req.out[-1] == self.eos_token_id:
            return "eos"
        if self.max_position is not None:
            # The next decode writes column ``prompt_len + len(out)``; past the last
            # column of the position table there is nothing left to write.
            if len(req.prompt_ids) + len(req.out) >= self.max_position:
                return "max_position"
        return None

    def _release(self, req: Request) -> None:
        """Retire the request's KV and return its row to the free list.

        Seat arm: freeze the seat (its KV stays put, P-26). Paged arm: ``release_request``
        returns the blocks to the free list; the seat number is still recycled here.
        """
        seat = req.seat
        if seat is not None:
            release = getattr(self.engine, "release_request", None)
            if self._paged_mode() and callable(release):
                # The real engine exposes both hooks; only the open arm's may run.
                release(req.id)
            else:
                self.engine.freeze_seat(seat)
            heapq.heappush(self._free, seat)
        req.seat = None
        if req in self._active:
            self._active.remove(req)
        self._retired_this_step += 1

    def _abort(self, req: Request, exc: BaseException) -> None:
        req.state = "aborted"
        req.error = f"{type(exc).__name__}: {exc}"
        self._release(req)
