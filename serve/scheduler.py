"""Static-wave vs continuous-refill scheduler for W3 (docs/decisions.md D5).

One :meth:`Scheduler.step` is exactly the sequence frozen in the plan:

1. **Admission** (policy-dependent, only for requests that have arrived):

   * ``continuous`` -- while ``len(active) < max_batch`` and the FIFO head has
     arrived, admit it. A seat freed during step *k* is refilled at step *k+1*,
     because admission runs before retirement inside a step.
   * ``static`` -- only when ``active == 0`` open one wave: wait until
     ``max_batch`` requests have arrived, or, once the trace is known to be
     exhausted (:meth:`Scheduler.close`, or :meth:`Scheduler.run`, which cannot
     receive submissions while it blocks), take whatever is left. A partial wave
     is never waited for forever.

   Each admitted request takes one free seat, records ``t_admitted`` and then
   ``t_prefill_start`` immediately before its own serial ``prefill_into_seat`` (no gap
   between them, so ``ttft == queue + prefill`` holds by construction), which writes
   the request's KV straight into that seat's row, samples the first token (argmax)
   and stamps ``t_first``.

2. **Batch decode** -- one ``decode_batch`` over the whole slab. ``tokens`` is
   always length ``max_batch``: active seats carry their last token, **idle
   seats carry the placeholder** :data:`IDLE_TOKEN`, and their row is frozen
   (mask closed, cursor parked; see :meth:`Scheduler._park_virgin_seats`). Their
   logits are discarded. One argmax over ``(B, vocab)`` does the device-to-host
   read.

3. **Retirement** -- ``budget`` / EOS / ``max_position`` all terminate. The
   seat is frozen (not reclaimed, P-26) and returned to the free list,
   ``t_finish`` is stamped. An engine exception during one request's prefill
   aborts only that request; an exception in the batch decode aborts every
   active request (it cannot be attributed to one row) and the run continues
   with whatever is still queued.

Timing
------
``prefill_ms`` is the sum over this step's admissions of
``t_first - t_prefill_start`` (an aborted admission contributes the time it
actually spent). ``decode_event_ms`` is ``engine.last_decode_event_ms`` when
the engine provides a float, otherwise the decode wall clock. ``step_ms`` is
measured from the last wake-up, so time spent sleeping for a future arrival is
not counted as compute. ``host_gap_ms = step_ms - decode_event_ms -
prefill_ms`` is recorded as-is, including when it is negative.

The module depends on the standard library only; the engine is duck-typed
(``prefill_into_seat`` / ``decode_batch`` / ``freeze_seat`` / ``slab`` /
``vocab_size`` / ``device`` / ``eos_token_id`` / ``max_position``), so the unit
tests can use a fake engine without torch.

W4 paged arm (``docs/w4-plan.md`` D6)
-------------------------------------
The same step applies to a paged engine, through optional hooks only -- nothing is
imported and no branch is taken when they are absent:

* **Admission asks before popping.** If the engine has ``can_admit(prompt_len, budget)``,
  the FIFO head is offered to it *before* the queue pop. A refusal leaves the request
  queued (so its wait is queue time and ``ttft == queue + prefill`` still holds) and
  **stops** the admission pass -- a head that does not fit is never skipped (FIFO).
* **Prefill goes into blocks.** If the engine has ``prefill_into_blocks``, the request is
  admitted with ``prefill_into_blocks(req.id, prompt_ids, seat=seat, budget=budget)``
  instead of ``prefill_into_seat(seat, ...)``; the upper-bound reservation needs the
  budget, and the seat is what lets decode scatter logits back by seat number.
* **Release frees blocks.** If the engine has ``release_request``, retirement calls it
  instead of ``freeze_seat`` (D6: blocks go back to the free list; the seat number still
  returns to the scheduler's free list).
* **Seat-only bookkeeping is skipped.** ``_park_virgin_seats`` and ``_check_pool`` read
  ``engine.slab``, which a paged engine does not have; they run only on the seat path.
* **The step row gains the pool's account** when the engine has ``paged_stats()``; every
  existing field keeps its name (the seat path's rows are bit-for-bit unchanged).
"""

from __future__ import annotations

import heapq
import inspect
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

    def _blocks_available(self, req: Request) -> bool:
        """Ask the engine's admission hook, if any, **before** a seat is popped (W4).

        Pure duck typing: a seat engine has no ``can_admit`` and this returns True, so the
        W3 path is unchanged. A refusal must not be turned into an exception: the request
        stays in the queue and its wait is queue time (``ttft == queue + prefill``).
        """
        can = getattr(self.engine, "can_admit", None)
        if not callable(can):
            return True
        return bool(can(len(req.prompt_ids), req.budget))

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
        self._step_origin = self.now()

        admitted = self._admit()
        prefill_ms = 0.0
        for req in admitted:
            started = req.t_prefill_start
            ended = req.t_first if req.t_first is not None else req.t_finish
            if started is not None and ended is not None:
                prefill_ms += (ended - started) * 1000.0

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
        self._attach_paged_stats(row)
        self._step_no += 1
        self._rows.append(row)
        return row

    def _attach_paged_stats(self, row: dict) -> None:
        """Add the paged pool's account to a step row, when the engine reports one (W4).

        The seat path appends nothing: no existing field is renamed or dropped. On the
        paged arm the row carries what the plan's §5 step row lists -- ``lpad``, block
        counts, the two fragmentation numbers and the three memory figures.
        """
        stats = getattr(self.engine, "paged_stats", None)
        if not self._paged_mode() or not callable(stats):
            return
        for key, value in stats().items():
            row.setdefault(key, value)

    # ------------------------------------------------------------ admission
    def _admit(self) -> list[Request]:
        if self.policy == "continuous":
            return self._admit_continuous()
        return self._admit_static()

    def _admit_continuous(self) -> list[Request]:
        admitted: list[Request] = []
        while len(self._active) < self.max_batch and self._queue:
            head = self._queue[0][0]
            if head > self.now():
                # With nothing running there is nothing to do but wait for the
                # arrival -- sleep, do not spin. With active rows the step starts
                # without the straggler and the next step picks it up if due.
                if self._active:
                    break
                self._wait_until(head)
                if self._queue and self._queue[0][0] > self.now():
                    break  # the injected sleep did not advance the clock: do not spin
                continue
            if not self._blocks_available(self._queue[0][2]):
                # Can't reserve blocks for the FIFO head: leave it queued (its wait counts
                # as queue time) and stop -- never skip it for a later, smaller request.
                self._blocked_this_step = True
                break
            admitted.append(self._start(heapq.heappop(self._queue)[2]))
        return admitted

    def _admit_static(self) -> list[Request]:
        if self._active or not self._queue:
            return []
        # A wave opens with ``max_batch`` requests, or with whatever is left once the
        # trace is known to be exhausted (``close()`` / ``run()``): a partial wave is
        # never waited for forever. Either way every member must have arrived.
        while True:
            target = self._wave_target()
            if target <= self.now():
                break
            self._wait_until(target)
            if self.now() < target:
                # The injected sleep did not advance the clock. Do not spin: either
                # wait for a later submission (trace open) or open with the arrived
                # prefix (trace closed) and pick the rest up on a later step.
                if len(self._queue) < self.max_batch and not self._closed:
                    return []
                break
        if len(self._queue) < self.max_batch and not self._closed:
            # Everything submitted so far has arrived; more must be submitted before
            # the wave can fill. Leave the step empty, do not spin.
            return []

        admitted: list[Request] = []
        now = self.now()
        while len(self._active) < self.max_batch and self._queue and self._queue[0][0] <= now:
            if not self._blocks_available(self._queue[0][2]):
                # FIFO: the wave stops at a member that cannot reserve its blocks; it is
                # opened by a later step once other requests have released theirs.
                self._blocked_this_step = True
                break
            admitted.append(self._start(heapq.heappop(self._queue)[2]))
        return admitted

    def _wave_target(self) -> float:
        """Arrival of the last member of the wave that would open next."""
        size = min(self.max_batch, len(self._queue))
        return max(item[0] for item in heapq.nsmallest(size, self._queue))

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

    def _start(self, req: Request) -> Request:
        if req.budget <= 0:
            # No generation at all (config validation forbids it, but never invent a
            # token past the budget): finish immediately without taking a seat.
            req.t_admitted = req.t_prefill_start = req.t_first = req.t_finish = self.now()
            req.state = "finished"
            self._admitted_this_step += 1
            self._retired_this_step += 1
            return req

        seat = heapq.heappop(self._free)
        if self._virgin_seats:
            self._virgin_seats.discard(seat)
        req.seat = seat
        req.state = "prefilling"
        self._admitted_this_step += 1
        req.t_admitted = self.now()
        # ``t_prefill_start`` is taken immediately before this request's own prefill
        # call, after the seat bookkeeping: ``ttft == queue + prefill`` is then an
        # identity, not a coincidence (C2). Serial prefill of *other* requests is
        # queue time -- exactly the tail continuous batching is meant to compress.
        req.t_prefill_start = self.now()
        try:
            # Prefill writes straight into the seat's row (or, on the paged arm, into
            # freshly reserved blocks); there is no separate hand-off step, so a failure
            # can only leave that row half-written -- the engine rolls it back before
            # re-raising (freeze for a seat, block release for the pool).
            logits = self._prefill(req, seat)
            first = _row_argmax(logits)[0]
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the run
            req.t_first = req.t_finish = self.now()
            self._abort(req, exc)
            return req
        req.out = [first]
        req.t_first = self.now()
        req.state = "decoding"
        self._active.append(req)
        return req

    def _prefill(self, req: Request, seat: int):
        """Admission prefill: into blocks on the paged arm, into the seat row otherwise.

        ``prefill_into_blocks`` gets the request id (the engine's block ledger key), the
        seat (so decode can scatter logits back) and the budget (the reservation is an
        upper bound over ``prompt + budget``, D6). The optional keywords are passed only if
        the engine's signature declares them: a fake engine that predates ``budget`` or
        ``seat`` keeps working, and this is decided by inspection, never by catching a
        ``TypeError`` raised inside the call.
        """
        blocks = getattr(self.engine, "prefill_into_blocks", None)
        if self._paged_mode() and callable(blocks):
            try:
                signature = inspect.signature(blocks)
            except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
                signature = None
            extra = {"seat": seat, "budget": req.budget} if signature is None else {}
            if signature is not None:
                accepted = signature.parameters
                if "seat" in accepted:
                    extra["seat"] = seat
                if "budget" in accepted:
                    extra["budget"] = req.budget
            return blocks(req.id, list(req.prompt_ids), **extra)
        return self.engine.prefill_into_seat(seat, list(req.prompt_ids))

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
