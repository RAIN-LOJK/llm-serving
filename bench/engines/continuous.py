"""The continuous-batching arm: one scheduler run over a trace (W3 / docs/decisions.md D5).

The arm is deliberately thin. Everything that decides what runs when lives in
``serve/scheduler.py``; everything that turns the resulting rows into numbers lives in
``bench/scenario.py``. This module only joins them to a model engine, so the static and
continuous arms share one code path and their ratio measures scheduling rather than two
different implementations (P-23).

``generate()`` raises on purpose: a scenario run is not a list of independent ``generate``
calls. Silently falling back to that would produce a "batch" number that never batched
anything (P-20).

Two backends:

* ``name != "dummy"`` -- the real decoder, through :class:`~bench.engines.minillm.ServeEngine`,
  so the scenario arm and the single-request arm tokenize and load identically;
* ``name == "dummy"`` -- a deterministic stand-in with the same pool interface. It exists so
  the runner/scenario/summary plumbing can be smoked on a CPU box with no weights. Its
  numbers are meaningless by construction, exactly like ``DummyEngine``.
"""

from __future__ import annotations

import hashlib
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Sequence

from bench.loadgen import Trace
from bench.scenario import aggregate
from serve.scheduler import Request, Scheduler

_DUMMY_NAME = "dummy"


@dataclass
class TraceResult:
    """One scheduler run: the three row kinds the runner writes, plus the summary block."""

    request_rows: list[dict[str, Any]] = field(default_factory=list)
    step_rows: list[dict[str, Any]] = field(default_factory=list)
    trace_row: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    repeat: int = 0
    #: Which arm produced the rows. ``_flatten_scenario`` stamps it onto every row; the
    #: default keeps every W3 continuous row byte-for-byte what it was.
    impl: str = "continuous"


class ContinuousEngine:
    """Runner adapter for ``[engine].impl = "continuous"``."""

    impl = "continuous"

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        dtype: str = "float32",
        device: str = "cpu",
        model_path: str | None = None,
        *,
        max_batch: int = 1,
        policy: str = "continuous",
        vocab_size: int = 1024,
        seed: int = 0,
        fill: bool = False,
        max_prefill_tokens: int = 8192,
    ) -> None:
        self.name = name
        self.revision = revision
        self.dtype_name = dtype
        self.device = "cpu" if device in {"", "auto"} else device
        self.max_batch = int(max_batch)
        self.policy = str(policy)
        # W4 E1 (`[exp].type = "fill"`): prefill + reserve only, no decode. The default
        # keeps every W3 scenario row on the original path (``run_fill_trace`` is only
        # entered when a fill config asks for it).
        self.fill = bool(fill)
        # W4.5 (D7): production prefill is always the packed FIFO group; there is no
        # serial control arm any more. ``max_prefill_tokens`` is the per-step packing
        # budget `serve/scheduler.py` reads from the engine. There is no decode switch
        # either: the seat arm always uses hand_baddbmm, the paged arm always Triton (the
        # gather control arm was deleted).
        self._max_prefill_tokens = int(max_prefill_tokens)
        self.model_path = model_path
        self._vocab = int(vocab_size)
        self._seed = int(seed)
        self._delegate = None  # type: Any
        self._engine = None  # type: Any
        self.pool_kv_mb: float | None = None
        self.max_len = 0

    # ------------------------------------------------------------------ interface
    @property
    def vocab_size(self) -> int:
        if self._delegate is not None:
            return int(self._delegate.vocab_size)
        return self._vocab

    @property
    def eos_token_id(self) -> int | None:
        return None if self._delegate is None else self._delegate.eos_token_id

    @property
    def max_prefill_tokens(self) -> int:
        """Packing budget per step; one prompt wider than this forms its own group."""
        return self._max_prefill_tokens

    @property
    def decode_impl(self) -> str:
        """Decode backend of the seat arm: no block table, so ``hand_baddbmm`` over KvSlab."""
        return "hand_baddbmm"

    @property
    def weights_mb(self) -> float:
        """Parameter footprint the memory account needs; the dummy arm has none."""
        return 0.0 if self._delegate is None else float(self._delegate.weights_mb)

    def load(self) -> None:
        if self.name == _DUMMY_NAME:
            self._engine = _DummyPoolEngine(self._vocab, self._seed)
            self.device = self._engine.device
            return
        from bench.engines.minillm import ServeEngine

        self._delegate = ServeEngine(
            name=self.name,
            revision=self.revision,
            dtype=self.dtype_name,
            device=self.device,
            model_path=self.model_path,
            max_prefill_tokens=self.max_prefill_tokens,
        )
        self._delegate.load()
        self.device = self._delegate.device
        self._engine = self._delegate.engine

    def encode(self, text: str) -> list[int]:
        if self._delegate is None:
            # The dummy arm never tokenizes; the length heuristic mirrors DummyEngine.
            length = max(1, min(len(text) // 2, 2048))
            base = zlib.crc32(text.encode("utf-8")) + self._seed
            return [(base + i) % self._vocab for i in range(length)]
        return self._delegate.encode(text)

    def attn_backends(self) -> dict[str, str]:
        if self._delegate is None:
            return {"prefill": "dummy", "decode": "dummy"}
        return self._delegate.attn_backends()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close_pool()
        if self._delegate is not None:
            self._delegate.close()
        self._engine = None
        self._delegate = None

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int):
        raise NotImplementedError(
            "impl='continuous' is a scenario arm: it runs a whole trace through one "
            "scheduler (run_trace). Use impl='minillm' for per-request generate()."
        )

    # ------------------------------------------------------------------------- trace
    def run_trace(self, trace: Trace, *, repeat: int = 0) -> TraceResult:
        """Run every arrival of ``trace`` through the scheduler and aggregate the rows."""
        if self.fill:
            # W4 E1: same pool/clock/trace plumbing, but the step loop stops after
            # admission. Lives in ``fill.py`` so the scenario path below stays untouched.
            from bench.engines.fill import run_fill_trace

            return run_fill_trace(self, trace, repeat=repeat)
        if self._engine is None:
            raise RuntimeError("call load() before run_trace()")
        engine = self._engine

        longest_prompt = max((len(a.prompt_ids) for a in trace.arrivals), default=0)
        longest_budget = max((a.budget for a in trace.arrivals), default=0)
        need = longest_prompt + longest_budget + 1
        slab = engine.open_pool(self.max_batch, need)
        t0 = time.perf_counter()
        self.pool_kv_mb = engine.pool_kv_mb
        self.max_len = int(getattr(slab, "max_len", need))

        scheduler = Scheduler(
            engine,
            max_batch=self.max_batch,
            policy=self.policy,
            # Trace arrivals are relative ("seconds after the run starts"); wall-clock
            # timestamps are absolute. Shifting ``now`` onto the trace's origin is what
            # makes ``t_arrival``, queue time and makespan comparable at all.
            now=_trace_clock(t0),
            eos_token_id=self.eos_token_id,
            max_position=int(engine.max_position),
        )
        prompt_ids_by_idx = {a.idx: a.prompt_id for a in trace.arrivals}
        for arrival in trace.arrivals:
            request = Request(
                id=arrival.idx,
                prompt_ids=list(arrival.prompt_ids),
                budget=int(arrival.budget),
                arrival_ts=float(arrival.at_s),
            )
            scheduler.submit(request)
        scheduler.close()

        requests = scheduler.run()

        t_arrival = min((r.t_arrival for r in requests), default=0.0)
        finishes = [r.t_finish for r in requests if r.t_finish is not None]
        t_finish = max(finishes, default=t_arrival)
        firsts = [r.t_first for r in requests if r.t_first is not None]
        t_first_token = min(firsts, default=t_arrival)
        out_total = sum(len(r.out) for r in requests)

        request_rows = [
            _request_row(r, prompt_ids_by_idx.get(r.id), self.pool_kv_mb, self.max_batch, repeat)
            for r in requests
        ]
        step_rows = [dict(row, row_kind="step", repeat=repeat) for row in scheduler.step_rows]
        trace_row = _trace_row(
            requests=requests,
            step_rows=step_rows,
            repeat=repeat,
            policy=self.policy,
            max_batch=self.max_batch,
            offered_rate=trace.rate,
            pool_kv_mb=self.pool_kv_mb,
            trace_sha256=trace.sha256,
            t_arrival=t_arrival,
            t_first_token=t_first_token,
            t_finish=t_finish,
            out_total=out_total,
        )

        summary = aggregate(
            request_rows,
            step_rows,
            [trace_row],
            budgets=trace.budgets,
            max_batch=self.max_batch,
        )
        latency = summary.get("latency") or {}
        trace_row.update(
            {
                "ttft_p99_ms": (latency.get("ttft_ms") or {}).get("p99"),
                "queue_p99_ms": (latency.get("queue_ms") or {}).get("p99"),
                "prefill_p99_ms": (latency.get("prefill_ms") or {}).get("p99"),
            }
        )
        return TraceResult(
            request_rows=request_rows,
            step_rows=step_rows,
            trace_row=trace_row,
            summary=summary,
            repeat=repeat,
        )

    @staticmethod
    def token_sha256(tokens: Sequence[int]) -> str:
        payload = ",".join(str(int(t)) for t in tokens)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- rows
def _trace_clock(origin: float):
    """``perf_counter`` relative to the trace's start, so arrivals are comparable."""

    def now() -> float:
        return time.perf_counter() - origin

    return now


def _rate(count: int, seconds: float) -> float | None:
    return (count / seconds) if seconds and seconds > 0 else None


def _queue_depth_at_last_arrival(requests: Sequence[Request], max_batch: int) -> int:
    """Backlog still waiting for a seat when the trace is fully submitted.

    The trace is "fully submitted" at ``t_last_arrival = max(t_arrival)``::

        backlog   = #{arrived by then and not yet finished}
        in_flight = #{prefill started by then and not yet finished}
        free      = max_batch - in_flight      (seats that can take a request now)
        depth     = max(0, backlog - in_flight - free)

    ``t_prefill_start <= t_last_arrival`` is deliberately inclusive: a request whose
    prefill starts at that same instant is being served, not queued.

    The ``free`` term is what makes the instant well defined. A seat that is free can
    take a waiting request immediately, so a request that is merely waiting for the
    next admission pass is not backlog; and a run that ends has always drained its
    queue, so the final step row's ``queued`` cannot answer this either (P-29). It is
    also why the E1 closed-batch shape (``requests == max_batch``, every arrival at
    t=0) reads 0: at that instant every seat is free, so nothing is waiting *for
    capacity*. A closed trace with more requests than seats, or arrivals that pile up
    between admission passes, still reads positive.

    Since ``in_flight <= max_batch`` always holds, the expression reduces to
    ``max(0, backlog - max_batch)``: the requests that cannot get a seat right now.
    """
    if not requests:
        return 0
    t_last_arrival = max(request.t_arrival for request in requests)
    backlog = sum(
        1
        for request in requests
        if request.t_arrival <= t_last_arrival
        and (request.t_finish is None or request.t_finish > t_last_arrival)
    )
    in_flight = sum(
        1
        for request in requests
        if request.t_prefill_start is not None
        and request.t_prefill_start <= t_last_arrival
        and (request.t_finish is None or request.t_finish > t_last_arrival)
    )
    free_seats = max(0, int(max_batch) - in_flight)
    return max(0, backlog - in_flight - free_seats)


def _trace_row(
    *,
    requests: Sequence[Request],
    step_rows: Sequence[dict[str, Any]],
    repeat: int,
    policy: str,
    max_batch: int,
    offered_rate: float | None,
    pool_kv_mb: float | None,
    trace_sha256: str,
    t_arrival: float,
    t_first_token: float,
    t_finish: float,
    out_total: int,
) -> dict[str, Any]:
    makespan = t_finish - t_arrival
    return {
        "row_kind": "trace",
        "repeat": repeat,
        "policy": policy,
        "max_batch": max_batch,
        "makespan_s": round(makespan, 6),
        "out_tokens_total": out_total,
        "tok_s": _rate(out_total, makespan),
        "tok_s_active": _rate(out_total, t_finish - t_first_token),
        "req_s": _rate(len(requests), makespan),
        "offered_rate": offered_rate,
        "achieved_rate": _rate(len(requests), makespan),
        "queue_depth_max": max((int(row.get("queued") or 0) for row in step_rows), default=0),
        # Saturation is "the trace ran out but the queue did not" (D5 / P-29); it never
        # consults a capacity estimate, which would be a per-arm self-comparison. It is
        # read at the moment the *last* request arrives, never at the end of the run:
        # a run that finished has necessarily drained its queue, so the final step row
        # would make saturation False by construction.
        "queue_depth_at_last_arrival": _queue_depth_at_last_arrival(requests, max_batch),
        "peak_concurrency": max((int(row.get("active") or 0) for row in step_rows), default=0),
        "kv_mb": pool_kv_mb,
        "trace_sha256": trace_sha256,
        "drops": 0,
    }


def _request_row(
    request: Request,
    prompt_id: str | None,
    pool_kv_mb: float | None,
    max_batch: int,
    repeat: int,
) -> dict[str, Any]:
    """One request row in the field names ``bench/scenario.py`` reads (D5).

    W4.5 adds the packed-prefill attribution (D7 TTFT row)::

        queue_ms          = t_admitted - t_arrival       (W3: t_prefill_start - t_arrival)
        pack_wait_ms      = t_prefill_start - t_admitted  (0.0 without a scheduler stamp)
        prefill_shared_ms = t_first - t_prefill_start    (same as prefill_ms for a group of 1)
        prefill_group     = the scheduler's packed group

    Every scheduler run is packed now (D7); the ``t_admitted`` fallback exists for
    historical W3/W4 request rows, which have no admission stamp and no packed group.
    ``prefill_ms``/``ttft_ms`` are unchanged: they are the frozen D5 names, not a second
    copy of the new attribution.
    """
    arrival = request.t_arrival
    first = request.t_first if request.t_first is not None else arrival
    finish = request.t_finish if request.t_finish is not None else first
    stamped_start = request.t_prefill_start
    prefill_start = stamped_start if stamped_start is not None else arrival
    admitted = getattr(request, "t_admitted", None)
    # Historical W3 rows had no admission stamp; falling back to ``t_prefill_start``
    # reproduces their queue_ms exactly.
    queue_start = admitted if admitted is not None else prefill_start
    # "0.0 if either timestamp is missing": a request that never started prefill has no
    # pack wait to attribute, and must not invent a negative one from the arrival fallback.
    pack_wait_ms = (
        (stamped_start - admitted) * 1000.0
        if stamped_start is not None and admitted is not None
        else 0.0
    )
    out_tokens = len(request.out)
    decode_s = max(0.0, finish - first)
    return {
        "row_kind": "request",
        "repeat": repeat,
        "arrival_idx": request.id,
        "prompt_id": prompt_id,
        "budget": request.budget,
        "state": request.state,
        "seat": request.seat,
        "out_tokens": out_tokens,
        # The scenario's `context` is the tokens the request actually occupied, not just the
        # prompt: `_summary` buckets on it, and a bucket that ignored the generated tokens
        # would put a 4-token and an 8-token decode in the same row (P-30).
        "context": len(request.prompt_ids) + out_tokens,
        "t_arrival_s": round(arrival, 6),
        "queue_ms": (queue_start - arrival) * 1000.0,
        "pack_wait_ms": pack_wait_ms,
        "prefill_ms": (first - prefill_start) * 1000.0,
        "prefill_shared_ms": (first - prefill_start) * 1000.0,
        "prefill_group": getattr(request, "prefill_group", None),
        "ttft_ms": (first - arrival) * 1000.0,
        "tpot_ms": (decode_s * 1000.0 / (out_tokens - 1)) if out_tokens > 1 else None,
        "e2e_ms": (finish - arrival) * 1000.0,
        # The pool's whole reservation, not this request's share: that is what
        # `memory_match_ratio` divides by (D5 / P-30). The share is `kv_own_mb`.
        "kv_mb": pool_kv_mb,
        "kv_own_mb": (pool_kv_mb / max(1, int(max_batch))) if pool_kv_mb else None,
        "token_ids": list(request.out),
        "token_sha256": ContinuousEngine.token_sha256(request.out),
        "ok": request.state != "aborted",
        "error": request.error,
    }


# --------------------------------------------------------------------------- dummy
class _DummyPoolEngine:
    """Deterministic stand-in with the pool interface; exercises the plumbing only."""

    def __init__(self, vocab_size: int, seed: int) -> None:
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.device = "cpu"
        self.max_position = 4096
        self.eos_token_id = None
        self.last_decode_event_ms = None
        #: Packing budget the scheduler reads; the stand-in never needs to split.
        self.max_prefill_tokens = 8192
        self._slab = None  # type: Any
        self._tokens: dict[int, list[int]] = {}

    # ---------------------------------------------------------------- pool
    def open_pool(self, max_batch: int, max_len: int):
        self._slab = _DummySlab(int(max_batch), int(max_len), self.vocab_size)
        self._tokens = {}
        return self._slab

    @property
    def slab(self):
        if self._slab is None:
            raise RuntimeError("open_pool() must be called before the scheduler runs")
        return self._slab

    @property
    def pool_kv_mb(self) -> float | None:
        return None if self._slab is None else self._slab.mb()

    def close_pool(self) -> None:
        self._slab = None

    # ---------------------------------------------------------------- steps
    def prefill_group(self, requests: list[dict]):
        """Packed admission for the stand-in: one first token per request, one row each.

        There is no KV to write, so this is exactly what the real engine's
        ``prefill_group`` does for the seat arm: activate the row, start its token history
        empty, and return one ``(1, vocab)`` row per request. The group is validated
        before any row is touched, so a rejected member cannot leave a half-activated seat.
        """
        if not requests:
            raise ValueError("prefill_group needs at least one request")
        seats: list[tuple[int, tuple[int, ...]]] = []
        for spec in requests:
            prompt = list(spec.get("prompt_ids") or [])
            if not prompt:
                raise ValueError("prompt_ids must not be empty")
            if len(prompt) > self.max_position:
                raise ValueError("prompt longer than max_position_embeddings")
            seat = spec.get("seat")
            if seat is None or not 0 <= int(seat) < self.slab.batch:
                raise IndexError(f"seat {seat!r} outside the slab's batch of {self.slab.batch}")
            seats.append((int(seat), tuple(prompt)))
        rows = []
        for seat, prompt in seats:
            self._tokens[seat] = []
            self.slab.activate(seat, len(prompt))
            first = (self.seed + prompt[-1] + sum(prompt)) % self.vocab_size
            rows.append(self._row(first))
        return rows

    def freeze_seat(self, slot: int) -> None:
        self.slab.freeze(slot)
        self._tokens.pop(slot, None)

    def decode_batch(self, tokens: list[int], slab=None):
        slab = self.slab if slab is None else slab
        out = []
        for seat, token in enumerate(tokens):
            history = self._tokens.setdefault(seat, [])
            nxt = (self.seed + int(token) + len(history)) % self.vocab_size
            if slab.is_active(seat):
                history.append(nxt)
            out.append(self._row(nxt))
        return out

    def _row(self, best: int) -> list[float]:
        """A one-hot row: ``argmax`` is the intent and the values are irrelevant."""
        row = [0.0] * self.vocab_size
        row[int(best) % self.vocab_size] = 1.0
        return row


class _DummySlab:
    def __init__(self, batch: int, max_len: int, vocab_size: int) -> None:
        self.batch = int(batch)
        self.max_len = int(max_len)
        self._vocab = int(vocab_size)
        self._active = [True] * self.batch
        self.pos = [0] * self.batch

    def activate(self, row: int, length: int) -> None:
        self._active[row] = True
        self.pos[row] = int(length)

    def freeze(self, row: int) -> None:
        self._active[row] = False
        self.pos[row] = 0

    def is_active(self, row: int) -> bool:
        return bool(self._active[row])

    def active_count(self) -> int:
        return sum(1 for a in self._active if a)

    def mb(self) -> float:
        # Arbitrary but monotone in the reservation; the dummy arm reports no real memory.
        return self.batch * self.max_len * 2 * 2 * 28 * 8 * 128 / 2**20 / 512
