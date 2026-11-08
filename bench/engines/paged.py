"""The paged-KV arm for W4 (``docs/decisions.md`` D6).

``PagedEngine`` is the W4 counterpart of :class:`bench.engines.continuous.ContinuousEngine`:
same trace path, same ``serve/scheduler.py`` admission loop, same three row kinds. The
differences are the pool (physical blocks instead of one ``KvSlab`` row per seat) and the
memory account:

* a request reserves ``ceil((prompt + budget) / block_size)`` blocks **at admission**
  (D6: no eviction, no mid-request allocation), so ``kv_own_mb`` is this request's
  reservation, not ``pool / max_batch``;
* ``kv_mb`` on the trace row is the **total budget** ``kv_mb + workspace_mb`` (P-36:
  charging KV alone re-creates the fake uniform-2048 "win"); the pool's own reservation
  is also recorded as ``kv_pool_mb``;
* every step row carries the engine's ``paged_stats()`` ledger (block counts, internal
  frag, reservation slack, live/alloc/workspace MiB; the Triton arm has no gather
  workspace, so ``workspace_mb`` is structurally zero);
* the summary gains a ``capacity`` block so ``bench/scenario.py`` and
  ``bench/thresholds.py`` can answer C1/C2 without re-parsing rows.

A fill run (``[exp].type = "fill"``) goes through
:class:`bench.engines.fill.FillScheduler`: prefill + reserve, no decode. Its capacity is
``max(step_row["active"])``, i.e. the allocator answer D6 asks for. W4.5's Triton decode
owns no gather workspace, so the paged arm reports ``workspace_mb = 0.0``: that is a real
zero, not a missing estimate. W4's gather-era ``workspace_estimated`` provenance is gone
with the control arm.

Two backends, exactly like the continuous arm:

* ``name != "dummy"`` -- :class:`~bench.engines.minillm.ServeEngine` plus
  ``engine.open_paged_pool(...)``;
* ``name == "dummy"`` -- :class:`_DummyPagedEngine`, a deterministic stand-in with the
  same pool interface (plumbing smoke only; its numbers are meaningless by construction).
"""

from __future__ import annotations

import time
from typing import Any

from bench.engines.continuous import (
    ContinuousEngine,
    TraceResult,
    _request_row,
    _trace_clock,
    _trace_row,
)
from bench.engines.fill import capacity_block
from bench.scenario import aggregate
from serve.scheduler import Request, Scheduler

_DUMMY_NAME = "dummy"

#: Dummy stand-in shape (2 layers, GQA 2/4). Only used to give the dummy arm non-zero,
#: monotone memory numbers; it is not the Qwen3-0.6B shape and must not be compared.
_DUMMY_SHAPE = {
    "n_layer": 2,
    "n_kv_head": 2,
    "n_head": 4,
    "head_dim": 16,
    "dtype_bytes": 2,
}


class PagedEngine(ContinuousEngine):
    """Runner adapter for ``[engine].impl = "paged"``."""

    impl = "paged"

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
        block_size: int = 16,
        kv_blocks: int = 1,
        fill: bool = False,
        max_prefill_tokens: int = 8192,
    ) -> None:
        super().__init__(
            name=name,
            revision=revision,
            dtype=dtype,
            device=device,
            model_path=model_path,
            max_batch=max_batch,
            policy=policy,
            vocab_size=vocab_size,
            seed=seed,
            fill=fill,
            max_prefill_tokens=max_prefill_tokens,
        )
        self.block_size = int(block_size)
        self.kv_blocks = int(kv_blocks)
        # request_id -> blocks reserved at admission (all of them, D6's upper bound).
        self._reserved_blocks: dict[int, int] = {}
        self.kv_pool_mb: float | None = None
        self._instrumented: tuple[Any, Any] = (None, None)

    # ------------------------------------------------------------------ interface
    @property
    def decode_impl(self) -> str:
        """Paged decode kernel recorded in the manifest: always Triton.

        This is constant provenance, not a config switch. W4's gather control arm was
        deleted; ``[engine].decode="triton"`` is tolerated only as a historical no-op and
        ``"gather"`` is rejected by the runner.
        """
        return "triton"

    def load(self) -> None:
        if self.name == _DUMMY_NAME:
            self._engine = _DummyPagedEngine(
                self._vocab, self._seed, block_size=self.block_size, kv_blocks=self.kv_blocks
            )
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

    def generate(self, prompt_ids, max_new_tokens):
        raise NotImplementedError(
            "impl='paged' is a scheduler arm: it runs a whole trace through one "
            "scheduler (run_trace). Use impl='minillm' for per-request generate()."
        )

    # ------------------------------------------------------------------------- trace
    def run_trace(self, trace, *, repeat: int = 0) -> TraceResult:
        if self._engine is None:
            raise RuntimeError("call load() before run_trace()")
        engine = self._engine

        longest_prompt = max((len(a.prompt_ids) for a in trace.arrivals), default=0)
        longest_budget = max((a.budget for a in trace.arrivals), default=0)
        need = longest_prompt + longest_budget + 1
        # D6: the pool is sized by the config's kv_blocks. The position bound is the
        # trace's worst prompt + budget (the RoPE table is a run bound, not
        # ``n_blocks * block_size`` -- P-34).
        engine.open_paged_pool(
            self.kv_blocks,
            self.block_size,
            self.max_batch,
            max_position=int(need),
        )
        self._instrument(engine)
        self.max_len = int(need)
        t0 = time.perf_counter()

        schedule = _fill_scheduler_class() if self.fill else Scheduler
        scheduler = schedule(
            engine,
            max_batch=self.max_batch,
            policy=self.policy,
            now=_trace_clock(t0),
            eos_token_id=self.eos_token_id,
            max_position=int(engine.max_position),
        )
        prompt_ids_by_idx = {a.idx: a.prompt_id for a in trace.arrivals}
        for arrival in trace.arrivals:
            scheduler.submit(
                Request(
                    id=arrival.idx,
                    prompt_ids=list(arrival.prompt_ids),
                    budget=int(arrival.budget),
                    arrival_ts=float(arrival.at_s),
                )
            )
        scheduler.close()
        try:
            requests = scheduler.run()
        finally:
            self._uninstrument()

        stats = self._engine_stats()
        kv_pool = float(stats.get("kv_mb") or 0.0)
        # Triton paged decode owns no gather workspace: this is a measured structural
        # zero (the engine reports it), not a W4 Appendix-C estimate.
        workspace = float(stats.get("workspace_mb") or 0.0)
        self.kv_pool_mb = kv_pool
        self.pool_kv_mb = kv_pool

        t_arrival = min((r.t_arrival for r in requests), default=0.0)
        finishes = [r.t_finish for r in requests if r.t_finish is not None]
        t_finish = max(finishes, default=t_arrival)
        firsts = [r.t_first for r in requests if r.t_first is not None]
        t_first_token = min(firsts, default=t_arrival)
        out_total = sum(len(r.out) for r in requests)

        request_rows = [
            _paged_request_row(
                r,
                prompt_ids_by_idx.get(r.id),
                kv_pool,
                self.max_batch,
                repeat,
                block_size=self.block_size,
                reserved_blocks=self._reserved_blocks.get(int(r.id)),
                per_token_bytes=self._per_token_bytes(),
            )
            for r in requests
        ]
        step_rows = [dict(row, row_kind="step", repeat=repeat) for row in scheduler.step_rows]
        for row in step_rows:
            row.setdefault("workspace_mb", workspace)
            row.setdefault("kv_pool_mb", kv_pool)

        trace_row = _trace_row(
            requests=requests,
            step_rows=step_rows,
            repeat=repeat,
            policy=self.policy,
            max_batch=self.max_batch,
            offered_rate=trace.rate,
            pool_kv_mb=kv_pool,
            trace_sha256=trace.sha256,
            t_arrival=t_arrival,
            t_first_token=t_first_token,
            t_finish=t_finish,
            out_total=out_total,
        )
        total = kv_pool + workspace
        trace_row.update(
            {
                # D6 / P-36: the budget number is KV + workspace; the pool alone is
                # kept beside it so the two are never confused.
                "kv_mb": total,
                "kv_pool_mb": kv_pool,
                "workspace_mb": workspace,
                "kv_total_mb": total,
                "block_size": self.block_size,
                "kv_blocks": self.kv_blocks,
                "fill_only": bool(self.fill),
            }
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
        peak = max((int(row.get("active") or 0) for row in step_rows), default=0)
        summary["capacity"] = capacity_block(
            peak_concurrency=peak,
            kv_mb=kv_pool,
            workspace_mb=workspace,
            kv_total_mb=total,
            block_size=self.block_size,
            kv_blocks=self.kv_blocks,
            max_batch=self.max_batch,
            frag_internal=_last_stat(step_rows, "frag_internal"),
            reservation_slack=_last_stat(step_rows, "reservation_slack"),
            seat_unused_mb=None,
            workload_sha256=trace.sha256,
        )
        return TraceResult(
            request_rows=request_rows,
            step_rows=step_rows,
            trace_row=trace_row,
            summary=summary,
            repeat=repeat,
            impl=self.impl,
        )

    # ------------------------------------------------------------------ internals
    def _instrument(self, engine: Any) -> None:
        """Record each request's reserved block count from a successful admission.

        The engine frees the blocks on release, so the reservation is gone by the time
        the run returns. The wrapper only reads back the (successful) call -- D6's
        reservation size is ``blocks_for(prompt, budget)`` either way, and an aborted
        prefill never gets a row to charge.

        W4.5 (D7) admits a whole packed group per forward, so the wrapper sits on
        ``prefill_group``: after a successful forward, every member of the group is
        charged its own ``blocks_for(prompt, budget)`` reservation. A failed group
        records nothing, which is the "aborted prefill gets no row" rule above.

        Wraps the instance, not the class, and restores the original in
        :meth:`_uninstrument` so repeated repeats do not stack wrappers.
        """
        self._reserved_blocks = {}
        original = getattr(engine, "prefill_group", None)
        if not callable(original):
            return
        block_size = self.block_size

        def wrapped(requests):
            logits = original(requests)
            for spec in requests:
                prompt = list(spec.get("prompt_ids") or [])
                budget = int(spec.get("budget") or 0)
                reserved = max(1, -(-(len(prompt) + budget) // block_size))
                request_id = spec.get("id")
                if request_id is not None:
                    self._reserved_blocks[int(request_id)] = int(reserved)
            return logits

        engine.prefill_group = wrapped
        self._instrumented = (engine, original)

    def _uninstrument(self) -> None:
        engine, original = getattr(self, "_instrumented", (None, None))
        if engine is not None and original is not None:
            engine.prefill_group = original
        self._instrumented = (None, None)

    def _engine_stats(self) -> dict[str, Any]:
        if self._engine is None:
            return {}
        stats = getattr(self._engine, "paged_stats", None)
        if not callable(stats):
            return {}
        return dict(stats() or {})

    def _kv_shape(self) -> dict[str, int]:
        """KV shape for the per-token byte account (dummy defaults, real model otherwise)."""
        engine = self._engine
        if engine is not None and engine.__class__.__name__ != "_DummyPagedEngine":
            model = getattr(engine, "model", None)
            config = getattr(model, "config", None)
            dtype = getattr(getattr(model, "token_embedding", None), "weight", None)
            dtype_bytes = int(getattr(dtype, "element_size", lambda: 2)())
            return {
                "n_kv_head": int(getattr(config, "n_kv_head", 8)),
                "n_head": int(getattr(config, "n_head", 16)),
                "head_dim": int(getattr(model, "head_dim", 128)),
                "dtype_bytes": dtype_bytes,
            }
        if engine is not None:
            return {
                "n_kv_head": int(getattr(engine, "n_kv_head", _DUMMY_SHAPE["n_kv_head"])),
                "n_head": int(getattr(engine, "n_head", _DUMMY_SHAPE["n_head"])),
                "head_dim": int(getattr(engine, "head_dim", _DUMMY_SHAPE["head_dim"])),
                "dtype_bytes": int(getattr(engine, "dtype_bytes", _DUMMY_SHAPE["dtype_bytes"])),
            }
        return dict(_DUMMY_SHAPE)

    def _per_token_bytes(self) -> int:
        """All layers, K and V: ``2 * n_layer * n_kv_head * head_dim * itemsize``."""
        engine = self._engine
        if engine is not None:
            n_layer = int(getattr(engine, "n_layer", 0) or 0)
            if n_layer <= 0:
                model = getattr(engine, "model", None)
                n_layer = int(getattr(getattr(model, "config", None), "n_layer", 0) or 0)
        else:
            n_layer = 0
        shape = self._kv_shape()
        return 2 * max(n_layer, 0) * shape["n_kv_head"] * shape["head_dim"] * shape["dtype_bytes"]


# --------------------------------------------------------------------------- rows
def _last_stat(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _paged_request_row(
    request: Request,
    prompt_id: str | None,
    kv_pool_mb: float,
    max_batch: int,
    repeat: int,
    *,
    block_size: int,
    reserved_blocks: int | None,
    per_token_bytes: int,
) -> dict[str, Any]:
    """``_request_row`` plus the D6 per-request block account.

    ``kv_mb`` keeps the continuous meaning (the whole pool's reservation, which is what
    the memory account divides by); ``kv_own_mb`` changes meaning on purpose -- it is
    this request's **reserved** bytes (``blocks_reserved * block_size * per_token``),
    not ``pool / max_batch`` (D6). ``blocks_written`` counts only blocks that hold a
    written token (the last partial block counts; the engine uses the same rule).
    """
    row = _request_row(request, prompt_id, kv_pool_mb, max_batch, repeat)
    out_tokens = len(request.out)
    written_tokens = len(request.prompt_ids) + out_tokens
    written_blocks = max(1, -(-written_tokens // block_size))
    reserved = reserved_blocks if reserved_blocks is not None else written_blocks
    row["kv_own_mb"] = reserved * block_size * per_token_bytes / 2**20 if per_token_bytes else None
    row["blocks_reserved"] = int(reserved)
    row["blocks_written"] = int(written_blocks)
    row["block_size"] = int(block_size)
    row["fill_only"] = not out_tokens
    return row


def _fill_scheduler_class():
    from bench.engines.fill import FillScheduler

    return FillScheduler


# --------------------------------------------------------------------------- dummy
class _DummyPagedEngine:
    """Deterministic stand-in with the paged pool interface (plumbing smoke only).

    Blocks are integers from :class:`serve.block_manager.BlockManager`, so admission,
    reservation, release and the ledger behave exactly like the real pool's. There is
    no KV; prefill returns the same one-hot row the continuous dummy does.
    """

    impl = "paged"
    paged = True

    def __init__(self, vocab_size: int, seed: int, *, block_size: int, kv_blocks: int) -> None:
        from serve.block_manager import BlockManager, FragStats

        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.device = "cpu"
        self.max_position = 4096
        self.eos_token_id = None
        self.last_decode_event_ms = None
        self.block_size = int(block_size)
        self.n_layer = _DUMMY_SHAPE["n_layer"]
        self.n_kv_head = _DUMMY_SHAPE["n_kv_head"]
        self.n_head = _DUMMY_SHAPE["n_head"]
        self.head_dim = _DUMMY_SHAPE["head_dim"]
        self.dtype_bytes = _DUMMY_SHAPE["dtype_bytes"]
        self._manager = BlockManager(int(kv_blocks), int(block_size))
        self._frag = FragStats(int(block_size))
        self._max_batch: int | None = None
        self._max_position: int | None = None
        self._state: dict[int, dict] = {}
        self._tokens: dict[int, list[int]] = {}

    # ---------------------------------------------------------------- pool
    def open_paged_pool(self, kv_blocks, block_size, max_batch, max_position=None):
        from serve.block_manager import BlockManager, FragStats

        self.block_size = int(block_size)
        self._manager = BlockManager(int(kv_blocks), int(block_size))
        self._frag = FragStats(int(block_size))
        self._max_batch = int(max_batch)
        self._max_position = int(self.max_position if max_position is None else max_position)
        self._state = {}
        self._tokens = {}
        return self._manager

    @property
    def paged_max_batch(self) -> int | None:
        return self._max_batch

    def close_pool(self) -> None:
        self._manager = None  # type: ignore[assignment]
        self._state = {}
        self._tokens = {}

    def _require_pool(self) -> None:
        if self._manager is None:
            raise RuntimeError("open_paged_pool() must be called before the scheduler runs")

    def can_admit_group(self, requests: list[dict]) -> bool:
        """Whether the whole packed group fits: row cap plus every member's blocks."""
        self._require_pool()
        if len(self._state) + len(requests) > int(self._max_batch or 0):
            return False
        needed = sum(
            self._manager.blocks_for(
                len(spec.get("prompt_ids") or []), int(spec.get("budget") or 0)
            )
            for spec in requests
        )
        return needed <= self._manager.stats()["free"]

    # ---------------------------------------------------------------- steps
    def prefill_group(self, requests: list[dict]):
        """Reserve every member's blocks, then register the group all-or-nothing."""
        self._require_pool()
        if not requests:
            raise ValueError("prefill_group needs at least one request")
        reserved: list[tuple[int, list[int], int, int, tuple[int, ...]]] = []
        used_seats: set[int] = set()
        try:
            for spec in requests:
                request_id = spec.get("id")
                if request_id is None:
                    request_id = -len(reserved) - 1
                request_id = int(request_id)
                if request_id in self._state:
                    raise ValueError(f"request {request_id} is already in the paged pool")
                prompt = list(spec.get("prompt_ids") or [])
                if not prompt:
                    raise ValueError("prompt_ids must not be empty")
                seat = spec.get("seat")
                if seat is None:
                    occupied = {state["seat"] for state in self._state.values()} | used_seats
                    seat = next(
                        (row for row in range(int(self._max_batch or 0)) if row not in occupied),
                        None,
                    )
                    if seat is None:
                        raise RuntimeError("no free seat id for the paged request")
                seat = int(seat)
                if not 0 <= seat < int(self._max_batch or 0):
                    raise IndexError(f"seat {seat} outside the paged pool's max_batch")
                if seat in {state["seat"] for state in self._state.values()}:
                    raise ValueError(f"seat {seat} is already occupied by a paged request")
                if seat in used_seats:
                    raise ValueError(f"two requests in one packed group share seat {seat}")
                used_seats.add(seat)
                needed = self._manager.blocks_for(len(prompt), int(spec.get("budget") or 0))
                block_table = self._manager.reserve(needed)
                if block_table is None:
                    raise RuntimeError(
                        f"cannot reserve {needed} blocks for request {request_id} "
                        "(can_admit_group should have refused the group)"
                    )
                reserved.append((request_id, list(block_table), len(prompt), seat, tuple(prompt)))
        except BaseException:
            for _, block_table, _, _, _ in reserved:
                self._manager.free(block_table)
            raise
        rows = []
        for request_id, block_table, length, seat, prompt in reserved:
            self._state[request_id] = {
                "block_table": block_table,
                "prompt_len": length,
                "length": length,
                "seat": seat,
            }
            self._tokens[seat] = []
            first = (self.seed + prompt[-1] + sum(prompt)) % self.vocab_size
            rows.append(self._row(first))
        return rows

    def release_request(self, request_id: int) -> None:
        if self._manager is None:
            return
        state = self._state.pop(int(request_id), None)
        if state is None:
            return
        self._manager.free(state["block_table"])

    def decode_batch(self, tokens, slab=None):
        self._require_pool()
        if slab is not None:
            raise ValueError("the paged arm has no seat slab to hand over")
        out = []
        active = sorted(self._state.items(), key=lambda item: item[1]["seat"])
        by_seat = {state["seat"]: (request_id, state) for request_id, state in active}
        for seat, token in enumerate(tokens):
            entry = by_seat.get(seat)
            if entry is None:
                out.append(self._row(0))
                continue
            request_id, state = entry
            history = self._tokens.setdefault(seat, [])
            nxt = (self.seed + int(token) + len(history)) % self.vocab_size
            history.append(nxt)
            state["length"] += 1
            out.append(self._row(nxt))
        return out

    # ---------------------------------------------------------------- ledger
    def paged_stats(self) -> dict:
        self._require_pool()
        in_use = self._manager.in_use
        written_blocks = sum(
            max(1, -(-int(state["length"]) // self.block_size)) for state in self._state.values()
        )
        written_tokens = sum(int(state["length"]) for state in self._state.values())
        per_token = (
            2 * self.n_layer * self.n_kv_head * self.head_dim * self.dtype_bytes
        )
        pool_mb = (
            self._manager.n_blocks * self.block_size * per_token / 2**20
            if per_token
            else 0.0
        )
        reserved_tokens = in_use * self.block_size
        return {
            "kv_mb": pool_mb,
            "kv_live_mb": written_tokens * per_token / 2**20,
            "kv_alloc_mb": reserved_tokens * per_token / 2**20,
            "workspace_mb": 0.0,
            "blocks_in_use": in_use,
            "free_blocks": self._manager.stats()["free"],
            "written_blocks": written_blocks,
            "frag_internal": 1.0 - (written_tokens / reserved_tokens) if reserved_tokens else None,
            "reservation_slack": (
                1.0 - written_tokens / reserved_tokens if reserved_tokens else None
            ),
            # The stand-in has the same key set as the real paged arm: no gather
            # workspace, and constant Triton provenance.
            "workspace_mb": 0.0,
            "decode_impl": "triton",
        }

    # ---------------------------------------------------------------- rows
    def _row(self, best: int) -> list[float]:
        row = [0.0] * self.vocab_size
        row[int(best) % self.vocab_size] = 1.0
        return row
