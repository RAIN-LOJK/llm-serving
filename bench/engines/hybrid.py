"""The hybrid-attention arm: Qwen3.5-0.8B under the same ``Scheduler`` (W7 Step 2/3).

``HybridEngine`` is the runner adapter for ``[engine].impl = "hybrid"``. It joins the two
W7 halves that live in ``serve/``:

* :class:`~serve.hybrid_engine.Qwen35Engine` -- pooled linear state (18 GDN layers) plus a
  six-layer full-attention KV container (paged block table, or the ``KvSlab`` seat
  baseline);
* :class:`~serve.scheduler.Scheduler` -- the **same** admission/decode/retire loop the 0.6B
  arms use. The scheduler sees only the engine protocol; nothing in this module (or in
  ``serve/scheduler.py``) branches on the model.

Like the ``paged`` arm, a scenario run is one whole trace through one scheduler;
``generate()`` raises on purpose. A trajectory that fell back to independent
``generate`` calls would produce a "batch" number that never batched anything (P-20).

Packed group prefill (W7 Step 3)
--------------------------------
``Qwen35Engine.prefill_group`` runs **one packed varlen forward per FIFO group** (W7
Step 3, D9-6): the 6 full-attention layers use ``cu_seqlens`` and write each request's KV
through its own sink; the 18 GDN layers loop over the group's segments, each from empty
state. The manifest therefore records ``prefill = "packed"`` and the actual varlen backend
(``flash-attn-varlen`` on CUDA, ``sdpa-varlen`` on CPU/tests) in ``engine.attn``. The
packing budget ``max_prefill_tokens`` is forwarded because the scheduler uses it to bound
group sizes.

Fill residency (W7 Step 3)
--------------------------
``[exp].type = "fill"`` runs through :class:`~bench.engines.fill.FillScheduler`, the same
fill-only scheduler the 0.6B arms use: one FIFO group per step, packed prefill, then the
whole group stays **resident** until a pass cannot admit anymore (``docs/w7-plan.md``
§5.4). That is what makes ``capacity.peak_concurrency`` the pool's cross-step capacity
(the number E2 predicts) instead of one group's size. Residents are released by the
scheduler's final drain; ``close_pool`` stays the adapter's job.

For a fill run the pool ledger has to be read *while the residents hold their state*, so
the capacity block takes its ``state_used_mb`` high-water from the step snapshots (linear
state + written KV at the same step) and its ``state_reserved_mb`` from the pool's own
tensors -- every linear slot plus the whole KV container (the run's final
``paged_stats()`` still has the reservation even though every slot was freed). The W7
state budget ``kv_mb + linear_reserved_mb <= 1640 MiB`` is recorded as
``over_budget``/``over_by_mb``; it is never silently clamped.

Memory account
--------------
The trace/request rows add W7's state ledger without redefining the W4 names:

* ``kv_mb`` / ``kv_pool_mb`` -- the KV pool reservation (six full-attention layers only);
* ``workspace_mb`` -- structural zero, as in W4.5 (no gather workspace);
* ``linear_state_mb`` / ``linear_used_mb`` -- high-water active linear state (pool slots);
* ``kv_used_mb`` -- written KV of the resident set at its peak step (not the reservation);
* ``state_used_mb`` -- ``linear_used_mb + kv_used_mb`` at that same step (plan §6.3);
* ``linear_reserved_mb`` / ``kv_reserved_mb`` -- every allocated linear slot and the whole
  KV container, from the pool's real tensors (never the analytic 18.633 MiB);
* ``state_reserved_mb`` / ``state_slack_mb`` -- their sum and the difference to
  ``state_used_mb``;
* ``state_total_mb`` -- the capacity basis (``state_reserved_mb + workspace``) on the
  trace/step rows, and this request's ``state_mb`` on a request row.

The per-request rows keep every D5/D7 field (``token_ids``, ``ttft_ms``, ``tpot_ms``, ...)
and add ``linear_mb`` (this request's slot, from the pool's ``per_slot_bytes``),
``kv_written_mb`` (its own written KV bytes) and ``state_mb`` (their sum).
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from bench.engines.continuous import (
    ContinuousEngine,
    TraceResult,
    _request_row,
    _trace_clock,
    _trace_row,
)
from bench.engines.fill import (
    FillScheduler,
    W4_TOTAL_BUDGET_MB,
    capacity_block,
)
from bench.scenario import aggregate
from serve.scheduler import Request, Scheduler

_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}
_ARMS = {"paged", "seat"}


class HybridEngine(ContinuousEngine):
    """Runner adapter for ``[engine].impl = "hybrid"`` (Qwen3.5-0.8B)."""

    impl = "hybrid"

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
        linear_slots: int = 1,
        kv_blocks: int = 1,
        block_size: int = 16,
        pool: str = "paged",
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
        if pool not in _ARMS:
            raise ValueError(f"pool must be one of {sorted(_ARMS)}, got {pool!r}")
        self.linear_slots = int(linear_slots)
        self.kv_blocks = int(kv_blocks)
        self.block_size = int(block_size)
        self.pool = str(pool)
        #: Whole KV container reservation (block pool or seat slab), set by ``run_trace``.
        self.kv_pool_mb: float | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        # ``ContinuousEngine`` derives these three from its per-request delegate; the
        # hybrid arm has no delegate, so it owns the loaded values directly.
        self._eos_token_id: int | None = None
        self._weights_mb: float | None = None

    # ------------------------------------------------------------------ interface
    @property
    def vocab_size(self) -> int:
        return int(self._vocab)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def weights_mb(self) -> float | None:
        return self._weights_mb

    @property
    def decode_impl(self) -> str:
        """Decode kernel this arm runs: Triton block-table (paged) or hand_baddbmm (seat)."""
        return "paged-triton" if self.pool == "paged" else "hand_baddbmm"

    def attn_backends(self) -> dict[str, str]:
        """Backends recorded in the manifest; the engine owns the real prefill arm."""
        engine = self._engine
        report = getattr(engine, "attn_backends", None)
        if callable(report):
            return dict(report())
        return {"prefill": "sdpa", "decode": self.decode_impl}

    def load(self) -> None:
        """Load the pinned Qwen3.5 snapshot through ``model.loading``.

        Same loader the W6/G0 tests use (``load_qwen35_model``): no ``from_pretrained``,
        no tokenizer training/chat template. The tokenizer is the snapshot's own, with
        ``local_files_only=True``, and ``encode`` uses ``add_special_tokens=False`` to
        match :class:`~bench.engines.minillm.ServeEngine` /
        :class:`~bench.engines.hf.HFEngine` (a different tokenization would silently
        measure different prompts).
        """
        import torch
        from transformers import AutoTokenizer

        from model.loading import cached_snapshot, load_qwen35_model
        from serve.hybrid_engine import Qwen35Engine

        if self.dtype_name not in _DTYPES:
            raise ValueError(
                f"unsupported dtype {self.dtype_name!r}; choose {sorted(_DTYPES)}"
            )
        torch_dtype = getattr(torch, _DTYPES[self.dtype_name])

        path = self.model_path or cached_snapshot(self.name, self.revision)
        if path is None:
            raise FileNotFoundError(
                f"{self.name}@{self.revision} is not in the HF cache; "
                "fetch it first (HF_ENDPOINT=https://hf-mirror.com) or pass model_path"
            )

        self.model = load_qwen35_model(path, dtype=torch_dtype, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._eos_token_id = self.tokenizer.eos_token_id
        self.engine = Qwen35Engine(
            self.model,
            eos_token_id=self._eos_token_id,
            max_prefill_tokens=self.max_prefill_tokens,
        )
        self._vocab = int(self.model.config.vocab_size)
        self._weights_mb = (
            sum(p.numel() * p.element_size() for p in self.model.parameters()) / 2**20
        )
        # The resolved device (``cuda:0``) is what the run must record, not the request
        # string ("cuda").
        self.device = str(self.engine.device)
        # ``self._engine`` is the scheduler-facing engine; ``self._delegate`` stays None
        # because there is no per-request adapter underneath (unlike the 0.6B arms).
        self._engine = self.engine

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("call load() before encode()")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int):
        raise NotImplementedError(
            "impl='hybrid' is a scheduler arm: it runs a whole trace through one "
            "scheduler (run_trace). Use impl='minillm' for per-request generate()."
        )

    def close(self) -> None:
        super().close()
        self.model = None
        self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------------- trace
    def run_trace(self, trace, *, repeat: int = 0) -> TraceResult:
        """Run every arrival of ``trace`` through one scheduler and aggregate rows.

        ``[exp].type = "fill"`` swaps the step loop for :class:`FillScheduler` (packed
        prefill + cross-step residency, no decode); the admission unit, pool sizing and
        timestamps are the scenario path's. ``open_hybrid_pool`` is sized from this
        adapter's config, not from the trace: the linear slots are the ``linear_slots``
        budget knob and the block pool is ``kv_blocks`` (paged arm only). ``max_len`` is
        the trace's worst ``prompt + budget + 1``, exactly like the paged arm's position
        bound (P-34).
        """
        if self._engine is None:
            raise RuntimeError("call load() before run_trace()")
        engine = self._engine

        longest_prompt = max((len(a.prompt_ids) for a in trace.arrivals), default=0)
        longest_budget = max((a.budget for a in trace.arrivals), default=0)
        need = int(longest_prompt + longest_budget + 1)
        engine.open_hybrid_pool(
            max_batch=int(self.max_batch),
            max_len=need,
            linear_slots=int(self.linear_slots),
            kv_blocks=int(self.kv_blocks) if self.pool == "paged" else None,
            block_size=int(self.block_size),
            pool=self.pool,
        )
        self.max_len = need
        self.pool_kv_mb = (
            None if engine.pool_kv_mb is None else float(engine.pool_kv_mb)
        )
        self.kv_pool_mb = self.pool_kv_mb
        t0 = time.perf_counter()

        schedule = FillScheduler if self.fill else Scheduler
        scheduler = schedule(
            engine,
            max_batch=int(self.max_batch),
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
        requests = scheduler.run()

        stats = self._engine_stats()
        kv_pool = float(stats.get("kv_mb") or 0.0)
        # Both hybrid arms have no gather workspace: the engine reports a structural zero.
        workspace = float(stats.get("workspace_mb") or 0.0)
        step_rows = [
            dict(row, row_kind="step", repeat=repeat) for row in scheduler.step_rows
        ]
        state = self._state_accounting(stats, step_rows)

        t_arrival = min((r.t_arrival for r in requests), default=0.0)
        finishes = [r.t_finish for r in requests if r.t_finish is not None]
        t_finish = max(finishes, default=t_arrival)
        firsts = [r.t_first for r in requests if r.t_first is not None]
        t_first_token = min(firsts, default=t_arrival)
        out_total = sum(len(r.out) for r in requests)

        per_slot_mb = state["per_slot_state_mb"]
        per_token_bytes = self._per_token_bytes()
        request_rows = [
            _hybrid_request_row(
                r,
                prompt_ids_by_idx.get(r.id),
                kv_pool,
                self.max_batch,
                repeat,
                linear_mb=per_slot_mb,
                per_token_bytes=per_token_bytes,
                fill_only=bool(self.fill),
                admitted=r.t_admitted is not None,
            )
            for r in requests
        ]

        # W7 state budget (plan §5.4/§6.4): KV pool + every linear slot, against the same
        # 1640 MiB as W4. Over budget is recorded, never clamped.
        state_reserved_mb = float(state["state_reserved_mb"] or 0.0)
        state_total_mb = state_reserved_mb + workspace
        over_budget = bool(state_total_mb > W4_TOTAL_BUDGET_MB)
        over_by_mb = max(0.0, state_total_mb - W4_TOTAL_BUDGET_MB)

        # Instrumentation rows: every field the W4 arms write keeps its name; the W7
        # state ledger is appended when the engine's step snapshot did not already carry
        # it (the paged arm's ``paged_stats`` hook does; the seat arm has no paged mode).
        for row in step_rows:
            row.setdefault("workspace_mb", workspace)
            row.setdefault("kv_pool_mb", kv_pool)
            row.setdefault("pool", self.pool)
            row.setdefault("linear_slots", state["linear_slots"])
            row.setdefault("state_reserved_mb", state["state_reserved_mb"])
            row.setdefault("state_total_mb", state_total_mb)
            if self.fill:
                row.setdefault("fill_only", True)
            # The paged arm's engine attaches a per-step ``linear_state_mb`` snapshot, so
            # ``state_used_mb`` on those rows is a real measurement. The seat arm has no
            # paged mode and therefore no per-step snapshot; it carries the reservation
            # only rather than repeating the run's high-water mark as if it were one.
            if row.get("linear_state_mb") is not None:
                row.setdefault("state_used_mb", row["linear_state_mb"])

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
        trace_row.update(
            {
                "kv_mb": kv_pool,
                "kv_pool_mb": kv_pool,
                "workspace_mb": workspace,
                "kv_total_mb": kv_pool + workspace,
                "pool": self.pool,
                "linear_slots": state["linear_slots"],
                "linear_state_mb": state["linear_state_mb"],
                "linear_used_mb": state["linear_used_mb"],
                "linear_reserved_mb": state["linear_reserved_mb"],
                "kv_used_mb": state["kv_used_mb"],
                "kv_live_mb": state["kv_used_mb"],
                "kv_reserved_mb": state["kv_reserved_mb"],
                "state_used_mb": state["state_used_mb"],
                "state_used_end_mb": state["state_used_end_mb"],
                "state_reserved_mb": state["state_reserved_mb"],
                "state_slack_mb": state["state_slack_mb"],
                "state_total_mb": state_total_mb,
                "peak_active": state["peak_active"],
                "leak": (None if "leak" not in stats else bool(stats["leak"])),
                "kv_blocks": int(self.kv_blocks) if self.pool == "paged" else None,
                "block_size": int(self.block_size) if self.pool == "paged" else None,
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
        capacity = capacity_block(
            peak_concurrency=peak,
            kv_mb=kv_pool,
            workspace_mb=workspace,
            kv_total_mb=kv_pool + workspace,
            block_size=int(self.block_size) if self.pool == "paged" else None,
            kv_blocks=int(self.kv_blocks) if self.pool == "paged" else None,
            max_batch=self.max_batch,
            frag_internal=_last_stat(step_rows, "frag_internal"),
            reservation_slack=_last_stat(step_rows, "reservation_slack"),
            seat_unused_mb=None,
            workload_sha256=trace.sha256,
        )
        # W7 capacity is KV + linear state (plan §2.3/§6.4), not KV alone. Both the W4
        # total and the W7 total are recorded so no reader has to guess the basis.
        capacity.update(
            {
                "pool": self.pool,
                "linear_slots": state["linear_slots"],
                "linear_state_mb": state["linear_state_mb"],
                "linear_used_mb": state["linear_used_mb"],
                "linear_reserved_mb": state["linear_reserved_mb"],
                "kv_used_mb": state["kv_used_mb"],
                "kv_live_mb": state["kv_used_mb"],
                "kv_reserved_mb": state["kv_reserved_mb"],
                "state_used_mb": state["state_used_mb"],
                "state_used_end_mb": state["state_used_end_mb"],
                "state_reserved_mb": state["state_reserved_mb"],
                "state_slack_mb": state["state_slack_mb"],
                "state_total_mb": state_total_mb,
                "per_slot_state_mb": state["per_slot_state_mb"],
                "peak_active": state["peak_active"],
                "leak": (None if "leak" not in stats else bool(stats["leak"])),
                # W7 budget accounting (E1/E2): the same 1640 MiB state budget W4 used,
                # now KV + linear slots. A run above it is reported, never re-run at a
                # smaller budget (plan §6.4: over-budget is invalid/record).
                "budget_mb": float(W4_TOTAL_BUDGET_MB),
                "over_budget": over_budget,
                "over_by_mb": over_by_mb,
                "fill_only": bool(self.fill),
            }
        )
        # W7 additive: one context means the fill probe has an unambiguous context for
        # the capacity/P1 evaluators. A mixed trace omits the key rather than asserting
        # a single number (the evaluator then has to refuse the pair).
        contexts = sorted({len(a.prompt_ids) for a in trace.arrivals})
        if len(contexts) == 1:
            capacity["context_tokens"] = contexts[0]
        summary["capacity"] = capacity
        return TraceResult(
            request_rows=request_rows,
            step_rows=step_rows,
            trace_row=trace_row,
            summary=summary,
            repeat=repeat,
            impl=self.impl,
        )

    # ------------------------------------------------------------------ internals
    def _engine_stats(self) -> dict[str, Any]:
        if self._engine is None:
            return {}
        stats = getattr(self._engine, "paged_stats", None)
        if not callable(stats):
            return {}
        return dict(stats() or {})

    def _state_accounting(
        self, stats: dict[str, Any], step_rows: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        """State ledger: used (linear + written KV), reserved, slack and per-slot bytes.

        ``paged_stats()['linear_state_mb']`` is the *instantaneous* active state, so the
        run's used high-water is read from the step rows (the paged arm attaches the
        engine's snapshot to every step). ``reserved``/``slack``/``per_slot_bytes`` come
        from the pool's own tensors; the analytic 18.633 MiB is never an input here.
        """
        state: dict[str, Any] = {
            "linear_slots": int(stats.get("linear_slots") or self.linear_slots),
            "linear_used_mb": _float_or_none(stats.get("linear_state_mb")) or 0.0,
            "linear_state_mb": _float_or_none(stats.get("linear_state_mb")) or 0.0,
            "kv_used_mb": _float_or_none(stats.get("kv_live_mb")) or 0.0,
            "state_used_mb": 0.0,
            "state_used_end_mb": 0.0,
            "linear_reserved_mb": None,
            "kv_reserved_mb": _float_or_none(stats.get("kv_mb")) or 0.0,
            "state_reserved_mb": None,
            "state_slack_mb": None,
            "per_slot_state_mb": None,
            "peak_active": int(stats.get("peak_active") or 0),
        }
        try:
            pool_stats = self._engine.state_pool.stats()
        except (AttributeError, RuntimeError):
            pool_stats = None
        if pool_stats is not None:
            state.update(
                {
                    "linear_slots": int(pool_stats.slot_count),
                    "linear_reserved_mb": float(pool_stats.reserved_mb),
                    "per_slot_state_mb": pool_stats.per_slot_bytes / 2**20,
                    "state_used_end_mb": float(pool_stats.used_mb),
                    "peak_active": int(pool_stats.peak_active),
                }
            )
        # ``used`` is linear state + **written** KV (plan §6.3), both measured at the
        # same step. The step snapshot is the only common point: after the trace every
        # request is released, so the end-of-run stats would report zero. A row that
        # carries only one of the two is not used to invent the other; the fallbacks
        # below cover a seat arm whose step rows lack the paged snapshot.
        step_used: list[float] = []
        for row in step_rows:
            linear = _float_or_none(row.get("linear_state_mb"))
            kv = _float_or_none(row.get("kv_live_mb"))
            if linear is not None and kv is not None:
                step_used.append(linear + kv)
        linear_peak = max(
            (
                float(row["linear_state_mb"])
                for row in step_rows
                if _float_or_none(row.get("linear_state_mb")) is not None
            ),
            default=0.0,
        )
        kv_peak = max(
            (
                float(row["kv_live_mb"])
                for row in step_rows
                if _float_or_none(row.get("kv_live_mb")) is not None
            ),
            default=0.0,
        )
        if pool_stats is not None:
            linear_peak = max(
                linear_peak, pool_stats.peak_active * pool_stats.per_slot_bytes / 2**20
            )
        if not step_used:
            step_used = [linear_peak + kv_peak]
        used_peak = max(step_used, default=0.0)
        state["linear_used_mb"] = linear_peak
        state["linear_state_mb"] = linear_peak
        state["kv_used_mb"] = kv_peak
        state["state_used_mb"] = used_peak
        linear_reserved = state["linear_reserved_mb"]
        if linear_reserved is not None:
            state["state_reserved_mb"] = float(linear_reserved) + float(
                state["kv_reserved_mb"]
            )
        # Slack is a reservation property at the run's high-water mark, not at the end:
        # after the trace all slots are free, so the pool's own ``slack_mb`` would read as
        # "the whole pool is unused" and hide the peak occupancy.
        if state["state_reserved_mb"] is not None:
            state["state_slack_mb"] = max(
                0.0, float(state["state_reserved_mb"]) - used_peak
            )
        return state

    def _per_token_bytes(self) -> int:
        """Written KV bytes per token: six full-attention layers, K and V, real dtype."""
        engine = self._engine
        if engine is None:
            return 0
        return (
            2
            * int(getattr(engine, "n_layer", 0) or 0)
            * int(getattr(engine, "n_kv_head", 0) or 0)
            * int(getattr(engine, "head_dim", 0) or 0)
            * int(getattr(getattr(engine, "dtype", None), "itemsize", 2))
        )


# --------------------------------------------------------------------------- rows
def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _last_stat(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _hybrid_request_row(
    request: Request,
    prompt_id: str | None,
    kv_pool_mb: float,
    max_batch: int,
    repeat: int,
    *,
    linear_mb: float | None,
    per_token_bytes: int,
    fill_only: bool = False,
    admitted: bool = True,
) -> dict[str, Any]:
    """``_request_row`` plus the W7 per-request state account.

    ``linear_mb`` is one pool slot's state (constant per request under the pool's dtype);
    ``kv_written_mb`` is this request's **written** KV (prompt + generated tokens), not
    the pool's reservation. ``state_mb`` is their sum -- the per-sequence state the W7
    comparison reports -- and ``state_total_mb`` is the same number under the capacity
    block's name. A request a fill probe never admitted (still ``queued`` when capacity
    stopped) owns no slot and wrote no KV, so all three are ``None`` rather than a
    fictional reservation.
    """
    row = _request_row(request, prompt_id, kv_pool_mb, max_batch, repeat)
    written_tokens = len(request.prompt_ids) + len(request.out)
    owns_state = bool(admitted) and request.state != "aborted"
    kv_written_mb = (
        written_tokens * per_token_bytes / 2**20
        if per_token_bytes and owns_state
        else None
    )
    slot_mb = linear_mb if owns_state else None
    row["admitted"] = owns_state
    row["linear_mb"] = slot_mb
    row["kv_written_mb"] = kv_written_mb
    row["state_mb"] = (
        slot_mb + kv_written_mb
        if slot_mb is not None and kv_written_mb is not None
        else None
    )
    row["state_total_mb"] = row["state_mb"]
    row["fill_only"] = bool(fill_only)
    return row
