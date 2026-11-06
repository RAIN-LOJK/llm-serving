"""Paged-KV block allocator and fragmentation accounting for W4 (D6).

Two small, dependency-free pieces live here:

* :class:`BlockManager` -- the physical block pool. A caller reserves a request's whole
  ``ceil((prompt + budget) / block_size)`` up front (D6: no eviction, no mid-request
  allocation failure), and returns every block at once when the request leaves. Block
  ids come out smallest-index-first so the capacity simulation and the unit tests can
  assert exact ids.
* :class:`FragStats` -- time-weighted bookkeeping for the two memory metrics in D6:
  ``internal_frag`` (only the unfilled tail of the last block, C2) and
  ``reservation_slack`` (reserved upper bound vs. actual tokens, reported but not judged).

The module is pure Python and imports the standard library only; ``serve/scheduler.py``
uses the same convention so the unit tests can run without torch. Blocks are opaque
integers here: mapping them to per-layer tensors is ``PagedPool``'s job (``model/paged.py``).

Accounting is maintained by the code, not reconstructed by hand:
``reserved`` counts blocks currently held by callers, ``peak`` is the high-water mark of
that value, and ``alloc_calls`` / ``free_calls`` count calls that moved at least one
block (``reserve(0)`` and ``free([])`` are no-ops and are not counted).
"""

from __future__ import annotations

import heapq
from typing import Sequence

__all__ = ["BlockManager", "FragStats"]


class BlockManager:
    """A deterministic free list of ``n_blocks`` physical blocks of ``block_size`` tokens.

    A block is either free or allocated to exactly one reservation; there is no shared
    block (the W4 zero block is read-only and lives outside this pool).
    """

    def __init__(self, n_blocks: int, block_size: int) -> None:
        if not isinstance(n_blocks, int) or isinstance(n_blocks, bool) or n_blocks < 1:
            raise ValueError(f"n_blocks must be a positive int, got {n_blocks!r}")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError(f"block_size must be a positive int, got {block_size!r}")
        self.n_blocks = n_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(n_blocks))
        heapq.heapify(self._free)
        self._allocated: set[int] = set()
        self._peak = 0
        self._alloc_calls = 0
        self._free_calls = 0

    # ------------------------------------------------------------ allocation

    def reserve(self, n: int) -> list[int] | None:
        """Reserve ``n`` distinct block ids, or return ``None`` without consuming any.

        All-or-nothing: a reservation either holds the full ``n`` blocks or leaves the
        pool exactly as it was. The capacity check happens before the state change, so
        a caller that fails admission can retry later as though the call never
        happened. ``reserve(0)`` returns ``[]``.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"n must be a non-negative int, got {n!r}")
        if n > len(self._free):
            return None
        if n == 0:
            return []
        ids = [heapq.heappop(self._free) for _ in range(n)]
        self._allocated.update(ids)
        self._alloc_calls += 1
        if len(self._allocated) > self._peak:
            self._peak = len(self._allocated)
        return ids

    def free(self, ids: Sequence[int]) -> None:
        """Return every id to the free list. Each id may be freed exactly once.

        The whole sequence is validated before anything changes, so a bad call
        (duplicate, unknown, already-free, out-of-range id) raises and leaves the
        ledger untouched. ``free([])`` is a no-op.
        """
        seen: set[int] = set()
        for block_id in ids:
            if not isinstance(block_id, int) or isinstance(block_id, bool):
                raise ValueError(f"block ids must be ints, got {block_id!r}")
            if block_id in seen:
                raise ValueError(f"duplicate block id {block_id} in one free call")
            seen.add(block_id)
            if block_id not in self._allocated:
                raise ValueError(f"block {block_id} is not allocated")
        if not seen:
            return
        self._allocated.difference_update(seen)
        self._free.extend(seen)
        heapq.heapify(self._free)
        self._free_calls += 1

    # ------------------------------------------------------------ accounting

    @property
    def in_use(self) -> int:
        """Blocks currently held by callers (same value as ``stats()["in_use"]``)."""
        return len(self._allocated)

    @property
    def peak(self) -> int:
        """High-water mark of :attr:`in_use` since construction; never decreases."""
        return self._peak

    def stats(self) -> dict:
        """Code-maintained capacity ledger (all counts in blocks)."""
        in_use = len(self._allocated)
        return {
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "in_use": in_use,
            "free": len(self._free),
            "reserved": in_use,
            "peak": self._peak,
            "alloc_calls": self._alloc_calls,
            "free_calls": self._free_calls,
        }

    def blocks_for(self, prompt_len: int, budget: int) -> int:
        """Blocks a request must hold for ``ceil((prompt_len + budget) / block_size)``.

        The reservation is an upper bound: the request gets exactly this many blocks at
        admission so decode can never fail, even if it stops early. The floor of one
        block keeps an empty request representable.
        """
        if not isinstance(prompt_len, int) or isinstance(prompt_len, bool) or prompt_len < 0:
            raise ValueError(f"prompt_len must be a non-negative int, got {prompt_len!r}")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError(f"budget must be a non-negative int, got {budget!r}")
        n = (prompt_len + budget + self.block_size - 1) // self.block_size
        return max(1, n)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BlockManager(n_blocks={self.n_blocks}, block_size={self.block_size}, "
            f"in_use={len(self._allocated)}, peak={self._peak})"
        )


class FragStats:
    """Time-weighted internal-fragmentation and reservation-slack accounting.

    ``observe`` takes one (step-level) sample; ``dt_s`` is the time the sample
    represents, so a slow step weighs more than a fast one. Samples with
    ``dt_s <= 0`` are ignored (they carry no time). The definitions are frozen in D6:

    * ``internal_frag = 1 - sum(written_tokens * dt) / sum(allocated_blocks * block_size * dt)``
      -- only the unfilled tail of the last block counts, never unused reservations;
    * ``reservation_slack = 1 - actual_tokens / reserved_tokens`` -- reported for
      context, not part of C2.
    """

    def __init__(self, block_size: int) -> None:
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError(f"block_size must be a positive int, got {block_size!r}")
        self.block_size = block_size
        self._written_dt = 0.0
        self._capacity_dt = 0.0  # allocated_blocks * block_size * dt
        self._seat_live_dt = 0.0
        self._seat_width_dt = 0.0  # row_width * dt

    def observe(self, dt_s: float, written_tokens: int, allocated_blocks: int) -> None:
        """Accumulate one sample; ``dt_s <= 0`` is ignored."""
        if not isinstance(written_tokens, int) or isinstance(written_tokens, bool) or written_tokens < 0:
            raise ValueError(f"written_tokens must be a non-negative int, got {written_tokens!r}")
        if (
            not isinstance(allocated_blocks, int)
            or isinstance(allocated_blocks, bool)
            or allocated_blocks < 0
        ):
            raise ValueError(
                f"allocated_blocks must be a non-negative int, got {allocated_blocks!r}"
            )
        if dt_s <= 0:
            return
        if allocated_blocks == 0:
            return  # no capacity to charge the tokens against; the sample carries no ratio
        self._written_dt += written_tokens * dt_s
        self._capacity_dt += allocated_blocks * self.block_size * dt_s

    def internal_frag(self) -> float | None:
        """Time-weighted fraction of allocated block capacity left unwritten.

        ``None`` when no positive-duration sample with capacity was observed. The
        value can be negative only if a caller reports more written tokens than the
        blocks it allocated, which is a bug in the caller.
        """
        if self._capacity_dt <= 0:
            return None
        return 1.0 - self._written_dt / self._capacity_dt

    def reservation_slack(self, reserved_tokens: int, actual_tokens: int) -> float | None:
        """``1 - actual / reserved`` for the admission upper bound; ``None`` if no reservation.

        Kept separate from :meth:`internal_frag` (D6): a request that stops early leaves
        reserved blocks untouched, and that waste is reported, not judged. No clipping:
        a negative value means the actual sequence outgrew its reservation, which
        should be visible.
        """
        if reserved_tokens <= 0:
            return None
        return 1.0 - actual_tokens / reserved_tokens

    def seat_row_slack(self, live_tokens: int, row_width: int, dt_s: float) -> float | None:
        """Time-weighted seat-row waste, for the same table as :meth:`internal_frag`.

        D6: ``seat_row_slack = 1 - live / row_width`` for a ``KvSlab`` row, where
        ``row_width`` is the bucket of the trace's worst prompt + budget. The seat arm
        reserves that width for every row, so this is the analogue of "allocated
        capacity never used" and is reported next to the paged metrics, not judged.

        Each call accumulates one sample weighted by ``dt_s`` and returns the running
        time-weighted value over every sample observed so far (``None`` until a sample
        with ``dt_s > 0`` and ``row_width > 0`` arrives). Samples with ``dt_s <= 0``
        carry no time and are ignored. No clipping: a negative value means the row
        outgrew its width.
        """
        if not isinstance(live_tokens, int) or isinstance(live_tokens, bool) or live_tokens < 0:
            raise ValueError(f"live_tokens must be a non-negative int, got {live_tokens!r}")
        if not isinstance(row_width, int) or isinstance(row_width, bool) or row_width < 0:
            raise ValueError(f"row_width must be a non-negative int, got {row_width!r}")
        if dt_s <= 0:
            return self._seat_slack_value()
        if row_width == 0:
            return self._seat_slack_value()  # a zero-width row has no ratio to weight
        self._seat_live_dt += live_tokens * dt_s
        self._seat_width_dt += row_width * dt_s
        return self._seat_slack_value()

    def _seat_slack_value(self) -> float | None:
        if self._seat_width_dt <= 0:
            return None
        return 1.0 - self._seat_live_dt / self._seat_width_dt
