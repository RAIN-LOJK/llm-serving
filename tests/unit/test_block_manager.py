"""``serve/block_manager.py``: the W4 paged-KV allocator and its accounting.

The allocator is the piece the capacity simulation imports directly (``docs/w4-plan.md``
§4.1), so these tests pin the properties the simulator relies on: deterministic block
ids, an atomic reservation (a failed ``reserve`` must not consume anything), a free
list that recycles, and counters maintained by the code. ``FragStats`` is checked
against hand-computed numbers; the metric definition is frozen in D6 (§2).
"""

from __future__ import annotations

import pytest

from serve.block_manager import BlockManager, FragStats


def _bm(n_blocks=8, block_size=16):
    return BlockManager(n_blocks, block_size)


# ---------------------------------------------------------------- construction

def test_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        BlockManager(0, 16)
    with pytest.raises(ValueError):
        BlockManager(8, 0)
    with pytest.raises(ValueError):
        BlockManager(-1, 16)
    with pytest.raises(ValueError):
        BlockManager(8, -16)


def test_initial_stats():
    stats = _bm(8, 16).stats()
    assert stats["n_blocks"] == 8
    assert stats["block_size"] == 16
    assert stats["in_use"] == 0
    assert stats["free"] == 8
    assert stats["reserved"] == 0
    assert stats["peak"] == 0
    assert stats["alloc_calls"] == 0
    assert stats["free_calls"] == 0


# ---------------------------------------------------------------- allocation

def test_reserve_is_smallest_index_first():
    bm = _bm(8)
    assert bm.reserve(3) == [0, 1, 2]
    assert bm.reserve(2) == [3, 4]


def test_reserve_zero_returns_empty_and_touches_nothing():
    bm = _bm(4)
    assert bm.reserve(0) == []
    assert bm.stats()["in_use"] == 0
    assert bm.stats()["free"] == 4
    assert bm.reserve(1) == [0]


def test_reserve_rejects_negative_counts():
    bm = _bm(4)
    with pytest.raises(ValueError):
        bm.reserve(-1)
    assert bm.stats()["in_use"] == 0


def test_reserve_is_atomic_when_capacity_is_short():
    bm = _bm(4)
    assert bm.reserve(3) == [0, 1, 2]
    before = bm.stats()

    assert bm.reserve(2) is None

    after = bm.stats()
    assert after == before, "a failed reserve must not consume or count anything"
    # the next successful reservation still sees the smallest free block
    assert bm.reserve(1) == [3]


def test_reservation_exhausts_and_does_not_overlap():
    bm = _bm(5)
    ids = bm.reserve(5)
    assert sorted(ids) == [0, 1, 2, 3, 4]
    assert bm.reserve(1) is None
    assert bm.stats()["free"] == 0


# ---------------------------------------------------------------- free / reuse

def test_free_returns_blocks_to_the_smallest_index_pool():
    bm = _bm(4)
    bm.reserve(4)
    bm.free([2, 0])  # free order must not matter
    assert bm.stats()["in_use"] == 2
    assert bm.reserve(2) == [0, 2]


def test_free_empty_sequence_is_a_noop():
    bm = _bm(4)
    bm.reserve(2)
    bm.free([])
    assert bm.stats()["in_use"] == 2
    assert bm.reserve(2) == [2, 3]


def test_double_free_raises_and_keeps_the_ledger_consistent():
    bm = _bm(4)
    bm.reserve(2)
    bm.free([0])
    with pytest.raises(ValueError, match="not allocated"):
        bm.free([0])
    assert bm.stats()["in_use"] == 1
    assert bm.stats()["free"] == 3


def test_free_of_an_unknown_or_out_of_range_block_raises():
    bm = _bm(4)
    with pytest.raises(ValueError, match="not allocated"):
        bm.free([3])
    bm.reserve(1)
    with pytest.raises(ValueError):
        bm.free([4])
    with pytest.raises(ValueError):
        bm.free([-1])
    assert bm.stats()["in_use"] == 1


def test_duplicate_id_in_one_call_raises_before_anything_is_freed():
    bm = _bm(4)
    bm.reserve(2)
    with pytest.raises(ValueError):
        bm.free([0, 0])
    # validation is upfront: block 0 is still allocated
    assert bm.stats()["in_use"] == 2
    bm.free([0, 1])


def test_free_accepts_any_sequence():
    bm = _bm(4)
    ids = bm.reserve(2)
    bm.free(tuple(ids))
    assert bm.stats()["in_use"] == 0


# ---------------------------------------------------------------- accounting

def test_capacity_counters_track_allocation_and_free():
    bm = _bm(8)
    bm.reserve(3)
    bm.reserve(2)
    stats = bm.stats()
    assert stats["in_use"] == 5
    assert stats["reserved"] == stats["in_use"]
    assert stats["free"] == 3
    assert stats["peak"] == 5
    assert stats["alloc_calls"] == 2

    bm.free([0, 1, 2])
    stats = bm.stats()
    assert stats["in_use"] == 2
    assert stats["free"] == 6
    assert stats["reserved"] == 2
    assert stats["peak"] == 5, "peak is a high-water mark and never decreases"
    assert stats["free_calls"] == 1

    bm.free([])
    assert bm.stats()["free_calls"] == 1, "an empty free is a no-op and is not counted"


def test_failed_reserve_is_not_counted_as_an_alloc_call():
    bm = _bm(2)
    assert bm.reserve(3) is None
    assert bm.stats()["alloc_calls"] == 0


def test_zero_block_reservations_are_not_counted():
    bm = _bm(2)
    assert bm.reserve(0) == []
    assert bm.stats()["alloc_calls"] == 0


def test_in_use_and_peak_properties_match_stats():
    bm = _bm(6)
    assert bm.in_use == 0 and bm.peak == 0
    bm.reserve(3)
    assert bm.in_use == 3 == bm.stats()["in_use"]
    assert bm.peak == 3
    bm.free([0])
    assert bm.in_use == 2 == bm.stats()["in_use"]
    assert bm.peak == 3, "peak is a high-water mark"
    assert bm.reserve(5) is None  # short -> atomic, nothing changes
    assert bm.in_use == 2 and bm.peak == 3


def test_peak_survives_a_full_cycle():
    bm = _bm(6)
    bm.reserve(4)
    bm.reserve(2)
    assert bm.stats()["in_use"] == 6
    bm.free([0, 1, 2, 3, 4, 5])
    stats = bm.stats()
    assert stats["peak"] == 6
    assert stats["in_use"] == 0
    assert stats["free"] == 6


# ---------------------------------------------------------------- blocks_for

@pytest.mark.parametrize(
    "prompt_len, budget, expected",
    [
        (0, 0, 1),        # reserved capacity is at least one block
        (1, 0, 1),
        (15, 0, 1),
        (16, 0, 1),       # exactly one full block
        (17, 0, 2),
        (16, 16, 2),
        (17, 15, 2),      # 32 tokens -> 2
        (17, 16, 3),      # 33 tokens -> 3
        (987, 128, 70),   # 1115 / 16 = 69.7 -> 70 (E2 prompt lengths)
        (1000, 24, 64),   # 1024 / 16 = 64 exactly
    ],
)
def test_blocks_for_rounds_up_and_has_a_floor_of_one(prompt_len, budget, expected):
    assert _bm(1024, 16).blocks_for(prompt_len, budget) == expected


def test_blocks_for_uses_the_configured_block_size():
    assert _bm(1024, 32).blocks_for(16, 16) == 1
    assert _bm(1024, 32).blocks_for(16, 17) == 2


def test_blocks_for_rejects_negative_lengths():
    bm = _bm(8)
    with pytest.raises(ValueError):
        bm.blocks_for(-1, 4)
    with pytest.raises(ValueError):
        bm.blocks_for(4, -1)


def test_blocks_for_matches_what_reserve_needs():
    bm = _bm(16, 16)
    need = bm.blocks_for(100, 28)  # ceil(128 / 16) == 8
    ids = bm.reserve(need)
    assert len(ids) == 8
    assert bm.reserve(1) is not None


# ---------------------------------------------------------------- FragStats

def test_internal_frag_is_time_weighted():
    """Two samples with dt 1 s / 3 s: 1 - (8*1 + 16*3) / (16*1 + 16*3) == 0.125.

    A plain average of the per-sample frags would say (0.5 + 0.0) / 2 == 0.25; the
    weighted number is the D6 definition.
    """
    frag = FragStats(block_size=16)
    frag.observe(1.0, written_tokens=8, allocated_blocks=1)
    frag.observe(3.0, written_tokens=16, allocated_blocks=1)
    assert frag.internal_frag() == pytest.approx(0.125)


def test_internal_frag_hand_example_with_multiple_blocks():
    """1 - (20*2 + 32*2) / (32*2 + 32*2) == 1 - 104/128 == 0.1875."""
    frag = FragStats(block_size=16)
    frag.observe(2.0, written_tokens=20, allocated_blocks=2)
    frag.observe(2.0, written_tokens=32, allocated_blocks=2)
    assert frag.internal_frag() == pytest.approx(0.1875)


def test_internal_frag_is_zero_when_every_block_is_full():
    frag = FragStats(block_size=16)
    frag.observe(2.0, written_tokens=32, allocated_blocks=2)
    assert frag.internal_frag() == pytest.approx(0.0)


def test_internal_frag_counts_only_the_tail_of_a_multi_block_request():
    """Written tokens are compared with whole blocks, so only the tail is slack."""
    frag = FragStats(block_size=16)
    frag.observe(1.0, written_tokens=20, allocated_blocks=2)
    assert frag.internal_frag() == pytest.approx(1 - 20 / 32)


def test_internal_frag_ignores_non_positive_dt():
    frag = FragStats(block_size=16)
    frag.observe(0.0, written_tokens=100, allocated_blocks=4)
    frag.observe(-1.0, written_tokens=100, allocated_blocks=4)
    assert frag.internal_frag() is None


def test_internal_frag_is_none_without_samples():
    assert FragStats(block_size=16).internal_frag() is None


def test_internal_frag_is_none_when_no_capacity_was_observed():
    frag = FragStats(block_size=16)
    frag.observe(1.0, written_tokens=0, allocated_blocks=0)
    assert frag.internal_frag() is None


def test_observe_rejects_negative_counts():
    frag = FragStats(block_size=16)
    with pytest.raises(ValueError):
        frag.observe(1.0, written_tokens=-1, allocated_blocks=1)
    with pytest.raises(ValueError):
        frag.observe(1.0, written_tokens=1, allocated_blocks=-1)


def test_fragstats_rejects_a_degenerate_block_size():
    with pytest.raises(ValueError):
        FragStats(0)


def test_reservation_slack_is_one_minus_actual_over_reserved():
    frag = FragStats(block_size=16)
    assert frag.reservation_slack(reserved_tokens=64, actual_tokens=48) == pytest.approx(0.25)
    assert frag.reservation_slack(reserved_tokens=64, actual_tokens=64) == pytest.approx(0.0)
    assert frag.reservation_slack(reserved_tokens=64, actual_tokens=0) == pytest.approx(1.0)


def test_reservation_slack_is_none_without_capacity():
    frag = FragStats(block_size=16)
    assert frag.reservation_slack(reserved_tokens=0, actual_tokens=0) is None


def test_reservation_slack_reports_overflow_instead_of_clipping():
    frag = FragStats(block_size=16)
    assert frag.reservation_slack(reserved_tokens=64, actual_tokens=80) == pytest.approx(-0.25)


# ---------------------------------------------------------------- seat row slack

def test_seat_row_slack_is_time_weighted():
    """1 - (1664*1 + 832*3) / (1664*1 + 1664*3) == 448 / 6656 == 0.067307..."""
    frag = FragStats(block_size=16)
    assert frag.seat_row_slack(live_tokens=1664, row_width=1664, dt_s=1.0) == pytest.approx(0.0)
    got = frag.seat_row_slack(live_tokens=832, row_width=1664, dt_s=3.0)
    assert got == pytest.approx(1 - (1664 * 1 + 832 * 3) / (1664 * 4))


def test_seat_row_slack_with_a_single_sample():
    frag = FragStats(block_size=16)
    assert frag.seat_row_slack(live_tokens=512, row_width=2048, dt_s=2.0) == pytest.approx(0.75)


def test_seat_row_slack_ignores_non_positive_dt_and_returns_none_until_a_sample():
    frag = FragStats(block_size=16)
    assert frag.seat_row_slack(live_tokens=10, row_width=100, dt_s=0.0) is None
    assert frag.seat_row_slack(live_tokens=10, row_width=100, dt_s=-1.0) is None
    # a zero-width seat row has no ratio; the sample is ignored rather than poisoning
    # the running sum
    assert frag.seat_row_slack(live_tokens=100, row_width=0, dt_s=1.0) is None
    assert frag.seat_row_slack(live_tokens=50, row_width=100, dt_s=1.0) == pytest.approx(0.5)


def test_seat_row_slack_validates_its_inputs():
    frag = FragStats(block_size=16)
    with pytest.raises(ValueError):
        frag.seat_row_slack(live_tokens=-1, row_width=10, dt_s=1.0)
    with pytest.raises(ValueError):
        frag.seat_row_slack(live_tokens=1, row_width=-10, dt_s=1.0)
