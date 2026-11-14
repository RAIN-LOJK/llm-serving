"""W7 Step 1: ``LinearStatePool``, request handles, and pooled ``Qwen35State`` storage.

Three things this file pins:

* **Lifecycle** -- allocation fills ``[0, capacity)``, exhaustion is atomic, releasing a
  middle slot swaps the last active row into the hole (``copy_``, not a rebind) and
  updates the moved handle's ``slot``, double release is idempotent, an unknown id
  raises, ``close`` is repeatable and leaves no leak. Releasing a merely *reserved*
  handle is the §3.3 rollback path: the slot goes back, and a later request's fresh
  pooled state reads ``None`` (not the abandoned bytes) for every layer.
* **Accounting** -- ``used``/``reserved``/``slack`` come from the real tensors, not from
  the analytic ``18.633 MiB``; the 0.8B geometry test builds the config only (no
  weights) and checks the exact integers: conv 0.633 MiB, recurrent 18.0 MiB.
* **Pooled state storage** -- getters are pool views, setters ``copy_`` into the pool
  and never rebind it, validation is the base class's, ``cull`` matches
  ``Qwen35State`` for the surviving rows (with the explicit gather fallback for a
  non-contiguous order), and a state object held across a release raises instead of
  reading another request's rows.

The numeric prefill/decode comparison against the single-sequence ``Qwen35State`` lives
in ``tests/unit/test_qwen35_pool_state.py``; everything here is tensor-level and fast.
"""

from __future__ import annotations

import math

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import KVcache
from model.qwen35_model import Qwen35State
from serve.state_pool import (
    ACTIVE,
    FREE,
    RELEASED,
    RESERVED,
    LinearStatePool,
    PooledQwen35State,
    PoolExhausted,
    RequestHandle,
)

TINY = dict(
    vocab_size=64,
    max_position_embeddings=64,
    n_embd=32,
    n_head=4,
    n_kv_head=2,
    n_layer=24,
    n_inner=64,
    dropout=0.0,
    bias=False,
    norm_eps=1e-6,
    head_dim=8,
    rope_theta=1e4,
    qk_norm=True,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    tie_word_embeddings=True,
)

#: The 0.8B text-decoder facts (config only; no weights are loaded anywhere in this
#: file). 24 layers on LLLF x 6, 18 linear layers at conv_dim 6144.
QWEN35_08B = dict(
    vocab_size=248320,
    max_position_embeddings=40960,
    n_embd=1024,
    n_head=8,
    n_kv_head=2,
    n_layer=24,
    n_inner=3584,
    norm_eps=1e-6,
    bias=False,
    head_dim=256,
    rope_theta=10_000_000.0,
    qk_norm=True,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
    linear_num_key_heads=16,
    linear_num_value_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    tie_word_embeddings=True,
)

FULL_POSITIONS = [3, 7, 11, 15, 19, 23]
# conv_dim = 2 * key_dim + value_dim = 2*(2*4) + (2*4) = 24; k - 1 = 3.
CONV_DIM = 24
# One slot, tiny geometry: 18 * (24*3*4 + 2*4*4*4) = 7488 B.
TINY_SLOT_BYTES = 18 * (CONV_DIM * 3 * 4 + 2 * 4 * 4 * 4)


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _linear_layers(config: MiniLLMConfig) -> list[int]:
    return [
        idx
        for idx, kind in enumerate(config.layer_types)
        if kind == "linear_attention"
    ]


def _pool(capacity: int = 2, *, dtype=torch.float32, config=None) -> LinearStatePool:
    return LinearStatePool(
        _config() if config is None else config,
        capacity,
        device="cpu",
        dtype=dtype,
    )


def _fill_by_row(pool: LinearStatePool, n_rows: int, *, offset: int = 0) -> None:
    """Write a distinct constant into every (layer, row) so swaps are visible."""
    for layer in pool.linear_layers:
        for row in range(n_rows):
            value = float(offset + layer * 10 + row)
            pool.conv[layer][row].fill_(value)
            pool.recurrent[layer][row].fill_(value + 1000.0)


# --------------------------------------------------------------- allocation


def test_allocation_fills_slots_in_order_and_exhaustion_is_atomic():
    pool = _pool(capacity=3)
    handles = [pool.alloc(f"r{index}") for index in range(3)]

    assert [handle.slot for handle in handles] == [0, 1, 2]
    assert [handle.phase for handle in handles] == [RESERVED] * 3
    assert [pool.slot_phase(slot) for slot in range(3)] == [RESERVED] * 3
    assert pool.active_count == 3
    assert pool.stats().active == 3 and pool.stats().free == 0

    with pytest.raises(PoolExhausted, match=r"full"):
        pool.alloc("overflow")

    # The failed allocation leaves the pool untouched: no phantom handle, no slot.
    assert pool.active_count == 3
    assert pool.stats().active == 3 and pool.stats().free == 0
    with pytest.raises(KeyError, match=r"unknown"):
        pool.handle("overflow")


def test_zero_capacity_pool_raises_exhaustion_but_has_stats():
    pool = _pool(capacity=0)
    stats = pool.stats()
    assert stats.slot_count == 0 and stats.active == 0 and stats.free == 0
    assert stats.used_bytes == 0 and stats.reserved_bytes == 0
    with pytest.raises(PoolExhausted, match=r"0/0"):
        pool.alloc("r")
    pool.close()
    pool.assert_no_leaks()


def test_pool_requires_the_linear_attention_fields():
    # The default config is all full attention and carries no linear_* geometry; a pool
    # validated against guessed head shapes would accept tensors the layer cannot use.
    with pytest.raises(ValueError, match=r"linear_num_key_heads"):
        LinearStatePool(
            MiniLLMConfig(n_layer=2), 1, device="cpu", dtype=torch.float32
        )


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_capacity_and_dtype_are_validated(bad):
    with pytest.raises((TypeError, ValueError), match=r"capacity"):
        LinearStatePool(_config(), bad, device="cpu", dtype=torch.float32)
    with pytest.raises((TypeError, ValueError), match=r"dtype"):
        LinearStatePool(_config(), 1, device="cpu", dtype=bad)


# ------------------------------------------------ release / compaction / phases


def test_free_middle_swaps_last_active_row_into_the_hole():
    pool = _pool(capacity=3)
    first, middle, last = (pool.alloc(name) for name in ("a", "b", "c"))
    for handle in (first, middle, last):
        pool.activate(handle)
    _fill_by_row(pool, n_rows=3)

    pool.free(middle)

    # The hole is filled by the last active slot's rows, per layer, with copy_.
    for layer in pool.linear_layers:
        torch.testing.assert_close(
            pool.conv[layer][1],
            torch.full((CONV_DIM, 3), float(layer * 10 + 2)),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            pool.recurrent[layer][1],
            torch.full((2, 4, 4), float(layer * 10 + 2 + 1000)),
            atol=0,
            rtol=0,
        )
        # The untouched first row is still row 0.
        assert pool.conv[layer][0].eq(float(layer * 10 + 0)).all()
    assert last.slot == 1 and last.phase == ACTIVE  # the moved handle is updated
    assert first.slot == 0 and first.phase == ACTIVE
    assert middle.phase == RELEASED and middle.slot is None
    assert pool.active_count == 2
    assert [pool.slot_phase(slot) for slot in range(3)] == [ACTIVE, ACTIVE, FREE]
    assert pool.stats().used_bytes == 2 * TINY_SLOT_BYTES

    # The moved handle's updated slot is what a later release uses.
    pool.free(last)
    assert last.phase == RELEASED and last.slot is None
    assert pool.active_count == 1 and first.slot == 0
    pool.free(first)
    pool.assert_no_leaks()


def test_double_free_is_idempotent_and_unknown_id_raises():
    pool = _pool(capacity=2)
    keep, gone = pool.alloc("keep"), pool.alloc("gone")
    pool.activate(gone)
    pool.free(gone)
    after_first = pool.stats()

    pool.free(gone)  # by handle
    pool.free("gone")  # by id
    assert pool.stats() == after_first
    assert pool.active_count == 1
    assert gone.phase == RELEASED and gone.slot is None

    with pytest.raises(KeyError, match=r"unknown request id 'missing'"):
        pool.free("missing")
    assert pool.stats() == after_first
    assert keep.slot == 0


def test_handle_from_another_pool_is_rejected():
    pool = _pool(capacity=1)
    other = _pool(capacity=1)
    pool.alloc("a")
    foreign = other.alloc("a")
    with pytest.raises(ValueError, match=r"does not belong to this pool"):
        pool.free(foreign)
    assert pool.active_count == 1


def test_phase_transitions_and_activate_guards():
    pool = _pool(capacity=2)
    handle = pool.alloc("r")
    assert handle.phase == RESERVED and pool.slot_phase(handle.slot) == RESERVED

    assert pool.activate(handle) is handle
    assert handle.phase == ACTIVE and pool.slot_phase(handle.slot) == ACTIVE
    assert pool.activate("r") is handle  # idempotent while active

    pool.free(handle)
    assert handle.phase == RELEASED and handle.slot is None
    assert pool.slot_phase(0) == FREE
    with pytest.raises(RuntimeError, match=r"already released"):
        pool.activate(handle)
    with pytest.raises(KeyError, match=r"unknown"):
        pool.activate("missing")


def test_handle_carries_step2_placeholders_without_allocating_blocks():
    handle = _pool(capacity=1).alloc(7)
    assert handle.request_id == 7
    assert handle.prompt_len is None
    assert handle.length is None
    assert handle.block_table is None
    # The class shape Step 2 extends in place (block_table stays a plain attribute).
    handle.prompt_len = 12
    handle.length = 140
    handle.block_table = [0, 1]
    assert (handle.prompt_len, handle.length, handle.block_table) == (12, 140, [0, 1])
    assert set(RequestHandle.__dataclass_fields__) == {
        "request_id",
        "slot",
        "phase",
        "prompt_len",
        "length",
        "block_table",
    }


# ----------------------------------------------------- rollback / close / leaks


def test_releasing_a_reserved_slot_leaves_no_half_state():
    """§3.3 rollback: a request that fails before prefill completes returns cleanly."""
    pool = _pool(capacity=2)
    handle = pool.alloc("failed")
    state = pool.state(handle.slot)

    # Simulate a prefill that wrote every layer and then failed: no activate() call.
    written = float(handle.slot + 7)
    for layer in pool.linear_layers:
        state.set_conv_state(layer, torch.full((1, CONV_DIM, 3), written))
        state.set_recurrent_state(layer, torch.full((1, 2, 4, 4), written))

    pool.free(handle)  # while still reserved -- the rollback path
    assert handle.phase == RELEASED and handle.slot is None
    assert pool.active_count == 0
    assert pool.stats().used_bytes == 0
    assert pool.slot_phase(0) == FREE

    # A later request reuses the slot; its fresh state presents None for every layer, so
    # the abandoned bytes are unreachable, and the next prefill overwrites them.
    replacement = pool.alloc("replacement")
    assert replacement.slot == 0
    fresh = pool.state(replacement.slot)
    for layer in pool.linear_layers:
        assert fresh.conv_state(layer) is None
        assert fresh.recurrent_state(layer) is None

    fresh.set_conv_state(0, torch.full((1, CONV_DIM, 3), 3.0))
    torch.testing.assert_close(pool.conv[0][0], torch.full((CONV_DIM, 3), 3.0), atol=0, rtol=0)


def test_close_releases_everything_and_is_repeatable():
    pool = _pool(capacity=3)
    first, second = pool.alloc("a"), pool.alloc("b")
    pool.activate(first)
    for layer in pool.linear_layers:
        pool.state(first.slot).set_conv_state(layer, torch.ones(1, CONV_DIM, 3))

    pool.close()

    assert pool.closed
    assert pool.active_count == 0
    assert first.phase == RELEASED and first.slot is None
    assert second.phase == RELEASED and second.slot is None
    assert [pool.slot_phase(slot) for slot in range(3)] == [FREE] * 3
    stats = pool.stats()
    assert stats.used_bytes == 0
    assert stats.reserved_bytes == 3 * TINY_SLOT_BYTES
    assert stats.slack_bytes == stats.reserved_bytes
    pool.assert_no_leaks()

    pool.close()  # repeated close is safe
    pool.assert_no_leaks()
    assert pool.stats() == stats

    with pytest.raises(RuntimeError, match=r"closed"):
        pool.alloc("c")
    pool.free(first)  # idempotent even after close
    pool.assert_no_leaks()


def test_assert_no_leaks_detects_a_live_handle():
    pool = _pool(capacity=2)
    pool.alloc("leaky")
    with pytest.raises(AssertionError, match=r"leaked.*active_count=1.*unreleased"):
        pool.assert_no_leaks()
    pool.free("leaky")
    pool.assert_no_leaks()


# -------------------------------------------------------------- accounting


def test_stats_used_reserved_slack_come_from_the_real_tensors():
    pool = _pool(capacity=3)
    assert pool.per_slot_bytes() == TINY_SLOT_BYTES
    expected_per_slot = sum(
        tensor[0].numel() * tensor.element_size()
        for tensor in (*pool.conv.values(), *pool.recurrent.values())
    )
    assert expected_per_slot == TINY_SLOT_BYTES

    pool.alloc("a")
    stats = pool.stats()
    assert stats.used_bytes == TINY_SLOT_BYTES
    assert stats.reserved_bytes == 3 * TINY_SLOT_BYTES
    assert stats.slack_bytes == 2 * TINY_SLOT_BYTES
    assert stats.used_mb == TINY_SLOT_BYTES / (1024 * 1024)
    assert stats.reserved_mb == 3 * TINY_SLOT_BYTES / (1024 * 1024)
    assert stats.slack_mb == stats.reserved_mb - stats.used_mb
    assert stats.slot_count == 3 and stats.active == 1 and stats.free == 2
    assert pool.linear_state_bytes() == stats.used_bytes

    pool.alloc("b")
    pool.alloc("c")
    full = pool.stats()
    assert full.used_bytes == full.reserved_bytes and full.slack_bytes == 0
    assert full.peak_active == 3


def test_real_08b_geometry_bytes_without_loading_weights():
    """The W7 budget arithmetic from the config alone: no weights, no forward."""
    config = MiniLLMConfig(**QWEN35_08B)
    pool = _pool(capacity=1, dtype=torch.bfloat16, config=config)
    handle = pool.alloc("r")
    assert handle.slot == 0

    # conv: [capacity, 6144, k - 1 = 3] bf16; recurrent: [capacity, 16, 128, 128] fp32.
    for layer in pool.linear_layers:
        assert pool.conv[layer].shape == (1, 6144, 3)
        assert pool.conv[layer].dtype == torch.bfloat16
        assert pool.recurrent[layer].shape == (1, 16, 128, 128)
        assert pool.recurrent[layer].dtype == torch.float32

    conv_bytes = 18 * 6144 * 3 * 2
    recurrent_bytes = 18 * 16 * 128 * 128 * 4
    assert conv_bytes == 663_552  # 0.6328125 MiB
    assert recurrent_bytes == 18_874_368  # 18.0 MiB
    assert conv_bytes + recurrent_bytes == 19_537_920  # 18.6328125 MiB

    stats = pool.stats()
    assert stats.per_slot_bytes == 19_537_920
    assert stats.used_bytes == 19_537_920
    assert stats.reserved_bytes == 19_537_920
    assert math.isclose(stats.used_mb, 18.6328125, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(conv_bytes / (1024 * 1024), 0.6328125, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(recurrent_bytes / (1024 * 1024), 18.0, rel_tol=0, abs_tol=1e-12)

    # The same geometry through the state interface: bf16 conv, fp32 recurrent, views.
    state = pool.state(handle.slot)
    state.set_conv_state(0, torch.zeros(1, 6144, 3, dtype=torch.bfloat16))
    state.set_recurrent_state(0, torch.zeros(1, 16, 128, 128, dtype=torch.float32))
    assert state.conv_state(0).shape == (1, 6144, 3)
    assert state.conv_state(0).dtype == torch.bfloat16
    assert state.recurrent_state(0).shape == (1, 16, 128, 128)
    assert state.recurrent_state(0).dtype == torch.float32
    assert state.conv_state(0).data_ptr() == pool.conv[0][0].data_ptr()
    assert state.linear_state_bytes() == 6144 * 3 * 2 + 16 * 128 * 128 * 4


# ---------------------------------------------------------- pooled state storage


def test_pooled_state_reads_views_and_writes_with_copy():
    pool = _pool(capacity=1)
    handle = pool.alloc("r")
    state = pool.state(handle.slot)

    assert state.rows == (0,) and state.batch == 1 and state.position == 0
    for layer in pool.linear_layers:
        assert state.conv_state(layer) is None
        assert state.recurrent_state(layer) is None

    conv = torch.arange(CONV_DIM * 3, dtype=torch.float32).reshape(1, CONV_DIM, 3)
    recurrent = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    state.set_conv_state(0, conv)
    state.set_recurrent_state(0, recurrent)

    conv_view = state.conv_state(0)
    recurrent_view = state.recurrent_state(0)
    assert conv_view.shape == (1, CONV_DIM, 3) and conv_view.dtype == torch.float32
    assert recurrent_view.shape == (1, 2, 4, 4)
    # A view of the pool row, not a per-request copy.
    assert conv_view.data_ptr() == pool.conv[0][0].data_ptr()
    assert recurrent_view.data_ptr() == pool.recurrent[0][0].data_ptr()

    # Only layer 0 is written, so the byte count matches the base class's rule
    # (unset layers do not count) rather than all 18 rows.
    assert state.linear_state_bytes() == CONV_DIM * 3 * 4 + 2 * 4 * 4 * 4

    # The setter copies into the existing storage: the view address never changes and
    # the pool tensor is not rebound.
    address = conv_view.data_ptr()
    pool_address = pool.conv[0].data_ptr()
    replaced = conv + 1.0
    state.set_conv_state(0, replaced)
    assert state.conv_state(0).data_ptr() == address
    assert pool.conv[0].data_ptr() == pool_address
    torch.testing.assert_close(state.conv_state(0), replaced, atol=0, rtol=0)
    torch.testing.assert_close(pool.conv[0][0], replaced[0], atol=0, rtol=0)


def test_pooled_state_preserves_base_validation():
    pool = _pool(capacity=1, dtype=torch.float32)
    state = pool.state(pool.alloc("r").slot)

    with pytest.raises(ValueError, match=r"full_attention"):
        state.set_conv_state(FULL_POSITIONS[0], torch.zeros(1, CONV_DIM, 3))
    with pytest.raises(ValueError, match=r"\(1, 24, 3\)"):
        state.set_conv_state(0, torch.zeros(1, CONV_DIM, 4))
    with pytest.raises(ValueError, match=r"pool/model dtype"):
        state.set_conv_state(0, torch.zeros(1, CONV_DIM, 3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=r"float32"):
        state.set_recurrent_state(0, torch.zeros(1, 2, 4, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=r"\(1, 2, 4, 4\)"):
        state.set_recurrent_state(0, torch.zeros(1, 2, 3, 4))
    with pytest.raises(IndexError):
        state.conv_state(24)


def test_pooled_conv_dtype_follows_the_pool():
    pool = _pool(capacity=1, dtype=torch.bfloat16)
    state = pool.state(pool.alloc("r").slot)
    conv = torch.zeros(1, CONV_DIM, 3, dtype=torch.bfloat16)
    state.set_conv_state(0, conv)
    assert state.conv_state(0).dtype == torch.bfloat16
    assert state.linear_state_bytes() == CONV_DIM * 3 * 2


def test_pooled_state_kv_is_injected_not_replaced():
    config = _config()
    pool = _pool(capacity=1, config=config)
    kv = KVcache(config.n_layer)
    state = pool.state(pool.alloc("r").slot, kv=kv)
    assert state.kv is kv

    key = torch.zeros(1, 2, 5, 8)
    value = torch.ones(1, 2, 5, 8)
    state.set_kv(FULL_POSITIONS[0], (key, value))
    assert state.get_kv(FULL_POSITIONS[0])[0] is key
    assert state.kv_bytes() == 2 * key.numel() * key.element_size()

    # The default is unchanged: no injection still means a fresh KVcache per state.
    default = Qwen35State(config)
    assert isinstance(default.kv, KVcache) and default.kv is not kv


def test_pooled_state_cull_matches_base_for_surviving_rows():
    config = _config()
    pool = _pool(capacity=2, config=config)
    first = pool.alloc("a")
    pool.alloc("b")
    pooled = pool.state([0, 1], kv=KVcache(config.n_layer))
    base = Qwen35State(config, batch=2)

    for layer in pool.linear_layers:
        conv = torch.arange(2 * CONV_DIM * 3, dtype=torch.float32).reshape(
            2, CONV_DIM, 3
        ) + layer
        recurrent = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(
            2, 2, 4, 4
        ) + layer
        pooled.set_conv_state(layer, conv)
        pooled.set_recurrent_state(layer, recurrent)
        base.set_conv_state(layer, conv.clone())
        base.set_recurrent_state(layer, recurrent.clone())
    for layer in FULL_POSITIONS:
        kv = (
            torch.arange(2 * 2 * 7 * 8, dtype=torch.float32).reshape(2, 2, 7, 8),
            torch.ones(2, 2, 7, 8),
        )
        pooled.set_kv(layer, kv)
        base.set_kv(layer, (kv[0].clone(), kv[1].clone()))
    pooled.advance(5)
    base.advance(5)

    keep = torch.tensor([1])
    pooled.cull(keep)
    base.cull(keep)

    assert pooled.batch == 1 and pooled.rows == (1,) and pooled.position == 5
    assert base.batch == 1 and base.position == 5
    for layer in pool.linear_layers:
        torch.testing.assert_close(
            pooled.conv_state(layer), base.conv_state(layer), atol=0, rtol=0
        )
        torch.testing.assert_close(
            pooled.recurrent_state(layer), base.recurrent_state(layer), atol=0, rtol=0
        )
        # A single surviving row is still a zero-copy slice of the pool.
        assert (
            pooled.conv_state(layer).data_ptr() == pool.conv[layer][1].data_ptr()
        )
    for layer in FULL_POSITIONS:
        pk, pv = pooled.get_kv(layer)
        bk, bv = base.get_kv(layer)
        torch.testing.assert_close(pk, bk, atol=0, rtol=0)
        torch.testing.assert_close(pv, bv, atol=0, rtol=0)
    assert pooled.linear_state_bytes() == base.linear_state_bytes() == TINY_SLOT_BYTES
    assert pooled.nbytes() == base.nbytes()
    assert first.slot == 0  # cull never releases a pool slot


def test_cull_to_a_non_contiguous_order_is_an_explicit_gather():
    config = _config()
    pool = _pool(capacity=3, config=config)
    for name in ("a", "b", "c"):
        pool.alloc(name)
    state = pool.state([0, 1, 2])

    for layer in pool.linear_layers:
        values = torch.stack(
            [
                torch.full((CONV_DIM, 3), float(row + layer * 10))
                for row in range(3)
            ]
        )
        state.set_conv_state(layer, values)
        state.set_recurrent_state(layer, torch.zeros(3, 2, 4, 4))

    state.advance(9)
    state.cull(torch.tensor([2, 0]))

    assert state.rows == (2, 0) and state.batch == 2 and state.position == 9
    view = state.conv_state(0)
    # Not a slice of the pool when the surviving rows are not contiguous: an exact
    # gather copy, which the module docstring calls out as the cull fallback.
    assert view.untyped_storage().data_ptr() != pool.conv[0].untyped_storage().data_ptr()
    expected = torch.stack([torch.full((CONV_DIM, 3), 2.0), torch.full((CONV_DIM, 3), 0.0)])
    torch.testing.assert_close(view, expected, atol=0, rtol=0)
    assert state.linear_state_bytes() == 2 * TINY_SLOT_BYTES

    # The setter still writes through to the right rows, in the culled order.
    state.set_conv_state(0, expected + 5.0)
    torch.testing.assert_close(pool.conv[0][2], expected[0] + 5.0, atol=0, rtol=0)
    torch.testing.assert_close(pool.conv[0][0], expected[1] + 5.0, atol=0, rtol=0)
    torch.testing.assert_close(
        pool.conv[0][1], torch.full((CONV_DIM, 3), 1.0), atol=0, rtol=0
    )


def test_rows_are_validated_and_a_state_is_bound_to_its_slots():
    pool = _pool(capacity=2)
    handle = pool.alloc("a")
    pool.alloc("b")
    state = pool.state(handle.slot)
    assert state.rows == (0,)

    with pytest.raises(ValueError, match=r"active region"):
        pool.state([2])
    with pytest.raises(ValueError, match=r"at least one slot"):
        pool.state([])
    with pytest.raises(ValueError, match=r"distinct"):
        pool.state([0, 0])
    with pytest.raises(TypeError, match=r"row indices must be ints"):
        pool.state([0.0])
    with pytest.raises(TypeError, match=r"sequence of slot indices"):
        PooledQwen35State(pool, "0")
    with pytest.raises(TypeError, match=r"LinearStatePool"):
        PooledQwen35State(object(), [0])

    # Releasing a lower slot moves the last slot's rows into it. A state object that was
    # built before the move holds the old row/epochs: it raises rather than reading the
    # moved request's state. A state built after the move sees the preserved data.
    state.set_conv_state(0, torch.full((1, CONV_DIM, 3), 4.0))
    moved = pool.state([1])
    moved.set_conv_state(0, torch.full((1, CONV_DIM, 3), 9.0))

    pool.free("a")
    for stale in (state, moved):
        with pytest.raises(RuntimeError, match=r"not the row this state was built on"):
            stale.conv_state(0)

    rebuilt = pool.state([0])
    torch.testing.assert_close(
        rebuilt.conv_state(0), torch.full((1, CONV_DIM, 3), 9.0), atol=0, rtol=0
    )
    assert rebuilt.linear_state_bytes() == CONV_DIM * 3 * 4


def test_batch_state_reads_states_written_through_other_states():
    """The readiness flag is per (slot, layer), so a batch view sees written rows."""
    config = _config()
    pool = _pool(capacity=2, config=config)
    first = pool.alloc("a")
    second = pool.alloc("b")
    for handle, value in ((first, 1.0), (second, 2.0)):
        state = pool.state(handle.slot)
        for layer in pool.linear_layers:
            state.set_conv_state(layer, torch.full((1, CONV_DIM, 3), value))
            state.set_recurrent_state(layer, torch.full((1, 2, 4, 4), value))
        state.advance(3)

    batch = pool.state([0, 1])
    assert batch.batch == 2 and batch.rows == (0, 1)
    conv = batch.conv_state(0)
    assert conv.shape == (2, CONV_DIM, 3)
    assert conv[0].eq(1.0).all() and conv[1].eq(2.0).all()
    assert batch.recurrent_state(0)[1].eq(2.0).all()
    # All 18 layers were written for both rows through the per-request states.
    assert batch.linear_state_bytes() == 2 * TINY_SLOT_BYTES
