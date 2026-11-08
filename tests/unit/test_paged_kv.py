"""W4 paged KV: physical block pool + the W4.5 Triton block-table container.

These tests pin the contract ``serve/`` relies on:

* a prefill/token write round-trips through the dense oracle bit for bit, and
  ``block_size`` 16 and 32 produce the same logical sequence;
* the same KV in ``PagedPool`` and in a ``KvSlab`` gathers to the same bytes;
* one logical token maps to ``(block_table[j // bs], j % bs)`` across block boundaries;
* ``PagedKV`` writes each step's token through one slot map and refreshes only the int32
  seqlen at the step boundary (no mask/bias, no ``lpad``);
* the paged container is decode-only on CUDA: the gather control arm was deleted, so a CPU
  pool is rejected rather than silently gathering;
* the decode forward matches the per-sequence ``KVcache`` reference (CUDA only).

The dense gather/oracle lives in ``tests/reference/paged.py``; production has no
``PagedPool.gather``, ``zero_block``, workspace or mask/bias pair any more. CPU tests use
the oracle as a *reader* of the pool's bytes, never as a production path.
"""

from __future__ import annotations

import pytest
import torch

from model import KVcache, KvSlab, MiniLLM, MiniLLMConfig
from model.paged import PagedKV, PagedPool
from model.triton_decode import TRITON_AVAILABLE
from serve.block_manager import BlockManager
from serve.engine import MiniLLMEngine
from tests.reference.paged import gather_pool, write_token_reference

N_LAYER = 2
N_KV_HEAD = 2
HEAD_DIM = 4
N_HEAD = 4
BS16 = 16
BS32 = 32

CUDA_READY = TRITON_AVAILABLE and torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(
    not CUDA_READY, reason="paged decode is the CUDA Triton block-table kernel"
)


def _pool(n_blocks: int = 8, block_size: int = BS16, n_layer: int = N_LAYER,
          dtype: torch.dtype = torch.float32, device: str = "cpu") -> PagedPool:
    return PagedPool(n_layer, n_blocks, block_size, N_KV_HEAD, HEAD_DIM,
                     dtype=dtype, device=device, n_head=N_HEAD)


def _kv(length: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(length, N_KV_HEAD, HEAD_DIM, generator=gen)
    v = torch.randn(length, N_KV_HEAD, HEAD_DIM, generator=gen)
    return k, v


def _split(gathered: torch.Tensor, length: int, rows: int = 1) -> torch.Tensor:
    """``(rows, H, lpad, D)`` -> ``(rows, length, H, D)`` for comparison with the source."""
    return gathered[:, :, :length, :].permute(0, 2, 1, 3)


def _reference_prefill(pool: PagedPool, layer: int, block_table, start: int,
                       k: torch.Tensor, v: torch.Tensor) -> None:
    """The pre-slot-map implementation, verbatim: one slice copy per physical block.

    Test oracle for ``write_prefill``: same mapping (``block_table[t // bs]``, column
    ``t % bs``), but per block instead of one advanced-index write, so comparing the whole
    pools pins the bytes, not just the gathered view.
    """
    k, v = k.to(pool.dtype), v.to(pool.dtype)
    written, length = 0, k.shape[0]
    while written < length:
        position = start + written
        block_id = block_table[position // pool.block_size]
        offset = position % pool.block_size
        n = min(pool.block_size - offset, length - written)
        pool.keys[layer][block_id, offset : offset + n] = k[written : written + n]
        pool.values[layer][block_id, offset : offset + n] = v[written : written + n]
        written += n


def _kv_engine_layout(length: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(length, H, D)`` non-contiguous views, the layout ``PagedWriteSink`` hands in."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(1, N_KV_HEAD, length, HEAD_DIM, generator=gen)[0].permute(1, 0, 2)
    v = torch.randn(1, N_KV_HEAD, length, HEAD_DIM, generator=gen)[0].permute(1, 0, 2)
    assert not k.is_contiguous() and not v.is_contiguous()
    return k, v


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(vocab_size=64, max_position_embeddings=128, n_embd=32, n_head=N_HEAD,
                         n_kv_head=N_KV_HEAD, n_layer=N_LAYER, dropout=0.0, bias=False)


# ------------------------------------------------------------------ pool: write -> oracle

@pytest.mark.parametrize("block_size", [BS16, BS32])
def test_prefill_write_then_oracle_gather_is_bit_identical(block_size):
    pool = _pool(block_size=block_size)
    length = block_size + 3  # crosses one block boundary
    block_table = [5, 1, 7]
    k, v = _kv(length)

    for layer in range(N_LAYER):
        pool.write_prefill(layer, block_table, 0, k, v)

    lpad = 64
    for layer in range(N_LAYER):
        got_k, got_v = gather_pool(pool, layer, [block_table], [length], lpad)
        assert got_k.shape == (1, N_KV_HEAD, lpad, HEAD_DIM) and got_v.shape == got_k.shape
        assert torch.equal(_split(got_k, length), k.unsqueeze(0))
        assert torch.equal(_split(got_v, length), v.unsqueeze(0))
        assert not bool(got_k[:, :, length:, :].any()), "padding columns must be zero"
        assert not bool(got_v[:, :, length:, :].any())


def test_block_size_does_not_change_the_logical_sequence():
    length = 40
    k, v = _kv(length, seed=3)
    small, large = _pool(block_size=BS16), _pool(block_size=BS32)
    tbl_small = [3, 0, 6]
    tbl_large = [1, 4]  # 40 tokens need ceil(40/32) == 2 blocks
    for layer in range(N_LAYER):
        small.write_prefill(layer, tbl_small, 0, k, v)
        large.write_prefill(layer, tbl_large, 0, k, v)
    for layer in range(N_LAYER):
        sk, sv = gather_pool(small, layer, [tbl_small], [length], 64)
        lk, lv = gather_pool(large, layer, [tbl_large], [length], 64)
        assert torch.equal(sk, lk) and torch.equal(sv, lv)


def test_prefill_slot_mapping_matches_per_block_reference():
    """One advanced-index write per layer == the per-block loop, including shuffled tables."""
    pool = _pool(n_blocks=8, block_size=BS16)
    reference = _pool(n_blocks=8, block_size=BS16)
    block_table = [7, 0, 5, 3]  # shuffled and non-contiguous: spelling order is not block order
    length = BS16 + 5  # crosses a block boundary and leaves a partial second block
    k, v = _kv_engine_layout(length, seed=71)

    for layer in range(N_LAYER):
        pool.write_prefill(layer, block_table, 0, k, v)
        _reference_prefill(reference, layer, block_table, 0, k, v)

    for layer in range(N_LAYER):
        assert torch.equal(pool.keys[layer], reference.keys[layer])
        assert torch.equal(pool.values[layer], reference.values[layer])

    # The formula itself, independent of the oracle: token 15 -> (7, 15), 16 -> (0, 0),
    # 20 -> (0, 4). Blocks 5 and 3 are allocated but never reached by these 21 tokens.
    assert torch.equal(pool.keys[0][7, 15], k[15])
    assert torch.equal(pool.keys[0][0, 0], k[16])
    assert torch.equal(pool.keys[0][0, 4], k[20])
    assert torch.equal(pool.values[0][7, 15], v[15])
    assert torch.equal(pool.values[0][0, 0], v[16])
    assert torch.equal(pool.values[0][0, 4], v[20])
    assert not bool(pool.keys[0][5].any()) and not bool(pool.keys[0][3].any())


def test_prefill_with_a_nonzero_start_appends_after_the_first_chunk():
    """Chunk 2 starts mid-block and crosses a boundary: the combined sequence is intact."""
    pool = _pool(n_blocks=8, block_size=BS16)
    reference = _pool(n_blocks=8, block_size=BS16)
    block_table = [6, 2, 4]
    first, second = 21, 14  # 21 ends at block 1, column 4; chunk 2 crosses into block 4
    k1, v1 = _kv(first, seed=72)
    k2, v2 = _kv(second, seed=73)

    for layer in range(N_LAYER):
        pool.write_prefill(layer, block_table, 0, k1, v1)
        pool.write_prefill(layer, block_table, first, k2, v2)
        _reference_prefill(reference, layer, block_table, 0, k1, v1)
        _reference_prefill(reference, layer, block_table, first, k2, v2)

    total = first + second
    combined_k, combined_v = torch.cat([k1, k2]), torch.cat([v1, v2])
    for layer in range(N_LAYER):
        got_k, got_v = gather_pool(pool, layer, [block_table], [total], 64)
        assert torch.equal(_split(got_k, total), combined_k.unsqueeze(0))
        assert torch.equal(_split(got_v, total), combined_v.unsqueeze(0))
        assert torch.equal(pool.keys[layer], reference.keys[layer])
        assert torch.equal(pool.values[layer], reference.values[layer])


def test_prefill_slot_cache_is_reused_per_prefill_and_invalidated_between():
    """One-entry cache: shared by every layer, rebuilt when the slot-map key changes."""
    pool = _pool(n_blocks=8, block_size=BS16)
    length = BS16 + 2
    k_a, v_a = _kv(length, seed=74)
    k_b, v_b = _kv(length, seed=75)
    table_a, table_b = [5, 1], [2, 6]

    pool.write_prefill(0, table_a, 0, k_a, v_a)
    slots_a = pool._prefill_slots
    pool.write_prefill(1, table_a, 0, k_a, v_a)
    assert pool._prefill_slots is slots_a, "a second layer rebuilt the same slot map"

    pool.write_prefill(0, table_b, 0, k_b, v_b)
    assert pool._prefill_slots is not slots_a, "a new block table reused the old slot ids"
    slots_b = pool._prefill_slots
    pool.write_prefill(1, table_b, 0, k_b, v_b)
    assert pool._prefill_slots is slots_b

    for layer in range(N_LAYER):
        assert torch.equal(pool.keys[layer][5, :BS16], k_a[:BS16])
        assert torch.equal(pool.keys[layer][1, :2], k_a[BS16:])
        assert torch.equal(pool.values[layer][5, :BS16], v_a[:BS16])
        assert torch.equal(pool.values[layer][1, :2], v_a[BS16:])
        assert torch.equal(pool.keys[layer][2, :BS16], k_b[:BS16])
        assert torch.equal(pool.keys[layer][6, :2], k_b[BS16:])
        assert torch.equal(pool.values[layer][2, :BS16], v_b[:BS16])
        assert torch.equal(pool.values[layer][6, :2], v_b[BS16:])
        # No block from the previous key was touched by the new write (or vice versa).
        for block_id in (0, 3, 4, 7):
            assert not bool(pool.keys[layer][block_id].any())
            assert not bool(pool.values[layer][block_id].any())


# ------------------------------------------------------------------ control arm deleted

def test_gather_control_arm_is_deleted_from_the_pool():
    """W4.5 step 2: no dense gather, workspace, zero block or gather account remains."""
    pool = _pool()
    for name in ("gather", "begin_step", "step_stats", "workspace_mb",
                 "workspace_peak_mb", "reset_workspace", "_workspace", "zero_block",
                 "write_token"):
        assert not hasattr(pool, name), f"PagedPool.{name} should have been deleted"
    assert not hasattr(PagedKV, "get")
    assert not hasattr(PagedKV, "set_buffers")
    assert not hasattr(PagedKV, "kernel_decode")


def test_paged_kv_rejects_a_cpu_pool_with_a_clear_error():
    """The gather arm was the CPU/MPS path; deleted means fail loudly, never fall back."""
    pool = _pool()
    with pytest.raises(ValueError, match="CUDA") as excinfo:
        PagedKV(pool, [[0]], [3], rope_len=16)
    message = str(excinfo.value)
    assert "gather control arm was deleted" in message
    assert "tests/reference/paged.py" in message


# ------------------------------------------------------------------ slab equivalence

def test_oracle_gather_equals_the_slab_for_the_same_kv():
    """Same bytes, two containers: the oracle view == the slab's masked view."""
    lengths = [13, 5]
    k0, v0 = _kv(lengths[0], seed=21)
    k1, v1 = _kv(lengths[1], seed=22)
    pool = _pool(n_blocks=4, block_size=BS16)
    slab = KvSlab(N_LAYER, 2, N_KV_HEAD, HEAD_DIM, 64, dtype=torch.float32, n_head=N_HEAD)

    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0, 1], 0, k0, v0)
        pool.write_prefill(layer, [2], 0, k1, v1)
        slab.keys[layer][0, :, : lengths[0]] = k0.permute(1, 0, 2)
        slab.values[layer][0, :, : lengths[0]] = v0.permute(1, 0, 2)
        slab.keys[layer][1, :, : lengths[1]] = k1.permute(1, 0, 2)
        slab.values[layer][1, :, : lengths[1]] = v1.permute(1, 0, 2)
    slab.valid.copy_(torch.tensor(lengths))
    slab._host_valid = list(lengths)

    lpad = 64
    for layer in range(N_LAYER):
        gathered_k, gathered_v = gather_pool(pool, layer, [[0, 1], [2]], lengths, lpad)
        slab_k, slab_v = slab.get(layer)
        columns = torch.arange(lpad)
        valid = columns.view(1, 1, -1, 1) < slab.valid.view(-1, 1, 1, 1)
        expected_k = torch.where(valid, slab_k[:, :, :lpad], torch.zeros(()))
        expected_v = torch.where(valid, slab_v[:, :, :lpad], torch.zeros(()))
        assert torch.equal(gathered_k, expected_k), f"layer {layer}: keys differ"
        assert torch.equal(gathered_v, expected_v), f"layer {layer}: values differ"


# ------------------------------------------------------------------ shape / bound guards

def test_shapes_and_bounds_fail_loudly():
    pool = _pool(block_size=BS16)
    k, v = _kv(4)
    with pytest.raises(ValueError):
        pool.write_prefill(0, [0], 0, k[:, :, :2], v)  # wrong head_dim
    with pytest.raises(IndexError):
        pool.write_prefill(0, [0], 0, *_kv(BS16 + 1))  # block table too short
    with pytest.raises(ValueError):
        gather_pool(pool, 0, [[0]], [9], 4)  # lpad does not cover the longest row
    with pytest.raises(IndexError):
        pool.write_prefill(0, [pool.n_blocks], 0, k, v)  # id outside the pool


# ------------------------------------------------------------------ decode container (CUDA)

def _seated_model_and_prompts():
    """Prefill three prompts into a CUDA pool; return ``(model, engine, pool, kv, seats)``."""
    prompts = [
        [3, 5, 7, 11, 13],
        [3, 5, 7, 11, 13, 17, 19, 23, 29],
        [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    ]
    torch.manual_seed(0)
    model = MiniLLM(_config()).to("cuda", dtype=torch.bfloat16).eval()
    engine = MiniLLMEngine(model)
    dtype = model.token_embedding.weight.dtype
    pool = PagedPool(N_LAYER, 12, BS16, N_KV_HEAD, model.head_dim,
                     dtype=dtype, device="cuda", n_head=N_HEAD)
    block_tables, seats = [], []
    for row, prompt in enumerate(prompts):
        logits, cache = engine.prefill(prompt)
        block_table = [row * 4 + i for i in range(4)]
        for layer in range(N_LAYER):
            k, v = cache.get_kv(layer)  # (1, H_kv, L, D)
            pool.write_prefill(layer, block_table, 0,
                               k[0].permute(1, 0, 2), v[0].permute(1, 0, 2))
        block_tables.append(block_table)
        seats.append((cache, int(logits.argmax(-1))))
    kv = PagedKV(pool, block_tables, [len(p) for p in prompts],
                 seats=list(range(len(prompts))), n_head=N_HEAD, rope_len=100)
    return model, engine, pool, kv, seats


@needs_cuda
def test_decode_container_matches_per_sequence_reference():
    model, engine, pool, kv, seats = _seated_model_and_prompts()
    assert kv._seqlen.dtype == torch.int32

    tokens = [token for _, token in seats]
    for step in range(4):
        got, _ = model(torch.tensor(tokens, device="cuda").view(-1, 1),
                       past_keys_values=kv, logits_to_keep=1)
        # The step boundary refreshed seqlen to the written column; ``advance`` moved pos
        # to the next one, so the two agree after the forward.
        assert torch.equal(kv._seqlen, kv.pos.to(torch.int32))
        assert not hasattr(kv, "lpad") and not hasattr(kv, "mask") and not hasattr(kv, "bias")

        ref = torch.cat([engine.decode(token, cache) for cache, token in seats], dim=0)
        torch.testing.assert_close(
            got[:, -1].float(), ref.float(), atol=2e-2, rtol=2e-2
        ), f"step {step} drifted"
        assert [int(r.argmax(-1)) for r in got] == [int(r.argmax(-1)) for r in ref]

        tokens = [int(r.argmax(-1)) for r in got]
        seats = [(cache, token) for (cache, _), token in zip(seats, tokens)]

    assert kv.pos.tolist() == [len(p) + 4 for p in
                               ([3, 5, 7, 11, 13],
                                [3, 5, 7, 11, 13, 17, 19, 23, 29],
                                [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])]


@needs_cuda
def test_build_mask_refreshes_only_the_int32_seqlen():
    """P-34: max_len is the RoPE bound; the step boundary writes seqlen, nothing else."""
    model, engine, pool, kv, _ = _seated_model_and_prompts()
    assert kv.max_len == 100
    kv.build_mask(next_token=True)
    assert kv._seqlen.dtype == torch.int32
    assert torch.equal(kv._seqlen, (kv.pos + 1).to(torch.int32))
    assert kv.build_mask(next_token=False) is None
    assert torch.equal(kv._seqlen, kv.pos.to(torch.int32))
    # No mask/bias/lpad state exists on the container.
    for name in ("lpad", "mask", "bias", "_buffers", "decode_impl"):
        assert not hasattr(kv, name), name


@needs_cuda
def test_paged_kv_rejects_unknown_shapes_and_overflow():
    model, engine, pool, kv, _ = _seated_model_and_prompts()
    kv.build_mask(next_token=True)
    with pytest.raises(ValueError):
        model(torch.tensor([[1], [2], [3], [4]], device="cuda"), past_keys_values=kv)
    bad = PagedPool(N_LAYER, 2, BS16, N_KV_HEAD, HEAD_DIM, device="cuda", n_head=N_HEAD)
    with pytest.raises((IndexError, ValueError)):
        PagedKV(bad, [[0]], [BS16 + 5], rope_len=16)  # length beyond the block table


# ------------------------------------------------------------------ container protocol

def test_container_protocol_is_explicit_not_isinstance():
    """``KVcache`` keeps the growing-cache path; an incomplete flagged container raises."""
    from model.slab import as_decode_container

    assert as_decode_container(None) is None
    assert as_decode_container(KVcache(N_LAYER)) is None, "KVcache must not look like a slab"
    slab = KvSlab(1, 1, 1, 4, 8)
    assert as_decode_container(slab) is slab

    class Incomplete:
        decode_container = True  # claims the protocol but lacks every hook
        pos = None

    with pytest.raises(TypeError, match="missing"):
        as_decode_container(Incomplete())


@needs_cuda
def test_paged_container_is_the_block_table_decoder():
    from model.slab import as_decode_container

    pool = _pool(device="cuda")
    paged = PagedKV(pool, [[0]], [3], rope_len=16)
    assert as_decode_container(paged) is paged
    assert paged.block_table_decode is True
    # The shared decode-container protocol does not require seat-only hooks.
    assert not hasattr(paged, "bias") and not hasattr(paged, "get")


def test_allocator_reuse_matches_a_never_used_block():
    """``free -> allocate`` on a dirty block must decode identically to a fresh one."""
    manager = BlockManager(4, BS16)
    dirty_ids = manager.reserve(1)
    assert dirty_ids == [0]
    dirty_pool = _pool(n_blocks=4, block_size=BS16)
    long_k, long_v = _kv(BS16, seed=31)
    for layer in range(N_LAYER):
        dirty_pool.write_prefill(layer, dirty_ids, 0, long_k, long_v)
    manager.free(dirty_ids)

    reused_ids = manager.reserve(1)
    fresh_ids = manager.reserve(1)
    assert reused_ids == [0] and fresh_ids == [1] and fresh_ids[0] != dirty_ids[0]

    short_k, short_v = _kv(7, seed=32)
    for layer in range(N_LAYER):
        dirty_pool.write_prefill(layer, reused_ids, 0, short_k, short_v)
        dirty_pool.write_prefill(layer, fresh_ids, 0, short_k, short_v)
    for layer in range(N_LAYER):
        old_k, old_v = gather_pool(dirty_pool, layer, [reused_ids], [7], 16)
        new_k, new_v = gather_pool(dirty_pool, layer, [fresh_ids], [7], 16)
        assert torch.equal(old_k, new_k) and torch.equal(old_v, new_v), "reuse leaked old KV"


# ---------------------------------------------------- decode write: slot map (W4.5 C1)
#
# The per-layer decode write used to build Python ``block_ids``/``offsets`` lists and do
# one advanced-index assignment per layer. It now derives one device slot tensor per step
# (``PagedKV._refresh_slots``) and scatters with ``PagedPool.write_slots``. These tests pin
# the bytes to the *same* mapping ``write_token_reference`` implements, so the optimization
# cannot change what lands in the pool. They run on CUDA because the container is CUDA-only.

@needs_cuda
def test_slot_write_is_bit_identical_to_write_token():
    torch.manual_seed(11)
    pool_slots, pool_ref = _pool(n_blocks=8, device="cuda"), _pool(n_blocks=8, device="cuda")
    assert torch.equal(pool_slots.keys[0], pool_ref.keys[0])
    tables = [[5, 1, 7], [2, 0, 6]]
    lengths = [20, 5]
    kv = PagedKV(pool_slots, tables, lengths, rope_len=64, n_head=N_HEAD)

    k = torch.randn(2, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    v = torch.randn(2, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    kv.write(0, k, v)

    for row, (table, position) in enumerate(zip(tables, lengths)):
        write_token_reference(
            pool_ref, 0, [row], [table], [position], k[row : row + 1, :, 0, :],
            v[row : row + 1, :, 0, :],
        )

    assert torch.equal(pool_slots.keys[0], pool_ref.keys[0])
    assert torch.equal(pool_slots.values[0], pool_ref.values[0])


@needs_cuda
def test_slot_write_crosses_a_block_boundary_with_a_shuffled_table():
    """The slot is ``table[pos // bs] * bs + pos % bs`` -- not the logical column."""
    pool = _pool(n_blocks=8, device="cuda")
    table = [5, 1, 7]  # deliberately not [0, 1, 2]
    kv = PagedKV(pool, [table], [BS16 - 1], rope_len=64, n_head=N_HEAD)
    before = pool.keys[0].clone()
    k = torch.randn(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    v = torch.randn(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    kv.write(0, k, v)

    flat = pool.keys[0].view(-1, N_KV_HEAD, HEAD_DIM)
    slot = table[(BS16 - 1) // BS16] * BS16 + (BS16 - 1) % BS16
    assert torch.equal(flat[slot], k[0, :, 0, :])
    changed = (pool.keys[0] != before).any(dim=-1).any(dim=-1).view(-1).nonzero().tolist()
    assert changed == [[slot]], f"only the mapped slot may change, got {changed}"


@needs_cuda
def test_slot_map_follows_a_rebuilt_container():
    """A new container (new block table) must write through *its* table, not a cached one."""
    pool = _pool(n_blocks=8, device="cuda")
    k_a = torch.randn(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    k_b = torch.randn(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    v = torch.zeros(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")

    first = PagedKV(pool, [[0, 1, 2]], [5], rope_len=64, n_head=N_HEAD)
    first.write(0, k_a, v)
    second = PagedKV(pool, [[3, 4, 5]], [5], rope_len=64, n_head=N_HEAD)
    second.write(0, k_b, v)

    flat = pool.keys[0].view(-1, N_KV_HEAD, HEAD_DIM)
    assert torch.equal(flat[0 * BS16 + 5], k_a[0, :, 0, :])
    assert torch.equal(flat[3 * BS16 + 5], k_b[0, :, 0, :])


@needs_cuda
def test_write_past_the_block_table_raises_before_touching_the_pool():
    pool = _pool(n_blocks=4, device="cuda")
    kv = PagedKV(pool, [[0]], [BS16], rope_len=32, n_head=N_HEAD)  # exactly one block
    before = pool.keys[0].clone()
    k = torch.randn(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")
    v = torch.randn(1, N_KV_HEAD, 1, HEAD_DIM, device="cuda")

    with pytest.raises(IndexError, match="block table has 1 blocks"):
        kv.write(0, k, v)
    assert torch.equal(pool.keys[0], before), "a refused write must not touch the pool"

    # The next position after the last token is out of range; the container parks there
    # without raising until a write is actually attempted (a request may end on it).
    kv2 = PagedKV(pool, [[1]], [BS16 - 1], rope_len=32, n_head=N_HEAD)
    kv2.build_mask(next_token=True)
    kv2.write(0, k, v)
    kv2.advance()
    assert kv2._slot_ids is None
    with pytest.raises(IndexError, match="needs block 1"):
        kv2.write(0, k, v)


def test_write_slots_rejects_bad_arguments():
    pool = _pool(n_blocks=4)
    k, v = _kv(2)  # (2, H_KV, D)
    slots = torch.tensor([0, 1], dtype=torch.long)
    pool.write_slots(0, slots, k, v)  # the accepted shape, for contrast
    slots32 = slots.to(torch.int32)
    pool.write_slots(0, slots32, k, v)  # int32 is accepted and cast to long
    assert torch.equal(pool.keys[0].view(-1, N_KV_HEAD, HEAD_DIM)[slots32.long()], k)

    with pytest.raises(ValueError, match="one entry per token"):
        pool.write_slots(0, slots[:1], k, v)
    with pytest.raises(ValueError, match="integer tensor"):
        pool.write_slots(0, slots.float(), k, v)
    with pytest.raises(ValueError, match="k/v shapes differ"):
        pool.write_slots(0, slots, k, v[:1])
    with pytest.raises(ValueError, match="head shape"):
        pool.write_slots(0, slots, k[:, :, :2], v[:, :, :2])
    with pytest.raises(IndexError, match="outside the pool's"):
        pool.write_slots(N_LAYER, slots, k, v)  # layer index is 0..N_LAYER-1
    with pytest.raises((IndexError, RuntimeError)):
        pool.write_slots(0, torch.tensor([0, 10_000], dtype=torch.long), k, v)
