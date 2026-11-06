"""W4 paged KV: the physical block pool and the decode container (docs/decisions.md D6).

These tests pin the contract that ``serve/`` will rely on later:

* a prefill/token write round-trips through ``gather`` bit for bit, and ``block_size`` 16
  and 32 produce the same logical sequence;
* the same KV in ``PagedPool`` and in a ``KvSlab`` gathers to the same bytes;
* a reused block's old data is masked out, never zeroed: padding comes from the dedicated
  zero block, so correctness never depends on "clear the block on allocation";
* the zero block is a separate, shared, read-only tile outside the allocator's free list;
* one logical token maps to ``(block_table[j // bs], j % bs)`` across block boundaries;
* ``PagedKV`` keeps ``gather`` columns == ``lpad`` == mask width, and the RoPE table length
  is the run's position bound -- a different number from ``lpad`` (P-34);
* the decode forward still dispatches ``hand_baddbmm`` once per layer (D4), and the paged
  decode equals the per-sequence ``KVcache`` reference.

CPU only, tiny model, no golden files, no allocator of our own (``BlockManager`` is used
where a free list matters).
"""

from __future__ import annotations

import pytest
import torch

from model import KVcache, KvSlab, MiniLLM, MiniLLMConfig
from model.attention import DECODE_BACKEND
from model.paged import PagedKV, PagedPool
from serve.block_manager import BlockManager
from serve.engine import MiniLLMEngine

N_LAYER = 2
N_KV_HEAD = 2
HEAD_DIM = 4
N_HEAD = 4
BS16 = 16
BS32 = 32


def _pool(n_blocks: int = 8, block_size: int = BS16, n_layer: int = N_LAYER,
          dtype: torch.dtype = torch.float32) -> PagedPool:
    return PagedPool(n_layer, n_blocks, block_size, N_KV_HEAD, HEAD_DIM,
                     dtype=dtype, device="cpu", n_head=N_HEAD)


def _kv(length: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(length, N_KV_HEAD, HEAD_DIM, generator=gen)
    v = torch.randn(length, N_KV_HEAD, HEAD_DIM, generator=gen)
    return k, v


def _split(gathered: torch.Tensor, length: int, rows: int = 1) -> torch.Tensor:
    """``(rows, H, lpad, D)`` -> ``(rows, length, H, D)`` for comparison with the source."""
    return gathered[:, :, :length, :].permute(0, 2, 1, 3)


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(vocab_size=64, max_position_embeddings=128, n_embd=32, n_head=N_HEAD,
                         n_kv_head=N_KV_HEAD, n_layer=N_LAYER, dropout=0.0, bias=False)


# ------------------------------------------------------------------ pool: write -> gather

@pytest.mark.parametrize("block_size", [BS16, BS32])
def test_prefill_write_then_gather_is_bit_identical(block_size):
    pool = _pool(block_size=block_size)
    length = block_size + 3  # crosses one block boundary
    block_table = [5, 1, 7]
    k, v = _kv(length)

    for layer in range(N_LAYER):
        pool.write_prefill(layer, block_table, 0, k, v)

    lpad = 64
    for layer in range(N_LAYER):
        got_k, got_v = pool.gather(layer, [block_table], [length], lpad)
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
        sk, sv = small.gather(layer, [tbl_small], [length], 64)
        lk, lv = large.gather(layer, [tbl_large], [length], 64)
        assert torch.equal(sk, lk) and torch.equal(sv, lv)


def test_token_offsets_cross_the_block_boundary():
    """One logical token is ``block_table[j // bs]``'s column ``j % bs`` (bs=16)."""
    pool = _pool(block_size=BS16)
    block_table = [2, 6]
    k, v = _kv(3, seed=11)
    positions = [15, 16, 17]  # 15 -> (2, 15), 16 -> (6, 0), 17 -> (6, 1)

    for layer in range(N_LAYER):
        pool.write_token(layer, [0, 0, 0], [block_table] * 3, positions, k, v)

    for layer in range(N_LAYER):
        assert torch.equal(pool.keys[layer][2, 15], k[0])
        assert torch.equal(pool.keys[layer][6, 0], k[1])
        assert torch.equal(pool.keys[layer][6, 1], k[2])
        assert torch.equal(pool.values[layer][2, 15], v[0])
        assert torch.equal(pool.values[layer][6, 0], v[1])
        assert torch.equal(pool.values[layer][6, 1], v[2])


# ------------------------------------------------------------------ stale blocks, zero block

def test_reused_block_keeps_stale_bytes_but_padding_is_masked():
    """A shorter request reusing a dirty block must not see the old tail (P-33)."""
    pool = _pool(n_blocks=4, block_size=BS16)
    # Dirty every physical block with finite junk; only the zero block stays zero.
    for layer in range(N_LAYER):
        pool.keys[layer].fill_(7.25)
        pool.values[layer].fill_(-3.5)
    assert not bool(pool.zero_block.any())

    old_k, old_v = _kv(20, seed=5)
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0, 1], 0, old_k, old_v)

    # The next request reuses block 0 but only owns its first 5 logical tokens.
    length = 5
    gathered_k, gathered_v = pool.gather(0, [[0]], [length], 16)
    assert torch.equal(_split(gathered_k, length), old_k[:length].unsqueeze(0))
    assert not bool(gathered_k[:, :, length:, :].any()), "stale block bytes leaked as padding"
    assert not bool(gathered_v[:, :, length:, :].any())


def test_zero_block_is_separate_shared_and_outside_the_free_list():
    pool = _pool(n_blocks=8, block_size=BS16)
    zero = pool.zero_block
    assert zero.shape == (BS16, N_KV_HEAD, HEAD_DIM)
    assert not bool(zero.any()), "the dedicated zero block must start zeroed"

    # Shared by every row and layer: one tile, not one per block/layer.
    for layer in range(N_LAYER):
        assert not any(zero.untyped_storage().data_ptr()
                       == p.untyped_storage().data_ptr()
                       for p in (pool.keys[layer], pool.values[layer]))

    # Not reachable through the allocator: every reserved id indexes a pool block.
    manager = BlockManager(pool.n_blocks, pool.block_size)
    ids = manager.reserve(3)
    assert ids == [0, 1, 2]
    for block_id in ids:
        assert 0 <= block_id < pool.n_blocks
    # The zero block is not part of the pool's byte account (P-35).
    expected = 2 * N_LAYER * pool.n_blocks * BS16 * N_KV_HEAD * HEAD_DIM * 4
    assert pool.bytes() == expected

    # Gathering through dirty blocks must not write to it.
    for layer in range(N_LAYER):
        pool.keys[layer].fill_(9.0)
    pool.gather(0, [[0, 1]], [20], 32)
    assert not bool(pool.zero_block.any()), "gather wrote into the shared zero block"


def test_padding_is_zero_even_when_the_pool_starts_dirty():
    """Correctness cannot depend on the pool being zero-initialised."""
    pool = _pool(block_size=BS16)
    for layer in range(N_LAYER):
        pool.keys[layer].fill_(1.5)
        pool.values[layer].fill_(2.5)
    k, v = _kv(3, seed=9)
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0], 0, k, v)
    got_k, got_v = pool.gather(0, [[0]], [3], 8)
    assert torch.equal(_split(got_k, 3), k.unsqueeze(0))
    assert not bool(got_k[:, :, 3:, :].any())
    assert not bool(got_v[:, :, 3:, :].any())


# ------------------------------------------------------------------ slab equivalence

def test_gather_equals_the_slab_for_the_same_kv():
    """Same bytes, two containers: the gathered view == the slab's masked view."""
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
        gathered_k, gathered_v = pool.gather(layer, [[0, 1], [2]], lengths, lpad)
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
        pool.write_token(0, [0], [[0]], [0, 1], k, v)  # rows/positions/k disagree
    with pytest.raises(ValueError):
        pool.write_prefill(0, [0], 0, k[:, :, :2], v)  # wrong head_dim
    with pytest.raises(IndexError):
        pool.write_prefill(0, [0], 0, *_kv(BS16 + 1))  # block table too short
    with pytest.raises(IndexError):
        pool.write_token(0, [0], [[0]], [BS16], *_kv(1))  # position past its block
    with pytest.raises(ValueError):
        pool.gather(0, [[0]], [9], 4)  # lpad does not cover the longest row
    with pytest.raises(IndexError):
        pool.write_prefill(0, [pool.n_blocks], 0, k, v)  # id outside the pool


# ------------------------------------------------------------------ decode container

def _seated_model_and_prompts():
    """Prefill three prompts into a pool; return ``(model, engine, pool, kv, seats)``."""
    prompts = [
        [3, 5, 7, 11, 13],
        [3, 5, 7, 11, 13, 17, 19, 23, 29],
        [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    ]
    torch.manual_seed(0)
    model = MiniLLM(_config()).eval()
    engine = MiniLLMEngine(model)
    dtype = model.token_embedding.weight.dtype
    pool = PagedPool(N_LAYER, 12, BS16, N_KV_HEAD, model.head_dim,
                     dtype=dtype, device="cpu", n_head=N_HEAD)
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


def test_decode_container_matches_per_sequence_and_keeps_lpad_aligned(monkeypatch):
    model, engine, pool, kv, seats = _seated_model_and_prompts()
    calls: list[int] = []
    original = DECODE_BACKEND.decode

    def counting_decode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(DECODE_BACKEND, "decode", counting_decode)

    tokens = [token for _, token in seats]
    widths: list[int] = []
    mask_ptrs: list[int] = []
    for step in range(4):
        got, _ = model(torch.tensor(tokens).view(-1, 1), past_keys_values=kv,
                       logits_to_keep=1)
        # Per-step alignment: gather columns == lpad == mask width == bias width, and lpad
        # covers every row's written length.
        assert kv.lpad >= max(kv.pos.tolist()), "lpad must cover the longest row"
        assert kv.mask.shape[-1] == kv.lpad == kv.bias.shape[-1]
        widths.append(kv.lpad)
        mask_ptrs.append(kv.mask.data_ptr())
        # Count *before* the reference loop (that loop also uses the decode backend).
        assert len(calls) == N_LAYER, "one hand_baddbmm per layer per step (D4)"
        calls.clear()

        ref = torch.cat([engine.decode(token, cache) for cache, token in seats], dim=0)
        calls.clear()  # the per-sequence reference also goes through the decode backend
        assert torch.allclose(got[:, -1], ref, atol=1e-5, rtol=1e-4), f"step {step} drifted"
        assert [int(r.argmax(-1)) for r in got] == [int(r.argmax(-1)) for r in ref]

        tokens = [int(r.argmax(-1)) for r in got]
        seats = [(cache, token) for (cache, _), token in zip(seats, tokens)]

    assert len(set(mask_ptrs)) == 1, "the mask buffer was reallocated between steps"
    assert widths == [64, 64, 64, 64], f"lpad moved inside one bucket: {widths}"
    assert kv.pos.tolist() == [len(p) + 4 for p in
                               ([3, 5, 7, 11, 13],
                                [3, 5, 7, 11, 13, 17, 19, 23, 29],
                                [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])]


def test_rope_table_length_is_the_position_bound_not_lpad():
    """P-34: the RoPE table covers the run's positions; ``lpad`` is a per-step number."""
    model, engine, pool, kv, _ = _seated_model_and_prompts()
    assert kv.max_len == 100 and kv.lpad is None

    first = None
    for _ in range(3):
        model(torch.tensor([[3], [5], [7]]), past_keys_values=kv, logits_to_keep=1)
        if first is None:
            first = model._rope_cache
        assert model._rope_cache is first, "the RoPE table was rebuilt mid-run"
        assert model._rope_key[0] == 100, "the table length is the run's position bound"
        assert kv.lpad == 64 and kv.lpad != model._rope_key[0], (
            "lpad (gather/mask width) and the RoPE table length are different numbers"
        )


def test_paged_kv_rejects_unknown_shapes_and_overflow():
    model, engine, pool, kv, _ = _seated_model_and_prompts()
    kv.build_mask(next_token=True)
    with pytest.raises(ValueError):
        model(torch.tensor([[1], [2], [3], [4]]), past_keys_values=kv)
    bad = PagedPool(N_LAYER, 2, BS16, N_KV_HEAD, HEAD_DIM, n_head=N_HEAD)
    with pytest.raises((IndexError, ValueError)):
        PagedKV(bad, [[0]], [BS16 + 5], rope_len=16)  # length beyond the block table


# ------------------------------------------------------------------ container protocol

def test_container_protocol_is_explicit_not_isinstance():
    """``KVcache`` keeps the growing-cache path; a fake flagged container is checked."""
    from model.slab import as_decode_container

    assert as_decode_container(None) is None
    assert as_decode_container(KVcache(N_LAYER)) is None, "KVcache must not look like a slab"
    slab = KvSlab(1, 1, 1, 4, 8)
    assert as_decode_container(slab) is slab
    pool = _pool()
    paged = PagedKV(pool, [[0]], [3], rope_len=16)
    assert as_decode_container(paged) is paged

    class Incomplete:
        decode_container = True  # claims the protocol but lacks every hook
        pos = None

    with pytest.raises(TypeError, match="missing"):
        as_decode_container(Incomplete())


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
        old_k, old_v = dirty_pool.gather(layer, [reused_ids], [7], 16)
        new_k, new_v = dirty_pool.gather(layer, [fresh_ids], [7], 16)
        assert torch.equal(old_k, new_k) and torch.equal(old_v, new_v), "reuse leaked old KV"


def test_workspace_is_reused_across_steps_and_layers():
    pool = _pool(n_blocks=4, block_size=BS16)
    k, v = _kv(5, seed=41)
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0], 0, k, v)
    first = pool.gather(0, [[0]], [5], 64)
    addresses = [t.data_ptr() for t in first]
    for layer in range(N_LAYER):
        for _ in range(3):
            again = pool.gather(layer, [[0]], [5], 64)
            assert [t.data_ptr() for t in again] == addresses, "gather reallocated its workspace"
    assert pool.workspace_mb() > 0
    # workspace is not part of the pool's KV bytes (P-35: reported separately).
    assert pool.mb() == pytest.approx(2 * N_LAYER * 4 * BS16 * N_KV_HEAD * HEAD_DIM * 4 / 2**20)


def test_workspace_rows_follow_active_rows_not_max_rows():
    """``max_rows`` is a hard cap, not the allocation size (W4 blocker)."""
    pool = PagedPool(N_LAYER, 16, BS16, N_KV_HEAD, HEAD_DIM, n_head=N_HEAD, max_rows=64)
    for row in range(8):
        for layer in range(N_LAYER):
            pool.write_prefill(layer, [row], 0, *_kv(5, seed=42 + row))
    ws_k, _ = pool.gather(0, [[0]] * 8, [5] * 8, 64)
    # Eight active rows: eight rows of workspace, not ``max_rows`` (64) rows.
    assert pool._workspace[64][0].shape[0] == 8 and ws_k.shape[0] == 8
    assert pool.workspace_mb() == pytest.approx(
        2 * 8 * 64 * N_KV_HEAD * HEAD_DIM * 4 / 2**20
    )
    # The obvious comparison: a run that allows 64 rows pays exactly what an 8-row run does.
    small = PagedPool(N_LAYER, 16, BS16, N_KV_HEAD, HEAD_DIM, n_head=N_HEAD, max_rows=8)
    small.gather(0, [[0]] * 8, [5] * 8, 64)
    assert small.workspace_mb() == pool.workspace_mb()

    # A step at the same bucket and row count reuses the same addresses.
    again_k, _ = pool.gather(0, [[0]] * 8, [5] * 8, 64)
    assert again_k.data_ptr() == ws_k.data_ptr(), "same bucket and row count: no rebuild"

    # A larger active batch rebuilds (and the numbers grow with the rows, not with the cap).
    for row in range(8, 12):
        for layer in range(N_LAYER):
            pool.write_prefill(layer, [row], 0, *_kv(5, seed=42 + row))
    grown_k, _ = pool.gather(0, [[0]] * 12, [5] * 12, 64)
    assert grown_k.data_ptr() != ws_k.data_ptr(), "the row cap grew, so the buffer was rebuilt"
    assert grown_k.shape[0] == 12
    assert pool.workspace_mb() == pytest.approx(
        2 * 12 * 64 * N_KV_HEAD * HEAD_DIM * 4 / 2**20
    )


def _workspace_mb(lpad: int, n_rows: int) -> float:
    """Appendix C gather workspace: k+v for ``n_rows`` active rows at ``lpad``."""
    return 2 * n_rows * lpad * N_KV_HEAD * HEAD_DIM * 4 / 2**20


def test_workspace_keeps_only_the_current_lpad_bucket():
    """Buckets do not accumulate: switching lpad releases the previous pair."""
    pool = _pool(n_blocks=8, block_size=BS16)
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0], 0, *_kv(5, seed=51))
        pool.write_prefill(layer, [1], 0, *_kv(5, seed=52))

    peak = 0.0
    for lpad in (64, 128, 256, 512, 1024, 1536, 1664):
        pool.gather(0, [[0], [1]], [5, 5], lpad)
        assert len(pool._workspace) == 1, f"{len(pool._workspace)} lpad buckets resident"
        expected = _workspace_mb(lpad, 2)
        assert pool.workspace_mb() == pytest.approx(expected)
        peak = max(peak, expected)
    assert pool.workspace_peak_mb() == pytest.approx(peak)
    assert pool.workspace_mb() == pytest.approx(_workspace_mb(1664, 2))


def test_workspace_does_not_accumulate_when_lpad_falls_back():
    """A long row leaving shrinks the bucket; the old one is not kept around."""
    pool = _pool(n_blocks=8, block_size=BS16)
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0], 0, *_kv(5, seed=61))
    pool.gather(0, [[0]], [5], 1024)
    assert len(pool._workspace) == 1
    pool.gather(0, [[0]], [5], 64)
    assert len(pool._workspace) == 1
    assert pool.workspace_mb() == pytest.approx(_workspace_mb(64, 1))
    assert pool.workspace_peak_mb() == pytest.approx(_workspace_mb(1024, 1))


def test_max_rows_is_a_hard_cap_on_the_gather_batch():
    pool = PagedPool(N_LAYER, 8, BS16, N_KV_HEAD, HEAD_DIM, n_head=N_HEAD, max_rows=2)
    for row in range(3):
        for layer in range(N_LAYER):
            pool.write_prefill(layer, [row], 0, *_kv(2, seed=70 + row))
    with pytest.raises(ValueError, match="max_rows"):
        pool.gather(0, [[0], [1], [2]], [2, 2, 2], 64)


# ------------------------------------------------------------------ gather account (§5)

def test_step_gather_traffic_is_shape_derived_and_read_equals_write():
    """Appendix C: one step reads and writes ``n_layer * n_rows * lpad * 2*H_kv*D*item``."""
    pool = _pool(n_blocks=8, block_size=BS16)  # n_layer=2, H_kv=2, D=4, float32
    n_rows, lpad = 2, 32
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0], 0, *_kv(5))
        pool.write_prefill(layer, [1], 0, *_kv(5, seed=1))

    pool.begin_step()
    for layer in range(N_LAYER):
        pool.gather(layer, [[0], [1]], [5, 5], lpad)

    per_token_per_layer = 2 * N_KV_HEAD * HEAD_DIM * torch.float32.itemsize  # k + v
    expected_bytes = N_LAYER * n_rows * lpad * per_token_per_layer
    stats = pool.step_stats()
    assert stats["gather_calls"] == N_LAYER
    assert stats["gather_read_mb"] == pytest.approx(expected_bytes / 2**20)
    assert stats["gather_write_mb"] == pytest.approx(expected_bytes / 2**20)
    assert stats["gather_read_mb"] == stats["gather_write_mb"]
    assert stats["gather_ms"] >= 0.0


def test_begin_step_zeroes_the_gather_account():
    pool = _pool(n_blocks=4, block_size=BS16)
    for layer in range(N_LAYER):
        pool.write_prefill(layer, [0], 0, *_kv(5))
    pool.begin_step()
    for layer in range(N_LAYER):
        pool.gather(layer, [[0]], [5], 64)
    assert pool.step_stats()["gather_calls"] == N_LAYER
    assert pool.step_stats()["gather_read_mb"] > 0.0

    pool.begin_step()
    assert pool.step_stats() == {
        "gather_ms": 0.0,
        "gather_read_mb": 0.0,
        "gather_write_mb": 0.0,
        "gather_calls": 0,
    }
