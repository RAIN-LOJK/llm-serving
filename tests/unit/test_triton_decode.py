"""W4.5 block-table decode: the Triton kernel against a hand-written dense oracle.

The paged arm has two implementations on purpose. ``paged_decode`` is the production
kernel (CUDA + Triton only); ``reference_paged_decode`` is the slow PyTorch oracle, and
this file uses a *second*, even more explicit dense oracle for the oracle itself. The
three-way split means a bug has to survive two independent formulations before it can
pass, and the CPU-only tests run on hosts where Triton is not installable.

What is pinned here:

* the reference gathers logical tokens through the block table (non-contiguous,
  duplicated and reused block ids, unused table entries, last-page truncation) and
  matches a dense per-token softmax;
* the Triton kernel matches the reference for block_size 16/32, D 64/128/256,
  GQA 1/2/4/8, B 1/3/8 and seqlen 1/15/16/17/100/1664, with fp32 and bf16 tolerances
  (bf16 also compares argmax), and never produces NaN/Inf;
* the kernel rejects CPU tensors, a non-int32 block table, a cross-device table and a
  seqlen larger than the table capacity instead of silently truncating or falling back;
* ``paged_decode`` does not mutate any input, including a non-contiguous q;
* the per-container launch plan (``paged_decode_plan``) launches exactly what
  ``paged_decode`` launches for every layer, refuses a query whose layout differs from
  the one it validated, and does not snapshot ``seqlen``: it is the same tensor the
  container updates in place every step.

CUDA cases are skipped on the CPU gate; they are the only tests that compile Triton.
"""

from __future__ import annotations

import math

import pytest
import torch

from model.triton_decode import (
    TRITON_AVAILABLE,
    paged_decode,
    paged_decode_plan,
    reference_paged_decode,
)

CUDA_READY = TRITON_AVAILABLE and torch.cuda.is_available()
CUDA = pytest.mark.skipif(not CUDA_READY, reason="needs Triton and CUDA")
#: Fast gate marker, same convention as the benchmark's ``gpu`` tests.
GPU = pytest.mark.gpu


# ------------------------------------------------------------------------------- builders


def _build(B, H, Hkv, D, block_size, lengths, *, device="cpu", dtype=torch.float32,
           seed=0, n_blocks=None):
    """Random pool plus per-row random block tables; rows may share physical blocks.

    The table has one column per logical page of the longest row. Shorter rows leave a
    tail that the kernel must not read; it is filled with a *valid* id so a wrong read
    would show up as a wrong value rather than an out-of-bounds crash.
    """
    gen = torch.Generator().manual_seed(seed)
    pages = [(int(length) + block_size - 1) // block_size for length in lengths]
    max_pages = max(max(pages), 1)
    if n_blocks is None:
        n_blocks = max(B * max_pages, max_pages + 4)
    k = torch.randn(n_blocks, block_size, Hkv, D, generator=gen).to(device=device, dtype=dtype)
    v = torch.randn(n_blocks, block_size, Hkv, D, generator=gen).to(device=device, dtype=dtype)
    q = torch.randn(B, H, D, generator=gen).to(device=device, dtype=dtype)

    table = torch.zeros(B, max_pages, dtype=torch.int32)
    for row, n_pages in enumerate(pages):
        if n_pages:
            table[row, :n_pages] = torch.randperm(n_blocks, generator=gen)[:n_pages].to(
                torch.int32
            )
    seqlen = torch.tensor(lengths, dtype=torch.int32)
    return q, k, v, table.to(device), seqlen.to(device)


def _dense_oracle(q, k_cache, v_cache, block_table, seqlen, scale):
    """Per-token dense softmax, written with plain indexing and float64 math.

    Deliberately different from ``reference_paged_decode`` (which uses ``index_select``
    and fp32 einsum): every logical token is looked up through the block table one at a
    time, so a wrong page/offset mapping cannot agree with the kernel by accident.
    """
    B, H, D = q.shape
    block_size = k_cache.shape[1]
    H_kv = k_cache.shape[2]
    group = H // H_kv
    out = torch.empty_like(q)
    for row in range(B):
        n = int(seqlen[row])
        if n == 0:
            out[row].zero_()
            continue
        keys, values = [], []
        for token in range(n):
            block_id = int(block_table[row, token // block_size])
            offset = token % block_size
            keys.append(k_cache[block_id, offset])
            values.append(v_cache[block_id, offset])
        k = torch.stack(keys).to(torch.float64)       # (n, H_kv, D)
        v = torch.stack(values).to(torch.float64)
        for head in range(H):
            kv_head = head // group
            scores = (q[row, head].to(torch.float64) @ k[:, kv_head, :].T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[row, head] = (probs @ v[:, kv_head, :]).to(q.dtype)
    return out


def _cpu_inputs(B=2, H=4, Hkv=2, D=8, block_size=16, lengths=(17, 5)):
    return _build(B, H, Hkv, D, block_size, lengths)


# ------------------------------------------------------------------ oracle: CPU, no Triton


@pytest.mark.parametrize(
    "B,H,Hkv,D,block_size,lengths",
    [
        pytest.param(1, 2, 1, 8, 16, [1], id="seqlen-1-gqa2-d8"),
        pytest.param(2, 4, 2, 8, 16, [15, 16], id="seqlen-15-16-gqa2-d8"),
        pytest.param(1, 4, 2, 16, 16, [17], id="seqlen-17-gqa2-d16"),
        pytest.param(2, 4, 4, 8, 16, [32, 0], id="seqlen-32-and-empty-gqa1"),
        pytest.param(2, 4, 2, 16, 32, [100, 33], id="seqlen-100-bs32"),
        pytest.param(3, 2, 2, 8, 32, [16, 17, 15], id="gqa1-bs32-ragged"),
    ],
)
def test_reference_matches_dense_oracle_on_cpu(B, H, Hkv, D, block_size, lengths):
    q, k, v, table, seqlen = _build(B, H, Hkv, D, block_size, lengths, seed=3)
    scale = 1.0 / math.sqrt(D)

    got = reference_paged_decode(q, k, v, table, seqlen)
    expected = _dense_oracle(q, k, v, table, seqlen, scale)

    assert got.shape == (B, H, D) and got.dtype == q.dtype
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)


def test_reference_honors_an_explicit_scale():
    q, k, v, table, seqlen = _cpu_inputs()
    scale = 0.375

    got = reference_paged_decode(q, k, v, table, seqlen, scale=scale)
    expected = _dense_oracle(q, k, v, table, seqlen, scale)

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)


def test_reference_handles_non_contiguous_and_duplicated_block_tables():
    """A block id may repeat, and the table's unused tail must never be read.

    Page 0 and page 2 of row 0 point at the same physical block, so the logical sequence
    contains those bytes twice; the last-page tail (``seqlen=50`` in a 16-column page) is
    dropped. Trailing table entries are ``-999`` on purpose: a kernel that walks the full
    table instead of ``ceil(seqlen / block_size)`` pages fails here.
    """
    block_size, D, H_kv, H = 16, 8, 2, 4
    q, k, v, _, _ = _build(2, H, H_kv, D, block_size, [50, 64], seed=7, n_blocks=6)
    table = torch.tensor(
        [
            [3, 0, 3, 5, -999, -999],
            [5, 1, 4, 0, -999, -999],
        ],
        dtype=torch.int32,
    )
    seqlen = torch.tensor([50, 64], dtype=torch.int32)

    got = reference_paged_decode(q, k, v, table, seqlen)
    expected = _dense_oracle(q, k, v, table, seqlen, 1.0 / math.sqrt(D))

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)


def test_reference_accepts_int64_seqlen_and_keeps_the_dtype():
    q, k, v, table, seqlen = _cpu_inputs()
    out32 = reference_paged_decode(q, k, v, table, seqlen)
    out64 = reference_paged_decode(q, k, v, table, seqlen.to(torch.int64))

    assert out64.dtype == q.dtype
    torch.testing.assert_close(out64, out32, rtol=0, atol=0)


def test_reference_does_not_mutate_inputs():
    q, k, v, table, seqlen = _cpu_inputs()
    snapshots = [t.clone() for t in (q, k, v, table, seqlen)]

    reference_paged_decode(q, k, v, table, seqlen)

    for tensor, before in zip((q, k, v, table, seqlen), snapshots):
        assert torch.equal(tensor, before)


def test_reference_accepts_a_non_contiguous_query():
    B, H, Hkv, D = 2, 4, 2, 8
    q, k, v, table, seqlen = _build(B, H, Hkv, D, 16, [17, 5], seed=11)
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    assert not q.is_contiguous()

    got = reference_paged_decode(q, k, v, table, seqlen)
    expected = _dense_oracle(q, k, v, table, seqlen, 1.0 / math.sqrt(D))

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)


# ------------------------------------------------------------------- oracle error surface


def test_reference_rejects_seqlen_beyond_capacity():
    q, k, v, table, _ = _cpu_inputs()
    capacity = table.shape[1] * k.shape[1]
    seqlen = torch.full((q.shape[0],), capacity + 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="capacity"):
        reference_paged_decode(q, k, v, table, seqlen)


def test_reference_rejects_negative_seqlen():
    q, k, v, table, _ = _cpu_inputs()
    seqlen = torch.tensor([-1, 3], dtype=torch.int32)

    with pytest.raises(ValueError, match=">= 0"):
        reference_paged_decode(q, k, v, table, seqlen)


def test_head_count_must_be_a_multiple_of_kv_heads():
    q, k, v, table, seqlen = _cpu_inputs(H=3, Hkv=2, lengths=(5, 5))

    with pytest.raises(ValueError, match="multiple of n_kv_head"):
        reference_paged_decode(q, k, v, table, seqlen)


def test_block_size_must_be_a_power_of_two():
    q, k, v, table, seqlen = _build(2, 4, 2, 8, 24, [10, 10], seed=1)

    with pytest.raises(ValueError, match="power of two"):
        reference_paged_decode(q, k, v, table, seqlen)


def test_cache_dtype_must_match_q():
    q, k, v, table, seqlen = _cpu_inputs()
    k = k.bfloat16()

    with pytest.raises(ValueError, match="share one dtype"):
        reference_paged_decode(q, k, v, table, seqlen)


# ------------------------------------------------------------------- kernel: CUDA only


@CUDA
@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "B,H,Hkv,D,block_size,lengths",
    [
        pytest.param(1, 2, 2, 64, 16, [1], id="b1-gqa1-d64-l1-bs16"),
        pytest.param(3, 8, 4, 128, 16, [15, 16, 17], id="b3-gqa2-d128-l15-17-bs16"),
        pytest.param(
            8, 8, 2, 64, 32, [1, 15, 16, 17, 100, 1, 15, 16], id="b8-gqa4-d64-bs32"
        ),
        pytest.param(1, 8, 1, 256, 16, [1664], id="b1-gqa8-d256-l1664-bs16"),
        pytest.param(3, 8, 1, 256, 32, [100, 1664, 1], id="b3-gqa8-d256-bs32"),
    ],
)
def test_paged_decode_matches_reference_on_cuda(B, H, Hkv, D, block_size, lengths, dtype):
    q, k, v, table, seqlen = _build(
        B, H, Hkv, D, block_size, lengths, device="cuda", dtype=dtype, seed=13
    )

    got = paged_decode(q, k, v, table, seqlen)
    expected = reference_paged_decode(q, k, v, table, seqlen)

    assert got.shape == (B, H, D) and got.dtype == dtype
    assert torch.isfinite(got).all(), "a finite input produced NaN/Inf"
    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    diff = (got.float() - expected.float()).abs().max().item()
    assert diff < tolerance, f"max|delta| {diff} exceeded {tolerance}"
    if dtype == torch.bfloat16:
        assert torch.equal(got.argmax(dim=-1), expected.argmax(dim=-1))


@CUDA
@GPU
def test_paged_decode_accepts_int64_seqlen_and_a_non_contiguous_query():
    q, k, v, table, seqlen = _build(
        2, 4, 2, 64, 16, [17, 15], device="cuda", seed=21
    )
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    assert not q.is_contiguous()

    got = paged_decode(q, k, v, table, seqlen.to(torch.int64))
    expected = reference_paged_decode(q.contiguous(), k, v, table, seqlen.to(torch.int64))

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)


@CUDA
@GPU
def test_paged_decode_does_not_mutate_its_inputs():
    q, k, v, table, seqlen = _build(
        3, 8, 4, 64, 16, [17, 15, 1], device="cuda", seed=17
    )
    snapshots = [t.clone() for t in (q, k, v, table, seqlen)]

    paged_decode(q, k, v, table, seqlen)

    for name, tensor, before in zip(
        ("q", "k_cache", "v_cache", "block_table", "seqlen"),
        (q, k, v, table, seqlen),
        snapshots,
    ):
        assert torch.equal(tensor, before), f"paged_decode mutated {name}"


@CUDA
@GPU
def test_zero_length_rows_return_zero_without_nan():
    q, k, v, table, seqlen = _build(
        3, 4, 2, 64, 16, [0, 16, 17], device="cuda", seed=23
    )

    out = paged_decode(q, k, v, table, seqlen)

    assert torch.isfinite(out).all()
    assert not out[0].any(), "a zero-length row must decode to exact zeros"
    torch.testing.assert_close(
        out[1:], reference_paged_decode(q, k, v, table, seqlen)[1:], rtol=0, atol=1e-4
    )


@CUDA
@GPU
def test_paged_decode_revalidates_after_an_in_place_seqlen_change():
    """The bound check is cached per tensor version, never across a mutation.

    In-place length updates are how the engine advances ``valid``; a stale cache here
    would silently decode the previous step's keys (and could read blocks that no longer
    belong to the request).
    """
    q, k, v, table, seqlen = _build(
        2, 4, 2, 64, 16, [17, 15], device="cuda", seed=43
    )
    paged_decode(q, k, v, table, seqlen)
    seqlen[0] = 1

    got = paged_decode(q, k, v, table, seqlen)
    expected = reference_paged_decode(q, k, v, table, seqlen)

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)


@CUDA
@GPU
def test_paged_decode_works_on_inference_tensors():
    """Inference tensors have no version counter; the wrapper must not assume one."""
    q, k, v, table, _ = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=47)
    with torch.inference_mode():
        seqlen = torch.tensor([17, 15], dtype=torch.int32, device="cuda")
        got = paged_decode(q, k, v, table, seqlen)
        expected = reference_paged_decode(q, k, v, table, seqlen)

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)


# ------------------------------------------------------------------- kernel error surface


def test_paged_decode_raises_when_triton_is_unavailable(monkeypatch):
    import model.triton_decode as module

    q, k, v, table, seqlen = _cpu_inputs()
    monkeypatch.setattr(module, "TRITON_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="Triton"):
        module.paged_decode(q, k, v, table, seqlen)


def test_paged_decode_rejects_cpu_tensors():
    q, k, v, table, seqlen = _cpu_inputs()

    with pytest.raises(RuntimeError, match="Triton|CUDA"):
        paged_decode(q, k, v, table, seqlen)


@CUDA
@GPU
def test_paged_decode_rejects_a_non_int32_block_table():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=29)

    with pytest.raises(ValueError, match="int32"):
        paged_decode(q, k, v, table.long(), seqlen)


@CUDA
@GPU
def test_paged_decode_rejects_a_non_cuda_block_table():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=31)

    with pytest.raises(RuntimeError, match="CUDA tensors"):
        paged_decode(q, k, v, table.cpu(), seqlen)


@CUDA
@GPU
def test_paged_decode_rejects_seqlen_beyond_the_table_capacity():
    q, k, v, table, _ = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=37)
    capacity = table.shape[1] * k.shape[1]
    seqlen = torch.full((q.shape[0],), capacity + 1, dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="capacity"):
        paged_decode(q, k, v, table, seqlen)


@CUDA
@GPU
def test_paged_decode_rejects_a_cache_with_the_wrong_head_dim():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=41)

    with pytest.raises(ValueError, match="head_dim"):
        paged_decode(
            q, k[:, :, :, :-1].contiguous(), v[:, :, :, :-1].contiguous(), table, seqlen
        )


# ------------------------------------------------------------ launch plan: CUDA only


@CUDA
@GPU
def test_decode_plan_matches_paged_decode_for_every_layer():
    """The cached launch metadata must not change a single output byte."""
    q, k, v, table, seqlen = _build(3, 8, 4, 64, 16, [17, 15, 32], device="cuda", seed=43)
    # A distinct pool per layer, like the engine's per-layer keys/values.
    gen = torch.Generator(device="cuda").manual_seed(44)
    k_layers = [
        torch.randn(k.shape, generator=gen, device="cuda", dtype=k.dtype) for _ in range(4)
    ]
    v_layers = [
        torch.randn(v.shape, generator=gen, device="cuda", dtype=v.dtype) for _ in range(4)
    ]

    plan = paged_decode_plan(q, k_layers, v_layers, table, seqlen)
    for layer in range(4):
        got = plan.run(layer, q)
        expected = paged_decode(q, k_layers[layer], v_layers[layer], table, seqlen)
        assert torch.equal(got, expected), f"layer {layer} differs from paged_decode"


@CUDA
@GPU
def test_decode_plan_rejects_int64_seqlen_instead_of_snapshotting_it():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=59)
    with pytest.raises(ValueError, match="int32 seqlen"):
        paged_decode_plan(q, [k], [v], table, seqlen.to(torch.int64))


@CUDA
@GPU
def test_decode_plan_rejects_non_contiguous_table_and_seqlen():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=61)
    with pytest.raises(ValueError, match="column-contiguous block table"):
        paged_decode_plan(q, [k], [v], table[:, ::2], seqlen)
    noncontiguous_seqlen = seqlen.repeat(2)[::2]
    assert noncontiguous_seqlen.stride(0) == 2
    with pytest.raises(ValueError, match="contiguous 1-D seqlen"):
        paged_decode_plan(q, [k], [v], table, noncontiguous_seqlen)


@CUDA
@GPU
def test_decode_plan_refuses_a_query_with_a_different_layout():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=47)
    plan = paged_decode_plan(q, [k], [v], table, seqlen)

    with pytest.raises(ValueError, match="does not match the validated layout"):
        plan.run(0, q.transpose(1, 2).contiguous().transpose(1, 2))  # non-contiguous
    with pytest.raises(ValueError, match="does not match the validated layout"):
        plan.run(0, q[:, :2, :].contiguous())  # different head count
    with pytest.raises(ValueError, match="does not match the validated layout"):
        plan.run(0, q.double())  # different dtype


@CUDA
@GPU
def test_decode_plan_rejects_an_unknown_layer_and_validates_every_pool():
    q, k, v, table, seqlen = _build(2, 4, 2, 64, 16, [17, 15], device="cuda", seed=53)
    plan = paged_decode_plan(q, [k, k], [v, v], table, seqlen)

    with pytest.raises(IndexError, match="outside the plan"):
        plan.run(2, q)

    bad_k, bad_v = k[:, :, :, :-1].contiguous(), v[:, :, :, :-1].contiguous()
    with pytest.raises(ValueError, match="head_dim"):
        paged_decode_plan(q, [k, bad_k], [v, bad_v], table, seqlen)


@CUDA
@GPU
def test_container_plan_tracks_in_place_seqlen_updates():
    """``build_mask`` updates ``seqlen`` in place; the plan must see the new value.

    This is the plan's invalidation contract: it caches *metadata*, never a snapshot of
    the seqlen values, so a step that appends a token changes the next ``attend``.
    """
    from model.paged import PagedKV, PagedPool

    torch.manual_seed(0)
    pool = PagedPool(1, 4, 16, 2, 32, dtype=torch.float32, device="cuda", n_head=4)
    kv = PagedKV(
        pool, [[0, 1, 2, 3]], [10], rope_len=64, n_head=4, decode_impl="triton"
    )
    q = torch.randn(1, 4, 1, 32, device="cuda")

    outputs = []
    for step in range(3):
        kv.build_mask(next_token=True)
        k = torch.randn(1, 2, 1, 32, device="cuda")
        v = torch.randn(1, 2, 1, 32, device="cuda")
        kv.write(0, k, v)
        outputs.append(kv.attend(0, q))
        # The plan's launch must equal the fully-revalidating entry point at this step.
        expected = paged_decode(
            q[:, :, 0, :].contiguous(),
            pool.keys[0],
            pool.values[0],
            kv._block_table_tensor,
            kv._seqlen,
        ).unsqueeze(2)
        assert torch.equal(outputs[-1], expected), f"step {step} diverged from paged_decode"
        kv.advance()

    assert not torch.equal(outputs[0], outputs[-1]), "attention ignored the new tokens"
