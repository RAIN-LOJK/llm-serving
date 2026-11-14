"""W7 Step 2: batched hybrid decode through the paged container (CUDA + Triton only).

This is the production arm: the 6 full-attention layers address a ``PagedPool`` of 6
layers through their **compact** index (D9-5), each row decoding through the Triton
block-table kernel (``PagedKV.attend``), while the 18 GDN layers read/write the
``LinearStatePool``'s compact ``[0, B)`` rows. The reference is the W6 single-sequence
path: two requests prefill into a growing ``KVcache`` each, their full-attention K/V is
written into the pool's physical blocks through their own block tables
(``PagedPool.write_prefill``), and every batched decode step must equal two dense
``Qwen35State`` decodes.

Prompts are 17 and 21 tokens with ``block_size=16``, so both rows span **two** physical
blocks: a container that only ever indexed block 0 would fail here. Decode positions
start at 17 and 21, so the per-row RoPE gather is exercised with unequal lengths.

Paged decode needs CUDA and Triton; there is deliberately no gather fallback (W4.5
deleted the control arm), so the module skips with an explicit reason elsewhere. The
tiny model's ``head_dim=8`` is a power of two, which is the kernel's only shape
requirement (``model/triton_decode.py::_validate``), so no geometry is weakened and no
real-0.8B substitution is needed.

Tolerance: fp32 both sides, but the Triton kernel's page walk and the dense
``hand_baddbmm`` accumulate over different key counts, so logits are compared at
``atol=1e-5`` (measured worst ~9e-7) and greedy ids must match unless the two candidates
are within ``2 * atol`` (a genuine near-tie).
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import KVcache
from model.paged import PagedKV, PagedPool
from model.qwen35_model import Qwen35Model, Qwen35State
from model.triton_decode import TRITON_AVAILABLE
from serve.state_pool import LinearStatePool

CUDA_READY = TRITON_AVAILABLE and torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason=(
        "paged decode is the CUDA Triton block-table kernel (model/paged.py); it has no "
        "CPU or gather fallback"
    ),
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

VOCAB = 64
FULL_POSITIONS = [3, 7, 11, 15, 19, 23]
BLOCK_SIZE = 16
#: 17 and 21 tokens: two physical blocks each at block_size=16, different decode starts.
PROMPTS = [list(range(3, 20)), list(range(2, 23))]
#: One block table per request; the two requests never share a physical block.
BLOCK_TABLES = [[0, 1], [2, 3]]
DECODE_STEPS = 4
ROPE_LEN = 64
#: fp32 logits from two different attention kernels; measured worst ~9e-7.
ATOL = 1e-5


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model() -> Qwen35Model:
    torch.manual_seed(0)
    return Qwen35Model(_config()).to("cuda", dtype=torch.float32).eval()


def _assert_row_matches(actual, expected, *, step: int, row: int) -> None:
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    drift = float((actual.detach() - expected.detach()).abs().max())
    torch.testing.assert_close(
        actual,
        expected,
        atol=ATOL,
        rtol=0,
        msg=f"step {step} row {row}: paged batch decode drifted {drift:.2e}",
    )
    got, want = int(actual.argmax()), int(expected.argmax())
    if got != want:
        top2 = expected.topk(2).values
        margin = float(top2[0] - top2[1])
        assert margin <= 2 * ATOL, (
            f"step {step} row {row}: greedy ids {got} != {want} with top1-top2 margin "
            f"{margin:.3e} (only a genuine near-tie may flip)"
        )


def _paged_batch(model: Qwen35Model):
    """Prefill two prompts, page their 6 full layers, and present one PagedKV batch.

    Returns ``(linear_pool, paged_pool, paged, batch_state, references, tokens)``.
    """
    linear_pool = LinearStatePool(
        model.config, len(PROMPTS), device="cuda", dtype=torch.float32
    )
    paged_pool = PagedPool(
        len(FULL_POSITIONS),
        len(BLOCK_TABLES) * len(BLOCK_TABLES[0]),
        BLOCK_SIZE,
        model.config.n_kv_head,
        model.head_dim,
        dtype=torch.float32,
        device="cuda",
        n_head=model.config.n_head,
    )

    references: list[Qwen35State] = []
    first_logits: list[torch.Tensor] = []
    for index, prompt in enumerate(PROMPTS):
        handle = linear_pool.alloc(f"req{index}")
        linear_pool.activate(handle)
        cache = KVcache(model.config.n_layer)
        state = linear_pool.state(handle.slot, kv=cache)
        ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
        pooled_logits = model(ids, state, logits_to_keep=1)[0]

        reference = Qwen35State(model.config)
        reference_logits = model(ids, reference, logits_to_keep=1)[0]
        # Same weights, same kernels, batch 1 both sides.
        torch.testing.assert_close(pooled_logits, reference_logits, atol=0.0, rtol=0.0)

        for compact, actual in enumerate(FULL_POSITIONS):
            k, v = cache.get_kv(actual)  # (1, n_kv_head, length, head_dim)
            paged_pool.write_prefill(
                compact,
                BLOCK_TABLES[index],
                0,
                k[0].transpose(0, 1),
                v[0].transpose(0, 1),
            )

        references.append(reference)
        first_logits.append(pooled_logits)

    paged = PagedKV(
        paged_pool,
        BLOCK_TABLES,
        [len(prompt) for prompt in PROMPTS],
        rope_len=ROPE_LEN,
        seats=[0, 1],
        n_head=model.config.n_head,
    )
    batch_state = linear_pool.state([0, 1], kv=paged)
    tokens = torch.tensor(
        [[int(logits.argmax())] for logits in first_logits],
        dtype=torch.long,
        device="cuda",
    )
    return linear_pool, paged_pool, paged, batch_state, references, tokens


# ------------------------------------------------------------ batched == separate


def test_paged_batch_decode_matches_per_request_decode():
    """Four steps of one block-table batch == four steps of each dense request."""
    model = _model()
    linear_pool, _, paged, batch_state, references, tokens = _paged_batch(model)

    assert [len(prompt) for prompt in PROMPTS] == [17, 21]
    assert paged.pos.tolist() == [17, 21], "rows must sit at different decode positions"
    assert paged.block_table_decode is True, "the paged container must take the kernel path"
    assert batch_state.batch == 2 and batch_state.rows == (0, 1)

    for step in range(DECODE_STEPS):
        logits, loss = model.forward_decode_batch(tokens, batch_state)
        assert loss is None
        assert tuple(logits.shape) == (len(PROMPTS), 1, VOCAB)
        assert torch.isfinite(logits).all()

        for row, reference in enumerate(references):
            reference_logits, _ = model(tokens[row : row + 1], reference, logits_to_keep=1)
            _assert_row_matches(
                logits[row : row + 1], reference_logits, step=step, row=row
            )

        assert paged.pos.tolist() == [17 + step + 1, 21 + step + 1]
        assert paged.valid.tolist() == [17 + step + 1, 21 + step + 1]
        assert batch_state.position == step + 1
        tokens = logits.argmax(dim=-1)

    linear_pool.free("req0")
    linear_pool.free("req1")
    linear_pool.close()
    linear_pool.assert_no_leaks()


def test_paged_batch_decode_launches_the_kernel_once_per_compact_layer(monkeypatch):
    """D9-5 at the model level: the 6 full layers call ``attend`` with layers 0..5.

    A model that passed the actual 24-layer ``idx`` would index a 6-layer pool at 23 and
    either raise or write into another layer's slot; asserting the launch order pins the
    compact mapping rather than merely "some number of launches happened".
    """
    from model.triton_decode import PagedDecodePlan

    model = _model()
    linear_pool, _, _, batch_state, _, tokens = _paged_batch(model)

    launched: list[int] = []
    original = PagedDecodePlan.run

    def counting_run(self, layer, q):
        launched.append(int(layer))
        return original(self, layer, q)

    monkeypatch.setattr(PagedDecodePlan, "run", counting_run)
    model.forward_decode_batch(tokens, batch_state)

    assert launched == list(range(len(FULL_POSITIONS))), (
        f"expected one launch per compact layer [0..5], got {launched}"
    )
    linear_pool.close()
