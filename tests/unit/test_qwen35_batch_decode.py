"""W7 Step 2: batched hybrid decode through a seat container == per-request decode.

``Qwen35Model.forward_decode_batch`` runs one decode token for every compact active row
against a decode container (here the CPU-capable seat arm, ``KvSlab``). The container
changes *where the bytes live* and *how the step is masked*, never the answer, so the
reference is the W6 single-sequence path: two prompts prefill through two one-slot
``PooledQwen35State`` objects (each with a growing ``KVcache``), their six full-attention
K/V layers are copied into one ``KvSlab(batch=2)`` with ``fill_from(row=i)``, and the
combined ``PooledQwen35State`` is decoded four steps against two dense ``Qwen35State``
references decoded one row at a time.

What this pins, and why each part is load-bearing:

* **Per-row positions.** The prompts have different lengths (5 and 9), so a shared scalar
  ``past_len`` would rotate row 0 at 9 or row 1 at 5 -- four positions wrong. The
  comparison is per row, and the negative control below forces row 1's cursor to row 0's
  length and requires the logits to move: the per-row comparison is not trivially
  satisfiable.
* **The GDN endpoints.** All 18 linear-attention layers get ``past_len=1`` (they read no
  positions, P-56) and read/write the pool's ``[0, B)`` view; a missing state raises
  before the first layer, so no half-advanced request can survive.
* **Failure guards.** Wrong ``T``, a wrong tokens batch, a container built for another
  batch, a non-container KV, and missing GDN state all raise.

Tolerance: fp32 CPU, same weights and same kernels, but the seat path attends over the
slab's fixed ``max_len`` columns with an additive mask while the reference attends over
``prompt + 1`` columns. That is a different summation width, not a different formula, so
the logits are compared at ``atol=1e-5`` (measured drift is ~1e-7, reported by the test
message if it ever grows) and greedy ids must match unless the two candidates are within
``2 * atol`` of each other (a genuine near-tie, D9-7 style). The tiny model is the existing
24-layer ``LLLF x 6`` config from ``test_qwen35_model.py``.
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import KVcache
from model.qwen35_model import Qwen35Model, Qwen35State
from model.slab import KvSlab
from serve.state_pool import LinearStatePool

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
#: Where the 6 full-attention layers sit; the compact index is the position here (D9-5).
FULL_POSITIONS = [3, 7, 11, 15, 19, 23]
#: Different lengths on purpose: decode starts at position 5 for row 0 and 9 for row 1.
PROMPTS = [
    [3, 5, 7, 11, 13],
    [2, 4, 6, 8, 10, 12, 14, 16, 18],
]
DECODE_STEPS = 4
#: prompt + 4 decode tokens + headroom, so the slab is never the thing under test.
SLAB_MAX_LEN = 24
#: fp32 logits from two attention widths (see the module docstring); measured ~1e-7.
ATOL = 1e-5


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model() -> Qwen35Model:
    torch.manual_seed(0)
    return Qwen35Model(_config()).eval()


def _prompt_tensor(prompt: list[int]) -> torch.Tensor:
    return torch.tensor([prompt], dtype=torch.long)


class _CompactFullLayers:
    """A 24-layer ``KVcache`` seen through the 6-layer compact full-attention mapping.

    ``KvSlab.fill_from(cache, layer, row=...)`` reads ``cache.get_kv(layer)``; the seat
    container's layer index is compact (0..5), the prefill cache's is the actual 24-layer
    one (3, 7, ...), so this adapter is the hand-off. It is test scaffolding, not a
    production path: Step 2b's engine owns the real hand-off.
    """

    def __init__(self, cache: KVcache):
        self._cache = cache

    def get_kv(self, layer: int):
        return self._cache.get_kv(FULL_POSITIONS[layer])


def _assert_row_matches(actual, expected, *, step: int, row: int) -> None:
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    drift = float((actual.detach() - expected.detach()).abs().max())
    torch.testing.assert_close(
        actual,
        expected,
        atol=ATOL,
        rtol=0,
        msg=f"step {step} row {row}: seat batch decode drifted {drift:.2e}",
    )
    got, want = int(actual.argmax()), int(expected.argmax())
    if got != want:
        top2 = expected.topk(2).values
        margin = float(top2[0] - top2[1])
        assert margin <= 2 * ATOL, (
            f"step {step} row {row}: greedy ids {got} != {want} with top1-top2 margin "
            f"{margin:.3e} (only a genuine near-tie may flip)"
        )


def _seat_batch(model: Qwen35Model):
    """Prefill two prompts into two pool slots, then present them as one slab batch.

    Returns ``(pool, slab, batch_state, references, tokens)``: the pool owns the 18 GDN
    states of both requests, ``references`` are independent dense ``Qwen35State`` objects
    prefilled with the same prompts, and ``tokens`` is each request's first greedy token.
    """
    pool = LinearStatePool(
        model.config, len(PROMPTS), device="cpu", dtype=torch.float32
    )
    caches: list[KVcache] = []
    first_logits: list[torch.Tensor] = []
    references: list[Qwen35State] = []

    for index, prompt in enumerate(PROMPTS):
        handle = pool.alloc(f"req{index}")
        pool.activate(handle)
        cache = KVcache(model.config.n_layer)
        state = pool.state(handle.slot, kv=cache)
        pooled_logits = model(_prompt_tensor(prompt), state, logits_to_keep=1)[0]

        reference = Qwen35State(model.config)
        reference_logits = model(_prompt_tensor(prompt), reference, logits_to_keep=1)[0]
        # Same weights, same kernels, batch 1 both sides: the pooled prefill is the
        # dense prefill, so the comparison below is about the decode path only.
        torch.testing.assert_close(
            pooled_logits, reference_logits, atol=0.0, rtol=0.0
        )
        caches.append(cache)
        first_logits.append(pooled_logits)
        references.append(reference)

    slab = KvSlab(
        len(FULL_POSITIONS),
        len(PROMPTS),
        model.config.n_kv_head,
        model.head_dim,
        SLAB_MAX_LEN,
        dtype=torch.float32,
        device="cpu",
    )
    for layer in range(len(FULL_POSITIONS)):
        for row, cache in enumerate(caches):
            slab.fill_from(_CompactFullLayers(cache), layer, row=row)

    batch_state = pool.state([0, 1], kv=slab)
    tokens = torch.tensor(
        [[int(logits.argmax())] for logits in first_logits], dtype=torch.long
    )
    return pool, slab, batch_state, references, tokens


# ------------------------------------------------------------ batched == separate


def test_batched_seat_decode_matches_per_request_decode():
    """Four batched steps: every row == its own dense decode, at its own position."""
    model = _model()
    pool, slab, batch_state, references, tokens = _seat_batch(model)

    assert [len(prompt) for prompt in PROMPTS] == [5, 9]
    assert slab.pos.tolist() == [5, 9], "rows must sit at different decode positions"
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

        assert slab.pos.tolist() == [5 + step + 1, 9 + step + 1]
        assert slab.valid.tolist() == [5 + step + 1, 9 + step + 1]
        assert batch_state.position == step + 1
        tokens = logits.argmax(dim=-1)

    pool.free("req0")
    pool.free("req1")
    pool.close()
    pool.assert_no_leaks()


def test_batch_decode_uses_each_rows_own_cursor():
    """Negative control: moving row 1's cursor to row 0's length must change its logits.

    Without this, "the rows matched" could also be true of an implementation that used
    one row's cursor (or the batch maximum) for every row on a model whose logits happened
    not to care. Row 1's prompt is 9 tokens; seating it at 5 truncates its context and
    rotates the new token at position 5 instead of 9, which moves fp32 logits far beyond
    the 1e-5 comparison band above.
    """
    model = _model()
    pool, slab, batch_state, references, tokens = _seat_batch(model)
    slab.activate(1, len(PROMPTS[0]))  # row 1 now claims row 0's length
    assert slab.pos.tolist() == [5, 5]
    logits, _ = model.forward_decode_batch(tokens, batch_state)

    reference_logits, _ = model(tokens[1:2], references[1], logits_to_keep=1)
    drift = float((logits[1:2].detach() - reference_logits.detach()).abs().max())
    assert drift > 1e-2, (
        "forcing row 1's cursor to row 0's length left its logits unchanged "
        f"(drift {drift:.2e}); the per-row position is not load-bearing"
    )

    pool.close()


# --------------------------------------------------------------------- guards


def test_decode_batch_rejects_bad_tokens():
    model = _model()
    pool, _, batch_state, _, tokens = _seat_batch(model)

    with pytest.raises(ValueError, match=r"decode-only"):
        model.forward_decode_batch(torch.zeros(2, 2, dtype=torch.long), batch_state)
    with pytest.raises(ValueError, match=r"state\.batch"):
        model.forward_decode_batch(tokens[:1], batch_state)
    with pytest.raises(ValueError, match=r"int64"):
        model.forward_decode_batch(tokens.float(), batch_state)
    with pytest.raises(ValueError, match=r"at least one row"):
        model.forward_decode_batch(torch.zeros(0, 1, dtype=torch.long), batch_state)
    with pytest.raises(TypeError, match=r"Tensor"):
        model.forward_decode_batch(tokens.tolist(), batch_state)
    pool.close()


def test_decode_batch_rejects_a_container_for_another_batch_or_geometry():
    model = _model()
    pool, _, _, _, tokens = _seat_batch(model)
    rows = [0, 1]

    wrong_batch = KvSlab(
        len(FULL_POSITIONS), 3, model.config.n_kv_head, model.head_dim, SLAB_MAX_LEN
    )
    state = pool.state(rows, kv=wrong_batch)
    with pytest.raises(ValueError, match=r"container has batch=3"):
        model.forward_decode_batch(tokens, state)

    wrong_layers = KvSlab(
        1, len(PROMPTS), model.config.n_kv_head, model.head_dim, SLAB_MAX_LEN
    )
    state = pool.state(rows, kv=wrong_layers)
    with pytest.raises(ValueError, match=r"1 layers"):
        model.forward_decode_batch(tokens, state)

    wrong_heads = KvSlab(
        len(FULL_POSITIONS), len(PROMPTS), 1, model.head_dim, SLAB_MAX_LEN, n_head=4
    )
    state = pool.state(rows, kv=wrong_heads)
    with pytest.raises(ValueError, match=r"n_kv_head"):
        model.forward_decode_batch(tokens, state)

    pool.close()


def test_decode_batch_needs_prefilled_gdn_state_and_a_decode_container():
    model = _model()
    pool = LinearStatePool(model.config, 1, device="cpu", dtype=torch.float32)
    handle = pool.alloc("fresh")
    tokens = torch.tensor([[1]], dtype=torch.long)

    # Fresh rows: the container is well formed, but no GDN layer has state yet.
    slab = KvSlab(
        len(FULL_POSITIONS), 1, model.config.n_kv_head, model.head_dim, SLAB_MAX_LEN
    )
    with pytest.raises(ValueError, match=r"linear states for layer 0"):
        model.forward_decode_batch(tokens, pool.state(handle.slot, kv=slab))

    # A growing KVcache is the single-sequence path, not a decode container.
    with pytest.raises(ValueError, match=r"decode container"):
        model.forward_decode_batch(tokens, pool.state(handle.slot))

    # The single-sequence forward refuses a decode container: it has one scalar
    # past_len and cannot express per-row positions.
    with pytest.raises(ValueError, match=r"forward_decode_batch"):
        model(tokens, pool.state(handle.slot, kv=slab))

    pool.close()


def test_partially_written_gdn_state_is_reported_before_the_first_layer():
    """A state with only some layers written must fail loudly, not decode from zeros."""
    model = _model()
    pool = LinearStatePool(model.config, 1, device="cpu", dtype=torch.float32)
    handle = pool.alloc("partial")
    slab = KvSlab(
        len(FULL_POSITIONS), 1, model.config.n_kv_head, model.head_dim, SLAB_MAX_LEN
    )
    state = pool.state(handle.slot, kv=slab)
    # Layer 0 has state; layer 1 (the next linear layer) does not. The guard must name
    # layer 1 before layer 0 has advanced the pool row.
    state.set_conv_state(0, torch.zeros(1, 24, 3))
    state.set_recurrent_state(0, torch.zeros(1, 2, 4, 4))

    with pytest.raises(ValueError, match=r"linear states for layer 1"):
        model.forward_decode_batch(torch.tensor([[1]], dtype=torch.long), state)

    pool.close()
