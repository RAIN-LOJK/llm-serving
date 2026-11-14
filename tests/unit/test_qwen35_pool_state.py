"""W7 Step 1: ``PooledQwen35State`` is numerically the W6 single-sequence state.

The pool changes *storage and lifetime*, never the math. This file proves that on the
real 24-layer ``LLLF x 6`` stack built by the existing tiny helper: one request prefills
through the normal ``Qwen35State`` and through a one-slot ``PooledQwen35State`` backed by
a ``LinearStatePool`` and an injected ``KVcache``, and every logit row, every conv state,
every recurrent state and every K/V tensor must agree exactly (fp32 CPU, same tensors,
same kernels). Then four decode steps repeat the comparison, which is the path where a
view/setter bug would show up as drift.

Two further checks cover the properties a single-request comparison cannot see:

* two requests in two slots hold independent state and a ``[0, B)`` batch view reads
  exactly the rows, with bytes equal to the sum of its row tensors;
* a reserved slot that was partially written and then released (the §3.3 rollback)
  cannot leak into the next request's prefill.

Coverage is the full tiny model, not a layer subset: hidden 32, 4/2 heads, conv kernel
4 (so ``k - 1 = 3``), all 24 layers, 6 full-attention layers. Seq len 7 prefill + 4
decode steps keeps the file fast; nothing is weakened to get there.
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import KVcache
from model.qwen35_model import Qwen35Model, Qwen35State
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
SEQ_LEN = 7
DECODE_STEPS = 4
CONV_DIM = 24
TINY_SLOT_BYTES = 18 * (CONV_DIM * 3 * 4 + 2 * 4 * 4 * 4)


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model() -> Qwen35Model:
    return Qwen35Model(_config()).eval()


def _ids(length: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, length))


def _pool(capacity: int, config: MiniLLMConfig | None = None) -> LinearStatePool:
    return LinearStatePool(
        _config() if config is None else config,
        capacity,
        device="cpu",
        dtype=torch.float32,
    )


def _assert_states_equal(actual, expected) -> None:
    """Every conv/recurrent tensor and every K/V tensor, exactly (same kernels)."""
    assert actual.config is expected.config
    for idx, kind in enumerate(actual.config.layer_types):
        if kind == "linear_attention":
            conv_actual = actual.conv_state(idx)
            conv_expected = expected.conv_state(idx)
            assert conv_actual is not None and conv_expected is not None
            torch.testing.assert_close(
                conv_actual,
                conv_expected,
                atol=0.0,
                rtol=0.0,
                msg=f"conv layer {idx}",
            )
            recurrent_actual = actual.recurrent_state(idx)
            recurrent_expected = expected.recurrent_state(idx)
            assert recurrent_actual is not None and recurrent_expected is not None
            torch.testing.assert_close(
                recurrent_actual,
                recurrent_expected,
                atol=0.0,
                rtol=0.0,
                msg=f"recurrent layer {idx}",
            )
        else:
            k_actual, v_actual = actual.get_kv(idx)
            k_expected, v_expected = expected.get_kv(idx)
            assert k_actual is not None and k_expected is not None
            torch.testing.assert_close(k_actual, k_expected, atol=0.0, rtol=0.0)
            torch.testing.assert_close(v_actual, v_expected, atol=0.0, rtol=0.0)


def _assert_logits_equal(actual, expected) -> None:
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def _row_tensor_bytes(state) -> int:
    """Bytes of the state's getters, i.e. the sum of its row tensors."""
    total = 0
    for idx, kind in enumerate(state.config.layer_types):
        if kind != "linear_attention":
            continue
        for tensor in (state.conv_state(idx), state.recurrent_state(idx)):
            assert tensor is not None
            total += tensor.numel() * tensor.element_size()
    return total


# ------------------------------------------------------- prefill + decode parity


def test_prefill_and_stepwise_decode_match_the_single_sequence_state():
    torch.manual_seed(0)
    model = _model()
    config = model.config
    ids = _ids(SEQ_LEN)

    reference = Qwen35State(config)
    reference_logits = model(ids, reference, logits_to_keep=None)[0]

    pool = _pool(1, config)
    handle = pool.alloc("request")
    kv = KVcache(config.n_layer)
    pooled = pool.state(handle.slot, kv=kv)
    assert pooled.kv is kv
    assert pooled.batch == 1 and pooled.rows == (handle.slot,)
    # Fresh state: the prefill must receive "no state" exactly like the base class, even
    # though the pool rows physically exist.
    assert pooled.conv_state(0) is None

    pooled_logits = model(ids, pooled, logits_to_keep=None)[0]
    _assert_logits_equal(pooled_logits, reference_logits)
    assert pooled.position == reference.position == SEQ_LEN
    _assert_states_equal(pooled, reference)

    # The pooled state's bytes are its real row tensors, not the analytic 18.633 MiB.
    viewed_bytes = _row_tensor_bytes(pooled)
    assert pooled.linear_state_bytes() == viewed_bytes == reference.linear_state_bytes()
    pooled_bytes = pooled.linear_state_bytes()
    assert pooled_bytes == TINY_SLOT_BYTES
    assert pool.stats().used_bytes == pooled_bytes
    assert pooled.nbytes() == pooled_bytes + pooled.kv_bytes()
    assert pooled.kv_bytes() == reference.kv_bytes()

    tokens = reference_logits[:, -1:, :].argmax(dim=-1)
    for step in range(DECODE_STEPS):
        reference_step, _ = model(tokens, reference, logits_to_keep=1)
        pooled_step, _ = model(tokens, pooled, logits_to_keep=1)
        _assert_logits_equal(pooled_step, reference_step)
        _assert_states_equal(pooled, reference)
        assert pooled.position == reference.position == SEQ_LEN + step + 1
        tokens = reference_step.argmax(dim=-1)

    pool.free(handle)
    pool.close()
    pool.assert_no_leaks()


def test_released_reserved_slot_does_not_leak_into_the_next_prefill():
    """A partial prefill that failed before activation is discarded by the pool."""
    torch.manual_seed(0)
    model = _model()
    config = model.config
    ids = _ids(SEQ_LEN)
    pool = _pool(1, config)

    failed = pool.alloc("failed")
    failed_state = pool.state(failed.slot)
    for layer in pool.linear_layers:
        failed_state.set_conv_state(layer, torch.full((1, CONV_DIM, 3), 5.0))
        failed_state.set_recurrent_state(layer, torch.full((1, 2, 4, 4), 5.0))
    pool.free(failed)  # reserved -> released, the rollback path

    replacement = pool.alloc("replacement")
    pooled = pool.state(replacement.slot, kv=KVcache(config.n_layer))
    reference = Qwen35State(config)
    pooled_logits = model(ids, pooled, logits_to_keep=None)[0]
    reference_logits = model(ids, reference, logits_to_keep=None)[0]

    _assert_logits_equal(pooled_logits, reference_logits)
    _assert_states_equal(pooled, reference)
    assert pooled.position == reference.position == SEQ_LEN


# ----------------------------------------------------- two requests / batch view


def test_two_requests_in_two_slots_hold_independent_state():
    torch.manual_seed(0)
    model = _model()
    config = model.config
    pool = _pool(2, config)
    down = pool.alloc("down")
    across = pool.alloc("across")
    down_state = pool.state(down.slot, kv=KVcache(config.n_layer))
    across_state = pool.state(across.slot, kv=KVcache(config.n_layer))

    down_ids = _ids(SEQ_LEN)
    across_ids = _ids(SEQ_LEN - 2)
    down_reference = Qwen35State(config)
    across_reference = Qwen35State(config)

    down_logits = model(down_ids, down_reference, logits_to_keep=None)[0]
    across_logits = model(across_ids, across_reference, logits_to_keep=None)[0]
    pooled_down = model(down_ids, down_state, logits_to_keep=None)[0]
    pooled_across = model(across_ids, across_state, logits_to_keep=None)[0]

    _assert_logits_equal(pooled_down, down_logits)
    _assert_logits_equal(pooled_across, across_logits)
    _assert_states_equal(down_state, down_reference)
    _assert_states_equal(across_state, across_reference)

    # The packed [0, B) view reads row 0 as the first request and row 1 as the second,
    # with no cross-talk at any conv/recurrent layer.
    batch = pool.state([0, 1])
    assert batch.batch == 2
    for layer in pool.linear_layers:
        conv = batch.conv_state(layer)
        recurrent = batch.recurrent_state(layer)
        assert conv.shape == (2, CONV_DIM, 3) and conv.dtype == torch.float32
        assert recurrent.shape == (2, 2, 4, 4) and recurrent.dtype == torch.float32
        torch.testing.assert_close(
            conv[0], down_state.conv_state(layer)[0], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            conv[1], across_state.conv_state(layer)[0], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            recurrent[0], down_state.recurrent_state(layer)[0], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            recurrent[1], across_state.recurrent_state(layer)[0], atol=0.0, rtol=0.0
        )

    # Bytes are the sum of the batch's real row tensors, and both active slots are full.
    assert batch.linear_state_bytes() == _row_tensor_bytes(batch)
    assert batch.linear_state_bytes() == 2 * TINY_SLOT_BYTES
    assert pool.stats().used_bytes == batch.linear_state_bytes()

    # Releasing the lower slot moves the surviving request's rows; a state object built
    # before the move is invalid, and a rebuilt one reads the preserved state.
    pool.free(down)
    assert across.slot == 0
    with pytest.raises(RuntimeError, match=r"not the row this state was built on"):
        across_state.conv_state(0)
    rebuilt = pool.state([0])
    for layer in pool.linear_layers:
        torch.testing.assert_close(
            rebuilt.conv_state(layer)[0],
            across_reference.conv_state(layer)[0],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            rebuilt.recurrent_state(layer)[0],
            across_reference.recurrent_state(layer)[0],
            atol=0.0,
            rtol=0.0,
        )
    assert pool.stats().used_bytes == TINY_SLOT_BYTES

    pool.free(across)
    pool.close()
    pool.assert_no_leaks()
