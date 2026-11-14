"""W6 Step 4: the 24-layer ``LLLF x 6`` hybrid model and its per-request state.

The W5/W6 layer tests already pin the two block kinds in isolation, and the operator
tests pin the delta rule against two references. What this file locks is the *assembly*,
where a plausible mistake stays invisible in every layer test:

* the real ``LLLF x 6`` schedule -- 18 GDN blocks, 6 full-attention blocks at positions
  3/7/11/.../23 -- not "some linear layers and some attention layers";
* one ``lm_head`` Parameter shared with the embedding (a copy costs ~1 GB fp32), and
  ``logits_to_keep`` slicing *before* that projection (P-13);
* chunked prefill vs stepwise decode over the whole stack (E4 / A2), including the
  1-token-prompt-then-decode case that a ``T == 1`` dispatch would get wrong (P-56);
* the state's lifecycle and exact byte accounting (W6 state bytes, D8), which is W7's input;
* ``cull`` shrinking rows without rewinding the cursor;
* every unsupported entry point raising: batch > 1, ``attention_mask``, a plain
  ``KVcache``, cross-call chunked prefill, non-positive ``logits_to_keep``;
* the state setters' shape/dtype guards (fp32 recurrent state, ``k - 1`` conv columns).

Tiny CPU config: hidden 32, 4 query / 2 KV heads of width 8, conv kernel 4. The 24 layers
are what makes the schedule real; each one is a handful of tiny matmuls, so the file stays
well inside the fast gate's budget (the tests are dominated by import time).
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import KVcache
from model.qwen35 import Qwen35FullAttentionBlock
from model.qwen35_gdn import Qwen35GDNBlock
from model.qwen35_model import Qwen35Model, Qwen35State

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
# conv_dim = 2 * key_dim + value_dim = 2*(2*4) + (2*4) = 24; k - 1 = 3.
CONV_DIM = 24
# Where the 6 full-attention layers sit in the LLLF x 6 schedule.
FULL_POSITIONS = [3, 7, 11, 15, 19, 23]
# configs/thresholds.toml [W6.gdn].decode_prefill_atol.
DECODE_PREFILL_ATOL = 1e-4


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model() -> Qwen35Model:
    return Qwen35Model(_config()).eval()


def _ids(length: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, length))


# ------------------------------------------------------------------- the schedule


def test_lllf_schedule_builds_18_linear_and_6_full_blocks():
    """The 0.8B schedule is real here: 24 blocks, full attention every 4th position."""
    model = _model()
    assert len(model.layers) == 24
    assert model.config.layer_types == TINY["layer_types"]

    kinds = [
        "linear_attention"
        if isinstance(layer, Qwen35GDNBlock)
        else "full_attention"
        if isinstance(layer, Qwen35FullAttentionBlock)
        else "unknown"
        for layer in model.layers
    ]
    assert kinds.count("linear_attention") == 18
    assert kinds.count("full_attention") == 6
    assert "unknown" not in kinds
    assert [i for i, kind in enumerate(kinds) if kind == "full_attention"] == FULL_POSITIONS
    assert model.head_dim == 8
    assert model.rotary_dim == 2  # int(8 * 0.25), partial RoPE
    assert model.rope_theta == 1e4


# ------------------------------------------------------------- tied head / slicing


def test_lm_head_is_tied_and_logits_to_keep_slices_before_it():
    model = _model()
    assert model.lm_head.weight is model.token_embedding.weight

    ids = _ids(SEQ_LEN)
    kept, kept_loss = model(ids, logits_to_keep=1)
    full, full_loss = model(ids, logits_to_keep=None)

    assert kept_loss is None and full_loss is None
    assert kept.shape == (1, 1, VOCAB)
    assert full.shape == (1, SEQ_LEN, VOCAB)
    # The last row of the full projection is the kept row: slicing happens after the norm
    # but before lm_head (P-13). Not bit-identical -- a 1-row GEMM rounds differently from
    # a 7-row one -- so this bounds the blocking noise rather than demanding equality.
    torch.testing.assert_close(kept, full[:, -1:], atol=1e-6, rtol=0)


# -------------------------------------------------------- prefill vs decode (E4/A2)


def test_prefill_matches_stepwise_decode():
    """One 7-token prefill == 7 single-token forwards against the same state."""
    model = _model()
    ids = _ids(SEQ_LEN)

    state_prefill = Qwen35State(model.config)
    prefill, _ = model(ids, state_prefill, logits_to_keep=None)

    state_decode = Qwen35State(model.config)
    steps = []
    for position in range(SEQ_LEN):
        step, _ = model(ids[:, position : position + 1], state_decode, logits_to_keep=1)
        steps.append(step)
    decoded = torch.cat(steps, dim=1)

    torch.testing.assert_close(
        decoded, prefill, atol=DECODE_PREFILL_ATOL, rtol=0
    )
    assert state_prefill.position == SEQ_LEN
    assert state_decode.position == SEQ_LEN


def test_one_token_prompt_then_decode_matches_two_token_prefill():
    """P-56 at the model level: a 1-token prompt prefills (past_len == 0), then decodes."""
    model = _model()
    ids = _ids(2)

    two_token, _ = model(ids, logits_to_keep=None)

    state = Qwen35State(model.config)
    first, _ = model(ids[:, :1], state, logits_to_keep=None)
    second, _ = model(ids[:, 1:2], state, logits_to_keep=1)

    assert state.position == 2
    torch.testing.assert_close(
        torch.cat([first, second], dim=1), two_token, atol=DECODE_PREFILL_ATOL, rtol=0
    )


# ------------------------------------------------------------- state and its bytes


def test_state_lifecycle_and_byte_accounting():
    model = _model()
    state = Qwen35State(model.config)
    assert state.position == 0
    assert state.past_len() == 0

    model(_ids(SEQ_LEN), state)

    assert state.position == SEQ_LEN
    assert state.past_len() == SEQ_LEN
    assert state.past_len(FULL_POSITIONS[0]) == SEQ_LEN

    for idx, kind in enumerate(TINY["layer_types"]):
        if kind == "linear_attention":
            conv = state.conv_state(idx)
            assert conv.shape == (1, CONV_DIM, 3)
            assert conv.dtype == torch.float32
            recurrent = state.recurrent_state(idx)
            assert recurrent.shape == (1, 2, 4, 4)
            assert recurrent.dtype == torch.float32
            assert state.get_kv(idx) == (None, None)
        else:
            k, v = state.get_kv(idx)
            assert k.shape == (1, 2, SEQ_LEN, 8)
            assert v.shape == (1, 2, SEQ_LEN, 8)
            assert state.conv_state(idx) is None
            assert state.recurrent_state(idx) is None

    # W6 state bytes (D8), exact integers: conv 24*3*4 B + recurrent 2*4*4*4 B per linear layer;
    # K+V 2*2*8*7*4 B per full layer.
    assert state.linear_state_bytes() == 18 * (24 * 3 * 4 + 2 * 4 * 4 * 4)
    assert state.kv_bytes() == 6 * (2 * 2 * 8 * SEQ_LEN * 4)
    assert state.nbytes() == state.linear_state_bytes() + state.kv_bytes()


def test_cull_keeps_rows_and_cursor():
    cfg = _config()
    state = Qwen35State(cfg, batch=2)

    conv = torch.arange(2 * CONV_DIM * 3, dtype=torch.float32).reshape(2, CONV_DIM, 3)
    recurrent = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
    for idx, kind in enumerate(cfg.layer_types):
        if kind == "linear_attention":
            state.set_conv_state(idx, conv.clone())
            state.set_recurrent_state(idx, recurrent.clone())
        else:
            state.set_kv(
                idx,
                (torch.arange(2 * 2 * 7 * 8, dtype=torch.float32).reshape(2, 2, 7, 8),
                 torch.ones(2, 2, 7, 8)),
            )
    state.advance(SEQ_LEN)
    before = state.nbytes()
    assert before == 2 * (18 * (24 * 3 * 4 + 2 * 4 * 4 * 4) + 6 * (2 * 2 * 8 * 7 * 4))

    state.cull(torch.tensor([1]))

    assert state.batch == 1
    assert state.position == SEQ_LEN  # culling rows does not rewind the cursor
    assert state.past_len() == SEQ_LEN
    assert state.conv_state(0).shape == (1, CONV_DIM, 3)
    assert state.recurrent_state(0).shape == (1, 2, 4, 4)
    k, v = state.get_kv(FULL_POSITIONS[0])
    assert k.shape == (1, 2, SEQ_LEN, 8) and v.shape == (1, 2, SEQ_LEN, 8)
    # Row 1 is what survived, not row 0.
    torch.testing.assert_close(state.conv_state(0), conv[1:2], atol=0, rtol=0)
    torch.testing.assert_close(
        state.recurrent_state(0), recurrent[1:2], atol=0, rtol=0
    )
    assert state.nbytes() == before // 2
    assert state.linear_state_bytes() == before // 2 - state.kv_bytes()


# ------------------------------------------------------------------- rejections


def test_batch_greater_than_one_is_not_implemented():
    model = _model()
    with pytest.raises(NotImplementedError, match=r"single sequence"):
        model(torch.zeros(2, 3, dtype=torch.long))


def test_attention_mask_is_not_implemented():
    model = _model()
    with pytest.raises(NotImplementedError, match=r"attention_mask"):
        model(torch.zeros(1, 3, dtype=torch.long), attention_mask=torch.ones(1, 3))


def test_plain_kvcache_is_not_a_qwen35_state():
    model = _model()
    with pytest.raises(TypeError, match=r"Qwen35State"):
        model(torch.zeros(1, 3, dtype=torch.long), state=KVcache(24))


def test_cross_call_chunked_prefill_is_not_implemented():
    """past_len > 0 with T > 1 is W7, even though every single layer could in principle."""
    model = _model()
    ids = _ids(SEQ_LEN)
    state = Qwen35State(model.config)
    model(ids[:, :1], state)

    with pytest.raises(NotImplementedError, match=r"W7"):
        model(ids[:, 1:3], state)


@pytest.mark.parametrize("bad", [0, -1, -SEQ_LEN])
def test_non_positive_logits_to_keep_is_refused(bad):
    model = _model()
    with pytest.raises(ValueError, match=r"logits_to_keep"):
        model(torch.zeros(1, 3, dtype=torch.long), logits_to_keep=bad)


# --------------------------------------------------------------- state guards


def test_conv_state_is_linear_attention_only():
    state = Qwen35State(_config())
    with pytest.raises(ValueError, match=r"full_attention"):
        state.set_conv_state(FULL_POSITIONS[0], torch.zeros(1, CONV_DIM, 3))


def test_conv_state_shape_is_checked():
    state = Qwen35State(_config())
    with pytest.raises(ValueError, match=r"\(1, 24, 3\)"):
        state.set_conv_state(0, torch.zeros(1, CONV_DIM, 4))


def test_recurrent_state_must_be_float32():
    """A bf16 recurrent state would silently change the delta rule's dtype (HF keeps fp32)."""
    state = Qwen35State(_config())
    with pytest.raises(ValueError, match=r"float32"):
        state.set_recurrent_state(0, torch.zeros(1, 2, 4, 4, dtype=torch.bfloat16))


@pytest.mark.parametrize("bad", [-1, True])
def test_advance_rejects_negative_and_bool(bad):
    state = Qwen35State(_config())
    with pytest.raises((TypeError, ValueError), match=r"n must"):
        state.advance(bad)
