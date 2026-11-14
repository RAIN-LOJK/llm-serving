"""W7 Step 3: packed group prefill for the hybrid Qwen3.5 decoder.

One ``Qwen35Model.forward_packed`` call prefills a whole FIFO group: the 6 full-attention
layers run **once** over a request-major packed token axis through ``PackedPrefill``
(varlen attention, K/V written through one-shot sinks), while the 18 GDN layers loop over
the group's segments, each request prefilling from empty state (``past_len=0``) and writing
its conv/recurrent endpoint into its own pool slot. The tiny 24-layer ``LLLF x 6`` model is
the one from ``test_qwen35_model.py``; the seat arm is the CPU-capable container.

What each part pins, and why it is load-bearing:

* **Packed == dense prefill.** Every row's last-token logits must equal the same prompt
  through ``Qwen35Model.forward`` with a dense ``Qwen35State`` (atol 1e-5; the measured
  drift is ~6e-8, from SDPA-varlen passing an explicit ``scale=`` where the dense backend
  leaves it to the library). The 18 GDN endpoints and the 6 layers of slab K/V are
  compared too: a segment loop that chained requests, used one shared state or wrote the
  wrong seat row would show up here even if the logits happened to look fine.
* **Request-major order and last-token selection.** The group is built request-major
  (``cu_seqlens`` + request-local positions), and only each request's final token reaches
  ``lm_head`` (P-13). The test reads the container the engine actually built and also
  checks that the *other* token positions would give different logits, so a wrong
  ``last_token_index`` could not pass silently.
* **No cross-request tokens.** Changing request 0's prompt leaves request 1's row
  bit-identical: ``cu_seqlens`` is a real boundary, not a mask over a shared axis.
* **Batched decode after packed prefill.** Two engine decode steps must equal per-request
  dense decodes, at each row's own position -- that is what proves the packed prefill wrote
  the right KV columns and the right GDN state, not just the right last-token logits.
* **Group budget.** A group above ``max_prefill_tokens`` is rejected before anything is
  reserved (the scheduler splits it); one over-budget prompt alone is legal (P-44).
* **Failure atomicity.** A failure inside the single packed forward (a GDN layer raising
  after earlier layers already wrote state) leaves no active slot, no held block and no
  live seat; the pool's per-(slot, layer) "written" flags are cleared, so the bytes cannot
  be read as a request. A dense state that was half written refuses a later prefill instead
  of reusing a mixed prefix.
* **Paged arm.** On CUDA + Triton the same geometry runs through the 6-layer paged block
  table (``block_size`` 16 and 4, so prompts span one and two-plus blocks) and its packed
  prefill + one decode step match the seat arm and the dense reference. The packed-prefill
  kernel is FlashAttention (fp16/bf16 only), so the CUDA case runs in bf16 -- D9-7's formal
  dtype -- and is compared at the bf16 tolerance; CPU runs fp32 through SDPA-varlen.
"""

from __future__ import annotations

import pytest
import torch

from model.config import MiniLLMConfig
from model.layers import rope_cos_sin
from model.packed import PackedPrefill
from model.qwen35 import Qwen35Attention, Qwen35FullAttentionBlock
from model.qwen35_model import Qwen35Model, Qwen35State
from model.slab import KvSlab
from model.triton_decode import TRITON_AVAILABLE
from serve.engine import SlabWriteSink
from serve.hybrid_engine import Qwen35Engine, _LayerMapSink

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
#: Placements of the 6 full-attention layers; the compact index is the list position.
FULL_POSITIONS = [3, 7, 11, 15, 19, 23]
LINEAR_POSITIONS = [idx for idx in range(24) if idx not in FULL_POSITIONS]
#: Three prompts (5, 9, 3 tokens) so a shared position axis or a chained segment state
#: would be wrong, and so "the row order is request order" is discriminating.
PROMPTS = [
    [3, 5, 7, 11, 13],
    [2, 4, 6, 8, 10, 12, 14, 16, 18],
    [21, 23, 25],
]
ATOL = 1e-5
#: bf16 logits from two different attention paths (flash-varlen vs dense SDPA).
ATOL_BF16 = 2e-2


def _config() -> MiniLLMConfig:
    return MiniLLMConfig(**TINY)


def _model(device: str = "cpu", dtype: torch.dtype = torch.float32) -> Qwen35Model:
    torch.manual_seed(0)
    return Qwen35Model(_config()).to(device=device, dtype=dtype).eval()


def _ids(prompt: list[int], device: str = "cpu") -> torch.Tensor:
    return torch.tensor([prompt], dtype=torch.long, device=device)


def _dense_prefill(model: Qwen35Model, prompt: list[int]):
    """The Step 2/3 reference: one dense growing-cache prefill, ``(state, last logits)``."""
    device = str(model.token_embedding.weight.device)
    state = Qwen35State(model.config, device=device)
    with torch.no_grad():
        logits, _ = model(_ids(prompt, device), state, logits_to_keep=1)
    return state, logits[0, -1]


def _dense_step(model: Qwen35Model, state: Qwen35State, token: int) -> torch.Tensor:
    device = str(state.device)
    with torch.no_grad():
        logits, _ = model(
            torch.tensor([[int(token)]], dtype=torch.long, device=device),
            state,
            logits_to_keep=1,
        )
    return logits[0, -1]


def _assert_allclose(actual, expected, what: str, atol: float = ATOL) -> None:
    actual = actual.reshape(-1).detach()
    expected = expected.reshape(-1).detach()
    drift = float((actual - expected).abs().max())
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=0,
        msg=f"{what}: packed prefill drifted {drift:.2e} (atol={atol:.1e})",
    )


def _assert_logits(actual, expected, what: str, atol: float = ATOL) -> None:
    _assert_allclose(actual, expected, what, atol=atol)
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    got, want = int(actual.argmax()), int(expected.argmax())
    if got != want:
        margin = float(expected.topk(2).values.diff().abs().item())
        assert margin <= 2 * atol, (
            f"{what}: greedy ids {got} != {want} with top1-top2 margin {margin:.3e} "
            "(only a genuine near-tie may flip)"
        )


def _specs(prompts, seats=None, budgets=None) -> list[dict]:
    seats = list(range(len(prompts))) if seats is None else seats
    budgets = [4] * len(prompts) if budgets is None else budgets
    return [
        {"id": index, "prompt_ids": list(prompt), "seat": seat, "budget": budget}
        for index, (prompt, seat, budget) in enumerate(zip(prompts, seats, budgets))
    ]


class _RecordingSink:
    """One-shot packed sink: records each layer's ``(1, n_kv_head, length, head_dim)``."""

    def __init__(self, length: int, n_kv_head: int = 2, head_dim: int = 8) -> None:
        self.length = int(length)
        self.n_kv_head = int(n_kv_head)
        self.head_dim = int(head_dim)
        self.layers: list[int] = []

    def set_kv(self, layer: int, kv) -> None:
        k, v = kv
        expected = (1, self.n_kv_head, self.length, self.head_dim)
        assert tuple(k.shape) == expected, f"layer {layer}: {tuple(k.shape)} != {expected}"
        assert tuple(v.shape) == expected, f"layer {layer}: v {tuple(v.shape)}"
        if layer in self.layers:
            raise AssertionError(f"layer {layer} was written twice (one-shot sink)")
        self.layers.append(layer)


def _pack(model, prompts, states=None, sinks=None):
    """Build one request-major ``PackedPrefill`` and run the packed forward."""
    cu = [0]
    positions: list[int] = []
    tokens: list[int] = []
    if sinks is None:
        sinks = [_RecordingSink(len(prompt)) for prompt in prompts]
    for prompt in prompts:
        cu.append(cu[-1] + len(prompt))
        positions.extend(range(len(prompt)))
        tokens.extend(prompt)
    if states is None:
        states = [Qwen35State(model.config) for _ in prompts]
    packed = PackedPrefill(
        cu_seqlens=cu,
        positions=torch.tensor(positions, dtype=torch.long),
        sinks=sinks,
    )
    with torch.no_grad():
        logits, loss = model.forward_packed(
            torch.tensor([tokens], dtype=torch.long), packed, states
        )
    return logits, loss, packed, states, sinks


# --------------------------------------------------- packed == dense (seat arm)


def test_seat_packed_group_matches_dense_prefill_and_pool_state(monkeypatch):
    """One packed group: last-token logits, 18 GDN endpoints and 6 K/V layers == dense."""
    model = _model()
    engine = Qwen35Engine(model, eos_token_id=None, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=3, max_len=32, linear_slots=3, pool="seat")

    seen: dict = {}
    real_forward = model.forward_packed

    def spy(tokens, packed, states):
        seen["tokens"] = tokens
        seen["packed"] = packed
        seen["states"] = list(states)
        return real_forward(tokens, packed, states)

    monkeypatch.setattr(model, "forward_packed", spy)
    logits = engine.prefill_group(_specs(PROMPTS, [0, 1, 2], [4, 3, 5]))
    monkeypatch.setattr(model, "forward_packed", real_forward)

    assert tuple(logits.shape) == (len(PROMPTS), VOCAB)
    assert torch.isfinite(logits).all()

    # The engine built one request-major group, not one forward per request.
    packed = seen["packed"]
    assert seen["tokens"].tolist() == [[token for prompt in PROMPTS for token in prompt]]
    assert packed.cu_seqlens == [0, 5, 14, 17]
    assert packed.n_seqs == 3 and packed.total_tokens == 17
    assert packed.positions.tolist() == [0, 1, 2, 3, 4] + list(range(9)) + [0, 1, 2]
    assert packed.last_token_index.tolist() == [4, 13, 16]
    assert len(seen["states"]) == 3
    # The sinks are addressed by the compact full-attention index, so the packed path
    # must not wrap them in the 24-layer adapter (index 3 would collide with a real key).
    assert all(type(sink) is SlabWriteSink for sink in packed.sinks)
    assert not any(isinstance(sink, _LayerMapSink) for sink in packed.sinks)
    assert engine.attn_backends()["prefill"] == "sdpa-varlen"

    references = []
    for row, prompt in enumerate(PROMPTS):
        state, dense = _dense_prefill(model, prompt)
        references.append((state, dense))
        _assert_logits(logits[row], dense, f"prefill row {row}")

        # Only the final prompt token may reach lm_head (P-13). Compare against the
        # dense forward's previous position too: if they agreed, a wrong last-token
        # index could not be detected here.
        with torch.no_grad():
            full, _ = model(_ids(prompt))
        _assert_logits(logits[row], full[0, -1], f"row {row} last-token selection")
        assert not torch.allclose(
            full[0, -1], full[0, -2], atol=0.0, rtol=0.0
        ), f"row {row}: positions -1 and -2 give identical logits; selection is untested"

        # The pool slots hold exactly the dense endpoints of this prompt's GDN layers.
        for idx in LINEAR_POSITIONS:
            _assert_allclose(
                engine.state_pool.conv[idx][row],
                state.conv_state(idx)[0],
                f"row {row} conv layer {idx}",
            )
            _assert_allclose(
                engine.state_pool.recurrent[idx][row],
                state.recurrent_state(idx)[0],
                f"row {row} recurrent layer {idx}",
            )
        assert engine.state_pool.handle(row).length == len(prompt)

    # Rows are request order, not "all prompts concatenated": row 0 and row 1 differ.
    assert not torch.allclose(logits[0], logits[1])

    assert engine.slab.pos.tolist() == [5, 9, 3]
    assert engine.slab.valid.tolist() == [5, 9, 3]
    stats = engine.paged_stats()
    assert stats["active_rows"] == 3 and stats["linear_state_mb"] > 0.0
    assert stats["workspace_mb"] == 0.0 and stats["leak"] is False

    # The packed prefill wrote the dense K/V into each seat row: the 6 compact layers
    # hold this request's own keys, at its own length (a shared sink would show here).
    for row, (state, _) in enumerate(references):
        length = len(PROMPTS[row])
        for compact, actual_layer in enumerate(FULL_POSITIONS):
            dense_k, dense_v = state.get_kv(actual_layer)
            slab_k = engine.slab.keys[compact][row, :, :length]
            slab_v = engine.slab.values[compact][row, :, :length]
            _assert_allclose(slab_k, dense_k[0], f"row {row} slab K layer {compact}")
            _assert_allclose(slab_v, dense_v[0], f"row {row} slab V layer {compact}")

    for row in range(len(PROMPTS)):
        engine.release_request(row)
    pool = engine.state_pool
    engine.close_pool()
    pool.assert_no_leaks()


def test_packed_prefill_then_batched_decode_matches_dense_per_request():
    """Two batched decode steps after the packed group == two dense per-row decodes."""
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=3, max_len=32, linear_slots=3, pool="seat")

    logits = engine.prefill_group(_specs(PROMPTS, [0, 1, 2], [4, 3, 5]))
    references = []
    for row, prompt in enumerate(PROMPTS):
        state, dense = _dense_prefill(model, prompt)
        references.append((state, dense))
        _assert_logits(logits[row], dense, f"prefill row {row}")

    tokens = [int(logits[row].argmax()) for row in range(len(PROMPTS))]
    for step in range(2):
        out = engine.decode_batch(tokens)
        assert tuple(out.shape) == (engine.max_batch, VOCAB)
        for row, (state, _) in enumerate(references):
            _assert_logits(
                out[row],
                _dense_step(model, state, tokens[row]),
                f"decode step {step} row {row}",
            )
        assert engine.slab.pos.tolist() == [len(p) + step + 1 for p in PROMPTS]
        tokens = [int(out[row].argmax()) for row in range(len(PROMPTS))]

    for row in range(len(PROMPTS)):
        engine.release_request(row)
    assert engine.paged_stats()["leak"] is False
    engine.close_pool()


# ------------------------------------------------------------- group boundaries


def test_over_budget_group_is_rejected_but_a_single_long_prompt_is_allowed():
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=8)
    engine.open_hybrid_pool(max_batch=2, max_len=32, linear_slots=2, pool="seat")

    # 5 + 9 = 14 tokens > 8: the scheduler must split the group.
    with pytest.raises(ValueError, match="max_prefill_tokens"):
        engine.prefill_group(_specs(PROMPTS[:2], [0, 1], [2, 2]))
    # Rejected before anything was reserved or written.
    assert engine.state_pool.stats().active == 0
    assert engine.paged_stats()["active_rows"] == 0
    # A fresh KvSlab's rows are live but empty; the group was rejected before any prefill,
    # so no row moved.
    assert engine.slab.pos.tolist() == [0, 0]
    assert engine.slab.valid.tolist() == [0, 0]

    # A single over-budget prompt is legal: it forms its own group (P-44).
    logits = engine.prefill_group(_specs(PROMPTS[1:2], [0], [2]))
    assert tuple(logits.shape) == (1, VOCAB)
    _, dense = _dense_prefill(model, PROMPTS[1])
    _assert_logits(logits[0], dense, "single over-budget prompt")

    engine.release_request(0)
    assert engine.paged_stats()["leak"] is False
    engine.close_pool()


def test_packed_group_does_not_share_tokens_across_requests():
    """Changing request 0 leaves request 1 bit-identical (cu_seqlens is a real boundary)."""
    model = _model()
    baseline, _, packed, states, sinks = _pack(model, PROMPTS[:2])
    # One write per compact full-attention layer per request (6 each, not 24).
    assert [sink.layers for sink in sinks] == [list(range(6)), list(range(6))]
    assert packed.cu_seqlens == [0, 5, 14]
    assert [state.position for state in states] == [5, 9]

    changed = [list(prompt) for prompt in PROMPTS[:2]]
    changed[0][-1] = (changed[0][-1] + 7) % VOCAB
    after, _, _, _, _ = _pack(model, changed)

    assert not torch.allclose(baseline[0], after[0]), "request 0 should have changed"
    torch.testing.assert_close(baseline[1], after[1], atol=0.0, rtol=0.0)

    # And both rows equal their own dense prefill: the boundary is not just "different".
    for row, prompt in enumerate(PROMPTS[:2]):
        _, dense = _dense_prefill(model, prompt)
        _assert_logits(baseline[row], dense, f"row {row}")


def test_packed_layer_requires_its_compact_layer_index():
    """A 6-layer sink needs the compact index; a directly-built layer has none."""
    layer = Qwen35Attention(_config())
    assert layer.layer_index is None
    block = Qwen35FullAttentionBlock(_config())
    prompt = PROMPTS[0]
    packed = PackedPrefill(
        cu_seqlens=[0, len(prompt)],
        positions=torch.arange(len(prompt)),
        sinks=[_RecordingSink(len(prompt))],
    )
    cos, sin = rope_cos_sin(
        layer.head_dim,
        layer.rope_theta,
        0,
        len(prompt),
        dtype=torch.float32,
        rotary_dim=layer.rotary_dim,
    )
    x = torch.zeros(1, len(prompt), layer.n_embd)
    with pytest.raises(ValueError, match="compact full-attention index"):
        layer(x, None, 3, packed, cos, sin)
    with pytest.raises(ValueError, match="compact full-attention index"):
        block(x, None, 3, packed, cos, sin)


# ------------------------------------------------------------------- guards


def test_forward_packed_validates_the_group_before_writing_state():
    """Every layout/state error is raised before the first layer touches a sink."""
    model = _model()
    prompt = PROMPTS[0]
    valid = PackedPrefill(
        cu_seqlens=[0, 5], positions=torch.arange(5), sinks=[_RecordingSink(5)]
    )

    def fresh():
        return [Qwen35State(model.config) for _ in PROMPTS[:2]]

    with pytest.raises(TypeError, match="Tensor"):
        model.forward_packed([[1, 2]], valid, fresh())
    with pytest.raises(TypeError, match="packed-prefill container"):
        model.forward_packed(torch.ones(1, 5, dtype=torch.long), None, fresh())
    with pytest.raises(ValueError, match=r"\(1, 5\)"):
        model.forward_packed(torch.ones(2, 5, dtype=torch.long), valid, fresh())
    with pytest.raises(ValueError, match="int64"):
        model.forward_packed(torch.ones(1, 5), valid, fresh())

    # Token ids outside the vocabulary, rejected before any layer or sink runs.
    sinks = [_RecordingSink(3)]
    bad = PackedPrefill(cu_seqlens=[0, 3], positions=torch.arange(3), sinks=sinks)
    with pytest.raises(ValueError, match="token ids must be in"):
        model.forward_packed(
            torch.tensor([[prompt[0], VOCAB, prompt[1]]]),
            bad,
            [Qwen35State(model.config)],
        )
    assert sinks[0].layers == []

    # One fresh state per request: a dense prefill already wrote this one.
    written, _ = _dense_prefill(model, prompt)
    group = [written, Qwen35State(model.config)]
    with pytest.raises(ValueError, match="not fresh"):
        _pack(
            model,
            PROMPTS[:2],
            states=group,
            sinks=[_RecordingSink(5), _RecordingSink(9)],
        )
    with pytest.raises(ValueError, match="packed request"):
        _pack(model, PROMPTS[:2], states=[Qwen35State(model.config)])
    with pytest.raises(TypeError, match="sequence"):
        model.forward_packed(
            torch.ones(1, 5, dtype=torch.long), valid, Qwen35State(model.config)
        )
    with pytest.raises(TypeError, match="Qwen35State"):
        model.forward_packed(torch.ones(1, 5, dtype=torch.long), valid, [object()])
    with pytest.raises(ValueError, match="batch=2"):
        model.forward_packed(
            torch.ones(1, 5, dtype=torch.long),
            valid,
            [Qwen35State(model.config, batch=2)],
        )

    # A decode container in state.kv would be silently unused by the packed path.
    slab = KvSlab(6, 1, 2, 8, 8)
    with pytest.raises(ValueError, match="through the group's sinks"):
        model.forward_packed(
            torch.ones(1, 5, dtype=torch.long),
            valid,
            [Qwen35State(model.config, kv=slab)],
        )


def test_forward_packed_rejects_a_container_with_bad_offsets_or_positions():
    """A hand-rolled container cannot make the model attend across requests."""

    class Fake:
        packed_prefill = True

        def __init__(self, cu, positions, last):
            self.cu_seqlens = cu
            self.positions = torch.tensor(positions, dtype=torch.long)
            self.total_tokens = len(positions)
            self.n_seqs = len(cu) - 1
            self.sinks = [_RecordingSink(2) for _ in range(self.n_seqs)]
            self.rope_len = int(self.positions.max().item()) + 1
            self.last_token_index = torch.tensor(last, dtype=torch.long)

        def past_len(self, layer=None):
            return 0

        def write(self, layer, k, v):
            raise AssertionError("write must not be reached by a rejected container")

        def attend(self, q, k, v):
            raise AssertionError("attend must not be reached by a rejected container")

    model = _model()
    tokens = lambda total: torch.ones(1, total, dtype=torch.long)  # noqa: E731
    states = lambda: [Qwen35State(model.config) for _ in range(2)]  # noqa: E731

    bad = Fake([1, 3, 4], [0, 1, 0], [2, 3])
    with pytest.raises(ValueError, match="span"):
        model.forward_packed(tokens(bad.total_tokens), bad, states())

    bad = Fake([0, 2, 2], [0, 1], [1, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        model.forward_packed(tokens(2), bad, states())

    with pytest.raises(ValueError, match="request-local"):
        model.forward_packed(
            tokens(4), Fake([0, 2, 4], [1, 2, 0, 1], [1, 3]), states()
        )
    with pytest.raises(ValueError, match="last_token_index"):
        model.forward_packed(
            tokens(4), Fake([0, 2, 4], [0, 1, 0, 1], [1, 2]), states()
        )
    narrow = Fake([0, 2, 4], [0, 1, 0, 1], [1, 3])
    narrow.rope_len = 1
    with pytest.raises(ValueError, match="rope_len"):
        model.forward_packed(tokens(4), narrow, states())
    one_sink = Fake([0, 2, 4], [0, 1, 0, 1], [1, 3])
    one_sink.sinks = [_RecordingSink(2)]
    with pytest.raises(ValueError, match="sink"):
        model.forward_packed(tokens(4), one_sink, states())


def test_a_failed_layer_leaves_no_reusable_half_written_dense_state(monkeypatch):
    """A mid-forward failure writes some layers, never advances, and cannot be reused."""
    model = _model()
    states = [Qwen35State(model.config)]
    sinks = [_RecordingSink(len(PROMPTS[0]))]

    real_linear = model.layers[5].linear_attn.forward

    def boom(*args, **kwargs):
        raise RuntimeError("gdn blew up")

    monkeypatch.setattr(model.layers[5].linear_attn, "forward", boom)
    with pytest.raises(RuntimeError, match="gdn blew up"):
        _pack(model, PROMPTS[:1], states=states, sinks=sinks)
    monkeypatch.setattr(model.layers[5].linear_attn, "forward", real_linear)

    # Layers 0..4 ran: their linear layers wrote state, later ones did not, and the
    # cursor never moved. That is "partial", and it must not be reusable by accident.
    assert states[0].position == 0
    assert states[0].conv_state(0) is not None
    assert states[0].recurrent_state(0) is not None
    assert states[0].conv_state(4) is not None
    assert states[0].conv_state(6) is None
    assert sinks[0].layers == [0]  # only full-attention layer 3 (compact 0) was reached

    # A later packed prefill through the same state rejects the mixed prefix ...
    with pytest.raises(ValueError, match="not fresh"):
        _pack(model, PROMPTS[:1], states=states, sinks=[_RecordingSink(5)])
    # ... and so does the dense path: past_len is still 0, so the GDN layer refuses a
    # non-empty state instead of treating it as this prompt's prefix (P-56).
    with pytest.raises(ValueError, match="must start from empty state"):
        model(_ids(PROMPTS[0]), states[0], logits_to_keep=1)

    # A fresh state still works: the failure poisoned the state, not the model.
    logits, _, _, _, _ = _pack(model, PROMPTS[:1])
    _, dense = _dense_prefill(model, PROMPTS[0])
    _assert_logits(logits[0], dense, "fresh state after a failed forward")


def test_engine_rollback_after_a_forward_failure_frees_the_whole_group(monkeypatch):
    """A failure inside the packed forward leaves no active slot, seat or reservation."""
    model = _model()
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(max_batch=3, max_len=32, linear_slots=3, pool="seat")
    specs = _specs(PROMPTS, [0, 1, 2], [4, 3, 5])

    class Boom(Exception):
        pass

    real_linear = model.layers[5].linear_attn.forward

    def boom(*args, **kwargs):
        # Dies after layers 0..4 wrote pool state for all three requests: the rollback
        # must clear those rows too, not just the first request's.
        raise Boom("prefill died mid-group")

    monkeypatch.setattr(model.layers[5].linear_attn, "forward", boom)
    with pytest.raises(Boom, match="mid-group"):
        engine.prefill_group(specs)
    monkeypatch.setattr(model.layers[5].linear_attn, "forward", real_linear)

    stats = engine.state_pool.stats()
    assert stats.active == 0 and stats.peak_active == 3
    # The pool's per-(slot, layer) written flags are cleared, so the partial bytes can
    # never be read as a live request.
    for idx in LINEAR_POSITIONS:
        assert engine.state_pool._conv_ready[idx] == [False, False, False]
        assert engine.state_pool._recurrent_ready[idx] == [False, False, False]
    assert engine.paged_stats()["active_rows"] == 0
    assert engine.paged_stats()["linear_state_mb"] == 0.0
    assert engine.paged_stats()["leak"] is False
    assert engine.slab.active.tolist() == [False, False, False]
    assert engine.slab.pos.tolist() == [0, 0, 0]
    assert engine.slab.valid.tolist() == [0, 0, 0]

    # The same ids and seats are immediately reusable and the rows are not stale.
    logits = engine.prefill_group(specs)
    references = []
    for row, prompt in enumerate(PROMPTS):
        state, dense = _dense_prefill(model, prompt)
        references.append(state)
        _assert_logits(logits[row], dense, f"retry row {row}")
    out = engine.decode_batch([int(logits[row].argmax()) for row in range(3)])
    for row in range(3):
        _assert_logits(
            out[row],
            _dense_step(model, references[row], int(logits[row].argmax())),
            f"retry decode row {row}",
        )

    for row in range(3):
        engine.release_request(row)
    assert engine.paged_stats()["leak"] is False
    engine.close_pool()


# ------------------------------------------------------------------- paged arm

_PAGED_SKIP = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason=(
        "paged packed prefill needs CUDA for the flash-attn varlen arm and Triton for the "
        "block-table decode kernel; there is no dense paged fallback"
    ),
)


@_PAGED_SKIP
@pytest.mark.parametrize("block_size", [16, 4])
def test_paged_packed_group_matches_seat_and_dense_then_decodes(block_size):
    """The 6-layer paged block table, filled by packed prefill, == seat and dense."""
    model = _model("cuda", torch.bfloat16)
    seat = Qwen35Engine(model, max_prefill_tokens=64)
    paged = Qwen35Engine(model, max_prefill_tokens=64)
    seat.open_hybrid_pool(max_batch=3, max_len=32, linear_slots=3, pool="seat")
    paged.open_hybrid_pool(
        max_batch=3,
        max_len=32,
        linear_slots=3,
        kv_blocks=32,
        block_size=block_size,
        pool="paged",
    )
    specs = _specs(PROMPTS, [0, 1, 2], [2, 2, 2])

    seat_logits = seat.prefill_group(specs)
    paged_logits = paged.prefill_group(specs)
    # Same packed varlen prefill on both arms (only the sink differs): identical logits.
    torch.testing.assert_close(paged_logits, seat_logits, atol=0.0, rtol=0.0)
    assert paged.attn_backends()["prefill"] == "flash-attn-varlen"
    assert paged.attn_backends()["decode"] == "paged-triton"

    for row, prompt in enumerate(PROMPTS):
        _, dense = _dense_prefill(model, prompt)
        _assert_logits(paged_logits[row], dense, f"paged row {row}", atol=ATOL_BF16)

    tokens = [int(paged_logits[row].argmax()) for row in range(len(PROMPTS))]
    seat_out = seat.decode_batch(tokens)
    paged_out = paged.decode_batch(tokens)
    torch.testing.assert_close(paged_out, seat_out, atol=ATOL_BF16, rtol=0)
    assert paged_out.argmax(-1).tolist() == seat_out.argmax(-1).tolist()
    for row, prompt in enumerate(PROMPTS):
        state, _ = _dense_prefill(model, prompt)
        _assert_logits(
            paged_out[row],
            _dense_step(model, state, tokens[row]),
            f"paged decode row {row}",
            atol=ATOL_BF16,
        )

    stats = paged.paged_stats()
    assert stats["blocks_in_use"] > 0 and stats["written_blocks"] > 0
    assert stats["decode_impl"] == "triton" and stats["workspace_mb"] == 0.0
    assert stats["kv_live_mb"] <= stats["kv_alloc_mb"] <= stats["kv_mb"]
    assert stats["leak"] is False

    for engine in (seat, paged):
        for row in range(len(PROMPTS)):
            engine.release_request(row)
        assert engine.paged_stats()["active_rows"] == 0
        assert engine.paged_stats()["leak"] is False
        engine.close_pool()


@_PAGED_SKIP
def test_paged_engine_rollback_returns_every_block_after_a_failed_forward(monkeypatch):
    """Paged arm: a failed packed forward returns every block and leaves no active row."""
    model = _model("cuda", torch.bfloat16)
    engine = Qwen35Engine(model, max_prefill_tokens=64)
    engine.open_hybrid_pool(
        max_batch=3, max_len=32, linear_slots=3, kv_blocks=32, block_size=16, pool="paged"
    )
    specs = _specs(PROMPTS, [0, 1, 2], [2, 2, 2])
    before = engine.block_manager.stats()["free"]

    class Boom(Exception):
        pass

    real_forward = model.forward_packed
    monkeypatch.setattr(
        model,
        "forward_packed",
        lambda *args, **kwargs: (_ for _ in ()).throw(Boom("forward died")),
    )
    with pytest.raises(Boom, match="forward died"):
        engine.prefill_group(specs)
    monkeypatch.setattr(model, "forward_packed", real_forward)

    assert engine.block_manager.stats()["free"] == before
    assert engine.block_manager.in_use == 0
    stats = engine.paged_stats()
    assert stats["active_rows"] == 0 and stats["blocks_in_use"] == 0
    assert stats["linear_state_mb"] == 0.0 and stats["leak"] is False

    # The same group is immediately admissible again (no orphaned reservation).
    logits = engine.prefill_group(specs)
    for row, prompt in enumerate(PROMPTS):
        _, dense = _dense_prefill(model, prompt)
        _assert_logits(logits[row], dense, f"retry row {row}", atol=ATOL_BF16)

    for row in range(len(PROMPTS)):
        engine.release_request(row)
    assert engine.block_manager.stats()["free"] == before
    assert engine.paged_stats()["leak"] is False
    engine.close_pool()
