"""W3 pool admission: prefill writes straight into the seat's row (w3-plan §2.2/§2.4).

The pool path must not build a second copy of the prompt's KV next to the pool row that is
already allocated (E0b: 166 MiB of prefill ``KVcache`` on top of the 182 MiB pool at ctx
1517, ratio 0.859). These tests pin the replacement:

* ``prefill_into_seat`` is **bit-identical** to the old ``prefill_one`` + ``fill_from``
  hand-off -- logits and the bytes in the row both;
* it constructs no ``KVcache`` and calls no ``fill_from``;
* a forward that dies half way through the layer stack leaves the row frozen (cursors at
  0, seat closed), and the next admission on that row works;
* the write-through sink refuses a second prefill instead of silently overwriting columns;
* the length / row guards fire before anything is written.

CPU only, tiny model, no golden files.
"""

from __future__ import annotations

import pytest
import torch

from model import KvSlab, MiniLLM, MiniLLMConfig
from serve import engine as engine_module
from serve.engine import MiniLLMEngine, SlabWriteSink

PROMPTS = [
    [3, 5, 7, 11, 13],
    [3, 5, 7, 11, 13, 17, 19, 23, 29],
]


def _config(n_layer: int = 2, max_position: int = 64) -> MiniLLMConfig:
    return MiniLLMConfig(vocab_size=64, max_position_embeddings=max_position, n_embd=32,
                         n_head=4, n_kv_head=2, n_layer=n_layer, dropout=0.0, bias=False)


def _model(dtype: torch.dtype = torch.float32, **kwargs) -> MiniLLM:
    torch.manual_seed(0)
    return MiniLLM(_config(**kwargs)).to(dtype).eval()


# --------------------------------------------------------- equivalence with the hand-off

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_into_seat_is_bit_identical_to_the_handoff(dtype):
    """Same prompt, (a) prefill+``fill_from`` vs (b) write-through: equal logits and KV."""
    model = _model(dtype)
    engine = MiniLLMEngine(model)
    n_layer = model.config.n_layer
    prompt = PROMPTS[1]
    slab = engine.open_pool(1, 32)

    with torch.inference_mode():
        # (a) the W2 hand-off: a prefill KVcache, then layer-by-layer copies into row 0.
        logits_handoff, cache = engine.prefill_one(prompt)
        for layer in range(n_layer):
            slab.fill_from(cache, layer, row=0)
        slab.activate(0, len(prompt))
        handed_kv = [
            (slab.keys[layer][0].clone(), slab.values[layer][0].clone())
            for layer in range(n_layer)
        ]
        row_before = slab.keys[0][0].clone()

        # (b) the pool path: the same forward, written into the (frozen) seat from column 0.
        slab.freeze(0)
        logits_pool = engine.prefill_into_seat(0, prompt)

    assert torch.equal(logits_pool, logits_handoff), (
        "write-through prefill changed the logits: "
        f"max |diff| = {(logits_pool.float() - logits_handoff.float()).abs().max().item():.2e}"
    )
    for layer in range(n_layer):
        assert torch.equal(slab.keys[layer][0], handed_kv[layer][0]), (
            f"layer {layer}: the pooled keys differ from what fill_from wrote"
        )
        assert torch.equal(slab.values[layer][0], handed_kv[layer][1]), (
            f"layer {layer}: the pooled values differ from what fill_from wrote"
        )
    assert torch.equal(slab.keys[0][0], row_before)  # the second write reproduced the bytes
    assert slab.pos.tolist() == [len(prompt)] and slab.valid.tolist() == [len(prompt)]


def test_pool_path_builds_no_kvcache_and_copies_nothing(monkeypatch):
    """The pool path must not construct a ``KVcache`` nor call ``fill_from``."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab = engine.open_pool(2, 32)

    def no_kvcache(*args, **kwargs):
        raise AssertionError("the pool path constructed a KVcache")

    def no_fill_from(*args, **kwargs):
        raise AssertionError("the pool path called KvSlab.fill_from")

    monkeypatch.setattr(engine_module, "KVcache", no_kvcache)
    monkeypatch.setattr(KvSlab, "fill_from", no_fill_from)

    logits = engine.prefill_into_seat(1, PROMPTS[0])

    assert tuple(logits.shape) == (1, model.config.vocab_size)
    assert torch.isfinite(logits).all()
    # Only the admitted row moved; the untouched seat is still where the pool left it.
    assert slab.pos.tolist() == [0, len(PROMPTS[0])]
    assert slab.valid.tolist() == [0, len(PROMPTS[0])]
    assert slab.active.tolist() == [True, True]
    assert torch.isfinite(engine.decode_batch([0, 0])).all()


# ------------------------------------------------------------------- half-write rollback

def test_failed_forward_freezes_the_half_written_row(monkeypatch):
    """A forward that raises in layer 1 leaves row 0 half written -> frozen, reusable."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab = engine.open_pool(2, 32)
    prompt = PROMPTS[1]
    row = 0
    state: dict[str, float] = {}

    def boom(x, attention_mask=None, idx=0, past_keys_values=None, cos=None, sin=None):
        # Layer 1 runs after layer 0's columns have landed, so the row really is half
        # written when the failure hits -- otherwise this test would prove nothing.
        state["layer0"] = float(slab.keys[0][row, :, : len(prompt)].float().abs().sum())
        raise RuntimeError("boom in layer 1")

    monkeypatch.setattr(model.layers[1], "forward", boom)
    with pytest.raises(RuntimeError, match="boom in layer 1"):
        engine.prefill_into_seat(row, prompt)

    assert state["layer0"] > 0.0, "expected the failed prefill to have written layer 0"
    assert slab.pos.tolist()[row] == 0
    assert slab.valid.tolist()[row] == 0
    assert slab.active.tolist()[row] is False
    assert slab._host_pos[row] == 0 and slab._host_valid[row] == 0

    # The same seat takes the next request normally: columns are overwritten from 0.
    monkeypatch.undo()
    logits = engine.prefill_into_seat(row, PROMPTS[0])
    assert slab.pos.tolist()[row] == len(PROMPTS[0])
    assert slab.valid.tolist()[row] == len(PROMPTS[0])
    assert slab.active.tolist()[row] is True
    assert torch.isfinite(logits).all()
    assert torch.isfinite(engine.decode_batch([0, 0])).all()


# ------------------------------------------------------------------------- sink contract

def test_sink_refuses_a_second_prefill():
    """Chunked prefill (past_len > 0) must fail loudly, not overwrite columns 0..L-1."""
    model = _model()
    engine = MiniLLMEngine(model)
    slab = engine.open_pool(1, 32)
    sink = SlabWriteSink(slab, 0, len(PROMPTS[0]))

    with torch.inference_mode():
        model(torch.tensor([PROMPTS[0]]), past_keys_values=sink, logits_to_keep=1)
        with pytest.raises(RuntimeError, match="chunked prefill"):
            model(torch.tensor([PROMPTS[1]]), past_keys_values=sink, logits_to_keep=1)

    # The sink is read-as-empty by construction: no history, so the layer never cats.
    assert sink.past_len(0) == 0
    assert sink.get_kv(0) == (None, None)


def test_length_and_row_guards_fire_before_anything_is_written():
    model = _model(max_position=100)
    engine = MiniLLMEngine(model)
    slab = engine.open_pool(1, 50)  # bucket 64 < max_position 100, so max_len is reachable
    assert slab.max_len == 64

    with pytest.raises(IndexError, match="exceeds the pool's max_len"):
        engine.prefill_into_seat(0, list(range(70)))
    with pytest.raises(IndexError, match="outside the slab"):
        engine.prefill_into_seat(3, PROMPTS[0])
    with pytest.raises(ValueError, match="must not be empty"):
        engine.prefill_into_seat(0, [])

    assert slab.pos.tolist() == [0]
    assert slab.valid.tolist() == [0]
    assert slab.active.tolist() == [True]
