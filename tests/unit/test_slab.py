"""``serve/slab.py``: a fixed-size KV buffer must be numerically identical to ``cat``.

The slab exists to replace ``torch.cat`` growth (address stability for CUDA graphs), so the
only property that matters is that the bytes end up where a reader expects them. These
tests pin that against the cat-grown ``model.KVcache``, plus the pieces a graph capture
depends on: stable tensor objects, an in-place mask, and a write that does not allocate
new storage per step.
"""

from __future__ import annotations

import pytest
import torch

from model import KVcache
from model.slab import KvSlab


def _mk(n_layer=3, batch=2, kv_head=2, head=4, max_len=8, dtype=torch.float32, n_head=None):
    return KvSlab(
        n_layer, batch, kv_head, head, max_len, dtype=dtype, device="cpu",
        n_head=kv_head * 2 if n_head is None else n_head,
    )


def test_written_history_equals_cat_grown_cache():
    slab = _mk()
    cache = KVcache(slab.n_layer)
    torch.manual_seed(0)

    for _ in range(5):
        token = torch.randn(slab.batch, slab.n_kv_head, 1, slab.head_dim)
        for layer in range(slab.n_layer):
            slab.write(layer, token, token * 2)
            past_k, past_v = cache.get_kv(layer)
            cache.set_kv(layer, (token.clone() if past_k is None else torch.cat([past_k, token], 2),
                                 token.clone() * 2 if past_v is None else torch.cat([past_v, token * 2], 2)))
        slab.advance()

    assert slab.past_len() == 5
    for layer in range(slab.n_layer):
        got_k, got_v = slab.get(layer)
        ref_k, ref_v = cache.get_kv(layer)
        assert torch.equal(got_k[:, :, :5], ref_k), f"layer {layer}: keys drifted"
        assert torch.equal(got_v[:, :, :5], ref_v), f"layer {layer}: values drifted"


def test_writes_do_not_replace_the_buffers():
    """A graph capture bakes in addresses; reallocating a buffer would silently break it."""
    slab = _mk()
    addresses = {(id(k), k.data_ptr()) for k in slab.keys + slab.values}
    for _ in range(3):
        token = torch.randn(slab.batch, slab.n_kv_head, 1, slab.head_dim)
        for layer in range(slab.n_layer):
            slab.write(layer, token, token)
        slab.advance()
        slab.build_mask()
    after = {(id(k), k.data_ptr()) for k in slab.keys + slab.values}
    assert addresses == after, "slab buffers were reallocated"


def test_per_row_positions_are_independent():
    """W3 admits and retires rows at different times: rows must not share a column."""
    slab = _mk(batch=3, max_len=8)
    rows = [0, 2, 1]  # three different target columns for the same step
    slab.pos.copy_(torch.tensor(rows))
    token = torch.arange(3 * slab.n_kv_head * slab.head_dim, dtype=torch.float32).view(
        3, slab.n_kv_head, 1, slab.head_dim
    )
    for layer in range(slab.n_layer):
        slab.write(layer, token, -token)

    for row, column in enumerate(rows):
        got_k, got_v = slab.get(0)
        assert torch.equal(got_k[row, :, column], token[row, :, 0])
        assert torch.equal(got_v[row, :, column], -token[row, :, 0])


def test_mask_matches_the_valid_lengths():
    slab = _mk(batch=3, max_len=8)
    slab.valid.copy_(torch.tensor([5, 2, 8]))
    mask = slab.build_mask()

    expected = torch.zeros(3, 1, 1, 8, dtype=torch.bool)
    expected[0, :, :, :5] = True
    expected[1, :, :, :2] = True
    expected[2, :, :, :] = True
    assert mask.dtype is torch.bool
    assert torch.equal(mask, expected)


def test_mask_is_updated_in_place():
    slab = _mk(batch=2, max_len=6)
    first = slab.build_mask()
    address = first.data_ptr()
    slab.valid.copy_(torch.tensor([1, 3]))
    second = slab.build_mask()

    assert second.data_ptr() == address, "build_mask allocated a new tensor"
    assert torch.equal(second[:, 0, 0, 0], torch.tensor([True, True]))
    assert second[1, 0, 0, 2] and not second[1, 0, 0, 3]


def test_reset_clears_cursors_and_mask():
    slab = _mk()
    token = torch.randn(slab.batch, slab.n_kv_head, 1, slab.head_dim)
    slab.write(0, token, token)
    slab.advance()
    slab.build_mask()

    slab.reset()

    assert int(slab.pos.max()) == 0 and int(slab.valid.max()) == 0
    assert not bool(slab.mask.any())


def test_overflow_fails_loudly():
    slab = _mk(max_len=2)
    token = torch.randn(slab.batch, slab.n_kv_head, 1, slab.head_dim)
    for _ in range(2):
        slab.write(0, token, token)
        slab.advance()
    with pytest.raises(IndexError, match="overflow"):
        for _ in range(2):
            slab.write(0, token, token)
            slab.advance()


def test_write_validates_the_shape():
    slab = _mk()
    with pytest.raises(ValueError, match="write needs"):
        slab.write(0, torch.zeros(slab.batch, slab.n_kv_head, 2, slab.head_dim), torch.zeros(1))


def test_bytes_follow_the_formula_account():
    slab = _mk(n_layer=3, batch=2, kv_head=2, head=4, max_len=8, dtype=torch.float32)
    per_token = 2 * slab.n_kv_head * slab.head_dim * 4  # k + v, one token, one layer
    assert slab.bytes() == per_token * 3 * 2 * 8


# ---------------------------------------------------------------- host bookkeeping

def test_bias_is_the_mask_in_additive_form():
    """The slab owns a persistent ``(B*H_kv, G, L)`` bias identical to ``additive_bias``."""
    from model.attention.hand_baddbmm import additive_bias

    slab = _mk(batch=3, head=4)
    slab.valid.copy_(torch.tensor([5, 2, 0]))  # row 2 is a dead (never-filled) seat
    slab._host_valid = [5, 2, 0]
    mask = slab.build_mask(next_token=True)
    address = slab.bias.data_ptr()

    expected = additive_bias(mask, slab.dtype, 3, slab.n_kv_head, slab.n_group)
    assert slab.bias.shape == (3 * slab.n_kv_head, slab.n_group, slab.max_len)
    assert torch.equal(slab.bias, expected)

    # A row that may attend nothing at all is zeroed by ``additive_bias`` (the NaN guard,
    # P-24); with ``next_token`` every row allows the column it is about to write.
    dead = slab.n_kv_head  # first kv-head slot of row 1 (valid=2 -> 2 allowed columns)
    assert torch.equal(slab.bias[dead, :, :2], torch.zeros(slab.n_group, 2))
    assert bool((slab.bias[dead] == float("-inf")).any())

    slab.build_mask()  # row 2 has valid == 0, so it is fully masked this time
    assert torch.isfinite(slab.bias[2 * slab.n_kv_head]).all(), "dead row must be zeroed"

    slab.build_mask(next_token=True)
    assert slab.bias.data_ptr() == address, "build_mask reallocated the bias buffer"


def test_decode_path_does_no_host_sync(monkeypatch):
    """Guards the 28 ``bool((pos >= max_len).any())`` syncs that made B=1 decode slower.

    ``.item()`` / ``bool(tensor)`` are the two host-sync forms; both are made to raise so a
    reintroduced sync fails here instead of hiding in a 25 ms step.
    """
    from model import MiniLLM, MiniLLMConfig
    from serve.engine import MiniLLMEngine

    def boom(*args, **kwargs):
        raise AssertionError("host sync on the eager slab decode path")

    torch.manual_seed(0)
    model = MiniLLM(
        MiniLLMConfig(vocab_size=64, max_position_embeddings=64, n_embd=32, n_head=4,
                      n_kv_head=2, n_layer=2, dropout=0.0, bias=False)
    ).eval()
    engine = MiniLLMEngine(model)
    prompt = [3, 5, 7, 11, 13, 17, 19]
    logits, cache = engine.prefill(prompt)
    slab = KvSlab(model.config.n_layer, 1, model.config.n_kv_head, model.head_dim, 24,
                  dtype=torch.float32, n_head=model.config.n_head)
    for layer in range(model.config.n_layer):
        slab.fill_from(cache, layer)

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    for _ in range(3):
        engine.decode_slab([int(logits.argmax(-1))], slab)


def test_bounds_are_tracked_without_reading_the_tensors():
    """Overflow is still reported from the Python counter (``check_bounds=True``)."""
    slab = _mk(max_len=2)
    token = torch.randn(slab.batch, slab.n_kv_head, 1, slab.head_dim)
    slab.write(0, token, token)
    slab.advance()
    slab.write(0, token, token)
    slab.advance()
    # the host mirror now says "next column is 2", i.e. past the last one
    with pytest.raises(IndexError, match="overflow"):
        slab.write(0, token, token)


def test_strict_mode_still_reads_the_tensors_back():
    """``check_bounds='strict'`` re-checks against ``pos``/``valid`` (the graph entry point).

    A caller that moves the cursors behind the class's back (as a captured graph does) is
    caught in strict mode, and ``sync_host`` realigns the fast path on request.
    """
    slab = _mk(max_len=2)
    slab.check_bounds = "strict"
    token = torch.randn(slab.batch, slab.n_kv_head, 1, slab.head_dim)
    slab.pos.fill_(2)  # behind the class's back: the Python mirror still says 0
    with pytest.raises(IndexError, match="overflow"):
        slab.write(0, token, token)

    slab.check_bounds = True
    slab.sync_host()
    with pytest.raises(IndexError, match="overflow"):
        slab.write(0, token, token)

