"""RoPE settings must be read from wherever transformers put them (PITFALLS P-03).

transformers 5.x moved ``rope_theta`` into ``rope_parameters``. A flat ``getattr`` with a
1e4 default does not fail — it silently feeds the wrong theta and turns the whole W1
diff red for the wrong reason. These guards are cheap and weight-free; they run in the
fast gate.

``scripts/align_hf.py`` only imports transformers inside ``load_qwen3``, so this stays
offline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.align_hf import config_from_hf, rope_parameters

BASE = dict(
    vocab_size=8,
    hidden_size=16,
    num_attention_heads=2,
    num_key_value_heads=1,
    num_hidden_layers=1,
    intermediate_size=32,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
)


def fake(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{**BASE, **overrides})


def test_transformers5_rope_parameters_is_read():
    cfg = config_from_hf(fake(rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"}))
    assert cfg.rope_theta == 1_000_000


def test_transformers4_flat_attribute_still_works():
    cfg = config_from_hf(fake(rope_theta=10_000.0))
    assert cfg.rope_theta == 10_000.0


def test_missing_theta_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="rope_theta"):
        config_from_hf(fake())


def test_scaled_rope_is_rejected():
    with pytest.raises(ValueError, match="rope_type"):
        config_from_hf(fake(rope_parameters={"rope_theta": 1e6, "rope_type": "linear"}))


def test_partial_rotary_factor_is_carried_for_w5():
    cfg = config_from_hf(
        fake(rope_parameters={"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": 0.25})
    )
    assert cfg.partial_rotary_factor == 0.25


def test_partial_rotary_defaults_to_full():
    cfg = config_from_hf(fake(rope_parameters={"rope_theta": 1e6}))
    assert cfg.partial_rotary_factor == 1.0


def test_rope_parameters_prefers_v5_dict_over_flat_attribute():
    """A stale flat attribute must not win over the dict transformers actually uses."""
    cfg = config_from_hf(fake(rope_theta=10_000.0, rope_parameters={"rope_theta": 1e6}))
    assert cfg.rope_theta == 1_000_000
    assert rope_parameters(fake(rope_theta=10_000.0, rope_parameters={"rope_theta": 1e6}))["rope_theta"] == 1e6
