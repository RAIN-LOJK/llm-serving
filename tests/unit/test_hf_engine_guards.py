"""HF engine guards: one logit position, text-only extraction, no silent fallbacks.

These run without network or real weights: a tiny Qwen3 and a tiny Qwen3.5 multimodal
checkpoint are built locally, so the fast gate covers the two failure modes that cost a
32k run (``[seq, vocab]`` logits) and a 0.8B load (vision tower in the text path).
"""

from __future__ import annotations

import types

import pytest
import torch

from bench.engines.hf import HFEngine, _assert_text_only

transformers = pytest.importorskip("transformers")


def _tiny_qwen3():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    return Qwen3ForCausalLM(config).eval()


def _bare_engine(model) -> HFEngine:
    engine = HFEngine("tiny", device="cpu")
    engine.model = model
    engine.weights_mb = 0.0
    engine.vocab_size = 64
    engine.eos_token_id = None
    engine._baseline_mem_mb = None
    return engine


def test_prefill_asks_for_exactly_one_logit():
    model = _tiny_qwen3()
    calls: list[dict] = []
    original = model.forward

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    model.forward = spy  # type: ignore[method-assign]
    engine = _bare_engine(model)

    result = engine.generate([1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=2)

    assert len(result.token_ids) == 2
    assert len(calls) == 2  # prefill + one decode step
    assert all(call["logits_to_keep"] == 1 for call in calls)
    assert result.meta["logits_to_keep"] == 1


def test_full_logits_raise_instead_of_silently_allocating():
    class IgnoresLogitsToKeep(torch.nn.Module):
        def forward(self, input_ids, **_kwargs):
            return types.SimpleNamespace(
                logits=torch.zeros(1, input_ids.shape[1], 64),
                past_key_values=None,
            )

    engine = _bare_engine(IgnoresLogitsToKeep())
    with pytest.raises(RuntimeError, match="logits_to_keep"):
        engine._forward(torch.zeros(1, 4, dtype=torch.long))


def test_guard_rejects_wrapper_class_and_vision_params():
    wrapper = type("Qwen3_5ForConditionalGeneration", (torch.nn.Module,), {})()
    with pytest.raises(RuntimeError, match="text-only"):
        _assert_text_only(wrapper, types.SimpleNamespace(vocab_size=8))

    leaked = torch.nn.Module()
    leaked.model = torch.nn.Module()
    leaked.model.visual = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="multimodal parameters"):
        _assert_text_only(leaked, types.SimpleNamespace(vocab_size=8))


def _save_tiny_multimodal(directory) -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

    text = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "intermediate_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
        },
    )
    Qwen3_5ForConditionalGeneration(config).save_pretrained(directory)

    vocab = {"<|endoftext|>": 0}
    for ch in "Ġhelowrd":
        if ch not in vocab:
            vocab[ch] = len(vocab)
    Qwen3_5Tokenizer(
        vocab=vocab,
        merges=[],
        unk_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ).save_pretrained(directory)


def test_multimodal_checkpoint_loads_text_only(tmp_path):
    _save_tiny_multimodal(tmp_path)
    engine = HFEngine(str(tmp_path), revision=None, dtype="float32", device="cpu")
    try:
        engine.load()
        assert engine.arch == "Qwen3_5ForCausalLM"
        assert [n for n, _ in engine.model.named_parameters() if "visual" in n] == []
        assert engine.vocab_size == 32
        result = engine.generate(engine.encode("hello world"), max_new_tokens=1)
        assert len(result.token_ids) == 1
        assert result.state_mb and result.state_mb > 0  # hybrid state is live
    finally:
        engine.close()
