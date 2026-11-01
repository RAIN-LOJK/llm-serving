"""附件：可插拔 attention 契约的 oracle 一致性与「不改数值」。

这份文件由验收方（reviewer）独立编写，**不依赖**任何后端内部实现细节，只走
``model.attention`` 的公开常量与 ``model.layers.Attention``。口径已定死（D4）：
prefill 只跑 ``sdpa``，decode 只跑 ``hand_baddbmm``，oracle 是 ``tests.reference.hand_legacy``。
它要挡住四类事故：

1. 后端换了实现、数值却漂了（对 oracle 逐形状对拍）；
2. 默认路径被这次重构改变了逐 bit 数值（对「重构前实现」的冻结副本）；
3. 解码只支持 T == 1 的后端被拿去跑 prefill（必须 raise，不许静默出数）；
4. 派发次数随 batch 线性增长（就是 D3 那版逐序列 attention 的病，成本 ∝ B 而非 ∝ B×L）。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model import MiniLLM, MiniLLMConfig
from model.attention import DECODE_BACKEND, PREFILL_BACKEND, provenance
from tests.reference.hand_legacy import BACKEND as ORACLE


# --------------------------------------------------------------------------- helpers
def _mk(B, H, Hkv, T, L, D, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, D, generator=g, dtype=dtype)
    k = torch.randn(B, Hkv, L, D, generator=g, dtype=dtype)
    v = torch.randn(B, Hkv, L, D, generator=g, dtype=dtype)
    return q, k, v


def _bool_mask_offsets(B, T, L, valid=None):
    """offset-causal 的 bool allow-mask，可选再叠一条「每行有效长度不同」的左 pad。"""
    qpos = torch.arange(L - T, L).view(1, T, 1)
    kpos = torch.arange(L).view(1, 1, L)
    allow = (kpos <= qpos).view(1, 1, T, L).expand(B, -1, -1, -1).clone()
    if valid is not None:
        for b, n in enumerate(valid):
            allow[b, :, :, n:] = False
    return allow


# --------------------------------------------------------------------------- fixed policy
def test_provenance_is_the_frozen_policy():
    """The only attention "configuration" left is a constant, and it must say so."""
    assert provenance() == {"prefill": "sdpa", "decode": "hand_baddbmm"}
    assert PREFILL_BACKEND.name == "sdpa"
    assert DECODE_BACKEND.name == "hand_baddbmm"
    assert PREFILL_BACKEND is not DECODE_BACKEND


def test_hand_baddbmm_refuses_prefill():
    q, k, v = _mk(2, 4, 2, 5, 5, 16)
    with pytest.raises(NotImplementedError, match="decode-only"):
        DECODE_BACKEND.prefill(q, k, v)
    with pytest.raises(NotImplementedError, match="decode-only"):
        DECODE_BACKEND.prefill(q, k, v, is_causal=True)


# --------------------------------------------------------------------------- oracle matrix
def _shapes():
    """(名字, B, H, Hkv, T, L, D, kwargs) —— 覆盖 prefill/decode/chunked/左 pad/GQA。

    D=128 与载体一致：`hand_baddbmm` 把 scale 折进 q，当 D 是 2 的偶次幂（如 8、16）时
    `1/sqrt(D)` 恰好是 2 的幂、缩放不产生舍入，会掩盖 bf16 的真实误差（实测 D=8 时最坏
    1.6e-2、D=128 时 5.4e-3）。用真实宽度测，容差才有意义。
    """
    return [
        ("prefill_causal", 2, 4, 2, 13, 13, 128, {"is_causal": True}),
        ("prefill_full", 2, 4, 4, 9, 9, 128, {}),
        ("chunked_offset", 2, 4, 2, 5, 12, 128, {"mask": _bool_mask_offsets(2, 5, 12)}),
        ("decode_first", 3, 4, 2, 1, 1, 128, {}),
        ("decode_with_past", 3, 4, 2, 1, 17, 128, {"mask": _bool_mask_offsets(3, 1, 17, valid=[17, 12, 7])}),
        ("leftpad_prefill", 3, 4, 2, 6, 6, 128, {"mask": _bool_mask_offsets(3, 6, 6, valid=[6, 4, 2])}),
    ]


# Each subject is exercised only on the shapes its phase owns: prefill shapes on sdpa,
# decode shapes on hand_baddbmm (the frozen policy), both against the same oracle.
SUBJECTS = [
    pytest.param(PREFILL_BACKEND, "sdpa", id="sdpa"),
    pytest.param(DECODE_BACKEND, "hand_baddbmm", id="hand_baddbmm"),
]


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("subject,name", SUBJECTS)
def test_backend_matches_oracle(subject, name, dtype, atol):
    """每个后端都必须与 oracle 在它负责的形状上一致；非它负责的 phase 必须 raise。"""
    for shape_name, B, H, Hkv, T, L, D, kwargs in _shapes():
        q, k, v = _mk(B, H, Hkv, T, L, D, dtype=dtype, seed=sum(map(ord, shape_name)))
        if T > 1 and name != "sdpa":
            with pytest.raises(NotImplementedError):
                subject.prefill(q, k, v, **kwargs)
            continue
        if T == 1 and name != "hand_baddbmm":
            # sdpa is only on the prefill path; the decode shape is not its contract here.
            continue
        if T > 1:
            ref = ORACLE.prefill(q, k, v, **kwargs)
            out = subject.prefill(q, k, v, **kwargs)
        else:
            m = kwargs.get("mask")
            ref = ORACLE.decode(q, k, v, mask=m)
            out = subject.decode(q, k, v, mask=m)
        assert out.shape == ref.shape == (B, H, T, D), f"{name}/{shape_name}: 形状错"
        drift = (out.float() - ref.float()).abs().max().item()
        assert drift <= atol, f"{name}/{shape_name}: 与 oracle 差 {drift:.2e} > {atol}"


def test_oracle_is_not_vacuous():
    """oracle 必须真的做事：打乱 V 的输出应与正常输出不同，否则上面的对拍无意义。"""
    q, k, v = _mk(2, 4, 2, 7, 7, 8, seed=3)
    base = ORACLE.prefill(q, k, v, is_causal=True)
    perm = ORACLE.prefill(q, k, torch.flip(v, dims=[2]), is_causal=True)
    assert not torch.allclose(base, perm), "oracle 对 V 的置换不敏感，说明对拍形同虚设"


def test_sdpa_rejects_float_masks():
    """P-24：bf16 的 -inf 加法 mask 会静默错 1.5e-1，只允许 bool mask。"""
    q, k, v = _mk(2, 4, 2, 1, 9, 8)
    bad = torch.zeros(2, 1, 1, 9)
    bad[..., 5:] = float("-inf")
    with pytest.raises((ValueError, TypeError)):
        PREFILL_BACKEND.decode(q, k, v, mask=bad)


# --------------------------------------------------------------------------- dispatch count
def test_backend_called_once_per_forward_regardless_of_batch(monkeypatch):
    """成本必须 ∝ B×L，不能 ∝ B 次派发（D3 那版逐序列的病根）。
    这里用计数替身：B=1 与 B=8 都必须只调用后端一次/层。"""
    model = MiniLLM(MiniLLMConfig(vocab_size=64, max_position_embeddings=64, n_embd=32,
                                  n_head=4, n_kv_head=2, n_layer=2, dropout=0.0, bias=False))
    model.eval()
    calls = {"n": 0}
    layer = model.layers[0].self_attn
    for backend, method in ((layer.backend_prefill, "prefill"), (layer.backend_decode, "decode")):
        real = getattr(backend, method)

        def counting(*a, _real=real, **kw):
            calls["n"] += 1
            return _real(*a, **kw)

        monkeypatch.setattr(backend, method, counting)

    with torch.no_grad():
        for B in (1, 8):
            calls["n"] = 0
            model(torch.randint(0, 64, (B, 5)))
            assert calls["n"] == model.config.n_layer, (
                f"B={B}: 每层应恰好 1 次后端调用，实际 {calls['n']} 次 / {model.config.n_layer} 层"
            )


# --------------------------------------------------------------------------- no silent fallback
def test_layer_does_not_swallow_backend_errors(monkeypatch):
    model = MiniLLM(MiniLLMConfig(vocab_size=64, max_position_embeddings=64, n_embd=32,
                                  n_head=4, n_kv_head=2, n_layer=1, dropout=0.0, bias=False))
    model.eval()
    layer = model.layers[0].self_attn

    def boom(*a, **kw):
        raise NotImplementedError("backend cannot do this shape")

    monkeypatch.setattr(layer.backend_prefill, "prefill", boom)
    with pytest.raises(NotImplementedError, match="cannot do this shape"):
        with torch.no_grad():
            model(torch.randint(0, 64, (1, 6)))


# --------------------------------------------------------------------------- default path frozen
def _reference_attention(q, k, v, *, is_causal=False, mask=None):
    """重构前 ``model/layers.py`` 第 4 步的冻结副本（SDPA 那一版），逐 op 复刻。"""
    need_mask = mask is not None
    attn_mask, causal = None, False
    if need_mask:
        allowed = mask
        attn_mask = allowed.masked_fill(~allowed.any(dim=-1, keepdim=True), True)
    else:
        causal = is_causal
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal,
        enable_gqa=(q.size(1) != k.size(1)),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_backend_is_bit_identical_to_pre_refactor(dtype):
    for name, B, H, Hkv, T, L, D, kwargs in _shapes():
        if T <= 1:
            continue  # decode is hand_baddbmm now; it is compared against the oracle
        q, k, v = _mk(B, H, Hkv, T, L, D, dtype=dtype, seed=11)
        got = PREFILL_BACKEND.prefill(q, k, v, **kwargs)
        ref = _reference_attention(q, k, v, **kwargs)
        assert torch.equal(got, ref), f"{name}: 默认后端与重构前不再是逐 bit 相同"
