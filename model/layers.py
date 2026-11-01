import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from model.attention import DECODE_BACKEND, PREFILL_BACKEND
from model.config import MiniLLMConfig
from model.slab import KvSlab


# ============================================================
# RoPE（旋转位置编码）
# ============================================================

def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """
    预计算旋转位置编码的 cos 和 sin 表
    - dim: head_dim（注意力头的维度）
    - max_seq_len: 最大序列长度
    - theta: base frequency（LLaMA 用 10000.0）

    返回:
        cos: shape (max_seq_len, dim // 2)
        sin: shape (max_seq_len, dim // 2)
    """
    cos, sin = rope_cos_sin(dim, theta, 0, max_seq_len, dtype=torch.float32)
    return sin, cos


def rope_cos_sin(head_dim, theta, start, length, device=None, dtype=torch.float32):
    """cos/sin for positions [start, start+length), computed on demand.

    The table belongs to the *model*, not to each layer: 28 layers x 2 x 40960 x 64 bf16 is
    280 MB on Qwen3-0.6B, all of it duplicates. ``MiniLLM`` computes this once per forward
    and hands ``(cos, sin)`` to every block; a block called directly still gets the same
    values by computing them itself.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(start, start + length, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary_emb(x: torch.Tensor, cos, sin):
    """
    对 Q 或 K 施加旋转位置编码（GPT-NeoX / HF 约定）

    约定说明: head_dim 维拆成"前一半 x1 + 后一半 x2"配对旋转 (k, k+head_dim//2)。
    这是 HF LLaMA / TinyLlama 使用的排列 (rotate_half)。
    另一种 interleaved 约定按 (2k, 2k+1) 配对，两者差一个固定维度置换，
    与同一组权重不兼容——本项目统一用 GPT-NeoX。

    x:   shape (B, T, n_head, head_dim)
    cos: shape (T, head_dim // 2)
    sin: shape (T, head_dim // 2)

    返回:
        shape (B, T, n_head, head_dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    # (T, D/2) is the shared-table form: broadcast over batch, as every measurement before
    # the slab path did. (B, T, D/2) is the per-row form the slab needs, because rows sit
    # at different positions; the two differ only in which dim was unsqueezed.
    if cos.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)
    else:
        sin = sin.unsqueeze(2)
        cos = cos.unsqueeze(2)
    out0 = x1 * cos - x2 * sin
    out1 = x1 * sin + x2 * cos
    return torch.cat([out0, out1], dim=-1)


# ============================================================
# KVcache
# ============================================================
class KVcache:
    """
    按层缓存注意力历史。kvs[l] = (k, v)，是该层"过去所有 token"的 K/V。
    - k/v 形状: (B, n_kv_head, past_len, head_dim)   # GQA 展开之前
    - 各层 past_len 相同，从第 0 层推导即可
    """
    def __init__(self, n_layer: int):
        self.kvs = [None] * n_layer

    def get_kv(self, idx: int):
        kv = self.kvs[idx]
        return (None, None) if kv is None else kv

    def set_kv(self, idx: int, kv: tuple[torch.Tensor, torch.Tensor]):
        self.kvs[idx] = kv

    def past_len(self, idx) -> int:
        kv = self.kvs[idx]
        return 0 if kv is None else kv[0].shape[2]

    def cull(self, keep_idx):
        """批量推理：只保留 batch 中 keep_idx 这些行（EOS 后用）"""
        self.kvs = [None if kv is None else (kv[0][keep_idx], kv[1][keep_idx]) for kv in self.kvs]


# ============================================================
# Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim or config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        if config.qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.head_dim, config.norm_eps), RMSNorm(self.head_dim, config.norm_eps)
        else:
            self.q_norm, self.k_norm = nn.Identity(), nn.Identity()
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE 表由 MiniLLM 每步算一次传进来；这里只记参数，不存表（见 rope_cos_sin）
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))

        # 注意力数学住在外挂后端里（契约见 model/attention/base.py）。口径已定死（D4）：
        # T > 1 走 PREFILL_BACKEND（sdpa），T == 1 走 DECODE_BACKEND（hand_baddbmm）。
        # 没有注册表、没有配置项、没有回退目标。
        self.backend_prefill = PREFILL_BACKEND  # T > 1
        self.backend_decode = DECODE_BACKEND  # T == 1（含 past_len == 0）

    def forward(self, x, attention_mask=None, idx = 0, past_keys_values:Optional[KVcache]=None,
                cos=None, sin=None):
        B, T, C = x.shape  # (batch, seq_len, n_embd)
        # ``KvSlab.past_len`` reads a tensor back to the host (``.item()``), which is illegal
        # inside a captured graph. In slab mode the cursor is only needed by the mask, and
        # the slab owns that, so skip the read entirely.
        slab = past_keys_values if isinstance(past_keys_values, KvSlab) else None
        past_len = 0 if (slab is not None or past_keys_values is None) else past_keys_values.past_len(idx)

        # 1. Q/K/V 投影并 reshape 为多头形状
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_head, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim))
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        # 2. 对 Q/K 施加 RoPE（cos/sin 由模型层算好传入；直接调用本层时现算）
        if cos is None or sin is None:
            if slab is not None:
                raise ValueError(
                    "the KvSlab path needs per-row cos/sin from the model (rows sit at "
                    "different positions); do not call the layer directly with a slab"
                )
            cos, sin = rope_cos_sin(self.head_dim, self.rope_theta, past_len, T, x.device, x.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 2b. Fixed-slab decode: write this token into the layer's buffer and attend over
        # the whole buffer with the slab's in-place mask, so there is no per-step ``cat``
        # and the tensor addresses never change (CUDA graph needs that).
        if slab is not None:
            if T != 1:
                raise ValueError(
                    "the KvSlab path is decode-only (T == 1); prefill keeps KVcache + flash "
                    "and hands over with KvSlab.fill_from()"
                )
            slab.write(idx, k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
            k_all, v_all = slab.get(idx)
            # The mask is identical for every layer, so the slab already turned it into the
            # additive bias once at the step boundary (``build_mask``). Passing ``bias=``
            # avoids rebuilding the same (B*H_kv, G, L) tensor 28 times per step.
            y = self.backend_decode.decode(
                q.transpose(1, 2),
                k_all,
                v_all,
                bias=slab.bias,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            return self.resid_dropout(self.o_proj(y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)))

        # 3. 转置为 (B, n_head, T, head_dim) 方便矩阵乘法
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past_keys_values is not None:
            past_k, past_v = past_keys_values.get_kv(idx)
            if past_k is not None:
                k = torch.cat((past_k, k), 2)
                v = torch.cat((past_v, v), 2)
            past_keys_values.set_kv(idx, (k, v))

        # 4. Mask decision stays here, not in the backends: a full prefill (past_len == 0)
        # lets the kernel build causality internally and never materializes a T x T mask;
        # decode (T == 1) needs none unless the batch is padded; chunked prefill
        # (past_len > 0 and T > 1) and padded batches need an explicit bool mask.
        need_offset_causal = past_len > 0 and T > 1
        attn_mask = None
        is_causal = False
        if attention_mask is not None or need_offset_causal:
            if attention_mask is not None and attention_mask.shape[-1] != past_len + T:
                raise ValueError(
                    "attention_mask must cover past and current keys: "
                    f"expected {past_len + T}, got {attention_mask.shape[-1]}"
                )
            query_pos = torch.arange(past_len, past_len + T, device=x.device).view(T, 1)
            key_pos = torch.arange(past_len + T, device=x.device).view(1, -1)
            allowed = key_pos <= query_pos
            if attention_mask is not None:
                key_valid = attention_mask[:, None, None, :].to(torch.bool)
                allowed = allowed & key_valid
            else:
                allowed = allowed.view(1, 1, T, past_len + T)
            # Left-padded queries can see no valid key at all; an all-masked softmax row
            # would be NaN. Keep the row finite (its output is discarded) exactly like the
            # old manual path's all-masked-row guard.
            attn_mask = allowed.masked_fill(~allowed.any(dim=-1, keepdim=True), True)
        else:
            is_causal = past_len == 0 and T > 1

        # 5. Dispatch to the fixed backend for this phase. One call per layer per forward.
        # The call is deliberately not wrapped in try/except: a backend error must surface,
        # never silently select another implementation (there is none to select).
        phase = "prefill" if T > 1 else "decode"
        backend = self.backend_prefill if phase == "prefill" else self.backend_decode
        dropout_p = self.attn_dropout.p if self.training else 0.0
        if phase == "prefill":
            y = backend.prefill(
                q, k, v, is_causal=is_causal, mask=attn_mask, dropout_p=dropout_p
            )
        else:
            y = backend.decode(q, k, v, mask=attn_mask, dropout_p=dropout_p)

        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y))


# ============================================================
# MLP（SwiGLU）
# ============================================================

class MLP(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        n_inner = config.n_inner if config.n_inner is not None else config.n_embd * 4
        self.gate_proj = nn.Linear(config.n_embd, n_inner, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, n_inner, bias=config.bias)
        self.down_proj = nn.Linear(n_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean(-1, True) + self.eps)
        return x/rms * self.weight


# ============================================================
# Transformer Block（残差结构）
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, idx=0, past_keys_values=None, cos=None, sin=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, idx, past_keys_values, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
