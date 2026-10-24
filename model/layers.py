import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from model.config import MiniLLMConfig


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
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
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

    def forward(self, x, attention_mask=None, idx = 0, past_keys_values:Optional[KVcache]=None,
                cos=None, sin=None):
        B, T, C = x.shape  # (batch, seq_len, n_embd)
        past_len = 0 if past_keys_values is None else past_keys_values.past_len(idx)

        # 1. Q/K/V 投影并 reshape 为多头形状
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_head, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim))
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        # 2. 对 Q/K 施加 RoPE（cos/sin 由模型层算好传入；直接调用本层时现算）
        if cos is None or sin is None:
            cos, sin = rope_cos_sin(self.head_dim, self.rope_theta, past_len, T, x.device, x.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
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

        # 4. GQA: 如果 n_kv_head < n_head，重复 K/V 的头
        if self.n_kv_head != self.n_head:
            n_reps = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_reps, dim=1)
            v = v.repeat_interleave(n_reps, dim=1)

        # 5. 计算注意力分数: att = Q @ K^T / sqrt(head_dim)
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 6. 因果掩码: 上三角设为 -inf
        # 掩码必须跟 q/k 同 dtype：fp32 掩码会把 bf16 的 attn 提升成 fp32，
        # 随后 attn @ v 就是 fp32 × bf16 直接报错（W2 的 bf16 跑法会撞上）。
        mask = torch.triu(torch.full((T, past_len+T), float('-inf'), device=x.device, dtype=x.dtype), diagonal=past_len+1)
        # 输入掩码
        if attention_mask is not None:
            pad = torch.where(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -torch.inf, 0.0).to(x.dtype)
            mask = mask + pad
            # 防止attention_mask+上三角mask将整行都mask掉之后softmax出现Nan (推理左pad才会出现)
            # 还有方法是第一key置0即直接将mask[0]设置为0，但是会将原本正常的行改错即第一key应该mask掉，不过误差较小
            mask = torch.where(mask.isinf().all(-1, True), 0.0, mask)
        attn = attn + mask

        # 7. softmax + dropout
        attn = self.attn_dropout(F.softmax(attn, dim = -1))

        # 8. 注意力输出: y = att @ V，然后 transpose + reshape 回 (B, T, C)
        y = attn @ v
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)

        # 9. 输出投影 + dropout
        y = self.resid_dropout(self.o_proj(y))
        return y


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
