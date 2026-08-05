import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    freqs = 1. / theta ** (torch.arange(0, dim, 2, dtype=torch.float)/dim)
    angles = freqs.unsqueeze(0) * torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)

    return angles.sin(), angles.cos()


def apply_rotary_emb(x: torch.Tensor, cos, sin):
    """
    对 Q 或 K 施加旋转位置编码

    x:   shape (B, T, n_head, head_dim)
    cos: shape (T, head_dim // 2)
    sin: shape (T, head_dim // 2)

    返回:
        shape (B, T, n_head, head_dim)
    """
    B, T, n_head, head_dim = x.shape
    x = x.view(B, T, n_head, head_dim//2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    out0, out1 = x0 * cos - x1 * sin, x0 * sin + x1 * cos
    out = torch.stack([out0, out1], -1).view(B, T, n_head, -1)

    return out


# ============================================================
# Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 预计算 RoPE
        sin, cos = precompute_rope_freqs(self.head_dim, config.block_size)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape  # (batch, seq_len, n_embd)

        # 1. Q/K/V 投影并 reshape 为多头形状
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2. 对 Q/K 施加 RoPE
        cos, sin = self.cos[:T, :], self.sin[:T, :]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 3. 转置为 (B, n_head, T, head_dim) 方便矩阵乘法
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. GQA: 如果 n_kv_head < n_head，重复 K/V 的头
        if self.n_kv_head != self.n_head:
            n_reps = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_reps, dim=1)
            v = v.repeat_interleave(n_reps, dim=1)

        # 5. 计算注意力分数: att = Q @ K^T / sqrt(head_dim)
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 6. 因果掩码: 上三角设为 -inf
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        # 输入掩码
        if attention_mask is not None:
            pad = torch.where(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -torch.inf, 0.0)
            mask = mask + pad
            # 防止attention_mask+上三角mask将整行都mask掉之后softmax出现Nan (推理左pad才会出现)
            # 还有方法是第一key置0即直接将mask[0]设置为0，但是会将原本正常的行改错即第一key应该mask掉，不过误差较小
            mask = torch.where(mask.isinf().all(-1, True), 0.0, mask)
        attn = attn + mask

        # 7. softmax + dropout
        attn = self.attn_dropout(F.softmax(attn, dim = -1))

        # 8. 注意力输出: y = att @ V，然后 transpose + reshape 回 (B, T, C)
        y = attn @ v
        y = y.transpose(1, 2).reshape(B, T, C)

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
    def __init__(self, dim, eps = 1e-6):
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
        self.input_layernorm = RMSNorm(config.n_embd)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
