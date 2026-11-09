"""Qwen3.5 full-attention layers (W5).

Qwen3.5-0.8B mixes 18 gated-delta-net layers with 6 full-attention layers
(``layer_types`` = ``LLLF`` x 6). W5 builds the full-attention half only, as a new
module rather than a flag on ``model/layers.py::Attention`` (D-W5-1), because three
things differ from the Qwen3-0.6B path and each changes the numbers:

* **Output gate.** ``q_proj`` emits query *and* gate (``2 * n_head * head_dim``); the
  two are split **per head** (``view(B, T, H, 2D).chunk(2, -1)``), and the sigmoid gate
  multiplies the attention output after attention and before ``o_proj``.
* **Zero-centered q/k norms.** ``Qwen3_5RMSNorm`` multiplies by ``(1 + weight)`` with
  ``weight`` initialized to zeros, and does the whole reduction in fp32.
  ``model/layers.py::RMSNorm`` multiplies by ``weight`` directly, so it cannot express
  this without changing the 0.6B path.
* **Partial RoPE.** Only the first ``rotary_dim = int(head_dim * 0.25) = 64`` of the 256
  head dimensions rotate; the tail passes through.

``Qwen35RMSNormGated`` is the linear-attention norm (``weight`` ones, norm before gate,
``silu(gate)`` in fp32). W5 only implements and unit-tests it -- W6 assembles it.

Scope (W5 plan §1): single-layer inference. A growing ``KVcache`` is supported because
W6's decode needs it. Chunked prefill (``past_len > 0`` with ``T > 1``), explicit
``attention_mask``, the packed-prefill container and the slab/paged decode containers
raise ``NotImplementedError`` -- silently treating them as a plain cache would produce
wrong numbers, and there is no fallback implementation to pick instead (D4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import DECODE_BACKEND, PREFILL_BACKEND
from model.config import MiniLLMConfig
from model.layers import MLP, apply_rotary_emb_partial, rope_cos_sin
from model.packed import as_packed_prefill
from model.slab import as_decode_container

__all__ = [
    "Qwen35RMSNorm",
    "Qwen35RMSNormGated",
    "Qwen35Attention",
    "Qwen35FullAttentionBlock",
]


# ============================================================
# Norms (HF Qwen3_5RMSNorm / Qwen3_5RMSNormGated semantics)
# ============================================================


class Qwen35RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * (1 + weight)``, computed in fp32.

    HF (``Qwen3_5RMSNorm``) initializes ``weight`` to **zeros**, so a fresh norm is a
    plain RMSNorm and the checkpoint's weight is a *delta* from that. The ``(1 + w)``
    multiplication and the reduction both happen in fp32, and the result is cast back to
    the input dtype (``output.type_as(x)``). This is deliberately a different class from
    ``model.layers.RMSNorm`` (which multiplies by ``weight``): mixing the two conventions
    would rescale every q/k head by ``1 + w`` vs ``w`` -- a silent, uniform error.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen35RMSNormGated(nn.Module):
    """``weight * norm(x) * silu(gate)``, norm before gate, fp32 reduction.

    HF (``Qwen3_5RMSNormGated``) is the gated-delta-net output norm: ``weight`` starts at
    **ones** and multiplies directly (no ``1 +``), the normed value is cast back to the
    input dtype *before* the weight multiply, and the gate's ``silu`` is evaluated in
    fp32. W5 implements and unit-tests it; nothing in the full-attention block uses it
    (W6's linear-attention layers do).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate.
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ============================================================
# Attention
# ============================================================


class Qwen35Attention(nn.Module):
    """Gated, partial-RoPE, GQA attention for a Qwen3.5 full-attention layer.

    ``forward`` is only the composition of the four helpers, so the W5 golden alignment
    test can ask the *same* code for each intermediate stage instead of re-deriving the
    arithmetic:

    ``_project`` -> ``q_norm``/``k_norm`` -> ``_rope`` -> ``_attend`` -> ``_merge``
    """

    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        if not config.attn_output_gate:
            raise ValueError(
                "Qwen35Attention requires config.attn_output_gate=True: this class only "
                "implements Qwen3.5's full-attention layer (q_proj emits query + gate). "
                "Qwen3/Qwen3.5 linear-attention layers have no output gate; use "
                "model.layers.Attention for those."
            )
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim or config.n_embd // config.n_head
        self.rotary_dim = config.rotary_dim
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))

        # q_proj carries query and gate: HF views it as (B, T, n_head, 2 * head_dim) and
        # chunks per head, so one matrix has 2x the rows of a plain q_proj.
        self.q_proj = nn.Linear(
            config.n_embd, self.n_head * self.head_dim * 2, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias
        )
        self.o_proj = nn.Linear(
            self.n_head * self.head_dim, config.n_embd, bias=config.bias
        )
        self.q_norm = Qwen35RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, config.norm_eps)

        # Same two constants as model/layers.py::Attention (D4); no knob, no fallback.
        # Qwen3.5 sets attention_dropout=0.0 and W5 is inference-only, so no dropout is
        # threaded through the backends.
        self.backend_prefill = PREFILL_BACKEND
        self.backend_decode = DECODE_BACKEND

    # ------------------------------------------------------------ helpers

    def _project(self, x: torch.Tensor):
        """``(q, gate, k, v)`` in ``(B, T, heads, head_dim)``, pre-norm.

        The per-head ``chunk`` is load-bearing: the flat ``q_proj`` output is laid out as
        ``[head0 query, head0 gate, head1 query, head1 gate, ...]``, **not** as
        ``[all queries, all gates]``. ``torch.cat([q, gate], dim=-1).reshape(B, T, -1)``
        inverts this layout exactly (pinned in ``tests/unit/test_qwen35_attention.py``).
        """
        batch, seq, _ = x.shape
        q_gate = self.q_proj(x).view(batch, seq, self.n_head, 2 * self.head_dim)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(x).view(batch, seq, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.n_kv_head, self.head_dim)
        return q, gate, k, v

    def _rope(self, q: torch.Tensor, k: torch.Tensor, cos, sin):
        """Partial RoPE over ``rotary_dim``; the tail of every head is untouched."""
        rotary_dim = self.rotary_dim
        return (
            apply_rotary_emb_partial(q, cos, sin, rotary_dim),
            apply_rotary_emb_partial(k, cos, sin, rotary_dim),
        )

    def _attend(self, q, k, v, idx: int = 0, past_keys_values=None):
        """Attention over ``(B, T, H, D)``; returns ``(B, T, H, D)``.

        ``T > 1`` is a full prefill (``past_len == 0``, causal built by the kernel);
        ``T == 1`` is decode and reads whatever is in ``KVcache`` after appending this
        step's K/V. Chunked prefill needs an offset causal mask that W5 does not build, so
        it raises instead of silently treating the past as visible or invisible.
        """
        seq = q.shape[1]
        past_len = 0 if past_keys_values is None else past_keys_values.past_len(idx)
        if seq > 1 and past_len > 0:
            raise NotImplementedError(
                "Qwen35Attention has no chunked-prefill path (past_len > 0 with T > 1): "
                "it would need the offset causal mask from model/layers.py::Attention, "
                "which W5 deliberately does not copy. Prefill in one call or decode one "
                "token at a time."
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past_keys_values is not None:
            past_k, past_v = past_keys_values.get_kv(idx)
            if past_k is not None:
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            past_keys_values.set_kv(idx, (k, v))

        if seq > 1:
            y = self.backend_prefill.prefill(
                q, k, v, is_causal=(past_len == 0), mask=None
            )
        else:
            y = self.backend_decode.decode(q, k, v)
        return y.transpose(1, 2)

    def _merge(self, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``o_proj(y * sigmoid(gate))`` -- the gate multiplies *after* attention.

        ``gate = -inf`` gives exactly zero here (``sigmoid(-inf) == 0``, and the
        checkpoint's ``attention_bias=False`` makes ``o_proj(0) == 0``). That is the unit
        test pinning the sigmoid's position: applied before attention, the same gate would
        poison the softmax instead of silencing the head's output.
        """
        batch, seq = y.shape[0], y.shape[1]
        width = self.n_head * self.head_dim
        gated = y.reshape(batch, seq, width) * torch.sigmoid(gate.reshape(batch, seq, width))
        return self.o_proj(gated)

    # ------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        idx: int = 0,
        past_keys_values=None,
        cos=None,
        sin=None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "Qwen35Attention does not take an attention_mask yet: W5 runs "
                "single-sequence prefill/decode, and padding support would have to "
                "follow model/layers.py::Attention's mask construction (W6)."
            )
        if as_packed_prefill(past_keys_values) is not None:
            raise NotImplementedError(
                "Qwen35Attention does not support the packed-prefill container yet: its "
                "varlen path lives in model/layers.py::Attention (W4.5) and has no "
                "Qwen3.5 wiring (W6)."
            )
        if as_decode_container(past_keys_values) is not None:
            raise NotImplementedError(
                "Qwen35Attention does not support the slab/paged decode containers yet "
                "(KvSlab.fill_from / PagedKV are W6/W7 work); pass a growing KVcache."
            )

        _, seq, _ = x.shape
        past_len = 0 if past_keys_values is None else past_keys_values.past_len(idx)
        if past_len > 0 and seq > 1:
            raise NotImplementedError(
                "Qwen35Attention has no chunked-prefill path (past_len > 0 with T > 1); "
                "see _attend()."
            )
        if cos is None or sin is None:
            cos, sin = rope_cos_sin(
                self.head_dim,
                self.rope_theta,
                past_len,
                seq,
                x.device,
                x.dtype,
                rotary_dim=self.rotary_dim,
            )

        q, gate, k, v = self._project(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._rope(q, k, cos, sin)
        y = self._attend(q, k, v, idx, past_keys_values)
        return self._merge(y, gate)


# ============================================================
# Block
# ============================================================


class Qwen35FullAttentionBlock(nn.Module):
    """Pre-norm residual block: ``x + attn(norm(x))``, ``x + mlp(norm(x))``.

    Same shape and forward signature as ``model/layers.py::TransformerBlock`` so a block
    can be dropped where the model expects one; the MLP is reused verbatim (Qwen3.5's
    SwiGLU has the same layout). The two norms are the zero-centered Qwen3.5 ones, and
    they are applied at ``n_embd`` width, not per head -- q_norm/k_norm live in
    ``self_attn`` and act on ``head_dim``.
    """

    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.input_layernorm = Qwen35RMSNorm(config.n_embd, config.norm_eps)
        self.self_attn = Qwen35Attention(config)
        self.post_attention_layernorm = Qwen35RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        idx: int = 0,
        past_keys_values=None,
        cos=None,
        sin=None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), attention_mask, idx, past_keys_values, cos, sin
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
