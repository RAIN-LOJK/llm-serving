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
``silu(gate)`` in fp32). W5 only implements and unit-tests it; W6 Step 4 assembles it with
the GDN blocks into the 24-layer hybrid model in ``model/qwen35_model.py``.

Scope (W5 plan §1): single-layer inference. A growing ``KVcache`` is supported because
W6's decode needs it. W7 Step 2 adds the two **decode containers** for ``T == 1``: the
seat arm (``model/slab.py::KvSlab`` + ``hand_baddbmm``) and the paged arm
(``model/paged.py::PagedKV`` + the Triton block-table kernel). Both address the six
full-attention layers by their **compact** index (``0..5``, D9-5), which
``Qwen35Model`` assigns at construction; the growing-cache path keeps using the actual
24-layer index. W7 Step 3 adds the **packed group prefill** container
(``model/packed.py::PackedPrefill``): one ``(1, total_tokens)`` token axis, varlen
attention per request segment, and the same compact index for the write sinks -- the
packed layer calls ``packed.write(self.layer_index, k, v)``, so the sinks it is given are
indexed compactly too (``PagedWriteSink``/``SlabWriteSink`` directly, no 24->6 adapter:
the packed path never sees the actual layer index).

Chunked prefill (``past_len > 0`` with ``T > 1``), explicit ``attention_mask``, and a
decode container with ``T > 1`` raise ``NotImplementedError`` -- silently treating them as
a plain cache would produce wrong numbers, and there is no fallback implementation to pick
instead (D4). The packed path is not a fallback either: a container that is neither a
packed prefill nor a decode container keeps the growing-cache path.
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

    ``layer_index`` is this block's **compact** full-attention index (``0..5`` for the
    24-layer ``LLLF x 6`` stack, D9-5), assigned by ``Qwen35Model``. The decode
    containers and the packed-prefill write sinks address the six full-attention layers
    by that index; the growing ``KVcache`` path keeps addressing them by the actual
    24-layer index (``idx``). A block built without one (the single-layer unit tests)
    still runs the growing-cache path and rejects a decode/packed container instead of
    writing into layer 0 of the wrong stack.
    """

    def __init__(self, config: MiniLLMConfig, layer_index: int | None = None):
        super().__init__()
        if layer_index is not None:
            if isinstance(layer_index, bool) or not isinstance(layer_index, int):
                raise TypeError(
                    f"layer_index must be an int (the compact full-attention index), got "
                    f"{layer_index!r}"
                )
            if layer_index < 0:
                raise ValueError(f"layer_index must be >= 0, got {layer_index}")
        self.layer_index = layer_index
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

    def _attend_decode(self, decode, q, k, v):
        """One decode token per row through a seat/paged container; ``(B, T, H, D)`` in/out.

        ``decode`` is a ``KvSlab`` (seat arm: write, then ``get`` + ``hand_baddbmm`` with
        the container's prebuilt additive bias) or a ``PagedKV`` (production arm: write,
        then the Triton block-table kernel, which reads the pool and masks by its int32
        seqlen). The container owns the mask, so nothing here reads ``past_len``; the
        position each row writes at is ``decode.pos[row]``, and this step's column is
        already allowed (``build_mask(next_token=True)`` ran once in
        ``Qwen35Model.forward_decode_batch``).

        The layer address is the **compact** full-attention index (D9-5), never ``idx``:
        a 24-layer ``idx`` would write into another layer's slot of a 6-layer container.
        """
        if self.layer_index is None:
            raise ValueError(
                "a slab/paged decode container needs this layer's compact full-attention "
                "index (0..5), but this Qwen35Attention was built without one; "
                "Qwen35Model assigns it at construction. A directly-built single layer "
                "has nowhere to write in a 6-layer container."
            )
        layer = self.layer_index
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        decode.write(layer, k, v)
        if getattr(decode, "block_table_decode", False):
            # W4.5 (D7): the paged container owns the kernel; it has no dense gather and
            # no additive bias, so ``attend`` is the whole call.
            y = decode.attend(layer, q)
        else:
            k_all, v_all = decode.get(layer)
            # The seat mask is identical for every layer, so the container already turned
            # it into the additive bias once at the step boundary (``build_mask``).
            y = self.backend_decode.decode(q, k_all, v_all, bias=decode.bias)
        return y.transpose(1, 2)

    def _attend_packed(self, packed, q, k, v):
        """Packed group prefill: write the group's K/V, then one varlen call; ``(1, T, H, D)``.

        The container slices the packed token axis by ``cu_seqlens`` and calls each
        request's one-shot ``set_kv`` with that request's ``(1, n_kv_head, length, D)``
        K/V, so the layer never materialises a padded ``(B, T_max)`` batch and never grows
        a ``KVcache``. ``attend`` returns the packed layout ``(1, T, H, D)`` directly (the
        varlen backend has no batch dimension), so unlike ``_attend`` there is no
        transpose back.

        The layer address is the **compact** full-attention index (D9-5), exactly like
        the decode containers: a 24-layer ``idx`` would miss every 6-layer sink.
        """
        layer = self.layer_index
        if layer is None:  # pragma: no cover - forward() rejects this before projecting
            raise ValueError(
                "the packed-prefill path needs this layer's compact full-attention index "
                "(0..5), but this Qwen35Attention was built without one"
            )
        packed.write(layer, k, v)
        return packed.attend(q, k, v)

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
        packed = as_packed_prefill(past_keys_values)
        if packed is not None:
            if x.shape[0] != 1 or x.shape[1] != int(packed.total_tokens):
                raise ValueError(
                    "the packed-prefill path is one token axis of packed.total_tokens "
                    f"tokens (B == 1); got x={tuple(x.shape)} for "
                    f"total_tokens={packed.total_tokens}. Pack every request's tokens on "
                    "that axis and pass cu_seqlens; do not pad a (B, T_max) batch."
                )
            if cos is None or sin is None:
                raise ValueError(
                    "the packed-prefill path needs per-token cos/sin from the model (each "
                    "request has its own 0-based position axis); do not call the layer "
                    "directly with a PackedPrefill"
                )
            if self.layer_index is None:
                raise ValueError(
                    "the packed-prefill path needs this layer's compact full-attention "
                    "index (0..5) to address the group's one-shot write sinks, but this "
                    "Qwen35Attention was built without one; Qwen35Model assigns it at "
                    "construction. A directly-built single layer has no 6-layer sink."
                )
        decode = as_decode_container(past_keys_values)
        if decode is not None:
            if packed is not None:
                raise ValueError(
                    "a container cannot be both a packed prefill and a decode container: "
                    "packed prefill writes whole prompts, decode advances one token per row"
                )
            if x.shape[1] != 1:
                raise NotImplementedError(
                    "Qwen35Attention's slab/paged decode containers are decode-only "
                    f"(T == 1); got T={x.shape[1]}. Prefill keeps the growing KVcache "
                    "and hands over with KvSlab.fill_from() / PagedPool.write_prefill."
                )
            if cos is None or sin is None:
                raise ValueError(
                    "the slab/paged decode path needs per-row cos/sin from the model: "
                    "rows sit at different positions, so the layer cannot derive one "
                    "table from a scalar past_len"
                )

        _, seq, _ = x.shape
        if decode is None and packed is None:
            past_len = 0 if past_keys_values is None else past_keys_values.past_len(idx)
            if past_len > 0 and seq > 1:
                raise NotImplementedError(
                    "Qwen35Attention has no chunked-prefill path (past_len > 0 with "
                    "T > 1); see _attend()."
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
        if packed is not None:
            y = self._attend_packed(packed, q, k, v)
        elif decode is not None:
            y = self._attend_decode(decode, q, k, v)
        else:
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
    ``self_attn`` and act on ``head_dim``. ``layer_index`` is the compact full-attention
    index (``0..5``) the W7 decode containers address; see ``Qwen35Attention``.
    """

    def __init__(self, config: MiniLLMConfig, layer_index: int | None = None):
        super().__init__()
        self.layer_index = layer_index
        self.input_layernorm = Qwen35RMSNorm(config.n_embd, config.norm_eps)
        self.self_attn = Qwen35Attention(config, layer_index=layer_index)
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
