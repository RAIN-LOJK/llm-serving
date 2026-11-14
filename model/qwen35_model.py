"""Whole-model assembly and per-request state for Qwen3.5-0.8B's text decoder (W6 Step 4).

Qwen3.5-0.8B is 24 layers on a repeating ``LLLF`` schedule: 18 gated-delta-net
(``linear_attention``) layers and 6 full-attention layers, with a full-attention layer at
every fourth position (3, 7, 11, ..., 23). This module puts the W5 full-attention block,
the W6 Step-2 GDN block, the zero-centered final norm and the embedding/lm_head together,
and owns the one per-request state object (D8 decision 5, D-W6-2).

Three conventions are load-bearing and each has its own test in
``tests/unit/test_qwen35_model.py``:

* **``lm_head`` shares the embedding Parameter** (``lm_head.weight is
  token_embedding.weight``). Qwen3.5-0.8B ties them; a copy would cost another
  ``248320 x 1024 x 4`` bytes (~1 GB) of fp32 weights and would let the two diverge.
* **``state=None`` is a one-shot forward.** The state is created, used and dropped, so a
  caller that wants decode must pass a ``Qwen35State`` and keep it. A provided state is
  mutated in place through the setters -- the model never reassigns ``state.position`` or
  stores tensors behind the interface's back.
* **``logits_to_keep`` slices before ``lm_head``** (P-13). Projecting all ``T`` rows to
  the 248320-wide vocabulary is pure bandwidth for rows nobody reads, and on a long
  prefill it is the difference between a small matmul and a multi-GiB temporary.

Scope (D8 decision 6): inference only, single sequence, no padding. There is no
targets/loss argument, no packed-prefill, no ``KvSlab``/``PagedKV`` container and no
``attention_mask`` path; each of those raises instead of silently taking the dense
growing-cache path, because a silent fallback would compute different numbers (D4).
Whole-model loading lives in ``model/loading.py::load_qwen35_model``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import MiniLLMConfig
from model.layers import KVcache, rope_cos_sin
from model.qwen35 import Qwen35FullAttentionBlock, Qwen35RMSNorm
from model.qwen35_gdn import Qwen35GDNBlock

__all__ = ["Qwen35State", "Qwen35Model"]

_FULL_ATTENTION = "full_attention"
_LINEAR_ATTENTION = "linear_attention"

#: The five ``linear_*`` fields that describe a linear-attention layer (same set the
#: loader uses). ``Qwen35State`` reads them from the config, never from globals.
_LINEAR_FIELDS = (
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)


def _linear_dims(config: MiniLLMConfig) -> tuple[int, int, int, int, int]:
    """``(conv_dim, num_v_heads, head_k_dim, head_v_dim, kernel)`` from the config.

    Raises instead of defaulting: a state validated against guessed head geometry would
    accept a tensor the layer cannot consume.
    """
    missing = [name for name in _LINEAR_FIELDS if getattr(config, name, None) is None]
    if missing:
        raise ValueError(
            f"config is missing {', '.join(missing)}; they describe every linear-attention "
            "layer (docs/decisions.md D8) and are needed to validate the state shapes"
        )
    num_k_heads = int(config.linear_num_key_heads)
    num_v_heads = int(config.linear_num_value_heads)
    head_k_dim = int(config.linear_key_head_dim)
    head_v_dim = int(config.linear_value_head_dim)
    kernel = int(config.linear_conv_kernel_dim)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    return 2 * key_dim + value_dim, num_v_heads, head_k_dim, head_v_dim, kernel


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class Qwen35State:
    """The only per-request state interface for the hybrid model (D-W6-2, D8 decision 5).

    All per-request state lives here and is reached only through the getters/setters:
    the model never keeps a second copy, and W7's ``StatePool`` replaces *storage and
    lifetime*, not the layer code. Concretely, one state holds

    * ``self.kv`` -- a ``model.layers.KVcache`` for the 6 full-attention layers
      (``(B, n_kv_head, past, head_dim)`` K/V per layer);
    * one ``conv_state`` per linear-attention layer, ``(B, conv_dim, k - 1)``. ``k - 1``
      is 3 for 0.8B: the minimal sufficient state. HF 5.20.2's ``DynamicCache`` stores
      ``kernel_size`` = 4 columns; the extra oldest column never enters the last
      convolution window, so the two lengths are output-equivalent and W6 freezes 3
      (P-49). The conv state follows the model dtype;
    * one ``recurrent_state`` per linear-attention layer, ``(B, num_v_heads, K, V)``
      **fp32 by contract** -- HF keeps the delta-rule matrix in fp32, and a bf16 state
      would silently change the recurrence's dtype (so the setter rejects it).

    ``position`` is the per-request cursor: it advances by the number of tokens each
    forward consumed and is what ``past_len()`` reports. ``layer`` is accepted by
    ``past_len`` only for API symmetry with ``KVcache`` -- one request has one cursor, so
    every layer sees the same value. ``cull(keep)`` is the W3-style batch shrink (drop
    finished rows); it does not rewind ``position``.
    """

    def __init__(
        self,
        config: MiniLLMConfig,
        batch: int = 1,
        device=None,
        dtype=None,
    ):
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise TypeError(f"batch must be an int, got {batch!r}")
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        self.config = config
        self.batch = batch
        self.position = 0
        # Allocation intent, recorded for W7's pool. The setters do not cast to these:
        # the conv state follows the layer's input dtype and the recurrent state is fp32.
        self.device = device
        self.dtype = dtype
        self._kv = KVcache(config.n_layer)
        self._conv: list[torch.Tensor | None] = [None] * config.n_layer
        self._recurrent: list[torch.Tensor | None] = [None] * config.n_layer

    # ---------------------------------------------------------------- internals

    @property
    def kv(self) -> KVcache:
        """The full-attention cache, exposed for ``Qwen35FullAttentionBlock``."""
        return self._kv

    def _check_layer(self, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer must be an int, got {layer!r}")
        if not 0 <= layer < self.config.n_layer:
            raise IndexError(
                f"layer={layer} is out of range for n_layer={self.config.n_layer}"
            )
        return layer

    def _check_linear_kind(self, layer: int, name: str) -> None:
        kind = self.config.layer_types[layer]
        if kind != _LINEAR_ATTENTION:
            raise ValueError(
                f"layer {layer} is {kind!r}; {name} belongs only to "
                f"{_LINEAR_ATTENTION!r} layers"
            )

    # ------------------------------------------------------------- cursor / KV

    def past_len(self, layer: int = 0) -> int:
        """Tokens already consumed by this request (``position``).

        ``layer`` is validated for API symmetry with ``KVcache.past_len``; the hybrid
        model has a single cursor, so the value is layer-independent.
        """
        self._check_layer(layer)
        return self.position

    def get_kv(self, layer: int):
        """``(k, v)`` for a full-attention layer, or ``(None, None)`` before any token."""
        return self._kv.get_kv(self._check_layer(layer))

    def set_kv(self, layer: int, kv) -> None:
        """Store the ``(k, v)`` tuple ``KVcache.set_kv`` takes."""
        self._kv.set_kv(self._check_layer(layer), kv)

    # --------------------------------------------------- linear-attention state

    def conv_state(self, layer: int) -> torch.Tensor | None:
        self._check_layer(layer)
        return self._conv[layer]

    def set_conv_state(self, layer: int, state) -> None:
        """Store a linear layer's ``(batch, conv_dim, k - 1)`` conv state, dtype as given.

        No dtype cast: the state is the *pre-activation* convolution input, so it follows
        the model dtype (bf16 model -> bf16 conv state), exactly as
        ``causal_conv_prefill``/``causal_conv_decode`` return it.
        """
        self._check_layer(layer)
        self._check_linear_kind(layer, "conv_state")
        conv_dim, _, _, _, kernel = _linear_dims(self.config)
        expected = (self.batch, conv_dim, kernel - 1)
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"conv_state for layer {layer} must be a Tensor, got {type(state).__name__}"
            )
        if tuple(state.shape) != expected:
            raise ValueError(
                f"conv_state for layer {layer} must be (batch, conv_dim, k - 1) = "
                f"{expected}, got {tuple(state.shape)}"
            )
        self._conv[layer] = state

    def recurrent_state(self, layer: int) -> torch.Tensor | None:
        self._check_layer(layer)
        return self._recurrent[layer]

    def set_recurrent_state(self, layer: int, state) -> None:
        """Store a linear layer's fp32 ``(batch, num_v_heads, K, V)`` delta-rule state.

        fp32 is enforced rather than cast: the recurrent endpoint runs the delta rule in
        fp32 and the operator returns fp32, so a bf16 state here means a caller
        downcast it somewhere -- casting it back would hide that, and using it as-is
        would silently change the recurrence's arithmetic.
        """
        self._check_layer(layer)
        self._check_linear_kind(layer, "recurrent_state")
        _, num_v_heads, head_k_dim, head_v_dim, _ = _linear_dims(self.config)
        expected = (self.batch, num_v_heads, head_k_dim, head_v_dim)
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"recurrent_state for layer {layer} must be a Tensor, got "
                f"{type(state).__name__}"
            )
        if tuple(state.shape) != expected:
            raise ValueError(
                f"recurrent_state for layer {layer} must be "
                f"(batch, num_v_heads, K, V) = {expected}, got {tuple(state.shape)}"
            )
        if state.dtype != torch.float32:
            raise ValueError(
                f"recurrent_state for layer {layer} must be float32 (the delta rule's "
                f"contract, HF behaviour), got {state.dtype}"
            )
        self._recurrent[layer] = state

    # -------------------------------------------------------------- bookkeeping

    def advance(self, n: int) -> None:
        """Move the per-request cursor by ``n`` consumed tokens (never negative)."""
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {n!r}")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.position += n

    def _checked_keep(self, keep) -> torch.Tensor:
        if not isinstance(keep, torch.Tensor):
            raise TypeError(f"keep must be a Tensor of row indices, got {type(keep).__name__}")
        if keep.dim() != 1:
            raise ValueError(f"keep must be 1-D, got shape {tuple(keep.shape)}")
        if keep.dtype != torch.long:
            raise ValueError(f"keep must be int64 (torch.long), got {keep.dtype}")
        if keep.numel():
            low = int(keep.min())
            high = int(keep.max())
            if low < 0 or high >= self.batch:
                raise ValueError(
                    f"keep indices must be in [0, {self.batch}), got [{low}, {high}]"
                )
        return keep

    def cull(self, keep: torch.Tensor) -> None:
        """Keep only the batch rows in the 1-D ``keep`` index tensor.

        Mirrors ``KVcache.cull`` (used by ``llm_utils/inference.py`` for finished
        requests) and extends it to the linear-attention states. ``position`` is not
        touched: culling rows does not rewind the surviving rows' cursor. ``batch`` is
        updated to the number of surviving rows so the shape checks keep accepting the
        next layer output, and every tensor keeps its device and dtype.
        """
        keep = self._checked_keep(keep)
        device = next(
            (
                tensor.device
                for tensor in (
                    *self._conv,
                    *self._recurrent,
                    *(t for kv in self._kv.kvs if kv is not None for t in kv),
                )
                if tensor is not None
            ),
            keep.device,
        )
        keep = keep.to(device)
        self._kv.cull(keep)
        for states in (self._conv, self._recurrent):
            for idx, tensor in enumerate(states):
                if tensor is not None:
                    states[idx] = tensor.index_select(0, keep)
        self.batch = int(keep.numel())

    # ------------------------------------------------------------------- bytes

    def linear_state_bytes(self) -> int:
        """Conv + recurrent bytes over the linear-attention layers (W6 state bytes, D8)."""
        total = 0
        for idx, kind in enumerate(self.config.layer_types):
            if kind != _LINEAR_ATTENTION:
                continue
            for tensor in (self._conv[idx], self._recurrent[idx]):
                if tensor is not None:
                    total += _tensor_nbytes(tensor)
        return total

    def kv_bytes(self) -> int:
        """K + V bytes over the full-attention layers."""
        total = 0
        for idx, kind in enumerate(self.config.layer_types):
            if kind != _FULL_ATTENTION:
                continue
            k, v = self._kv.get_kv(idx)
            for tensor in (k, v):
                if tensor is not None:
                    total += _tensor_nbytes(tensor)
        return total

    def nbytes(self) -> int:
        """Every stored-state byte, counted once (conv, recurrent, K and V)."""
        return self.linear_state_bytes() + self.kv_bytes()


class Qwen35Model(nn.Module):
    """The 24-layer ``LLLF x 6`` decoder: embedding -> hybrid blocks -> norm -> lm_head.

    ``config.layer_types`` selects the block for each position -- ``linear_attention``
    gives :class:`~model.qwen35_gdn.Qwen35GDNBlock` (it returns the carried state),
    ``full_attention`` gives :class:`~model.qwen35.Qwen35FullAttentionBlock` (it writes
    ``state.kv`` directly). The embedding and ``lm_head`` share one Parameter
    (``lm_head.weight = token_embedding.weight``): tying is what the 0.8B checkpoint
    ships, and a copy would add ~1 GB of fp32 weights.

    ``forward(tokens, state=None, ...)`` is a one-shot forward when ``state`` is None --
    the created ``Qwen35State`` is discarded, so the next call starts from position 0.
    A provided state is mutated **in place** through its setters: full-attention layers
    read/write ``state.kv``, GDN layers consume ``state.conv_state(i)`` /
    ``state.recurrent_state(i)`` and write the returned pair back. That is the whole
    point of keeping storage outside the blocks (D-W6-2): W7's ``StatePool`` swaps the
    backing store without touching layer code.

    Inference only: no ``targets``/loss, exactly one sequence per call (``B == 1``), no
    padding mask, no packed prefill, no slab/paged containers. Each unsupported input
    raises instead of falling back -- a dense growing-cache fallback would compute
    different numbers, not merely slower ones (D4). ``logits_to_keep`` is applied before
    ``lm_head`` (P-13).
    """

    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        for idx, kind in enumerate(config.layer_types):
            if kind == _LINEAR_ATTENTION:
                layers.append(Qwen35GDNBlock(config))
            elif kind == _FULL_ATTENTION:
                layers.append(Qwen35FullAttentionBlock(config))
            else:
                # MiniLLMConfig already rejects unknown kinds; this only fires if the
                # list was mutated after construction. Guessing a block would be worse.
                raise ValueError(
                    f"layer_types[{idx}]={kind!r} is not a supported layer type; "
                    f"expected {_FULL_ATTENTION!r} or {_LINEAR_ATTENTION!r}"
                )
        self.layers = nn.ModuleList(layers)

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.norm = Qwen35RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # One Parameter in two modules, not a copy: 248320 x 1024 fp32 is ~1 GB.
        self.lm_head.weight = self.token_embedding.weight

        self.head_dim = config.head_dim or config.n_embd // config.n_head
        self.rotary_dim = config.rotary_dim
        self.rope_theta = float(config.rope_theta)

        self._init_weights()

    def _init_weights(self) -> None:
        """Same 0.02-std init as ``MiniLLM`` (a random 24-layer stack stays sane).

        Every tensor is overwritten by ``load_qwen35_model``; this only matters for the
        unit tests that build a tiny random model.
        """
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        tokens: torch.Tensor,
        state: Qwen35State | None = None,
        logits_to_keep: int | None = None,
        attention_mask=None,
    ):
        """``(logits, None)`` for one sequence, optional ``state`` carried across calls.

        ``logits`` is ``(B=1, T, vocab_size)``, or ``(1, logits_to_keep, vocab_size)``
        when ``logits_to_keep`` is given (the slice happens before ``lm_head``, P-13).
        The second element is always ``None``: this model has no training path.
        """
        if attention_mask is not None:
            raise NotImplementedError(
                "Qwen35Model does not take an attention_mask: W6 runs one unpadded "
                "sequence, and padding support would need a mask path in every block "
                "(W7 scheduling work)."
            )
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be a Tensor, got {type(tokens).__name__}")
        if tokens.dim() != 2:
            raise ValueError(
                f"tokens must be (B, T), got shape {tuple(tokens.shape)}"
            )
        batch, seq_len = tokens.shape
        if batch > 1:
            raise NotImplementedError(
                f"Qwen35Model got batch={batch}: single sequence; W6 has no "
                "batch/padding path. Forward one sequence at a time."
            )
        if batch < 1 or seq_len < 1:
            raise ValueError(f"tokens must be a non-empty (B, T), got {(batch, seq_len)}")
        if tokens.dtype != torch.long:
            raise ValueError(f"tokens must be int64 (torch.long), got {tokens.dtype}")
        # Validate the slice before touching the state: the layers mutate it in place, so
        # a late error would leave a half-advanced request behind.
        if logits_to_keep is not None:
            if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):
                raise TypeError(
                    f"logits_to_keep must be an int, got {logits_to_keep!r}"
                )
            if logits_to_keep < 1:
                raise ValueError(
                    f"logits_to_keep must be >= 1, got {logits_to_keep}"
                )

        if state is None:
            state = Qwen35State(
                self.config,
                batch=batch,
                device=tokens.device,
                dtype=self.token_embedding.weight.dtype,
            )
        else:
            if not isinstance(state, Qwen35State):
                raise TypeError(
                    "state must be a Qwen35State (the hybrid model's per-request state), "
                    f"got {type(state).__name__}"
                )
            if state.batch != batch:
                raise ValueError(
                    f"state.batch={state.batch} does not match tokens batch={batch}; a "
                    "state belongs to one request"
                )
            if state.config.n_layer != self.config.n_layer:
                raise ValueError(
                    f"state was built for n_layer={state.config.n_layer}, this model has "
                    f"n_layer={self.config.n_layer}"
                )

        past_len = state.past_len(0)
        if past_len > 0 and seq_len > 1:
            raise NotImplementedError(
                f"Qwen35Model got past_len={past_len} with T={seq_len}: cross-call "
                "chunked prefill is W7 (P-56). Decode one token at a time, or prefill "
                "the whole prefix in a single call with past_len=0."
            )

        x = self.token_embedding(tokens)
        # One table for the whole stack (P-18); the linear layers ignore it.
        cos, sin = rope_cos_sin(
            self.head_dim,
            self.rope_theta,
            past_len,
            seq_len,
            x.device,
            x.dtype,
            rotary_dim=self.rotary_dim,
        )

        for idx, (kind, layer) in enumerate(zip(self.config.layer_types, self.layers)):
            if kind == _LINEAR_ATTENTION:
                x, conv, recurrent = layer(
                    x,
                    state.conv_state(idx),
                    state.recurrent_state(idx),
                    past_len,
                )
                state.set_conv_state(idx, conv)
                state.set_recurrent_state(idx, recurrent)
            elif kind == _FULL_ATTENTION:
                x = layer(x, None, idx, state.kv, cos, sin)
            else:  # pragma: no cover - __init__ already rejected this kind
                raise ValueError(
                    f"layer_types[{idx}]={kind!r} is not a supported layer type"
                )

        state.advance(seq_len)

        x = self.norm(x)
        if logits_to_keep is not None:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(x)
        return logits, None
