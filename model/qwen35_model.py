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

Scope (D8 decision 6): inference only, one sequence per ``forward`` call, no padding.
There is no targets/loss argument and no ``attention_mask`` path; each raises instead of
silently taking the dense growing-cache path, because a silent fallback would compute
different numbers (D4). A group prefill is not a mode of ``forward`` either: it has its own
method (:meth:`forward_packed`, W7 Step 3) with a packed token axis, per-token positions
and one state per request.

W7 Step 2 adds :meth:`Qwen35Model.forward_decode_batch`: one decode token for each of
the ``B`` compact active rows, with ``state.kv`` a decode container (``KvSlab`` or
``PagedKV``) instead of a growing ``KVcache``. The per-row RoPE positions come from the
container's ``pos`` (one table, gathered per row, like ``MiniLLM._rope_rows``); the GDN
layers read the pool's ``[0:B)`` views and write the returned states back. The existing
single-sequence ``forward`` is unchanged and rejects a decode container rather than
letting a scalar ``past_len`` stand in for per-row positions.

W7 Step 3 adds :meth:`Qwen35Model.forward_packed`: one varlen forward over a FIFO group of
whole prompts (D9-6). The 6 full-attention layers go through ``PackedPrefill`` (varlen
attention, K/V written through the group's sinks); the 18 GDN layers loop over the group's
segments, each request starting at ``past_len=0`` from empty state and writing its conv /
recurrent endpoint into its own state. Only each request's final prompt token reaches
``lm_head`` (P-13). The single-sequence ``forward`` and ``forward_decode_batch`` are
unchanged by Step 3.
Whole-model loading lives in ``model/loading.py::load_qwen35_model``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from model.config import MiniLLMConfig
from model.layers import KVcache, rope_cos_sin
from model.packed import as_packed_prefill
from model.qwen35 import Qwen35FullAttentionBlock, Qwen35RMSNorm
from model.qwen35_gdn import Qwen35GDNBlock
from model.slab import as_decode_container

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
      (``(B, n_kv_head, past, head_dim)`` K/V per layer); the constructor's optional
      ``kv`` argument lets a caller inject that container instead (W7 pooling): a plain
      ``KVcache`` for prefill, or a seat/paged decode container consumed by
      :meth:`Qwen35Model.forward_decode_batch`;
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
        kv=None,
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
        # ``kv=None`` keeps the W6 behaviour (a fresh growing ``KVcache``). W7 Step 1's
        # ``PooledQwen35State`` passes a caller-owned container so the pooled and the
        # single-sequence reference share nothing but the numbers; Step 2's
        # ``forward_decode_batch`` passes a paged/seat container through the same
        # argument. The container is duck-typed on purpose (``get_kv``/``set_kv``/
        # ``past_len``/``cull``), not isinstance-checked.
        self._kv = KVcache(config.n_layer) if kv is None else kv
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
    padding mask, no packed prefill. Each unsupported input raises instead of falling
    back -- a dense growing-cache fallback would compute different numbers, not merely
    slower ones (D4). ``logits_to_keep`` is applied before ``lm_head`` (P-13). Batched
    decode is a separate method (:meth:`forward_decode_batch`) because it needs a decode
    container and per-row positions, not a scalar ``past_len``.
    """

    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        # Actual 24-layer indices of the full-attention blocks, in stack order. Their
        # compact indices are the positions in this list (D9-5: paged/seat KV covers
        # ``[3, 7, 11, 15, 19, 23] -> 0..5``).
        self.full_attention_layers = [
            idx for idx, kind in enumerate(config.layer_types) if kind == _FULL_ATTENTION
        ]
        full_index = 0
        for idx, kind in enumerate(config.layer_types):
            if kind == _LINEAR_ATTENTION:
                layers.append(Qwen35GDNBlock(config))
            elif kind == _FULL_ATTENTION:
                # The decode containers address full-attention layers compactly; the
                # growing-KVcache path still passes the actual 24-layer ``idx``.
                layers.append(Qwen35FullAttentionBlock(config, layer_index=full_index))
                full_index += 1
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

        # Per-row RoPE needs a table to index into (``forward_decode_batch`` gathers
        # ``cos[positions]``). One table for the whole model, keyed by
        # ``(length, device, dtype)`` -- not one per layer (P-18), and not rebuilt per
        # step: a seat/paged container's row width is fixed for the run.
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._rope_key: tuple | None = None

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

    def _rope_rows(
        self, positions: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row cos/sin gathered from one table: ``(B, 1, rotary_dim // 2)``.

        Mirrors ``MiniLLM._rope_rows``; the only difference is ``rotary_dim`` (Qwen3.5
        rotates 64 of 256 head dimensions). The table is built for ``[0, length)`` where
        ``length`` is the container's row bound (``max_len``: prompt + budget, or the
        model's position limit), so a row's position is an index, not a table offset.
        """
        key = (length, positions.device, self.token_embedding.weight.dtype)
        if self._rope_key != key:
            self._rope_cache = rope_cos_sin(
                self.head_dim,
                self.rope_theta,
                0,
                length,
                positions.device,
                self.token_embedding.weight.dtype,
                rotary_dim=self.rotary_dim,
            )
            self._rope_key = key
        cos, sin = self._rope_cache
        return cos[positions].unsqueeze(1), sin[positions].unsqueeze(1)

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
            if as_decode_container(state.kv) is not None:
                raise ValueError(
                    "state.kv is a seat/paged decode container; this single-sequence "
                    "forward derives RoPE from one scalar past_len, which is wrong for "
                    "per-row positions. Use Qwen35Model.forward_decode_batch instead."
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

    # ------------------------------------------------------ batched decode (W7)

    def forward_decode_batch(self, tokens: torch.Tensor, state: Qwen35State):
        """One decode token for each of the ``B`` compact active rows; ``(B, 1, vocab)``.

        ``tokens`` is ``(B, 1)`` int64 and ``B == state.batch``. ``state.kv`` must be a
        **decode container** (``KvSlab`` for the seat baseline or ``PagedKV`` for the
        production paged arm): its ``pos`` is the per-row RoPE source and its
        ``build_mask(next_token=True)`` is the step boundary, called exactly once here
        (not per layer). The 24 layers then run as usual, except that:

        * the 6 full-attention blocks address the container by their **compact** index
          (``0..5``), and get per-row ``cos``/``sin`` of shape ``(B, 1, rotary_dim // 2)``
          gathered from one table -- never a scalar ``past_len``;
        * the 18 GDN layers are told ``past_len=1`` for every row: the operator reads no
          positions at all, it only uses ``past_len == 0`` vs ``> 0`` to pick the
          prefill/recurrent endpoint (P-56), and every active row has prefill state.

        The state mutates in place: conv/recurrent outputs are written back through the
        setters (which ``copy_`` into the pool for a ``PooledQwen35State``), then
        ``state.kv.advance()`` and ``state.advance(1)`` move the cursors.

        Everything that would leave a half-advanced request behind is checked *before*
        the first layer: ``B >= 1``, ``T == 1``, ``state.batch == B``, a decode
        container with ``batch == B`` and matching geometry, and one conv + recurrent
        state for every one of the 18 GDN layers. A missing state raises instead of
        decoding from zeros (which would silently produce a different answer).
        """
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be a Tensor, got {type(tokens).__name__}")
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (B, T), got shape {tuple(tokens.shape)}")
        batch, seq_len = tokens.shape
        if batch < 1:
            raise ValueError(f"tokens must have at least one row, got {tuple(tokens.shape)}")
        if seq_len != 1:
            raise ValueError(
                "forward_decode_batch is decode-only (T == 1); got T="
                f"{seq_len}. Prefill keeps Qwen35Model.forward and hands over with "
                "KvSlab.fill_from / PagedPool.write_prefill; packed prefill is Step 3."
            )
        if tokens.dtype != torch.long:
            raise ValueError(f"tokens must be int64 (torch.long), got {tokens.dtype}")
        if not isinstance(state, Qwen35State):
            raise TypeError(
                "state must be a Qwen35State (or a PooledQwen35State over the active "
                f"rows), got {type(state).__name__}"
            )
        if state.config.n_layer != self.config.n_layer:
            raise ValueError(
                f"state was built for n_layer={state.config.n_layer}, this model has "
                f"n_layer={self.config.n_layer}"
            )
        if state.batch != batch:
            raise ValueError(
                f"state.batch={state.batch} does not match the batch decode's {batch} "
                "rows; a batch state presents exactly the compact active rows"
            )

        decode = as_decode_container(state.kv)
        if decode is None:
            raise ValueError(
                "forward_decode_batch needs state.kv to be a decode container (KvSlab "
                f"or PagedKV), got {type(state.kv).__name__}; a growing KVcache only "
                "supports the single-sequence forward"
            )
        if int(decode.batch) != batch:
            raise ValueError(
                f"the decode container has batch={decode.batch} rows but the batch "
                f"decode got {batch}; build the container for exactly the active rows"
            )
        container_layers = getattr(decode, "n_layer", None)
        if container_layers is not None and int(container_layers) != len(
            self.full_attention_layers
        ):
            raise ValueError(
                f"the decode container has {container_layers} layers but the model has "
                f"{len(self.full_attention_layers)} full-attention layers (D9-5: only "
                "the full-attention layers are paged/seat)"
            )
        if getattr(decode, "n_kv_head", self.config.n_kv_head) != self.config.n_kv_head:
            raise ValueError(
                f"the decode container has n_kv_head={decode.n_kv_head}, the model "
                f"{self.config.n_kv_head}"
            )
        if getattr(decode, "head_dim", self.head_dim) != self.head_dim:
            raise ValueError(
                f"the decode container has head_dim={decode.head_dim}, the model "
                f"{self.head_dim}"
            )

        # Collect the GDN state once, before the first layer: a missing layer must fail
        # here, not after ten layers have already written their new state.
        linear_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for idx, kind in enumerate(self.config.layer_types):
            if kind != _LINEAR_ATTENTION:
                continue
            conv = state.conv_state(idx)
            recurrent = state.recurrent_state(idx)
            if conv is None or recurrent is None:
                raise ValueError(
                    f"forward_decode_batch needs both linear states for layer {idx} "
                    f"before the first decode (conv_state="
                    f"{'set' if conv is not None else 'None'}, recurrent_state="
                    f"{'set' if recurrent is not None else 'None'}); prefill the "
                    "request through these pool rows first"
                )
            linear_states[idx] = (conv, recurrent)

        decode.build_mask(next_token=True)
        cos, sin = self._rope_rows(decode.pos, decode.max_len)

        x = self.token_embedding(tokens)
        for idx, (kind, layer) in enumerate(zip(self.config.layer_types, self.layers)):
            if kind == _LINEAR_ATTENTION:
                conv, recurrent = linear_states[idx]
                # past_len=1 for every row: GDN only uses the 0 vs >0 endpoint switch
                # (P-56); positions belong to the full-attention RoPE.
                x, conv, recurrent = layer(x, conv, recurrent, 1)
                state.set_conv_state(idx, conv)
                state.set_recurrent_state(idx, recurrent)
            elif kind == _FULL_ATTENTION:
                x = layer(x, None, idx, state.kv, cos, sin)
            else:  # pragma: no cover - __init__ already rejected this kind
                raise ValueError(
                    f"layer_types[{idx}]={kind!r} is not a supported layer type"
                )

        state.kv.advance()
        state.advance(1)

        logits = self.lm_head(self.norm(x))
        return logits, None

    # ------------------------------------------------- packed group prefill (W7 Step 3)

    def _check_packed_group(self, packed, tokens: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        """Validate one packed group's layout; returns ``(cu_seqlens, positions)``.

        Every check runs before the first layer, because a packed forward writes the
        group's KV through one-shot sinks and each request's linear state as it goes: a
        late reject would leave partial bytes in the pools. ``PackedPrefill`` already
        enforces this layout at construction; re-deriving it here is what stops a
        hand-rolled container (or a ``PackedPrefill`` built for a different token axis)
        from silently attending across a request boundary.
        """
        total = int(packed.total_tokens)
        n_seqs = int(packed.n_seqs)
        if tokens.shape[0] != 1 or tokens.shape[1] != total:
            raise ValueError(
                "forward_packed takes one packed token axis (B == 1): tokens must be "
                f"(1, {total}), got {tuple(tokens.shape)}"
            )
        if len(packed.sinks) != n_seqs:
            raise ValueError(
                f"packed group has {len(packed.sinks)} sink(s) for n_seqs={n_seqs}; the "
                "sink list is the request order"
            )
        try:
            cu = [int(value) for value in packed.cu_seqlens]
        except (TypeError, ValueError):
            raise ValueError(
                f"packed.cu_seqlens must be a sequence of ints, got {packed.cu_seqlens!r}"
            ) from None
        if len(cu) != n_seqs + 1:
            raise ValueError(
                f"packed.cu_seqlens has {len(cu)} entries for n_seqs={n_seqs}; it needs "
                "n_seqs + 1 cumulative offsets"
            )
        if cu[0] != 0 or cu[-1] != total:
            raise ValueError(
                f"packed.cu_seqlens must span [0, {total}] (total_tokens), got {cu}"
            )
        if any(end <= start for start, end in zip(cu, cu[1:])):
            raise ValueError(
                "packed.cu_seqlens must be strictly increasing (one non-empty segment per "
                f"request), got {cu}"
            )

        positions = packed.positions
        if not isinstance(positions, torch.Tensor):
            raise TypeError(
                f"packed.positions must be a Tensor, got {type(positions).__name__}"
            )
        if positions.dim() != 1 or int(positions.numel()) != total:
            raise ValueError(
                f"packed.positions must be a 1-D tensor of {total} entries, got "
                f"{tuple(positions.shape)}"
            )
        if positions.device != tokens.device:
            raise ValueError(
                f"packed.positions is on {positions.device}, tokens on {tokens.device}; "
                "one packed group lives on one device"
            )
        positions = positions.to(dtype=torch.long)
        if int(positions.min()) < 0:
            raise ValueError("packed.positions must be >= 0 (they are RoPE positions)")
        for row, (start, end) in enumerate(zip(cu, cu[1:])):
            expected = torch.arange(
                end - start, dtype=torch.long, device=positions.device
            )
            if not torch.equal(positions[start:end], expected):
                raise ValueError(
                    f"request {row}: packed.positions must be request-local "
                    f"0..{end - start - 1} (a packed layout does not share one increasing "
                    "position axis)"
                )
        highest = int(positions.max())
        rope_len = int(packed.rope_len)
        if rope_len <= highest:
            raise ValueError(
                f"packed.rope_len={rope_len} does not cover the highest position "
                f"{highest}; it must be at least positions.max() + 1"
            )
        if highest >= int(self.config.max_position_embeddings):
            raise ValueError(
                f"position {highest} is outside the model's max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        last = packed.last_token_index
        if not isinstance(last, torch.Tensor):
            raise TypeError(
                f"packed.last_token_index must be a Tensor, got {type(last).__name__}"
            )
        expected_last = [end - 1 for end in cu[1:]]
        if last.dim() != 1 or last.numel() != n_seqs:
            raise ValueError(
                f"packed.last_token_index must be 1-D with {n_seqs} entries, got "
                f"{tuple(last.shape)}"
            )
        if last.to(dtype=torch.long).tolist() != expected_last:
            raise ValueError(
                "packed.last_token_index must hold each request's final packed index "
                f"{expected_last}, got {last.tolist()}"
            )
        return cu, positions

    def _check_packed_states(self, states, n_seqs: int) -> list[Qwen35State]:
        """Validate one fresh, single-row state per request (before the first layer)."""
        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise TypeError(
                "states must be a sequence with one Qwen35State per packed request, got "
                f"{type(states).__name__}"
            )
        state_list = list(states)
        if len(state_list) != n_seqs:
            raise ValueError(
                f"{len(state_list)} state(s) for {n_seqs} packed request(s); the order is "
                "the packed request order (cu_seqlens)"
            )
        linear_layers = [
            idx
            for idx, kind in enumerate(self.config.layer_types)
            if kind == _LINEAR_ATTENTION
        ]
        for row, state in enumerate(state_list):
            if not isinstance(state, Qwen35State):
                raise TypeError(
                    f"states[{row}] must be a Qwen35State (or a pool subclass over one "
                    f"slot), got {type(state).__name__}"
                )
            if state.config.n_layer != self.config.n_layer:
                raise ValueError(
                    f"states[{row}] was built for n_layer={state.config.n_layer}, this "
                    f"model has n_layer={self.config.n_layer}"
                )
            if state.batch != 1:
                raise ValueError(
                    f"states[{row}].batch={state.batch}; a packed prefill holds one "
                    "single-row state per request (rows=[slot])"
                )
            if as_decode_container(state.kv) is not None:
                raise ValueError(
                    f"states[{row}].kv is a seat/paged decode container; packed prefill "
                    "writes K/V through the group's sinks, never through state.kv, so a "
                    "container there would be silently unused. Pass a fresh state "
                    "(kv=None or a plain growing KVcache)."
                )
            for idx in linear_layers:
                conv = state.conv_state(idx)
                recurrent = state.recurrent_state(idx)
                if conv is not None or recurrent is not None:
                    raise ValueError(
                        f"states[{row}] is not fresh: linear layer {idx} already holds "
                        f"state (conv={'set' if conv is not None else 'None'}, "
                        f"recurrent={'set' if recurrent is not None else 'None'}). A "
                        "packed prefill starts every prompt at position 0 from empty "
                        "state; feeding a previous request's endpoint is not supported "
                        "(P-56)."
                    )
        return state_list

    def forward_packed(
        self,
        tokens: torch.Tensor,
        packed,
        states: Sequence[Qwen35State],
    ):
        """One packed forward over a FIFO group; ``(n_seqs, vocab)`` last-token logits.

        ``tokens`` is ``(1, total_tokens)`` int64, request-major, matching
        ``packed.cu_seqlens``; ``packed`` is a :class:`~model.packed.PackedPrefill` whose
        sinks are one-shot KV writers indexed by the **compact** full-attention layer
        (0..5, D9-5); ``states`` holds one fresh single-row ``Qwen35State`` per request,
        in the same order. The second return value is always ``None`` (no training path).

        The 6 full-attention layers run **once over the whole packed axis**: each writes
        its K/V through ``packed.write`` and attends with ``packed.attend``, so the varlen
        kernel sees ``cu_seqlens`` and no query can read another request's keys. The 18
        GDN layers **loop over the segments**: every request starts at ``past_len=0`` from
        empty state, gets its own chunked prefill (W6's WY/UT form), and writes the
        returned conv/recurrent endpoint into its own state row. A shared or chained
        segment state would silently treat the previous request as this one's prefix
        (P-56). Outputs are concatenated back in request order.

        RoPE is per-token: one table over ``[0, packed.rope_len)`` is indexed by
        ``packed.positions`` to give ``(total_tokens, rotary_dim // 2)`` cos/sin, the
        shared-table form ``apply_rotary_emb_partial`` broadcasts over ``(1, T, H, D)``.
        A ``(B, 1, rotary_dim // 2)`` per-row table here would align the token axis against
        the batch axis and produce a ``(T, T, ...)`` rotation. Positions are request-local
        by contract: every request starts at 0.

        After the last layer the norm runs over the packed axis, then each request's
        ``last_token_index`` row is selected **before** ``lm_head`` (P-13): the vocabulary
        projection is ``n_seqs`` rows, not ``total_tokens``. Each state is advanced by its
        own prompt length at the very end; the state rows already hold the written
        conv/recurrent endpoint.

        Failure: everything above is validated before the first layer. If a layer still
        raises, the exception propagates unchanged and **no position is advanced** -- but
        the group's KV sinks and any linear state written by an earlier layer hold partial
        bytes, so the owner of those rows must discard them. The engine does exactly that
        (it frees the slots, which clears the pool's ready flags, and freezes the seats);
        a plain dense ``Qwen35State`` that was half written refuses a later ``past_len==0``
        prefill instead of reusing a mixed prefix.
        """
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be a Tensor, got {type(tokens).__name__}")
        if tokens.dim() != 2:
            raise ValueError(
                f"tokens must be (1, total_tokens), got shape {tuple(tokens.shape)}"
            )
        if tokens.dtype != torch.long:
            raise ValueError(f"tokens must be int64 (torch.long), got {tokens.dtype}")
        container = as_packed_prefill(packed)
        if container is None:
            raise TypeError(
                "forward_packed needs a packed-prefill container "
                f"(model.packed.PackedPrefill), got {type(packed).__name__}; the "
                "single-sequence path is forward(), batched decode is "
                "forward_decode_batch()"
            )
        packed = container

        vocab = int(self.config.vocab_size)
        low = int(tokens.min())
        high = int(tokens.max())
        if low < 0 or high >= vocab:
            raise ValueError(
                f"token ids must be in [0, {vocab}), got [{low}, {high}]; the packed "
                "group cannot be forwarded through an embedding it does not have"
            )

        cu, positions = self._check_packed_group(packed, tokens)
        state_list = self._check_packed_states(states, int(packed.n_seqs))
        bounds = list(zip(cu, cu[1:]))

        # One table for the whole stack (P-18), gathered per token. ``_rope_rows`` returns
        # the (T, 1, D/2) per-row form the slab path needs; the packed axis is the token
        # axis, so squeeze the batch dim to the (T, D/2) shared-table form.
        cos, sin = self._rope_rows(positions, int(packed.rope_len))
        cos, sin = cos.squeeze(1), sin.squeeze(1)

        x = self.token_embedding(tokens)
        for idx, (kind, layer) in enumerate(zip(self.config.layer_types, self.layers)):
            if kind == _LINEAR_ATTENTION:
                segments: list[torch.Tensor] = []
                for row, (start, end) in enumerate(bounds):
                    # Fresh state per segment (past_len=0): never the previous request's
                    # endpoint, never one shared state across the group.
                    out, conv, recurrent = layer(x[:, start:end], None, None, 0)
                    segments.append(out)
                    state = state_list[row]
                    state.set_conv_state(idx, conv)
                    state.set_recurrent_state(idx, recurrent)
                # A single-request group (the common single long prompt) needs no copy:
                # ``cat`` would duplicate the whole (1, T, D) tensor 18 times, once per
                # GDN layer, for a segment list of one.
                x = segments[0] if len(segments) == 1 else torch.cat(segments, dim=1)
            elif kind == _FULL_ATTENTION:
                # One varlen call for the whole group; the container writes each request's
                # K/V through its own sink and addresses the compact layer index.
                x = layer(x, None, idx, packed, cos, sin)
            else:  # pragma: no cover - __init__ already rejected this kind
                raise ValueError(
                    f"layer_types[{idx}]={kind!r} is not a supported layer type"
                )

        for row, (start, end) in enumerate(bounds):
            state_list[row].advance(end - start)

        x = self.norm(x)
        logits = self.lm_head(x[0, packed.last_token_index])
        return logits, None
