"""Packed whole-prompt prefill (W4.5): one varlen forward for a group of requests.

W4 prefills one request at a time, so the GPU sees a ``(1, T, ...)`` forward per
admission and every other queued prompt waits for it. The scheduler's ``ttft`` is
therefore ``queue + prefill`` by construction, and a burst of short prompts pays one
dispatch and one weight read per prompt. W4.5 packs a FIFO group of whole prompts
into **one** token axis -- request-major, no padding, no chunking -- and computes
their attention with a varlen kernel (FlashAttention-2 on CUDA; a per-sequence SDPA
loop as the CPU/tests reference). The packed tensor carries ``cu_seqlens``, so the
kernel never attends across a request boundary and never materializes a
``(B, T_max)`` padded batch.

What lives here
---------------
``PackedPrefill``
    The prefill container ``model/layers.py`` / ``model/minillm.py`` consume. It
    owns the request boundaries (``cu_seqlens``), the per-token RoPE positions and
    the list of **write sinks** -- one per request, each already pointing at that
    request's destination (a ``KvSlab`` row or a ``PagedPool`` block table). The
    attention backend is selected explicitly at construction (``backend_for``),
    never by trying one kernel and catching an error.

This is deliberately separate from the two decode containers (``KvSlab`` /
``PagedKV``): a prefill container has no mask, no cursor and no ``advance``. The
layer loop writes each layer's K/V through the sinks and then drops it, exactly like
``SlabWriteSink`` / ``PagedWriteSink`` already do for the serial path.

The contract the layer relies on
--------------------------------
* ``q``, ``k``, ``v`` are packed ``(1, total_tokens, H, D)`` (``H_kv`` for k/v).
* ``write(layer, k, v)`` slices the packed token axis by ``cu_seqlens`` and calls
  each sink's one-shot ``set_kv(layer, (k, v))`` with ``(1, H, length, D)``.
* ``attend(q, k, v)`` calls the varlen backend with the same ``cu_seqlens`` and
  returns ``(1, total_tokens, H, D)``.
* ``last_token_index`` is the packed index of every request's final prompt token;
  ``MiniLLM`` selects those rows **before** ``lm_head``, so the vocabulary
  projection runs ``n_seqs`` times, not ``total_tokens`` times.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["PackedPrefill", "as_packed_prefill", "PACKED_PREFILL_FLAG"]

#: A prefill container opts in by carrying ``packed_prefill = True``. The layer and
#: model check this flag explicitly, exactly like ``decode_container``; an object
#: that claims it but misses a hook raises instead of being treated as a ``KVcache``.
PACKED_PREFILL_FLAG = "packed_prefill"

_PACKED_HOOKS = (
    "cu_seqlens",
    "positions",
    "last_token_index",
    "total_tokens",
    "n_seqs",
    "past_len",
    "write",
    "attend",
)


def as_packed_prefill(past):
    """Return ``past`` if it is a packed-prefill container, else ``None``.

    ``None`` and every ordinary cache keep the growing-``KVcache`` path. A container
    that sets the flag but is missing a hook raises ``TypeError``: an incomplete
    container is a caller bug, not a reason to fall back to a different attention
    computation (D4's no-silent-fallback rule).
    """
    if past is None or not getattr(past, PACKED_PREFILL_FLAG, False):
        return None
    missing = [name for name in _PACKED_HOOKS if not hasattr(past, name)]
    if missing:
        raise TypeError(
            f"container {type(past).__name__} sets {PACKED_PREFILL_FLAG}=True but is "
            f"missing packed-prefill hook(s) {missing}; implement the PackedPrefill "
            "protocol or drop the flag"
        )
    return past


class PackedPrefill:
    """One varlen forward over a FIFO group of whole prompts.

    Parameters
    ----------
    cu_seqlens:
        ``n_seqs + 1`` cumulative token counts, starting at 0 and ending at the
        packed token count. Each adjacent pair is one request; no token is shared.
    positions:
        ``(total_tokens,)`` long tensor of **request-local** RoPE positions. A
        packed request-major layout means request ``s`` occupies
        ``[cu[s], cu[s+1])`` and its positions are ``0 .. length-1``. They are not
        ``arange(total_tokens)``: every request starts at position 0.
    sinks:
        One write sink per request. A sink implements
        ``set_kv(layer, (k, v))`` with ``k``/``v`` shaped ``(1, H_kv, length, D)``
        -- the same one-shot contract as ``SlabWriteSink`` / ``PagedWriteSink``.
    backend:
        A varlen backend (``SdpaVarlen`` / ``FlashVarlen``). When omitted it is
        selected from ``positions.device`` through ``backend_for``; the selection is
        recorded in ``backend_name`` so the manifest can report it.
    """

    #: Explicit protocol marker checked by ``as_packed_prefill``.
    packed_prefill = True

    def __init__(
        self,
        *,
        cu_seqlens: Sequence[int],
        positions: torch.Tensor,
        sinks: Sequence[object],
        backend=None,
    ) -> None:
        cumulative = [int(value) for value in cu_seqlens]
        if len(cumulative) < 2:
            raise ValueError("PackedPrefill needs at least one request (cu_seqlens length >= 2)")
        if cumulative[0] != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {cumulative[0]}")
        if any(b <= a for a, b in zip(cumulative, cumulative[1:])):
            raise ValueError(f"cu_seqlens must be strictly increasing, got {cumulative}")
        total = int(cumulative[-1])
        if len(sinks) != len(cumulative) - 1:
            raise ValueError(
                f"PackedPrefill needs one sink per request ({len(cumulative) - 1}), "
                f"got {len(sinks)}"
            )
        if positions.dim() != 1 or int(positions.numel()) != total:
            raise ValueError(
                f"positions must be a 1-D tensor of {total} entries, got "
                f"{tuple(positions.shape)}"
            )
        positions = positions.to(dtype=torch.long)
        for row, (start, end) in enumerate(zip(cumulative, cumulative[1:])):
            expected = torch.arange(end - start, dtype=torch.long, device=positions.device)
            if not torch.equal(positions[start:end], expected):
                raise ValueError(
                    f"request {row}: positions must be request-local 0..{end - start - 1} "
                    "(a packed layout does not share one increasing position axis)"
                )

        if backend is None:
            from model.attention.varlen import backend_for

            backend = backend_for(positions.device)
        self.backend = backend
        self.cu_seqlens = cumulative
        self.positions = positions
        self.sinks = list(sinks)
        self.total_tokens = total
        self.n_seqs = len(self.sinks)
        self.max_seqlen = max(b - a for a, b in zip(cumulative, cumulative[1:]))
        self.rope_len = max(int(positions.max().item()) + 1, 1)
        self.last_token_index = torch.tensor(
            [end - 1 for end in cumulative[1:]], dtype=torch.long, device=positions.device
        )

    # ---------------------------------------------------------------- layer hooks
    def past_len(self, layer: int | None = None) -> int:
        """A packed prefill has no history: every prompt starts at position 0."""
        return 0

    def get_kv(self, layer: int) -> tuple[None, None]:
        """No growing cache: K/V is written through the sinks and dropped."""
        return (None, None)

    def set_kv(self, layer: int, kv) -> None:
        """Compatibility hook: write a dense ``(1, H, L, D)`` layer through the sinks.

        The packed path in ``Attention.forward`` calls :meth:`write` with the packed
        token axis instead; this method exists so the container still satisfies the
        ordinary cache protocol if a caller reaches it.
        """
        k, v = kv
        if k.dim() != 4 or k.shape[0] != 1:
            raise ValueError(f"set_kv needs (1, H, L, D) k/v, got {tuple(k.shape)}")
        start = 0
        for sink, end in zip(self.sinks, self.cu_seqlens[1:]):
            length = end - start
            sink.set_kv(layer, (k[:, :, start:end], v[:, :, start:end]))
            start = end

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Slice the packed token axis and write each request's K/V through its sink."""
        if k.dim() != 4 or k.shape[0] != 1 or k.shape[1] != self.total_tokens:
            raise ValueError(
                f"write needs packed (1, {self.total_tokens}, H, D) k/v, got "
                f"{tuple(k.shape)} / {tuple(v.shape)}"
            )
        if tuple(k.shape) != tuple(v.shape):
            raise ValueError(f"k/v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        start = 0
        for sink, end in zip(self.sinks, self.cu_seqlens[1:]):
            # (1, H, L, D) is the sink contract; the layer's k is (1, L, H, D).
            layer_k = k[0, start:end].transpose(0, 1).unsqueeze(0)
            layer_v = v[0, start:end].transpose(0, 1).unsqueeze(0)
            sink.set_kv(layer, (layer_k, layer_v))
            start = end

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Per-request causal attention over the packed axis, no cross-request keys."""
        if q.dim() != 4 or q.shape[0] != 1 or q.shape[1] != self.total_tokens:
            raise ValueError(
                f"attend needs packed (1, {self.total_tokens}, H, D) q, got {tuple(q.shape)}"
            )
        out = self.backend.prefill_varlen(
            q[0].contiguous(),
            k[0].contiguous(),
            v[0].contiguous(),
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.max_seqlen,
            causal=True,
        )
        return out.unsqueeze(0)

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "name", type(self.backend).__name__))
