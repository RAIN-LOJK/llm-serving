"""Capture one decode step as a CUDA graph.

**PARKED — not wired into any measurement; W4 will connect it.** Nothing on the run path
imports this module (D4: CUDA graphs do not enter W3 numbers, because a captured step
removes the very host cost W3 exists to measure). It stays in the tree with its own
``tests/integration/test_graph_decode.py`` so the W4 wiring has something to build on;
do not call it from ``serve/engine.py``.

Why it might matter later: a 0.6B model's decode step is host-bound (measured: 18.6 ms/step
at B=8 while the GPU busy time was ~3.3 ms). Batching amortises that host cost, which makes
every W3/W4 ratio measure *Python*, not the mechanism. Capturing the step removes the host
cost from the loop entirely: the step becomes one graph replay.

What it needs, and why the slab exists:

* **stable addresses** — the KV buffers, the token buffer, the mask and the RoPE table are
  allocated once and updated in place (``model/slab.py``); ``torch.cat`` growth would move
  them every step and silently invalidate the capture;
* **no host synchronisation inside the step** — no ``.item()``, no ``bool(tensor.any())``.
  ``KvSlab.check_bounds`` is that sync, so it is switched off here after the budget is
  asserted statically, and restored afterwards. (The eager path now guards with a Python
  counter, but the strict check must still stay out of a capture.)
* **a fixed shape** — batch is a bucket, not a value: callers pad to the bucket and drop the
  padding rows from the sampling step (batch-dimension padding only; the length direction is
  still never padded beyond the slab's own allocation).

Scope: decode only. Prefill keeps its per-request SDPA call (one big kernel per sequence),
so there is nothing to gain from capturing it and W3 does not run it inside the loop.
"""

from __future__ import annotations

import torch

from model.slab import KvSlab

__all__ = ["GraphDecoder"]


class GraphDecoder:
    """Replay a captured decode step against a fixed ``KvSlab``.

    Usage::

        decoder = GraphDecoder(model, slab, max_steps=budget)
        decoder.capture()          # advances the slab, then restores it
        logits = decoder.step([token])   # one token per row, returns (batch, vocab)
    """

    def __init__(self, model, slab: KvSlab, *, max_steps: int | None = None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs need a CUDA device")
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError(f"the model must be on CUDA, got {device}")
        start = int(slab.valid.max().item())
        if max_steps is not None and start + max_steps > slab.max_len:
            raise ValueError(
                f"budget does not fit the slab: {start} prefilled + {max_steps} steps > {slab.max_len}; "
                "allocate max_len = prompt_len + max_new_tokens"
            )
        self.model = model
        self.slab = slab
        self.device = device
        self.batch = slab.batch
        self._tokens = torch.zeros(self.batch, 1, dtype=torch.long, device=device)
        self._logits: torch.Tensor | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._snapshot: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ capture
    def capture(self) -> "GraphDecoder":
        """Warm up, capture, then restore the slab to its pre-capture state."""
        slab = self.slab
        self._snapshot = (slab.pos.clone(), slab.valid.clone())
        self._tokens.zero_()
        self._bounds = slab.check_bounds  # restore what the caller had (True or "strict")
        slab.check_bounds = False  # the eager guard; replay never runs this Python
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):  # warm up allocators, rope table, backend buffers
                    self.model(self._tokens, past_keys_values=slab, logits_to_keep=1)
            torch.cuda.current_stream(self.device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                logits, _ = self.model(self._tokens, past_keys_values=slab, logits_to_keep=1)
            self._graph = graph
            self._logits = logits
        finally:
            # The warm-up and the capture both advanced the slab; put it back so the caller
            # can start from the state it measured. Addresses never changed, so the captured
            # graph stays valid. The eager host mirrors are re-read afterwards (one sync,
            # outside the graph) so the Python-side bounds guard stays truthful.
            slab.pos.copy_(self._snapshot[0])
            slab.valid.copy_(self._snapshot[1])
            slab.sync_host()
            slab.check_bounds = self._bounds
        return self

    # ------------------------------------------------------------------ replay
    @torch.no_grad()
    def step(self, tokens) -> torch.Tensor:
        """One decode step for one token per row; returns ``(batch, vocab)``."""
        if self._graph is None:
            raise RuntimeError("call capture() before step()")
        tensor = torch.as_tensor(tokens, dtype=torch.long, device=self.device).view(-1)
        if tensor.numel() != self.batch:
            raise ValueError(f"step needs {self.batch} tokens (one per row), got {tensor.numel()}")
        self._tokens[:, 0] = tensor
        slab = self.slab
        slab.check_bounds = False
        try:
            self._graph.replay()
        finally:
            slab.check_bounds = self._bounds
        # The replay advanced the cursors on the device without running ``KvSlab.advance``,
        # so keep the Python mirrors in step with it -- an increment is exact (one token per
        # row per replay) and costs no synchronisation, unlike re-reading the tensors.
        slab.advance_host()
        return self._logits[:, -1, :]

    # ------------------------------------------------------------------ inspection
    @property
    def captured(self) -> bool:
        return self._graph is not None
