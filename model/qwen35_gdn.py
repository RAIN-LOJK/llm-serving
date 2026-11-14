"""Gated DeltaNet operators for Qwen3.5-0.8B's linear-attention layers (W6 Step 1).

Qwen3.5-0.8B is 18 gated-delta-net layers + 6 full-attention layers (``LLLF`` x 6).
W5 built the full-attention half; this module is the numerical core of the other 18,
written as pure functions so Step 2's ``Qwen35GDN`` layer only has to wire projections
and the gated norm (D8, decision 1):

* :func:`causal_conv_prefill` / :func:`causal_conv_decode` -- the depthwise short
  convolution (``groups=conv_dim``, kernel 4 for 0.8B) plus its minimal ``k - 1`` token
  state.
* :func:`gated_delta_recurrent` -- the token-by-token delta rule (decode gold standard,
  and the oracle the chunked form is checked against).
* :func:`gated_delta_chunked` -- the WY/UT chunked prefill (``chunk_size=64``).
* :func:`l2norm` / :func:`compute_beta_g` -- the two shared helpers whose exact form is
  part of the contract.

References and licenses
-----------------------

* **HF** ``torch_chunk_gated_delta_rule`` / ``torch_recurrent_gated_delta_rule`` /
  ``causal_conv1d_fn`` / ``causal_conv1d_update`` / ``l2norm`` in
  ``transformers/models/qwen3_5/modeling_qwen3_5.py`` (transformers 5.20.2,
  **Apache-2.0**). The chunked algorithm below is that function formula-for-formula:
  the same pad-then-reshape chunking, the same ``exp(cum_i - cum_j)`` pairwise decay
  (strictly-upper masked to ``-inf`` before ``exp``), the same UT system
  ``k_beta @ k^T * pairwise``, the same split between the undecayed ``q`` in
  ``intra`` and the decayed ``q`` in the inter-chunk read, and the same per-chunk
  ``v_new = new_values - k_cumdecay @ S`` / ``S = S * chunk_decay + key_decayed^T @ v_new``
  scan. License: Apache License 2.0, Copyright HuggingFace / the Qwen team.
* **fla** (flash-linear-attention) ``naive_recurrent_gated_delta_rule`` /
  ``naive_chunk_gated_delta_rule`` in ``fla/ops/gated_delta_rule/naive.py``
  (fla 0.5.2, **MIT**), the WY/UT derivation this and HF's version follow. We
  deliberately do **not** copy fla's forward-substitution loop: the triangular solve
  uses ``torch.linalg.solve_triangular(..., upper=False, unitriangular=True)`` so the
  two references stay numerically independent (D8, decision 2).
* The delta rule itself: Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving
  Mamba2 with Delta Rule* (ICLR 2025) -- the WY representation that turns ``C``
  sequential rank-1 state updates into one triangular solve.

License notice for the derived formula: the algorithm above is a translation of
Apache-2.0 (transformers) and MIT (fla) code into this repo's own structure; no source
text is copied verbatim, but the derivation and the ordering of operations are theirs.

Contract (load-bearing, see ``docs/decisions.md`` D8 and ``tests/conftest.py``)
-----------------------------------------------------------------------------

* **Shapes**: ``q``, ``k`` are ``(B, T, H_k, K)``; ``v`` is ``(B, T, H_v, V)``; ``g`` and
  ``beta`` are ``(B, T, H_v)``; the recurrent state is ``(B, H_v, K, V)``. ``H_v`` must
  be an integer multiple of ``H_k``; when ``ratio = H_v // H_k > 1`` the operators
  ``repeat_interleave(ratio, dim=2)`` q/k themselves, so the layer passes raw heads.
* **q/k normalization happens here, exactly once**: after the fp32 cast and after the
  GQA repeat, ``q = l2norm(q) * K**-0.5`` and ``k = l2norm(k)`` (eps ``1e-6``, fp32).
  The layer must not pre-normalize. This is not about algebraic equivalence -- the delta
  rule is defined with normalized q/k -- but about numerical conditioning of the WY
  triangular solve, whose condition number grows with ``||k||``: the G0 dry run saw
  unnormalized k differ by ``6.1e19`` at scale 1.0 and produce NaN at scale 2.0
  (P-58).
* **dtypes**: everything is computed in fp32; the output is cast back to ``q.dtype`` and
  the returned state is **always fp32** (HF behaviour). ``initial_state`` may have any
  floating dtype and is only read.
* **No mutation**: ``q``/``k``/``v``/``g``/``beta``/``initial_state`` are never written
  to. All state updates are out-of-place, and the output buffer is freshly allocated.
* ``g`` is log-space (``<= 0``) and ``beta`` is already sigmoid-ed; the operators apply
  neither sigmoid nor softplus (P-50). ``T >= 1``: there is no empty-sequence path.

This module is inference-only and has no ``nn.Module`` yet (Step 2 adds
``Qwen35GDN``/``Qwen35GDNBlock``); it imports neither ``fla`` nor ``transformers``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "l2norm",
    "causal_conv_prefill",
    "causal_conv_decode",
    "gated_delta_recurrent",
    "gated_delta_chunked",
    "compute_beta_g",
]

#: fp32 l2norm epsilon, fixed by HF ``l2norm`` / fla's kernel (D8, decision 3).
L2NORM_EPS = 1e-6

#: The only supported short-conv activation: 0.8B sets ``hidden_act="silu"``.
_CONV_ACTIVATION = "silu"


# ============================================================
# Shared helpers
# ============================================================


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """``x * rsqrt(sum(x^2, dim) + eps)`` -- HF/fla's l2norm, in the input dtype.

    The expression is deliberately identical to HF (``Qwen3_5``'s ``l2norm``) and to
    fla's kernel: ``rsqrt`` of the *sum plus eps*, not ``x / ||x||``, and the ``eps``
    sits inside the ``rsqrt`` so a zero vector maps to zero instead of NaN.

    Arithmetic happens in whatever dtype ``x`` has. The delta-rule operators call this
    only after casting to fp32, which is what makes the contract "l2norm in fp32"
    (D8, decision 3); the operator, not this helper, owns the cast.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"l2norm expects a torch.Tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        raise TypeError(f"l2norm expects a floating-point tensor, got {x.dtype}")
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _check_activation(activation: str, function: str) -> None:
    if activation != _CONV_ACTIVATION:
        raise ValueError(
            f"{function} only supports activation={_CONV_ACTIVATION!r} (Qwen3.5-0.8B sets "
            f"hidden_act='silu'); got {activation!r}. A different activation would "
            "silently change every linear-attention layer."
        )


def _as_conv_weight(weight: torch.Tensor, conv_dim: int) -> torch.Tensor:
    """Normalize the depthwise weight to ``(conv_dim, 1, k)`` (HF ``nn.Conv1d`` layout).

    Accepts ``(conv_dim, 1, k)`` (the checkpoint layout) or ``(conv_dim, k)``. There is
    no bias parameter in 0.8B's ``conv1d``, so a ``(conv_dim, 1, k)`` weight whose middle
    axis is not 1 is rejected rather than reinterpreted: it would mean grouped or
    multi-tap-per-channel conv, which this operator does not implement.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if not weight.is_floating_point():
        raise TypeError(f"weight must be a floating-point tensor, got {weight.dtype}")
    if weight.dim() == 2:
        weight = weight.unsqueeze(1)
    elif weight.dim() != 3:
        raise ValueError(
            f"weight must be (conv_dim, 1, k) or (conv_dim, k), got shape "
            f"{tuple(weight.shape)}"
        )
    if weight.shape[0] != conv_dim or weight.shape[1] != 1:
        raise ValueError(
            f"weight must be depthwise with shape ({conv_dim}, 1, k) to match "
            f"x.shape[1]={conv_dim}; got {tuple(weight.shape)}. The 0.8B conv has no "
            "bias and groups=conv_dim."
        )
    if weight.shape[-1] < 1:
        raise ValueError(f"conv kernel size must be >= 1, got {weight.shape[-1]}")
    return weight


def _check_conv_x(x: torch.Tensor, function: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{function} expects a torch.Tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        raise TypeError(f"{function} expects a floating-point tensor, got {x.dtype}")
    if x.dim() != 3:
        raise ValueError(
            f"{function} expects x of shape (B, conv_dim, T) -- already transposed, as "
            f"the layer does before the conv; got shape {tuple(x.shape)}"
        )
    if x.shape[-1] < 1:
        raise ValueError(f"{function} needs T >= 1, got T={x.shape[-1]}")


# ============================================================
# Short causal convolution (kernel k, groups=conv_dim, no bias)
# ============================================================


def causal_conv_prefill(
    x: torch.Tensor, weight: torch.Tensor, activation: str = "silu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-sequence depthwise causal conv + SiLU, plus the minimal decode state.

    ``x`` is ``(B, conv_dim, T)`` (already transposed by the layer), ``weight`` is
    ``(conv_dim, 1, k)`` or ``(conv_dim, k)``. Returns ``(out, conv_state)`` with ``out``
    of shape ``(B, conv_dim, T)`` and the same dtype as ``x``.

    The convolution is HF's ``causal_conv1d_fn``: ``padding = k - 1`` then slice the
    first ``T`` columns, so ``out[t]`` sees ``x[t - k + 1 .. t]`` only. The state is the
    **input** columns (pre-activation) ``[x[t - k + 1 .. t - 1]]``, i.e. the last
    ``k - 1`` columns of ``x`` left-padded with zeros when ``T < k - 1``. 0.8B freezes
    ``k - 1 = 3`` columns: HF's cache keeps 4 (``kernel_size``) and the oldest column
    never enters the next token's window, so the two lengths are equivalent for decode
    but not interchangeable as blobs (P-49).
    """
    _check_activation(activation, "causal_conv_prefill")
    _check_conv_x(x, "causal_conv_prefill")
    conv_dim, seq_len = x.shape[1], x.shape[-1]
    weight3 = _as_conv_weight(weight, conv_dim)
    kernel = weight3.shape[-1]

    out = F.conv1d(
        x.to(weight3.dtype),
        weight3,
        bias=None,
        padding=kernel - 1,
        groups=conv_dim,
    )[:, :, :seq_len]
    out = F.silu(out)

    state_len = kernel - 1
    if state_len == 0:
        # k == 1 has no history: the state is an empty (B, conv_dim, 0) tensor.
        state = x.new_zeros((x.shape[0], conv_dim, 0))
    else:
        state = F.pad(x, (state_len, 0))[..., -state_len:]
    return out.to(x.dtype), state


def causal_conv_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One or more tokens through the same causal conv, reading/updating ``conv_state``.

    ``x`` is ``(B, conv_dim, T)`` (``T >= 1``), ``conv_state`` is ``(B, conv_dim, k - 1)``
    as returned by :func:`causal_conv_prefill` -- the **pre-SiLU** input columns. The
    state must have exactly ``k - 1`` columns: accepting ``k`` (HF's 5.20.2 cache length)
    would pass the length check and then silently shift the window by one token. The
    caller's tensor is not mutated; a new state is returned.

    Feeding one token at a time reproduces :func:`causal_conv_prefill` up to fp32
    rounding (``tests/unit/test_qwen35_gdn.py``), which is the whole point of keeping the
    state pre-activation.
    """
    _check_activation(activation, "causal_conv_decode")
    _check_conv_x(x, "causal_conv_decode")
    conv_dim, seq_len = x.shape[1], x.shape[-1]
    weight3 = _as_conv_weight(weight, conv_dim)
    state_len = weight3.shape[-1] - 1

    if not isinstance(conv_state, torch.Tensor):
        raise TypeError(
            f"conv_state must be a torch.Tensor, got {type(conv_state).__name__}"
        )
    if not conv_state.is_floating_point():
        raise TypeError(
            f"conv_state must be a floating-point tensor, got {conv_state.dtype}"
        )
    if conv_state.dim() != 3 or conv_state.shape[:2] != (x.shape[0], conv_dim):
        raise ValueError(
            f"conv_state must be (B={x.shape[0]}, conv_dim={conv_dim}, {state_len}), got "
            f"{tuple(conv_state.shape)}"
        )
    if conv_state.shape[-1] != state_len:
        raise ValueError(
            f"conv_state must have exactly k - 1 = {state_len} columns (the minimal "
            f"sufficient state, P-49); got {conv_state.shape[-1]}. HF's cache keeps "
            "kernel_size columns; slice it with [..., -3:] before calling this."
        )

    new = torch.cat([conv_state.to(x.dtype), x], dim=-1)
    out = F.conv1d(
        new.to(weight3.dtype), weight3, bias=None, padding=0, groups=conv_dim
    )[:, :, -seq_len:]
    out = F.silu(out)

    if state_len == 0:
        new_state = new.new_zeros((new.shape[0], conv_dim, 0))
    else:
        new_state = new[..., -state_len:]
    return out.to(x.dtype), new_state.to(x.dtype)


# ============================================================
# Gated delta rule: shared validation / preparation
# ============================================================


def _check_gdn_operands(q, k, v, g, beta) -> tuple[int, int, int, int, int, int, int]:
    """Validate ranks/shapes/dtypes; return ``(B, T, H_k, H_v, K, V, ratio)``."""
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_floating_point():
            raise TypeError(
                f"{name} must be a floating-point tensor, got {tensor.dtype}; these "
                "operators do no integer/bool math and do not cast silently"
            )
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            f"q and k must be (B, T, H_k, K), got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if v.dim() != 4:
        raise ValueError(f"v must be (B, T, H_v, V), got {tuple(v.shape)}")
    if g.dim() != 3 or beta.dim() != 3:
        raise ValueError(
            f"g and beta must be (B, T, H_v), got {tuple(g.shape)} and {tuple(beta.shape)}"
        )

    batch, seq_len, num_k_heads, head_k_dim = q.shape
    if tuple(k.shape) != tuple(q.shape):
        raise ValueError(
            f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if v.shape[0] != batch or v.shape[1] != seq_len:
        raise ValueError(
            f"v must be (B={batch}, T={seq_len}, H_v, V) to match q/k, got "
            f"{tuple(v.shape)}"
        )
    num_v_heads, head_v_dim = v.shape[-2], v.shape[-1]
    if seq_len < 1:
        raise ValueError("the gated delta rule needs T >= 1; empty sequences are not a path")
    if num_k_heads < 1 or num_v_heads < 1 or head_k_dim < 1 or head_v_dim < 1:
        raise ValueError(
            "head counts and head dims must all be >= 1, got "
            f"H_k={num_k_heads}, H_v={num_v_heads}, K={head_k_dim}, V={head_v_dim}"
        )
    decay_shape = (batch, seq_len, num_v_heads)
    if tuple(g.shape) != decay_shape or tuple(beta.shape) != decay_shape:
        raise ValueError(
            f"g and beta must be (B={batch}, T={seq_len}, H_v={num_v_heads}) -- one "
            f"scalar per value head, no broadcasting; got {tuple(g.shape)} and "
            f"{tuple(beta.shape)}"
        )
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num value heads ({num_v_heads}) must be an integer multiple of num key "
            f"heads ({num_k_heads}) for the GQA repeat"
        )
    ratio = num_v_heads // num_k_heads
    if ratio < 1:
        raise ValueError(
            f"num value heads ({num_v_heads}) must be >= num key heads ({num_k_heads}); "
            f"ratio = H_v // H_k = {ratio} would repeat q/k zero times"
        )
    return batch, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, ratio


def _check_initial_state(initial_state, batch, num_v_heads, head_k_dim, head_v_dim):
    if initial_state is None:
        return
    if not isinstance(initial_state, torch.Tensor):
        raise TypeError(
            "initial_state must be a torch.Tensor or None, got "
            f"{type(initial_state).__name__}"
        )
    if not initial_state.is_floating_point():
        raise TypeError(
            f"initial_state must be a floating-point tensor, got {initial_state.dtype}"
        )
    expected = (batch, num_v_heads, head_k_dim, head_v_dim)
    if tuple(initial_state.shape) != expected:
        raise ValueError(
            f"initial_state must be (B, H_v, K, V) = {expected} -- K before V, not the "
            f"transposed layout; got {tuple(initial_state.shape)}"
        )


def _to_head_major(x: torch.Tensor) -> torch.Tensor:
    """``(B, T, H, D)``/``(B, T, H)`` -> ``(B, H, T, D)``/``(B, H, T)``, fp32 contiguous.

    Matches HF's ``x.transpose(1, 2).to(torch.float32, memory_format=contiguous_format)``
    so the two implementations walk the same memory and the comparisons in
    ``tests/unit/test_qwen35_gdn.py`` are bit-stable.
    """
    return x.transpose(1, 2).to(torch.float32).contiguous()


def _normalized_qk(
    q: torch.Tensor, k: torch.Tensor, ratio: int, head_k_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """GQA repeat, then fp32 l2norm, then the ``K**-0.5`` scale on q only.

    Order and single-application are part of the contract (D8, decision 3): the layer
    hands over raw q/k, the repeat comes first so every value head sees its own key
    head, and the scale multiplies the *normalized* q exactly once (P-51, P-58).
    """
    if ratio > 1:
        q = q.repeat_interleave(ratio, dim=1)
        k = k.repeat_interleave(ratio, dim=1)
    q = l2norm(q, dim=-1, eps=L2NORM_EPS)
    k = l2norm(k, dim=-1, eps=L2NORM_EPS)
    q = q * (head_k_dim**-0.5)
    return q, k


# ============================================================
# Gated delta rule: recurrent form (decode gold standard)
# ============================================================


def gated_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Token-by-token gated delta rule, all math in fp32.

    Per token: ``h *= exp(g_t)``; ``kv = (h * k_t).sum(K)``; ``delta = (v_t - kv) * beta_t``;
    ``h += k_t (x) delta``; ``o_t = (h * q_t).sum(K)``. The loop slices ``g[:, :, i]`` and
    friends -- never the whole ``(B, H, T)`` tensor -- so no decay is applied twice.

    ``initial_state`` (``(B, H_v, K, V)``, any floating dtype) is cast to fp32 and read
    only; the returned state is fp32 even when the output is cast back to ``q.dtype``.
    Returns ``(out, None)`` when ``output_final_state`` is false.
    """
    batch, seq_len, _, num_v_heads, head_k_dim, head_v_dim, ratio = _check_gdn_operands(
        q, k, v, g, beta
    )
    _check_initial_state(initial_state, batch, num_v_heads, head_k_dim, head_v_dim)
    initial_dtype = q.dtype

    q, k, v, g, beta = (_to_head_major(x) for x in (q, k, v, g, beta))
    q, k = _normalized_qk(q, k, ratio, head_k_dim)

    if initial_state is None:
        h = torch.zeros(
            (batch, num_v_heads, head_k_dim, head_v_dim),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        # Clone so the fp32 cast of an already-fp32 state cannot alias the caller's
        # tensor and turn a future in-place update into a silent mutation.
        h = initial_state.to(torch.float32).clone()

    out = torch.empty(
        (batch, num_v_heads, seq_len, head_v_dim), dtype=torch.float32, device=q.device
    )
    for i in range(seq_len):
        h = h * g[:, :, i].exp()[..., None, None]
        kv_mem = (h * k[:, :, i, :, None]).sum(dim=-2)
        delta = (v[:, :, i] - kv_mem) * beta[:, :, i][..., None]
        h = h + k[:, :, i, :, None] * delta[:, :, None, :]
        out[:, :, i] = (h * q[:, :, i, :, None]).sum(dim=-2)

    out = out.transpose(1, 2).to(initial_dtype).contiguous()
    return out, (h if output_final_state else None)


# ============================================================
# Gated delta rule: chunked WY/UT form (prefill)
# ============================================================


def gated_delta_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked gated delta rule through the WY/UT representation, ``chunk_size=64``.

    Same arguments, shapes, dtype rules and q/k-normalization contract as
    :func:`gated_delta_recurrent`; the two must agree to ~1e-7 on fp32 CPU, and the
    chunked form is the production prefill path (D8, decision 4).

    The algorithm is HF's ``torch_chunk_gated_delta_rule`` step for step (see the module
    docstring for the license); the only intentional difference is the triangular solve,
    which uses ``torch.linalg.solve_triangular`` instead of fla's forward substitution.
    Two orderings are easy to get wrong and are pinned by unit tests:

    * ``pairwise[..., i, j] = exp(cum_i - cum_j)`` -- ``cum.unsqueeze(-1)`` is the row
      index ``i``, ``cum.unsqueeze(-2)`` the column index ``j``, and the strictly upper
      triangle is masked **before** the ``exp``.
    * ``intra`` uses the **undecayed** ``q``; only the inter-chunk read uses
      ``q * exp(cum)``. Feeding the decayed q into both double-applies the chunk decay.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    batch, seq_len, _, num_v_heads, head_k_dim, head_v_dim, ratio = _check_gdn_operands(
        q, k, v, g, beta
    )
    _check_initial_state(initial_state, batch, num_v_heads, head_k_dim, head_v_dim)
    initial_dtype = q.dtype

    q, k, v, g, beta = (_to_head_major(x) for x in (q, k, v, g, beta))
    q, k = _normalized_qk(q, k, ratio, head_k_dim)

    # Right-pad the sequence to a whole number of chunks; padded rows produce garbage
    # that is sliced off at the end (P-52). beta=0/g=0 makes the padded rows inert for
    # the UT solve too (their k_beta/v_beta are zero).
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    num_chunks = (seq_len + pad) // chunk_size

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    def _by_chunk(x, width):
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, width)

    q, k = _by_chunk(q, head_k_dim), _by_chunk(k, head_k_dim)
    v_beta, k_beta = _by_chunk(v_beta, head_v_dim), _by_chunk(k_beta, head_k_dim)
    decay = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    strict_upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=q.device
    ).triu(1)

    # cum[..., i] = log of the decay product from the chunk start to position i.
    cum = decay.cumsum(-1)
    pairwise = (
        (cum.unsqueeze(-1) - cum.unsqueeze(-2))
        .masked_fill(strict_upper, float("-inf"))
        .exp()
    )

    ut = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)

    # (I + strict_lower(ut)) is unit lower triangular, so the diagonal of `ut`
    # (beta * ||k||^2, which the WY form ignores) is not part of the system.
    new_values = torch.linalg.solve_triangular(
        ut, v_beta, upper=False, unitriangular=True
    )
    k_cumdecay = torch.linalg.solve_triangular(
        ut, decayed_k_beta, upper=False, unitriangular=True
    )

    query_decayed = q * cum.exp().unsqueeze(-1)
    key_decayed = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]

    if initial_state is None:
        state = torch.zeros(
            (batch, num_v_heads, head_k_dim, head_v_dim),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.to(torch.float32).clone()

    core = torch.empty_like(new_values)
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        core[:, :, i] = query_decayed[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + (
            key_decayed[:, :, i].transpose(-1, -2) @ v_new
        )

    core = core.reshape(batch, num_v_heads, -1, head_v_dim)[:, :, :seq_len]
    core = core.transpose(1, 2).to(initial_dtype).contiguous()
    return core, (state if output_final_state else None)


# ============================================================
# beta / g projections
# ============================================================


def compute_beta_g(
    b: torch.Tensor, a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``beta = sigmoid(b)`` and ``g = -exp(A_log) * softplus(a + dt_bias)``, as HF does.

    ``b``/``a`` are ``(B, T, H)``; ``A_log``/``dt_bias`` are ``(H,)``. ``beta`` keeps
    ``b.dtype``; ``g`` is fp32 (the log-space decay is the numerically sensitive one, and
    HF's ``A_log.float()`` exists so a fp16 model cannot produce ``-inf``).

    Applying sigmoid or softplus a second time anywhere downstream is the P-50 bug: the
    operators take ``beta`` already in ``(0, 1)`` and ``g`` already ``<= 0``.
    """
    for name, tensor in (
        ("b", b),
        ("a", a),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_floating_point():
            raise TypeError(
                f"{name} must be a floating-point tensor, got {tensor.dtype}"
            )
    if b.dim() != 3 or a.dim() != 3:
        raise ValueError(
            f"b and a must be (B, T, H), got {tuple(b.shape)} and {tuple(a.shape)}"
        )
    if tuple(b.shape) != tuple(a.shape):
        raise ValueError(
            f"b and a must have the same shape, got {tuple(b.shape)} and {tuple(a.shape)}"
        )
    num_heads = b.shape[-1]
    if A_log.dim() != 1 or dt_bias.dim() != 1 or (
        A_log.shape[0] != num_heads or dt_bias.shape[0] != num_heads
    ):
        raise ValueError(
            f"A_log and dt_bias must be (H={num_heads},), got {tuple(A_log.shape)} and "
            f"{tuple(dt_bias.shape)}"
        )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return beta, g
