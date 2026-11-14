#!/usr/bin/env python
"""Generate the W6 Gated DeltaNet single-layer golden fixture from Qwen3.5-0.8B.

W6 aligns the hand-written ``Qwen35GDNBlock`` against HF *stage by stage*, the same way
W5 aligned the full-attention block (docs/w6-plan.md §6). This script runs the HF text
decoder once per prompt on CPU in float32 with eager attention and hooks every stage of
the two linear-attention layers ``L0`` and ``L2``:

    layer_in -> input_layernorm -> in_proj_qkv -> conv_out -> q_pre/k_pre, beta/g
             -> core_attn_out -> norm_out -> layer_out (+ final_state, final_conv_state)

Step 3 only proves the *generator* and the stage table. The formal fixture
``tests/golden/qwen35-0-8b/0002`` is generated once by the user in Step 5 and is read-only
from then on (tests/golden/README.md); ``--out`` therefore defaults to the throw-away
``/tmp/w6-golden-dry``:

    python scripts/make_golden_qwen35_w6.py \\
        --revision 2fc06364715b967f1860aea9cf38778875588b17 \\
        --layers 0 2 --ids s01,m02 --out /tmp/w6-golden-dry

Environment facts this script has to neutralise (PITFALLS P-14/P-15/P-48):

* **P-48.** ``causal-conv1d`` / ``fla`` are installed on the 4090; transformers resolves
  their kernels eagerly and both the chunked *and* the recurrent delta rule are replaced
  by fla's Triton kernels, which assert on CPU tensors. **Four** symbols are pinned back
  to their torch fallbacks before the first forward -- W5 pinned three and never needed
  the recurrent one because it had no linear-attention layer. This script does not change
  ``tests/conftest.py`` or ``scripts/make_golden_qwen35_layer.py``; the pins are local.
* **P-14.** The checkpoint is ``Qwen3_5ForConditionalGeneration``; the decoder is loaded
  from ``config.get_text_config()`` explicitly (via W5's ``load_text_decoder``), so the
  vision tower and the MTP head are not part of the reference run.

Two capture rules are load-bearing:

* **Hook attribution.** ``layer.input_layernorm`` runs *before* ``layer.linear_attn``, so
  a single module-level ``current_layer`` cannot key the module hooks -- it would record
  layer 2's ``input_layernorm`` under whichever layer ran last. Every module hook is a
  closure over its own layer index. Only the delta-rule wrapper needs a module-level
  ``current_layer``, because all 18 linear layers share one rule symbol; it is set by
  ``layer.linear_attn.register_forward_pre_hook``, which fires immediately before the
  wrapped call, and the wrapper records only for the requested layers.
* **``conv_out`` is derived, not hooked.** It is recomputed from the captured
  ``in_proj_qkv`` output with the same ``F.conv1d`` formula the production operator uses,
  then ``F.silu``: it is **post-SiLU**. The conv *state* is the pre-SiLU input columns, so
  ``conv_out`` and ``conv_state`` are different measurements and are never compared
  (P-49). ``final_conv_state`` is read from the returned ``DynamicCache`` as
  ``conv_states[0][..., -3:]`` because HF 5.20.2 stores ``kernel_size = 4`` columns while
  W6 freezes the minimal sufficient ``kernel - 1 = 3``.

``q_pre``/``k_pre`` are the q/k **at the rule entry**: the split after the conv, reshaped
to ``(B, T, H, K)``, *before* the in-kernel l2norm and before the ``K**-0.5`` q scale
(they are captured as the wrapper's arguments, so the operator's own normalization
cannot have run yet). ``beta`` is already sigmoid-ed and ``g`` is already log-space
``<= 0``: both are captured after HF's projection, never re-derived.

npz keys: ``L{layer}/{prompt}/{stage}`` plus ``{prompt}/input_ids`` and
``{prompt}/positions``. Everything is stored at the three sampled rows
``sorted({0, T // 2, T - 1})`` except ``layer_in`` (all ``T``: the test replays the whole
block) and the two final states (full tensors, ``m02`` only, to keep the fixture small).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_golden import git_state, resolve_snapshot, slug_of  # noqa: E402
from scripts.make_golden_qwen35_layer import (  # noqa: E402
    cap_torch_threads,
    load_text_decoder,
)

SPEC = "w6-gdn-v1"
DEFAULT_REPO = "Qwen/Qwen3.5-0.8B"
DEFAULT_LAYERS = (0, 2)
DEFAULT_IDS = "s01,m02"
DEFAULT_OUT = "/tmp/w6-golden-dry"
MAX_BYTES = 10 * 1024 * 1024

#: P-48: all four Qwen3.5 kernel symbols, in the order the layer uses them.
PINNED_KERNELS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)

#: Stages read straight off a module hook (``conv_out`` is derived, see the docstring).
HOOKED_STAGES = (
    "input_layernorm",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "norm_out",
    "layer_out",
)
#: Stages stored at the three sampled rows, in data-flow order.
SAMPLED_STAGES = (
    "input_layernorm",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "conv_out",
    "q_pre",
    "k_pre",
    "beta",
    "g",
    "core_attn_out",
    "norm_out",
    "layer_out",
)
#: Stages stored in full (no row sampling). ``layer_in`` feeds the whole block back.
FULL_STAGES = ("layer_in", "final_state", "final_conv_state")
#: Final states are only stored for the long prompt: two layers x 2 states would add
#: ~2.3 MB for a 36-token prompt that has no extra chunk boundary to exercise.
FINAL_STAGES = ("final_state", "final_conv_state")
FINAL_PROMPTS = ("m02",)


# ------------------------------------------------------------------ environment pins


def pin_torch_kernels() -> tuple[list[str], list[str]]:
    """P-15/P-48: force transformers' four Qwen3.5 kernels back to torch fallbacks.

    Returns ``(pinned, changed)``: ``pinned`` is always all four names (they are the
    contract the fixture records and the test asserts on), ``changed`` is the subset that
    actually had a hub/fla wrapper to strip in this environment. A symbol that is already
    the raw fallback is left alone but still counts as pinned.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    changed: list[str] = []
    for name in PINNED_KERNELS:
        resolved = getattr(q35, name, None)
        if resolved is None:
            raise SystemExit(
                f"transformers.models.qwen3_5.modeling_qwen3_5 has no {name!r}; this "
                "script pins four symbols (P-48) -- check the transformers version"
            )
        fallback = getattr(resolved, "__wrapped__", None)
        if fallback is not None and fallback is not resolved:
            setattr(q35, name, fallback)
            changed.append(name)
    return list(PINNED_KERNELS), changed


# ------------------------------------------------------------------------ capturing


def _store_output(store: dict, layer_idx: int, name: str):
    def hook(_module, _inputs, output):
        store[(layer_idx, name)] = output.detach()

    return hook


def _store_input(store: dict, layer_idx: int, name: str):
    def hook(_module, inputs):
        store[(layer_idx, name)] = inputs[0].detach()

    return hook


class RuleCapture:
    """Wrap the two pinned delta-rule symbols and record the requested layers' stages.

    HF calls the rule with positional ``query, key, value`` and keyword ``g=``, ``beta=``
    (both rules); the wrapper accepts either convention so a future call-site change
    cannot silently record nothing. Inputs are cloned *before* the call (the operators
    promise not to mutate, but a clone keeps a capture bug from corrupting the fixture),
    outputs are read after it. ``current_layer`` is set by the ``linear_attn`` pre-hook,
    which is the only attribution signal that survives 18 layers sharing one symbol.
    """

    RULE_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")

    def __init__(self, layers: list[int], store: dict):
        self.layers = set(layers)
        self.current_layer: int | None = None
        self.enabled = False
        # Same dict the module hooks write to: one store per forward, cleared together.
        self.store: dict[tuple[int, str], object] = store
        self._original: dict[str, object] = {}

    def install(self, module) -> None:
        for name in self.RULE_NAMES:
            original = getattr(module, name)
            self._original[name] = original
            setattr(module, name, self._wrap(name, original))

    def restore(self, module) -> None:
        for name, original in self._original.items():
            setattr(module, name, original)
        self._original.clear()

    def _wrap(self, name: str, original):
        def wrapper(query, key, value, *args, **kwargs):
            g = kwargs["g"] if "g" in kwargs else (args[0] if len(args) > 0 else None)
            beta = kwargs["beta"] if "beta" in kwargs else (args[1] if len(args) > 1 else None)
            layer_idx = self.current_layer
            recording = self.enabled and layer_idx in self.layers
            if recording:
                if g is None or beta is None:
                    raise SystemExit(
                        f"{name} was called without g/beta (positional or keyword); "
                        "cannot capture beta/g for the golden"
                    )
                self.store[(layer_idx, "q_pre")] = query.detach().clone()
                self.store[(layer_idx, "k_pre")] = key.detach().clone()
                self.store[(layer_idx, "v")] = value.detach().clone()
                self.store[(layer_idx, "beta")] = beta.detach().clone()
                self.store[(layer_idx, "g")] = g.detach().clone()

            out = original(query, key, value, *args, **kwargs)

            if recording:
                core, final_state = out
                self.store[(layer_idx, "core_attn_out")] = core.detach()
                self.store[(layer_idx, "final_state")] = final_state
                self.store[(layer_idx, "rule")] = name
            return out

        wrapper.__name__ = f"capture_{name}"
        return wrapper


def register_hooks(model, layers: list[int], store: dict, rule: RuleCapture) -> list:
    """Per-layer closures for the module stages; only the rule needs ``current_layer``."""
    handles = []
    # Attribution has to cover **every** linear-attention layer, not just the captured
    # ones: all 18 share one rule symbol, so a pre-hook registered only on L0/L2 would
    # leave ``current_layer`` at 0 while L1 runs (L1's rule call would overwrite L0's
    # stages) and at 2 while L3..L23 run (overwriting L2). The cache-state invariant in
    # ``_check_cache_attribution`` is what caught exactly this during development.
    for layer_idx, layer in enumerate(model.model.layers):
        linear = getattr(layer, "linear_attn", None)
        if linear is None:
            continue
        handles.append(
            linear.register_forward_pre_hook(
                lambda _module, _inputs, idx=layer_idx: setattr(rule, "current_layer", idx)
            )
        )
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        linear = layer.linear_attn
        handles += [
            layer.register_forward_pre_hook(_store_input(store, layer_idx, "layer_in")),
            layer.register_forward_hook(_store_output(store, layer_idx, "layer_out")),
            layer.input_layernorm.register_forward_hook(
                _store_output(store, layer_idx, "input_layernorm")
            ),
            linear.in_proj_qkv.register_forward_hook(
                _store_output(store, layer_idx, "in_proj_qkv")
            ),
            linear.in_proj_z.register_forward_hook(
                _store_output(store, layer_idx, "in_proj_z")
            ),
            linear.in_proj_b.register_forward_hook(
                _store_output(store, layer_idx, "in_proj_b")
            ),
            linear.in_proj_a.register_forward_hook(
                _store_output(store, layer_idx, "in_proj_a")
            ),
            # The gated norm returns (T * H_v, head_v_dim); it is reshaped to (T, value_dim)
            # when the stage is saved.
            linear.norm.register_forward_hook(_store_output(store, layer_idx, "norm_out")),
        ]
    return handles


def sampled_positions(length: int) -> list[int]:
    return sorted({0, length // 2, length - 1})


def derive_conv_out(store: dict, layer_idx: int, conv_weight: torch.Tensor, length: int):
    """``conv_out`` from the captured ``in_proj_qkv``, exactly as the operator computes it.

    Channels-first, ``padding = k - 1``, slice the first ``T`` columns, then SiLU. This is
    the **post-SiLU** activation; the conv *state* is the pre-SiLU input columns, so the
    two are not the same measurement (P-49). Derived instead of hooked because HF has no
    module boundary after the conv, and because deriving it keeps the fixture free of one
    more hook that could silently capture the pre-SiLU buffer.
    """
    mixed = store[(layer_idx, "in_proj_qkv")]  # (1, T, conv_dim)
    conv_dim = conv_weight.shape[0]
    kernel = conv_weight.shape[-1]
    if tuple(mixed.shape) != (1, length, conv_dim):
        raise SystemExit(
            f"L{layer_idx}: in_proj_qkv is {tuple(mixed.shape)}, expected "
            f"{(1, length, conv_dim)}"
        )
    conv = F.conv1d(
        mixed.transpose(1, 2), conv_weight, bias=None, padding=kernel - 1, groups=conv_dim
    )[:, :, :length]
    return F.silu(conv).transpose(1, 2)  # back to (1, T, conv_dim)


def _rows(tensor: torch.Tensor, positions: list[int]) -> np.ndarray:
    """Drop the batch axis, sample the three rows, fp32 numpy."""
    if tensor.shape[0] != 1:
        raise SystemExit(f"expected batch 1, got {tuple(tensor.shape)}")
    rows = tensor[0].to(torch.float32)
    index = torch.tensor(positions, dtype=torch.long)
    return rows.index_select(0, index).cpu().numpy()


def _all_rows(tensor: torch.Tensor) -> np.ndarray:
    if tensor.shape[0] != 1:
        raise SystemExit(f"expected batch 1, got {tuple(tensor.shape)}")
    return tensor[0].to(torch.float32).cpu().numpy()


# ---------------------------------------------------------------------------- main


def main() -> int:
    # P-57: sample git state *first*. The new fixture directory is itself untracked, so a
    # ``git status`` run after writing it would report dirty=true even on a clean tree.
    source = {**git_state(), "command": " ".join(sys.argv)}

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=None, help="commit hash; recorded verbatim in meta")
    ap.add_argument("--model-path", default=None, help="local snapshot dir (skips cache lookup)")
    ap.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda when RAM is short")
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"output dir; must NOT exist yet (default {DEFAULT_OUT}). The formal 0002 is "
        "generated by the user, never into tests/golden/ by default",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    if args.model_path:
        snapshot = Path(args.model_path)
        revision = args.revision or snapshot.name
    else:
        snapshot, revision = resolve_snapshot(args.model, args.revision)
    if args.revision and revision != args.revision:
        raise SystemExit(
            f"--revision {args.revision} resolved to snapshot {revision}; refusing to "
            "record a different commit in meta (P-57)"
        )

    # Not "exists but empty": an empty directory is already untracked and would make the
    # formal fixture's meta dirty (docs/w6-plan.md §6.2, P-57).
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(
            f"{out_dir} already exists; golden is immutable — remove it or pick a new id "
            "(tests/golden/README.md)"
        )

    prompt_ids = [p.strip() for p in args.ids.split(",") if p.strip()]
    if not prompt_ids:
        raise SystemExit("--ids is empty")
    layers = [int(layer) for layer in args.layers]
    if not layers:
        raise SystemExit("--layers is empty")

    threads = cap_torch_threads()
    pinned, changed = pin_torch_kernels()
    model, tokenizer, text_config = load_text_decoder(snapshot, device)
    torch.set_grad_enabled(False)

    for layer_idx in layers:
        kind = text_config.layer_types[layer_idx]
        if kind != "linear_attention":
            raise SystemExit(
                f"layer {layer_idx} is {kind!r}; the W6 golden needs linear_attention layers"
            )

    n_head = int(text_config.num_attention_heads)
    n_kv_head = int(text_config.num_key_value_heads)
    head_dim = int(getattr(text_config, "head_dim", None) or text_config.hidden_size // n_head)
    num_k_heads = int(text_config.linear_num_key_heads)
    num_v_heads = int(text_config.linear_num_value_heads)
    key_head_dim = int(text_config.linear_key_head_dim)
    value_head_dim = int(text_config.linear_value_head_dim)
    kernel = int(text_config.linear_conv_kernel_dim)
    conv_dim = 2 * num_k_heads * key_head_dim + num_v_heads * value_head_dim
    value_dim = num_v_heads * value_head_dim
    chunk_size = 64  # D-W6-6; the layer's _CHUNK_SIZE

    prompt_texts = _load_prompt_texts(prompt_ids)
    arrays: dict[str, np.ndarray] = {}
    prompt_meta = []
    store: dict[tuple[int, str], object] = {}
    rule = RuleCapture(layers, store)
    rule.install(sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"])

    try:
        for prompt_id in prompt_ids:
            tokens = tokenizer(prompt_texts[prompt_id], return_tensors="pt").input_ids
            input_ids = tokens.to(device)
            length = int(input_ids.shape[1])
            positions = sampled_positions(length)

            store.clear()
            handles = register_hooks(model, layers, store, rule)
            rule.enabled = True
            try:
                # use_cache=True is what makes the HF delta rule return its final state and
                # write conv/recurrent states into the DynamicCache; logits_to_keep=1 trims
                # the lm_head matmul we do not need (the fixture stores no logits).
                cache = model(
                    input_ids, use_cache=True, logits_to_keep=1
                ).past_key_values
            finally:
                rule.enabled = False
                rule.current_layer = None
                for handle in handles:
                    handle.remove()

            arrays[f"{prompt_id}/input_ids"] = np.asarray(
                input_ids[0].cpu(), dtype=np.int32
            )
            arrays[f"{prompt_id}/positions"] = np.asarray(positions, dtype=np.int32)

            for layer_idx in layers:
                linear = model.model.layers[layer_idx].linear_attn
                conv_weight = linear.conv1d.weight.detach()
                if int(conv_weight.shape[-1]) != kernel:
                    raise SystemExit(
                        f"L{layer_idx}: conv weight kernel is {conv_weight.shape[-1]}, "
                        f"config says {kernel}"
                    )
                conv_out = derive_conv_out(store, layer_idx, conv_weight, length)
                rule_name = store.get((layer_idx, "rule"))
                if rule_name is None:
                    raise SystemExit(
                        f"L{layer_idx}: the delta rule was never captured — the wrapper "
                        "did not fire for this layer (hook attribution bug?)"
                    )

                base = f"L{layer_idx}/{prompt_id}"
                arrays[f"{base}/layer_in"] = _all_rows(store[(layer_idx, "layer_in")])

                stages: dict[str, np.ndarray] = {
                    "input_layernorm": _rows(store[(layer_idx, "input_layernorm")], positions),
                    "in_proj_qkv": _rows(store[(layer_idx, "in_proj_qkv")], positions),
                    "in_proj_z": _rows(store[(layer_idx, "in_proj_z")], positions),
                    "in_proj_b": _rows(store[(layer_idx, "in_proj_b")], positions),
                    "in_proj_a": _rows(store[(layer_idx, "in_proj_a")], positions),
                    "conv_out": _rows(conv_out, positions),
                    "q_pre": _rows(store[(layer_idx, "q_pre")], positions),
                    "k_pre": _rows(store[(layer_idx, "k_pre")], positions),
                    "beta": _rows(store[(layer_idx, "beta")], positions),
                    "g": _rows(store[(layer_idx, "g")], positions),
                    "core_attn_out": _rows(store[(layer_idx, "core_attn_out")], positions),
                    # (T * H_v, head_v_dim) -> (T, value_dim), then sample rows.
                    "norm_out": _rows(
                        store[(layer_idx, "norm_out")].reshape(1, length, value_dim),
                        positions,
                    ),
                    "layer_out": _rows(store[(layer_idx, "layer_out")], positions),
                }

                if prompt_id in FINAL_PROMPTS:
                    final_state = store.get((layer_idx, "final_state"))
                    if final_state is None:
                        raise SystemExit(
                            f"L{layer_idx}/{prompt_id}: the rule returned no final state; "
                            "use_cache=True should have set output_final_state"
                        )
                    _check_cache_attribution(
                        cache, layer_idx, final_state, conv_dim, kernel, num_v_heads,
                        key_head_dim, value_head_dim, prompt_id,
                    )
                    layer_cache = cache.layers[layer_idx]
                    # P-49: HF stores kernel_size=4 columns; W6's contract is the minimal
                    # sufficient kernel-1=3. The oldest column never enters the next
                    # token's window, so the slice is exact, not an approximation.
                    conv_state = layer_cache.conv_states[0]
                    stages["final_state"] = (
                        final_state[0].to(torch.float32).cpu().numpy()
                    )
                    stages["final_conv_state"] = (
                        conv_state[0, :, -(kernel - 1):].to(torch.float32).cpu().numpy()
                    )

                # P-50: beta is already sigmoid-ed and g is already log-space <= 0. A
                # positive g would mean HF changed the projection and the fixture's
                # "beta/g" rows would silently encode a different contract.
                if stages["g"].max() > 0.0:
                    raise SystemExit(
                        f"L{layer_idx}/{prompt_id}: g has positive entries; it must be "
                        "log-space <= 0 (P-50)"
                    )
                if stages["beta"].min() <= 0.0 or stages["beta"].max() >= 1.0:
                    raise SystemExit(
                        f"L{layer_idx}/{prompt_id}: beta is outside (0, 1); it must be "
                        "sigmoid-ed (P-50)"
                    )

                for name, tensor in stages.items():
                    arrays[f"{base}/{name}"] = tensor

                print(
                    f"  L{layer_idx}/{prompt_id}: {length} tokens, {rule_name}, "
                    f"positions {positions}"
                )
                for name in SAMPLED_STAGES + FINAL_STAGES:
                    if name not in stages:
                        continue
                    tensor = stages[name]
                    print(
                        f"      {name:<17} {str(tensor.shape):<18} "
                        f"max|ref|={np.abs(tensor).max():.3e}"
                    )

            prompt_meta.append(
                {
                    "id": prompt_id,
                    "n_tokens": length,
                    "truncation": "none (full prompt)",
                    "text_sha256": hashlib.sha256(prompt_texts[prompt_id].encode()).hexdigest(),
                    "token_sha256": hashlib.sha256(
                        np.asarray(input_ids[0].cpu(), dtype=np.int32).tobytes()
                    ).hexdigest(),
                    "sampled_positions": positions,
                    "input_ids": [int(token) for token in input_ids[0].cpu()],
                }
            )
    finally:
        rule.restore(sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"])

    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir / "tensors.npz", **arrays)

    meta = {
        "spec": SPEC,
        "model": {
            "repo": args.model,
            "revision": revision,
            "slug": slug_of(args.model),
            "class": type(model).__name__,
            "config_class": type(text_config).__name__,
            "dtype": "float32",
            "device": str(device),
            "attn_implementation": "eager",
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_layer": int(text_config.num_hidden_layers),
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "head_dim": head_dim,
            "linear_num_key_heads": num_k_heads,
            "linear_num_value_heads": num_v_heads,
            "linear_key_head_dim": key_head_dim,
            "linear_value_head_dim": value_head_dim,
            "linear_conv_dim": conv_dim,
            "linear_value_dim": value_dim,
            "linear_conv_kernel_dim": kernel,
            "chunk_size": chunk_size,
        },
        "tokenizer": {
            "repo": args.model,
            "revision": revision,
            "class": type(tokenizer).__name__,
        },
        "prompts": prompt_meta,
        "layers": layers,
        "stages": {
            "sampled": list(SAMPLED_STAGES),
            "hooked": list(HOOKED_STAGES),
            "full": list(FULL_STAGES),
            # Only the prompts this run actually has: --ids s01 alone stores no states.
            "final_prompts": [pid for pid in prompt_ids if pid in FINAL_PROMPTS],
            "rules": {
                "layer_in": "decoder layer input (residual stream), all T positions (T, hidden)",
                "input_layernorm": "layer.input_layernorm output (n_pos, hidden)",
                "in_proj_qkv": "layer.linear_attn.in_proj_qkv output, raw projection "
                "(n_pos, conv_dim)",
                "in_proj_z": "in_proj_z output, the gated norm's gate logits "
                "(n_pos, value_dim)",
                "in_proj_b": "beta logits (in_proj_b output), before sigmoid (n_pos, n_v_heads)",
                "in_proj_a": "decay logits (in_proj_a output), before softplus "
                "(n_pos, n_v_heads)",
                "conv_out": "conv_out is derived and post-SiLU: F.conv1d(in_proj_qkv, "
                "conv1d.weight, padding=k-1, groups=conv_dim)[:, :, :T] then F.silu, "
                "(n_pos, conv_dim). Not hooked, and not comparable to conv_state, which "
                "is the pre-SiLU input columns (P-49)",
                "q_pre": "q_pre/k_pre are raw, before l2norm: split after the conv and "
                "reshaped (B, T, H_k, K), captured at the rule entry (n_pos, n_k_heads, "
                "head_k_dim)",
                "k_pre": "q_pre/k_pre are raw, before l2norm (n_pos, n_k_heads, head_k_dim)",
                "beta": "beta already sigmoid-ed (n_pos, n_v_heads)",
                "g": "g in log space, <= 0: -exp(A_log) * softplus(a + dt_bias) "
                "(n_pos, n_v_heads)",
                "core_attn_out": "delta-rule output before the gated norm "
                "(n_pos, n_v_heads, head_v_dim)",
                "norm_out": "linear_attn.norm(core, z) output, reshaped (T, value_dim) "
                "then sampled (n_pos, value_dim)",
                "layer_out": "decoder layer output, residual and MLP included (n_pos, hidden)",
                "final_state": "m02 only; final_state is HF cache['...'].recurrent_states[0] "
                "== the chunk rule's final state (B, H_v, K, V) = (16, 128, 128), fp32",
                "final_conv_state": "m02 only; final_conv_state is HF cache[..., -3:]: "
                "conv_states[0][..., -(k-1):] = (conv_dim, 3), pre-SiLU input columns "
                "(P-49: HF stores 4)",
            },
            "sampling": "rows sorted({0, T // 2, T - 1}) per prompt; see "
            "{prompt}/positions in the npz",
        },
        "tensors": {
            "npz_keys": "L{layer}/{prompt}/{stage}; plus {prompt}/input_ids and "
            "{prompt}/positions",
            "input_ids": "int32 (T,) token ids the HF reference consumed",
            "positions": "int32 (n_sampled,) rows present for the sampled stages",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "environment": {
            "pinned_kernels": pinned,
            "pinned_rewrapped": changed,
            "torch_threads": threads,
        },
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    size = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"\n✓ {out_dir}  ({size / 1e6:.2f} MB, limit {MAX_BYTES / 1e6:.0f} MB)")
    if size > MAX_BYTES:
        raise SystemExit(
            f"fixture is {size / 1e6:.1f} MB > {MAX_BYTES / 1e6:.0f} MB; shrink the "
            "sampled stages before committing"
        )
    print(
        f"   layers {layers}, prompts {prompt_ids}, pinned {len(pinned)} kernels "
        f"({', '.join(changed) if changed else 'already raw'})"
    )
    return 0


def _check_cache_attribution(
    cache, layer_idx, final_state, conv_dim, kernel, num_v_heads, key_head_dim,
    value_head_dim, prompt_id,
):
    """The rule's final state must be the one HF stored for *this* layer.

    This is the strongest available check that the module-level ``current_layer``
    attribution is right: if the wrapper recorded layer 2's rule under layer 0's key, the
    cache at index 0 would disagree. Shapes are checked as well so a bad slice fails here
    instead of in the alignment table.
    """
    # HF and the production operator both return (B, H_v, K, V); the fixture drops B.
    expected_state = (1, num_v_heads, key_head_dim, value_head_dim)
    if tuple(final_state.shape) != expected_state:
        raise SystemExit(
            f"L{layer_idx}/{prompt_id}: final state is {tuple(final_state.shape)}, "
            f"expected {expected_state}"
        )
    conv_state = cache.layers[layer_idx].conv_states[0]
    if tuple(conv_state.shape) != (1, conv_dim, kernel):
        raise SystemExit(
            f"L{layer_idx}/{prompt_id}: HF conv cache is {tuple(conv_state.shape)}, "
            f"expected {(1, conv_dim, kernel)}"
        )
    if not torch.equal(
        cache.layers[layer_idx].recurrent_states[0], final_state.to(torch.float32)
    ):
        raise SystemExit(
            f"L{layer_idx}/{prompt_id}: the rule's final state differs from HF's "
            "recurrent_states[0] for this layer index — capture attribution is wrong"
        )


def _load_prompt_texts(ids: list[str]) -> dict[str, str]:
    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    by_id = {p["id"]: p["text"] for p in spec["prompts"]}
    missing = [prompt_id for prompt_id in ids if prompt_id not in by_id]
    if missing:
        raise SystemExit(f"unknown prompt ids {missing}; bench/prompts.json v{spec['version']}")
    return {prompt_id: by_id[prompt_id] for prompt_id in ids}


if __name__ == "__main__":
    raise SystemExit(main())
