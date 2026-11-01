"""Probe: where does a B=1 decode step spend its time? (NOT a run; writes no results/)

Three decode paths, all teacher-forced (every step feeds the same fixed token, so the paths
cannot diverge and no sampling / ``.item()`` enters the timed region):

  (a) sdpa + KVcache(cat)         -- the W2-0003 path (probe-local backend swap)
  (b) hand_baddbmm + KVcache(cat) -- kernel swapped only, no mask (probe-local)
  (c) hand_baddbmm + KvSlab       -- the shipped fixed path (D4), with the slab mask/bias

The three-way split is pure attribution: (a) vs (b) is the kernel at B=1, (b) vs (c) is the
slab's host bookkeeping. (a) and (b) no longer exist in the shipped engine (D4 deleted the
knob), so they are built *here* by pointing ``Attention.backend_decode`` at the other
backend on the in-process model. Nothing is added to ``model/`` for them.

(b) is conditional: it is only an attribution tool, so the script measures (a) and (c)
first and runs (b) only when the residual ``(c)/(a)`` gap exceeds ``--threshold``
(default 5%). ``--all`` forces it.

Timing: one CUDA-event interval around a loop of ``--steps`` steps, after ``--warmup``
untimed steps; the reported number is ``elapsed / steps``. ``--ctx`` and ``--batch`` are
comma-separated.

    python scripts/probe_decode_paths.py
    python scripts/probe_decode_paths.py --ctx 100,512,2048 --batch 1,8 --steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import KVcache, KvSlab, MiniLLM  # noqa: E402
from model.attention.hand_baddbmm import BACKEND as HAND  # noqa: E402
from model.attention.sdpa import BACKEND as SDPA  # noqa: E402
from model.loading import cached_snapshot, load_checkpoint  # noqa: E402
from serve.engine import SLAB_GRANULARITY, MiniLLMEngine  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


# ---------------------------------------------------------------- model / state helpers

def load_model(device: str = "cuda") -> MiniLLM:
    path = cached_snapshot(MODEL, REVISION) or cached_snapshot(MODEL)
    if path is None:
        raise SystemExit(f"{MODEL}@{REVISION} is not in the HF cache")
    return load_checkpoint(path, dtype=torch.bfloat16, device=device)


def _set_decode_backend(model: MiniLLM, backend) -> None:
    """Probe-local: route every layer's T==1 dispatch to ``backend``."""
    for layer in model.layers:
        layer.self_attn.backend_decode = backend


def _prompt_ids(ctx: int, vocab: int, seed: int = 0) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (ctx,), generator=gen).tolist()


def _cat_cache(engine: MiniLLMEngine, prompt: list[int], batch: int) -> KVcache:
    """Prefill once, then broadcast the history over identical rows (no length padding).

    All rows share one prompt and one decode token, so the decode call needs no mask
    (P-05/P-22 do not apply). This is the container the W2-0003 path used.
    """
    _, cache = engine.prefill(prompt)
    if batch > 1:
        for layer in range(engine.model.config.n_layer):
            k, v = cache.get_kv(layer)
            cache.set_kv(
                layer,
                (k.expand(batch, -1, -1, -1).contiguous(), v.expand(batch, -1, -1, -1).contiguous()),
            )
    return cache


def _slab(engine: MiniLLMEngine, prompt: list[int], batch: int, budget: int) -> KvSlab:
    cache = _cat_cache(engine, prompt, batch)
    need = len(prompt) + budget
    with torch.inference_mode():  # the slab's buffers are born inside inference mode
        if batch == 1:
            slab = engine._slab(need)  # same bucket policy as the shipped engine
        else:
            bucket = min(max(-(-need // SLAB_GRANULARITY) * SLAB_GRANULARITY, need), engine.max_position)
            slab = KvSlab(
                engine.model.config.n_layer,
                batch,
                engine.model.config.n_kv_head,
                engine.model.head_dim,
                bucket,
                dtype=engine.model.token_embedding.weight.dtype,
                device=engine.device,
                n_head=engine.model.config.n_head,
            )
        for layer in range(engine.model.config.n_layer):
            slab.fill_from(cache, layer)  # rows share a length, so the whole-row copy is exact
    return slab


# --------------------------------------------------------------------------- timing

def _ms_per_step(step, *, warmup: int, steps: int) -> float:
    """CUDA-event interval around ``steps`` calls; warmup calls are untimed.

    Everything runs under ``torch.inference_mode`` because the engine's methods are, and a
    tensor born inside it cannot be written outside it.
    """
    with torch.inference_mode():
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            step()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / steps


# --------------------------------------------------------------------------- paths

def measure_a(model, engine, prompt, batch, args) -> float:
    """(a) sdpa + cat: swap the T==1 backend to SDPA, keep the growing KVcache."""
    cache = _cat_cache(engine, prompt, batch)
    ids = torch.full((batch, 1), prompt[-1], dtype=torch.long, device=engine.device)

    def step():
        model(ids, past_keys_values=cache, logits_to_keep=1)

    _set_decode_backend(model, SDPA)
    try:
        return _ms_per_step(step, warmup=args.warmup, steps=args.steps)
    finally:
        _set_decode_backend(model, HAND)


def measure_b(model, engine, prompt, batch, args) -> float:
    """(b) hand_baddbmm + cat: same tensors/container as (a), kernel swapped only."""
    cache = _cat_cache(engine, prompt, batch)
    ids = torch.full((batch, 1), prompt[-1], dtype=torch.long, device=engine.device)

    def step():
        model(ids, past_keys_values=cache, logits_to_keep=1)

    return _ms_per_step(step, warmup=args.warmup, steps=args.steps)


def measure_c(engine, prompt, batch, args) -> float:
    """(c) hand_baddbmm + KvSlab: the shipped path, via ``decode_slab``."""
    slab = _slab(engine, prompt, batch, args.warmup + args.steps + 8)
    tokens = [prompt[-1]] * batch
    return _ms_per_step(lambda: engine.decode_slab(tokens, slab), warmup=args.warmup, steps=args.steps)


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", default="100,512,2048")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100, help="must be >= 50")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="run (b) only when (c)/(a) - 1 exceeds this")
    ap.add_argument("--all", action="store_true", help="always run (b)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.steps < 50:
        raise SystemExit("--steps must be >= 50")
    if not torch.cuda.is_available():
        raise SystemExit("this probe needs CUDA (CUDA-event timing)")

    model = load_model(args.device)
    engine = MiniLLMEngine(model)
    ctxs = [int(x) for x in args.ctx.split(",")]
    batches = [int(x) for x in args.batch.split(",")]

    print(f"model {MODEL} bf16, {model.config.n_layer} layers, teacher-forced (one fixed "
          f"token per step), CUDA events, warmup={args.warmup} excluded, steps={args.steps}")
    print(f"{'ctx':>5} {'B':>3} {'(a) sdpa+cat':>13} {'(c) hand+slab':>14} "
          f"{'(c)/(a)':>8} {'(b) hand+cat':>13}  note")

    for ctx in ctxs:
        for batch in batches:
            prompt = _prompt_ids(ctx, model.config.vocab_size)
            try:
                a = measure_a(model, engine, prompt, batch, args)
                c = measure_c(engine, prompt, batch, args)
            except (RuntimeError, IndexError) as exc:  # e.g. OOM at B=8/ctx2048
                print(f"{ctx:>5} {batch:>3}  skipped: {str(exc).splitlines()[0]}")
                continue
            ratio = c / a
            if args.all or (ratio - 1.0) > args.threshold:
                b = measure_b(model, engine, prompt, batch, args)
                b_txt = f"{b:>13.2f}"
                note = f"(b)/(a) {b / a:.3f}x kernel, (c)/(b) {c / b:.3f}x slab bookkeeping"
            else:
                b_txt = f"{'--':>13}"
                note = "(b) skipped: residual below threshold"
            print(f"{ctx:>5} {batch:>3} {a:>13.2f} {c:>14.2f} {ratio:>8.3f} {b_txt}  {note}")

    print("\nms/step. (a) sdpa+KVcache(cat) = W2-0003; (c) hand_baddbmm+KvSlab = shipped (D4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
