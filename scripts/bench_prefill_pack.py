"""G1 spike: packed FA2 varlen prefill vs per-request SDPA prefill (W4.5).

Run on the CUDA host::

    python scripts/bench_prefill_pack.py --spike fa2
    python scripts/bench_prefill_pack.py --spike fa2 --counts 2,4,8 --json docs/w4_5_prefill_pack.json

The script is mechanism evidence, not a formal run: it uses the frozen prompt set but a
single warm model, no scheduler and no trace. It answers the G1 question -- can the packed
forward replace the per-request one without changing token ids, and does it get cheaper as
the group grows -- and it reports the *forward* time only (the scheduler's queue/pack wait
lives in ``bench/``).

Packed and serial both write into a real ``KvSlab`` (the serial arm through
``prefill_into_seat``, the packed arm through ``prefill_group``), so the comparison includes
the KV write that the serving path pays. A recording-only sink would measure a kernel that
the engine never runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "bench" / "prompts.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _load(model_name: str, revision: str | None, device: str, dtype: str):
    sys.path.insert(0, str(ROOT))
    from bench.engines.minillm import ServeEngine

    adapter = ServeEngine(
        name=model_name,
        revision=revision,
        dtype=dtype,
        device=device,
    )
    adapter.load()
    return adapter


def _prompt_map(path: Path, adapter) -> dict[str, list[int]]:
    from bench.loadgen import PromptSet

    prompt_set = PromptSet.load(path)
    return {prompt.id: adapter.encode(prompt.text) for prompt in prompt_set.prompts}


def _pick(prompts: dict[str, list[int]], count: int) -> list[tuple[str, list[int]]]:
    """A mixed set: short, medium, long, repeated until ``count`` entries."""
    order = ["s01", "m01", "l01", "s05", "m04", "l03", "m08", "l05", "s02", "m02", "l02"]
    chosen = [key for key in order if key in prompts][:count]
    if len(chosen) < count:
        chosen += [key for key in prompts if key not in chosen][: count - len(chosen)]
    return [(key, prompts[key]) for key in chosen]


def _time_cuda(fn, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


@torch.inference_mode()
def _serial_arm(engine, prompts: list[list[int]], lengths: list[int], *, warmup: int, repeats: int):
    slab = engine.open_pool(len(prompts), max(lengths) + 8)

    def run():
        slab.reset()
        return [engine.prefill_into_seat(row, prompt) for row, prompt in enumerate(prompts)]

    times = _time_cuda(run, warmup=warmup, repeats=repeats)
    logits = run()
    return times, torch.cat(logits)


@torch.inference_mode()
def _packed_arm(engine, prompts: list[list[int]], lengths: list[int], *, warmup: int, repeats: int):
    engine.open_pool(len(prompts), max(lengths) + 8)
    specs = [
        {"id": index, "prompt_ids": prompt, "seat": index, "budget": 0}
        for index, prompt in enumerate(prompts)
    ]

    def run():
        engine.open_pool(len(prompts), max(lengths) + 8)
        return engine.prefill_group(specs)

    times = _time_cuda(run, warmup=warmup, repeats=repeats)
    logits = run()
    return times, logits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike", default="fa2", choices=["fa2"], help="G1 spike mode")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--counts", default="2,4,8", help="comma-separated group sizes")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-prefill-tokens", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.device != "cuda" or not torch.cuda.is_available():
        print("error: this spike is CUDA-only (FA2 varlen); use --device cuda", file=sys.stderr)
        return 2

    adapter = _load(args.model, args.revision, args.device, args.dtype)
    engine = adapter.engine
    engine.prefill_mode = "packed"
    engine.max_prefill_tokens = int(args.max_prefill_tokens)
    prompts = _prompt_map(args.prompts, adapter)

    rows = []
    for count in [int(value) for value in args.counts.split(",") if value.strip()]:
        selected = _pick(prompts, count)
        ids = [key for key, _ in selected]
        prompt_ids = [value for _, value in selected]
        lengths = [len(value) for value in prompt_ids]
        if sum(lengths) > args.max_prefill_tokens:
            print(
                f"warning: {count} prompts total {sum(lengths)} tokens > "
                f"max_prefill_tokens={args.max_prefill_tokens}; skipping",
                file=sys.stderr,
            )
            continue
        serial_times, serial_logits = _serial_arm(
            engine, prompt_ids, lengths, warmup=args.warmup, repeats=args.repeats
        )
        packed_times, packed_logits = _packed_arm(
            engine, prompt_ids, lengths, warmup=args.warmup, repeats=args.repeats
        )
        serial_ids = [int(row.argmax()) for row in serial_logits]
        packed_ids = [int(row.argmax()) for row in packed_logits]
        diff = float((serial_logits.float() - packed_logits.float()).abs().max())
        row = {
            "count": count,
            "prompt_ids": ids,
            "tokens": sum(lengths),
            "max_seqlen": max(lengths),
            "serial_ms_median": statistics.median(serial_times),
            "packed_ms_median": statistics.median(packed_times),
            "speedup": statistics.median(serial_times) / statistics.median(packed_times),
            "serial_ms": serial_times,
            "packed_ms": packed_times,
            "token_ids_equal": serial_ids == packed_ids,
            "logits_max_abs_diff": diff,
            "serial_token_ids": serial_ids,
            "packed_token_ids": packed_ids,
        }
        rows.append(row)
        print(
            f"B={count:>2} tokens={row['tokens']:>5} serial={row['serial_ms_median']:.3f} ms "
            f"packed={row['packed_ms_median']:.3f} ms speedup={row['speedup']:.3f}x "
            f"ids={'ok' if row['token_ids_equal'] else 'MISMATCH'} max|d|={diff:.2e}"
        )

    result = {
        "spike": "fa2-varlen",
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "prefill_backend": engine.attn_backends()["prefill"],
        "rows": rows,
        "all_token_ids_equal": all(row["token_ids_equal"] for row in rows),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if result["all_token_ids_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
