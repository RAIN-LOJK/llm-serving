"""E0a: packed whole-prompt prefill must equal the per-request prefill (W4.5).

    python scripts/check_packed_equiv.py --counts 2,4,8
    python scripts/check_packed_equiv.py --counts 8 --json /tmp/w45_packed_equiv.json

Compares the last-token logits and greedy ids of ``prefill_group`` against the
``KVcache`` per-sequence reference (``prefill_one``) for mixed prompt sets, then checks
that changing request 0's tokens cannot move request 1's logits. It is a dry-run gate for
E0a, not a formal run: exit code 1 means stop and fix the packing/position/slot map before
any performance run.
"""

from __future__ import annotations

import argparse
import json
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

    adapter = ServeEngine(name=model_name, revision=revision, dtype=dtype, device=device)
    adapter.load()
    return adapter


def _prompts(adapter, path: Path, count: int):
    from bench.loadgen import PromptSet

    prompt_set = PromptSet.load(path)
    by_id = {prompt.id: adapter.encode(prompt.text) for prompt in prompt_set.prompts}
    order = ["s01", "m01", "l01", "s05", "m04", "l03", "m08", "l05", "s02", "m02", "l02"]
    keys = [key for key in order if key in by_id][:count]
    return [(key, by_id[key]) for key in keys]


def _check(engine, selected, *, max_new_tokens: int = 1):
    ids = [key for key, _ in selected]
    prompt_ids = [value for _, value in selected]
    lengths = [len(value) for value in prompt_ids]
    engine.open_pool(len(prompt_ids), max(lengths) + 8)
    engine.prefill_mode = "packed"
    engine.max_prefill_tokens = max(8192, sum(lengths))

    reference = torch.cat([engine.prefill_one(prompt)[0] for prompt in prompt_ids])
    packed = engine.prefill_group(
        [
            {"id": index, "prompt_ids": prompt, "seat": index, "budget": max_new_tokens}
            for index, prompt in enumerate(prompt_ids)
        ]
    )
    diff = float((reference.float() - packed.float()).abs().max())
    ref_ids = [int(row.argmax()) for row in reference]
    packed_ids = [int(row.argmax()) for row in packed]

    # No cross-request attention: changing segment 0 must leave every later segment
    # bit-identical (the segments share one packed buffer, not one attention problem).
    changed = [list(prompt) for prompt in prompt_ids]
    changed[0][-1] = (changed[0][-1] + 7) % engine.vocab_size
    engine.open_pool(len(changed), max(len(value) for value in changed) + 8)
    after = engine.prefill_group(
        [
            {"id": index, "prompt_ids": prompt, "seat": index, "budget": max_new_tokens}
            for index, prompt in enumerate(changed)
        ]
    )
    leak = float((packed[1:] - after[1:]).abs().max()) if len(selected) > 1 else 0.0
    return {
        "prompt_ids": ids,
        "tokens": sum(lengths),
        "max_abs_diff": diff,
        "token_ids": packed_ids,
        "reference_ids": ref_ids,
        "token_ids_equal": ref_ids == packed_ids,
        "cross_request_max_abs_diff": leak,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--counts", default="2,4,8")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    adapter = _load(args.model, args.revision, args.device, args.dtype)
    engine = adapter.engine
    rows = []
    for count in [int(value) for value in args.counts.split(",") if value.strip()]:
        selected = _prompts(adapter, args.prompts, count)
        if len(selected) < count:
            print(f"error: only {len(selected)} prompts available, asked {count}", file=sys.stderr)
            return 2
        row = _check(engine, selected)
        rows.append(row)
        print(
            f"B={count:>2} tokens={row['tokens']:>5} max|d|={row['max_abs_diff']:.2e} "
            f"ids={'ok' if row['token_ids_equal'] else 'MISMATCH'} "
            f"leak={row['cross_request_max_abs_diff']:.2e}"
        )

    ok = all(row["token_ids_equal"] for row in rows) and all(
        row["cross_request_max_abs_diff"] == 0.0 for row in rows
    )
    result = {
        "device": args.device,
        "dtype": args.dtype,
        "prefill_backend": engine.attn_backends()["prefill"],
        "rows": rows,
        "ok": ok,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
