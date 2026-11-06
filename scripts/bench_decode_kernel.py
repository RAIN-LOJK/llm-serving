#!/usr/bin/env python3
"""G2 spike for the W4.5 block-table decode kernel: correctness + per-shape cost.

This is a **spike**, not a run: it writes no ``results/`` directory, records no
provenance and is not a benchmark arm. It answers the two questions the W4.5 plan puts in
front of the Triton decode path:

1. does ``paged_decode`` agree with ``reference_paged_decode`` on the shapes the engine
   will feed it (0.6B plus a wider D=256 synthetic case)?
2. how long does one kernel launch take, and how close is it to the KV bytes the kernel
   actually reads?

Shapes
------
* the Qwen3-0.6B decode shape ``(B=8, H=16, H_kv=8, D=128, L=1664, block_size=16)``;
* a synthetic ``D=256, H_kv=2`` (GQA 8) sweep over ``B=1/8/32`` and ``L=512/2048/8192``,
  which is the shape family the block-table kernel has to stay linear on.

Every row builds a fresh random pool and random block table (reused and non-contiguous
block ids are normal: rows draw independently from one shared pool). ``seqlen`` is random
per row in the last two pages, so the last page is partial in almost every row. The row is
checked against the dense oracle *before* it is timed; a correctness failure makes the
script exit non-zero.

Timing is one CUDA-event pair per call (``--warmup`` untimed calls, then ``--iters``
timed calls), and the report is nearest-rank p50/p95 like ``bench/metrics.py``. The
reported GB/s uses the valid K+V bytes (``sum(seqlen) * 2 * H_kv * D * itemsize``): the
kernel's masked tail loads no invalid column, and GQA heads share the same KV bytes, so
this is the traffic the memory system is actually asked for, not an ``H``-expanded count.

Usage::

    python scripts/bench_decode_kernel.py --spike triton
    python scripts/bench_decode_kernel.py --spike triton --json /tmp/w45_decode_spike.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.metrics import percentiles  # noqa: E402
from model.triton_decode import (  # noqa: E402
    TRITON_AVAILABLE,
    paged_decode,
    reference_paged_decode,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 2e-2, torch.float16: 2e-2}
MIB = 2**20


def _sweep() -> list[dict[str, int]]:
    """The G2 shape list: one 0.6B anchor plus the D=256 synthetic grid."""
    shapes = [
        {
            "name": "qwen3-0.6b",
            "B": 8,
            "H": 16,
            "H_kv": 8,
            "D": 128,
            "L": 1664,
            "block_size": 16,
        }
    ]
    for batch in (1, 8, 32):
        for length in (512, 2048, 8192):
            shapes.append(
                {
                    "name": "synth-d256",
                    "B": batch,
                    "H": 16,
                    "H_kv": 2,
                    "D": 256,
                    "L": length,
                    "block_size": 16,
                }
            )
    return shapes


def _build(shape: dict[str, int], dtype: torch.dtype, seed: int, device: torch.device):
    """Random pool, random table and ragged seqlen for one sweep row.

    The pool is shared by every row of the table (``B * max_pages`` blocks), so a row's
    logical pages reuse physical blocks other rows also use -- the same blocks the real
    allocator would hand out from a common free list. Each row's page list is a random
    permutation, hence non-contiguous and, across rows, overlapping.
    """
    gen = torch.Generator().manual_seed(seed)
    block_size = shape["block_size"]
    max_pages = (shape["L"] + block_size - 1) // block_size
    n_blocks = max(shape["B"] * max_pages, max_pages + 1)

    elements = n_blocks * block_size * shape["H_kv"] * shape["D"]
    k = torch.randn(elements, generator=gen, dtype=torch.float32).reshape(
        n_blocks, block_size, shape["H_kv"], shape["D"]
    )
    v = torch.randn(elements, generator=gen, dtype=torch.float32).reshape_as(k)
    q = torch.randn(shape["B"], shape["H"], shape["D"], generator=gen, dtype=torch.float32)
    k = k.to(device=device, dtype=dtype)
    v = v.to(device=device, dtype=dtype)
    q = q.to(device=device, dtype=dtype)

    # Ragged lengths in the last two pages: every row has a partial last page, and the
    # final row keeps the sweep's nominal L so the reported shape is the max, not the mean.
    low = max(1, shape["L"] - 2 * block_size + 1)
    lengths = torch.randint(
        low, shape["L"] + 1, (shape["B"],), generator=gen, dtype=torch.int32
    )
    lengths[-1] = shape["L"]

    table = torch.empty(shape["B"], max_pages, dtype=torch.int32)
    for row in range(shape["B"]):
        table[row] = torch.randperm(n_blocks, generator=gen)[:max_pages].to(torch.int32)
    return q, k, v, table.to(device), lengths.to(device)


def _time_calls(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    table: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    """Per-call CUDA-event intervals; warmup calls are untimed.

    The wrapper's seqlen bound check is cached per tensor version, so after warmup the
    timed region is the empty-output allocation plus the kernel launch, which is what the
    engine pays once per layer.
    """
    out = None
    for _ in range(warmup):
        out = paged_decode(q, k, v, table, seqlen)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        starts[index].record()
        out = paged_decode(q, k, v, table, seqlen)
        ends[index].record()
    torch.cuda.synchronize()
    # Keep the last output alive across the event synchronisation; it is not read again.
    assert out is not None
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _run_row(
    shape: dict[str, int],
    dtype: torch.dtype,
    *,
    seed: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    q, k, v, table, seqlen = _build(shape, dtype, seed, device)
    tolerance = TOLERANCE[dtype]

    with torch.inference_mode():
        reference = reference_paged_decode(q, k, v, table, seqlen)
        got = paged_decode(q, k, v, table, seqlen)
    max_diff = (got.float() - reference.float()).abs().max().item()
    finite = bool(torch.isfinite(got).all().item())
    argmax_equal = None
    if dtype is not torch.float32:
        argmax_equal = bool(torch.equal(got.argmax(dim=-1), reference.argmax(dim=-1)))
    ok = finite and max_diff < tolerance and argmax_equal is not False

    times = _time_calls(q, k, v, table, seqlen, warmup=warmup, iters=iters)
    stats = percentiles(times, qs=(0.50, 0.95))
    p50 = float(stats["p50"])  # nearest-rank; None only for an empty sample
    p95 = float(stats["p95"])

    tokens = int(seqlen.sum().item())
    traffic_bytes = tokens * 2 * shape["H_kv"] * shape["D"] * dtype.itemsize
    traffic_mb = traffic_bytes / MIB
    gbps = traffic_bytes / (p50 / 1000.0) / 1e9

    row = {
        **shape,
        "n_blocks": int(k.shape[0]),
        "max_blocks": int(table.shape[1]),
        "seqlen_mean": tokens / shape["B"],
        "seqlen_max": int(seqlen.max().item()),
        "dtype": str(dtype).removeprefix("torch."),
        "finite": finite,
        "argmax_equal": argmax_equal,
        "max_abs_diff": max_diff,
        "tolerance": tolerance,
        "ok": ok,
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": min(times),
        "max_ms": max(times),
        "traffic_mb": traffic_mb,
        "gbps": gbps,
        "warmup": warmup,
        "iters": iters,
    }
    return row


def _print_table(rows: list[dict[str, Any]], device_name: str, dtype: str) -> None:
    print(f"device: {device_name} · dtype: {dtype} · times are one paged_decode call")
    header = (
        f"{'shape':<38} {'max|d|':>9} {'p50 ms':>9} {'p95 ms':>9} "
        f"{'GB/s':>8} {'MiB':>8} {'ok':>3}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        label = (
            f"B{row['B']} H{row['H']}/{row['H_kv']} D{row['D']} "
            f"L{row['L']} bs{row['block_size']}"
        )
        print(
            f"{label:<38} {row['max_abs_diff']:>9.2e} {row['p50_ms']:>9.4f} "
            f"{row['p95_ms']:>9.4f} {row['gbps']:>8.1f} {row['traffic_mb']:>8.1f} "
            f"{'yes' if row['ok'] else 'NO':>3}"
        )


def _clock_warmup(device: torch.device, rounds: int = 200) -> None:
    """Lift the GPU out of idle clocks before a row's timed section.

    A 4090 idles around 210 MHz; the B=1/L=512 row is short enough that ten warmup calls
    would otherwise be measured before the clock governor reacts. This burn-in sits
    outside every reported interval and uses a separate allocation.
    """
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    for _ in range(rounds):
        a = torch.mm(a, b)
    torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spike",
        choices=("triton",),
        default="triton",
        help="which decode backend to spike (only 'triton' exists)",
    )
    parser.add_argument("--json", default=None, help="write the rows to this JSON path")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not TRITON_AVAILABLE:
        print("paged_decode needs Triton; this host cannot import it", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("paged_decode needs CUDA; torch.cuda is unavailable", file=sys.stderr)
        return 1
    if args.warmup < 1 or args.iters < 1:
        parser.error("--warmup and --iters must be positive")

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    device_name = torch.cuda.get_device_name(device)

    rows: list[dict[str, Any]] = []
    for index, shape in enumerate(_sweep()):
        # The small rows are shorter than the clock governor's reaction time; a short
        # burn-in before the per-shape 10-call warmup keeps their p50 comparable to the
        # large rows. It is outside every reported interval.
        _clock_warmup(device, rounds=50)
        row = _run_row(
            shape,
            dtype,
            seed=args.seed + index,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        rows.append(row)
        if not row["ok"]:
            print(
                f"correctness failed for {shape['name']} {shape}: "
                f"max|delta|={row['max_abs_diff']:.3e} > {row['tolerance']:.1e} "
                f"(finite={row['finite']}, argmax_equal={row['argmax_equal']})",
                file=sys.stderr,
            )

    _print_table(rows, device_name, args.dtype)

    payload = {
        "spike": args.spike,
        "device": device_name,
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")

    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
