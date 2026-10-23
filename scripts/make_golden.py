#!/usr/bin/env python
"""Generate the W1 golden fixture from the HF reference (ROADMAP §5 W1).

The reference runs in **float32 on CPU with eager attention**: the checkpoint ships bf16,
so the upcast is exact (bf16 -> fp32 loses nothing), and the fixture stays reproducible on
any machine without renting a GPU. Golden is committed; tests only read it — never
regenerate silently (tests/golden/README.md).

    python scripts/make_golden.py --model Qwen/Qwen3-0.6B --golden-id 0001

Output: ``tests/golden/<slug>/<golden-id>/{meta.json,tensors.npz}``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROMPT_IDS = ("s01", "s02", "s03", "s04")
MAX_NEW_TOKENS = 64


def slug_of(repo: str) -> str:
    return repo.split("/")[-1].lower().replace(".", "-")


def resolve_snapshot(repo: str, revision: str | None) -> tuple[Path, str]:
    """Return (snapshot_dir, commit) from the local HF cache; never touches the network."""
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    base = cache / ("models--" + repo.replace("/", "--"))
    if revision is None:
        ref = base / "refs" / "main"
        revision = ref.read_text().strip() if ref.exists() else None
    snapshots = base / "snapshots"
    dirs = sorted(p for p in snapshots.glob("*") if p.is_dir()) if snapshots.exists() else []
    if not dirs:
        raise SystemExit(
            f"{repo} is not in the HF cache ({snapshots}).\n"
            f"Fetch it first, e.g.  HF_ENDPOINT=https://hf-mirror.com "
            f"python -c \"from huggingface_hub import snapshot_download as d; "
            f"print(d('{repo}', revision='<commit>'))\""
        )
    if revision is None:
        path = max(dirs, key=lambda p: p.stat().st_mtime)
    else:
        path = snapshots / revision
        if not path.exists():
            raise SystemExit(f"revision {revision} not cached; available: {[p.name for p in dirs]}")
    return path, path.name


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return ""

    dirty = bool(run("status", "--porcelain"))
    return {"commit": run("rev-parse", "HEAD"), "dirty": dirty}


def load_prompts(ids: tuple[str, ...]) -> list[dict]:
    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    by_id = {p["id"]: p for p in spec["prompts"]}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown prompt ids {missing}; bench/prompts.json v{spec['version']}")
    return [by_id[i] for i in ids]


def block_outputs(model, ids):
    """Every decoder block's output, captured with hooks.

    Do **not** use ``output_hidden_states`` for this: in transformers 5.x its last entry is
    the *post-final-norm* tensor, not the last block output. On Qwen3-0.6B that row has
    ``max|x| ≈ 54`` while the real block output is ``≈ 1819`` — storing it makes the last
    layer look catastrophically wrong when the model is fine.

    Hooks also make the fixture and ``scripts/align_hf.layer_hidden`` the same measurement
    by construction, instead of two conventions that have to be kept in sync.
    """
    captured: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out)
        )
        for layer in model.model.layers
    ]
    try:
        with torch.no_grad():
            logits = model(ids, logits_to_keep=1).logits
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == model.config.num_hidden_layers, len(captured)
    return torch.stack(captured)[:, 0], logits  # (n_layer, seq, hidden), (1, 1, vocab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None, help="commit hash; defaults to the cached refs/main")
    ap.add_argument("--model-path", default=None, help="local snapshot dir (skips cache lookup)")
    ap.add_argument("--golden-id", default="0001")
    ap.add_argument("--ids", default=",".join(DEFAULT_PROMPT_IDS))
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--out", default=None, help="override the golden root (default: tests/golden)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.model_path:
        model_path, revision = Path(args.model_path), Path(args.model_path).name
    else:
        model_path, revision = resolve_snapshot(args.model, args.revision)

    prompt_ids = tuple(p.strip() for p in args.ids.split(",") if p.strip())
    prompts = load_prompts(prompt_ids)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    )
    model.eval()
    torch.set_grad_enabled(False)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    rows = []
    for prompt in prompts:
        ids = tokenizer(prompt["text"], return_tensors="pt").input_ids
        length = ids.shape[1]
        positions = sorted({0, length // 2, length - 1})
        with torch.no_grad():
            blocks, last_logits = block_outputs(model, ids)
            hidden = blocks.index_select(1, torch.tensor(positions))  # (n_layer, n_pos, hidden)
            logits_last = last_logits[0, -1]
            generated = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                logits_to_keep=1,
                pad_token_id=pad_id,
            )
        new_ids = generated[0, length:]
        greedy = torch.full((args.max_new_tokens,), -1, dtype=torch.int32)
        greedy[: new_ids.numel()] = new_ids.to(torch.int32)

        rows.append(
            {
                "id": prompt["id"],
                "input_ids": ids[0],
                "hidden": hidden.to(torch.float32),  # (n_layer, n_positions, hidden)
                "positions": positions,
                "logits": logits_last.to(torch.float32),
                "greedy": greedy,
                "sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
            }
        )

    width = max(int(r["input_ids"].numel()) for r in rows)
    n_pos = len(rows[0]["positions"])
    block_hidden = torch.stack([r["hidden"] for r in rows])
    n_layer = block_hidden.shape[1]
    assert block_hidden.shape == (len(rows), n_layer, n_pos, block_hidden.shape[-1])
    assert n_layer == len(model.model.layers), (n_layer, len(model.model.layers))

    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.int32)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.int8)
    lengths = torch.tensor([int(r["input_ids"].numel()) for r in rows], dtype=torch.int32)
    for i, row in enumerate(rows):
        length = int(lengths[i])
        input_ids[i, width - length :] = row["input_ids"]
        attention_mask[i, width - length :] = 1

    arrays = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "lengths": lengths.numpy(),
        "hidden": block_hidden.numpy(),  # (prompts, layers, positions, hidden)
        "positions": torch.tensor([r["positions"] for r in rows], dtype=torch.int32).numpy(),
        "logits_last": torch.stack([r["logits"] for r in rows]).numpy(),
        "greedy_ids": torch.stack([r["greedy"] for r in rows]).numpy(),
    }

    out_dir = Path(args.out or ROOT / "tests" / "golden") / slug_of(args.model) / args.golden_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already has a fixture; golden is immutable — bump --golden-id")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "tensors.npz", **arrays)

    meta = {
        "spec": "w1-golden-v1",
        "model": {
            "repo": args.model,
            "revision": revision,
            "dtype": "float32",
            "device": "cpu",
            "attn_implementation": "eager",
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "tokenizer": {"repo": args.model, "revision": revision, "class": type(tokenizer).__name__},
        "prompts": [
            {"id": r["id"], "sha256": r["sha256"], "n_tokens": int(lengths[i])}
            for i, r in enumerate(rows)
        ],
        "generation": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens},
        "tensors": {
            "input_ids": "int32 (n_prompts, width) left-padded; pad id in the fixture, not used by tests",
            "hidden": f"float32 (n_prompts, {n_layer} block outputs, {n_pos} positions, hidden_size)",
            "positions": "int32 (n_prompts, positions) absolute prefill positions sampled from hidden",
            "logits_last": "float32 (n_prompts, vocab_size) last prefill position",
            "greedy_ids": "int32 (n_prompts, max_new_tokens), -1 past EOS",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "source": {**git_state(), "command": " ".join(sys.argv)},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    size_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"✓ {out_dir.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    for row, n in zip(rows, lengths.tolist()):
        n_new = int((row["greedy"] >= 0).sum())
        print(f"   {row['id']}: {n:>4} tokens, greedy {n_new} ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
