"""评估解耦 CLI：从 checkpoint 加载模型，输出 val ppl + 固定 prompt 生成 + 多样性。

用法:
    python eval.py --exp exp001                          # 从 results/exp001/ 评估
    python eval.py --exp exp001 exp002 exp003            # 多实验对比表（同数据同协议）
    python eval.py --checkpoint path --config config.toml  # 自由路径 + 显式配置

评估协议（结构对比的尺子）:
    1. val ppl      —— 客观主指标（同 val 划分、同 seed，跨配置可比 = 语言的 EPE）
    2. 生成样本     —— 固定 prompt 集（data/eval_prompts.txt），decode 肉眼解释 ppl 差异
    3. 多样性       —— TODO(用户)：ppl 偏爱保守模型的盲区补偿
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import torch

from config import Config, DataConfig, TrainConfig, WeightConfig
from data import build_data
from train import evaluate, load_model
from tools import prune as prune_tools

RESULT_MD = Path("result.md")
# config.json 里 MiniLLMConfig 的"结构字段"子集（跳过 transformers 序列化样板字段）
MODEL_KEYS = ("vocab_size", "max_position_embeddings", "n_embd", "n_head",
              "n_kv_head", "n_layer", "n_inner", "dropout", "bias",
              "norm_eps", "tie_word_embeddings")


def config_from_dict(d: dict) -> Config:
    """从 exp config.json 的 Config 段重建 Config（data/train/weight 三块）。"""
    return Config(
        data=DataConfig(**d.get("data", {})),
        train=TrainConfig(**d.get("train", {})),
        weight=WeightConfig(**d.get("weight", {})),
    )


# ---------- 实验记录：评估后自动追加 result.md ----------

def git_state() -> str:
    """当前分支 + 短 SHA + 未提交标记，记进实验的代码状态行。"""
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
        return f"{branch}@{sha}{'（+未提交改动）' if dirty else ''}"
    except Exception:
        return "unknown"


def structure_str(mc: dict) -> str:
    """MiniLLMConfig → 结构简写，如 4L×128d×4h；GQA/MQA 时附 kv 头数。"""
    s = f"{mc['n_layer']}L×{mc['n_embd']}d×{mc['n_head']}h"
    if mc["n_kv_head"] != mc["n_head"]:
        s += f"/{mc['n_kv_head']}kv"
    return s


def train_stats(exp_dir: Path) -> dict:
    """从 train_log.csv 统计 (最优 val ppl, step) 与 (末步 val ppl, step)。"""
    best = last = None
    if (exp_dir / "train_log.csv").exists():
        with open(exp_dir / "train_log.csv") as f:
            for row in csv.DictReader(f):
                if not row.get("ppl"):
                    continue
                pp, st = float(row["ppl"]), int(row["step"])
                if best is None or pp < best[0]:
                    best = (pp, st)
                last = (pp, st)
    return {"best_ppl": best, "last_ppl": last}


def corpus_label(cfg: Config) -> str:
    """语料标签：取自 tokenizer 所在目录名（如 luxun）+ 字符规模。"""
    name = Path(cfg.data.tokenizer_path).parent.name
    # 按字符数计（st_size 是 UTF-8 字节，中文 1 字 3 字节，语义误导）
    chars = sum(len(p.read_text(encoding="utf-8")) for p in Path(cfg.data.train_dir).rglob("*.txt"))
    return f"{name} {chars/1e4:.1f}万字符"


def record_result(exp_dir: Path, cfg: Config, result: dict, args) -> bool:
    """把实验追加进 result.md：索引表行（表尾）+ 实验节（文件末尾），两者顺序一致。

    同 exp = 同源实验：写入前先把旧记录整段移入 archive/result.md（纯追加），再写新记录。
    无幂等判断；机器字段自动填，判断字段留 TODO(用户)。"""
    if not RESULT_MD.exists():
        print(f"! {RESULT_MD} 不存在，跳过记录"); return False
    mc = json.loads((exp_dir / "config.json").read_text())["MiniLLMConfig"]
    md_id = f"Exp-{exp_dir.name[3:]}"

    # 同 exp 旧记录 → archive（纯追加），再写新记录
    removed = prune_tools.extract_record_text(md_id)
    if removed:
        prune_tools.append_archive(md_id, removed)
    text = RESULT_MD.read_text(encoding="utf-8")

    # 索引表行：追加到索引表末尾（表头分隔行后的连续 | 行块之后；无表头则直接追加）
    tr = train_stats(exp_dir)
    best_cell = f"{tr['best_ppl'][0]:.0f} @{tr['best_ppl'][1]}" if tr["best_ppl"] else "—"
    row = (f"| {md_id} | {date.today().isoformat()} | {corpus_label(cfg)} | {structure_str(mc)} | "
           f"{cfg.train.max_iters} iters | {best_cell} | TODO(用户) | TODO(用户) |")
    lines = text.splitlines()
    sep = next((i for i, ln in enumerate(lines) if re.match(r"^\|[\s\-:|]+\|$", ln)), None)
    if sep is not None:
        j = sep + 1
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        lines.insert(j, row)
        text = "\n".join(lines)
    else:
        text = text.rstrip() + "\n" + row + "\n"

    # 实验节：追加到文件末尾（前补 --- 分隔线；末尾已是 --- 则不重复）
    section = build_record(exp_dir, cfg, mc, result, args)
    text = text.rstrip()
    text = text if text.endswith("---") else text + "\n---"
    RESULT_MD.write_text(text + "\n\n" + section + "\n", encoding="utf-8")
    print(f"✓ {md_id} 已追加 → result.md（TODO 项待人工填）")
    return True


def build_record(exp_dir: Path, cfg: Config, mc: dict, result: dict, args) -> str:
    """构造 result.md 的实验节文本（机器可填的填上，需判断的留 TODO）。"""
    md_id = f"Exp-{exp_dir.name[3:]}"
    tr = train_stats(exp_dir)
    best_ppl = f"{tr['best_ppl'][0]:.0f} @ step {tr['best_ppl'][1]}" if tr["best_ppl"] else "—"
    last_ppl = f"{tr['last_ppl'][0]:.0f} @ step {tr['last_ppl'][1]}" if tr["last_ppl"] else "—"
    div = "/".join(f"{d:.2f}" for d in result["diversity"]) if result["diversity"] else "—"
    model_json = json.dumps({k: mc[k] for k in MODEL_KEYS}, ensure_ascii=False, indent=1)
    train_json = json.dumps(cfg.train.to_dict(), ensure_ascii=False, indent=1)

    return (
        f"## {md_id}\n"
        f"- **日期**：{date.today().isoformat()}　**代码状态**：{git_state()}\n"
        f"- **目的**：TODO(用户)\n"
        f"- **数据**\n"
        f"  - 来源：`{cfg.data.train_dir}/` + `{cfg.data.val_dir}/`　block={cfg.data.block_size}\n"
        f"  - tokenizer：`{cfg.data.tokenizer_path}`　vocab={mc['vocab_size']}\n"
        f"  - 划分：TODO(用户)\n"
        f"- **模型**（`MiniLLMConfig`，完整见 `{exp_dir}/config.json`）\n"
        f"  ```json\n  {model_json}\n  ```\n"
        f"- **训练**（`Config.train`，完整见 `{exp_dir}/config.json`）\n"
        f"  ```json\n  {train_json}\n  ```\n"
        f"- **结果**\n"
        f"  | 项 | 值 |\n  |---|---|\n"
        f"  | 最优 val ppl | {best_ppl} |\n"
        f"  | 末步 val ppl | {last_ppl} |\n"
        f"  | checkpoint ppl / acc | {result['ppl']:.0f} / {result['acc']:.3f} |\n"
        f"  | 多样性 | {div} |\n"
        f"  | 拐点 | TODO(用户) |\n"
        f"  | 轨迹 | `{exp_dir}/train_log.csv` |\n"
        f"  | 生成样本 | `{exp_dir}/samples.txt`（temp={args.temperature}, max_out={args.max_output}） |\n"
        f"  | 配置 | `{exp_dir}/config.json` |\n"
        f"- **观察**：TODO(用户)\n"
        f"- **结论**：TODO(用户)\n"
        f"- **复现**：`python eval.py --exp {exp_dir.name}` | `python train.py --config <对应toml>`\n"
    )


def load_prompts(path: str, tok) -> list[list[int]]:
    """eval_prompts.txt → 每段 encode 成 token 序列（空行分隔段落，# 开头段为注释忽略）。"""
    text = Path(path).read_text(encoding="utf-8")
    return [tok.encode(seg.strip()).ids for seg in text.split("\n\n")
            if seg.strip() and not seg.strip().startswith("#")]


def make_prompt_batch(prompt_ids: list[list[int]], pad_id: int, device, max_output: int, max_pos: int):
    """变长 prompt → 左 pad 等长 batch + attention_mask（A3.1 约定）。
    护栏：prompt_len + max_output <= max_position_embeddings，否则截断 prompt 尾部。"""
    cap = max_pos - max_output                     # 单条 prompt 允许的最大长度
    trimmed = [ids[:cap] for ids in prompt_ids]    # 超预算截断（保留开头，语义完整优先）
    max_len = max(len(ids) for ids in trimmed)
    B = len(trimmed)
    tokens = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, ids in enumerate(trimmed):
        tokens[i, max_len - len(ids):] = torch.tensor(ids)   # 左 pad
        mask[i, max_len - len(ids):] = 1
    return tokens.to(device), mask.to(device)


def output_diversity(sample_ids: list[list[int]]) -> float:
    """度量同一 prompt 采样多次的输出多样性（0~1，越高越多样）。

    为什么需要它 —— val ppl 偏爱"保守"模型：只敢预测高频词、生成千篇一律
    的模型 ppl 反而低，但文本无聊。多样性补偿这个盲区。

    实现(方案 B：输出 token 分布熵): 合并所有输出的 token 频率 → 熵，归一化：
        diversity = H / log2(去重 token 数)   （H = -Σ p log2 p）
    度量"用词整体的集中度"：越高 = 覆盖越多不同 token = 越不保守。
    选它而非"序列内容重合"(Jaccard/去重) 的原因 —— 后者度量"两条具体延续
    像不像"，但长输出一旦分叉就大量不同，把有共同前缀的相似输出误判为迥异；
    保守性本质是"用词集中"，分布熵直接抓这一点。

    已知局限: 模型对 prompt 有多条高概率路径时（数据不足的 toy），temperature
    效应被自身不确定性稀释，分层收窄（实测 exp001: t0.1=0.94 / t1.0=0.98，
    方向对但仅弱分层）。正常训练模型上会更明显。作辅助观察指标，不主判。

    验收: 对同一 prompt、temperature=1.0 采样 5 次，值应 ≥ temperature=0.1 采样 5 次
    """
    import math
    from collections import Counter
    n_out = len(sample_ids)
    if n_out < 2:
        return 1.0
    toks = [t for ids in sample_ids for t in ids]   # 合并所有输出 token
    n = len(toks)
    if n == 0:
        return 1.0
    cnt = Counter(toks)
    if len(cnt) < 2:
        return 1.0
    H = -sum((c / n) * math.log2(c / n) for c in cnt.values())
    return H / math.log2(len(cnt))


def evaluate_model(model, data, device, args, exp_name: str):
    """评估已加载模型：val ppl + 固定 prompt 生成 + 多样性。"""
    # 层 1: val ppl（复用 train.evaluate，10 batch 平均）
    acc, val_loss = evaluate(model, data)
    ppl = float(torch.exp(torch.tensor(val_loss)))

    # 层 2 + 3: 固定 prompt 生成
    tok = data.tok
    prompt_ids = load_prompts(args.prompts, tok)
    max_pos = model.config.max_position_embeddings
    prompts, prompt_mask = make_prompt_batch(
        prompt_ids, tok.token_to_id("<pad>"), device, args.max_output, max_pos)

    from llm_utils.inference import generate
    gen = generate(model, prompts, args.max_output, temperature=args.temperature,
                   attention_mask=prompt_mask)

    # 层 3: 多样性 —— 同一 prompt 复制 num_samples 份，采样多次
    samples_per = []
    if args.num_samples > 1:
        for pid in prompt_ids[:args.prompt_rows]:
            ids = [pid] * args.num_samples
            pb, pm = make_prompt_batch(ids, tok.token_to_id("<pad>"), device,
                                       args.max_output, max_pos)
            g = generate(model, pb, args.max_output, temperature=args.temperature,
                         attention_mask=pm)
            clean = [[t for t in g[i].tolist() if t != tok.token_to_id("<pad>")]
                     for i in range(args.num_samples)]
            samples_per.append(output_diversity(clean))

    clean = [[t for t in gen[i].tolist() if t != tok.token_to_id("<pad>")]
             for i in range(len(gen))]
    texts = [tok.decode(ids) for ids in clean]
    prompt_texts = [tok.decode(ids) for ids in prompt_ids]

    return {"exp": exp_name, "val_loss": val_loss, "ppl": ppl, "acc": acc,
            "samples": list(zip(prompt_texts, texts)), "diversity": samples_per}


def evaluate_one(exp_dir: Path, device, args):
    """从实验目录加载（config.json 重建配置 + model.pth 重建模型）。"""
    cfg = config_from_dict(json.loads((exp_dir / "config.json").read_text())["Config"])
    model = load_model(exp_dir / "model.pth").to(device)
    data = build_data(cfg, device)
    return evaluate_model(model, data, device, args, exp_dir.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", help="实验 id（如 exp001），可多个做对比")
    ap.add_argument("--checkpoint", help="自由 checkpoint 路径（配合 --config）")
    ap.add_argument("--config", default="config/experiment.toml", help="训练配置 toml")
    ap.add_argument("--prompts", default="data/eval_prompts.txt")
    ap.add_argument("--max-output", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--num-samples", type=int, default=1, help="多样性采样次数（>1 才启用）")
    ap.add_argument("--prompt-rows", type=int, default=2, help="多样性用前几条 prompt")
    ap.add_argument("--no-record", action="store_true",
                    help="不写入 result.md（默认评估后自动追加：索引表行 + 实验节，TODO 留人工）")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = []
    if args.exp:
        for e in args.exp:
            exp_dir = Path("results") / e
            if not (exp_dir / "model.pth").exists():
                print(f"!! {exp_dir} 无 model.pth，跳过"); continue
            r = evaluate_one(exp_dir, device, args)
            results.append(r)
            if not args.no_record:
                cfg = config_from_dict(json.loads((exp_dir / "config.json").read_text())["Config"])
                record_result(exp_dir, cfg, r, args)
    elif args.checkpoint:
        # 自由 checkpoint：模型结构在 checkpoint 内，数据/训练配置由 --config toml 提供
        cfg = Config.from_toml(args.config)
        model = load_model(args.checkpoint).to(device)
        data = build_data(cfg, device)
        results.append(evaluate_model(model, data, device, args, Path(args.checkpoint).name))
    else:
        ap.print_help(); sys.exit(1)

    # 输出：对比表 + 生成样本
    print(f"\n{'exp':<10}{'val_loss':>10}{'ppl':>10}{'acc':>8}  {'diversity':>10}")
    for r in results:
        div = "/".join(f"{d:.2f}" for d in r["diversity"]) or "-"
        print(f"{r['exp']:<10}{r['val_loss']:>10.4f}{r['ppl']:>10.1f}{r['acc']:>8.3f}  {div:>10}")
    for r in results:
        print(f"\n=== {r['exp']} 生成样本 (temp={args.temperature}) ===")
        for i, (pt, gt) in enumerate(r["samples"]):
            print(f"--- prompt[{i}] ---\n{pt}\n--- 生成 ---\n{gt}")


if __name__ == "__main__":
    main()
