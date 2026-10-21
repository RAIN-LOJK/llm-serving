import csv
import json
import math
import os
import shutil
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer

from data import build_data
from llm_utils.inference import generate
from config import Config, WeightConfig
from model import MiniLLM, MiniLLMConfig

def save_weights(model, cfg: WeightConfig, resume=False, optimizer=None, scheduler=None, step=None):
    """保存模型参数 + config, 供后续加载 (KV cache 阶段要用)"""
    # TODO: torch.save({"config": ..., "state_dict": ...}, path)
    if resume:
        torch.save({
            "config":     model.config.to_dict(),
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict() if optimizer else None,
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "step":       step + 1 if step is not None else None,
        }, cfg.resume_path)
    else:
        torch.save({"config": model.config.to_dict(), "model": model.state_dict()}, cfg.path)

def resume_weights(path):
    weights = torch.load(path)
    return weights

def load_model(path):
    """从 checkpoint 重建模型"""
    # TODO: 读取文件 -> 从 config 构建 MiniLLM -> load_state_dict -> 返回 model
    weights = torch.load(path)
    model = MiniLLM(MiniLLMConfig(**weights["config"]))
    model.load_state_dict(weights["model"])
    return model

def next_exp_id(results_dir="results") -> int:
    """从 results/ 现有 expXXX 目录推下一个编号（Exp-001 起始）。"""
    if not os.path.isdir(results_dir):
        return 1
    ids = [int(p.name[3:]) for p in Path(results_dir).iterdir()
           if p.is_dir() and p.name.startswith("exp")]
    return max(ids) + 1 if ids else 1

@torch.no_grad()
def evaluate(model: MiniLLM, data, times=10):
    model.eval()
    total, corrects, loss_sum = 0, 0, 0.
    for _ in range(times):
        tokens, targets, mask = data.get_batch("val")
        logits, loss = model(tokens, targets, mask)
        loss_sum += loss.item()
        preds = logits.argmax(-1)
        m = mask.bool()
        corrects += (preds[m] == targets[m]).sum().item()
        total += m.sum().item()
    model.train()
    return corrects/total, loss_sum/times

@torch.no_grad()
def evaluate_generation(model, data, n_rows=3, max_output=64):
    """真实语料：从 val prompt 续写，decode 成文本肉眼检查可读性。
    返回拼接好的文本块，供落盘 samples.txt；同时打印到终端。"""
    model.eval()
    tokens, _, mask = data.get_batch("val")
    # prompt 只取前 128 token，留出生成空间：护栏要求
    #   prompt_len + max_output <= max_position_embeddings(256)，否则第一步就 break
    prompts, prompt_mask = tokens[:n_rows, :128], mask[:n_rows, :128]
    gen = generate(model, prompts, max_output, temperature=0.8, attention_mask=prompt_mask)
    model.train()
    blocks = []
    for i in range(n_rows):
        pt = data.tok.decode(prompts[i, :40].tolist())
        gt = data.tok.decode(gen[i].tolist())
        print(f"--- 行 {i} prompt(前40) ---")
        print(pt)
        print(f"--- 行 {i} 生成 ---")
        print(gt)
        blocks.append(f"### 行 {i} prompt(前40)\n{pt}\n\n### 行 {i} 生成\n{gt}")
    return "\n\n".join(blocks)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="训练 MiniLLM（实验落盘到 results/expXXX/）")
    ap.add_argument("--config", default="config/experiment.toml", help="实验配置 toml")
    ap.add_argument("--exp", help="实验编号（如 exp010）；已存在则先 prune 目录+台账再重建（覆盖重训）")
    args = ap.parse_args()
    cfg = Config.from_toml(args.config)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 实验名 = 配置文件名（不含 .toml）：一个实验一个 config，文件名即身份证
    exp_name = Path(args.config).stem

    # 实验归档：本次训练写入 results/expXXX/（自包含文件夹）
    #   weights/ 是开发工作区，不再被训练写入
    if cfg.weight.resume:
        # 续训 = 从原实验目录接着写（不新建 exp）
        exp_dir = Path(cfg.weight.resume_path).parent
        exp_id = int(exp_dir.name[3:])
    elif args.exp:
        # 显式指定编号：已存在 → 删目录重建（覆盖重训；result.md 旧记录不动，留待 eval 归档）
        exp_dir = Path("results") / args.exp
        if exp_dir.exists():
            print(f"! {args.exp} 已存在 → 删除目录重建（覆盖重训；result.md 旧记录留待 eval 归档）")
            shutil.rmtree(exp_dir)
        exp_id = int(exp_dir.name[3:])
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_id = next_exp_id()
        exp_dir = Path("results") / f"exp{exp_id:03d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    wcfg = WeightConfig(directory=str(exp_dir), name="model.pth", resume_name="resume.pt")
    print(f"实验目录: {exp_dir}")

    torch.manual_seed(cfg.train.seed)
    tok = Tokenizer.from_file(str(cfg.data.tokenizer_path))   # vocab 契约基准
    if cfg.weight.resume:
        weights = resume_weights(cfg.weight.resume_path)
        model = MiniLLM(MiniLLMConfig(**weights["config"])).to(device)
        model.load_state_dict(weights["model"])
        # 校验 checkpoint 与当前 tokenizer/数据配置一致，防静默不匹配
        wcfg = weights["config"]
        assert wcfg["vocab_size"] == tok.get_vocab_size(), \
            f"checkpoint vocab {wcfg['vocab_size']} != tokenizer {tok.get_vocab_size()}"
        assert wcfg["max_position_embeddings"] == cfg.data.block_size, \
            f"checkpoint 窗口 {wcfg['max_position_embeddings']} != block_size {cfg.data.block_size}"
    else:
        # 结构参数来自实验 toml 的 [model] 段，桥接进 MiniLLMConfig
        model = MiniLLM(cfg.model.to_minillm_config(
            tok.get_vocab_size(), cfg.data.block_size)).to(device)

    # 实验档案落盘：config.json 训练开始即写（配置快照：数据+训练+权重+模型结构 + 实验名）
    #   不依赖训练正常结束——中断/打断也能 eval（历史坑：Ctrl-C 后 config.json 缺失）
    with open(exp_dir / "config.json", "w") as f:
        json.dump({
            "meta": {
                "experiment": f"exp{exp_id:03d}",
                "name": exp_name,          # 实验名 = 配置文件名（如 fw）
                "config": args.config,     # 训练用 toml 路径（复现行可回放）
            },
            "Config": cfg.to_dict(),
            "MiniLLMConfig": model.config.to_dict(),
        }, f, ensure_ascii=False, indent=2)
    print(f"配置快照 → {exp_dir/'config.json'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # for lr scheduler
    optimizer_steps = cfg.train.max_iters // cfg.train.accum_steps
    warmup_steps_opt = cfg.train.warmup_steps // cfg.train.accum_steps
    def lr_lambda(step):
        if step < warmup_steps_opt:
            return step / warmup_steps_opt
        t = (step - warmup_steps_opt) / (optimizer_steps - warmup_steps_opt)
        t = max(0.0, min(1.0, t))
        # 设置min_lr是lr的几倍
        min_ratio = cfg.train.min_lr / cfg.train.lr
        return (1 + math.cos(math.pi * t)) / 2 * (1 - min_ratio) + min_ratio
    scheduler = LambdaLR(optimizer, lr_lambda)
    data = build_data(cfg, device)

    # 迭代式训练循环（LLM 预训练标准形态，非 epoch 式）：
    #   总预算 = max_iters 步；train/val 分流；每 eval_interval 步在 val 流评估
    optimizer.zero_grad()

    # 实时轨迹：train_log.csv，append 模式 + 游标 last_written（resume 防重复行）
    csv_path = exp_dir / "train_log.csv"
    csv_handle = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_handle)
    last_written = -1
    if csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "train_loss", "val_loss", "ppl", "acc"])
    else:
        with open(csv_path) as f:   # resume：从末尾读已写最大 step
            rows = list(csv.reader(f))
        if len(rows) > 1:
            last_written = int(rows[-1][0])

    start_step = 0
    if cfg.weight.resume:
        optimizer.load_state_dict(weights["optimizer"])
        scheduler.load_state_dict(weights["scheduler"])
        start_step = weights["step"]
        print(f"resuming from step {start_step}")
    for step in range(start_step, cfg.train.max_iters):
        tokens, targets, attention_mask = data.get_batch("train")
        _, loss = model(tokens, targets, attention_mask)
        (loss / cfg.train.accum_steps).backward()
        if (step + 1) % cfg.train.accum_steps == 0:
            if cfg.train.max_norm > 0:   # 0 = 关闭裁剪（迁移到其他任务时可按需关）
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if step % cfg.train.eval_interval == 0:
            eval_res, val_loss = evaluate(model, data)
            print(f"step {step:4d} | train {loss.item():.4f} | val {val_loss:.4f} | ppl {math.exp(val_loss):.1f} | acc {eval_res:.3f}")
        # 实时写 csv：每 log_interval 行，eval 步带 val 列；step<=last_written 跳过（resume 防重）
        #   约定 eval_interval % log_interval == 0（val 列只在 log 步落盘）
        if step % cfg.train.log_interval == 0 and step > last_written:
            row = [step, f"{loss.item():.4f}", "", "", ""]
            if step % cfg.train.eval_interval == 0:
                row[2:] = f"{val_loss:.4f}", f"{math.exp(val_loss):.1f}", f"{eval_res:.3f}"
            csv_writer.writerow(row)
            csv_handle.flush()   # 实时落盘：训练中断不丢轨迹
            last_written = step
        if step > 0 and step % cfg.weight.resume_interval == 0:
            save_weights(model, wcfg, True, optimizer, scheduler, step)

    csv_handle.close()
    save_weights(model, wcfg)

    # 最终验证: load_checkpoint 加载后, 从 val prompt 续写, decode 看可读性
    model = load_model(wcfg.path)
    model.to(device)
    samples = evaluate_generation(model, data)
    (exp_dir / "samples.txt").write_text(samples)
    print(f"生成样本 → {exp_dir/'samples.txt'}")

    # 训练曲线不再落盘：需要时 `python tools/plot_curve.py <train_log.csv>` 现画现看
    print(f"轨迹 → {csv_path}（绘图: python tools/plot_curve.py {csv_path}）")



if __name__ == "__main__":
    main()
