"""读实验 train_log.csv 画训练曲线（train/val loss + ppl）。默认 plt.show 现画现看，--save 时落盘。

用法（支持多实验叠图对比，同一份数据不同画法无需重训）:
    python tools/plot_curve.py results/exp001/train_log.csv
    python tools/plot_curve.py results/exp001/train_log.csv results/exp002/train_log.csv
    python tools/plot_curve.py .../train_log.csv --save curve.png   # 存图不弹窗

约定（与 train.py 落盘一致）:
    train_loss 每 log_interval 行有值；val_loss/ppl/acc 每 eval_interval 行有值（其余为空）。
    y 轴 log：train/val 分离的拐头在 loss 上更易读。
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_traj(path):
    """解析 csv → {step, train_loss, val_loss, ppl}；空值过滤为独立点列。"""
    steps, train_pts, val_pts, ppl_pts = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            s = int(r["step"])
            steps.append(s)
            if r["train_loss"]:
                train_pts.append((s, float(r["train_loss"])))
            if r["val_loss"]:
                val_pts.append((s, float(r["val_loss"])))
            if r["ppl"]:
                ppl_pts.append((s, float(r["ppl"])))
    return steps, train_pts, val_pts, ppl_pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="train_log.csv 路径（多个叠图对比）")
    ap.add_argument("--save", help="保存到指定路径（如 curve.png），否则 plt.show 现画现看")
    args = ap.parse_args()
    paths = [Path(p) for p in args.csvs]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for p in paths:
        steps, train_pts, val_pts, ppl_pts = load_traj(p)
        label = p.parent.name  # exp001 / exp002 ...

        if train_pts:
            ts, tl = zip(*train_pts)
            ax1.plot(ts, tl, label=f"{label} train")
        if val_pts:
            vs, vl = zip(*val_pts)
            ax1.plot(vs, vl, marker="o", ls="--", label=f"{label} val")
        if ppl_pts:
            ps, pp = zip(*ppl_pts)
            ax2.plot(ps, pp, marker="o", ls="--", label=f"{label} ppl")

    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax2.set_ylabel("ppl")
    ax2.set_yscale("log")
    ax2.set_xlabel("step")
    if len(paths) > 1:
        ax2.legend()

    plt.tight_layout()
    if args.save:                       # 指定 --save 才落盘（默认现画现看）
        plt.savefig(args.save, dpi=150)
        print(f"✓ 已保存 → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
