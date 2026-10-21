"""A3.3 KV cache 加速比基准：cache vs 全量重算，随 seq 长度画曲线。"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent   # tests/ -> 仓库根
sys.path.insert(0, str(REPO_ROOT))

import time
import torch
import matplotlib.pyplot as plt

from model import MiniLLM, MiniLLMConfig
from llm_utils.inference import generate

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

weights = torch.load("weights/minillm.pth", map_location=device)
model = MiniLLM(MiniLLMConfig(**weights["config"])).to(device)
model.load_state_dict(weights["model"])
model.eval()

N = 16  # 生成 token 数


def ref_greedy(prompt, n):
    """无 cache：每步把整段上下文喂进模型，取 argmax。"""
    out = prompt.clone()
    for _ in range(n):
        logits, _ = model(out)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out


def timeit(fn, reps=5):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()  # MPS 异步 kernel，计时前必须同步
        ts.append(time.perf_counter() - t0)
    return min(ts)  # min 抗噪声


# 主循环：变 prompt 长 T，测全量 vs cache 耗时
Ts = [8, 16, 24, 32, 40]  # 约束: T + N <= max_position_embeddings=256
speedups = []
for T in Ts:
    prompt = torch.randint(0, 64, (1, T)).to(device)
    t_full = timeit(lambda: ref_greedy(prompt, N))
    t_cache = timeit(lambda: generate(model, prompt, N, temperature=0.0))
    speedup = t_full / t_cache
    speedups.append(speedup)
    print(f"T={T:3d} | 全量 {t_full*1e3:7.2f}ms | cache {t_cache*1e3:7.2f}ms | 加速比 {speedup:.2f}x")

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(Ts, [timeit(lambda: ref_greedy(torch.randint(0, 64, (1, T)).to(device), N)) * 1e3
              for T in Ts], label="full recompute", marker="o")
ax1.plot(Ts, [timeit(lambda: generate(model, torch.randint(0, 64, (1, T)).to(device), N, temperature=0.0)) * 1e3
              for T in Ts], label="with kv cache", marker="o")
ax1.set_ylabel("time (ms)")
ax1.set_yscale("log")
ax1.legend()

ax2.plot(Ts, speedups, marker="o", color="orange")
ax2.set_ylabel("speedup")
ax2.set_xlabel("prompt length T")
ax2.axhline(1.0, color="gray", linestyle="--")

plt.tight_layout()
plt.savefig("bench_kvcache.png", dpi=150)
print("saved bench_kvcache.png")
