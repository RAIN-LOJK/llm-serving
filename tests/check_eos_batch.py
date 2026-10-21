"""A3.2 验收：EOS 发散——batch 各行按各自 EOS 提前结束，输出长度各自正确。

模型学的是 y = x+1（从 prompt 末值 v 起连续递增）。设 eos_token=100：
  - 每行从 v 递增，到 100 时写出 EOS token 并提前结束
  - 不同 v → 到达 100 的步数不同（12/15/23）→ 同一 batch 内各行长度发散
对比：batch 左 pad 生成 vs 逐条无 pad 单序列参考（含 EOS token）。
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent   # tests/ -> 仓库根
sys.path.insert(0, str(REPO_ROOT))

import torch

from model import MiniLLM, MiniLLMConfig
from llm_utils.inference import generate

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

weights = torch.load("weights/minillm.pth", map_location=device)
model = MiniLLM(MiniLLMConfig(**weights["config"])).to(device)
model.load_state_dict(weights["model"])
model.eval()

EOS = 100          # 模型自然递增到 100 时触发 EOS
N = 32             # 足够大，保证各行都在 max_output 内到达 100
LASTS = [85, 77, 88]        # prompt 末值 → 期望输出长度 = 100 - v = 15/23/12

# 造 3 条不同长度、末值不同的序列（末值决定何时撞 EOS）
seqs = []
for l, v in zip([12, 20, 32], LASTS):
    s = torch.randint(0, 128, (l,))
    s[-1] = v
    seqs.append(s)

# 左 pad 成 batch
T_max = max(s.numel() for s in seqs)
B = len(seqs)
tokens = torch.full((B, T_max), 255, dtype=torch.long)
attention_mask = torch.zeros(B, T_max, dtype=torch.long)
for i, s in enumerate(seqs):
    tokens[i, -s.numel():] = s
    attention_mask[i, -s.numel():] = 1

# 被测：batch 一次生成（左 pad + mask + EOS）
got = generate(model, tokens.to(device), N, temperature=0.0, eos_token=EOS,
               attention_mask=attention_mask.to(device))

# 参考：逐条无 pad 生成（同 EOS）
ref = [generate(model, s.unsqueeze(0).to(device), N, temperature=0.0, eos_token=EOS)[0]
       for s in seqs]

# 断言
all_ok = True
for i in range(B):
    exp_len = EOS - LASTS[i]
    len_ok = (len(got[i]) == exp_len == len(ref[i]))
    tok_ok = (got[i] == ref[i]).all().item()
    all_ok = all_ok and len_ok and tok_ok
    print(f"seq {i}: last={LASTS[i]} | 期望长 {exp_len:2d} | got {len(got[i]):2d} ref {len(ref[i]):2d} | "
          f"tok一致 {tok_ok} | 末尾={got[i][-3:].tolist()}")

print(f"\n{'✓' if all_ok else '✗'} EOS 发散验收: batch 各行按各自 EOS 提前结束，与参考一致")
