"""A3.1 验收：左 pad 变长 batch vs 逐条无 pad 单序列参考（greedy）

验证两点：
  1. generate 接受 attention_mask，prefill 与 decode 都带全长度 mask
  2. 左 pad batch 的生成结果 == 每条单独无 pad 生成（逐 token）
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

# 1. 加载权重（与 check_kvcache.py 同一套）
weights = torch.load("weights/minillm.pth", map_location=device)
model = MiniLLM(MiniLLMConfig(**weights["config"])).to(device)
model.load_state_dict(weights["model"])
model.eval()

# 2. 造左 pad 变长 batch：值域对齐训练 VOCAB_MAX=128，PAD_ID=255
N = 16
seqs = [torch.randint(0, 128, (l,)) for l in [12, 20, 32]]   # 3 条变长
T_max = max(s.numel() for s in seqs)
B = len(seqs)
tokens = torch.full((B, T_max), 255, dtype=torch.long)
attention_mask = torch.zeros(B, T_max, dtype=torch.long)
for i, s in enumerate(seqs):
    tokens[i, -s.numel():] = s
    attention_mask[i, -s.numel():] = 1
print(f"batch 形状: tokens {tuple(tokens.shape)}, mask {tuple(attention_mask.shape)}")

# 3. 被测：batch 一次生成（左 pad，带 mask，greedy，eos 关闭）
got = generate(model, tokens.to(device), N, temperature=0.0,
               attention_mask=attention_mask.to(device))

# 4. 参考：逐条无 pad 生成（不传 mask）
ref = [generate(model, s.unsqueeze(0).to(device), N, temperature=0.0)[0] for s in seqs]

# 5. 断言 + 打印
all_ok = True
for i in range(B):
    ok = (got[i] == ref[i]).all().item()
    all_ok = all_ok and ok
    print(f"seq {i} len={seqs[i].numel():2d} | batch==ref: {ok} | "
          f"got len={len(got[i])} ref len={len(ref[i])}")

print(f"\n{'✓' if all_ok else '✗'} 左 pad batch 与逐条无 pad 参考逐 token 一致: {all_ok}")

# 肉眼复核 x+1 递增规律
for i in range(B):
    print(f"\nseq {i}: prompt {seqs[i][-8:].tolist()}")
    print(f"        out   {got[i].tolist()}")
