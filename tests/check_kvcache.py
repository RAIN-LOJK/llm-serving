"""KV cache 一致性校验：cache 生成 vs 全量重算（greedy），逐 token 对比。"""
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

prompt = torch.randint(0, 64, (1, 16)).to(device)
N = 24


def ref_greedy(prompt, n):
    """无 cache：每步把整段上下文喂进模型，取 argmax。"""
    out = prompt.clone()
    for _ in range(n):
        logits, _ = model(out)          # 全量重算
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out[0][prompt.shape[1]:]


ref = ref_greedy(prompt, N)
got = generate(model, prompt, N, temperature=0.0)[0]

ok = (ref == got).all().item()
print(f"prompt : {prompt[0].tolist()}")
print(f"ref    : {ref.tolist()}")
print(f"cache  : {got.tolist()}")
print(f"一致: {ok}  ({int((ref == got).sum())}/{N} tokens)")
