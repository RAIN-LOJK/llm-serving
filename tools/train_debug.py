import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent   # tools/ -> 仓库根
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from model import MiniLLM, MiniLLMConfig

device = "mps" if torch.backends.mps.is_available() else "cpu"

config = MiniLLMConfig()
model = MiniLLM(config).to(device)

context = torch.arange(0, 63).unsqueeze(0)
target = torch.arange(1, 64).unsqueeze(0)
context, target = context.to(device), target.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(1000):
    logits, loss = model(context, targets=target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

logits, _ = model(context)
preds = logits.argmax(dim=-1)
print(f"\ninput:   {context[0].tolist()}")
print(f"predict: {preds[0].tolist()}")
print(f"target:  {target[0].tolist()}")
print(f"correct: {(preds == target).sum().item()} / {target.numel()}")
