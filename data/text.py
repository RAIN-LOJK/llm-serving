"""B2d 真实语料 dataloader（方案 A：BOS/EOS 包裹 + 不跨文档滑窗）。

预处理（只做一次）：每篇文档 encode 成 [<bos>, ...ids, <eos>]，
太短放不下一个窗口的文档丢弃（方案 A 的代价，长文足够）。
get_batch：batch 内每个样本独立随机选一篇文档 + 随机偏移，
取连续 block_size 个 token 为输入、右移一位为 target。
无 pad → attention_mask 全 1。
"""
import torch
from pathlib import Path
from tokenizers import Tokenizer


class TextData:
    def __init__(self, tokenizer_path, train_dir, val_dir, block_size,
                 batch_size, device="cpu", seed=0):
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.bos = self.tok.token_to_id("<bos>")
        self.eos = self.tok.token_to_id("<eos>")
        self.docs = {
            "train": self._encode_dir(train_dir),
            "val":   self._encode_dir(val_dir),
        }
        self.gens = {                       # train 随机、val 固定（可复现）
            "train": torch.Generator().manual_seed(seed),
            "val":   torch.Generator().manual_seed(seed + 1),
        }

    def _encode_dir(self, dir) -> list[torch.Tensor]:
        """每篇文档 → [<bos>, ...ids, <eos>] 的 tensor；太短的丢弃。"""
        docs = []
        for p in sorted(Path(dir).rglob("*.txt")):
            ids = self.tok.encode(p.read_text(encoding="utf-8")).ids
            # 严格大于：滑窗随机偏移需 len(d) - block_size > 0（等于则 randint(0,0) 崩）
            if len(ids) + 2 > self.block_size:
                docs.append(torch.tensor([self.bos] + ids + [self.eos], dtype=torch.long))
        return docs

    def get_batch(self, split="train"):
        docs = self.docs[split]
        g = self.gens[split]
        B, T = self.batch_size, self.block_size
        tokens = torch.zeros(B, T, dtype=torch.long)
        targets = torch.zeros(B, T, dtype=torch.long)
        for i in range(B):
            d = docs[torch.randint(0, len(docs), (1,), generator=g).item()]
            off = torch.randint(0, len(d) - T, (1,), generator=g).item()
            tokens[i] = d[off:off + T]
            targets[i] = d[off + 1:off + T + 1]
        mask = torch.ones(B, T, dtype=torch.long)   # 无 pad → 全 1
        return tokens.to(self.device), targets.to(self.device), mask.to(self.device)
