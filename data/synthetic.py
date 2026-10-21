import torch
from dataclasses import dataclass

PAD_ID = 255        # tokens 的 pad token (合法 id)
IGNORE = -100      # targets 的 pad (cross_entropy ignore_index)


@dataclass
class EchoData:
    vocab_max: int
    min_len: int
    max_len: int
    device: str
    batch_size: int
    val_gen: torch.Generator

    def get_batch(self, split="train"):
        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        g = None if split == "train" else self.val_gen
        lengths = torch.randint(self.min_len, self.max_len, (self.batch_size,), generator=g)
        sequences = [ torch.randint(0, self.vocab_max, (l,), generator=g) for l in lengths ]
        B, T = self.batch_size, int(lengths.max().item())
        tokens = torch.full((B, T), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        targets = torch.full((B, T), IGNORE, dtype=torch.long)

        for (idx, sequence) in enumerate(sequences):
            end = len(sequence)
            if split == "train":
                tokens[idx, :end] = sequence
                targets[idx, :end] = sequence + 1
                attention_mask[idx, :end] = 1
            else:
                tokens[idx, -end:] = sequence
                targets[idx, -end:] = sequence + 1
                attention_mask[idx, -end:] = 1

        return tokens.to(self.device), targets.to(self.device), attention_mask.to(self.device)

