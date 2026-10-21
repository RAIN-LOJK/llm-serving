import torch
from typing import Protocol
from config import Config
from data.text import TextData

class DataProtocol(Protocol):
    """所有数据类必须实现的接口"""
    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:...

def build_data(cfg: Config, device="cpu") -> DataProtocol:
    d = cfg.data
    return TextData(d.tokenizer_path, d.train_dir, d.val_dir, d.block_size,
                    cfg.train.batch_size, device, cfg.train.seed)
