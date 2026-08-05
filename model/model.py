import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from model.config import MiniLLMConfig
from model.layers import RMSNorm, TransformerBlock


class MiniLLM(PreTrainedModel):
    config_class = MiniLLMConfig
    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}

    def __init__(self, config: MiniLLMConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, tokens, targets=None, attention_mask=None):
        B, T = tokens.shape
        x = self.token_embedding(tokens)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 使用mask实现loss过滤，但是更加适合生成环境的方法是使用label pad的-100来mask loss
            # 这里处于学习目的
            if attention_mask is not None:
                per_loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    reduction="none"
                )
                loss_mask = attention_mask.view(-1).float()
                if (val_loss := loss_mask.sum()) > 0:
                    loss = (per_loss * loss_mask).sum()/val_loss 
                else:
                    # 全掩行
                    loss = None
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1)
                )
        return logits, loss
