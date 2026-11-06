import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from model.config import MiniLLMConfig
from model.layers import RMSNorm, TransformerBlock, rope_cos_sin
from model.slab import as_decode_container


class MiniLLM(PreTrainedModel):
    config_class = MiniLLMConfig
    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}

    def __init__(self, config: MiniLLMConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.head_dim = getattr(config, "head_dim", None) or config.n_embd // config.n_head
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-row RoPE needs a table to index into (positions arrive as a tensor, so the
        # table cannot be rebuilt per forward inside a captured graph). One table for the
        # whole model, keyed by (length, device, dtype) -- not one per layer (P-18).
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._rope_key: tuple | None = None

        self.post_init()

    def _rope_rows(self, positions: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row cos/sin gathered from one table: (B, 1, head_dim // 2)."""
        key = (length, positions.device, self.token_embedding.weight.dtype)
        if self._rope_key != key:
            self._rope_cache = rope_cos_sin(
                self.head_dim, self.rope_theta, 0, length,
                positions.device, self.token_embedding.weight.dtype,
            )
            self._rope_key = key
        cos, sin = self._rope_cache
        return cos[positions].unsqueeze(1), sin[positions].unsqueeze(1)

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

    def forward(self, tokens, targets=None, attention_mask=None, past_keys_values=None, logits_to_keep=None):
        B, T = tokens.shape
        x = self.token_embedding(tokens)

        # Explicit protocol dispatch (``model.slab.as_decode_container``): a ``KvSlab`` and
        # W4's ``PagedKV`` both carry ``decode_container = True`` and take the decode path;
        # a plain ``KVcache`` (or anything unrecognised) keeps the growing-cache path.
        slab = as_decode_container(past_keys_values)
        if slab is None:
            past_len = 0 if past_keys_values is None else past_keys_values.past_len(0)
            cos, sin = rope_cos_sin(self.head_dim, self.rope_theta, past_len, T, x.device, x.dtype)
        else:
            if B != slab.batch:
                raise ValueError(
                    f"the KvSlab path is one whole-block forward: tokens batch {B} must equal "
                    f"slab.batch {slab.batch} (the pool is forwarded as a block, including its "
                    "idle seats); slice/select rows instead of forwarding a subset"
                )
            if T != 1:
                raise ValueError(
                    "the KvSlab path is decode-only (D4: T == 1); prefill via KVcache then "
                    "fill_from(), never pad a prefill into (B, T_max)"
                )
            # The mask must already allow the column this step writes, and it is rebuilt
            # in place so a captured graph only reads it.
            slab.build_mask(next_token=True)
            cos, sin = self._rope_rows(slab.pos, slab.max_len)

        for idx, layer in enumerate(self.layers):
            x = layer(x, attention_mask, idx, past_keys_values, cos, sin)
        if slab is not None:
            slab.advance()

        x = self.norm(x)
        if logits_to_keep is not None:
            # 训练形状的 API 默认给全序列 logits（算 loss 需要每个位置）；推理只要末位，
            # 必须在 lm_head 之前切，否则 593 MiB 的 [T, vocab] 张量和那次大 matmul 已经付掉了（P-13）。
            if targets is not None:
                raise ValueError("logits_to_keep is an inference-only slice; it cannot feed a loss")
            x = x[:, -int(logits_to_keep):]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 使用mask实现loss过滤，但是更加适合生成环境的方法是使用label pad的-100来mask loss
            # 这里处于学习目的
            if attention_mask is not None:
                loss_mask = attention_mask.view(-1).float()
                if (val_loss := loss_mask.sum()) > 0:
                    per_loss = F.cross_entropy(
                        logits.view(-1, self.config.vocab_size),
                        targets.view(-1),
                        reduction="none"
                    )
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
