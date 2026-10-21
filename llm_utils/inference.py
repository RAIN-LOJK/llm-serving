import torch
import torch.nn.functional as F
from model import MiniLLM, KVcache

def sample(logits, temperature=1.0, top_k=None, top_p=None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)   # greedy: 跳过后面所有步骤
    logits = logits / temperature

    if top_k is not None:
        topk_val, _ = torch.topk(logits, top_k, dim=-1)
        kth = topk_val[..., -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p is not None and top_p <= 1:
        sorted_logits, original_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        p_sum = torch.cumsum(probs, dim=-1)
        # 这样如果最大的概率大于top_p就会全部删除： is_remove = p_sum > top_p
        is_remove = (p_sum - probs) > top_p
        sorted_logits = torch.where(is_remove, -torch.inf, sorted_logits)
        # inverse_idx = original_idx.argsort(-1)
        # logits = torch.gather(sorted_logits, -1, inverse_idx)
        logits = torch.zeros_like(sorted_logits)
        logits.scatter_(-1, original_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

@torch.no_grad()
def generate(model: MiniLLM, input_ids: torch.Tensor, max_output: int, attention_mask=None,
        temperature=1.0, eos_token=None, top_k=None, top_p=None):
    model.eval()
    B, T = input_ids.shape
    past_kv = KVcache(model.config.n_layer)
    if attention_mask is None:
        attention_mask = input_ids.new_ones(B, T)
    logits, _ = model(input_ids, attention_mask=attention_mask, past_keys_values=past_kv)

    orign = torch.arange(B, device=input_ids.device)
    buffer = torch.zeros(B, max_output, dtype=torch.long, device=input_ids.device)
    outputs = [None] * B
    # 防止因守卫直接break，之后将buffer剩余内容移动到output时不加尾部的pad
    sampled = 0

    for step in range(max_output):
        if buffer.shape[0] == 0:
            break
        next_ids = sample(logits[:, -1, :], temperature, top_k, top_p)
        buffer[:, step] = next_ids.squeeze(-1)
        sampled = step + 1
        if eos_token is not None:
            finished = next_ids.squeeze(-1) == eos_token
            if finished.any():
                done_idx = finished.nonzero().flatten()
                for idx, org in zip(done_idx.tolist(), orign[done_idx].tolist()):
                    outputs[org] = buffer[idx, :sampled].clone()
                unfinished = ~finished
                buffer = buffer[unfinished]
                orign = orign[unfinished]
                attention_mask = attention_mask[unfinished]
                past_kv.cull(unfinished)
                next_ids = next_ids[unfinished]
                if buffer.shape[0] == 0: break
        attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, :1])), -1)
        # 防止额外计算一次forward or 防止输出长度越界
        if step == max_output - 1 or past_kv.past_len(0) + 1 >= model.config.max_position_embeddings: break
        logits, _ = model(next_ids, attention_mask=attention_mask, past_keys_values=past_kv)
    for idx, org in enumerate(orign):
        outputs[org] = buffer[idx, :sampled].clone()
    return outputs
