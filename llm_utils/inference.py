from model import MiniLLM
import torch
import torch.nn.functional as F

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
def generate(model: MiniLLM, input_ids: torch.Tensor, max_output: int,
        temperature: float=1.0, eos_token=None, top_k=None, top_p=None):
    model.eval()
    B, T = input_ids.shape
    pad_id = eos_token if eos_token is not None else 0
    buffer = torch.full((B, T + max_output), pad_id, dtype=input_ids.dtype, device=input_ids.device)
    buffer[:, :T] = input_ids
    last_token_idx = T
    output_len = torch.zeros(B, dtype=torch.long, device=input_ids.device)

    un_finished = torch.ones(B, dtype=torch.bool, device=input_ids.device)
    for _ in range(max_output):
        active_rows = un_finished.nonzero().flatten()
        if len(active_rows) == 0: break
        ctx = buffer[active_rows][..., max(0, last_token_idx - model.config.block_size):last_token_idx]
        logits, _ = model.forward(ctx)
        logits = logits[:, -1, :]
        next_idx = sample(logits, temperature, top_k, top_p)
        if eos_token is not None:
            finished = next_idx.squeeze(-1) == eos_token
            un_finished[active_rows[finished]] = False
        buffer[active_rows, last_token_idx] = next_idx.squeeze(-1)
        output_len[active_rows] += 1
        last_token_idx += 1
    return [buffer[i, T:T+output_len[i]] for i in range(B)]
