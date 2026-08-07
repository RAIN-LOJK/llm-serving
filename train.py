"""
批量变长训练: 复读机任务 (随机值序列, 规律 y = x + 1), 每条样本长度随机
目的:
  1. 激活 attention_mask: 右 pad 训练 (loss 用 -100 ignore), 左 pad 评估
  2. 建立"每序列独立长度"心智模型, 为 KV cache 铺路
  3. 产出 checkpoint, 供 KV cache 阶段在同一份权重上对比

约定 (HF 风格):
  tokens  pad 用 PAD_ID (合法 token id, embedding 要查表)
  targets pad 用 IGNORE (-100, cross_entropy ignore_index 自动忽略)
  attention_mask 只管 attention 层 (左 pad 时必须, 右 pad 时 causal 已挡住)
"""
import torch
import matplotlib.pyplot as plt

from model import MiniLLM, MiniLLMConfig
from llm_utils.inference import generate

PAD_ID = 255        # tokens 的 pad token (合法 id)
IGNORE = -100      # targets 的 pad (cross_entropy ignore_index)
VOCAB_MAX = 128     # 输入值域 [0, VOCAB_MAX)

device = "mps" if torch.backends.mps.is_available() else "cpu"

def make_data(batch_size, min_len=32, max_len=64, pad_side="right", device=None):
    """
    采样一个 batch 的变长随机序列样本。
    规律: y = x + 1, 输入值域 [0, VOCAB_MAX-1] 保证 x+1 不越界
    ！函数最后才to device，故中间数据生成处理都在cpu进行

    返回:
        tokens:         (B, T_max)
        targets:        (B, T_max), pad 位置填 IGNORE
        attention_mask: (B, T_max), 真实位置 1, pad 位置 0
    """
    # TODO 1: 采样 batch_size 个长度 L_i, 均匀取自 [min_len, max_len - 1]
    # TODO 2: 逐条生成随机序列 tokens_i (值域 [0, VOCAB_MAX-1]), 存入 list
    # TODO 3: targets_i = tokens_i + 1 (值域自动在 [1, VOCAB_MAX], 不越界)
    # TODO 4: T_max = max(L_i), 按 pad_side 构造定长 tokens / targets
    #   右 pad: 内容靠左, pad 在右 -> tokens[i] = [内容..., PAD_ID...]
    #   左 pad: 内容靠右, pad 在左 -> tokens[i] = [PAD_ID..., 内容...]
    #   targets 与 tokens 同位对齐, pad 处填 IGNORE
    # TODO 5: attention_mask: 真实位置 1, pad 位置 0
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    lengths = torch.randint(min_len, max_len, (batch_size,))
    sequences = [ torch.randint(0, VOCAB_MAX, (l,)) for l in lengths ]
    B, T = batch_size, int(lengths.max().item())
    tokens = torch.full((B, T), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)
    targets = torch.full((B, T), IGNORE, dtype=torch.long)

    for (idx, sequence) in enumerate(sequences):
        end = len(sequence)
        if pad_side == "right":
            tokens[idx, :end] = sequence
            targets[idx, :end] = sequence + 1
            attention_mask[idx, :end] = 1
        else:
            tokens[idx, -end:] = sequence
            targets[idx, -end:] = sequence + 1
            attention_mask[idx, -end:] = 1

    return tokens.to(device), targets.to(device), attention_mask.to(device)


def evaluate(model: MiniLLM, num_batch=10, batch_size=64):
    """
    在左 pad 的 batch 上统计 next-token 正确率 (masked accuracy)。
    左 pad 同时验证 attention 层的 pad mask 生效 (causal 挡不住左边的 pad)。
    """
    model.eval()
    # TODO: 循环 num_batch 次: make_data(pad_side="left")
    #   -> 前向 -> argmax 与 targets 比较 -> 只统计 attention_mask==1 的位置
    #   -> 累计正确数 / 累计有效数 得到 acc
    total, corrects = 0, 0
    with torch.no_grad():
        for _ in range(num_batch):
            tokens, targets, mask = make_data(batch_size, 32, 64, "left", device)
            logits, _ = model(tokens, targets, mask)
            preds = logits.argmax(-1)
            m = mask.bool()
            corrects += (preds[m] == targets[m]).sum().item()
            total += m.sum().item()
    model.train()
    return corrects/total


def save_checkpoint(model, path):
    """保存模型参数 + config, 供后续加载 (KV cache 阶段要用)"""
    # TODO: torch.save({"config": ..., "state_dict": ...}, path)
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, path)


def load_checkpoint(path):
    """从 checkpoint 重建模型"""
    # TODO: 读取文件 -> 从 config 构建 MiniLLM -> load_state_dict -> 返回 model
    weights = torch.load(path)
    model = MiniLLM(MiniLLMConfig(**weights["config"]))
    model.load_state_dict(weights["state_dict"])
    return model


def main():
    torch.manual_seed(0)
    config = MiniLLMConfig()
    model = MiniLLM(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # TODO: 训练循环 (约 2000 步, 每 200 步打印 loss + evaluate)
    #   1. make_data(batch_size=32, pad_side="right")
    #   2. model(tokens, targets=targets, attention_mask=attention_mask)
    #   3. zero_grad -> backward -> step
    all_step = 1200
    loss_log, eval_log = list(), list()
    for step in range(all_step):
        tokens, targets, attention_mask = make_data(32, 32, 65, device=device)
        _, loss = model(tokens, targets, attention_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            loss_log.append((step, loss.item()))
            if step % 200 == 0:
                eval_res = evaluate(model)
                eval_log.append((step, eval_res))
                print(f"step {step:4d} | loss {loss.item():.4f} | acc {eval_res:.3f}")

    # TODO: 保存 checkpoint 到 checkpoints/train.pt
    save_checkpoint(model, "weights/minillm.pth")

    # 绘制训练曲线
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    steps, losses = zip(*loss_log)
    ax1.plot(steps, losses)
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")

    esteps, accs = zip(*eval_log)
    ax2.plot(esteps, accs, color="orange")
    ax2.set_ylabel("acc")
    ax2.set_xlabel("step")
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)

    # TODO: 最终验证: load_checkpoint 加载后, 用 generate 从几个等长 prompt 续写
    #   期望输出: 每个位置 +1 的递增序列
    model = load_checkpoint("weights/minillm.pth")
    model.to(device)
    prompts = torch.randint(0, VOCAB_MAX, (3, 32)).to(device)
    res = generate(model, prompts, 5, 0.01)
    print(f"input: {prompts}")
    print(res)



if __name__ == "__main__":
    main()
