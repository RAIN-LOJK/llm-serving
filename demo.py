"""A4 交互 demo：终端流式续写。

REPL：用户输入整数序列（复读机任务，值域 [0, 128)），
模型逐 token 流式续写，肉眼看到 token 一个个冒出来。
B3 换成真实 tokenizer 后，同一脚本就是"看得懂的续写"。
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from llm_utils import sample
from model import KVcache
from train import load_model

VOCAB_MIN = 0          # 复读机任务值域下界（含）
VOCAB_MAX = 128        # 复读机任务值域上界（不含）——模型只见过 [0,128) 的 token
MAX_OUTPUT = 32        # 每轮最多续写多少 token
STREAM_DELAY = 0.05    # 特效延迟：让"流"肉眼可见（真实模型生成本来就慢，天然流式）
device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    """加载复读机任务 checkpoint，挪到 device 并置 eval。"""
    model = load_model("weights/minillm.pth").to(device)
    model.eval()
    return model


def parse_input(line: str):
    """原始输入字符串 -> 合法 token 张量 [1, T]；非法则打印提示并返回 None。

    合法域 = [VOCAB_MIN, VOCAB_MAX)：模型训练时只见过这个区间的 token 当内容
    （255 虽不崩，但只被当 pad 用、从没被当内容见过）。
    校验用白名单思路：放行合法域，其余全拒——而不是列非法值，那样列不完。
    """
    # 第一层：解析。非整数输入在这里被 try/except 拦住。
    try:
        nums = [int(x) for x in line.split()]   # split() 按任意空白切，支持多空格/tab
    except ValueError:
        print("✗ 包含非整数，请重新输入")
        return None

    # 第二层：校验。越界值在这里被范围检查拦住。
    bad = [x for x in nums if not (VOCAB_MIN <= x < VOCAB_MAX)]
    if bad:
        print(f"✗ 越界值 {bad}，合法范围 [{VOCAB_MIN}, {VOCAB_MAX})")
        return None

    return torch.tensor(nums, device=device).unsqueeze(0)   # [1, T]


def stream_predict(model, tokens, max_output):
    """逐 token 续写并实时打印，返回续写的整数列表。"""
    past_kv = KVcache(model.config.n_layer)
    logits, _ = model(tokens, past_keys_values=past_kv)   # prefill：为第 1 个生成 token 备好 logits

    res = []
    for step in range(max_output):
        next_ids = sample(logits[:, -1, :], temperature=0.0)   # greedy：诚实展示学到的规律
        tok = next_ids.item()
        res.append(tok)
        print(tok, end=" ", flush=True)     # flush=True 让 token 立刻上屏
        time.sleep(STREAM_DELAY)            # 特效延迟：让"流"可见
        if step == max_output - 1:
            break                           # 尾部挡板：不再 forward 预测不存在的下一个
        logits, _ = model(next_ids, past_keys_values=past_kv)   # decode 步：长度 1，复用同一 KV
    print()
    return res


def main():
    model = load_model()
    print("MiniLLM 终端续写 demo（复读机任务，输入 0-127 整数序列，空行退出）")
    while True:
        try:
            line = input("> ")
        except EOFError:                    # Ctrl-D 也算退出
            break
        if not line.strip():
            break                           # 空行退出（exit 归 main 管）
        tokens = parse_input(line)
        if tokens is None:
            continue                        # 非法输入：提示已打印，重输，不退出
        stream_predict(model, tokens, MAX_OUTPUT)


if __name__ == "__main__":
    main()
