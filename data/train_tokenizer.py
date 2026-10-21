"""B2c/B2c2：用 HF tokenizers 训练中文 BPE tokenizer（参数化，支持多语料 + 词表档位对比）。

设计（对齐 B2a 定案 + 之前手写版的约束）：
- 单字起始：每个 Unicode 字符（含换行）初始就是一个独立 token
- 换行独立：\n 单独成 token，不参与合并 → 段落边界完整保留
- 特殊 token 固定占 id 0-3：<pad> <bos> <eos> <unk>（正文从 4 开始）
- 训练只用 train 语料，不混 val（避免 tokenizer 学到 val 分布 → 泄漏）

用法：
  python data/train_tokenizer.py --data luxun --vocab 8192   # B2c 原版
  python data/train_tokenizer.py --data fineweb_cn --vocab 8192   # 词表档位对比
  python data/train_tokenizer.py --data fineweb_cn --vocab 16384  # 词表档位对比

验收（主判据是 val 压缩率，val 语料训练未见）：
  vocab 精确、特殊 token id 0-3、单句往返一致、val 压缩率（tokens/字符，<1 说明多字合并）
"""
import argparse
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
DATA_DIRS = {"luxun": "datasets/luxun", "fineweb_cn": "datasets/fineweb_cn"}


def build(train_dir: Path, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.decoder = decoders.BPEDecoder()

    # 单字起始：按 \n 切分 → 每段文本是一个 word，BPE 在段内合并单字；
    # \n 独立成 token（段落边界保留、不参与合并）。不能按单字符切——
    # 那样每个 word 只有一个 token，没有 pair 可合并（BPE 按 word 内统计）。
    tok.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"\n"), behavior="isolated"
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,   # 按序占 id 0-3
        show_progress=True,
    )
    files = [str(p) for p in train_dir.rglob("*.txt")]
    tok.train(files, trainer)
    return tok


def val_compression(tok: Tokenizer, val_dir: Path, limit: int = 30) -> float:
    """val 语料抽样测压缩率（tokens/字符；语料未见，防 tokenizer 过拟合 train 分布）。"""
    n_tok = n_chr = 0
    for p in sorted(val_dir.rglob("*.txt"))[:limit]:
        t = p.read_text(encoding="utf-8")
        n_chr += len(t)
        n_tok += len(tok.encode(t).ids)
    return n_tok / n_chr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=DATA_DIRS.keys(), default="luxun")
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    root = Path(DATA_DIRS[args.data])
    tok = build(root / "train", args.vocab)

    # 1. vocab 与特殊 token id 校验
    assert tok.get_vocab_size() == args.vocab, tok.get_vocab_size()
    print(f"vocab = {tok.get_vocab_size()}")
    ids = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    print("特殊 token id:", ids)
    assert ids == {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}, ids

    # 2. 往返验收（含换行，验证段落边界保留）
    s = "氢能是清洁能源\n碳达峰、碳中和目标下，能源体系如何转型？\n这是现代中文测试段落。"
    enc = tok.encode(s)
    print("encode ids:", enc.ids)
    print("decode    :", tok.decode(enc.ids))
    assert tok.decode(enc.ids) == s, "往返不一致！"
    print("往返 OK")

    # 3. val 压缩率（主判据）
    comp = val_compression(tok, root / "val")
    print(f"val 压缩率: {comp:.3f} tokens/字符  (luxun 参考 0.73；>0.75 建议升词表档)")
    if args.data == "fineweb_cn" and comp > 0.75:
        print("! 压缩率偏高：8192 词表偏挤，建议 --vocab 16384 对比")

    out = root / "tokenizer.json"
    tok.save(str(out))
    print(f"saved -> {out}")