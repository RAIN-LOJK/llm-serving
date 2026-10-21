"""B2b 清洗：datasets/luxun_raw 的「全集」→ datasets/luxun 干净语料。

只处理鲁迅原著（全集 13 部），剔除 meta 文件（简介/目录），
剥围栏 + 段落重排（恢复折行排版） + 内容去重，按文集切 train/val。

train = 11 部文集；val = 整本的《朝花夕拾》《故事新编》（模型未见文本）。
"""
import hashlib
from pathlib import Path

RAW = Path("datasets/luxun_raw/全集")
OUT = Path("datasets/luxun")
VAL_COLLECTIONS = {"朝花夕拾", "故事新编"}   # val 整本文集（train 不出现）
META_KEYWORDS = ("简介", "目录")             # 非鲁迅原文，剔除
MIN_LEN = 20                                 # 过短篇章（<20 字符）视为空/损坏


def clean_article(text: str) -> str:
    """单篇清洗：剥围栏 + 段落重排。

    原格式：行首行尾 ``` 围栏；正文按 ~20 字/行硬折行（断点随机，
    标点会被顶到下一行行首）。段落 = 空行分隔的块，块内拼接即复原原句
    ——折行点是纯排版噪声，不需要按句尾标点判断断句。
    """
    paras, cur = [], []
    for raw_line in text.splitlines():
        s = raw_line.strip()                # 去全角/半角空格缩进
        if s == "```":                      # 剥围栏行
            continue
        if not s:                           # 空行 = 段落边界
            if cur:
                paras.append("".join(cur))
                cur = []
        else:
            cur.append(s)
    if cur:
        paras.append("".join(cur))
    return "\n\n".join(paras)


def main():
    n_train = n_val = n_skip = n_dup = n_meta = 0
    n_chars_train = n_chars_val = 0
    seen = set()                            # 内容级 MD5 去重（防重复篇目）

    for f in sorted(RAW.glob("*/*.md")):
        coll = f.parent.name

        if any(k in f.stem for k in META_KEYWORDS):
            META_KEYWORDS and (n_meta := n_meta + 1)
            n_skip += 1
            continue

        text = clean_article(f.read_text(encoding="utf-8"))
        if len(text) < MIN_LEN:
            n_skip += 1
            continue

        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            n_dup += 1
            n_skip += 1
            continue
        seen.add(h)

        is_val = coll in VAL_COLLECTIONS
        target = OUT / ("val" if is_val else "train") / coll / f"{f.stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")

        if is_val:
            n_val += 1
            n_chars_val += len(text)
        else:
            n_train += 1
            n_chars_train += len(text)

    print(f"train: {n_train} 篇, {n_chars_train:,} 字符")
    print(f"val  : {n_val} 篇, {n_chars_val:,} 字符")
    print(f"跳过 : {n_skip}（其中 meta={n_meta}, 重复={n_dup}, 其余=短篇）")


if __name__ == "__main__":
    main()