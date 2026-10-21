"""C 阶段基准语料：Fineweb-Edu-Chinese 5% 采样 → 清洗 → seed 固定切片。

产出 datasets/fineweb_cn/{train,val}/*.txt（每文件一篇文档，对齐 TextData 输入），
+ manifest.json 记录切片协议（seed / hash 规则 / 规模）→ 实验矩阵固定可比。

切片协议（永久固定，不可重跑改参数）：
- 清洗（去乱码/url/markdown 残留，见 clean()）→ 每篇文档指纹 h = sha256(seed + text)
- 入选：h % 65536 < S（S 按目标字符量推算）；val：入选且 (h>>16) % 8 == 0
- train/val 由哈希分桶决定 → 天然不相交，且与文档顺序无关
- 过短（< MIN_LEN）与完全重复的文档丢弃
"""
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("datasets/fineweb_cn_raw/fineweb_edu_sampled_5_percent/fineweb_edu_sampled_5_percent_001.parquet")
OUT = Path("datasets/fineweb_cn")
TRAIN_CHARS = 4_000_000          # 目标 train 字符量（≈ luxun 4.5x，3-4 分钟/配置档）
VAL_DENOM = 8                    # 入选文档 1/8 划为 val
SEED = b"minillm-b3-fineweb-edu-cn-v2"
MIN_LEN = 400                    # 过短丢弃（TextData 会丢 < block_size 的文档）
MIN_ZH = 0.3                     # 中文字符占比下限（滤掉混入的英文论文/外文文档）
SAMPLE_N = 65536                 # 哈希取模空间

_RE_URL = re.compile(r"https?://\S+")
_RE_HEAD = re.compile(r"^#{1,6}\s*", re.M)
_RE_WIKI = re.compile(r"\[\[([^\[\]]+)\]\]")
_RE_STAR = re.compile(r"\*{1,2}")
_RE_BLANK = re.compile(r"\n{3,}")
_RE_REF = re.compile(r"\[\[@[^\[\]]+\](?:,\[[^\[\]]+\])*\]|\[\[@ref\d+\](?:\[[^\]]*\])*\]")
_RE_FIG = re.compile(r"\[\[[^\]]*\]\(#[^)]*\)\{[^}]*\}")
_RE_REFATTR = re.compile(r"\{ref-type=\"[^\"]*\"\}")
_RE_ANCHOR = re.compile(r"\{#[^}\n]*\}")
_RE_RULE = re.compile(r"^\s*[-=]{3,}\s*$", re.M)


def zh_ratio(text: str) -> float:
    n = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return n / max(1, len(text))


def clean(text: str) -> str:
    t = text.replace("\uFFFD", "")
    t = t.replace("\\[", "[").replace("\\]", "]")   # JATS 转义括号还原
    t = _RE_URL.sub("", t)
    t = _RE_WIKI.sub(lambda m: m.group(1).split("|")[-1], t)
    t = _RE_STAR.sub("", t)
    t = _RE_HEAD.sub("", t)
    t = _RE_REF.sub("", t)
    t = _RE_FIG.sub("", t)
    t = _RE_REFATTR.sub("", t)
    t = _RE_ANCHOR.sub("", t)
    t = _RE_RULE.sub("", t)
    t = _RE_BLANK.sub("\n\n", t)
    return t.strip()


def doc_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(SEED + text.encode("utf-8")).digest(), "big")


def iter_docs(pf):
    for batch in pf.iter_batches(batch_size=5000, columns=["text"]):
        for t in batch["text"].to_pylist():
            t = clean(t) if t else ""
            if len(t) < MIN_LEN or zh_ratio(t) < MIN_ZH:
                continue
            yield t


def pass1() -> tuple[int, int]:
    """全量扫描：统计候选文档数与总字符量（不含重复）。"""
    pf = pq.ParquetFile(RAW)
    seen, n, chars = set(), 0, 0
    for t in iter_docs(pf):
        h = doc_hash(t)
        if h in seen:
            continue
        seen.add(h)
        n += 1
        chars += len(t)
    return n, chars


def main():
    n, chars = pass1()
    val_share = 1 / VAL_DENOM
    s = max(1, round(SAMPLE_N * TRAIN_CHARS / chars * (1 + val_share)))
    print(f"candidates={n} chars={chars:,}  S={s}  (入选 {s/ SAMPLE_N:.2%})")

    pf = pq.ParquetFile(RAW)
    seen, out = set(), {"train": [], "val": []}
    for t in iter_docs(pf):
        h = doc_hash(t)
        if h in seen or h % SAMPLE_N >= s:
            continue
        seen.add(h)
        split = "val" if (h >> 16) % VAL_DENOM == 0 else "train"
        out[split].append(t)

    for split in ("train", "val"):
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(sorted(out[split])):
            (d / f"doc_{i:06d}.txt").write_text(t, encoding="utf-8")

    manifest = {
        "source": "wangkunqing/mini_llm_dataset → fineweb_edu_sampled_5_percent.zip",
        "upstream": "opencsg/Fineweb-Edu-Chinese-V2.1 (score 4-5, 5% 等比例采样)",
        "seed": SEED.decode(),
        "cleaning": "去�/JATS转义括号/url/`**`/`[[..]]`(取显示文本)/`#`标题行首/`{#anchor}`/`---===`分隔线/JATS引用/折叠3+空行",
        "filter": f"len>={MIN_LEN} 且 中文字符占比>={MIN_ZH}（滤英文混入）",
        "rule": "h=sha256(seed+cleaned_text); include: h%65536<S; val: included and (h>>16)%8==0",
        "S": s,
        "min_len": MIN_LEN,
        "n_train": len(out["train"]),
        "n_val": len(out["val"]),
        "chars_train": sum(map(len, out["train"])),
        "chars_val": sum(map(len, out["val"])),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
