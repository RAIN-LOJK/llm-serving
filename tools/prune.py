"""清理实验档案：删除 results/expXXX/，其 result.md 记录默认移入 archive/result.md。

只动 results/、result.md、archive/ —— 不碰 config/（实验配方，永不删）、不碰 weights/（开发工作区）。

用法:
    python tools/prune.py exp003 exp005          # 删指定实验（有台账记录 → 归档；无 → 仅删目录）
    python tools/prune.py exp003 --force         # 跳过确认直接删
    python tools/prune.py exp003 --no-archive    # 强制不归档：目录 + 台账记录全删
    python tools/prune.py --list                 # 列出 results/ 现有实验

校验: expXXX 目录存在、编号合法；result.md 同步移除该节与索引表行（精确匹配防误删同名前缀）。
"""
import re
import shutil
import sys
from pathlib import Path

RESULTS = Path("results")
RESULT_MD = Path("result.md")
ARCHIVE_MD = Path("archive") / "result.md"


def list_exps() -> list[str]:
    if not RESULTS.is_dir():
        return []
    return sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and re.fullmatch(r"exp\d+", p.name))


def md_id_for(exp_id: str) -> str:
    """目录形 exp999 → 台账形 Exp-999（result.md 节标题/索引表用带连字符的大写形）。"""
    m = re.search(r"\d+", exp_id)
    return f"Exp-{m.group(0)}" if m else exp_id


def extract_record_text(md_id: str) -> str:
    """从 result.md 提取该实验的索引表行 + 全部节，移除并返回待归档文本。

    匹配加固：索引行锚定 `| Exp-XXX |` 前缀（防 Exp-002 误删 Exp-0020 的行）；
    节标题锚定 `## Exp-XXX` 且循环提取全部（同 exp 多次记录 = 同源实验，全部迁移）。
    无记录返回 ""，不写回。
    """
    if not RESULT_MD.exists():
        return ""
    text = RESULT_MD.read_text(encoding="utf-8")
    key = f"| {md_id.lower()} |"

    # 1. 索引表行
    lines = text.splitlines()
    rows, keep = [], []
    for ln in lines:
        if ln.lstrip().lower().startswith(key):
            rows.append(ln)
        else:
            keep.append(ln)
    text = "\n".join(keep)

    # 2. 全部实验节（循环提取，避免只取第一节）
    pattern = re.compile(
        rf"^## {re.escape(md_id)}\n.*?(?=^## |\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    sections = []
    while True:
        m = pattern.search(text)
        if not m:
            break
        sections.append(m.group(0).rstrip())
        text = text[:m.start()] + text[m.end():]

    if not rows and not sections:
        return ""
    # 折叠残留分隔线
    text = re.sub(r"-{3,}(\s*-{3,})+", "---", text)
    text = re.sub(r"---(\s*\n)*---", "---", text)
    RESULT_MD.write_text(text.rstrip() + "\n", encoding="utf-8")
    return "\n\n".join(rows + ["---"] + sections)


def append_archive(md_id: str, record_text: str):
    """归档文本纯追加到 archive/result.md（日志语义，允许同 exp 重复累积，可整体删除）。"""
    ARCHIVE_MD.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_MD.exists():
        ARCHIVE_MD.write_text("# 实验归档（阶段性记录日志，可整体删除；重要实验不会复用 exp id）\n\n",
                              encoding="utf-8")
    text = ARCHIVE_MD.read_text(encoding="utf-8").rstrip()
    ARCHIVE_MD.write_text(text + "\n---\n\n" + record_text + "\n", encoding="utf-8")
    print(f"→ {md_id} 旧记录已归档 → archive/result.md")


def prune_exp(exp_id: str, force: bool, no_archive: bool = False) -> bool:
    """删除一个实验目录；result.md 记录默认归档（--no-archive 则全删）。返回是否执行了删除。"""
    if not re.fullmatch(r"exp\d+", exp_id):
        print(f"!! 非法编号: {exp_id}（应为 expXXX）")
        return False
    target = RESULTS / exp_id
    if not target.is_dir():
        print(f"!! {exp_id} 不存在于 results/")
        return False

    print(f"将删除: {target}/（含 config.json / model.pth / train_log.csv / samples.txt）")
    if not force:
        reply = input("确认删除? [y/N] ").strip().lower()
        if reply != "y":
            print("已取消")
            return False

    # 台账：有记录 → 归档（或 --no-archive 全删）；无记录 → 仅删目录
    md_id = md_id_for(exp_id)
    removed = extract_record_text(md_id)
    if removed:
        if no_archive:
            print(f"✓ result.md 已移除 {md_id} 记录（--no-archive：不归档）")
        else:
            append_archive(md_id, removed)
    else:
        print(f"! result.md 未找到 {md_id} 记录（仅删目录）")

    shutil.rmtree(target)
    print(f"✓ 已删除 {target}/")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="*", help="实验 id，如 exp003")
    ap.add_argument("--force", action="store_true", help="跳过确认直接删")
    ap.add_argument("--no-archive", action="store_true",
                    help="强制不归档：目录 + result.md 记录全删")
    ap.add_argument("--list", action="store_true", help="列出 results/ 现有实验")
    args = ap.parse_args()

    if args.list:
        for e in list_exps():
            print(e)
        return

    if not args.exps:
        ap.print_help()
        sys.exit(1)

    for e in args.exps:
        prune_exp(e, args.force, args.no_archive)


if __name__ == "__main__":
    main()
