# archive

旧训练学习项目的代码/数据/结果已从工作树删除。

可恢复的**文本快照**在 git tag：

```text
legacy-learning-2026-08
```

查看方式：

```bash
git ls-tree -r --name-only legacy-learning-2026-08 | head
git show legacy-learning-2026-08:model/layers.py
git worktree add /tmp/minillm-legacy legacy-learning-2026-08
```

注意：

- `datasets/` 和模型权重当年就在 `.gitignore` 里，未进 Git；删除后不可从 tag 恢复。
- 旧代码不作为可运行入口；路径、依赖、数据都已变化。
- 摘要与仍然有效的结论见 [docs/history.md](../docs/history.md)。
