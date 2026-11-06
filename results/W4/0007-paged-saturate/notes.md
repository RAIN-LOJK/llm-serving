# W4-0007 · E3 系统吞吐（paged 臂）：frozen 32 / poisson / B=64 / 820 块

- 状态：`aborted`；`verdict` 保持 `unreviewed`
- 配置：`config.toml`（运行目录 `results/W4/0007-paged-saturate`）
- 类型：`scenario`；workload：`prompts`（frozen 32）
- 证据：只有部分 `metrics.jsonl`，**没有 `summary.json`**

## 中断说明

本 run 在 2026-11-06 13:43–13:45 被工具调用中断（SIGKILL），只完成了 2/3 条 measured trace，
没有写出 `summary.json`。按 run 契约保留目录与部分指标，`manifest.status = aborted`；
**它不是 E3 的证据，也不用于任何结论**。

同 workload 的 paged 臂由 **`W4-0008`** 重跑；`0007` 的部分 `tok_s ≈ 330.7` 只能说明
“中断前看起来和 seat 的 333.4 接近”，不足以作为系统吞吐结论。

## 下一步

- 见 `W4-0008`（同 seed/rate/requests/budgets，干净树正式跑）。
