# W45-0011 · E4b（可选）continuous / B=8 / mixed / packed prefill

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0011-continuous-b8-mixed`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：record-only；只报 continuous/static 比值与 TTFT，不设通过线。
- 观察：**292.80 tok/s** vs static **157.15** = **1.863×**；两臂都 `saturated=true`、峰值并发 8；TTFT p50 6.03 s / p99 9.82 s（static 14.57 / 30.29）；`n_errors = 0`。
- 结论：E4b **pass（record-only）**。packed prefill 没有抹掉连续批优势：1.863× 高于配置里的 1.2× 负结果线，p99 也明显改善。作为背景，W3 的 C1 是 2.01×，但代码/prefill 路径不同，不能把这个差值直接归因于 packing。
- 限制：token hash 24/32（75.0%）只作诊断；63 条 CV warning 同 static；trace 级 tok_s CV **0.69%**。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
