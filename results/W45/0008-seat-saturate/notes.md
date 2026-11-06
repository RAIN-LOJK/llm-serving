# W45-0008 · E4 端到端（seat）：W4 trace / packed prefill / B=9

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0008-seat-saturate`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：record-only 基线；与 `W45-0009` 同 trace（`d6391043…`）、同 λ=30、同 budgets，两臂都须 `saturated=true`。
- 观察：**330.53 tok/s**、`saturated=true`、峰值并发 9、makespan 20.77 s；`n_errors = 0`。TTFT p50 8.03 s / p99 14.21 s 是开放环排队，不是单请求服务时间。
- 结论：seat E4 基线 **pass**。61 条 CV warning 是 mixed-context 分桶的 artifact；trace 级 tok_s CV 仅 **0.21%**，不套 W2 的 0.10 延迟 CV 目标。
- 限制：open-loop Poisson 的 128 请求/λ=30 工作点；不要把它与 `W4-0006` 的跨 commit 数字混成同 commit 配对。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
