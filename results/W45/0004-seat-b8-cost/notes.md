# W45-0004 · E2 decode 基线（seat）：8 条最长 / B=8 / 预算 128

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0004-seat-b8-cost`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：record-only 基线；`active=8` 的稳态 decode `step_ms` 是 `W45-0005/0012` 的分母。
- 观察：`step_ms = 22.749`、`decode_event_ms = 22.337`、TPOT median **22.674 ms**；8/8 prompt token hash；`n_errors = 0`。5 个 repeat 稳定在 22.5–22.9 ms。
- 结论：seat 基线 **pass**。注意不能用 dry run 的 24.888 ms 作分母：那是含离群值的脏数据；正式基线是 22.749 ms。
- 限制：closed、单 workload、B=8；`warmup=3/repeats=5`。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
