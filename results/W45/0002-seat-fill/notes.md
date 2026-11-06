# W45-0002 · E1 容量（seat）：fill-only / B=9 / serial

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0002-seat-fill`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：record-only 基线；与 `W45-0003` 同 trace、同 1640 MiB 总预算，只验证 allocator 未被注意力路径改动。
- 观察：`peak_concurrency = 9`、`max_batch = 9`；`kv_mb = 1638`、`kv_total_mb = 1640`、`seat_unused_mb = 2`；`n_errors = 0`。
- 结论：seat 基线 **pass**；9 行是 `W45-0003` 的容量分母。与 W4 的 9 行一致，说明新注意力路径没有改变 seat 的分配路径。
- 限制：fill-only，不 decode；`frag_internal` 为空是预期。`prefill=serial` 仅为与 W4 准入路径对齐。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
