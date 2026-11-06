# W45-0010 · E4b（可选）static / B=8 / mixed / packed prefill

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0010-static-b8-mixed`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：record-only；与 `W45-0011` 同 W3 trace（rate 11、128 请求、B=8），只差 `policy=static|continuous`。
- 观察：**157.15 tok/s**、`saturated=true`、峰值并发 8、makespan 43.68 s；TTFT p50 14.57 s / p99 30.29 s；`n_errors = 0`。
- 结论：static E4b 基线 **pass**；63 条 CV warning 是 mixed-context bucket artifact，trace 级 tok_s CV **0.86%**。
- 限制：E4b 只报不判；不能把 static 的 TTFT 与 packed 的 closed-loop E3 混读。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
