# W45-0003 · E1 容量（paged）：fill-only / 800 块 / serial

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0003-paged-fill`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：`capacity_speedup_min = 1.50`；与 `W45-0002` 同 trace、同总预算；`kv_total <= 1640 MiB`。
- 观察：`peak_concurrency = 28` vs seat **9** = **3.111×**；`kv_mb = 1400` + workspace **201.97** = **1601.97 MiB**；`reservation_slack = 2.68%`；`n_errors = 0`。
- 结论：C1 容量 **pass**，3.111× 与 W4 的 28 vs 9 一致；新路径没有碰块分配器。
- 限制：fill-only，`frag_internal` 在 summary 中为空；C2 碎片在 E2（`W45-0012`）判定。`prefill=serial` 是控制臂，不是生产回退。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
