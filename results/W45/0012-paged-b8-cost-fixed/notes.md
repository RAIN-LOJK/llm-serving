# W45-0012 · E2 decode（paged，C1 修复后）：Triton block-table / B=8 / 832 块

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0012-paged-b8-cost-fixed`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：`decode_step_ratio_max <= 1.15`（目标 1.0）；E2 同组 8/8 token 一致；frag <= 0.05；gather 流量 0。
- 观察：`step_ms = 23.651` vs seat **22.749** = **1.040× → pass（接受线）**；`decode_event_ms = 23.073`；`frag = 0.0055`；8/8 token hash 全等；`n_errors = 0`。
- 修复：`PagedPool.write_slots` + 每步 slot map；`PagedDecodePlan` 每容器全量校验一次、每层只查 query 布局；block id 每容器校验一次（`01e5512`）。profile 归因见 `W45-0005` / P-47。
- 结论：C1 从 **1.203 fail** 修到 **1.040 pass**；但 1.040 仍高于 1.0 目标，报告必须写“acceptance-pass, not target-pass”。
- 限制：单 workload、B=8、832 块；5 个 repeat 稳定；正式证据用本 run，`W45-0005` 只作修复前对照。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
