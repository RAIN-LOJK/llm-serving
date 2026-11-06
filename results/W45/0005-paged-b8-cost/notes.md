# W45-0005 · E2 decode（paged，pre-fix）：Triton block-table / B=8 / 832 块

- 状态：`reviewed`；verdict = `fail`
- 配置：`config.toml`（运行目录 `results/W45/0005-paged-b8-cost`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：`decode_step_ratio_max <= 1.15`（目标 1.0）；同组 8/8 token 一致；frag <= 0.05；gather 流量 0。
- 观察：`step_ms = 27.379` vs seat **22.749** = **1.203× → fail**；`decode_event_ms = 26.798` vs 22.337。`frag = 0.0055` pass；8/8 token hash 全等；`n_errors = 0`。
- 归因（修复前 profile）：decode 写入 3.17 ms CPU/step + 每层 launch/重校验 1.83 ms + kernel 2.85 ms GPU；step 是 **CPU-dispatch-bound**，不是 kernel 单点慢。`bench_decode_kernel.py` 的 0.089 ms 是同一层 L2 热读，不能直接外推 28 层冷读。
- 结论：**C1 fail**，保留为修复前证据；由 `W45-0012` supersede。不能引用本 run 的吞吐做结论。
- 限制：单 workload、B=8；失败原因是集成 host 开销，修复见 `01e5512` 与 P-47。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
