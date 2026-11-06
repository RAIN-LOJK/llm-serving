# W45-0009 · E4 端到端（paged）：W4 trace / packed prefill + Triton / 800 块

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0009-paged-saturate`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：两臂 `saturated=true`；paged tok/s >= seat；`kv_total <= 1640`；无错误。E4 token 一致率只作诊断（D7）。
- 观察：**678.59 tok/s** vs seat **330.53** = **2.053×**；峰值并发 34、makespan 10.12 s；`kv_total = 1400 MiB`、`frag_internal = 1.58%`；`n_errors = 0`。也高于 `W4-0009` 的 358.18 tok/s（跨 commit，仅参考）。
- 结论：E4 **pass**（record-only 的吞吐结论）。容量并非同内存对照：paged 峰值 34 条/1400 MiB，seat 9 条/1640 MiB；吞吐比必须和 E1 的 3.111× 容量比分开读。
- 限制：token hash 27/32（84.4%）只作诊断；跨调度器 group 组成不同，bf16 near-tie 会翻转。69 条 CV warning 同 seat，trace 级 tok_s CV **0.50%**。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
