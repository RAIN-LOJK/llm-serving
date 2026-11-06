# W45-0007 · E3 prefill A/B（packed）：static B=8 mixed / 1 token

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0007-static-packed-prefill`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：`packed_prefill_ttft_p99_speedup_min = 1.15`；`packed_prefill_ttft_p50_ratio_max = 1.10`。
- 观察：TTFT **p50 53.859 ms / p99 55.149 ms**；vs serial 105.143 / 207.378 → **p99 3.760×**、**p50 0.512×**，两项均 pass。packed `prefill_ms` p50 53.740、`pack_wait` p50 0.064 ms；`n_errors = 0`。
- 结论：packed **pass**。整组一次 varlen forward 把 TTFT 从“随长度/排队位置拉开”收敛到 51.9–55.1 ms；尾部收益远大于中位数收益。
- 限制：同一 static B=8 mixed workload、`max_new_tokens=1`；不要把这组 TTFT 直接外推到开放负载或 decode。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
