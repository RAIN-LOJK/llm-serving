# W8-0005 · 判据矩阵（pure 0.6B）：ctx 511 / B=8

- 状态：`unreviewed`（agent 起草；`reviewer` 留空，等用户确认）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`b66950c`

## 观察

- TTFT p50 **48.5 ms**（queue share p99 0.00）；decode TPOT p50 **23.87 ms**；out_tokens 128
- 峰值并发 **8**；torch 峰值 **1942 MiB**（driver 2599 MiB）；tok/s **326.7**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- pure 短行用 511；B=8 单波，TTFT 干净。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
