# W8-0004 · 判据矩阵（pure 0.6B）：ctx 511 / B=1

- 状态：`pass`（reviewer: rainhome，2026-11-17；D10）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`ee54402`

## 观察

- TTFT p50 **3015.7 ms**（queue share p99 1.00）；decode TPOT p50 **22.94 ms**；out_tokens 128
- 峰值并发 **1**；torch 峰值 **1507 MiB**（driver 2021 MiB）；tok/s **42.8**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- pure 短行用 511（512 EOS，P-31）；B=1 的 TTFT 含前序请求 128-token decode。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
