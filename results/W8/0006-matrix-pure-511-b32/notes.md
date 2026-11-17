# W8-0006 · 判据矩阵（pure 0.6B）：ctx 511 / B=32

- 状态：`pass`（reviewer: rainhome，2026-11-17；D10）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`eadc76a`

## 观察

- TTFT p50 **92.7 ms**（queue share p99 0.57）；decode TPOT p50 **24.28 ms**；out_tokens 128
- 峰值并发 **32**；torch 峰值 **3614 MiB**（driver 4323 MiB）；tok/s **1222.2**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- pure 短行用 511；B=32 满池（1300 块），峰值并发 32。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
