# W8-0007 · 判据矩阵（pure 0.6B）：ctx 2048 / B=32

- 状态：`pass`（reviewer: rainhome，2026-11-17；D10）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`4ba82f8`

## 观察

- TTFT p50 **462.8 ms**（queue share p99 0.90）；decode TPOT p50 **24.85 ms**；out_tokens 128
- 峰值并发 **32**；torch 峰值 **9068 MiB**（driver 9701 MiB）；tok/s **977.7**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- pure 2k B=32 满池（4400 块）；峰值并发 32。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
