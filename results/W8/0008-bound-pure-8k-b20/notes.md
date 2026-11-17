# W8-0008 · 边界探针（pure 0.6B）：ctx 8192 / B=20（8k×B=32 不可行）

- 状态：`pass`（reviewer: rainhome，2026-11-17；D10）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`9fcd669`

## 观察

- TTFT p50 **1634.4 ms**（queue share p99 0.96）；decode TPOT p50 **25.42 ms**；out_tokens 2
- 峰值并发 **0**；torch 峰值 **19716 MiB**（driver 20453 MiB）；tok/s **12.2**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- 边界探针：8k 纯臂第 2 个 token 即 EOS，peak_concurrency=0，decode/TPOT 不可引用；只作内存上界（19.25 GiB）。B=32 光 KV 需 28.1 GiB。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
