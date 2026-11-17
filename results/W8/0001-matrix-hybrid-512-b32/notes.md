# W8-0001 · 判据矩阵（hybrid 0.8B）：ctx 512 / B=32

- 状态：`unreviewed`（agent 起草；`reviewer` 留空，等用户确认）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`39b67d5`

## 观察

- TTFT p50 **1027.9 ms**（queue share p99 0.51）；decode TPOT p50 **33.50 ms**；out_tokens 128
- 峰值并发 **32**；torch 峰值 **2944 MiB**（driver 3635 MiB）；tok/s **620.3**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- B=32 满池（1400 块）；hybrid 512 的 TTFT 含同批 prefill 排队，架构成本看 TPOT。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
