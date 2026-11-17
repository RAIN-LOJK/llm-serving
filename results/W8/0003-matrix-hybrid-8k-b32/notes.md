# W8-0003 · 判据矩阵（hybrid 0.8B）：ctx 8192 / B=32

- 状态：`unreviewed`（agent 起草；`reviewer` 留空，等用户确认）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`54c1e88`

## 观察

- TTFT p50 **9261.3 ms**（queue share p99 0.97）；decode TPOT p50 **38.33 ms**；out_tokens 128
- 峰值并发 **32**；torch 峰值 **6406 MiB**（driver 7403 MiB）；tok/s **172.1**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- B=32 满池（16700 块，KV 6.1 GiB）；8k 单 prompt 独占一个 prefill 组。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
