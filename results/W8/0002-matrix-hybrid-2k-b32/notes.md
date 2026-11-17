# W8-0002 · 判据矩阵（hybrid 0.8B）：ctx 2048 / B=32

- 状态：`unreviewed`（agent 起草；`reviewer` 留空，等用户确认）
- 证据：`summary.json` / `metrics.jsonl`；口径：`docs/decisions.md` D10、`docs/w8-plan.md`
- 配置：`config.toml`；类型：`scenario`；base commit：`442acbf`

## 观察

- TTFT p50 **2762.9 ms**（queue share p99 0.88）；decode TPOT p50 **37.25 ms**；out_tokens 128
- 峰值并发 **32**；torch 峰值 **3506 MiB**（driver 4197 MiB）；tok/s **384.6**

## 结论

- 判据矩阵的一格（`reports/matrix.md`）；`record_only`，无同批对照臂，不单独下 pass/fail。
- B=32 满池（4400 块）；2k 的 TTFT 主要由同批 8k-token prefill 组贡献。

## 限制

- 单卡 4090、bf16、synthetic prompt；不与其它设备 / dtype 比较。
