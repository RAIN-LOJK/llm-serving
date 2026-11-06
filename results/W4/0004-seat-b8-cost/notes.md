# W4-0004 · E2 代价基线（seat）：8 条最长 / B=8 / budget 128

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W4/0004-seat-b8-cost`）
- 类型：`scenario`；workload：`prompts`（`l01`–`l08`，long 桶 8 条）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `seat_decode_step_cost_is_the_cost_baseline_for_paged`
- E2 seat 臂，作为 C4 的基线；`record_only`，不单独判 pass/fail。
- 固定 8 条最长 frozen prompt、budget 128、closed、`max_batch=8`：8 行全忙，
  seat 行宽 `bucket(1517 + 128 + 1, 64) = 1664`，因此每一步的 attention 宽度都是 1664。
- 只统计 decode 步的 `step_ms`；`decode_event_ms` 只作诊断（D6）。

## 观察

- **decode step 中位数 21.32 ms**（625 个 decode step，5 次重复；范围 21.25–21.35 ms，
  CV 0.16%）；`decode_event_ms` 21.19 ms、`host_gap_ms` 0.126 ms。
- `tok_s = 355.93`（decode-only 379.78），makespan 2.877 s。
- 显存：`kv_mb = 1456.0`（8 × 182 MiB）、`workspace_mb = 0`、
  `bytes_per_active_request = 190.84 MB`（约 182 MiB/请求）。
- 8 行在 625 个 decode step 中始终活跃；`summary.steps.by_active` 里 active=8 的
  `step_ms = 22.64` 把 prefill 的 step 0（prefill 约 4.4 ms）也算进去了，
  C4 用的是 decode-only 的 21.32 ms。
- 8 条 prompt × 5 次重复的 `token_ids` 完全一致；与 `W4-0005` 逐 token 相同（见那边）。
- `trace_sha256 = 25cd4f62...`、`prompt_sha256 = 6c40a2eb...`，与 `W4-0005` 相同。

## 结论（含“什么时候不划算”）

- 这份 run 是 C4 的分母：同并发（B=8）、同预算、同 8 条 prompt，seat 的稳态 decode
  成本为 **21.32 ms/step**。
- 它本身不判任何东西；是否「划算」由 `W4-0005` 的 `step_ms` 比值回答。
- 注意：seat 的 `lpad` 不逐 step 记录（行宽固定 1664）；做 C4 的 matched-width 比较时，
  只能用 paged 的 `lpad=1664` 子集对齐。

## 限制与偏差

- closed 波次而不是 Poisson：没有准入/排队噪声，正是 E2 想要的；代价是它不代表
  稳态系统吞吐。
- 只有 8 条长 prompt，不代表 32 条混合长度；E2 只回答「同并发、同宽度的 step 代价」。
- 设备：单张 RTX 4090 24 GB，bf16，`device_class=measure-cuda`，commit `9dc5211`，clean。

## 下一步

- `W4-0005`：同 workload 的 paged 臂，832 块；比较 decode `step_ms`、frag 与 workspace。
