# W4-0002 · E1 容量（seat）：W3-like / fill-only / B=9

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W4/0002-seat-fill-max2048`）
- 类型：`fill`；workload：`prompts`（32 条 frozen prompt）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `seat_capacity_is_limited_by_whole_trace_max_length`
- E1 seat 臂，作为 C1 的分母；`record_only`，不单独判 pass/fail。
- D6：seat 的一行 = 一块 `KvSlab` 的 `max_batch` 行，行宽 =
  `bucket(trace 最长 prompt + budget + 1, 64)`。最长 prompt 为 1517（l05），
  `bucket(1517 + 128 + 1, 64) = 1664` 列 → **182 MiB/行**；
  `floor(1640 / 182) = 9`。`max_batch=9` 是让**显存**先于 `max_batch` 成为约束，
  不是把 seat 人为调小。

## 观察

- `capacity.peak_concurrency = 9`；`kv_mb = 1638.0`、`seat_unused_mb = 2.0`、
  `kv_total_mb = 1640.0`（1640 是两臂共同的上限，seat 实际 KV 为 1638）。
- **`seat_row_slack = 0.7244`**（时间加权）：平均只有约 27.6% 的行宽写了字。
  这是「按 trace 最坏长度预留」的直接读数，也是 C1 想消除的浪费。
- `capacity_per_gib = 5.62`；`bytes_per_active_request = 191.07 MB`（约 182.2 MiB）。
- `trace_sha256 = 2c047181...`、`prompt_sha256 = 7548f3a6...`，与 `W4-0003` 完全相同。
- step 账：先准入 9 条、再 9 条、再 9 条、最后 5 条；`peak = 9`，`out_tokens = 0`
  （fill-only 不 decode，正常）。

## 结论（含“什么时候不划算”）

- seat 在 1640 MiB 总预算下只能装 **9** 条并发；其中约 **72.4%** 的行宽是空的。
  这就是 C1 的基线，也是分页收益的来源。
- 该 arm 不判 pass/fail；它只回答「同预算下 seat 的 allocator capacity 是多少」。

## 限制与偏差

- `warmup=1, repeats=3`：fill-only 是确定性准入，不需要默认的 3/5；3 次重复的
  `peak_concurrency` 都是 9。
- fill-only 不 decode、不 gather；因此没有 `frag_internal` / `reservation_slack`，
  `seat_row_slack` 是 seat 自己的对照口径。
- 设备：单张 RTX 4090 24 GB，bf16，`device_class=measure-cuda`，commit `9dc5211`，clean。

## 下一步

- `W4-0003`：同 trace、同预算上限，paged 800 块；C1 由 `bench/thresholds.py` 对它求值。
