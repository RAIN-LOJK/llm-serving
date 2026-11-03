# W3-0006 · E1 长上下文参照：ctx2048 / B=1

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0006-batch-b1-ctx2048`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=128`（5/5 行）；127 step/trace，`active==1` 共 630 行。
- `tok_s=45.27`、`decode_only_tok_s=45.93`、`step_ms=22.02`、`decode_event=21.70`、
  `host_gap=0.114`、`prefill_ms=0.198`（均值）。
- 无 CV warning；trace `tok_s` CV 1.72%。

## 结论（含“什么时候不划算”）

建议：**record_only**（E1 基线）。ctx2048 的单请求 decode 成本 ≈21.8 ms/token，
是 `W3-0007` 的 B=1 对照。

## 限制与偏差

- 只有 B=1；synthetic 输出是 128 个换行，内容无意义，只用于长度/计时。
- B=32 × ctx2048 不跑（KV 预留 ≈7.6 GiB），留给 W4。

## 下一步

与 `W3-0007` 一起算 P1（B=1→8）。
