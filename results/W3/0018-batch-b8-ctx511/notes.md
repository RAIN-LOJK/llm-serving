# W3-0018 · E1 批处理摊薄 host：ctx511 / B=8（P1，ctx512 EOS 后的替代）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0018-batch-b8-ctx511`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=128`（40/40 行）；127 step/trace，`active==8` 630 行。
- `tok_s=361.28`、`decode_only_tok_s=384.74`、`step_ms=22.41`、`decode_event=21.04`、
  `host_gap=0.110`、`prefill_ms=1.261`（均值）。
- **P1 = 384.74 / 48.46 = 7.94×**（≥5×）；`step_ms` B8/B1 = 1.06×，R1 未触发。
- 有 `ttft_ms CV=0.509` 警告（饱和/串行 prefill 排队），trace `tok_s` CV 1.08%。

## 结论（含“什么时候不划算”）

建议：**record_only**（E1）。ctx511 下批处理把每 token decode 成本从 20.7 ms 摊到 2.6 ms，
7.94× 支持 P1；`host_gap` 0.11 ms 基本不动。

## 限制与偏差

- TTFT CV 警告来自 prefill 排队，不是吞吐不稳；不要用 W2 的延迟 CV 套它。
- 只有 B=1/8/32 三点，ctx511 的 KV 读取还没成为限制。

## 下一步

与 `W3-0019` 配对算 P2。
