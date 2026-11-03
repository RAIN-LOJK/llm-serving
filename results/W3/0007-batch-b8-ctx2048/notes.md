# W3-0007 · E1 长上下文 batch 收益：ctx2048 / B=8

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0007-batch-b8-ctx2048`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=128`（40/40 行）；127 step/trace，`active==8` 共 630 行。
- `tok_s=337.89`、`decode_only_tok_s=363.00`、`step_ms=24.19`、`decode_event=22.52`、
  `host_gap=0.121`、`prefill_ms=1.550`（均值）。
- **P1 = 363.00 / 45.93 = 7.90×**（≥5×）；`step_ms` B8/B1 = 1.10×，R1 未触发。
- 有 `ttft_ms CV=0.503` 警告（饱和排队），trace `tok_s` CV 2.57%。

## 结论（含“什么时候不划算”）

建议：**record_only**（E1）。ctx2048 下批处理把每 token decode 成本从 21.8 ms 摊到 2.76 ms，
7.90× 支持 P1；host 摊销成立，KV 读取尚未成为限制。

## 限制与偏差

- TTFT CV 警告来自串行 prefill 的排队顺序，不是吞吐不稳；不要用 W2 的 0.10 延迟 CV 套它。
- 只有 B=1/8；B=32 × ctx2048 未测。

## 下一步

与 `W3-0017–0019` 一起出 E1 图。
