# W3-0017 · E1 批处理摊薄 host：ctx511 / B=1 参照（ctx512 EOS 后的替代）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0017-batch-b1-ctx511`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=128`（5/5 行）；127 step/trace，`active==1` 630 行。
- `tok_s=47.83`、`decode_only_tok_s=48.46`、`step_ms=21.06`、`decode_event=20.80`、
  `host_gap=0.106`、`prefill_ms=0.161`（均值）。
- 无 CV warning；trace `tok_s` CV 约 0.1%。

## 结论（含“什么时候不划算”）

建议：**record_only**（E1 短上下文基线）。ctx511 是 512 的 -1 token 替代（0.2%），
用来绕开 ctx512 synthetic prompt 的 EOS；这条 prompt 稳定生成 128 token。

## 限制与偏差

- 与计划里的 ctx512 差 1 token；远小于 trace 间 CV，不影响 P1/P2。
- synthetic 输出是退化换行串，只用于长度/计时。

## 下一步

与 `W3-0018` 配对算 P1。
