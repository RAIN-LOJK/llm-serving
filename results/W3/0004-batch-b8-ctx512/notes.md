# W3-0004 · E1 批处理摊薄 host：ctx512 / B=8（P1）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0004-batch-b8-ctx512`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=3`（40/40 行），同一 token 序列；ctx512 synthetic prompt EOS。
- 每 trace 2 个 step；`active==8` 行 `step_ms=184.73`、`prefill_ms=162.50`、`decode_event=22.04`。
- P1 字面 7.67× 只是 8×B 的算术镜像；`active=0` 空转步使 `decode_only_tok_s` 偏差约 50%。

## 结论（含“什么时候不划算”）

建议：**invalid**（不用于 E1）。与 `W3-0003` 同因（P-31），由 `W3-0018`（ctx511/B=8）取代。

## 限制与偏差

- 3 token/trace；R1 字面 4.46× 是 prefill 随 B 放大，不是 host 未摊薄。
- TTFT/TPOT CV 警告（0.508/0.402）也是 3-token 数据造成的。

## 下一步

由 `W3-0018` 取代。
