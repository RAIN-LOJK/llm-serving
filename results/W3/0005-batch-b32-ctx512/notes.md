# W3-0005 · E1 批处理摊薄 host：ctx512 / B=32（P2）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0005-batch-b32-ctx512`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=3`（160/160 行），同一 token 序列；ctx512 synthetic prompt EOS。
- 每 trace 2 个 step；`active==32` 行 `step_ms=675.17`、`prefill_ms=651.70`、`decode_event=22.99`。
- P2 字面 3.85× 只是 4×B 的算术镜像；`decode_only_tok_s` 同样被 `active=0` 步扭曲。

## 结论（含“什么时候不划算”）

建议：**invalid**（不用于 E1）。与 `W3-0003/0004` 同因（P-31），由 `W3-0019`（ctx511/B=32）取代。

## 限制与偏差

- 3 token/trace；R1 字面 4.46× 是 prefill 步，不是 host 失败。
- 该 run 的 `step_ms` 不能与 128-token 的 run 横比。

## 下一步

由 `W3-0019` 取代。
