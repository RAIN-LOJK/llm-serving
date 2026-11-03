# W3-0019 · E1 批处理摊薄 host：ctx511 / B=32（P2，ctx512 EOS 后的替代）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0019-batch-b32-ctx511`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=128`（160/160 行）；127 step/trace，`active==32` 630 行。
- `tok_s=1177.64`、`decode_only_tok_s=1454.12`、`step_ms=27.43`、`decode_event=22.23`、
  `host_gap=0.124`、`prefill_ms=5.071`（均值）。
- **P2 = 1454.12 / 384.74 = 3.78×**（高于预注册 2.5–3×）；`step_ms` B32/B8 = 1.22×，仍 <3×。
- 有 `ttft_ms CV=0.559` 警告；trace `tok_s` CV 0.59%。

## 结论（含“什么时候不划算”）

建议：**record_only**（E1）。B=32 仍近乎线性摊薄（3.78×），说明 ctx511 下 KV 读取
还不是限制项；P2 的高侧 miss 应记为“host 仍主导，slab 读取的代价要到更长 ctx 才显形”，
这正是 W4 分页要回答的问题。

## 限制与偏差

- 3.78× 高于预注册区间，但不是负结果；不要为了贴合 2.5–3× 调数据。
- B=32 × ctx2048 未测（KV 预留 ≈7.6 GiB），留给 W4。

## 下一步

E1 图；之后 E2 与 W4。
