# W3-0013 · E2 调度收益：continuous / B=32 / mixed

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0013-continuous-b32-mixed`）
- 类型：`scenario`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `continuous_beats_static_only_with_length_heterogeneity`

## 观察

- mixed、B=32、λ=28.0；`tok_s=718.56`、`req_s=13.400`。
- 与 `W3-0012` 同 trace `7d8cf5ec…`；**比值 718.56/447.54 = 1.606×**。
- `saturated=true`、`qd_last=46`；TTFT p99 2.33 s（static 8.24 s）。
- `out_tokens` 逐请求 == budget；trace `tok_s` CV 0.27%。

## 结论（含“什么时候不划算”）

建议：**record_only**，不判 C1。连续批在 B=32 仍赢（1.606×），但低于 B=8 的 2.01×；
P4 在 B=32 的 1.8–2.2× 区间未命中，属于 scale 效应：静态 B=32 本身已很高效，
128 条 trace 的波次太少。不是 C1 失败。

## 限制与偏差

- 只有 B=32 / 128 requests；没有 B=32 × ctx2048。
- 饱和排队 CV 警告；不是 R5。

## 下一步

W4 分页 KV（B=32 × ctx2048 留给 W4）。
