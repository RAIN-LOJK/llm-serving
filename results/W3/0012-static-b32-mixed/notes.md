# W3-0012 · E2 调度收益：static / B=32 / mixed

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0012-static-b32-mixed`）
- 类型：`scenario`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `continuous_beats_static_only_with_length_heterogeneity`

## 观察

- mixed、B=32、λ=28.0；`tok_s=447.54`、`req_s=8.346`、`offered=28.0`。
- `saturated=true`、`qd_last=73`；TTFT p99 8.24 s、queue p99 8.22 s、prefill p99 24.25 ms。
- `out_tokens` 逐请求 == budget；trace `tok_s` CV 0.23%。

## 结论（含“什么时候不划算”）

建议：**record_only**（B=32 基线）。B=32 的静态批已经比 B=8 静态快 2.9×，
留给连续批的相对回填空间变小。

## 限制与偏差

- 128 条 trace 在 B=32 下只有约 4 波，波次取整放大了比值噪声。
- 饱和排队 CV 警告；不是 R5。

## 下一步

与 `W3-0013` 配对只报比值。
