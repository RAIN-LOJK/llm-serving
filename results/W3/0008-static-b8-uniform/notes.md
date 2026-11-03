# W3-0008 · E2 调度收益：static / B=8 / uniform

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0008-static-b8-uniform`）
- 类型：`scenario`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `continuous_beats_static_only_with_length_heterogeneity`

## 观察

- uniform `[128]`、B=8、λ=6.0；`tok_s=341.48`、`req_s=2.668`、`offered=6.0`。
- `saturated=true`、`qd_last=72`；TTFT p99 25.00 s、queue p99 24.98 s、prefill p99 31.75 ms；
  `queue_share_p99=0.9992`。
- `out_tokens` 640/640 == 128；trace `tok_s` CV 0.11%。

## 结论（含“什么时候不划算”）

建议：**record_only**。均匀长度下静态批的基线；与 `W3-0009` 一起验证 P3（≈1.0×）。
饱和排队导致 per-prompt TTFT/e2e CV 警告，属开放环预期。

## 限制与偏差

- 只有一种 budget（128），不能代表真实长度分布；`requests=128`、5 repeats。
- 饱和队列的 per-prompt CV 不是 R5；R5 看 trace `tok_s` CV。

## 下一步

与 `W3-0009` 配对算 P3。
