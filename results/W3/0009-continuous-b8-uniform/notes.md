# W3-0009 · E2 调度收益：continuous / B=8 / uniform

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0009-continuous-b8-uniform`）
- 类型：`scenario`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `continuous_beats_static_only_with_length_heterogeneity`

## 观察

- uniform `[128]`、B=8、λ=6.0；`tok_s=349.37`、`req_s=2.730`。
- `saturated=true`、`qd_last=64`；TTFT p99 23.44 s、queue p99 23.42 s；`queue_share_p99=0.9991`。
- 与 `W3-0008` 同 trace `918b8c52…`；**比值 349.37/341.48 = 1.023×**。

## 结论（含“什么时候不划算”）

建议：**record_only**；P3 负结果成立——均匀长度下连续批只比静态批快 2.3%，
差异来自波次取整与少量回填，不是调度机制的大收益。C1 不在这一对判。

## 限制与偏差

- 1.023× 不进 `acceptance`；不要把它写成“连续批无效”，它只说明**长度异质**是收益前提。
- 饱和排队导致 TTFT CV 警告；trace `tok_s` CV 0.13%。

## 下一步

进入 mixed B=8 对（C1）。
