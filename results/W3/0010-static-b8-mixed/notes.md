# W3-0010 · E2 调度收益：static / B=8 / mixed

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0010-static-b8-mixed`）
- 类型：`scenario`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `continuous_beats_static_only_with_length_heterogeneity`

## 观察

- mixed `{16,32,64,128}`、B=8、λ=11.0；`tok_s=154.94`、`req_s=2.889`、`offered=11.0`。
- `saturated=true`、`qd_last=86`；TTFT p99 30.56 s、queue p99 30.54 s、prefill p99 28.77 ms；
  `queue_share_p99=0.9994`。
- `out_tokens` 逐请求 == budget（分布 16:205 / 32:150 / 64:160 / 128:125）。

## 结论（含“什么时候不划算”）

建议：**record_only**（C1 的基线臂）。静态批在长度异质 + 过载下被波次取整拖住：
每波要等最慢请求，短请求的席位空转。

## 限制与偏差

- 合成 budgets、自然 prompt；不是真实负载吞吐声明。
- 饱和排队导致大量 TTFT/e2e CV 警告；C2 是结构判据，不看这个 CV。

## 下一步

与 `W3-0011` 配对判 C1。
