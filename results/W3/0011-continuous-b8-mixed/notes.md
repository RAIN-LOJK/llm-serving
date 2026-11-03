# W3-0011 · E2 调度收益：continuous / B=8 / mixed

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0011-continuous-b8-mixed`）
- 类型：`scenario`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `continuous_beats_static_only_with_length_heterogeneity`

## 观察

- mixed、B=8、λ=11.0；`tok_s=311.43`、`req_s=5.808`。
- **C1 `throughput_speedup_min = 2.010×`（pass，target 1.30；ours 311.43 / baseline 154.94）**。
- **C2 `ttft_p99_decomposed=pass`**：`decomposed_rate=1.0`，逐请求最大偏差 1.8e-12 ms。
- **C3**：与 `W3-0010` 同 trace `33bc2397…`；request-row `token_sha256` 89/89 keys 0 mismatch。
- `saturated=true`、`qd_last=60`；TTFT p99 8.42 s（static 30.56 s），`queue_share_p99=0.9974`。

## 结论（含“什么时候不划算”）

建议：**pass**（W3 的 C1/C2/C3 判定点）。连续批在长度异质 + 过载下把吞吐提到 2.01×，
p99 TTFT 压到 1/3.6；收益来自短请求释放席位后立即回填。**什么时候不划算**：长度均匀时
（`W3-0009` 1.023×）或未饱和时，收益消失。

## 限制与偏差

- 合成 budgets，qwen3-0.6b 在自然 prompt 上不 EOS；不是真实负载吞吐声明。
- per-prompt TTFT/e2e CV 警告是饱和排队所致；trace `tok_s` CV 0.53%。
- summary 的 `queue_depth_max` / `protocol.policy` 为 null，`consistency` 无 token-hash 字段；
  C3 用 request rows 核对（D5 后果 12）。

## 下一步

B=32 放大对（record_only）；随后 W4 分页 KV。
