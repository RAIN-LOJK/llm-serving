# W4-0006 · E3 同预算系统吞吐（seat）：frozen 32 / poisson / B=9

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W4/0006-seat-saturate`）
- 类型：`scenario`；workload：`prompts`（frozen 32）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `paged_system_throughput_at_equal_budget_beats_seat`
- E3 原计划的可选吞吐臂：同总预算、同 trace、同 Poisson λ=30、128 请求、
  budgets `[16,32,64,128]`；seat `max_batch=9`。两臂都必须 `saturated=true`，否则
  λ 加倍、两臂一起重跑；`tok_s` 只作 record-only，不进 W4 判据。
- **执行口径**：E3 是在 C4 失败后按用户指示补跑的，属于 **post-hoc record-only**；
  它不改变 C4 的负结果，也不把 W4 的结论改成“吞吐提升”。

## 观察

- `throughput.saturated = true`；`tok_s = 333.44`（3 次重复，trace 级 CV **0.117%**）；
  decode-only `386.96 tok/s`、`req_s = 6.218`、makespan `20.585 s`。
- 动态峰值 `peak_concurrency = 9`（= `max_batch`）；最后到达时队列深度 99。
- EOS 检查：384/384 个 request 的 `out_tokens == budget`（无提前 EOS），所以吞吐
  没有被早停污染。
- `memory.kv_mb = 1638.0`（9 行 × 182 MiB），`kv_total = 1638.0`，`seat_unused = 2`。
- 同 paged 臂的 `trace_sha256 = d6391043...`、`prompt_sha256 = 7548f3a6...` 完全相同。
- `summary.warnings` 里大量 TTFT/e2e per-prompt CV > 0.10：这是**开放环饱和**的预期现象
  （docs/experiments §5.1 明确例外），只看 trace 级 `tok_s` CV（0.117%），不重跑。

## 结论（含“什么时候不划算”）

- 这是 E3 的 seat 基线：**333.44 tok/s**（B=9、λ=30、同预算）。
- 本 run 不判 pass/fail；它只回答“同预算下 seat 的系统吞吐是多少”。
- E3 的结论只作 record-only；最终 W4 结论仍由 C1–C4 决定。

## 限制与偏差

- `warmup=1, repeats=3`：E3 的文件头口径（不是默认 3/5），已在 warning 中说明。
- 无 paired baseline：seat 臂本身是基线；`no paired baseline` warning 不适用。
- λ=30 是预注册初值；没有做 λ sweep，不能外推到其他到达率。
- 单张 RTX 4090 24 GB、bf16、`device_class=measure-cuda`，commit `c8cbe53`，clean。

## 下一步

- `W4-0008`（800 块，同预算）是有效的 paged 对比；`W4-0007`（820 块）峰值超预算，
  只保留为 R5 记录。
