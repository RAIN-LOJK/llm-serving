# W3 · 连续批处理（开工方案）

> 前置决策：[D3](decisions.md)（注意力口径）。基线：`W2-0003`（`W2-0002` 已被它取代）。
> 本文件是“新开一个对话就能直接开工”的自包含方案；阶段目标与判据仍以 `ROADMAP.md` §5 W3 为准。

## 目标与判据

- 高 λ 吞吐 ≥ 1.3× 静态批（`configs/thresholds.toml` → `[W3.continuous].throughput_speedup_min`）；
- p99 TTFT 能分解为“排队 + 计算”；
- 不同到达顺序下输出一致；
- 到达过程（`poisson` λ）在 W3 接线——`bench/loadgen.poisson_arrivals` 已留接口，
  `bench/run.py` 目前对 `arrival="poisson"` 直接报错，要一起改。

## 不可改的约束（来自 D3）

1. **长度方向禁止 pad**；批维度可分桶（为将来的 CUDA graph）。
2. **线性层 / MLP 按批执行，attention 逐序列**（每条 `(1, T)`、无 mask、`is_causal`）。
3. 只有一条注意力实现；后端逐 run 记录。

违反第 1 条的典型写法是“把一批请求左 pad 成 `(B, T_max)` 一次 forward”——那样会退回 math 后端
（实测 6.3×），和 `W2-0003` 的基线不可比。

## 要新增/改动的模块

| 文件 | 改动 |
|---|---|
| `model/minillm.py` | `forward` 接受 `(B, T, H)` 的批输入；KV cache 支持逐序列 |
| `serve/engine.py` | `step(requests)`：一次前向 = B 条序列各一个 token |
| `serve/scheduler.py` | 准入/退出、per-request 状态机（新增） |
| `bench/engines/minillm.py` | `impl="continuous"` 走 scheduler；保留 `minillm` 作为单请求臂 |
| `bench/loadgen.py` | `poisson_arrivals` 接线（λ、seed 可复现） |
| `bench/run.py` | 放行 `arrival="poisson"`，加 `[workload].rate` |
| `configs/experiments/W3/` | 静态批与连续批两条配对 run |

## 必须先写的测试（红 → 绿）

1. **批 forward 等价性**：B 条序列一次 `step()` == 逐条 `generate()` 的同一步输出（fp32 `atol=1e-5`）；
2. **调度等价性**：同 workload、不同到达顺序 → 逐 token 相同输出；
3. **逐序列开销**：B=1/8/32 的单步时间拆分，attention 占比 **< 15%**（D3 的判据）；
4. **静态批 vs 连续批**：同 B、同总 token 数下的 token/s 比值。

## 测量清单

- 单步拆分：host / GEMM / attention / KV 读取（`torch.profiler` 或两段计时差值）；
- 吞吐、p99 TTFT（排队 + 计算）、TPOT；
- B ∈ {1, 8, 32} × ctx ∈ {512, 2048}。

## 已知风险

- **0.6B 的 host 开销占 86%**（TPOT 15.1 ms 中约 13 ms 是 Python/launch）。批处理能否摊薄它，
  就是 W3 的核心结论；摊不薄就如实记负结果（ROADMAP §8 已预设“W3 收益 <1.2× 不硬调”）。
- 若 B=32 时逐序列占比超过 20%，**不要换 varlen kernel**：等 W4 的固定地址 KV 后上 CUDA graph。
- 任何为了“先跑通”而引入的长度 pad，都会让结论作废。

## 开工顺序

1. 批 forward 等价性测试（先红）→ 改 `model/minillm.py` 接受批输入；
2. scheduler 最小版（FCFS 准入/退出），只支持 decode 批；
3. runner 接线 + 静态批对照臂（同一注意力路径）；
4. `W3-9001` 冒烟（synthetic、小 B）→ 正式 run。

## 不在 W3 做

- chunked prefill（掩码路径已支持 `past_len>0, T>1`，但不接入调度）；
- 分页 KV（W4）、CUDA graph（W4 之后）、varlen kernel（D3 已排除）；
- 任何形式的长度方向 pad。
