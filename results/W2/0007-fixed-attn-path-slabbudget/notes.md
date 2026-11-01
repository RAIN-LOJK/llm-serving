# W2-0007 · 固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab（逐层分配、随请求释放）

- 状态：`pass`（reviewer: rainhome, 2026-11-01）
- 配置：`config.toml`（运行目录 `results/W2/0007-fixed-attn-path-slabbudget`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `one_attention_path_does_not_move_w2_numbers`
- 配对基线 `W2-0005`（HF 臂，同机；HF 引擎本 run 未改动，分母不变）。
- 取代 `W2-0006`：代码变了（`d8361d4`，slab 逐层分配 + 随请求释放），TTFT 因此移动，
  旧基线不再描述 shipped 代码。

## 观察

- 160 行 / 0 错误 / 0 warnings。
- TTFT **21.85** ms（HF 23.46）、TPOT **21.27** ms（HF 21.43）、e2e 2723.9 ms、
  peak 1196 MB、driver 1727 MB。
- 判据 4/4 pass：`ttft_cv_max` 1.12%（short）、`tpot_cv_max` 0.44%（medium）、
  `memory_match_ratio` **0.992**、`decode_speed_vs_hf_min` **1.013**。
- 按长度桶（每桶取各 prompt 中位数的中位数）：

  | 桶 | TTFT 我们 / HF | TPOT 我们 / HF |
  |---|---|---|
  | short | 21.97 / 23.85 = **0.921** | 21.30 / 21.60 = 0.986 |
  | medium | 21.56 / 23.34 = **0.923** | 21.09 / 21.45 = 0.984 |
  | long | 22.01 / 23.13 = **0.952** | 21.82 / 21.34 = 1.023 |

- **显存随 ctx 的斜率消失**：`memory_match_ratio` 最差 4 格为 0.979（ctx 1457/1517/1336/1338），
  最好 0.992（ctx 19/21/25），全区间 0.979–0.992。
  `W2-0006` 同口径下是 0.859–0.987（ctx 1517 → 0.859）。

## 结论（含“什么时候不划算”）

- 假设成立，且比 `W2-0006` 更干净：decode 与 HF 臂持平（1.013×，逐桶 0.984–1.023），
  TTFT 三个桶都比 HF 快 5–8%，显存账在整个上下文范围里是平的。
- **两处改动必须一起做**（ctx 1517 的 2×2，走真实 `generate()`）：

  | | slab 保留到下次请求 | 请求结束释放 slab |
  |---|---|---|
  | 整块分配 | peak 1534.6 / ratio 0.859 | peak 1498.9 / ratio 0.880 |
  | **逐层分配** | peak 1532.1 / ratio 0.861 | **peak 1348.0 / ratio 0.978** |

  逐层交错单独用没有收益：slab 若留着，下次 prefill 开始时它已完整驻留，交错无从发生。
  只释放也不够：整块目标与整份源在记峰值那一刻同时在场。
- **代价**：每个请求重建 28 层 slab，TTFT 21.25 → 21.85 ms（+2.8%），TPOT 不动。
  换来最差 context 0.859 → 0.979。这是测量协议的取舍，不是引擎的最终形态。
- **什么时候不划算**：真实 serving 的 KV 必须跨请求复用（每个并发槽一块），
  本 run 的「随请求释放」是为让单请求峰值描述该请求本身；W3 会改回复用并显式记账，
  届时 `memory_match_ratio` 这个口径要重新定义（分子的预留会覆盖所有活跃槽）。

## 限制与偏差

- 设备 / dtype / 样本量：RTX 4090 / bf16 / 32 prompt × 5 次 / `max_new_tokens=128`；
  与 `W2-0005` 同机同会话，`device_class=measure-cuda`，配对有效。
- 本 run 与 `W2-0005` 之间隔若干 commit（配置与修复），但两次 run 期间 host 无其他负载，
  HF 引擎未改动，分母可比。
- V4（与更早 run 比 token 序列）仍无法从落盘证据复核：`metrics.jsonl` 只存 `out_tokens`
  （本 run 160/160 均为 128）。跨 kernel 的数值门由 `test_align_qwen3` 承担。
- `peak_mem_mb` 的 overall 中位数被短上下文主导，长上下文看 `by_context`。

## 下一步

- `W3-0001` 的 `[baseline]` 挂本 run。W3 改回「每并发槽一块 slab」时，
  要同时定义新的显存口径：分子必须覆盖所有活跃槽的预留，否则会重演 `W2-0004` 的错配。
