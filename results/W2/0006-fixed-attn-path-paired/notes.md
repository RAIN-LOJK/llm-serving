# W2-0006 · 固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab（本机配对）

- 状态：`pass`（reviewer: rainhome, 2026-11-01）
- 配置：`config.toml`（运行目录 `results/W2/0006-fixed-attn-path-paired`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `one_attention_path_does_not_move_w2_numbers`
- 配对基线 `W2-0005`（HF 臂，同机同会话重测；`W2-0001` 是另一台 4090 实例，已退役）。

## 观察

- 160 行 / 0 错误 / 0 warnings。
- TTFT **21.25** ms（HF `W2-0005` 23.46），TPOT **21.43** ms（HF 21.43），e2e 2743 ms，
  peak 1238 MB，driver 1801 MB。
- 判据 4/4 pass：`ttft_cv_max` 1.22%（short 桶）、`tpot_cv_max` 0.42%（long 桶）、
  `memory_match_ratio` **0.957**（最差 context 0.859）、`decode_speed_vs_hf_min` **1.001**。
- 逐 prompt 配对（32 条）：TPOT 比值 **0.985–1.031**，中位数 1.001——不是中位数抵消，
  是每一条都在 ±3% 内。TTFT 比值 1.072–1.138，每条都优于 HF。
- 按长度桶（每桶取各 prompt 中位数的中位数）：

  | 桶 | TTFT 我们 / HF | TPOT 我们 / HF |
  |---|---|---|
  | short | 21.32 / 23.85 = **0.894** | 21.38 / 21.60 = 0.990 |
  | medium | 21.29 / 23.34 = **0.912** | 21.50 / 21.45 = 1.002 |
  | long | 20.93 / 23.13 = **0.905** | 21.44 / 21.34 = 1.005 |

- 数值门（V5）：`MINILLM_RUN_HF=1 pytest tests/integration/test_align_qwen3.py` → 4 passed。
- 与 `W2-0004` 的差：同一台机器上 TPOT 21.30 → 21.43 ms，只差 0.6%；
  唯一改动是单一 slab（换桶即重建，commit `ce0af17`）。

## 结论（含“什么时候不划算”）

- **单一注意力路径不移动 W2 的数字。** 假设成立：decode 与 HF 臂逐 prompt 持平（1.001×），
  TTFT 三个桶都比 HF 快 9–11%。`W2-0003` 的 decode 1.113× 不能外推——那次是另一台实例，
  本机的 HF 臂自己就慢 1.28×（16.78 → 21.43 ms），两个 host-bound 数字跨实例不可比。
- **本 run 取代 `W2-0003` 成为 W3 的配对基线。** 理由不是"数字更漂亮"，是口径：
  `W2-0003` 是 sdpa/cat，与 D4 定死的 hand_baddbmm/slab 不是同一个 kernel，
  机制键天然不相等；本 run 与后续 W3 各臂共享同一组常量 provenance。
- **什么时候不划算**：本次没有观察到交叉点。B=1 下"单一代码路径的税"（`w2-rework` §4.1
  探针估的 kernel 4–6% + slab 记账 4–13%）在配对后没有显形——因为那 1.09–1.17× 是对
  `W2-0003` 的 cat 路径测的，而 cat 路径在本进程里本身也要 21.24 ms/step。
  这个税的绝对值仍在，只是被 host 侧的共同成本淹没了；**B ≥ 2 时 slab 的收益要重测**，
  这正是 W3 的事。

## 限制与偏差

- 设备 / dtype / 样本量：RTX 4090 / bf16 / 32 prompt × 5 次 / `max_new_tokens=128`。
  `W2-0005` 与 `W2-0006` 同机、同会话、两次 run 之间隔一个 commit（`results/INDEX.md`
  被上一次 run 改写，第二次启动需要干净树），不是严格背靠背；host 侧没有其他负载介入。
- **显存账在长上下文仍然偏低，原因已定位**：`generate()` 里 prefill 的 `KVcache`
  在交接（`fill_from`）之后不再被读——decode 只读 slab——但它的引用留到了函数返回。
  于是这次请求的**峰值**落在「slab 已整块分配、`KVcache` 尚未释放」的那一刻，
  同一份 KV 在峰值里存在两份：slab（预留桶）+ 原 `KVcache`（实际长度）。
  这不是泄漏：`generate()` 返回时引用计数归零、立即释放；范围只在单个请求内部，
  也**不是**"请求结束了没回收"。但 `memory_match_ratio` 的分子只有 `weights + slab`，
  分母却是含两份 KV 的峰值，所以比值随 ctx 单调下滑：

  | ctx | peak | weights | slab | prefill 的 `KVcache` | 残差 | ratio |
  |---|---|---|---|---|---|---|
  | 441 | 1272.3 | 1136.9 | 63.0 | 48.2 | 24.2 | 0.943 |
  | 987 | 1405.6 | 1136.9 | 126.0 | 108.0 | 34.8 | 0.898 |
  | 1517 | 1534.9 | 1136.9 | 182.0 | 165.9 | 50.1 | 0.859 |

  `prefill 的 KVcache` 按 `28 层 × 2 × 8 kv_head × ctx × 128 × 2 B` 算，与实测吻合；
  残差 24–50 MB 是 prefill 的 `T²` 激活工作区（本来就记不进公式账）。

- **交接后释放原件降的是驻留，不是峰值（实测，ctx 1517，每次新进程）**：

  | 交接方式 | prefill 自身峰值 | 整个请求峰值 | decode 期间驻留 |
  |---|---|---|---|
  | 不动（当前） | 1389.0 | **1539.9** | 1498.2 |
  | 循环后 `del cache` | 1389.0 | **1539.9** | 1332.3 |
  | 每层拷完即 `cache.kvs[l] = None` | 1389.0 | **1539.9** | 1332.0 |

  三种顺序的峰值完全相同：峰值在 slab 整块分配那一刻就已定下（分配后 alloc 1497.9、
  peak 1539.9），之后怎么释放都追不回来。**所以「释放原件就能把 ratio 拉回 0.95」
  是错的**——它只让 decode 期间少占 166 MB，`peak_mem_mb` 不动。
  要真正降峰值，得让 slab 在交接过程中**逐层惰性分配**、并同步释放对应层的 `KVcache`；
  这条没测过，W3 若要做再验证，不要当成已知收益。
- V4（逐 prompt 输出与 `W2-0003` 比 token 序列）**无法从落盘证据复核**：`metrics.jsonl`
  不存 token id，只存 `out_tokens`（两臂均为 128/128）。跨 kernel 的数值门由 V5 的
  golden 对齐承担，不靠这条。
- `peak_mem_mb` 的 overall 中位数被短上下文主导，长上下文必须看 `by_context`。

## 下一步

- W3 以本 run 为配对基线（`W3-0001` 的 `[baseline]`）；机制键 `attn_prefill=sdpa`、
  `attn_decode=hand_baddbmm` 由常量 provenance 给出，W3 各臂必须相等。
- W3 接线时逐层释放交接后的 `KVcache`（降低驻留，不是降峰值——见上表），并在 B ≥ 2 时
  重测 slab 相对 cat 的收益：B=1 下它的正当性只是"少一次 cat 拷贝"，没有数据支撑。
