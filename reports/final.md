# 单卡 LLM Serving：机制收益边界实测（W0–W8）

> 载体：`Qwen/Qwen3-0.6B`（纯注意力，28 层）与 `Qwen/Qwen3.5-0.8B`（3:1 混合，24 层）。
> 设备：单张 RTX 4090（`measure-cuda`），bf16，同一 `serve/` 调度器。
> 每个数字绑定一次干净树上的 run 或经评审的 golden；目标值与实测值分列。
> 矩阵表由 `scripts/w8_matrix.py` 生成（`reports/matrix.md`），图 `figures/fig_w8_matrix.png`。

## 1. 问题与口径

问的是**边界**，不是“快了多少”：什么规模、什么上下文、什么批次下，连续批处理 / 分页 KV /
packed prefill / 混合注意力双状态才回本；不划算时，代价是什么。

- **两代载体**：先用 0.6B 把 `serve/` 骨架与测量框架跑通（引擎层与架构无关），再移植 0.8B
  混合层——W0 的 P1 实测 7.80×（8k）/8.90×（32k）后，D1 才接受移植。
- **判据先行**：每个机制先写“什么时候不划算”的线，再跑；负结果与正结果同篇幅（§4）。
- **双 TPOT**：service TPOT（wall，含其他请求 prefill）回答体验，decode TPOT（每请求
  `decode_event` 累加）回答架构成本；两者不互相替代（P-60）。
- **矩阵**：2 臂 × 3 ctx × 3 B = 18 格；B=1/8 的 2k/8k 来自 W7 的多 ctx run，B=32 来自 W8
  的单 ctx run。pure 短行用 **511**（512 合成 prompt 第 3 个 token 即 EOS，P-31）。
- 报 p95 不报 p99（n 不够）；不跨设备 / dtype / regime / 引擎路径配对；B=32 行用满池，
  **不按 1640 MiB 判**，容量结论只来自 W7 的 fill run。

## 2. 数值对拍：可信度地基

| 门 | 结果 | 证据 |
|---|---|---|
| 0.6B 贪心逐 token | **64/64** 全等；末层 logits `max|Δ|` 2.0e-5；逐层 9.8e-4–1.95e-3（线 1e-2） | `W1` golden |
| paged vs seat 输出 | 40/40 request 逐 id 相同 | `W4-0004/0005` |
| 0.8B 全模型 | 贪心 **64/64**；逐层 hidden 最差 2.861e-6；logits 1.061e-5 | `W6` golden `0002` |
| 分块 prefill vs 逐步 decode | 3.040e-5；chunk vs recurrent 2.6e-7 | `W6` |
| 服务化哈希 | 0.8B B=1/B=8 **8/8** token hash 全等；G0 64/64 | `W7-0001/0002` |
| packed prefill + Triton decode | E0 **64/64**；E2 8/8 token hash | `W45-0001/0012` |

## 3. 纯注意力：机制各自买到什么

| 机制 | 对照 | 结果 | 判据 |
|---|---|---|---|
| 连续批处理 | static vs continuous，B=8 mixed | 154.94 → **311.43 tok/s（2.01×）** | `W3-0011` |
| 批摊薄 | B=1→8→32，ctx511 | **7.94×**、再 **3.78×** | `W3-0017–0019` |
| 分页 KV | seat 9 vs paged 28，同 1640 MiB | **3.111×**；internal frag **0.55%** | `W4-0002/0003` |
| packed prefill | serial vs packed，B=8 mixed | TTFT p99 **3.760×**、p50 0.512× | `W45-0006/0007` |
| block-table decode | paged vs seat，B=8 | **1.040×**（接受线 <1.15，1.0 目标未达） | `W45-0012` |
| 端到端 | seat 330.53 → paged 678.59 tok/s | **2.053×**（both saturated，record-only） | `W45-0008/0009` |

**W8 矩阵（decode TPOT p50，ms）**：pure 511 行 22.94/23.87/24.28（B=1/8/32），2k 行
22.07/22.89/24.85——**对批次几乎平**；8k 的 decode 因 EOS 不可测（§4），只报状态 910 MiB/序列
与 B=20 上界。

边界：paging 只在**长度异质**时回本（uniform-2048 模拟 **1.00×**）；packed prefill 买的是尾延迟，
不是吞吐；block-table decode 只能把代价压到 ~1.04×，不能归零。

## 4. 负结果：哪些优化在这个规模下无效

| 负结果 | 数字 | 为什么不划算 |
|---|---|---|
| uniform 负载下的连续批 | **1.023×** | 同长度、无排队，批处理没有可摊薄的空洞（P3） |
| 分页 KV 的 decode 代价 | **1.671×**（35.63 vs 21.32 ms） | 每步按 block table 取 KV，读放大；>1.5× 线上**撤回吞吐结论**，只报容量 |
| 分页的系统吞吐（pre-fix） | **0.975×**（325.19 vs 333.44） | prefill 3.39× 把 decode 的 1.103× 拉回负值；post-fix 1.074× 只是 record-only |
| packed decode 的预修版本 | **1.203× fail** → 1.040× | 修复只把代价从 1.20 压到 1.04，**1.0 目标仍未达** |
| 混合架构的连续批收益 | **1.498×**（纯臂 1.863×） | 18 层 GDN 的计算占比高，批内摊薄变少 |
| 混合架构的分页收益 | **1.297×**（纯臂 3.111×） | 分页只覆盖 6/24 层，KV 只占 25% 状态 |
| P2-memory | **false**：512 混合已小 **2.27×** | 18.63 MiB 常数状态不是劣势；短上下文也不贵 |
| P3 归因 | 两口径均**不可归因** | 纯臂只 +10.0%/+4.3%（未达 1.15）；8k 纯臂 EOS，decode 无样本 |
| pure 8k decode | **N/A** | 合成 prompt 第 2 个 token 即 EOS（P-31）；不是模型结论，是负载边界 |
| pure 8k × B=32 | **N/A**：28.1 GiB KV > 24.5 GiB | 结构性不可行；实测上界 B=20（19.25 GiB，`W8-0008`） |

## 5. 混合注意力：KV 只剩 25% 的层之后

- **P1 成立**：8k 每序列状态 **7.816×**；容量比 **2.70/6.17/14.0×**（512/2k/8k，同 1640 MiB）。
- **交叉点 190.8 prompt token**（实测区间 [128,192]）：≈191 token 以下纯注意力更省，以上混合。
- **P2-latency 口径依赖**：service B1 4.14×/1.39×、B8 8.82×/0.49×；decode B1 1.285×、
  B8 1.283×（decode true，service 不可下单一结论）。
- **P3**：B=1 时 TPOT 28.82/28.88/29.12（512/2k/8k），**平坦**；B=8 三 ctx 都是 31.00；
  B=32 才随 ctx 升到 33.50→38.33（+14%）。所以“decode 不随上下文增长”在固定 B 下成立，
  增长来自批内 KV 读取，不是线性层的循环状态。
- **判据**：混合赢在**状态容量**，不赢在单请求延迟（B=1 时 TPOT 28.8 vs pure 22.9）。
  分页在混合上只能拿到 1.297×，因为 KV 只剩 6 层；调度要按**两种资源**分别准入（KV 块 +
  常数状态槽），`StatePool` 与 `BlockManager` 绑定同一生命周期（D9）。
- 对应图：`figures/fig_w7_capacity.png`、`figures/fig_w7_crossover.png`、
  `figures/fig_w8_matrix.png`。

## 6. 复现

```bash
# 快速门禁（无 HF 权重）
python -m pytest -m "not slow and not hf"

# 一个 pure 矩阵格 / 一个 hybrid 矩阵格（正式 run 需干净树，revision 已 pin）
python -m bench.run --config configs/experiments/W8/0006-matrix-pure-511-b32.toml
python -m bench.run --config configs/experiments/W8/0003-matrix-hybrid-8k-b32.toml

# 矩阵表 + 图（只读 summary）
python scripts/w8_matrix.py

# 索引自检
python scripts/exp_index.py --check
```

HF 侧数值门（需 `MINILLM_RUN_HF=1` + 本地权重）：`tests/integration/test_align_qwen3.py`、
`test_align_qwen35_model.py`、`test_paged_reopen.py`。每个 run 目录自带 `reproduce.sh`
（记录提交号，重跑输出到 `results-repro/`）。

## 附：交给 W10 的边界表

| 轴 | W8 给出 | W10 需要补 |
|---|---|---|
| ctx × B | 3×3 实测格点（含 N/A） | 格点间扫点求 `x*` + 区间 |
| 容量 | 1640 MiB 预留口径 | 按需分配（W9b）第二条曲线 |
| decode | Regime A（host-bound） | graph-on 重锚（W9a） |
| H / λ / graph | 未测 | 全部 |
| prefill / TTFT | 已测，W9 不影响 | 直接引用，不重测 |
