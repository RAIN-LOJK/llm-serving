# 混合注意力（hybrid attention）

> 状态：**W5–W7 已完成（2026-11-14）**。W5/W6 是数值门；W7 的数字来自 `results/W7/`
> 的 bf16/CUDA 4090 run，D9 已 accepted，`[W7.state]/[W7.hybrid]` 已 calibrated。

## 1. 目标

Qwen3.5 的 3:1 混合注意力：75% 层是 Gated DeltaNet（线性注意力），25% 是全注意力。
serving 侧要同时管理：

- 分页 KV：只覆盖全注意力层；
- 线性状态池：每请求常数大小，覆盖线性层；
- 统一句柄与生命周期（`serve/state_pool.py`）。

## 2. 实现清单

| # | 算子 | 验收 | 状态 |
|---|---|---|---|
| 1 | 全注意力层（GQA, head_dim 256, 输出门控） | 单层对拍 HF | ✅ 2026-11-09 |
| 2 | partial-RoPE（1/4 维）+ M-RoPE（文本退化 1D） | 位置编码单独测 | ✅ 2026-11-09 |
| 3 | 短卷积（kernel 4, causal, q/k/v） | 因果性：改第 t token 不影响 t−1 | ✅ 2026-11-14 |
| 4 | Gated DeltaNet decode 形式 | 逐步对拍 | ✅ 2026-11-14 |
| 5 | Gated DeltaNet prefill 分块形式 | 分块 prefill == 逐步 decode | ✅ 2026-11-14 |
| 6 | MTP 头 | 跳过，加载时忽略 | N/A |

**W5 实测（2026-11-09）**：

- golden `tests/golden/qwen35-0-8b/0001`：L3/L23 × s01/m01，9 个 stage
  （`input_layernorm` / `q_proj` / `q_norm` / `k_norm` / `gate` / `q_rope` / `k_rope` /
  `attn_gated` / `layer_out`）逐项 `max|Δ| = 0`（CPU fp32；head_dim 256 的 softmax
  scale = 1/16 是 2 的幂，SDPA 与 eager 的缩放顺序不引入舍入差）。
- 因果性：同形状替换第 t 个 token，位置 < t 的输出 `torch.equal`；增量 decode
  （`KVcache`）与整段 prefill 一致（fp32 atol 1e-5）。
- 两种 norm 分开实现：`Qwen3_5RMSNorm` 的 `(1+w)`（zeros 初始化）与
  `Qwen3_5RMSNormGated` 的直接 `w`（ones 初始化）；gate 按**头**切分，sigmoid 在
  attention 之后、`o_proj` 之前。
- 阈值 `configs/thresholds.toml [W5.layer]` 预注册值未改（全 0，无放宽）。
- W7 已接：packed prefill / `KvSlab` / `PagedKV` / `StatePool`；见第 4 节。

**W6 实测（2026-11-14）**：

- golden `tests/golden/qwen35-0-8b/0002`（`spec=w6-gdn-v1`，7.16 MB）：L0/L2 × s01/m02
  的 56 行 stage 表全过；投影/卷积 `max|Δ|=0`，`core_attn_out`/`norm_out` ≤3.6e-7，
  `layer_out` ≤3.0e-7，`final_state` ≤6.0e-7，`final_conv_state=0`。
- 三算子对拍：0.8B 生产几何（T=240、H=16、K=V=128）下 chunk vs HF chunk 的输出与终态
  均 **0.0**（bit-identical），recurrent vs HF `1.5e-8`，fla naive `≈4.5e-8`，
  chunk vs recurrent `2.6e-7`；`chunk_size=64`，`solve_triangular` 是唯一生产路径。
- 全模型：24 层（18 GDN + 6 全注意力）加载成功；逐层 hidden 最差 `2.861e-6`，
  末位 logits `1.061e-5`/`1.121e-5`，贪心 **64/64**（s01/m02），
  prefill-vs-decode `3.040e-5`。
- 每请求状态（0.8B，fp32）：18 个线性层 `conv_state 6144×3` + `recurrent 16×128×128`
  = **19.27 MiB**；6 个全注意力层 KV = **24 KiB/token**（7 token 时 168 KiB）。
  `Qwen35State.linear_state_bytes()/kv_bytes()/nbytes()` 把这两个数锁在单测里。
- `[W6.gdn]` 预注册值未改，`status=calibrated`；快速门禁 **874 passed**，
  W5 的 9-stage 0 与 0.6B 64/64 不变。
- **未做**：packed/varlen GDN prefill、`StatePool`、`KvSlab`/`PagedKV` 接线由 W7 完成；
  W6 本身没有 `results/W6` run。

## 3. 自洽测试

**分块 prefill 与逐步 decode 必须给出同样的 logits。** 这比对齐 HF 更重要；
一旦通过，数值正确性基本锁住。W6 的层级测试在 `test_align_qwen35_gdn.py`（L0/L2 stage），
全模型级在 `test_align_qwen35_model.py`（贪心 64/64 + prefill-vs-decode 3.040e-5），
均标记 `slow`/`hf`。

## 4. W7 实测结论（2026-11-14，bf16 / CUDA 4090）

**口径**：状态预算 1640 MiB（KV + 线性状态 + workspace，权重不计）；容量按
`prompt + 128` 预留、P1/交叉点按 prompt 的 prefill 状态；E3 的 TPOT 用 P-60 的
**每请求 decode step 累加**口径，`W7-0026…0029` 是修正后的正式 run。旧
`W7-0010/0011/0018/0019`（wall-clock TPOT）和 `W7-0013`（只到 4 行）均为
`invalid`，不作证据。

### 4.1 状态、P1–P3

| 项 | 实测 | 判定 / 证据 |
|---|---|---|
| 每序列状态 vs 解析式 | 512/2k/8k 混合 **24.63 / 42.63 / 114.63 MiB**；解析式 24.84 / 42.84 / 114.84；误差 **0.83% / 0.48% / 0.18%** | pass（<20%）；`W7-0004/0006/0008` |
| 纯注意力分母 | **56 / 224 / 896 MiB** | `W7-0005/0007/0009` |
| **P1** | 8k 每序列纯/混合 **7.816×**（≥5.0） | **true**；`W7-0008/0009` |
| **P2-memory** | 512 混合 24.63 vs 纯 56.0 MiB，混合反而小 **2.27×** | 原判“混合更大”**false**，与 W0 的 2.25× 一致；`W7-0004/0005` |
| **P2-latency** | 512 B=1：TTFT **3.89×**、TPOT **1.285×**；B=8：TTFT **8.78×**、TPOT **1.283×**（均 ≥1.10） | **true**：两个 batch 下混合都更贵；修正后 B=8 不再出现 TPOT 反转。`W7-0026/0027/0028/0029` |
| **P3** | B=1：混合 TPOT(8k)/TPOT(512)=**1.010**（≤1.10），纯=**1.100**（<1.15）；B=8：混合 **1.000**、纯 **1.043** | 混合平坦通过；但纯也没有 15% 增长，按预案判 **不可归因**（不是 P3 true）。`W7-0026…0029` |

W0 的历史 P3 是纯 0.6B **+20.7%**、混合 −3.0%；W7 的 bf16/paged 纯臂 B=1 只有
**+10.0%**、B=8 只有 **+4.3%**，因此“混合更平坦”不能单独归因于 18 层线性状态。
P2/P3 的此前“口径依赖/污染”结论来自旧 wall-clock TPOT，已被 P-60 修复和
`W7-0026…0029` 取代。

### 4.2 交叉点与容量

| 项 | 值 | 说明 |
|---|---|---|
| 内存交叉点（prefill） | **190.8 prompt token**（实测区间 [128,192]） | 128：混合 20.13 vs 纯 14.0；192：20.88 vs 21.0；线性插值 190.8；256：21.63 vs 28.0；512：24.63 vs 56.0。W0 用 18.84 得 192.9。`W7-0020…0025` |
| 预留口径交叉点 | **62.8 token** | `190.8 − 128` 的解析值；容量报告口径 |
| 512 容量 | 62 vs 23 = **2.70×** | `W7-0004/0005`，两臂 ≤1640 |
| 2k 容量 | 37 vs 6 = **6.17×** | `W7-0006/0007` |
| 8k 容量 | 14 vs 1 = **14.0×** | `W7-0008/0009`；整数 floor 放大，P1 的 7.82× 是每序列口径 |
| 吞吐交叉点 | **未观察到** | mixed workload 的端到端 tok/s 受调度/队列影响，不能当架构交叉 |

### 4.3 E4 机制迁移（record-only）

| 机制 | hybrid 实测 | 纯臂历史 | 说明 |
|---|---|---|---|
| 连续批处理 vs static | **1.498×**（111.76 / 74.59 tok/s，Poisson B=8） | W45-0010/0011 = 1.863× | 收益下降但方向不变；`W7-0014/0015` |
| paged vs seat 容量 | **48 / 37 = 1.297×**，state 1638.75 / 1578.28 MiB | W4-0002/0003 = 3.111× | paged 仍是生产路径，但线性状态摊薄了 KV 分页的容量优势；`W7-0016/0017` |

**结论边界**：这些是机制迁移的 record-only 数字，不是通过/失败线；纯臂是历史同机、
不同 trace，只作方向参照。E4b 的 seat 是 baseline-only。

### 4.4 正确性与未决

- G0：0.8B/0.6B bf16 与 HF 同 dtype 贪心均 **64/64**；E0b 的 8 条 prompt 在 B=1 与
  B=8 的 token hash **8/8 全等**（`W7-0001/0002`）。
- E1 三点误差均 <1%，`state_used_mb` 来自池张量实际字节，不是解析式反算。
- 内存交叉点已由 `W7-0020…0025` 的 128/192/256/512 实测 bracket 到 [128,192]，
  线性插值 190.8 ≤256；`memory_crossover_ctx_max` evaluator 仍 pending，只是因为当前
  run summary 没有该复合字段，Step 6 再决定接线。
- **未决**：`W7-0013` 的 B=8 请求数口径错误；`W7-0010/0011/0018/0019` 的 TPOT
  归因错误（P-60），两组旧 run 都只保留为 invalid 证据；正式 E3 证据是
  `W7-0026…0029`。

## 5. 风险与止损

- 分块形式的 WY 表示是最难的一段；W6 已用 `solve_triangular` 走通，并与循环形式、
  HF torch fallback 互拍，**没有**回退到“prefill 走逐步循环”。
- packed/varlen GDN prefill 与跨调用 chunked prefill 留 W7；W6 的 `past_len>0` 且 T>1
  显式 raise，不静默走慢路径。
- 若 W0 的 D1 判定不投入移植，本文件只保留 HF 侧对照分析。
