# W7 计划 · 双状态管理与跨代对照

> **状态**：**active**（2026-11-14 用户批准；**Step 0–3 已完成**；**Step 4 正式 run 已完成**，
> 23 条有效 + 1 条 invalid 修正；Step 5 判定表已填草稿，**等待用户复核**）。
> **权威**：阶段目标与判据以 [ROADMAP.md](../ROADMAP.md) §5 W7、§6 测量协议、§8 失败与裁剪为准；
> 本文件只展开步骤、门禁、证据与生命周期。若本文件与 ROADMAP 冲突，以 ROADMAP 为准，
> 并在 §12 记录修订。
> **生命周期**：本文件是**临时执行计划**。W7 收口、阻塞或放弃时**删除**；不保留“历史 plan”。
> 持久结论只进：`docs/decisions.md` **D9**、`ROADMAP.md` §5/§12/§13、
> `docs/hybrid_attention.md` §4、`PITFALLS.md`、`configs/thresholds.toml [W7.*]`、
> `tests/`、`results/W7/`、`figures/`、`scripts/plot_w7.py`。
> **范围**：`Qwen/Qwen3-0.6B`（纯注意力）与 `Qwen/Qwen3.5-0.8B`（3:1 混合，text decoder，D2）
> 在**同一个 `Scheduler`** 下的 serving；正式数字用 bf16 / CUDA 4090（`measure-cuda`），
> `dev-mps` 只做正确性/冒烟。**E4 机制迁移按用户 2026-11-14 决定进入 W7。**
> **不做**：训练/权重、视觉/MTP、4B 载体、chunked prefill、prefix cache、CUDA graph、
> 抢占/swap、Triton GDN kernel、量化、W8 报告正文。

---

## 0. 一页纸

### 0.1 现状

- W0–W6 已完成。0.6B 纯注意力已有完整的 serving 证据：W3 连续批处理（C1 **2.01×**）、
  W4 分页 KV（C1 **3.111×**、C2 **0.55%**、C4 **1.671×** 负结果）、W4.5 packed prefill +
  Triton block-table decode（C1 **1.040×** 接受线、E3 p99 **3.760×**、E4 **2.053×**）。
- 0.8B 混合模型已完成 **数值层**：W5 的 6 层全注意力 + partial-RoPE + 输出门控，W6 的
  18 层 Gated DeltaNet（短卷积 / decode 循环 / chunked prefill）+ 24 层 `Qwen35Model` +
  `Qwen35State`；golden `tests/golden/qwen35-0-8b/0002` 单层 56 行 stage 表、全模型
  logits、贪心 **64/64**、prefill-vs-decode 均过，快速门禁 **874 passed**。
- **但 0.8B 只有单序列、CPU fp32、数值门**：没有调度器、没有状态池、没有分页/批量路径，
  `bench` 的 `impl="hybrid"` 仍是 `NotImplementedError`。W7 是 W8 报告前最后一个实现阶段。

### 0.2 目标

把 0.8B 从“能算对”推进到“能在同一个调度器下服务”，并把混合架构的机制收益量出来：

1. **双状态管理**：线性层状态（每请求常数大小）与全注意力 KV 块（分页）统一句柄与生命周期；
2. **同一调度器**：0.6B 纯注意力与 0.8B 混合共用 `Scheduler`，不出现模型分支；
3. **packed/varlen GDN 接线**：packed group prefill 中全注意力层走 varlen，GDN 先按
   segment 循环正确写入池状态；真正 fused varlen GDN 不是本期硬门；
4. **跨代对照**：状态占用、容量、交叉点、P1–P3、以及 W3/W4 机制在混合架构下的迁移（E4）。

### 0.3 成功定义（A1–A6，缺一不算 ✅）

| # | 判据 | 硬线 / 证据 |
|---|---|---|
| **A1** | 同一 `Scheduler` 下两种模型都能跑，输出正确 | 0.8B bf16 与 HF 同 dtype 贪心 **64/64**；B=8 与逐请求 token id 全等；0.6B 既有 64/64 不回归 |
| **A2** | 双状态生命周期正确，每序列状态与解析式误差 <20% | 分配/释放/回滚/复用/无泄漏测试；`[W7.state].state_estimate_error_max = 0.20`；used/reserved/slack 分开记账 |
| **A3** | 容量与交叉点有实测 | 同 **1640 MiB 状态预算**，512/2k/8k 容量 ≥**2.0×/4.0×/6.0×**；内存交叉点 ≤**256** prompt token；吞吐交叉点给出或明确“未观察到” |
| **A4** | P1–P3 逐条判真假 | 服务化复核口径（§2.2）；P2 拆 memory/latency；每条 verdict + run id |
| **A5** | E4 机制迁移有数（用户决定进 W7） | hybrid continuous/static、paged/seat 容量均有数；**seat 是 baseline-only，生产仍 paged**；E4 record-only；E4a/E4b 的纯臂引用同机历史 |
| **A6** | W8 交接与生命周期收口 | D9 **accepted**；`[W7.*]` calibrated；ROADMAP/hybrid_attention/PITFALLS/图/索引回填；临时 plan 删除 |

### 0.4 推进方式

**不设日历时间盒。** 以门禁 G0–G6 为推进条件：每一步在测试/run 证据上通过后进入下一步；
不按日期承诺，也不为了“赶时间”跳过门禁。范围裁剪只按 §0.5/§7 的触发条件，不按日期。

### 0.5 止损（与 ROADMAP §8 对齐）

| 触发 | 处置 |
|---|---|
| **G0** bf16 与 HF 同 dtype 对拍不过 | 先修数值（GDN fp32 内部、RoPE/norm、loader），不进入池/调度实现；W7 暂标 `blocked` |
| **G1** 批量 decode 或调度器集成不绿 | 回退“逐请求混合引擎”；W7 只做正确性 + 状态记账 + P1 分析，**不伪造容量结论**，标部分完成 |
| **G2** packed prefill 不绿 | 先修 packed；**不实现逐请求 prefill 控制臂**；修不动则停在 E4a/E4b，不伪造 packed 收益 |
| **G3** 容量/交叉点测不出来 | 按 §5.5 报实测或明确“未观察到”；**不砍 E4a/E4b 的 hybrid/seat 臂**（用户已定 E4 进 W7，E4 不整块删）；纯臂已引用历史；按 §8 写分析并标明未实测 |

> ROADMAP §8 的兜底（只做 P1 + 交叉点、P2/P3 写分析）仍然有效；本计划优先实测，未实测的条目标注，不写 ✅。

---

## 1. 现状核查（2026-11-14，只读）

### 1.1 证据

| 项 | 值 / 命令 | 说明 |
|---|---|---|
| HEAD | `8dc4cd0 docs(w6): close Gated DeltaNet stage (D8 accepted, thresholds calibrated)` | 工作树干净；本 plan 为未跟踪文件 |
| W6 快速门禁 | **874 passed**（W6 收口记录） | 本计划不重跑；W7 改动后按受影响范围重跑 |
| W6 全模型 golden | `tests/golden/qwen35-0-8b/0002`（`spec=w6-gdn-v1`，7.16 MB） | 单层 56 行 stage 表、全模型 logits、贪心 64/64 |
| W5 HF 门禁 | **2 passed**（W6 收口记录） | `MINILLM_RUN_HF=1` |
| W1 0.6B 对齐 | 64/64，logits max abs(Δ) 2.0e-5（逐层 9.8e-4–1.95e-3） | W6 复核未变 |
| run 契约 / prompt 集 | `exp_index.py --check` ✓；`32 prompts, sha256=7548f3a6c7e6…` | W6 记录；W7 不修改 prompt 集 |
| `results/` | W0/W2/W3/W4/W45；**无 W6** | W6 是 golden+pytest 数值门，不是 run |

### 1.2 0.8B 架构事实（`config.get_text_config()` 直读，W6 冻结）

| 字段 | 值 | W7 的影响 |
|---|---|---|
| `num_hidden_layers` / `layer_types` | 24；`LLLF × 6` | 18 线性 + 6 全注意力；全注意力层号 `3,7,11,15,19,23` |
| 全注意力 | `head_dim=256`、`8` heads、`2` KV heads、输出门控、partial-RoPE(1/4) | 复用 W5 层；分页 decode 复用 W4.5 Triton（D=256/GQA） |
| 线性层 | 16 key heads / 16 value heads、`K=V=128`、conv kernel 4 | `conv_state` 只存 **3** 列（D8/P-49）；`conv_dim=6144` |
| 状态 | `conv_state [B,6144,3]` + `recurrent [B,16,128,128]` fp32 | 每请求线性状态 bf16 下 **18.63 MiB**（fp32 19.27 MiB） |
| KV | 6 层 × `2×2×256` × dtype | bf16 **12 KiB/token**；fp32 24 KiB/token |
| 权重 | text-only bf16 ≈ **1435.1 MiB**（W0） | 0.6B 约 **1136.9 MiB**；容量口径不计权重 |
| 其他 | `tie_word_embeddings=true`；MTP 1 层 / vision 12 层忽略 | D2；W7 只测 text decoder |

### 1.3 缺什么

| 缺口 | 位置 | 现状 |
|---|---|---|
| 线性状态池 + 统一句柄 | `serve/state_pool.py` | **不存在**；`Qwen35State` 每请求自己分配 `KVcache` + 18 层状态 |
| 批量/打包 forward | `model/qwen35_model.py` | `forward` 拒绝 `batch>1`、`attention_mask`、`past_len>0 且 T>1` |
| 混合引擎 | `serve/engine.py` | `MiniLLMEngine` 硬编码 `MiniLLM`/`KVcache`/`KvSlab`/`PagedPool` |
| 调度器协议 | `serve/scheduler.py` | 已是 duck-typed（文档化可选钩子），但没有 `Qwen35Engine` 实现，协议未固化成 ABC/测试 |
| bench 入口 | `bench/engines/__init__.py` | `impl="hybrid"` → `NotImplementedError("... W7")` |
| W7 配置/阈值/run | `configs/experiments/W7/`、`[W7.*]` | 无配置；阈值只有 `[W7.state].state_estimate_error_max=0.20` |
| bf16 GPU 数值门 | `tests/integration/` | W6 只有 CPU fp32；bf16/CUDA 的 64/64 未建立 |

### 1.4 环境

| 项 | 值 |
|---|---|
| GPU | 4090 24 GB（`measure-cuda`）；M1 只做 `dev-mps` 冒烟 |
| torch / CUDA | 2.8.0+cu128 |
| transformers | 5.20.2 |
| fla / causal_conv1d | 0.5.2 / 1.7.0 |
| 0.8B snapshot | `2fc06364715b967f1860aea9cf38778875588b17` |
| 0.6B snapshot | `c1899de289a04d12100db370d81485cdf75e47ca` |

### 1.5 状态字节算术（W7 记账基准）

```text
线性状态（每请求，18 层）:
  conv bf16 = 18 × 6144 × 3 × 2  =   663,552 B =  0.633 MiB
  recurrent = 18 × 16 × 128 × 128 × 4 = 18,874,368 B = 18.000 MiB
  合计 bf16 = 18.633 MiB；fp32 conv = 19.266 MiB

全注意力 KV（每请求）:
  bf16 = 6 层 × 2(K,V) × 2 kv_head × 256 × 2 B = 12 KiB/token
  fp32 = 24 KiB/token

W0 解析式（对照）:
  混合 = 18.84 MiB + 12 KiB/token
  纯注意力 0.6B = 112 KiB/token
  预填交叉点 ≈ 18.84 MiB / (112 − 12) KiB ≈ 193 prompt token
```

> **两个交叉点不要混**：`193` 是“prefill 状态、按 prompt token”的交叉点；
> 如果按“prompt + 128 decode 预算”的**预留**状态算，常数被预算摊薄，交叉点约
> `193 − 128 ≈ 65` prompt token。W7 两个都报，但标题数字必须注明口径（§6.5）。

---

## 2. 范围与口径裁定（active 时冻结）

### 2.1 D9 草案（Step 0 写入 `docs/decisions.md`，proposed；收口时 accepted）

| # | 决定 | 理由 / 边界 |
|---|---|---|
| **D9-1** | `StatePool` 管线性状态槽；`BlockManager`/`PagedPool` 管 KV 块；`RequestHandle` 把 `{slot, block_table, length}` 绑成一个生命周期 | 不复制块管理器；释放必须同时归还两者，且幂等 |
| **D9-2** | 紧凑活跃槽：释放时把最后一个活跃槽换入空洞；GDN 批量直接读 `[0:B)` 视图 | 避免每步 gather 18 MiB 状态；换入成本只发生在释放 |
| **D9-3** | `PooledQwen35State` 实现/继承 `Qwen35State` 的公开接口（getter/setter/advance/cull），只替换存储与生命周期；`model/qwen35.py` 只加 **paged / varlen / seat 容器分支**，不改 GDN 算子与注意力数学 | D8 决定 5；`Qwen35Model.forward` 的 `isinstance` 由子类满足（或改结构化检查，Step 1 定）；容器分支是 D9-5/D9-6 的前提 |
| **D9-4** | `Scheduler` 只依赖引擎协议；`MiniLLMEngine` 与 `Qwen35Engine` 两个实现，调度器不出现模型分支 | 同一份 trace 可跑两模型；“同一调度器”不要求同进程双模型 |
| **D9-5** | 混合模型的 paged KV 只覆盖 6 个全注意力层；18 个线性层走 StatePool | 全注意力层号映射为紧凑 0..5；`PagedPool` 不按 24 层分配 |
| **D9-6** | packed group prefill：全注意力层 varlen；GDN 先按 segment 循环（chunked），写回池状态 | 先正确后融合；fused varlen GDN 不是硬门，prefill 代价单独报 |
| **D9-7** | 正式数字统一 bf16 / CUDA；A0 是“与 HF 同 dtype 贪心 64/64”，W6 fp32 golden 只作交叉核对 | bf16 near-tie 必须记录 margin；不静默放宽 |
| **D9-8** | 容量主口径 = 同 **1640 MiB 状态预算**（KV + 线性状态 + workspace）；权重不计入；总卡预算只作辅助列 | 两模型权重不同，权重差不是机制；**纯臂与 W4 C1 同口径，混合臂把线性状态并入，是扩展口径** |

### 2.2 P1–P3：服务化复核定义

保留 W0（`docs/carrier.md` §4）的历史判定，W7 不重复单请求 HF 探针，而是把三条预测
改写成可测的 serving 命题。`docs/hybrid_attention.md` §4 的“待填”按此口径填。

| # | W0 原判 | W7 服务化命题 | 判定线 |
|---|---|---|---|
| **P1** | 成立：8k **7.80×**、32k **8.90×** | 同 1640 MiB 状态预算下，8k 每序列状态混合 ≤ 纯注意力的 **1/5** | `[W7.hybrid].p1_state_ratio_8k_min = 5.0` |
| **P2-memory** | 不成立（方向相反）：512 混合已小 **2.25×** | 512 时混合每序列状态是否反而更大？ | 预期 **false**；与 W0 一致才算复核通过 |
| **P2-latency** | W0 未测 serving | 512、batch=1/8 时混合 TTFT/TPOT 是否比纯注意力差 ≥10% | `p2_latency_material_ratio = 1.10`；batch 依赖时判“口径依赖” |
| **P3** | 数值成立但不可归因：混合 TPOT 22.41→**21.74** ms（8k；32k 为 21.89）；0.6B **16.55→19.97** ms（8k，+20.7%；32k 为 18.31，+10.6%） | batched/paged 下 `TPOT(8k)/TPOT(512)`：混合 ≤1.10，纯 ≥1.15；端点统一 **512→8k**，32k 不进判据；用每步 KV/状态字节归因 | `hybrid_tpot_growth_max=1.10`、`pure_tpot_growth_min=1.15`；纯也平坦则仍判“不可归因” |

> **P2 拆开是重点**：W0 的“短上下文混合更贵”在**显存**上已被证伪；W7 要回答的是
> **延迟**上是否成立。若 batch=1 成立、batch=8 反转，报告写“口径依赖”，不强行二选一。

### 2.3 容量口径

- **主口径**：同 **1640 MiB 状态预算**（KV 块 + 线性状态槽 + workspace），
  纯臂与 W4 C1/D6 同口径（W4.5 之后 workspace=0），混合臂把线性状态并入，是**扩展口径**；
  权重（1.14 vs 1.44 GiB）**不计入**，因为权重差不是机制。
- `max_batch` 是调度上限，**不是容量上限**；容量探针用 `linear_slots` / `kv_blocks`
  作为预算旋钮，fill-only 到不能再准入，记 `peak_concurrency`。
- 同一 context 的纯/混合两臂必须用**同一份合成 trace**（同 token 长度、同 seed、
  同 `trace_sha256`）；纯臂 `linear_slots=0`。
- 辅助列：同“总卡预算”（24 GB 减权重）下的容量；只报不判。两套数字不同表。
- 容量按 `prompt + 128` 的**预留**状态算；P1/交叉点按 prompt 的**prefill**状态算；
  两者都在 summary 里标明。

### 2.4 dtype / bf16 口径

- 正式 run：`dtype="bfloat16"`，`device="cuda"`，pin revision；0.6B 与 0.8B 同 dtype。
- **G0**：0.8B bf16 与 HF 同 dtype、同设备贪心 **64/64**（s01/m02）；0.6B bf16 64/64。
  near-tie（top1−top2 margin 很小）逐步记录并解释；解释只用于诊断，**不改变 strict 64/64 硬门**，
  tie-adjusted 计数可同时报，但不作通过依据。
- W6 的 fp32 golden 只作交叉核对：bf16 与 fp32 的 token 差异必须用 top1−top2 margin
  解释；不重生成 golden、不静默放宽 64/64。
- GDN 算子内部已是 fp32（W6），bf16 只影响输入/输出与 conv_state；W7 复核
  `conv_state` dtype 与 recurrent fp32 契约不被池化破坏。
- 若 G0 失败：先查数值，不进入池/调度实现；必要时退回 fp32 做正确性、bf16 只做性能，
  并在报告显著标注（不静默混用）。

### 2.5 模型 / 引擎边界

- 同一个 `Scheduler` 类、同一套 trace/config 骨架；两个引擎实例分别跑，
  **不做同进程同时加载两个模型**。
- `MiniLLMEngine`（0.6B）行为冻结，只抽协议，不改 W3/W4.5 路径。
- `Qwen35Engine`（0.8B）新模块；`bench` 的 `impl="hybrid"` 走它。
- `serve/` 只依赖 `model/`，不反向依赖 `bench/`/`scripts/`。

### 2.6 workload / trace 口径

| 用途 | workload | 规则 |
|---|---|---|
| E0 正确性 | `bench/prompts.json`（冻结 32 条） | 不改 prompt 集；`s01/m02` 对 golden；8 条 mixed 对 batch/逐请求 |
| E1/E2 容量、E3 交叉点 | `kind="synthetic"`，固定 seed | 同模型对同 trace；`trace_sha256` 必须一致；不与 W3/W4 的 prompt-hash run 直接配对 |
| E4 机制迁移 | 合成 mixed 长度 + Poisson/closed | 纯/混合两臂用**同一 token 长度 trace**（避免 tokenizer 差异）；历史 W3/W4 只作方向参照 |

### 2.7 E4 范围（用户 2026-11-14 决定：进 W7）

| 子实验 | 问题 | 臂 | 判据 |
|---|---|---|---|
| **E4a** | 连续批处理在混合架构下收益如何变 | hybrid × static/continuous，B=8 mixed | record-only；报 tok/s 比值；纯臂引用 W45-0010/0011（1.863×，同机） |
| **E4b** | 分页 KV 在混合架构下容量收益如何变 | hybrid × paged/seat，fill mixed | record-only；报 paged/seat 容量比；纯臂引用 W4-0002/0003（3.111×，同机）；seat 是 baseline-only |

E4a/E4b 的纯臂引用同机历史（`W45-0010/0011` 1.863×、`W4-0002/0003` 3.111×），只跑 hybrid 臂；历史 trace 与 W7 synthetic 不同，notes 标“方向参照”。
**E4 不整块删；E4a/E4b 的 hybrid/seat 臂不砍；hybrid seat 是 baseline-only，不是生产路径。**

### 2.8 不做

- 不训练、不提交权重、不改 `bench/prompts.json`、不重生成 `tests/golden/`。
- 不做 chunked prefill、prefix cache、CUDA graph、抢占/swap、Triton GDN kernel。
- **不做逐请求 prefill 控制臂**（W4.5 已删 serial prefill；E4d 取消，packed 与 serial 的结论引用 W4.5 E3）。
- 不做视觉/MTP、4B 载体、量化（W8 可选）。
- 不把 0.6B 与 0.8B 的绝对 tok/s 当“架构因果”；跨代对照只在状态预算/同 trace 下做，
  吞吐数字标注“端到端，含模型规模差异”。

---

## 3. 架构设计

### 3.1 模块

| 模块 | 职责 | 状态 |
|---|---|---|
| `serve/state_pool.py` | 线性状态槽池、紧凑活跃槽、请求句柄、used/reserved/slack 记账 | **新增** |
| `serve/hybrid_engine.py` | `Qwen35Engine`：`prefill_group` / `decode_batch` / `can_admit_group` / `release_request` / stats；paged 池（6 层）+ seat 池（6 层 `KvSlab`，baseline-only） | **新增** |
| `serve/engine.py` | 抽 `SchedulerEngine` 协议；`MiniLLMEngine` 行为冻结 | 改 |
| `serve/scheduler.py` | 只依赖协议；不增加模型分支 | 改（最小） |
| `model/qwen35_model.py` | `PooledQwen35State`、`forward_decode_batch`、`forward_packed` | 改 |
| `model/qwen35.py` | 全注意力层的 varlen / paged / seat 容器分支；单序列路径不动 | 改 |
| `model/qwen35_gdn.py` | 批量 decode / segment prefill 调用；算子本身不动 | 尽量不动 |
| `bench/engines/hybrid.py` | `impl="hybrid"` 的 scenario/fill 适配 | **新增** |
| `bench/engines/__init__.py` | 移除 `hybrid` 的 `NotImplementedError`，接工厂 | 改 |
| `bench/engines/fill.py` | 混合臂 fill-only 容量；W7 改为**跨步驻留**（见 §5.4） | 改 |
| `bench/thresholds.py` | `[W7.hybrid]` 求值 | 改 |
| `scripts/plot_w7.py` | 容量/交叉点图 | **新增** |

### 3.2 `StatePool`

- 容量 `linear_slots`；每线性层预分配：
  - `conv [capacity, 6144, 3]`，dtype 跟随模型（bf16/ fp32）；
  - `recurrent [capacity, 16, 128, 128]`，**fp32 契约**（D8）。
- 槽位状态：`free / reserved / active / releasing`；free list + 紧凑活跃区 `[0:B)`。
- `alloc()` 从 free list 取；`free(slot)` 把最后一个活跃槽换入该洞，更新被换请求的
  `handle.slot`，再归还洞；释放是 O(状态大小) 的一次拷贝，不在每步 decode 做 gather。
- `state(slot)` 返回 `PooledQwen35State` 的视图，不复制；setter 把算子返回的新张量 `copy_` 回槽
  （`gated_delta_recurrent` 是 clone + 返回新 fp32 状态，不是原地写），每步写回约
  `18 层 × B × 1 MiB`（B=8 约 144 MiB/step），在 TPOT 归因里单列；in-place 变体是 stretch，不设硬门。
- `stats()` 返回 `used_mb / reserved_mb / slot_count / active / free / slack_mb`。
- `close()` 释放全部；`assert_no_leaks()` 要求所有槽 free、所有 block table 归还。

### 3.3 请求句柄生命周期

```text
reserve ──> prefilled ──> active ──> finished/aborted ──> released
   │            │             │
   │            └── 失败 ─────┴──> rollback（归还 slot + blocks，幂等）
   └── 任一资源不足：整组拒绝，不留半分配
```

| 阶段 | 允许操作 | 不变量 |
|---|---|---|
| `reserve` | 原子预留 `linear_slot` + `block_table`；写 prompt_len/length | 两者要么都拿到，要么都没有 |
| `prefilled` | 写 KV/线性状态；失败回滚 | 不允许 decode；半写状态必须可丢弃 |
| `active` | decode；更新 length | slot/block_table 不被其他请求复用 |
| `finished` / `aborted` | 停止 decode；记录终止原因 | 不再写状态 |
| `released` | 归还 slot + blocks；幂等 | 重复 release 不二次释放；`handle.phase=released` |

- `release_request(id)` 幂等：已释放直接返回；未知 id 报错，不静默。
- 释放顺序：先标记 `releasing`，再归还 blocks，再 free slot；异常时保留可诊断状态。
- 调度器在 `finish`、`abort`、`close` 三条路径都调用同一个 release。

### 3.4 `PooledQwen35State`

- 实现 `Qwen35State` 的同一接口：`.kv`、`.past_len(layer)`、`.conv_state(layer)`、
  `.recurrent_state(layer)`、`.set_conv_state`、`.set_recurrent_state`、`.advance(n)`、`.cull`。
- `.kv` 返回 paged 视图（6 层 block table + seqlen），不是 `KVcache`；全注意力层的
  decode 分支读 paged 视图，prefill 分支写 block table。
- seat 臂的 `.kv` 返回 6 层 `KvSlab` 视图（行宽 = 该 run 的 trace 最大 prompt + budget）；
  seat/paged 共用同一 `Qwen35State` 接口，层代码只按容器类型分支（D9-3）。
- `.conv_state(layer)` / `.recurrent_state(layer)` 返回 `[B,...]` 池视图；
  批量 decode 时 `B = active_count`，与紧凑槽 `[0:B)` 对齐。
- `positions` 是 per-row 向量（不是单个 `past_len`）；RoPE 按行取位置，参照 W3
  `MiniLLM._rope_rows` 的做法。
- **GDN 层不读 per-row positions**：它只用 `past_len == 0` vs `>0` 选端点（P-56），
  批量 decode 对所有活跃行统一传标量 `1`；状态批量在 `[0:B)`。per-row positions
  只服务全注意力层的 RoPE 和 paged seqlen。
- `cull(keep)` 只用于 W3 风格的 batch 收缩；W7 的调度器用 handle release，不依赖 cull。
- 单序列测试仍可用原 `Qwen35State`；两条路径的 state_dict/数值必须一致。

### 3.5 批量 decode

一次 forward 覆盖 `B=active_count` 行：

1. 6 个全注意力层：`q_norm`/`k_norm` → partial-RoPE（per-row positions）→
   `paged_decode`（D=256、GQA 8/2、bs16）→ 输出门控 → `o_proj`；
2. 18 个 GDN 层：`causal_conv_decode` + `gated_delta_recurrent`，状态读 `[0:B)` 池视图；
3. final norm → `lm_head` 只对**紧凑活跃行**取最后位置 → `(B_active, vocab)`；引擎再按 seat
   散回 `(max_batch, vocab)`（idle 行写占位、scheduler 丢弃），见 §3.7 的 `decode_batch` 契约；
4. 写回 conv/recurrent 状态（`copy_` 回池，见 §3.2）；KV 新 token 走 `write_slots`（每步一次 slot map）。

`Qwen35Model.forward` 的单序列路径保留，作为测试参考；批量路径新方法
`forward_decode_batch(tokens, state, positions)`，不改变单序列语义。

### 3.6 packed group prefill

- 输入：一个 FIFO group（`max_prefill_tokens` / `max_batch` 限制），每成员有
  `{id, prompt_ids, handle}`。
- 全注意力层：request-major packed，`cu_seqlens` + 段内 causal；复用 W4.5 的
  varlen SDPA/FA2 容器；partial-RoPE 按段内位置。
- GDN 层：**先按 segment 循环**，每段 `past_len=0` 走 W6 chunked prefill，写回该请求
  的 conv/recurrent 终态；输出按段拼回 packed 布局。这样正确性先锁住；真正 fused
  varlen GDN 留作 stretch（不设硬门）。
- **prefill 段必须从空状态开始**：layer 调用传 `conv_state=None`/`recurrent_state=None`
  （或 pool 的 fresh 视图返回 `None`），不能把槽里上一个请求的旧值当 prefix；
  W6 的 `past_len=0` 会显式拒绝非空状态。槽在 `reserve` 时视为空，prefill 成功后才写终态。
- 全注意力 KV（paged）：prefill 写穿到该请求的 paged block table；不建 `KVcache`、不做交接拷贝。
- 全注意力 KV（seat）：prefill 写 per-request `KvSlab` 行（6 层、2 KV head、D=256，
  行宽 = trace 最大 prompt + budget），复用 `SlabWriteSink` 的 `set_kv`；GDN 状态仍走 `StatePool`。
  seat 钩子（`slab`/`freeze_seat`/`activate_seat`/`prefill_into_seat`）与 `MiniLLMEngine` 同名，
  `Scheduler` 不需要模型分支。
- 每组只对 last token 过 `lm_head`；返回 `(B, vocab)` + 每请求长度。
- 失败：组内任一请求 prefill 异常 → 该请求 rollback；组内其他请求已写的状态由调度器
  统一 abort/release；不允许半写状态进入 active。

### 3.7 引擎协议与 `Scheduler`

`Scheduler` 只依赖以下协议（名字可微调，语义冻结）：

```text
paged: bool
max_prefill_tokens: int
vocab_size / device / eos_token_id / max_position
can_admit_group(requests) -> bool
prefill_group(requests) -> logits (n_group, vocab)
decode_batch(tokens) -> logits (max_batch, vocab)   # seat 索引；idle 行占位
release_request(request_id) -> None
paged_stats() -> dict
pool_kv_mb: float | None
close_pool() -> None
# 两个引擎都有的 seat 钩子：slab / freeze_seat / activate_seat / prefill_into_seat
# （hybrid seat 是 baseline-only，生产默认 paged）
```

- `MiniLLMEngine` 已实现上述行为；把现有 duck-type 契约固化成 `SchedulerEngine` Protocol/ABC + 测试，不改行为。
  `linear_state_mb`/`weights_mb` 在 bench adapter 上，不在引擎协议里。
- `Qwen35Engine` 实现同一协议；`Scheduler` 不出现 `isinstance(model, Qwen35Model)`。
- 调度器的 `_paged_mode` 只看 `engine.paged`；混合臂默认 paged，E4b 的 seat 臂切 `paged=False`
  并走同一引擎的 seat 池；调度器不出现模型分支。
- `decode_batch` 的 `tokens` 长度恒为 `max_batch`（idle 位为 `IDLE_TOKEN`）；引擎负责 seat→紧凑 row
  映射并把 logits **散回 seat 行**——`Scheduler._decode` 要求 `len(ids)==max_batch` 且用 `ids[req.seat]`
  取 token（`_decode_paged` 已这样做）。
- `can_admit_group` 必须原子：linear slot + blocks 任一不足 → 整组拒绝，队头留队列。

### 3.8 KV / 线性状态映射与记账

- 全注意力层号 `[3,7,11,15,19,23]` → 紧凑层号 `0..5`；`PagedPool` 只分配 6 层。
- 每个 block（bs16）字节：`6 × 2 × 2 × 256 × 2 × 16 = 196,608 B = 192 KiB`
  （bf16，6 层合计）；纯 0.6B 每 block 为 `28 × 2 × 8 × 128 × 2 × 16 = 1,835,008 B = 1.75 MiB`
  （8 个 KV head，与 112 KiB/token 一致）。两套 block 不混用。
- `used` = 活跃槽线性状态（按实际 dtype）+ 已写 KV token 字节；最后一块的未写尾巴
  算 `reserved`/`slack`，**不算 used**（否则会把碎片塞进与解析式的误差）。
- `reserved` = `linear_slots` 全部槽 + 已预留 block 数 × block 字节；
  `slack` = reserved − used。
- `workspace`：W4.5 之后 paged/seat 的**持久池** workspace 是 **0**
  （`paged_stats()['workspace_mb']=0.0`，结构性零，不是未估），W7 沿用；容量预算 = KV + 线性状态。
  GDN chunked prefill 的 transient 激活（8k 时 `T×conv_dim` 量级）不进持久预算，
  用 torch allocator 峰值单列，与权重/KV/线性状态分开报。
- 解析式对照：`18.84 MiB + 12 KiB/token`（W0）；W7 实测 bf16 为
  `18.63 MiB + 12 KiB/token`，差 1.1%（来自 conv 列数：W0 按 kernel 4 = 0.84 MiB，
  W7 按 k−1 = 3 = 0.633 MiB，不是 dtype/测量差）；误差按 `used` 对解析式算，reserved/slack 只报。

### 3.9 失败原子性与泄漏检查

- `reserve` 失败：整组拒绝；不产生孤儿 block/slot。
- prefill 失败：rollback 该请求；组内其他请求由调度器 abort；池状态回到 reserve 前。
- decode 异常：调度器 abort 该请求；release 幂等；不让坏状态留在 active 区。
- `close_pool()`：归还全部 block、清空槽、`assert_no_leaks()`；测试覆盖重复 close。
- 每个 formal run 的 summary 记录 `linear_slots`、`kv_blocks`、`peak_active`、
  `state_used_mb`、`state_reserved_mb`、`leak=false`。
- `paged_stats()` 复用现有键（`kv_mb` / `kv_live_mb` / `kv_alloc_mb` / `workspace_mb` /
  `blocks_in_use` / `free_blocks` / `frag_internal` / `reservation_slack` / `decode_impl`）；
  W7 只增量加 `linear_state_mb` / `linear_slots` / `active_rows` / `peak_active` / `leak`，
  `bench/engines/paged.py` 的读取路径不改。

### 3.10 不变量

1. 一个请求同时最多一个 `linear_slot` 与一张 `block_table`。
2. 活跃槽恒为 `[0:B)`；`handle.slot` 与池内位置一致。
3. `release` 幂等；已释放的 slot/block 不再被任何 handle 引用。
4. 池化 `Qwen35State` 与单序列 `Qwen35State` 对同一输入给出相同 logits/状态（fp32 测试）。
5. 批量 decode 的每行 == 单独 decode；packed prefill 的每段 == 单独 prefill。
6. 任何路径都不静默回退到逐请求/稠密 gather；失败必须显式报错或按计划回退并记录。
7. seat 与 paged 两臂对同一 trace 的逐请求 token id 一致（E4b 的配对门）；hybrid seat 只作基线。

---

## 4. 工作分解

### Step 0 · 基线与口径冻结

**目标**：确认 W7 起点、建立 bf16 GPU 数值门、把 §2/§5/§9 冻结成可执行口径。

**改动**
- `docs/decisions.md`：新增 **D9（proposed）**，内容按 §2.1。
- `configs/thresholds.toml`：新增 `[W7.hybrid]`（target），保留 `[W7.state]`；
  值按 §5.7，正式 run 前不再改。
- `tests/unit/test_thresholds_w7.py`：锁 section/key/值。
- `tests/integration/test_align_qwen35_bf16.py`（或参数化现有 W6 全模型测试）：
  0.8B bf16/CUDA 与 HF 同 dtype 贪心 64/64；0.6B bf16 64/64；记录 top1−top2 margin。
- `configs/experiments/W7/`：空目录/模板，不写正式配置。

**门禁 G0**
- 0.8B bf16 与 HF 同 dtype **64/64**（s01/m02）；0.6B bf16 64/64；
- W6 fp32 golden 交叉核对：token 差异必须由 margin 解释；
- 快速门禁保持绿（不重跑 874 全量作为前置，只跑受影响 + 阈值测试）。

**证据**：pytest 输出；D9 proposed；阈值 diff。
**回滚**：G0 不过 → 不进 Step 1；W7 标 `blocked`，先修数值。

### Step 1 · `StatePool` + 请求句柄

**目标**：线性状态池化、紧凑活跃槽、统一句柄生命周期；不改调度器。

**改动**
- `serve/state_pool.py`：`LinearStatePool`、`PooledQwen35State`、`RequestHandle`、
  `PoolStats`、`assert_no_leaks`。
- `model/qwen35_model.py`：`PooledQwen35State` 接入；单序列路径不动。
- `tests/unit/test_state_pool.py`、`tests/unit/test_qwen35_pool_state.py`：
  分配/释放/复用/双释放/中途失败回滚/换入/无泄漏；池化 vs 单序列数值。

**门禁 G1a**
- 生命周期测试全绿；`used/reserved/slack` 与手算一致；
- 池化状态与 W6 单请求 state 字节/数值一致（fp32）；
- `close()` 后 `assert_no_leaks()` 通过，重复 close 安全。

**证据**：单测；状态字节对照表。
**回滚**：池化数值不一致 → 不改层代码，先修视图/setter；必要时退回“每请求独立 state”只做正确性（W7 部分完成）。

### Step 2 · 批量 decode + 引擎协议 + 同一调度器

**目标**：0.8B 能在 `Scheduler` 下 decode；0.6B 路径不回归；E0 干跑。

**改动**
- `serve/engine.py`：抽 `SchedulerEngine` 协议；`MiniLLMEngine` 实现声明；行为不变。
- `serve/hybrid_engine.py`：`Qwen35Engine`（`open_hybrid_pool`、`can_admit_group`、
  `prefill_group` 先逐请求、`decode_batch`、`release_request`、stats）。
- `model/qwen35_model.py`：`forward_decode_batch`；per-row positions。
- `model/qwen35.py`：全注意力层接 paged decode 容器（Triton D=256/GQA）；同时加 seat `KvSlab`
  分支（6 层、2 KV head、D=256；复用 `SlabWriteSink`/`KvSlab` 的 prefill/activate/freeze 语义；
  baseline-only，不是生产路径）。
- `serve/scheduler.py`：只改协议调用/`paged` 判定，不加模型分支。
- `bench/engines/hybrid.py`、`bench/engines/__init__.py`：`impl="hybrid"` 可用；
  `bench/run.py` 校验 `linear_slots` 等键。
- `tests/unit/test_qwen35_batch_decode.py`、`test_qwen35_seat.py`、`test_hybrid_engine.py`、
  `test_scheduler_hybrid.py`。

**门禁 G1b**
- 批量 decode 每行 == 单独 decode（fp32 小模型 + bf16 0.8B 各一组）；
- 同一份 trace 在 0.6B/0.8B 上都能跑完；`Scheduler` 源码无模型分支；
- E0 干跑：0.8B B=1/B=8 token id 全等；槽位复用不串状态；0.6B 64/64 不变；
- hybrid seat 与 paged 对同一 trace 的逐请求 token id 全等；两臂 release 后无泄漏。

**证据**：单测 + 干跑（dirty，仅校准，不进报告）。
**回滚**：G1b 不过 → 按 §0.5 回退逐请求混合引擎；W7 部分完成。

### Step 3 · packed group prefill + fill/容量

**目标**：packed group prefill 正确；fill-only 容量可测；KV/线性状态写穿到池。

**改动**
- `model/qwen35_model.py::forward_packed`；`model/qwen35.py` varlen 分支；
  GDN segment 循环写回池。
- `serve/hybrid_engine.py::prefill_group`：全注意力 varlen + GDN 分段；
  失败 rollback；last-token logits。
- `bench/engines/fill.py` / `paged.py`：混合 fill 路径 + **跨步驻留**（§5.4）；`linear_slots`/`kv_blocks`
  预算检查；`state_used/reserved` 进 summary。
- `tests/unit/test_qwen35_packed_prefill.py`、`test_hybrid_fill.py`。

**门禁 G2/G3**
- packed prefill 每段 == 单独 prefill（token id + KV/状态容差）；
- 失败注入：组内某请求异常，池无半写状态、无泄漏；
- fill 能在 512/2k/8k 跑到预算上限，`peak_concurrency` 与 §5.4 预测差 ≤1；
  `kv_total + linear ≤1640`（持久 workspace=0）。

**证据**：单测 + fill 干跑。
**回滚**：G2 不过 → 先修 packed；**不实现逐请求 prefill 控制臂**；修不动则停在 E4a/E4b，不伪造 packed 收益。

### Step 4 · 正式 run（E1/E2/E4）

**目标**：干净树跑完核心 run，评审 summary/verdict。

**改动**
- `configs/experiments/W7/0001…`（按 §5.1 矩阵）；每条自包含、pin revision、
  `dtype=bfloat16`、`device=cuda`。
- `bench/thresholds.py`：接 `[W7.hybrid]` 容量/P1/P3 判据；缺 key 必须报错。
- `scripts/plot_w7.py`：容量/交叉点图。
- 正式 run 由用户在干净树执行/确认；agent 只准备配置、dry run、初稿 notes/verdict
  （`reviewer=null`、`verdict=unreviewed` 直到用户确认）。

**门禁 G4**
- E0/E1/E2/E4a/E4b 全部有有效 run；配对 trace_sha256 一致；
- 无 over-budget、无泄漏、无 invalid 未解释；

**证据**：`results/W7/` + INDEX；summary/verdict。
**回滚**：某 run invalid → `supersedes` 重跑，不覆盖；预算/口径错误按 D6/W4 规则处理。

### Step 5 · E3 / P1–P3 / 交叉点 / 图

**目标**：把 E0–E4 的数字变成 W8 可用的判定表。

**改动**
- `docs/hybrid_attention.md` §4：填每序列状态、P1/P2/P3、交叉点、同显存最大并发；
  标明实测/分析、口径、run id。
- `scripts/plot_w7.py`：`fig_w7_capacity.png`（容量 vs context）、
  `fig_w7_crossover.png`（每序列状态/吞吐 vs context）。
- D9 补证据与后果；ROADMAP §5 W7 实测段落草稿。

**门禁 G5**
- P1/P2-memory/P2-latency/P3 每条有 verdict（true/false/口径依赖/不可归因）+ 证据；
- 交叉点给区间 + 口径（prefill vs reservation；memory vs throughput）；
- 图能从已提交 summary 复现，不手改数字。

**证据**：判定表、图、D9 草稿。
**回滚**：某条证据不足 → 标“未实测/分析”，不硬判；不补造数字。

### Step 6 · 收口

**目标**：W7 生命周期闭环，W8 可直接写报告。

**改动**
- D9 → **accepted**；`[W7.state]` calibrated（basis run）；`[W7.hybrid]` calibrated
  （数值不改，只改状态/基准；若需改值，必须回 D9 修订并在下一次正式 run 前）。
- ROADMAP §5 W7 实测、§12 W7 ✅、§13 v31；PITFALLS 新增条目；
  `results/INDEX.md`/`index.csv` 由脚本重生成。
- 删除 `docs/w7-plan.md`；检查无悬空引用。

**门禁 G6**：§9.4 删除清单全过。
**证据**：最终 diff、run 索引、图、D9。

---

## 5. 实验

### 5.1 run 矩阵（连续编号，从 0001 开始；E4c 已删除）

| 计划 ID | 实验 | 臂 | type | workload | 目的 |
|---|---|---|---|---|---|
| `W7-0001` | E0b | hybrid | scenario | 8 条 mixed prompts，B=1，128 | 逐请求参考 |
| `W7-0002` | E0b | hybrid | scenario | 同 8 条，B=8 continuous，128 | batch == 逐请求 |
| `W7-0003` | E0c | pure 0.6B | scenario | 同文本 8 条，B=8，128 | 调度器回归 |
| `W7-0004` | E1/E2 | hybrid | fill | synthetic 512 | 容量 + 状态 |
| `W7-0005` | E1/E2 | pure | fill | synthetic 512 | 基线 |
| `W7-0006` | E1/E2 | hybrid | fill | synthetic 2k | 容量 + 状态 |
| `W7-0007` | E1/E2 | pure | fill | synthetic 2k | 基线 |
| `W7-0008` | E1/E2 | hybrid | fill | synthetic 8k | 容量 + 状态 |
| `W7-0009` | E1/E2 | pure | fill | synthetic 8k | 基线 |
| `W7-0010` | E3 | hybrid | scenario | synthetic [512,2k,8k]，B=1 | P2/P3/交叉点 |
| `W7-0011` | E3 | pure | scenario | 同上 | 基线 |
| `W7-0012` | E3 | hybrid | scenario | 同上，B=8 | P2/P3/交叉点 |
| `W7-0013` | E3 | pure | scenario | 同上，B=8 | 基线 |
| `W7-0014` | E4a | hybrid | scenario | synthetic mixed Poisson，B=8，continuous | 机制迁移 |
| `W7-0015` | E4a | hybrid | scenario | 同上，static | 机制迁移 |
| `W7-0016` | E4b | hybrid | fill | synthetic mixed，paged | 容量 |
| `W7-0017` | E4b | hybrid | fill | 同上，seat | 容量（baseline） |
| `W7-0018` | E3 修正 | hybrid | scenario | synthetic [512,2k,8k]，N=24，B=8 | B=8 实际并发 8；替代未跑的 0012 |
| `W7-0019` | E3 修正 | pure 0.6B | scenario | 同上 | B=8 分母；`supersedes=W7-0013` |
| `W7-0020/0022/0024` | 交叉点扫描 | hybrid | fill | synthetic 128/192/256 | 内存交叉点实测；64 槽 + 2048 块 |
| `W7-0021/0023/0025` | 交叉点扫描 | pure 0.6B | fill | 同上 | 纯臂 936 块；与相邻 hybrid 同 trace |

> 共 **23 个有效正式 run**：hybrid 14、pure 9。G0 是测试门禁，不占 run。
>
> **执行修正（2026-11-14）**：`W7-0012` 未运行；`W7-0013` 的 `requests=8` 被 3 个
> context 分掉，实测 `peak_concurrency=4`，已标 `invalid`；B=8 改用 `W7-0018/0019`
> （`requests=24`，每 context 8 条，`W7-0018/0019` 的 B=8 峰值均为 8）。因此
> 另加 128/192/256 的交叉点扫描 `W7-0020…0025`（hybrid/pure 各 3 条）。
> `results/W7/` 实际有 **24 个 run 目录**（23 有效 + 1 invalid），E3 的 B=8 证据引用
> `W7-0018/0019`，不引用 `W7-0013`。
> 0.6B 纯臂共 **9 个同工况对照**：E0c 1 个（同文本 prompt）、E1/E2 3 个 fill、E3 2 个 scenario、
> 交叉点扫描 3 个；E4a/E4b 的纯臂继续引用同机历史，不重跑。
> 配对规则：同一实验的实跑纯/混合臂必须同 workload、同 seed、同 `max_new_tokens`、
> 同 `trace_sha256`；引用的历史纯臂标注 trace 差异，不按 `invalid` 处理。运行顺序不共享进程状态。
> `W7-0004…0009` 的 `linear_slots/kv_blocks/max_batch/requests` 按 §5.4 预注册表；
> 不在 run 之间改预算。
>
> **为什么保留 9 个 0.6B 纯臂**：它们不是重复 W3/W4，而是同 trace 的分母——
> E0c 1 个证明调度器协议没破坏纯路径；E1/E2 的 3 个 fill 提供 `capacity_hybrid/capacity_pure`
> 与 P1 的每序列比；E3 的 2 个 scenario 提供 P2/P3；3 个 128/192/256 扫描用于内存交叉点。
> E4a/E4b 的纯臂已引用同机历史，不再重跑。

### 5.2 E0 · 正确性

- **E0b**：8 条 mixed prompts，B=1 与 B=8 continuous；逐请求 token id 必须全等；
  顺序/槽位复用不得改变结果。
- **E0c**：0.6B 同调度器 B=8；64/64 不回归；W3/W4.5 机制测试保持绿。
- **E0d**（测试，不是 run）：packed prefill vs 逐请求；批量 decode vs 单请求；
  pool state vs 单序列 state。

### 5.3 E1 · 状态记账

- 用 `W7-0004…0009` 的 fill summary；每个 context 报：
  `linear_mb`、`kv_written_mb`、`kv_reserved_mb`、`slack_mb`、`workspace_mb`、
  `state_used_mb`、`state_reserved_mb`、`peak_concurrency`。
- 解析式：混合臂 `18.84 MiB + 12 KiB/token`（W0）；纯臂 `112 KiB/token`；
  实测 bf16 混合预期 `18.63 MiB + 12 KiB/token`。
- 误差 = `|used − analytic| / analytic`；硬线 `<20%`；reserved/slack 只报。
- **`used` 从池张量实际字节读**（`element_size()*numel()`，或 allocator 差值），
  不用解析式反算；否则该门只能抓到 dtype/层数错误。
- 额外核对：`linear_state_bytes()`/`kv_bytes()` 的 W6 单测仍过；池化后不破坏
  conv 3 列、recurrent fp32、6 层 KV 的契约。

### 5.4 E2 · 容量

- `W7-0004…0009`：512/2k/8k，fill-only，同 1640 MiB 状态预算。
- 容量 = `peak_concurrency`（fill-only 最大同时准入）；报告：
  - `capacity_hybrid / capacity_pure`；
  - 每序列 `state_used` 与 `state_reserved`；
  - `kv_total + linear_total ≤1640`（持久 workspace=0；超出即 invalid/over-budget，不静默改预算）。
- 判据：`[W7.hybrid].capacity_speedup_512_min=2.0`、`2k=4.0`、`8k=6.0`；
  负线见 §5.7。整数 floor 会放大 8k 比值，必须同时报每序列字节比（P1）。

**驻留口径**：现有 `FillScheduler.step` 在 snapshot 后 release 全部驻留请求，`peak_concurrency` 等于单组大小；
uniform 2k/8k 下会被 `max_prefill_tokens`（默认 8192）卡成 3/1，测不出 37/14。W7 的 fill 改为
**跨步保留驻留请求**（只 prefill + 预留、不 decode，直到 `can_admit_group` 拒绝；run 在
`admitted=0` 时停止，即使仍有 resident；结束时统一 `close()`）。这是 `fill.py` 的改动，
并用 W4 的 mixed 配置做回归（`kv_blocks=800` 时 peak 仍 28±1，不重跑正式 run）。

**fill 旋钮（预注册，Step 0 定稿）**：预算约束
`linear_slots × 18.633 MiB + kv_blocks × 0.1875 MiB ≤ 1640`；`max_batch = 64`、
`requests = 64`。fill 不 decode，所以 `linear_slots < max_batch` 合法。混合臂按预测容量定槽/块，
纯臂 `linear_slots = 0`。

| run | 臂 | ctx | linear_slots | kv_blocks | 预测 peak_concurrency |
|---|---|---|---|---|---|
| `W7-0004` | hybrid | 512 | 62 | 2480 | 62 |
| `W7-0005` | pure | 512 | 0 | 936 | 23 |
| `W7-0006` | hybrid | 2k | 37 | 5032 | 37 |
| `W7-0007` | pure | 2k | 0 | 936 | 6 |
| `W7-0008` | hybrid | 8k | 14 | 7280 | 14 |
| `W7-0009` | pure | 8k | 0 | 936 | 1 |

> 例：512 混合 = 62×18.633 + 2480×0.1875 ≈ 1620 MiB；8k 纯 = 936×1.75 ≈ 1638 MiB。
> 实测 `peak_concurrency` 与预测差 >1 时先查记账（W4 D6 的 ±1 口径），不调预算重跑同一 run。
> 交叉点扫描 `W7-0020…0025` 的旋钮：hybrid `linear_slots=64, kv_blocks=2048`（1576.5 MiB），
> pure `kv_blocks=936`（1638 MiB）；`requests=max_batch=64`，1/3、closed。
> **不要沿用 W4 的 800 块**：800 块下 512 的纯/混合都只能到 20，比值 1.0×，那是配置卡住而不是机制。

### 5.5 E3 · 交叉点 / P2 / P3

- `W7-0010…0013`：synthetic contexts `[512, 2048, 8192]`，B=1 与 B=8；
  纯/混合配对。
- **内存交叉点**：用 prefill 每序列状态（prompt token）扫 `128/192/256/512`；
  报相等点区间；预期 190–200 token；同时报“预留口径”交叉点（约 65）。
- **吞吐交叉点**：同 context/batch 下 hybrid/pure 的 tok/s 或 TPOT 比值；找 1.0 交叉；
  若在 512–8k 内不交叉，明确写“未观察到”，不硬造。
- **P2-latency**：512、B=1/8 的 TTFT/TPOT 比值；≥1.10 记“混合更贵”，
  batch 依赖时判“口径依赖”。
- **P3**：`TPOT(8k)/TPOT(512)`（端点固定 512→8k，32k 不进判据）；混合 ≤1.10、纯 ≥1.15；
  用每步 KV 字节 + 线性状态字节归因；纯也平坦则“不可归因”。

### 5.6 E4 · 机制迁移

- **E4a**：`W7-0014/0015`，synthetic mixed + Poisson，B=8，hybrid only；
  比 `tok/s(continuous)/tok/s(static)`；纯臂引用 `W45-0010/0011`（1.863×，同机，trace 不同标方向）。
- **E4b**：`W7-0016/0017`，synthetic mixed，fill-only，hybrid paged/seat；
  纯臂引用 `W4-0002/0003`（3.111×，同机，trace 不同标方向）。
  hybrid seat 是 baseline-only；fill 旋钮按同一 1640 预算反解，配置头写死
  `linear_slots/kv_blocks/max_batch`；record-only，不设阈值。
- E4 全部 `record_only`；不设通过/失败线，但必须有数、有 notes、有“与纯代差异”的解释。

### 5.7 预注册阈值（Step 0 写入 `configs/thresholds.toml`，正式 run 前冻结）

```toml
[W7.state]
status = "target"
basis_run = ""
state_estimate_error_max = 0.20

[W7.hybrid]
status = "target"
basis_run = ""
greedy_tokens_exact = 64

# P1 / 容量（同 1640 MiB 状态预算）
p1_state_ratio_8k_min = 5.0
capacity_speedup_512_min = 2.0
capacity_speedup_2k_min = 4.0
capacity_speedup_8k_min = 6.0

# 交叉点 / P3 / P2
memory_crossover_ctx_max = 256
p2_latency_material_ratio = 1.10
hybrid_tpot_growth_max = 1.10
pure_tpot_growth_min = 1.15
```

> 容量阈值以 §5.4 的 fill 旋钮为前提；P3 端点统一 **512→8k**（32k 不进判据）。

负线（写进注释/plan，不新增未使用 key）：
- 512 容量 `<1.3×`、2k `<2.0×`、8k `<3.0×` → 记负结果；
- P1 `<5×` → P1 false；
- 内存交叉点 `>256` prompt token → 与 W0 不一致，必须解释；
- 混合 TPOT 增长 `>1.10` → P3 false（混合不平坦）。

> 收口只允许把 `status` 改 `calibrated`、填 `basis_run`；**不得下调数值**。
> 若要改，走 D9 修订并在下一次正式 run 前完成。

### 5.8 verdict 规则

- `pass`：硬线全过且配对有效。
- `fail`：硬线未过，但 run 有效；保留为负结果。
- `inconclusive`：样本/重复不足、预算边界、或 workload 不匹配。
- `invalid`：配对 trace/dtype/device/max_new_tokens 不一致；over-budget；泄漏。
- E4 全部 `record_only`：`pass` 只表示 run 有效且配对成立，不表示机制收益达标；无效记 `inconclusive`/`invalid`。
- `reviewer` 保持 `null`、`verdict` 保持 `unreviewed`，直到用户确认。

---

## 6. 测量与记账

### 6.1 协议

- 预热 3、重复 5、报中位数 + p95（ROADMAP §6）；fill 沿用 W4 协议
  （warmup 1 / repeats 3，容量是确定性上限）。
- 计时：CUDA event + `synchronize()`；TTFT 含排队；`queue_ms` / `pack_wait_ms` /
  `prefill_shared_ms` 归因沿用 D7。
- 显存：`torch_peak` + driver；权重 / KV / 线性状态 / workspace 分开；
  `nvidia-smi` 只作辅助。
- 方差：同进程同顺序；跨配置重开进程；不跨设备比较。

### 6.2 配对

- 同一实验的纯/混合臂：同 workload、seed、`max_new_tokens`、`out_tokens`、
  `trace_sha256`、dtype、device；否则 `invalid`。
- 跨代绝对 tok/s 不混表；比值必须绑定“同机、同 trace、同 dtype”。
- E4 的纯臂引用同机历史（W45/W4），标 trace 差异；W3/W4 只作方向对照。

### 6.3 状态记账

```text
used(seq, ctx) = linear_bytes(dtype) + kv_written_bytes(ctx)   # 都从池张量实测
reserved       = linear_slots × linear_bytes + kv_blocks × block_bytes
slack          = reserved − used            # 含最后一块未写尾巴 + 空闲槽/块
error          = |used − (18.84 MiB + 12 KiB × ctx)| / (18.84 MiB + 12 KiB × ctx)
```

- `ctx` 要写清是 prompt 还是 prompt+budget；P1/交叉点用 prompt，容量用 prompt+budget。
- 纯臂的解析式是 `112 KiB × ctx`；混合臂是 `18.84 MiB + 12 KiB × ctx`，两者不混用。
- `linear_bytes` 用实际 dtype（bf16 conv + fp32 recurrent）；不拿 W6 fp32 数字冒充 bf16。
- `used` 来自实际张量字节，不用本节的解析式反算；解析式只作对照。

### 6.4 容量

- fill-only：只 prefill + 预留，不 decode；`peak_concurrency` 是最大同时准入。
- 预算：`kv_total + linear_total ≤1640 MiB`（持久池 workspace=0；transient 激活单列）；
  over-budget 记 `invalid`/record，不降预算重跑同一 run。
- `linear_slots/kv_blocks/max_batch/requests` 按 §5.4 表；两臂 reserved 各自 ≤1640。
- 同一 context 的纯/混合用同 trace；纯臂 `linear_slots=0`。
- 报 `capacity_ratio` + `per_seq_state_ratio` + `reserved/used/slack`；
  整数 floor 导致的比值放大必须注明。

### 6.5 交叉点

- **prefill 交叉点**：`linear_bytes + 12KiB×ctx = 112KiB×ctx` → `ctx ≈ 193`；
  用 128/192/256/512 扫，给相等区间。
- **reservation 交叉点**：`linear + 12KiB×(ctx+128) = 112KiB×(ctx+128)` →
  `ctx ≈ 65`；容量报告用这个口径。
- **吞吐交叉点**：同 context/batch 的 tok/s 比值；1.0 处即交叉；没有就写“未观察到”。
- 三个数分别命名，禁止只写一个“交叉点”让读者猜口径。

### 6.6 统计

- 比值用中位数；延迟报 p50/p95；容量报峰值（不是末步）。
- CV 按长度桶，不把 16–8k 混成一个 overall；饱和场景只判 trace 级 tok_s CV。
- 每个数字带 run id；图只从已提交 summary 生成。

### 6.7 run contract

- 正式 run 必须干净树、pin revision、`--allow-dirty` 只用于 exploratory；
  dirty run 不进报告。
- run 目录不可变；失败/中断保留；重跑用 `supersedes`。
- `notes.md`/`verdict.json` 由人确认；agent 可起草，但不代签。
- `results/INDEX.md`/`index.csv` 由 `scripts/exp_index.py` 生成，不手改。

---

## 7. 风险与止损

| 风险 | 概率/影响 | 触发信号 | 缓解 / 止损 |
|---|---|---|---|
| bf16 与 HF 64/64 翻转 | 中 / 高 | G0 有 token 差 | 记录 margin；先查 GDN/RoPE/norm/loader；必要时 fp32 正确性 + bf16 性能并显著标注 |
| Qwen35 全注意力层接 paged decode 复杂 | 高 / 高 | Step 2 数值/形状错 | 先单层对拍，再接批量；Triton D=256/GQA 复用 W4.5；失败则 seat 臂先做正确性，paged 降级 |
| GDN packed prefill segment 循环慢 | 中 / 中 | prefill TTFT 明显差 | 先正确；prefill 代价记负结果；fused varlen 留 stretch，不阻塞容量 |
| StatePool 换入/映射错 | 中 / 高 | 槽位复用串状态/泄漏 | 生命周期单测 + 失败注入；active `[0:B)` 不变量；每次 release 校验 |
| 容量受整数 floor/块粒度影响 | 中 / 中 | 8k 比值虚高 | 同时报 per-seq 字节比与 reserved/slack；over-budget 判 invalid |
| 0.6B/0.8B 规模混淆 | 高 / 中 | 把 tok/s 差归因架构 | 状态预算隔离显存；吞吐标“端到端含规模差”；不做因果断言 |
| E4 workload 与 W3/W4 不同 | 中 / 中 | 比值不可比 | 同 trace 纯/混合配对；历史只作方向；notes 写清 |
| 4090 租期/环境漂移 | 中 / 高 | 跨实例比值不稳 | 同机同日配对；不引用跨实例绝对数；断租即停 |
| 范围蔓延 | 中 / 中 | 实现中出现新的优化想法 | 不进 W7；记 future work；E4a/E4b 的 hybrid/seat 臂与 A1–A3 优先 |

---

## 8. 交付物

### 8.1 代码

- `serve/state_pool.py`（新）
- `serve/hybrid_engine.py`（新；paged + seat baseline）
- `serve/engine.py`（协议抽取，行为冻结）
- `serve/scheduler.py`（协议调用，无模型分支）
- `model/qwen35_model.py`（池化 state / batch decode / packed prefill）
- `model/qwen35.py`（varlen / paged / seat 容器分支）
- `bench/engines/hybrid.py`（新）、`bench/engines/__init__.py`、`bench/engines/fill.py`
- `bench/thresholds.py`、`bench/run.py`（W7 键校验）
- `scripts/plot_w7.py`（新）

### 8.2 测试

- `tests/unit/test_state_pool.py`
- `tests/unit/test_qwen35_pool_state.py`
- `tests/unit/test_qwen35_batch_decode.py`
- `tests/unit/test_qwen35_seat.py`
- `tests/unit/test_qwen35_packed_prefill.py`
- `tests/unit/test_hybrid_engine.py`
- `tests/unit/test_scheduler_hybrid.py`
- `tests/unit/test_hybrid_fill.py`
- `tests/unit/test_thresholds_w7.py`
- `tests/integration/test_align_qwen35_bf16.py`（`hf`/`slow`）
- 现有 874 快速门禁保持绿；不静默重生成 golden。

### 8.3 配置 / 阈值 / 证据

- `configs/experiments/W7/0001…`（§5.1）
- `configs/thresholds.toml [W7.state]` / `[W7.hybrid]`
- `results/W7/`（run 目录 + summary + notes + verdict + reproduce）
- `results/INDEX.md` / `index.csv`（脚本生成）
- `figures/fig_w7_capacity.png`、`figures/fig_w7_crossover.png`

### 8.4 文档

- `docs/decisions.md` **D9**（proposed → accepted）
- `ROADMAP.md` §5 W7 实测 / §12 ✅ / §13 v31
- `docs/hybrid_attention.md` §4 填实
- `PITFALLS.md` 新增条目（池生命周期、bf16 GDN、paged 全注意力、per-row positions、fill 记账）
- `docs/w7-plan.md` **删除**

---

## 9. 生命周期

### 9.1 计划文件状态机

```text
draft ──用户评审 §2/§5/§9──> active ──全部门禁──> closing ──删除──> closed
                                │  ↑
                                │  └── blocked（门禁失败/等决策）
                                └───── abandoned（按 §8 裁剪/放弃）
```

### 9.2 状态转换

| 状态 | 进入条件 | 允许动作 | 退出 |
|---|---|---|---|
| `draft` | 文件创建；当前状态 | 编辑 §2/§5/§9；不改代码/不跑正式 run | 用户批准 → `active`；放弃 → `abandoned` |
| `active` | 用户批准；§2/§5/§9 冻结 | 实现、dry run、正式 run；改动走 §12 + 用户确认 | 门禁全绿 → `closing`；阻塞 → `blocked` |
| `blocked` | 门禁失败/等决策 | 记录 blocker、做分析、准备回退；**不伪造 ✅** | 解决 → `active`；持续失败且无法在门禁内解决 → 按 §7 裁剪或 `abandoned` |
| `closing` | 所有 run 已审、D9 accepted、阈值 calibrated | 回填 ROADMAP/hybrid_attention/PITFALLS、图、索引 | 删除 plan → `closed` |
| `closed` | 文件已删除 | 只读持久文档 | — |
| `abandoned` | 决定不做/砍掉 | 在 ROADMAP §13 记录原因与剩余工作 | 删除 plan |

> 本计划已于 2026-11-14 转 `active`；Step 0–3 已按用户“先完成到可以正式开始实验之前”
> 的指令执行完毕。用户已于 2026-11-14 明确指令开始 Step 4 正式 run；本文件随后回填执行结果。

**编辑规则**
- `draft`：可改范围/阈值/步骤；不改代码。
- `active`：范围冻结；阈值只能按 D9 修订流程改，且必须在正式 run 前；步骤可因门禁
  回退调整，但要在 §12 记一笔。
- `blocked`：只记录，不扩范围；超时按 §0.5/§7 处置。
- `closing`：只做文档/证据，不加功能。

### 9.3 请求状态生命周期（技术核心）

| 状态 | 拥有者 | 允许操作 | 失败处理 |
|---|---|---|---|
| `free` | `StatePool` | alloc | — |
| `reserved` | `RequestHandle` | 写 prompt/长度；prefill | rollback：归还 slot + blocks |
| `prefilled` | `RequestHandle` | 校验 KV/状态；activate | 丢弃半写状态，release |
| `active` | `Scheduler` | decode；更新 length | abort → release |
| `finished` | `Scheduler` | 记录终止原因 | release |
| `released` | `RequestHandle` | 幂等；不再写 | 重复 release 安全 |

不变量：见 §3.10；每个 formal run 的 summary 必须有 `leak=false`。

### 9.4 删除清单（`closing` → `closed`）

- [ ] D9 accepted，引用 run id；决策与后果写全。
- [ ] `[W7.state]` calibrated（basis run）；`[W7.hybrid]` calibrated 或 target 且数值未下调。
- [ ] ROADMAP §5 W7 实测、§12 W7 ✅、§13 修订记录。
- [ ] `docs/hybrid_attention.md` §4 填实（状态、P1–P3、交叉点、同显存并发）。
- [ ] PITFALLS 新增条目并映射测试。
- [ ] 新增测试 + 现有快速门禁绿；golden 未静默重生成。
- [ ] `results/W7/` run 已审；INDEX 由脚本重生成；over-budget/invalid 有说明。
- [ ] 图能从 summary 复现；脚本提交。
- [ ] 无文档/代码引用 `docs/w7-plan.md`；删除后 `git status` 干净。
- [ ] 计划文件删除；持久结论只在上方列出的位置。

### 9.5 持久产物

`D9`、ROADMAP §5/§12/§13、`hybrid_attention.md` §4、PITFALLS、`[W7.*]`、
tests、`results/W7/`、figures、`scripts/plot_w7.py`。
**计划文件本身不是证据；run id / test 才是证据。**

---

## 10. 已确认与待确认

### 10.1 已确认（用户 2026-11-14）

1. P1–P3 按“服务化复核”写；保留 W0 历史判定，P2 拆 memory/latency。
2. 容量主口径 = 同 1640 MiB 状态预算；权重不计入；总卡预算只作辅助。
3. W7 必做 A1–A3；E4 机制迁移**进入 W7**（用户明确）。
4. 正式数字统一 bf16；G0 与 HF 同 dtype 64/64；W6 fp32 golden 只作交叉核对。
5. **不设日历时间盒**，按门禁推进；**E4 不整块删**；plan 已转 `active`，但按用户要求**暂不执行**。

### 10.2 active 启动检查（执行前复核）

> 用户已于 2026-11-14 批准进入 `active`；按指示暂不执行。启动 Step 0 前按此表再确认一遍。

- [x] §2 的 D9 八条、P1–P3 定义、容量口径、bf16 口径无异议。
- [x] §5.1 run 矩阵与 §5.7 阈值无异议（尤其 2/4/6× 与 5×、1.10/1.15）。
- [x] §9 生命周期/删除清单无异议。
- [x] §3.7 的 seat 臂范围、§5.4 的 fill 旋钮、§5.7 阈值无异议；seat 仅作 baseline。
- [x] E4a/E4b 两个子实验范围可接受（E4c/E4d 已取消）。

> 评审通过后把本文件头改为 `active`，从 Step 0 开始；未通过则先在 §12 记修订，
> 不改代码。

---

## 11. 推进方式

- **顺序**：Step 0 → Step 6；每一步以门禁为出口，不按日历排期。
- **红灯**：先按 §0.5/§7 处理；可以并行/重排，但不跳门、不伪造 ✅。
- **正式 run**：由用户在干净树执行；agent 准备配置、dry run、初稿 notes/verdict。
- **范围**：只按 §0.5/§7 的触发条件裁剪；E4a/E4b 的 hybrid/seat 臂与 A1–A3 不砍。

---

## 12. 修订记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-14 | v1 | **draft 创建**。承接 W6 收口；用户确认 E4 进 W7，其余按推荐：P1–P3 服务化复核、1640 MiB 状态预算、bf16 G0、12–15 工作日。范围/阈值/生命周期待用户评审 §2/§5/§9 后转 `active`。 |
| 2026-11-14 | v2 | 按评审修正：0.6B block = 1.75 MiB（8 个 KV head）；`decode_batch` 散回 seat 行；D9-3 改为只加容器分支；setter 改为 `copy_` 写回；§5.4 预注册 fill 旋钮；E4 不砍；时间盒 15 天、**缓冲 0**；workspace=0 口径；E4c 基线改 W45-0012；P3 端点统一 8k；E1 `used` 从实际张量读。 |
| 2026-11-14 | v3 | 按用户决定**删除 hybrid seat 臂**（paged 是唯一生产路径）：E4b/E4c 改为 paged hybrid/pure；run 矩阵 0019–0023 重排；Step 2/协议/交付物去掉 seat 代码与测试；D9-3 只加 paged/varlen 容器分支。 |
| 2026-11-14 | v4 | 按用户决定**恢复 hybrid seat 基线**（不是生产路径）：6 层 `KvSlab` + `StatePool`，baseline-only；E4b/E4c 恢复 paged/seat 四臂，run 矩阵 0019–0027；Step 2 加 seat 分支与 `test_qwen35_seat.py`；G1b 加 seat==paged token id。 |
| 2026-11-14 | v5 | 取消 E4d（packed vs 逐请求 prefill）：W4.5 已用 E3 给出 packed/serial 结论，且 serial prefill 控制臂已删除；G2 回退同步收紧为“不实现逐请求 prefill 控制臂”，run 矩阵止于 0026。 |
| 2026-11-14 | v6 | 按用户决定：删除 E0a（G0 覆盖 64/64）；E1/E2 纯容量保留实测 3 条；E4a/E4b/E4c 纯臂引用同机历史（W45-0010/0011、W4-0002/0003、W45-0004/0012），只跑 hybrid 臂；矩阵删除 0001/0017/0018/0021/0022/0025/0026，空号不复用；G2 去掉逐请求 prefill 回退。 |
| 2026-11-14 | v7 | 按用户决定删除日历时间盒、时间表与日期性止损；§0.4/§11 改为门禁驱动。v1/v2 中的 12–15 工作日/缓冲 0 表述仅作历史记录，已作废。 |
| 2026-11-14 | v8 | 删除 E4c（paged decode 代价）；run 矩阵连续编号为 `W7-0001…0017`（17 个：hybrid 11 / pure 6）。保留 6 个 0.6B 同 trace 对照（E0c 1、E1/E2 3、E3 2），E4a/E4b 纯臂继续引用同机历史、不重跑。状态转 `active`，按用户要求暂不执行。 |
| 2026-11-14 | v9 | **Step 0 执行**：D9 写入 `docs/decisions.md`（proposed，未 accepted）；`configs/thresholds.toml` 新增并冻结 `[W7.hybrid]`（target，`[W7.state]` 未改），key/值由 `tests/unit/test_thresholds_w7.py` 锁定；G0 bf16/CUDA 与 HF 同 dtype 贪心 **64/64**（0.8B s01/m02 min margin 1.250e-1 / 0.0，m02 step 0 为 exact tie；0.6B s01/m02 0.0 / 6.250e-2；W6 fp32 golden 诊断 64/64）。未启动 Step 1。 |
| 2026-11-14 | v10 | **Step 2/3 实现完成**：packed group prefill（varlen 全注意力 + GDN segment）与 hybrid fill 跨步驻留接通；`bench/engines/hybrid.py` 可用，fill 读真实池账；17 条 `W7-0001…0017` 正式配置已准备并 dry-run 通过（**未跑正式 run，未创建 `results/W7`**）；`bench/thresholds.py` 增加 W7 evaluator（`p1_state_ratio_8k_min`、`capacity_speedup_512/2k/8k_min`、`hybrid_tpot_growth_max`、`pure_tpot_growth_min`、`p2_latency_material_ratio`、`memory_crossover_ctx_max`）与 `tests/unit/test_thresholds_w7_eval.py`；`scripts/plot_w7.py` 就绪。E3/E4a 的 B=8 臂取 `kv_blocks=4200`（8k 预留 520 块/请求，8×520=4160）避免准入节流污染 TPOT/P2/P3；E3 纯臂因此不按 1640 容量口径判，配置头已注明；E4b 改为同 1640 预算的 record-only 配对（0016 paged 48 slots/3970 blocks ≈1638.8 MiB；0017 seat 37 行 ≈1578.3 MiB），不再用会反转方向的 800 块/64 行。`_build_trace` 的 synthetic id 上限改为 `min(vocab_size, 151936)`，使 0.6B/0.8B 同 workload 的 `trace_sha256` 一致（prompt 路径不变；新增 `tests/unit/test_w7_synthetic_trace.py`）。状态保持 **active**。 |
| 2026-11-14 | v11 | **可开跑前复核**：17 条正式配置的 `[measure]` 对齐 §6.1（scenario warmup=3/repeats=5；fill 1/3）；E4b 改为同 1640 MiB 预算的 record-only 配对（paged 48 slots/3970 blocks ≈1638.8 MiB；seat 37 行 ≈1578.3 MiB），E3 纯臂 4200 blocks 明确为延迟探针、不参与容量判据；§10.2 启动检查全部勾选。全量快速门禁 **1008 passed**；G0 bf16/CUDA 仍 **64/64**；0.8B packed fill 实测 512/2k/8k = 62/37/14，pure = 23/6/1，配对 `trace_sha256` 一致、`state_total ≤1640`。仍未启动正式 run。 |
| 2026-11-14 | v12 | **Step 4 启动**：用户明确指令开始 W7 正式实验；提交 Step 0–3 实现与 17 条配置，在干净树上按 §5.1 顺序执行 `W7-0001…0017`。正式 run 结果、配对复核与 verdict 回填见后续修订。 |
| 2026-11-14 | v13 | **Step 4 完成 / Step 5 草稿**：17 条有效 run + `W7-0013` invalid；E3 B=8 的 `requests=8` 口径错误经用户确认，`W7-0018/0019`（N=24）重跑并 `supersedes`；`[W7.hybrid]` 数值未改。实测：E0b 8/8 token hash 全等；E1 误差 0.83/0.48/0.18%；容量 2.70/6.17/14.0×；P1 7.816×；P2-memory 512 混合小 2.27×；P2-latency B=1 材料差、B=8 口径依赖；P3 纯臂 +9.8% → 不可归因；E4a 1.498×、E4b 48/37=1.297×。内存交叉点 190.8 为解析值（无 128/192/256 扫描）。D9 仍 proposed，待用户复核后收口。 |
| 2026-11-14 | v14 | **交叉点实测补齐**：按 §5.5/§6.5 增跑 128/192/256（`W7-0020…0025`，hybrid/pure 各 3 条，同 trace、1/3、64 请求）。128 混合 20.13 > 纯 14.0；192 混合 20.88 < 纯 21.0；256 混合 21.63 < 纯 28.0；512 混合 24.63 < 纯 56.0 → 交叉区间 **[128,192]**，线性插值 **190.8** prompt token（预留口径 62.8），≤256。累计 23 有效 + 1 invalid；D9 仍 proposed。 |
