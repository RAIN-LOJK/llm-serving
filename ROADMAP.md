# minillm · 推理引擎路线图（W0–W8）

> **本文件是 W0–W8 的唯一权威来源**。历史学习主线（A–E）已归档，见 `docs/history.md` 与 git tag `legacy-learning-2026-08`；本文件是其中 D5 的独立扩写。
>
> **一句话**：把 minillm 从「玩具解码器」变成「**能同时装下 KV cache 和线性注意力状态的单卡 serving 引擎**」，
> 用同一套调度器测两代架构（Qwen3 全注意力 / Qwen3.5 混合注意力），给出「什么规模、什么上下文下
> 哪种优化才回本」的判据。
>
> **不训练任何模型**：载体是真实开源权重；"模型会不会说话"不是本项目的自变量。

---

## 0. 三条铁律

1. **测试先行，再动结构。** `tests/` 是规格。每步重构前先让护栏变绿，改完立刻重跑；回归红了不许进下一步
   （`PITFALLS.md` 里那十几条坑，**重写前必须逐条变成测试**，否则等于交学费重读一遍）。
2. **判据优先于数字。** 每个机制必须回答"**什么时候不划算**"；只报"快了多少"视为未完成。
3. **所有阈值是目标值与判失败线，不是实测结果。** W1 基线跑完后校准一次，写进 `configs/thresholds.toml`（并在 §13 记一笔）。

---

## 1. 载体

### 1.1 事实（2026-10 核查，`config.json` 实测字段）

**Qwen3.5 有小模型，但全系都是混合注意力。** 以 0.8B / 4B 的 `text_config` 为例：

| 字段 | 0.8B | 4B | 含义 |
|---|---|---|---|
| `num_hidden_layers` | 24 | 32 | 层数 |
| `layer_types` | LLLF × 6 | LLLF × 8 | **3:1 混合**：18 / 24 层线性注意力，6 层全注意力 |
| `hidden_size` | 1024 | 2560 | |
| `head_dim` | **256** | **256** | 显式 head_dim，且远大于 hidden/n_head |
| `num_attention_heads` / `num_key_value_heads` | 8 / 2 | 16 / 4 | GQA 4:1 |
| `attn_output_gate` | true | true | attention 输出带门控（**新算子**） |
| `rope_theta` | 1e7 | 1e7 | |
| `partial_rotary_factor` | **0.25** | **0.25** | **只有 1/4 维度做 RoPE** |
| `rope_parameters` | `mrope_interleaved: true`, `mrope_section: [11,11,10]` | 同 | **M-RoPE（多模态 3D 位置）** |
| `vocab_size` | 248320 | 248320 | 词表极大；`tie_word_embeddings: true` |
| `rms_norm_eps` | 1e-6 | 1e-6 | |
| `linear_conv_kernel_dim` | 4 | 4 | 线性层的短卷积核 |
| `linear_*_head_dim` / `num_*_heads` | 128 / 16key / 16val | 128 / 16key / 32val | 线性层头配置（与全注意力层不同） |
| `mtp_num_hidden_layers` | 1 | 1 | 自带 **MTP 头**（投机解码用，推理可跳过） |
| `vision_config.depth` | 12 | 24 | **原生多模态**（`Qwen3_5ForConditionalGeneration`） |
| `max_position_embeddings` | 262144 | 262144 | 256k |

依据：[Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)、[Qwen3.5-4B config](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json)、
[全系 75% 线性注意力](https://zhuanlan.zhihu.com/p/2011914556159582443)、[架构解析](https://huggingface.co/blog/mlabonne/qwen35)。

### 1.2 由此得出的三条判断

| 判断 | 说明 |
|---|---|
| **算力不再是问题** | 0.8B/2B/4B 都能进 4090（bf16 分别约 1.6G/4G/8G），甚至 M1 上能跑 0.8B。**量化重新变成自变量**，可以在 bf16 前提下测服务机制 |
| **架构成本是真问题** | 相比 Qwen3，需要新增 **5 类算子**：分块式 delta rule 线性注意力、短卷积、输出门控、partial-RoPE、M-RoPE。移植从 ~4 天涨到 **2–3 周** |
| **这恰恰是项目的价值** | 2026 年服务端的真问题就是「**KV cache 只剩 25% 的层**」——调度、缓存、显存管理全部要重写。nano-vllm / mini-sglang 都不覆盖这块 |

### 1.3 决定：双载体，先测后实现

| 角色 | 载体 | 阶段 | 理由 |
|---|---|---|---|
| **A · 骨架载体** | Qwen3-0.6B（纯全注意力） | W1–W4 | 架构简单（GQA + QK-Norm + head_dim），2–4 天内可对齐；用它把 `serve/` 骨架与测量框架跑通——**引擎层与架构无关** |
| **B · 目标载体** | Qwen3.5-0.8B（混合） | W0 测、W5–W7 实现 | 满足"用最新"；尺寸友好；混合结构是差异化来源 |
| **C · 对照臂** | A / B 互为对照 | W7 | 同量级、同代际、只差注意力机制 → **干净的受控对比** |
| 回退 | TinyLlama-1.1B | 仅当 A/B 都卡住 | C1 已对齐，可当"已知正确"参照物 |

> **为什么不是一上来就 All-in Qwen3.5：**
> ① 5 类算子 + 分块 delta rule，首次实现 2–3 周，失败则前 3 周零产出；
> ② `serve/` 层（调度、块管理、指标）与架构无关，先在简单载体上做出来，后面只是"多接一种状态"；
> ③ W0 会先用 HF 粗测混合架构的显存/延迟曲线——**数据说值得投入，再投入**（这正是你 `protocol.md` 里 D1 的做法）。

### 1.4 移植工作分解（W5–W7 的依据）

| # | 算子 | 难度 | 参考 | 验收 |
|---|---|---|---|---|
| 1 | 全注意力层（GQA, head_dim 256, 输出门控） | 中 | HF `qwen3_5` 源码 | 单层对拍 |
| 2 | partial-RoPE（只转 1/4 维）+ M-RoPE（文本退化为 1D） | 中 | 同上 | 位置编码单独测 |
| 3 | 短卷积（kernel 4, causal, 作用于 q/k/v） | 低 | 同上 | 因果性测试（改第 t 个 token 不影响 t−1） |
| 4 | Gated DeltaNet **decode 形式**（单步循环） | 中 | HF / `fla` | 逐步对拍 |
| 5 | Gated DeltaNet **prefill 分块形式**（chunked） | **高** | 同上 | 与 decode 形式逐 token 一致（自洽测试） |
| 6 | MTP 头 | 跳过 | — | 推理不需要；权重里有，加载时忽略 |

**关键自洽测试**（比对齐 HF 更重要）：分块 prefill 与逐步 decode 必须给出同样的 logits——
这条一旦通过，数值正确性基本就锁住了。

---

## 2. 现状盘点

| 已有（直接复用） | 位置 |
|---|---|
| 手写 RoPE / GQA / KV cache / SwiGLU / RMSNorm | `model/layers.py` |
| 采样、批量左 pad + EOS 提前退出 | `llm_utils/inference.py` |
| KV cache 一致性 / 左 pad / EOS / RoPE / 采样回归 | `tests/unit/`（pytest，快速门禁） |
| 实验工程纪律（自包含目录、config 快照、git 状态、人工结论） | `docs/experiments.md`；训练实现已归档，不复用 |
| run 契约 + runner + 生成索引 | `bench/run.py`、`scripts/exp_index.py` |
| 12+ 条已踩过的坑 | `PITFALLS.md` |

| 缺（本项目要补） | 阶段 |
|---|---|
| 真实权重加载与对拍框架 | W1 |
| 基准框架（负载 + 指标） | W2 |
| 请求级调度器（连续批） | W3 |
| 分页 KV（只覆盖全注意力层） | W4 |
| 混合架构（5 类算子） | W5–W6 |
| **双状态管理**（KV 块 + 循环状态池） | W7 |
| 跨代对照测量 | W7–W8 |

> 实验契约（run 目录 / 配置 / 评审 / 索引）见 `docs/experiments.md`；它是操作手册，§6 仍是测量口径的权威来源。

---

## 3. W0 · 双载体探测（3–4 天，先测后实现）

**目标**：用 HF 直接把两个模型跑起来，量出混合架构的收益曲线，**再决定 W5 要不要投入移植**。

### 3.1 动作

1. `scripts/probe_checkpoint.py`：读两个模型 `config.json`，落盘 `docs/carrier.md`
   （字段、层型分布、每 token 状态量的**解析式估算**）
2. `bench/engines/hf.py` + `bench.run`：HF 加载 → 上下文 512 / 2k / 8k / 32k → 记录
   每序列状态字节（KV / 线性状态 / 卷积）、进程峰值显存、prefill 延迟、decode 延迟（每 token）
3. 解析式估算（写进文档，作为待验证的预测）：

```
纯注意力每 token KV = 2 × n_kv_head × head_dim × dtype_bytes × n_full_layers
混合架构每 token   = 同上（但 n_full_layers 只有 1/4）
                   + 常数：n_linear_layers × n_v_head × key_head_dim × value_head_dim × dtype_bytes
```

对 0.8B 粗估：KV ≈ 12 KiB/token（6 层），固定状态 ≈ 18.84 MiB/序列（18 层 fp32 + 短卷积）。
→ 与纯注意力的 112 KiB/token 相比，**交叉点约 193 token**（`W0-0001` 实测 ≈193，prefill 口径）：
受测上下文（≥512）里混合始终更省，短上下文的额外成本来自固定状态与 prefill 启动开销。

### 3.2 待验证预测（W0 的判据）

| # | 预测 | 若成立 |
|---|---|---|
| P1 | 长上下文（≥8k）每序列显存比同量级纯注意力小 **5–8×** | KV 不再是主要显存项 |
| P2 | 短上下文（≤512）混合架构**反而更贵**（固定状态开销） | 报告里最漂亮的一张交叉图 |
| P3 | decode 延迟随上下文几乎**不增长**（线性层无 KV 读取） | 长上下文服务范式变化 |

### 3.3 决策点 D1（记录：`docs/decisions.md`）

- P1 成立（≥3×）→ **W5–W7 投入移植**，主线按 §5 走
- P1 不成立（<2×）→ 砍掉移植，Qwen3.5 只作 HF 侧对照，主线改纯注意力引擎
- Qwen3.5-0.8B 在 4090 上跑不动 → 退 2B 或纯注意力
- **判定（2026-10-22）**：P1 实测 7.80×（8k）/ 8.90×（32k）→ **接受移植**，见 `docs/decisions.md` D1。

**产出**：`docs/carrier.md`、`docs/fig_w0_crossover.png`（`scripts/plot_w0_crossover.py` 生成）、
`results/W0/0001-carrier-probe/`（summary + metrics）、`docs/decisions.md` D1

---

## 4. 重构纪律（S0–S7，不可跳步）

现状：`generate()` 是「整批同步 + 预分配 buffer」，**结构上不支持**连续批；
`KVcache` 是每层一个大张量，**不支持**分页。所以 decode 循环与状态管理必须重写——但**分层推进**：

| 步 | 动作 | 护栏（必须保持绿） | 动哪些文件 |
|---|---|---|---|
| S0 | 冻结规格：现有 pytest 快速门禁 + golden 对拍 | 全绿 | `tests/` 只增不改 |
| S1 | 真实权重加载（Qwen3-0.6B）+ 对齐 | 贪心 64 token 全等 | `scripts/`、`model/` |
| S2 | 抽出 `Engine` 接口（仍单请求） | 与 S1 逐 token 一致 | `serve/engine.py` |
| S3 | `Scheduler` 连续批 | 批内每个请求 == 单独跑 | `serve/scheduler.py` |
| S4 | `BlockManager` 替换 `KVcache` | paged vs 连续输出逐 token 一致 | `serve/block_manager.py` |
| S5 | 混合层接入（逐层替换，先全套 HF 权重） | **分块 prefill == 逐步 decode**（自洽） | `model/linear_attn.py` |
| S6 | 全模型对齐 Qwen3.5-0.8B | 贪心 64 token 与 HF 一致 | `scripts/align_hf.py` |
| S7 | `StatePool`：线性层状态池化 | 双模型在同一 `serve/` 下跑通 | `serve/state_pool.py` |

**铁律**：每步只改一件事；S4/S5/S7 的等价性测试**先写、后实现**（红 → 绿）。

---

## 5. 阶段明细

### W1 · 对齐与 golden（4 天）

- Qwen3-0.6B 加载进自己的实现：QK-Norm、显式 `head_dim`、`rope_theta`、`tie`、`attention_bias` 逐项接线
- `scripts/make_golden.py` → `tests/golden/`（prompt + 前 64 个贪心 token + 每层 hidden 抽样）
- **验收**：贪心 64 token 逐 id 全等；每层 `max|Δ| ≤ 1e-2`（2026-10-23 校准，依据见 §13）；末层 logits `max|Δ| ≤ 0.05`；batch 1/2/4 一致
- **失败处置**：5 天对不齐 → 逐层二分；仍不行退 TinyLlama，迁移成本写进报告

### W2 · 基准框架

- `bench/loadgen.py`：32 条固定 prompt（长度桶 16–64 / 256–512 / 1024–2048），closed-loop；到达率 λ 推到 W3（没有调度器时 λ 没有意义）
- `bench/metrics.py`：TTFT / TPOT / tokens-s / 峰值显存 / KV 占用（公式账 + 设备级显存对照；`torch.cuda.mem_get_info`，不 fork 子进程，见 P-27）
- 对照臂：HF `generate`（同模型、同 batch、同 `max_new_tokens`）
- **验收**：5 次重复的 TTFT/TPOT 抖动 ≤ 10%（**按长度桶**取每 prompt 的 CV 中位数，`overall` 会被 16–2048 的长度跨度污染）；`weights+KV` 占 torch 分配器峰值的 ≥90%（按 context 分组取中位数，设备级显存只作对照——它含 ~0.5 GB CUDA context，不能当分母）；decode ≥ 0.8× HF（按 prompt 配对取中位数，配对前校验 prompt hash / dtype / device / `max_new_tokens` / `out_tokens`）
- **返工已完结（2026-11-01）**：D3 驳回后注意力口径定死为 **prefill=逐序列 SDPA / decode=hand_baddbmm+KvSlab**
  （`docs/decisions.md` **D4**），旋钮全部删除。`W2-0004` 因两处**与注意力路径无关**的原因判 `invalid`
  （slab 三桶驻留让 `memory_match_ratio` 的分母含账外内存；HF 分母来自另一台 4090 实例）→
  改为单一 slab 后在**同一台机器**上重测两臂：`W2-0005`（HF，TPOT 21.43 ms）+ **`W2-0006`（自实现，21.43 ms）**，
  四条判据全 pass，decode **1.001×**（逐 prompt 0.985–1.031）、TTFT 三桶 0.89–0.91×、memory 0.957。
  **判据训：跨实例的 host-bound 比值不稳定**——同一份 HF 代码在两台 4090 上差 1.28×，
  凡引用「相对 HF 多少倍」必须绑定跑它的机器；`W2-0001`/`W2-0003` 的比值从此只作历史记录。
  随后修掉 slab 的两处记账缺陷（三桶驻留 → 单一 slab；整块分配 → 逐层分配 + 随请求释放），
  代码变动使 `W2-0006` 的 TTFT 失效，故由 **`W2-0007`** 取代为 W3 的配对基线：
  decode **1.013×**、TTFT 三桶 0.92–0.95×、memory **0.992**（全区间 0.979–0.992，与 ctx 无关）。

### W3 · 连续批处理

- **前置**：配对基线 **`W2-0007`**（固定注意力路径；HF 分母用同机的 `W2-0005`；
  `W2-0003`/`W2-0004`/`W2-0006` 是旧口径、无效配对或被取代，都不能当基线）；
  注意力口径见 `docs/decisions.md` **D4（accepted）**——**D3 已于 2026-10-31 rejected**（逐序列 attention
  实测为 28×B 次派发，B=8 时占 step 的 25–50%）；开工清单与 W3 专用参考实现见 `docs/w3-plan.md`
- 请求队列 + 每步准入/退出 + per-request 状态机；到达过程（poisson / gamma λ）——`loadgen.poisson_arrivals`
  的占位实现将由带 `trace_sha256` 的 `poisson_trace` 取代（见 `docs/w3-plan.md`）；对照静态批
- **验收**：高 λ 吞吐 ≥ 1.3×；p99 TTFT 分解为排队+计算；不同到达顺序输出一致
- **预期风险**：0.6B 上 Python 开销占比大，收益可能不显著——**把交叉点测出来就是结论**

### W4 · 分页 KV

- `BlockManager`（block_size 16/32 可配）、block table、按需分配/回收、物理块池（参考实现见 §14）
- **验收**：同显存 max concurrency ≥ 1.5×（ctx=2048）；碎片率 < 5%；paged vs 连续输出逐 token 一致
- 这一格复现 vLLM 论文的核心论点：**浪费来自"按最坏长度预留"**

> **W4 末检查点**：若 W0 的 D1 判定不投入移植 → 直接跳 W8 打包，主线结束（已是完整交付）。

### W5 · 混合层实现（一）：全注意力层改造

- head_dim 256、输出门控、partial-RoPE（1/4 维）、M-RoPE（文本退化）；只读本地 HF 源码（§14）
- **验收**：与 HF 单层对拍；因果性测试；改第 t token 不影响 t−1 的输出

### W6 · 混合层实现（二）：Gated DeltaNet

- 短卷积（kernel 4）→ decode 循环形式 → **分块 prefill 形式**；两套纯 torch 参考互拍（§14）
- **验收**：分块 prefill 与逐步 decode **逐 token 一致**；全模型贪心 64 token 与 HF 对齐
- **风险**：分块形式的 WY 表示是最难的一段。**止损**：若 W6 末只有 decode 形式可用，
  就接受"prefill 走逐步循环"的慢速实现（报告里注明性能代价），不硬啃

### W7 · 双状态管理与跨代对照

- `StatePool`：线性层状态（每请求常数大小）与 KV 块（分页）统一句柄与生命周期（唯一先例见 §14）
- 重跑 W3/W4 的机制，得到"混合架构下机制收益如何变化"
- **验收**：
  - [ ] 同一调度器下两种模型都能跑，输出正确
  - [ ] 每序列状态占用与 §3.1 解析式估算误差 < 20%
  - [ ] P1–P3 三条预测逐条判真假（`docs/hybrid_attention.md`）
  - [ ] 给出**交叉点**：上下文多长时混合架构开始占优

### W8 · 报告与判据矩阵

- 矩阵：架构（纯注意力 / 混合）× 模型档 × 上下文（512/2k/8k）× batch（1/8/32）
- 报告六节：口径 → 数值对拍 → 各机制收益边界 → **负结果专章** → 跨代对照 → 复现
- **可选（仅余力）**：权重量化（int8/int4），指标同 v1 计划

---

## 6. 测量协议（写死）

| 项 | 规定 |
|---|---|
| 预热 | 每配置先跑 3 次丢弃 |
| 重复 | 5 次，报中位数 + p95 |
| prompt 集 | 32 条，长度桶固定，落盘 `bench/prompts.json` |
| 生成 | greedy，`max_new_tokens=128` |
| 计时 | CUDA：`torch.cuda.Event` + `synchronize()`；MPS：`perf_counter` + `torch.mps.synchronize()`；TTFT 含排队 |
| 显存 | CUDA：`max_memory_allocated()` + `torch.cuda.mem_get_info()` 采样的设备级数字；MPS：`driver_allocated_memory()`；区分权重 / KV / 线性状态 / 激活 |
| 环境 | GPU、驱动、torch/CUDA、transformers 版本、git commit 写进结果行 |
| 设备 | `dev-mps`（M1：正确性/冒烟）与 `measure-cuda`（4090：正式数字）分表；**不跨设备比较** |
| 方差 | 同进程同顺序；跨配置重开进程 |
| 口径 | **HF 与自实现数字不混用**：比值可以，绝对值分开报 |

---

## 7. 报告结构与判据

1. 问题与口径（为什么两代载体、为什么先纯后混）
2. 数值对拍（可信度地基）
3. 连续批与分页 KV 在纯注意力下的收益边界
4. **负结果专章**：哪些优化在这个规模下无效、为什么
5. 混合注意力：P1–P3 判定 + **显存交叉点** + 对调度/缓存的含义
6. 复现说明（一条命令一档）

---

## 8. 失败与裁剪

| 触发 | 处置 |
|---|---|
| W0 的 P1 < 2× | 砍移植，Qwen3.5 只作 HF 对照，主线收在 W4 |
| W1 五天未对齐 | 退 TinyLlama，迁移成本写成报告一节 |
| W3 收益 < 1.2× | 不硬调，记录负结果，转 W4 |
| W6 分块 prefill 啃不动 | 用逐步循环实现，报告注明性能代价 |
| W7 时间不够 | 只做 P1 + 交叉点，P2/P3 写分析不实测 |
| 任一周租卡无数字 | 立即停租，改本地开发 |
| 时间被实习挤压 | W0–W4 已是完整交付（引擎 + 对拍 + 两张图） |

---

## 9. 简历映射（数字留空，实测后填）

> **统一 serving 引擎：跨两代注意力架构的单卡推理实现** 2026.10 — 2027.01
> - 基于手写 PyTorch 解码器实现单卡 serving：连续批处理、分页 KV cache、双状态管理；
>   与 HF 逐 token 数值对齐（贪心 64 token 完全一致，每层 `max|Δ|` `<实测>`）
> - 实现 Qwen3.5 的 **3:1 混合注意力**（Gated DeltaNet 分块 prefill + 逐步 decode、短卷积、输出门控、partial/M-RoPE），
>   分块与逐步两条路径逐 token 自洽
> - 统一状态管理：分页 KV 覆盖 25% 全注意力层 + 常数大小状态池覆盖 75% 线性层；
>   相对静态批处理吞吐 `<实测>`×，同显存最大并发 `<实测>` → `<实测>`
> - 跨代对照：同量级下测得混合架构每序列显存 `<实测>`×、显存**交叉点**约 `<实测>` token，
>   并给出「KV cache 退场后调度与缓存如何改写」的判据

技能栏：`推理工程（PagedAttention / 连续批处理 / 混合注意力状态管理 / KV cache）`、
`Gated DeltaNet（分块并行 + 循环形式）`、`性能测量（TTFT/TPOT、P50–P95、memory/compute-bound 判定）`、
`数值对拍与回归测试驱动重构`

**禁止措辞**：不写「比 vLLM 快」。写「实现并测出收益边界」。

---

## 10. 与 D / E 阶段的关系（代码复用，不是口号）

| 本项目产出 | 在 D / E 复用在哪 |
|---|---|
| `BlockManager` / `StatePool` | E4「复用参考帧特征缓存，省 ~50% 前向」——块管理与引用计数直接可用 |
| 混合层实现（delta rule + 短卷积） | **D2（CSA / HCA 序列压缩）的地基**：线性注意力与压缩注意力的实现手艺同源 |
| 分块 prefill vs 逐步 decode 的自洽测试法 | D 阶段所有算子实现的通用验收手段 |
| 跨代对照的测量结论 | E 阶段选型：流场解码器该用什么注意力 |

> 附带发现：Qwen3.5 自带 **MTP 头**（`mtp_num_hidden_layers: 1`）。若将来要做投机解码，
> 可以直接用它的 MTP 头当 draft，不必另训小模型——这是 W8 之后的可选项。

---

## 11. 目录与命令

```text
minillm/
├── model/            # layers.py（+输出门控/partial-RoPE）、linear_attn.py（新增）
├── llm_utils/        # inference.py（过渡；S2 后由 serve/engine.py 接管）
├── serve/            # engine / scheduler / block_manager / state_pool
├── bench/            # run / metrics / loadgen / prompts.json
├── configs/          # experiments/ + thresholds.toml
├── scripts/          # env_snapshot / exp_index / probe_* / make_golden / align_hf
├── tests/            # unit/ + integration/ + golden/
├── results/          # W0..W8/NNNN-slug/（run 目录）+ INDEX.md
├── docs/             # experiments / decisions / environments / carrier / hybrid_attention
├── reports/
├── archive/          # 仅 README 指向 legacy tag
├── README.md
├── AGENTS.md
├── PITFALLS.md
├── ROADMAP.md
└── requirements*.txt / pytest.ini
```

```bash
# 快速门禁（不需要 HF 权重）
python -m pytest -m "not slow and not hf"

# 环境快照
python scripts/env_snapshot.py

# 一个 run（W0 示例；正式跑前 pin revision）
python -m bench.run --config configs/experiments/W0/0001-carrier-probe.toml

# 索引
python scripts/exp_index.py
python scripts/exp_index.py --check
```

---

## 12. 进度

| 阶段 | 内容 | 状态 | 证据 run |
|---|---|---|---|
| W0 | 双载体探测 + D1 决策 | ✅ | `W0-0001` |
| W1 | Qwen3-0.6B 对齐 + golden | ✅ | `tests/golden/qwen3-0-6b/0001` |
| W2 | 基准框架 | ✅ | `W2-0007`（HF 分母 `W2-0005`） |
| W3 | 连续批处理 | ⬜ | — |
| W4 | 分页 KV | ⬜ | — |
| W5 | 混合层（一）：全注意力层改造 | ⬜ | — |
| W6 | 混合层（二）：Gated DeltaNet | ⬜ | — |
| W7 | 双状态管理 + 跨代对照 | ⬜ | — |
| W8 | 报告与判据矩阵 | ⬜ | — |
| W9–W11 | 打包：README / 30 秒 demo / 投递 | ⬜ | — |

> `results/INDEX.md` 是全部 run 的索引；阶段结束时把证据 run id 回填到本表。

---

## 13. 修订记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-21 | v1 | 初稿：误以为 Qwen3.5 最小 27B，把混合注意力放到 W7 做对照 |
| 2026-10-21 | v2 | **更正**：Qwen3.5 有 0.8B/2B/4B/9B，但全系 3:1 混合 → 改为「骨架载体 + 目标载体」双轨；W0 先测后实现（D1 决策点）；删除投机解码，量化降为可选 |
| 2026-10-21 | v3 | **项目重组**：旧训练线删除（文本快照 tag `legacy-learning-2026-08`）；实验契约改为不可变 run 目录 + 生成索引（`docs/experiments.md`）；测试改 pytest 自包含护栏；设备分层 `dev-mps` / `measure-cuda`；W0 双载体 revision 已 pin |
| 2026-10-22 | v4 | **W0 完成**：`W0-0001` 实测 P1 成立（8k 7.80× / 32k 8.90×）、P2 不成立（交叉点 ≈193 token）、P3 数值成立但不可归因；修正 §3.1 交叉点（1.5k → 193）、产出清单与 `probe_memory.py` 的描述；D1 接受 |
| 2026-10-23 | v5 | **W1 完成**：Qwen3-0.6B 逐层对齐（4 prompt × 贪心 64/64 id，logits `max|Δ|` 2.0e-5，每层 9.8e-4–1.95e-3）；阈值 `1e-3 → 1e-2`——首 token 有 attention-sink 离群值（\|x\|≈7e3），fp32 一个 ulp 就有 9.8e-4，原阈值等于要求"≤1 ulp"；golden 改用 hook 采块输出（transformers 5.x 的 `hidden_states[-1]` 是 post-norm，见 P-16）、`rope_theta` 改从 `rope_parameters` 读（P-17） |
| 2026-10-24 | v6 | **W2 工具链**：`impl="minillm"` 适配器、`cuda_event` 计时接线、`[accept]` 真求值；判据口径三处修正——CV 改按长度桶（`overall` 混了 16–2048 的长度跨度）、显存对齐改为 `weights+KV` / **torch 分配器峰值**（`nvidia-smi` 含 ~0.5 GB CUDA context，不能当分母，只作对照）、配对 baseline 先校验 prompt hash/dtype/device/`max_new_tokens`/`out_tokens`，不符记 `invalid`；另修 RoPE 每层一张 40960 行表（-280 MB，P-18）与 `logits_to_keep`（P-13） |
| 2026-10-24 | v7 | **W2 完成**：`W2-0001`（HF 参考臂，TPOT 16.78 ms）与 `W2-0002`（自实现，17.10 ms）四条判据全 pass——decode **0.976×** HF、TTFT 持平（short/medium 1.00–1.01×）、桶级 CV 6.8%/3.6%、memory 0.988；**长上下文 prefill 1.49× 慢**（物化 `T²`、无 SDPA），记为负结果并作为 W3/W4 结论的前提；`[W2.bench]` 标 calibrated（basis `W2-0002`），memory 0.9 只对 context 中位数成立（最长 1517 token 为 0.889） |
| 2026-10-31 | v9 | **D3 驳回 + 参考实现索引**：逐序列 attention 实测为 `28×B` 次派发（B=8 占 step 25%、B=32 占 50%，与 ctx 无关），D3 的 `<15%` 判据在 B=8 即不成立、且本机 torch 2.8 无可用的「一次 ragged 调用」（nested 非同构 / flex eager 慢 20–50× / flash-attn 未装）→ D3 改 `rejected`，W3 正式 run 等 W2 返工后的新 ADR；新增 §14 跨阶段「可抄代码」索引（来源 `docs/benchmark-resources.md`，W3 专用参考在 `docs/w3-plan.md`） |
| 2026-10-25 | v8 | **W2 收尾（注意力路径）**：`W2-0003` 换上 SDPA 后四条判据全 pass，decode **1.113×** HF、TTFT 三桶 0.86–0.90×；`W2-0002` 的 long-prefill 1.49× 归因为 attention 物化 `T²`（31.8→19.5 ms）。**pad 会让 flash/efficient 退出、只剩 math（6.3×）**，故立 D3：长度方向禁止 pad、线性层批 + attention 逐序列、全仓一条注意力路径、不引入 varlen kernel，CUDA graph 留到 W4 之后；`W2-0002` 被 `W2-0003` 取代 |
| 2026-10-31 | v10 | **D4：注意力路径定死**。prefill（T>1）只用逐序列 SDPA（无 mask 才有 flash）；decode（T==1）只用 `hand_baddbmm` + `KvSlab`（一次调用覆盖整批，mask 融进 score GEMM）；**不做 sdpa+slab**（SDPA 拿 mask 即退 math，slab 的收益正好被抵消）。删掉 `attn_prefill/attn_decode/[engine].decode` 三个旋钮、`Caps` 能力声明与封闭注册表（「不回退」改由构造保证）、`slab_graph` 运行路径（`serve/graph.py` 保留为 parked）；`hand_legacy` 移到 `tests/reference/` 只做 oracle。**W2 回到返工中**：`W2-0003` 是旧口径，不能当 W3 的配对基线 → 新开 `W2-0004`（`supersedes`）重建锚点，清单见 `docs/w2-rework.md`。配对校验拆成通用键 / 机制键两层，否则自实现臂与 HF 臂的 `decode_speed_vs_hf_min` 会被误判 invalid |
| 2026-11-01 | v11 | **W2 完成（返工后）**：`W2-0004` 判 `invalid`——两处失败都与注意力路径无关（slab 三桶驻留把账外内存放进 `memory_match_ratio` 的分母；HF 分母取自另一台 4090 实例）。改为单一 slab（`ce0af17`）后同机重测：`W2-0005`（HF）与 **`W2-0006`（自实现）**四条判据全 pass——decode **1.001×**（逐 prompt 0.985–1.031，32 条全在 ±3%）、TTFT 三桶 0.89–0.91×、memory 0.957、CV 1.22%/0.42%；`[W2.bench].basis_run` → `W2-0006`。**判据训：跨实例的 host-bound 比值不稳定**（同一份 HF 代码两台 4090 差 1.28×），引用「相对 HF 多少倍」必须绑定机器；`W2-0001`/`W2-0003` 的比值降级为历史记录。`W3-0001` 的基线改为 `W2-0006` |
| 2026-11-01 | v12 | **W2 收尾（slab 生命周期）**：`W2-0006` 判 pass 后发现 slab 记账有两处缺陷——① 三桶缓存让 `memory_match_ratio` 的分母含账外内存（已在 `ce0af17` 改单一 slab）；② 交接时 slab 整块分配、且随请求保留，同一份 KV 两次同时在场。②修完才彻底：逐层分配 + 逐层拷 + 逐层释放 + 请求结束释放 slab（`d8361d4`），ctx1517 峰值 1534.6→1348.0 MB、ratio 0.859→0.978；**两处必须同时改**（各自单独无效，2×2 实测）。代价是每请求重建 28 层、TTFT 21.25→21.85 ms。代码变更使 `W2-0006` 不再描述 shipped 状态 → 新开 **`W2-0007`**（`supersedes`）作为 W3 配对基线：四条判据 pass、decode 1.013×、TTFT 0.92–0.95×、memory 0.992（全区间 0.979–0.992）、CV 1.12%/0.44%；`[W2.bench].basis_run` → `W2-0007`。**留待 W3**：真实 serving 的 KV 必须跨请求复用（每并发槽一块），届时 `memory_match_ratio` 的分子需覆盖所有活跃槽的预留，否则重演 `W2-0004` 的错配 |

---

## 14. 参考实现索引（跨阶段，只列「可抄代码」）

> 来源：`docs/benchmark-resources.md`（2026-10-30 调研）里的**代码类**资源，按阶段裁剪。
> 规则：只作机制参照与对拍；直接复制代码需保留原许可证（下表均为 Apache-2.0 / MIT）。
> 路径以本机 `transformers 5.20.2` / `fla 0.5.2` 为准，升级后需重新定位（先 `python -c "import transformers;print(transformers.__version__)"`）。
> 数据类资源（ShareGPT / LongBench / RULER / BurstGPT）不在本表：它们只影响负载真实度，不影响任何判据。
> W3 的调度/到达过程参考不在这里，见 `docs/w3-plan.md`。

| 阶段 | 参考实现 | 位置 | 抄什么 | **不抄什么** |
|---|---|---|---|---|
| W2/W3 | `model/attention/base.py`（本项目自建，非外部参考） | 本仓库 `model/attention/` | 注意力后端的契约：**两个入口**（prefill / decode）与「不回退」的构造性保证（`T>1` → SDPA，`T==1` → `hand_baddbmm`，见 D4）；`model/attention/__init__.py::provenance()` 写进 manifest/summary；`hand_legacy` 移到 `tests/reference/` 冻结为**数值 oracle**（与 git `26b268d^` 逐 bit 相同） | 不做可选后端的注册表 / 能力声明（D4 定死后是死重量）、不做 entry-point 自动发现、不允许静默回退、不让 oracle 进性能结论 |
| W3 | nano-vllm `engine/{scheduler,sequence,llm_engine}.py` | `github.com/GeeeekExplorer/nano-vllm`（~1200 行） | `Sequence` 字段与调度主循环的**接口形状**（为 W4 的 block table 预留） | preempt / chunked prefill / CUDA graph；W3 实现细节见 `docs/w3-plan.md` |
| W4 | nano-vllm `engine/block_manager.py` | 同上 | 物理块池、block table、free list、按需分配/回收 | 其 attention/kernel 部分 |
| W4 | LightLLM `kv_cache_mem_manager/{allocator,mem_manager}.py` | `github.com/ModelTC/LightLLM` | 纯 Python、token/块级分配器（比固定 block 更轻），与逐序列 attention 的 gather 更贴 | 模型专属子类（deepseek/glm…）、多进程架构 |
| W4 | PagedAttention（SOSP'23） | `arxiv.org/abs/2309.06180` | block_size 16/32、block table、碎片率定义 | 自定义 CUDA kernel（本项目不写 kernel） |
| W5 | `transformers/models/qwen3_next/modular_qwen3_next.py` | `site-packages/transformers/models/qwen3_next/` | 全注意力层装配、输出门控（`chunk(q_proj(x),2)` + `sigmoid(gate)`）、`Qwen3NextRMSNormGated`（norm before gate） | MoE / 多卡分支 |
| W5 | `transformers/models/qwen3_5/modular_qwen3_5.py` | `site-packages/transformers/models/qwen3_5/` | 0.8B 的覆盖：`Qwen3_5GatedDeltaNet`、partial-RoPE（`dim=int(head_dim*partial_rotary_factor)` ⇒ inv_freq 只按 64 维算）、`rope_parameters` 读取、mrope_section | 视觉塔 |
| W6 | `qwen3_5/modeling_qwen3_5.py`: `torch_chunk_gated_delta_rule`(L287) / `torch_recurrent_gated_delta_rule`(L424) | 同上 | **「分块 == 循环」两个端点的纯 torch 实现**；`chunk_size=64` 默认；`conv_state` 长度 = kernel−1 | 其 Triton/CUDA 分支 |
| W6 | fla `fla/ops/gated_delta_rule/naive.py` | `site-packages/fla/`（0.5.2） | **第二套独立实现**，与上一行互拍；状态布局 `[K,V]`、g 在 log-space、beta 已 sigmoid | `chunk.py` / `fused_recurrent.py`（Triton kernel） |
| W7 | LightLLM `common/state_cache_manager.py` + `qwen3next_mem_manager.py` | 同上 | 唯一把「KV 块池 + 线性常数状态池」放进同一 manager 的开源实现：常数状态池的句柄与生命周期 | 其调度器与 kernel |
| W7 | vLLM V1 `v1/core/kv_cache_coordinator.py` | `github.com/vllm-project/vllm` | 多 cache type 的协调抽象（概念） | 整个 V1 引擎（分布式 / CUDA graph / 后端全家桶） |

两件已核实、可省弯路的事：

- `olmo_hybrid/modeling_olmo_hybrid.py` 里的同名 `torch_*_gated_delta_rule` 与 `qwen3_5` 版**逐行相同**，不能当第三个独立对拍端点。
- 0.8B 的 `config.json` 里**没有** `use_qk_l2norm_in_kernel`（走 `Qwen3NextGatedDeltaNet` 的类默认），`partial_rotary_factor` / `mrope_section` 也**不在顶层**（在 `rope_parameters` 内）——与 P-17 同类坑。
