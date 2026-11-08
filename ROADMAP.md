# minillm · 推理引擎路线图（W0–W8，含 W4.5）

> **本文件是 W0–W8（含 W4.5）的唯一权威来源**。历史学习主线（A–E）已归档，见 `docs/history.md` 与 git tag `legacy-learning-2026-08`；本文件是其中 D5 的独立扩写。
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
| **A · 骨架载体** | Qwen3-0.6B（纯全注意力） | W1–W4.5 | 架构简单（GQA + QK-Norm + head_dim），2–4 天内可对齐；用它把 `serve/` 骨架与测量框架跑通——**引擎层与架构无关**；W4.5 的 packed prefill / 内核也先在它上面做 |
| **B · 目标载体** | Qwen3.5-0.8B（混合） | W0 测、W5–W7 实现 | 满足"用最新"；尺寸友好；混合结构是差异化来源 |
| **C · 对照臂** | A / B 互为对照 | W7 | 同量级、同代际、只差注意力机制 → **干净的受控对比** |
| 回退 | TinyLlama-1.1B | 仅当 A/B 都卡住 | C1 已对齐，可当"已知正确"参照物 |

> **为什么不是一上来就 All-in Qwen3.5：**
> ① 5 类算子 + 分块 delta rule，首次实现 2–3 周，失败则前 3 周零产出；
> ② `serve/` 层（调度、块管理、指标）与架构无关，先在简单载体上做出来，后面只是"多接一种状态"；
> ③ W0 会先用 HF 粗测混合架构的显存/延迟曲线——**数据说值得投入，再投入**（这正是你 `protocol.md` 里 D1 的做法）。

> **W4.5 的载体边界**：只用 A（Qwen3-0.6B）做 packed prefill + block-table decode；B（Qwen3.5-0.8B）的 packed prefill 在 W6 接同一接口，不在 W4.5 重复造。

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
| packed prefill + block-table decode（注意力路径 v2） | W4.5 |
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

**产出**：`docs/carrier.md`、`figures/fig_w0_crossover.png`（`scripts/plot_w0_crossover.py` 生成）、
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

### W3 · 连续批处理（2026-11-03 完成）

- **前置**：配对基线 **`W2-0007`**（固定注意力路径；HF 分母用同机的 `W2-0005`）；
  注意力口径见 `docs/decisions.md` **D4（accepted）**；负载/调度口径见 **D5（accepted）**
  （trace/席位/饱和/吞吐双口径）。D3 已于 2026-10-31 rejected。
- **实现**：请求队列 + per-request 状态机 + 每步准入/退出；`bench/loadgen.py::poisson_trace`
  （gamma + 总时长归一化）带 `trace_sha256`；静态/连续两臂共用同一个 `Scheduler` 与
  `decode_batch`，只差 `policy`；席位 = 一块 `KvSlab` 的 `max_batch` 行（准入 prefill 写穿、
  退出 freeze 不回收，回收留给 W4 块池）。
- **验收（实测）**：
  - **C1 吞吐 2.01×**：`W3-0011`（continuous）vs `W3-0010`（static），B=8 mixed，同 trace
    `33bc2397…`、同 max_batch=8；311.43 / 154.94 tok/s，target 1.30。
  - **C2 pass**：`ttft_p99_decomposed`，逐请求 `ttft == queue + prefill`，最大偏差 1.8e-12 ms。
  - **C3 pass**：三对配对 run 的 request-row `token_sha256` 全等（32/89/89 keys，0 mismatch）
    + 机制层单测（批 decode == 逐序列、冻结席位不改其它行）。
  - **E1**：P1 **7.94×**（ctx511）/ **7.90×**（ctx2048）；P2 **3.78×**（高于预注册 2.5–3×）；
    R1 未触发（`step_ms` 1.06×/1.10×）。
  - **负结果/边界**：uniform 1.023×（P3 如设计，负结果）；B=32 mixed 1.606×（低于 P4 的
    1.8–2.2×，record-only）；B=32 × ctx2048 留给 W4。
- **教训**：E1 的 ctx512 合成 prompt 在第 3 个 token EOS，`W3-0003/0004/0005` 判 invalid，
  改用 ctx511 重跑（P-31）；R1 的 `step_ms` 含 prefill，判读改用稳态 decode 步（D5 后果 10）。
- **证据 run**：`W3-0001/0002`（E0）、`W3-0017–0019` + `0006/0007`（E1）、
  `W3-0010/0011`（C1/C2/C3）、`W3-0008/0009`（P3）、`W3-0012/0013`（B=32）。

### W4 · 分页 KV

- `BlockManager`（block_size 16/32 可配）、block table、按需分配/回收、物理块池（参考实现见 §14）
- **验收**（口径见 `docs/decisions.md` **D6**）：
  - 同**总预算**（KV + workspace = 1640 MiB）、**W3-like 混合长度** trace 下，fill-only 的
    peak concurrency **≥1.5×**（<1.2× 记负结果）；
  - **uniform 2048 只作负对照**（预测 0.8–1.0×），只报不判——它不是失败线；
  - **internal frag ≤5%**（只算最后一块没写满；reservation slack 单独报，不判）；
  - paged vs 连续输出**逐 token 一致**（E0 对 golden）；
  - B=8、两边 `lpad=1664` 的 decode `step_ms` 只记录；**>1.5× 撤回吞吐结论**。
- 这一格复现 vLLM 论文的核心论点：**浪费来自"按最坏长度预留"**，收益只来自长度异质性。
- 容量比值随预算单调下降（真实 trace、预算反解：1200/1456/1640/2400 MiB → 4.33×/3.50×/**3.22×**/3.54×；
  ≥2400 后 paged 撞到请求数或 `max_batch`，不再是显存受限），所以**报告主图是容量曲线**，C1 只是 1640 那条竖线。
- E1 实机门禁：`sim_paged_capacity.py --trace <本 run 的 metrics.jsonl> --paged-blocks 800 --expect-concurrency <实机数>`，
  期望 28 ±1（预算反解的 29 不是配置块数下的答案）；实测 **28（delta 0）**。
- **W4 实测（2026-11-06，核心代码 commit `9dc5211`；post-fix 0009 在 `b8fa5c1`）**：
  - **C1 pass · 3.111×**：`W4-0002` seat **9** 行 vs `W4-0003` paged **28** 条
    （同 trace `2c0471…`、同预算上限、800 块 bs16）；paged `kv_total = 1601.97 MiB ≤ 1640`
    （其中 workspace 201.97 MiB 是 fill 的公式估算，不是实测）；同 trace 模拟门禁 `28±0`。
    uniform-2048 负对照 **1.00×**（模拟，只报不判）。
  - **C2 pass · 0.55%**：`W4-0005` internal frag 0.55%（target 5%）；reservation slack
    按 D6 时间加权 5.01%（只报不判；runner summary 的 0.68% 是末步瞬时值）。
  - **C3 pass · 全等**：`W4-0001` E0 对 golden **64/64**；`W4-0005` 的 40/40 request
    `token_ids` 与 `W4-0004` 逐 id 相同（8 prompt × 5 repeats）。
  - **C4 负结果 · 1.671×**：paged decode step **35.63 ms** vs seat **21.32 ms**；
    `lpad=1664` 子集 1.673×，逐重复 1.661–1.676。>1.5× → **撤回吞吐结论**；
    **E3 原未跑**（plan §4.3 只在 A/B 都通过时跑），后按用户指示补跑、仅 record-only（见下）。
    同并发多付 workspace **52.42 MiB（+3.6%）**，这是代价，不是 C1 的收益。
    **prefill 也不便宜（pre-fix）**：paged 551.8 ms vs seat 163.0 ms（**3.39×**），
    使 makespan 比 1.773×；C4 仍只按 decode step 判。修复后的 prefill 见下面的 0009。
  - **E3 record-only（post-hoc，pre-fix）**：`W4-0006` seat **333.44 tok/s** vs `W4-0008`
    paged 800 **325.19 tok/s = 0.975×**（同 trace、两臂都 `saturated`）；decode-only
    **1.103×**，但 prefill 3.39× 把整体拉回略负。`W4-0007`（820 块）动态峰值
    `kv_total = 1661.8 > 1640`（R5），只保留为 over-budget 记录；`W4-0008` 是有效的
    pre-fix 同预算臂（峰值 1626.8），由 `W4-0009` `supersedes`。
  - **E3′ post-fix（`W4-0009`，slot-mapping）**：`tok_s = 358.18` vs `W4-0006` **333.44**
    = **1.074×**，两臂 `saturated`；峰值 `1626.8 ≤ 1640`。decode-only 1.094×；
    E3 per-request prefill 中位数 **76.4 → 23.0 ms**（seat 21.4）。prefill 修复把旧的
    0.975× 翻成 +7.4%，但 **C4 仍是 1.671×**（本次只改 prefill，decode 未动），
    所以 W4 的 C4 负结果/不报吞吐不变；0009 是 post-hoc post-fix 实验，不是 C4 pass。
  - 图：`figures/fig_w4_capacity.png`、`figures/fig_w4_memory.png`、`figures/fig_w4_step.png`。
- **C4 口径偏差**：plan 假设两边 `lpad=1664`；实机 paged 的 `lpad` 是**本步最长活跃长度**，
  实际为 1536/1600/1664。按 1664 子集复核仍为 1.673×，所以结论不依赖该偏差；
  后续引用必须写实际桶，不能写「全程 1664」。
- **注**：原条款的 `ctx=2048` 被读成 uniform，测不到「按最坏长度预留」的浪费，与同段论点矛盾；
  改法见 §13 v14。

> **W4 末检查点**：D1 已接受（2026-10-22）→ 主线先走 **W4.5**（注意力路径 v2），再进 W5；若 D1 为否，则直接跳 W8 打包（W0–W4 已是完整交付）。

### W4.5 · 注意力路径 v2：packed prefill + block-table decode（✅ 2026-11-06）

- **为什么单开一格**：W4 的两个问题都是**路径级**——① decode 的 gather 是 dense `hand_baddbmm` 的必然代价（D6 / P-32），C4 1.671× 使 paged 只能报容量；② prefill 逐请求串行（`Scheduler` 对每个 admitted request 逐个 `prefill_into_*`），`ttft == queue + prefill` 只在串行下成立。只换 decode 内核 = `W4-0010`，不够开阶段；**packed prefill 才是能力变化**。
- **目标**：把 attention 路径从「逐请求 SDPA prefill + gather decode」升级为「**packed whole-prompt prefill + Triton block-table decode**」；seat/paged 两臂共用同一接口；不 chunk、不与 decode 混批。
- **范围**：
  - **packed prefill**：FA2 `flash_attn_varlen_func`（不读 paged cache，因此不受 256 page 约束）、无 pad、`cu_seqlens`/`max_seqlen`；seat/paged 两臂；scheduler 按 `max_prefill_tokens` FIFO 打包，超 budget 的 prompt 单独 prefill。
  - **paged decode**：自写 Triton kernel，直接吃 `(n_blocks, bs, H_kv, D)` + `block_table` + `seqlens`；bs16/32、GQA、D=128/256、bf16；保留新 token 的 slot 写入；生产路径不读 gather / `lpad` / 零块 / mask-bias。**serial prefill 与 gather decode 两个控制臂均已删除**（收口后 step 1/2，见 v24/v25）；gather/lpad/零块/mask-bias 的纯稠密 oracle 在 `tests/reference/paged.py`，只被测试导入。
  - **测量契约**：`queue_ms` / `pack_wait_ms` / `prefill_shared_ms` + 每请求归因；`ttft == queue + pack_wait + prefill_shared` 为唯一口径（逐请求 serial 臂已删除）。D7 冻结。
- **门禁（Day 1–2，go/no-go）**：① FA2 varlen 在 0.6B（D=128、pack 2/4/8）对拍逐序列 SDPA，数值 / 无 pad / TTFT 过线；② Triton decode 对拍 dense oracle，B=8 / L=1664 的 `step_ms` ≤ seat。**两个都过才开阶段**；①/② 任一不过，按 §8 降级（decode 走 `W4-0010`，或只做 packed prefill），不硬写 prefill kernel。
- **验收**（阈值在 `configs/thresholds.toml [W45.paged]` 预注册，收口时未改值）：
  - P1 packed == serial（token ids；logits 容差）；无 pad。（历史判据：serial 控制臂已删除，证据在 `W45-0001…0012` 与 git 历史。）
  - P2 mixed load 下 TTFT p99 / tok/s 改善 ≥ 预注册线。
  - **W4 的 C1/C2 不变**（同 bs16、800 块、28 并发、0.55%）；**W4 的 E3 必跑**；W3 的 C1 降为**可选 E4b**（static vs continuous 的 Poisson 回归，只报不判）。
  - C3：golden 64/64 + paged vs seat 容差；C4：paged decode ≤ seat（目标 ≤1.0×，接受 <1.15×）；E3：≥ seat。
  - 生产路径不回退 gather；paged 只有 CUDA Triton 一条路径，CPU/MPS 上 `open_paged_pool`/`PagedKV` 直接报错（用 seat 臂），不再 silent gather。**serial prefill 控制臂已删除**（收口后 step 1）：生产 `Scheduler` 只有 packed group 一条路径，历史 serial 配置（`W45-0002/0003/0006`）不可运行但保留作历史；W3/W4 配置复跑会从 serial 默认变成 packed，不能与历史 run 直接配对。**gather decode 控制臂已删除**（收口后 step 2，见 v25）：`PagedPool.gather`、`lpad`、零块、mask-bias、`gather_*` 记账与 `decode="gather"` 生产分支全部移除，oracle 在 `tests/reference/paged.py`；历史 `W45-0003` 不可运行，写 `decode="triton"` 的配置仍作 legacy no-op 通过校验。
- **不做**：chunked prefill、prefix cache、CUDA graph、抢占/swap、FA2 paged decode/bs256；hybrid GDN 的 packed prefill 留 W6（W4.5 只冻结接口）。
- **正式结果（2026-11-06，clean tree）**：
  - E0 `W45-0001` golden **64/64**；E1 `W45-0002/0003` **9 vs 28 = 3.111×**，`kv_total 1601.97 ≤ 1640`；C2 frag **0.55%**。
  - E2 `W45-0005` pre-fix **1.203× fail** → `W45-0012` post-fix **1.040× pass（接受线）**，但仍高于 1.0 目标；E2 8/8 token 全等。
  - E3 `W45-0006/0007` TTFT p99 **3.760×**、p50 **0.512×**，两项 pass。
  - E4 `W45-0008/0009` seat **330.53** vs paged **678.59 tok/s = 2.053×**，both saturated，paged 峰值 34、`kv_total=1400`；token 27/32 只作诊断。
  - E4b `W45-0010/0011` static **157.15** vs continuous **292.80 tok/s = 1.863×**，both saturated；只报不判。
- **文档/证据**：D7 **accepted**；`W45-0001…0012` 已审（0005 为 pre-fix fail，0012 supersedes）；临时执行计划已删除，结论见 D7 与本段。
  - 图：`figures/fig_w4_5_decode.png`（C1 1.203× → 1.040×）、`figures/fig_w4_5_prefill.png`（E3 p99 3.76×）；由 `scripts/plot_w45.py` 从已提交 summary 生成。
- **时间盒/止损**：spike 1–2 天 + 实现 7–9 天 + 重跑 2–3 天 ≈ 2 周；超时先砍 prefill 保 decode，不让 W5/W6 买单。

> W4.5 不改 W4 的 **C1 3.111× / C2 0.55%**；paged decode 从 W4 的 1.671× 负结果变为 `W45-0012` 的 **1.040×（接受线，非 1.0 目标）**。

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
| W4.5 的 FA2 varlen 不过（安装/数值/TTFT） | 不开 W4.5：decode 走 `W4-0010`，直接 W5；packed prefill 记 future work |
| W4.5 的 Triton decode 不过（正确性/性能） | 只做 packed prefill（W4.5a），C4 维持 W4 负结果；不硬调 kernel |
| W4.5 两周不绿 | 先砍 packed prefill，保 Triton decode；仍不绿则回 W4 结论 |
| W6 分块 prefill 啃不动 | 用逐步循环实现，报告注明性能代价 |
| W7 时间不够 | 只做 P1 + 交叉点，P2/P3 写分析不实测 |
| 任一周租卡无数字 | 立即停租，改本地开发 |
| 时间被实习挤压 | W0–W4 已是完整交付（引擎 + 对拍 + 两张图） |

---

## 9. 简历映射（数字留空，实测后填）

> **状态（2026-11-08）**：W0–W4.5 已完成；W5–W7 未开始。下面带 `<实测>` 的条目是简历模板，不是已完成事实；投递时只写 W0–W4.5 的数字，W5–W7 留在“计划/进行中”。

> **统一 serving 引擎：跨两代注意力架构的单卡推理实现** 2026.10 — 2027.01
> - 基于手写 PyTorch 解码器实现单卡 serving：packed prefill + 连续批处理、分页 KV cache、双状态管理；
>   与 HF 逐 token 数值对齐（贪心 64 token 完全一致，每层 `max|Δ|` `<实测>`）
> - 实现 **packed varlen prefill** 与 **block-table decode 的 Triton 内核**，去掉分页 gather；
>   decode `step_ms` `<实测>`、TTFT p99 `<实测>`
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
llm-serving/
├── model/            # layers.py（+输出门控/partial-RoPE）、linear_attn.py（新增）
├── llm_utils/        # inference.py（过渡；S2 后由 serve/engine.py 接管）
├── serve/            # engine / scheduler / block_manager / state_pool
├── bench/            # run / metrics / loadgen / prompts.json
├── configs/          # experiments/ + thresholds.toml
├── scripts/          # env_snapshot / exp_index / probe_* / make_golden / align_hf
├── tests/            # unit/ + integration/ + golden/
├── results/          # W0..W8/NNNN-slug/（run 目录）+ INDEX.md
├── figures/          # 结果图 + 架构示意图
├── docs/             # experiments / decisions / environments / carrier / hybrid_attention
├── reports/
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
| W3 | 连续批处理 | ✅ | `W3-0010/0011`（C1 2.01×）、`W3-0001/0002`（E0）、E1 `W3-0017–0019` + `0006/0007` |
| W4 | 分页 KV | ✅ | C1 **3.111×**、C2 0.55%、C3 全等；C4 **1.671×** 负结果（按预注册不报吞吐）；post-fix `0009` **1.074×** 为 record-only addendum；`W4-0001…0006/0008/0009`（0007 over-budget，0008 pre-fix/superseded） |
| W4.5 | 注意力路径 v2：packed prefill + block-table decode | ✅ | E0 **64/64**；E1 **3.111×**（9/28）；E2 pre-fix **1.203× fail** → post-fix **1.040× pass**（接受线；1.0 目标未达）；E3 p99 **3.760×**/p50 0.512×；E4 **2.053×**（330.53→678.59 tok/s，both saturated）；E4b **1.863×**（record-only）；D7 accepted；`W45-0001…0012` |
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
| 2026-11-03 | v13 | **W3 完成**：C1 **2.01×**（`W3-0011`/`W3-0010`，B=8 mixed，同 trace `33bc2397…`）、C2 pass、C3 0 mismatch；E1 P1 7.94×/7.90×、P2 3.78×；P3 uniform 1.023×（负结果）；B=32 mixed 1.606×（低于 P4 的 1.8–2.2×，record-only）；E0 四条判据 pass。**发现 synthetic ctx512 EOS**（`W3-0003–0005` invalid，ctx511 重跑，P-31）；R1 的 `step_ms` 含 prefill，判读改为稳态 decode 步；D5 改 accepted，`[W3.continuous]` 标 calibrated（basis `W3-0011`）。 |
| 2026-11-06 | v14 | **W4 口径冻结（D6）**：原验收的 `ctx=2048` 被读成 uniform，测不到「按最坏长度预留」的浪费 → 改为**同总预算（KV + workspace = 1640 MiB）+ W3-like 混合长度**下的 fill-only peak concurrency ≥1.5×；**uniform 2048 降为负对照**（只报不判）；internal frag ≤5%（reservation slack 只报）；B=8 `lpad=1664` 的 `step_ms` 只记录、>1.5× 撤回吞吐结论。定稿前修正 plan 两处：E1 seat `max_batch` 6→**9**（实际最长 prompt 1517 → 行宽 1664 → 182 MiB/行），E3 paged `kv_blocks` 840→**820**（840 块在 24 活跃行时 1646 MiB 超预算）。**±1 门禁改为对同一真实 trace 重跑模拟**（合成分布比实机量的是 workload 建模差异）；并且模拟必须用**与配置相同的 `kv_blocks`**（`--paged-blocks`），否则预算反解出的块数会让 delta 恰好卡在 ±1 边界——E1 的预期因此是 **28±1（800 块）**，不是预算反解的 29，比值 **3.11×**（预言线 ≥1.5× 仍大幅成立）。 |
| 2026-11-06 | v15 | **W4 实验收口**：`W4-0001` E0 对 golden **64/64**；`W4-0002/0003` E1 seat **9** vs paged **28** = **3.111×**（同 trace、800 块，`kv_total 1601.97 ≤1640`，同 trace 模拟门禁 28±0）；`W4-0004/0005` E2 同并发 decode step **21.32 → 35.63 ms = 1.671×**（`lpad=1664` 子集 1.673×，逐重复 1.661–1.676），internal frag **0.55%**，40/40 token id 全等，workspace **52.42 MiB（+3.6%）**。**C4 >1.5× → 撤回吞吐结论，E3 不跑**（plan §4.3）；W4 以「容量成立、代价负结果」收口。口径偏差：paged `lpad` 实为 1536/1600/1664，不是全程 1664；按 1664 子集复核结论不变。`[W4.paged]` 标 calibrated（basis `W4-0003`）；`docs/w4-plan.md` 收口删除，新增 P-37。 |
| 2026-11-06 | v16 | **E3 补跑（post-hoc record-only）**：按用户指示在 C4 失败后补跑，不改变 C4。`W4-0006` seat **333.44 tok/s** vs `W4-0008` paged 800 **325.19 tok/s = 0.975×**（同 trace、both saturated）；decode-only **1.103×**，但 prefill 3.39× 把整体拉回略负。`W4-0007` 820 块峰值 **1661.8 MiB > 1640**（R5，动态 36 活跃；summary 的 1455.16 是末步，不是峰值），只保留为 over-budget 记录；`W4-0008` 800 块峰值 **1626.8 ≤ 1640**，是有效同预算臂。E3 是 **post-hoc**，不进 C1–C4；W4 仍不报吞吐。docs 清理：`w4-handoff.md` 与两个容量 sweep JSON 删除，只保留三张图；新增 P-38。 |
| 2026-11-06 | v17 | **post-fix：prefill slot mapping + E3′**。`PagedPool.write_prefill` 从 per-block 循环改成 slot mapping + 一次 advanced indexing（每层 K/V 各一次；E2 探索 551.8→176.8 ms，seat 163.0，1.08×；40/40 token 不变）。正式 `W4-0009`（800 块、同 `0006` trace）**358.18 tok/s** vs `0006` **333.44** = **1.074×**，both saturated，峰值 **1626.8 ≤ 1640**；decode-only 1.094×，E3 per-request prefill **76.4→23.0 ms**。`0007` 820 块峰值 1661.8>1640（R5）仅 over-budget record；`0008` 是 pre-fix 0.975×，由 `0009` `supersedes`。**C4 仍 1.671×**（只改 prefill，decode 未动），W4 的 C4 负结果/不报吞吐不变；0009 是 post-hoc post-fix 实验，不是 C4 pass。新增 P-39。 |
| 2026-11-06 | v18 | **W4 评审收口**：9 个 verdict 已审（`0001–0006/0009` pass；`0007` invalid/over-budget；`0008` inconclusive/pre-fix/superseded）；§12 W4 → **✅**。最终结论：C1 **3.111×**、C2 **0.55%**、C3 全等；C4 **1.671×** 负结果，按预注册**不报吞吐**；post-fix `0009` **1.074×** 为 record-only addendum，不改变 C4。 |
| 2026-11-06 | v19 | **W4.5 立项**：W4 的 gather decode（C4 1.671×）与逐请求 prefill 是路径级问题 → packed whole-prompt prefill（FA2 varlen、无 pad、seat/paged 两臂）+ Triton block-table decode（bs16/32、GQA、D=128/256），删 gather/lpad/零块/mask-bias。门禁：FA2 varlen 在 0.6B 对拍逐序列 SDPA + Triton decode ≤ seat；不过则 decode 走 `W4-0010`、直接 W5。不做 chunked prefill/prefix cache/CUDA graph/bs256；hybrid GDN packed prefill 留 W6。测量契约（queue/pack_wait/prefill_shared）与临时执行计划待写（D7）。 |
| 2026-11-06 | v20 | **W4.5 plan 落地**：临时执行计划写入（E0–E4 + 可选 E4b、双 spike 门禁、测量契约、控制臂、配置/阈值、文档生命周期）；W3 C1 降为可选 E4b（只报不判），W4 E3 必跑；plan 收口删除，D7/PITFALLS/ROADMAP 持久。 |
| 2026-11-06 | v21 | **W4.5 命名对齐 run 契约**：`bench/run.py::RUN_ID_RE` 只接受 `[A-Z]\d+`，点不合法 → stage/id 用 **`W45`**（`results/W45/`、`W45-NNNN`、`[exp].stage="W45"`、阈值段 `[W45]`）；人类可读名仍是 W4.5，plan 已同步。 |
| 2026-11-06 | v22 | **W4.5 代码完成 + dry run 校准**：`model/attention/varlen.py`（FA2/SDPA varlen）、`model/packed.py`、`model/triton_decode.py`、`serve/engine.py::prefill_group`、`serve/scheduler.py` 的 packed admission 与 `pack_wait/prefill_shared` 记账、bench plumbing 与 `[W45.paged]` 阈值、W45 配置 0001–0011 全部落地；快速门禁 627 passed。dry run（dirty，仅校准）：G1 1.93–3.98×、G2 0.089 ms、E0 64/64、E1 3.111×、E2 1.089–1.097×、E3 p99 3.49×/p50 0.558×、E4 590 vs 327 tok/s、E4b 1.86×（record-only）。**C3 口径修正**：E4 跨调度器 token 一致率（75.8%）只作诊断，精确门禁只判 E0/E2；D7 proposed，P-40…P-45 落地。正式 `W45-0001…0011` 需干净树重跑。 |
| 2026-11-06 | v23 | **W4.5 正式收口**：`W45-0001…0012` 全部干净树跑完并审。E0 **64/64**；E1 **9/28 = 3.111×**；E2 pre-fix `W45-0005` **1.203× fail** → post-fix `W45-0012` **1.040× pass**（接受线，1.0 目标未达；frag 0.55%、8/8 token）；E3 p99 **3.760×**/p50 **0.512×**；E4 **330.53 → 678.59 tok/s = 2.053×**（both saturated，paged 峰值 34、kv_total 1400）；E4b **157.15 → 292.80 = 1.863×**（record-only）。E4/E4b token 27/32、24/32 只作诊断。D7 accepted；`[W45.paged]` calibrated（basis `W45-0012`，阈值未改）；plan 删除；图 `figures/fig_w4_5_decode.png`、`figures/fig_w4_5_prefill.png`；serial/gather 控制臂当时待 W5 开工前清理（后续由 v24/v25 删除）。 |
| 2026-11-08 | v24 | **W4.5 收口后清理 step 1：删除 serial prefill 控制臂**。`MiniLLMEngine` 不再接受/宣告 `prefill` 模式；`Scheduler` 删除 `_packed_mode` 与逐请求准入分支，只走 `prefill_group`；`[engine].prefill` 不再是开关（`serial` 被校验拒绝，`packed` 只作历史 no-op），`max_prefill_tokens` 保留；bench 适配器/manifest 记录常量 `packed`。历史 `W45-0002/0003/0006` 配置保留但不可运行，注释指向 git 历史。gather/lpad/零块/mask-bias 与 `decode="gather"` 留到 step 2。 |
| 2026-11-08 | v25 | **W4.5 收口后清理 step 2：删除 gather decode 控制臂**。`PagedPool.gather` / gather workspace / `lpad` / `zero_block` / mask/bias pair / `gather_*` 记账，以及 `PagedKV.get`/`set_buffers`/`decode_impl`/`kernel_decode` 均从生产删除；paged decode 恒为 CUDA Triton block-table kernel，CPU/MPS 上 `open_paged_pool`/`PagedKV` 明确报错（seat 臂才是 CPU/MPS 路径）。纯稠密 `gather_pool` / `write_token_reference` / `reference_paged_decode` 移到 `tests/reference/paged.py`，只被测试导入。`[engine].decode` 不再是开关：`gather` 被校验拒绝，`triton`/`auto` 只作历史 no-op；`paged_stats()`/summary 不再有 `lpad`/`gather_*`/`workspace_estimated`，`workspace_mb` 在新 paged 行是结构零（D6 总预算口径保留，历史 summary 仍可读）。`W45-0003` 加 HISTORICAL banner，保留作历史但不可运行。快速门禁保持绿。
| 2026-11-08 | v26 | **投递打包**：README 改为「单卡 LLM Serving：机制收益边界实测」，加入架构图与 W3/W4/W4.5 结果图，W4 负结果折叠；§9 标注 W5–W7 未完成；删除 `archive/` 与 `docs/w4-plan.md`，`background.md` 移出版本控制；MIT LICENSE；GitHub 只读镜像 `RAIN-LOJK/llm-serving`（不推 legacy tag）。 |

---

## 14. 参考实现索引（跨阶段，只列「可抄代码」）

> 来源：`docs/benchmark-resources.md`（2026-10-30 调研）里的**代码类**资源，按阶段裁剪。
> 规则：只作机制参照与对拍；直接复制代码需保留原许可证（下表均为 Apache-2.0 / MIT）。
> 路径以本机 `transformers 5.20.2` / `fla 0.5.2` 为准，升级后需重新定位（先 `python -c "import transformers;print(transformers.__version__)"`）。
> 数据类资源（ShareGPT / LongBench / RULER / BurstGPT）不在本表：它们只影响负载真实度，不影响任何判据。
> W3 的调度/到达过程参考不在这里，见 `docs/decisions.md` **D5**（负载与调度口径）。

| 阶段 | 参考实现 | 位置 | 抄什么 | **不抄什么** |
|---|---|---|---|---|
| W2/W3 | `model/attention/base.py`（本项目自建，非外部参考） | 本仓库 `model/attention/` | 注意力后端的契约：**两个入口**（prefill / decode）与「不回退」的构造性保证（`T>1` → SDPA，`T==1` → `hand_baddbmm`，见 D4）；`model/attention/__init__.py::provenance()` 写进 manifest/summary；`hand_legacy` 移到 `tests/reference/` 冻结为**数值 oracle**（与 git `26b268d^` 逐 bit 相同） | 不做可选后端的注册表 / 能力声明（D4 定死后是死重量）、不做 entry-point 自动发现、不允许静默回退、不让 oracle 进性能结论 |
| W3 | nano-vllm `engine/{scheduler,sequence,llm_engine}.py` | `github.com/GeeeekExplorer/nano-vllm`（~1200 行） | `Sequence` 字段与调度主循环的**接口形状**（为 W4 的 block table 预留） | preempt / chunked prefill / CUDA graph；W3 的实现与判据见 `docs/decisions.md` **D5**，结果见 §5 W3 |
| W4 | nano-vllm `engine/block_manager.py` | 同上 | 物理块池、block table、free list、按需分配/回收 | 其 attention/kernel 部分 |
| W4 | nano-vllm `layers/attention.py`（`store_kvcache` / `slot_mapping`） | 同上 | 写 KV 的扁平物理槽映射（`block_id*bs + offset`）；`block_table` 与 `context_lens` 的语义 | `flash_attn_with_kvcache(block_table=...)` 的 kernel 级 paged attention（W4 走稠密 gather；W4.5 改为 Triton block-table decode） |
| W4 | LightLLM `kv_cache_mem_manager/{allocator,mem_manager}.py` | `github.com/ModelTC/LightLLM` | 纯 Python、token/块级分配器（比固定 block 更轻），与逐序列 attention 的 gather 更贴 | 模型专属子类（deepseek/glm…）、多进程架构 |
| W4 | PagedAttention（SOSP'23） | `arxiv.org/abs/2309.06180` | block_size 16/32、block table、碎片率定义 | 自定义 CUDA kernel（W4 不写；W4.5 只写 Triton decode，不写 prefill kernel） |
| W4.5 | nano-vllm `layers/attention.py`（`store_kvcache` / varlen prefill / `with_kvcache` decode） | `../nano-vllm` | packed prefill 的 `cu_seqlens`/`max_seqlen` 形状、slot 写入、decode 的 `block_table`/`cache_seqlens` | 其 `kvcache_block_size=256`（FA2 paged KV 的 256 page 约束）；本项目保留 bs16/32，decode 自写 Triton |
| W4.5 | FA2 `flash_attn_varlen_func` | `github.com/Dao-AILab/flash-attention` | prefill 的 varlen 接口（无 `block_table` 时不受 256 约束）、GQA、`causal=True` | `flash_attn_with_kvcache(block_table=...)`（page_block_size 必须 256 的倍数，会废掉 W4 的 bs16 容量结论） |
| W5 | `transformers/models/qwen3_next/modular_qwen3_next.py` | `site-packages/transformers/models/qwen3_next/` | 全注意力层装配、输出门控（`chunk(q_proj(x),2)` + `sigmoid(gate)`）、`Qwen3NextRMSNormGated`（norm before gate） | MoE / 多卡分支 |
| W5 | `transformers/models/qwen3_5/modular_qwen3_5.py` | `site-packages/transformers/models/qwen3_5/` | 0.8B 的覆盖：`Qwen3_5GatedDeltaNet`、partial-RoPE（`dim=int(head_dim*partial_rotary_factor)` ⇒ inv_freq 只按 64 维算）、`rope_parameters` 读取、mrope_section | 视觉塔 |
| W6 | `qwen3_5/modeling_qwen3_5.py`: `torch_chunk_gated_delta_rule`(L287) / `torch_recurrent_gated_delta_rule`(L424) | 同上 | **「分块 == 循环」两个端点的纯 torch 实现**；`chunk_size=64` 默认；`conv_state` 长度 = kernel−1 | 其 Triton/CUDA 分支 |
| W6 | fla `fla/ops/gated_delta_rule/naive.py` | `site-packages/fla/`（0.5.2） | **第二套独立实现**，与上一行互拍；状态布局 `[K,V]`、g 在 log-space、beta 已 sigmoid | `chunk.py` / `fused_recurrent.py`（Triton kernel） |
| W7 | LightLLM `common/state_cache_manager.py` + `qwen3next_mem_manager.py` | 同上 | 唯一把「KV 块池 + 线性常数状态池」放进同一 manager 的开源实现：常数状态池的句柄与生命周期 | 其调度器与 kernel |
| W7 | vLLM V1 `v1/core/kv_cache_coordinator.py` | `github.com/vllm-project/vllm` | 多 cache type 的协调抽象（概念） | 整个 V1 引擎（分布式 / CUDA graph / 后端全家桶） |

> W4 的实现口径（总预算 / 容量 / 碎片 / ±1 门禁）见 `docs/decisions.md` **D6**；W4.5 的测量契约与 attention 条款见 **D7**（accepted，2026-11-06 收口）。

两件已核实、可省弯路的事：

- `olmo_hybrid/modeling_olmo_hybrid.py` 里的同名 `torch_*_gated_delta_rule` 与 `qwen3_5` 版**逐行相同**，不能当第三个独立对拍端点。
- 0.8B 的 `config.json` 里**没有** `use_qk_l2norm_in_kernel`（走 `Qwen3NextGatedDeltaNet` 的类默认），`partial_rotary_factor` / `mrope_section` 也**不在顶层**（在 `rope_parameters` 内）——与 P-17 同类坑。
