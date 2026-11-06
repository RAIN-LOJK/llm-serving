# 决策记录（ADR）

> 记录「为什么这样选」而不是「跑了多少」。状态：`proposed / accepted / rejected / superseded`。
> 每个决定引用具体 run id；run 的契约见 [experiments.md](experiments.md)。

---

## D0 · 主线从训练学习项目切换到单卡 serving 引擎

- **日期**：2026-10-21
- **状态**：accepted
- **背景**：旧仓库是手写 LLaMA + 训练管线 + 玩具实验；ROADMAP 的目标是实习简历导向的
  单卡 serving 引擎（连续批处理、分页 KV、Qwen3.5 混合注意力双状态）。
- **决定**：训练线不迁移；旧代码/数据/结果从工作树删除，文本快照保留在 git tag
  `legacy-learning-2026-08`。保留 `model/` 手写算子、`llm_utils/` 生成与采样、
  `PITFALLS.md` 的工程结论。
- **后果**：
  - 仓库不再有 `train.py` / `eval.py` / 数据集 / 训练台账。
  - 实验约定重写为 run 契约（见 [experiments.md](experiments.md)）。
  - 旧训练数字不进入新报告；只在 [history.md](history.md) 里作为背景。
- **复盘点**：W1 对齐完成后，确认手写算子是否值得继续维护，还是转向 HF 适配层。

---

## D1 · 是否投入 Qwen3.5 混合注意力移植

- **日期**：2026-10-21 提出；2026-10-22 判定
- **状态**：accepted
- **判据**（预注册于 ROADMAP §3.3）：
  - P1 成立（长上下文每序列状态 ≥ 3× 优于同量级纯注意力）→ W5–W7 投入移植；
  - P1 < 2× → 砍掉移植，Qwen3.5 只作 HF 侧对照，主线收在 W4；
  - 交叉点先用解析式估算（≈193 token），再由 `W0-0001` 实测复核。
- **证据**：`W0-0001`（4090，bf16，batch=1，warmup=3 / repeats=5）：
  - 每序列 cache：8k **7.80×**、32k **8.90×**（预测 5–8× 成立并略超上限）；
  - 512 上混合已小 **2.25×**，实测交叉点 **≈193 token**（prefill 口径，与解析式一致）→ P2 不成立；
  - 32k 每序列省 **3.11 GiB**；进程峰值只差 1.39×（权重 1136.9 / 1435.1 MiB 是固定成本）；
  - 0.8B decode TPOT 22.41 → 21.89 ms（−2.3%），但 0.6B 也近似平（+10.6%）：P3 数值成立、暂不可归因。
- **决定**：**接受**。W5–W7 按 ROADMAP §5 推进；主线保持「双载体、同一 `serve/`」，不收敛到单一架构。
- **载体选择的理由**：主线必须用当年主流开源小模型（`Qwen3-0.6B` / `Qwen3.5-0.8B`）——这既是项目定位
  （回答「为什么不用最新模型」），也是受控对照的前提（同代际、同量级、只差注意力机制）。
  `TinyLlama-1.1B` 只作工程兜底，不进入对外叙事。
- **后果（若接受）**：W5 全注意力层改造（head_dim 256 / 输出门控 / partial-RoPE / M-RoPE）；
  W6 Gated DeltaNet（短卷积 → decode 循环 → 分块 prefill）；W7 StatePool + 跨代对照。
  收益论证必须限定在「长上下文 + 多并发」：上下文 < ≈193 token 时混合更贵，≤2k 时 prefill 慢 1.5–2.2×。
- **后果（若拒绝）**：W8 直接收口；报告只覆盖纯注意力引擎 + HF 侧混合架构对照。
- **复盘点**：W7 末复核一次「移植成本是否被收益覆盖」；若 W6 只有 decode 形式，按 ROADMAP §8 止损。

---

## D2 · 主线只测 Qwen3.5 的文本分支，多模态不进 W0–W8

- **日期**：2026-10-21（提出）
- **状态**：proposed
- **背景**：目标载体 `Qwen/Qwen3.5-0.8B` 是原生多模态 checkpoint
  （`Qwen3_5ForConditionalGeneration`，视觉塔 100.6M 参数 / ~192 MiB bf16）。
  W0 的双载体对照里，A 是纯文本 `Qwen3-0.6B`；是否要把图像输入纳入测量，需要明确。
- **决定**：
  - W0–W8 只加载 **text decoder**（`config.get_text_config()`），视觉塔与 MTP 头不进显存；
  - 图像输入、视觉编码器、图像 prompt 集都不在主线；报告里写明这是范围边界；
  - 但 W5 **仍实现 M-RoPE**（文本退化为 1D），因为它是 checkpoint 的位置编码，不实现就无法与 HF 对齐。
- **理由**：
  1. **自变量是注意力架构，不是模态。** 对照臂 A 只能看文本；B 若看图像，"混合 vs 纯注意力"的差异会和"模态"混在一起，D1 的显存比就不可归因。
  2. **视觉不进 decode 的状态生命周期。** 视觉塔是 prefill 前的一次性编码器，产出的是普通 token 前缀；KV cache、线性状态、调度、分页都不因它改变形状。它改变的是上下文长度分布，不是被研究的机制。
  3. **成本闸门。** W5 已有 5 类新算子；再加视觉预处理、图像 token 展开、M-RoPE 3D 位置，会把 D1 之前的投入翻倍——而 D1 的意义正是"先测再决定要不要投入"。
- **后果**：
  - `weights_mb` 是 text-only（实测 0.8B = 1435.1 MiB），与 0.6B 的 1136.9 MiB 可比；
  - 图像能力不进入简历/报告的性能结论，只在"后续工作"里出现；
  - `serve/` 的引擎接口保持 token 流，不引入视觉预处理阶段。
- **后续可选（未进 ROADMAP）**：W8 之后的 **W9 · 图像前缀复用**——1024×1024 图像约 1024 个 visual token，
  重复图像跨请求时可复用 vision embedding / KV 前缀，与 W4 的 prefix cache、W7 的 StatePool 同源。
  需要多模态 baseline、图像 prompt 集、TTFT 分解里增加 vision encode 分量，约 1 周。
- **复盘点**：W8 结束；若 D1 通过且时间有余，再评估 W9。

---

## D3 · 注意力实现与批处理的口径

- **日期**：2026-10-25（2026-10-31 **rejected**）
- **状态**：**rejected**（驳回依据见本节末；替代口径待 W2 返工后另立 ADR）
- **背景**：`W2-0002` 基线里 long 桶 prefill 比 HF 慢 1.49×（TTFT 31.8 vs 21.7 ms），
  根因是手写 attention 物化 `T×T` 分数 + 显式 fp32 掩码。换 SDPA 修复了这一点，
  但批处理一旦左 pad，flash / efficient 两个融合后端都退出、只剩 math（4 条序列：
  按序列循环 80.3 ms，pad 后 506.7 ms）。W3 要做连续批，必须先定“注意力在哪、怎么批”，
  否则机制收益会和 kernel 差异混在一起。
- **证据**：`W2-0002`（手写）、`W2-0003`（SDPA）；2048 prefill：flash 25.4 ms / math 125.5 ms；
  4 条序列（512–2048）循环 80.3 ms vs pad 506.7 ms；TPOT 17.10 → 15.08 ms（vs HF 1.113×）。
- **判据**：逐序列 attention 占单步时间 < 15%（B≤8）；W3 吞吐 ≥ 1.3× 静态批
  （`configs/thresholds.toml` 的 `[W3.continuous]`）。
- **决定**：
  1. **长度方向禁止 pad**；批维度可分桶（为将来的 CUDA graph）。
  2. **线性层 / MLP 按批执行，attention 逐序列调用**：每条保持 `(1, T)`、无 mask、
     `is_causal`，与 `W2-0003` 同一条代码路径。
  3. 全仓库只有一条注意力实现（SDPA）；手写版本只作为 `tests/` 的参考对拍。
     实际后端逐 run 记录在 manifest/summary 里。
  4. **不引入 flash-attn varlen**：prefill 每序列已是一个大 kernel（4 条 80 ms 足够），
     decode 每步打包 KV 反而更慢；CUDA graph 作为 W4 之后的可选项。
  5. 注意力路径任何变更都要新开 run + `supersedes`，旧基线不复用。
- **后果**：
  - W3 的调度器只能改“这一步有谁”，不能改注意力的张量形状；
  - W4 分页后是“按 block table gather → 逐序列 SDPA”，与 W3 同形，不是新问题；
  - W7 的线性注意力状态本来就逐请求，逐序列不是妥协而是架构要求；
  - W8 的 B=32 列，逐序列 launch 估算占比 20–30%，届时用 CUDA graph（整步捕获）
    而不是换 varlen kernel。
- **复盘点**：W3 结束时用实测占比校验 < 15%（**该判据已随本决策 rejected 一并作废，见下方驳回**）；
  W4 结束后看 gather 是否引入新热点；
  torch 升级导致 SDPA 后端支持表变化时重新确认。

### 驳回（2026-10-31）

- **驳回理由**：第 2 条（attention 逐序列）只测过「prefill 时 pad 会退 math」，**从没量过 decode 的代价**。
  实测（Qwen3-0.6B 形状，28 层，bf16，纯 attention 微bench）：
  - 逐序列 SDPA 每 step = `28×B` 次调用，B=8 → **5.8–8.8 ms**，B=32 → **21–37 ms**，且**与 ctx 几乎无关**
    （ctx512 5.77 ms vs ctx2048 5.36 ms）→ 是**派发开销**，不是算力/带宽；
  - 放进 step 里：B=8 约占 **25%**、B=32 约占 **50%** → **第 3 条自己写的「< 15%」在 B=8 就不成立**；
  - 「一次 fused ragged 调用」在本机 torch 2.8 上**没有可用实现**：jagged nested 要求 q/kv 同一 ragged 维
    （decode 的 q=1、kv=L_i 不同构）；`flex_attention` eager 慢 20–50×，`create_block_mask` 在
    B×H×Q×KV 上物化（B=32/ctx2048 要 35 GiB，OOM）；`flash_attn` 未安装。
  - 第 4 条把一个**可测的成本**（28×B 次派发）当成了「不值得做」，方向反了：它恰恰是 W3 归因的主要污染源。
- **后果**：本决策不再约束后续阶段。W3 的注意力口径、是否 pad、是否引入 varlen/graph，由 W2 返工后的新 ADR 定；
  在那之前 W3 只允许跑 exploratory，正式数字不引用。D3 里仍然成立的部分：prefill 的长度 pad 会让
  SDPA 退 math（实测 B=32/ctx2048 慢 8×），这一条作为**测量事实**保留。

---

## D4 · 注意力路径定死为「prefill=逐序列 SDPA / decode=hand_baddbmm+slab」

- **日期**：2026-10-31
- **状态**：**accepted**（口径已定；数值判定见 `W2-0007`——四条判据 pass，decode 1.013× HF、TTFT 0.92–0.95×、memory 0.992）
- **取代**：D3（rejected）留下的空位。D3 第 2 条「attention 逐序列」被它驳回的原因（`28×B` 次派发）
  在本决策里针对 **decode** 修正，prefill 继续保持逐序列。
- **背景**：D3 驳回后 W3 没有注意力口径，而 W3 的配对判据要求两臂同 kernel。当前工作树上有
  `{attn_prefill, attn_decode, decode}` 三个旋钮、8 种组合，其中 7 种没人跑，且 `decode_mode`
  与 `attn_decode` 部分重叠——这正是「配置面本身成为变量」的病。
- **证据**（均为已有实测，未新跑）：
  - prefill-2048：flash 25.4 ms / math 125.5 ms；4 条序列逐序列 80.3 ms vs pad 506.7 ms（`W2-0002`/`W2-0003` 归因）；
  - decode（Qwen3-0.6B 形状，28 层，bf16）：逐序列 SDPA 5.8–8.8 ms/step（B=8），带 mask 的批 SDPA 7.4 ms，
    `hand_baddbmm` 0.7 ms/step（B=8/L=512）；step 总量 ≈20 ms；
  - 单请求 `W2-0003`：TPOT 15.08 ms、TTFT 三桶 0.86–0.90× HF。
- **判据**（预注册，`W2-0006` 判定）：
  - V1 四条 W2 判据对 HF 臂全 pass（`decode_speed_vs_hf_min ≥ 0.80`）；
  - V2 与 `W2-0003` 的 TPOT 差 **记录不判**；预测 ≤ +10%，> 15% 记负结果，**不为 B=1 开 cat 分支**；
  - V5 `test_align_qwen3` 贪心 64 token 全等（新路径的数值门）。
- **决定**：
  1. **prefill（T > 1）只用逐序列 SDPA**，`is_causal=True`、无 mask，长度方向**永不 pad**。
     flash 不是「prefill 的选项」而是「无 mask 的结果」，pad 一次就没了。
  2. **decode（T == 1）只用 `hand_baddbmm` + `KvSlab`**，一次调用覆盖整个 batch，bool mask 融进 score GEMM。
     不做 `sdpa + slab`：SDPA 拿到 mask 后 flash/efficient 全部退出，slab 的收益被 mask 惩罚抵消。
  3. **删掉全部注意力旋钮**（`attn_prefill` / `attn_decode` / `[engine].decode`）。后端常量写死在
     `model/attention/__init__.py`，provenance 记录常量名而不是选择值。
  4. **删掉能力声明（`Caps`）与封闭注册表**。「不允许静默回退」改由构造保证：`Attention.forward` 按
     `T > 1` 直接调对应常量，没有可回退的目标。
  5. **CUDA graph 不进 W3 数字**（`serve/graph.py` 保留为 parked，W4 接线）。graph 会消灭 W3 要测的
     host 成本本身，且关掉 host 后瓶颈转移到 KV 读取——不是「去掉隐患」，是把隐患挪个位置。
  6. 旧手写实现 `hand_legacy` **从运行路径移出**，搬到 `tests/reference/` 作数值 oracle。
- **后果**：
  - 配置面从 8 种组合收到 1 条，manifest 的 `attn` 变成常量；
  - `W2-0003` 不能作为 W3 的配对基线 → **`W2-0007`**（`supersedes = "W2-0006"`，HF 分母 `W2-0005`），`W3-0001` 挂它；
  - 配对校验必须拆两层：**通用键**（prompt hash / dtype / device / out_tokens）与**机制键**
    （attn / decode / batch / trace），否则自实现臂与 HF 臂的 `decode_speed_vs_hf_min` 会被误判 `invalid`；
  - `kv_mb` 改为「预留字节」（桶粒度 64），`memory_match_ratio` 的口径随之变化，如实报告；
  - `hand_baddbmm` 的 `baddbmm` 形状就是 W4 paged decode 的原型（块表 gather 插在 `k`/`v` 读取处）。
- **复盘点**：`W2-0007` 的 V1/V2/V5；W4 结束后看分页是否让 slab 的桶粒度失去意义；
  torch 升级导致 SDPA 后端支持表变化时重新确认「无 mask 才走 flash」。

---

## D5 · W3 负载与调度口径（trace / 席位 / 饱和 / 判据）

- **日期**：2026-11-01
- **状态**：accepted（2026-11-03；C1/C2/C3 全部成立）
- **取代/关系**：**D4 仍然有效且不被本决策修改**：D4 定注意力路径（prefill=T>1 逐序列 SDPA /
  decode=T==1 `hand_baddbmm`+`KvSlab`），本决策只定 W3 的负载与指标口径。本决策不碰 D4 的任何一条
  （prefill 形状、decode kernel、旋钮删除、graph 不进 W3 数字都原样保留）。
- **背景**：W3 的配对判据要求两臂在**同一份 trace、同一个 max_batch** 上跑，而「batch」在当前的
  runner 里只是行标签（P-20：`batch_sizes=[B]` 会把同一条 prompt 重复 B 条 case 再逐条串行
  `generate`，不是并发度）。不先冻结口径，C1 的比值就会把「机制收益」与「trace 不同 / 并发度不同 /
  吞吐怎么算」混在一起，得到无法归因、也无法复现的数。因此 W3 先立本决策，再动实现。
- **判据**（引用 `configs/thresholds.toml` 的 `[W3.continuous]`，本决策**不复制数值、只引用**）：
  - C1 吞吐：`throughput_speedup_min ≥ 1.30`（连续批 / 静态批，同 trace 同 B 的配对 run）；
    **< 1.20 记负结果**并按 ROADMAP §8「不硬调」收口；
  - C2 TTFT 可分解：`ttft_p99_decomposed` —— `queue_ms` / `prefill_ms` / `ttft_ms` 三者的 p99 都存在
    且逐请求 `ttft ≈ queue + prefill`（误差 < 1 ms）；
  - C3 到达顺序不变性：单测（两种到达顺序逐 id 相同）+ 配对 run 的 `token_sha256` 交叉核对；
  - C5 批处理摊薄 host（Q1，非机制判据）：记录 `tok/s(B)` 曲线，`record_only`，不设阈值键。
- **证据（2026-11-03）**：E0 门禁 `W3-0001/0002`；E1 `W3-0017/0018/0019`（ctx511）+
  `W3-0006/0007`（ctx2048）——P1 **7.94×/7.90×**（≥5×），P2 **3.78×**（高于预注册 2.5–3×），
  R1 未触发（`step_ms` 1.06×/1.10×）；E2 `W3-0010/0011`（B=8 mixed，同 trace `33bc2397…`、
  同 max_batch=8）——C1 **2.01×**（311.43 / 154.94 tok/s），C2 pass，C3 request-row
  `token_sha256` 89/89 keys 0 mismatch；uniform `W3-0008/0009` = **1.023×**（P3 负结果）；
  B=32 `W3-0012/0013` = **1.606×**（record_only，低于 P4 的 1.8–2.2×）。
  `W3-0003/0004/0005` 因 synthetic prompt EOS 判 invalid（P-31），由 `W3-0017..0019` 取代。
- **决定**（W3 开工前冻结的负载与调度口径）：

  | 项 | 定义 |
  |---|---|
  | **trace** | `n` 条请求的 `(arrival_ts, prompt_id, budget)` 序列，由 `(seed, rate, n, budgets, prompt_sha256, burstiness, shuffle)` 完全确定；`trace_sha256` 记录在 manifest/summary 里，**配对比较的第一道校验** |
  | **席位（W3 的并发单位）** | **一块 `KvSlab`，`batch = max_batch`，每行一个席位**（不是 B 块单行 slab，也不是 `list[KVcache]`）。`pos`/`valid`/`mask` 本来就逐行独立，行与行处在不同位置、走同一套 `hand_baddbmm`（D4）。**准入 = prefill 写穿到该行（`prefill_into_seat(row, prompt_ids)`：SDPA prefill 逐层把 K/V 写进 `slab.keys/values[layer][row, :, :L]`），不建 `KVcache`、不 `fill_from` 拷贝；成功后置 `pos[row]=valid[row]=L` 并解冻游标；退出 = 冻结游标（不再 advance）+ mask 屏蔽该行，席位不回收**（P-26；席位回收是 W4 块池的事）。**同一轮 `model(...)` 永远覆盖全部 `max_batch` 行**，空闲行也在，只是被 mask 挡住 |
  | **空转行** | 空闲席位每步仍在 forward 里，token 是占位、输出被丢弃（冻结行 `valid=0`，全 mask → softmax 无可用 key 的行 bias 被清零，输出是确定值不是 NaN）。这是定长 shape 的代价，也是 D4 选择的结果；**不做**「按 active 行重排 KV」，**不做**逐行 forward |
  | **静态批** | 同一份 trace、同一个 `max_batch`，准入策略 `static`：**一波跑完再收下一波**（波内所有请求解码到各自 budget）；波次开始前需凑满 `max_batch`，**trace 已发完则以现有请求开波**（否则末波永远等不满） |
  | **连续批** | 同 trace、同 `max_batch`，准入策略 `continuous`：任一席位释放后**下一步立即**补入（下一轮 forward 就该行已写穿 prefill 并解冻） |
  | **两臂共用代码** | 同一个 `Scheduler` 类、同一个 `decode_batch`；差异只有 `policy` 一个字段（构造上保证同 kernel 同数值路径，P-23） |
  | **吞吐（双口径，两个都报）** | ① `tok_s = Σ out_tokens / makespan`，`makespan = t_last_finish − t_first_arrival`（**含收尾 drain**）；② `tok_s_active = Σ out_tokens / (t_last_finish − t_first_token_global)`（**活跃窗口**，去掉首批 prefill 之前的空转）。**C1 判据用 ①**；② 只作口径说明，避免「drain 段怎么算」的争议。同时报 `req_s`、`offered_rate`、`achieved_rate`、`queue_depth_max`（已到达、仍在排队的最大数；未到达的不算，见 step row 的 `pending`）、`queue_depth_at_last_arrival` |
  | **TTFT 分解** | `t_admitted` = 准入决定（分到席位）；`t_prefill_start` = **该请求自己的** prefill 真正开始（打点即在此）；`prefill_ms = t_first_token − t_prefill_start`；`queue_ms = t_prefill_start − t_arrival`。**`ttft = queue_ms + prefill_ms` 按构造恒等**（C2 要的就是这个），另记 `admit_wait_ms = t_prefill_start − t_admitted` 作诊断，**不进分解**。串行 prefill 期间别的请求在队列里等的时间算**排队**——这正是连续批要压缩的尾部 |
  | **TPOT** | `(t_finish − t_first_token)/(out_tokens − 1)`，不含 prefill，不含排队 |
  | **budgets** | 每条请求的生成上限（`≤ max_new_tokens`），由 trace 按 seed 抽样；`budgets=[128]` 即协议默认（与 W2 同口径）；**一个 run 内只用一种分布**，记录进 manifest。缺省或空列表在 `validate_config` 里**先归一化成 `[max_new_tokens]` 再校验非空** |
  | **计时** | 墙钟用 `perf_counter`（含排队，本来就是墙钟）；单步 GPU 时间用 `torch.cuda.Event`；step row = `active / queued / admitted / retired / step_ms / prefill_ms / decode_event_ms / host_gap_ms`（`host_gap_ms = step_ms − decode_event_ms − prefill_ms`，即 **decode event 窗口之外的 host 开销**，不是 `cat`。CUDA event 覆盖整个 decode 时间线，host 饥饿造成的 GPU 空转落在 `decode_event_ms` **里面**，所以它**不是**「slab 读写之外的 host 开销」） |
  | **饱和判据（每臂各自判，不用共享常量）** | `queue_depth_at_last_arrival` = 最后一条请求到达时刻仍在排队（已到达、未开始 prefill）的请求数（即 `max(0, backlog − in_flight)`，再扣掉当时仍空着的席位；见 `bench/engines/continuous.py`）；`saturated ⇔ queue_depth_at_last_arrival > 0`：trace 已经发完、仍有请求在等容量 = 过载。**不能取 run 结束时最后一条 step row 的 `queued`**——run 结束必然排空，那样 `saturated` 按构造恒为 False，C1 每对都 `invalid`。**不看** `capacity_estimate_req_s` 是否达标——两臂 capacity 本来就不同，用同一个 capacity 会把弱臂按构造判成 `not_saturated`，C1 直接变 `invalid`。`capacity_estimate_req_s` 只在冒烟里量一次、只用于挑 λ 的初值，**不进判据**；若连续臂在 2× 估计值下队列仍会排空 → 该对记 `not_saturated`，λ 加倍**新开 run**（不挑点） |
  | **λ 与 B 的关系** | λ 只在**同一 B** 的一对（static/continuous）内必须相同（`trace_sha256` 可比的前提）。B=8 对与 B=32 对**各自**用该 B 的冒烟 capacity 定 λ（`λ = 2 × capacity(B)` 取整），并写进 manifest；跨 B 不比较绝对吞吐，只比较**同一 B 内的比值** |

- **后果**：
  1. **两臂必须同 trace**：`trace_sha256` 是第一道配对校验，不同就直接 `invalid`、不给比值；
     `max_batch` 同为机制键，`policy` 只记录不判等。
  2. **空闲席位在每步 forward 里**：定长 shape 是 D4 decode 路径的直接代价，`B` 越大空转越贵
     （`budgets` 混合时中段席位会稀疏），这部分成本计入 `step_ms`，不做任何「按 active 行压缩」的补救。
  3. **`memory_match_ratio` 的口径要跟着改**：池化之后分子是**所有席位预留之和**
     （`2·n_layer·max_batch·n_kv_head·max_len·head_dim·itemsize`），仍是「本请求份额」就会重演
     W2-0004 的账外席位（P-30）。
  4. **席位回收不在 W3**：冻结即退出，回收与按需分配留给 W4 块池；W3 的内存图因此是「池的定长预留」，
     不承诺与负载相关。
  5. **p99 只作标注**：640 个 request row（`requests=128 × repeats=5`）下报 p95 与 CV，合法 p99
     由可选的单臂 `requests ≥ 1024` run 产出，且不进比值判据表。
  6. C1 的比值只在**过载 + 混合长度**的一对上判；均匀长度的那一对期望 ≈1.00×，按设计进负结果专章，
     不进 `acceptance`（否则一份 `fail` 会污染判据表）。
  7. **C4 已随 D3 rejected 删除**：W3 不再由 profiler 集成测试断言注意力占比。注意力占比的实测留在
     `model/attention/hand_baddbmm.py` 的模块文档里（B=8/L=512 ≈0.7 ms/step），属于 D4 的证据，
     不是 W3 判据；`[W3.continuous].attn_share_max` 随之删除。
  8. **输出长度口径**：Qwen3-0.6B 在固定 32 条自然 prompt 上不吐 EOS，所以 `out_tokens` 恒等于
     `budget`（`W2-0007` 实测 160/160 条 measured row 都是 `out_tokens=128`）；**合成 prompt 不保证**
     （见第 9 条）。因此 C1 只证明「合成异质长度负载下调度开销足够小」，**不是真实负载吞吐声明**；
     P4 预注册的 1.8–2.2× 由 `budgets` 分布直接决定，是**构造性推论而非独立预测**。
  9. **E1 的 synthetic prompt 可能 EOS（2026-11-03）**：`synthetic_ids` 的随机前缀对 base 模型是 OOD；
     ctx512 的 E1 三格（`W3-0003/0004/0005`）实际只解码 3 token，已判 invalid。E1 的短上下文改用
     最近的 ctx511（`W3-0017/0018/0019`）；1 token ≈0.2%，远小于 trace 间 CV。**E1 每个 run 必须先查
     `out_tokens` 中位数 == `budget`**，不等就不进图/结论（PITFALLS P-31）；根治的 `ignore_eos`
     workload 控制留到 W7 再评估。
  10. **R1 的 `step_ms` 含 admission/prefill 步**：`active==B` 的第一行是 prefill 步；ctx512 的 3-token
     数据使该行 prefill 占 88%，字面 4.46× 是假阳性。判读 R1 用稳态 decode 步（排除 prefill 行）或
     `decode_event_ms`；ctx2048 的 24.19/22.02 = 1.10× 是有效对照。`decode_only_tok_s` 同理不能把
     `active=0` 空转步算进分母（128 token 时 ≈0.8%，3 token 时 ≈50%）。
  11. **饱和 scenario 的 per-prompt CV 警告不是 W2 的延迟 CV 判据**：E2 六格各 61–76 条
     `ttft_ms/e2e_ms CV>0.10` 警告，是开放环过载下队列位置随 repeat 抖动的必然结果；C2 是
     结构判据（逐请求恒等），R5 只看 trace 级 `tok_s` CV（实测 0.11–0.53%），未触发。
     不要把 W2 的单请求 0.10 CV 目标套到饱和 scenario 上，也不要因此重跑。
  12. **summary 汇总层有三个已知缺口（不影响判据，报告用 trace/request rows 补）**：
     `throughput.queue_depth_max` 为 `null`（trace row 有值：72/67/88/62/96/48）；
     `protocol.policy` 为 `null`（`manifest.scheduler.policy` 正确）；
     `consistency` 没有 token-hash 对比字段（C3 用 request rows 按 `(prompt_id,budget)` 核对）。
     收口时在 notes 里注明，不在 W3 改代码。
- **复盘点（2026-11-03 已回填）**：C1 2.01×、C2 pass、C3 89/89 keys 0 mismatch；
  E1 P1 7.94×/7.90×、P2 3.78×；P3 1.023×；P4 B=8 2.01× 命中、B=32 1.606× 低于 1.8–2.2×
  （scale 效应，record_only，不是 C1 失败）；P5 queue share 99.1–99.9%。
  未决：P6（注意力占比）未单独测；B=32 × ctx2048 留给 W4；summary 三处汇总缺口（见后果 12）。
  若后续复跑发现 C1 <1.20，仍按 ROADMAP §8「不硬调」收口，不改判据、不挑点。

---

## D6 · W4 分页口径（预算 / 容量 / 碎片 / 判据）

- **日期**：2026-11-06
- **状态**：accepted（口径冻结；W4 plan v2（已删除，见 git history）的两处偏差在定稿前已修正，见「与 plan 的偏差」）
- **取代/关系**：不改 **D4**（注意力路径：prefill=逐序列 SDPA / decode=hand_baddbmm），
  也不改 **D5**（trace / 席位 / 饱和口径）。D6 只定 W4 新增的三件事：**总预算怎么算、容量怎么量、
  碎片怎么判**。D5 的「席位回收留给 W4 块池」由本决策兑现。
- **背景**：W3 的席位是「一块 `KvSlab`、每行按 trace 最坏长度预留、退出 freeze 不回收」（P-26），
  所以并发上限是 `floor(预算 / 行宽)`,与负载无关。W4 要证明的是 vLLM 的核心论点：
  **浪费来自「按最坏长度预留」，换成「按每条请求的上界预留」后，同预算能装下更多并发**。
  不先把「预算包含什么、容量取哪个数、碎片指什么」冻结，C1 的比值就会把 workspace、
  行粒度浪费、workload 顺序混在一起，得到无法归因也无法复现的数。

- **判据**（引用 `configs/thresholds.toml` 的 `[W4.paged]`，本决策不复制数值）：

  | # | 判据 | 口径 | 失败线 |
  |---|---|---|---|
  | C1 | 同预算容量 | 总预算 **1640 MiB**（KV + workspace），W3-like 混合长度，fill-only 最大同时准入（paged 用配置的 `kv_blocks=800`） | ≥1.50×；<1.20× 记负结果 |
  | C2 | 碎片 | **internal frag**（只算最后一块没写满），时间加权 | ≤5% |
  | C3 | 数值 | paged vs seat 逐 token 一致；E0 对 golden | 全等 |
  | C4 | 代价 | B=8、两边 `lpad=1664` 的 decode `step_ms` | 只记录；>1.5× 撤回吞吐结论 |

- **证据（定稿前，均为解析/模拟，不是实测）**：
  - `scripts/sim_paged_capacity.py`：W3-like @1640/bs16 → seat 6 / paged 26（合成分布，固定 budget=128）；
  - **真实 trace 重算**（`results/W3/0011` 的 prompt 长度 + seed 0 的 budget 抽样）
    → seat **9**（行宽 1664、182 MiB/行）；
  - **E1 配置的固定块数**（`--trace ... --paged-blocks 800`）→ paged **28** → **3.11×**，`kv_total ≈1595 MiB`；
    预算反解（≈822 块）则是 29 条 → 3.22×。E1 的实机门禁比的是**前者**（28 ±1）；
  - uniform-2048 负对照 6 vs 6 = **1.00×**（paged 不多装，与设计一致）；
  - internal frag：bs16 ≈1.4%、bs32 ≈2.7%（与 plan P3 的预测吻合）。

- **决定**：

  | 项 | 定义 |
  |---|---|
  | **总预算** | `kv_total_mb = kv_mb + workspace_mb`，W4 固定 **1640 MiB**；seat 与 paged 同一**上限**，实际用量分别记 |
  | **seat 单位** | 一块 `KvSlab` 的 `max_batch` 行，行宽 = `bucket(trace 最大 prompt + budget + 1, 64)`；每行按最坏长度预留 |
  | **paged 单位** | 每层 `(n_blocks, bs, n_kv_head, head_dim)`；`kv_blocks` 由配置给定，须满足 `kv_mb + workspace ≤ 1640` |
  | **块表** | 每请求一张逻辑→物理映射；第 `j` 个 token 在 `blocks[j//bs]` 的第 `j%bs` 列；**块分配后不移动** |
  | **准入** | 一次性预留 `ceil((prompt + budget) / bs)` 块；同一 step 的多次准入必须原子；无抢占 ⇒ 不允许中途分配失败 ⇒ 无死锁 |
  | **退出** | 立即归还全部块；`release_request` 幂等；下一请求可直接复用 |
  | **decode gather** | 每步按块表把**活跃行**搬成 `(n_active, H_kv, lpad, D)`，`lpad = bucket(本步最长活跃长度, 64)`；不足 `lpad` 的行用**专用零块**补齐，逻辑列 mask 关掉；然后走现有 `hand_baddbmm`（一层一次，D4 不变） |
  | **解码批** | 只压缩到活跃行；logits 散回 `(max_batch, vocab)`；`max_batch` 只是调度上限，不再是显存上限 |
  | **容量口径** | fill-only 下最大同时准入请求数（allocator capacity），**不是**动态峰值；动态峰值只作 E3 诊断 |
  | **±1 门禁** | 模拟必须读**同一条真实 trace** 的 `(prompt_len, budget)` 序列、并用**与配置相同的 `kv_blocks`**（`--paged-blocks`）再与实机比较；用合成分布比实机、或用预算反解的块数比固定块数是无效门禁（前者量 workload 建模差异，后者让 delta 恰好卡在 ±1 边界） |
  | **internal frag** | `1 − 已写 token / (已写块 × bs)`，时间加权；**C2 只判这个** |
  | **reservation slack** | `1 − (prompt + generated) / (reserved_blocks × bs)`；时间加权；单独报，不判 |
  | **seat row slack** | `1 − live / row_width`；与 reservation slack 同表，作为「seat 的浪费」对照 |
  | **代价** | B=8、两边 `lpad=1664`、只统计 decode 步的 `step_ms`；`decode_event_ms` 只作诊断，不当 GPU 时间 |
  | **gather 记账** | `gather_read_mb` / `gather_write_mb` 是**流量**（`n_layer × n_active × lpad × kv_bytes_per_token_per_layer`）；`workspace_mb` 是**占用**；两者不混 |
  | **fill-only** | 只 prefill + 预留，不 decode；synthetic prompt 的 EOS 不影响容量 |
  | **EOS** | E2/E3 用 frozen prompt，先查 `out_tokens` 中位数 == budget；E1 不 decode，不受 P-31 影响 |

- **与 plan（W4 plan v2（已删除，见 git history））的两处偏差，已在定稿前修正**：

  1. **E1 seat 臂的 `max_batch`：6 → 9。** plan §4.1 假设「最长 prompt = 2048」，
     行宽 2240。实际 `bench/prompts.json` 的最长 prompt 是 **1517**（l05），
     `bucket(1517 + 128 + 1, 64) = 1664`，每行 **182 MiB**，1640 MiB 能放 **9 行**。
     若照 6 跑，C1 会从 3.22× 虚高到 4.67×——多出来的部分是「把 seat 的 `max_batch` 人为设小」，
     不是分页收益。原则：**两臂的 `max_batch` 都必须让显存先于它成为约束**。
  2. **E3 paged 臂的 `kv_blocks`：840 → 820。** 840 块 = 1470 MiB，24 活跃行时 workspace
     176 MiB，合计 **1646 MiB > 1640**。820 块 + 26 活跃行 ≈ **1624 MiB**。
     （E1 的 800 块是保守值，不是预算上限：它给 workspace 留 45 MiB 余量。）

- **后果**：
  1. **容量比值随预算单调下降，必须报曲线**：真实 trace（预算反解）下 1200/1456/1640/2400/4096 MiB 对应
     4.33×/3.50×/**3.22×**/3.54×/2.86×（≥2400 后 paged 撞到请求数或 `max_batch`，不再是显存受限）。原因是 seat 每 182 MiB 稳定加 1 行，而 paged 会撞到
     trace 的请求数（32）或 `max_batch`（64）。C1 判的是 1640 那个切片，**报告主图是曲线**。
  2. **uniform-2048 不是判据**：seat 的 9 行正好覆盖均匀长度，paged 还要付 workspace，
     预测 0.8–1.0×；报告必须给出，但不拿它判 1.5×。
  3. **`memory_match_ratio` 的口径沿用 D5/P-30**：paged 臂的分母是 `kv_total_mb`（池 + workspace），
     不是「本请求份额」。
  4. **C4 不进硬验收**：它是「能不能报吞吐」的门槛，不是容量判据；避免 host/GPU 抖动把容量结论一起拖死。
  5. **P-26 由此关闭（不是作废）**：`freeze_seat` 保留「冻结游标、保留预留」的原语义给 W3 席位路径；
     paged 路径用 `release_request` 归还块，两条路径不共用函数名。
  6. **W4 不做**：抢占/swap/recompute、chunked prefill、prefix cache、CUDA graph、
     flash-attn/paged-attention kernel（换 kernel 会同时废掉 D4 与 W2/W3 的全部基线）。
  7. **新增坑**：P-32…P-36（见 `PITFALLS.md`），核心是「gather 宽 = 最大活跃长度」「零块 + 逻辑 mask」
     「RoPE 表长 ≠ lpad」「`kv_mb` 必须同时报三个数」「固定 KV 预算而不计 workspace 会读出假比值」。

- **复盘点**：
  - E1 实机 `peak_concurrency` 与「同 trace 模拟」是否在 ±1 内；
  - E2 的 `step_ms` 比值是否 ≤1.15×（健康线）、是否 >1.5×（撤回吞吐结论）；
  - E3（可选）两臂是否都 `saturated`；
  - `kv_total_mb` 实测是否 ≤1640（超了下一 run 降 `kv_blocks`，不静默改预算）；
  - W4 结束后回看：分页是否让 slab 的 64 列桶粒度失去意义（D4 的复盘点之一）。

---

## D7 · W4.5 注意力路径 v2：packed prefill + block-table decode 的口径

- **日期**：2026-11-06
- **状态**：proposed（口径已按 dry run 校准；正式 run 前由用户确认后转 accepted）
- **取代/关系**：
  - **部分取代 D4**：W4.5 的 seat/paged 两臂 prefill 改为 packed varlen（FA2 on CUDA /
    SDPA-per-segment on CPU），paged decode 改为 Triton block-table kernel；D4 的
    「prefill=逐序列 SDPA / decode=hand_baddbmm+slab」只保留给 W2/W3/W4 的历史 run 与
    seat 臂的 decode。**W2/W3/W4 的结果不因本决策改写**，但用新 commit 复跑旧配置会走新
    路径，不能与历史数字直接配对。
  - **反转 D6 的一条边界**：D6 写「W4 不做 flash-attn/paged-attention kernel」；W4.5 正是
    这个 kernel 阶段。D6 的预算/容量/碎片/配对口径**不变**，只有注意力条款被本决策取代。
  - **扩展 D5 的 TTFT 分解**：串行臂仍满足 `ttft == queue + prefill`；packed 臂改为
    `ttft == queue + pack_wait + prefill_shared`。W3/W4 的 request/step row 语义不变。
- **背景**：W4 的 paged 臂在 decode 前把块表 gather 成 `(B, H_kv, lpad, D)`，再走
  `hand_baddbmm`。gather 的读写流量与 `lpad` 桶宽是分页的代价，也把 decode 实验和
  prefill 的逐请求 SDPA 混在一起。W4.5 要把 attention 路径本身换成
  「整 prompt 打包的 varlen prefill + 直接读块表的 decode」，因此必须先冻结：打包/准入
  策略、TTFT 的归属、控制臂、C3 的适用范围，以及哪些数字可以跨臂比较。

- **证据（dry run，dirty，仅用于校准；正式证据须干净树重跑）**：
  - G1 `scripts/bench_prefill_pack.py --spike fa2`：B=2/4/8 的 packed/serial
    `prefill_ms` = 1.93× / 3.47× / 3.98×；token ids 全等；`max|Δ|` = 0.30 / 0.44 / 0.50
    （bf16，Qwen3 logits 的 |max|≈28，属于 kernel 归约顺序差异，不是接线错）。
  - G2 `scripts/bench_decode_kernel.py --spike triton`：0.6B anchor（B=8, L=1664, D=128）
    单次 kernel p50 0.089 ms；对 dense oracle 的 bf16 `max|Δ|` = 2.44e-4。
  - `W45-0001`：packed + Triton，s01 64 token，golden **64/64**（match_rate=1.0）。
  - `W45-0002/0003`：E1 容量 **9 vs 28 = 3.111×**；`kv_total = 1601.97 ≤ 1640`。
  - `W45-0004/0005`：E2 `step_ms` paged/seat = **1.097×**（active=8，27.293 / 24.888 ms）；
    8/8 first-repeat prompt token hash 全等；frag 0.0055。
  - `W45-0006/0007`：E3 TTFT p99 **3.491×**（208.02 / 59.59 ms），p50 **0.558×**
    （58.24 / 104.37 ms）。
  - `W45-0008/0009`：E4 seat **327.37 tok/s**、paged **590.04 tok/s**（1.80×），两臂
    `saturated=true`，paged `kv_total=1400`；paged/seat 的逐请求 token 一致率
    **75.8%（31/128 不同）**、token 级 **78.9%**，见下面 C3 条款，**不作为门禁**。
  - 同组对照（两臂相同 packed group、相同 prompt/预算、128 步）：seat vs paged gather
    **8/8 全等**；seat vs paged Triton 3/8 prompt 翻转，首个翻转的 top-2 margin 为
    0.0 / 0.0625 / 0.0，单步 kernel 差 2.44e-4。翻转发生在 bf16 的 near-tie 上，
    不是块表/位置/页掩码错误。

- **判据（引用 `configs/thresholds.toml` 的 `[W45.paged]`，本决策不复制数值）**：

  | # | 判据 | 口径 | 失败线 |
  |---|---|---|---|
  | G1 | FA2 varlen 可用 | 与 serial 比 token ids 全等、`max\|Δ\|` 在容差内、B≥4 有收益 | 不过则不开 W4.5 |
  | G2 | Triton decode 可用 | 单步 vs dense oracle：无 NaN，argmax 全等，bf16 `max\|Δ\| ≤2e-2`；B=8/L=1664 `step_ms` ≤1.15× seat | 不过则只做 packed prefill（W4.5a） |
  | E0 | 数值门 | `W45-0001` golden 64/64；packed vs serial token 全等 | 不过不跑性能 |
  | E1 | 容量不变 | 9/28、3.111×、`kv_total ≤1640` | 查 allocator，不改预算 |
  | E2 | decode 代价 | `step_ms` paged/seat ≤1.15×（目标 ≤1.0×）；E2 同组 token 全等；gather 流量 0 | 调 kernel / 退回 W4.5a |
  | E3 | packed prefill | TTFT p99 改善 ≥1.15×；p50 ≤1.10× | 查打包策略；不过则只做 decode |
  | E4 | 端到端 | 两臂 saturated；paged tok/s ≥ seat；`kv_total ≤1640`；无错误 | 查 kernel/prefill/峰值，不调参 |

- **决定**：

  | 项 | 定义 |
  |---|---|
  | **prefill 路径** | packed whole-prompt varlen：`q/k/v` 为 `[total_tokens, H, D]`，`cu_seqlens` 按请求切段；CUDA 用 `flash_attn_varlen_func`（fp16/bf16），CPU/MPS 用逐段 SDPA reference。**不使用** `block_table` 读 paged cache，因此不触发 FA2 的 256 page 约束。 |
  | **packing** | step 开始按 FIFO 取已到达请求，填到 `max_prefill_tokens`（默认 8192）或 `max_batch`；不等待填满、不跳队；队头单独放不下时自己成组；一个 prompt 不拆（无 chunk）；不与 decode 混一个 forward。 |
  | **KV 写入** | 每层 K/V 通过 per-request sink 写 seat 行（`SlabWriteSink`）或 paged 块（`PagedWriteSink` + 预计算 slot map）；两臂共用同一 varlen 计算，只差写入目标。 |
  | **logits / 首 token** | 每组只取每个请求的最后一个 token（`last_token_index`）过 `lm_head`；一次批量 greedy argmax；同组 `t_first` 同一时间戳。 |
  | **decode 路径** | paged：Triton `paged_decode` 直接读 `(n_blocks, bs, H_kv, D)` 池 + int32 block table + int32 seqlen，无 gather/lpad/zero block/causal mask；seat：`hand_baddbmm` 不变。paged 在 CUDA 上不允许自动回退 gather。 |
  | **TTFT 归属** | `queue_ms = t_admitted − t_arrival`；`pack_wait_ms = t_prefill_start − t_admitted`；`prefill_shared_ms = t_first − t_prefill_start`。串行臂 `t_admitted == t_prefill_start`，退化为 `ttft == queue + prefill`。step 的 `prefill_ms` 是**组耗时之和**，不是逐请求相加。 |
  | **step/request row** | 新增 `prefill_group`、`pack_wait_ms`、`prefill_shared_ms`、`prefill_groups`、`prefill_ids_per_group`；旧字段不改名。`host_gap_ms = step_ms − decode_event_ms − prefill_ms`。 |
  | **计时** | 沿用 D5：墙钟 `perf_counter`，单步 GPU `cuda_event`；报 p50/p95/p99/max，不报单一均值。 |
  | **C3 数值** | 精确 token 一致只在两种同组场景判：① `W45-0001` golden 64/64；② E2（closed、同 prompt 集、同 packed group 组成）8/8 first-repeat prompt hash、W4 口径 40/40 row。**E4 开放负载的 paged vs seat token 一致率只作诊断**：两臂 group 组成不同（B=9 vs B=64），且 Triton 与 hand 的 bf16 归约不同，near-tie 翻转不是接线错误；不设 0.999 门禁。 |
  | **单步 kernel 容差** | bf16：`max|Δ| ≤ 2e-2`、argmax 全等、无 NaN/Inf；fp32：`≤1e-4`。packed vs serial 的 logits 容差按 dry run 记为 `max|Δ| ≤ 1.0`（Qwen3 bf16 的 |logit|≈28），**不用于替代 token 门禁**。 |
  | **控制臂** | `[engine].prefill="serial"` 只在 W4.5 实验期存在，写入 provenance；它不是生产回退。`[engine].decode="gather"` 只作 oracle/对照；`auto` 按 device 解析（CUDA→triton，其他→gather），不按可用性回退。 |
  | **阈值** | `[W45.paged]` 于 2026-11-06 dry run 后冻结：`decode_step_ratio_max=1.15`（目标 1.0）、`packed_prefill_ttft_p99_speedup_min=1.15`、`p50_ratio_max=1.10`、`capacity_speedup_min=1.50`、`fragmentation_max=0.05`、`golden_token_match_min=1.0`、`paged_vs_seat_token_match_min=0.999`（**只用于 E2**）。`decode_step_ratio_max` 取两臂最大公共、prefill 占比 ≤10% 的 active bucket，不取全 bucket 中位数。 |

- **后果**：
  1. **D4 的注意力条款被部分取代**：W4.5 的 seat prefill 也不再是逐请求 SDPA，而是
     `SdpaVarlen` 的单段（CPU）或 FA2 varlen（CUDA）。用新 commit 复跑 W2/W3/W4 配置会
     改变 `attn_prefill` provenance 与数值路径，不能再与历史 run 做 mechanism 配对。
  2. **D6 的容量/碎片口径继续有效**：E1 的 9/28、`kv_total` 与 E2 的 internal frag 仍是
     C1/C2 的证据；本决策只改 attention kernel。
  3. **E4 的 token 一致率不是正确性门禁**：跨调度器/跨 kernel 的 greedy id 在 bf16
     near-tie 上会翻转；正确性由 E0 golden、E2 同组全等与单步 kernel 容差承担。报告必须
     同时给出一致率和首个翻转的 top-2 margin，不能只报一个百分比。
  4. **`decode="gather"` 是 oracle**：paged 生产路径在 CUDA 上必须走 Triton；`auto` 只按
     device 解析，不因 Triton 缺失而静默切 gather（缺失时构造/调用直接 `RuntimeError`）。
  5. **阶段结束删除**：`[engine].prefill` 开关与 serial 生产分支；`PagedPool.gather`、
     `lpad`、`zero_block`、`mask-bias`、`gather_*` 记账从生产路径移到
     `tests/reference/`（oracle 保留）。`max_prefill_tokens` 保留。删除前先 `grep` 确认
     没有生产引用，并跑完整快速门禁。
  6. **新坑（收口时编号，候选 P-40…P-45）**：
     - P-40 packed 的 TTFT 是共享时间，p50 可能变差、p99 可能变好；只报均值会掩盖；
     - P-41 FA2 的 256 page 约束只在 `block_table` 路径触发，varlen prefill 不受影响，
       但一旦加 prefix cache 就会重新适用；
     - P-42 Triton decode 的最后页必须按 `seqlen` mask，block table 必须 int32，不能依赖
       零块；
     - P-43 static 臂也从 packed prefill 受益，continuous/static 比值会被稀释；
     - P-44 超 `max_prefill_tokens` 的队头单独成组，不是 chunked prefill；
     - P-45 跨 kernel/跨调度器的 greedy token 全等不是良定义判据，near-tie 会翻转。

- **复盘点**：
  - 正式 E4 两臂是否都 `saturated=true`；paged tok/s 是否 ≥ seat；`kv_total ≤1640`；
  - E2 的 1.097× 是否稳定在 1.15× 以内；若 >1.15×，先加 split-K/瓦片，不调阈值；
  - E4 的 token 一致率与首个翻转 margin 是否与 dry run 同量级；若出现 early-token
    大面积翻转（不是 near-tie），按 kernel bug 查，不引用吞吐；
  - E4b（可选）continuous/static 比值是否被 packed 稀释；
  - 收口时删除 `docs/w4.5-plan.md`，本决策、PITFALLS、ROADMAP、`results/W45/` 保留。

---

## 模板

```markdown
## Dx · <一句话决定>

- **日期**：
- **状态**：proposed / accepted / rejected / superseded
- **背景**：
- **证据**：W?-????（run id）、summary 关键数字
- **判据**：目标值 / 判失败线；引用 configs/thresholds.toml
- **决定**：
- **后果**：
- **复盘点**：
```
