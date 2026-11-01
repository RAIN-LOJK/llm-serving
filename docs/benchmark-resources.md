# minillm 测试数据集与参考资源调研报告（2026-10）

> **调研时间**：2026-10-30（JST）· **调研范围**：四类——① serving 负载/测试数据集；② 参考项目（连续批/分页 KV/块管理/双状态管理）；③ 算子实现参考（Gated DeltaNet 分块 prefill + 逐步 decode）；④ 评测方法（负载生成/到达过程/指标口径）。
>
> **证据分级**：**【已查证】** = 直接读过 HF 数据卡 / GitHub 仓库页 / 官方文档 / 源码；**【一方称】** = 仅来自搜索摘要、第三方页面或版本间可能漂移的说法，引用前需复核。
>
> **适配性标签**：**可直接用**（格式/许可/逻辑拿来即用）／ **需改造**（需抽子集、裁剪、重写或对齐配置）／ **仅参考**（只借鉴机制、口径或思路，不直接采用）。
>
> **用途**：给 W3–W8 的工程决策提供「哪个资源对哪个阶段有用、怎么用、有什么坑」。本报告不修改任何现有代码/配置/测试，仅作资源索引。

---

## 0. 阶段 × 资源总览矩阵

| 资源 | W3 连续批 | W4 分页 KV | W5 全注意力层 | W6 Gated DeltaNet | W7 StatePool | W8 报告 |
|---|---|---|---|---|---|---|
| ShareGPT_Vicuna_unfiltered | 🟢 可直接用 | — | — | — | — | 🟢 可直接用 |
| LMSYS-Chat-1M | 🟡 需改造（取统计） | — | — | — | — | — |
| BurstGPT（Azure trace） | 🟢 可直接用（到达时间戳） | — | — | — | — | — |
| LongBench v1（含 LongBench-E） | — | 🟢 可直接用 | — | — | — | 🟢 可直接用 |
| LongBench v2 | — | 🟡 需改造（只取 ≤32k 档） | — | — | — | — |
| ∞Bench / InfiniteBench | — | 🔵 仅参考（抽极小子集） | — | — | — | — |
| RULER（合成生成器） | — | 🟢 可直接用 | — | — | — | — |
| vLLM（V1 engine） | 🔵 仅参考（机制） | 🔵 仅参考 | — | — | 🔵 仅参考 | — |
| SGLang | — | 🔵 仅参考 | 🔵 仅参考 | 🔵 仅参考 | — | — |
| mini-sglang | 🟡 需改造 | 🟡 需改造 | 🟡 需改造 | 🟡 需改造 | — | — |
| nano-vllm | 🟡 需改造（主抄） | 🟡 需改造（主抄） | — | — | — | — |
| LightLLM | 🟡 需改造 | 🟡 需改造 | — | — | 🟡 需改造（主参考） | — |
| vattention | — | 🔵 仅参考（概念） | — | — | — | — |
| TensorRT-LLM | 🔵 仅参考 | 🔵 仅参考 | — | — | — | — |
| PagedAttention 论文 (SOSP'23) | — | 🔵 仅参考（理论依据） | — | — | — | — |
| Gated DeltaNet 论文 (arXiv 2412.06464) | — | — | — | 🔵 仅参考（公式） | — | — |
| fla naive.py（纯 PyTorch） | — | — | — | 🟢 可直接抄 | — | — |
| fla chunk/fused_recurrent（Triton） | — | — | — | 🔵 仅参考（kernel 层） | — | — |
| HF modular_qwen3_next.py fallback | — | — | 🟢 可直接抄（部分） | 🟢 可直接抄 | — | — |
| HF Qwen3NextGatedDeltaNet 层装配 | — | — | 🟡 需对齐配置 | 🟡 需对齐配置 | — | — |
| 短因果卷积（kernel 4）实现要点 | — | — | — | 🟢 可直接抄 | — | — |
| partial-RoPE / M-RoPE 实现要点 | — | — | 🟢/🟡 可直接抄/需改造 | — | — | — |
| NVIDIA AIPerf（LLMPerf 血脉） | 🔵 仅参考（口径） | — | — | — | — | 🔵 仅参考 |
| vLLM bench serve 方法论 | 🟡 算法直接用 | 🟡 算法直接用 | — | — | — | 🟡 算法直接用 |
| SGLang bench_serving | 🔵 仅参考 | — | — | — | — | 🔵 仅参考（goodput 范式） |
| IETF LLM Benchmarking Terminology 草案 | 🟢 可直接用（口径仲裁） | 🟢 可直接用 | — | — | — | 🟢 可直接用 |
| 动态请求流建模（Poisson/gamma + 成对采样） | 🟡 需改造 | — | — | — | — | — |

🟢 可直接用　🟡 需改造　🔵 仅参考

---

## 1. Serving 负载 / 测试数据集

### 1.1 总览速查

| # | 资源 | 类型 | 对应阶段 | 适配性 | 许可证 | 一句话理由 |
|---|---|---|---|---|---|---|
| 1 | ShareGPT_Vicuna_unfiltered | 真实对话（英文） | W3/W8 | 🟢 可直接用 | Apache-2.0（HF badge 已查证） | vLLM 事实标准负载，固定 seed 抽 500 条即行业基线（avg 202/179 token） |
| 2 | LMSYS-Chat-1M | 真实对话（154 语言） | W3 | 🟡 需改造 | 自定义协议（研究+商业可用，禁止转授，非 gated） | 官方均值 prompt 69.5 / resp 214.5 token 是引用长度分布的最干净数字 |
| 3 | BurstGPT | 生产请求 trace（Azure） | W3 | 🟢 可直接用 | CC-BY-4.0（GitHub 已查证） | 唯一带真实时间戳的成熟公开 trace，W3 poisson 到达直接 replay |
| 4 | LongBench v1 | 长文本评测（中/英） | W4/W8 | 🟢 可直接用 | MIT（GitHub 已查证） | 4,750 条 5k–15k，LongBench-E 按 0–4k/4–8k/8k+ 三档均匀采样，与项目长度桶协议天然对齐 |
| 5 | LongBench v2 | 长上下文 MCQ（zai-org） | W4 | 🟡 需改造 | Apache-2.0（HF badge 已查证） | 503 条 8k–2M words（多数 <128k），只取 context 压 KV，不评对错 |
| 6 | ∞Bench / InfiniteBench | 超长上下文（100k–2M token） | W4 | 🔵 仅参考 | MIT（GitHub 已查证） | Retrieve.PassKey 类合成任务适合画 100k 档显存曲线，单卡只宜抽极小子集 |
| 7 | RULER（NVIDIA） | 合成长上下文生成器（非静态集） | W4 | 🟢 可直接用 | Apache-2.0（GitHub 已查证） | 按目标长度（4k→128k+）逐档生成、seed 可复现，匹配 W4 长度扫描 |
| 8 | 动态长度生成法（方法） | 方法论 | W3 | 🟡 需改造 | — | 从真实语料成对采样 (prompt_len, completion_len)，而非解析分布随机数 |

### 1.2 真实请求轨迹 / 对话数据集

#### A1. ShareGPT_Vicuna_unfiltered —— vLLM 业界事实标准负载

- **对应阶段**：W3（变长请求流的长度分布来源）、W8（上下文×batch 矩阵中的对话类负载）。
- **适配性**：🟢 **可直接用**。vLLM 官方 benchmark 文档直接给出下载命令，是 serving 论文最常引用的负载集。
- **许可证**：**【已查证】** HF 数据卡 license badge = `apache-2.0`；非 gated。注意：该 badge 是数据集上传者（anon8231489123）的声明；数据本体源自用户分享的 ChatGPT 对话，再分发时应保留来源说明。
- **规模**：**【已查证】** HF 页面 Total file size = 4.35 GB；数据卡自述约 10 万条 ShareGPT 对话经英文过滤、去重、去「AI 说教」后剩约 **53k 条**。vLLM 实际只用其中单个文件 `ShareGPT_V3_unfiltered_cleaned_split.json`。另：NVIDIA AIPerf 文档称其自动预 tokenize **73,499 段**对话（口径与数据卡自述不同，两处数字各自来源于官方文档）。
- **获取**：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered（vLLM 文档单文件直链：`.../resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json`）。
- **时效性**：数据集冻结于 2023 年，无更新但持续被 vLLM/SGLang 默认引用（2026-10 vLLM 文档仍在分发）。
- **值得借鉴**：vLLM perf 报告标准用法 = 固定随机种子抽 500 条，平均输入 202 / 输出 179 token——可直接作为 closed-loop loadgen 的对话类基线长度。
- **坑**：纯英文（已剔除中/韩文），与 `bench/prompts.json` 的中文集互补、不要混用比较；对话已按 2048 token 切块，取首条 user query 作单请求 prompt 即可；单卡引擎**无需全量 53k**，抽 200–500 条按长度分桶即可满足 repeats=5。
- **来源**：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered ；https://docs.vllm.ai/en/stable/contributing/benchmarks.html ；https://blog.vllm.ai/2024/09/05/perf-update.html

#### A2. LMSYS-Chat-1M

- **对应阶段**：W3（真实 prompt/response 长度分布统计）。
- **适配性**：🟡 **需改造**。适合取子集或只引用其分布统计，不建议整包当 loadgen 语料。
- **许可证**：**【已查证】** HF 数据卡**没有标准 SPDX badge**，而是附《LMSYS-Chat-1M Dataset License Agreement》自定义协议：授予研究与商业使用的有限、不可转让、不可再许可权利，**明确禁止 "distribute, copy, disclose, transfer, sublicense, embed, host…to any third party"**，并保留随时要求删除数据的权利。非 gated；但未脱敏原始对话需填表申请，HF 公开版本已做姓名脱敏（NAME_1/NAME_2）。**第三方页面流传的 CC-BY-NC 说法与官方自定义协议不符，以官方页面为准。**
- **规模**：**【已查证】** 1,000,000 条对话、25 个模型、210,479 个用户；Total file size = 1.49 GB（parquet）；官方统计：平均每对话 2.0 轮，**平均 prompt 69.5 token、平均 response 214.5 token**。
- **获取**：https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- **时效性**：采集于 2023 年 4–8 月，静态；仍在正常分发。
- **值得借鉴**：69.5/214.5 这对官方 token 均值是引用「真实对话长度分布」时最干净的数字；多轮结构可用于 W3 模拟会话式负载。
- **坑**：协议禁止再分发——**不要把整包数据提交进仓库**，只本地使用、报告引用统计数字；含未过滤的不安全对话，随机抽取前按 moderation tag 过滤；全量 100 万条对单卡无意义，建议 `language=="English"` 过滤后随机抽 1–5k 条。
- **来源**：https://huggingface.co/datasets/lmsys/lmsys-chat-1m ；论文 https://arxiv.org/abs/2309.11998

#### A3. BurstGPT（Azure OpenAI 生产请求轨迹）

- **类型**：真实生产 LLM serving 请求 trace（带到达时间戳），**不是文本语料**。
- **对应阶段**：W3（到达过程负载——唯一带真实时间戳的成熟公开 trace）。
- **适配性**：🟢 **可直接用**（作为到达时间与请求/响应长度序列）；需改造的只是「补 prompt 文本」（见坑）。
- **许可证**：**【已查证】** GitHub 仓库 license = `CC-BY-4.0`。
- **规模**：**【已查证】** README：BurstGPT_1（142.97 万行，约 2 个月）+ BurstGPT_2（385.84 万行）+ BurstGPT_3（534.4 万行，另 110 天）；单文件 CSV 约 188–220 MB。论文（KDD'25）总结共 **10.31M 条 trace、覆盖 213 天**。
- **Schema（已查证）**：`Timestamp`（秒）、`Session ID`、`Elapsed time`、`Model`（ChatGPT/GPT-4）、`Request tokens`、`Response tokens`、`Total tokens`、`Log Type`。
- **获取**：https://github.com/HPMLL/BurstGPT（Release v2.0 CSV；vLLM 用 `BurstGPT_without_fails_2.csv`）。
- **时效性**：KDD 2025 发表；vLLM（`burstgpt` dataset 类型）与 NVIDIA AIPerf 2026 文档均在引用，活跃。
- **值得借鉴**：README 官方建议两种用法——① 按目标 RPS 整体缩放时间轴重放；② 按论文拟合到达/会话模型后参数化生成。这正是 W3 接 poisson 到达率时最该抄的作业：**不必自己造到达过程，直接 replay BurstGPT 的 Timestamp 序列**。
- **坑**：**trace 里没有 prompt 文本，只有 token 数**。vLLM 做法是用随机 token id 按 Request tokens 长度合成假 prompt（只压 prefill/KV）；若 minillm 要「长度真实 + 文本真实」，可把 (到达时刻, 输入长度, 输出长度) 三元组取出，再从 ShareGPT 按长度就近抽样真实文本配对。vLLM 默认只用 `Model=="GPT-4"` 且 response tokens>0 的行。
- **来源**：https://github.com/HPMLL/BurstGPT ；https://docs.vllm.ai/en/v0.10.2/api/vllm/benchmarks/datasets.html ；https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-burst-gpt-traces.md

### 1.3 长上下文基准（显存探测 / 分页 KV 压力）

#### B1. LongBench v1（THUDM/LongBench）

- **对应阶段**：W4（分页 KV 长上下文压力）、W8（上下文 2k/8k 档）。
- **适配性**：🟢 **可直接用**（把 context 字段直接当长 prompt 喂引擎，不需要跑评测答案）。
- **许可证**：**【已查证】** GitHub `THUDM/LongBench` license badge = `MIT`；HF 数据卡未打 badge（ModelScope/CAICT 镜像标 Apache-2.0 为【一方称】，以 GitHub MIT 为准）。非 gated。
- **规模**：**【已查证】** 共 **4,750 条测试数据**，多数任务平均长度 5k–15k（英文按词、中文按字符）；HF 侧 8,418 rows、parquet 161 MB、原始文件 3.87 GB。
- **获取**：https://huggingface.co/datasets/THUDM/LongBench（按任务名 load_dataset）
- **值得借鉴**：**LongBench-E 子集**在 0–4k / 4–8k / 8k+ 三档长度均匀采样——与项目「按长度桶报告 CV」的测量协议天然对齐，W4 做「同显存 max concurrency ≥1.5×」可直接用它选样本；含中文任务（multifieldqa_zh、dureader、vcsum、lsht、passage_retrieval_zh），与现有中文 prompts.json 衔接。
- **坑**：长度单位是英文单词数/中文字符数、不是 token 数，进 loadgen 前需用 Qwen tokenizer 实测重分桶；只取 `context`（+短 `input`）字段即可。
- **来源**：https://huggingface.co/datasets/THUDM/LongBench ；https://github.com/THUDM/LongBench

#### B2. LongBench v2（zai-org/LongBench-v2）

- **对应阶段**：W4。
- **适配性**：🟡 **需改造**。它是 MCQ 评测集，serving 用途只取超长 context 当 KV 压力负载，不评对错。
- **许可证**：**【已查证】** HF 数据卡 license badge = `apache-2.0`，非 gated。注：HF 上已从 `THUDM/LongBench-v2` 迁移到 **`zai-org/LongBench-v2`**。
- **规模**：**【已查证】** **503 条 MCQ**，context 8k–2M words、**多数 <128k**；Total file size = 465 MB（单行 parquet 达 384 MiB，说明 context 极长）。
- **获取**：https://huggingface.co/datasets/zai-org/LongBench-v2
- **值得借鉴**：长度跨度覆盖 W4 的 8k→128k 扫描带；长对话/代码库/长结构化数据等真实场景比合成 NIAH 更接近真实 KV 访问模式。
- **坑**：单条 384 MiB，0.6B/0.8B 在 4090 上单条 128k+ prefill 很慢——W4 只取 8k–32k 档样本做 concurrency 扫描，2M words 档对单卡不现实。
- **来源**：https://huggingface.co/datasets/zai-org/LongBench-v2 ；论文 https://arxiv.org/abs/2412.15204

#### B3. ∞Bench / InfiniteBench（OpenBMB）

- **对应阶段**：W4（极端长度显存探测，与 W0「显存交叉点 ≈193 token」那类扫描互补）。
- **适配性**：🔵 **仅参考**（单卡只宜抽极小子集）。
- **许可证**：**【已查证】** GitHub `OpenBMB/InfiniteBench` license = `MIT`。
- **规模**：**【已查证】** 各任务 50–590 条，平均输入 token 从 43.9k（Math.Calc）到 2,068.6k（Zh.QA），多数任务 75k–200k。
- **获取**：https://github.com/OpenBMB/InfiniteBench（数据在 HF `xufu1999/InfiniteBench` 等镜像，镜像许可【一方称】未逐一核实）
- **值得借鉴**：Retrieve.PassKey / Retrieve.Number 这类合成任务输入固定、输出仅几 token，是干净的「只测 prefill+KV 显存」探针，适合测分页 KV 在 100k 档的显存占用曲线。
- **坑**：200k+ token 单条 prefill 在单卡上要分钟级；0.6B/0.8B 原生窗口（256k）虽够，实测需显式开长上下文 scaling。
- **来源**：https://github.com/OpenBMB/InfiniteBench ；论文 https://arxiv.org/abs/2402.13718

#### B4. RULER（NVIDIA/RULER）—— 合成长上下文 prompt 生成器

- **类型**：**不是静态数据集，而是可配置长度的合成数据生成器**（13 任务：NIAH 检索、variable_tracking 多跳追踪、聚合、QA）。
- **对应阶段**：W4（最匹配：按目标上下文长度精确生成 KV 压力）。
- **适配性**：🟢 **可直接用**。
- **许可证**：**【已查证】** GitHub `NVIDIA/RULER` license = `Apache-2.0`。
- **规模**：无固定规模——输入长度、任务复杂度（needle 数、chain 数、haystack 类型）全部通过 `scripts/synthetic.yaml` 配置，4k 到 128k+ 按需生成、给定 tokenizer+seed 可复现。
- **获取**：https://github.com/NVIDIA/RULER
- **值得借鉴**：项目 W0 做过「8k/32k 上下文显存节省 ×」扫描，RULER 这种**按目标长度逐档生成、seed 可复现**的模式，正是把显存探测从「几个点」变成「连续长度曲线」时该用的生成器；synthetic 文本无版权负担，可自由落盘进测量目录。
- **坑**：生成依赖 tokenizer（需接 Qwen3 tokenizer）；其 scorer/评测答案逻辑对 serving 无用，只取生成的 context 文本。
- **来源**：https://github.com/NVIDIA/RULER

### 1.4 动态长度负载生成（业界做法，方法类）

1. **vLLM 的 ShareGPT 采样法**【已查证 vLLM blog/文档】：固定随机种子抽 N 条（标准 N=500，avg input 202 / output 179 token）；`--random-input-len`/`--random-output-len` 可把样本截断/对齐到目标长度。即「真实语料 + 受控长度」。
2. **BurstGPT 重放**【已查证 README】：保留真实 Timestamp 序列按目标 RPS 整体缩放；同时保留真实 Request/Response tokens 分布。W3 接 poisson 到达时可直接以其到达间隔为基线做叠加。
3. **真实长度分布数字（引用用）**：LMSYS-Chat-1M 官方均值 prompt 69.5 / response 214.5 token；ShareGPT（vLLM 实测）202 / 179。两者差异来自过滤口径，报告时注明出处。
4. **IETF benchmarking 草案建议**（见 §4.5）：对话类负载的输入/输出长度直接取 ShareGPT vicuna_cleaned 子集的经验分布，并让 50% 请求共享同一段 200-token system prompt 来模拟 prefix sharing——对后续测前缀缓存复用率有参考价值。

### 1.5 单卡（0.6B/0.8B @ 4090）取用建议速查

| 资源 | 全量规模 | 单卡建议用量 |
|---|---|---|
| ShareGPT_Vicuna_unfiltered | ~53k 条 / 4.35 GB | 抽 200–500 条做 W3 对话请求池 |
| LMSYS-Chat-1M | 1M 条 / 1.49 GB | 不整包下载；引用 69.5/214.5 统计即可，或本地过滤英文后抽 1–5k |
| BurstGPT | 10.31M 行 / ~600 MB CSV | 取 `BurstGPT_without_fails_2.csv` 一个文件即可，replay 其时间戳 |
| LongBench v1 | 4,750 条 | 用 LongBench-E 三档各抽 20–50 条 |
| LongBench v2 | 503 条 / 465 MB | 只取 ≤32k 档样本做 W4 concurrency 扫描 |
| ∞Bench | 数千条、100k–2M token | 仅抽 Retrieve 类短例子画显存曲线 |
| RULER | 按需生成 | 按 2k/8k/32k/128k 目标长度各生成几十条，seed 固定 |

---

## 2. 参考项目（连续批处理 / 分页 KV / 块管理 / 双状态管理）

### 2.1 总览速查

| # | 项目 | 一句话理由 | 阶段 | 适配性 | 许可证 |
|---|---|---|---|---|---|
| 1 | vLLM (V1 engine) | 分页 KV + 连续批的工业界标准答案，但代码量与分布式/CUDA graph 耦合重 | W3/W4/W7 | 🔵 仅参考（机制） | Apache-2.0 |
| 2 | SGLang | RadixAttention 前缀树复用 + chunked prefill，前缀共享思路对 W4/W7 有价值 | W4/W5-W6 | 🔵 仅参考 | Apache-2.0 |
| 3 | mini-sglang | SGLang 官方教学版 ~5000 行 Python，默认跑 Qwen3-0.6B，模块边界干净 | W3/W4/W5-W6 | 🟡 需改造（Linux/CUDA kernel 依赖） | MIT |
| 4 | nano-vllm | ~1200 行手写 vLLM，scheduler/block_manager 各一个文件，W3/W4 最直接参照系 | W3/W4 | 🟡 需改造（可选关掉 torch.compile/CUDA graph） | MIT |
| 5 | LightLLM | token 级细粒度调度 + 纯 Python KV 管理，已有 state_cache_manager 与 qwen3next 混合注意力适配 | W3/W4/W7 | 🟡 需改造 | Apache-2.0 |
| 6 | vattention | CUDA 虚拟内存 API 替代 PagedAttention 块重排，思路新颖但依赖驱动级 API | W4 | 🔵 仅参考（与纯 PyTorch 约束冲突） | MIT |
| 7 | TensorRT-LLM | NVIDIA 官方 batching + KV cache manager，C++/TensorRT 编译链与纯 PyTorch 路线不匹配 | W3/W4 | 🔵 仅参考 | Apache-2.0 |
| 8 | PagedAttention 论文 | vLLM 机制的原始出处，W4 块池/block table/refcount 的概念依据 | W4 | 🔵 仅参考（论文） | — |

### 2.2 条目详情

#### vLLM（V1 Engine）

- **仓库**：https://github.com/vllm-project/vllm 【已查证】；**许可证**：Apache-2.0【已查证】；**时效性**：极度活跃，95.9k stars / 24,218 commits；README 列 2026 年新闻（Day-0 support 等）。
- **值得抄的机制（→ 阶段）**：
  1. **waiting/running/swapped 三队列调度**（→ W3）：入口 `vllm/v1/core/sched/scheduler.py`、`request_queue.py`。每 iteration 先摘除完成序列、回收块，再从 waiting 按优先级提升，块不够时抢占。minillm W3 只需 waiting/running 两队列 + 按可用块数准入，**不需要 swapping**（单卡显存不是瓶颈）。
  2. **BlockPool + KVCacheManager**（→ W4）：入口 `vllm/v1/core/block_pool.py`、`single_type_kv_cache_manager.py`。物理块池一次性申请连续 GPU 内存切 `block_size`（vLLM 默认 16）物理块；block table = sequence → 物理块 id；refcount 处理前缀共享与 COW。W4 验收「同显存 max concurrency ≥1.5×」的核心机制就是这套。
  3. **前缀缓存**（→ W4 可选加分项）：V1 起零开销 CPU 侧前缀匹配；minillm W4 可先不做 COW，只做「相同前缀序列复用已分配块」的 refcount。
  4. **多类型 KV cache manager**（→ W7）：`vllm/v1/core/kv_cache_coordinator.py` 协调多种 cache type（full attention KV + encoder cache + SSM state），W7「KV 块池 + 线性状态池」统一句柄可参考这个协调器抽象。
- **与单卡 0.6B/0.8B 不匹配**（【已查证】README/目录树）：分布式并行全家桶（TP/PP/EP、Ray、disaggregated）、CUDA graph 全流程（D3 铁律推迟到 W4 后）、varlen/FlashAttention/FlashInfer/Triton kernel 后端（D3 禁止 varlen kernel）、量化/MoE/投机/LoRA/多模态——全部不看；代码量几十万行，必须按上表精确文件路径单点切入。
- **坑**：V1 引擎（2025-01 alpha 起）已是默认，老的 `vllm/engine/`、`vllm/core/`（V0）代码已过时，**不要读错目录**；vLLM 的 schedule 混 prefill+decode（chunked prefill），minillm W3 可先把 prefill 和 decode 严格分开。

#### SGLang

- **仓库**：https://github.com/sgl-project/sglang 【已查证】；Apache-2.0；极度活跃（38.3k stars）。
- **值得抄的机制（→ 阶段）**：
  1. **RadixAttention / RadixCache**（→ W4 前缀共享）：基数树组织存活序列的 KV 块，公共前缀自动复用、LRU 淘汰尾节点。比 vLLM 哈希前缀匹配更显式、数据结构直观；0.6B/0.8B 单卡前缀命中率不高，但作为机制展示是加分项。
  2. **Chunked Prefill**（→ W5-W6 / W3 后段）：长 prefill 切 chunk 与 decode 混跑，避免长 prefill 独占 GPU。minillm D3 无 varlen kernel，chunked prefill 在 minillm 就是「把一条长 prompt 拆成几次 forward」，调度器层面实现。
  3. **Overlap Scheduling**（→ W3）：CPU 调度与 GPU 计算重叠【一方称，lmsys 博客】；0.6B 量级下可先同步调度，overlap 作 W3 后段优化。
- **不匹配**：生产代码耦合 FlashInfer / sgl-kernel JIT、CUDA graph、TP/PP/DP、RL/diffusion/多模态——全部不读；调度器与模型执行深度耦合，单读成本高，**建议直接读 mini-sglang**。
- **坑**：RadixCache 的块复用嵌在 radix tree 节点分裂/合并里，比 vLLM refcount 更绕；只做最简前缀共享时 vLLM 的 block_pool refcount 更好抄。

#### mini-sglang（SGLang 官方教学版）

- **仓库**：https://github.com/sgl-project/mini-sglang 【已查证】；MIT；活跃（5.5k stars / 171 commits）；2025-12-17 LMSYS 博客发布，README 主推 **`Qwen/Qwen3-0.6B` 单卡部署——与载体模型 A 完全一致**。
- **值得抄**：~5000 行全类型标注 Python，模块边界干净（`python/minisgl/{scheduler, kvcache, engine, attention, kernel, layers, models}`）；`--cache radix` / `--cache naive` 开关对 W4 很有参考价值（先 naive 块池再加 radix 前缀复用）；chunked prefill 纯调度层实现；默认跑 Qwen3-0.6B 可直接对照模型加载与注意力路径。
- **不匹配**：**仅支持 Linux**（依赖 sgl-kernel/flashinfer JIT，M1 上不能跑，只能 4090 跑或读代码）；集成 FlashAttention/FlashInfer 后端需剥掉（D3 要求纯 PyTorch 逐序列 attention）；有 TP 分支不看。
- **坑**：171 commits 发布不到一年，API 可能变动，读时 checkout 一个 tag；kernel 目录是 C++/CUDA JIT，**不要陷入 kernel 实现**。
- **代码切入点**：`python/minisgl/scheduler/`（调度主循环）、`python/minisgl/kvcache/`（naive vs radix 对照）、`python/minisgl/attention/`（block table 数据结构，不抄 kernel）、`docs/features.md`。

#### nano-vllm（GeeeekExplorer）

- **仓库**：https://github.com/GeeeekExplorer/nano-vllm 【已查证】；MIT；16.2k stars / 67 commits（代码稳定、不再高频迭代，适合做冻结参照系）；README benchmark 用 Qwen3-0.6B 在 8GB 笔记本 GPU 上跑，与单卡小模型场景吻合。
- **值得抄（这五个文件几乎就是 W3+W4 手写实现的骨架）**：
  - `nanovllm/engine/scheduler.py` —— waiting/running 队列、iteration 级连续批调度、preempt（**W3 主抄**）
  - `nanovllm/engine/block_manager.py` —— 物理块池、block table、按需分配/回收（**W4 主抄**）
  - `nanovllm/engine/llm_engine.py` —— engine 主循环 schedule → model_runner → postprocess
  - `nanovllm/engine/model_runner.py` —— scheduled seqs 拼 batch、调模型
  - `nanovllm/engine/sequence.py` —— Sequence 数据类（状态、block table、输出 token）
  - 模块边界与 `serve/{scheduler,block_manager,engine,state_pool}` 命名可直接对照。
- **不匹配**：README 显示开启 torch.compile / CUDA graph 才能接近 vLLM 吞吐——minillm D3 推迟 CUDA graph，W3/W4 用 `enforce_eager=True` 跑即可；有 TP 分支不看；其 attention 是简化手写版（非 FlashAttention），反而接近 minillm 的「纯 PyTorch 一条注意力路径」约束。
- **坑**：第三方解读博客提到 Qwen3 加载有 `rope_scaling` 默认值 bug 需 patch【一方称，cefboud.com 博客】——minillm 已 W1 对齐 HF，不会走它的模型加载路径；注意 **原版是 `GeeeekExplorer/nano-vllm`**，`ovshake/nano-vllm` 是带教学文章的 fork。

#### LightLLM（ModelTC）

- **仓库**：https://github.com/ModelTC/LightLLM 【已查证】；Apache-2.0；4.3k stars / 1,029 commits；最近 release v1.1.0（2025-09）；README 自称「pure-python design and token-level KV cache management make it easy to use as the basis for research projects」——定位正合需要。
- **值得抄**：
  1. **token 级 KV cache 管理**（→ W4）：`lightllm/common/kv_cache_mem_manager/allocator.py` + `mem_manager.py` 纯 Python 块分配器，比 vLLM 轻；不是 PagedAttention 式 block，而是 token 级细粒度管理——D3 禁止 varlen kernel 的纯 PyTorch 路径下，token 级/小块管理反而更自然。
  2. **`state_cache_manager.py`**（→ **W7 直接相关**）：为 Mamba/SSM 类线性注意力模型做的**常数大小状态池**，与 W7「18 层 Gated DeltaNet 线性状态池」直接对应。
  3. **`qwen3next_mem_manager.py`**【已查证】：LightLLM 已为 Qwen3-Next 混合注意力模型做过 KV + state **双内存管理器**适配——W7 最直接参考。
  4. **Past-Future Scheduler**（ASPLOS'25，→ W3）：带预测的峰值显存调度；可借鉴「预测下一步需要多少块」思路，不必照搬 SLA 复杂度。
- **不匹配**：TP/DP/EP、MoE、量化、Triton kernel——不看；多进程架构（Triton server、输入编码/推理分离）——minillm 单进程即可。
- **坑**：mem_manager 有大量模型专属子类（deepseek2/3/4、glm5_next 等），**只看基类 `mem_manager.py` + `qwen3next_mem_manager.py`**；其 KV 布局是 token 级而非 vLLM block 级——W4 选型要决定走哪条：考虑到 D3，**LightLLM 式纯 Python 块管理 + 逐序列 attention 可能比 vLLM 的 PagedAttention 更适配 minillm**。

#### vattention（Microsoft）

- **仓库**：https://github.com/microsoft/vattention 【已查证】；MIT；526 stars / 126 commits，研究原型、不再活跃迭代。
- **核心思路**【已查证 README】：不用用户态块重排，而用 **CUDA 虚拟内存 API**——申请连续虚拟地址、运行时按需映射物理页，KV 在虚拟地址上连续，未修改的 attention kernel（FlashAttention-2/FlashInfer）直接跑。
- **适配判断**：**不匹配**——依赖 CUDA driver 级 API（`cuMemCreate`/`cuMemMap`）、C++/CUDA 扩展编译，与纯 PyTorch 约束冲突；小页需替换 NVIDIA UVM 内核驱动，学生项目不可能做；它解决的问题对 minillm 不存在（minillm 自己写逐序列 attention，block gather 在 PyTorch 里做）。
- **值得借鉴的思想**（→ W4 设计讨论）：「虚拟连续、物理离散」抽象是 W4 block table 的理论背书（软件版虚拟→物理映射）；论文对 PagedAttention「破坏虚拟连续性导致 kernel 重写成本」的批评提醒：**不要为了凑「PagedAttention」名词硬上 gather kernel**。读论文 https://arxiv.org/abs/2405.04437 即可，不建议读代码。

#### TensorRT-LLM（NVIDIA）

- **仓库**：https://github.com/NVIDIA/TensorRT-LLM 【已查证】；Apache-2.0；15k stars / 10,748 commits；2026-03 起完全开源迁 GitHub，v1.3（2026-09）发布周期内。
- **值得参考**：**In-flight batching + Paged KV cache**（KvCacheManager 预分配固定块、运行时动态分发；支持 `enable_block_reuse`/`enable_partial_reuse` 前缀复用）【已查证 NVIDIA 文档】；2026 年版本已迁移 PyTorch native runtime，距离比从前近。
- **不匹配**：重度 NVIDIA 栈（TensorRT 编译、C++ runtime、Blackwell/H100 优化）；万级 commits 双语言代码库学习成本高；KV cache manager v2/block reuse 是生产吞吐优化，0.6B/0.8B 用不到；`use_uvm`/`sink_token_length` 已 deprecated【一方称】，API 剧烈变动不适合做稳定参照。
- **代码切入点**：不建议读代码；仅参考 KV cache 文档 https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html（了解工业界 block reuse 参数语义）。

#### PagedAttention 论文（机制依据）

- **论文**：Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023。【已查证 vLLM README 引用】；arXiv https://arxiv.org/abs/2309.06180 （arXiv 编号【一方称】，未单独 fetch 验证）。
- **对应阶段**：W4。物理块池、block table、固定 size block、内部碎片 <4%、preempt with recompute/swapping 概念全部出自此文。minillm W4 不做 swapping（单卡显存不紧张），但 block table + free list + refcount 的原始定义以这篇为准。
- **与 D3 的关系**：论文的 PagedAttention kernel（自定义 CUDA kernel 做块间 attention）minillm 不实现；只做**内存管理抽象层**（block table），attention 走纯 PyTorch 逐序列 gather——这是合法的「抄机制、不抄 kernel」。

### 2.3 给 minillm 的选型建议（按阶段）

- **W3 连续批处理**：主抄 nano-vllm 的 `scheduler.py` + `sequence.py`（~1200 行总量、文件最少）；对照 mini-sglang `scheduler/`；概念参考 vLLM V1 三队列与 preempt 思路（跳过 swapping/分布式/CUDA graph）；不要 TensorRT-LLM、vattention 的调度部分。
- **W4 分页 KV**：主抄 nano-vllm `block_manager.py`；对照 LightLLM `kv_cache_mem_manager/mem_manager.py` + `allocator.py`（纯 Python、更贴合纯 PyTorch）；前缀共享加分项用 mini-sglang `kvcache/` radix；理论依据 PagedAttention 论文（block_size 16/32、碎片率定义）；不要 vattention 的 CUDA VMM 路线、不要 TRT-LLM KvCacheManager v2。
- **W7 StatePool 双状态管理**：主参考 **LightLLM `state_cache_manager.py` + `qwen3next_mem_manager.py`**——这是唯一一个把「全注意力 KV 块池 + 线性注意力常数状态池」放在同一 mem manager 体系里的开源项目，且模型族（Qwen3-Next 混合注意力）与 Qwen3.5-0.8B（3:1 混合）最接近；对照 vLLM V1 `kv_cache_coordinator.py`（多 cache type 协调抽象）。机制映射：6 层全注意力 → W4 已建分页 KV 块池；18 层 Gated DeltaNet → 常数大小状态张量池（每序列固定 shape，不分块）；统一句柄由 scheduler 分配/回收。
- **规模匹配提醒**：0.6B/0.8B bf16 权重仅 ~1.6G，4090 显存不是瓶颈；W4「max concurrency ≥1.5×」验收靠**消除内部碎片**，不靠压显存上限。简历项目规模：nano-vllm（1200 行）+ mini-sglang（5000 行）已演示全部核心机制；vLLM/SGLang/TRT-LLM 只作背景阅读，**不要试图复刻其功能广度**。

---

## 3. 算子实现参考（Gated DeltaNet 分块 prefill + 逐步 decode）

### 3.1 总览速查

| # | 资源 | 类型 | 阶段 | 适配性 | 许可证 |
|---|---|---|---|---|---|
| 1 | Gated DeltaNet 论文 (arXiv 2412.06464) | 论文 | W6 | 🔵 仅参考（公式定义） | CC BY 4.0 |
| 2 | fla naive.py（纯 PyTorch 参考） | 代码库 | W6 | 🟢 **可直接抄** | MIT |
| 3 | fla chunk.py / fused_recurrent.py（Triton） | 代码库 | W6 | 🔵 仅参考（kernel 层） | MIT |
| 4 | HF `modular_qwen3_next.py` 纯 PyTorch fallback | 官方源码 | W5+W6 | 🟢 **可直接抄** | Apache-2.0 |
| 5 | HF `Qwen3NextGatedDeltaNet` 层装配 | 官方源码 | W6 | 🟡 需改造（对齐 0.8B 配置） | Apache-2.0 |
| 6 | 短因果卷积（kernel=4） | 官方源码片段 | W6 | 🟢 可直接抄（几行 F.conv1d） | Apache-2.0 |
| 7 | partial-RoPE（1/4 维旋转） | 官方源码片段 | W5 | 🟢 可直接抄 | Apache-2.0 |
| 8 | M-RoPE（mrope_interleaved，文本退化 1D） | 官方源码 | W5 | 🟡 需改造（文本路径） | Apache-2.0 |

### 3.2 Gated DeltaNet 论文（公式权威定义）

- **标题**：Gated Delta Networks: Improving Mamba2 with Delta Rule；**作者**：Songlin Yang, Jan Kautz, Ali Hatamizadeh（NVIDIA Research）；**arXiv**：2412.06464（v1 2024-12-09，v3 2025-03-06）；**会议**：ICLR 2025 camera ready（arXiv Comments 字段标注）；**许可**：CC BY 4.0。
- **来源**：https://arxiv.org/abs/2412.06464
- **核心公式（gated delta rule）**：遗忘 `h_t = α_t · h_{t−1}`（α_t∈(0,1)，来自 g 的 log-space 指数）；读-改-写 `ν_t = β_t · (v_t − h_{t−1}ᵀ k_t)`；写 `h_t += k_t ⊗ ν_t`（秩-1 外积）。**分块并行**：块内用 WY 表示（C 步串行秩-1 更新压缩成一次矩阵乘），块间用常数大小状态 h 递推——这正是 W6「分块 prefill == 逐步 decode」的算法基础。Q/K 做 ℓ2 归一化（Qwen3.5 配置 `use_qk_l2norm_in_kernel=True`）。
- **坑**：论文记号 g 与实现不完全一致——论文写 `α_t = exp(g_t)`，实现里 g 已是 log-space 预计算量（负值），直接 `g.exp()` 即 α，**不要重新套 softplus/sigmoid**；论文公式是训练视角（含反向），推理只需前向，WY 表示里的逆在纯 PyTorch 参考实现里已用 C 步串行三角循环替代，可读性更高。

### 3.3 fla（flash-linear-attention）—— 最重要的参考实现

- **仓库**：https://github.com/fla-org/flash-linear-attention ；**许可证**：MIT（`fla/ops/gated_delta_rule/naive.py` 文件头明确标注）；**时效性**：PyPI `flash-linear-attention==0.5.1` 发布于 2026-10-21，main 活跃（含 GDN-2、OSDN 等 2026 年新算子），活跃度高。
- **对应目录**：`fla/ops/gated_delta_rule/`——`naive.py`（**纯 PyTorch 参考：recurrent + chunk 两个版本，全 fp32**）、`chunk.py`（Triton 封装 `chunk_gated_delta_rule`）、`chunk_fwd.py`（块内 WY）、`fused_recurrent.py`（Triton decode kernel）、`gate.py`、`wy_fast.py`。

**3.3.1 `naive_recurrent_gated_delta_rule`（decode 循环形式，可直接抄）**

接口约定（**decode 循环必须严格对齐这个状态布局**）：输入 `q,k [B,T,H,K]`、`v [B,T,H,V]`、`beta [B,T,H]`（已 sigmoid）、`g [B,T,H]`（log-space 负值）；状态 `h [B,H,K,V]` 初始化为 0 或传入 initial_state；`scale = 1/sqrt(K)` 在循环外对 q 乘一次；所有张量循环前 `.to(torch.float32)`。

逐 token 循环（W6「逐步 decode」金标准）：
```python
h = h * g[:, :, i].exp()[..., None, None]                    # 遗忘
b_v = v[:, :, i].clone() - (h.clone() * k[:, :, i][..., None]).sum(-2)  # δ = v − hᵀk
b_v = b_v * beta[:, :, i][..., None]                         # ν = β·δ
h = h.clone() + k[:, :, i].unsqueeze(-1) * b_v.unsqueeze(-2) # h += k⊗ν
o[:, :, i] = torch.einsum('bhd,bhdm->bhm', q[:, :, i], h)    # o = q·h
```
**状态布局是 `[K,V]`（K=key_dim=128, V=value_dim=128），不是 `[V,K]`**（fla 有 `state_v_first=True` 切换，minillm 直接用默认）。`h.clone()` 是为 autograd 正确，推理可 inplace，但对拍建议先不 inplace。naive 版**没有 q/k L2 归一化**——假设调用方已归一化（Qwen3.5 在 HF 侧做了，见 §3.5）。

**3.3.2 `naive_chunk_gated_delta_rule`（分块 prefill 形式，可直接抄）**

chunk_size 参数化（Triton 版只允许 16/32/64，naive 版任意）。两阶段：
- **阶段 A（块内 WY，并行）**：`pad_len = (C − T % C) % C` 对 q/k/v/beta/g 做 F.pad（beta/g 补 0）；`v = v * beta`、`k_beta = k * beta`（β 吸收进 v/k）；块内 decay `= g.cumsum(-1)`、`decay_exp = decay.exp()`；`L_mask = tril(exp(decay_i − decay_j))`；`attn = −(k_beta @ kᵀ * L_mask)` 对角线 masked_fill(0)；**WY 三角递推**（C 步串行，C=64 很小纯 PyTorch 可接受）；`k_cumsum = attn @ v_beta`、`k_cumdecay = attn @ (k_beta * decay_exp)`。
- **阶段 B（块间递推，串行 num_chunks 步）**：`v_new = v_i − k_cumdecay_i @ S`；`o_i = (q_i * decay_exp_i) @ S + intra_attn_i @ v_new`；`S = S * chunk_decay_i + (k_i 带衰减)ᵀ @ v_new`。最后 `o[:, :, :T]` 去掉 padding。
- **这就是 W6「分块 prefill == 逐步 decode」对拍的两端**：阶段 B 的 S 递推必须与 §3.3.1 的逐 token 循环逐 token 一致。

**3.3.3 chunk_size 建议**：minillm 先固定 **chunk_size=64**（与 HF/fla 默认一致），对拍通过后再参数化。chunk_size 不影响数学正确性，只影响块内并行度；**对拍时两端必须用同一个 chunk_size**。

**3.3.4 fla kernel 层（仅参考）**：`chunk.py`/`fused_recurrent.py` 处理 bf16 精度、TMA/warp specialization 等工程细节，minillm 不引入；但接口签名（`initial_state [N,HV,K,V]`、`output_final_state`、`cu_seqlens`）值得借鉴用于未来扩展。

### 3.4 HF transformers `modular_qwen3_next.py` —— 官方纯 PyTorch fallback

- **文件**：`src/transformers/models/qwen3_next/modular_qwen3_next.py`（`modeling_qwen3_next.py` 是它自动生成的，**改代码要改 modular 版**）；raw URL：https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modular_qwen3_next.py ；Apache-2.0（文件头标注）；2026-10 仍在维护。
- **为什么极有价值**：它用 `@use_kernel_func_from_hub_with_fallback("fused_recurrent_gated_delta_rule", "fla")` 装饰——hub kernel 找不到时 fallback 到纯 PyTorch 的 `torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule`。这两个 fallback 就是 **HF 自己写的、与 fla naive 等价但更贴近 Qwen3-Next 实际装配**的参考实现，可直接抄。
- **`torch_recurrent_gated_delta_rule`（decode）**：与 fla naive 几乎逐行等价，但多了：① `l2norm(query/key, eps=1e-6)` 在 fp32 下做（`use_qk_l2norm_in_kernel=True` 时）；② `query = query / sqrt(K)`。递归步与 fla naive 完全一致——**两端可以互相对拍**。
- **`torch_chunk_gated_delta_rule`（prefill）**：块内 WY 同一套三角递推；块间 `v_new = new_values − k_cumdecay @ state`、`out = inter_chunk_attn + intra_chunk_attn @ v_new`、`state = state * chunk_decay + keyᵀ @ v_new`。注意 HF 把 decay 乘在 q/key 上、fla naive 乘在输出 S 上——**数学等价，但对拍时不要混用两种写法**。

### 3.5 `Qwen3NextGatedDeltaNet` 层装配（W6 主要参照）

- **投影与短卷积**：`in_proj_qkvz`：hidden → key_dim*2 + value_dim*2（q,k,v,z 四股）；`in_proj_ba`：hidden → num_v_heads*2（b=beta logits, a=decay logits）；`conv_dim = key_dim*2 + value_dim`（只对 q,k,v 卷积，z 和 ba 不卷积）；`nn.Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim, bias=False, padding=3)` —— **深度维卷积**；激活 SiLU。
- **短因果卷积确切形式（可直接抄）**：Prefill `F.conv1d(x, weight, padding=3, groups=hidden)[:, :, :seq_len]` 后接 SiLU（左 pad 3 再取前 seq_len，标准 causal）。Decode 单 token：`hidden_new = cat([conv_state, x], dim=-1)`；`conv_state.copy_(hidden_new[:, :, -3:])`；`F.conv1d(hidden_new, weight, padding=0)[:, :, -seq_len:]`。**必须实现这个 conv_state：长度 = kernel_size − 1 = 3**。
- **g 与 beta 计算（已抓取）**：`beta = b.sigmoid()`；`g = −A_log.float().exp() * F.softplus(a.float() + dt_bias)`。A_log 可学习（初始化 `log(U(0.01,16))`）、dt_bias 初始化 1。**g 已是 log-space 负值，直接喂算子；beta 已 sigmoid 到 (0,1)。**
- **KV 头数**：Qwen3.5-0.8B 16 key / 16 value 头（1:1），无需 repeat_interleave；HF 代码里 `num_v_heads // num_k_heads > 1` 分支可保留备用。
- **输出门控 Norm（Qwen3NextRMSNormGated）**：`h = weight * h / sqrt(mean(h²) + eps)` 后乘 `silu(z)`——**norm before gate**。
- **全注意力层输出门控（W5 参照）**：`query_states, gate = chunk(q_proj(x).view(..., head_dim*2), 2, -1)`；attention 输出乘 `sigmoid(gate)` 再过 o_proj。**W5 验收「attn 输出门控」就照这个**。
- **Qwen3.5-0.8B 配置（用于对齐超参）**：以下来自第三方重实现 mfai（https://meteofrance.github.io/mfai/_modules/mfai/pytorch/models/llms/qwen3_5.html ）【**一方称**，非 HF config.json 直读】，但与任务给出的架构参数吻合：vocab 248320、context 262144、emb 1024、24 层、全注意力 n_heads=8/GQA 2/head_dim 256/qk_norm、rope_base 1e7、partial_rotary_factor 0.25、线性层 conv kernel 4 / key·value head_dim 128 / 16 头、`layer_types = (["linear_attention"]*3 + ["full_attention"])*6`。**真正对拍前以 HF `Qwen/Qwen3.5-0.8B` config.json 为准。**

### 3.6 partial-RoPE / M-RoPE

- **partial-RoPE（1/4 维，可直接抄）**：head_dim=256 → `rotary_dim = 64`（32 个复数对）；`inv_freq = 1/(theta_base^(arange(0,64,2)/64))`；只旋转 `x[..., :64]`，`x[..., 64:]` 原样透传；**theta_base = 1e7（不是常见的 1e4）**。
- **M-RoPE（mrope_interleaved，需改造）**：HF 通过 `rope_parameters["rope_type"]="mrope_interleaved"` 选择；section `[11,11,10]` 对应时间/高度/宽度三段；interleaving 模式把 inv_freq 按 3 路交错赋值（第三方 Qwen3-ASR 实现片段可参考）。**文本退化**：纯文本时三路坐标全等于同一个 t，mrope_interleaved 退化为标准 1D RoPE——minillm 只取 text model，**可先用标准 1D RoPE 对拍，再补 mrope 路径**。
- **测试建议**：① 文本退化测试：同一纯文本序列分别走 1D RoPE 与 mrope_interleaved（三路坐标都=t），cos/sin 必须逐元素相等；② section 分块测试：构造三路不同坐标，验证第 0..10 频率对用 t、11..21 用 h、22..31 用 w；③ 位置错位风险：**对拍时 assert `sum(mrope_section) == rotary_dim // 2`（=32）**，off-by-one 是高频 bug。

### 3.7 「分块 prefill == 逐步 decode」自洽性测试（W6 硬判据）

1. **状态缓存对齐**：prefill 结束取 `final_state [B,H,K,V]`（H=16, K=128, V=128，fp32；bf16 对拍时 upcast）；decode 第一步把它作为 `initial_state` 喂入循环。**必须确认 chunk prefill 输出的 final_state 是最后一个 chunk 结束后的 S**（fla naive 与 HF chunk 都在循环结束后返回 S）。
2. **chunk 边界处理**：T 非 chunk_size 整数倍时补零到 ceil(T/C)·C；**padding 位置输出必须丢弃**（取 `o[:, :, :T]`）；pad 位置 beta=0、g=0 数学上「不写」，但要确认 pad 的 q/k/v 不参与块内输出贡献（fla naive 直接 F.pad 后 mask 输出）。**边界 token 测试**：专门构造 T = C−1, C, C+1, 2C−1, 2C+1 验证边界处 final_state 一致。
3. **误差控制与容差**：参考实现全部 fp32（fla naive / HF fallback 都强制 `.to(torch.float32)`）。引擎 bf16 对拍时：引擎输出 upcast fp32 再与 fp32 参考比较；**推荐容差 `atol=1e-3, rtol=1e-3`**（bf16 累积 64 chunk 常见量级）；状态 h 单独比较 `atol=5e-3` 起步，不行放宽到 1e-2 并在 notes.md 记录实测误差。fp32 引擎对 fp32 参考 `atol=1e-5, rtol=1e-5` 应轻松通过。
4. **测试组织（建议顺序）**：① 随机权重、单头小 K/V（H=1,K=16,V=16,C=8,T=32）验证数学正确 → ② 随机权重真实头数（H=16,K=128,V=128,C=64,T=64/128/65）→ ③ 随机权重长序列 T=1024 测累积误差 → ④ 真实 Qwen3.5-0.8B 权重单层 GDN 与 HF 单层输出对拍 → ⑤ 全模型贪心 64 token 与 HF `generate(do_sample=False)` 对比，至少前 32 token 完全一致。
5. **因果性测试（W5/W6 通用）**：替换第 t 个 token，输出 logits 在位置 < t 必须逐元素相等（atol=1e-7 内）；对 GDN 层额外验证状态 h 在第 t 步前不受第 t 个 token 影响。

### 3.8 坑与注意事项汇总

1. **g 是 log-space**：fla/HF 都要求传入 log-space（负值），直接 `g.exp()`；不要在外面再套 logsigmoid/softplus。
2. **beta 已 sigmoid**：统一在外部 sigmoid 一次，算子内不再处理。
3. **q/k L2 归一化**：Qwen3.5 开启（eps=1e-6），在 fp32 下做，归一化后再 scale=1/sqrt(K)。
4. **状态布局 [K,V] vs [V,K]**：默认 [K,V]，缓存对齐时两端必须一致。
5. **conv_state 长度 = kernel_size − 1 = 3**，不是 4。
6. **partial-RoPE 只旋转前 1/4 维，theta_base=1e7**。
7. **M-RoPE section 和 = rotary_dim/2 = 32**，off-by-one 高频 bug。
8. **chunk_size 固定 64 先对拍**；改 chunk_size 会改变浮点累积顺序，对拍失败难以定位。
9. **naive 实现的 `.clone()`**：推理可 inplace，对拍版本保留 clone 避免 autograd 别名误导。
10. **HF `modeling_qwen3_next.py` 是自动生成的**——读参考直接读 modular 版。
11. **Qwen3.5-0.8B 的 model_type 是否复用 `qwen3_next` 目录未完全查证**（见 §6）——实际加载时 `AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")` 看 model_type 再定位确切源码。

---

## 4. 评测方法（负载生成 / 到达过程 / 指标口径）

### 4.1 总览速查

| # | 名称 | 类型 | 阶段 | 适配性 | 许可证 | 一句话理由 |
|---|---|---|---|---|---|---|
| 1 | NVIDIA AIPerf（LLMPerf 血脉） | 工具 | W3/W8 | 🔵 仅参考 | Apache-2.0（已查证） | TTFT 含排队 / ITL 排除 TTFT / 吞吐窗口口径的官方定义 |
| 2 | vLLM bench serve | 工具+方法 | W3/W4/W8 | 🟡 算法直接用、代码重写 | Apache-2.0（一方称） | gamma→Poisson 到达+总时长归一化、ShareGPT 成对采样、goodput、并发峰值全有源码级范本 |
| 3 | SGLang bench_serving | 工具+方法 | W3/W8 | 🔵 仅参考 | Apache-2.0（一方称） | goodput 扫 λ 找「SLO 内可持续吞吐」峰的报告范式 |
| 4 | IETF LLM Benchmarking Terminology 草案 | 标准草案 | 全程 | 🟢 可直接用 | 开放 IETF 草案 | open/closed-loop 定义、p99≥1000 样本量、warmup 必剔除的权威仲裁 |
| 5 | 动态流建模（Poisson/gamma + 成对采样） | 方法 | W3 | 🟡 需改造 | — | 间隔指数分布+总时长归一化；长度从真实语料成对取而非解析生成 |

> **命名澄清**：原始 `llmperf` 是 **ray-project/llmperf**（已归档【已查证，arXiv 2601.02522 引用该地址】）；NVIDIA 现役客户端工具叫 **AIPerf**（`pip install aiperf`，打任意 OpenAI 兼容端点）。**未查证到独立名为 "openai-perf" 的成熟包**——不要在报告里写它是一个独立工具。

### 4.2 NVIDIA AIPerf（口径，已从官方文档逐字核对）

- **核心口径**：**TTFT** = 请求发出→收到第一个**有内容**的 token（丢弃首包为空的响应），含排队+prefill+网络；**ITL（=TPOT）** = `(e2e − TTFT) / (output_tokens − 1)`，显式排除 TTFT、分母减 1（文档原话 "AIPerf excludes TTFT"）；**TPS（系统）** = total_output_tokens /（末响应结束 − 首请求发出），是批处理区间口径、配 warmup 就剔除；**RPS** = 完成请求数 / 同上区间。
- **负载侧**：支持 ShareGPT（自动下载并预 tokenize 全部 73,499 段对话）；到达形态 constant / Poisson / gamma，burstiness 可调【已查证 NVIDIA 博客】。
- **对 minillm 的建议**：直接借三个公式——`bench/metrics.py` 的 TPOT 口径应对齐 `(e2e−ttft)/(out−1)`，确认 out_len≤1 时记 0/跳过（vLLM 同）。
- **坑**：TPS 窗口「首请求→末响应」自动丢掉末尾 drain 时间；open-loop 过载时队列不空，窗口口径失真——W3 报告吞吐时**同时给「墙钟全程吞吐」和「活跃窗口吞吐」两个数**；AIPerf 是客户端视角含网络/反序列化，minillm 进程内测量不要把这部分算进 GPU 指标。
- **来源**：https://docs.nvidia.com/nim/benchmarking/llm/metrics.md ；https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf/ ；https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-share-gpt-dataset.md

### 4.3 vLLM `bench serve` 方法论（源码级核对）

- **到达过程 `get_request()`**：`request_rate=inf` → 所有请求 t=0 瞬间发出（burst，近似 closed-loop 满载）；`request_rate=r` → 相邻间隔采样 `gamma(shape=burstiness, scale=1/(r·burstiness))`；**burstiness=1 时退化为指数分布 = Poisson 过程**（<1 更突发、>1 更平滑）。**关键稳定性技巧：把累积 delay 归一化使总时长严格等于 num_prompts/r**——源码注释直说直接累加随机间隔有 1–2% 偏差，归一化是为了让不同随机种子的吞吐数据稳定。可选 ramp_up 从 start_rps 爬升到 end_rps。
- **ShareGPT 采样 `ConversationDataset.sample()`**【已查证源码】：过滤 `len(conversations)>=2`；取 `conv[0].value` 作 prompt、`conv[1].value` 作 completion（第一轮用户问 + 第一轮助手答）；分别 tokenize 得 prompt_len/completion_len；`--output-len` 不传时动态输出 = 真实 completion_len；长度门控不合规就跳过，样本不够时**有放回重采样**补够。**具体门控阈值随版本漂移【一方称】，引用前 pin release**。
- **指标 `calculate_metrics()`**【已查证源码】：`tpot = (latency − ttft)/(output_len − 1)`（out_len≤1 记 0）；`request_throughput = completed/dur_s`；`output_throughput = Σout/dur_s`；`total_token_throughput = (Σin+Σout)/dur_s`；**goodput**：给定 `{ttft:ms, tpot:ms, e2el:ms}` SLO，逐请求判定「同时满足所有 SLO」再算 `request_goodput = good/dur_s`；**max_concurrent_requests**：把请求按「首 token 时刻 + 各 ITL」展开成 token 时间线按秒分桶求重叠并发数峰值。
- **对 minillm 的建议**：W3 接 `loadgen.poisson_arrivals` 直接照抄 gamma 采样 + **归一化总时长**（小模型吞吐本来就抖，不归一化不同 seed 直接污染 CV）；W3「高 λ 吞吐 ≥1.3×」的 λ 要扫到**未饱和区**（吞吐随 λ 上升）和**饱和区**（吞吐压平）两段；p99 TTFT 分解靠进程内三时间戳：`arrival_ts → scheduled_ts → first_token_ts`，则 queue_wait = scheduled−arrival、prefill = first_token−scheduled。
- **坑**：ShareGPT 是英文长对话分布，与冻结中文三桶**不可直接比绝对值，只能借方法不能借数字**；vLLM 默认动态输出长度方差很大，minillm 的 greedy+max_new=128 固定输出更干净，**不要为了「像 vLLM」改回去**。
- **来源**：https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/ ；https://docs.vllm.ai/en/v0.10.1/api/vllm/benchmarks/datasets.html ；https://docs.vllm.ai/en/v0.20.2/cli/bench/serve/

### 4.4 SGLang `bench_serving`

- **核心要点**【已查证官方指南】：`--request-rate inf` = burst，非 inf = Poisson 到达；`--max-concurrency` 硬性截断在飞请求数（与到达率独立）——**这是 W4「同显存并发上限」的直接对应参数**；`--warmup-requests N` 预热 + `--flush-cache` 预热后清 prefix cache（vLLM 要 `VLLM_SERVER_DEV_MODE=1`，否则直接报错而不是偷偷测热缓存）；报告 e2el/TTFT 给 mean/median/std/p99、ITL 给 p95/p99/max；并发 = Σ各请求在飞时长/墙钟。
- **goodput 范式最值得抄**：`--goodput ttft:500 tpot:50 e2el:20000`（毫秒），一个请求 good = 成功**且**同时满足所有 SLO；额外打印逐 SLO 达标率指出**哪个 SLO 是瓶颈**。方法论原话：「扫 request-rate，吞吐继续升或压平，goodput 先升后降——goodput 的峰就是 SLO 内可持续吞吐」。
- **对 minillm 的建议**：W8 报告加一条 goodput 曲线——`metrics.py` 已逐请求记 ttft/tpot/e2e ms，加 per-bucket SLO 阈值就能算 good requests/s，几乎零成本。
- **坑**：SGLang 把 ITL 报到 p99/max、vLLM 默认百分位可配；minillm 现有 p95 口径要在 W8 显式声明「报 p95 不报 p99，因为 n 不够」。
- **来源**：https://docs.sglang.io/docs/developer_guide/bench_serving

### 4.5 指标口径共识（IETF Internet-Draft）

- **文件**：draft-gaikwad-llm-benchmarking-terminology-01（2026-08 更新，2027-02 到期）【已查证】。
- **核心定义（逐字核对）**：
  - **open-loop vs closed-loop**：open-loop 按固定到达率发请求、与系统响应无关，**会暴露过载/队列增长**；closed-loop 固定并发数、一个完成才发下一个，**构造上不可能超过系统容量**。二者 MUST 显式声明。
  - **TTFT**：含网络、鉴权、准入、**排队等待**、prefill；MUST 声明测的是「任意 token」还是「第一个非空 token」。
  - **ITL vs TBT**：ITL 是服务端 token 间隔（decode 效率）；TBT 是客户端观察间隔（含网络抖动），二者不同。聚合 MUST 声明 token 加权 vs 请求加权。
  - **TPOT** = `(e2e − TTFT)/(out−1)`，请求加权，单 token 输出无定义。
  - **百分位样本量硬要求**：p50 ≥ 100 样本；p99 ≥ 1000；p99.9 ≥ 10000。**MUST 同时报 n**。
  - **预热**：MUST 从测量中剔除预热或单列，写明显式流程和时长。
  - **Goodput** = 吞吐 × SLO 达标率。
  - **工作负载 MUST 声明**：输入长度分布（均值/百分位/min/max）、输出长度分布或生成参数、到达模式、数据集来源、tokenizer。
- **对 minillm 的建议（直接击中现状）**：现有 warmup=3/repeats=5 → 每 prompt 只有 5 个样本，**只够 median 和 CV，根本不够报 p99**（p95 也偏险，经验需 ~400 样本）。**分两层解决，而不是把 repeats 加到几百**：
  1. **正确性/趋势层**（现有协议保留）：warmup=3/repeats=5/median+p95/CV≤10% 按长度桶，用于 W0–W2 式「自实现 vs HF」配对。
  2. **开放负载层**（W3 新增）：open-loop Poisson run 把请求数提到 **n≥1000**（靠 32 条冻结 prompt 循环复用 + 随机到达时刻），p99 TTFT 才合法；p99.9 不报。
- **来源**：https://www.ietf.org/archive/id/draft-gaikwad-llm-benchmarking-terminology-01.txt

### 4.6 动态请求流建模（request-level vs token-level、Poisson/gamma、长度分布）

- **共识**：到达间隔用 **Poisson（指数间隔）** 是无记忆到达的默认假设；vLLM 的 gamma(burstiness) 是其推广（burstiness=1 退化为 Poisson），选 gamma 是为可调突发度【已查证 vLLM 源码】。**长度分布采样的业界做法：不是从解析分布随机数生成长度，而是从真实会话语料统计每轮 (prompt_len, completion_len) 再成对采样**——保持「长问倾向长答」的真实相关结构【已查证 ShareGPT sample】。request-level 到达是主流；token-level 到达是 trace replay 类（Mooncake/Baseten），minillm 阶段不需要。
- **对 minillm 的建议（不破坏 prompts.json 冻结约定）**：W3 open-loop 负载**复用 32 条冻结文本**按到达顺序循环分配（seed 固定）；需要变长时 tokenize 后把文本**重复拼接**到目标桶长度（不要截断，截断改语义），记录真实 tokenized 长度——loadgen 注释已写「跨模型比较用实测长度而非桶标签」，直接沿用。到达间隔实现：`intervals = np.random.exponential(scale=1/lambda_, size=n)`，`cum *= (n/lambda_)/cum[-1]` 归一化保证总时长=理论值。

### 4.7 对 minillm 的可执行方案汇总

**W3（连续批 + Poisson 到达）**：
1. 实现 `poisson_arrivals(lam, n, seed, prompt_set)`：指数间隔 + 累积归一化（§4.3a、§4.6）；配置校验现在拒绝 `arrival="poisson"`，W3 放开。
2. open-loop run 请求数 **n≥1000**（循环复用 32 条 prompt，seed 写进 provenance），否则 p99 非法。
3. 每请求记三时间戳 arrival/scheduled/first_token：queue_wait = scheduled−arrival、prefill = first_token−scheduled、TTFT = first_token−arrival，与现有「TTFT 含排队」口径闭合。
4. λ 扫描 ~6 个点，画「吞吐 vs λ」和「p99 TTFT vs λ」，吞吐压平处即饱和点；「≥1.3×」在未饱和段读。
5. 预热：open-loop run 开头加一段 closed-loop warmup（复用 warmup=3），聚合前剔除。

**W4（同显存 max concurrency ≥1.5×）**：
1. closed-loop **并发阶梯**：ctx=2048、权重固定（峰值显存恒定），in-flight 并发取 1/2/4/8/16/32，每档 repeats=5 取 median 输出 token 吞吐。
2. closed-loop 构造上不会过载，正合 W4 意图：测「固定显存下能同时喂多少请求」。
3. paged vs 连续两条臂在**同一并发档**配对比较，报告 median 比值；碎片率 <5% 用现有 `cache_usage.py`。
4. **不要用 open-loop λ 扫回答 W4**——那是 W3 的事；W4 输入是并发数、输出是吞吐。

**W8（判据矩阵 + 六节报告）**：
1. 正确性层（n=5，median/p95/CV）与开放层（n≥1000，p99）**分两表**，与「HF/自实现绝对值不混表」同纪律。
2. 加 goodput 列：每桶给 SLO 阈值（ttft/tpot 预算），报 good req/s 和达标率、指出瓶颈 SLO（§4.4）。
3. 负结果专章沿用 W2「长 prefill 慢 1.49×」先例。

**小模型（0.6B）噪声控制**：单步 decode 可能只有几 ms，Python/scheduler 开销同量级——**scheduler 开销本身是被测对象，不是噪声**。逐请求记录 `scheduler_step_ms`（CPU perf_counter）与 GPU kernel 时间（cuda.Event）分列报，不合并；CUDA Event 只包 GPU forward，queue_wait/scheduler 用 CPU 时钟，口径标注清楚。repeats=5 对 median/CV 仍够（W2 实测桶级 CV 6.8%/3.6%），不要为 p99 加 repeats；尾延迟交给 n≥1000 open-loop run。

---

## 5. 结论与优先级建议

按阶段给出「最值得优先用」的资源（各 1–2 个 + 一句话）：

| 阶段 | 首选资源 | 理由 |
|---|---|---|
| **W3** | ① vLLM bench serve 到达过程算法（gamma→Poisson + 总时长归一化）＋ ② BurstGPT 时间戳 replay | 前者是 0 成本接入 `loadgen.poisson_arrivals` 的源码级范本；后者提供真实到达间隔作对照；长度分布用 ShareGPT 抽 200–500 条或复用冻结 32 条拼接 |
| **W4** | ① nano-vllm `block_manager.py` ② LightLLM `mem_manager.py`（对照） | 最简 block pool + block table 骨架；纯 Python token/块级管理更贴合 D3 无 varlen kernel 约束；压力数据用 LongBench-E 三档 + RULER 按长度生成 |
| **W5** | HF `modular_qwen3_next.py`（全注意力层装配 + 输出门控 + partial-RoPE） | 官方纯 PyTorch 源码，W5 层改造逐项对拍的唯一权威参照 |
| **W6** | ① fla `naive.py`（recurrent+chunk 双版本）② HF `torch_recurrent/chunk_gated_delta_rule` fallback | 两端纯 PyTorch、互相等价可交叉对拍；自洽测试按 §3.7 的五步走，先固定 chunk_size=64 |
| **W7** | LightLLM `state_cache_manager.py` + `qwen3next_mem_manager.py` | 唯一已实现「全注意力 KV 块池 + 线性常数状态池」统一管理的开源项目，模型族最接近 |
| **W8** | IETF 草案口径（n 分层、goodput、warmup 剔除）+ SGLang goodput 报告范式 | 报告六节的口径仲裁与「SLO 内可持续吞吐」呈现方式，两处都零成本落地 |

**总体判断**：minillm 的规模（0.6B/0.8B 单卡、纯 PyTorch、不引入 varlen kernel / CUDA graph / 多卡）意味着四类资源中**最该深读的是「教学级实现」（nano-vllm、mini-sglang、fla naive、HF modular 源码）和「方法论文档」（vLLM/SGLang/AIPerf 的 benchmark 口径、IETF 草案）**；vLLM/SGLang/TensorRT-LLM 等生产系统只取其机制与口径，不取其工程。数据集层面，**RULER（W4 长度扫描）+ ShareGPT（W3 变长负载）+ BurstGPT（W3 到达过程）** 三件套已覆盖项目全部测量需求，LongBench 系列作中文长文本补充，其余无需引入。

---

## 6. 未查证 / 待确认项清单（引用前需复核）

| # | 事项 | 说明 |
|---|---|---|
| 1 | Qwen3.5-0.8B 在 HF 的 `model_type` 是否复用 `qwen3_next` 目录 | 本次抓取确认 transformers 主仓有 `qwen3_next/`（对应 Qwen3-Next-80B MoE）；0.8B dense 小模型可能复用同一套 modeling。**实际加载时 `AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")` 看 model_type 再定位** |
| 2 | `modeling_rope_utils.py` 中 `mrope_interleaved` 官方实现函数体 | 未逐行抓取；接入时直接读 `src/transformers/modeling_rope_utils.py` 的 `_compute_rope` 分支，不要凭第三方片段重写 |
| 3 | mfai 的 Qwen3.5-0.8B 配置 | 第三方重实现【一方称】，与任务参数吻合但以 HF config.json 为准 |
| 4 | vLLM ShareGPT 长度截断门控具体阈值（2048/32000 等） | 随版本漂移【一方称】，引用前 pin 到对应 release |
| 5 | SGLang 生产版 `radix_cache.py` 文件路径 | 目录路径为【一方称】（deepwiki/目录惯例），未逐文件 fetch；建议直接读 mini-sglang |
| 6 | "openai-perf" 独立包 | **未查证到**；现役工具是 NVIDIA AIPerf，不要引用不存在的包名 |
| 7 | PagedAttention arXiv 编号 2309.06180 | 会议论文（SOSP'23）已查证于 vLLM README 引用；arXiv 编号本身未单独 fetch 验证 |
| 8 | LongBench v1 镜像站 Apache 许可 | 以 GitHub MIT 为准，镜像标 Apache 为【一方称】 |

---

## 7. 来源清单（分类汇总）

**数据集**：
- ShareGPT_Vicuna_unfiltered：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered ；https://docs.vllm.ai/en/stable/contributing/benchmarks.html ；https://blog.vllm.ai/2024/09/05/perf-update.html
- LMSYS-Chat-1M：https://huggingface.co/datasets/lmsys/lmsys-chat-1m ；https://arxiv.org/abs/2309.11998
- BurstGPT：https://github.com/HPMLL/BurstGPT ；https://docs.vllm.ai/en/v0.10.2/api/vllm/benchmarks/datasets.html ；https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-burst-gpt-traces.md
- LongBench v1：https://huggingface.co/datasets/THUDM/LongBench ；https://github.com/THUDM/LongBench
- LongBench v2：https://huggingface.co/datasets/zai-org/LongBench-v2 ；https://arxiv.org/abs/2412.15204
- ∞Bench：https://github.com/OpenBMB/InfiniteBench ；https://arxiv.org/abs/2402.13718
- RULER：https://github.com/NVIDIA/RULER

**参考项目**：
- vLLM：https://github.com/vllm-project/vllm
- SGLang：https://github.com/sgl-project/sglang ；mini-sglang：https://github.com/sgl-project/mini-sglang
- nano-vllm：https://github.com/GeeeekExplorer/nano-vllm
- LightLLM：https://github.com/ModelTC/LightLLM
- vattention：https://github.com/microsoft/vattention ；https://arxiv.org/abs/2405.04437
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM ；KV cache 文档 https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html
- PagedAttention 论文：https://arxiv.org/abs/2309.06180

**算子实现**：
- Gated DeltaNet 论文：https://arxiv.org/abs/2412.06464
- fla：https://github.com/fla-org/flash-linear-attention ；naive.py https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/naive.py ；PyPI https://pypi.org/project/flash-linear-attention/0.5.1/
- HF modular_qwen3_next.py：https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
- Qwen3-Next 博客：https://qwen.ai/blog?id=qwen3-next ；Qwen3.5 博客：https://qwen.ai/blog?id=qwen3.5
- mfai（一方称）：https://meteofrance.github.io/mfai/_modules/mfai/pytorch/models/llms/qwen3_5.html

**评测方法**：
- NVIDIA AIPerf：https://docs.nvidia.com/nim/benchmarking/llm/metrics.md ；https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf/ ；https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-share-gpt-dataset.md
- vLLM bench serve：https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/ ；https://docs.vllm.ai/en/v0.10.1/api/vllm/benchmarks/datasets.html ；https://docs.vllm.ai/en/v0.20.2/cli/bench/serve/
- SGLang bench_serving：https://docs.sglang.io/docs/developer_guide/bench_serving
- IETF 草案：https://www.ietf.org/archive/id/draft-gaikwad-llm-benchmarking-terminology-01.txt
