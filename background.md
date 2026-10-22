# Background

> 用途：让后续 agent / 合作者快速了解项目主人的身份、目的、目标和已有知识，避免从零讲基础或重复猜测。
>
> 更新时间：2026-10-21
>
> 依据：当前仓库、git tag `legacy-learning-2026-08`、旧 `todo.md` / `result.md` / `CLAUDE-old.md`，以及近期对话。

---

## 1. 身份

- Git 提交者：`shirone`。
- 正在准备 **LLM 推理 / serving 工程方向实习** 的学生或求职者；具体学校、年级、专业未在仓库中记录。
- 有完整的手写 LLaMA 风格 decoder 与训练实验经历，不是零基础转码。
- 当前主要工作环境：Apple M1 / 16GB / MPS，无本地 CUDA；正式性能数字计划在租用 4090 上产出。

---

## 2. 当前目的

按优先级：

1. **求职目的**：做一个能讲清楚、有真实测量数据的 LLM serving 项目，用于推理 / Infra / 平台方向实习简历与面试。
2. **学习目的**：真正理解连续批处理、分页 KV、混合注意力状态管理，而不是只会调用 HF `generate`。
3. **工程目的**：在单卡上实现一个可运行的 serving 引擎，支持请求级调度与可控的 KV/state 生命周期。
4. **实验目的**：回答“这些机制在什么规模、什么上下文、什么并发下才回本”，并保留负结果。
5. **长期兴趣**（旧项目记录，非当前主线）：LLM 与光流 / PIV 的结合，用自回归 next-state 范式做流场/帧预测。

当前项目明确 **不训练模型、不做 SFT、不提交权重**。训练主线已归档到 git tag `legacy-learning-2026-08`。

---

## 3. 目标

### 3.1 项目目标

把 minillm 从“手写玩具解码器”推进到“能同时管理 KV cache 和线性注意力状态的单卡 serving 引擎”。

双载体：

| 角色 | 模型 | 用途 |
|---|---|---|
| 骨架载体 | `Qwen/Qwen3-0.6B` | 纯全注意力，先跑通 engine / scheduler / paged KV |
| 目标载体 | `Qwen/Qwen3.5-0.8B` | 3:1 混合注意力，最终对照对象 |
| 回退 | `TinyLlama-1.1B-Chat` | A/B 卡住时的已知正确参照 |

### 3.2 阶段目标

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| W0 | 用 HF 测双载体显存/延迟 vs 上下文，判定 P1–P3，做 D1 决策 | 下一步 |
| W1 | Qwen3-0.6B 权重加载与数值对齐，生成 golden | 未开始 |
| W2 | 基准框架：固定 prompt、TTFT/TPOT/显存、HF 对照 | 未开始 |
| W3 | 连续批处理：请求队列、每步准入/退出 | 未开始 |
| W4 | 分页 KV：block table、按需分配/回收 | 未开始 |
| W5 | Qwen3.5 全注意力层改造 | 视 D1 决定 |
| W6 | Gated DeltaNet：短卷积、decode 循环、分块 prefill | 视 D1 决定 |
| W7 | 双状态管理：分页 KV + 线性状态池 | 视 D1 决定 |
| W8 | 报告与判据矩阵 | 未开始 |

### 3.3 交付策略

- **必做**：W0–W4。完成后已经是一个完整的纯注意力 serving 引擎 + 测量报告。
- **加分**：W5–W7。只有 W0 的 D1 判定“值得投入混合注意力”才做。
- **不承诺**：W8 之前不声称“比 vLLM 快”；只报告实现并测出的收益边界。
- **简历原则**：没有真实 run 之前，不写“已实现连续批 / 分页 KV”。

---

## 4. 知识储备

### 4.1 已经掌握 / 有实践

来自旧项目 `legacy-learning-2026-08` 与当前仓库：

| 领域 | 具体内容 | 证据 |
|---|---|---|
| PyTorch 模型实现 | 手写 RoPE（GPT-NeoX）、GQA、KVcache、RMSNorm、SwiGLU、TransformerBlock | 旧 `model/layers.py`、当前 `model/` |
| 自回归生成 | prefill + 逐 token decode、采样 greedy / temperature / top-k / top-p | 旧 A1–A2、`llm_utils/inference.py` |
| KV cache | 一致性校验：cache 生成 vs 全量重算逐 token 对比，24/24 一致 | 旧 A3、当前 `tests/unit/test_kvcache.py` |
| 批量生成 | 左 pad 变长 batch、attention mask 全长度传递 | 旧 A3.1、`test_generate_leftpad.py` |
| EOS 处理 | batch 内各行提前结束、buffer / outputs / orig 账本对齐 | 旧 A3.2、`test_generate_eos.py` |
| 数值对齐 | TinyLlama 前两层对齐，max diff ~1e-5，cosine=1.0 | 旧 C1 |
| 结构校验 | GQA 锚点：`n_kv_head=2` 与 MHA 训练轨迹重合 | 旧 B3 |
| 训练管线 | BPE tokenizer、语料清洗、train/val、断点续训、val ppl、过拟合判断 | 旧 B1–B2 |
| 实验方法 | 配置快照、实验索引、固定 prompt、负结果、规模结论 | 旧 `result.md`、`todo.md` |
| 基准测试 | KV cache vs 全量重算、MPS 同步、warmup/repeats、加速比曲线 | 旧 A3.3、`bench_kvcache.py` |
| 工程踩坑 | RoPE 约定、RMSNorm eps、KV cache 层间独立、左 pad mask、EOS 发散、RoPE 越界 | `PITFALLS.md` |
| 当前 serving 骨架 | run 契约、provenance、HF baseline、固定 prompt、pytest 门禁 | 当前 `bench/`、`tests/` |

一个重要的已有认知：

> toy 规模下 KV cache 加速比只有约 0.9×，瓶颈是 kernel launch / Python 开销；机制收益必须在大模型、长上下文、真实并发下才显现。

这正是当前 serving 项目的动机。

### 4.2 正在补 / 尚未在项目中验证

这些不是“完全不会”，而是“还没有在当前项目中实现或验证”：

| 缺口 | 对应阶段 |
|---|---|
| 连续批处理的请求状态机、准入/退出、抢占 | W3 |
| 分页 KV：block table、allocator、碎片、最大并发 | W4 |
| serving 指标：queue_ms、吞吐、SLO、p99、CUDA event 计时 | W2–W3 |
| HF `past_key_values` / `DynamicCache` 的实际内存结构 | W0–W1 |
| Qwen3 权重加载：QK-Norm、显式 head_dim、rope_theta、tie、attention_bias | W1 |
| Qwen3.5 复合 checkpoint 的 text-only 加载路径 | W0 |
| Gated DeltaNet、短卷积、输出门控、partial/M-RoPE | W5–W6 |
| 线性状态池与 KV 块的统一生命周期 | W7 |
| CUDA profiling / Nsight / kernel 级优化 | 当前未涉及 |
| 最终报告、判据矩阵、复现文档 | W8 |

### 4.3 工具与环境

- Python 3.12.15
- torch 2.13.0
- transformers 5.14.1（已内置 `qwen3_5` 官方实现）
- pytest、tokenizers、safetensors、numpy、matplotlib
- 本机：macOS / Apple M1 / 16GB / MPS
- 正式数字：租用 4090（`measure-cuda`）
- 仓库内统一使用 conda base 的 `python`（不建 `.venv`）
- 当前没有 lint / formatter / typecheck / CI；pytest 是唯一门禁

---

## 5. 学习与协作偏好

从旧 `CLAUDE-old.md`、`AGENTS.md` 和近期对话整理：

- 使用中文交流。
- 希望 **先讲原理，再落地代码/实验**，不要直接贴完整实现。
- 喜欢逐步推进，不要一次批量改很多文件而不同步。
- 核心组件希望自己写；agent 应给骨架、提示、TODO、review 和 debug，而不是代写全部。
- 关键实验命令优先由用户自己执行；agent 给命令和结果解读。
- 重视可复现和负结果；不接受“只看一个数字就下结论”。
- 会主动追问“为什么”和挑战设计，喜欢诚实的边界说明。
- 当前不需要从 Transformer / KV cache 基础重新讲；需要的是 serving 层的 just-in-time 学习。
- 学习节奏建议：每个阶段开始前补该阶段最小概念，然后立即读代码 / 做实验，不要先系统学完。

---

## 6. 当前进度

- 当前仓库已完成从训练项目到 serving 项目的重组。
- `bench/` 的 runner、run 契约、dummy/HF engine、指标聚合、prompt 集、索引已就位。
- `model/` 的玩具 MiniLLM 与 `tests/unit/` 回归已就位，快速门禁 26 passed。
- `serve/` 只有模块边界，尚无 `engine.py` / `scheduler.py` / `block_manager.py` / `state_pool.py`。
- `results/` 还没有正式 run；`W0-0001` 未执行。
- `tests/golden/` 为空，W1 未开始。
- 已知待验证：Qwen3.5 文本加载路径、HF cache 显存分项、32k 在 M1 上可能 OOM。

### 下一步

1. 验证 Qwen3.5 text-only 加载路径。
2. 写 `scripts/probe_checkpoint.py`，填 `docs/carrier.md` 的解析式估算。
3. 在 M1 上用 `--limit 1` 做 W0 冒烟。
4. 扩展 `HFEngine` 记录 KV / linear state bytes。
5. 在 4090 上跑完整 W0，产出正式 run。
6. 填写 `notes.md` / `carrier.md` / D1，再决定 W5–W7 是否投入。

---

## 7. 给后续 agent 的注意事项

- 不要从零讲 Transformer、RoPE、KV cache 基础；这些已经做过。
- 不要把“计划”说成“已实现”；`serve/` 目前是空骨架。
- W0 用官方 HF 代码是预期行为，不是“偷懒”；W1 之后才需要自研引擎。
- 官方 `qwen3_5` 代码是参考实现和 baseline，不是要替代的“错误实现”。
- 不要伪造 W0/W1 数据；没有 run 就写“未测”。
- 不要手改 `results/INDEX.md`、`index.csv` 或已完成的 run。
- 不要在没有 D1 数据前承诺 W5–W7。
- 涉及关键实验命令时，先给用户命令和解释，由用户执行。
- 如果更新了用户背景或目标，同步更新本文件。
