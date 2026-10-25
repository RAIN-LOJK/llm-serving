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

- **日期**：2026-10-25
- **状态**：accepted
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
- **复盘点**：W3 结束时用实测占比校验 < 15%；W4 结束后看 gather 是否引入新热点；
  torch 升级导致 SDPA 后端支持表变化时重新确认。

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
