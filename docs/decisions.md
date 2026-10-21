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

- **日期**：2026-10-21（提出）
- **状态**：proposed
- **判据**（来自 ROADMAP §3.3）：
  - P1 成立（长上下文每序列显存 ≥ 3× 优于同量级纯注意力）→ W5–W7 投入移植；
  - P1 < 2× → 砍掉移植，Qwen3.5 只作 HF 侧对照，主线收在 W4；
  - 交叉点约 1.5k token 是待验证预测，不是结论。
- **证据**：`W0-0001`（待跑）；`docs/carrier.md`；`results/W0/0001-carrier-probe/summary.json`。
- **后果（若接受）**：W5 全注意力层改造（head_dim 256 / 输出门控 / partial-RoPE / M-RoPE）；
  W6 Gated DeltaNet（短卷积 → decode 循环 → 分块 prefill）；W7 StatePool + 跨代对照。
- **后果（若拒绝）**：W8 直接收口；报告只覆盖纯注意力引擎 + HF 侧混合架构对照。
- **复盘点**：W0 结束；W7 末复核一次「移植成本是否被收益覆盖」。

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
