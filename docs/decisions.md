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
