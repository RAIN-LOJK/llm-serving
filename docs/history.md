# 历史：从训练学习项目到 serving 引擎

> 旧仓库的完整文本快照在 git tag **`legacy-learning-2026-08`**。
> 工作树里的训练代码/数据/结果已删除；本文件只留摘要，避免新项目被旧叙事带偏。

---

## 1. 旧项目做了什么

一个手写 LLaMA 风格 decoder 的学习项目，主线是「手写全链路 → 对齐 HF → 结构实验」：

- **A 生成引擎**：采样（greedy/temperature/top-k/top-p）、KV cache、批量左 pad、EOS 发散。
- **B 训练管线**：迭代式训练、val 划分、断点续训、BPE tokenizer、鲁迅/fineweb_cn 语料。
- **C 架构对齐**：TinyLlama 前 2 层对齐（max diff ~1e-5）；GQA 锚点校验。
- **D/E 计划**：DeepSeek 结构（MLA/CSA/MoE/GRPO）→ 光流 LLM；未完成。

实验 exp001–exp005：规模扫、GQA 校验、fineweb 语料升级。产出过 `result.md` 台账、
`training_curve.png`、`bench_kvcache.png` 和若干生成样本。

## 2. 仍然有效的结论

| 结论 | 对新项目的意义 |
|---|---|
| Toy 规模上 KV cache 加速比 ~0.9×（kernel launch/Python 开销主导） | 机制收益要看规模/上下文；W2/W3 要把交叉点测出来 |
| Toy 上结构性对比（ALiBi/窗口/pre-post-norm）无信息增益 | “结论已知 + 测不到收益维度”的实验不做；W 阶段只做可测的机制 |
| 120 万 → 234 万 token 只推进一档词法，句级需 20–50M 参数 + 1–3 亿 token | 不做训练；载体改用真实开源权重 |
| 左 pad / EOS / KV cache 的一致性校验方法有效 | 直接迁移为 pytest 护栏 |

## 3. 被保留的代码

- `model/layers.py`：RoPE（GPT-NeoX）、GQA、KVcache、SwiGLU、RMSNorm、TransformerBlock。
- `model/config.py`、`model/minillm.py`：玩具模型装配；单元测试的 tiny model 底座。
- `llm_utils/inference.py`：采样 + 批量 generate；S0–S2 的参考实现，之后由 `serve/engine.py` 接管。
- `PITFALLS.md`：工程坑清单；已加 ID 并映射到 pytest。
- 三个回归的**语义**：KV cache 一致性、左 pad、EOS 发散；已重写为自包含 pytest。

## 4. 被删除的内容

- `train.py` / `eval.py` / `demo.py` / `config/` / `data/` / `tools/`
- `results/exp001–005`（含 95MB 的 `.pth/.pt`）
- `datasets/`（28MB，含内嵌 `.git`）
- 旧 `tests/check_*.py`、`tests/bench_kvcache.py`
- 旧文档 `CLAUDE-old.md`、`todo.md`、`thought.md`、`result.md`、`archive/`

> 说明：Git 只能恢复**曾被跟踪**的文本；`datasets/` 和权重本来就在 `.gitignore` 里，
> 删除后不可从 git 恢复。旧语料来源见下方的“旧数据来源（仅备查）”。

## 5. 为什么转向 serving

- 实习简历要的是**可验证的工程能力**：调度、缓存、显存、测量、数值对拍，而不是再跑一个 toy 训练。
- ROADMAP 的 W0–W8 有清晰的判据与失败线；训练线的“句级可读”在 M1 + 小语料上不可达。
- 旧项目的算子/回归/坑清单正好是 serving 引擎的地基；训练管线不是。

## 6. 恢复旧快照

```bash
# 看旧树
git ls-tree -r --name-only legacy-learning-2026-08 | head
# 取单个文件
git show legacy-learning-2026-08:model/layers.py
# 在临时 worktree 里完整查看（不影响当前工作树）
git worktree add /tmp/minillm-legacy legacy-learning-2026-08
```

旧代码不作为可运行入口：路径、依赖、数据都已变化。只在需要考古/写历史时使用。

## 7. 旧数据来源（仅备查）

新项目不训练，以下仅用于需要复现旧实验时重新获取。`datasets/` 当年被 `.gitignore`，
不在 tag 里，无法从 Git 恢复。

- 鲁迅（md）：<https://github.com/Ac-heron/luxun.git>（旧项目的 `datasets/source.md` 曾记录）
- 鲁迅（json）：<https://github.com/sun510001/luxun_dataset.git>
- Fineweb-Edu-Chinese 5% 采样：
  `wangkunqing/mini_llm_dataset` → `fineweb_edu_sampled_5_percent.zip`
  - source_url: <https://modelscope.cn/datasets/wangkunqing/mini_llm_dataset/resolve/master/fineweb_edu_sampled_5_percent.zip>
  - sha256: `bd9bc3d46eb0908a93b5cb107c0bfc1f0a28981b1089ab90e399e36eaf487bef`
  - upstream: `opencsg/Fineweb-Edu-Chinese-V2.1`（score 4–5，5% 等比例采样）
  - seed: `minillm-b3-fineweb-edu-cn-v2`；清洗/筛选规则见旧 `data/slice_fineweb_cn.py`
    （该脚本在 tag `legacy-learning-2026-08` 里）

