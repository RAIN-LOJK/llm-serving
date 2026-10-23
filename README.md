# minillm · 单卡 serving 引擎

把 minillm 从「手写 LLaMA 的玩具解码器」推进到「能同时装下 KV cache 和线性注意力状态的单卡推理引擎」。
用同一套调度器测两代注意力架构（Qwen3 全注意力 / Qwen3.5 混合注意力），回答一个工程问题：

> **什么规模、什么上下文下，连续批处理 / 分页 KV / 混合注意力状态管理才回本？**

- 技术主计划：[ROADMAP.md](ROADMAP.md)（W0–W8 的唯一权威来源）
- 实验约定：[docs/experiments.md](docs/experiments.md)
- 决策记录：[docs/decisions.md](docs/decisions.md)
- 环境与设备口径：[docs/environments.md](docs/environments.md)
- 历史与归档：[docs/history.md](docs/history.md)

> 本项目**不训练任何模型**。载体是真实开源权重；模型会不会说话不是自变量。

## 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| W0 | 双载体探测 + D1 决策 | ✅ `W0-0001` |
| W1 | Qwen3-0.6B 对齐 + golden | ⬜ 下一步 |
| W2 | 基准框架 | ⬜ |
| W3 | 连续批处理 | ⬜ |
| W4 | 分页 KV | ⬜ |
| W5–W7 | 混合层 + 双状态管理 + 跨代对照 | ⬜ |
| W8 | 报告与判据矩阵 | ⬜ |

W0 已完成（双载体探测，D1 接受移植）：P1 成立（8k 7.80× / 32k 8.90×），P2 不成立（交叉点 ≈193 token），
P3 数值成立但暂不可归因。下一步 W1：把 `Qwen3-0.6B` 装进自己的解码器并对齐。

## 快速开始

```bash
# 1. 依赖（miniconda base 已带 torch；这里补 transformers / tokenizers / pytest）
python -m pip install -r requirements-dev.txt

# 2. 快速门禁：不需要任何 HF 权重
python -m pytest -m "not slow and not hf"

# 3. 环境快照（设备 / git / 版本）
python scripts/env_snapshot.py

# 4. 跑一个不依赖模型的 runner 冒烟（写到临时目录，不污染 results/）
python -m bench.run \
  --config configs/experiments/W0/9001-runner-smoke.toml \
  --out /tmp/minillm-smoke --allow-dirty

# 5. 生成 / 校验实验索引
python scripts/exp_index.py
python scripts/exp_index.py --check
```

正式测量请在干净 git 树上跑；官方数字以 `measure-cuda`（4090）为准，本机 M1 只做正确性与冒烟。口径见 [docs/environments.md](docs/environments.md)。

## 目录

```text
model/         手写算子：RoPE / GQA / KVcache / SwiGLU / RMSNorm / TransformerBlock
llm_utils/     过渡层：sampling + 批量 generate（S2 后由 serve/engine.py 接管）
serve/         W2–W7：engine / scheduler / block_manager / state_pool（待实现）
bench/         run / metrics / loadgen / prompts.json（runner 骨架已就位）
configs/       实验配置（experiments/）与集中阈值（thresholds.toml）
scripts/       env_snapshot / exp_index；W0/W1 再加 probe_* / make_golden / align_hf
tests/         pytest：unit + integration + golden
results/       run 目录（W*/NNNN-slug/）+ 自动生成的 INDEX.md
docs/          实验约定 / 决策 / 环境 / 载体 / 混合注意力
reports/       最终报告与判据矩阵
archive/       仅保留 legacy tag 的说明（旧训练资料已删除）
```

## 历史

前身是一个手写 LLaMA + 训练学习项目（A–E 阶段、exp001–005、TinyLlama 对齐）。
训练主线已结束，代码/数据/结果已从工作树删除；可恢复的文本快照在 git tag
`legacy-learning-2026-08`。摘要见 [docs/history.md](docs/history.md)。

## 不做的事

- 不训练模型、不扩语料、不做 SFT。
- 不声称「比 vLLM 快」；只报告**实现并测出的收益边界**。
- 不在 M1 上产出的数字后面暗示 4090；设备是结果的一等字段。
