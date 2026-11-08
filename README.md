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
| W1 | Qwen3-0.6B 对齐 + golden | ✅ `tests/golden/qwen3-0-6b/0001` |
| W2 | 基准框架 | ✅ `W2-0007` |
| W3 | 连续批处理 | ✅ `W3-0010/0011`（C1 2.01×） |
| W4 | 分页 KV | ✅ `W4-0002/0003`（C1 3.111×；C4 负结果，详见 ROADMAP） |
| W4.5 | 注意力路径 v2：packed prefill + block-table decode | ✅ `W45-0001…0012`（C1 1.040× 接受线；E3 p99 3.760×；E4 2.053×） |
| W5–W7 | 混合层 + 双状态管理 + 跨代对照 | ⬜ |
| W8 | 报告与判据矩阵 | ⬜ |

W0–W4.5 已完成。W3 的连续批在 B=8、混合长度、过载 trace 上做到 **2.01×** 吞吐
（`W3-0011` vs `W3-0010`；311.43 / 154.94 tok/s）。W4 的分页 KV 在同 1640 MiB 总预算下
把 allocator capacity 从 9 提到 28（**3.111×**），internal frag **0.55%**、逐 token 全等；
但 B=8 decode step **1.671×**，按预注册规则**不报吞吐**。W4.5 把注意力路径换成 packed
varlen prefill + Triton block-table decode：decode step 从 **1.203× fail** 修到 **1.040×
pass（接受线，1.0 目标未达）**，E3 TTFT p99 **3.760×**，E4 **2.053×**（both saturated，
record-only）。W4.5 收口后已删除 serial prefill 控制臂：生产调度器只有 packed group，
`[engine].prefill` 不再是开关；历史 `W45-0002/0003/0006` 配置保留作历史但不可运行，
旧代码在 git 历史。下一步 W5：混合层。

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
serve/         W2–W4.5：engine / scheduler / block_manager 已落地；state_pool(W7) 待实现
bench/         run / metrics / loadgen（含 poisson_trace）/ prompts.json
configs/       实验配置（experiments/）与集中阈值（thresholds.toml）
scripts/       env_snapshot / exp_index / probe_* / make_golden / align_hf / plot_*
tests/         pytest：unit + integration + golden
results/       run 目录（W*/NNNN-slug/）+ 自动生成的 INDEX.md
figures/       从 summary 生成的报告图
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
