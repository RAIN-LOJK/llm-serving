# AGENTS.md · 协作约定

本文件是 agent 在本仓库工作的唯一入口约定。`CLAUDE.md` 只是指向本文件的指针。

## 项目是什么

minillm 是一个**不训练模型**的单卡 serving 引擎项目：手写 PyTorch 解码器 + 请求级调度 +
分页 KV + 混合注意力（Gated DeltaNet）状态管理。技术主计划见 [ROADMAP.md](ROADMAP.md)。

目标不是「跑得比 vLLM 快」，而是给出**收益边界**：什么规模、什么上下文、什么 batch 下，
连续批处理 / 分页 KV / 双状态管理才值得做。

## 三条铁律

1. **测试先行，再动结构。** `tests/` 是规格。每步重构前先让护栏变绿，改完立刻重跑；
   回归红了不许进下一步。`PITFALLS.md` 每条坑必须有对应测试（未覆盖标 `OPEN`）。
2. **判据优先于数字。** 每个机制必须回答「**什么时候不划算**」；只报「快了多少」视为未完成。
3. **所有阈值是目标值与判失败线，不是实测结果。** W1 基线跑完后校准一次，写进
   `configs/thresholds.toml` 与 `ROADMAP.md` §13。

## 用户偏好

- **交互式、逐步推进**：不要一次批量改多个文件而不同步；每步先讲原理再实现。
- **核心算子由用户手写**：attention / 线性注意力 / 引擎关键路径由用户实现；agent 负责
  骨架、测试、测量、文档、review。用户写出问题后 agent 做 review 和 debug。
- **正式实验由用户执行**：agent 给命令、给配置、给数据结论，不代跑长 benchmark。
  agent 可以跑快速 pytest、dry-run、几分钟内的 smoke，并在结果里标注 `operator: agent`。
- **有获得感优先**：不要为了「快」把用户的学习环节全部代劳。

## 实验工作流

1. 写/改 `configs/experiments/<stage>/<id>-<slug>.toml`。
2. 用户执行：
   ```bash
   .venv/bin/python -m bench.run --config <config>
   ```
3. runner 生成 `results/<stage>/<seq>-<slug>/`（manifest / env / metrics / summary /
   notes / verdict / reproduce.sh）。
4. 人看 `summary.json`，在 `notes.md` 写观察与结论，在 `verdict.json` 填
   `pass | fail | inconclusive`；agent 可以起草，但不能替用户定稿。
5. `scripts/exp_index.py` 重新生成 `results/INDEX.md`。
6. 若触发阶段决策，写 `docs/decisions.md` 并回填 `ROADMAP.md` §12。

**禁止**：
- 不覆盖、不复用 run id；失败/作废也保留。
- 不手改 `results/INDEX.md`（它是生成物）。
- 不在 run 完成后改 `manifest.json`。
- 不把 `verdict` 自动填成 `pass`；没有人的判断就保持 `unreviewed`。
- 不把 HF 与自实现的绝对值混在一张表里；比值可以，绝对值分表。
- 不把 M1 数字写成 4090 数字；设备是结果的一等字段。
- 不提交权重、数据集、HF cache；不重新引入训练线。

## 目录与命令

```bash
# 快速门禁
.venv/bin/python -m pytest -m "not slow and not hf"

# 全量（需要 HF 权重时）
.venv/bin/python -m pytest

# 环境快照
.venv/bin/python scripts/env_snapshot.py

# 一个 run
.venv/bin/python -m bench.run --config configs/experiments/W0/0001-carrier-probe.toml

# 索引
.venv/bin/python scripts/exp_index.py --check

# 单个 pytest 文件 / 用例
.venv/bin/python -m pytest tests/unit/test_kvcache.py -q
```

目录职责见 [README.md](README.md)；实验契约见 [docs/experiments.md](docs/experiments.md)；
环境口径见 [docs/environments.md](docs/environments.md)。

## 当前阶段

- W0 尚未开跑；下一步是 `scripts/probe_checkpoint.py` + `scripts/probe_memory.py`。
- 现有 runner 是骨架：`impl = "dummy"` 可跑通契约；`impl = "hf"` 的最小 HF 引擎已存在，
  W0 在其上补 Qwen3.5 的加载分支与显存/延迟分项。
- 测试快速门禁必须始终绿；`slow` / `hf` / `gpu` 标记的测试不在快速门禁里。

## 代码风格

- Python 3.12+，类型标注适度；不要为了类型而类型。
- 新增模块先写 docstring：它回答什么问题、输入输出边界、失败时怎么表现。
- 注释只写「为什么」，不写「这行在做什么」。
- 任何数值结论必须带：设备、dtype、commit、模型 revision、重复次数、聚合方式。
