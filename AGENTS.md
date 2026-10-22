## 项目定位

- 单卡 serving 引擎实验，**不训练模型、不提交权重**。真实载体：`Qwen/Qwen3-0.6B`（纯注意力骨架）与 `Qwen/Qwen3.5-0.8B`（3:1 混合注意力目标）。
- `ROADMAP.md` 是 W0–W8 阶段、判据与测量协议的唯一权威；与 `docs/` 冲突时以 ROADMAP 为准并修文档。先读 `ROADMAP.md` 再动阶段目标。
- 当前是可执行骨架：runner、run 契约、玩具模型和三组回归已就位；`serve/` 只有模块边界，W0 探测是下一步。
- 训练主线已删除。git tag `legacy-learning-2026-08` 只用于考古，旧代码不是可运行入口。

## 命令（conda base 的 `python`）

```bash
python -m pip install -r requirements-dev.txt               # base 已有 torch；补 transformers/tokenizers/pytest
python -m pytest -m "not slow and not hf"                   # 快速门禁，无 HF 权重/网络
python -m pytest tests/unit/test_kvcache.py -q              # 单文件
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_hf_engine.py -q
python scripts/env_snapshot.py
python scripts/exp_index.py --check                         # 校验 run 契约
python bench/make_prompts.py --check                        # 校验固定 prompt 集
python -m bench.run --config configs/experiments/W0/9001-runner-smoke.toml \
  --out /tmp/minillm-smoke --allow-dirty                    # smoke 不写 results/
```

- 没有配置 lint / formatter / typecheck / CI；pytest 是唯一门禁，不要声称其他检查已配置。
- HF 集成测试即使显式指定文件也会被 skip，必须同时设 `MINILLM_RUN_HF=1`。

## 结构

- `bench/run.py`：`python -m bench.run` 的入口，负责配置校验、provenance、指标聚合、run 目录与索引。`bench/engines/build_engine` 目前只实现 `dummy` 和 `hf`；`minillm/continuous/paged/hybrid` 会按阶段抛 `NotImplementedError`，不要静默回退。
- `serve/`：只固定 `engine/scheduler/block_manager/state_pool` 边界。调度/缓存逻辑写这里，不要塞回 `llm_utils/`。
- `model/`：手写玩具 MiniLLM（GPT-NeoX RoPE、GQA、KVcache）；`llm_utils/inference.py` 是过渡版批量 generate，S2 后由 `serve/engine.py` 接管。
- `bench/engines/hf.py` 只处理 text-only causal LM；Qwen3.5 的多模态 wrapper 需要显式抽取 text model。

## 实验与结果约定

- run 目录不可变：`results/<stage>/<4位序号>-<slug>/`，`run_id` 形如 `W0-0001`。目录或 id 已存在时 runner 直接拒绝；失败/中断的 run 也保留，用 `supersedes` 指回旧 run。
- 非 `smoke` 的正式 run 要求 git 干净；dirty 只能用 `--allow-dirty` 做 exploratory，报告不得引用。正式探测前 pin HF `revision` 为 commit hash，`main` 会被配置校验拒绝。
- `arrival = "poisson"`（W3）和 `timer = "cuda_event"`（W2）尚未接线，配置校验会直接报错，不要伪造。
- `results/INDEX.md` 和 `index.csv` 由 `scripts/exp_index.py` 生成，**不要手改**。
- `notes.md` / `verdict.json` 是人的判断。agent 可以起草，但 `reviewer` 保持 `null`、`verdict` 保持 `unreviewed`，直到用户确认。
- `configs/thresholds.toml` 的值是目标值/判失败线，不是实测结果。M1（`dev-mps`）只做正确性/冒烟/趋势；正式数字只来自 `measure-cuda`（4090），两套数字永不同表。
- `bench/prompts.json` 是冻结的 32 条 prompt 集（文本而非 token id）。改动必须 bump version 并用 `bench/make_prompts.py` 重生成；prompt hash 变化会使跨 run 比较失效。
- 关键命令交给用户来跑

## 测试与实现纪律

- `tests/` 是规格：修 bug 或重构前先补/跑回归，快速门禁必须保持绿。`PITFALLS.md` 把已知坑映射到测试，动相关代码前先读。
- 加载 HF 权重时保持既有约定：RoPE 用 GPT-NeoX（P-02），RMSNorm 必须显式接 `norm_eps`（P-03）。
- golden 只由脚本生成并经 review 提交，测试只读；不要静默重生成 `tests/golden/`。
