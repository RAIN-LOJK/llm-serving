# 实验约定（run contract）

> 本文件回答：**一次实验怎样算完成、产物长什么样、什么数字可以进报告。**
> 测量口径的权威来源仍是 [ROADMAP.md](../ROADMAP.md) §6（L229–L242）；本文件负责把它落成可执行的目录/配置/评审契约。
> 两者冲突时以 ROADMAP 为准，并修本文件。

---

## 1. 三种记录，不要混

| 类型 | 回答什么 | 产物 | 进 `results/` 吗 |
|---|---|---|---|
| **test** | 代码有没有回归 | `pytest` pass/fail | 否 |
| **run / experiment** | 一个具体工程问题 | 自包含 run 目录 + summary + notes | 是 |
| **decision** | 阶段要不要继续投入 | `docs/decisions.md` 的 ADR | 引用 run id |

- 测试是门禁，跑得频繁，不需要归档成 run。
- run 是证据，必须可复现、可追溯；没有 summary/notes/verdict 的 run 不算完成。
- decision 是人的取舍；一个 decision 可以引用多个 run，一个 run 也可以支撑多个 decision。

---

## 2. run 目录契约

```text
results/W0/0001-carrier-probe/
├── config.toml          # 本次配置快照（原样复制，不引用外部文件）
├── manifest.json        # 机器事实：id/stage/type/git/env/model/workload/engine/status
├── env.json             # 设备、OS、python、torch/transformers、CUDA/MPS、内存
├── metrics.jsonl        # 原始逐请求/逐配置指标（一行一个 JSON）
├── summary.json         # 聚合：median/p95/min/max/CV + 判据结果
├── stdout.log
├── notes.md             # 人写：观察、结论、边界、下一步
├── verdict.json         # 人写：pass/fail/inconclusive + decision_id
├── reproduce.sh         # 一条命令重跑（校验 commit）
└── figures/             # 可选：从 summary 生成的图
```

### 2.1 不可变规则

- `run_id` 形如 `W0-0001`，目录 `results/W0/0001-carrier-probe/`。
- **不覆盖、不复用 id。** 目录已存在时 runner 直接拒绝。
- 失败/中断/作废也保留：`manifest.status = failed | aborted`，新 run 用 `supersedes` 指回旧 id。
- `manifest.json` 在 run 开始时写 `status: running`，结束时写 `completed | failed`；之后不再改。
- `metrics.jsonl` 只追加；不回头改数。
- 人只写 `notes.md` 和 `verdict.json`；机器产物由 runner 写。

### 2.2 状态与 verdict 是两件事

| 字段 | 取值 | 谁写 |
|---|---|---|
| `manifest.status` | `running / completed / failed / aborted` | runner |
| `verdict.verdict` | `unreviewed / pass / fail / inconclusive / invalid` | 人 |

`completed` 不等于 `pass`。没经过人评审的 run 在索引里显示 `unreviewed`。

---

## 3. 配置契约

一个 run 一个 TOML，自包含；模板见 `configs/experiments/_template.toml`。

```toml
schema = 1

[exp]
id = "W0-0001"                 # <stage>-<4位序号>
stage = "W0"
type = "probe"                 # probe | microbench | scenario | smoke
slug = "carrier-probe"         # 目录名用，英文/数字/连字符
title = "双载体：显存/延迟 vs 上下文"
hypothesis = ["P1", "P2", "P3"]
seed = 0

[[models]]
name = "Qwen/Qwen3-0.6B"
revision = "<hf-commit>"       # 必须是 commit hash，不能是 main
dtype = "bfloat16"

[workload]
kind = "synthetic"             # synthetic | prompts
context_lengths = [512, 2048, 8192]
max_new_tokens = 128
batch_sizes = [1]

[engine]
impl = "hf"                    # dummy | hf | minillm | continuous | paged | hybrid
device = "auto"                # auto | cuda | mps | cpu
max_batch = 1

[measure]
warmup = 3
repeats = 5
timer = "auto"                 # auto | perf_counter | cuda_event（cuda_event 需 device=cuda）
memory = ["torch_peak", "driver"]

[accept]
mode = "record_only"           # record_only | threshold
criteria = ["P1.peak_mem_ratio", "P2.crossover_tokens"]
```

### 3.1 必填与校验

- `[exp]`：`id / stage / type / slug / title` 必填；`id` 的前缀必须等于 `stage`。
- `[[models]]`：非 `smoke` 至少一个；`revision` 不能是 `main` 或空。
- `[engine].impl`：必须是 `bench.engines.build_engine` 认识的实现。
- `[measure]`：默认 `warmup=3, repeats=5`；偏离默认必须在 `notes.md` 写原因。
- `[accept]`：`mode=record_only` 的 run 不判 pass/fail；`mode=threshold` 必须引用
  `configs/thresholds.toml` 的键，不能内联魔法数字。`bench/thresholds.py` 负责求值：
  `*_cv_max` 直接对 summary 求值，需要配对 baseline 或驱动侧显存的判据保持
  `pending` 并写明原因——不会假装通过。

### 3.2 prompt 集

- 固定 prompt 集在 `bench/prompts.json`，带 `version` 和 `sha256`。
- 每次 run 记录 `prompt_sha256`；prompt 集一变，跨 run 比较自动失效。
- prompt 存文本，不存 token id：不同模型 tokenizer 不同；每个 run 记录**实际 token 长度**，
  跨模型比较用实际长度，不用桶名。
- `synthetic` workload 不读 prompt 集，记录 `context_lengths` 与 `seed`。

### 3.3 阈值

- 所有目标值/判失败线集中在 `configs/thresholds.toml`，字段含
  `value / unit / status(target|calibrated) / basis_run`。
- W1 基线跑完后校准一次；`status` 从 `target` 改为 `calibrated`，并写 `basis_run`。
- 报告里必须区分“目标值”与“实测值”；不得把目标值写成实测结果。

---

## 4. provenance（每个 run 必须可追溯）

`manifest.json` + `env.json` 至少包含：

| 项 | 例子 |
|---|---|
| git commit | `1ce0bf5...` |
| git dirty | `false`（官方 run 必须 false） |
| 设备 | `cuda:0 / NVIDIA 4090 / 24GB` 或 `mps / Apple M1 / 16GB` |
| python / torch / transformers | 精确版本 |
| 模型 | repo id + revision commit + dtype |
| tokenizer | repo id + revision |
| prompt 集 | version + sha256 |
| seed | workload / torch |
| 命令 | 完整 argv |
| 配置 | `config.toml` + `config_sha256` |

- **官方性能 run 必须在干净 git 树上跑**；dirty run 只能 exploratory，索引标 `dirty`，报告不引用。
- `reproduce.sh` 先校验 commit，再用 `--out results-repro` 重跑到新目录；不覆盖原 run。
- 权重只放 HF cache，不进仓库；run 里只记录 revision。

---

## 5. 测量协议

### 5.1 默认参数

| 项 | 默认 | 说明 |
|---|---|---|
| warmup | 3 | 每个 case 先跑 3 次丢弃；loadgen 用 3 个完整波次或 10s，取长 |
| repeats | 5 | 每个 case 测 5 次 |
| 聚合 | median + p95 | 同时报 min/max/n/CV；不只报 mean |
| 生成 | greedy | `max_new_tokens=128`；正确性 run 可不同，但必须记录 |
| 进程 | 一个配置一个进程 | 跨配置重开进程，避免 allocator/thermal 污染 |
| 顺序 | 同进程同顺序 | 基线在前、候选在后；怀疑顺序效应时反向再跑一个新 run |
| 方差 | CV ≤ 10%（**按长度桶**算，每 prompt 一个 CV） | 超了标 `warning`，重跑整个配置，不挑点；`overall` 会把 16–2048 token 的长度跨度算进 CV |

### 5.2 指标定义

| 指标 | 定义 | 采集 |
|---|---|---|
| TTFT | 请求入队 → 首 token | `perf_counter` + 设备同步；CUDA 目标用 event |
| queue_ms | 入队 → 实际开始 prefill | 仅在 loadgen/调度器可观测时 |
| prefill_ms | prefill 前向 | 引擎内计时 |
| TPOT | `(末 token − 首 token)/(N−1)` | 不含 prefill |
| E2E | 入队 → 末 token | |
| throughput | 输出 token/s、请求/s | 报 offered 与 achieved |
| peak mem | 权重 / KV / 线性状态 / 激活 | `weights_mb`（参数字节）、`kv_mb / state_mb / conv_mb`（直接数 `past_key_values`，与设备无关）、`peak_mem_mb`（设备计数器）、`mem_delta_mb`（相对加载后的 driver 增量）；分项能测就测，不能测就标 `unavailable` |
| CV | `std/mean` | summary 自动算 |

### 5.3 设备分层

| 设备类 | 用途 | 计时/显存 |
|---|---|---|
| `dev-mps`（本机 M1 16GB） | 正确性、冒烟、相对趋势 | `perf_counter` + `torch.mps.synchronize()`；`driver_allocated_memory` |
| `measure-cuda`（4090 24GB） | 正式吞吐/显存数字 | `torch.cuda.Event` + `synchronize()`；`max_memory_allocated` + `nvidia-smi` |
| `cpu` | 单元测试/确定性对比 | `perf_counter`；RSS |

- **不同设备类的数字永不同表。** 报告表格必须带 `device_class` 列。
- M1 上的吞吐/延迟只说明趋势；不把 M1 数字写成 4090 数字。
- 若没有 4090，W2 的 `nvidia-smi` 验收项改为 MPS 口径并在报告里注明，不能假装有 CUDA 数据。

### 5.4 HF 与自实现

- 不混用绝对值：HF 一张表，自实现一张表；比值可以放同一张表。
- HF baseline 必须记录：`attn_implementation`、dtype、device_map、batch、`max_new_tokens`、transformers 版本。
- 自实现 attention 从 `W2-0003` 起固定为 PyTorch SDPA；`W2-0002` 是手写 attention 的历史 run，两者的绝对数字不可直接混读。
- 自实现必须记录：block_size、max_batch、scheduler 策略、preemption、dtype、prefix cache 开关。

### 5.5 正确性先于性能

- 性能 run 前，先跑对应的 pytest 正确性门禁；红了不测性能。
- 正确性 run 不报性能数字；性能 run 不拿来证明数值正确。
- 正确性验收：贪心 64 token 逐 id 全等；每层 `max|Δ|`、末层 logits `max|Δ|`；
  batch 1/2/4 一致；容差写进 `thresholds.toml`，W1 校准。

---

## 6. 四类 run

| 类型 | 回答 | 必需 | 不算完成的情况 |
|---|---|---|---|
| `probe` | 探索性曲线（W0 P1–P3） | `record_only`；记录每个 context 的显存/延迟 | 只有单点、没有交叉点分析 |
| `microbench` | 单个机制的相对收益 | 配对 baseline；5 次重复；CV | 缺 baseline、只报绝对值 |
| `scenario` | 负载下的调度/缓存收益 | 固定 prompt 集；到达率或并发；TTFT/TPOT/吞吐 | 没有排队/饱和说明 |
| `smoke` | 契约/管线是否通 | 可直接用 `dummy` 引擎 | 拿 smoke 数字当证据 |

- `scenario` 的 `arrival = "poisson"` 在 W3 才实现；当前 runner 只支持 closed-loop。
- `microbench` / `scenario` 必须有**配对 baseline**（同 commit、同设备、同 workload、同 dtype）。
- 配对前 `bench/thresholds.py` 会校验 `prompt_sha256` / `dtype` / `device_class` /
  `max_new_tokens` 以及每条 prompt 的 `out_tokens`；任一不符判据记 `invalid`，不给比值。
- 每个机制 run 的 `notes.md` 必须回答“**什么时候不划算**”；答不出就标 `inconclusive`。

---

## 7. 人工评审

`notes.md` 模板：

```markdown
# W0-0001 · 双载体：显存/延迟 vs 上下文

## 假设与判据
- P1: ...
- 判失败线: ...

## 观察
- 数值（引用 summary.json）：
- 与预测的偏差：

## 结论（含“什么时候不划算”）

## 限制与偏差
- 设备 / dtype / 样本量 / 已知噪声：

## 下一步
```

`verdict.json`：

```json
{
  "run_id": "W0-0001",
  "verdict": "unreviewed",
  "reviewer": null,
  "reviewed_at": null,
  "decision_id": null,
  "notes": ""
}
```

- `verdict` 只由人改；agent 可以起草，但必须保留 `reviewer` 为空直到用户确认。
- 触发阶段决策时，写 `docs/decisions.md`，并在 `ROADMAP.md` §12 回填证据 run。

---

## 8. 索引与保留

- `results/INDEX.md` 由 `scripts/exp_index.py` 生成，**不手改**。
- 索引列：run / stage / type / title / model / engine / key metric / verdict / date / commit。
- `scripts/exp_index.py --check` 在测试/CI 中校验：必需文件齐全、manifest 可解析、run_id 与目录一致。
- run 不删除；作废用 `verdict=invalid` + `supersedes`。只有明确的隐私/体积问题才归档到仓库外。
- `metrics.jsonl` 若超过 5MB，gzip 后保留；`summary.json` 永远保留明文。
- 提交：config/manifest/env/summary/notes/verdict/reproduce.sh/小图；不提交权重、HF cache、原始大 trace。

---

## 9. 复现检查清单

一次 run 声称可复现，至少满足：

- [ ] git clean，commit 已记录
- [ ] 模型 revision 是 commit hash
- [ ] tokenizer revision / prompt sha256 已记录
- [ ] env.json 含设备、python、torch、transformers
- [ ] 命令完整，`reproduce.sh` 可执行
- [ ] warmup/repeats/聚合方式已记录
- [ ] 失败/异常样本没有被静默丢弃
- [ ] 结论引用了具体 run id，而不是“之前跑过”

---

## 10. 反模式（来自旧训练项目）

| 反模式 | 为什么不行 | 新约定 |
|---|---|---|
| print-only 测试 | 失败也 0 退出，不能当门禁 | pytest `assert`，非零退出 |
| `train.py --exp` 覆盖目录 | 历史被删，无法复现 | run id 不可复用，失败也保留 |
| 手改 `result.md` 台账 | 合并冲突、格式漂移、TODO 永远不补 | 生成 `INDEX.md` + `verdict.json` |
| 单次 eval 当结论 | 没有方差，噪声当收益 | warmup + 5 次 + median/p95/CV |
| 只报 mean | 长尾被平均掉 | median + p95 + min/max |
| 缺 baseline | 不知道收益从哪来 | 每个机制配对 baseline |
| 混设备/混 dtype | 数字不可比 | device_class/dtype 是一等字段 |
| 静默重生成 golden | 测试跟着实现走，失去意义 | golden 由脚本生成并 review，测试只读 |

---

## 11. 旧约定 → 新约定

| 旧 | 新 | 说明 |
|---|---|---|
| `Exp-XXX` 全局流水号 | `W*-NNNN` + `results/<stage>/<seq>-<slug>/` | 阶段内编号，可重跑 |
| `result.md` 自动追加 + 手填 TODO | `INDEX.md`（生成）+ `notes.md` + `verdict.json` | 机器事实与人的判断分离 |
| `config.json`（data/train/weight） | `config.toml`（exp/model/workload/engine/measure/accept） | 训练配置不再适用 |
| `train_log.csv` | `metrics.jsonl` + `summary.json` | 从 loss 轨迹变成服务指标 |
| `samples.txt` | 可选 `figures/` / 生成样本 | 不是主证据 |
| `model.pth` | 无；权重只在 HF cache | 不训练、不存权重 |
| `TODO(用户)` | `verdict: unreviewed` | 显式状态，索引可见 |

保留的旧优点：一个实验一个自包含目录、config 快照、git 状态、固定 prompt、人的结论、归档而非静默删除。
