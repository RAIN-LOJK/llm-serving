# 单卡 LLM Serving：机制收益边界实测

在单张 RTX 4090 上实现一个可运行的 serving 引擎，用同一套调度器测连续批处理、分页 KV、packed prefill、block-table decode 与混合注意力双状态管理的收益边界。载体是 `Qwen/Qwen3-0.6B`（全注意力骨架）与 `Qwen/Qwen3.5-0.8B`（3:1 混合）；每个数字都来自 `results/` 里一次干净树上的 run，或 `tests/golden/` 下经评审的数值夹具，绑定 run id / fixture、设备与判据。

> **什么规模、什么上下文下，连续批处理 / 分页 KV / 混合注意力状态管理才回本？**

![引擎结构](figures/fig_architecture.png)
<sub>请求 → FIFO → Scheduler → packed prefill → batch decode；KV 层有 seat（连续 slab）和 paged（块表）两条后端。示意图，不是测量结果。</sub>

- 技术主计划：[ROADMAP.md](ROADMAP.md)（W0–W8 的唯一权威）
- 实验约定：[docs/experiments.md](docs/experiments.md)
- 决策记录：[docs/decisions.md](docs/decisions.md)
- 环境与设备口径：[docs/environments.md](docs/environments.md)
- 全部 run：[results/INDEX.md](results/INDEX.md)

## 实测结果（RTX 4090 · Qwen3-0.6B / Qwen3.5-0.8B · bf16）

| 机制 | 对照 | 结果 | 判据 |
|---|---|---|---|
| W3 连续批 | static vs continuous，B=8 mixed | 154.94 → 311.43 tok/s（**2.01×**） | pass |
| W4 分页 KV | seat vs paged，同 1640 MiB | 9 → 28 并发（**3.111×**），internal frag 0.55% | C1–C3 pass；C4 **1.671×** 负结果，按预注册不报吞吐 |
| W4.5 路径 v2 | serial vs packed / seat vs paged | TTFT p99 **3.760×**；decode **1.040×**（接受线 <1.15，1.0 目标未达）；E4 **2.053×**（both saturated，record-only） | D7 accepted |
| W7 双状态管理 | 混合 0.8B vs 纯 0.6B，同一 `Scheduler` | 每序列状态误差 0.83/0.48/0.18%（512/2k/8k）；同 1640 MiB 容量比 **2.70/6.17/14.0×**；P1 **7.816×**，交叉点 190.8 prompt token | A1 8/8 token hash、G0 64/64；P2/P3 见下 |
| W8 判据矩阵 | 2 臂 × 3 ctx × 3 B（18 格） | hybrid 8k×B=32 可达（6.4 GiB、172.1 tok/s、TPOT 38.33 ms）；pure 8k×B=32 结构性 N/A（28.1 GiB），上界 B=20 | 矩阵 `reports/matrix.md`；pure 8k decode 记 N/A（EOS） |

正确性门禁：W1 贪心 64/64（logits `max|Δ|` 2.0e-5）；W4 paged/seat 40/40 token 全等；W4.5 E0 64/64、E2 8/8 token hash 全等；W5 单层 9 个 stage `max|Δ|=0`；W6 分块 prefill 与逐步 decode 自洽（`3.040e-5`）、贪心 64/64；W7 0.8B B=1/B=8 **8/8** token hash、G0 bf16 与 HF **64/64**。

### 为什么批处理有效

![W3 批处理摊薄](figures/fig_w3_amortization.png)
<sub>W3 机制图 · B=1→8 吞吐 7.9×，B=8→32 再 3.8×；per-token 成本 21.1 → 0.86 ms（B=1→32，ctx511）。continuous vs static 的 2.01× 是另一组对照（`W3-0010/0011`）。</sub>

### 分页 KV 买到多少并发

![W4 容量](figures/fig_w4_capacity.png)
<sub>W4 · 同 1640 MiB 总预算、同一 trace：seat 9 → paged 28 并发（3.111×），frag 0.55%。曲线为确定性 allocator 模拟，星标为 `W4-0003` 实测；uniform-2048 是负对照（≈1.0×），收益只来自长度异质性。</sub>

### 路径 v2：packed prefill + block-table decode

![W4.5 decode](figures/fig_w4_5_decode.png)
<sub>W4.5 E2 · 27.379 ms（1.203× fail）→ 23.651 ms（1.040× pass，接受线 <1.15）；seat 22.749 ms；1.0× 目标未达。`W45-0004/0005/0012`。</sub>

![W4.5 prefill](figures/fig_w4_5_prefill.png)
<sub>W4.5 E3 · TTFT p99 207.4 → 55.1 ms（3.760×），p50 105.1 → 53.9 ms（0.512×）；static B=8 mixed。`W45-0006/0007`。</sub>

### 混合注意力：KV 只剩 1/4 的层

![W7 容量](figures/fig_w7_capacity.png)
<sub>W7 · 同 1640 MiB 总预算：混合 0.8B / 纯 0.6B 的容量比 **2.70/6.17/14.0×**（ctx 512/2k/8k）；混合内部 paged/seat 为 48/37 = **1.297×**（纯注意力的历史值是 3.111×）——分页只能捞 6/24 层的 KV。`W7-0016/0017`。</sub>

![W7 交叉点](figures/fig_w7_crossover.png)
<sub>W7 · 每序列状态随上下文：prefill 口径交叉点 **190.8 prompt token**（实测区间 [128,192]），8k 处每序列显存 **7.816×**（P1）。`W7-0020…0025`。</sub>

### 18 格判据矩阵：什么时候能跑、什么时候不划算

![W8 矩阵](figures/fig_w8_matrix.png)
<sub>W8 · decode TPOT 随 ctx × batch：pure 在 22–25 ms 基本平；hybrid 从 B=1 的 ~29 ms 升到 B=32 的 33.5/37.3/38.3 ms（512/2k/8k）。pure 8k decode 因 EOS 记 N/A；pure 8k×B=32 需 28.1 GiB KV，结构性 N/A（上界 B=20）。`W8-0001…0008`。</sub>

负结果：E4a 连续批收益从纯注意力的 1.863× 降到 **1.498×**；P2-memory **不成立**（512 下混合已更省，2.27×）；P2-latency 依赖口径（service 与 decode 两臂方向不一致）；P3 两口径均**不可归因**。

## 范围与边界

当前版本覆盖 W0–W8：Qwen3-0.6B 上的连续批、分页 KV、packed prefill + block-table decode；
Qwen3.5 的全注意力层、短卷积 + Gated DeltaNet（分块 prefill / 逐步 decode）与 24 层混合模型均已对齐 HF。
W7 把两代架构放进了同一个 `Scheduler`：分页 KV 覆盖 6/24 全注意力层，常数大小的 `StatePool` 覆盖 18 层线性状态。
W8 收口为 18 格判据矩阵与六节报告（`reports/final.md`）；进度与判据以 [ROADMAP.md](ROADMAP.md) §12 为准。

W4.5 收口后，生产调度器只有 packed group，paged decode 只有 CUDA Triton block-table kernel；
CPU/MPS 上 paged 直接报错，seat 才是 CPU/MPS 路径。

## 快速开始

```bash
# 1. 依赖（miniconda base 已带 torch；这里补 transformers / tokenizers / pytest）
python -m pip install -r requirements-dev.txt

# 2. 快速门禁：不需要任何 HF 权重
python -m pytest -m "not slow and not hf"

# 3. 环境快照
python scripts/env_snapshot.py

# 4. runner 冒烟（写到临时目录，不污染 results/）
python -m bench.run \
  --config configs/experiments/W0/9001-runner-smoke.toml \
  --out /tmp/llm-serving-smoke --allow-dirty

# 5. 生成 / 校验实验索引
python scripts/exp_index.py
python scripts/exp_index.py --check
```

正式测量请在干净 git 树上跑；官方数字以 `measure-cuda`（4090）为准，本机 M1 只做正确性与冒烟。口径见 [docs/environments.md](docs/environments.md)。

## 负结果与边界

<details>
<summary><b>W4 C4：分页 decode 的 gather 代价 1.671×（点击展开）</b></summary>

![W4 decode cost](figures/fig_w4_step.png)
<sub>W4 C4 · 同并发 B=8 decode step 21.32 → 35.63 ms（1.671×）；按预注册不报吞吐。`W4-0004/0005`。</sub>

W4 的容量结论（3.111×）成立，但 decode 代价超过预注册的 1.5× 撤回线，所以 W4 只报容量与碎片，不报吞吐。W4.5 的 packed prefill + Triton block-table decode 正是为这条负结果做的路径级返工；返工后 decode 回到 1.040×（接受线，仍不是 1.0 目标）。

</details>

## 目录

```text
model/         手写算子：RoPE / GQA / KVcache / SwiGLU / RMSNorm / TransformerBlock；
               以及 Qwen3.5 混合层（全注意力 / GDN / 短卷积 / partial-M-RoPE）
llm_utils/     过渡层：sampling + 批量 generate（S2 后由 serve/engine.py 接管）
serve/         engine / scheduler / block_manager / state_pool（+ hybrid_engine）
bench/         run / metrics / loadgen（含 poisson_trace）/ prompts.json；
               engines/：dummy·hf·minillm·continuous·fill·paged·hybrid
configs/       实验配置（experiments/）与集中阈值（thresholds.toml）
scripts/       env_snapshot / exp_index / probe_* / make_golden / align_* / plot_* / sim_paged_capacity
tests/         pytest：unit + integration + golden
results/       run 目录（W*/NNNN-slug/）+ 自动生成的 INDEX.md
figures/       报告图与架构示意图
docs/          实验约定 / 决策 / 环境 / 载体 / 混合注意力
reports/       最终报告与判据矩阵
```

## License

MIT，见 [LICENSE](LICENSE)。仓库不包含模型权重；Qwen3-0.6B 与 Qwen3.5-0.8B 的使用遵循各自上游许可证。
