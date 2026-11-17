# W8 执行计划 · 报告与判据矩阵

> 状态：active（2026-11-17 立项）。**收口时保留本文件**（报告口径的唯一记录），不删。
> 口径由 `docs/decisions.md` **D10** 冻结；本文件只列范围、run 表与出口判据。
> 进度以 `ROADMAP.md` §12 为准；本文件不复制阶段状态。

## 1. 范围

- **做**：18 格判据矩阵、六节报告、复现说明、D10 收口、ROADMAP/README 同步。
- **不做**：量化（int8/int4）、goodput/SLO、CUDA graph、按需分配、chunked prefill、
  prefix cache、4B 档、`x*` 求解（W10）、任何引擎重构。
- 唯一新增代码：`scripts/w8_matrix.py`（只读 `summary.json`，出表 + 出图）。

## 2. 矩阵

2 臂 × 3 ctx（512 / 2k / 8k）× 3 B（1 / 8 / 32）= **18 格**。
模型档只有 0.6B / 0.8B（4B 从未加载权重，D10 记 out of scope）。
batch = `[engine].max_batch`；每格只报**实测**，列定义见 D10-2。

| 臂 | ctx | B=1 | B=8 | B=32 |
|---|---|---|---|---|
| pure 0.6B | 511 / 2k / 8k | `W8-0004` / `W7-0027` / `W7-0027`（8k decode N/A） | `W8-0005` / `W7-0029` / `W7-0029`（8k decode N/A） | `W8-0006` / `W8-0007` / **N/A**（上界 B=20，`W8-0008`） |
| hybrid 0.8B | 512 / 2k / 8k | `W7-0026` | `W7-0028` | `W8-0001/0002/0003` |

pure 短行用 **511**（512 合成 prompt 第 3 个 token 即 EOS，P-31）；hybrid 无此问题，保持 512。
`W7-0027/0029` 的 8k 行 `out_tokens=2`（EOS），8k pure 的 decode 列记 **N/A**（D10-3）。

## 3. run 表（`configs/experiments/W8/`）

统一：`type=scenario`、`arrival=closed`、`max_new_tokens=128`、`block_size=16`、
`max_prefill_tokens=8192`、`warmup=3 / repeats=5`、`mode=record_only`、`shuffle=false`。

| id | 臂 | ctx | B | requests | pool | blocks | slots | 备注 |
|---|---|---|---|---|---|---|---|---|
| 0001 | hybrid 0.8B | 512 | 32 | 32 | paged | 1400 | 32 | 32×40 块 |
| 0002 | hybrid 0.8B | 2048 | 32 | 32 | paged | 4400 | 32 | 32×136 块 |
| 0003 | hybrid 0.8B | 8192 | 32 | 32 | paged | 16700 | 32 | 32×520 块；KV 池 ≈6.1 GiB |
| 0004 | pure 0.6B | 511 | 1 | 3 | paged | 200 | — | P-31 短行 |
| 0005 | pure 0.6B | 511 | 8 | 8 | paged | 400 | — | P-31 短行 |
| 0006 | pure 0.6B | 511 | 32 | 32 | paged | 1300 | — | P-31 短行 |
| 0007 | pure 0.6B | 2048 | 32 | 32 | paged | 4400 | — | |
| 0008 | pure 0.6B | 8192 | 20 | 20 | paged | 10500 | — | 边界探针；EOS 早退，只作内存上界 |

B=32 行用满池，**不按 1640 MiB 判**：summary 会记 `over_budget=true`，报告必须同时给
池大小与 `over_by_mb`；容量结论仍来自 W7 的 fill run（1640 MiB 口径）。

## 4. 预检（2026-11-17，`/tmp`，不入证据）

| 探针 | 结论 | 峰值（alloc / reserved / driver） |
|---|---|---|
| hybrid 8k×32，64 new | 峰值并发 **32**、`step_ms 54.8`、pass 20.8 s | 6.26 / 6.38 / 6.85 GiB |
| pure 8k×16 | 可加载；8k prompt 2 token 即 EOS | 15.5 / 15.7 / 16.1 GiB |
| pure 8k×20 | 可加载，余 ~4.3 GiB | 19.1 / 19.3 / 19.8 GiB |
| pure 8k×32 | **不可行**：KV 池 32×897.75 MiB = 28.1 GiB > 24.5 | — |

结论：8k×32 的 hybrid 格保留；pure 的 N/A 上界写 **B=20**；pure 的 `kv_blocks` 按
128 new tokens 取 `(ctx+128)/16 × B`，8k B=20 为 10400 → 配置留 10500。

## 5. 步骤

| 步 | 内容 | 出口 |
|---|---|---|
| P0 | D10 提交、6 个配置、`--dry-run` 全过 | 预注册早于正式 run |
| P1 | 逐条跑正式 run；**每个 run 后提交结果目录**（否则下一次 run 判 dirty），我起草 notes/verdict（`reviewer: null`） | 6 个 summary + `exp_index --check` 绿 |
| P2 | `scripts/w8_matrix.py` 出 `reports/matrix.md` + `figures/fig_w8_matrix.png` | 18 格无空格；口径断言全过 |
| P3 | `reports/final.md` 六节（2500–3500 字）+ `reproduce.md` | 每个数字带 run id |
| P4 | §12 → ✅、§13 v33、README/§9 回填、6 个 verdict 复核 | 用户确认 |

## 6. 出口判据

- 18 格全部有结论：数值，或「N/A + 解析式/实测上界 + 原因」；
- 报告每个数字可点回 run id，带 `device_class`；无跨 regime / 设备 / dtype 混表；
- 负结果专章与正结果同篇幅，每条写清「什么时候不划算」；
- `reports/reproduce.md` 每条命令在干净树上实跑过；图/表由脚本生成；
- D10 accepted、6 个 verdict 经用户确认、快速门禁与 `exp_index --check` 绿。

## 7. 交给 W10 的边界表

| 轴 | W8 给出 | W10 需要补 |
|---|---|---|
| ctx × B | 3×3 实测格点（含 N/A） | 格点间扫点求 `x*` + 区间 |
| 容量 | 1640 MiB 的预留口径曲线 | 按需分配（W9b）第二条曲线 |
| decode | Regime A（host-bound） | graph-on 后重锚（W9a） |
| H / λ / graph | 未测 | 全部 |
| prefill / TTFT | 已测，W9 不影响 | 直接引用，不重测 |
