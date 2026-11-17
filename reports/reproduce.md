# 复现说明

> 一条命令一档。所有正式数字来自 `measure-cuda`（单张 RTX 4090）、bf16；**不跨设备比较**。
> 运行契约见 [docs/experiments.md](../docs/experiments.md)；矩阵与报告见
> [reports/final.md](final.md) / [reports/matrix.md](matrix.md)。

## 0. 前置

```bash
conda activate base
python -m pip install -r requirements-dev.txt   # base 已有 torch
python -m pip install -r requirements-cuda.txt  # flash-attn / fla / triton（4090）
python scripts/env_snapshot.py                  # GPU / 驱动 / torch / transformers / commit
```

权重（HF 缓存，revision 已 pin 在配置里）：

- `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
- `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`

正式 run 要求**干净树**；dirty 只能加 `--allow-dirty` 做 exploratory，报告不得引用。
每个 run 目录自带 `reproduce.sh`（记录提交号，输出到 `results-repro/`，避免覆盖原 run）。

## 1. 快速门禁（无 HF 权重、无网络）

```bash
python -m pytest -m "not slow and not hf"
```

## 2. 数值门（需要 HF 权重，`MINILLM_RUN_HF=1`）

```bash
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py -q        # 0.6B 64/64
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen35_model.py -q # 0.8B 64/64
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_paged_reopen.py -q       # 池重开（W8 回归）
```

## 3. 纯注意力：机制档

```bash
# 连续批（C1）：static vs continuous
python -m bench.run --config configs/experiments/W3/0010-e2-static-b8-mixed.toml
python -m bench.run --config configs/experiments/W3/0011-e2-continuous-b8-mixed.toml

# 分页容量（C1）+ 碎片（C2）：同 1640 MiB
python -m bench.run --config configs/experiments/W4/0002-seat-fill-max2048.toml
python -m bench.run --config configs/experiments/W4/0003-paged-fill-max2048.toml

# packed prefill + block-table decode（E2/E3）
python -m bench.run --config configs/experiments/W45/0004-seat-b8-cost.toml
python -m bench.run --config configs/experiments/W45/0012-paged-b8-cost-fixed.toml
python -m bench.run --config configs/experiments/W45/0006-static-serial.toml
python -m bench.run --config configs/experiments/W45/0007-static-packed.toml
```

配置文件号以 `configs/experiments/<stage>/` 为准；历史 run 的 `config.toml` 也在各自目录里。

## 4. 混合注意力：双状态与跨代对照

```bash
python -m bench.run --config configs/experiments/W7/0026-e3-hybrid-b1-tpot.toml   # B=1 TPOT
python -m bench.run --config configs/experiments/W7/0028-e3-hybrid-b8-tpot.toml   # B=8 TPOT
python -m bench.run --config configs/experiments/W7/0016-e4b-hybrid-paged.toml    # paged 容量
python -m bench.run --config configs/experiments/W7/0017-e4b-hybrid-seat.toml     # seat 基线
```

## 5. W8 判据矩阵（18 格）

```bash
# B=32 行（hybrid 三格 + pure 511/2k）
python -m bench.run --config configs/experiments/W8/0001-matrix-hybrid-512-b32.toml
python -m bench.run --config configs/experiments/W8/0002-matrix-hybrid-2k-b32.toml
python -m bench.run --config configs/experiments/W8/0003-matrix-hybrid-8k-b32.toml
python -m bench.run --config configs/experiments/W8/0006-matrix-pure-511-b32.toml
python -m bench.run --config configs/experiments/W8/0007-matrix-pure-2k-b32.toml
# pure 短路（511，P-31）与 8k 边界（B=20）
python -m bench.run --config configs/experiments/W8/0004-matrix-pure-511-b1.toml
python -m bench.run --config configs/experiments/W8/0005-matrix-pure-511-b8.toml
python -m bench.run --config configs/experiments/W8/0008-bound-pure-8k-b20.toml
```

B=1/8 的 2k/8k 格来自 W7 的 `0026…0029`（一个 run 覆盖三个 ctx），不需要重跑。

## 6. 表、图、索引

```bash
python scripts/w8_matrix.py                 # reports/matrix.md + figures/fig_w8_matrix.png
python scripts/plot_w3_amortization.py
python scripts/plot_w4.py
python scripts/plot_w45.py
python scripts/plot_w7.py
python scripts/exp_index.py --check         # 索引与 run 契约自检
```

## 7. 口径提醒

- 目标值在 `configs/thresholds.toml`，与实测值分列；不要把目标写成结果。
- `service` 与 `decode` 两种 TPOT 不混用；`p95` 不报 `p99`。
- B=32 行的池是满池（1400/4400/16700/10500 块），不按 1640 MiB 判；容量结论只来自 W7 fill。
- pure 8k 的 decode 是 N/A（EOS 早退），不是“混合更快”的证据。
