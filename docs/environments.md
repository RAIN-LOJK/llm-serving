# 环境与设备口径

> 结论先行：**M1 只做正确性/冒烟/趋势；正式吞吐与显存数字在 4090 上出。**
> 两套数字永不同表；设备是结果的一等字段。

---

## 1. 两套环境

| | `dev-mps` | `measure-cuda` |
|---|---|---|
| 机器 | MacBook Air M1，16GB 统一内存 | 单卡 4090 24GB（租用/远程） |
| 后端 | MPS（`torch.backends.mps`） | CUDA |
| 用途 | 单元/集成测试、W0 逻辑验证、冒烟 | W0 正式探测、W2–W7 正式 benchmark |
| 计时 | `perf_counter` + `torch.mps.synchronize()` | `torch.cuda.Event` + `synchronize()` |
| 显存 | `torch.mps.driver_allocated_memory()` / `current_allocated_memory()` | `max_memory_allocated()` + `nvidia-smi` 采样 |
| 报告标签 | `device_class = "dev-mps"` | `device_class = "measure-cuda"` |

本机实测（2026-10）：macOS / Apple M1 / 16GB / 无 CUDA / Python 3.12.15 /
torch 2.13.0 / transformers 5.14.1 / MPS 可用。HF cache 在 `~/.cache/huggingface`。

4090 实测（2026-10-22）：Ubuntu 22.04 / RTX 4090 24GB / 驱动 580.105.08 /
Python 3.12.3（**miniconda base**，`/root/miniconda3/bin/python`）/ torch 2.8.0+cu128 /
CUDA 12.8。**不建 `.venv`**，命令里的 `python` 就是 base 解释器。
直连 `huggingface.co` 超时，**必须** `export HF_ENDPOINT=https://hf-mirror.com`；
PyPI 走清华镜像（`pip config` 已全局配置）。
本机 CUDA 是否可用以 `python scripts/env_snapshot.py` 的 `cuda_available` 为准——
`nvidia-smi` 与 `torch.cuda` 都依赖 `/dev/nvidia*`，在受限 shell/容器里会报
`NVML: Unknown Error` 或 `cuInit -> 304`，那是执行环境问题，不是机器问题。

GPU 加速包走单独文件：[requirements-cuda.txt](../requirements-cuda.txt)
（`flash-linear-attention==0.5.2` + `causal-conv1d==1.7.0`，W0-0001 实测版本）。
它们**只装 4090**：causal-conv1d 要编译 CUDA 扩展、fla 要 Triton，CPU/MPS 装不上；
不装也能跑，只是 Qwen3.5 的短卷积/delta rule 退回 torch 实现。版本由 `env.json` 记录。

---

## 2. 铁律

1. **不跨设备比较。** 报告表必须有 `device_class` 列；M1 与 4090 不放在同一列比较。
2. **不在 M1 上产出的数字后面暗示 4090。** 简历/报告写“M1 实测”或“4090 实测”。
3. **没有 4090 时不要伪造 CUDA 口径。** W2 的 `nvidia-smi` 验收项改为 MPS 口径并在
   报告里注明；W3/W4 的吞吐阈值相应降级为“趋势观察”，不写绝对值。
4. **官方 run 在干净 git 树上跑。** dirty run 只做 exploratory。
5. **权重不进仓库。** 只记录模型 repo + revision commit；HF cache 可随时重建。

---

## 3. 模型与缓存

- 载体：
  - A 骨架：`Qwen/Qwen3-0.6B`（纯全注意力）
  - B 目标：`Qwen/Qwen3.5-0.8B`（3:1 混合）
  - 回退：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`（本机已缓存，可用于测试基建）
- 下载时 pin revision：
  ```bash
  python - <<'PY'
  from huggingface_hub import model_info
  for name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]:
      info = model_info(name)
      print(name, info.sha)
  PY
  ```
  把 sha 写进 `configs/experiments/...toml` 的 `revision`，不要写 `main`。
- 磁盘：M1 剩约 14GB，4090 剩约 22GB；0.6B + 0.8B bf16 下载约 3GB（4090 的 HF cache
  目前没有 Qwen 权重）。下载前先 `df -h`，不要把磁盘跑满。
- 32k 上下文在 16GB M1 上可能 OOM；W0 若 OOM，记录为失败样本（`ok=false`），
  不要静默跳过。必要时把 `context_lengths` 上限降到 8k/16k，并在 `notes.md` 说明。

---

## 4. 远程 4090 工作流（建议）

1. 本地 commit + push；远程 `git pull`（不要 rsync 未提交改动）。
2. 远程用 conda base（不建 venv），`python -m pip install -r requirements-dev.txt` 补齐依赖。
3. 跑 `scripts/env_snapshot.py`，确认设备是 `cuda` 且 `device_class=measure-cuda`。
4. 按 `configs/experiments/...` 跑 `bench.run`；产物在 `results/`。
5. 把 `results/<stage>/<run>/` 的文本产物（config/manifest/env/summary/notes/verdict）提交回仓库；
   `metrics.jsonl` 若小于 5MB 一并提交，否则 gzip。
6. 本地 `scripts/exp_index.py` 重建索引；报告引用 run id。

> 远程机器只跑，不手改结果；结论在本地/评审时写。

### 4.1 4090 启动清单（W0）

```bash
# 0) 代码必须是已提交的 commit（run 契约要求干净树）；依赖装在 conda base
git clone <repo> && cd minillm && git checkout <commit>
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-cuda.txt   # 仅 4090：fla + causal-conv1d（CPU/MPS 不要装）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 必须是 True
python scripts/env_snapshot.py | grep -E "cuda_available|cuda_devices|memory_gb"

# 0.5) 快速门禁先绿（不需要权重）
python -m pytest -m "not slow and not hf"

# 1) 权重（4090 直连 huggingface.co 不通，必须走镜像；revision 必须与 config 一致）
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17

# 2) 冒烟（只跑 512/2k；不写 results/）
python -m bench.run \
  --config configs/experiments/W0/0001-carrier-probe.toml \
  --out /tmp/minillm-w0-smoke --limit 2 --allow-dirty

# 3) 正式 W0-0001（warmup=3 / repeats=5 / 32k）
python -m bench.run --config configs/experiments/W0/0001-carrier-probe.toml
```

口径与预期（2026-10-21 的代码状态）：

- prefill 只保留 **1 个位置的 logits**（`logits_to_keep=1`）；32k 时省下 9.3 GiB（0.6B）
  / 15.2 GiB（0.8B）。冒烟若看到 `logits_to_keep` 相关报错，说明模型 forward 不认这个参数，
  不要退回全量 logits。
- 0.8B 只加载 **text decoder**（`text_config`），视觉塔与 MTP 头不进显存；text-only 权重
  实测 1435 MiB（0.6B 为 1137 MiB）。
- 32k 时 cache 上限：0.6B ≈ 3.5 GiB（KV），0.8B ≈ 0.4 GiB（KV + 18 MiB 状态），均能进 24 GB。
- 未安装 `flash-linear-attention` / `causal-conv1d` 时，Qwen3.5 的 delta rule 走 torch 回退，
  prefill 会慢一些；这是**已知限制**，写进 `notes.md`，不要当成架构结论。

---

## 5. 环境快照

每次 run 自动写 `env.json`；也可以手动：

```bash
python scripts/env_snapshot.py
python scripts/env_snapshot.py --out /tmp/env.json
```

字段包括：git commit/branch/dirty、python、torch、CUDA/MPS、GPU 名称与显存、
CPU/内存、transformers/tokenizers/numpy/matplotlib 版本、`HF_HOME`、`CUDA_VISIBLE_DEVICES`。

---

## 6. 已知限制

- M1 的 Python/kernel launch 开销占比高，连续批处理的吞吐收益可能被掩盖；
  这正是 W3 要测的“交叉点”，不是失败。
- MPS 的 bf16 支持取决于 macOS/torch 版本；若不可用，退回 fp16/fp32 并记录 dtype。
- `nvidia-smi` 的进程显存包含 CUDA context，与 `torch.cuda.max_memory_allocated()`
  不是同一口径；报告两个数，不要互相“对上 90%”式硬凑。
- HF `generate` 与手写 decode 的 TTFT 口径不同；比较时只比同口径，或明确注明。
