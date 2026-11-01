# W2 返工 · 固定注意力路径（W2-0004）

> **生命周期：临时文件，W2 阶段完成即删。** 它是这次返工的施工清单，不是长期文档：
> `W2-0004` 判 pass、结论回填 `ROADMAP.md` §12/§13 与 `docs/decisions.md` **D4** 之后，删除本文件
> （若 W2 的返工方案被整体推翻，也直接删，不留半死的清单）。
> **删除时必须同步修掉的引用**（本文件被链接的地方）：
>
> | 位置 | 内容 | 删除时怎么改 |
> |---|---|---|
> | `ROADMAP.md` §5「W2 · 基准框架」 | 「返工（2026-10-31 起）」段的 `[docs/w2-rework.md]` 链接 | 改成「返工已完成，证据 `W2-0004`」并去掉链接 |
> | `ROADMAP.md` §12 进度表 | W2 行由「🔁 返工中」改回 ✅ | 回填证据 run |
> | `ROADMAP.md` §13 v10 修订记录 | 历史条目里的 `docs/w2-rework.md` | 历史不改写；断链可接受，或保留文字去掉链接 |
> | `docs/decisions.md` **D4** | 「**清单**：[w2-rework.md](w2-rework.md)」 | 整行删除（D4 本身保留） |
> | `serve/engine.py` 顶部 | `SLAB_GRANULARITY` 注释里的 `w2-rework B1` | 改成 `D4`（D4 是长期记录） |
> | 本文件 → `docs/w3-plan.md` | §2 E6 与 §0 的前置引用 | 两个临时文件一起删，反向引用随之消失 |

> 状态：**代码已落地（2026-10-31），`W2-0004` 未跑**。本文件是这次返工的清单与规格；阶段判据仍以 `ROADMAP.md` §5 为准，
> 决策记录见 `docs/decisions.md` **D4**。数字分三类：`实测`（有 run）、`探针`（不进报告）、
> `预测`（预注册，待验证）。落地后的门禁与 V2 实测见 §4.1 / §4.2。

---

## 0. 为什么要返工

`W2-0003` 的基线是「一条注意力实现（SDPA）+ 逐序列」。D3 被驳回后（`docs/decisions.md`），
W3 的注意力口径悬空，而 W3 的吞吐判据 C1 需要一个**同 kernel 的锚点**：

- `W2-0003`（sdpa/cat）已经封版，不能改；
- W3 的配对校验要求两臂 `attn_prefill/attn_decode` 相等——换 kernel 后对 `W2-0003` 的配对
  按构造 `invalid`；
- 所以先在新路径上建立 W2 基线 `W2-0004`，`W3-0001`（重构无害）才有基线可挂。

`W2-0004` 用 `supersedes = "W2-0003"`，与 `W2-0002 → W2-0003` 的先例同形。

---

## 1. 冻结的口径（写进 D4，跑之前定死）

| 项 | 规定 | 依据 |
|---|---|---|
| prefill（T > 1） | **逐序列 SDPA**，`is_causal=True`、`attn_mask=None` | 只有无 mask 的调用才可能落到 flash；长度方向**永不 pad**（pad 后 flash/efficient 退出、只剩 math，实测 prefill-2048 25.4 → 125.5 ms） |
| decode（T == 1） | **`hand_baddbmm`**，容器 **`KvSlab`** | 一次 `baddbmm` 覆盖整个 batch，mask 融进 score GEMM（实测 B=8/L=512/28 层 0.7 ms/step，逐序列 SDPA 5.8–8.8 ms，带 mask 的批 SDPA 7.4 ms） |
| 容器 | prefill 用 `KVcache`，decode 用 `KvSlab`（`fill_from` 交接） | slab 地址稳定、无 `cat`；`cat` 只保留给过渡版 `llm_utils/inference.py` 与测试参考 |
| 桶粒度 | 64 列，封顶 `max_position_embeddings`，最多缓存 3 块 | 原实现 256 起按 2 倍翻：2048 + 128 → **4096**，B=8 时每 step 白读 ~448 MiB |
| 每步收 L | **不做**（见 P-25） | 切片后 `kT` 不再是连续视图，`bmm` 会隐式拷贝整个 K，比 padding 更贵 |
| CUDA graph | **不进 W3 数字**；`serve/graph.py` 保留为 parked | graph 会消灭 W3 要测的 host 成本本身；W4 再接线。**注意：graph 只影响 slab 的内部实现，不影响 slab 是否该存在**——slab 的理由是无 `cat` 拷贝 + 给 `hand_baddbmm` 一个稠密调用，两者都不依赖 graph |
| slab 的正当性前提 | **W3 会跑 B ≥ 2**；B=1 时它是净负担（实测 +9~17%，见 §4.1） | 单请求下 `cat` 每步只复制 ctx×`2·n_kv·D`,ctx2048 约 28 MiB ≈ 0.04 ms；slab 的固定记账反而更贵。**若 W3 退化成只跑 B=1，应重新考虑 slab 的去留** |
| CUDA graph | **不进 W3 数字**；`serve/graph.py` 保留为 parked | graph 会消灭 W3 要测的 host 成本本身；W4 再接线 |
| 旋钮 | 全部删除 | `{attn_prefill, attn_decode, decode}` 有 8 种组合、7 种没人跑 |

**不做 `sdpa + slab`**：SDPA 一旦拿到 mask，flash/efficient 全部退出（只剩 math），
而 slab 的全部收益恰恰依赖 mask（稠密桶 + 按行有效长度）。两者组合是纯亏。

---

## 2. 清单

### A · 口径定死（删旋钮）

| # | 内容 | 文件 | 验收 |
|---|---|---|---|
| A1 | 删 `attn_prefill` / `attn_decode`；`T>1 → PREFILL_BACKEND`（sdpa），`T==1 → DECODE_BACKEND`（hand_baddbmm） | `model/config.py`、`model/layers.py` | `grep -r attn_decode model/` 归零 |
| A2 | 删 `BACKENDS` / `build` / `available` / `describe` / `Caps` / 各后端的 `caps`；契约只留两个方法 | `model/attention/*` | 后端模块只暴露 `BACKEND`、`name`、`prefill`/`decode` |
| A3 | 删 `[engine].decode` 旋钮与 `decode_mode` provenance；`attn` 由常量 `provenance()` 给出 | `bench/run.py`、`bench/engines/minillm.py` | `[engine]` 只剩 impl/device/max_batch |

> provenance **不删**：它从「可选旋钮的记录」变成「常量的记录」，保护的是「以后有人静默换 kernel、
> run 之间还按老规矩比」。

### B · slab

| # | 内容 | 验收 |
|---|---|---|
| B1 | `_slab_bucket` 改 64 列粒度、封顶 `max_position`、缓存上限 3（FIFO 淘汰） | 桶 2176 而不是 4096；`cache_mb` = 预留字节，写进 notes |
| B2 | **明确不做**按步收 L，登记 P-25 | PITFALLS 有这一条 |
| B3 | `KvSlab.fill_from(cache, layer, row=None)`：按席位写入，`pos[row]` 独立 | CPU 单测：两行不同长度各填各的 |
| B4 | `kiv`/`v` 的 `write` 已经是按行 scatter，`pos` 是 `(B,)`，批 decode 只需要 engine 层循环 | `decode_slab(tokens: list[int], slab) -> (B, V)` |
| B5 | **删掉「为 graph 准备」的每步记账**（2026-10-31 补，实测来源）：bias 提到 `build_mask` 里每步一次（`slab.bias` + `hand_baddbmm.decode(bias=...)`）、`_columns` 改成 `pos` 的广播视图、`check_bounds` 用 Python 镜像 `_host_pos/_host_valid` 做越界判断（`"strict"` 仍读张量，`False` 留给 graph） | B=1 从 25.3 → 21–23 ms；每步 D2H 同步从 ~30 次降到 0 |

### C · 删代码

| # | 内容 | 处置 |
|---|---|---|
| C1 | `decode="slab_graph"` 分支、`MODES`、`_graph_decoder`、`_decoders` | 删；`serve/graph.py` 保留并标注 parked（W4 接线） |
| C2 | `model/attention/hand_legacy.py` | **搬到 `tests/reference/hand_legacy.py`**，从运行路径消失，但保留为独立数值 oracle |

> C2 的理由：golden 只覆盖一条固定路径；两条路径**同向出错**时 golden 查不出，oracle 能。
> 搬走后 `model/` 里确实不再有旧实现。

### D · 引擎

| # | 内容 |
|---|---|
| D1 | `generate()` = prefill(sdpa) → `fill_from` → 循环 `decode_slab`；`generate_cat` 删除，`decode(token, cache)` 降级为「过渡版/测试参考」 |
| D2 | `decode_slab` 支持 B ≥ 1，一次 `model(...)` 覆盖所有行（RoPE 按 `slab.pos` 逐行 gather） |
| D3 | TTFT 覆盖「prefill + 交接拷贝」，TPOT 只覆盖 decode 循环（口径不变） |

### E · 测试与文档

| # | 内容 |
|---|---|
| E1 | `test_attention_*` 去 Caps/注册表；oracle 改为 `tests.reference.hand_legacy`；`hand_baddbmm.prefill` 必须 raise |
| E2 | `test_slab_decode` 去 `attn_decode` 参数化（内核已固定），保留「slab vs cat 容器等价」 |
| E3 | `test_decode_modes.py` 改为 `test_engine_decode_path.py`：engine 输出 == 过渡版 `generate`（cat 容器）逐 id；`cache_mb` 语义按预留桶更新 |
| E4 | `test_thresholds`：协议键拆两层（通用 / 机制）；`decode_speed_vs_hf_min` 只看通用键 |
| E5 | `MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py` 必须绿——定死路径后它**就是**新 decode 路径的数值门 |
| E6 | `docs/decisions.md` D4、`PITFALLS.md` P-25/P-26、`docs/w3-plan.md` 口径段改写、`ROADMAP.md` §12/§13/§14 |

### F · run（4090）

> 代码已落地、门禁已绿，但 **`W2-0004` 尚未跑**（2026-10-31 由用户决定延到后续会话）。

| # | run | 说明 |
|---|---|---|
| F1 | **`W2-0004`** `fixed-attn-path` | B=1、32 prompt、`supersedes="W2-0003"`、`[baseline]="W2-0001"`（HF 臂）、复用 W2 四条判据 |
| F2 | ~~`W2-0005` decode-kernel~~ | **取消**：`W2-0005` 想回答的问题已由 `scripts/probe_decode_paths.py` 的探针答掉（§4.1），不必再开一个 run。探针不进报告、不进 results/ |
| F3 | `W3-0001` 的 `[baseline]` 改 `W2-0004` | 它此时才真的是「重构无害」 |

---

## 3. `W2-0004` 配置规格

```toml
schema = 1
[exp]
id = "W2-0004"
stage = "W2"
type = "microbench"
slug = "fixed-attn-path"
title = "固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab"
hypothesis = ["one_attention_path_does_not_move_w2_numbers"]
supersedes = "W2-0003"
seed = 0

[[models]]
name = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
dtype = "bfloat16"
trust_remote_code = false

[workload]
kind = "prompts"
prompts = "bench/prompts.json"
max_new_tokens = 128
batch_sizes = [1]
arrival = "closed"

[engine]
impl = "minillm"
device = "cuda"
max_batch = 1

[measure]
warmup = 3
repeats = 5
timer = "auto"
memory = ["torch_peak", "driver"]

[baseline]
run_id = "W2-0001"

[accept]
mode = "threshold"
criteria = ["ttft_cv_max", "tpot_cv_max", "memory_match_ratio", "decode_speed_vs_hf_min"]
```

---

## 4. 验收与预期（预注册）

| # | 判据 | 目标 / 失败线 |
|---|---|---|
| V1 | 四条 W2 判据（对 HF 臂） | 全 pass；`decode_speed_vs_hf_min ≥ 0.80` |
| V2 | 与 `W2-0003` 的 TPOT 差 | **记录，不判**。单请求也要读满桶列 + 一次 `fill_from`，预测 **≤ +10%**；超 15% 记负结果，**不为 B=1 开 cat 分支** |
| V3 | `memory_match_ratio` | 口径变了：`kv_mb` 现在是**预留**字节（桶粒度 64）。目标值不动，若因此跌破 0.90 就如实报。另：`driver_mem_mb` 已从 `nvidia-smi` 子进程（按 pid）换成 `torch.cuda.mem_get_info()`（整卡，P-27），与 `W2-0003` 的 1716 MB 同轴但不再按进程过滤——独占 GPU 时两者可比，notes 里注明换了来源 |
| V4 | 逐 prompt 输出 | `W2-0004` vs `W2-0003` 的 token 序列**不要求全等**（跨 kernel），差异条数写进 notes |
| V5 | `test_align_qwen3`（贪心 64 token） | 全等——这是新路径的数值门 |

### 4.1 V2 的实测（2026-10-31，探针；不进报告）

代码落地后 B=1 的 decode step 一度比 `W2-0003` 慢 **25%**，拆开后是四项 host 记账（每层重建
additive bias 1.95 ms/step、`check_bounds` 每步约 30 次 D2H 同步、`write` 每层复制列索引 2.29 ms/step、
`build_mask`/`advance` 0.11 ms），而 28 层的 attention 数学本身只有 0.17 ms。修掉记账后：

`scripts/probe_decode_paths.py`（teacher forcing，CUDA events，warmup 10 步排除，80 步）：
(a) `sdpa + KVcache(cat)` = `W2-0003` 路径；(b) `hand_baddbmm + KVcache(cat)` 只换 kernel、无 mask；
(c) `hand_baddbmm + KvSlab` = 落地路径。

| ctx | B | (a) | (c) | (c)/(a) | kernel 贡献 (b)/(a) | slab 记账 (c)/(b) |
|---|---|---|---|---|---|---|
| 100 | 1 | 19.55 | 22.92 | **1.172** | 1.038 | 1.130 |
| 512 | 1 | 19.37 | 21.59 | **1.114** | 1.055 | 1.057 |
| 2048 | 1 | 19.67 | 21.47 | **1.092** | 1.039 | 1.051 |
| 512 | 8 | 19.91 | 21.83 | **1.096** | 1.056 | 1.038 |

**判定**：残差不是单一的，也不是 bug——

- **kernel 4–6%**：`hand_baddbmm` 在 B=1 走两条独立 kernel（score `baddbmm` + `bmm`），而 SDPA 在无 mask
  的单请求下能选到融合后端。这是形状的性质，改实现改不掉。
- **slab 记账 4–13%**：已从 25% 降到这水平；短 ctx 那格偏高是因为固定开销（一步的 mask/bias/guard）
  摊在更少的历史列上，与预算无关。

两者都记为**「单一代码路径的税」**，按预注册的处置**不优化、不为 B=1 开 `cat` 分支**。判据余量仍在：
1.17 × 15.08 ms ≈ 17.6 ms 对 HF 的 16.78 ms ≈ **0.95×**，高于 0.80 的失败线。

**注意 (b) 的定位**：它永远不会出现在引擎里（D4 把 decode 定死为 hand+slab），只是这一次的归因工具，
脚本里条件执行（残差 ≤ 5% 时跳过），且不往 `model/` 加任何为它服务的代码。

### 4.2 落地后的门禁（2026-10-31）

| 门禁 | 结果 |
|---|---|
| `pytest -m "not slow and not hf"` | 184 passed |
| `pytest -m "gpu and not hf"` | 4 passed |
| `MINILLM_RUN_HF=1 pytest tests/integration/test_align_qwen3.py` | 4 passed |
| 新 `generate()` 路径 vs W1 golden 贪心 64 token | **4/4 逐 id 全等** |

---

## 5. 风险

| # | 风险 | 触发 | 处置 |
|---|---|---|---|
| R1 | `hand_baddbmm` 在 B=1 明显慢于 SDPA（派发主导） | V2 超 15% | 记录为「单一代码路径的税」；不按形状分叉（那会重新引入旋钮）。**已实测 1.09–1.17×，见 §4.1** |
| R2 | 跨 kernel 后 greedy 与 W2-0003 不一致 | V4 有差异 | 正常，只要求 golden 全等（V5）。**已实测：teacher forcing 下 logits 相对差 1.3–2.1e-2，部分 prompt 第 0–5 步 argmax 分叉** |
| R3 | 桶粒度改小后 slab 重建频繁、首步变慢 | W2-0004 的 TTFT p95 抬升 | 预热覆盖（warmup=3），桶缓存上限提到 4；不改回 2 倍翻 |
| R4 | `memory_match_ratio` 因预留字节下滑 | V3 | 如实报；W4 分页是根治手段 |
| R5 | 把「归因探针」当成优化任务，越拆越细 | 探针迭代超过 1 轮 | 这类探针的产出只用于**二选一**（记负结果 vs 修一处）；不追毫秒级分解。`scripts/probe_decode_paths.py` 已内置「残差 ≤ 5% 时跳过 (b)」 |
