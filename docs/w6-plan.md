# W6 计划 · 混合层实现（二）：Gated DeltaNet

> **状态**：**active**（2026-11-09 创建；2026-11-14 冻结范围并开工）。§2/§12 已按推荐裁定，
> 当前从 Step 0 执行；Step 1 起由 subagent 分步实现，主 agent 逐门禁验收。
> **权威**：阶段目标与判据以 `ROADMAP.md` §5 W6 为准；本文件只展开步骤、门禁、证据与生命周期。
> 若本文件与 ROADMAP 冲突，以 ROADMAP 为准，并在 §13 记录修订。
> **生命周期**：本文件是**临时执行计划**。W6 收口、阻塞或放弃时**删除**；不保留"历史 plan"。
> 持久结论只进：`docs/decisions.md` **D8**、`PITFALLS.md`、`ROADMAP.md` §5/§12/§13、
> `docs/hybrid_attention.md`、`tests/golden/qwen35-0-8b/0002/`、
> `configs/thresholds.toml [W6.gdn]`、`scripts/make_golden_qwen35_w6.py` 与新增测试。
> **范围**：Qwen3.5-0.8B 的 **text decoder**（D2），**单序列**，fp32 / CPU 数值门。
> **不做**：调度器/StatePool（W7）、packed/varlen GDN prefill（已定 W7，见 §12 第 1 条）、
> `KvSlab`/`PagedKV` 接入、padding/attention_mask、bf16/GPU 性能结论、Triton GDN kernel、
> 视觉/MTP、4B 载体、任何 `results/` run。

---

## 0. 一页纸

- **现状**：W5 已完成——`model/qwen35.py` 有全注意力 block（24 层中的 6 层）+ 两种 norm + partial-RoPE，
  `tests/golden/qwen35-0-8b/0001` 的 9 个 stage 逐项 `max|Δ|=0`；快速门禁 **712 passed**。
  0.8B 仍不能整模型加载：18 个 `linear_attn` 层没有实现，loader 明确报 W6 gap。
- **目标**：补齐 5 类算子中的最后 3 类（短卷积、GDN decode 循环、GDN 分块 prefill），
  装配 24 层 3:1 混合模型，并让 **分块 prefill == 逐步 decode**、**全模型贪心 64 token == HF**。
- **成功定义**（A1–A6，缺一不算 ✅）：
  - **A1** 卷积 / 循环 / 分块三个算子与**两套独立纯 torch 参考**（HF fallback + fla naive）对拍过；
  - **A2** 分块 prefill 与逐步 decode 的输出、conv_state、recurrent_state 在 chunk 边界逐 token 一致；
  - **A3** 单层 GDN（L0/L2 × s01/m02）对 HF golden 的 stage 表过；
  - **A4** 全 24 层模型加载 0.8B text decoder，贪心 **64/64** 与 HF 相同（硬线；near-tie 翻转只改变归因，**不放宽 A4**：翻转即判失败，按 §11.5 部分收口），末位 logits 在容差内；
  - **A5** 快速门禁保持绿；0.6B 的 64/64 与 W5 的 9-stage 0 误差不变；
  - **A6** W7 交接：状态形状/字节数、`Qwen35State` 接口、命令与已知边界写进 D8 与 `hybrid_attention.md`。
- **时间盒**：**8–10 个工作日**（计划窗口 2026-11-16 → 2026-11-27；2026-11-14 复核时仍未开工，
  旧窗口 11-20 已不可行）。
  Step 0 0.5 天；Step 1–2 算子 3 天；Step 3 单层 golden 1.5 天；Step 4–5 全模型 2.5 天；
  Step 6 收口/缓冲 1–1.5 天；合计约 8.5–9.5 天，留 0.5–1.5 天缓冲到 10 天。
- **止损**（ROADMAP §8）：分块形式在 Step 1 结束仍不正确 → **生产 prefill 走逐步循环**，
  记录性能代价，不硬啃；全模型在 Step 5 仍不过 → 以"单层 GDN + 状态接口"部分收口，
  W6 标 `blocked`/部分完成，不伪造 ✅，W7 第一件事是修全模型。

---

## 1. 现状核查（2026-11-14）

### 1.1 已完成的证据

| 项 | 值 / 命令 | 说明 |
|---|---|---|
| HEAD | `f59bbaf feat(w5): Qwen3.5 full-attention layer + single-layer HF golden` | 代码树干净（本计划文件未跟踪） |
| 快速门禁 | `712 passed, 14 deselected in 12.02s`（2026-11-14 复核） | `python -m pytest -m "not slow and not hf" -o addopts="" -q` |
| W5 HF 门禁 | `2 passed in 3.24s`（2026-11-14 复核） | `MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen35_layer.py -q` |
| run 契约 | `✓ run contract OK` | `python scripts/exp_index.py --check` |
| prompt 集 | `32 prompts, sha256=7548f3a6c7e6…` | `python bench/make_prompts.py --check` |
| W5 golden | `tests/golden/qwen35-0-8b/0001`（3.1 MB） | L3/L23 × s01/m01，9 stage 全 0 |
| 0.6B 回归 | 64/64、`max abs(Δ)` 2.0e-5 | W1，未受 W5 影响 |

### 1.2 0.8B 的架构事实（`config.get_text_config()` 直读）

| 字段 | 值 | W6 的影响 |
|---|---|---|
| `num_hidden_layers` | 24 | 18 linear + 6 full |
| `layer_types` | `LLLF × 6` | 每 4 层一个全注意力；L0–L2 为 linear，L3 为 full |
| `hidden_size` / `intermediate_size` | 1024 / 3584 | MLP 复用 `model/layers.py::MLP` |
| `head_dim` / `num_attention_heads` / `num_key_value_heads` | 256 / 8 / 2 | W5 已覆盖 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 16 | 0.8B 无需 GQA repeat；代码保留 ratio>1 分支 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | 状态 `[K,V] = [128,128]` |
| `linear_conv_kernel_dim` | 4 | 我们冻结 `conv_state` 长度 = **3**（kernel−1，最小充分状态）；HF 5.20.2 的 cache 实际存 4，见 P-49 |
| `hidden_act` | `silu` | 只支持 silu，其他值显式报错 |
| `attn_output_gate` / `partial_rotary_factor` | true / 0.25 | W5 已覆盖 |
| `tie_word_embeddings` | true | checkpoint 无 `lm_head.weight`，加载时从 embedding 填 |
| `mtp_num_hidden_layers` | 1 | D2：忽略 |
| 视觉塔 | 12 层 | D2：忽略 |

### 1.3 缺什么

| 缺口 | 位置 | 现状 |
|---|---|---|
| 短因果卷积（prefill + decode 状态） | 无 | 未实现 |
| GDN decode 循环 | 无 | 未实现 |
| GDN 分块 prefill（WY 表示） | 无 | 未实现 |
| 24 层混合模型装配 + 逐请求状态 | 无 | `MiniLLM` 只支持全注意力 |
| 0.8B 整模型加载 | `model/loading.py` | `load_layer_weights` 拒绝 linear 层；`load_checkpoint` 对 hybrid 报 gap |

### 1.4 环境与参考实现

| 项 | 值 |
|---|---|
| `transformers` | 5.20.2 |
| `fla` | 0.5.2（`fla/ops/gated_delta_rule/naive.py`，纯 torch） |
| `causal_conv1d` | 1.7.0 |
| torch / CUDA | 2.8.0+cu128 / 4090 24 GB |
| 0.8B snapshot | `2fc06364715b967f1860aea9cf38778875588b17`（本地缓存） |
| HF GDN 参考 | `transformers/models/qwen3_5/modeling_qwen3_5.py`：`torch_chunk_gated_delta_rule`(L287)、`torch_recurrent_gated_delta_rule`(L424)、`Qwen3_5GatedDeltaNet`(L490) |
| fla 参考 | `naive_recurrent_gated_delta_rule` / `naive_chunk_gated_delta_rule` |

**已做的 G0 预演（不是正式数字，只用于确认门禁量级）**：

- 随机 fp32、`H=2,K=V=32,C=64`（q/k 已 l2norm），T ∈ {1, 63, 64, 65, 127, 128, 129, 1024}：
  HF chunk vs HF recurrent 的 `max|Δ|` = **5.6e-9 … 1.0e-7**，final state = **0 … 8.3e-7**；
  fla naive recurrent vs HF recurrent = **4.5e-8 … 6.0e-8**；fla chunk vs HF chunk = **0 … 7.5e-8**。
  → `[W6.gdn]` 的 `operator_atol=1e-5` 对**输出**有约 100× 余量、对 **final state** 有约 12× 余量；
  这是预注册值，不是拟合值。
- **未归一化 k 的稳定性预演**（H=2、K=V=32、T=65）：对 q/k 乘随机幅度 scale 后不做 l2norm，
  scale=0.1/0.5 的 chunk vs recurrent 差 2.6e-8/8.9e-4，scale=1.0 已差 **6.1e19**，
  scale=2.0 直接 NaN。→ q/k l2norm 不是可选项；G2 的 fla/HF 对拍必须使用归一化后的 q/k
  （我们的算子传原始 q/k，内部归一化）；W6 不对未归一化 k 做跨端点对拍（P-58）。
- **未钉 kernel 时**，HF 的 `torch_recurrent_gated_delta_rule` 已被解析成 fla 的 Triton kernel，
  CPU 上直接报 `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`。
  `tests/conftest.py` 目前只钉了 3 个符号（P-15），W6 必须补第 4 个（见 P-48）。
- HF 全模型 fp32 CPU 冒烟（7-token prompt，16 threads，钉 4 个符号）：
  加载 1.5 s、prefill 0.2 s、4 个 decode step 0.2 s；`DynamicCache` 的 linear 层类型为
  `LinearAttentionLayer`，conv/recurrent 状态可读。→ golden 生成在 CPU 上可行，不需要 GPU。

---

## 2. 范围裁定（2026-11-14 冻结）

> 以下裁定已按用户 2026-11-14 的"按推荐"确认冻结；转 `active` 时原样写进 D8。
> §12 的六项也已裁定，不再等待确认。若与 ROADMAP 的措辞冲突，在 §13 记一笔
> （例如 W4.5 段"B 的 packed prefill 在 W6 接同一接口"）。

| ID | 裁定 | 理由 |
|---|---|---|
| D-W6-1 | 新写 **`model/qwen35_gdn.py`** 与 **`model/qwen35_model.py`**，不在 `model/layers.py` 上加开关，不改 0.6B 路径 | W5 已确立"新模块承载 Qwen3.5"；D4 约束的是**全注意力层的 prefill/decode 后端选择**（不注册、不回退），不是禁止新增混合层；把 GDN 塞进 `layers.py` 会让 0.6B 路径承担 18 层无用状态 |
| D-W6-2 | `Qwen35State` 是**唯一的每请求状态接口**：全注意力层用 `KVcache`，线性层用 `conv_state` + `recurrent_state`；层代码只调用接口，不自己分配 | W7 的 `StatePool` 替换的是**存储/生命周期**，不是层代码；现在冻结接口，W7 才能只换 allocator |
| D-W6-3 | **W6 只做单序列数值门**；packed/varlen GDN prefill、调度器/`StatePool`、`KvSlab`/`PagedKV` 全部留 W7 | 这三件事都需要"每请求状态句柄"的分配与生命周期，正是 W7 的 `StatePool`；W6 先锁数学，避免把算法 bug 和状态管理 bug 混在一起 |
| D-W6-4 | 证据 = **golden + pytest**，不新开 `results/` run | 与 W1/W5 同型；W6 没有性能问题要回答，runner/threshold evaluator 反而是额外风险 |
| D-W6-5 | 参考路径 **fp32 / CPU**；bf16/GPU 只做冒烟，不出性能数字 | golden 要跨机器可复现（W1/W5 口径）；fp32 下 chunk vs recurrent 的误差已被 G0 预演量化 |
| D-W6-6 | `chunk_size = 64` 固定；"分块 prefill"是**一次 forward 内的算法**，不是跨调用的 partial prefill | 与 HF/fla 默认一致；跨调用 chunked prefill 需要 offset 状态与调度，留 W7 |
| D-W6-7 | 一个 fixture **`tests/golden/qwen35-0-8b/0002`**，同时装 GDN 单层 stage 与全模型 greedy/logits | 同一 revision、同一 prompt、同一次生成；拆成两个 fixture 只会多一次用户生成与审查 |
| D-W6-8 | 写 **D8**（proposed → accepted），并把 W5 遗留的 `D-W5-1`/`D-W5-2` 引用收编 | `model/qwen35.py` 引用了这两个 ID，但 `docs/decisions.md` 没有对应条目；W6 收口时不能继续留悬空引用 |

**明确不做**：packed/varlen GDN prefill（默认）、`serve/` 改动、bench 适配器、`results/W6/`、
`configs/experiments/W6/`、bf16 性能、Triton GDN kernel、视觉/MTP、4B、quantization。

---

## 3. 门禁

### G0 · 基线与环境（Step 0，0.5 天）

- 快速门禁 712 passed；W5 HF 门禁 2 passed；`exp_index --check` / `make_prompts --check` 绿。
- `tests/conftest.py` 与 golden 脚本都钉 **4 个**符号：
  `causal_conv1d_fn`、`causal_conv1d_update`、`torch_chunk_gated_delta_rule`、`torch_recurrent_gated_delta_rule`。
- 本地 0.8B snapshot 可按 revision 解析；`fla` / `causal_conv1d` 可导入。
  若 `fla` 不可导入：把 `naive.py` 冻结到 `tests/reference/gdn.py`（保留 MIT 头，只被测试导入），
  不跳过第二套参考。
- **不过**：先修环境，不写模型代码。

### G1 · 卷积算子（Step 1）

- 因果性：改第 t 个 token，位置 < t 的输出 `torch.equal`。
- 状态等价：整段 prefill 的 `conv_state` == 逐 token decode 后的 `conv_state`；两者输出一致。
- `conv_state` 形状 `[B, conv_dim, 3]`，是**卷积前**的输入列，不是 SiLU 后的输出。
- **不过**：先修 conv，不进 GDN。

### G2 · GDN 循环与分块（Step 1，硬门禁）

- 随机形状（B=1/2、T=1/63/64/65/127/128/129/257/1024、H=1/16、K=V=16/128）：
  - 循环 vs fla `naive_recurrent`：`atol=rtol=1e-5`；
  - 分块 vs 循环：输出与 final state 均 `atol=rtol=1e-5`；
  - 分块 vs HF `torch_chunk_gated_delta_rule`：同上；
  - 非零 `initial_state`、GQA ratio=2 各一组。
- **归一化契约**：我们的算子接原始 q/k，在算子内做 l2norm + q scale；fla naive 不做 l2norm，
  测试端先按其契约手动归一化 q/k。调 fla 时**只能选一种 q scale 口径**：推荐 q 不预乘、
  给 fla 传 `scale=K**-0.5`；或者 q 预乘 `K**-0.5`、给 fla 传 `scale=1.0`；两者都做会二次缩放。
  HF `torch_chunk_gated_delta_rule` 用 `use_qk_l2norm_in_kernel=True` 接原始 q/k。
- **快速门禁只跑小形状**（H≤2、K=V≤32、T≤257），避免把 10 s 门禁拖成分钟级；
  `T=1024 / H=16 / K=V=128` 的完整对拍（尤其 fla 的逐步循环）标 `slow`，在 Step 1 收口前至少跑一次。
- **不过**：按 §9 止损。分块失败不阻塞循环路径，但 W6 不能标 ✅。

### G3 · 单层 GDN 对 HF golden（Step 3 dry run；Step 5 正式）

- L0/L2 × s01/m02 的 stage 表全过 `[W6.gdn]` 的 rel 线；
- Step 3 先用 `/tmp` 草稿对拍；正式 G3 必须等 Step 5 的 `0002` 提交后跑（一个 fixture 同时含单层与全模型，D-W6-7）；
- `final_state`（m02）× 2 层在 `state_atol` 内；
- `layer_out` 与 block 输出一致；因果性测试在层级别通过。
- **不过**：先定位第一个错的 stage，不重生成 golden、不改容差。

### G4 · 全模型对 HF（Step 5，最终门禁）

- 24 层加载成功；末位 logits `max|Δ| ≤ full_logits_max_abs`；
- 贪心 **64/64** 与 HF 相同；若翻转，记录该步 top-2 margin，按 §9 处理；
- 逐层 hidden（24 层 × 3 位置）在 `layer_hidden_max_abs` 内；
- 全模型 prefill vs 逐 token decode（接 `Qwen35State`）在 `decode_prefill_atol` 内；
- 0.6B W1 64/64 与 W5 9-stage 0 误差不变。
- **不过**：W6 部分收口，W7 第一件事修全模型；不伪造 ✅。

---

## 4. 步骤与文件

| Step | 天 | 内容 | 产物 / 门禁 |
|---|---|---|---|
| 0 | 0.5 | 按 §2/§12 冻结口径启动；ROADMAP §13 记 v29、§12 W6 → 🟡；写 D8 proposed；补 conftest 第 4 个 kernel pin；记录基线 | G0 |
| 1 | 2.0 | `model/qwen35_gdn.py`：`causal_conv_prefill/decode`、`gated_delta_recurrent`、`gated_delta_chunked`、`l2norm`；单测对 HF + fla | G1/G2 |
| 2 | 1.0 | `Qwen35GDN` 模块 + `Qwen35GDNBlock`；`load_layer_weights` 支持 linear 层；层单测（因果、decode==prefill、state） | 层单测绿 |
| 3 | 1.5 | `scripts/make_golden_qwen35_w6.py` 的 GDN 部分；agent dry run 到 `/tmp`；`tests/integration/test_align_qwen35_gdn.py`（正式 G3 等 Step 5 的 `0002`） | G3（dry run） |
| 4 | 1.5 | `model/qwen35_model.py`：`Qwen35State` + `Qwen35Model`（LLLF 调度、RoPE 表、tied lm_head）；`load_qwen35_model`；模型单测 | prefill==decode |
| 5 | 1.0 | 同一 golden 脚本的全模型部分（logits/greedy/layer hidden）；agent 重新 dry run 到 `/tmp`；`test_align_qwen35_model.py`；用户生成并提交 `0002`（唯一一次，含单层 + 全模型） | G4；正式 G3+G4 |
| 6 | 0.5–1.0 | 收口：阈值 calibrated、D8 accepted、ROADMAP/PITFALLS/hybrid_attention/README、删本文件、全门禁；缓冲 | 收口清单 |

### 新增/修改文件

| 文件 | 动作 | 用途 |
|---|---|---|
| `model/qwen35_gdn.py` | 新增 | 3 个算子 + `Qwen35GDN` + `Qwen35GDNBlock` |
| `model/qwen35_model.py` | 新增 | `Qwen35State` + `Qwen35Model` |
| `model/qwen35.py` | 改 docstring | 指向 W6 装配；不改 W5 数值代码 |
| `model/loading.py` | 扩展 | linear 层 shape/加载；`load_qwen35_model`；hybrid 的显式入口 |
| `tests/conftest.py` | 改 | 第 4 个 kernel pin（P-48） |
| `tests/unit/test_qwen35_gdn.py` | 新增 | 算子对拍、边界、状态、GQA |
| `tests/unit/test_qwen35_gdn_layer.py` | 新增 | 模块/block 因果、decode==prefill |
| `tests/unit/test_qwen35_model.py` | 新增 | LLLF 调度、状态生命周期、模型级一致性 |
| `tests/unit/test_qwen35_loading.py` | 改 | W6 的整模型成功路径；保留 shape 错误护栏 |
| `tests/unit/test_thresholds_w6.py` | 新增 | 预注册值不被松动 |
| `tests/integration/test_align_qwen35_gdn.py` | 新增 | 单层 stage 表（slow/hf） |
| `tests/integration/test_align_qwen35_model.py` | 新增 | 全模型 logits/贪心（slow/hf） |
| `scripts/make_golden_qwen35_w6.py` | 新增 | 唯一 fixture 生成器 |
| `configs/thresholds.toml` | 加段 | `[W6.gdn]` |
| `docs/decisions.md` | 加段 | D8 |
| `PITFALLS.md` / `ROADMAP.md` / `docs/hybrid_attention.md` / `tests/golden/README.md` / `README.md` | 收口更新 | 持久结论与状态 |

---

## 5. 机制设计（写进 D8）

### 5.1 短卷积

- **prefill**（T≥1，要求 `past_len == 0`）：`conv_state` 从零开始；
  `F.conv1d(x, weight, bias=None, padding=k-1, groups=conv_dim)[:, :, :T]` → SiLU；
  `conv_state = cat([zeros, x], -1)[:, :, -(k-1):]`（**卷积前**的输入，T<k−1 时左补零）。
- **decode**（T≥1，带状态）：
  `new = cat([conv_state, x], -1)`；`out = F.conv1d(new, weight, padding=0, groups=conv_dim)[:, :, -T:]`；
  `conv_state = new[:, :, -(k-1):]`；SiLU。
- 0.8B：`conv_dim = 2*16*128 + 16*128 = 6144`，`k=4`，我们的状态 `[B,6144,3]`。
  **HF 5.20.2 的 `DynamicCache.conv_states` 实际存最后 4 列**（`kernel_size`，不是 `kernel−1`）；
  多出的最旧一列不进入"下一个 token"的最后一个卷积窗，所以两种长度对输出等价。
  W6 冻结为 **3**（最小充分状态）；与 HF golden 比较时只取 HF 状态的 `[..., -3:]`（P-49）。
- 模型层：`past_len > 0` 且 `T > 1` 直接 raise（跨调用 chunked prefill 留 W7）；
  算子单测可以用 decode 路径逐 token 推进，验证与 prefill 的状态/输出一致。
- 只支持 `hidden_act="silu"`；其他值 raise。

### 5.2 GDN 循环（decode 金标准）

输入 `q,k [B,T,H,K]`、`v [B,T,H,V]`、`beta [B,T,H]`（已 sigmoid）、`g [B,T,H]`（log-space ≤0）；
状态 `h [B,H,K,V]`；所有计算 fp32。**q 和 k 都在算子入口做 fp32 l2norm（eps=1e-6），然后只对 q 乘 `K^-0.5`**；
layer 不再重复归一化（§5.4）：

```python
h = h * g.exp()[..., None, None]
kv_mem = (h * k[..., None]).sum(-2)
delta = (v - kv_mem) * beta[..., None]
h = h + k.unsqueeze(-1) * delta.unsqueeze(-2)
o = (h * q[..., None]).sum(-2)
```

- 上面是第 i 步的切片公式；循环里写 `g[:, i]`、`k[:, i]`、`v[:, i]`、`beta[:, i]`、`q[:, i]`，
  不要拿整段 `[B,T,H]` 直接广播。
- 输出 cast 回输入 dtype；**state 保持 fp32**（HF 行为）。
- 0.8B ratio=1；`num_v_heads // num_k_heads > 1` 时 q/k `repeat_interleave`（保留备用 + 单测）。

### 5.3 GDN 分块 prefill（WY 表示）

按 HF `torch_chunk_gated_delta_rule` 的**同一套公式**实现（Apache-2.0，注释保留来源）：

- `pad_size = (C - T % C) % C`；q/k/v 右 pad，beta/g 补 0；输出只取 `[:T]`。
- 块内：`k_beta = k * beta`、`v_beta = v * beta`；`cum_decay = g.cumsum(-1)`；`pairwise_decay = exp(cum_i - cum_j)`（严格上三角 mask 为 -inf）；
  这里 `cum_i` 是位置 i 的块内累计 decay，`cum_last` 是块末累计 decay。
  `ut_system = (k_beta @ k^T) * pairwise_decay`；`intra = (q @ k^T) * pairwise_decay`
  （q/k 已按 §5.2 归一化，q 已乘 `K^-0.5`）；
  `decayed_k_beta = k_beta * exp(cum)`；
  `new_values = solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)`；
  `k_cumdecay = solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)`。
  `ut_system` 的对角线被 `unitriangular=True` 忽略：求解时按单位三角系统处理（对角视为 1，
  不来自 `beta·‖k‖²`），这是 HF 的 WY 写法；分块与循环在**代数上**对任意 k 等价。但 WY 三角求解的数值条件随
  `‖k‖`（以及 beta）恶化：G0 预演里未归一化 k 在 scale=1.0 时 chunk/recurrent 已差到 O(1e19)，
  scale=2.0 直接 NaN；归一化后回到 ~1e-7。**HF 的 GDN 定义包含 q/k l2norm**，两个端点必须用同一份
  归一化后的 q/k；W6 不对未归一化 k 做跨端点对拍。
- 块间：`v_new = new_values_i - k_cumdecay_i @ S`；
  `o_i = (q_i * exp(cum_i)) @ S + intra_i @ v_new`；
  `S = S * exp(cum_last) + key_decayed_i^T @ v_new`，其中 `key_decayed_i = k_i * exp(cum_last - cum_i)`。
- **两个 q 不能混用**：`intra` 用未乘 `exp(cum_i)` 的 q；块间 `o_i` 用乘了 `exp(cum_i)` 的 q
  （HF 在 chunk 循环外先 `query *= exp(cum_decay)`）。
- `chunk_size=64`；`solve_triangular` 是主路径；fla 的逐步前代法只作 oracle，**不做生产回退**（D4 精神）。
- 用 `torch.linalg.solve_triangular` 而不是照抄 fla 的循环，正是为了让两套参考在数值上真正独立。
- **分派看 `past_len`，并区分 T**：`past_len == 0` 走 chunk（prefill，含 T=1 的 1-token prompt）；
  `past_len > 0` 且 `T == 1` 走 recurrent（decode）；`past_len > 0` 且 `T > 1` 在模型层 raise
  （跨调用 chunked prefill 留 W7）。HF 的 decode 分支判断是
  `use_precomputed_states and seq_len == 1`；不要只看 `T == 1`，否则 1-token prompt 的 prefill
  会走错端点，golden 对拍会多一个不必要的差异源。

### 5.4 层装配

`Qwen35GDNBlock`：

```
x + linear_attn(input_layernorm(x)) ; x + mlp(post_attention_layernorm(x))
```

`Qwen35GDN`：`in_proj_qkv` → conv → split(q,k,v)；并行的 `in_proj_z`（输出门控）与
`in_proj_b/a`（beta/g）；**原始 q/k 直接进 GDN（l2norm 与 q scale 只在算子内做一次，layer 不预归一化）**
→ `Qwen35RMSNormGated(core, z)` → `out_proj`。
权重是 **4 个独立投影**（`in_proj_qkv/z/b/a`），不是 Qwen3-Next 的 `qkvz/ba` 合并；z/b/a 不卷积。

### 5.5 全模型与状态接口

```python
class Qwen35State:
    position: int
    def past_len(self, layer: int = 0) -> int
    def get_kv(self, layer: int) -> tuple[Tensor, Tensor] | tuple[None, None]
    def set_kv(self, layer: int, kv) -> None
    def conv_state(self, layer: int) -> Tensor | None
    def set_conv_state(self, layer: int, state: Tensor) -> None
    def recurrent_state(self, layer: int) -> Tensor | None
    def set_recurrent_state(self, layer: int, state: Tensor) -> None
    def advance(self, n: int) -> None
    def cull(self, keep: Tensor) -> None
    def nbytes(self) -> int
    def linear_state_bytes(self) -> int
    def kv_bytes(self) -> int
```

- `Qwen35Model.forward(tokens, state=None, logits_to_keep=None) -> (logits, None)`；
  只接受单序列、无 mask、无 packed/slab；其他容器显式 `NotImplementedError`。
- 全注意力层复用 `Qwen35Attention` + `KVcache`；GDN 层复用上面的 block；final norm 用
  `Qwen35RMSNorm`（`(1+w)`）；`lm_head` 与 embedding **共享 Parameter**（否则 fp32 多占 1 GB）。
- RoPE 表由模型每步算一次传给所有层（P-18），GDN 层忽略。

### 5.6 状态字节数（W7 的解析式输入）

| 项 | 公式 | 0.8B fp32 | 0.8B bf16 模型（估算） |
|---|---|---|---|
| conv_state / linear 层 | `6144 × 3 × dtype` | 72 KiB | 36 KiB |
| recurrent_state / linear 层 | `16 × 128 × 128 × 4` | 1 MiB | 1 MiB（保持 fp32） |
| 18 个 linear 层 / 请求 | 18 × 上述两项 | **19.27 MiB** | **18.63 MiB** |
| KV / token（6 个 full 层） | `2 heads × 2(KV) × 256 × 6 × dtype` | 24 KiB | 12 KiB |
| KV @2048 / 请求 | 2048 × 上述 | 48 MiB | 24 MiB |

> 表按 W6 的 `conv_state=3` 算；HF cache 存 4 列，若直接对齐 HF 每个 linear 层会多 24 KiB（fp32），
> 18 层多 432 KiB。W7 若要直接用 HF kernel，再把契约改成 4；W6 不为这点内存放弃最小状态。

> 仅"线性状态 vs KV"两项的解析交叉点约 **1.6k token**（bf16 KV + fp32 状态）。
> 这是 W7 要实测的量级，**不是** W0 的 193-token 总显存交叉点，也不进 W6 判据。
> W6 的 `Qwen35State.nbytes()` 分项（`linear_state_bytes()` / `kv_bytes()`）把这张表锁住，
> W7 的 `[W7.state].state_estimate_error_max=0.20` 从这里起步。

---

## 6. 数值与 golden

### 6.1 fixture

`tests/golden/qwen35-0-8b/0002/`，`spec = "w6-gdn-v1"`，由用户生成并提交，测试只读。

| 范围 | prompt | 内容 |
|---|---|---|
| GDN 单层 | `s01`（36 tok）、`m02`（240 tok） | L0/L2：`layer_in`（全 T）、`input_layernorm`、`in_proj_qkv/z/b/a`、`conv_out`、`q_pre/k_pre`、`beta/g`、`core_attn_out`、`norm_out`、`layer_out`、`final_state`（m02）、final `conv_state`（m02，HF cache 的后 3 列） |
| 全模型 | `s01`、`m02` | `input_ids`、末位 logits（`248320`）、贪心 64 ids + 每步 top-2 margin、24 层 × 3 位置 hidden |

- 采样位置：`0 / T//2 / T-1`；`layer_in` 存全 T（测试要跑完整前向）；`final_state` 只存 m02
  （s01 不存，控制体积）。`q_pre/k_pre` 是 `in_proj_qkv` split 后的原始 q/k（算子入口，未 l2norm）；
  l2norm 在算子内做，避免与 §5.4 重复。

单层 stage 的形状与口径（`T` = 36 / 240；除 `layer_in`/终态外均为 3 个采样位置）：

| stage | shape | 口径 |
|---|---|---|
| `layer_in` | `(T,1024)` | 全 T；block 的残差输入 |
| `input_layernorm` | `(3,1024)` | `layer.input_layernorm` 输出 |
| `in_proj_qkv` / `in_proj_z` | `(3,6144)` / `(3,2048)` | 原始投影输出 |
| `in_proj_b` / `in_proj_a` | `(3,16)` | beta logits / decay logits |
| `conv_out` | `(3,6144)` | **post-SiLU**；由 `in_proj_qkv` + conv 权导出 |
| `q_pre` / `k_pre` | `(3,16,128)` | split 后、l2norm 前 |
| `beta` / `g` | `(3,16)` | beta 已 sigmoid；g 是 log-space ≤0 |
| `core_attn_out` | `(3,16,128)` | GDN 输出、gated norm 前 |
| `norm_out` | `(3,2048)` | `Qwen35RMSNormGated` 输出 |
| `layer_out` | `(3,1024)` | 残差 + MLP 后 |
| `final_state` | `(16,128,128)` | 仅 m02；fp32 |
| `final conv_state` | `(6144,3)` | 仅 m02；HF cache `[..., -3:]` |

- 体积估算：`layer_in` 2.3 MB + final state 2.1 MB + final conv 0.15 MB + 全模型 logits 2.0 MB +
  hidden 0.6 MB + sampled stages ≈ 1.2 MB ≈ **8.3 MB**（上限 10 MB）。除 `layer_in` 外所有 stage
  只存 3 个采样位置；`in_proj_qkv`/`conv_out`/`q_pre/k_pre` 等宽 stage 存全 T 会超 10 MB。
- 不存全词表 logits 的每一步；贪心由 `greedy_ids` 逐步判，logits 只存末位（同 W1）。
- 贪心 **不特殊处理 EOS**：固定走 64 步，避免"早停/不早停"两边不一致；这是数值门，不是聊天 demo。

### 6.2 生成方式

- 脚本 `scripts/make_golden_qwen35_w6.py`；复用 `make_golden_qwen35_layer.py` 的
  `cap_torch_threads`、`load_text_decoder`、`pin_torch_kernels`，**在新脚本里再补钉**
  `torch_recurrent_gated_delta_rule`（不改 W5 脚本的 3 符号口径）；
  模型 `fp32 / CPU / eager`，线程上限与测试一致（16）。
  生成 `0002` 前先提交 W6 代码与本计划文件，确保 `git status --porcelain` 为空。
  **但 `git_state` 必须在创建 `out_dir` 之前采样（或显式排除 `out_dir`）**：脚本写完后才跑
  `git status` 会把新建的 fixture 目录本身算成未跟踪 → `dirty=true`，无论代码多干净。
  `out_dir` 还必须**不存在**（不是"存在但为空"）；已存在的空目录同样会被算成未跟踪。
  W5 的 `0001/meta.json` 是 `source.commit=3e564f4`、`dirty=true`：当时 W5 代码尚未提交；
  即使代码干净，旧脚本的采样时点也会让新目录污染 dirty。W6 脚本在 `main()` 开头、建目录之前
  采样，并把结果写进 meta；`dirty` 必须为 false。
- 生成时**显式 pin revision**：`--revision 2fc06364715b967f1860aea9cf38778875588b17`
  （或 `--model-path` 指向该 snapshot），不要依赖 cache 的 `refs/main`；meta 必须记录这个 hash。
- 单层 stage 用 forward hook + 包装 `torch_chunk_gated_delta_rule`：
  记录进入算子的 `q/k/v/g/beta`、输出的 `core_attn_out` 与 `final_state`；
  `conv_out` 由 `in_proj_qkv` + `conv1d.weight` 用同一 `F.conv1d` 口径导出（**post-SiLU**；脚本里注明是派生量）。
- 18 个 linear 层共用同一个 rule 符号：给每个 `layer.linear_attn` 注册 forward_pre_hook，把
  `layer_idx` 写进模块级 `current_layer`；rule/conv wrapper 按 `current_layer` 记录，只保留
  L0/L2，其余丢弃。不要用调用顺序硬猜。
- 全模型用 `use_cache=True` + `DynamicCache` 手动贪心，`logits_to_keep=1`；
  读 `cache.layers[i].conv_states[0][..., -3:]`（HF 存 4 列，取我们的 3 列契约）与
  `recurrent_states[0]` 得到终态。
- 命令（**最终生成由用户执行**）：

```bash
# agent 只写 /tmp（dry run；用于出 stage 表）
python scripts/make_golden_qwen35_w6.py \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --out /tmp/w6-golden-dry

# 用户 review 后生成正式 fixture（tests/golden 不可静默重生成）
python scripts/make_golden_qwen35_w6.py \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --out tests/golden/qwen35-0-8b/0002
```

- 测试用 `MINILLM_W6_GOLDEN=/tmp/w6-golden-dry` 指向草稿；正式门禁必须指向已提交的 `0002`。

### 6.3 阈值（预注册，先冻结再测）

```toml
[W6.gdn]
status = "target"                      # 收口时改 calibrated，数值不动
basis_golden = "tests/golden/qwen35-0-8b/0002"
operator_atol = 1e-5                   # 循环/分块 vs 两套参考
operator_rtol = 1e-5
state_atol = 1e-5                      # final_state
conv_atol = 1e-6                       # conv 因果/状态等价
causal_atol = 0.0                      # fp32 CPU，必须 bit-identical
proj_rel_max = 1e-5                    # in_proj_* / q_pre / k_pre
conv_rel_max = 1e-5                    # conv_out
norm_rel_max = 1e-5                    # gated norm
gdn_out_rel_max = 1e-5                 # core_attn_out
layer_out_rel_max = 1e-4               # 残差 + MLP 后
layer_hidden_max_abs = 1e-2            # 全模型逐层 hidden
full_logits_max_abs = 0.05             # 末位 logits（同 W1）
greedy_tokens_exact = 64
decode_prefill_atol = 1e-4             # 全模型 prefill vs 逐步 decode
```

- **不得事后放宽**。G0 预演的 ~1e-7 说明 `operator_atol=1e-5` 对输出约 100×、对 final state 约 12× 余量；若正式值超线，先查实现，
  不先改阈值。确需改，走 D8 修订并在 `ROADMAP §13` 记录，且必须在正式 golden 之前。
- `tests/unit/test_thresholds_w6.py` 锁死上述值、`status`、`basis_golden` 与 fixture 存在性；
  **fixture 存在性只在 `status="calibrated"` 时断言**——Step 0–4 还没有 `0002`（Step 5 才提交），快速门禁必须保持绿。

---

## 7. 测试清单与命令

### 7.1 快速门禁（无权重）

| 文件 | 锁什么 |
|---|---|
| `tests/unit/test_qwen35_gdn.py` | conv 因果/状态；循环 vs fla；分块 vs 循环/HF；边界 T=C±1；非零初态；GQA ratio=2；l2norm；beta/g 公式；state 布局 |
| `tests/unit/test_qwen35_gdn_layer.py` | 层形状；gate norm；残差；因果性；prefill vs 逐步 decode；`Qwen35State` 生命周期 |
| `tests/unit/test_qwen35_model.py` | LLLF 调度；tiny 模型（含 24 层调度分支）；tied lm_head；`logits_to_keep`；prefill vs decode；cull；显式拒绝 packed/slab/mask |
| `tests/unit/test_qwen35_loading.py` | linear 层 shape；整模型加载成功；缺 key/错 shape 报错；0.6B 路径不变 |
| `tests/unit/test_thresholds_w6.py` | 预注册值 / status / fixture（仅 `calibrated` 后断言） |

```bash
python -m pytest -m "not slow and not hf" -q
python -m pytest tests/unit/test_qwen35_gdn.py tests/unit/test_qwen35_gdn_layer.py \
  tests/unit/test_qwen35_model.py tests/unit/test_qwen35_loading.py -q
```

### 7.2 慢门禁（本地 0.8B 权重）

```bash
# 单层 GDN stage 表（先对 /tmp 草稿）
MINILLM_RUN_HF=1 MINILLM_W6_GOLDEN=/tmp/w6-golden-dry \
  python -m pytest tests/integration/test_align_qwen35_gdn.py -q -s

# 全模型 logits / 贪心 / 逐层 hidden
MINILLM_RUN_HF=1 MINILLM_W6_GOLDEN=/tmp/w6-golden-dry \
  python -m pytest tests/integration/test_align_qwen35_model.py -q -s

# 正式门禁（0002 已提交）
MINILLM_RUN_HF=1 python -m pytest \
  tests/integration/test_align_qwen35_gdn.py \
  tests/integration/test_align_qwen35_model.py \
  tests/integration/test_align_qwen35_layer.py \
  tests/integration/test_align_qwen3.py -q
```

### 7.3 收口检查

```bash
python scripts/exp_index.py --check
python bench/make_prompts.py --check
python -m pytest -m "not slow and not hf" -q          # 必须仍绿
git status --short                                     # 正式 golden 之外不应有未提交改动
```

> 关键命令（正式 golden、正式 HF 门禁、提交）由用户执行；agent 给命令、跑 dry run、读结果、起草文档。

---

## 8. 实验矩阵（证据，不是 results run）

| 组 | 回答什么 | 形式 | 判据 |
|---|---|---|---|
| **E0** 算子 | 循环/分块是否与两套参考同解 | 单测 + 随机形状 | `operator_atol/rtol` |
| **E1** 卷积 | 因果性与状态推进是否正确 | 单测 | `conv_atol`、`causal_atol=0` |
| **E2** 单层 | 真实权重下每个 stage 是否对 HF | golden 集成测试 | stage rel 线 + `state_atol` |
| **E3** 全模型 | 24 层装配 + 贪心是否对 HF | golden 集成测试 | 64/64、logits、hidden |
| **E4** 自洽 | 分块 prefill / 逐步 decode 是否同一模型 | 单测 + 全模型 | `decode_prefill_atol` |
| **E5** 回归 | W5/W1 是否被碰坏 | 既有门禁 | 712+、64/64、9-stage 0 |

> packed/varlen GDN prefill 不在 W6 证据矩阵：已裁定留 W7（§12 第 1 条）；W6 不跑 E6。

**没有 `results/W6/NNNN`**：W6 的结论是"数值正确 + 接口冻结"，不是吞吐；把 golden 与 pytest
输出当证据，避免为了一个 run 去改 runner/threshold evaluator。

---

## 9. 风险、止损、时间盒

| 风险 | 触发 | 处置 |
|---|---|---|
| 分块 WY 推不对 | G2 在 Step 1 末仍红 | **止损**：生产 prefill 用逐步循环（数学正确、慢），`ROADMAP §8` 记负结果；W6 不标 ✅；分块代码只留 `tests/reference` 或删除，不静默回退 |
| `solve_triangular` 数值/后端问题 | 与 fla 循环差 >1e-5 | 先用 fla 的逐步前代法作 oracle 定位；仍不过则改用前代法（显式、无回退开关），并在 D8 记 |
| HF kernel 在 CPU 被换成 fla Triton | 加载/解码报 `x.is_cuda()` / Triton pointer | G0 钉 4 个符号；golden 脚本与 conftest 同一口径（P-48） |
| 全模型贪心翻转 | 64 步中某步 id 不同 | 先看该步 top-2 margin；early-token 大面积翻转按 bug 查；margin <1e-3 的 near-tie 只改变归因（记录 margin，不按实现错误查），但 **A4 仍以 64/64 为硬线**，翻转即不能标 ✅，按 §11.5 部分收口；≥1e-3 视为实现错误 |
| 全模型 logits 不达 0.05 | G4 | 用逐层 hidden 定位第一个偏层；GDN 层查 conv/state，full 层查 W5 口径；不调 golden |
| 0.8B fp32 CPU 太慢/OOM | 生成/测试超时 | 本机 755 GB RAM，实测 prefill 0.2 s/7 tok；若慢，只降低采样 stage 数，不降 dtype/设备（golden 必须 CPU fp32） |
| fixture 超 10 MB | 生成脚本报错 | 先减 sampled stage（尤其宽 stage）；**不截断 `layer_in`**（会让层测试无法复现）；`final_state` 已只存 m02，不能再靠删 s01 省空间 |
| 用户生成 golden 排队 | Step 5 正式生成等待 | agent 先把 dry run 与 stage 表做完；正式 fixture 未提交前不跑正式门禁，不把 /tmp 结果写进结论 |
| 旧测试锁死 W5 的"整模型失败"行为 | Step 4 改 loader 后红 | 这是 W6 的预期行为变化：把失败测试改成"`load_checkpoint` 对 hybrid 显式拒绝 + `load_qwen35_model` 成功"；0.6B 测试必须原样绿 |
| 时间超 10 天 | Step 5 仍未绿 | 按 §11.5 部分收口；不把 W7 的 StatePool/调度拖进 W6 补锅 |

---

## 10. PITFALLS 候选（收口按实际踩坑编号，当前从 P-48 起）

| 候选 | 坑 | 回归 |
|---|---|---|
| P-48 | HF 的 **recurrent** GDN 也会被换成 fla Triton；只钉 chunk 不够 | `tests/conftest.py` 4 符号；`test_qwen35_gdn.py` |
| P-49 | **参考文档与 5.20.2 实际实现不一致**：ROADMAP §14 / `benchmark-resources` 写 `conv_state` 长度 = kernel−1 = 3，但 HF 5.20.2 的 `DynamicCache.conv_states` 实际存最后 **4** 列（kernel_size）。两长度对"下一个 token"的输出等价（最旧一列不进入最后一个卷积窗）。W6 冻结为 **3**（最小充分状态），golden 只取 HF 状态的 `[..., -3:]`；若未来直接用 HF 的 `causal_conv1d_update` 或要对齐其 cache，再改成 4 | `test_qwen35_gdn.py` 状态等价 / 因果；golden 切片单测 |
| P-50 | `g` 已是 log-space 负值、`beta` 已 sigmoid；再套 sigmoid/softplus 会改变算子 | 公式单测 + 与 fla/HF 对拍 |
| P-51 | 状态布局是 `[B,H,K,V]`（K 在 V 前）；q/k 的 l2norm 在 fp32、scale 在 norm 之后；ratio>1 要 repeat_interleave | 形状/数值单测 |
| P-52 | 分块 padding 的输出必须丢弃；beta/g 补 0；两端 chunk_size 必须相同；T=C−1/C/C+1 是高频 off-by-one | 参数化边界单测 |
| P-53 | 0.8B 是 4 个独立投影（`in_proj_qkv/z/b/a`），不是 `qkvz/ba`；z/b/a 不卷积 | shape/键名单测 + 单层 golden |
| P-54 | 全模型加载：`model.language_model.` 前缀、tied `lm_head` 缺失、MTP/vision 丢弃；hybrid 不能静默走 `MiniLLM` | `test_qwen35_loading.py` |
| P-55 | 全模型贪心全等是数值门，不是聊天 demo；EOS 不特殊处理；near-tie 必须记录 margin | golden 集成测试 |
| P-56 | GDN 的 prefill/decode 分派看 `past_len`，不是 `T`；1-token prompt 的 prefill 走 chunk（HF 同） | `test_qwen35_gdn_layer.py` 1-token 用例 + 全模型 golden |
| P-57 | golden 的 `git_state` 采样时点：旧脚本在写完 `tensors.npz` 后才跑 `git status`，新建的 fixture 目录本身是未跟踪的 → `dirty=true`，即使代码树干净。W5 `0001` 的 `dirty=true` 因此不能**单独**当作"代码脏"的证据。W6 在创建 `out_dir` 前采样（或排除 `out_dir`） | `scripts/make_golden_qwen35_w6.py` 的 `git_state` 单测；`0002/meta.json` 的 `dirty=false` |
| P-58 | WY 分块的三角求解条件数随 `‖k‖`（以及 beta）恶化；未归一化 k 时 chunk vs recurrent 虽代数等价，数值可差 O(1e19)/NaN。必须保持 HF 的 q/k l2norm 契约；测试不要用未归一化 k 做跨端点对拍 | `test_qwen35_gdn.py` 归一化契约 / 与 fla 对拍 |

**已有条的修订**：P-15 的"三个符号"改为"四个符号"（补 `torch_recurrent_gated_delta_rule`），
测试映射指向 `tests/conftest.py` 与新的 GDN 单测。

---

## 11. 生命周期

### 11.1 状态机

| 状态 | 进入条件 | 允许动作 | 退出条件 |
|---|---|---|---|
| `draft` | 本文件创建（2026-11-09） | 用户 review；只读；不改代码/ROADMAP | 用户确认 §2/§12 → `ready`；或否决 → 删除 |
| `ready` | 2026-11-14 用户"按推荐"确认；范围/时间盒/证据冻结，未写代码 | 只维护本文件；不改代码/ROADMAP/D8；不跑 golden 生成 | 用户给出"开始" → `active`；或否决 → 删除 |
| `active` | 用户批准范围与时间盒并给出开始信号 | 按 Step 0–6 实现；小步提交；每步门禁绿 | G4 绿 → `closed`；触发止损 → `blocked`/部分收口；放弃 → `abandoned` |
| `blocked` | G2/G4 在时间盒内不过，且已按止损降级 | 只做已批准的最小交付；记录负结果 | 用户决定：部分收口 / 转 W7 / 放弃 |
| `closed` | A1–A6 全绿 + 收口清单完成 | 删除本文件；只留持久产物 | — |
| `abandoned` | 环境/权重/时间不可行 | 记录原因；不伪造 ✅ | 删除本文件 |

### 11.2 角色

- **agent**：起草/维护本文件；实现与单测；dry run 到 `/tmp`；起草 D8/PITFALLS/ROADMAP 文本；
  跑快速门禁与（在用户要求下）慢门禁；**不写** `tests/golden/`。
- **用户**：已于 2026-11-14 确认 §2/§12（按推荐）；准备开工时给出"开始"信号；运行正式 golden 生成；
  review 并提交 fixture；确认 D8 与 verdict；批准删除本文件。

### 11.3 临时 vs 持久

| 临时（收口删除/丢弃） | 持久（不随 plan 删除） |
|---|---|
| `docs/w6-plan.md` | `docs/decisions.md` **D8** |
| `/tmp/w6-golden-dry/`、stage 表日志 | `tests/golden/qwen35-0-8b/0002/`（用户提交） |
| 一次性 perf 计时（若有） | `configs/thresholds.toml [W6.gdn]` |
| — | `PITFALLS.md` P-48… |
| — | `ROADMAP.md` §5/§12/§13、§2/§9 状态 |
| — | `docs/hybrid_attention.md`、`tests/golden/README.md`、`README.md` 状态 |
| — | `model/qwen35_gdn.py`、`model/qwen35_model.py`、`scripts/make_golden_qwen35_w6.py`、测试 |

### 11.4 收口清单（按顺序）

1. **A1–A6 全绿**：快速门禁、E0–E4、W5/W1 回归；命令与输出粘进收口 commit/PR 描述。
2. 用户已提交 `tests/golden/qwen35-0-8b/0002`；`git status --porcelain` 为空（fixture 与代码都已提交）。
3. `[W6.gdn]` 改 `status="calibrated"`、`calibrated="2026-11-27"`（按实际收口日填），**数值不动**；
   `test_thresholds_w6.py` 绿。
4. **D8 accepted**：写清 3 算子、q/k l2norm + q scale 契约、状态接口、chunk_size、fp32 口径、W6/W7 边界、容差；
   收编 W5 的 `D-W5-1/2` 引用。
5. `ROADMAP.md`：§5 W6 加"实测（日期）"段；§12 W6 → **✅**；§13 记 v29（立项）/v30（收口）；
   §2 的"混合架构"行与 §9 的状态行同步。**不要改写 W5 v28 的历史 `708 passed`**：那是 W5
   收口时的记录；W6 的 v30 里写 W6 收口时的实际数，并注明 708 是 W5 收口时的记录、712 是
   开工前基线，两者不是同一时点的门禁数。
6. `PITFALLS.md`：按实际踩坑编号 P-48…，每条有测试映射；P-15 改 4 符号。
7. `docs/hybrid_attention.md`：状态改"W5/W6 已完成；W7 待填"；表 3–5 ✅；补 W6 实测与
   状态字节数；`tests/golden/README.md` 加 0002；`README.md` 的范围行更新。
8. **删除本文件**；`grep -R "w6-plan"` 确认无悬空引用；最终快速门禁 + 慢门禁 + 两个 check 绿。

### 11.5 部分收口 / 放弃

- 只有循环、没有分块：仍可交付单层 + 全模型（prefill 走循环），但 `hybrid_attention.md` 的
  "分块 prefill" 保持 ⬜，`ROADMAP §12` W6 不得写 ✅（写"部分完成：decode/全模型；分块未达"），
  W7 第一件事是补分块；不把慢速实现说成"已优化"。
- 全模型不过：交付算子 + 单层 golden + 状态接口；全模型 gap 写进 D8/§13，W7 续。
- 放弃：在 §13 记原因；删除本文件；已提交的 golden/测试保留（不可变），不删历史。

### 11.6 变更控制

- 范围变更：改本文件 §2 + 在 `ROADMAP §13` 记一笔；不口头扩大。
- 阈值：只在正式 golden **之前**、有 dry run 证据时调整；之后只能通过 D8 修订并记录。
- golden：不可覆盖、不可静默重生成；prompt/revision 变化才 bump `0003`（`tests/golden/README.md`）。
- 提交：每个 Step 一个 commit（测试绿才提交）；失败/回退用新 commit 修正，不 rewrite 历史；
  正式 fixture 单独 commit，不夹带代码改动。

---

## 12. 已裁定项（2026-11-14 冻结）

> 用户 2026-11-14 指示"需要确认的就按推荐的"；以下六项已冻结，不再等待确认。
> 变更只能走 §11.6。

1. **packed/varlen GDN prefill**：**已定 W7**（D-W6-3）。W6 不实现；旧 Step 6（packed spike）取消，
   收口顺延为 Step 6。W7 先补 `StatePool`，再接 packed/varlen，避免把算法 bug 与状态生命周期混在一起。
2. **fixture 布局**：**已定一个 `0002`**（D-W6-7），Step 5 一次生成，含单层 stage 与全模型
   logits/greedy/layer hidden；不拆 `0003`。
3. **prompt 选择**：**已定 `s01`（36 tok）+ `m02`（240 tok）**。m02 跨 3 个 chunk 边界、
   尾巴 48 token，比 W5 的 `m01@64` 更能暴露分块 off-by-one；不采用 `m01@64`。
4. **E6（packed 内部 spike）**：**已定不做**。W6 证据矩阵只到 E5；packed 属 W7。
5. **GQA ratio>1**：**已定实现 + 单测**。0.8B 是 16/16，不走该分支；为 4B/其他 checkpoint 留接口。
6. **时间盒**：**已定 8–10 个工作日，窗口 2026-11-16 → 2026-11-27**（11-14 复核仍未开工，
   旧窗口 11-20 已不可行）。超时按 §9/§11.5 部分收口，不把 W7 的 StatePool/调度拖进 W6。

---

## 13. 与 ROADMAP 的差异记录

| ROADMAP 措辞 | 本计划 | 处理 |
|---|---|---|
| §1.3 / §5 W4.5："B 的 packed prefill 在 W6 接同一接口" | D-W6-3：留 W7 | 转 active 时在 `ROADMAP §13` 记 v29；W7 段补"第一件事：GDN packed/varlen 接线" |
| §5 W6 只写"短卷积 → decode → 分块；两套参考互拍；全模型 64 token" | 本计划增加单层 stage golden、状态接口、逐层 hidden、状态字节数 | 不改变验收，只把证据做厚；不新增 results run |
| §2 "混合架构（5 类算子）W5–W6" | W6 收口时改为 ✅（5 类算子全部完成；packed 接线仍 W7） | 收口清单第 5 条 |
| §14 与 `benchmark-resources` §3.5 写 `conv_state` 长度 = kernel−1 | HF 5.20.2 实际 cache 存 kernel=4；W6 取最小充分状态 3，golden 切片 | 收口时修 §14 / 参考文档，并记 P-49 |
| §9 状态"W5–W7 未开始"（2026-11-08） | W5 已完成，W6 计划 **ready**/未开工 | 收口时更新；这是已过期的状态行 |

---

## 14. 参考实现与许可

| 来源 | 版本/位置 | 用途 | 许可 |
|---|---|---|---|
| HF `qwen3_5/modeling_qwen3_5.py` | 5.20.2，L287/L424/L490 | 分块/循环公式、层装配、conv_state 口径 | Apache-2.0 |
| `fla/ops/gated_delta_rule/naive.py` | 0.5.2 | 第二套独立实现，测试 oracle | MIT |
| Qwen3.5-0.8B config/checkpoint | `2fc0636…` | 形状、权重、golden | 以 HF repo 的 LICENSE 为准 |

> 抄公式可以，抄代码保留来源与许可注释；两套参考**只作测试与对拍**，不 import 进生产路径
> （`fla` 的 Triton kernel 不接，`tests/` 里可以 import）。

---

## 15. 附：当前代码地图

```
model/
  qwen35.py          W5：两种 norm + Qwen35Attention + Qwen35FullAttentionBlock（不动数值）
  qwen35_gdn.py      W6 新增：conv + recurrent + chunked + Qwen35GDN + Qwen35GDNBlock
  qwen35_model.py    W6 新增：Qwen35State + Qwen35Model（LLLF 调度）
  loading.py         W6 扩展：linear 层 + 整模型 + 显式 hybrid 入口
  layers.py          复用：MLP / KVcache / rope_cos_sin / apply_rotary_emb_partial
  minillm.py         0.6B 全注意力模型（不动）
tests/
  conftest.py        4 个 kernel pin
  unit/test_qwen35_gdn*.py / test_qwen35_model.py / test_thresholds_w6.py
  integration/test_align_qwen35_gdn.py / test_align_qwen35_model.py
scripts/
  make_golden_qwen35_w6.py
tests/golden/qwen35-0-8b/0002/
```
