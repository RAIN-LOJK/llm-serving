# W4 · 分页 KV（实验设计 v2）

> **生命周期：临时文件，W4 阶段完成即删。** 判据以 `ROADMAP.md` §5 W4 为准；
> 本文件只把判据操作化。**ROADMAP 未按 §1.4 改完之前，不跑任何正式 run。**
> **删除时同步修掉**：`ROADMAP.md` §5 W4 的方案链接、§12 的 W4 行（⬜→✅）、§13 一条修订；
> §14 的 W4 参考实现保留，另加一句「W4 口径见 `docs/decisions.md` **D6**」。
>
> **状态：计划（v2，未跑任何 W4 run）。** 数字分三类：`实测`（已有 run）、`解析`（由长度与公式算出）、
> `预测`（预注册待验）。前置：`W3-0011`（continuous B=8 mixed，311.43 tok/s，池 1456 MiB）只作为
> **预算锚点**，**不再作为 W4 的配对基线**。D4（注意力路径）不动；W4 新增 **D6**（分页口径）。

---

## 0. 一页速览

**做什么**：新增 `serve/block_manager.py` 与 `model/paged.py`，在 `serve/engine.py` 加 paged 池，
`scheduler` 加准入钩子，`bench` 接 `impl="paged"`。**不换 kernel、不改席位路径。**

**判什么**：

1. **容量**：同总预算（KV + workspace）下，mixed max-2048 的 peak concurrency ≥ **1.50×**；
2. **碎片**：paged 的 internal frag ≤ **5%**；
3. **数值**：paged vs seat 逐 token 全等；
4. **代价**：B=8、两边 `lpad=1664` 的 `step_ms` 记录；>1.5× 则撤回吞吐结论。

**怎么测**：容量用**确定性 allocator 模拟**（扫预算与长度分布）+ **2 个实机 fill run** 校验；
代价用 **2 个 closed run**；吞吐是可选的 **2 个 scenario run**。不再用一个 scenario 同时回答三件事。

**关键口径**：总预算 **1640 MiB = KV + workspace**，paged 的 workspace 必须计入；uniform 2048 是
**负对照**（预测 0.8–1.0×），不判 1.5×。收益只来自长度异质性。

---

## 1. 主张与判据

### 1.1 主张

W4 要证明的不是「分页省显存」，而是：

> **同样的总显存预算，分页把「按 trace 最坏长度预留」换成「按每条请求的上界预留」，
> 从而在长度异质时装下更多并发；长度均匀时它不占优，甚至因 workspace 略亏。**

### 1.2 判据（预注册）

| # | 判据 | 口径 | 目标 / 失败线 |
|---|---|---|---|
| C1 | 同预算容量 | **总预算 1640 MiB（KV + workspace）**，W3-like max-2048，fill-only 最大并发 | **≥1.50×**；<1.20× 记负结果收口 |
| C2 | 碎片 | paged **internal frag**（只算最后一块没写满），时间加权 | **≤5%**；reservation slack 单独报，不判 |
| C3 | 数值 | paged vs seat 逐 token 一致；E0 对 golden | `token_sha256` 全等 |
| C4 | 代价 | B=8、两边 `lpad=1664` 的 decode `step_ms` | 记录；≤1.15× 健康，>1.5× 撤回吞吐结论 |

**C1 比的是同一「上限」，不是同一「实际用量」。** seat 因行粒度用不满的 170 MiB 单列
`seat_unused_mb`，不补给 paged；两边实际 `kv_total_mb` 都报。C1 的 `peak_concurrency` 是
**allocator capacity**（fill-only 的最大同时准入）；E3 另报**动态** `peak_concurrency`，两者不混。

**uniform 2048 不是判据。** 它是负对照：seat 的 6 行正好覆盖均匀长度，paged 还要付 workspace，
预测 0.8–1.0×；报告必须给出，但不拿它判 1.5×。

C4 不进 W4 硬验收（ROADMAP 原本没有），但它是「能不能报吞吐」的门槛。这样不会因为 host/GPU
抖动把容量判据一起拖死。

### 1.3 预测（待验，不是目标）

| # | 预测 | 依据 |
|---|---|---|
| P1 | W3-like max-2048 容量 **约 3.0–3.5×**（20 左右 vs 6） | 800 块 ÷ 约 40 块/请求；seat 6 行 |
| P2 | uniform 2048 容量 **0.8–1.0×** | 6 vs 6（保守 5 vs 6），paged 多付 workspace |
| P3 | internal frag bs16 **约 1.4%**，bs32 **约 2.8%** | mixed 平均长度约 565 的估计；E2 用长 prompt，实测会更低 |
| P4 | reservation slack **约 14%**（时间加权） | `W3-0011` 640 条 request row 按同一公式实算 |
| P5 | B=8 同并发 `step_ms` **1.05–1.15×**，workspace **约 +55 MiB** | 8 行、lpad 1664、max_batch=8；gather 只增加少量派发 |
| P6 | 同预算系统吞吐 **2.2–2.6×**（record-only） | 容量约 2.6×，step 略慢；仅 E3 通过才报 |

P1–P6 都不进判据；判据只有 C1–C4。

### 1.4 开工前要同步的 ROADMAP

把 §5 W4 的验收改成：

> 同总预算（KV + workspace）、**最大上下文 2048 的混合长度 trace**，peak concurrency ≥1.5×；
> uniform 2048 作为负对照（只报不判）；internal frag ≤5%；paged vs seat 逐 token 全等；
> B=8 matched-`lpad` 的 step 代价记录，>1.5× 时撤回吞吐结论。

理由写进 §13：原条款的 `ctx=2048` 被解释成 uniform，测不到「按最坏长度预留」的浪费，
与同段论点矛盾。**这一步由用户确认后再改，不由 agent 静默改。**

---

## 2. 口径冻结（D6，proposed → accepted 后才能跑正式 run）

| 项 | 定义 |
|---|---|
| **总预算** | `kv_total_mb = kv_mb + workspace_mb`，W4 固定 **1640 MiB**；seat 与 paged 同一上限，实际用量分别记 |
| **seat 单位** | 一块 `KvSlab` 的 `max_batch` 行，行宽 = `bucket(trace 最大 prompt+budget, 64)`；每行按最坏长度预留 |
| **paged 单位** | 每层 `(n_blocks, bs, n_kv_head, head_dim)`；`kv_blocks = floor((1640 − workspace) / (bs × 112 KiB))`；workspace 实测后定 |
| **块表** | 每请求一张逻辑→物理映射；第 `j` 个 token 在 `blocks[j//bs]` 的第 `j%bs` 列；**块分配后不移动**，不做压缩/搬迁 |
| **准入** | 一次性预留 `ceil((prompt + budget) / bs)` 块；同一 step 的多次准入必须原子；无抢占 ⇒ 不允许中途分配失败 ⇒ 无死锁 |
| **退出** | 立即归还全部块到 free list；`release_request` 幂等；下一请求可直接复用 |
| **decode gather** | 每步按块表把**活跃行**搬成 `(n_active, H_kv, lpad, D)`，`lpad = bucket(本步最长活跃长度, 64)`；不足 `lpad` 的行用**专用零块**补齐，逻辑列 mask 关掉；然后走现有 `hand_baddbmm` |
| **解码批** | 只压缩到活跃行（席号→批行映射在引擎内），logits 散回 `(max_batch, vocab)`；`max_batch` 只是调度上限，不再是显存上限 |
| **容量** | fill-only 下最大同时准入请求数；W3-like 主判据；uniform 2048 负对照；预算 sweep 只进模拟 |
| **同预算** | 两臂同一 `1640 MiB` **上限**；实际 `kv_total_mb` 都报；seat 因行粒度用不满的部分单独列 `seat_unused_mb` |
| **internal frag** | `1 − 已写 token / (已写块 × bs)`，时间加权；**C2 只判这个** |
| **reservation slack** | `1 − (prompt + generated) / (reserved_blocks × bs)`；时间加权；单独报，不判 |
| **seat row slack** | `1 − live / row_width`；与 reservation slack 同表，作为「seat 的浪费」对照 |
| **代价** | B=8、两边 `lpad=1664`、只统计 decode 步的 `step_ms`；`decode_event_ms` 只作诊断，不当 GPU 时间 |
| **配对** | 容量：同分布、同预算上限、`max_batch` 可不同；吞吐：同 `trace_sha256` / `prompt_sha256` / dtype / device_class / 预算，两臂都 `saturated` |
| **fill-only** | E1 只 prefill + 预留，不 decode；synthetic prompt 的 EOS 不影响容量 |
| **EOS** | E2/E3 用 frozen prompt，先查 `out_tokens` 中位数 == budget；E1 不 decode，不受 P-31 影响 |

**D6 必须在跑 E0 之前 accepted。** 跑完只把 `configs/thresholds.toml` 的 `[W4.paged]` 标 calibrated，
不允许在看完数字后再改判据定义。

---

## 3. 实现

```text
serve/block_manager.py   free list + 块表 + 准入块数 + 计数（reserved/written/active）。纯 Python，不碰张量
model/paged.py           物理块张量、prefill 写穿、decode gather、本步 mask/bias/pos、workspace_mb()
serve/engine.py          open_paged_pool / prefill_into_blocks / decode_batch / release_request / can_admit + 账
serve/scheduler.py       准入前问 can_admit（拿不到块留队列，算排队）；step row 加块统计。policy/终止/时间戳不动
model/{layers,minillm}.py  decode 容器从「只认 KvSlab」改成「认容器协议」；KvSlab 路径逐 bit 不变
bench/engines/paged.py   impl="paged" 适配器；scenario 复用 ContinuousEngine 的 trace 路径，fill/closed 走固定批
bench/run.py             scenario 谓词纳入 paged；[scheduler] 允许 paged；新增 type="fill"；_flatten_scenario 读 impl
bench/scenario.py        throughput 补 peak_concurrency；新增 capacity/memory 块（workspace/frag/slack）
bench/thresholds.py      两个 W4 判据 + 容量配对校验（允许 max_batch 不同）
configs/experiments/W4/  7 个配置（5 核心 + 2 可选）+ 1 个 dummy 冒烟
```

关键接口（签名先定，实现随意）：

```python
BlockManager(n_blocks, block_size)
  .reserve(n) -> list[int] | None      # 原子；不够返回 None
  .free(ids)                           # 只归还一次；重复 free 报错
  .stats() -> dict                     # reserved/written/free/peak

PagedPool(n_layer, n_blocks, bs, n_kv_head, head_dim, dtype, device)
  .write_prefill(layer, block_table, start, k, v)   # 按块表写 L 个 token
  .write_token(layer, row, block_table, pos, k, v)  # decode 单 token
  .gather(layer, block_table, lpad) -> (k, v)       # 返回 workspace 视图
  .workspace_mb() / .bytes()

MiniLLMEngine.open_paged_pool(kv_blocks, block_size, max_batch)
  .can_admit(prompt_len, budget) -> bool
  .prefill_into_blocks(request_id, prompt_ids) -> logits
  .decode_batch(tokens, cache) -> (max_batch, vocab)
  .release_request(request_id)                       # 归还块，幂等；不要复用 freeze_seat
```

不变量（直接抄进 `tests/`）：

- `active ≤ max_batch`、`reserved_blocks ≤ n_blocks`、块表长度 == `ceil((prompt+budget)/bs)`；
- 一个块同时只属于一个请求；**专用零块除外**（只读、共享、不在 free list）；
- `free → allocate` 复用后输出与「从未用过」逐 bit 相同；
- 每步 `gather` 列数 == `lpad` == mask 宽度，且 `lpad ≥ 本步最长行`；
- **RoPE 表长是「本 run 的位置上界」，不是 `n_blocks × bs`**；`lpad` 与 RoPE 上界是两个数；
- 空块只被 mask；**不靠「每次分配清零」保证正确性**；池初始化时全零，复用块里是旧有限数据；
- `ttft == queue + prefill` 仍按构造成立（等块算 queue）；
- paged 的 64 token 贪心输出 == `tests/golden/qwen3-0-6b/0001` 且与 seat 臂逐 id 相同。

**账（必须由代码产出，不能只在文档里）**：`kv_mb`（池预留）、`kv_live_mb`（写了字的 token）、
`kv_alloc_mb`（已分配块 × bs）、`workspace_mb`（分配器实测）、`gather_read_mb` / `gather_write_mb`
（块表算出）、`frag_internal`、`reservation_slack`、`blocks_in_use`、`written_blocks`、`lpad`。

---

## 4. 实验

### 4.1 A · 容量：确定性模拟 + 2 个 fill run

**模拟**：`scripts/sim_paged_capacity.py`，直接 import 真实 `BlockManager` + 一个 seat 行模型，
不跑模型、不跑 Poisson。扫：

| 维度 | 取值 |
|---|---|
| 总预算 | 1200 / 1456 / 1640 / 2400 MiB |
| 长度分布 | uniform-2048；mild-mixed（16×512 + 16×2048）；W3-like max-2048（12 短 / 12 中 / 8 长，长桶 1536–2048） |
| block size | 16（主）、32（敏感性） |

输出：`capacity_ratio vs budget`、`capacity_per_GiB`、internal frag、reservation slack、
`fig_w4_capacity.png`。**报告主图是容量曲线，C1 只是 1640 MiB 那条竖线。**

**实机校验**（只验 W3-like @ 1640, bs16）：

- `W4-0002` seat：`max_batch=6`，行宽 2240，6×245 = 1470 MiB；`seat_unused_mb=170`；
- `W4-0003` paged：`max_batch=64`，`kv_blocks=800`（保守 workspace 240 MiB；实测后报实际
  `kv_total_mb`；若 workspace ≤240，总量 ≤1640）。fill 到不能再准入，记 `peak_concurrency`。

两边同一 `seed=0` 请求顺序、同一预算上限。预测：seat 6、paged 约 20。

### 4.2 B · 代价：2 个 closed run

固定 **8 条最长 frozen prompt**（987–1517），budget 128，closed，`max_batch=8`：

- `W4-0004` seat：8 行 × 1664 = 1456 MiB；
- `W4-0005` paged：`kv_blocks=832`（8×103=824，刚好放得下），1456 MiB KV + workspace。

两边每一步 `lpad` 都是 **1664**、active 都是 8，没有准入/排队噪声；只统计 decode 步。
测 C2（internal frag）、C4（`step_ms`），同时产出同并发显存开销：

| 项 | seat | paged |
|---|---|---|
| `kv_mb` | 1456 | 1456 |
| `workspace_mb` | 0 | 约 55（实测） |
| `kv_total_mb` | 1456 | 约 1511（+3.8%） |
| `bytes/active_request` | 182 MiB | 约 189 MiB |

**同样 8 条并发，paged 多花约 4% 显存**——这是代价，不是收益；必须和 C1 的容量收益分开报。

### 4.3 C · 吞吐（可选，record-only）

只有 A/B 都通过才跑。frozen 32 prompts、Poisson、同一 λ、两臂都重跑：

- `W4-0006` seat：`max_batch=9`，9×1664 = 1638 MiB；
- `W4-0007` paged：`kv_blocks=840` + workspace ≈ 1639 MiB；
- λ 由冒烟容量定，取 **1.25–1.5× paged capacity**（例如 30）；两臂都必须 `saturated=true`，
  否则 λ 加倍、**两臂一起重跑**。**不复用 `W3-0011`。**

`tok_s` 只作 record-only：这是「同总预算的系统吞吐」，不是 D5 的调度器机制对比。
时间紧就删掉，W4 的结论不依赖它。

### 4.4 Run matrix

| ID | 类型 | 臂 | 工作负载 | 用途 | 门禁 |
|---|---|---|---|---|---|
| `W4-0001` | microbench | paged | 1 请求，golden prompt，64 token | 数值门 | token 全等；不过不许跑性能 run |
| `W4-0002` | fill | seat | W3-like max-2048，32 请求 | 容量校验 | 并发 = 模拟 ±1 |
| `W4-0003` | fill | paged | 同上 | 容量 + workspace 实测 | 同上；`kv_total ≤1640` |
| `W4-0004` | closed | seat | 8 条最长，B=8，128 token | 代价基线 | — |
| `W4-0005` | closed | paged | 同上，832 块 | 代价 + 碎片 | step ≤1.15×；internal frag ≤5% |
| `W4-0006` | scenario | seat | frozen 32，λ，B=9 | 吞吐基线 | `saturated` |
| `W4-0007` | scenario | paged | 同 trace，B=64，840 块 | 系统吞吐 | `saturated`；record-only |

核心 5 个；`0006/0007` 可选。模拟脚本不占 run id。

### 4.5 配置字段

```toml
[engine]
impl = "paged"          # paged 臂；seat 臂用 continuous
max_batch = 64          # E1/E3；E2 固定 8
block_size = 16         # 16 | 32
kv_blocks = 800         # E1；E2 832；E3 840；由总预算和 workspace 反推
[workload]
kind = "fill"           # E1；E2/E3 分别 closed / poisson
[scheduler]
policy = "continuous"   # E2 可 static；E3 continuous
```

`bench/run.py` 必须允许 `impl="paged"` + `[scheduler]`，并把 `paged` 纳入 scenario 谓词；
否则配置在 runner 校验阶段就会被拒。

---

## 5. 指标与 summary

**step row 增记**（判据 run 不开逐层计时，只在 E2 与 E3 开）：

```jsonc
{"row_kind":"step", "active":8, "lpad":1664, "blocks_in_use":824,
 "written_blocks":824, "kv_alloc_mb":1442.0, "kv_live_mb":1094.0,
 "frag_internal":0.014, "reservation_slack":0.24,
 "gather_ms":1.8, "gather_read_mb":52.0, "gather_write_mb":52.0,
 "step_ms":24.2, "decode_event_ms":23.9, "prefill_ms":0.1, "host_gap_ms":0.2}
```

**summary 的 `memory` 块**：`kv_mb`、`workspace_mb`、`kv_total_mb`、`seat_unused_mb`、
`frag_internal`、`reservation_slack`、`seat_row_slack`、`capacity_per_gib`、`bytes_per_active_request`。

**summary 的 `throughput` 块**：补 `peak_concurrency`（`ContinuousEngine` 已算，`scenario.py` 现在没放进去）。

**notes 必给的分解表**：

| 项 | 来源 | seat | paged |
|---|---|---|---|
| 同并发 `step_ms` | E2 | ? | ? |
| 其中 gather | E2 | 0 | ? |
| 其中 event 窗口其余 | `decode_event_ms − Σgather_ms` | ? | ? |
| host（窗口外） | `host_gap_ms` | ? | ? |
| KV 字节/step | `gather_read + gather_write + kernel 读` | ? | ? |
| workspace | `workspace_mb` | 0 | ? |
| 同并发总显存 | `kv_total_mb` | 1456 | ? |

**图**：`fig_w4_capacity.png`（容量 vs 预算/长度 CV，标 1640 与 1.5× 线）、
`fig_w4_memory.png`（同并发显存开销 + capacity/GiB）、`fig_w4_step.png`（step 分解）。

---

## 6. 测试清单（红 → 绿，先写后实现）

| 文件 | 覆盖 |
|---|---|
| `tests/unit/test_block_manager.py`（新） | 分配/回收/复用/重复 free 报错/容量记账/准入块数/时间加权碎片 |
| `tests/unit/test_paged_kv.py`（新） | 写穿→gather 往返；bs16 与 bs32 逐 bit 相同；复用块旧数据被 mask；专用零块；gather 后与 `KvSlab` 逐 bit 相同 |
| `tests/unit/test_paged_decode.py`（新，toy 模型） | 多步 decode == 逐序列参考；席号有空缺（0/2/5）时 logits 散回正确行；块被下一请求复用后输出不变 |
| `tests/unit/test_paged_capacity.py`（新） | 模拟的 seat/paged 容量与手算小例子一致；预算 sweep 单调；workspace 计入总预算 |
| `tests/unit/test_scheduler.py`（扩展） | 拿不到块时留队列且 `ttft == queue + prefill`；同 step 多次准入不超卖；`active ≤ max_batch` |
| `tests/unit/test_run_contract.py`（扩展） | `impl="paged"` 校验（`block_size ∈ {16,32}`、`kv_blocks ≥ 1`、`type="fill"`、`[scheduler]` 合法、`batch_sizes=[1]`） |
| `tests/unit/test_thresholds.py`（扩展） | 容量判据（允许 max_batch 不同；分布/预算不符→invalid；缺基线→pending）；internal frag >5% → fail；reservation slack 不判 |

硬门禁：`python -m pytest -m "not slow and not hf"` 全绿；
`MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py -q` 全绿；
`test_batch_forward.py` / `test_prefill_into_seat.py` 零回归。

---

## 7. 施工顺序与门禁（约 5–6 天，正式 run 由用户在 4090 上跑）

| 天 | 动作 | 门禁 |
|---|---|---|
| D1 | 改 ROADMAP §5/§13；立 **D6（accepted）**；写 §6 的 block_manager / paged_kv / capacity 用例（红）；实现 `serve/block_manager.py` | D6 定稿；该两文件用例绿 |
| D2 | `model/paged.py` + decode 容器协议（`KvSlab` 逐 bit 不变） | paged_kv + batch_forward 全绿；`test_align_qwen3`（HF）全绿 |
| D3 | engine 的池与 gather、`scheduler.can_admit`、压缩批、`release_request` | paged_decode / scheduler 全绿；E0 |
| D4 | `bench/run.py` 契约 + paged adapter + fill/closed + dummy 冒烟 | `exp_index --check` 绿；paged 冒烟通 |
| D5 | 用户跑 E0 → E1 → E2 | E0 pass 才允许性能 run；E1 `kv_total ≤1640`；E2 `lpad` 两边 1664 |
| D6 | 可选 E3；回填 D6（calibrated）、`thresholds.toml`、ROADMAP §5/§12/§13、PITFALLS、删本文件 | 索引与判据表一致；notes 我起草，`verdict` 保持 `unreviewed` |

---

## 8. 风险与止损

| # | 风险 | 触发 | 处置 |
|---|---|---|---|
| R1 | gather 太贵，容量赢了吞吐没赢 | E2 `step_ms` >1.5× | 记录负结果（容量照报）；撤回吞吐结论；**不换 kernel、不加 graph** |
| R2 | 分页与席位数值不一致 | E0/E2 token 不匹配 | **停性能 run**；按 mask 宽 / lpad / 块内偏移 / RoPE 行位置 / logits 散回五处二分；不许放宽 atol |
| R3 | 块泄漏 | 容量明显低于模拟，或 `blocks_in_use` 单调涨 | 查退出 / prefill 失败 / 半写块三条路径；泄漏是硬 bug，不出报告 |
| R4 | workspace 反复重分配 | step 时间抖动 | `lpad` 按 64 桶、预分配到本 run 上界；`workspace_mb` 实测 |
| R5 | 总预算被 workspace 顶穿 | `kv_total >1640` | 下一 run 降 `kv_blocks`；**不静默改预算**；报告实际值 |
| R6 | 容量 <1.2× | E1/模拟 | 按 ROADMAP §8「不硬调」收口，W4 以负结果结题，直接进 W5 |
| R7 | 单一预算/单一分布得出巧合结论 | 评审 | 报 budget sweep + 长度分布曲线；C1 只是切片 |
| R8 | 跨阶段复用基线被质疑 | `W3-0011` 的 device/commit 与本次不同 | **本版不复用 W3-0011**；E3 同机、同 trace、两臂重跑 |

---

## 9. 不在 W4 做

- 抢占 / swap / recompute（无抢占 ⇒ 准入预留满，D6 主动选的边界）；
- chunked prefill、prefix cache / 引用计数 / COW；
- **CUDA graph**（会消灭 host 成本，改变要测的东西；要做必须两臂都做 + 另开 run）；
- **flash-attn / paged-attention kernel**：那是另一个实验（换 kernel 会同时废掉 D4 的单一注意力路径、
  C3 的逐 token 一致，以及 W2-0007 / W3-0010/0011 的全部基线），**W8 之后另立，不放进 W8 判据矩阵**；
- 按长度分组 decode（host-bound 时加组必亏，等 graph 之后）；
- 量化、MTP/投机、混合架构状态（W7）；块大小自适应、`max_batch` 自动调参、多队列。

---

## 10. 交付物与报告骨架

- 代码：`serve/block_manager.py`、`model/paged.py`、engine 的 paged 池、scheduler 的准入钩子；
  新测试 4 个文件 + 3 处扩展，快速门禁与 golden 全绿
- run：`W4-0001`(E0) / `W4-0002,0003`(E1) / `W4-0004,0005`(E2) / 可选 `W4-0006,0007`(E3)
- 图：`fig_w4_capacity.png`、`fig_w4_memory.png`、`fig_w4_step.png`
- `docs/decisions.md` **D6**；`configs/thresholds.toml` `[W4.paged]` → calibrated + `basis_run`
- ROADMAP §5/§12/§13；`serve/README.md`；PITFALLS P-32…P-36；删本文件

**报告这一节（数字待填）**：

> 同样 **1640 MiB** 总预算，W3-like max-2048 下分页把并发从 **6** 提到约 **20（3.3×）**；
> uniform 2048 只有 **0.8–1.0×**——分页不创造容量，它把「按 trace 最坏长度预留」换成
> 「按请求上界预留」，收益只来自长度异质性。
> 代价：同并发 B=8 时 paged 多花约 **4%** 显存（workspace），每步 **? ms（?×）**；
> gather 每步搬 **? MB**，workspace **? MiB**。
> 边界：① 吞吐收益依赖 step 仍未被 gather 顶穿（C4）；② 预算越松、长度越均匀，收益越小；
> ③ 预算远大于实际输出（早停）、单请求、极大并发都会让收益收窄。

---

## 11. 命令

```bash
python -m pytest -m "not slow and not hf"
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py -q
python scripts/sim_paged_capacity.py --out docs/fig_w4_capacity.png
python -m bench.run --config configs/experiments/W4/9001-paged-smoke.toml --out /tmp/minillm-w4-smoke --allow-dirty
python -m bench.run --config configs/experiments/W4/0001-paged-numeric-gate.toml       # E0
python -m bench.run --config configs/experiments/W4/0002-seat-fill-max2048.toml        # E1
python -m bench.run --config configs/experiments/W4/0003-paged-fill-max2048.toml       # E1
python -m bench.run --config configs/experiments/W4/0004-seat-b8-cost.toml             # E2
python -m bench.run --config configs/experiments/W4/0005-paged-b8-cost.toml            # E2
python -m bench.run --config configs/experiments/W4/0006-seat-saturate.toml            # E3（可选）
python -m bench.run --config configs/experiments/W4/0007-paged-saturate.toml           # E3（可选）
python scripts/exp_index.py && python scripts/plot_w4.py
```

---

## 附录 A · 预算算术（为什么固定总预算）

上一版固定的是 **KV 预算 1456 MiB**，workspace 没进分母；这是 uniform 2048 读出 1.2× 的原因：

| 口径 | seat | paged | 结论 |
|---|---|---|---|
| KV-only 1456 | 5 行 × 245 = 1225 MiB | 6 请求：1428 KV + 约 71 ws = **1499 MiB** | 6 vs 5 = 1.2×，但**不是同显存** |
| total 1456 | 5 行 = 1225 MiB | 5 请求：1190 KV + 约 63 ws = 1253 MiB | 5 vs 5 = **1.0×** |
| total 1640 | 6 行 = 1470 MiB | 6 请求：1428 KV + 约 71 ws = 1499 MiB | 6 vs 6 = **1.0×** |

- 表中 ws 是 **6 active / lpad 2240 / max_batch=64**；183 MiB 是 25 active / lpad 1664，不能混。
- E1 混合容量是 20 active，ws 约 195–212 MiB，故保守取 `kv_blocks=800`（ws 按 240 留余量）；
  800 块只够 5 条 2048 请求，uniform 5 vs 6 = 0.83×；816 块才是 6 vs 6。
- W3-like max-2048：seat 6 行（1640 MiB 上限，余 170）；paged 800 块 / 平均约 40 块/请求
  → 约 20 并发 → **3.3×**。uniform 2048：136 块/请求 → 800/136 ≈ 5.9 → 5；这是负对照，不是失败。

---

## 附录 B · 为什么 decode 必须稠密

`hand_baddbmm` 的输入是 `(B·H_kv, D, L)`，`L` 是共享直角边；seat 的行等距，零拷贝 `view`；
paged 的逻辑相邻 token 物理不相邻，没有 view，只能先物化到 `lpad`。

| 方案 | 每步流量 | 派发 | 裁决 |
|---|---|---|---|
| A 稠密 gather（W4） | gather 读+写 ≈2× KV；加 kernel 读 ≈3× | 1×/层 + 少量 gather 调用 | 选它：保住 D4「一层一次 baddbmm」 |
| B 逐行 attention | ~1.0× | 28×B | 否：D3 实测 B=32 时纯派发 21–37 ms |
| C kernel 级 paged attention | **1.0×** | 1×/层 | 真解，但换数值/基线 → W8 之后 |
| D 每请求连续段（免 gather） | ~1.0× | 28×B | 等价 B：起点不同，批不成一个 stride |

`K` 先 `reshape(B*H, L, D)` 再 `transpose(-2,-1)` 是合法 view；先转置再 reshape 会拷贝。
P-25 的隐式拷贝只发生在 `k[:, :, :L]` 切片之后——本设计不切片。

---

## 附录 C · 成本模型（先测再预测）

**workspace（随 active / lpad / max_batch 变）**

```
K+V:    n_active × lpad × 4 KiB
bias:   n_active × H_kv × G × lpad × 2 B      # G = n_head / n_kv_head
logits: max_batch × vocab × dtype_bytes
```

B=8 / lpad 1664 / max_batch=8 ≈ **55 MiB**；20 active / lpad 2240 / max_batch=64 ≈ **195 MiB**；
23 active / lpad 1664 / max_batch=64 ≈ **170 MiB**。每个 run 实测，不复用同一个数。

**gather 流量（单层）**：`read = write = n_active × lpad × 4 KiB`，28 层累加。
`gather_read_mb` / `gather_write_mb` 是流量，`workspace_mb` 是占用，不混。

**时间**：不手算「705 op × 25 µs」；先测 E2 的 `step_ms` / `decode_event_ms`。
event 是窗口（含 host 间隙），不是 GPU kernel 时间和；C4 用 `step_ms`。

`capacity_per_GiB = peak_concurrency / (kv_total_mb / 1024)`；budget sweep 见 §4.1，不在这里重复。

---

## 附录 D · 参考与坑

参考实现见 ROADMAP §14；坑的 canonical home 是 `PITFALLS.md`。W4 新增、尚未登记的：

| ID | 坑 | 回归 |
|---|---|---|
| P-32 | 分页把「按最坏长度预留」从显存搬到带宽：gather 宽 = 最大活跃长度 | `test_paged_decode.py` + `steps.by_active` |
| P-33 | 块表 padding 用专用零块 + 逻辑 mask，不能靠「块内容清零」 | `test_paged_kv.py`（旧数据被 mask） |
| P-34 | 「RoPE 表长」与「每步 lpad」是两个数，混用会让表每步重建或 mask 错位 | `test_paged_kv.py`（表只建一次） |
| P-35 | `kv_mb` 必须同时报池预留、活跃占用、workspace，否则重演 W2-0004 / P-30 | `test_paged_kv.py` |
| P-36 | 固定 KV 预算而不计 workspace，会让 uniform 2048 读出假的 1.2×；必须用 `kv_total_mb` | `test_paged_capacity.py` |

P-26 由 W4 的块池**关闭**（不是作废）：`freeze_seat` 保留「冻结游标、保留预留」的原语义，
paged 路径用 `release_request` 归还块；两条路径不要复用同一个函数名。
