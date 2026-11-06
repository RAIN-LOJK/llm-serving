# W4-0005 · E2 代价（paged）：8 条最长 / B=8 / budget 128 / 832 块

- 状态：`reviewed`；verdict = `pass`（C2/C3；C4 负结果见结论）
- 配置：`config.toml`（运行目录 `results/W4/0005-paged-b8-cost`）
- 类型：`scenario`；workload：`prompts`（`l01`–`l08`，与 `W4-0004` 同一份）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `paged_matches_seat_step_cost_at_the_same_lpad`
- C2：paged 的 internal frag（只算最后一块没写满，时间加权）**≤ 5%**；
  reservation slack 单独报，不判。
- C4：B=8、两边同为 1664 宽的 decode `step_ms`，**≤1.15× 健康；>1.5× 撤回吞吐结论**；
  C4 本身不进 W4 硬验收，但它是「能不能报吞吐」的门槛。
- C3：paged 与 seat 的逐 token 输出全等；E0 对 golden。
- 同 workload 门禁：`trace_sha256` / `prompt_sha256` 必须与 `W4-0004` 相同。

## 观察

### C2 · 碎片（pass）

- `fragmentation_max = 0.0055`（0.55%，target 5%，**pass**）；
  `memory.frag_internal = 0.00549`，逐 step 时间加权 0.5566%，与 summary 一致。
- reservation slack：按 D6 的时间加权口径为 **5.01%**；`summary.memory.reservation_slack`
  给的是最后一步瞬时值 0.68%。两者都只报不判；这里以 5.01% 作为 D6 口径的读数，
  0.68% 是 runner 的末步快照（见「限制」）。

### C4 · 代价（负结果）

- **decode step 中位数 35.63 ms** vs seat **21.32 ms** → **1.671×**。
  逐重复：1.6676 / 1.6704 / 1.6760 / 1.6764 / 1.6611，比值稳定。
- paged 的 `lpad` 不是恒为 1664：1536（90 步）/1600（320 步）/1664（215 步），
  因为 `lpad = bucket(本步最长活跃长度, 64)`，生成到末尾才到 1664。
  **按 `lpad=1664` 子集对齐后仍是 1.673×**（35.664 / 21.317），不是宽度差造成的。
- 结论：>1.5× → **撤回吞吐结论**；也远高于 1.15× 健康线。
- 同并发 `tok_s = 200.99` vs seat 355.93（0.565×）只是记录；它不是系统吞吐结论，
  也不能拿来抵消 C4（见 plan §4.3：E3 只在 A/B 都通过时跑）。

### 分解表（C4 用 decode-only 的 step 行；不是 summary 的 by_active 均值）

| 项 | 来源 | seat | paged |
|---|---|---|---|
| 同并发 `step_ms` | E2 decode 中位数 | **21.32** | **35.63** |
| 其中 `gather_ms` | E2 step 行（host 时间） | 0 | **9.44** |
| 其中 event 窗口其余 | `decode_event_ms − gather_ms` | 21.19 | 25.92 |
| host（窗口外） | `host_gap_ms` | 0.13 | 0.27 |
| KV 字节/step（记账） | gather read + gather write + attention read | ≈1.46 GiB | ≈4.16 GiB |
| `workspace_mb` | E2 实测 | 0 | **52.42** |
| 同并发 `kv_total_mb` | `kv_mb + workspace_mb` | **1456.0** | **1508.42** |

- `gather_read_mb` / `gather_write_mb` 各约 **1400 MiB/step**（lpad 1600；1536 时 1344、
  1664 时 1456），是**流量**；`workspace_mb = 52.42 MiB` 是**占用**，两者不混（P-35）。
- `decode_event_ms` 的差值（paged − seat）= **14.17 ms**；`gather_ms` 的 9.44 ms 是
  `perf_counter` 的 host 派发时间，不是 GPU 时间。事件窗口里还有约 4.7 ms 的 gather GPU
  工作没被 `gather_ms` 单独计到（见 `docs/fig_w4_step.png` 的脚注）。
- 实现里 `gather` 先 `index_select` 到中间 buffer，再 `permute + copy_` 到 workspace；
  `gather_read_mb` 只记源/目的各一遍，中间 buffer 的额外读写未计入，所以上表 4.16 GiB
  是**记账值**，不是实测 DRAM 流量。

### C3 · 数值（pass）

- E2 的 8 条 prompt × 5 次重复共 **40/40 个 request 的 `token_ids` 与 `W4-0004` 逐 id 相同**；
  两臂输出确定（同一 prompt 的 5 次重复也完全相同）。
- E0（`W4-0001`）另对 `tests/golden/qwen3-0-6b/0001` 64/64 全等。

### 显存

- `kv_mb = 1456.0`（与 seat 同预算）、`workspace_mb = 52.42`、
  `kv_total_mb = 1508.42`（+52.42 MiB，**+3.6%**）；`bytes_per_active_request = 197.71 MB`
  （seat 190.84 MB，约 +3.6%）。
- 这 3.6% 是**同并发的代价**，不是 C1 的容量收益；两者必须分开报（plan §4.2）。

## 结论（含“什么时候不划算”）

- **C2 pass、C3 pass、C4 负结果**（1.671× > 1.5×）。
- 容量结论仍然成立：同 1640 MiB 总预算下 paged 能装 28 条，seat 只能装 9 条（C1 3.111×）。
  但**稠密 gather 让 B=8 的每一步慢 67%**；按预注册规则，W4 **不报吞吐**，
  也不跑 E3（E3 只在 A/B 都通过时才跑）。
- 什么时候可能不亏：本 run 的 B=8 是 **host-bound** 的（W3 已实测每步 host 成本占主导），
  gather 每层多次小 kernel 的派发成本被放大；更大并发下 attention 的 GPU 工作变大，
  派发可能被摊薄。但这是**未验证**的推断，不是本 run 的结论；要验证必须另开 run，
  不能把 C4 的失败改写成「gather 没问题」。
- 另一条边界：分页的容量收益要求负载长度异质；uniform-2048 负对照只有 1.00×（模拟），
  长度均匀时 paged 还要多付 workspace。

## 限制与偏差

- **C4 的 lpad 口径与 plan 原文不一致**：plan 说「两边 lpad 都是 1664」，但 paged 的 `lpad`
  是**本步最长活跃长度**，实际为 1536/1600/1664。已用 `lpad=1664` 子集复核（1.673×），
  结论不变；后续报告不能再写「全程 1664」。
- `gather_ms` 是 host 时间，`decode_event_ms` 是 event 窗口（含 host 间隙）；
  C4 只用 `step_ms`，分解表的「event 窗口其余」不能当纯 GPU kernel 时间。
- 单张 RTX 4090 24 GB，bf16，`device_class=measure-cuda`，commit `9dc5211`，clean；
  5 次重复的 CV 0.13–0.17%，step 比值稳定。
- `reservation_slack` 的 D6 定义是时间加权（5.01%），但 runner 的 summary 取末步
  （0.68%）；该项不进判据，未在跑完后改代码。
- paged 的 prefill 也慢得多：5 个 prefill 步的 `prefill_ms` 中位数 **551.8 ms**
  vs seat **163.0 ms**（**3.39×**）；`summary.steps.by_active` 里的 4.38 / 1.29 ms
  是把这 5 个 prefill 步摊到 630 个 active=8 步后的均值，不是 step 0 的值。
  prefill 的成本使 `tok_s` 的 makespan 比为 1.773×，比 decode-only 的 1.671× 更大；
  C4 按 D6 只看 decode step，prefill 不在 C4 内，但它是「不报吞吐」的第二个理由。

## 下一步

- W4 以「容量成立（C1 3.111×）、碎片成立（C2 0.55%）、数值成立（C3 全等）、
  代价负结果（C4 1.671×，撤回吞吐）」收口。
- E3 不跑（预注册门禁未过）；若要探索高并发下 gather 是否被摊薄，必须另立 run，
  并明确它不是 C4 的补救。gather 的 PyTorch 级优化（去掉中间 buffer、合并零块填充）
  或 paged-attention kernel 都属于 W4 之后的新实验，不在本阶段改代码。
