# W4-0008 · E3 同预算系统吞吐（paged，800 块，pre-fix；已被 0009 supersede）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W4/0008-paged-saturate-800`）
- 类型：`scenario`；workload：`prompts`（与 `W4-0006` 同一份 frozen 32）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `paged_system_throughput_at_equal_budget_beats_seat`
- E3 的 **pre-fix** paged 臂：`kv_blocks=800`（1400 MiB KV）、`max_batch=64`、与
  `W4-0006` 同 trace / λ=30 / 128 请求；两臂都必须 `saturated=true`。
- 同预算门禁：峰值 `kv_total = kv_mb + workspace_mb ≤ 1640`（D6/R5）。
- E3 是 **post-hoc record-only**（C4 失败后按用户指示补跑）；不改变 W4 的 C4 结论。
- **prefill slot-mapping 修复后，本 run 由 `W4-0009` 取代**（0009 的 `supersedes` 指向本 run）；
  0008 保留为“修复前”的对照记录，不再作为 post-fix 吞吐证据。

## 观察

- `saturated = true`；`tok_s = 325.19` vs seat **333.44** → **0.975×**。
  decode-only `426.86` vs seat `386.96` → **1.103×**；`req_s = 6.064` vs `6.218`；
  makespan **21.107 s** vs `20.585 s`。
- 动态峰值 `peak_concurrency = 36`（seat 为 9）；最后到达时队列深度 49–52（seat 99）。
- 预算：峰值在 step 52，`active=36`、`lpad=1600`、`kv_mb=1400.0`、
  `workspace_mb=226.81` → **`kv_total=1626.81 ≤ 1640`**，没有 step 超预算。
  `summary.memory.kv_total_mb = 1420.16` 是最后一步（active=0）的快照，不是峰值；
  报告峰值必须从 step rows 取。
- EOS：384/384 request 的 `out_tokens == budget`，无提前 EOS；trace 与 `W4-0006`
  相同（`d6391043...`），prompt hash 相同（`7548f3a6...`）。
- trace 级 `tok_s` CV = 0.27%（3 次重复）；per-prompt TTFT/e2e CV 高是开放环饱和的
  预期现象，按 docs/experiments §5.1 不套 W2 的 0.10 目标。

## 结论（含“什么时候不划算”）

- **pre-fix E3 没有显示系统吞吐提升**：同预算、同 trace、两臂都饱和，paged **0.975×** seat。
  这是 prefill per-block 拷贝（551.8 ms vs 163.0 ms）拖累的结果。
- **post-fix `W4-0009` 是 1.074×**（358.18 vs 333.44 tok/s；prefill 修复到 23.0 ms/request）。
  所以 0008 的 0.975× 是“修复前”的 before record，不是最终结论；最终 post-fix 结论看 0009。
- decode-only 是 **1.103×**，说明 3.11× 容量确实在纯 decode 阶段兑现了一部分；
  但 prefill（`551.8 ms` vs `163.0 ms`，**3.39×**）和调度/排队把整体 `tok_s` 拉回到
  略低于 seat。**容量收益被 prefill + gather 成本吃掉。**
- 这对“直接上 paged-attn kernel”的判断很关键：decode kernel 只能改善 1.103× 那一段，
  **无法修复 prefill 的 3.39×**；即使 kernel 把 decode 再加快，整体系统吞吐也未必赢。
  要做，应该同时处理 prefill 写块路径，并且另立新实验，而不是把它当作 W4 的补丁。
- E3 是 post-hoc record-only，不能反过来改变 C4 的 1.671× 负结果；W4 仍然**不报吞吐**。

## 限制与偏差

- 单一 λ=30、单一 workload、128 请求、3 次重复；没有 λ sweep，不能外推到其他到达率。
- `0007`（820 块）峰值 1661.8 > 1640，是 R5 over-budget record；本 run 才是有效的
  同预算 paged 臂（P-38）。
- 单张 RTX 4090 24 GB、bf16、`device_class=measure-cuda`，commit `c301b26`，clean。
- `summary.memory.kv_total_mb` 是末步值；本 notes 的 1626.81 是 step 峰值。

## 下一步

- W4 以「C1 3.111× / C2 0.55% / C3 全等 / C4 1.671× 负结果 / E3 0.975×（record-only）」
  收口；不报吞吐。
- 若要追吞吐，优先级应是 **prefill 写块 + gather 成本** 的新实验，而不是单换
  decode attention kernel。

## 附：prefill 3.39× 的根因（代码层，E2/E3 同一条路径）

- seat `SlabWriteSink.set_kv`：每层对 slab 的**一行**做 2 次连续 slice 写（K/V），
  一个 prompt 共 28×2 = **56 次**大拷贝。
- paged `PagedWriteSink` → `PagedPool.write_prefill`：按**物理块**循环，每个 block、
  每层、K/V 各一次小拷贝。E2 的 8 条 long prompt 共 **715 个 reserved blocks**
  → `715 × 28 × 2 = 40,040` 次小拷贝；同 workload 的 seat 是 `8 × 56 = 448` 次。
- E2 prefill 差值 `551.8 − 163.0 = 388.7 ms`，摊到 40,040 次约 **9.7 µs/次**。
  源张量是 `k[0].permute(1,0,2)`（非连续），每次是 strided copy；Python 循环、
  block-table 索引和 `_check_block_ids` 也都在 host 上。**不是 attention 算力，
  也不是 KV 字节数**——同样的字节，seat 用 56 次连续写入，paged 用 4 万次 16-token 小块。
- 候选修复（属于新实验，不是 W4 补丁）：prefill 先算 `slot_ids`
  （`block_table[t//bs]*bs + t%bs`），再用**一次 advanced indexing** 写整层 K/V，
  把 4 万次降到 56 次；这是 nano-vllm 的 `slot_mapping` 思路，纯 PyTorch，
  数值上只搬字节、应逐 bit 不变，但仍要用 E0 复核。它比 decode paged-attn kernel
  更直接地打中 E3 的瓶颈。
