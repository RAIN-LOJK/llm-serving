# W4-0009 · E3′ 同预算系统吞吐（paged，prefill 修复后，800 块）

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W4/0009-paged-saturate-fixed`）
- 类型：`scenario`；workload：`prompts`（与 `W4-0006` 同一份 frozen 32）
- 证据：`summary.json` / `metrics.jsonl`
- 关系：`supersedes = "W4-0008"`（0008 是 pre-fix paged 记录）

## 假设与判据

- `paged_system_throughput_at_equal_budget_beats_seat`
- **E3′**：在 `PagedPool.write_prefill` 改为 slot mapping 之后，用与 `W4-0006`
  完全相同的 workload（同 trace / λ=30 / 128 请求 / budgets）重跑 paged 臂。
- seat 路径（`KvSlab` / `SlabWriteSink`）没有改动，所以直接对 **`W4-0006`**
  比较；不重跑 seat。
- `kv_blocks=800`（1400 MiB KV），峰值 workspace 226.8 → 1626.8 ≤ 1640；两臂都必须
  `saturated=true`。E3′ 是 **post-fix、post-hoc record-only**，不改变 W4 的 C4。

## 观察

- **`tok_s = 358.18`** vs seat `333.44` → **1.0742×**；两臂都 `saturated=true`。
- decode-only `423.40` vs seat `386.96` → **1.094×**；`req_s = 6.679` vs `6.218`；
  makespan **19.16 s** vs `20.59 s`；动态峰值并发 36 vs 9。
- E3 的 per-request prefill 中位数 **22.97 ms** vs seat **21.39 ms**（约 1.074×）；
  pre-fix `W4-0008` 是 **76.41 ms**。这与 E2 微基准的 551.8 → 176.8 ms 方向一致：
  prefill 已经回到 seat 附近。
- 预算：峰值在 step 55，`active=36`、`lpad=1600`、`kv_mb=1400.0`、
  `workspace_mb=226.81` → **`kv_total=1626.81 ≤ 1640`**，没有 step 超预算。
  `summary.memory.kv_total_mb = 1420.16` 是末步（active=0）快照，不是峰值。
- EOS：384/384 request 的 `out_tokens == budget`，无提前 EOS；trace 与 `W4-0006`
  相同（`d6391043...`），prompt hash 相同（`7548f3a6...`）。
- trace 级 `tok_s` CV = 0.43%（3 次重复）；per-prompt TTFT/e2e CV 高是开放环饱和的
  预期现象，不套 W2 的 0.10 目标。

## 结论（含“什么时候不划算”）

- **prefill 修复后，E3′ 从旧的 0.975× 翻到 1.074×**：同预算、同 trace、两臂饱和，
  paged 的系统吞吐比 seat 高 **7.4%**。这验证了旧 E3 的负结果主要是 prefill 的
  3.39× 造成的，而不是容量收益不存在。
- decode-only 只小幅变化（1.094× vs 0008 的 1.103×）；主要改善来自 prefill
  （76.4 → 23.0 ms/request）。所以 **slot mapping 是有效的，且它只需要改 prefill，
  不需要 paged-attn kernel**。
- **C4 仍然是 1.671×**：本次修复不碰 decode/gather，W4 的 C4 负结果和「不报吞吐」
  结论仍然成立；0009 是一个 **post-fix、post-hoc** 的新实验，不是 W4 C4 的 pass。
- 什么时候不划算：单一 λ=30、单一 workload、128 请求；没有 λ sweep，也没有验证
  更短/更长 prompt 分布。7.4% 是这一个工作点的 record-only 结果，不能外推成
  「所有负载都提升 7.4%」。

## 限制与偏差

- E3′ 是 post-hoc，没有预注册的通过线；`0009/0006` 是直接对比，但 seat 是早先
  `0006` 的数字（seat 路径未改动；commit 不同，配对看 trace/dtype/device_class）。
- `0007`（820 块）峰值 1661.8 > 1640，是 R5 over-budget record；`0008` 是 pre-fix
  paged record（0.975×），由本 run `supersedes`。
- 单张 RTX 4090 24 GB、bf16、`device_class=measure-cuda`，commit `b8fa5c1`，clean。
- `summary.memory.kv_total_mb` 是末步值；本 notes 的 1626.81 是 step 峰值。

## 下一步

- 如果要把这个结果升级成正式结论，需要新 ADR/预注册（明确通过线、λ/长度 sweep），
  不能直接把 post-hoc 0009 当作 W4 的 C4 pass。
- 若要继续追吞吐，下一步是把 gather/decode 的 1.67× 也压下来；prefill 已不再是主瓶颈。
