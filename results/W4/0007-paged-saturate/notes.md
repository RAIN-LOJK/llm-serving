# W4-0007 · E3 同预算系统吞吐（paged）：frozen 32 / poisson / B=64 / 820 块

- 状态：`reviewed`；verdict = `invalid`（over-budget，R5）
- 配置：`config.toml`（运行目录 `results/W4/0007-paged-saturate`）
- 类型：`scenario`；workload：`prompts`（frozen 32）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `paged_system_throughput_at_equal_budget_beats_seat`
- E3 paged 臂的预注册配置：`kv_blocks=820`、`max_batch=64`、与 `W4-0006` 同 trace。
- 同预算门禁：**峰值** `kv_total = kv_mb + workspace_mb` 必须 ≤ 1640 MiB（D6/R5）。

## 观察

- `saturated = true`；`tok_s = 321.46`、decode-only `423.06`、`req_s = 5.99`、
  makespan `21.353 s`；`peak_concurrency = 36`。
- 384/384 request 的 `out_tokens == budget`，无提前 EOS。
- **R5 触发**：峰值出现在 step 51，`active=36`、`lpad=1600`、
  `workspace_mb=226.81`、`kv_mb=1435.0` → **`kv_total=1661.81 > 1640`**；
  共有 148 个 step 超过 1640。
- 原因：配置按「26 活跃行」估 workspace（约 189 MiB），但动态负载在短请求/混合 budget
  下峰值到了 36 活跃行；`summary.memory.kv_total = 1455.16` 是**最后一步**（active=0）的
  快照，不是峰值，不能拿它当预算读数。
- trace 与 `W4-0006` 相同（`d6391043...`）；prompt hash 相同（`7548f3a6...`）。

## 结论（含“什么时候不划算”）

- 这条 run **不是有效的同预算 E3 证据**：它在峰值多用了 21.8 MiB（1661.8 vs 1640），
  所以 `0.964×` 不能拿来和 seat 比。
- 按 plan R5，保留为 **over-budget record**，不静默改预算；下一 run 用
  `W4-0008`（800 块，峰值 1626.8 ≤ 1640）作为 **pre-fix** paged 臂；prefill 修复后
  由 `W4-0009` 取代。
- 它唯一的科学价值是：**动态峰值活跃数高于 fill-only capacity（36 vs 28）**，
  静态按 26 行估 workspace 会顶穿预算；这是 P-38 的来源。

## 限制与偏差

- 预注册的 820 块本身是按 26 活跃行算的；实际 36 行说明 E3 的 paged 预算校验必须用
  动态峰值，而不是 fill capacity。
- 单张 RTX 4090 24 GB、bf16、commit `c301b26`、clean；3 次重复。
- 不要引用 `summary.memory.kv_total_mb` 作峰值；峰值从 step rows 取
  `max(kv_mb + workspace_mb)`。

## 下一步

- `W4-0008` 是 pre-fix 同预算 paged 臂；prefill 修复后的 E3′ 看 `0006` vs `0009`
  （1.074×）。0007 本身只作 over-budget 记录。
