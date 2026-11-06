# W4-0003 · E1 容量（paged）：W3-like max-2048 / fill-only / 800 块

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W4/0003-paged-fill-max2048`）
- 类型：`fill`；workload：`prompts`（32 条 frozen prompt，与 `W4-0002` 同一 trace）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `paged_capacity_beats_seat_under_the_same_total_budget`
- C1（D6）：同总预算 **1640 MiB（KV + workspace）**、W3-like max-2048、fill-only 的
  allocator capacity `paged / seat` **≥ 1.50×**；`< 1.20×` 记负结果。
- paged：`block_size=16`、`kv_blocks=800`、`max_batch=64`（调度上限，不是显存上限）。
- 同 trace 门禁：`trace_sha256` 必须与 `W4-0002` 相同。

## 观察

- **`capacity.peak_concurrency = 28`** vs seat **9** → **3.111×**；
  `acceptance: max_concurrency_speedup_min = pass`（value 3.111，target 1.50）。
- 账：`kv_mb = 1400.0`（800 × 1.75 MiB）、`workspace_mb = 201.97`（fill 不 gather，
  `workspace_estimated = true`，按附录 C 公式估算）、
  **`kv_total_mb = 1601.97 ≤ 1640`**。
- `capacity_per_gib = 17.90` vs seat `5.62`；`reservation_slack = 2.68%`（最后一步的瞬时值；
  D6 只报不判，C2 只看 internal frag）。
- 28 < `max_batch=64`，所以容量由**块池**（allocator）顶住，不是调度上限顶住；
  这正是 D6 要量的 allocator capacity。
- 同 trace 模拟门禁（与配置相同的 800 块）：
  `python scripts/sim_paged_capacity.py --trace results/W4/0003-paged-fill-max2048/metrics.jsonl --paged-blocks 800 --expect-concurrency 28`
  → **`sim=28 expected=28 delta=0 gate=PASS(±1)`**。
- `trace_sha256 = 2c047181...`、`prompt_sha256 = 7548f3a6...`，与 `W4-0002` 相同。
- 负对照：同一模拟器在 1640 MiB 下 uniform-2048 为 **6 vs 6 = 1.00×**（只报不判）；
  见 `figures/fig_w4_capacity.png`。

## 结论（含“什么时候不划算”）

- **C1 pass：3.111×（28 vs 9）**。同总预算下，分页把「按 trace 最坏长度预留」换成
  「按每条请求的上界预留」，长度异质时能多装 3.11 倍。
- 收益不是「分页创造了显存」，而是把 seat 每行 72.4% 的空闲行宽换成了按请求的块。
  长度越均匀收益越小：uniform-2048 是 **1.00×** 的负对照，不是失败线。
- 预算越松、长度越均匀、请求数越少，比值都会下降；C1 只是 1640 MiB 这条竖线，
  报告主图是 `figures/fig_w4_capacity.png`。

## 限制与偏差

- **workspace 是估算值**：fill-only 只 prefill + 预留，不 decode/gather，池不会分配 gather
  workspace；`kv_total = 1601.97` 里的 201.97 MiB 来自附录 C 公式，不是实测。
  实测的 workspace 见 `W4-0005`（B=8、52.42 MiB）；28 活跃行的真实 workspace 没有在本 run 量到。
- `warmup=1, repeats=3`：确定性 fill，不需要默认 3/5；3 次 `peak_concurrency` 都是 28。
- 800 块是**保守值**：预算反解约 822 块（29 条），配置固定 800，所以门禁比的是 28 而不是 29；
  这是 D6 预注册的口径，不是事后挑选。
- 设备：单张 RTX 4090 24 GB，bf16，`device_class=measure-cuda`，commit `9dc5211`，clean。

## 下一步

- C1 已过；进入 E2（`W4-0004` seat / `W4-0005` paged）量 C2/C4。E3 是否跑见 `W4-0005` 的 C4 结论。
