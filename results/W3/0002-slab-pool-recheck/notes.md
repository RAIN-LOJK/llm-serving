# W3-0002 · E0b slab 池化后单请求复核：kv_mb 记池的总预留

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0002-slab-pool-recheck`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `pool_reservation_explains_the_allocator_peak`

## 观察

- `n_rows=645`、`n_errors=0`；`impl=continuous`、`max_batch=1`、`measure-cuda`、`dirty=false`。
- `memory_match_ratio=0.963`（ctx=1645，分子为池总预留；target 0.90；`peak_over_driver=0.731`）。
- `tpot_ms.median=21.2118`（W2-0007 21.271，**-0.28%**，落在 ±2%）；`ttft_ms.median=21.1204`（只记录）。
- warning `no paired baseline run_id` 是该格故意不写 `[baseline]` 的设计，不是失败。

## 结论（含“什么时候不划算”）

建议：**pass**。池化后 `kv_mb` 记池的总预留，`memory_match_ratio` 不再含账外席位；
`max_batch=1` 时池路径与旧单请求路径等价（TPOT 漂移 -0.28%），E0b 门禁通过。

## 限制与偏差

- 只测 `max_batch=1`，不覆盖并发；池的定长预留与负载无关，真正的回收/按需分配在 W4。
- memory 比值是池预留口径，不能与 W2 的“单请求 slab”绝对值直接混读。

## 下一步

E1（`W3-0017–0019` + `0006/0007`）与 E2。
