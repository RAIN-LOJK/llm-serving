# W4-0001 · E0 数值门：paged 64 token 贪心输出与 golden 逐 id 全等

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W4/0001-paged-numeric-gate`）
- 类型：`microbench`；workload：`prompts`（`prompt_ids = ["s01"]`）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `paged_kv_is_bitwise_equivalent_to_seat_on_golden_prompt`
- C3/E0 口径（D6）：paged 的 64 token 贪心输出必须与
  `tests/golden/qwen3-0-6b/0001` 的 `greedy_ids[0][:64]` 逐 id 全等；这是 E1/E2 的前置门，
  不过就不许跑性能 run。
- 该 run 只判数值，不判延迟/显存；`type=microbench`、`record_only`。

## 观察

- `python scripts/check_paged_golden.py results/W4/0001-paged-numeric-gate` → `s01 pass 64/64 ids`（exit 0）。
- summary：TTFT **213.96 ms**、TPOT **30.43 ms**、e2e **2131 ms**、`tok_s` 30.03（1 请求、
  warmup=0，只作记录）。
- 账：池 `kv_mb = 112.0`（64 块 × 1.75 MiB）、末尾 `kv_live_mb ≈ 11.0`、
  `workspace_mb = 0.504`、`kv_total_mb = 112.5`；step 的 `lpad` 从 64 升到 128，
  `blocks_in_use = 7`，`written_blocks = 3→7`。
- `frag_internal = 9.8%` 是 `budget=64` 的单请求尾块，不是 C2 的样本；C2 在 E2 判。
- 与预测：无独立预测。`lpad` 随活跃长度按 64 桶增长，符合 D6「gather 宽 = 本步最长活跃长度」。

## 结论（含“什么时候不划算”）

- **C3/E0 pass**：块表、专用零块 + 逻辑 mask、gather、logits 散回整条 paged 路径在 golden
  prompt 上与 seat/golden 逐 id 一致。允许进入 E1/E2。
- 这份 run 不能用于性能判断：单请求、无重复、`timer=auto`，且 213.96 ms 的 prefill 里包含
  paged 池的初始化；只能说明“数值正确”。

## 限制与偏差

- `warmup=0, repeats=1`：确定性贪心数值门，不测方差；runner 的
  `measure protocol deviates...` warning 就是这条，已在此说明。
- 无 paired baseline：E0 的分母是 golden 文件，不是另一个 run；warning 不适用于本 run。
- 设备：单张 RTX 4090 24 GB，bf16，`device_class=measure-cuda`，commit `9dc5211`，工作树 clean。

## 下一步

- `W4-0002`（seat fill）→ `W4-0003`（paged fill），再做 E2 0004/0005。
