# W3-0001 · E0a 重构无害性：席位 API 落地后单请求路径不动数字

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0001-single-request-recheck`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `seat_api_does_not_move_w2_numbers`

## 观察

- `n_rows=160`、`n_errors=0`；`device_class=measure-cuda`、`git.dirty=false`（commit `552c72e`）。
- 四条判据全 pass：`ttft_cv_max=0.009`、`tpot_cv_max=0.0051`、`memory_match_ratio=0.992`、
  `decode_speed_vs_hf_min=1.018`（同实现配对，0.97–1.03）。
- `ttft_ms.median=21.184`（W2-0007 21.847，-3.0%）；`tpot_ms.median=20.828`（W2-0007 21.271，-2.08%）。

## 结论（含“什么时候不划算”）

建议：**pass**。席位 API 只加了 `activate/freeze/advance(all_rows=)` 与逐行 mask，
单请求路径的 op 序列和数字没有漂移；TPOT 的 -2% 是同机不同会话的抖动（W2-0007 自身 CV 1.69%），
且方向是变快，不是回归。E0a 门禁通过后才允许 E0b 与性能 run。

## 限制与偏差

- 只覆盖 model 层（`model/slab.py` 的席位 API），**不覆盖 slab 池化**；池化归 E0b。
- 绝对 TPOT 差是跨会话的 host 抖动，不能当“更快 2%”引用；E0a 判的是配对 decode 比值。

## 下一步

E0b（`W3-0002`）：池化后的内存账与单请求 TPOT 漂移。
