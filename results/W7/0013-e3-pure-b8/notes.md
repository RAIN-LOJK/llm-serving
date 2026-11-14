# W7-0013 · E3 交叉点/P2/P3（pure 0.6B）：synthetic [512,2k,8k] / B=8 / 4200 块

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W7/0013-e3-pure-b8`）
- 类型：`scenario`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `pure_attention_is_the_paired_baseline_for_the_crossover`

## 观察

- **invalid**：`requests=8` 被 3 个 context（512/2k/8k）分掉，`peak_concurrency=4`，不是 B=8；
  本 run 只保留为口径错误证据，不作 E3 B=8 基线。
- 关键数字（引用 summary.json）：`tok_s=112.33`，`peak_concurrency=4`，`out_tokens_total=397`。
- 与预测的偏差：B=8 未填满；由 `W7-0019`（N=24，每 context 8 条）supersede。

## 结论（含“什么时候不划算”）

（待填）

## 限制与偏差

- 设备 / dtype / 样本量 / 已知噪声：

## 下一步

（待填）
