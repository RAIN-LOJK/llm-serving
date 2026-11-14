# W7-0019 · E3 交叉点/P2/P3（pure 0.6B）：synthetic [512,2k,8k] / N=24 / B=8 / 4200 块

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W7/0019-e3-pure-b8-n24`）
- 类型：`scenario`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `pure_attention_is_the_paired_baseline_for_the_crossover`

## 观察

- **invalid（P-60）**：旧 TPOT 用 `finish-first`，把其他请求的 prefill 算进本请求；仅保留为旧归因证据。
- 由 `W7-0029` 替代；旧 summary 的 TPOT 不作判据。
- 关键数字（引用 summary.json）：
- 与预测的偏差：

## 结论（含“什么时候不划算”）

（待填）

## 限制与偏差

- 设备 / dtype / 样本量 / 已知噪声：

## 下一步

（待填）
