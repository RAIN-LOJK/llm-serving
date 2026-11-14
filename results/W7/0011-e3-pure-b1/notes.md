# W7-0011 · E3 交叉点/P2/P3（pure 0.6B）：synthetic [512,2k,8k] / B=1 / 936 块

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W7/0011-e3-pure-b1`）
- 类型：`scenario`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `pure_attention_is_the_paired_baseline_for_the_crossover`

## 观察

- **service TPOT 有效证据**：`tpot_ms` 是 wall inter-token latency，包含其他请求 prefill；不用于 decode 成本归因。
- 关键数字（引用 summary.json）：
- 与预测的偏差：

## 结论（含“什么时候不划算”）

（待填）

## 限制与偏差

- 设备 / dtype / 样本量 / 已知噪声：

## 下一步

（待填）
