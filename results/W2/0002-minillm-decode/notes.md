# W2-0002 · 自实现解码器：TTFT / TPOT @ 同一 prompt 集

- 状态：`pass`（reviewer: rainhome, 2026-10-24）
- 配置：`config.toml`（运行目录 `results/W2/0002-minillm-decode`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `decode_speed_vs_hf`

## 观察

- 关键数字（引用 summary.json，单位 ms）：TTFT 中位数 **20.02**（HF `W2-0001` 20.02），
  TPOT **17.10**（HF 16.78），e2e 2260（HF 2213），peak 1197 MB（HF 1192），driver 1716 MB（HF 1706）。
- 判据：`ttft_cv_max` 6.8%、`tpot_cv_max` 3.6%、`memory_match_ratio` 0.988、
  `decode_speed_vs_hf_min` **0.976**（32 条 prompt 配对）—— 四条全 pass，无 warning。
- 分桶（ours/HF，中位数比值）：TTFT short 1.00× / medium 1.01× / **long 1.49×**；
  TPOT short 1.04× / medium 1.00× / long 1.03×。
- 与预测的偏差：decode 与 HF 同级（−2.4%），符合预期（都是单请求、bf16、同一权重）；
  长上下文 prefill 慢 1.49× 是**新增的已知代价**——手写 attention 物化 `T×T` 分数并显式构造因果掩码，
  HF 走 SDPA。最慢的 3 条 prompt 是 l02(1.20×)、s08(1.08×)、l01(1.08×)，都在 128 token 输出上。

## 结论（含“什么时候不划算”）

- 解码速度不值得再优化：单请求 bf16 下与 HF 差 2.4%，已经落在测量噪声量级；
  真正要动的是 **prefill**——≥1k 上下文慢 1.49×，且随 T 线性恶化（`T²` 分数矩阵）。
- 什么时候不划算：短上下文（≤512）本实现与 HF 无差别，任何额外的调度/分页开销都必须是净收益；
  长上下文（≥1k）当前是明确的负项，W3/W4 的分页 KV 与 kernel 优化要以“补掉 prefill 差距”为前提，
  否则同规模下加调度只会更慢。

## 限制与偏差

- 设备 / dtype / 样本量：RTX 4090 / bf16 / 32 prompt × 5 次重复 / `max_new_tokens=128`；
  `device_class=measure-cuda`，与 `W2-0001` 同设备同 dtype，可比。
- 已知噪声：短桶最差单条 prompt 的 TTFT CV 为 **0.219**（19 token 级别，固定开销占比大），
  判据取桶内中位数（6.8%）所以通过；`peak_over_driver=0.698` 是 CUDA context 不在公式账内，
  判据只用 torch 分配器峰值。
- 显存口径：`memory_match_ratio` 按 context 取中位数 0.988，但**最长 context（1517 token）是 0.889**，
  低于目标——差值就是 `T²` attention 工作区，属于实现性质，已在 `by_context` 里逐条列出，不做平均掩盖。

## 下一步

- W3：连续批处理。注意本 run 是 batch=1 的基线，`batch_sizes=[1]`，真批处理要等调度器。
- 若要修 prefill：先换 SDPA/flash 注意力，或至少不在每个 prefill 里显式构造 `T×T` 掩码；
  修完重跑本配置，`decode_speed_vs_hf_min` 之外的 TTFT 比值应回到 1.0× 附近。
