# W2-0001 · HF generate 参考臂：TTFT / TPOT @ 32 prompt × 128 token

- 状态：`pass`（reviewer: rainhome, 2026-10-24）
- 配置：`config.toml`（运行目录 `results/W2/0001-hf-reference`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `decode_speed_vs_hf`

## 观察

- 关键数字：160 行（32 prompt × 5 次），0 错误；TTFT 中位数 20.02 ms、TPOT 16.78 ms、
  e2e 2213 ms、peak 1192 MB、driver 1706 MB；权重 1136.9 MB（bf16 596M 参数）。
- 稳定性：桶级 CV short 7.2% / medium 3.4% / long 5.9%（TTFT），TPOT 1.9% / 1.1% / 2.7%。
- 与预测的偏差：`ttft_cv_max` 的 worst_prompt_cv 是 0.173（短桶单条 19 token），
  说明短 prompt 的固定开销会让单条 CV 偏大——判据用桶内中位数是对的。

## 结论（含“什么时候不划算”）

本 run 只提供参考臂，不作机制结论：它是 `W2-0002` 的配对基线（同权重、同 prompt、
同 dtype、同设备）。唯一可下的判断是：HF `generate` 在 0.6B 单请求上 TPOT ≈16.8 ms，
这给出了“自实现不得慢于 21 ms”的判失败线（0.8×）。

## 限制与偏差

- 设备 / dtype / 样本量 / 已知噪声：

## 下一步

（待填）
