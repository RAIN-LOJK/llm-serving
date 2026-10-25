# W2-0003 · SDPA attention 路径：long-prefill 归因（对 W2-0002）

- 状态：`pass`（reviewer: rainhome, 2026-10-25）
- 配置：`config.toml`（运行目录 `results/W2/0003-sdpa-attn`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `attn_path_explains_long_prefill`

## 观察

- 关键数字（引用 summary.json）：160 行 / 0 错误；TTFT **17.37** ms（HF `W2-0001` 20.02）、
  TPOT **15.08** ms（HF 16.78）、e2e 1996 ms、peak 1193 MB、driver 1716 MB。
- 判据 4/4 pass：`ttft_cv_max` 6.1%（medium 桶）、`tpot_cv_max` 2.7%、`memory_match_ratio` 0.991
  （最差 context 0.978）、`decode_speed_vs_hf_min` **1.113**（32 条 prompt 配对）。
- 归因（与 `W2-0002` 逐 prompt 配对，同 workload）：TTFT short 19.7→17.4、medium 19.7→17.0、
  **long 31.8→19.5（−39%）**；TPOT 17.06/16.92/17.03 → 14.83/15.13/15.14。
- 与预测的偏差：预估 prefill 降到 12–20 ms，实测 19.5（long 桶）/ 25.0（2048 合成）；
  TPOT 原以为不变，实际快了 11–13%——这部分不是 SDPA 的功劳，是 `enable_gqa` 去掉了
  每步 28 层 × 16 头的 K/V `repeat_interleave` 复制。
- 后端确认（P-19 要求）：default = **flash**（2048 合成 prefill 25.4 ms），
  `SDPBackend.MATH` = 125.5 ms，`EFFICIENT_ATTENTION` 无可用 kernel。

## 结论（含“什么时候不划算”）

- `W2-0002` 的 long-prefill 1.49× **几乎全部来自 attention 路径**（31.8→19.5，−39%）。
  换 SDPA 后三个桶的 TTFT 都优于 HF（0.86–0.90×），"prefill 是负项"这条结论作废，
  本 run 取代 `W2-0002` 成为 W3 的对照基线。
- decode 反超 11%（1.113×）**不能归给 SDPA**：那是 `enable_gqa` 省掉了每步的 K/V 复制。
  两者在同一次提交里，报告里要分开写，否则下一个项目会以为"换 SDPA 能提速解码"。
- 什么时候不划算：显存不再被 `T²` 工作区支配（最差 context 0.978 vs 之前 0.889），
  但计算仍是 O(T²)。超过 2k 之后 flash 的时间重新成为主要项；更要紧的是 **W3 的批处理会走显式
  bool mask 路径**（左 pad），flash 的可用性会下降，prefill 的优势要重新测，不能直接外推。

## 限制与偏差

- 设备 / dtype / 样本量：RTX 4090 / bf16 / 32 prompt × 5 次 / `max_new_tokens=128`；
  `device_class=measure-cuda`，与 `W2-0001` 同设备同 dtype 同 prompt hash，配对有效。
- 口径差异：我们的 TTFT 含首个 token 的 argmax，HF 的到 prefill 结束为止——**对我们不利**，
  在这个前提下仍然更快。
- `peak_mem_mb` 中位数被短上下文主导（1193 MB ≈ 权重 + 小 KV），长上下文要看 `by_context`。
- 后端可用性依赖 CUDA/dtype/shape/torch 版本（此处 torch 2.8）：换设备必须重测，别外推结论。

## 下一步

- W3 以本 run 为对照基线；批处理引入左 pad 后测一次 prefill，确认 flash 是否仍可用。
- 若 W3 发现 prefill 重新成为瓶颈，再考虑 chunked prefill（掩码已支持 `past_len>0, T>1`）。
