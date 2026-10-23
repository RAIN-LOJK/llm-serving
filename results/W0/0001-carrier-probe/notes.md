# W0-0001 · 双载体：显存/延迟 vs 上下文

- 状态：`pass`（2026-10-23 由 rainhome 评审；判定见 `verdict.json`，决策 D1）
- 设备：RTX 4090 24GB / 驱动 580.105.08 / torch 2.8.0+cu128 / transformers 5.20.2 / bf16 / sdpa
- 协议：batch=1，greedy，`max_new_tokens=128`，warmup=3，repeats=5；synthetic 随机 token id
  （`bench/loadgen.synthetic_ids`，内容无意义，只测状态与延迟形状）；上下文 512 / 2k / 8k / 32k
- 证据：`summary.json` / `metrics.jsonl`；整理与判定见 [docs/carrier.md](../../../docs/carrier.md)

## 假设与判据

- `P1`：≥8k 时每序列状态比同量级纯注意力小 5–8×。
- `P2`：≤512 时混合反而更贵（固定状态开销）。
- `P3`：decode 延迟随上下文几乎不增长（线性层无 KV 读取）。
- D1 判据：P1 ≥ 3× → 投入 W5–W7 移植。

## 观察（同配置 5 次重复的中位数）

| context | 0.6B cache | 0.8B cache | 比 | 0.6B prefill / tpot | 0.8B prefill / tpot |
|---|---|---|---|---|---|
| 512 | 56.0 MiB | 24.8 MiB | 2.25× | 19.8 / 16.55 ms | 44.5 / 22.41 ms |
| 2k | 224.0 MiB | 42.8 MiB | 5.23× | 29.5 / 16.73 ms | 44.1 / 21.59 ms |
| 8k | 896.0 MiB | 114.8 MiB | 7.80× | 152.9 / 19.97 ms | 135.4 / 21.74 ms |
| 32k | 3584.0 MiB | 402.8 MiB | 8.90× | 1282.5 / 18.31 ms | 729.5 / 21.89 ms |

> cache 列用 `cache_prefill_mb`：prefill 结束时、恰好 N 个 token 的状态。`cache_mb`（decode 结束后）
> 比它多 `(out_tokens−1)` 个 token 的 KV——0.8B 恒多 1.49 MiB，0.6B 因早停多 0–13.9 MiB。
> 早期稿把 0.8B 这 1.49 MiB 当作“固定开销”，实为 127 个 decode token，已更正。

- 权重：1136.9 MiB（0.6B）/ 1435.1 MiB（0.8B，text-only）。32k 进程峰值 5578 / 4002 MiB（1.39×），
  差值 ≈ cache 差 3.11 GiB——**峰值口径看不出架构差异，判据必须用每序列 cache**。
- 重复性：同配置 TTFT CV ≤ 8.1%、TPOT CV ≤ 6.8%、cache 字节完全一致。
- `summary.json.warnings` 里的 `ttft_ms CV=1.41` 是跨上下文/跨模型混合口径的 artifact，不是噪声。

## 与预测的偏差

- **P1（≥8k 小 5–8×）成立**：7.80×（8k）/ 8.90×（32k），渐近比 `112/12 = 9.33×`。
- **P2（≤512 混合更贵）不成立，方向相反**：512 上混合已小 2.25×；交叉点 ≈193 token
  （prefill 口径，与解析式 `18.84 / (112 − 12) KiB` 一致），低于最小测试桶。
- **P3（decode 不随上下文增长）数值成立但不可归因**：0.8B TPOT 22.41 → 21.89 ms（−2.3%）；
  0.6B 也只 +10.6%。batch=1 时 decode 被 kernel launch + 权重读取支配（0.6B 32k 的 KV 读取理论值
  3.6 ms，总耗时 18 ms），分辨不出线性层“无 KV 可读”的贡献。

## 结论（含“什么时候不划算”）

- D1 接受（预注册判据 3×，实测 7.8–8.9×）：W5–W7 投入移植；收益来源是**同显存下的并发数**
  （32k 每序列省 3.11 GiB），不是单请求更快。
- **不划算的边界**：上下文 < ≈193 token 时混合更贵；≤2k 时 prefill 慢 1.5–2.2×；
  batch=1 的短上下文没有收益。W5–W7 的论证必须限定在「长上下文 + 多并发」。

## 限制与偏差

- batch=1，未测并发；P3 的机制归因需要 batch≥8 才能验。
- 0.6B 在 512 / 8192 两格 2–3 token 后 EOS（synthetic 随机 id 触发），TPOT 只由 1–2 步平均；
  重复性仍好，但这两格的 decode 结论要弱化。
- synthetic 随机 token 不代表真实 prompt 分布，只用于状态字节与延迟曲线。
- 未做“有/无 `flash-linear-attention`”的对照（本机装有 fla 0.5.2 / causal-conv1d 1.7.0）。
- 跨上下文比较用 prefill cache；final cache 含 decode token（0.8B +1.49 MiB），不可拿它除以 prompt 长度。
- 未测 2B/4B 档；`Qwen3.5-0.8B` 的选择是尺寸友好，不代表其它档位。

## 下一步

- **W1**：Qwen3-0.6B 装进自己的解码器并对齐（贪心 64 token 逐 id 全等）。
- **W2**：CV 改成按配置分组；P3 归因加一组 batch≥8 的探测（待定）。
- **W5**：按 D1 推进全注意力层改造。
