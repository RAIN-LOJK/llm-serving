# 载体（carrier）

> 状态：**待 W0 填写**。本文件是 `W0-0001` 的人类可读产物，不是数字的唯一来源；
> 数字以 `results/W0/0001-carrier-probe/summary.json` 为准。

## 1. 模型

| 角色 | 模型 | revision | dtype | 参数量 | 备注 |
|---|---|---|---|---|---|
| A 骨架 | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | bf16 | ~0.6B | 纯全注意力，W1–W4 骨架 |
| B 目标 | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | bf16 | ~0.8B | 3:1 混合，W5–W7 |
| 回退 | `TinyLlama-1.1B-Chat` | 本机缓存 | fp16/bf16 | 1.1B | 仅当 A/B 卡住 |

## 2. 解析式估算（待验证）

```text
纯注意力每 token KV = 2 × n_kv_head × head_dim × dtype_bytes × n_full_layers
混合架构每 token   = 同上（n_full_layers 只有 1/4）
                   + 常数：n_linear_layers × n_v_head × key_head_dim × value_head_dim × dtype_bytes
```

- 0.8B 粗估：KV ≈ 12 KB/token（6 层），线性状态 ≈ 19 MB/序列（18 层，fp32）。
- 交叉点约 1.5k token：短上下文混合反而更“贵”。

## 3. W0 要填的表

| context | Qwen3-0.6B peak mem | Qwen3.5-0.8B peak mem | 显存比 | Qwen3 prefill/decode | Qwen3.5 prefill/decode |
|---|---|---|---|---|---|
| 512 | 待填 | 待填 | | | |
| 2k | 待填 | 待填 | | | |
| 8k | 待填 | 待填 | | | |
| 32k | 待填 / OOM | 待填 / OOM | | | |

## 4. 预测判定

| # | 预测 | 判定 | 证据 |
|---|---|---|---|
| P1 | 长上下文（≥8k）每序列显存比同量级纯注意力小 5–8× | 待测 | `W0-0001` |
| P2 | 短上下文（≤512）混合反而更贵 | 待测 | `W0-0001` |
| P3 | decode 延迟随上下文几乎不增长 | 待测 | `W0-0001` |

## 5. D1 决策

见 [decisions.md](decisions.md) D1。结论只在 `W0-0001` 的 summary + notes 完成后写。
