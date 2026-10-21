# 载体（carrier）

> 状态：**§2 解析式估算已完成（2026-10-21）**，§3 实测表待 `W0-0001`。
> 本文件是 `W0-0001` 的人类可读产物，不是数字的唯一来源；
> 实测数字以 `results/W0/0001-carrier-probe/summary.json` 为准。

## 1. 模型

| 角色 | 模型 | revision | dtype | 参数量 | 备注 |
|---|---|---|---|---|---|
| A 骨架 | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | bf16 | ~0.6B | 纯全注意力，W1–W4 骨架 |
| B 目标 | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | bf16 | ~0.8B | 3:1 混合，W5–W7 |
| 回退 | `TinyLlama-1.1B-Chat` | 本机缓存 | fp16/bf16 | 1.1B | 仅当 A/B 卡住 |

## 2. 解析式估算（未实测）

> 由 `scripts/probe_checkpoint.py --offline --write` 从 pin 住的 `config.json` 算出，
> 机器可读版本见 [carrier_estimates.json](carrier_estimates.json)。**这是预测，不是测量**；
> 实测数字以 `results/W0/0001-carrier-probe/summary.json` 为准。

```text
纯注意力每 token KV = 2 × n_kv_head × head_dim × dtype_bytes × n_full_layers
混合架构每 token   = 同上（n_full_layers 只有 1/4）
                   + 常数：n_linear_layers × n_v_head × key_head_dim × value_head_dim × state_bytes
                   + 短卷积窗口（kernel × (2·d_k + d_v)，可忽略但记一笔）
```

| 模型 | 层型 | 全注意力层 | KV / token | 线性状态 / 序列 | 常数项 / 序列 | 512 | 2k | 8k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `F`×28 | 28 | 112 KiB | 0 | 0 | 56 MiB | 224 MiB | 896 MiB | 3.50 GiB |
| `Qwen/Qwen3.5-0.8B` | `LLLF`×6 | 6 | 12 KiB | 18 MiB | 18.84 MiB | 24.84 MiB | 42.84 MiB | 114.84 MiB | 402.84 MiB |

**交叉点 ≈ 193 token**（`18.84 MiB / (112 − 12) KiB`），渐近 KV 比 **9.33×**。

> **更正**：ROADMAP §3.1 写的「交叉点约 1.5k」比的是「混合的线性状态 vs 混合自己的 KV」
> （19 MB / 12 KB ≈ 1.5k），比较对象错了。要和**纯注意力**的 KV 比，差值只有 100 KiB/token，
> 交叉点在 200 token 以内。因此按「每序列状态」口径，P2（≤512 混合更贵）在 512 上不成立，
> 只有上下文 ≲193 token 时才成立。

### 2.1 口径警告：别拿「进程峰值」比

权重文件本身 ~1.4 GiB（0.6B）/ ~1.7 GiB（0.8B，含视觉塔）；若直接比 `peak_memory_allocated`，
0.8B 在**任何**上下文都比 0.6B 大，KV 的差异被权重淹没。W0 必须把状态拆成
**权重 / KV / 线性状态 / 激活**，P1、P2 用「每序列状态（cache）」判定，
总量只作辅助列。这条决定了 `bench/engines/hf.py` 要能从 `past_key_values` 里数字节。

## 3. W0 要填的表

`weights_mb` 与 context 无关，单独记一行；P1/P2 只看 **cache** 列（KV + 线性状态 + 卷积窗口），
`peak_mem_mb` / `mem_delta_mb` 作辅助，不参与判定。

| context | 0.6B cache (KV) | 0.8B cache (KV+state) | cache 比 0.6B/0.8B | 0.6B prefill / decode | 0.8B prefill / decode |
|---|---|---|---|---|---|
| 512 | 待填 | 待填 | | | |
| 2k | 待填 | 待填 | | | |
| 8k | 待填 | 待填 | | | |
| 32k | 待填 / OOM | 待填 / OOM | | | |

> 字段来自 `bench/engines/hf.py` 对 `past_key_values` 的直接计数（`kv_mb / state_mb / conv_mb`），
> 与设备无关；所以即使 `W0-0001` 只在 `dev-mps` 上跑，P1/P2 的字节比仍然可引用，
> 延迟结论只作趋势。

## 4. 预测判定

| # | 预测 | 判定 | 证据 |
|---|---|---|---|
| P1 | 长上下文（≥8k）每序列显存比同量级纯注意力小 5–8× | 待测 | `W0-0001` |
| P2 | 短上下文（≤512）混合反而更贵 | 待测 | `W0-0001` |
| P3 | decode 延迟随上下文几乎不增长 | 待测 | `W0-0001` |

## 5. D1 决策

见 [decisions.md](decisions.md) D1。结论只在 `W0-0001` 的 summary + notes 完成后写。
