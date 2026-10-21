# 混合注意力（hybrid attention）

> 状态：**待 W5–W7 填写**。数字以 `results/W5..W7/` 的 summary 为准。

## 1. 目标

Qwen3.5 的 3:1 混合注意力：75% 层是 Gated DeltaNet（线性注意力），25% 是全注意力。
serving 侧要同时管理：

- 分页 KV：只覆盖全注意力层；
- 线性状态池：每请求常数大小，覆盖线性层；
- 统一句柄与生命周期（`serve/state_pool.py`）。

## 2. 实现清单

| # | 算子 | 验收 | 状态 |
|---|---|---|---|
| 1 | 全注意力层（GQA, head_dim 256, 输出门控） | 单层对拍 HF | ⬜ |
| 2 | partial-RoPE（1/4 维）+ M-RoPE（文本退化 1D） | 位置编码单独测 | ⬜ |
| 3 | 短卷积（kernel 4, causal, q/k/v） | 因果性：改第 t token 不影响 t−1 | ⬜ |
| 4 | Gated DeltaNet decode 形式 | 逐步对拍 | ⬜ |
| 5 | Gated DeltaNet prefill 分块形式 | 分块 prefill == 逐步 decode | ⬜ |
| 6 | MTP 头 | 跳过，加载时忽略 | N/A |

## 3. 自洽测试

**分块 prefill 与逐步 decode 必须给出同样的 logits。** 这比对齐 HF 更重要；
一旦通过，数值正确性基本锁住。测试放在 `tests/integration/`，标记 `slow`。

## 4. W7 要填的结论

| 项 | 目标 / 判据 | 实测 | 判定 |
|---|---|---|---|
| 每序列状态占用 vs 解析式估算 | 误差 < 20% | 待填 | |
| P1 / P2 / P3 | 见 carrier.md | 待填 | |
| 交叉点 | 上下文多长混合开始占优 | 待填 | |
| 同显存最大并发 | 混合 vs 纯注意力 | 待填 | |

## 5. 风险与止损

- 分块形式的 WY 表示是最难的一段。若 W6 末只有 decode 形式可用，接受
  “prefill 走逐步循环”的慢速实现，报告里注明性能代价，不硬啃。
- 若 W0 的 D1 判定不投入移植，本文件只保留 HF 侧对照分析。
