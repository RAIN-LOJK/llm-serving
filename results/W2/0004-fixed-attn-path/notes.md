# W2-0004 · 固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab

- 状态：`invalid`（配对无效；reviewer: rainhome, 2026-11-01）
- 配置：`config.toml`（运行目录 `results/W2/0004-fixed-attn-path`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `one_attention_path_does_not_move_w2_numbers`
- 配对基线 `W2-0001`（HF 臂，**在另一台 4090 实例上**测的）。

## 观察

- 160 行 / 0 错误 / 0 warnings。
- TTFT 21.16 ms，TPOT 21.30 ms，peak 1330 MB，driver 1855 MB。
- 判据 2/4 pass：`ttft_cv_max` 1.53%、`tpot_cv_max` 0.40%；
  `memory_match_ratio` **0.889** 失败、`decode_speed_vs_hf_min` **0.781** 失败。
- 长上下文比值塌陷：ctx 987 → 0.737、1249 → 0.721、1338 → 0.740、1517 → 0.723。

## 结论（含“什么时候不划算”）

本 run **不作为 W2 的基线**，两条失败各自有独立原因，都不是注意力路径本身：

1. **`memory_match_ratio` 0.889 是记账缺陷。** 当时 slab 保留 3 个桶（`MAX_CACHED_SLABS = 3`），
   本 run 出现 11 个不同桶，长上下文时同时驻留 3 块大桶（最大三块合计 518 MB）。
   而分母是全进程分配器峰值、分子是单请求的 `weights + 预留 KV`，驻留的邻居桶永远解释不了。
   单跑 ctx 987 是 0.870，run 里是 0.737——差的 263 MB 全是邻居桶。
   又因为 prompt 按组跑、`warmup=3` 已把建桶挡在计时区外，这个缓存**零收益**。
2. **`decode_speed_vs_hf_min` 0.781 是跨实例配对。** `W2-0001` 的 HF 臂是 16.78 ms，
   同一份 HF 代码在本机是 21.43 ms（`W2-0005`）——分母来自一台更快的机器。
   本 run 的 21.30 ms 与 `W2-0006` 的 21.43 ms 只差 0.6%，说明代码没有退化。

处置：`MAX_CACHED_SLABS` 改为单一 slab（换桶即重建，commit `ce0af17`），
并在同一台机器上重测两臂——`W2-0005`（HF）+ `W2-0006`（自实现）。本 run 由 `W2-0006` 取代。

## 限制与偏差

- 设备 / dtype / 样本量：RTX 4090 / bf16 / 32 prompt × 5 次 / `max_new_tokens=128`。
  与本机 `W2-0005`/`W2-0006` 同卡不同实例，**数字不可与它们直接比绝对值**。
- 本 run 保留为失败记录（run 目录不可变），不重跑、不改 `metrics.jsonl`。

## 下一步

- 见 `W2-0006`：修好记账、同机配对后四条判据全 pass，由它作为 W3 的配对基线。
