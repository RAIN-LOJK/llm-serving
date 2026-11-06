# 实验索引

> 本文件由 `scripts/exp_index.py` 生成，请勿手改。
> 新 run 请放到 `results/<stage>/<4位序号>-<slug>/`，契约见 [docs/experiments.md](../docs/experiments.md)。

| run | stage | type | title | model | engine | key metric | verdict | status | date | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| W0-0001 | W0 | probe | 双载体：显存/延迟 vs 上下文 | Qwen/Qwen3-0.6B +1 | hf | ttft_ms 93.7 | pass | completed | 2026-10-22 | ecdf506 |
| W2-0001 | W2 | microbench | HF generate 参考臂：TTFT / TPOT @ 32 prompt × 128 token | Qwen/Qwen3-0.6B | hf | ttft_ms 20 | pass | completed | 2026-10-24 | f32bdb0 |
| W2-0002 | W2 | microbench | 自实现解码器：TTFT / TPOT @ 同一 prompt 集 | Qwen/Qwen3-0.6B | minillm | ttft_ms 20 | pass | completed | 2026-10-24 | 7892322 |
| W2-0003 | W2 | microbench | SDPA attention 路径：long-prefill 归因（对 W2-0002） | Qwen/Qwen3-0.6B | minillm | ttft_ms 17.4 | pass | completed | 2026-10-24 | c833013 |
| W2-0004 | W2 | microbench | 固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab | Qwen/Qwen3-0.6B | minillm | ttft_ms 21.2 | invalid | completed | 2026-11-01 | e97649e |
| W2-0005 | W2 | microbench | HF generate 参考臂（本机重测，W2-0006 的配对基线） | Qwen/Qwen3-0.6B | hf | ttft_ms 23.5 | pass | completed | 2026-11-01 | 71887a8 |
| W2-0006 | W2 | microbench | 固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab（本机配对） | Qwen/Qwen3-0.6B | minillm | ttft_ms 21.2 | pass | completed | 2026-11-01 | 50e3623 |
| W2-0007 | W2 | microbench | 固定注意力路径：prefill=sdpa / decode=hand_baddbmm+slab（逐层分配、随请求释放） | Qwen/Qwen3-0.6B | minillm | ttft_ms 21.8 | pass | completed | 2026-11-01 | 50e0dcd |
| W3-0001 | W3 | microbench | E0a 重构无害性：席位 API 落地后单请求路径不动数字 | Qwen/Qwen3-0.6B | minillm | ttft_ms 21.2 | pass | completed | 2026-11-03 | 552c72e |
| W3-0002 | W3 | microbench | E0b slab 池化后单请求复核：kv_mb 记池的总预留 | Qwen/Qwen3-0.6B | continuous | tok_s 47.1 | pass | completed | 2026-11-03 | 9ca5ffe |
| W3-0003 | W3 | microbench | E1 批处理摊薄 host：ctx512 / B=1 参照 | Qwen/Qwen3-0.6B | continuous | tok_s 47.9 | invalid | completed | 2026-11-03 | ecc9b14 |
| W3-0004 | W3 | microbench | E1 批处理摊薄 host：ctx512 / B=8（P1） | Qwen/Qwen3-0.6B | continuous | tok_s 116 | invalid | completed | 2026-11-03 | e7553e9 |
| W3-0005 | W3 | microbench | E1 批处理摊薄 host：ctx512 / B=32（P2） | Qwen/Qwen3-0.6B | continuous | tok_s 137 | invalid | completed | 2026-11-03 | a69c475 |
| W3-0006 | W3 | microbench | E1 长上下文参照：ctx2048 / B=1 | Qwen/Qwen3-0.6B | continuous | tok_s 45.3 | pass | completed | 2026-11-03 | 36ced60 |
| W3-0007 | W3 | microbench | E1 长上下文 batch 收益：ctx2048 / B=8 | Qwen/Qwen3-0.6B | continuous | tok_s 338 | pass | completed | 2026-11-03 | 95c95a9 |
| W3-0008 | W3 | scenario | E2 调度收益：static / B=8 / uniform | Qwen/Qwen3-0.6B | continuous | tok_s 341 | pass | completed | 2026-11-03 | 15e9c2c |
| W3-0009 | W3 | scenario | E2 调度收益：continuous / B=8 / uniform | Qwen/Qwen3-0.6B | continuous | tok_s 349 | pass | completed | 2026-11-03 | ed9378b |
| W3-0010 | W3 | scenario | E2 调度收益：static / B=8 / mixed | Qwen/Qwen3-0.6B | continuous | tok_s 155 | pass | completed | 2026-11-03 | 19971ad |
| W3-0011 | W3 | scenario | E2 调度收益：continuous / B=8 / mixed | Qwen/Qwen3-0.6B | continuous | tok_s 311 | pass | completed | 2026-11-03 | 632f395 |
| W3-0012 | W3 | scenario | E2 调度收益：static / B=32 / mixed | Qwen/Qwen3-0.6B | continuous | tok_s 448 | pass | completed | 2026-11-03 | 84431e9 |
| W3-0013 | W3 | scenario | E2 调度收益：continuous / B=32 / mixed | Qwen/Qwen3-0.6B | continuous | tok_s 719 | pass | completed | 2026-11-03 | eba8732 |
| W3-0017 | W3 | microbench | E1 批处理摊薄 host：ctx511 / B=1 参照（ctx512 EOS 后的替代） | Qwen/Qwen3-0.6B | continuous | tok_s 47.8 | pass | completed | 2026-11-03 | 295a8df |
| W3-0018 | W3 | microbench | E1 批处理摊薄 host：ctx511 / B=8（P1，ctx512 EOS 后的替代） | Qwen/Qwen3-0.6B | continuous | tok_s 361 | pass | completed | 2026-11-03 | d9e505a |
| W3-0019 | W3 | microbench | E1 批处理摊薄 host：ctx511 / B=32（P2，ctx512 EOS 后的替代） | Qwen/Qwen3-0.6B | continuous | tok_s 1.18e+03 | pass | completed | 2026-11-03 | 7a70e19 |
| W4-0001 | W4 | microbench | E0 数值门：paged 64 token 贪心输出与 golden 逐 id 全等 | Qwen/Qwen3-0.6B | paged | tok_s 30 | pass | completed | 2026-11-06 | 9dc5211 |
| W4-0002 | W4 | fill | E1 容量（seat）：W3-like / fill-only / B=9 | Qwen/Qwen3-0.6B | continuous | peak_concurrency 9 | pass | completed | 2026-11-06 | 6652ef4 |
| W4-0003 | W4 | fill | E1 容量（paged）：W3-like max-2048 / fill-only / 800 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 28 | pass | completed | 2026-11-06 | 0d800b9 |
| W4-0004 | W4 | scenario | E2 代价基线（seat）：8 条最长 / B=8 / budget 128 | Qwen/Qwen3-0.6B | continuous | tok_s 356 | pass | completed | 2026-11-06 | 1a97b6f |
| W4-0005 | W4 | scenario | E2 代价（paged）：8 条最长 / B=8 / budget 128 / 832 块 | Qwen/Qwen3-0.6B | paged | tok_s 201 | pass | completed | 2026-11-06 | 956ff85 |
| W4-0006 | W4 | scenario | E3 同预算系统吞吐（seat，可选）：frozen 32 / poisson / B=9 | Qwen/Qwen3-0.6B | continuous | tok_s 333 | pass | completed | 2026-11-06 | adffb81 |
| W4-0007 | W4 | scenario | E3 同预算系统吞吐（paged，可选）：frozen 32 / poisson / B=64 / 820 块 | Qwen/Qwen3-0.6B | paged | tok_s 321 | invalid | completed | 2026-11-06 | b7be3b4 |
| W4-0008 | W4 | scenario | E3 同预算系统吞吐（paged，800 块）：frozen 32 / poisson / B=64 | Qwen/Qwen3-0.6B | paged | tok_s 325 | inconclusive | completed | 2026-11-06 | c301b26 |
| W4-0009 | W4 | scenario | E3′ 同预算系统吞吐（paged，prefill 修复后，800 块）：frozen 32 / poisson / B=64 | Qwen/Qwen3-0.6B | paged | tok_s 358 | pass | completed | 2026-11-06 | b8fa5c1 |
| W45-0001 | W45 | microbench | E0 数值门：packed prefill + Triton decode 的 s01 64 token 与 golden 全等 | Qwen/Qwen3-0.6B | paged | tok_s 27.6 | unreviewed | completed | 2026-11-06 | ab4aef1 |
| W45-0002 | W45 | fill | E1 容量（seat）：W3-like / fill-only / B=9 / serial | Qwen/Qwen3-0.6B | continuous | peak_concurrency 9 | unreviewed | completed | 2026-11-06 | 0b04ae7 |

_下一步与阶段进度见 [ROADMAP](../ROADMAP.md) §12。_
