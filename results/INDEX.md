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
| W45-0001 | W45 | microbench | E0 数值门：packed prefill + Triton decode 的 s01 64 token 与 golden 全等 | Qwen/Qwen3-0.6B | paged | tok_s 27.6 | pass | completed | 2026-11-06 | ab4aef1 |
| W45-0002 | W45 | fill | E1 容量（seat）：W3-like / fill-only / B=9 / serial | Qwen/Qwen3-0.6B | continuous | peak_concurrency 9 | pass | completed | 2026-11-06 | 0b04ae7 |
| W45-0003 | W45 | fill | E1 容量（paged）：W3-like / fill-only / 800 块 / serial | Qwen/Qwen3-0.6B | paged | peak_concurrency 28 | pass | completed | 2026-11-06 | 5a9b4e1 |
| W45-0004 | W45 | scenario | E2 decode 基线（seat）：8 条最长 / B=8 / budget 128 | Qwen/Qwen3-0.6B | continuous | tok_s 340 | pass | completed | 2026-11-06 | 3fcca5e |
| W45-0005 | W45 | scenario | E2 decode 内核（paged）：Triton block-table / 8 条最长 / B=8 / 832 块 | Qwen/Qwen3-0.6B | paged | tok_s 285 | fail | completed | 2026-11-06 | c19a5f3 |
| W45-0006 | W45 | scenario | E3 prefill A/B（serial）：static B=8 mixed / 1 token | Qwen/Qwen3-0.6B | continuous | tok_s 39.1 | pass | completed | 2026-11-06 | 7021e02 |
| W45-0007 | W45 | scenario | E3 prefill A/B（packed）：static B=8 mixed / 1 token | Qwen/Qwen3-0.6B | continuous | tok_s 148 | pass | completed | 2026-11-06 | dd9206c |
| W45-0008 | W45 | scenario | E4 端到端（seat）：W4 trace / packed prefill / B=9 | Qwen/Qwen3-0.6B | continuous | tok_s 331 | pass | completed | 2026-11-06 | 548a216 |
| W45-0009 | W45 | scenario | E4 端到端（paged）：W4 trace / packed prefill + Triton decode / 800 块 | Qwen/Qwen3-0.6B | paged | tok_s 679 | pass | completed | 2026-11-06 | bb30f0d |
| W45-0010 | W45 | scenario | E4b（可选）static / B=8 / mixed / packed prefill | Qwen/Qwen3-0.6B | continuous | tok_s 157 | pass | completed | 2026-11-06 | 0141fce |
| W45-0011 | W45 | scenario | E4b（可选）continuous / B=8 / mixed / packed prefill | Qwen/Qwen3-0.6B | continuous | tok_s 293 | pass | completed | 2026-11-06 | b97c45d |
| W45-0012 | W45 | scenario | E2 decode 内核（paged，C1 修复后）：Triton block-table / 8 条最长 / B=8 / 832 块 | Qwen/Qwen3-0.6B | paged | tok_s 329 | pass | completed | 2026-11-06 | f333483 |
| W7-0001 | W7 | scenario | E0b 正确性（hybrid）：8 条 mixed prompts / B=1 / 128 new | Qwen/Qwen3.5-0.8B | hybrid | tok_s 31.9 | unreviewed | completed | 2026-11-14 | 1a3e902 |
| W7-0002 | W7 | scenario | E0b 正确性（hybrid）：8 条 mixed prompts / B=8 / 128 new | Qwen/Qwen3.5-0.8B | hybrid | tok_s 219 | unreviewed | completed | 2026-11-14 | 22b31fa |
| W7-0003 | W7 | scenario | E0c 调度器回归（pure 0.6B）：同 8 条 mixed prompts / B=8 / 128 new | Qwen/Qwen3-0.6B | continuous | tok_s 340 | unreviewed | completed | 2026-11-14 | c8cc70b |
| W7-0004 | W7 | fill | E1/E2 容量（hybrid fill）：synthetic 512 / B=64 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 62 | unreviewed | completed | 2026-11-14 | ed7b6bd |
| W7-0005 | W7 | fill | E1/E2 容量（pure 0.6B fill）：synthetic 512 / B=64 / 936 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 23 | unreviewed | completed | 2026-11-14 | 43eb9e7 |
| W7-0006 | W7 | fill | E1/E2 容量（hybrid fill）：synthetic 2k / B=64 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 37 | unreviewed | completed | 2026-11-14 | 90f379d |
| W7-0007 | W7 | fill | E1/E2 容量（pure 0.6B fill）：synthetic 2k / B=64 / 936 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 6 | unreviewed | completed | 2026-11-14 | 4fdcca4 |
| W7-0008 | W7 | fill | E1/E2 容量（hybrid fill）：synthetic 8k / B=64 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 14 | unreviewed | completed | 2026-11-14 | 1b2c4de |
| W7-0009 | W7 | fill | E1/E2 容量（pure 0.6B fill）：synthetic 8k / B=64 / 936 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 1 | unreviewed | completed | 2026-11-14 | 5a61a92 |
| W7-0010 | W7 | scenario | E3 交叉点/P2/P3（hybrid）：synthetic [512,2k,8k] / B=1 / 128 new | Qwen/Qwen3.5-0.8B | hybrid | tok_s 29.4 | unreviewed | completed | 2026-11-14 | 580276a |
| W7-0011 | W7 | scenario | E3 交叉点/P2/P3（pure 0.6B）：synthetic [512,2k,8k] / B=1 / 936 块 | Qwen/Qwen3-0.6B | paged | tok_s 42.9 | unreviewed | completed | 2026-11-14 | 1b2c4de |
| W7-0013 | W7 | scenario | E3 交叉点/P2/P3（pure 0.6B）：synthetic [512,2k,8k] / B=8 / 4200 块 | Qwen/Qwen3-0.6B | paged | tok_s 112 | invalid | completed | 2026-11-14 | 9c05ecc |
| W7-0014 | W7 | scenario | E4a 机制迁移（hybrid）：synthetic mixed Poisson / B=8 / continuous | Qwen/Qwen3.5-0.8B | hybrid | tok_s 112 | unreviewed | completed | 2026-11-14 | db11f05 |
| W7-0015 | W7 | scenario | E4a 机制迁移（hybrid）：synthetic mixed Poisson / B=8 / static | Qwen/Qwen3.5-0.8B | hybrid | tok_s 74.6 | unreviewed | completed | 2026-11-14 | 7037a3d |
| W7-0016 | W7 | fill | E4b 容量（hybrid paged）：synthetic mixed [512,2k] / fill / 3970 块 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 48 | unreviewed | completed | 2026-11-14 | 8aea209 |
| W7-0017 | W7 | fill | E4b 容量（hybrid seat baseline）：synthetic mixed [512,2k] / fill / 37 行 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 37 | unreviewed | completed | 2026-11-14 | c7c59d2 |
| W7-0018 | W7 | scenario | E3 交叉点/P2/P3（hybrid）：synthetic [512,2k,8k] / N=24 / B=8 / 128 new | Qwen/Qwen3.5-0.8B | hybrid | tok_s 166 | unreviewed | completed | 2026-11-14 | b48b811 |
| W7-0019 | W7 | scenario | E3 交叉点/P2/P3（pure 0.6B）：synthetic [512,2k,8k] / N=24 / B=8 / 4200 块 | Qwen/Qwen3-0.6B | paged | tok_s 224 | unreviewed | completed | 2026-11-14 | c363767 |
| W7-0020 | W7 | fill | E1/E2 容量（hybrid fill）：synthetic 128 / B=64 / 64 槽 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 64 | unreviewed | completed | 2026-11-14 | 82823f3 |
| W7-0021 | W7 | fill | E1/E2 容量（pure 0.6B fill）：synthetic 128 / B=64 / 936 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 58 | unreviewed | completed | 2026-11-14 | ae88d94 |
| W7-0022 | W7 | fill | E1/E2 容量（hybrid fill）：synthetic 192 / B=64 / 64 槽 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 64 | unreviewed | completed | 2026-11-14 | ce27595 |
| W7-0023 | W7 | fill | E1/E2 容量（pure 0.6B fill）：synthetic 192 / B=64 / 936 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 46 | unreviewed | completed | 2026-11-14 | 04f7c79 |
| W7-0024 | W7 | fill | E1/E2 容量（hybrid fill）：synthetic 256 / B=64 / 64 槽 | Qwen/Qwen3.5-0.8B | hybrid | peak_concurrency 64 | unreviewed | completed | 2026-11-14 | fc3b9f0 |
| W7-0025 | W7 | fill | E1/E2 容量（pure 0.6B fill）：synthetic 256 / B=64 / 936 块 | Qwen/Qwen3-0.6B | paged | peak_concurrency 39 | unreviewed | completed | 2026-11-14 | 497807b |
| W7-0026 | W7 | scenario | E3 P2/P3 修正（hybrid）：B=1，每请求 decode 归因 | Qwen/Qwen3.5-0.8B | hybrid | tok_s 32 | unreviewed | completed | 2026-11-14 | 54a3c7f |
| W7-0027 | W7 | scenario | E3 P2/P3 修正（pure）：B=1，每请求 decode 归因 | Qwen/Qwen3-0.6B | paged | tok_s 43.4 | unreviewed | completed | 2026-11-14 | 3f4cd9a |
| W7-0028 | W7 | scenario | E3 P2/P3 修正（hybrid）：B=8，N=24，每请求 decode 归因 | Qwen/Qwen3.5-0.8B | hybrid | tok_s 168 | unreviewed | completed | 2026-11-14 | f55c119 |
| W7-0029 | W7 | scenario | E3 P2/P3 修正（pure）：B=8，N=24，每请求 decode 归因 | Qwen/Qwen3-0.6B | paged | tok_s 226 | unreviewed | completed | 2026-11-14 | f975590 |

_下一步与阶段进度见 [ROADMAP](../ROADMAP.md) §12。_
