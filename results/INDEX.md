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
| W3-0001 | W3 | microbench | E0a 重构无害性：席位 API 落地后单请求路径不动数字 | Qwen/Qwen3-0.6B | minillm | ttft_ms 21.2 | unreviewed | completed | 2026-11-03 | 552c72e |
| W3-0002 | W3 | microbench | E0b slab 池化后单请求复核：kv_mb 记池的总预留 | Qwen/Qwen3-0.6B | continuous | tok_s 47.1 | unreviewed | completed | 2026-11-03 | 9ca5ffe |
| W3-0003 | W3 | microbench | E1 批处理摊薄 host：ctx512 / B=1 参照 | Qwen/Qwen3-0.6B | continuous | tok_s 47.9 | unreviewed | completed | 2026-11-03 | ecc9b14 |
| W3-0004 | W3 | microbench | E1 批处理摊薄 host：ctx512 / B=8（P1） | Qwen/Qwen3-0.6B | continuous | tok_s 116 | unreviewed | completed | 2026-11-03 | e7553e9 |
| W3-0005 | W3 | microbench | E1 批处理摊薄 host：ctx512 / B=32（P2） | Qwen/Qwen3-0.6B | continuous | tok_s 137 | unreviewed | completed | 2026-11-03 | a69c475 |
| W3-0006 | W3 | microbench | E1 长上下文参照：ctx2048 / B=1 | Qwen/Qwen3-0.6B | continuous | tok_s 45.3 | unreviewed | completed | 2026-11-03 | 36ced60 |
| W3-0007 | W3 | microbench | E1 长上下文 batch 收益：ctx2048 / B=8 | Qwen/Qwen3-0.6B | continuous | tok_s 338 | unreviewed | completed | 2026-11-03 | 95c95a9 |
| W3-0017 | W3 | microbench | E1 批处理摊薄 host：ctx511 / B=1 参照（ctx512 EOS 后的替代） | Qwen/Qwen3-0.6B | continuous | tok_s 47.8 | unreviewed | completed | 2026-11-03 | 295a8df |

_下一步与阶段进度见 [ROADMAP](../ROADMAP.md) §12。_
