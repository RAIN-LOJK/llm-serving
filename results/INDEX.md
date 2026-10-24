# 实验索引

> 本文件由 `scripts/exp_index.py` 生成，请勿手改。
> 新 run 请放到 `results/<stage>/<4位序号>-<slug>/`，契约见 [docs/experiments.md](../docs/experiments.md)。

| run | stage | type | title | model | engine | key metric | verdict | status | date | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| W0-0001 | W0 | probe | 双载体：显存/延迟 vs 上下文 | Qwen/Qwen3-0.6B +1 | hf | ttft_ms 93.7 | pass | completed | 2026-10-22 | ecdf506 |
| W2-0001 | W2 | microbench | HF generate 参考臂：TTFT / TPOT @ 32 prompt × 128 token | Qwen/Qwen3-0.6B | hf | ttft_ms 20 | pass | completed | 2026-10-24 | f32bdb0 |
| W2-0002 | W2 | microbench | 自实现解码器：TTFT / TPOT @ 同一 prompt 集 | Qwen/Qwen3-0.6B | minillm | ttft_ms 20 | pass | completed | 2026-10-24 | 7892322 |

_下一步与阶段进度见 [ROADMAP](../ROADMAP.md) §12。_
