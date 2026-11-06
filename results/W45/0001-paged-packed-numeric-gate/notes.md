# W45-0001 · E0 数值门：packed prefill + Triton decode

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0001-paged-packed-numeric-gate`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：`golden_token_match_min = 1.0`；paged + packed + Triton 的 s01 64-token greedy 前缀必须与 `tests/golden/qwen3-0-6b/0001` 全等。
- 观察：**64/64 token ids**，1/1 prompt，`match_rate = 1.0`；`n_errors = 0`。provenance 为 `attn_prefill=flash-attn-varlen`、`attn_decode=paged-triton`、`prefill_mode=packed`。
- 结论：E0 **pass**。新注意力路径的端到端 greedy 输出与 golden 一致；这是 E1–E4 的前置门。
- 限制：单 prompt、64 token、`warmup=0`；TTFT 282.5 ms 含首次 kernel/JIT 开销，**不是性能证据**。`summary.warnings` 的测量协议偏差与无 paired baseline 对数值门无影响。
- 下一步：容量与 decode 性能对照见 `W45-0002…0005/0012`。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
