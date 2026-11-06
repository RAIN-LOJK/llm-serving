# W45-0006 · E3 prefill A/B（serial）：static B=8 mixed / 1 token

- 状态：`reviewed`；verdict = `pass`
- 配置：`config.toml`（运行目录 `results/W45/0006-static-serial-prefill`）
- 决策：D7（W4.5 注意力路径 v2）
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- 阶段口径见 `docs/decisions.md` D7；本 run 的 acceptance 见 `summary.json`。

## 观察

- 判据：record-only 基线；与 `W45-0007` 同 prompt/trace，只差 `prefill=serial|packed`。
- 观察：TTFT **p50 105.143 ms / p99 207.378 ms**；per-request prefill p50 25.417 / p99 28.837 ms；`pack_wait = 0`（串行臂 `t_admitted == t_prefill_start`）；`n_errors = 0`。
- 结论：serial 基线 **pass**；TTFT 的尾部由排队/逐请求 prefill 叠加造成，不是单个 kernel 慢。
- 限制：`max_new_tokens=1`，只量首 token；3 个 repeat；`warmup=1`；不能把 TTFT 与 decode 混读。

## 限制与偏差

- 单张 RTX 4090、bf16、`device_class=measure-cuda`；所有数字只用于本机同口径比较。
- 本 run 的 summary 是聚合证据；逐请求细节在 `metrics.jsonl`，不在这里重算。

## 下一步

- 阶段收口以 `ROADMAP.md` §12 与 D7 为准；历史 run 不重写。
