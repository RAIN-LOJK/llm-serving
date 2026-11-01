# W2-0005 · HF generate 参考臂（本机重测）

- 状态：`pass`（reviewer: rainhome, 2026-11-01）
- 配置：`config.toml`（运行目录 `results/W2/0005-hf-reference-this-host`）
- 类型：`microbench`；workload：`prompts`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `hf_arm_on_this_host_is_the_decode_baseline`
- 取代 `W2-0001`：那一份是在另一台 4090 实例上测的。

## 观察

- 160 行 / 0 错误；warnings 仅 "no paired baseline run_id"（本 run 是基线臂，预期如此）。
- TTFT **23.46** ms，TPOT **21.43** ms，e2e 2745.6 ms，peak 1192 MB，driver 1715 MB。
- 三条判据 pass：`ttft_cv_max` 0.81%（short 桶）、`tpot_cv_max` 0.52%（short 桶）、
  `memory_match_ratio` **0.992**（最差 context 0.974）。
- 与 `W2-0001` 的差：TPOT 16.78 → **21.43 ms（1.28×）**，TTFT 20.02 → 23.46 ms。
  代码、dtype、prompt hash、`max_new_tokens` 全部相同，差别只有机器实例。

## 结论（含“什么时候不划算”）

- 本 run 是 `W2-0006`/`W2-0007` 的配对基线，也是之后 W3 各臂与 HF 比较时该用的分母。
- **跨实例的 host-bound 比值不是稳定量。** decode 每步的墙钟由 host 主导（每步 28 层
  Python 派发 + `.item()` 同步），两台 4090 的 CPU/驱动/抢占差异足以让它挪 28%。
  所以凡是引用 "decode 相对 HF 多少倍" 的结论，必须带上跑它的那台机器；
  `W2-0001` 的 16.78 ms 从此只作为历史记录，不再作为分母。
- 什么时候不划算：HF 臂本身在长上下文没有优势（memory 0.974，`T²` 工作区仍在），
  但它不是本次的判据对象——它只提供分母。

## 限制与偏差

- 设备 / dtype / 样本量：RTX 4090 / bf16 / 32 prompt × 5 次 / `max_new_tokens=128`。
  与本机的 `W2-0006` 同卡同会话（两次 run 之间只隔一个把 `results/INDEX.md` 提交的 commit）。
- `attn_prefill` / `attn_decode` 记为 `hf:sdpa`：机制键检查对 HF 臂不生效
  （`bench/thresholds.py::_is_inhouse`），这是有意的——HF 没有本仓库可替换的 kernel。
- `peak_over_driver` 0.695：设备级数字含 ~0.5 GB CUDA context，不进判据。

## 下一步

- `W2-0007` 挂本 run 作 `[baseline]`（`W2-0006` 已被它取代）。W3 若重开新臂，同样先在本机重测 HF 分母。
