# golden fixtures

W1 的数值对拍基准放在这里。目前**还没有** golden；W1 用 `scripts/make_golden.py` 生成。

## 格式

```text
tests/golden/<model-slug>/<golden-id>/
├── meta.json      # 模型 repo + revision、dtype、tokenizer、prompt sha256、生成命令、版本
└── tensors.npz    # input_ids / greedy_ids / 抽样层的 hidden / 末层 logits
```

约定：

- golden 由脚本生成并**提交**；测试只读，不允许静默重新生成。
- `meta.json` 必须记录 HF commit hash（不能用 `main`）、transformers 版本、dtype、设备。
- 只抽样保存需要的层/位置，控制体积；`.npz` 很小，已在 `.gitignore` 中显式放行。
- 测试标记 `slow`；快速门禁不跑。
- 容差写 `configs/thresholds.toml` 的 `[W1.align]`；W1 校准后从 `target` 改 `calibrated`。

## 验收

- 贪心 64 token 逐 id 全等；
- 每层 `max|Δ|` ≤ 阈值；末层 logits `max|Δ|` ≤ 阈值；
- batch 1/2/4 结果一致；
- 改第 t 个 token 不影响 t−1（因果性）。
