# golden fixtures

W1 的数值对拍基准放在这里，由 `scripts/make_golden.py` 生成，测试只读。

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

## 0001 · Qwen3-0.6B

`Qwen/Qwen3-0.6B@c1899de`（596.0M 参数），**float32 / CPU / eager**：checkpoint 是 bf16，
upcast 到 fp32 无损，所以参考值与 GPU、驱动无关，换机器也能复现。3.5 MB。

| 字段 | 内容 |
|---|---|
| prompts | `bench/prompts.json` 的 s01–s04（短桶），token 数 39 / 40 / 25 / 31 |
| `hidden` | 28 层的**块输出**（不含 embedding 行），每 prompt 抽 3 个绝对位置：首、中、尾 |
| `logits_last` | 只存 prefill 末位，`(4, 151936)` fp32；64 步 logits 全存会到 39 MB/份 |
| `greedy_ids` | 64 步贪心，`-1` 表示参考早已 EOS（本 fixture 四条都走满 64 步） |
| `input_ids` | 左 pad 到公共宽度，pad id 在 fixture 里；测试不用 pad 列 |

为什么只存末位 logits：`max|Δ|` 是对整个词表取的最大值，末位就足够判；逐步 logits
既不必要，也把体积放大一个量级。逐步正确性由 `greedy_ids` 的逐 id 全等来保证。

块输出用 forward hook 采，不用 `output_hidden_states`：transformers 5.x 的
`hidden_states[-1]` 已经过 final norm（PITFALLS P-16），拿它当末层会让正确的实现
显示成差 1.9e3。生成侧（`make_golden.py`）与对拍侧（`align_hf.py`）共用同一套 hook
口径，不会再各算一套。

`[W1.align]` 的 `layer_max_abs_diff` 已按实测校准到 `1e-2`：首 token 的 attention-sink
离群值到 7e3，fp32 一个 ulp 就是 9.8e-4，原来的 `1e-3` 等于要求"≤1 ulp"。依据写在
`configs/thresholds.toml` 的注释里。

重新生成只在 prompt 集或模型 revision 变了时发生，并且要新开 `--golden-id`（目录不可覆盖）。

## 验收

- 贪心 64 token 逐 id 全等；
- 每层 `max|Δ|` ≤ 阈值；末层 logits `max|Δ|` ≤ 阈值；
- batch 1/2/4 结果一致；
- 改第 t 个 token 不影响 t−1（因果性）。

入口：`tests/integration/test_align_qwen3.py`（`MINILLM_RUN_HF=1` 才跑）。
排版对照/逐层定位：`python scripts/align_hf.py`。

