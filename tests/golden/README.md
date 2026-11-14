# golden fixtures

W1/W5/W6 的数值对拍基准放在这里，由对应脚本生成，测试只读。

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

## 0002 · Qwen3.5-0.8B · W6 GDN + 全模型

`Qwen/Qwen3.5-0.8B@2fc0636`（超短线是 `2fc06364715b967f1860aea9cf38778875588b17`），
**float32 / CPU / eager**，`spec = w6-gdn-v1`，7.16 MB。由
`scripts/make_golden_qwen35_w6.py` 在干净树生成（`meta.source.dirty=false`，
`source.commit=024d58a`），用户 review 后提交；测试只读。

| 字段 | 内容 |
|---|---|
| prompts | `s01`（36 token）、`m02`（240 token；3 个 chunk 边界 + 48-token 尾巴） |
| `layers` | L0/L2（linear_attention） |
| 单层 stage | `layer_in`（全 T）、`input_layernorm`、`in_proj_qkv/z/b/a`、`conv_out`（post-SiLU，派生）、`q_pre`/`k_pre`（raw，l2norm 前）、`beta`/`g`、`core_attn_out`、`norm_out`、`layer_out`；`final_state`/`final_conv_state` 仅 m02 |
| 全模型 | `logits_last`、64 步 `greedy_ids` + `greedy_margins`、24 层 × 3 位置 `layer_hidden` |
| 状态口径 | `final_conv_state` 取 HF cache 的 `[..., -3:]`（P-49）；`final_state` fp32 |

为什么 `conv_out` 是派生的：HF 在 conv 之后没有模块边界，直接用 `in_proj_qkv` +
`conv1d.weight` 重算 `F.conv1d(..., padding=k-1)[:, :, :T]` 再 `F.silu`，与生产
`causal_conv_prefill` 逐元素一致；它**不是** `conv_state`（后者是 pre-SiLU 输入列）。

全模型上界：逐层 hidden `1e-2`、末位 logits `0.05`、贪心 64/64；正式实测远低于上界
（hidden 2.861e-6、logits 1.061e-5/1.121e-5、贪心 64/64）。

入口：`tests/integration/test_align_qwen35_gdn.py`（单层 stage）与
`tests/integration/test_align_qwen35_model.py`（全模型）；两者都要求
`MINILLM_RUN_HF=1`，草稿可用 `MINILLM_W6_GOLDEN=/tmp/w6-golden-dry` 指向。

重新生成只在 revision/prompt/契约变化时发生，并且要新开 `0003`；目录不可覆盖。
`git_state` 必须在创建 `out_dir` 之前采样（P-57），否则新目录本身会让
`dirty=true`。`final_state`/`final_conv_state` 在 `.numpy()` 前必须 `.clone()`
（P-59：HF cache 在 greedy decode 时会原地更新）。

## 验收

- 贪心 64 token 逐 id 全等；
- 每层 `max|Δ|` ≤ 阈值；末层 logits `max|Δ|` ≤ 阈值；
- batch 1/2/4 结果一致；
- 改第 t 个 token 不影响 t−1（因果性）。

入口：`tests/integration/test_align_qwen3.py`（`MINILLM_RUN_HF=1` 才跑）。
排版对照/逐层定位：`python scripts/align_hf.py`。

