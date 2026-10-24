# 踩过的坑（高信号，按需读取）

> 约定：每条坑有 ID，并尽量映射到 `tests/` 的回归用例。
> `status = covered` 表示已有自动护栏；`OPEN` 表示尚未覆盖，动相关代码前先补测试。
> 这些坑来自旧训练项目（tag `legacy-learning-2026-08`），技术结论在 serving 项目里仍然适用。

| ID | 坑 | 根因 / 解法 | 回归测试 | 状态 |
|---|---|---|---|---|
| P-01 | transformers v5 的 `_tied_weights_keys` 是 dict | 现在是 `{目标: 源}`（如 `{"lm_head.weight": "token_embedding.weight"}`），不再是 list；写 list 会报 `'list' object has no attribute 'keys'` | W1 加载 HF 权重时覆盖 | OPEN |
| P-02 | RoPE 两种约定不兼容 | interleaved（按 `(2k, 2k+1)` 配对）vs GPT-NeoX（前后两半配对 `rotate_half`）。两者等价但差一个固定维度置换，**与同一组权重不兼容**。本项目统一 GPT-NeoX；加载 HF 权重前先确认 `apply_rotary_emb` 约定 | `tests/unit/test_rope.py::test_gpt_neox_convention` | covered |
| P-03 | `config.norm_eps` 必须接线 | RMSNorm 构造要显式传 `config.norm_eps`（TinyLlama 是 1e-5），否则默默用默认值，和 HF 对不上 | `tests/unit/test_attention.py::test_norm_eps_wired` | covered |
| P-04 | KV cache 三个隐性坑 | ① 层间缓存绝不拼接，`kvs[l]` 只存本层随时间累积的 `(k, v)`；② 因果 mask 的 diagonal 要偏移 `past_len+1`；③ `past_len` 要按层取，prefill 时各层历史为空 | `tests/unit/test_kvcache.py` | covered |
| P-05 | decode 的 pad mask 必须“全长度” | key 序列 = 历史 `past_len` 列 + 新 1 列，pad 全在左侧。只传新列 mask 会泄漏 cached pad 列；解法是每步 `concat` 追加一列 1，宽度 = key 总长。该做法依赖左 pad 前对齐（RoPE 共享右列 → 共享 past_len） | `tests/unit/test_attention.py::test_leftpad_mask_matches_unpadded`、`tests/unit/test_generate_leftpad.py` | covered |
| P-06 | EOS 发散后账本错位 | `buffer` + `outputs` + `orig` 映射统一到压缩空间，行结束即 flush；`break` 前 `buffer` 与 `orig` 必须同形 | `tests/unit/test_generate_eos.py` | covered |
| P-07 | RoPE 表越界不报错，空切片级联成形状错误 | decode 到位置 ≥ `max_position_embeddings` 时 `cos[past_len:past_len+T]` 返回空张量；空维广播静默把 q/k 的 T 维抹成 0，最终在 `attn + mask` 报“64 vs 65”这种看似 mask 公式错误的形状不匹配。根因是越界。解法：`generate` 加护栏 `past_len+1 >= max_position_embeddings` 就停 | `tests/unit/test_generate_guard.py` | covered |
| P-08 | `generate` 输出契约：护栏中断必须截断 | `buffer` 是预分配 zeros，早停后整行 flush 会把没采样过的 0 当结果。解法：循环外声明 `sampled` 计数器，EOS 与最终 flush 统一 `buffer[idx, :sampled]` | `tests/unit/test_generate_eos.py` | covered |
| P-09 | 批量左 pad 的生成预算是共享的 | 所有行的生成预算 = `max_position_embeddings - T_max`（含 pad 的宽度），不是每条按自己的长度算 | `tests/unit/test_generate_guard.py` | covered |
| P-10 | TinyLlama 权重缓存与内存 | 权重在 `~/.cache/huggingface/`；对齐脚本约需 4GB 内存。M1 16GB 上只做正确性，不做吞吐结论 | 环境注意，非测试 | N/A |
| P-11 | 旧测试是 print-only | 三个 `check_*.py` 没有 `assert`/`sys.exit`，打印 `False` 仍 0 退出；且依赖不存在的 `weights/minillm.pth`。新项目改为 pytest，禁止 print-only | `tests/conftest.py` + 全部 unit 测试 | covered |
| P-12 | 实验目录可被覆盖 | 旧 `train.py --exp` 会 `rmtree` 重建；新约定禁止覆盖/复用 run id，失败也保留 | `tests/unit/test_run_contract.py` | covered |
| P-13 | prefill 默认物化整段 `[seq, vocab]` logits | `model(input_ids)` 返回全序列 logits：32k × 152k 词表 × bf16 ≈ 9.3 GiB，0.8B 的 248k 词表 ≈ 15.2 GiB，24 GB 卡在 0.8B@32k 上直接 OOM。解法：`logits_to_keep=1`，并断言返回的 logits 只有 1 个位置 | `tests/unit/test_hf_engine_guards.py::test_prefill_asks_for_exactly_one_logit`、`::test_full_logits_raise_instead_of_silently_allocating` | covered |
| P-14 | 多模态 checkpoint 的 text 分支 | Qwen3.5 是 `Qwen3_5ForConditionalGeneration`，直接把 `Qwen3_5Config` 传给 `Qwen3_5ForCausalLM` 会报 `vocab_size` 不存在（字段在 `text_config` 里）。解法：`AutoConfig` → `config.get_text_config()` → 显式加载 text decoder，并校验没有 `visual`/`mtp` 参数；否则 vision 塔会静默进显存 | `tests/unit/test_hf_engine_guards.py::test_multimodal_checkpoint_loads_text_only`、`::test_guard_rejects_wrapper_class_and_vision_params` | covered |
| P-15 | 装了 `causal-conv1d` / `fla` 后 CPU 单元测试报 `Expected x.is_cuda()` | transformers 在 import 时把 Qwen3.5 的短卷积/delta rule 解析成 CUDA kernel 并无条件调用，CPU 上的 tiny 模型必炸；不是模型代码错。解法：`tests/conftest.py::_torch_kernel_fallbacks` 把三个符号钉回 `__wrapped__` 的 torch 实现，让快速门禁在有无加速包时都绿 | `tests/conftest.py`、`tests/unit/test_cache_usage.py::test_tiny_qwen35_hybrid_cache_matches_config`、`test_hf_engine_guards.py::test_multimodal_checkpoint_loads_text_only` | covered |
| P-16 | `output_hidden_states` 的最后一项不是最后一块的输出 | transformers 5.x 里 `hidden_states[-1]` 已过 final norm（Qwen3-0.6B：norm 后 `max\|x\|≈54`，真实块输出 `≈1819`）。拿它当 golden 末层，会让"模型其实全对"的实现显示成最后一层差 1.9e3，白查一轮。解法：forward hook 采 `model.layers[i]`，生成侧与对拍侧同一口径 | `scripts/make_golden.py::block_outputs`（hook + 层数断言）、`scripts/align_hf.py::layer_hidden` | covered |
| P-17 | transformers 5.x 把 `rope_theta` 挪进 `rope_parameters` | 扁平属性在 5.x 不存在，`getattr(cfg, "rope_theta", 1e4)` 会**静默**用 1e4（Qwen3 是 1e6），整张逐层表全红且看不出原因；`partial_rotary_factor` / `mrope_section` 也在同一个 dict 里。解法：先读 `rope_parameters`，读不到直接 raise 不猜默认值；非 `default` 的 rope_type 也拒绝 | `tests/unit/test_align_config.py`（7 条） | covered |
| P-18 | RoPE 表按层各存一份 | `Attention.__init__` 每层都 `register_buffer` 一张 40960×64 的 sin/cos：Qwen3-0.6B 有 28 层，bf16 下 56 张表 = **280 MB**，占模型常驻显存的 25%，而每张表逐元素相同。它会伪装成"三分项对不上的显存缺口"，让 `memory_match_ratio` 从 0.99 掉到 0.79。解法：模型层每步算一次 `rope_cos_sin(start, T)` 传给所有 block，层内只留 `head_dim`/`theta` 标量 | `tests/unit/test_rope.py::test_model_keeps_no_rope_table` | covered |
| P-19 | SDPA 的 `is_causal` 不是永远成立 | `is_causal=True` 只假设 query/key 从同一位置开始：full prefill（`past_len=0`）可以，decode（`T=1`）不需要 mask，但 chunked prefill（`past_len>0` 且 `T>1`）必须用带 offset 的显式 mask；padding 也要显式 key mask。写错时不会报错，只会让未来 token 泄漏或 attention 变错。另：SDPA 是 dispatcher，不保证落到 flash（约束不满足会退 math），性能结论前要确认后端 | `tests/unit/test_sdpa_attention.py`（causal / decode / left-pad / chunked / 全掩行） | covered |
