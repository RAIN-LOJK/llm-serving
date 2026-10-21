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
