# 踩过的坑（高信号，按需读取）

> 从 CLAUDE.md 剥离，仅在排查 transformers / RoPE / RMSNorm / KV cache 相关 bug 时读取，
> 不常驻上下文以节省开销。

- **transformers v5 变更**：`_tied_weights_keys` 现在是 dict `{目标: 源}`（如 `{"lm_head.weight": "token_embedding.weight"}`），不再是 list。写成 list 会报 `'list' object has no attribute 'keys'`
- **RoPE 两种约定不兼容**：interleaved（LLaMA 论文伪代码，按 `(2k, 2k+1)` 配对）vs GPT-NeoX（HF/TinyLlama 用，前后两半配对 `rotate_half`）。两者等价但差一个固定维度置换，**与同一组权重不兼容**。本项目统一用 GPT-NeoX；加载 HF 权重前先确认 `apply_rotary_emb` 的约定
- **`config.norm_eps` 必须接线**：RMSNorm 构造要显式传 `config.norm_eps`（TinyLlama 是 1e-5），否则默默用类默认值，和 HF 对不上，L1 会差一个量级
- **KV cache 三个隐性坑**：① 层间缓存绝不拼接，`kvs[l]` 只存本层随时间累积的 (k, v)；② 因果 mask 的 `diagonal` 要偏移 `past_len+1`，否则 decode 新 token 只能看到历史开头；③ `past_len` 要按层取（`kvs[idx]` 的长度），prefill 时各层历史为空，用全局长度会让后续层 RoPE/mask 错位
- **decode 的 pad mask 必须"全长度"**：key 序列 = [历史 past_len 列] + [新 1 列]，pad 全在历史左侧（左 pad 前对齐）。只传新列 mask `(B,1,1,1)` 会泄漏 cached pad 列，新 token 能看穿到 pad 里的随机信息。解法：`generate` 每步 `concat` 追加一列 1，宽度 = key 总长；"只能追加 1"成立依赖左 pad 前对齐（RoPE 共享右列 → 共享 past_len）
- **A3.2 账本对齐**（已修）：EOS 发散后原始 B 空间与压缩空间错位。解法：buffer + orig 映射统一到压缩空间，行结束即 flush；**`break` 前 buffer 与 orig 必须同形**（全行同时到 EOS 时若提前 break 会索引空 buffer）
- **RoPE 表越界不报错，空切片级联成形状错误**：decode 到位置 ≥ max_position_embeddings 时 `cos[past_len:past_len+T]` 返回**空张量** (0, dim)（`[64:65]` 是合法切片不是越界异常）。空维与广播合法扩张成 0，q/k 的 T 维被静默抹成 0 → attn 的 key 维（来自被污染的实际张量）与 mask 的 key 维（来自计数器 `past_len+T`）分歧，最终在 `attn + mask` 报"64 vs 65"这种看似 mask 公式错误的形状不匹配。**根因是越界不是 mask**。解法：`generate` 加护栏 `past_len(0)+1 >= max_position_embeddings` 就停；批量左 pad 时护栏按 KV 宽度（含 pad 的 T_max）判断，**所有行的生成预算 = max_position_embeddings - T_max（共享预算）**
- **`generate` 输出契约：护栏中断必须截断到实际采样数**：`buffer` 是 `torch.zeros` 预分配，早停后整行 flush 会把"没采样过的 0"当结果返回，而 token 0 合法无法区分。解法：循环外声明 `sampled` 计数器（不要用逃逸的循环变量 `step+1`），EOS 与最终 flush 统一 `buffer[idx, :sampled]`
- TinyLlama 权重缓存于 `~/.cache/huggingface/`，对比脚本约需 4GB 内存
