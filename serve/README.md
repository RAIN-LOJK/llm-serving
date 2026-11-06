# serve/ · 推理引擎

> 状态：`engine.py` / `scheduler.py` 已完成 W3（连续批 C1 **2.01×**，C2/C3 pass）；
> W4 的 paged 池已跑完 5 个正式 run（C1 **3.111×**、C2 **0.55%**、C3 全等；
> C4 **1.671×** 负结果，撤回吞吐），口径见 `docs/decisions.md` **D6**；
> `state_pool.py`(W7) 待实现。本文件只固定模块边界，避免把调度/缓存逻辑塞回 `llm_utils/`。

| 模块 | 职责 | 阶段 |
|---|---|---|
| `engine.py` | forward、prefill/decode、采样；单请求 `generate()`；W3 的席位池（`open_pool` / `prefill_into_seat` / `freeze_seat` / `decode_batch`）；W4 的分页池（`open_paged_pool` / `can_admit` / `prefill_into_blocks` / `release_request` / `paged_stats`）。`activate_seat` 与 `fill_from` 只留给 W2 单请求路径与等价性测试 | S2 / W2 / W3 / W4 |
| `scheduler.py` | 请求状态机、准入/退出、static 与 continuous 两条策略的 step 主循环；准入前问引擎的可选 `can_admit`（拿不到块则队头留队列，不跳过） | W3 / W4 |
| `block_manager.py` | 分页 KV 的块分配器：free list、原子预留、整张块表归还、`FragStats`（internal frag / reservation slack / seat row slack，时间加权） | W4 |
| `state_pool.py` | 线性层状态池 + KV 块统一句柄与生命周期 | W7 |

约定：

- `serve/` 只依赖 `model/`，不反向依赖 `bench/` 或 `scripts/`。
- 引擎内部不打印；指标通过返回值/回调交给 `bench/metrics.py`。
- 任何调度/缓存改动先过 `tests/` 的等价性测试（batch 内每请求 == 单独跑）。
- **`freeze_seat(slot)` 必须幂等**，且必须能作用于「从未准入过的行」（`open_pool()` 后 `KvSlab.reset()` 全行 `active=True`，调度器第一次 decode 前先 park 一遍）和「prefill 失败已回退的行」；任何一次准入都必须在 freeze 之后能完全覆盖 `pos` / `valid` / `active`（不依赖行的旧状态）。
- **席位 = `KvSlab` 的一行**（口径见 `docs/decisions.md` D5）：准入 = `prefill_into_seat(row, prompt_ids)`
  把 prefill 的 K/V 直接写穿到该行（`SlabWriteSink` 只写不读，池路径不建 `KVcache`、不 `fill_from`，
  也就没有交接拷贝），成功后才 `activate`；forward 中途失败则由引擎先 `freeze(row)` 回滚半写的行。
  退出 freeze（不回收，回收是 W4 块池的事）；一次 forward 永远覆盖全部 `max_batch` 行，
  空闲行只是被 mask 挡住。不做逐行 forward、不按 active 行重排 KV。

W4 分页路径（`docs/decisions.md` D6）
---

- **池 = `PagedPool`（`model/paged.py`）+ `BlockManager`（`serve/block_manager.py`）**：每层
  `(n_blocks, bs, n_kv_head, head_dim)`；准入一次性预留 `ceil((prompt+budget)/bs)` 块，退出整张归还。
  不做抢占、不做中途分配，所以准入失败只能发生在开头，不会死锁。
- **decode 压缩到活跃行**：`PagedKV` 的 `batch` 是活跃请求数（不是 `max_batch`）；引擎把席号→批行映射
  留在自己手里，logits 散回 `(max_batch, vocab)`。`max_batch` 只是调度上限，不再是显存上限。
- **gather 是稠密的**：每步按块表把活跃行搬成 `(n_active, H_kv, lpad, D)`，`lpad = bucket(本步最长活跃长度, 64)`；
  不足列由**专用零块**填充、逻辑列 mask 关掉。然后仍是一层一次 `hand_baddbmm`（D4 不变）。
  这是「把按最坏长度的预留从显存搬到带宽」的代价（P-32），`gather_read_mb` / `gather_write_mb` 单独记流量。
- **两个池互斥**：`decode_batch(tokens, slab)` 在 `slab is None` 且 paged 池已开时走分页路径；
  `serve/scheduler.py` 用显式 `paged` 属性判定，不靠 `hasattr(engine, "slab")`（未开席位池时该属性会 raise）。
- **两套名字不共用**：W3 席位退出用 `freeze_seat`（冻结游标、保留预留，P-26 原语义），
  W4 请求退出用 `release_request`（归还块、幂等）。不要互相复用。

**W4 实测（`W4-0001…0005`）**：同 1640 MiB 总预算、W3-like trace，seat **9** vs paged **28**
（**3.111×**）；uniform-2048 负对照 1.00×（模拟）；internal frag **0.55%**；paged vs seat
40/40 request `token_ids` 全等；B=8 decode step **21.32 → 35.63 ms（1.671× > 1.5×）**，
因此**不报吞吐**，E3 未跑。同并发多付 workspace 52.42 MiB（+3.6%）。详见
`ROADMAP.md` §5 W4、`results/W4/`、`docs/fig_w4_{capacity,memory,step}.png`。
