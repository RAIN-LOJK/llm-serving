# serve/ · 推理引擎

> 状态：`engine.py` / `scheduler.py` 已落地并完成 W3（连续批 C1 **2.01×**，C2/C3 pass；见 `ROADMAP.md` §5）；
> `block_manager.py`(W4) 与 `state_pool.py`(W7) 待实现。
> 本文件只固定模块边界，避免把调度/缓存逻辑塞回 `llm_utils/`。

| 模块 | 职责 | 阶段 |
|---|---|---|
| `engine.py` | forward、prefill/decode、采样；单请求 `generate()`，以及 W3 的 KV 池（`open_pool` / `prefill_into_seat` / `freeze_seat` / `decode_batch`；`activate_seat` 与 `fill_from` 只留给 W2 单请求路径与等价性测试） | S2 / W2 / W3 |
| `scheduler.py` | 请求状态机、席位准入/退出、static 与 continuous 两条准入策略的 step 主循环 | W3 |
| `block_manager.py` | 分页 KV：block table、按需分配/回收、物理块池 | W4 |
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
