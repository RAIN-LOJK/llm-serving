# serve/ · 推理引擎

> 状态：**待 W2–W7 实现**。本文件只固定模块边界，避免把调度/缓存逻辑塞回 `llm_utils/`。

| 模块 | 职责 | 阶段 |
|---|---|---|
| `engine.py` | 单请求 forward、prefill/decode、采样；对外 `generate()` | S2 / W2 |
| `scheduler.py` | 请求队列、每步准入/退出、连续批处理 | W3 |
| `block_manager.py` | 分页 KV：block table、按需分配/回收、物理块池 | W4 |
| `state_pool.py` | 线性层状态池 + KV 块统一句柄与生命周期 | W7 |

约定：

- `serve/` 只依赖 `model/`，不反向依赖 `bench/` 或 `scripts/`。
- 引擎内部不打印；指标通过返回值/回调交给 `bench/metrics.py`。
- 任何调度/缓存改动先过 `tests/` 的等价性测试（batch 内每请求 == 单独跑）。
