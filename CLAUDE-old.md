# MiniLLM — Learning Project

手写 LLaMA 风格 decoder-only transformer，用于学习 LLM 架构，未来延伸到 DeepSeek 方向（MLA / DeepSeekMoE / R1-RL）。

## ⚠️ 最重要的约定：这是学习项目

核心组件（RoPE、Attention、RMSNorm、SwiGLU MLP、采样、训练循环）**由用户手写**，不使用 HF 封装组件。agent 的任务是：
- 讲原理、给骨架、留 TODO，让用户自己实现
- 不要直接往学习文件里贴完整 HF 实现
- 用户写出问题后做 review 和 debug

## 实验导向：模型结构是临时的

当前 model 结构（vocab / 层数 / 维度）是**随意设计的实验配置**，非最优调参。
全流程整合完成后，会**重新设计不同网络结构**适配不同任务、对比观察效果。
因此组件与数据设计（tokenizer 粒度、vocab 规模等）要为**未来的结构实验**保留通用性，
不被当前 toy 配置锁死。

## Current state

- 设备Mac M1，mps已安装
- `model/`：模型定义包。`layers.py`（RoPE、GQA Attention、KVcache、SwiGLU MLP、RMSNorm、TransformerBlock）全手写；`minillm.py`（MiniLLM + HF 集成，原 `model.py` 改名）；`config.py`（MiniLLMConfig）
- `llm_utils/`：对模型的操作。`inference.py` 手写采样（temperature / top-k / top-p）+ KV cache 生成（prefill + 逐 token decode，支持左 pad + EOS 发散）
- `tools/train_debug.py`：复读机任务（输入 0..62，预测 1..63），loss 从 ~5.5 降至 < 0.001，63/63 全对（D4 GRPO 沙盒可复用其任务定义）
- `tests/`：验收/回归/基准脚本（均已加 sys.path 引导，从 repo 根运行）
  - `check_kvcache.py`：KV cache 一致性校验（cache 生成 vs 全量重算逐 token 对比）
  - `check_leftpad_batch.py`：A3.1 验收，左 pad 变长 batch vs 单序列参考
  - `check_eos_batch.py`：A3.2 验收，EOS 发散 batch 各行按各自 EOS 提前结束
  - `bench_kvcache.py`：A3.3 加速比基准（cache vs 全量，随 seq 长画曲线）

> 代码结构随学习推进会变化，以上目录仅供参考，不要写死。

## Run

```bash
.venv/bin/python tools/train_debug.py      # 训练复读机任务
.venv/bin/python train.py                  # 变长训练，产出 weights/minillm.pth
.venv/bin/python tests/check_kvcache.py    # KV cache 一致性校验（回归）
.venv/bin/python tests/check_leftpad_batch.py  # A3.1 左 pad 验收（回归）
.venv/bin/python tests/check_eos_batch.py  # A3.2 EOS 发散验收（回归）
.venv/bin/python tests/bench_kvcache.py    # A3.3 加速比基准，产出 bench_kvcache.png
```

Venv at `.venv/`，所有命令通过 `.venv/bin/python`。

## 踩过的坑

见 `PITFALLS.md`（已剥离，按需读取，不常驻上下文以节省开销）。
遇到 transformers 兼容 / RoPE / RMSNorm / KV cache 相关 bug 时，先读它排查已知坑。

## User preference

- 交互式、逐步推进：不要一次批量改多个文件而不同步
- 每步先解释原理再实现，用户自己驱动节奏
- 优先让用户自己写，agent 提示 + 把关
- **实验操作命令（训练 / eval / 可视化）由用户自己执行**：agent 只给命令、给数据结论，不代跑（有获得感优先）
