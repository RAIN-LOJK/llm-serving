# 实验归档（阶段性记录日志，可整体删除；重要实验不会复用 exp id）
---

| Exp-005 | 2026-08-14 | fineweb_cn 1071.1万 | 4L×128d×4h | 600 iters | 1438 @400 | TODO(用户) | TODO(用户) |

---

## Exp-005
- **日期**：2026-08-14　**代码状态**：main@4c3cd46（+未提交改动）
- **目的**：TODO(用户)
- **数据**
  - 来源：`datasets/fineweb_cn/train/` + `datasets/fineweb_cn/val/`　block=512
  - tokenizer：`datasets/fineweb_cn/tokenizer.json`　vocab=16384
  - 划分：TODO(用户)
- **模型**（`MiniLLMConfig`，完整见 `results/exp005/config.json`）
  ```json
  {
 "vocab_size": 16384,
 "max_position_embeddings": 512,
 "n_embd": 128,
 "n_head": 4,
 "n_kv_head": 4,
 "n_layer": 4,
 "n_inner": null,
 "dropout": 0.0,
 "bias": false,
 "norm_eps": 1e-05,
 "tie_word_embeddings": true
}
  ```
- **训练**（`Config.train`，完整见 `results/exp005/config.json`）
  ```json
  {
 "batch_size": 32,
 "max_iters": 600,
 "warmup_steps": 30,
 "lr": 0.003,
 "min_lr": 0.0001,
 "accum_steps": 1,
 "max_norm": 1.0,
 "log_interval": 50,
 "eval_interval": 200,
 "seed": 0
}
  ```
- **结果**
  | 项 | 值 |
  |---|---|
  | 最优 val ppl | 1438 @ step 400 |
  | 末步 val ppl | 1438 @ step 400 |
  | checkpoint ppl / acc | 1218 / 0.049 |
  | 多样性 | — |
  | 拐点 | TODO(用户) |
  | 轨迹 | `results/exp005/train_log.csv` |
  | 生成样本 | `results/exp005/samples.txt`（temp=0.8, max_out=64） |
  | 配置 | `results/exp005/config.json` |
- **观察**：TODO(用户)
- **结论**：TODO(用户)
- **复现**：`python eval.py --exp exp005` | `python train.py --config <对应toml>`
