# 实验记录

> 规则：ID 流水号 Exp-XXX，由 eval.py / train.py 自动分配（读本文件最大号 +1）。
> 每个实验 = `results/expXXX/` 自包含文件夹：config.json（配置快照）、train_log.csv（逐 step 轨迹）、samples.txt（固定 prompt 生成）、model.pth（checkpoint）、resume.pt（续训状态）。
> 改动代码须记 git 状态；结论只记可复现的（归因要能对上具体配置）。

## 索引表

| ID | 名称 | 日期 | 语料 | 结构 | 预算 | 最优 val ppl | 拐点 | 结论 |
|---|---|---|---|---|---|---|---|---|
| Exp-004 | gqa-2 | 2026-08-13 | luxun 258.4万 | 2L×128d×4h/2kv | 500 iters | 1012 @450 | ≥450 未见拐点（欠拟合端） | TODO(用户) |
| Exp-003 | scale-128 | 2026-08-13 | luxun 258.4万 | 2L×128d×4h | 500 iters | 975 @450 | TODO(用户) | TODO(用户) |
| Exp-002 | scale-64 | 2026-08-13 | luxun 258.4万 | 2L×64d×4h | 500 iters | 1043 @450 | TODO(用户) | TODO(用户) |
| Exp-001 | baseline-b2e | 2026-08-12 | luxun 87.9万 | 4L×128d×4h | 1200 iters | 1214 @400 | ~400 | 过拟合拐头 400；语料只够词级档位 |
| Exp-005 | 2026-08-14 | fineweb_cn 385.7万字符 | 4L×128d×4h | 3000 iters | 857 @600 | 拐点@600，词级✓句级✗ | 规模不足，玩具定位，C2 收尾 |

---

## Exp-001
- **日期**：2026-08-12　**代码状态**：kv-cache@4c3cd46（+未提交的 B2 全链路改动）
- **目的**：真实语料上验证全链路 + 建立 val ppl 基线；观察过拟合点
- **数据**
  - 来源：`datasets/luxun/{train,val}`（B2b 产物）
  - tokenizer：`datasets/luxun/tokenizer.json`　vocab=8192　压缩率 0.73
  - 规模：train 285篇 87.9万字符 / val 21篇 12万字符（≈120 万 token）
  - 划分：固定 train/val，朝花夕拾+故事新编 → val
- **模型**（`MiniLLMConfig.to_dict()`，完整见 `results/exp001/config.json`）
  ```json
  {"vocab_size": 8192, "max_position_embeddings": 256, "n_embd": 128,
   "n_head": 4, "n_kv_head": 4, "n_layer": 4, "dropout": 0.0, "bias": false,
   "norm_eps": 1e-5, "tie_word_embeddings": true}
  ```
- **训练**（`Config.to_dict()` 的 train 段，完整见 `results/exp001/config.json`）
  ```json
  {"batch_size": 32, "max_iters": 1200, "warmup_steps": 50, "lr": 3e-3,
   "min_lr": 1e-4, "accum_steps": 1, "max_norm": 1.0,
   "log_interval": 50, "eval_interval": 200, "seed": 0}
  ```
- **结果**
  | 项 | 值 |
  |---|---|
  | 最优 val ppl | 1214 @ step 400 |
  | 末步 val ppl | 1848 @ step 1000 |
  | 拐点 | ~400（train 继续降 / val 回升 → 过拟合） |
  | 轨迹 | `results/exp001/train_log.csv`（6 个 eval 点，补记） |
  | 生成样本 | `results/exp001/samples.txt`（temp=0.8，prompt 前128 续写64） |
  | 配置 | `results/exp001/config.json` |
- **观察**：词级像中文（书名号/`〔5〕` 注释格式都学到）；句级/段落级乱
- **结论**：管线闭环成立，val ppl 主指标可用；语料 120 万 token 只够词法档位，段落可读需扩语料/SFT（移出主线验收线）；预算 1200 偏大（过拟合），结构实验须重定基线到欠拟合端（→ B3a）
- **复现**：`python train.py`（当时 config/experiment.toml，budget 已改；轨迹为补记非实时）

---

## Exp-002
- **日期**：2026-08-13　**代码状态**：main@4c3cd46（+未提交改动）
- **目的**：TODO(用户)
- **数据**
  - 来源：`datasets/luxun/train/` + `datasets/luxun/val/`　block=256
  - tokenizer：`datasets/luxun/tokenizer.json`　vocab=8192
  - 划分：TODO(用户)
- **模型**（`MiniLLMConfig`，完整见 `results/exp002/config.json`）
  ```json
  {
 "vocab_size": 8192,
 "max_position_embeddings": 256,
 "n_embd": 64,
 "n_head": 4,
 "n_kv_head": 4,
 "n_layer": 2,
 "n_inner": null,
 "dropout": 0.0,
 "bias": false,
 "norm_eps": 1e-05,
 "tie_word_embeddings": true
}
  ```
- **训练**（`Config.train`，完整见 `results/exp002/config.json`）
  ```json
  {
 "batch_size": 32,
 "max_iters": 500,
 "warmup_steps": 50,
 "lr": 0.003,
 "min_lr": 0.0001,
 "accum_steps": 1,
 "max_norm": 1.0,
 "log_interval": 50,
 "eval_interval": 50,
 "seed": 0
}
  ```
- **结果**
  | 项 | 值 |
  |---|---|
  | 最优 val ppl | 1043 @ step 450 |
  | 末步 val ppl | 1043 @ step 450 |
  | checkpoint ppl / acc | 1056 / 0.056 |
  | 多样性 | — |
  | 拐点 | TODO(用户) |
  | 轨迹 | `results/exp002/train_log.csv` |
  | 生成样本 | `results/exp002/samples.txt`（temp=0.8, max_out=64） |
  | 配置 | `results/exp002/config.json` |
- **观察**：TODO(用户)
- **结论**：TODO(用户)
- **复现**：`python eval.py --exp exp002` | `python train.py --config config/scale-64.toml`（名称 scale-64）
---

## Exp-003
- **日期**：2026-08-13　**代码状态**：main@4c3cd46（+未提交改动）
- **目的**：TODO(用户)
- **数据**
  - 来源：`datasets/luxun/train/` + `datasets/luxun/val/`　block=256
  - tokenizer：`datasets/luxun/tokenizer.json`　vocab=8192
  - 划分：TODO(用户)
- **模型**（`MiniLLMConfig`，完整见 `results/exp003/config.json`）
  ```json
  {
 "vocab_size": 8192,
 "max_position_embeddings": 256,
 "n_embd": 128,
 "n_head": 4,
 "n_kv_head": 4,
 "n_layer": 2,
 "n_inner": null,
 "dropout": 0.0,
 "bias": false,
 "norm_eps": 1e-05,
 "tie_word_embeddings": true
}
  ```
- **训练**（`Config.train`，完整见 `results/exp003/config.json`）
  ```json
  {
 "batch_size": 32,
 "max_iters": 500,
 "warmup_steps": 50,
 "lr": 0.003,
 "min_lr": 0.0001,
 "accum_steps": 1,
 "max_norm": 1.0,
 "log_interval": 50,
 "eval_interval": 50,
 "seed": 0
}
  ```
- **结果**
  | 项 | 值 |
  |---|---|
  | 最优 val ppl | 975 @ step 450 |
  | 末步 val ppl | 975 @ step 450 |
  | checkpoint ppl / acc | 994 / 0.064 |
  | 多样性 | — |
  | 拐点 | TODO(用户) |
  | 轨迹 | `results/exp003/train_log.csv` |
  | 生成样本 | `results/exp003/samples.txt`（temp=0.8, max_out=64） |
  | 配置 | `results/exp003/config.json` |
- **观察**：TODO(用户)
- **结论**：TODO(用户)
- **复现**：`python eval.py --exp exp003` | `python train.py --config config/scale-128.toml`（名称 scale-128）
---

## Exp-004
- **日期**：2026-08-13　**代码状态**：main@4c3cd46（+未提交改动）
- **目的**：TODO(用户)
- **数据**
  - 来源：`datasets/luxun/train/` + `datasets/luxun/val/`　block=256
  - tokenizer：`datasets/luxun/tokenizer.json`　vocab=8192
  - 划分：TODO(用户)
- **模型**（`MiniLLMConfig`，完整见 `results/exp004/config.json`）
  ```json
  {
 "vocab_size": 8192,
 "max_position_embeddings": 256,
 "n_embd": 128,
 "n_head": 4,
 "n_kv_head": 2,
 "n_layer": 2,
 "n_inner": null,
 "dropout": 0.0,
 "bias": false,
 "norm_eps": 1e-05,
 "tie_word_embeddings": true
}
  ```
- **训练**（`Config.train`，完整见 `results/exp004/config.json`）
  ```json
  {
 "batch_size": 32,
 "max_iters": 500,
 "warmup_steps": 50,
 "lr": 0.003,
 "min_lr": 0.0001,
 "accum_steps": 1,
 "max_norm": 1.0,
 "log_interval": 50,
 "eval_interval": 50,
 "seed": 0
}
  ```
- **结果**
  | 项 | 值 |
  |---|---|
  | 最优 val ppl | 1012 @ step 450 |
  | 末步 val ppl | 1012 @ step 450 |
  | checkpoint ppl / acc | 1036 / 0.062 |
  | 多样性 | — |
  | 拐点 | ≥450 未见拐点（欠拟合端） |
  | 轨迹 | `results/exp004/train_log.csv` |
  | 生成样本 | `results/exp004/samples.txt`（temp=0.8, max_out=64） |
  | 配置 | `results/exp004/config.json` |
- **观察**：TODO(用户)
- **结论**：TODO(用户)
- **复现**：`python eval.py --exp exp004` | `python train.py --config config/gqa-2.toml`（名称 gqa-2）
---

## Exp-005
- **日期**：2026-08-14　**代码状态**：main@4c3cd46（+未提交改动）
- **目的**：C2 语料升级（fineweb_cn，234 万 token）验证"规模→句级通顺"假设；全链路（tokenizer 重训→训练→拐点提档→eval 归档）打通
- **数据**
  - 来源：`datasets/fineweb_cn/train/` + `datasets/fineweb_cn/val/`　block=512
  - tokenizer：`datasets/fineweb_cn/tokenizer.json`　vocab=16384
  - 划分：train 1499 篇 385 万字符 / val 233 篇 57 万字符（≈234 万 / 35 万 token，压缩率 0.607）
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
 "max_iters": 3000,
 "warmup_steps": 150,
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
  | 最优 val ppl | 857 @ step 600 |
  | 末步 val ppl | 857 @ step 600 |
  | checkpoint ppl / acc | 885 / 0.075 |
  | 多样性 | — |
  | 拐点 | @600 ppl 856（3000 调度全轨迹最低；@800 901 / @1000 977 单调回升）
  | 轨迹 | `results/exp005/train_log.csv` |
  | 生成样本 | `results/exp005/samples.txt`（temp=0.8, max_out=64） |
  | 配置 | `results/exp005/config.json` |
- **观察**：生成词级像现代中文（词序/标点/结构词"金融风险""欧盟国家"），句级语义跳脱（"其国体的合作性承担,商品定价,性监管"）、偶发低概率生造词；@601 提档 eval ppl 885 / acc 0.075；过拟合损失小（885 vs 951）不改变生成质量台阶
- **结论**：234 万 token 到不了句级——数据翻倍（luxun 120万→234万）仅推进一档词法；句级需 20-50M 参数 + 1-3 亿 token（M1 单次 14h+ 不可行）→ 玩具定位成立，规模结论入档，C2 收尾
- **复现**：`python eval.py --exp exp005` | `python train.py --config <对应toml>`

