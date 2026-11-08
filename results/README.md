# results/ · run 目录

这里放**不可变**的实验 run。一个目录 = 一次 run = `config.toml` + `manifest.json` +
`env.json` + `metrics.jsonl` + `summary.json` + `notes.md` + `verdict.json` + `reproduce.sh`。

- 命名：`results/<stage>/<4位序号>-<slug>/`，例如 `results/W0/0001-carrier-probe/`。
- `run_id`：`<stage>-<4位序号>`，例如 `W0-0001`。
- 不覆盖、不复用 id；失败/作废也保留。
- `INDEX.md` 由 `scripts/exp_index.py` 生成，**不要手改**。
- 契约全文见 [docs/experiments.md](../docs/experiments.md)。

## 状态

| 字段 | 取值 | 谁写 |
|---|---|---|
| `manifest.status` | `running / completed / failed / aborted` | runner |
| `verdict.verdict` | `unreviewed / pass / fail / inconclusive / invalid` | 人 |

## 常用命令

```bash
# 一个 run
python -m bench.run --config configs/experiments/W0/0001-carrier-probe.toml

# 重建索引
python scripts/exp_index.py

# 校验（CI/测试用）
python scripts/exp_index.py --check
```

## 当前

W0–W4.5 的正式 run 已全部落在本目录；完整清单与关键指标见 [INDEX.md](INDEX.md)。
`9001-runner-smoke` 只用于验证管线，不作为证据。
