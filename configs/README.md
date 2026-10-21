# configs/

| 路径 | 用途 |
|---|---|
| `thresholds.toml` | 目标值/判失败线；W1 后校准 |
| `experiments/_template.toml` | 新 run 的模板 |
| `experiments/W0/0001-carrier-probe.toml` | W0 双载体探测（正式） |
| `experiments/W0/9001-runner-smoke.toml` | runner 契约冒烟（dummy，不是证据） |

约定：

- 一个 run 一个 TOML，自包含；run 目录里会复制一份 `config.toml` 快照。
- 模型 `revision` 必须是 HF commit hash，不能是 `main`。
- 阈值不写进配置内联；`[accept].criteria` 引用 `thresholds.toml` 的键。
- 配置改了就是新 run；不要拿旧 run 的 summary 配新配置。
