# W3-0003 · E1 批处理摊薄 host：ctx512 / B=1 参照

- 状态：`unreviewed`
- 配置：`config.toml`（运行目录 `results/W3/0003-batch-b1-ctx512`）
- 类型：`microbench`；workload：`synthetic`
- 证据：`summary.json` / `metrics.jsonl`

## 假设与判据

- `host_side_cost_is_amortized_by_batching`

## 观察

- `out_tokens=3`（5/5 行），token 序列恒为 `[545, 198, 151645]` = `"ime\n<|im_end|>"`；
  ctx512 的 synthetic prompt 在第 3 个 token 就 EOS。
- 每 trace 只有 2 个 step 行；`active==1` 的那行是 prefill+首个 decode，`prefill_ms=20.25`。
- `decode_only_tok_s=71.04` 把 `active=0` 空转步也算进分母；P1/P2 的 7.67× 只是 B 比的算术镜像。

## 结论（含“什么时候不划算”）

建议：**invalid**（不用于 E1）。根因是 synthetic 随机 prompt 对 base 模型 OOD + greedy + EOS
终止（P-31），不是引擎问题；已有 `W3-0017`（ctx511）取代。

## 限制与偏差

- 3 token/trace，绝对 ms/token 与 R1 都不可用；R1 字面 4.46× 来自 prefill 步，是假阳性。
- scenario 路径每个 ctx 只用一条 prompt，B 行是重复同一条，不是 32 条不同 prompt。

## 下一步

由 `W3-0017`（ctx511/B=1）取代；旧 run 保留为 P-31 证据。
