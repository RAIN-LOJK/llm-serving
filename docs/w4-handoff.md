# W4 开跑交接（新对话读这个）

## 当前状态
W4 分页 KV 的代码、配置、判据、文档全部落地，**尚未跑任何正式 run**。
快速门禁全绿（`python -m pytest -m "not slow and not hf"` → 477 passed, exit 0）。
**工作树是 dirty 的，正式 run 前必须 git commit**（否则 runner 拒绝，除非 `--allow-dirty`，而 dirty 的 run 不能引用）。

## 口径（唯一权威）
- `docs/decisions.md` **D6** ← 总预算 / 容量 / 碎片 / ±1 门禁
- `ROADMAP.md` §5 W4 ← 验收条件；§13 v14 ← 为什么改
- `docs/w4-plan.md` ← 实施计划（W4 完成后要删）

## 关键数字（解析/模拟，**不是实测**）
- 总预算 **1640 MiB**（KV + workspace），W3-like 混合长度
- seat 行宽 **1664**（最长 prompt 1517 + budget 128 + 1，桶到 64）= **182 MiB/行** → `max_batch=9`
- paged `kv_blocks=800` → 模拟准入 **28** 条 → **3.11×**
- uniform-2048 是负对照（≈1.0×），只报不判

## 开跑顺序（不要跳）
```bash
# 0. 先提交，否则正式 run 会被拒
git add -A && git commit

# 1. 门禁
python -m pytest -m "not slow and not hf"
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py -q

# 2. E0 数值门（不过就停，不许跑 E1/E2/E3）
python -m bench.run --config configs/experiments/W4/0001-paged-numeric-gate.toml
python scripts/check_paged_golden.py results/W4/0001-paged-numeric-gate   # 期望 exit 0

# 3. E1 容量
python -m bench.run --config configs/experiments/W4/0002-seat-fill-max2048.toml
python -m bench.run --config configs/experiments/W4/0003-paged-fill-max2048.toml
python scripts/sim_paged_capacity.py \
  --trace results/W4/0003-paged-fill-max2048/metrics.jsonl \
  --paged-blocks 800 --expect-concurrency <0003 的 capacity.peak_concurrency>
# 期望 28 ±1；同时核对 0003 的 capacity.kv_total_mb ≤ 1640

# 4. E2 代价（可选，但 C4 靠它）
python -m bench.run --config configs/experiments/W4/0004-seat-b8-cost.toml
python -m bench.run --config configs/experiments/W4/0005-paged-b8-cost.toml
# 判：fragmentation_max ≤5%；step_ms 比值 >1.5× 则撤回吞吐结论

# 5. E3 吞吐（可选，record-only）
# λ=30 是预注册估计；若 E1 实测容量与 28 差得多，先按 λ=2×capacity 重算再跑
python -m bench.run --config configs/experiments/W4/0006-seat-saturate.toml
python -m bench.run --config configs/experiments/W4/0007-paged-saturate.toml
```

## 已经踩过的坑（不要重踩）
1. **`gather_ms` / `gather_read_mb` / `gather_write_mb` 必须存在**——在正式 run 之前补的，
   所以现在所有 run 在同一 commit 上。别在 E1 之后改 engine 代码。
2. **workspace 只保留当前 lpad 桶**（修过一个会累积到 5.6 GiB 的 bug）；
   `workspace_mb()` 是占用，`gather_read_mb` 是流量，两者不混。
3. **E0 用 `prompt_ids = ["s01"]` 锁定**，不靠 `--limit`（scenario 分发下 `--limit` 不生效）。
4. **`fill` 的容量 = `max(step_row.active)`**；fill 不 decode，所以 `out_tokens=0` 是正常的。
5. **E2 的 8 条长 prompt** 用 `prompt_ids = ["l01"..."l08"]` 锁定（runner 的 `prompt_ids` 键）。

## 还没做
- `scripts/plot_w4.py` + `fig_w4_memory.png` / `fig_w4_step.png`（要有 E2 数据才画）
- 删 `docs/w4-plan.md`（plan 要求 W4 完成即删，删时同步 ROADMAP §5 链接 / §12 W4 行 / §13 一条修订）
- ROADMAP §12 的 W4 目前是 🟡，跑完回填 ✅ 与证据 run id
