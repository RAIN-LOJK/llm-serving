# W3 · 连续批处理（详细开工方案）

> **生命周期：临时文件，W3 阶段完成即删。** 它是「新开一个对话就能开工」的施工方案，不是长期文档：
> W3 的三个实验（E0–E2）判完、结论回填 `ROADMAP.md` §12/§13 与 `docs/decisions.md` 之后删除本文件
> （若 W3 的方案被整体推翻，也直接删）。**判据仍以 `ROADMAP.md` §5 W3 为准**——本文件不承载任何判据。
> **删除时必须同步修掉的引用**（本文件被链接的地方）：
>
> | 位置 | 内容 | 删除时怎么改 |
> |---|---|---|
> | `ROADMAP.md` §5「W3 · 连续批处理」 | 开工清单与参考实现的两处 `docs/w3-plan.md`；`poisson_trace` 的占位说明 | 把已落地的结论就地写进 §5，去掉链接；`poisson_trace` 的 `trace_sha256` 约定搬到 `docs/experiments.md` |
> | `ROADMAP.md` §12 进度表 | W3 行由 ⬜ 改 ✅ | 回填证据 run |
> | `ROADMAP.md` §13 v9 修订记录 | 历史条目里的 `docs/w3-plan.md` | 历史不改写；断链可接受 |
> | `ROADMAP.md` §14 参考实现索引 | W3 行的「实现细节见 `docs/w3-plan.md`」 | 改成指向 `docs/decisions.md` 的 **D5**（W3 负载与口径） |

> **前置**：配对基线 **`W2-0007`**（固定注意力路径，四条判据 pass；HF 分母用同机的 `W2-0005`）。
> `W2-0003`（旧口径 sdpa）、`W2-0004`（跨实例配对，`invalid`）与 `W2-0006`（被取代）都不能当基线。
> **两份 ADR 分工，不要混**：
> - **[D4](decisions.md)（accepted，不要改）** 只定注意力路径——prefill（T>1）逐序列 SDPA，
>   decode（T==1）`hand_baddbmm` + `KvSlab`。本文件凡说「D4」处都指这一层。
> - **D5（W3 开工第一件事新立，proposed）** 定 W3 自己的负载与指标口径：trace/slot/饱和/吞吐双口径/判据。
>   §2.1 就是 D5 的正文草案；**不要把它写进 D4**，否则 accepted ADR 会被改乱或出现两个 D4。
> W3 的调度/到达过程参考见 §13。
> 本文件是「新开一个对话就能直接开工」的自包含方案；阶段目标与判据仍以 `ROADMAP.md` §5 W3 为准。
> **状态：计划（未跑任何 W3 run）**。文中所有数字分成三类，读的时候不要混：
> `实测`（已有 run）、`探针`（规划期一次性 ad-hoc 测量，**不进报告**）、`预测`（预注册，待验证）。
> **§0.1/§0.2 的探针是 D4 之前测的**（表里还留着 `SDPA 0.24 ms` 的旧 decode 路径），只当量级参考；
> W3 的定量依据是 E1 的实测，不是它们。

---

## 0. 规划期核对到的三件事（决定了 W3 的重心）

> 本节全部是 **D4 之前的探针**（decode 那时还走 SDPA + `cat` 的 `KVcache`）。量级结论（host 主导、
> GPU 侧成本对 B 不敏感）仍然成立，但「SDPA 0.24 ms / `cat` 成本」这两项**不代表 shipped 路径**：
> D4 之后 decode 是 `hand_baddbmm`（0.7 ms/step，B=8/L=512）+ 定长 slab（**没有 `cat`**）。
> 引用时只引用趋势；W3 的定量依据是 E1。

### 0.1 现有基线的真实结构（探针，不引用进报告）

在 4090 上用 `serve/engine.py` 的单请求路径测一个 decode step（ctx≈90）：

| 项 | 值 | 含义 |
|---|---|---|
| 每 step 墙钟 | **19.6 ms** | 与 `.item()` 同步的版本相同（19.6），说明瓶颈不在同步 |
| 每 step GPU 忙时间 | **≈3.3 ms** | GEMM 1.66 + 逐元素 1.4 + attention 0.24（**D4 前的 SDPA 路径；现在 `hand_baddbmm`**） |
| 每 step CPU op 数 | **≈705**（其中 197 个 matmul） | 28 层 × 25 个 eager op |
| 单 kernel launch 成本 | 5.2 µs | 空的 `add_` 实测（本机 CPU 不算慢） |
| 权重 roofline | 1137 MiB / ≈700 GB/s ≈ **1.7 ms** | bf16 一 token 的理论下界 |

→ **host 侧约 16 ms/step 是纯粹按「每 token」付的钱**。这 16 ms 与 batch 大小几乎无关（op 数不变），
这就是 W3 唯一真正大的杠杆：**批处理摊薄的是 host，而不是 GPU**。

### 0.2 同一组权重做 B 条序列的 GPU 时间几乎不涨（探针）

用真实 Qwen3-0.6B 权重形状跑「28 层 × (qkv+o+mlp) + lm_head」，M=B、T=1（**纯 GEMM 形状探针，不含注意力；
当时 decode 还是 SDPA，现在按 D4 换成 `hand_baddbmm`，趋势不变但绝对值要重测**）：

| B | GPU 侧 step 时间 | 每 token GPU |
|---|---|---|
| 1 | 6.17 ms | 6.17 ms |
| 8 | 6.30 ms | 0.79 ms |
| 32 | 6.93 ms | 0.22 ms |
| 64 | 6.99 ms | 0.11 ms |

（B=1 那一格本身已经混了 host 饥饿，所以看的是**趋势**，不是绝对值：B=8→32 只涨 10%。）
→ 权重读取带宽受限，因此「同一 step 的 GPU 成本几乎与 B 无关」。
结合 §0.1：B=8 每 step ≈ 16(host) + 4~7(GPU) ≈ 20~23 ms，摊到 8 个 token ≈ **2.5~2.9 ms/token**。
**注意**：这是「静态批也能拿到的收益」，不是「连续批的收益」。本方案把两件事分开测（§3 E1 / E2）。

### 0.3 现有 prompt 集的输出长度完全一致（实测 `W2-0003`）

32 条 prompt 贪心 128 token，**全部 32 条 `out_tokens = 128`**（base 模型不吐 EOS）。
这条直接决定了一件容易被忽略的事：

> 如果所有请求都恰好 128 token，那么「静态批 vs 连续批」的吞吐差异只剩下**波次取整**（64/8=8 整）
> 和 prefill 覆盖，理论比值 ≈ **1.00×**。此时 1.3× 的判据**按构造成立不了**。

所以 W3 必须显式区分两个问题（否则会得到一个「假阴性」）：
- **Q1 批处理本身能否摊薄 host**（与长度分布无关）；
- **Q2 调度在什么长度分布下回本**（需要长度异质性；用 trace 的 `budgets` 显式表达，
  并把「均匀长度 → 收益≈1.0×」作为负结果专章的一行）。

### 0.4 顺带发现的一个契约陷阱

`[workload].batch_sizes` **目前只是行标签，不是并发度**：`_cases_for_model` 会把同一 prompt
重复 `B` 条 case，`_execute` 逐条串行 `generate`。W2 的配置注释已经写了「真批处理要等 W3」。
W3 起：**并发度 = `[engine].max_batch`**，`batch_sizes` 在 scenario 模式下必须为 `[1]`，
否则 runner 报错（登记为 P-20，见 §9）。

---

## 1. 目标与判据（预注册）

| # | 判据 | 目标 / 失败线 | 评测方式 | 阈值键 |
|---|---|---|---|---|
| C1 | 高 λ 吞吐（连续批 / 静态批，同 trace 同 B） | **≥ 1.30×**；< 1.20× 记负结果收口 | 配对 run 比值 | `[W3.continuous].throughput_speedup_min` |
| C2 | p99 TTFT 可分解为「排队 + 计算」 | 结构判据：`queue_ms`、`prefill_ms`、`ttft_ms` 三者的 p99 都存在且 `ttft ≈ queue + prefill`（逐请求误差 < 1 ms） | summary 结构检查 | `[W3.continuous].ttft_p99_decomposed` |
| C3 | 不同到达顺序输出一致 | 逐 request 的 token 序列**逐 id 全等** | 单测（两种 trace）+ 配对 run 的 `token_sha256` 交叉核对 | 单测 + `consistency` 块 |
| C5 | 批处理摊薄 host（Q1，非机制判据） | 记录 `tok/s(B)` 曲线；预测 B=8 时 ≥ 5× | E1 五个 run，record_only | 无（图） |

**预注册预测（publish-then-measure，跑完逐条判真假）**

| # | 预测 | 依据 |
|---|---|---|
| P1 | B=1→8 每 token 成本降 ≥ 5×（ctx 512） | 0.1/0.2 探针：host 16 ms 被摊成 2 ms |
| P2 | B=8→32 每 token 再降 2.5~3×，但 step 时间开始被 slab 读取量抬高 | slab 的 K/V 读取量 ∝ B×ctx（**不是 `cat`**：D4 之后 decode 写定长 slab，没有每步 concat） |
| P3 | 均匀 128 token 负载：连续/静态 ≈ 1.00×（**负结果**） | 0.3 实测 + 波次取整 |
| P4 | 混合长度（{16,32,64,128} 均匀抽样）饱和负载：连续/静态 **1.8–2.2×**（解析上限 E[max₈]/E[mean] = **2.02×**；B→∞ 的极限是 2.13×，B=32 时为 2.13×） | 解析式（`bench/scenario.py::analytic_straggler_ratio`，按真实 `max_batch` 算，不是硬编码常量）。该比值由 `budgets` 分布直接决定，属**构造性推论**，不是独立预测 |
| P5 | 饱和 trace 的 p99 TTFT 中排队占比 > 60% | 过载必然排队 |
| P6 | B=8 时 decode 阶段的注意力占比 < 10% | `hand_baddbmm` 实测 0.7 ms/step（B=8/L=512）；D4 已把它定为唯一 decode 路径 |

---

## 2. 设计（先定口径，再定接口）

### 2.1 口径冻结（写成 **D5（proposed）**，跑之前定死；**D4 已 accepted，不要再改**）

| 项 | 定义 |
|---|---|
| **trace** | `n` 条请求的 `(arrival_ts, prompt_id, budget)` 序列，由 `(seed, rate, n, budgets, prompt_sha256, burstiness, shuffle)` 完全确定；`trace_sha256` 记录在 manifest/summary 里，**配对比较的第一道校验** |
| **席位（W3 的并发单位）** | **一块 `KvSlab`，`batch = max_batch`，每行一个席位**（不是 B 块单行 slab，也不是 `list[KVcache]`）。`pos`/`valid`/`mask` 本来就逐行独立（`model/slab.py`），因此行与行处在不同位置、走同一套 `hand_baddbmm`（D4）。**准入 = prefill 直接写穿到该行（`prefill_into_seat(row, prompt_ids)`：SDPA prefill 逐层把 K/V 写进 `slab.keys/values[layer][row, :, :L]`，不建 `KVcache`、不 `fill_from` 拷贝），成功后置 `pos[row]=valid[row]=L` 并解冻游标；退出 = 冻结游标（不再 advance）+ mask 屏蔽该行，席位不回收**（P-26；席位回收是 W4 块池的事）。**同一轮 `model(...)` 永远覆盖全部 `max_batch` 行**，空闲行也在，只是被 mask 挡住。**注意**：`open_pool()` 之后 `KvSlab.reset()` 让**所有**行 `active=True`，所以调度器在第一次 decode 前会先把从未用过的席位 `freeze_seat` 一次（park），准入时才 `activate` |
| **空转行** | 空闲席位每步仍在 forward 里，token 是占位、输出被丢弃（冻结行 `valid=0`，全 mask → softmax 无可用 key 的行 bias 被清零，输出是确定值不是 NaN）。这是定长 shape 的代价，也是 D4 选择的结果；**不做**「按 active 行重排 KV」，**不做**逐行 forward |
| **静态批** | 同一份 trace、同一个 `max_batch`，准入策略 `static`：**一波跑完再收下一波**（波内所有请求解码到各自 budget）；波次开始前需凑满 `max_batch`，**trace 已发完则以现有请求开波**（否则末波永远等不满） |
| **连续批** | 同 trace、同 `max_batch`，准入策略 `continuous`：任一席位释放后**下一步立即**补入（下一轮 forward 就该行已写穿 prefill 并解冻） |
| **两臂共用代码** | 同一个 `Scheduler` 类、同一个 `decode_batch`；差异只有 `policy` 一个字段（构造上保证同 kernel 同数值路径，P-23） |
| **吞吐（双口径，两个都报）** | ① `tok_s = Σ out_tokens / makespan`，`makespan = t_last_finish − t_first_arrival`（**含收尾 drain**）；② `tok_s_active = Σ out_tokens / (t_last_finish − t_first_token_global)`（**活跃窗口**，去掉首批 prefill 之前的空转）。**C1 判据用 ①**；② 只作口径说明，避免「drain 段怎么算」的争议。同时报 `req_s`、`offered_rate`、`achieved_rate`、`queue_depth_max`（已到达、仍在排队的最大数；未到达的不算，见 step row 的 `pending`）、`queue_depth_at_last_arrival` |
| **TTFT 分解** | `t_admitted` = 准入决定（分到席位）；`t_prefill_start` = **该请求自己的** prefill 真正开始；`prefill_ms = t_first_token − t_prefill_start`；`queue_ms = t_prefill_start − t_arrival`。**`ttft = queue_ms + prefill_ms` 按构造成立**（C2 要的恒等就是这个），另记 `admit_wait_ms = t_prefill_start − t_admitted` 作诊断，**不进分解**。串行 prefill 期间别的请求在队列里等的时间算**排队**——这正是连续批要压缩的尾部 |
| **TPOT** | `(t_finish − t_first_token)/(out_tokens − 1)`，不含 prefill，不含排队 |
| **budgets** | 每条请求的生成上限（`≤ max_new_tokens`），由 trace 按 seed 抽样；`budgets=[128]` 即协议默认（与 W2 同口径）；**一个 run 内只用一种分布**，记录进 manifest。缺省或空列表在 `validate_config` 里**先归一化成 `[max_new_tokens]` 再校验非空** |
| **计时** | 墙钟用 `perf_counter`（含排队，本来就是墙钟）；单步 GPU 时间用 `torch.cuda.Event`；step row = `active / queued / admitted / retired / step_ms / prefill_ms / decode_event_ms / host_gap_ms`（`host_gap_ms = step_ms − decode_event_ms − prefill_ms`，即 **decode event 窗口之外的 host 开销**，不是 `cat`。CUDA event 覆盖整个 decode 时间线，host 饥饿造成的 GPU 空转落在 `decode_event_ms` **里面**，所以它**不是**「slab 读写之外的 host 开销」） |
| **饱和判据（每臂各自判，不用共享常量）** | `queue_depth_at_last_arrival` = 最后一条请求到达时刻仍在排队（已到达、未开始 prefill）的请求数（即 `max(0, backlog − in_flight)`，再扣掉当时仍空着的席位；见 `bench/engines/continuous.py`）；`saturated ⇔ queue_depth_at_last_arrival > 0`：trace 已经发完、仍有请求在等容量 = 过载。**不能取 run 结束时最后一条 step row 的 `queued`**——run 结束必然排空，那样 `saturated` 按构造恒为 False，C1 每对都 `invalid`。**不看 `capacity_estimate_req_s` 是否达标**——两臂 capacity 本来就不同，用同一个 capacity 会把弱臂按构造判成 `not_saturated`，C1 直接变 `invalid`。`capacity_estimate_req_s` 只在冒烟里量一次、只用于挑 λ 的初值，**不进判据**；若连续臂在 2× 估计值下队列仍会排空 → 该对记 `not_saturated`，λ 加倍**新开 run**（不挑点） |
| **λ 与 B 的关系** | λ 只在**同一 B** 的一对（static/continuous）内必须相同（`trace_sha256` 可比的前提）。B=8 对与 B=32 对**各自**用该 B 的冒烟 capacity 定 λ（`λ = 2 × capacity(B)` 取整），并写进 manifest；跨 B 不比较绝对吞吐，只比较**同一 B 内的比值** |

### 2.2 模块改动（只动这些文件）

```text
model/slab.py          席位（seat）语义补齐：activate(row, length) / freeze(row) / advance(all_rows) /
                       build_mask(next_token=True) 认 active 标记。**不做逐行 forward、不做 list[KVcache]**
                       `advance()` 默认只推进 active 行；`serve/graph.py`（parked）与 W2 单请求路径需要
                       「全行 +1」时显式 `advance(all_rows=True)`，并在 `tests/integration/test_graph_decode.py`
                       里确认冻结行不会被误推进（该测试 gpu 标记，不进快速门禁）
model/minillm.py       forward 只接受单个 KvSlab（batch = max_batch 行席位），T 必须为 1（D4：decode-only 批）
                       非 slab 路径（KVcache）保持与今天完全相同的 op 序列（W1 的 batch 1/2/4 门禁靠它）
model/layers.py        Attention 的 slab 分支已经是「一次 hand_baddbmm 覆盖整批」，W3 不改；
                       RoPE 已按 slab.pos 逐行 gather（_rope_rows），W3 不改
serve/scheduler.py     新增：Request 状态机 + 席位分配 + 准入/退出 + step 主循环（static/continuous）
serve/engine.py        新增 prefill_one()（= 今天的 prefill，逐序列）；decode_batch(tokens, slab)（= 今天的
                       decode_slab，一行 token 对一行席位）；generate() 保留为单请求路径（E0 用）
                       **交接口径（W3 改）**：W2 是「每个请求新建并释放一块 slab」（逐层交错，W2-0007 实测
                       ctx1517 峰值 1534.6→1348.0 MB、ratio 0.859→0.978）。W3 改成**一个池 = 一块
                       B 行 slab，随 scheduler 生命周期存在**：请求准入时 prefill **写穿**到该行
                       （`prefill_into_seat`；容器 `SlabWriteSink` 只写不读，Attention 不需要知道池），
                       池路径上**不建 `KVcache`、也不 `fill_from`**——否则同一份 KV 会在已分配的池
                       （ctx1517 为 182 MiB）之外再存在一份（E0b 实测 166 MiB 的 prefill 临时 KV，
                       ratio 0.859 未达标）。退出时冻结席位（不释放）。W2 单请求路径的 `fill_from`
                       与逐层交错保持原样，只是池不再经过它。
                       **memory 账必须同步**：`kv_mb` 从「本请求的预留」改成「池的总预留，
                       = 2·n_layer·max_batch·n_kv_head·max_len·head_dim·itemsize」，否则
                       `memory_match_ratio` 的分母里会出现账外的其他席位（W2-0004 的同一个病）。
                       池按 B 与桶（64 列）确定 max_len，超出预算时**拒绝该 run 而不是缩批**
bench/loadgen.py      新增 Trace/poisson_trace/closed_trace + trace_sha256；poisson_arrivals 由
                       poisson_trace 取代（旧的 Iterator[(float, Prompt)] 带不了 budget 和稳定 hash，
                       保留一个会误导的「已留接口」不如删掉，并在 ROADMAP/本文件注明）
                      gamma(burstiness) 采样 + 总时长归一化到 n/λ（vLLM serve.py 的做法，§13）是
                      签名的一部分：`poisson_trace(..., burstiness=1.0)`，burstiness=1 即 Poisson
bench/scenario.py     新增：trace 级聚合（makespan/吞吐双口径/p99/分解），纯函数、可单测
bench/engines/continuous.py  新增适配器：run_trace(trace) -> TraceResult；generate() 明确报错
bench/engines/__init__.py    impl="continuous" 接上新适配器（去掉 _NOT_YET 里的条目）
bench/run.py          放行 arrival="poisson"；scenario 执行路径；summary 扩块；key_metric
bench/metrics.py      新增 percentiles()（p50/p95/p99），不动 summarize()
bench/thresholds.py   新增三个评测器；_protocol_mismatches 拆通用/机制两层（D4 的后果，不是改 D4）
configs/thresholds.toml      跑完校准 W3 段（status=calibrated）
configs/experiments/W3/*.toml  13 个正式配置（E0 ×2 + E1 ×5 + E2 ×6）+ 可选 3 个 + 1 个 dummy 冒烟（§3/§4）
```

### 2.3 关键接口（签名先定，再实现）

```python
# model/slab.py —— 席位生命周期（W3 唯一需要新增的容器语义）
class KvSlab:
    active: Tensor            # (batch,) bool，与 pos/valid 同设备、原地更新（graph 可捕获）
                              # 构造与 reset() 后**全 True** → W2 单请求路径的行为逐 bit 不变
    def activate(self, row: int, length: int) -> None
        # = 该行字节已在（prefill 写穿，或 W2 路径的 fill_from）之后置 pos[row]=valid[row]=length、active[row]=True
    def freeze(self, row: int) -> None
        # pos[row]=valid[row]=0、active[row]=False；不搬 KV、不回收席位（P-26）
        # 空闲行仍会被 write 写到第 0 列（占位 token），下次准入的写穿 prefill 会从第 0 列覆盖它
    def advance(self) -> None
        # 默认只推进 active 行（pos += active.to(long)，原地，不读回 host）；
        # MiniLLM.forward 末尾的那一句 slab.advance() 不用改签名。
        # 需要「全行 +1」的旧路径（W2 单请求、serve/graph.py 的捕获）显式调 advance(all_rows=True)
    def build_mask(self, *, next_token: bool = False) -> torch.Tensor
        # limit = (valid + next_token) * active；冻结行整行 False，
        # 且 hand_baddbmm 会把「无可用 key」的行 bias 清零（已有行为，不需要新代码）
        # 冻结行于是输出「V 的均值」，是确定值不是 NaN，只是被调度器丢弃

# model/minillm.py —— 只保留一个容器
def forward(self, tokens, targets=None, attention_mask=None,
            past_keys_values: KVcache | KvSlab | None = None,
            logits_to_keep: int | None = None):
    # slab 路径：B 必须 == slab.batch（整块，含空闲行），T == 1
    # 空闲行照常前向、结果丢弃；不按 active 行重排、不逐行 forward（P-23/D4）
    # 非 slab 路径：KVcache，行为与今天逐 bit 相同（W1 的 batch 1/2/4 门禁依赖它）

# model/layers.py —— 已满足，W3 不改
class Attention:
    def forward(self, x, attention_mask, idx, past_keys_values, cos, sin):
        # slab 分支：slab.write(idx, ...) -> hand_baddbmm.decode(q, k_all, v_all, bias=slab.bias)
        # 一次调用覆盖整批；cos/sin 来自 MiniLLM._rope_rows(slab.pos, slab.max_len)，逐行
        # 直接调用本层且传 slab 时拒绝（必须由模型层给逐行 cos/sin）

# serve/engine.py
def prefill_one(self, prompt_ids: list[int]) -> tuple[Tensor, KVcache]   # = 今天的 prefill（逐序列 SDPA；W2 单请求路径用）
def prefill_into_seat(self, row: int, prompt_ids: list[int]) -> Tensor   # W3 准入：SDPA prefill 写穿进池的第 row 行，(1, vocab)
class SlabWriteSink:                     # 只写不读的 prefill 容器（放在 serve/，Attention 不感知池）
    def past_len(self, layer=None) -> 0          # 于是 is_causal=True、不传 mask（D4 的 prefill 调用不变）
    def get_kv(self, layer) -> (None, None)      # 于是层内不 cat
    def set_kv(self, layer, kv) -> None          # kv=(k, v)：slab.keys[layer][row, :, :L] = k（v 同理）；同一层写两次即报错（拒绝 chunked prefill）
def decode_batch(self, tokens: Sequence[int], slab: KvSlab) -> Tensor     # = 今天的 decode_slab，(B, vocab)
def generate(self, prompt_ids, max_new_tokens) -> EngineOutput            # 单请求臂，保留给 E0/W2 对照

# serve/scheduler.py
@dataclass
class Request:
    id: int; prompt_ids: list[int]; budget: int; arrival_ts: float
    state: Literal["queued","prefilling","decoding","finished","aborted"]
    seat: int | None; out: list[int]
    t_arrival: float; t_admitted: float | None
    t_prefill_start: float | None; t_first: float | None; t_finish: float | None

class Scheduler:
    def __init__(self, engine, *, max_batch: int, policy: str = "continuous",
                 now=time.perf_counter, sleep=time.sleep, eos_token_id=None)
    def submit(self, req: Request) -> None
    def run(self) -> list[Request]          # 全部 finished/aborted 后返回
    def step(self) -> dict                  # 一步：准入→prefill→整批 decode→采样→退出

# bench/loadgen.py
@dataclass(frozen=True)
class Arrival:  idx: int; at_s: float; prompt_id: str; budget: int
@dataclass(frozen=True)
class Trace:
    arrivals: list[Arrival]; rate: float | None; budgets: tuple[int, ...]
    prompt_sha256: str | None; seed: int; burstiness: float
    @property sha256 -> str      # 覆盖 arrivals(到 1e-6)+rate+budgets+prompt_sha256+seed+burstiness
def poisson_trace(prompt_set | None, *, rate, n, budgets, seed,
                  shuffle=False, burstiness=1.0) -> Trace
def closed_trace(...) -> Trace   # 全部 at_s = 0
```

### 2.4 一步长什么样（口径落地）

```text
step k:
  1. 准入（按 policy；只动 queued 里的请求）
       continuous: while active < max_batch and queue[0].arrival_ts <= now: admit
       static    : if active == 0: 凑一波（队列不足则等到够或 trace 结束）；
                   若队列不足 max_batch，要等到「够一个满波」或「trace 已 submit 完」，
                   两种情形下都要**等到这些成员的到达时间**（真实等待，不 busy-wait）
     每条新准入：占一个空席位 → t_admitted → t_prefill_start（就在 prefill 调用前）→
                 prefill_into_seat(row=r)（逐条串行、无 pad；SDPA prefill 逐层写穿到该行，
                 不建 KVcache、不拷贝；失败则引擎先 freeze 该行再抛）→ 采首 token → t_first
  2. 批 decode（整块，一次 forward）：
       tokens 长度必须 == max_batch；空闲席位填一个占位 token（被 mask 挡住，结果丢弃）
       logits = decode_batch(tokens, slab) → (B, V) → 一次 D2H 取 argmax → 逐行 append
  3. 退出：budget / EOS / max_position → t_finish，slab.freeze(row)，席位空出
  每步记一行 step row：active / queued / admitted / retired / step_ms / prefill_ms /
                      decode_event_ms / host_gap_ms
```

不变式（测试断言）：
- `active ≤ max_batch` 恒成立；
- 每条请求恰好 `finish` 一次，`t_arrival ≤ t_admitted ≤ t_prefill_start ≤ t_first ≤ t_finish`；
- `t_first − t_prefill_start == prefill_ms`、`t_prefill_start − t_arrival == queue_ms`（C2 的恒等按构造成立）；
- 冻结席位的 `valid` 不前进、mask 整行为 False（测试直接读 slab 状态）；
- `out_tokens ≤ budget ≤ max_new_tokens`；
- 同一请求在 static 臂与 continuous 臂的 token 序列**逐 id 相同**（同 prompt + 同 budget）；
- 不同到达顺序下同一请求的 token 序列逐 id 相同。

### 2.5 为什么不需要 chunked prefill / pad / varlen

- prefill 逐条做：本来长度就不同，**没有 pad 的需求**，也就不用掩码（逐序列 SDPA，
  `is_causal=True`、`attn_mask=None`）；
- 批只发生在 decode（T=1）：所有请求形状天然一致，长度方向没有可 pad 的东西。decode 的容器是
  `KvSlab`，每行一个席位、按桶（64 列）预留；bool allow-mask 直接融进 `hand_baddbmm` 的
  score GEMM。所以 decode 确实是「slab+mask 的一次调用」——但吃到 mask 的是 `hand_baddbmm`，
  不是 SDPA；它本来就不依赖 flash/efficient，没有「拿到 mask 就退 math」的惩罚；
- 因此「禁长度 pad / 一遍注意力」这类约束在 W3 **构造上满足**，不靠纪律：prefill 永远逐序列、
  长度方向永不 pad（P-22 只约束 SDPA），decode 永远一次 `baddbmm` 覆盖整个 batch。
  这个**注意力口径**已冻结在 `docs/decisions.md` **D4**；`W2-0007`（`fixed-attn-path-slabbudget`）是它的基线。
  W3 自己的负载/调度口径（席位、饱和、吞吐双口径）在 **D5**，见 §2.1。

---

## 3. 三个实验（把「批」和「调度」分开）

### E0 · 重构无害性（1 run，microbench，threshold）

`W3-0001`：`impl="minillm"`，与 `W2-0007` 同 prompt / 同 dtype / 同 batch=1 / 同 max_new_tokens，
`[baseline] run_id="W2-0007"`，`[accept]` 复用 W2 四条判据（`ttft_cv_max/tpot_cv_max/memory_match_ratio/decode_speed_vs_hf_min`）。
**通过即证明 model 层改动没有动数字**；不通过就停下修，不做任何性能 run。
（`decode_speed_vs_hf_min` 在这里当**漂移比值**用：两臂同 kernel 后它比较的是
「W2-0007 的 TPOT / 本次的 TPOT」，≈1.0 即无漂移；名字沿用旧键，避免新增判据键。
注意它**只在同实现配对时**校验 `attn_prefill/attn_decode`——对 HF 臂的正式判据只看通用键，见 D4 后果。）

**E0 只覆盖 model 层（`model/slab.py` 的席位 API + `model/` 的路径），不覆盖 slab 池化。**
池化（§2.2 `serve/engine.py` 那段）会让 TTFT 和 memory 的账都移动，混在同一次 run 里就无法归因。
因此门禁是两条：
- **E0a（必跑，`W3-0001`）**：在**未池化**的 engine 上跑（此时 `generate()` 仍是 W2 的路径，只换了 slab 的
  席位 API），四条判据必须与 `W2-0007` 一致；
- **E0b（必跑，顺序在 E0a 之后，`W3-0002`）**：池化落地后新开的一格（`supersedes` 不需要；它是不同机制），
  只判 `memory_match_ratio`（分子改成池的总预留）并记录 TTFT；它的 TPOT **不与 W2-0007 比**——
  `max_batch=1` 时池与旧路径等价，差异应 < 2%，超过就停下来归因。

### E1 · 批处理摊薄 host（Q1，5 runs，microbench，record_only）

工作负载：`kind="synthetic"`、`arrival="closed"`（全部 t=0）、**`requests=B`（显式写，closed synthetic
不走默认推导**，见 §4 规则 1b）、`budgets=[128]`；`[engine].impl="continuous"`，
`[engine].max_batch=B`（n=B 时两 policy 等价，这一组只测批，不测调度）。

| run | ctx | B | 用途 |
|---|---|---|---|
| W3-0003 | 512 | 1 | 单请求参照（E1 自身基线） |
| W3-0004 | 512 | 8 | P1 |
| W3-0005 | 512 | 32 | P2 |
| W3-0006 | 2048 | 1 | 长上下文参照（**与 W3-0003 比的是 slab 读取量随 ctx 的增长**） |
| W3-0007 | 2048 | 8 | 长上下文 batch 收益 |

口径：E1 同时报三个数——
- `tok_s`：含 prefill 的墙钟口径（`Σout / makespan`）；
- `tok_s_active`：活跃窗口口径（§2.1）；
- **仅 decode 的每 token 成本**：由 step rows 的 `Σout_tokens / Σdecode_event_ms` 换算；**这是「批摊薄
  host」的干净证据**。口径偏差：每条请求的第 1 个 token 是 prefill 后采样的，不在任何 decode event
  里，所以 budget=128 时偏低约 0.8%、budget=16 时偏低约 6.7%；E1 只用 `budgets=[128]`，按 0.8% 读。

B=8/ctx2048 时 prefill 约占 7%，不扣掉会把曲线压低；但**要同时报 `step_ms` 与 `decode_event_ms`**，
两者的差是 `host_gap_ms`（= `decode event 窗口之外`的开销，含 slab 读写之外的 host 部分与空闲等待），
不是「纯 host」的干净度量——CUDA event 覆盖整段 decode 时间线，host 饥饿造成的 GPU 空转落在
`decode_event_ms` 里面。**W3 唯一想压的量是 `step_ms` 本身随 B 的增长斜率**。
**这里没有 `cat`**：D4 之后 decode 写在定长 slab 上，KV 成本表现为「每步读 `B×ctx` 的 K/V」+ 常数量级的
`scatter_` 写入，所以 E1 的长上下文参照测的是**slab 读取**，不是 `cat` 拷贝。

产出：`fig_w3_amortization.png`（x=B；**分面而不是双轴**——左图 ms/token、右图 tok/s、下方面板
`decode_event_ms` 与 `host_gap_ms` 随 B 的变化；每张图按 ctx 分线），由
`scripts/plot_w3_amortization.py` 从 **E1 的 5 份 summary** 的 `steps` 块生成（脚本不读 metrics.jsonl 原表）。
**B=32 × ctx2048 不跑**：KV 预留 ≈ 7.6 GiB（且池按 `max_batch` 一次分配），每步读的量也同量级，
会把「host 摊薄」和「KV 读取带宽」混在一起；那一格留给 W4（分页 + 按需分配）后再测，记在 notes 里。

### E2 · 调度在什么条件下回本（Q2，6 runs，scenario，配对）

> **输出长度口径**：Qwen3-0.6B 是 base 权重、无 chat template、不吐 EOS，`out_tokens` 恒等于 `budget`
> （`W2-0007` 实测 160/160 条 measured row 都是 `out_tokens=128`）。所以 E2 的比值只证明「合成异质长度
> 负载下调度开销足够小」，**不是真实负载吞吐声明**；P4 的 1.8–2.2× 由 `budgets` 分布直接决定，
> 属构造性推论（与 D5 一致）。

工作负载：`kind="prompts"`（32 条固定集）、`arrival="poisson"`、`requests=128`、`max_new_tokens=128`、
**每种长度分布一对，每对独立定 λ**：`rate = 2 × capacity(B)`（capacity 由该 B 的冒烟测得，取整）。
λ 必须写进 manifest 的 `trace.rate`，并作为 `trace_sha256` 的输入——所以**同一对内的两臂 trace 完全相同**，
跨 B 的两对不要求相同（§2.1「λ 与 B 的关系」）。

| run | policy | B | budgets | accept | 用途 |
|---|---|---|---|---|---|
| W3-0008 | static | 8 | `[128]` | record_only | 均匀长度基线（P3） |
| W3-0009 | continuous | 8 | `[128]` | record_only | **P3：期望 ≈1.00×，负结果** |
| W3-0010 | static | 8 | `[16,32,64,128]` | record_only | 混合长度基线 |
| W3-0011 | continuous | 8 | `[16,32,64,128]` | **threshold** | **C1 判据就在这一对**（P4），`[baseline] run_id="W3-0010"` |
| W3-0012 | static | 32 | `[16,32,64,128]` | record_only | B 放大 |
| W3-0013 | continuous | 32 | `[16,32,64,128]` | record_only | B 放大（P2/P4；不判 C1，只报比值） |

**P3 那对不跑 ratio 判据**：`W3-0009` 的 `[accept].mode="record_only"`，结论写进 notes 的负结果专章。
**C1 只在 `W3-0011` 判**（`throughput_speedup_min` + `ttft_p99_decomposed`）。
若 `W3-0008/0009` 也想给出数值，由 notes 里的手算比值承载，**不进 `acceptance`**——否则一份 1.00× 的
`fail` 会污染判据表。

可选（有余力再做，不进退稿）：`W3-0014/0015` 在 `rate = 0.8 × capacity(B=8)` 下的一对，
用来说明「不饱和时连续批没有吞吐收益、只有 TTFT 尾部差异」（record_only）。

**样本量与 p99 口径**：`requests=128 × repeats=5 = 640` 条 request row，**报 p95 与 CV，p99 只标注**
（`draft-gaikwad` 要求 p99 需 ≥1000 样本）。C2 是**结构判据**（`ttft ≈ queue + prefill` 的恒等），
不受样本量影响。若要一个「合法 p99」数字，另开**单臂** `W3-0016`（continuous，`requests=1024`，
record_only，不进任何比值判据），结论回填 notes 而不是判据表。这一格在 §3 里即可选，不阻断 W3 收口。

**每对 run 的协议一致性**由 `thresholds._protocol_mismatches` 校验（D4 的后果两层，未改 D4）：
**通用键** = `prompt_sha256 / dtype / device_class / max_new_tokens`，**机制键** = `attn_prefill / attn_decode`
（同为自实现的两臂必须相等），**scenario 追加** `trace_sha256 / max_batch / policy`（policy 允许不同，
只记录不判等）；不一致 → 判据 `invalid`，不给比值（沿用 W2 的规矩）。

---

## 4. 配置契约（新增字段，逐条校验）

```toml
[workload]
kind = "prompts"
prompts = "bench/prompts.json"
max_new_tokens = 128
arrival = "poisson"        # closed | poisson；poisson 只允许 type="scenario"
rate = 20.0                # req/s；poisson 必填且 > 0
# requests = 128           # trace 的请求数。**不是新必填项**，见下面规则 1
budgets = [16, 32, 64, 128]  # 每条请求的生成上限，seed 抽样；缺省/[] 归一化为 [max_new_tokens]
# shuffle = true           # 可选：prompt 顺序按 seed 打乱（用于到达顺序不变性）
# burstiness = 1.0         # 可选：gamma 形状参数，1.0 = Poisson（§13）
[scheduler]
policy = "continuous"      # static | continuous
[engine]
impl = "continuous"        # continuous | minillm（minillm = 单请求臂，行为不变）
max_batch = 8              # 并发度；scenario 模式下 batch_sizes 必须为 [1]
```

校验规则（`validate_config` 新增；**1 与 2 的顺序固定：先归一化再校验**）：

1. **`requests` 的三种情形**（`closed` 的旧配置必须继续通过，W0/W2 的重跑与
   `tests/unit/test_run_contract.py` 依赖这条）：
   - a. `arrival="poisson"` 或 `impl="continuous"`（scenario 路径，含 E1 的 closed synthetic）：
     **必填且 `>= 1`**，缺了报错。理由：trace 长度决定重复次数与 λ 的总量，不能从 prompt 集推；
     E1 的 `requests` 必须等于 `max_batch`（B 行席位每行一条请求），配置校验同时检查这一点；
   - b. `arrival="closed"` + `impl != "continuous"`（W0/W2 的全部历史配置）：**完全可选**，
     缺省时行为与今天逐字相同（`_cases_for_model` 的 prompt × batch 遍历），runner 不会补一个隐式 trace。
2. `arrival="poisson"` ⇒ `type="scenario"`；
3. `budgets` 先归一化（缺省或 `[]` → `[max_new_tokens]`），再校验每项 `1 <= b <= max_new_tokens` 且非空；
4. `impl="continuous"` ⇒ `[scheduler].policy ∈ {static, continuous}` 且 `max_batch >= 1`；
5. scenario 模式（poisson 或 impl=continuous）⇒ `batch_sizes == [1]`，否则报 P-20 的错；
6. `timer="cuda_event"` 仍要求 `device=cuda`（不动）；
7. `[accept].mode="threshold"` 时，`throughput_speedup_min` 只能出现在 scenario run；
8. `[scheduler]` 出现在非 continuous 配置里 ⇒ 报错（不要留一个不生效的旋钮）。

**配置不动的部分**：`[workload].batch_sizes` 的语义仍是「行标签」（P-20），并发度只看 `[engine].max_batch`；
所以 `batch_sizes=[1]` 是 scenario 的硬要求，**不是**把 batch 从配置里删掉。

---

## 5. 测试清单（红 → 绿，先写后实现）

### 5.1 `tests/unit/test_batch_forward.py`（新）

| 用例 | 断言 | 防的坑 |
|---|---|---|
| `test_seated_decode_matches_per_sequence` | 一片 B 行 slab，3 条**不同长度**的 prompt（5/9/13）各占一行，decode 6 步：批 vs 逐条 logits `atol=1e-5`、id 全等 | 整条批路径的等价性 |
| `test_frozen_seat_is_masked_out` | 冻结一行后：`advance()` 不动该行 `pos/valid`，`build_mask()` 该行整行 False；其它行结果与「该座位从未存在」逐 bit 相同 | 席位生命周期（§2.1）；P-26 |
| `test_slab_rejects_multi_token_forward` | 传 slab 且 `T > 1` 抛 `ValueError`（信息里带 D4）。**不要写成「任意 B>1,T>1 都被拒」**——`KVcache` 路径的 left-pad 批 prefill 是 W1 的验收项（`test_align_qwen3.py::test_batch_sizes_agree`），必须继续通过 | 有人偷偷 pad 成 `(B,T_max)` 一次 slab forward（会让 SDPA 退 math，6.3×；见 P-22）；同时不许误伤 W1 |
| `test_rope_uses_per_row_positions` | 两条 past_len 不同（5 和 9）时批 decode == 逐条；且故意把两行都当成同一位置会不等 | **P-21：整批共用一张 RoPE 表 → 静默错 token** |
| `test_single_seat_is_bit_identical` | `torch.equal` 与今天 `KVcache` 单请求路径逐 bit 相同 | 重构不得改数字（E0a 的单测版） |
| `test_attention_dispatched_once_per_batch` | 用 monkeypatch 计数：`hand_baddbmm.decode` 每层每步只调用 1 次（不是 B 次），`MLP.forward` == n_layer | D3 驳回的病（`28×B` 次派发）不能从 W3 后门回来 |

### 5.2 `tests/unit/test_scheduler.py`（新，全部用 fake engine，不需要权重）

| 用例 | 断言 |
|---|---|
| `test_capacity_never_exceeded` | 每步 `active ≤ max_batch` |
| `test_each_request_finishes_once` | 每 id 恰好一次 finish；无重复采样、无丢失 |
| `test_each_request_holds_one_seat` | 任意时刻 `len({r.seat}) == active`；退出后席位被下一请求复用 |
| `test_static_runs_waves` | `active` 在波次边界回到 0；最大并发 == max_batch |
| `test_continuous_refills_next_step` | 短请求在第 k 步退出 → 队首在第 k+1 步被准入（fake clock 精确到步） |
| `test_lifecycle_timestamps_monotone` | `arrival ≤ admitted ≤ prefill_start ≤ first ≤ finish`，`ttft == queue + prefill`（<1e-9 s，fake clock） |
| `test_budget_eos_and_max_position_terminate` | 三种终止条件各一条；EOS 不产生后续 token（P-06/P-07 的调度器版） |
| `test_arrival_order_invariance` | 同一批请求，两种到达顺序 → 逐 id 相同（C3 的单测） |
| `test_static_and_continuous_ids_equal` | 同 budget 下两 policy 的 token 序列逐 id 相同 |

### 5.3 不新增文件：C3 由既有测试承担

C3（不同到达顺序输出一致）**不另写 `test_scheduler_equivalence.py`**。它由两条已存在的机制层
测试传递推出，论证写进 W3 的 notes：

- `test_batch_forward.py::test_seated_decode_matches_per_sequence`：批 decode 每行 == 逐序列 decode
  （含不同 past_len 的行，走的是真实的 mask/RoPE 路径）；
- `test_batch_forward.py::test_frozen_seat_is_masked_out`：冻结/激活一个席位不改变其它行的输出。

席位之间互不影响 ⇒ 请求的输出只取决于它自己的 prompt、budget 和它被准入时的位置，与同批还有谁
在跑无关 ⇒ 到达顺序不影响逐 id 结果。`test_scheduler.py::test_arrival_order_invariance` 在 fake
engine 上验证调度层（准入顺序、席位复用）不引入顺序依赖；`test_static_and_continuous_ids_equal`
验证两 policy 同 id。**这三条合起来覆盖 C3，不再加第四个文件。**

### 5.4 `tests/unit/test_loadgen_trace.py`（新）

同 seed 同参数 → 同 `trace_sha256`；改任一参数（含 `burstiness`、`rate`、`budgets`、`prompt_sha256`）→ hash 变；
`mean(interarrival) ≈ 1/rate`（1e-2 相对误差，n=5000）；**总时长归一化**：`Σ interarrival ≈ n/rate`
（1e-3 相对误差，vLLM `serve.py` 的做法，§13）；`burstiness=4` 时超几何变差但均值不变；
budgets 抽样频率与权重一致（±3%）；`shuffle` 可复现；prompt 轮转覆盖全部 32 条。

### 5.5 `tests/unit/test_scenario_summary.py`（新，纯聚合）

手工构造 6 条 request row + 2 条 trace row → 断言 `makespan/tok_s/tok_s_active/req_s`、
p99 的 `ttft/queue/prefill`、`queue_share`、`ttft == queue+prefill` 检查，
另用 dummy 臂实跑 E2 形状的 trace，断言「run 跑完（最后一步 `queued=0`）但最后到达时刻有积压」仍判 `saturated=true`，低率 Poisson 或闭合 trace（`n == max_batch`）则 `saturated=false`；
以及**饱和标记只看 `queue_depth_at_last_arrival > 0`**（给一条「achieved_rate 低于任何 capacity 估计但队列
非空」的行，必须判 `saturated=true`；正是旧定义会误判的那种情形）。

### 5.6 扩展既有测试

- `tests/unit/test_thresholds.py`：`throughput_speedup_min`（有基线/无基线/协议不符→invalid）、
  `ttft_p99_decomposed`（bool 判定 + 缺项 → pending）；
- `tests/unit/test_run_contract.py`：poisson 配置被接受、scenario run 的 manifest 含
  `scheduler.policy/trace_max_batch/trace_sha256`、`batch_sizes=[2]` 在 scenario 下被拒；
  **且必须保留并扩充「历史 closed 配置不写 `requests` 仍通过」的用例**（W0/W2 的重跑契约，§4 规则 1b），
  外加「poisson 缺 `requests` → 报错」「E1 的 `requests != max_batch` → 报错」两条；
- 五种非法配置：`poisson` + `type="microbench"`；`budgets` 越界；`[scheduler]` 出现在非 continuous 配置；
  scenario 下 `batch_sizes` 非 `[1]`；`max_batch<1`。

---

## 6. runner / summary / thresholds 的契约扩展（字段级）

### 6.1 `metrics.jsonl` 新增 `row_kind`

```jsonc
// row_kind="request"（每请求每 repeat 一行）
{"row_kind":"request","run_id":"W3-0011","model":"Qwen/Qwen3-0.6B","impl":"continuous",
 "policy":"continuous","repeat":0,"arrival_idx":17,"seat":3,"prompt_id":"medium-03","budget":64,
 "context":421,"out_tokens":64,"t_arrival_s":0.42,"queue_ms":1180.3,"prefill_ms":21.4,
 "ttft_ms":1201.7,"tpot_ms":2.6,"e2e_ms":1366.0,"kv_own_mb":49.1,"token_sha256":"…","ok":true}
// queue_ms = t_prefill_start − t_arrival（串行 prefill 期间等别人才算排队，§2.1）；
// 本行的 kv_own_mb 是本请求的份额；`memory_match_ratio` 的分子用 trace row 的 kv_mb（池总预留）

// row_kind="step"（每 step 一行；W7 也要用）
{"row_kind":"step","run_id":"W3-0011","repeat":0,"step":311,"active":8,"queued":41,
 "admitted":1,"retired":1,"pending":96,"step_ms":20.9,"prefill_ms":21.4,"decode_event_ms":3.9,
 "host_gap_ms":16.6,"kv_mb_sum":612.4}
// host_gap_ms = step_ms − decode_event_ms − prefill_ms；slab 读写含在 decode_event_ms 里

// row_kind="trace"（每 repeat 一行 = 聚合口径的权威行）
{"row_kind":"trace","run_id":"W3-0011","repeat":0,"makespan_s":18.32,"out_tokens_total":3841,
 "tok_s":209.7,"tok_s_active":214.2,"req_s":3.49,"offered_rate":20.0,"achieved_rate":3.49,
 "queue_depth_max":56,"queue_depth_at_last_arrival":12,
 "ttft_p99_ms":5210.0,"queue_p99_ms":5180.0,"prefill_p99_ms":24.0,"peak_concurrency":8,
 "saturated":true,"drops":0}
```

### 6.2 `summary.json` 新增块（`bench/scenario.py` 负责算，纯函数）

```jsonc
"throughput": {"n_traces":5,"tok_s":{…median/p95/cv…},"tok_s_active":{…},"decode_only_tok_s":{…},
               "req_s":{…},"makespan_s":{…},"offered_rate":20.0,"achieved_rate":{…},
               "saturated":true,"queue_depth_at_last_arrival":{…},
               "analytic_straggler_ratio":2.02},      // 理想波次比 E[max_B]/E[mean]，按 max_batch 解析算
"latency":    {"ttft_ms":{"p50":…,"p95":…,"p99":…},"queue_ms":{…},"prefill_ms":{…},
               "queue_share_p99":0.98,"decomposed":true},
"steps":      {"by_active":[{"active":8,"n":2410,"step_ms":20.9,"prefill_ms":0.4,
                             "decode_event_ms":3.9,"host_gap_ms":16.6}]},
"consistency":{"token_sha256_match_vs_baseline":128,"mismatch":0}
```

`overall`/`groups`/`stability` 保持只吃 `row_kind="request"` 的行（W2 口径完全不变）；
scenario run 的 `groups` 按 **bucket** 分组（而不是精确 ctx），避免 32 个 context 把 summary 撑爆。
`key_metric`：`throughput.tok_s.median` 优先（scenario run 的索引就该显示 tok/s）。

### 6.3 `thresholds.py` 新增评测器

| 判据键 | 逻辑 |
|---|---|
| `throughput_speedup_min` | `baseline.throughput.tok_s.median / ours.throughput.tok_s.median`（**墙钟口径**，§2.1 的 ①）；先校验 `_protocol_mismatches`（scenario 层新增 `trace_sha256`、`max_batch`）→ 不符记 `invalid`；缺基线 → `pending`；两个 run 都要 `saturated=true`（按 §2.1 的 `queue_depth_at_last_arrival > 0`，**不看 capacity 估计**） |
| `ttft_p99_decomposed` | 结构检查：`latency.ttft_ms.p99`、`queue_ms.p99`、`prefill_ms.p99` 都存在，且逐请求 `|ttft−(queue+prefill)| < 1 ms` 的比例 ≥ 99% |

### 6.4 manifest 增补

```jsonc
"scheduler": {"policy":"continuous","max_batch":8,"seats":"one_kvslab_one_row_each",
              "prefill":"per_sequence","free_rows":"masked_in_forward"},
"trace": {"arrival":"poisson","rate":20.0,"burstiness":1.0,"requests":128,
          "budgets":[16,32,64,128],"seed":0,"sha256":"…",
          "capacity_estimate_req_s":3.5}
```

`capacity_estimate_req_s` 只作记录：它由冒烟量一次、用于挑 λ，**不参与饱和判据**（§2.1）。

---

## 7. 施工顺序与门禁（约 5 天）

| 天 | 动作 | 门禁（必须绿才进下一步） |
|---|---|---|
| D1 上午 | 在 `docs/decisions.md` **新立 D5（proposed）**，正文 = §2.1 口径 + §1 判据；**不动 D4**；先写 §5.1 全部用例（红） | `pytest -m "not slow and not hf"` 只红在新增用例 |
| D1 下午 | `model/slab.py` 席位 API（`activate/freeze/advance(all_rows=)` + `active` 标记）+ §5.1 | §5.1 全绿；`test_align_qwen3.py`（`MINILLM_RUN_HF=1`）仍全绿（**W1 的 batch 1/2/4 是硬门禁**）；`tests/unit/test_slab*.py` 与 `test_graph_decode.py` 不回归；单请求路径逐 bit 不变 |
| D2 | `serve/scheduler.py` + `engine.prefill_into_seat/decode_batch` + slab 池化；§5.2/5.3 | §5.2 全绿；既有 24 个单测文件全绿 |
| D3 上午 | `loadgen` trace + hash（gamma + 归一化），§5.4 | §5.4 全绿 |
| D3 下午 | `bench/scenario.py` + `bench/engines/continuous.py` + runner 接线，§5.5 | §5.5 全绿；`--out /tmp/... --allow-dirty` 的 dummy 冒烟能跑通 |
| D4 上午 | `thresholds.py` 三个评测器 + §5.6 扩展；`W3/9001-scheduler-smoke.toml`（dummy） | 契约测试全绿；`scripts/exp_index.py --check` 绿 |
| D4 下午 | 用户在 4090 上跑 **E0a**（`W3-0001`）→ **E0b**（`W3-0002`） | E0a 四条判据 pass；E0b 的 memory 分子含池预留且单请求 TPOT 偏差 < 2%，才允许正式性能 run |
| D5 | E1 → 用户跑 5 个 run（`W3-0003…0007`）→ 出图；**E1 结论回填 D5** | 每 token 成本曲线与 P1/P2 对照 |
| D5 末 | E2 冒烟定 capacity(B) → 用户跑 6 个 run（`W3-0008…0013`，+2 可选） | C1/C2/C3 判定；`thresholds.toml` 的 W3 段改 `calibrated` + `basis_run` |

每个正式 run 之后：`python scripts/exp_index.py` → 检查 `--check`；`notes.md` 由 runner 生成模板，
**observation 段我起草，`verdict.json` 保持 `unreviewed` + `reviewer: null`**，直到用户确认。

---

## 8. 风险与止损

| # | 风险 | 触发信号 | 处置 |
|---|---|---|---|
| R1 | host 摊薄不了（step 时间随 B 线性涨） | E1：B=8 的 `step_ms` > 3× B=1 | 记负结果，W3 收口（ROADMAP §8：< 1.2× 不硬调）。探针显示不会发生 |
| R2 | 均匀长度下 1.3× 不可达 | E2 均匀对 ≈1.0× | **按设计如此**：判据只在混合对判（`W3-0011`）；均匀对进负结果专章（P3） |
| R3 | 注意力在 B=32 成瓶颈 | B=32 时注意力占比若成瓶颈（W4 分页阶段再测） | 先记数。**B=32 时的注意力占比若成瓶颈属 KV 读取问题**，而非派发（D4 已定 decode 走 `hand_baddbmm`，B=8 实测 0.7 ms/step）——那是 W4 分页的问题，不在 W3 改 |
| R4 | 长上下文的 **slab 读取**（每步读 `B×max_len` 的 K/V）吃掉落差 | ctx 2048 的 K/V 读取占比偏高（W4 分页阶段再测） | 记录为 W4（分页按需分配）的目标；**不在 W3 改 KV 布局**（否则机制混在一起）。**这里没有 `cat`**：D4 之后 decode 不 concat，成本是带宽不是拷贝 |
| R5 | 实时钟抖动导致 tok/s CV > 10% | 5 个 repeat 的 CV 超线 | 加长 trace（`requests` 256/512，**poisson 必填所以直接改字段**）**新开 run**，不挑点 |
| R6 | B=32 × ctx2048 显存/带宽爆炸 | OOM 或 step 异常长 | 该格不跑（已排除在 E1 矩阵外） |
| R7 | 静态臂被质疑「被削弱」 | 评审意见 | 过载 trace（λ=2×capacity(B)）下静态臂的 `queue_depth_at_last_arrival > 0` 恒成立，凑一波不会因等填充而空闲；把这条写进 notes 的「限制」段 |
| R8 | 席位/池化让 W2 数字漂移 | E0a/E0b 判据红 | 停性能 run，E0a 归因 model 层、E0b 归因池化，分开定位（W1 的 golden 是对拍地基） |

---

## 9. 要登记到 `PITFALLS.md` 的新坑（实现时补，附回归测试）

| ID | 坑 | 回归 |
|---|---|---|
| P-20 | `[workload].batch_sizes` 只是行标签，不是并发度；把它当并发会得到「假的 batch=4」 | `test_run_contract.py` |
| P-21 | 批 decode 里各请求位置不同，**RoPE 表不能整批共用一张**（静默错 token，且单请求测试查不出） | `test_batch_forward.py::test_rope_uses_per_row_positions` |
| P-22 | 为了「先跑通」把批内 pad 成 `(B,T_max)`：SDPA 退 math、闸值全作废 | `test_batch_forward.py::test_slab_rejects_multi_token_forward` |
| P-23 | 静态批与连续批若各写一份 decode 循环，收益会混进 kernel 差异 | `test_static_and_continuous_ids_equal` + 构造上同一个 `decode_batch` |
| P-28 | **「把 `requests` 加成全局必填」会静默改掉历史 closed 配置的语义**（W0/W2 的重跑契约与 `test_run_contract.py` 都会红）。只有 poisson / `impl="continuous"` 需要它 | `test_run_contract.py`（历史配置不写 `requests` 仍通过） |
| P-29 | **用单一 `capacity` 常量判饱和会把弱臂按构造判成 `not_saturated`**，C1 直接 `invalid`。饱和只看「trace 发完队列仍非空」 | `test_scenario_summary.py`（capacity 低于 achieved 仍判 saturated） |
| P-30 | **slab 池化后 `kv_mb` 若仍记「本请求份额」，`memory_match_ratio` 的分母会含账外的其他席位**（W2-0004 同病） | `test_engine_decode_path.py`（池预留 == 各席位之和） |

---

## 10. 不在 W3 做

- chunked prefill（掩码路径已支持 `past_len>0, T>1`，但不接入调度）；
- 分页 KV（W4）、CUDA graph 与 varlen kernel（**D4 已定：均不进 W3 数字**；graph 保留为 parked，W4 接线）、prefix cache；
- 任何形式的长度方向 pad；
- 量化、MTP/投机解码、多模态（D2）；
- 改 KV 布局（slab 的定长预留保持原样，只测其读取占比；按需分配是 W4）；
- 按 active 行重排 KV / 逐行 forward（D4 的两条推论，写进 §2.1 就是为了不许做）。

---

## 11. 命令（冒烟我来跑；正式 run 由你在 4090 上跑）

```bash
# 快速门禁（无权重）
python -m pytest -m "not slow and not hf"

# golden 对拍（改完 model 必跑）
MINILLM_RUN_HF=1 python -m pytest tests/integration/test_align_qwen3.py -q

# 冒烟（dummy，不写 results/）
python -m bench.run --config configs/experiments/W3/9001-scheduler-smoke.toml \
  --out /tmp/minillm-w3-smoke --allow-dirty

# 正式 run（干净树；模型 revision 已 pin）
python -m bench.run --config configs/experiments/W3/0001-single-request-recheck.toml   # E0a
python -m bench.run --config configs/experiments/W3/0002-slab-pool-recheck.toml        # E0b
python -m bench.run --config configs/experiments/W3/0003-batch-b1-ctx512.toml          # E1
# … 0004 … 0007（E1 矩阵）、0008 … 0013（E2 矩阵）
python scripts/exp_index.py

# 出图
python scripts/plot_w3_amortization.py
```

---

## 12. 交付物

- `serve/scheduler.py`、`bench/scenario.py`、`bench/engines/continuous.py`、`bench/loadgen.py`（trace）
- `model/slab.py` 的席位 API + `serve/engine.py` 的 slab 池化
- 新测试 4 个文件 + 2 处扩展（`tests/unit/` 下：`test_batch_forward.py`、`test_scheduler.py`、
  `test_loadgen_trace.py`、`test_scenario_summary.py`，加扩展的 `test_thresholds.py` /
  `test_run_contract.py`；不含已删除的 profiler 集成测试，也不新增 `test_scheduler_equivalence.py`
  ——C3 由既有测试传递推出，见 §5.3），快速门禁全绿
- run：`W3-0001`（E0a）+ `W3-0002`（E0b）+ E1 ×5（`W3-0003…0007`）+ E2 ×6（`W3-0008…0013`，+2 可选）
- `docs/fig_w3_amortization.png`（E1）、`results/W3/*/notes.md` 草稿 + `verdict.json`（unreviewed）
- `docs/decisions.md` **D5**（W3 负载与口径预注册 + 实测结论）；**D4 保持原样**
- `configs/thresholds.toml` `[W3.continuous]` → `calibrated` + `basis_run`
- `ROADMAP.md` §12 回填 W3 证据 run、§13 一条修订；`AGENTS.md` / `serve/README.md` 的模块状态更新
- `PITFALLS.md` P-20…P-23、P-28…P-30 登记并挂回归测试

---

## 13. 参考实现（W3 专用）

> 只列 W3 要用的；跨阶段（W4–W7）的代码参考在 `ROADMAP.md` §14。
> 来源：`docs/benchmark-resources.md`。规则同上：抄机制与口径，不抄生产系统的工程。

| 用途 | 参考 | 抄什么 | 不抄什么 |
|---|---|---|---|
| **到达过程** | vLLM `benchmarks/serve.py::get_request()` | 间隔采样 `gamma(shape=burstiness, scale=1/(rate·burstiness))`（`burstiness=1` 即 Poisson）+ **把累积间隔归一化到总时长 = n/λ**（源码注释：直接累加有 1–2% 偏差，归一化才让不同 seed 的吞吐可比） | 其 dataset 加载、tokenize 流程 |
| **调度接口** | nano-vllm `engine/{scheduler,sequence,llm_engine}.py` | waiting/running 两队列的 iteration 级准入；`Sequence` 的字段（**给 W4 的 block table 留位置**） | preempt / chunked prefill / CUDA graph；nano-vllm 的 attention |
| 调度（可选） | mini-sglang `python/minisgl/scheduler/` | 只在 W3 卡住时读：`--cache naive` 的块池与 `radix` 的对照 | sgl-kernel / FlashInfer 依赖（与本项目口径冲突） |
| **指标口径** | NVIDIA AIPerf 文档 | TTFT 含排队；`ITL = (e2e − TTFT)/(out−1)`；系统 `TPS = Σout/(末响应 − 首请求)` | AIPerf 工具本体（客户端，需 OpenAI 兼容端点；我们是进程内） |
| **统计合法性** | IETF `draft-gaikwad-llm-benchmarking-terminology` | open/closed-loop 必须声明；**p99 需 ≥1000 样本**、p95 需 ~400；warmup 必须剔除并写明；工作负载须声明到达模式/长度分布/tokenizer | — |
| 报告范式（留给 W8） | SGLang `bench_serving` 的 goodput | 「扫 λ，goodput 先升后降，峰即 SLO 内可持续吞吐」 | 其 e2el/ITL 的百分位选择 |

据此已经改掉本方案三处（不重复讨论）：

1. 到达过程：gamma + 总时长归一化，替换裸指数间隔（§2.3 `poisson_trace`）；
2. p99 双轨：判据用配对 trace（n=128×5=640，报 p95+CV），另设一个 n≥1000 的 open-loop
   单臂 run 专门产出合法 p99 TTFT（不进比值判据）；
3. 吞吐双口径：墙钟 makespan 与「活跃窗口」（首请求→末响应）并报，避免 drain 段口径争议。

**不引入**：AIPerf 作为工具、BurstGPT 时间戳 replay（可选只取 `(input_tokens, output_tokens)` 经验分布当
`budgets` 来源）、LongBench / RULER / ShareGPT（W4+ 可选，不影响任何 W3 判据）。
