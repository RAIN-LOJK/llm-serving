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
> | `ROADMAP.md` §14 参考实现索引 | W3 行的「实现细节见 `docs/w3-plan.md`」 | 改成指向 `docs/decisions.md` 的对应 D 条目 |
> | `docs/w2-rework.md` §0 / §2 E6 | 前置与「口径段改写」引用 | 两个临时文件一起删，反向引用随之消失 |

> **前置**：配对基线 **`W2-0006`**（固定注意力路径，四条判据 pass；HF 分母用同机的 `W2-0005`）。
> `W2-0003`（旧口径 sdpa/cat）与 `W2-0004`（跨实例配对，判 `invalid`）都不能当基线。
> **D3 已于 2026-10-31 [rejected](decisions.md)，口径由 [D4](decisions.md) 定死**——prefill（T>1）逐序列 SDPA、
> decode（T==1）`hand_baddbmm` + `KvSlab`。本文件里凡引用 D3 的段落都标成「当前口径（待替换）」，现在替换为 D4。
> W3 的调度/到达过程参考见 §13。
> 本文件是「新开一个对话就能直接开工」的自包含方案；阶段目标与判据仍以 `ROADMAP.md` §5 W3 为准。
> **状态：计划（未跑任何 W3 run）**。文中所有数字分成三类，读的时候不要混：
> `实测`（已有 run）、`探针`（规划期一次性 ad-hoc 测量，**不进报告**）、`预测`（预注册，待验证）。

---

## 0. 规划期核对到的三件事（决定了 W3 的重心）

### 0.1 现有基线的真实结构（探针，不引用进报告）

在 4090 上用 `serve/engine.py` 的单请求路径测一个 decode step（ctx≈90）：

| 项 | 值 | 含义 |
|---|---|---|
| 每 step 墙钟 | **19.6 ms** | 与 `.item()` 同步的版本相同（19.6），说明瓶颈不在同步 |
| 每 step GPU 忙时间 | **≈3.3 ms** | GEMM 1.66 + 逐元素 1.4 + SDPA 0.24 |
| 每 step CPU op 数 | **≈705**（其中 197 个 matmul） | 28 层 × 25 个 eager op |
| 单 kernel launch 成本 | 5.2 µs | 空的 `add_` 实测（本机 CPU 不算慢） |
| 权重 roofline | 1137 MiB / ≈700 GB/s ≈ **1.7 ms** | bf16 一 token 的理论下界 |

→ **host 侧约 16 ms/step 是纯粹按「每 token」付的钱**。这 16 ms 与 batch 大小几乎无关（op 数不变），
这就是 W3 唯一真正大的杠杆：**批处理摊薄的是 host，而不是 GPU**。

### 0.2 同一组权重做 B 条序列的 GPU 时间几乎不涨（探针）

用真实 Qwen3-0.6B 权重形状跑「28 层 × (qkv+o+mlp) + lm_head」，M=B、T=1：

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
| C4 | decode 阶段注意力占单步时间 | **< 15%**（B ≤ 8；原 D3 第 3 条保留为目标） | `tests/integration/test_step_split.py`（gpu 标记）断言；数字进 notes | `[W3.continuous].attn_share_max`（新增 0.15，测试直接读它） |
| C5 | 批处理摊薄 host（Q1，非机制判据） | 记录 `tok/s(B)` 曲线；预测 B=8 时 ≥ 5× | E1 五个 run，record_only | 无（图） |

**预注册预测（publish-then-measure，跑完逐条判真假）**

| # | 预测 | 依据 |
|---|---|---|
| P1 | B=1→8 每 token 成本降 ≥ 5×（ctx 512） | 0.1/0.2 探针：host 16 ms 被摊成 2 ms |
| P2 | B=8→32 每 token 再降 2.5~3×，但 step 时间开始被 KV `cat` 抬高 | `cat` 拷贝量 ∝ B×ctx |
| P3 | 均匀 128 token 负载：连续/静态 ≈ 1.00×（**负结果**） | 0.3 实测 + 波次取整 |
| P4 | 混合长度（{16,32,64,128} 均匀抽样）饱和负载：连续/静态 **1.8–2.2×**（解析上限 E[max₈]/E[mean] = 2.13×，扣掉连续臂收尾 drain） | 解析式；max 在 B≥8 时几乎必为 128，故与 B 基本无关 |
| P5 | 饱和 trace 的 p99 TTFT 中排队占比 > 60% | 过载必然排队 |
| P6 | B=8 时 decode 阶段的注意力占比 < 10%（远低于 15% 上限） | `hand_baddbmm` 实测 0.7 ms/step（B=8/L=512）；D4 已把它定为唯一 decode 路径 |

---

## 2. 设计（先定口径，再定接口）

### 2.1 口径冻结（写进 D4，跑之前定死）

| 项 | 定义 |
|---|---|
| **trace** | `n` 条请求的 `(arrival_ts, prompt_id, budget)` 序列，由 `(seed, rate, n, budgets, prompt_sha256)` 完全确定；`trace_sha256` 记录在 manifest/submary 里，**配对比较的第一道校验** |
| **静态批** | 同一份 trace、同一个 `max_batch`，准入策略 `static`：**一波跑完再收下一波**（波内所有请求解码到各自 budget）；波次开始前需凑满 `max_batch`，**trace 已发完则以现有请求开波**（否则末波永远等不满） |
| **连续批** | 同 trace、同 `max_batch`，准入策略 `continuous`：任一 slot 释放后**下一步立即**补入 |
| **两臂共用代码** | 同一个 `Scheduler` 类、同一个 `decode_batch`；差异只有 `policy` 一个字段（构造上保证同 kernel 同数值路径） |
| **吞吐** | `tok/s = Σ out_tokens / makespan`；`makespan = t_last_finish − t_first_arrival`；同时报 `req/s`、`offered_rate`、`achieved_rate`、`queue_depth_max` |
| **TTFT** | `t_first_token − t_arrival`；`queue_ms = t_admitted − t_arrival`；`prefill_ms = t_first_token − t_admitted` |
| **TPOT** | `(t_finish − t_first_token)/(out_tokens − 1)`，不含 prefill，不含排队 |
| **budgets** | 每条请求的生成上限（`≤ max_new_tokens`），由 trace 按 seed 抽样；`budgets=[128]` 即协议默认（与 W2 同口径）；**一个 run 内只用一种分布**，记录进 manifest |
| **计时** | 墙钟用 `perf_counter`（含排队，本来就是墙钟）；单步 GPU 时间用 `torch.cuda.Event`；两侧分别记录，不互相替代 |
| **饱和判据** | `achieved_rate ≥ 0.95 × capacity` 且 `queue_depth_max > 0`；否则该 run 标记 `not_saturated`，吞吐比值不引用 |

### 2.2 模块改动（只动这些文件）

```text
model/minillm.py      forward 接受 KvSlab（B 行席位），T 必须为 1（D4：decode-only 批）
model/layers.py       TransformerBlock：decode 走 slab（attention 一次覆盖整批、MLP 按批）；RoPE 按 slab.pos 逐行
serve/scheduler.py    新增：Request 状态机 + 准入/退出 + step 主循环（static/continuous）
serve/engine.py       新增 prefill_one()/decode_slab()（已有单行版）；generate() 在 D4 定死后即 slab 路径
bench/loadgen.py      新增 Trace/poisson_trace/closed_trace + trace_sha256；poisson_arrivals 由
                      poisson_trace 取代（旧的 Iterator[(float, Prompt)] 带不了 budget 和稳定 hash，
                      保留一个会误导的「已留接口」不如删掉，并在 ROADMAP/本文件注明）
bench/scenario.py     新增：trace 级聚合（makespan/吞吐/p99/分解），纯函数、可单测
bench/engines/continuous.py  新增适配器：run_trace(trace) -> TraceResult；generate() 明确报错
bench/engines/__init__.py    impl="continuous" 接上新适配器（去掉 _NOT_YET 里的条目）
bench/run.py          放行 arrival="poisson"；scenario 执行路径；summary 扩块；key_metric
bench/metrics.py      新增 percentiles()（p50/p95/p99），不动 summarize()
bench/thresholds.py   新增三个评测器；_protocol_mismatches 拆通用/机制两层（D4）
configs/thresholds.toml      新增 attn_share_max；跑完校准 W3 段（status=calibrated）
configs/experiments/W3/*.toml  13 个配置（§4）
```

### 2.3 关键接口（签名先定，再实现）

```python
# model/minillm.py
def forward(self, tokens, targets=None, attention_mask=None,
            past_keys_values: KVcache | Sequence[KVcache] | None = None,
            logits_to_keep: int | None = None):
    # B = tokens.shape[0]; T = tokens.shape[1]
    # if past_keys_values is a sequence: assert len == B, assert T == 1
    #   (batched forward is decode-only; prefill stays per sequence —— 当前口径，待 W2 返工后可能改)

# model/layers.py
class Attention:
    def forward(self, x, attention_mask, idx, past_keys_values, cos, sin):
        # cos/sin 现在是 per-row：cos 形状 (n_pos, head_dim//2)，第 i 行取 [pos_i - pos_min]
        # 单请求路径（KVcache）保持与今天完全相同的 op 序列

class TransformerBlock:
    def forward(self, x, ...):
        # caches 为 list 时：
        #   attn = torch.stack([self.self_attn(x[i:i+1], ..., cache=caches[i], cos=cos[i:i+1], ...) for i])
        #   mlp 仍然一次吃完 (B,T,C)

# serve/engine.py
def prefill_one(self, prompt_ids: list[int]) -> tuple[Tensor, KVcache]     # = 今天的 prefill
def decode_batch(self, tokens: list[int], caches: list[KVcache]) -> Tensor  # (B, vocab)

# serve/scheduler.py
@dataclass
class Request:
    id: int; prompt_ids: list[int]; budget: int; arrival_ts: float
    state: Literal["queued","prefilling","decoding","finished","aborted"]
    cache: KVcache | None; out: list[int]
    t_admitted: float | None; t_first: float | None; t_finish: float | None
    step_rows: list[dict] = field(default_factory=list)

class Scheduler:
    def __init__(self, engine, *, max_batch: int, policy: str = "continuous",
                 now=time.perf_counter, sleep=time.sleep, eos_token_id=None)
    def submit(self, req: Request) -> None
    def run(self) -> list[Request]          # 全部 finished/aborted 后返回
    def step(self) -> dict                  # 一步：准入→prefill→批 decode→采样→退出

# bench/loadgen.py
@dataclass(frozen=True)
class Arrival:  idx: int; at_s: float; prompt_id: str; budget: int
@dataclass(frozen=True)
class Trace:
    arrivals: list[Arrival]; rate: float | None; budgets: tuple[int, ...]
    prompt_sha256: str | None; seed: int
    @property sha256 -> str      # 覆盖 arrivals(到 1e-6)+rate+budgets+prompt_sha256+seed
def poisson_trace(prompt_set | None, *, rate, n, budgets, seed, shuffle=False) -> Trace
def closed_trace(...) -> Trace   # 全部 at_s = 0
```

### 2.4 一步长什么样（口径落地）

```text
step k:
  1. 准入（按 policy）
       continuous: while active < max_batch and queue 非空 and queue[0].arrival_ts <= now: admit
       static    : if active == 0: 凑一波（队列不足则等到够或 trace 结束）
     每条新准入：prefill_one()（逐条，串行，无 pad）→ 采首 token → t_first
  2. 批 decode：decode_batch([r.last for r in active], [r.cache for r in active])
     → (B, V) → 逐行 argmax（一次 D2H）→ append
  3. 退出：budget / EOS / max_position → t_finish，释放 cache
每步记一行 step row：active / queued / admitted / retired / step_ms / prefill_ms / decode_event_ms
```

不变式（测试断言）：
- `active ≤ max_batch` 恒成立；
- 每条请求恰好 `finish` 一次，`t_arrival ≤ t_admitted ≤ t_first ≤ t_finish`；
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
  这个口径已冻结在 `docs/decisions.md` **D4**；`W2-0006`（`fixed-attn-path-paired`）是它的基线。

---

## 3. 三个实验（把「批」和「调度」分开）

### E0 · 重构无害性（1 run，microbench，threshold）

`W3-0001`：`impl="minillm"`，与 `W2-0006` 同 prompt / 同 dtype / 同 batch=1 / 同 max_new_tokens，
`[baseline] run_id="W2-0006"`，`[accept]` 复用 W2 四条判据（`ttft_cv_max/tpot_cv_max/memory_match_ratio/decode_speed_vs_hf_min`）。
**通过即证明 §2.2 的 model 改动没有动数字**；不通过就停下修，不做任何性能 run。
（`decode_speed_vs_hf_min` 在这里当**漂移比值**用：两臂同 kernel 后它比较的是
「W2-0006 的 TPOT / 本次的 TPOT」，≈1.0 即无漂移；名字沿用旧键，避免新增判据键。
注意它**只在同实现配对时**校验 `attn_prefill/attn_decode`——对 HF 臂的正式判据只看通用键，见 D4 后果。）

### E1 · 批处理摊薄 host（Q1，5 runs，microbench，record_only）

工作负载：`kind="synthetic"`、`arrival="closed"`（全部 t=0）、`requests=B`、`budgets=[128]`；
`[engine].impl="continuous"`（n=B 时两 policy 等价，这一组只测批，不测调度）。

| run | ctx | B | 用途 |
|---|---|---|---|
| W3-0002 | 512 | 1 | 单请求参照（E1 自身基线） |
| W3-0003 | 512 | 8 | P1 |
| W3-0004 | 512 | 32 | P2 |
| W3-0005 | 2048 | 1 | 长上下文的 `cat` 成本参照 |
| W3-0006 | 2048 | 8 | 长上下文 batch 收益 |

口径：E1 同时报两个数——`tok_s`（含 prefill 的墙钟口径）与**仅 decode 的每 token 成本**
（由 step rows 的 `Σstep_ms / Σtokens` 推得，或 `Σout/(makespan−Σprefill)`），
后者才是「批摊薄 host」的干净证据；B=8/ctx2048 时 prefill 约占 7%，不扣掉会把曲线压低。

产出：`fig_w3_amortization.png`（x=B、y=ms/token 与 tok/s，双轴；按 ctx 分线），
由 `scripts/plot_w3_amortization.py` 从三份 summary 的 `steps` 块生成（脚本不读 metrics.jsonl 原表）。
**B=32 × ctx2048 不跑**：KV ≈ 7.6 GiB + 每步 `cat` 7.6 GiB，会把「host 摊薄」和「拷贝成本」混在一起，
那一格留给 W4（预分配块）后再测，记在 notes 里。

### E2 · 调度在什么条件下回本（Q2，6 runs，scenario，配对）

工作负载：`kind="prompts"`（32 条固定集）、`arrival="poisson"`、`requests=64`、`max_new_tokens=128`、
`rate = 2 × capacity`（capacity 由冒烟测得，取整）、每种长度分布一对。

| run | policy | B | budgets | 用途 |
|---|---|---|---|---|
| W3-0007 | static | 8 | `[128]` | 均匀长度基线（P3） |
| W3-0008 | continuous | 8 | `[128]` | **P3：期望 ≈1.00×，负结果** |
| W3-0009 | static | 8 | `[16,32,64,128]` | 混合长度基线 |
| W3-0010 | continuous | 8 | `[16,32,64,128]` | **C1 判据就在这一对**（P4） |
| W3-0011 | static | 32 | `[16,32,64,128]` | B 放大 |
| W3-0012 | continuous | 32 | `[16,32,64,128]` | B 放大（P2/P4） |

可选（有余力再做，不进退稿）：`W3-0013/0014` 在 `rate = 0.8 × capacity` 下的一对，
用来说明「不饱和时连续批没有吞吐收益、只有 TTFT 尾部差异」。

**每对 run 的协议一致性**由 `thresholds._protocol_mismatches` 校验（D4 起分两层）：
**通用键** = 上面这些，**机制键** = `attn_prefill / attn_decode`（同为自实现的两臂必须相等）；
`prompt_sha256 / dtype / device_class / max_new_tokens / trace_sha256 / max_batch` 全等；
不一致 → 判据 `invalid`，不给比值（沿用 W2 的规矩）。

---

## 4. 配置契约（新增字段，逐条校验）

```toml
[workload]
kind = "prompts"
prompts = "bench/prompts.json"
max_new_tokens = 128
arrival = "poisson"        # closed | poisson；poisson 只允许 type="scenario"
rate = 20.0                # req/s；poisson 必填且 > 0
requests = 64              # 一个 trace 的请求数；closed 必填且 >= 1
budgets = [16, 32, 64, 128]  # 每条请求的生成上限，seed 抽样；缺省/[] 视为 [max_new_tokens]
# shuffle = true           # 可选：prompt 顺序按 seed 打乱（用于到达顺序不变性）
[scheduler]
policy = "continuous"      # static | continuous
[engine]
impl = "continuous"        # continuous | minillm（minillm = 单请求臂，行为不变）
max_batch = 8              # 并发度；scenario 模式下 batch_sizes 必须为 [1]
```

校验规则（`validate_config` 新增）：
1. `arrival="poisson"` ⇒ `type="scenario"` 且 `rate>0` 且 `requests>=1`；
2. `budgets` 每项 `1 <= b <= max_new_tokens`，非空；
3. `impl="continuous"` ⇒ `[scheduler].policy ∈ {static, continuous}`，`max_batch>=1`；
4. scenario 模式（poisson 或 impl=continuous）⇒ `batch_sizes == [1]`，否则报 P-20 的错；
5. `timer="cuda_event"` 仍要求 `device=cuda`（不动）；
6. `[accept].mode="threshold"` 时，`throughput_speedup_min` 只能出现在 scenario run。

---

## 5. 测试清单（红 → 绿，先写后实现）

### 5.1 `tests/unit/test_batch_forward.py`（新）

| 用例 | 断言 | 防的坑 |
|---|---|---|
| `test_batched_decode_matches_per_sequence` | 3 条**不同长度**的 prompt（5/9/13），decode 6 步：批 vs 逐条 logits `atol=1e-5`、id 全等 | 整条批路径的等价性 |
| `test_batched_prefill_is_rejected` | `B>1, T>1` 抛 `ValueError`（信息里带 D4） | 有人偷偷 pad 成 `(B,T_max)` 一次 forward（会让 SDPA 退 math，6.3×；见 P-22） |
| `test_rope_uses_per_row_positions` | 两条 past_len 不同（5 和 9）时批 decode == 逐条；且故意把两行都当成同一位置会不等 | **P-21：整批共用一张 RoPE 表 → 静默错 token** |
| `test_batch_of_one_is_bit_identical` | `torch.equal` 与今天单请求路径逐 bit 相同 | 重构不得改数字（E0 的单测版） |
| `test_attention_dispatched_once_per_batch` | 用 monkeypatch 计数：`hand_baddbmm.decode` 每层每步只调用 1 次（不是 B 次），`MLP.forward` == n_layer | D3 驳回的病（`28×B` 次派发）不能从 W3 后门回来 |

### 5.2 `tests/unit/test_scheduler.py`（新，全部用 fake engine，不需要权重）

| 用例 | 断言 |
|---|---|
| `test_capacity_never_exceeded` | 每步 `active ≤ max_batch` |
| `test_each_request_finishes_once` | 每 id 恰好一次 finish；无重复采样、无丢失 |
| `test_static_runs_waves` | `active` 在波次边界回到 0；最大并发 == max_batch |
| `test_continuous_refills_next_step` | 短请求在第 k 步退出 → 队首在第 k+1 步被准入（fake clock 精确到步） |
| `test_lifecycle_timestamps_monotone` | `arrival ≤ admitted ≤ first ≤ finish`，`ttft ≈ queue+prefill`（<1e-9 s，fake clock） |
| `test_budget_eos_and_max_position_terminate` | 三种终止条件各一条；EOS 不产生后续 token（P-06/P-07 的调度器版） |
| `test_arrival_order_invariance` | 同一批请求，两种到达顺序 → 逐 id 相同（C3 的单测） |
| `test_static_and_continuous_ids_equal` | 同 budget 下两 policy 的 token 序列逐 id 相同 |

### 5.3 `tests/unit/test_scheduler_equivalence.py`（新，tiny_model）

`policy="continuous"` 跑 3 条不同长度的请求（同 budget、closed trace），每条的输出
== `MiniLLMEngine.generate()` 单独跑的结果（逐 id）。这是 S3 在 ROADMAP §4 的护栏。

### 5.4 `tests/unit/test_loadgen_trace.py`（新）

同 seed 同参数 → 同 `trace_sha256`；改任一参数 → hash 变；`mean(interarrival) ≈ 1/rate`（1e-2 相对误差，n=5000）；
budgets 抽样频率与权重一致（±3%）；`shuffle` 可复现；prompt 轮转覆盖全部 32 条。

### 5.5 `tests/unit/test_scenario_summary.py`（新，纯聚合）

手工构造 6 条 request row + 2 条 trace row → 断言 `makespan/tok_s/req_s`、
p99 的 `ttft/queue/prefill`、`queue_share`、`ttft ≈ queue+prefill` 检查、`not_saturated` 标记。

### 5.6 扩展既有测试

- `tests/unit/test_thresholds.py`：`throughput_speedup_min`（有基线/无基线/协议不符→invalid）、
  `ttft_p99_decomposed`（bool 判定 + 缺项 → pending）；`attn_share_max` 只断言键存在且为 float（评测在 §5.7）；
- `tests/unit/test_run_contract.py`：poisson 配置被接受、五种非法配置被拒、scenario run 的
  manifest 含 `scheduler.policy/trace_sha256`、`batch_sizes=[2]` 在 scenario 下被拒。

### 5.7 `tests/integration/test_step_split.py`（新，`gpu` + `hf` 标记，不进快速门禁）

B ∈ {1,8,32} 各测 20 个 decode step，用 `torch.profiler` 拆 `GEMM / attention / 逐元素 / KV cat / host gap`，
断言 **B≤8 时 attention 占比 < `thresholds.lookup("attn_share_max")`**（C4；常量从
`configs/thresholds.toml` 读，不写死在测试里），并把整张表贴进 W3 的 notes（不进 summary）。

---

## 6. runner / summary / thresholds 的契约扩展（字段级）

### 6.1 `metrics.jsonl` 新增 `row_kind`

```jsonc
// row_kind="request"（每请求每 repeat 一行）
{"row_kind":"request","run_id":"W3-0010","model":"Qwen/Qwen3-0.6B","impl":"continuous",
 "policy":"continuous","repeat":0,"arrival_idx":17,"prompt_id":"medium-03","budget":64,
 "context":421,"out_tokens":64,"t_arrival_s":0.42,"queue_ms":1180.3,"prefill_ms":21.4,
 "ttft_ms":1201.7,"tpot_ms":2.6,"e2e_ms":1366.0,"kv_mb":97.2,"token_sha256":"…","ok":true}

// row_kind="step"（每 step 一行；W7 也要用）
{"row_kind":"step","run_id":"W3-0010","repeat":0,"step":311,"active":8,"queued":41,
 "admitted":1,"retired":1,"step_ms":20.9,"prefill_ms":21.4,"decode_event_ms":3.9,"kv_mb_sum":612.4}

// row_kind="trace"（每 repeat 一行 = 聚合口径的权威行）
{"row_kind":"trace","run_id":"W3-0010","repeat":0,"makespan_s":18.32,"out_tokens_total":3841,
 "tok_s":209.7,"req_s":3.49,"offered_rate":20.0,"achieved_rate":3.49,"queue_depth_max":56,
 "ttft_p99_ms":5210.0,"queue_p99_ms":5180.0,"prefill_p99_ms":24.0,"peak_concurrency":8,
 "saturated":true,"drops":0}
```

### 6.2 `summary.json` 新增块（`bench/scenario.py` 负责算，纯函数）

```jsonc
"throughput": {"n_traces":5,"tok_s":{…median/p95/cv…},"decode_only_tok_s":{…},"req_s":{…},
               "makespan_s":{…},"offered_rate":20.0,"achieved_rate":{…},"saturated":true,
               "analytic_straggler_ratio":2.13},      // 理想波次比 E[max_B]/E[mean]，从 trace 算
"latency":    {"ttft_ms":{"p50":…,"p95":…,"p99":…},"queue_ms":{…},"prefill_ms":{…},
               "queue_share_p99":0.98,"decomposed":true},
"steps":      {"by_active":[{"active":8,"n":2410,"step_ms":20.9,"prefill_ms":0.4,
                             "decode_event_ms":3.9,"host_gap_ms":16.6}]},   // step = prefill + decode + gap
"consistency":{"token_sha256_match_vs_baseline":64,"mismatch":0}   // 与配对基线交叉核对
```

`overall`/`groups`/`stability` 保持只吃 `row_kind="request"` 的行（W2 口径完全不变）；
scenario run 的 `groups` 按 **bucket** 分组（而不是精确 ctx），避免 32 个 context 把 summary 撑爆。
`key_metric`：`throughput.tok_s.median` 优先（scenario run 的索引就该显示 tok/s）。

### 6.3 `thresholds.py` 新增评测器

| 判据键 | 逻辑 |
|---|---|
| `throughput_speedup_min` | `baseline.throughput.tok_s.median / ours.throughput.tok_s.median`；先校验 `_protocol_mismatches`（新增 `trace_sha256`、`max_batch`）→ 不符记 `invalid`；缺基线 → `pending`；两个 run 都要 `saturated=true` |
| `ttft_p99_decomposed` | 结构检查：`latency.ttft_ms.p99`、`queue_ms.p99`、`prefill_ms.p99` 都存在，且逐请求 `|ttft−(queue+prefill)| < 1 ms` 的比例 ≥ 99% |

> `attn_share_max` **不是 run 判据**：它由 §5.7 的集成测试直接从 `thresholds.toml` 取值断言，
> 避免把 profiler 拆分开销塞进每一次性能 run。

### 6.4 manifest 增补

```jsonc
"scheduler": {"policy":"continuous","max_batch":8,"admit":"fill_free_slots","prefill":"per_sequence"},
"trace": {"arrival":"poisson","rate":20.0,"requests":64,"budgets":[16,32,64,128],
          "seed":0,"sha256":"…","capacity_estimate_req_s":3.5}
```

---

## 7. 施工顺序与门禁（约 5 天）

| 天 | 动作 | 门禁（必须绿才进下一步） |
|---|---|---|
| D1 上午 | 把 §2.1 口径 + §1 判据写进 `docs/decisions.md` **D4（proposed）**；新建 `configs/thresholds.toml` 的 `[W3.continuous].attn_share_max`；先写 §5.1 全部用例（红） | `pytest -m "not slow and not hf"` 只红在新增用例 |
| D1 下午 | 实现 model 批 forward + per-row RoPE | §5.1 全绿；`test_align_qwen3.py`（`MINILLM_RUN_HF=1`）仍全绿；单请求路径逐 bit 不变 |
| D2 | `serve/scheduler.py` + `engine.prefill_one/decode_batch`；§5.2/5.3 | §5.2/5.3 全绿；既有 22 个单测文件全绿 |
| D3 上午 | `loadgen` trace + hash，§5.4 | §5.4 全绿 |
| D3 下午 | `bench/scenario.py` + `bench/engines/continuous.py` + runner 接线，§5.5 | §5.5 全绿；`--out /tmp/... --allow-dirty` 的 dummy 冒烟能跑通 |
| D4 上午 | `thresholds.py` 三个评测器 + §5.6 扩展；`W3/9001-scheduler-smoke.toml`（dummy） | 契约测试全绿；`scripts/exp_index.py --check` 绿 |
| D4 下午 | 用户在 4090 上跑 **E0**（W3-0001） | 四条判据 pass，才允许正式性能 run |
| D5 | E1 → 用户跑 5 个 run → 出图；写 D4 的 E1 结论 | 每 token 成本曲线与 P1/P2 对照 |
| D5 末 | E2 冒烟定 capacity → 用户跑 6 个 run（+2 可选） | C1/C2/C3 判定；`thresholds.toml` 的 W3 段改 `calibrated` + `basis_run` |

每个正式 run 之后：`python scripts/exp_index.py` → 检查 `--check`；`notes.md` 由 runner 生成模板，
**observation 段我起草，`verdict.json` 保持 `unreviewed` + `reviewer: null`**，直到用户确认。

---

## 8. 风险与止损

| # | 风险 | 触发信号 | 处置 |
|---|---|---|---|
| R1 | host 摊薄不了（step 时间随 B 线性涨） | E1：B=8 的 step_ms > 3× B=1 | 记负结果，W3 收口（ROADMAP §8：< 1.2× 不硬调）。探针显示不会发生 |
| R2 | 均匀长度下 1.3× 不可达 | E2 均匀对 ≈1.0× | **按设计如此**：判据只在混合对判；均匀对进负结果专章（P3） |
| R3 | 注意力在 B=32 成瓶颈 | step_split 显示 > 20% | 先记数。D4 已定 decode 走 `hand_baddbmm`（B=8 实测 0.7 ms/step），> 20% 说明是 KV 读取而非派发——那是 W4 分页的问题，不在 W3 改 |
| R4 | KV `cat` 吃掉落差 | ctx 2048 的 `kv_cat_ms` > 20% | 记录为 W4 的目标；**不在 W3 改 KV 分配**（否则机制混在一起） |
| R5 | 真实时钟抖动导致 tok/s CV > 10% | 5 个 repeat 的 CV 超线 | 加长 trace（`requests` 96/128）**新开 run**，不挑点 |
| R6 | B=32 × ctx2048 显存/带宽爆炸 | OOM 或 step 异常长 | 该格不跑（已排除在 E1 矩阵外） |
| R7 | 静态臂被质疑「被削弱」 | 评审意见 | 饱和 trace 下静态臂永远能凑满一波（λ=2×capacity），等待填充不会触发；把这条写进 notes 的「限制」段 |
| R8 | 重构让 W2 数字漂移 | E0 判据红 | 停性能 run，用 `git bisect` 式的逐层对拍定位（W1 的 golden 是对拍地基） |

---

## 9. 要登记到 `PITFALLS.md` 的新坑（实现时补，附回归测试）

| ID | 坑 | 回归 |
|---|---|---|
| P-20 | `[workload].batch_sizes` 只是行标签，不是并发度；把它当并发会得到「假的 batch=4」 | `test_run_contract.py` |
| P-21 | 批 decode 里各请求位置不同，**RoPE 表不能整批共用一张**（静默错 token，且单请求测试查不出） | `test_batch_forward.py::test_rope_uses_per_row_positions` |
| P-22 | 为了「先跑通」把批内 pad 成 `(B,T_max)`：SDPA 退 math、闸值全作废 | `test_batch_forward.py::test_batched_prefill_is_rejected` |
| P-23 | 静态批与连续批若各写一份 decode 循环，收益会混进 kernel 差异 | `test_static_and_continuous_ids_equal` + 构造上同一个 `decode_batch` |

---

## 10. 不在 W3 做

- chunked prefill（掩码路径已支持 `past_len>0, T>1`，但不接入调度）；
- 分页 KV（W4）、CUDA graph 与 varlen kernel（**D4 已定：均不进 W3 数字**；graph 保留为 parked，W4 接线）、prefix cache；
- 任何形式的长度方向 pad；
- 量化、MTP/投机解码、多模态（D2）；
- 改 KV 分配策略（`cat` 保持原样，只测其占比）。

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
python -m bench.run --config configs/experiments/W3/0001-single-request-recheck.toml
python -m bench.run --config configs/experiments/W3/0002-batch-b1-ctx512.toml
# … 0003 … 0012（§3 矩阵）
python scripts/exp_index.py

# 出图
python scripts/plot_w3_amortization.py
```

---

## 12. 交付物

- `serve/scheduler.py`、`bench/scenario.py`、`bench/engines/continuous.py`、`bench/loadgen.py`（trace）
- 新测试 6 个文件 + 2 处扩展，快速门禁全绿
- run：`W3-0001`（重构无害）+ E1 ×5 + E2 ×6（+2 可选）
- `docs/fig_w3_amortization.png`（E1）、`results/W3/*/notes.md` 草稿 + `verdict.json`（unreviewed）
- `docs/decisions.md` **D4**（W3 负载与口径预注册 + 实测结论）
- `configs/thresholds.toml` `[W3.continuous]` → `calibrated` + `basis_run`
- `ROADMAP.md` §12 回填 W3 证据 run、§13 一条修订；`AGENTS.md` / `serve/README.md` 的模块状态更新

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
