# MiniLLM 路线图

> 最终目的（北极星）：**光流 LLM**——语言与图片预测同源，用 transformer 自回归 next-state 范式做光流/帧预测，回到 PIV 做算法创新。
> 路径：手写 LLaMA 全链路 → 读懂 HF 源码 → DeepSeek 结构与训练（MLA/MoE/预训练-微调）作为"结构肌肉训练"，迁移到光流预测。
> 每条都带验收标准，不验收不下结论。回归测试（check_kvcache.py + check_leftpad_batch.py + check_eos_batch.py）每次改 layers 后跑。

## 阶段 A：生成引擎（推理侧）

- [✓] A1 手写采样（greedy → temperature → top-k → top-p）
- [✓] A2 手写自回归生成循环 + attention/loss mask
- [✓] A3 KV cache 版 Attention + 一致性校验（check_kvcache.py，cache vs 全量重算 24/24 一致）
- [✓] A3.1 批量解码·左 pad + attention_mask 贯通
  - generate 增加 attention_mask；prefill 与 decode 都带
  - **decode 的 pad mask 必须覆盖 cached pad 列**：每步 concat 追加 1 列，宽度 = key 总长
  - 验收：左 pad 变长 batch，逐 token 与"单序列无 pad"参考一致（check_leftpad_batch.py 3/3 过）
- [✓] A3.2 批量解码·EOS 发散（todo.md 原题）
  - alive/cull 账本对齐：buffer + outputs + orig 映射，压缩空间当工作区，行结束即 flush
  - 验收：batch 各行按各自 EOS 提前结束，输出长度正确（check_eos_batch.py 3/3 过，15/23/12）
- [✓] A3.3 KV cache 加速比基准（cache vs 全量重算，随 seq 长度画曲线）
  - 产物 `tests/bench_kvcache.py` + `bench_kvcache.png`
  - toy 规模加速比 ~0.9x（cache 略慢）：瓶颈是 kernel launch/Python 循环，非 FLOPs；
    KV cache 价值在"大 T × 大模型"才显现（阶段 D MLA/PagedAttention 动机）
- [✓] A4 交互 demo（终端流式续写）
  - 验收：demo.py REPL——合法输入流式续写（greedy x+1），非法输入（非整数/越界）拦截重输不崩，空行/Ctrl-D 退出

## 阶段 B：训练管线（生产级）

- [✓] B1 循环工程化：迭代式（max_iters + eval_interval）/ val 划分 / 多 batch / grad clip / LR 调度 / 断点续训
  - 验收通过：全新训 loss 5.58→0.0006 acc 1.0；断点续训从 step 1001 续跑；回归 3/3（check_kvcache / leftpad / eos）
  - 产物：weights/minillm.pth + resume.pt + training_curve.png；checkpoint 格式统一为 dict{config, model, optimizer, scheduler, step}（key="model"）
  - 坑：resume 后剩余步数 < eval_interval 时 eval_log 为空 → 绘图 zip(*[]) 崩，已防御
- [✓] B2 真实数据：BPE tokenizer + 真实中文语料全链路
  - [✓] B2a 定案：单字起始；target_vocab=8192；特殊 token PAD=0/BOS=1/EOS=2/UNK=3；语料 2-3MB；max_position_embeddings 256
  - [✓] B2b 语料：`datasets/luxun_raw`（全集 13 部）→ `datasets/luxun/{train,val}` 清洗完成；train 285 篇 87.9万字符 / val 21 篇 12万字符（朝花夕拾+故事新编）；脚本 `data/preprocess_luxun.py`
  - [✓] B2c tokenizer：官方 `tokenizers`(BpeTrainer) 单字起始 BPE；vocab 8192；特殊 token 0-3；换行独立（按 \n 切块，段落内合并）；往返一致（长文 0.73 压缩率）；产出 `datasets/luxun/tokenizer.json`。坑：pre-tokenizer 按单字符切则无 pair 可合并（BPE 按 word 内统计）；decode 默认插空格需 BPEDecoder
  - [✓] B2d dataloader：`TextData`（方案 A：BOS/EOS 包裹 + 不跨文档滑窗，定长 256）；train 277 篇 / val 21 篇；`MiniLLMConfig(vocab_size=8192, max_position_embeddings=256)` 对齐 tokenizer 契约；`build_data` 切换；train.py 跑通（250 iters 冒烟 loss 9.02→6.77）。坑：**prompt 占满窗口则生成预算 = max_position_embeddings − T_max = 0**（护栏立即 break），evaluate_generation 需 prompt 只取 128 留出生成空间
  - [✓] B2e 训练 + 调参 + 评估：正式跑 1200 iters（`training_curve.png`；train.log 暂存 /tmp）。val ppl 8292→~1300，**过拟合拐头 step~400**（train loss 继续降 3.39、val 回升 7.55）→ 预算 1200 偏大，B3 按 ~500 重定；当前 weights/minillm.pth 是过拟合版非最优
    - 生成：词级像中文（书名号/`〔5〕` 注释形态都学到）、句级/段落级乱 → 语料 120 万 token 只够词法，不够段落组织（低频长程证据稀疏）
    - 结论：**minillm 定位 = 条件语言建模**（非复读机）；"输出可读"降级为附带观察（需扩语料千万级/SFT 才可及，移出主线验收线）；评估以 **val ppl 为主指标**（同协议同 seed 跨配置可比 = 语言的 EPE）
- [✓] B3 结构实验矩阵（语料固定 = 现有 luxun，不扩）：为光流 LLM 练"结构→行为"的测量方法论
  - [✓] B3a 基线重定 + 规模扫（B3a/B3b 合并执行）：三档 2L×64d / 2L×128d / 4L×128d 各 500 iters（exp002/exp003/exp001）
    - 拐点漂移规律：容量越大拐点越早——4L×128d 拐 ~400（过拟合）、2L×128d 500 步内趋平、2L×64d 全程下降（欠拟合）
    - 操作点：**2L×64d（0.65M）欠拟合端**；语料 120 万 token 支撑上限 ~1.3M 参数 500 步不拐头
    - 产物：exp002/exp003 归档 result.md；`tools/plot_curve.py` 多档叠图（显示 dpi 恢复默认、保存 300dpi）
  - [✓] GQA 实现校验：n_kv_head=2 一档（exp004），同 seed 下训练轨迹与 MHA 逐点重合（末值 1012 vs 975）→ 实现正确，锚点入 result.md
  - [✗→已砍] B3c 组成对比（ALiBi / Pre-Post-norm / 窗口注意力）：按筛选标准全部否决——**结论已知 + toy 不可测 + 收益维度测不到**（ppl 测不出位置编码/局部性差异，2-4 层深度 norm 差异不显现）
    - **B3 核心结论：toy 上结构性对比无信息增益，组件学习转为"理论理解 + 阅读代码"**（ALiBi/窗口/pre-post-norm 原理已讲，D 阶段实现时直接落地）
    - 方法论产出：**实验筛选两问**（① 结论已知吗？② 测量工具测得到收益维度吗？）——GQA 锚点式"实现校验"保留，效果对比仅限"结论未知且可测"
  - [✓] B3d 评估协议基建：`eval.py`（--exp 评估/多实验对比表）+ `data/eval_prompts.txt`（固定 prompt）+ `tools/plot_curve.py`（读 csv 现画现 show，不落盘）= val ppl + 生成样本 + 多样性三层
  - 多样性 `output_diversity` 落地：方案 B 输出分布熵（归一化）。诊断过 C 序列去重/完全失效、C′ n-gram Jaccard/长输出稀释重合，最终 B 分布熵度量"用词集中度"才对准 ppl 保守盲区。验收 t0.1=0.94<t1.0=0.98 单调通过；toy 模型上仅弱分层（多条高概率延续稀释温度效应）
  - 坑：toy 上形态收益有时不可见（同 A3.3 KV cache）——看性质（低秩损失/专家专化/路由发散）而非收益

## 阶段 C：架构对齐

- [✓] C1 前 2 层对齐 TinyLlama（max diff ~1e-5，cosine=1.0）
- [✓] C2 语料升级全链路：`datasets/fineweb_cn`（Fineweb-Edu-Chinese-V2.1，train 385万字符/val 57万）→ 词级✓ / 句级✗（规模证据）
  - tokenizer 重训：`data/train_tokenizer.py` 参数化（--data/--vocab）；vocab **16384** 定案（压缩率 0.607 vs luxun 0.73，8192 档 alphabet 膨胀 merge 不够）；特殊 token 0-3 契约不变；val 压缩率验收防泄漏
  - 结构：4L×128d×4h×窗口512，总参 2.8M（嵌入 16384×128×tie=2.1M）；batch 32×512=16K token/iter；~1.15s/iter
  - 流程：冒烟测速 → 正式训练。**lr 调度归一化到 max_iters**（改预算=改 lr 轨迹，拐点随之漂移）→ 定案：3000 调度 + resume_interval=200 在拐点窗口 Ctrl-C 提 resume.pt 为拐点模型
  - 结果（exp005）：拐点 @600 ppl 856 → 取 @601 提档 eval **ppl 885 / acc 0.075**；生成词级像现代中文、句级语义跳脱（未达验收）
  - **规模结论**：luxun(120万tok)→fineweb(234万tok) 数据翻倍只推进一档词法；句级需 20-50M 参数 + 1-3 亿 token（M1 单次 14h+ 不可行）→ 玩具定位成立，C2 收尾
  - 机制补丁：config.json 训练开始即写（不依赖正常结束）；eval.py 默认记录 + --no-record；台账 archive 归档语义（tools/prune.py + eval 自动归档）
  - 注意：ppl 与 luxun 实验不可比（vocab/语料不同），指标只在 fineweb 内部有意义
- [ ] C2b（原 C2）全模型对齐 / 零样本续写对比 —— 已放弃（TinyLlama 不再需要；C1 已覆盖 block 逻辑，结构差异仅 tie 开关与规模）

## 阶段 D：DeepSeek 结构迁移（光流 LLM 的结构训练）

> D 主线对齐 DeepSeek-V4（2026-04）：混合注意力 CSA/HCA、DeepSeekMoE、GRPO 后训练。
> 学习路径沿压缩谱系递进：MLA 前置（维度压缩）→ CSA/HCA（序列压缩）→ MoE → GRPO。
> D1 为 V3 谱系前置（V4 已弃 MLA，学它是为理解 CSA 算子基础与"压缩演进"直觉）；D5 为推理优化收尾（非 V4 架构）。
> 方法论沿用 B3：锚点校验（还原/退化时与基线轨迹重合）+ 效率/性质测量；效果收益 toy 不可测，不作验收。

- [ ] D1 MLA 前置节（V3 谱系起点；V4 已弃，不替换注意力、不做完整切换）
  - 原理：c_kv 低秩投影（每 token 存 d_c 而非 n_kv×d_head）；RoPE 只在 query 加
  - **吸收 cache 技巧**：W^UK 吸收进 W^Q，decode 不物化 K/V——CSA/HCA 同样依赖的算子级手艺，需亲手实现
  - 理解 V4 为何弃用：维度压缩后 KV 仍随序列线性增长，1M 上下文必须转向序列压缩（这是最大的学习价值）
  - 验收：锚点（低秩还原满秩时与 MHA 轨迹重合）+ KV 压缩率实测；效果收益不可测
  - 迁移：压缩条件历史（前 k 帧）→ 预测器记忆
- [ ] D2 序列压缩注意力 CSA/HCA（V4 核心，简化版）
  - CSA：softmax 门控池化压缩（m=4，相邻块重叠）+ 简化 indexer 选 top-k 块（稀疏）；HCA：重度压缩（m'=128）全量稠密注意力；交替分层
  - 验收：锚点（压缩率=1 时与 MHA 轨迹重合）+ KV 显存压缩率实测 + 压缩条目注意力分布性质
  - 迁移：光流帧序列压缩——每帧→压缩条目，"压缩历史"直接对应
- [ ] D3 DeepSeekMoE（V4 版改进）
  - 细粒度路由专家 + 1 共享专家；前 1-3 层 **hash 路由**（token-id 静态映射，后续层学习路由）；affinity `Sqrt(Softplus)` + aux-loss-free（e_score_correction_bias 偏置不流梯度）
  - 验收：性质观察——路由专化/稀疏度/负载均衡收敛、hash vs learned 路由行为对比；效果收益不可测
  - 迁移：流态专家路由（层流/湍流/涡/剪切）
- [ ] D4 GRPO（V4 后训练用 RL）
  - 复读机沙盒（0..62→1..63，自动可验证奖励）；组内相对优势 + KL 控制
  - 验收：正确率提升、KL 稳定、组内优势行为；域专家→on-policy distillation 可作进阶
  - 迁移：弱关联；物理一致性用 NS loss 约束更有效
- [ ] D5 PagedAttention / token 级 continuous batching（推理引擎优化，非 V4 架构；A3.3 动机落地，结构主线后收尾）
- [ ] 可选理论/实验：mHC（多流残差 + Birkhoff 双随机投影，深网络稳定化，toy 只学原理）、Muon vs AdamW、MTP 多 token 预测

## 阶段 E：光流 LLM（最终目标）

> E 方向 outline 依用户初稿定（2026-08）：冻结编码器当"嵌入层"+ 纯解码器堆叠；D 完成后详议。
> 核心哲思：智能 = 有限算力下对海量数据的最优压缩；结构工程定义"思考方式"而非智能本身。

- [ ] E1 架构形态：冻结预训练视觉编码器（DINOv2 类）当嵌入层 + 纯解码器堆叠（非因果全注意力 + 2D RoPE，无 encoder-decoder 分离）
- [ ] E2 双帧交互与解码：帧间交叉/拼接自注意力找对应；**空间并行直接解码**输出光流（拒绝像素级自回归），帧级自回归保留（t→t+1 = 光流的 next-token）
- [ ] E3 上采样头：RAFT 式 convex upsampling / 轻量 CNN（低分辨率特征 → 稠密逐像素光流）
- [ ] E4 训练与推理：冻结编码器只训解码器；L1/L2 + smoothness loss；推理复用参考帧特征缓存（节省 ~50% 前向）
- [ ] E5 物理约束：NS 残差/散度 loss 防物理漂移（语言无硬约束、光流必须有——最大差异点，也最大创新空间）
- [ ] E6 数据与评估：合成流场预训练 + 真实 PIV 微调（PIV 真值稀缺是社区痛点）；指标 EPE；评估协议同 B3d

## 横切关注

- 回归：改 layers 后跑 check_kvcache.py + check_leftpad_batch.py + check_eos_batch.py
- 文档随动：CLAUDE.md / PITFALLS.md 按阶段更新
- 左 pad 的 RoPE 约定（共享右列 → 共享 past_len）在 A3.1 落地时一并锁死
