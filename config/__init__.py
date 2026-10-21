import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    tokenizer_path: str = "datasets/luxun/tokenizer.json"
    train_dir: str = "datasets/luxun/train"
    val_dir: str = "datasets/luxun/val"
    block_size: int = 256
    def to_dict(self):
        # asdict 递归转 dict：字段全部落到字面量，缺省即默认 → dump 出来完整无缺省模糊
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_iters: int = 1200       # 训练预算：总优化步数（LLM 预训练标准形态，非 epoch 式）
    warmup_steps: int = 50
    lr: float = 3e-3
    min_lr: float = 3e-4
    accum_steps: int = 1
    max_norm: float = 1.0   # 梯度裁剪上限；0 = 关闭（约定，非 torch 语义：torch 的 0 会清零梯度）
    log_interval: int = 50      # 每多少步记录一次 train loss
    eval_interval: int = 200    # 每多少步在 val 流评估一次
    seed: int = 0
    def to_dict(self):
        return asdict(self)


@dataclass
class ModelConfig:
    """模型结构参数（B3 结构矩阵的主变量）。
    不含 vocab_size / max_position_embeddings —— 它们分别由 tokenizer 契约、block_size 派生，
    训练时注入，不进实验 toml。
    """
    n_embd: int = 128            # 宽度维度
    n_head: int = 4              # query 头数
    n_kv_head: int = 4           # KV 头数（= n_head 时 MHA；调小即 GQA/MQA 实验）
    n_layer: int = 4             # 深度
    n_inner: int | None = None   # SwiGLU 中间维；None = n_embd * 4
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self):
        # toml 无 null：n_inner=0 约定"默认 n_embd*4"，此处归一为 None（MiniLLMConfig 语义）
        if self.n_inner == 0:
            self.n_inner = None

    def to_dict(self):
        return asdict(self)

    def to_minillm_config(self, vocab_size: int, max_position_embeddings: int):
        """桥接：toml 的 [model] → MiniLLMConfig（模型真正使用的对象）。
        vocab_size / max_position_embeddings 由调用方注入（tokenizer 契约 / block_size）。"""
        from model.config import MiniLLMConfig
        return MiniLLMConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            **self.to_dict(),
        )


@dataclass
class WeightConfig:
    directory: str = "weights"
    name: str = "minillm.pth"
    resume: bool = False
    resume_name: str = "resume.pt"
    resume_interval: int = 500
    @property
    def path(self) -> Path:
        return Path(self.directory)/self.name
    @property
    def resume_path(self) -> Path:
        return Path(self.directory)/self.resume_name
    def to_dict(self):
        # path/resume_path 是 property，不在 dataclass 字段里，asdict 不会误带
        return asdict(self)


@dataclass
class Config:          # 你也可以直接命名为 Config
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    weight: WeightConfig = field(default_factory=WeightConfig)

    @classmethod
    def from_toml(cls, path: Path | str) -> "Config":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        # 仅提取 TOML 中存在的字段，忽略未定义的额外字段（提高容错）
        data_kwargs = {
            k: v for k, v in raw.get("data", {}).items()
            if k in DataConfig.__dataclass_fields__
        }
        model_kwargs = {
            k: v for k, v in raw.get("model", {}).items()
            if k in ModelConfig.__dataclass_fields__
        }
        train_kwargs = {
            k: v for k, v in raw.get("train", {}).items()
            if k in TrainConfig.__dataclass_fields__
        }
        save_kwargs = {
            k: v for k, v in raw.get("weight", {}).items()
            if k in WeightConfig.__dataclass_fields__
        }

        return cls(
            data=DataConfig(**data_kwargs),
            model=ModelConfig(**model_kwargs),
            train=TrainConfig(**train_kwargs),
            weight=WeightConfig(**save_kwargs),
        )

    def to_dict(self):
        # 递归聚合 data/model/train/weight 四块 —— 与 toml 四段结构一一对应
        return asdict(self)
