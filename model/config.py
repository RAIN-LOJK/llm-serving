from transformers import PretrainedConfig


class MiniLLMConfig(PretrainedConfig):
    model_type = "minillm"

    def __init__(
        self,
        vocab_size=256,
        max_position_embeddings=256,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_layer=4,
        n_inner=None,
        dropout=0.0,
        bias=False,
        norm_eps=1e-5,
        tie_word_embeddings=True,
        head_dim:int|None=None,
        rope_theta: float = 1e4,
        qk_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.dropout = dropout
        self.bias = bias
        self.norm_eps = norm_eps
        self.tie_word_embeddings = tie_word_embeddings

        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        # Attention is not configurable (docs/decisions.md D4): model/layers.py dispatches on
        # T to the two constant backends. There is deliberately no attn_prefill/attn_decode.
