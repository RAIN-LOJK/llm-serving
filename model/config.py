from transformers import PretrainedConfig


class MiniLLMConfig(PretrainedConfig):
    model_type = "minillm"

    def __init__(
        self,
        vocab_size=256,
        block_size=64,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_layer=4,
        n_inner=None,
        dropout=0.0,
        bias=False,
        norm_eps=1e-5,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.dropout = dropout
        self.bias = bias
        self.norm_eps = norm_eps
        self.tie_word_embeddings = tie_word_embeddings
