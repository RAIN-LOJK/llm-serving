from transformers import PretrainedConfig

_LAYER_TYPES = ("full_attention", "linear_attention")


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
        layer_types: list[str] | None = None,
        attn_output_gate: bool = False,
        partial_rotary_factor: float = 1.0,
        mrope_section: list[int] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int | None = None,
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

        # Architecture facts, not backend switches. Attention is still not configurable
        # (docs/decisions.md D4): model/layers.py dispatches on T to the two constant
        # backends, and there is deliberately no attn_prefill/attn_decode.
        #
        # ``layer_types`` is the per-layer schedule Qwen3.5 declares ("linear_attention" or
        # "full_attention"); None keeps the all-attention Qwen3-0.6B/toy shape. The linear_*
        # fields are read and validated only because W6 needs them; nothing in W5 consumes
        # them. Defaults are chosen so that passing no new field reproduces the old config.
        self.layer_types = self._checked_layer_types(layer_types)
        self.attn_output_gate = attn_output_gate
        self.partial_rotary_factor = partial_rotary_factor
        self.mrope_section = None if mrope_section is None else [int(x) for x in mrope_section]
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        if self.mrope_section is not None:
            total = sum(self.mrope_section)
            expected = self.rotary_dim // 2
            if total != expected:
                raise ValueError(
                    f"mrope_section={self.mrope_section} sums to {total}, but "
                    f"rotary_dim={self.rotary_dim} requires "
                    f"sum(mrope_section) == rotary_dim // 2 == {expected}"
                )

    def _checked_layer_types(self, layer_types: list[str] | None) -> list[str]:
        if layer_types is None:
            return ["full_attention"] * self.n_layer
        layer_types = list(layer_types)
        if len(layer_types) != self.n_layer:
            raise ValueError(
                f"layer_types must have exactly n_layer={self.n_layer} entries, "
                f"got {len(layer_types)}"
            )
        for index, kind in enumerate(layer_types):
            if kind not in _LAYER_TYPES:
                raise ValueError(
                    f"layer_types[{index}]={kind!r} is not a supported layer type; "
                    f"expected 'full_attention' or 'linear_attention'"
                )
        return layer_types

    @property
    def rotary_dim(self) -> int:
        """Width of the rotated slice of each head.

        Qwen3.5-0.8B rotates 64 of its 256 head dimensions
        (``int(head_dim * partial_rotary_factor)``); the tail passes through unchanged.
        ``head_dim`` falls back to ``n_embd // n_head`` exactly like ``model/layers.py``.
        """
        head_dim = self.head_dim or self.n_embd // self.n_head
        return int(head_dim * self.partial_rotary_factor)
