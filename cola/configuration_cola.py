import json
from transformers import LlamaConfig
from transformers.modeling_rope_utils import rope_config_validation


class ColaConfig(LlamaConfig):
    model_type = "cola"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0,
        mlp_bias=False,
        head_dim=None,
        rank=None,
        lr_act_type=None,
        only_lr_act=True,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            attention_dropout,
            mlp_bias,
            head_dim,
            **kwargs,
        )
        self.rank = rank if rank is not None else hidden_size // 4
        self.lr_act_type = lr_act_type if lr_act_type is not None else hidden_act
        self.only_lr_act = only_lr_act