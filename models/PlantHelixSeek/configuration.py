# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HelixSeek model configuration"""

import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HelixSeekConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HelixSeekModel`]. It is used to instantiate a
    HelixSeek model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the HelixSeek model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the HelixSeek model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HelixSeekModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        ffn_hidden_size (`int`, *optional*, defaults to 4 * hidden_size):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if config.is_decoder=True.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 5):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 4):
            The id of the "end-of-sequence" token.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `None`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        add_bias_linear (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in various layers
        model_mode (`str`, *optional*, defaults to `"causal"`):
            Model mode - "causal" for autoregressive generation, "mlm" for masked language modeling
        classification_pooling (`str`, *optional*, defaults to `"last"`):
            Pooling strategy for sequence classification. Options: "last" (last non-padding token),
            "cls" (use <cls> token), "mean" (mean pooling), "max" (max pooling).
    """

    model_type = "HelixSeek"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        hidden_size=2560,
        add_bias_linear=False,
        ffn_hidden_size=None,
        num_attention_heads=32,
        num_key_value_heads=None,
        sliding_window=None,
        attention_dropout=0.0,
        hidden_act="silu",
        num_mem_blocks=1,
        use_mem_rope=False,
        rope_theta=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=1,
        bos_token_id=5,
        eos_token_id=4,
        use_long_context=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        model_mode="causal",
        classification_pooling="mean",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.num_mem_blocks = num_mem_blocks
        self.use_mem_rope = use_mem_rope
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.add_bias_linear = add_bias_linear
        self.use_long_context = use_long_context
        if use_long_context:
            self.max_position_embeddings = 16384

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_attention_heads = num_attention_heads
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act

        if ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.model_mode = model_mode
        self.classification_pooling = classification_pooling

        if self.model_mode not in ["causal", "mlm"]:
            raise ValueError(
                f"model_mode must be 'causal' or 'mlm', got {self.model_mode}"
            )

        if self.classification_pooling not in ["last", "cls", "mean", "max"]:
            raise ValueError(
                f"classification_pooling must be 'last', 'cls', 'mean', or 'max', got {self.classification_pooling}"
            )

        # HelixSeekDelta specific configuration (initialize with defaults)
        self.linear_attn_config = None
        self.q_lora_rank = None
        self.kv_lora_rank = None
        self.qk_nope_head_dim = None
        self.qk_rope_head_dim = None
        self.v_head_dim = None
        self.mla_use_nope = False

        # MoE configuration (initialize with defaults)
        self.moe_intermediate_size = None
        self.moe_renormalize = True
        self.moe_router_activation_func = "sigmoid"
        self.num_experts = None
        self.num_experts_per_tok = None
        self.num_shared_experts = 0
        self.routed_scaling_factor = 1.0
        self.first_k_dense_replace = 0
        self.moe_layer_freq = 1
        self.num_expert_group = 1
        self.topk_group = 1

        # Override with kwargs if provided
        if "linear_attn_config" in kwargs:
            self.linear_attn_config = kwargs.pop("linear_attn_config")
        if "q_lora_rank" in kwargs:
            self.q_lora_rank = kwargs.pop("q_lora_rank")
        if "kv_lora_rank" in kwargs:
            self.kv_lora_rank = kwargs.pop("kv_lora_rank")
        if "qk_nope_head_dim" in kwargs:
            self.qk_nope_head_dim = kwargs.pop("qk_nope_head_dim")
        if "qk_rope_head_dim" in kwargs:
            self.qk_rope_head_dim = kwargs.pop("qk_rope_head_dim")
        if "v_head_dim" in kwargs:
            self.v_head_dim = kwargs.pop("v_head_dim")
        if "mla_use_nope" in kwargs:
            self.mla_use_nope = kwargs.pop("mla_use_nope")
        if "moe_intermediate_size" in kwargs:
            self.moe_intermediate_size = kwargs.pop("moe_intermediate_size")
        if "moe_renormalize" in kwargs:
            self.moe_renormalize = kwargs.pop("moe_renormalize")
        if "moe_router_activation_func" in kwargs:
            self.moe_router_activation_func = kwargs.pop("moe_router_activation_func")
        if "num_experts" in kwargs:
            self.num_experts = kwargs.pop("num_experts")
        if "num_experts_per_tok" in kwargs:
            self.num_experts_per_tok = kwargs.pop("num_experts_per_tok")
        if "num_shared_experts" in kwargs:
            self.num_shared_experts = kwargs.pop("num_shared_experts")
        if "routed_scaling_factor" in kwargs:
            self.routed_scaling_factor = kwargs.pop("routed_scaling_factor")
        if "first_k_dense_replace" in kwargs:
            self.first_k_dense_replace = kwargs.pop("first_k_dense_replace")
        if "moe_layer_freq" in kwargs:
            self.moe_layer_freq = kwargs.pop("moe_layer_freq")
        if "num_expert_group" in kwargs:
            self.num_expert_group = kwargs.pop("num_expert_group")
        if "topk_group" in kwargs:
            self.topk_group = kwargs.pop("topk_group")

        self.layers_block_type = [
            "Transformer_1",
            "Transformer_2",
            "Transformer_3",
            "Transformer_4",
        ]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_chaoba_mla(self) -> bool:
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    def get_layer_type(self, layer_idx: int) -> str:
        if layer_idx < 0 or layer_idx >= len(self.layers_block_type):
            return "Unknown"
        layer_name = self.layers_block_type[layer_idx]
        if "_" in layer_name:
            return layer_name.split("_")[0]
        return layer_name

    def is_chaoba_delta_layer(self, layer_idx: int) -> bool:
        layer_name = (
            self.layers_block_type[layer_idx]
            if layer_idx < len(self.layers_block_type)
            else ""
        )
        return layer_name.startswith("HelixSeekDelta")

    def is_chaoba_mla_layer(self, layer_idx: int) -> bool:
        layer_name = (
            self.layers_block_type[layer_idx]
            if layer_idx < len(self.layers_block_type)
            else ""
        )
        return layer_name.startswith("HelixSeekMLA")
