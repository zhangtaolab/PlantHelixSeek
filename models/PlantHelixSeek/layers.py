import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import HelixSeekRMSNorm
from .mlp import HelixSeekMoeSparseMoeBlock, HelixSeekMLP
from .attention import HelixSeekAttention, HelixSeek_ATTENTION_CLASSES
from .cache import HybridMambaAttentionDynamicCache


logger = logging.getLogger(__name__)


class HelixSeekTransformerDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.self_attn = HelixSeek_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx=-1
        )
        self.feed_forward = HelixSeekMLP(config)
        self.input_layernorm = HelixSeekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_ff_layernorm = HelixSeekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cache_layer_idx: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input of shape `(batch, seq_len, embed_dim)`
            layer_idx (int): layer index for identification
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Cache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*`):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*`):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids,
            layer_idx=layer_idx,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_layer_idx=cache_layer_idx,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # if router_logits is not None:
        #     outputs += (router_logits,)

        return outputs
