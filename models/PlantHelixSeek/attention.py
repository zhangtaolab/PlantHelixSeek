import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from typing import Optional, Tuple, Any
from collections.abc import Callable
from einops import rearrange, repeat


class PytorchCausalConv1d(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        W = self.kernel_size
        is_decoding = (B * T == N) and (T == 1)

        if is_decoding:
            if output_final_state and cache is None:
                cache = x.new_zeros(N, D, W)
            if cache is None:
                cache = x.new_zeros(N, D, W)
            cache = cache.roll(shifts=-1, dims=-1)
            cache[:, :, -1] = x.squeeze(1) if cu_seqlens is None else x.squeeze(0)
            w = self.weight.squeeze(1)
            y = torch.einsum("ndw,ndw->nd", cache, w.unsqueeze(0).expand(N, -1, -1))
            if self.bias is not None:
                y = y + self.bias
            if self.activation == "silu":
                y = F.silu(y)
            y = y.unsqueeze(1) if cu_seqlens is None else y.unsqueeze(0)
            return y, cache

        x_t = x.transpose(1, 2)
        y = F.conv1d(x_t, self.weight, self.bias, padding=W - 1, groups=D)
        y = y[:, :, :T]

        if self.activation == "silu":
            y = F.silu(y)

        y = y.transpose(1, 2)

        if output_final_state or cache is not None:
            if cache is None:
                cache = x.new_zeros(N, D, W)
            if cu_seqlens is not None:
                for i in range(N):
                    bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    seq_len = eos - bos
                    src = x_t[i]
                    avail = min(seq_len, W)
                    cache[i] = torch.cat([
                        cache[i, :, : max(0, W - avail)],
                        src[:, max(0, seq_len - W): seq_len],
                    ], dim=1)
            else:
                avail = min(T, W)
                cache = torch.cat([
                    cache[:, :, : max(0, W - avail)],
                    x_t[:, :, max(0, T - W): T],
                ], dim=2)

        return y, cache

class PytorchRMSNormGated(nn.Module):
    """Pure PyTorch implementation of FLA's FusedRMSNormGated.

    Computes: sigmoid(gate) * (weight * rms_norm(x))

    This replaces FusedRMSNormGated on platforms where FLA is unavailable
    (e.g., Huawei NPU), ensuring identical mathematical semantics so that
    weights trained with FLA produce consistent results.

    Args:
        hidden_size: Dimension of the input.
        eps: Epsilon for numerical stability in RMS norm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid-gated RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).
            gate: Gate tensor of shape (..., hidden_size).

        Returns:
            sigmoid(gate) * (weight * rms_norm(x))
        """
        input_dtype = x.dtype
        gate = torch.sigmoid(gate)
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (gate * (self.weight * x)).to(input_dtype)


from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    is_flash_attn_2_available,
    logging,
)
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)

from .utils import rotate_half, apply_rotary_pos_emb, repeat_kv, _get_unpad_data

from .cache import HybridMambaAttentionDynamicCache

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )

try:
    from fla.ops.kda.chunk import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda
    from fla.modules import FusedRMSNormGated

    FLA_AVAILABLE = True
except Exception:
    FLA_AVAILABLE = False


class HelixSeekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HelixSeekRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HelixSeekRotaryEmbedding(nn.Module):
    def __init__(
        self, config, dim, max_position_embeddings=4096, base=10000, device=None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        if config.use_long_context:
            a = 8  # Alpha value
            base = base * a ** (dim / (dim - 2))  # Base change formula
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class HelixSeekAttention(nn.Module):
    def __init__(self, config, layer_idx=None, num_mem_blocks=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.rope_theta = config.rope_theta

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = config.model_mode == "causal"
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        if config.use_mem_rope:
            self.rotary_emb = HelixSeekRotaryEmbedding(
                config,
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=self.rope_theta,
            )

        # 构建layer_block_map: 将层名称中的索引映射到连续的缓存索引
        # 例如: Transformer_1 -> 0, Transformer_2 -> 1
        self.layer_block_map = {}
        cache_idx = 0
        seen_layers = set()
        for layer_name in config.layers_block_type:
            if layer_name.startswith("Transformer"):
                # 从层名称中提取索引,例如 "Transformer_1" -> 1
                name_idx = int(layer_name.split("_")[1])
                if name_idx not in seen_layers:
                    self.layer_block_map[name_idx] = cache_idx
                    seen_layers.add(name_idx)
                    cache_idx += 1

    def _project_qkv(
        self, hidden_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self._project_qkv(
            hidden_states, layer_idx
        )

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class HelixSeekFlashAttention2(HelixSeekAttention):
    """
    HelixSeek flash attention module. This module inherits from `HelixSeekAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        cache_layer_idx: Optional[int] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self._project_qkv(
            hidden_states, layer_idx
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            # Use model-wide layer index for cache slot to avoid collisions
            # with other layer types (e.g. HelixSeekMLA) that also use model-wide indices
            cache_idx = cache_layer_idx if cache_layer_idx is not None else layer_idx
            key_states, value_states = past_key_value.update(
                key_states, value_states, cache_idx
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        softmax_scale = 1 / query_states.shape[-1] ** 0.5

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=softmax_scale,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

        return attn_output

    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2._upad_input
    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
# added scale = 1 / (query_states.shape[-1]/2)**0.5 to the arguments of torch.nn.functional.scaled_dot_product_attention
class HelixSeekSdpaAttention(HelixSeekAttention):
    """
    HelixSeek attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `HelixSeekAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        cache_layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "HelixSeekModel is using HelixSeekSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_layer_idx=cache_layer_idx,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self._project_qkv(
            hidden_states, layer_idx
        )

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            cache_idx = cache_layer_idx if cache_layer_idx is not None else layer_idx
            key_states, value_states = past_key_value.update(
                key_states, value_states, cache_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            # NPU fused SDPA needs a bool mask; a float additive mask silently falls
            # back to math and materializes the full [B,H,N,N] matrix (O(seq^2) HBM).
            if query_states.device.type == "npu" and causal_mask.dtype.is_floating_point:
                causal_mask = causal_mask >= 0

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        softmax_scale = 1 / query_states.shape[-1] ** 0.5

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
            scale=softmax_scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


HelixSeek_ATTENTION_CLASSES = {
    "eager": HelixSeekAttention,
    "flash_attention_2": HelixSeekFlashAttention2,
    "sdpa": HelixSeekSdpaAttention,
}


def index_first_axis(x, indices):
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"),
        0,
        repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def index_put_first_axis(x, indices, first_axis_dim):
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    y[indices] = x
    return y


def get_unpad_data(attention_mask: torch.Tensor):
    lens = torch.sum(attention_mask != 0, dim=1)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item() if lens.numel() > 0 else 0
    cu_seqlens = F.pad(lens, pad=(1, 0), mode="constant")
    cu_seqlens = torch.cumsum(cu_seqlens, dim=0)
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: tuple,
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError(
            "We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)"
        )

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


class HelixSeekMLAAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.rope_theta = config.rope_theta
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", None)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", None)
        self.v_head_dim = getattr(config, "v_head_dim", None)
        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", None)
        self.q_head_dim = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
            if self.qk_nope_head_dim and self.qk_rope_head_dim
            else self.hidden_size // self.num_heads
        )
        self.use_nope = getattr(config, "mla_use_nope", False)
        self.scaling = (
            self.q_head_dim ** (-0.5)
            if self.qk_nope_head_dim
            else self.hidden_size ** (-0.5)
        )

        if self.q_lora_rank is not None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim
            if self.kv_lora_rank and self.qk_rope_head_dim
            else self.hidden_size,
            bias=False,
        )

        if self.kv_lora_rank:
            self.kv_a_layernorm = HelixSeekRMSNorm(self.kv_lora_rank)
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_heads
                * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim)
                if self.use_nope
                else self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.kv_a_layernorm = None
            self.kv_b_proj = None

        self.o_proj = nn.Linear(
            self.num_heads
            * (
                self.v_head_dim
                if self.v_head_dim
                else self.hidden_size // self.num_heads
            ),
            self.hidden_size,
            bias=False,
        )
        self.is_causal = config.model_mode == "causal"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim
            if self.use_nope
            else self.q_head_dim,
        )

        q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)

        if self.qk_nope_head_dim and self.qk_rope_head_dim:
            q_pass, q_rot = torch.split(
                q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
        else:
            q_pass, q_rot = q_states, None

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        if self.kv_lora_rank and self.qk_rope_head_dim:
            k_pass, k_rot = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pass = (
                self.kv_b_proj(self.kv_a_layernorm(k_pass))
                .view(key_shape)
                .transpose(1, 2)
            )
            if self.use_nope:
                k_pass, value_states = torch.split(
                    k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )
            else:
                value_states = k_pass
                k_pass = None
            k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
            k_rot = (
                k_rot.expand(*k_pass.shape[:-1], -1)
                if k_pass is not None
                else k_rot.expand(batch_size, 1, seq_length, self.qk_rope_head_dim)
            )
        else:
            k_rot = None
            k_pass = None
            value_states = compressed_kv

        if self.use_nope and q_pass is not None and k_pass is not None:
            query_states = (
                torch.cat((q_pass, q_rot), dim=-1) if q_rot is not None else q_pass
            )
            key_states = (
                torch.cat((k_pass, k_rot), dim=-1) if k_rot is not None else k_pass
            )
        else:
            query_states = q_states
            key_states = compressed_kv

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if (
            self.config._attn_implementation == "flash_attention_2"
            and hasattr(self, "v_head_dim")
            and self.q_head_dim != self.v_head_dim
        ):
            value_states = F.pad(
                value_states, [0, self.q_head_dim - value_states.shape[-1]]
            )

        attn_output, _ = self._attention_forward(
            query_states, key_states, value_states, attention_mask
        )

        if (
            self.config._attn_implementation == "flash_attention_2"
            and hasattr(self, "v_head_dim")
            and self.q_head_dim != self.v_head_dim
        ):
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _attention_forward(self, query, key, value, attention_mask):
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, : key.shape[-2]]
            # NPU fused SDPA needs a bool mask; a float additive mask silently falls
            # back to math and materializes the full [B,H,N,N] matrix (O(seq^2) HBM).
            if query.device.type == "npu" and attn_mask.dtype.is_floating_point:
                attn_mask = attn_mask >= 0

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, None


class HelixSeekDeltaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.head_dim = (
            getattr(config, "linear_attn_config", {}).get("head_dim", 64)
            if isinstance(getattr(config, "linear_attn_config", None), dict)
            else 64
        )
        self.num_heads = (
            getattr(config, "linear_attn_config", {}).get("num_heads", 8)
            if isinstance(getattr(config, "linear_attn_config", None), dict)
            else 8
        )
        self.conv_size = (
            getattr(config, "linear_attn_config", {}).get("short_conv_kernel_size", 4)
            if isinstance(getattr(config, "linear_attn_config", None), dict)
            else 4
        )
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        if FLA_AVAILABLE:
            try:
                from fla.modules import FusedRMSNormGated, ShortConvolution

                self.q_conv1d = ShortConvolution(
                    hidden_size=projection_k_size,
                    kernel_size=self.conv_size,
                    activation="silu",
                )
                self.k_conv1d = ShortConvolution(
                    hidden_size=projection_k_size,
                    kernel_size=self.conv_size,
                    activation="silu",
                )
                self.v_conv1d = ShortConvolution(
                    hidden_size=projection_size,
                    kernel_size=self.conv_size,
                    activation="silu",
                )
            except Exception:
                self.q_conv1d = PytorchCausalConv1d(
                    hidden_size=projection_k_size,
                    kernel_size=self.conv_size,
                    activation="silu",
                )
                self.k_conv1d = PytorchCausalConv1d(
                    hidden_size=projection_k_size,
                    kernel_size=self.conv_size,
                    activation="silu",
                )
                self.v_conv1d = PytorchCausalConv1d(
                    hidden_size=projection_size,
                    kernel_size=self.conv_size,
                    activation="silu",
                )
        else:
            self.q_conv1d = PytorchCausalConv1d(
                hidden_size=projection_k_size,
                kernel_size=self.conv_size,
                activation="silu",
            )
            self.k_conv1d = PytorchCausalConv1d(
                hidden_size=projection_k_size,
                kernel_size=self.conv_size,
                activation="silu",
            )
            self.v_conv1d = PytorchCausalConv1d(
                hidden_size=projection_size,
                kernel_size=self.conv_size,
                activation="silu",
            )

        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            ).view(1, 1, self.num_heads, 1)
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        nn.init.zeros_(self.dt_bias)

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = (
            FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
            if FLA_AVAILABLE
            else PytorchRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")
            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len]"
                )

        use_cache = past_key_values is not None
        batch_size, q_len, _ = hidden_states.shape
        # fused_recurrent mode has no backward pass, so only use it during inference
        if hidden_states.device.type == "npu":
            mode = "fused_recurrent"
        else:
            mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if past_key_values is not None:
            if (
                hasattr(past_key_values, "conv_states")
                and past_key_values.conv_states[self.layer_idx] is not None
            ):
                conv_state_q, conv_state_k, conv_state_v = past_key_values.conv_states[
                    self.layer_idx
                ]
            if hasattr(past_key_values, "recurrent_states"):
                recurrent_state = past_key_values.recurrent_states[self.layer_idx]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if hasattr(self, "q_conv1d") and hasattr(self.q_conv1d, "kernel_size"):
            q, conv_state_q = self.q_conv1d(
                q, conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                k, conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                v, conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens
            )
            q = rearrange(
                q, "b s (h d) -> b h d s", h=self.num_k_heads, d=self.head_k_dim
            )
            k = rearrange(
                k, "b s (h d) -> b h d s", h=self.num_k_heads, d=self.head_k_dim
            )
            v = rearrange(v, "b s (h d) -> b h d s", h=self.num_heads, d=self.head_dim)
        else:
            q = rearrange(
                q, "b s (h d) -> b h d s", h=self.num_k_heads, d=self.head_k_dim
            )
            k = rearrange(
                k, "b s (h d) -> b h d s", h=self.num_k_heads, d=self.head_k_dim
            )
            v = rearrange(v, "b s (h d) -> b h d s", h=self.num_heads, d=self.head_dim)
            conv_state_q, conv_state_k, conv_state_v = None, None, None

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "b s (h d) -> b s h d", h=self.num_k_heads, d=self.head_k_dim)
        if FLA_AVAILABLE:
            g = fused_kda_gate(g, self.A_log, dt_bias=self.dt_bias)
        else:
            A_decay = torch.exp(-torch.exp(self.A_log.float()))
            dt = F.softplus(self.dt_bias.view(self.num_k_heads, self.head_k_dim))
            g = g * dt * A_decay
        beta = self.b_proj(hidden_states).float().sigmoid()

        if FLA_AVAILABLE:
            # FLA expects (B, T, H, K) format — rearrange from (B, H, K, T)
            q = rearrange(q, "b h d s -> b s h d")
            k = rearrange(k, "b h d s -> b s h d")
            v = rearrange(v, "b h d s -> b s h d")

            if mode == "chunk":
                o, recurrent_state = chunk_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                )
            else:
                o, recurrent_state = fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                )
        else:
            # Pure PyTorch fallback matching FLA KDA semantics
            q = rearrange(q, "b h d s -> b s h d")
            k = rearrange(k, "b h d s -> b s h d")
            v = rearrange(v, "b h d s -> b s h d")

            q_norm = F.normalize(q, p=2, dim=-1)
            k_norm = F.normalize(k, p=2, dim=-1)

            b_size, seq_len, n_heads, head_dim = q_norm.shape
            decay = g.sigmoid()

            if mode == "fused_recurrent" and recurrent_state is not None:
                # Single-token decode: update state incrementally
                kv_state = recurrent_state
                decay_scalar = decay[:, 0].mean(dim=-1, keepdim=True).unsqueeze(-1)
                kv_state = kv_state * decay_scalar

                k_t = k_norm[:, 0].unsqueeze(-1)
                v_t = v[:, 0].unsqueeze(-2)
                kv_state = kv_state + torch.matmul(k_t, v_t)

                o_t = torch.einsum("bhd,bhde->bhe", q_norm[:, 0], kv_state)
                o_t = o_t * beta[:, 0].unsqueeze(-1)
                o = o_t.unsqueeze(1)
                recurrent_state = kv_state
            else:
                # Parallel scan: o_t = sum_{k<=t} exp(cum_log[t]-cum_log[k]) * (q_t . k_k) * v_k
                # Equivalent to S_t = decay_t * S_{t-1} + k_t (x) v_t;  o_t = q_t^T @ S_t
                decay_scalar = decay.mean(dim=-1)

                log_decay = torch.log(decay_scalar.clamp(min=1e-20))
                cum_log_decay = torch.cumsum(log_decay, dim=1)
                cum_p = cum_log_decay.permute(0, 2, 1)

                exp_diff = (cum_p.unsqueeze(-1) - cum_p.unsqueeze(-2)).clamp(min=-20, max=0)
                decay_weights = torch.exp(exp_diff)

                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                decay_weights = decay_weights.masked_fill(causal_mask, 0.0)

                qk = torch.einsum("bshd,bthd->bhst", q_norm, k_norm)
                attn = decay_weights * qk
                o = torch.einsum("bhst,bthd->bshd", attn, v)

                if recurrent_state is not None:
                    state_decay = torch.exp(cum_log_decay.clamp(min=-20))
                    state_out = torch.einsum("bshd,bhde->bshe", q_norm, recurrent_state)
                    o = o + state_out * state_decay.unsqueeze(-1)

                o = o * beta.unsqueeze(-1)

                final_exp_diff = (cum_log_decay[:, -1:, :] - cum_log_decay).clamp(min=-20)
                final_decay = torch.exp(final_exp_diff)
                weighted_k = k_norm * final_decay.unsqueeze(-1)
                new_recurrent_state = torch.matmul(
                    weighted_k.permute(0, 2, 3, 1),
                    v.permute(0, 2, 1, 3),
                )

                if recurrent_state is not None:
                    new_recurrent_state = (
                        new_recurrent_state
                        + recurrent_state * torch.exp(cum_log_decay[:, -1:, :].clamp(min=-20).unsqueeze(-1))
                    )

                recurrent_state = new_recurrent_state

        if past_key_values is not None:
            if hasattr(past_key_values, "recurrent_states"):
                past_key_values.recurrent_states[self.layer_idx] = recurrent_state
            if hasattr(past_key_values, "conv_states"):
                past_key_values.conv_states[self.layer_idx] = (
                    conv_state_q,
                    conv_state_k,
                    conv_state_v,
                )

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, "b s h d -> b s (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


class HelixSeekDeltaMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = (
            ACT2FN[config.hidden_act]
            if hasattr(ACT2FN, config.hidden_act)
            else nn.GELU()
        )

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class HelixSeekDeltaMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = getattr(config, "num_experts_per_tok", 1)
        self.num_experts = getattr(config, "num_experts", 1)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.moe_router_activation_func = getattr(
            config, "moe_router_activation_func", "sigmoid"
        )
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = getattr(config, "moe_renormalize", True)
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, return_router_logits=False):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)
        logits = F.linear(hidden_states_flat.float(), self.weight.float(), None)

        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            scores = logits.sigmoid()

        scores_for_choice = scores.view(bsz * seq_len, -1)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len,
                self.num_expert_group,
                self.num_experts // self.num_expert_group,
            )
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        if return_router_logits:
            return topk_idx, topk_weight, logits.view(bsz, seq_len, -1)
        return topk_idx, topk_weight


class HelixSeekDeltaSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = getattr(config, "num_experts", 1)
        self.top_k = getattr(config, "num_experts_per_tok", 1)
        self.moe_renormalize = getattr(config, "moe_renormalize", True)
        self.ep_size = 1
        self.experts_per_rank = self.num_experts
        self.ep_rank = 0
        # Get intermediate size safely
        moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
        if moe_intermediate_size is None:
            if config.hidden_size is not None:
                moe_intermediate_size = 4 * config.hidden_size
            else:
                moe_intermediate_size = 2048  # Fallback default

        self.experts = nn.ModuleList(
            [
                HelixSeekDeltaMLP(
                    config,
                    intermediate_size=moe_intermediate_size,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate = HelixSeekDeltaMoEGate(config)

        # Add shared experts mechanism
        if getattr(config, "num_shared_experts", None) is not None:
            shared_intermediate_size = getattr(
                config, "moe_intermediate_size", moe_intermediate_size
            )
            intermediate_size = shared_intermediate_size * config.num_shared_experts
            self.shared_experts = HelixSeekDeltaMLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states, output_router_logits=False):
        identity = hidden_states
        orig_shape = hidden_states.shape

        if output_router_logits:
            topk_idx, topk_weight, router_logits = self.gate(
                hidden_states, return_router_logits=True
            )
        else:
            topk_idx, topk_weight = self.gate(hidden_states, return_router_logits=False)
            router_logits = None

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

        # Add shared experts output if configured
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity)

        if output_router_logits:
            return y, router_logits
        return y

    def moe_infer(self, x, topk_ids, topk_weight):
        num_tokens, num_experts = topk_ids.shape[1], len(self.experts)

        cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        outputs = []
        start_idx = 0
        tokens_per_expert_list = tokens_per_expert.tolist()
        for i, num_tokens in enumerate(tokens_per_expert_list):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        if len(outputs) == 0:
            final_out = torch.zeros(
                (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            )
        else:
            outs = torch.cat(outputs, dim=0)
            new_x = torch.empty_like(outs)
            new_x[idxs] = outs
            final_out = (
                new_x.view(*topk_ids.shape, -1)
                .type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(dim=-1))
                .sum(dim=1)
                .type(new_x.dtype)
            )
        return final_out
