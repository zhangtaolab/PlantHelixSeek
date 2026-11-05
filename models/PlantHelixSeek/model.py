import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MaskedLMOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from .configuration import HelixSeekConfig
from .cache import HybridMambaAttentionDynamicCache, HelixSeekDeltaDynamicCache
from .attention import (
    HelixSeekRMSNorm,
    HelixSeekMLAAttention,
    HelixSeekDeltaAttention,
    HelixSeekDeltaSparseMoeBlock,
    HelixSeekDeltaMLP,
)
from .layers import (
    HelixSeekTransformerDecoderLayer,
)
from .utils import load_balancing_loss_func, count_mem_blocks_in_config, layer_type_list

logger = logging.get_logger(__name__)


class HelixSeekPreTrainedModel(PreTrainedModel):
    config_class = HelixSeekConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "HelixSeekTransformerDecoderLayer",
        "HelixSeekDeltaDecoderLayer",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HelixSeekDeltaDecoderLayer(nn.Module):
    def __init__(self, config: HelixSeekConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        layer_type = config.get_layer_type(layer_idx)

        if layer_type == "HelixSeekDelta":
            self.is_linear_attn = True
            self.self_attn = HelixSeekDeltaAttention(config=config, layer_idx=layer_idx)
        elif layer_type == "HelixSeekMLA":
            self.is_linear_attn = False
            self.self_attn = HelixSeekMLAAttention(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unsupported HelixSeekDelta layer type: {layer_type}")

        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = HelixSeekDeltaSparseMoeBlock(config)
        else:
            self.mlp = HelixSeekDeltaMLP(config)

        self.input_layernorm = HelixSeekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = HelixSeekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: bool = False,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor,
        tuple[torch.FloatTensor, torch.FloatTensor] | None,
        torch.Tensor | None,
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        router_logits = None
        if hasattr(self, "block_sparse_moe"):
            if output_router_logits:
                hidden_states, router_logits = self.block_sparse_moe(
                    hidden_states, output_router_logits=True
                )
            else:
                hidden_states = self.block_sparse_moe(
                    hidden_states, output_router_logits=False
                )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_router_logits:
            return hidden_states, router_logits
        return hidden_states


class HelixSeekModel(HelixSeekPreTrainedModel):
    def __init__(self, config: HelixSeekConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if config.use_long_context:
            logger.warning_once(
                f"`use_long_context` has been set to True, therefore `max_position_embeddings` will be set to 16384."
            )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        layers_modules = []

        for i in range(len(config.layers_block_type)):
            name = config.layers_block_type[i]
            layer_idx = int(name.split("_")[1])
            layer_type = config.get_layer_type(i)
            if layer_type == "HelixSeekDelta":
                layer = HelixSeekDeltaDecoderLayer(config, layer_idx=i)
            elif layer_type == "HelixSeekMLA":
                layer = HelixSeekDeltaDecoderLayer(config, layer_idx=i)
            elif layer_type == "Transformer":
                layer = HelixSeekTransformerDecoderLayer(config)
            elif layer_type == "Linear":
                layer = nn.Linear(
                    self.config.hidden_size, self.config.hidden_size, bias=False
                )
            else:
                raise ValueError(f"Invalid layer name: {name}")
            layers_modules.append(layer)
        # print(layers_modules)

        self.layers = nn.ModuleList(layers_modules)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = HelixSeekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config.model_mode == "mlm":
            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            if attention_mask is None:
                return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            target_length = cache_position[-1] + 1 if cache_position is not None else sequence_length

            # Bidirectional mask (all positions visible to each other)
            attention_bias = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )
            attention_bias = attention_bias[None, None, :, :]

            # Apply padding mask only
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
                if mask_len < target_length:
                    padding = torch.zeros(
                        (attention_mask.shape[0], target_length - mask_len),
                        dtype=attention_mask.dtype, device=device,
                    )
                    attention_mask_full = torch.cat([attention_mask, padding], dim=-1)
                else:
                    attention_mask_full = attention_mask[:, :target_length]

                attention_mask_expanded = (
                    (1.0 - attention_mask_full[:, None, None, :].float()) * min_dtype
                ).to(dtype)
                attention_bias = attention_bias + attention_mask_expanded

            attention_bias = attention_bias.expand(input_tensor.shape[0], 1, -1, -1)

            return attention_bias

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )

        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length
                ].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            logger.warning_once(
                "HelixSeek requires an initialized cache to return a cache. None was "
                "provided, so no cache will be returned."
            )

        if cache_position is None:
            cache_position = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for i, layer in enumerate(self.layers):
            layer_name = self.config.layers_block_type[i]
            layer_type = self.config.get_layer_type(i)
            layer_idx = int(layer_name.split("_")[1]) if "_" in layer_name else i
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # print(layer.__class__.__name__)

            if layer.__class__.__name__ == "HelixSeekTransformerDecoderLayer":
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        layer_idx,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        i,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        layer_idx=layer_idx,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        cache_layer_idx=i,
                    )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    if layer_outputs[1] is not None:
                        all_self_attns += (layer_outputs[1],)

            if layer.__class__.__name__ == "Linear":
                if self.gradient_checkpointing and self.training:
                    hidden_states = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                    )
                else:
                    hidden_states = layer(
                        hidden_states,
                    )

            if layer.__class__.__name__ == "HelixSeekDeltaDecoderLayer":
                is_linear_attn = getattr(layer, "is_linear_attn", False)
                if self.gradient_checkpointing and self.training:
                    if is_linear_attn:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            past_key_values,
                            output_router_logits,
                        )
                    else:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            causal_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                            output_router_logits,
                        )
                else:
                    if is_linear_attn:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            output_router_logits=output_router_logits,
                        )
                    else:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            output_router_logits=output_router_logits,
                        )

                if output_router_logits and isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    if layer_outputs[1] is not None:
                        all_router_logits += (layer_outputs[1],)
                else:
                    hidden_states = layer_outputs

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if (
            past_key_values
            and hasattr(past_key_values, "has_previous_state")
            and not past_key_values.has_previous_state
        ):
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class HelixSeekForCausalLM(HelixSeekPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HelixSeekConfig):
        super().__init__(config)
        self.model = HelixSeekModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def _get_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        device=None,
        dtype=None,
    ):
        return HybridMambaAttentionDynamicCache(
            self.config,
            batch_size,
            dtype=dtype or self.dtype,
            device=device or self.device,
        )

    def _supports_default_dynamic_cache(self):
        return False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: Optional[Union[int, None]] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if num_logits_to_keep is None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -num_logits_to_keep:, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if not empty_past_kv:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            from .cache import HybridMambaAttentionDynamicCache

            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {
                "input_ids": input_ids.contiguous()
            }  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "num_logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs


class HelixSeekForMaskedLM(HelixSeekPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HelixSeekConfig):
        super().__init__(config)
        self.model = HelixSeekModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: Optional[Union[int, None]] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if num_logits_to_keep is None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -num_logits_to_keep:, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            logits_view = logits.view(-1, self.config.vocab_size)
            labels_view = labels.view(-1)

            active_labels = labels_view != -100
            active_logits = logits_view[active_labels]
            active_labels = labels_view[active_labels]

            if active_labels.sum() == 0:
                loss = torch.zeros(
                    (), device=logits.device, dtype=logits.dtype, requires_grad=True
                )
            else:
                loss = loss_fct(active_logits, active_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        output = MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # Attach router_logits for MoE debugging (not in standard MaskedLMOutput)
        if output_router_logits and return_dict:
            output.router_logits = outputs.router_logits if hasattr(outputs, "router_logits") else None
        return output


class HelixSeekForSequenceClassification(HelixSeekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = HelixSeekModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )

        if self.config.classification_pooling == "last":
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1)
                        - 1
                    )
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]

        elif self.config.classification_pooling == "cls":
            cls_token_id = getattr(self.config, "cls_token_id", 3)
            if input_ids is not None:
                cls_positions = (input_ids == cls_token_id).int().argmax(-1)
                cls_positions = cls_positions.to(logits.device)
                pooled_logits = logits[
                    torch.arange(batch_size, device=logits.device), cls_positions
                ]
            else:
                raise ValueError("CLS pooling requires input_ids")

        elif self.config.classification_pooling == "mean":
            attention_mask_expanded = None
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled_hidden_states = (hidden_states * attention_mask_expanded).sum(
                    dim=1
                )
                pooled_hidden_states = (
                    pooled_hidden_states / attention_mask_expanded.sum(dim=1)
                )
            else:
                pooled_hidden_states = hidden_states.mean(dim=1)
            pooled_logits = self.score(pooled_hidden_states)

        elif self.config.classification_pooling == "max":
            attention_mask_expanded = None
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_hidden_states = hidden_states.masked_fill(
                    attention_mask_expanded == 0, float("-inf")
                )
                pooled_hidden_states = masked_hidden_states.max(dim=1)[0]
            else:
                pooled_hidden_states = hidden_states.max(dim=1)[0]
            pooled_logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class HelixSeekForTokenClassification(HelixSeekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, "num_labels") else 2
        self.model = HelixSeekModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout
            if hasattr(config, "classifier_dropout")
            else config.hidden_dropout_prob
            if hasattr(config, "hidden_dropout_prob")
            else 0.1
        )
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
