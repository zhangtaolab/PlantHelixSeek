import torch
from typing import Optional, Tuple, Dict, Any, List


class HybridMambaAttentionDynamicCache:
    def __init__(self, config, batch_size, dtype=torch.bfloat16, device=None):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.device = device
        self.has_previous_state = False
        self.max_batch_size = batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0

        self.transformer_layers = []
        for i in range(len(self.layers_block_type)):
            name = self.layers_block_type[i]
            layer_type = name.split("_")[0] if "_" in name else name
            if layer_type in ("Transformer", "HelixSeekMLA"):
                self.transformer_layers.append(i)

        num_layers = len(config.layers_block_type)
        self.conv_states = [None for _ in range(num_layers)]
        self.recurrent_states = [None for _ in range(num_layers)]
        self.key_cache = [None for _ in range(num_layers)]
        self.value_cache = [None for _ in range(num_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx_device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx_device)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx_device = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx_device),
                    k_conv.index_select(0, beam_idx_device),
                    v_conv.index_select(0, beam_idx_device),
                )

            if self.recurrent_states[layer_idx] is not None:
                device = self.recurrent_states[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx_device)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self.transformer_layers:
            if layer_idx is None or layer_idx not in self.transformer_layers:
                layer_idx = self.transformer_layers[0]
            if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
                return 0
            return self.key_cache[layer_idx].shape[-2]
        for states in (self.conv_states, self.recurrent_states):
            for s in states:
                if s is not None:
                    if isinstance(s, tuple):
                        s = s[0]
                    return s.shape[1] if s.dim() >= 2 else s.shape[0]
        return 0

    def get_max_length(self) -> Optional[int]:
        return None

    def __len__(self) -> int:
        return len(self.layers_block_type)

    def to_legacy_cache(self):
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None):
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")


class HelixSeekDeltaDynamicCache:
    is_compileable = False

    def __init__(self, config, batch_size=1, dtype=torch.bfloat16, device=None):
        self.config = config
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self._has_previous_state = False

        num_layers = len(config.layers_block_type)
        layer_types = []
        for i in range(num_layers):
            layer_name = config.layers_block_type[i] if hasattr(config, 'layers_block_type') else ""
            if layer_name.startswith("HelixSeekDelta"):
                layer_types.append("linear_attention")
            else:
                layer_types.append("full_attention")
        self.layer_types = layer_types

        self.transformer_layers = [
            i for i in range(num_layers) if self.layer_types[i] == "full_attention"
        ]

        linear_layers = [i for i in range(num_layers) if self.layer_types[i] == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else -1

        self.conv_states = [None for _ in range(num_layers)]
        self.recurrent_states = [None for _ in range(num_layers)]
        self.key_cache = [None for _ in range(num_layers)]
        self.value_cache = [None for _ in range(num_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx] = key_states if self.key_cache[layer_idx] is None else torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        self.value_cache[layer_idx] = value_states if self.value_cache[layer_idx] is None else torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx_device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx_device)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx_device = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx_device),
                    k_conv.index_select(0, beam_idx_device),
                    v_conv.index_select(0, beam_idx_device),
                )

            if self.recurrent_states[layer_idx] is not None:
                device = self.recurrent_states[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx_device)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if not self.transformer_layers:
            layer_idx = 0
        elif layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        if layer_idx is None:
            return 0
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        if self.last_linear_layer == -1:
            return False
        return self.conv_states[self.last_linear_layer] is not None
