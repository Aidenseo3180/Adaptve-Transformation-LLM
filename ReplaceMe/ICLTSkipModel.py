
# ICLTSkipModel.py
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from .ICLTAdapter import ICLTAdapter

class ICLTSkipModel(nn.Module):
    def __init__(self, base_model, adapter_dir, K, rank, start_idx=3, end_idx=8):
        super().__init__()
        self.base = base_model
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_layers = base_model.config.num_hidden_layers
        self.hidden_size = base_model.config.hidden_size

        self.adapter = ICLTAdapter(
            U_dir=adapter_dir,
            d_model=self.hidden_size,
            K=K,
            rank=rank,
            device=next(base_model.parameters()).device,
            dtype=next(base_model.parameters()).dtype
        )

        self.embed_tokens = base_model.model.embed_tokens
        self.embed_positions = getattr(base_model.model, "embed_positions", None)
        self.layers = base_model.model.layers
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        hidden_states = self.embed_tokens(input_ids)

        if self.embed_positions is not None:
            hidden_states += self.embed_positions(input_ids)

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(input_ids.shape[0], -1)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -1e9

        for i in range(len(self.layers)):
            if i == self.start_idx:
                out = self.layers[i](hidden_states, attention_mask=attention_mask, position_ids=position_ids)
                hidden_states = out[0] if isinstance(out, tuple) else out
                hidden_states = self.adapter(hidden_states)
                i = self.end_idx - 1
            elif self.start_idx < i < self.end_idx:
                continue
            else:
                out = self.layers[i](hidden_states, attention_mask=attention_mask, position_ids=position_ids)
                hidden_states = out[0] if isinstance(out, tuple) else out

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits)
