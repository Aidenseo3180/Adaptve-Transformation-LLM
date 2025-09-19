"""Layer Collapse (LaCo) - Merge multiple layers into one while preserving differences

Based on the LaCo paper's concept of Reserving-Differences-while-Seeking-Common.
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all,
                    select_non_overlapping_blocks)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def extract_layer_weights(model, start_idx: int, end_idx: int) -> Tuple[List, List]:
    """Extract weights from layers to be collapsed"""
    
    print(f"[DEBUG] Extracting weights from layers {start_idx} to {end_idx}")
    
    attention_weights = []
    ffn_weights = []
    
    for idx in range(start_idx, end_idx):
        layer = model.model.layers[idx]
        
        # Extract attention weights
        attn_dict = {
            'q_proj': layer.self_attn.q_proj.weight.data.clone(),
            'k_proj': layer.self_attn.k_proj.weight.data.clone(),
            'v_proj': layer.self_attn.v_proj.weight.data.clone(),
            'o_proj': layer.self_attn.o_proj.weight.data.clone(),
        }
        attention_weights.append(attn_dict)
        
        # Extract FFN weights
        ffn_dict = {
            'gate_proj': layer.mlp.gate_proj.weight.data.clone(),
            'up_proj': layer.mlp.up_proj.weight.data.clone(),
            'down_proj': layer.mlp.down_proj.weight.data.clone(),
        }
        ffn_weights.append(ffn_dict)
    
    print(f"[DEBUG] Extracted {len(attention_weights)} attention layers and {len(ffn_weights)} FFN layers")
    
    return attention_weights, ffn_weights


class CollapsedLayer(nn.Module):
    """Single layer that represents multiple collapsed layers"""
    
    def __init__(self, 
                 config,
                 attention_weights: List,
                 ffn_weights: List,
                 preserve_ratio: float = 0.1,
                 dtype=torch.bfloat16):
        super().__init__()
        
        self.num_collapsed = len(attention_weights)
        print(f"[DEBUG] Creating CollapsedLayer from {self.num_collapsed} layers")
        
        # Compute common components (average)
        self.init_common_weights(attention_weights, ffn_weights, dtype)
        
        # Store differences (residuals) - this is the key innovation
        self.init_difference_weights(attention_weights, ffn_weights, preserve_ratio, dtype)
        
        # Layer norms (use first layer's)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, dtype=dtype)
        
        # Progressive mixing weights
        self.layer_weights = nn.Parameter(
            torch.ones(self.num_collapsed, dtype=dtype) / self.num_collapsed
        )
        
        print(f"[DEBUG] CollapsedLayer initialized with preserve_ratio={preserve_ratio}")
    
    def init_common_weights(self, attention_weights, ffn_weights, dtype):
        """Initialize common (averaged) weights"""
        
        # Average attention weights
        avg_attn = {}
        for key in attention_weights[0].keys():
            stacked = torch.stack([w[key] for w in attention_weights])
            avg_attn[key] = stacked.mean(dim=0).to(dtype)
        
        # Create attention projection layers
        hidden_size = avg_attn['q_proj'].shape[1]
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        
        self.q_proj.weight.data = avg_attn['q_proj']
        self.k_proj.weight.data = avg_attn['k_proj']
        self.v_proj.weight.data = avg_attn['v_proj']
        self.o_proj.weight.data = avg_attn['o_proj']
        
        # Average FFN weights
        avg_ffn = {}
        for key in ffn_weights[0].keys():
            stacked = torch.stack([w[key] for w in ffn_weights])
            avg_ffn[key] = stacked.mean(dim=0).to(dtype)
        
        # Create FFN layers
        intermediate_size = avg_ffn['gate_proj'].shape[0]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        
        self.gate_proj.weight.data = avg_ffn['gate_proj']
        self.up_proj.weight.data = avg_ffn['up_proj']
        self.down_proj.weight.data = avg_ffn['down_proj']
    
    def init_difference_weights(self, attention_weights, ffn_weights, preserve_ratio, dtype):
        """Initialize difference (residual) weights"""
        
        if preserve_ratio <= 0:
            self.use_differences = False
            return
        
        self.use_differences = True
        self.preserve_ratio = preserve_ratio
        
        # Compute differences from average for each layer
        self.attn_diffs = nn.ParameterList()
        self.ffn_diffs = nn.ParameterList()
        
        for i in range(self.num_collapsed):
            # Attention differences (compressed)
            attn_diff = {}
            q_diff = attention_weights[i]['q_proj'] - self.q_proj.weight.data
            
            # Use low-rank approximation to compress differences
            rank = max(1, int(q_diff.shape[0] * preserve_ratio))
            U, S, V = torch.svd_lowrank(q_diff, q=rank)
            
            # Store low-rank factors
            self.attn_diffs.append(nn.ParameterList([
                nn.Parameter(U.to(dtype)),
                nn.Parameter(S.to(dtype)),
                nn.Parameter(V.t().to(dtype))
            ]))
            
            # Similarly for FFN (simplified for now)
            down_diff = ffn_weights[i]['down_proj'] - self.down_proj.weight.data
            self.ffn_diffs.append(nn.Parameter(down_diff.to(dtype) * preserve_ratio))
        
        print(f"[DEBUG] Initialized {len(self.attn_diffs)} difference components")
    
    def forward(self, hidden_states, layer_idx: Optional[int] = None):
        """Forward pass through collapsed layer"""
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Compute attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Apply differences if specified
        if self.use_differences and layer_idx is not None and layer_idx < self.num_collapsed:
            # Add low-rank correction
            U, S, Vt = self.attn_diffs[layer_idx]
            correction = hidden_states @ Vt.t() @ torch.diag(S) @ U.t()
            q = q + correction * self.layer_weights[layer_idx]
        
        # Simplified attention computation
        batch_size, seq_len, hidden_dim = q.shape
        q = q.view(batch_size, seq_len, -1, hidden_dim // 32).transpose(1, 2)
        k = k.view(batch_size, seq_len, -1, hidden_dim // 32).transpose(1, 2)
        v = v.view(batch_size, seq_len, -1, hidden_dim // 32).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        down = self.down_proj(gate * up)
        
        # Apply FFN differences
        if self.use_differences and layer_idx is not None and layer_idx < self.num_collapsed:
            down = down + F.linear(gate * up, self.ffn_diffs[layer_idx])
        
        hidden_states = residual + down
        
        return hidden_states


def layer_collapse_compress(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "train",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    # Layer Collapse specific
    preserve_ratio: float = 0.1,
    fine_tune_epochs: int = 5,
    learning_rate: float = 1e-5,
    **kwargs
) -> str:
    """Layer Collapse compression"""
    
    print(f"\n{'='*60}")
    print(f"Starting Layer Collapse (LaCo)")
    print(f"Collapsing layers {start_id} to {end_id}")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logging.info(f"{Fore.GREEN}Loading model for Layer Collapse...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",  # Load on CPU for weight extraction
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract weights from layers to collapse
    attention_weights, ffn_weights = extract_layer_weights(
        model, 
        start_id - num_layer, 
        end_id - num_layer
    )
    
    # Create collapsed layer
    logging.info(f"{Fore.YELLOW}Creating collapsed layer...{Fore.RESET}")
    
    collapsed_layer = CollapsedLayer(
        model.config,
        attention_weights,
        ffn_weights,
        preserve_ratio=preserve_ratio,
        dtype=torch.bfloat16
    )
    
    print(f"[DEBUG] Collapsed layer created with {collapsed_layer.num_collapsed} layers merged")
    
    # Replace layers with collapsed version
    print(f"[DEBUG] Replacing layers {start_id - num_layer} to {end_id - num_layer}")
    
    # Create a wrapper that applies the collapsed layer multiple times
    class CollapsedLayerWrapper(nn.Module):
        def __init__(self, collapsed_layer, num_applications):
            super().__init__()
            self.collapsed_layer = collapsed_layer
            self.num_applications = num_applications
            
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                   past_key_value=None, output_attentions=False, use_cache=False,
                   cache_position=None, **kwargs):
            
            # Apply collapsed layer with different residuals
            for i in range(self.num_applications):
                hidden_states = self.collapsed_layer(hidden_states, layer_idx=i)
            
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (None,)
            
            return outputs
    
    # Replace first layer with collapsed version
    model.model.layers[start_id - num_layer - 1] = CollapsedLayerWrapper(
        collapsed_layer,
        end_id - start_id
    )
    
    # Remove the collapsed layers
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Optional fine-tuning
    if fine_tune_epochs > 0:
        logging.info(f"{Fore.CYAN}Fine-tuning collapsed layer...{Fore.RESET}")
        
        model = model.to(device)
        dataloader = get_calib_dataloader(
            dataset, dataset_subset, dataset_column,
            min(dataset_size, 500), batch_size, tokenizer
        )
        
        # Only optimize the collapsed layer parameters
        optimizer = torch.optim.AdamW(
            collapsed_layer.parameters(),
            lr=learning_rate
        )
        
        for epoch in range(fine_tune_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_text in tqdm(list(dataloader)[:20], desc=f"Epoch {epoch+1}/{fine_tune_epochs}"):
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(collapsed_layer.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"[DEBUG] Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        model = model.cpu()
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_LaCo"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}")
    tokenizer.save_pretrained(f"{save_path}")
    
    # Save collapse info
    torch.save({
        'preserve_ratio': preserve_ratio,
        'num_collapsed': collapsed_layer.num_collapsed,
        'layer_weights': collapsed_layer.layer_weights.data
    }, f"{save_path}/laco_info.pth")
    
    logging.info(f"{Fore.GREEN}Model saved to {save_path}{Fore.RESET}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path