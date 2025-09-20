"""Differential Transformer Blocks (DTB) implementation.

This module implements a novel approach where instead of computing full transformer outputs,
we only compute the 'delta' (change) between input and output, leveraging the high similarity
observed in deeper layers.
"""

import gc
import logging
import os
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

from .utils import get_calib_dataloader, truncate_model, seed_all

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class DeltaNetwork(nn.Module):
    """Lightweight network to compute delta (change) instead of full output."""
    
    def __init__(self, hidden_size: int, rank: int, use_attention: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.use_attention = use_attention
        
        if use_attention:
            # Low-rank attention-based delta
            self.query = nn.Linear(hidden_size, rank, bias=False)
            self.key = nn.Linear(hidden_size, rank, bias=False)
            self.value = nn.Linear(hidden_size, rank, bias=False)
            self.output = nn.Linear(rank, hidden_size, bias=False)
            print(f"[DTB] Created attention-based delta network with rank {rank}")
        else:
            # Simple low-rank projection
            self.down_proj = nn.Linear(hidden_size, rank, bias=False)
            self.activation = nn.GELU()
            self.up_proj = nn.Linear(rank, hidden_size, bias=False)
            print(f"[DTB] Created projection-based delta network with rank {rank}")
        
        # Initialize with small values to start with small deltas
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to produce small deltas initially."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                print(f"[DTB] Initialized {module} with small random values (std=0.01)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute delta to be added to input."""
        if self.use_attention:
            # Simplified attention mechanism
            batch_size, seq_len, _ = x.shape
            
            q = self.query(x)  # [B, S, rank]
            k = self.key(x)    # [B, S, rank] 
            v = self.value(x)  # [B, S, rank]
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.rank ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            delta = self.output(attn_output)
        else:
            # Simple MLP-based delta
            delta = self.down_proj(x)
            delta = self.activation(delta)
            delta = self.up_proj(delta)
        
        return delta


def analyze_layer_changes(
    model,
    dataloader,
    tokenizer,
    max_length: int,
    device: str
) -> List[float]:
    """Analyze how much each layer changes its input."""
    
    print(f"\n{Fore.YELLOW}[DTB] Analyzing layer-wise changes...{Fore.RESET}")
    
    layer_changes = [[] for _ in range(model.config.num_hidden_layers)]
    
    for batch in tqdm(dataloader, desc="Analyzing layers", colour="yellow"):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Compare input and output of each layer
            for i in range(len(hidden_states) - 1):
                input_hidden = hidden_states[i]
                output_hidden = hidden_states[i + 1]
                
                # Compute relative change
                change = torch.norm(output_hidden - input_hidden, dim=-1) / (torch.norm(input_hidden, dim=-1) + 1e-8)
                layer_changes[i].append(change.mean().item())
    
    # Average changes per layer
    avg_changes = [np.mean(changes) for changes in layer_changes]
    
    # Print analysis results
    print(f"\n{Fore.GREEN}[DTB] Layer Change Analysis:{Fore.RESET}")
    for i, change in enumerate(avg_changes):
        color = Fore.RED if change > 0.1 else Fore.YELLOW if change > 0.05 else Fore.GREEN
        print(f"  Layer {i+1}: {color}{change:.4f}{Fore.RESET}")
    
    return avg_changes


def compute_delta_ranks(
    layer_changes: List[float],
    min_rank: int = 4,
    max_rank: int = 128
) -> List[int]:
    """Compute appropriate rank for each layer's delta network."""
    
    print(f"\n{Fore.YELLOW}[DTB] Computing delta ranks...{Fore.RESET}")
    
    # Normalize changes to [0, 1]
    min_change = min(layer_changes)
    max_change = max(layer_changes)
    
    if max_change - min_change < 1e-6:
        # All layers have similar change, use uniform rank
        ranks = [max_rank // 2] * len(layer_changes)
    else:
        normalized = [(c - min_change) / (max_change - min_change) for c in layer_changes]
        
        # Map to rank range (higher change -> higher rank)
        ranks = [int(min_rank + n * (max_rank - min_rank)) for n in normalized]
    
    # Print rank assignments
    print(f"{Fore.GREEN}[DTB] Rank assignments:{Fore.RESET}")
    for i, rank in enumerate(ranks):
        print(f"  Layer {i+1}: rank {rank}")
    
    return ranks


def extract_principal_directions(
    model,
    layer_idx: int,
    dataloader,
    tokenizer,
    max_length: int,
    rank: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract principal directions of change for a layer using PCA/SVD."""
    
    print(f"\n{Fore.CYAN}[DTB] Extracting principal directions for layer {layer_idx+1}...{Fore.RESET}")
    
    deltas = []
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:  # Use limited batches for PCA
            break
            
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            input_hidden = hidden_states[layer_idx]
            output_hidden = hidden_states[layer_idx + 1]
            
            # Compute delta
            delta = output_hidden - input_hidden
            deltas.append(delta.reshape(-1, delta.shape[-1]).cpu())
    
    # Concatenate all deltas
    all_deltas = torch.cat(deltas, dim=0)
    print(f"[DTB] Collected {all_deltas.shape[0]} delta samples")
    
    # Perform SVD to get principal components
    U, S, V = torch.svd(all_deltas.float())
    
    # Keep top-k components
    principal_dirs = V[:, :rank]
    singular_values = S[:rank]
    
    print(f"[DTB] Top {rank} singular values: {singular_values[:5].tolist()[:5]}...")
    print(f"[DTB] Explained variance ratio: {(S[:rank].sum() / S.sum()).item():.4f}")
    
    return principal_dirs, singular_values


def initialize_delta_network_from_pca(
    delta_net: DeltaNetwork,
    principal_dirs: torch.Tensor,
    singular_values: torch.Tensor
):
    """Initialize delta network using PCA directions."""
    
    print(f"[DTB] Initializing delta network from PCA...")
    
    if not delta_net.use_attention:
        # Initialize projection matrices using principal components
        # Down projection maps to principal component space
        delta_net.down_proj.weight.data = principal_dirs.T.to(delta_net.down_proj.weight.dtype).to(delta_net.down_proj.weight.device)
        
        # Up projection reconstructs from principal components
        # Scale by singular values for importance weighting
        scaled_dirs = principal_dirs * singular_values.sqrt().unsqueeze(0)
        delta_net.up_proj.weight.data = scaled_dirs.to(delta_net.up_proj.weight.dtype).to(delta_net.up_proj.weight.device)
        
        print(f"[DTB] Initialized projection matrices with PCA components")
    else:
        # For attention-based delta, use principal directions to initialize value/output
        delta_net.value.weight.data = principal_dirs.T[:delta_net.rank].to(delta_net.value.weight.dtype).to(delta_net.value.weight.device)
        delta_net.output.weight.data = principal_dirs[:, :delta_net.rank].to(delta_net.output.weight.dtype).to(delta_net.output.weight.device)
        print(f"[DTB] Initialized attention matrices with PCA components")


def apply_dtb_transformation(
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
    min_rank: int = 4,
    max_rank: int = 64,
    use_attention_delta: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    **kwargs
) -> str:
    """Apply Differential Transformer Blocks transformation to model."""
    
    print(f"\n{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}[DTB] Starting Differential Transformer Blocks{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}\n")
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DTB] Using device: {device}")
    
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"[DTB] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get calibration data
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Analyze layer changes
    layer_changes = analyze_layer_changes(model, dataloader, tokenizer, max_length, device)
    
    # Compute appropriate ranks for each layer
    ranks = compute_delta_ranks(layer_changes, min_rank, max_rank)
    
    # Reload dataloader for PCA
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column, 
        min(dataset_size, 500),  # Use less data for PCA
        batch_size,
        tokenizer
    )
    
    # Apply DTB to specified layers
    if end_id == 0:
        end_id = model.config.num_hidden_layers
    
    print(f"\n{Fore.YELLOW}[DTB] Replacing layers {start_id} to {end_id} with delta networks{Fore.RESET}")
    
    for layer_idx in range(start_id - 1, end_id - 1):
        print(f"\n{Fore.CYAN}[DTB] Processing layer {layer_idx + 1}{Fore.RESET}")
        
        # Get rank for this layer
        rank = ranks[layer_idx] if layer_idx < len(ranks) else min_rank
        
        # Extract principal directions
        principal_dirs, singular_values = extract_principal_directions(
            model, layer_idx, dataloader, tokenizer, max_length, rank, device
        )
        
        # Create delta network
        delta_net = DeltaNetwork(
            model.config.hidden_size,
            rank,
            use_attention=use_attention_delta and layer_changes[layer_idx] > 0.05
        ).to(device).to(torch.bfloat16)
        
        # Initialize from PCA
        initialize_delta_network_from_pca(delta_net, principal_dirs, singular_values)
        
        # Replace layer with delta network wrapper
        original_layer = model.model.layers[layer_idx]
        
        class DeltaLayerWrapper(nn.Module):
            def __init__(self, delta_network):
                super().__init__()
                self.delta_network = delta_network
                # Keep layer norm from original
                self.input_layernorm = original_layer.input_layernorm
                self.post_attention_layernorm = original_layer.post_attention_layernorm
            
            def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                       output_attentions=False, use_cache=False, cache_position=None, **kwargs):
                # Compute delta instead of full layer
                delta = self.delta_network(hidden_states)
                
                # Add delta to input (residual connection is already there)
                output = hidden_states + delta
                
                # Return in expected format
                if use_cache:
                    return (output, None, None)
                elif output_attentions:
                    return (output, None)
                else:
                    return (output,)
        
        # Replace the layer
        model.model.layers[layer_idx] = DeltaLayerWrapper(delta_net)
        
        print(f"[DTB] Layer {layer_idx + 1} replaced with rank-{rank} delta network")
        
        # Measure memory/parameter reduction
        original_params = sum(p.numel() for p in original_layer.parameters())
        delta_params = sum(p.numel() for p in delta_net.parameters())
        reduction = (1 - delta_params / original_params) * 100
        print(f"[DTB] Parameter reduction: {reduction:.1f}% ({original_params} -> {delta_params})")
    
    # Save model
    if save_path is None:
        save_path = f"output_models/{model_path.replace('/', '_')}_DTB_{start_id}_{end_id}"
    
    print(f"\n{Fore.GREEN}[DTB] Saving transformed model to {save_path}{Fore.RESET}")
    
    # Ensure the model is on CPU before saving
    model = model.to('cpu')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save DTB metadata
    import json
    metadata = {
        "method": "DTB",
        "layers_replaced": f"{start_id}-{end_id}",
        "ranks": ranks[start_id-1:end_id-1],
        "layer_changes": layer_changes[start_id-1:end_id-1],
        "use_attention_delta": use_attention_delta
    }
    
    with open(f"{save_path}/dtb_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"{Fore.GREEN}[DTB] Transformation complete!{Fore.RESET}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path