"""Attention Head Lifecycle (AHL) Module

Implements progressive attention head reduction based on layer depth.
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def calibrate_head_importance(model, dataloader, num_samples=100):
    """Measure importance of each attention head in each layer."""
    
    print(f"{Fore.CYAN}[AHL] Calibrating head importance...{Fore.RESET}")
    
    head_importance = {}
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
        layer = model.model.layers[layer_idx]
        head_scores = torch.zeros(num_heads)
        
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            inputs = model.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get hidden states before this layer
                hidden_states = model.model.embed_tokens(inputs['input_ids'])
                
                for i in range(layer_idx):
                    hidden_states = model.model.layers[i](hidden_states)[0]
                
                # Get attention weights from this layer
                layer_output = layer(
                    hidden_states,
                    output_attentions=True
                )
                
                if len(layer_output) > 1 and layer_output[1] is not None:
                    attention_weights = layer_output[1]  # [batch, num_heads, seq_len, seq_len]
                    
                    # Calculate importance as entropy of attention patterns
                    for head_idx in range(num_heads):
                        head_attn = attention_weights[:, head_idx, :, :]
                        # Entropy as importance metric
                        entropy = -(head_attn * torch.log(head_attn + 1e-9)).sum(dim=-1).mean()
                        head_scores[head_idx] += entropy.item()
                
                sample_count += inputs['input_ids'].shape[0]
        
        head_importance[layer_idx] = head_scores / max(1, sample_count)
        
    return head_importance


def create_head_schedule(head_importance, num_layers, num_heads, strategy='progressive'):
    """Create schedule for which heads are active at each layer."""
    
    schedule = {}
    
    for layer_idx in range(num_layers):
        importance_scores = head_importance.get(layer_idx, torch.ones(num_heads))
        
        if strategy == 'progressive':
            # Progressive reduction based on depth
            if layer_idx < num_layers * 0.3:  # First 30%
                # Keep all heads
                active_ratio = 1.0
            elif layer_idx < num_layers * 0.6:  # Middle 30%
                # Keep 75% of heads
                active_ratio = 0.75
            else:  # Last 40%
                # Keep 50% of heads
                active_ratio = 0.5
        elif strategy == 'linear':
            # Linear reduction
            active_ratio = 1.0 - (layer_idx / num_layers) * 0.5
        else:
            active_ratio = 1.0
        
        num_active = max(1, int(num_heads * active_ratio))
        
        # Select top-k important heads
        _, top_indices = torch.topk(importance_scores, num_active)
        schedule[layer_idx] = top_indices.tolist()
        
    return schedule


def modify_attention_layer(layer, layer_idx, active_heads, num_heads):
    """Modify a single attention layer to use AHL."""
    
    original_forward = layer.self_attn.forward
    
    def ahl_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        head_dim = hidden_dim // num_heads
        
        # Compute Q, K, V for all heads initially
        query_states = layer.self_attn.q_proj(hidden_states)
        key_states = layer.self_attn.k_proj(hidden_states)
        value_states = layer.self_attn.v_proj(hidden_states)
        
        # Reshape to separate heads
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Process active heads normally, bypass inactive heads
        attn_output = torch.zeros_like(query_states)
        
        for head_idx in range(num_heads):
            if head_idx in active_heads:
                # Active head: compute attention
                scores = torch.matmul(query_states[:, head_idx], key_states[:, head_idx].transpose(-1, -2))
                scores = scores / torch.sqrt(torch.tensor(head_dim, dtype=scores.dtype))
                
                if attention_mask is not None:
                    scores = scores + attention_mask
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_output[:, head_idx] = torch.matmul(attn_weights, value_states[:, head_idx])
            else:
                # Inactive head: identity mapping (bypass)
                attn_output[:, head_idx] = value_states[:, head_idx]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Output projection
        attn_output = layer.self_attn.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, None  # We don't compute full attention weights
        return attn_output,
    
    # Replace forward method
    layer.self_attn.forward = ahl_forward
    
    # Log modification
    print(f"{Fore.YELLOW}[AHL] Layer {layer_idx}: {len(active_heads)}/{num_heads} heads active{Fore.RESET}")


def ahl_transform(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    strategy: str = 'progressive',
    min_heads_ratio: float = 0.5,
    **kwargs
) -> str:
    """Main AHL transformation function."""
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"[AHL] Starting Attention Head Lifecycle Transform")
    print(f"[AHL] Strategy: {strategy}")
    print(f"[AHL] Minimum heads ratio: {min_heads_ratio}")
    print(f"{'='*60}{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    print(f"[AHL] Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
        token=token,
        torch_dtype=torch.bfloat16 if not use_4bit else None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.tokenizer = tokenizer  # Attach for convenience
    model.eval()
    
    # Get calibration data
    print(f"[AHL] Loading calibration dataset")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Calibrate head importance
    head_importance = calibrate_head_importance(model, dataloader, num_samples=100)
    
    # Create head schedule
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_schedule = create_head_schedule(head_importance, num_layers, num_heads, strategy)
    
    # Calculate statistics
    total_heads = num_layers * num_heads
    active_heads = sum(len(heads) for heads in head_schedule.values())
    reduction_ratio = 1 - (active_heads / total_heads)
    
    print(f"\n{Fore.GREEN}[AHL] Head Reduction Statistics:")
    print(f"  Total heads: {total_heads}")
    print(f"  Active heads: {active_heads}")
    print(f"  Reduction: {reduction_ratio*100:.1f}%{Fore.RESET}")
    
    # Clean up calibration model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for modification
    print(f"\n[AHL] Reloading model for modification...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token
    )
    
    # Apply AHL modifications
    print(f"[AHL] Applying head lifecycle modifications...")
    for layer_idx in range(num_layers):
        active_heads = head_schedule[layer_idx]
        modify_attention_layer(
            model.model.layers[layer_idx],
            layer_idx,
            active_heads,
            num_heads
        )
    
    # Test modified model
    print(f"\n[AHL] Testing modified model...")
    test_text = "The capital of France is"
    test_inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**test_inputs)
        print(f"[AHL] Test passed! Output shape: {outputs.logits.shape}")
    
    # Save
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        model_name = model_path.split('/')[-1]
        save_path = f"output_models/{model_name}_AHL_{strategy}_r{int(reduction_ratio*100)}"
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"[AHL] Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save AHL config
    ahl_config = {
        'method': 'ahl',
        'strategy': strategy,
        'min_heads_ratio': min_heads_ratio,
        'head_schedule': head_schedule,
        'head_importance': {k: v.tolist() for k, v in head_importance.items()},
        'reduction_ratio': reduction_ratio,
        'active_heads': active_heads,
        'total_heads': total_heads
    }
    
    with open(f"{save_path}/ahl_config.json", 'w') as f:
        json.dump(ahl_config, f, indent=2)
    
    print(f"{Fore.GREEN}[AHL] Transformation complete!{Fore.RESET}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path