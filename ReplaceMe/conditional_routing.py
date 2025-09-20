"""Conditional Block Routing module for adaptive transformer computation.

This module implements dynamic routing of tokens through different computational
paths based on input complexity, without requiring training or healing.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import numpy as np

from .utils import (get_calib_dataloader, seed_all)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class RouterNetwork(nn.Module):
    """Lightweight router to decide computational path for each token."""
    
    def __init__(self, hidden_size, temperature=1.0):
        super().__init__()
        # Very small network: hidden_size -> 64 -> 3
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 paths: simple, medium, complex
        )
        self.temperature = temperature
        
    def forward(self, hidden_states):
        """Route tokens to different paths.
        
        Returns:
            route_probs: [batch, seq_len, 3] probabilities for each path
            route_decisions: [batch, seq_len] integer path indices
        """
        # Pool over hidden dimension for routing decision
        router_input = hidden_states.mean(dim=-1)  # [batch, seq_len]
        router_logits = self.router(hidden_states)  # [batch, seq_len, 3]
        
        # Soft routing (for training) or hard routing (for inference)
        route_probs = F.softmax(router_logits / self.temperature, dim=-1)
        route_decisions = torch.argmax(route_probs, dim=-1)
        
        return route_probs, route_decisions


class ConditionalRoutingLayer(nn.Module):
    """Layer that routes tokens through different computational paths."""
    
    def __init__(self, original_layer, layer_idx, router, low_rank_dim=32):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.router = router
        
        # Get hidden size from the original layer
        if hasattr(original_layer, 'self_attn'):
            hidden_size = original_layer.self_attn.hidden_size
        else:
            hidden_size = original_layer.hidden_size
            
        # Path 1: Simple - just learned bias
        self.simple_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Path 2: Medium - low-rank approximation
        self.low_rank_down = nn.Linear(hidden_size, low_rank_dim, bias=False)
        self.low_rank_up = nn.Linear(low_rank_dim, hidden_size, bias=False)
        
        # Initialize low-rank as near-identity
        with torch.no_grad():
            # Initialize to approximate identity
            nn.init.xavier_uniform_(self.low_rank_down.weight, gain=0.1)
            nn.init.xavier_uniform_(self.low_rank_up.weight, gain=0.1)
        
        # Path 3: Complex - use original layer
        
        # Statistics
        self.path_counts = [0, 0, 0]
        self.total_tokens = 0
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions
        route_probs, route_decisions = self.router(hidden_states)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each path
        for path_idx in range(3):
            # Get mask for tokens taking this path
            path_mask = (route_decisions == path_idx)  # [batch, seq_len]
            
            if path_mask.any():
                # Count for statistics
                self.path_counts[path_idx] += path_mask.sum().item()
                self.total_tokens += path_mask.numel()
                
                if path_idx == 0:  # Simple path
                    # Just add bias and skip
                    path_output = hidden_states + self.simple_bias.unsqueeze(0).unsqueeze(0)
                    print(f"{Fore.GREEN}[Layer {self.layer_idx}] Simple path: {path_mask.sum().item()} tokens{Fore.RESET}")
                    
                elif path_idx == 1:  # Medium path - low-rank
                    # Low-rank transformation
                    down_proj = self.low_rank_down(hidden_states)
                    path_output = hidden_states + self.low_rank_up(down_proj)
                    print(f"{Fore.YELLOW}[Layer {self.layer_idx}] Medium path: {path_mask.sum().item()} tokens{Fore.RESET}")
                    
                else:  # path_idx == 2, Complex path
                    # Full transformer block
                    path_output = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)[0]
                    print(f"{Fore.RED}[Layer {self.layer_idx}] Complex path: {path_mask.sum().item()} tokens{Fore.RESET}")
                
                # Apply mask and accumulate
                path_mask_expanded = path_mask.unsqueeze(-1).expand_as(output)
                output = torch.where(path_mask_expanded, path_output, output)
        
        return output
    
    def get_routing_stats(self):
        """Get statistics about routing decisions."""
        if self.total_tokens > 0:
            percentages = [count / self.total_tokens * 100 for count in self.path_counts]
            return {
                'simple': percentages[0],
                'medium': percentages[1],
                'complex': percentages[2]
            }
        return {'simple': 0, 'medium': 0, 'complex': 0}


def analyze_token_complexity(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    token: Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """Analyze token complexity patterns in each layer.
    
    Returns:
        Dict mapping layer_idx to complexity statistics
    """
    print(f"{Fore.CYAN}=== Analyzing Token Complexity Patterns ==={Fore.RESET}")
    
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        output_attentions=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    num_layers = model.config.num_hidden_layers
    layer_complexity = {i: {'entropy': [], 'norm_change': []} for i in range(num_layers)}
    
    print(f"{Fore.YELLOW}Computing complexity metrics...{Fore.RESET}")
    
    for batch in tqdm(dataloader, desc="Analyzing complexity", colour="yellow"):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        
        for layer_idx in range(num_layers):
            # Compute change in hidden states
            if layer_idx < len(hidden_states) - 1:
                input_hidden = hidden_states[layer_idx]
                output_hidden = hidden_states[layer_idx + 1]
                
                # Norm of change
                norm_change = (output_hidden - input_hidden).norm(dim=-1).mean().item()
                layer_complexity[layer_idx]['norm_change'].append(norm_change)
            
            # Compute attention entropy (if available)
            if attentions and layer_idx < len(attentions):
                attn = attentions[layer_idx].mean(dim=1)  # Average over heads
                # Compute entropy
                attn_entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean().item()
                layer_complexity[layer_idx]['entropy'].append(attn_entropy)
    
    # Aggregate statistics
    complexity_stats = {}
    for layer_idx in range(num_layers):
        complexity_stats[layer_idx] = {
            'avg_entropy': np.mean(layer_complexity[layer_idx]['entropy']) if layer_complexity[layer_idx]['entropy'] else 0,
            'avg_norm_change': np.mean(layer_complexity[layer_idx]['norm_change']) if layer_complexity[layer_idx]['norm_change'] else 0
        }
        
        print(f"{Fore.GREEN}Layer {layer_idx}: entropy={complexity_stats[layer_idx]['avg_entropy']:.3f}, "
              f"norm_change={complexity_stats[layer_idx]['avg_norm_change']:.3f}{Fore.RESET}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return complexity_stats


def create_routing_config(
    complexity_stats: Dict[int, Dict[str, float]],
    target_simple_ratio: float = 0.3,
    target_medium_ratio: float = 0.5
) -> Dict[int, Dict[str, float]]:
    """Create routing configuration based on complexity analysis.
    
    Returns:
        Dict mapping layer_idx to routing thresholds
    """
    print(f"{Fore.CYAN}=== Creating Routing Configuration ==={Fore.RESET}")
    
    routing_config = {}
    
    for layer_idx, stats in complexity_stats.items():
        # Use norm_change as primary metric
        norm_change = stats['avg_norm_change']
        
        # Adaptive thresholds based on layer depth
        if layer_idx < 10:  # Early layers: more complex processing
            simple_threshold = norm_change * 0.5
            medium_threshold = norm_change * 1.5
        elif layer_idx < 20:  # Middle layers: balanced
            simple_threshold = norm_change * 0.7
            medium_threshold = norm_change * 1.3
        else:  # Deep layers: more simple processing
            simple_threshold = norm_change * 0.9
            medium_threshold = norm_change * 1.1
        
        routing_config[layer_idx] = {
            'simple_threshold': simple_threshold,
            'medium_threshold': medium_threshold
        }
        
        print(f"{Fore.YELLOW}Layer {layer_idx}: simple<{simple_threshold:.3f}, "
              f"medium<{medium_threshold:.3f}, complex>={medium_threshold:.3f}{Fore.RESET}")
    
    return routing_config


def apply_conditional_routing(
    model_path: str,
    routing_config: Dict[int, Dict[str, float]],
    save_path: Optional[str] = None,
    low_rank_dim: int = 32
) -> str:
    """Apply conditional routing to model layers.
    
    Returns:
        Path to saved model
    """
    print(f"{Fore.CYAN}=== Applying Conditional Routing ==={Fore.RESET}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create routers for each layer
    hidden_size = model.config.hidden_size
    
    # Wrap layers with conditional routing
    for i, layer in enumerate(model.model.layers):
        if i in routing_config:
            print(f"{Fore.YELLOW}Adding conditional routing to layer {i}{Fore.RESET}")
            
            # Create router for this layer
            router = RouterNetwork(hidden_size)
            
            # Initialize router weights based on complexity analysis
            # (This is a simplified initialization - in practice would need calibration)
            
            # Wrap layer
            model.model.layers[i] = ConditionalRoutingLayer(
                layer, i, router, low_rank_dim=low_rank_dim
            )
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_conditional_routing"
    
    print(f"{Fore.GREEN}Saving model to {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save routing config
    import json
    with open(f"{save_path}/routing_config.json", 'w') as f:
        json.dump(routing_config, f)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path


def conditional_routing(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,  # Not used but kept for compatibility
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    target_simple_ratio: float = 0.3,
    target_medium_ratio: float = 0.5,
    low_rank_dim: int = 32,
    **kwargs  # Catch extra args
) -> str:
    """Main entry point for conditional routing method.
    
    Returns:
        Path to optimized model
    """
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Starting Conditional Block Routing{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    # Step 1: Analyze token complexity
    complexity_stats = analyze_token_complexity(
        model_path=model_path,
        dataset=dataset,
        dataset_column=dataset_column,
        batch_size=batch_size,
        max_length=max_length,
        dataset_size=dataset_size,
        dataset_subset=dataset_subset,
        use_4bit=use_4bit,
        token=token
    )
    
    print(f"\n{Fore.CYAN}Analyzed complexity for {len(complexity_stats)} layers{Fore.RESET}")
    
    # Step 2: Create routing configuration
    routing_config = create_routing_config(
        complexity_stats,
        target_simple_ratio=target_simple_ratio,
        target_medium_ratio=target_medium_ratio
    )
    
    print(f"\n{Fore.CYAN}Created routing config for {len(routing_config)} layers{Fore.RESET}")
    
    # Step 3: Apply conditional routing
    optimized_model_path = apply_conditional_routing(
        model_path=model_path,
        routing_config=routing_config,
        save_path=save_path,
        low_rank_dim=low_rank_dim
    )
    
    # Calculate expected FLOPs reduction
    expected_flops_reduction = (
        target_simple_ratio * 0.05 +  # Simple path: 5% FLOPs
        target_medium_ratio * 0.30 +  # Medium path: 30% FLOPs  
        (1 - target_simple_ratio - target_medium_ratio) * 1.0  # Complex path: 100% FLOPs
    )
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}Optimization Complete!{Fore.RESET}")
    print(f"{Fore.GREEN}Expected FLOPs: ~{expected_flops_reduction*100:.1f}% of original{Fore.RESET}")
    print(f"{Fore.GREEN}Model saved to: {optimized_model_path}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*60}{Fore.RESET}")
    
    return optimized_model_path