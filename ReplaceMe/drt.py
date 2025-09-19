"""Dynamic Resolution Transformer (DRT) Module

This module implements token merging strategies to reduce computation
while preserving model performance through adaptive resolution reduction.
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, seed_all, truncate_model)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class TokenMergeStrategy:
    """Manages token merging decisions based on attention patterns."""
    
    def __init__(self, merge_threshold: float = 0.7, min_tokens_ratio: float = 0.25):
        """
        Args:
            merge_threshold: Minimum attention weight to consider merging
            min_tokens_ratio: Minimum ratio of tokens to preserve
        """
        self.merge_threshold = merge_threshold
        self.min_tokens_ratio = min_tokens_ratio
        self.merge_history = {}
        
        print(f"{Fore.GREEN}[DRT] Initialized TokenMergeStrategy:")
        print(f"  - Merge threshold: {merge_threshold}")
        print(f"  - Min tokens ratio: {min_tokens_ratio}{Fore.RESET}")
    
    def analyze_attention_patterns(
        self, 
        attention_weights: torch.Tensor,
        layer_idx: int
    ) -> List[Tuple[int, int]]:
        """
        Analyze attention patterns to find mergeable tokens.
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            layer_idx: Current layer index
            
        Returns:
            List of token pairs to merge
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average across heads and batch
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        
        # Find strongly connected token pairs
        merge_candidates = []
        used_tokens = set()
        
        # Sort by attention strength
        values, indices = avg_attention.flatten().sort(descending=True)
        
        for idx in range(len(values)):
            if values[idx] < self.merge_threshold:
                break
                
            # Get token pair
            i = indices[idx] // seq_len
            j = indices[idx] % seq_len
            
            # Skip if same token or already used
            if i == j or i in used_tokens or j in used_tokens:
                continue
            
            # Skip special positions (first and last tokens often important)
            if i == 0 or j == 0 or i == seq_len-1 or j == seq_len-1:
                continue
                
            merge_candidates.append((i.item(), j.item()))
            used_tokens.add(i.item())
            used_tokens.add(j.item())
            
            # Limit merging to preserve minimum tokens
            if len(used_tokens) >= seq_len * (1 - self.min_tokens_ratio):
                break
        
        print(f"{Fore.YELLOW}[DRT] Layer {layer_idx}: Found {len(merge_candidates)} merge pairs{Fore.RESET}")
        return merge_candidates
    
    def create_merge_groups(
        self, 
        merge_candidates: List[Tuple[int, int]],
        seq_len: int
    ) -> Dict[int, List[int]]:
        """
        Create groups of tokens to merge (handle transitivity).
        
        Returns:
            Dictionary mapping group_id to list of token indices
        """
        groups = {}
        token_to_group = {}
        next_group_id = 0
        
        for i, j in merge_candidates:
            if i in token_to_group and j in token_to_group:
                # Merge two groups
                group_i = token_to_group[i]
                group_j = token_to_group[j]
                if group_i != group_j:
                    # Merge smaller into larger
                    if len(groups[group_i]) < len(groups[group_j]):
                        group_i, group_j = group_j, group_i
                    groups[group_i].extend(groups[group_j])
                    for token in groups[group_j]:
                        token_to_group[token] = group_i
                    del groups[group_j]
            elif i in token_to_group:
                # Add j to i's group
                group_id = token_to_group[i]
                groups[group_id].append(j)
                token_to_group[j] = group_id
            elif j in token_to_group:
                # Add i to j's group
                group_id = token_to_group[j]
                groups[group_id].append(i)
                token_to_group[i] = group_id
            else:
                # Create new group
                groups[next_group_id] = [i, j]
                token_to_group[i] = next_group_id
                token_to_group[j] = next_group_id
                next_group_id += 1
        
        # Add unmerged tokens as single-token groups
        for idx in range(seq_len):
            if idx not in token_to_group:
                groups[next_group_id] = [idx]
                token_to_group[idx] = next_group_id
                next_group_id += 1
        
        return groups


def apply_drt_transform(
    hidden_states: torch.Tensor,
    attention_weights: torch.Tensor,
    merge_strategy: TokenMergeStrategy,
    layer_idx: int,
    depth_ratio: float
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply DRT transformation to hidden states.
    
    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_weights: [batch, heads, seq_len, seq_len]
        merge_strategy: Token merging strategy
        layer_idx: Current layer index
        depth_ratio: Depth ratio (0.0=shallow, 1.0=deep)
        
    Returns:
        Merged hidden states and merge map for restoration
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Adjust merge threshold based on depth
    adaptive_threshold = merge_strategy.merge_threshold * (1 - 0.5 * depth_ratio)
    merge_strategy.merge_threshold = adaptive_threshold
    
    # Find merge candidates
    merge_candidates = merge_strategy.analyze_attention_patterns(
        attention_weights, layer_idx
    )
    
    if not merge_candidates:
        print(f"{Fore.BLUE}[DRT] Layer {layer_idx}: No merging needed{Fore.RESET}")
        return hidden_states, {}
    
    # Create merge groups
    merge_groups = merge_strategy.create_merge_groups(merge_candidates, seq_len)
    
    # Apply merging
    merged_states = []
    merge_map = {}
    
    # Sort groups by first token index for consistent ordering
    sorted_groups = sorted(merge_groups.items(), 
                          key=lambda x: min(x[1]))
    
    for new_idx, (group_id, token_indices) in enumerate(sorted_groups):
        if len(token_indices) == 1:
            # No merge - keep original
            merged_states.append(hidden_states[:, token_indices[0], :])
        else:
            # Weighted average based on attention importance
            weights = attention_weights[:, :, token_indices, :].sum(dim=(1, 3))  # [batch, tokens]
            weights = F.softmax(weights, dim=1)  # Normalize
            
            # Compute weighted average
            merged = torch.zeros(batch_size, hidden_dim, device=hidden_states.device)
            for i, idx in enumerate(token_indices):
                merged += hidden_states[:, idx, :] * weights[:, i].unsqueeze(1)
            merged_states.append(merged)
        
        # Store mapping for restoration
        for idx in token_indices:
            merge_map[idx] = new_idx
    
    merged_hidden = torch.stack(merged_states, dim=1)
    
    print(f"{Fore.GREEN}[DRT] Layer {layer_idx}: Merged {seq_len} → {len(merged_states)} tokens "
          f"(reduction: {100*(1-len(merged_states)/seq_len):.1f}%){Fore.RESET}")
    
    return merged_hidden, merge_map


def restore_resolution(
    merged_states: torch.Tensor,
    merge_map: Dict[int, int],
    original_seq_len: int
) -> torch.Tensor:
    """
    Restore merged states back to original sequence length.
    
    Args:
        merged_states: [batch, merged_seq_len, hidden_dim]
        merge_map: Mapping from original to merged indices
        original_seq_len: Original sequence length
        
    Returns:
        Restored states [batch, original_seq_len, hidden_dim]
    """
    batch_size, _, hidden_dim = merged_states.shape
    
    # Create restored tensor
    restored = torch.zeros(
        batch_size, original_seq_len, hidden_dim,
        device=merged_states.device,
        dtype=merged_states.dtype
    )
    
    # Fill in values
    for orig_idx, merged_idx in merge_map.items():
        restored[:, orig_idx, :] = merged_states[:, merged_idx, :]
    
    return restored


def drt_transform(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    merge_threshold: float = 0.7,
    min_tokens_ratio: float = 0.25,
    start_merge_layer: int = 10,
    **kwargs
) -> str:
    """
    Apply Dynamic Resolution Transformer to model.
    
    Args:
        model_path: Path to pretrained model
        dataset: Calibration dataset name
        dataset_column: Text column in dataset
        batch_size: Processing batch size
        max_length: Maximum sequence length
        layers_to_skip: Layers to skip (compatibility)
        dataset_size: Size of calibration set
        dataset_subset: Dataset split to use
        use_4bit: Use 4-bit quantization
        save_path: Where to save transformed model
        token: HuggingFace token
        merge_threshold: Attention threshold for merging
        min_tokens_ratio: Minimum tokens to preserve
        start_merge_layer: Layer to start merging from
        
    Returns:
        Path to saved model
    """
    print(f"{Fore.CYAN}{'='*60}")
    print(f"[DRT] Starting Dynamic Resolution Transformer")
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
    print(f"{Fore.YELLOW}[DRT] Loading model: {model_path}{Fore.RESET}")
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
    
    # Get calibration data
    print(f"{Fore.YELLOW}[DRT] Loading calibration dataset{Fore.RESET}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Initialize merge strategy
    merge_strategy = TokenMergeStrategy(
        merge_threshold=merge_threshold,
        min_tokens_ratio=min_tokens_ratio
    )
    
    # Collect merge statistics
    layer_merge_stats = {}
    num_layers = model.config.num_hidden_layers
    
    print(f"{Fore.CYAN}[DRT] Analyzing attention patterns...{Fore.RESET}")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calibration", colour="cyan")):
        if batch_idx >= 10:  # Use first 10 batches for calibration
            break
            
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
        
        # Analyze attention patterns for each layer
        if outputs.attentions:
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                if layer_idx < start_merge_layer:
                    continue
                    
                depth_ratio = layer_idx / num_layers
                
                # Analyze this layer's patterns
                merge_candidates = merge_strategy.analyze_attention_patterns(
                    attn_weights, layer_idx
                )
                
                if layer_idx not in layer_merge_stats:
                    layer_merge_stats[layer_idx] = []
                layer_merge_stats[layer_idx].append(len(merge_candidates))
    
    # Compute average merge statistics
    print(f"\n{Fore.GREEN}[DRT] Merge Statistics:{Fore.RESET}")
    for layer_idx in sorted(layer_merge_stats.keys()):
        avg_merges = sum(layer_merge_stats[layer_idx]) / len(layer_merge_stats[layer_idx])
        print(f"  Layer {layer_idx}: avg {avg_merges:.1f} merge pairs")
    
    # Apply DRT transformation to model
    print(f"\n{Fore.CYAN}[DRT] Applying transformations...{Fore.RESET}")
    
    # Inject DRT hooks into model
    inject_drt_hooks(model, merge_strategy, start_merge_layer)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_DRT_{merge_threshold}_{min_tokens_ratio}"
    
    print(f"{Fore.GREEN}[DRT] Saving transformed model to: {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save DRT configuration
    drt_config = {
        'merge_threshold': merge_threshold,
        'min_tokens_ratio': min_tokens_ratio,
        'start_merge_layer': start_merge_layer,
        'layer_merge_stats': layer_merge_stats
    }
    
    import json
    with open(f"{save_path}/drt_config.json", 'w') as f:
        json.dump(drt_config, f, indent=2)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}[DRT] Transformation complete!{Fore.RESET}")
    return save_path


def inject_drt_hooks(model, merge_strategy: TokenMergeStrategy, start_layer: int):
    """
    Inject DRT hooks into model layers.
    
    This modifies the model to apply token merging during forward passes.
    """
    print(f"{Fore.YELLOW}[DRT] Injecting hooks into model layers...{Fore.RESET}")
    
    def create_hook(layer_idx: int, original_forward):
        """Create a forward hook for a specific layer."""
        
        def drt_forward(module, args, kwargs=None):
            if kwargs is None:
                kwargs = {}
                
            # Get hidden states from args
            hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            
            # Run original forward to get attention weights
            with torch.no_grad():
                # Temporarily enable attention output
                original_output_attentions = kwargs.get('output_attentions', False)
                kwargs['output_attentions'] = True
                
                outputs = original_forward(*args, **kwargs)
                
                # Restore original setting
                kwargs['output_attentions'] = original_output_attentions
            
            # Apply DRT if we have attention weights and past merge layer
            if layer_idx >= start_layer and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                depth_ratio = layer_idx / len(model.model.layers)
                
                # Apply token merging
                merged_hidden, merge_map = apply_drt_transform(
                    hidden_states,
                    outputs.attentions[-1],  # Use last attention weights
                    merge_strategy,
                    layer_idx,
                    depth_ratio
                )
                
                # Store merge map for potential restoration
                if not hasattr(module, '_drt_merge_maps'):
                    module._drt_merge_maps = {}
                module._drt_merge_maps[layer_idx] = merge_map
                
                # Update args with merged hidden states
                if len(args) > 0:
                    args = (merged_hidden,) + args[1:]
                else:
                    kwargs['hidden_states'] = merged_hidden
                
                # Re-run forward with merged states
                outputs = original_forward(*args, **kwargs)
            
            return outputs
        
        return drt_forward
    
    # Apply hooks to transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for idx, layer in enumerate(model.model.layers):
            if idx >= start_layer:
                # Store original forward
                original_forward = layer.forward
                # Replace with DRT forward
                layer.forward = create_hook(idx, original_forward)
                print(f"  Injected hook at layer {idx}")
    
    print(f"{Fore.GREEN}[DRT] Hooks injected successfully{Fore.RESET}")


def read_config(config_path: str) -> dict:
    """Read YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run DRT from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run Dynamic Resolution Transformer from config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    return drt_transform(**config)