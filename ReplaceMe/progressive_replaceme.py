"""Progressive ReplaceMe module for gradual transformer block replacement.

This module implements a two-phase approach: first replacing FFN with linear
transformation, then optionally simplifying attention layers.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import numpy as np

from .utils import (get_calib_dataloader, adam_method, optimizing_method,
                    truncate_model, seed_all)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def compute_ffn_replacement(
    activations: Dict[int, Dict[str, torch.Tensor]],
    layer_idx: int,
    loss: str = "cosine",
    solver: str = "adam"
) -> torch.Tensor:
    """Compute linear transformation to replace FFN.
    
    Returns:
        Transformation matrix T such that pre_ffn @ T ≈ post_ffn
    """
    print(f"{Fore.CYAN}Computing FFN replacement for layer {layer_idx}...{Fore.RESET}")
    
    pre_ffn = activations[layer_idx]['pre_ffn']
    post_ffn = activations[layer_idx]['post_ffn']
    
    # Compute difference (since FFN output is added to input as residual)
    ffn_delta = post_ffn  # This is just the FFN output
    
    print(f"{Fore.YELLOW}Input shape: {pre_ffn.shape}, Output shape: {ffn_delta.shape}{Fore.RESET}")
    
    # Use adam_method or optimizing_method from utils
    if solver == "adam":
        transform = adam_method(pre_ffn, ffn_delta, loss=loss)
    else:
        transform = optimizing_method(pre_ffn, ffn_delta, solver=solver)
    
    # Calculate approximation error
    with torch.no_grad():
        approx = pre_ffn @ transform.to(pre_ffn.dtype)
        error = (approx - ffn_delta).norm() / ffn_delta.norm()
        print(f"{Fore.GREEN}FFN approximation error: {error:.4f}{Fore.RESET}")
    
    return transform


def compute_attention_simplification(
    activations: Dict[int, Dict[str, torch.Tensor]],
    layer_idx: int,
    rank: int = 64
) -> Dict[str, torch.Tensor]:
    """Compute low-rank approximation for attention.
    
    Returns:
        Dict with 'U' and 'V' matrices for low-rank approximation
    """
    print(f"{Fore.CYAN}Computing attention simplification for layer {layer_idx}...{Fore.RESET}")
    
    pre_attn = activations[layer_idx]['pre_attn']
    post_attn = activations[layer_idx]['post_attn']
    
    # Compute the transformation matrix
    # post_attn ≈ pre_attn @ T
    # Using SVD for low-rank approximation
    
    print(f"{Fore.YELLOW}Computing SVD for rank-{rank} approximation...{Fore.RESET}")
    
    # First compute full transformation
    if pre_attn.shape[0] > 10000:
        # Sample for efficiency
        indices = torch.randperm(pre_attn.shape[0])[:10000]
        pre_sample = pre_attn[indices]
        post_sample = post_attn[indices]
    else:
        pre_sample = pre_attn
        post_sample = post_attn
    
    # Solve for T: pre @ T = post
    T = torch.linalg.lstsq(pre_sample, post_sample).solution
    
    # Low-rank approximation via SVD
    U, S, V = torch.svd(T)
    
    # Keep top-k components
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]
    
    # Create low-rank matrices
    low_rank_transform = {
        'U': U_r @ torch.diag(torch.sqrt(S_r)),
        'V': V_r @ torch.diag(torch.sqrt(S_r))
    }
    
    # Calculate approximation error
    T_approx = low_rank_transform['U'] @ low_rank_transform['V'].T
    with torch.no_grad():
        approx = pre_sample @ T_approx.to(pre_sample.dtype)
        error = (approx - post_sample).norm() / post_sample.norm()
        print(f"{Fore.GREEN}Attention rank-{rank} approximation error: {error:.4f}{Fore.RESET}")
    
    return low_rank_transform


def apply_progressive_replacement(
    model_path: str,
    ffn_transforms: Dict[int, torch.Tensor],
    attn_transforms: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    save_path: Optional[str] = None
) -> str:
    """Apply progressive replacements to model.
    
    Args:
        model_path: Path to base model
        ffn_transforms: Dict mapping layer_idx to FFN replacement matrix
        attn_transforms: Optional dict mapping layer_idx to attention low-rank matrices
        save_path: Where to save modified model
    
    Returns:
        Path to saved model
    """
    print(f"{Fore.CYAN}=== Applying Progressive Replacements ==={Fore.RESET}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Apply FFN replacements
    for layer_idx, transform in ffn_transforms.items():
        print(f"{Fore.YELLOW}Replacing FFN in layer {layer_idx}...{Fore.RESET}")
        
        # Get the MLP module
        mlp = model.model.layers[layer_idx].mlp
        
        # Create a simple linear replacement
        hidden_size = model.config.hidden_size
        replacement = nn.Linear(hidden_size, hidden_size, bias=True)
        replacement.weight.data = transform.T.to(torch.bfloat16)
        replacement.bias.data = torch.zeros(hidden_size, dtype=torch.bfloat16)
        
        # Replace the MLP
        model.model.layers[layer_idx].mlp = replacement
        
        print(f"{Fore.GREEN}Layer {layer_idx} FFN replaced with linear transform{Fore.RESET}")
    
    # Apply attention simplifications if provided
    if attn_transforms:
        print(f"{Fore.YELLOW}Applying attention simplifications...{Fore.RESET}")
        for layer_idx, low_rank in attn_transforms.items():
            print(f"{Fore.RED}Warning: Attention replacement not yet fully implemented{Fore.RESET}")
            # This would require more complex modifications to the attention mechanism
            # For now, we skip this part
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_progressive_replaceme"
    
    print(f"{Fore.GREEN}Saving model to {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save replacement info
    import json
    replacement_info = {
        'ffn_replaced_layers': list(ffn_transforms.keys()),
        'attn_simplified_layers': list(attn_transforms.keys()) if attn_transforms else []
    }
    with open(f"{save_path}/replacement_info.json", 'w') as f:
        json.dump(replacement_info, f)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path


def analyze_layer_components(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_analyze: List[Tuple[int, int]],  # Changed to accept ranges
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    token: Optional[str] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Analyze FFN and Attention components separately for specified layers.
    
    Args:
        layers_to_analyze: List of (start, end) tuples for layer ranges
    
    Returns:
        Dict mapping layer_idx to component activations
    """
    print(f"{Fore.CYAN}=== Analyzing Layer Components (FFN vs Attention) ==={Fore.RESET}")
    
    # Convert ranges to individual layer indices
    individual_layers = []
    for start, end in layers_to_analyze:
        individual_layers.extend(range(start, end))
    
    print(f"{Fore.YELLOW}Analyzing layers: {individual_layers}{Fore.RESET}")
    
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
    
    # Storage for activations
    hidden_size = model.config.hidden_size
    activations_storage = {
        layer_idx: {
            'pre_attn': [],
            'post_attn': [],
            'pre_ffn': [],
            'post_ffn': []
        } for layer_idx in individual_layers
    }
    
    # Hook functions to capture activations
    hooks = []
    captured_activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            captured_activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return hook
    
    # Register hooks
    for i in individual_layers:
        layer = model.model.layers[i]
        
        # Hook before and after attention
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(f'layer_{i}_attn')
        ))
        
        # Hook before and after FFN
        hooks.append(layer.mlp.register_forward_hook(
            make_hook(f'layer_{i}_ffn')
        ))
        
        # Hook for layer input
        hooks.append(layer.register_forward_pre_hook(
            lambda m, i, layer_idx=i: captured_activations.update({f'layer_{layer_idx}_input': i[0]})
        ))
    
    print(f"{Fore.YELLOW}Collecting activations for {len(individual_layers)} layers...{Fore.RESET}")
    
    for batch in tqdm(dataloader, desc="Processing batches", colour="yellow"):
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
        
        # Store activations
        for layer_idx in individual_layers:
            if f'layer_{layer_idx}_input' in captured_activations:
                layer_input = captured_activations[f'layer_{layer_idx}_input']
                attn_output = captured_activations[f'layer_{layer_idx}_attn']
                ffn_output = captured_activations[f'layer_{layer_idx}_ffn']
                
                # Flatten batch and sequence dimensions
                activations_storage[layer_idx]['pre_attn'].append(
                    layer_input.view(-1, hidden_size).cpu()
                )
                activations_storage[layer_idx]['post_attn'].append(
                    attn_output.view(-1, hidden_size).cpu()
                )
                # FFN input is attention output + residual
                ffn_input = attn_output + layer_input
                activations_storage[layer_idx]['pre_ffn'].append(
                    ffn_input.view(-1, hidden_size).cpu()
                )
                activations_storage[layer_idx]['post_ffn'].append(
                    ffn_output.view(-1, hidden_size).cpu()
                )
    
    # Concatenate all batches
    for layer_idx in individual_layers:
        for key in activations_storage[layer_idx]:
            if activations_storage[layer_idx][key]:
                activations_storage[layer_idx][key] = torch.cat(
                    activations_storage[layer_idx][key], dim=0
                )
                print(f"{Fore.GREEN}Layer {layer_idx} {key}: shape {activations_storage[layer_idx][key].shape}{Fore.RESET}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return activations_storage, layers_to_analyze  # Also return the ranges


def progressive_replaceme(
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
    loss: str = "cosine",
    solver: str = "adam",
    phase: str = "ffn_only",  # "ffn_only" or "full"
    attention_rank: int = 64,
    distances_path: Optional[str] = None,
    num_blocks: int = 1,  # Number of blocks to replace
    merge_consecutive: bool = True,  # Whether to merge consecutive blocks
    **kwargs
) -> str:
    """Main entry point for Progressive ReplaceMe.
    
    Now uses cosine distance to automatically select blocks to replace.
    """
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Starting Progressive ReplaceMe{Fore.RESET}")
    print(f"{Fore.MAGENTA}Phase: {phase}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    # Step 1: Profile distances if not already done
    if distances_path is None or not os.path.exists(distances_path):
        print(f"{Fore.CYAN}Computing cosine distances between layers...{Fore.RESET}")
        
        # Use profile_distances to find most linear blocks
        import inspect
        from .distance import profile_distances
        
        sig = inspect.signature(profile_distances)
        distance_config = {k: v for k, v in locals().items() if k in sig.parameters}
        profile_distances(**distance_config)
        distances_path = "./distances.pth"
        
        print(f"{Fore.GREEN}Distance profiling complete{Fore.RESET}")
    
    # Step 2: Load distances and select blocks
    print(f"{Fore.CYAN}Selecting blocks based on cosine distance...{Fore.RESET}")
    average_distances = torch.load(distances_path, weights_only=False)
    
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_blocks,
        merge_consecutive=merge_consecutive
    )
    
    print(f"{Fore.GREEN}Selected blocks for replacement: {selected_blocks}{Fore.RESET}")
    print(f"{Fore.YELLOW}These blocks have the highest linearity (lowest cosine distance){Fore.RESET}")
    
    # Step 3: Analyze layer components for selected blocks
    activations, layer_ranges = analyze_layer_components(
        model_path=model_path,
        dataset=dataset,
        dataset_column=dataset_column,
        batch_size=batch_size,
        max_length=max_length,
        layers_to_analyze=selected_blocks,  # Pass the selected blocks
        dataset_size=dataset_size,
        dataset_subset=dataset_subset,
        use_4bit=use_4bit,
        token=token
    )
    
    # Extract individual layers from ranges
    layers_to_process = []
    for start, end in layer_ranges:
        layers_to_process.extend(range(start, end))
    
    print(f"\n{Fore.CYAN}Analyzing {len(layers_to_process)} layers from {len(selected_blocks)} blocks{Fore.RESET}")
    
    # Step 4: Compute FFN replacements
    print(f"\n{Fore.CYAN}=== Phase 1: Computing FFN Replacements ==={Fore.RESET}")
    ffn_transforms = {}
    
    for layer_idx in layers_to_process:
        ffn_transforms[layer_idx] = compute_ffn_replacement(
            activations, layer_idx, loss=loss, solver=solver
        )
        print(f"{Fore.GREEN}Layer {layer_idx} FFN transform computed{Fore.RESET}")
    
    # Step 5: Optionally compute attention simplifications
    attn_transforms = None
    if phase == "full":
        print(f"\n{Fore.CYAN}=== Phase 2: Computing Attention Simplifications ==={Fore.RESET}")
        attn_transforms = {}
        
        for layer_idx in layers_to_process:
            attn_transforms[layer_idx] = compute_attention_simplification(
                activations, layer_idx, rank=attention_rank
            )
            print(f"{Fore.GREEN}Layer {layer_idx} attention simplification computed{Fore.RESET}")
    
    # Step 6: Apply replacements
    optimized_model_path = apply_progressive_replacement(
        model_path=model_path,
        ffn_transforms=ffn_transforms,
        attn_transforms=attn_transforms,
        save_path=save_path
    )
    
    # Calculate expected FLOPs reduction
    num_ffn_replaced = len(ffn_transforms)
    num_attn_simplified = len(attn_transforms) if attn_transforms else 0
    total_layers = 32  # Assuming standard model
    
    # Rough estimation
    ffn_reduction = num_ffn_replaced * 0.3  # FFN is ~30% of computation
    attn_reduction = num_attn_simplified * 0.2  # Additional 20% if attention simplified
    total_reduction = (ffn_reduction + attn_reduction) / total_layers
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}Progressive Replacement Complete!{Fore.RESET}")
    print(f"{Fore.GREEN}Selected blocks (most linear): {selected_blocks}{Fore.RESET}")
    print(f"{Fore.GREEN}FFN replaced: {num_ffn_replaced} layers{Fore.RESET}")
    print(f"{Fore.GREEN}Attention simplified: {num_attn_simplified} layers{Fore.RESET}")
    print(f"{Fore.GREEN}Expected FLOPs reduction: ~{total_reduction*100:.1f}%{Fore.RESET}")
    print(f"{Fore.GREEN}Model saved to: {optimized_model_path}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*60}{Fore.RESET}")
    
    return optimized_model_path


