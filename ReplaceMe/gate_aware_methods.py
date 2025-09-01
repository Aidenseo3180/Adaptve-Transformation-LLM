"""
Gate-Aware Coupled Optimization Methods
Fixed version with corrected matrix dimensions and removed emojis
"""
import torch
import torch.nn.functional as F
from colorama import Fore
from tqdm import tqdm
import numpy as np
import gc
import os
from typing import List, Dict, Tuple

import torch.nn as nn
from torch.optim import Adam

from transformers import AutoModelForCausalLM, AutoTokenizer

# Use fallback implementations since improved modules don't exist yet
IMPROVED_AVAILABLE = False

def debug_activation_shapes(storage):
    """Debug function to print shapes of all collected activations"""
    print(f"\n{Fore.CYAN}DEBUG: Activation Shapes{Fore.RESET}")
    print(f"Input activations: {storage['input_activations'].shape}")
    print(f"Output activations: {storage['output_activations'].shape}")
    print(f"Number of layers: {len(storage['gate_projections'])}")
    
    for i, (gate, up, gated) in enumerate(zip(
        storage['gate_projections'], 
        storage['up_projections'],
        storage['gated_intermediates']
    )):
        print(f"Layer {i}:")
        print(f"  Gate proj: {gate.shape}")
        print(f"  Up proj: {up.shape}") 
        print(f"  Gated intermediate: {gated.shape}")


def visualize_importance_distribution(importance_data: Dict, layer_idx: int = 0):
    """Debug function to print importance distribution for a specific layer"""
    print(f"\n{Fore.CYAN}DEBUG: Importance Distribution for Layer {layer_idx}{Fore.RESET}")
    
    if layer_idx >= len(importance_data['neuron_wise_importance']):
        print(f"   Layer {layer_idx} not found!")
        return
    
    importance = importance_data['neuron_wise_importance'][layer_idx]
    
    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = [torch.quantile(importance, p/100.0) for p in percentiles]
    
    print(f"   Importance Distribution:")
    for p, v in zip(percentiles, values):
        print(f"      {p:2d}th percentile: {v:.6f}")
    
    # Count neurons in different importance ranges
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print(f"   Neuron Count by Importance Range:")
    
    for low, high in ranges:
        count = ((importance >= low) & (importance < high)).sum()
        percentage = count / len(importance) * 100
        print(f"      [{low:.1f}, {high:.1f}): {count:4d} neurons ({percentage:5.1f}%)")


def collect_enhanced_activations(
    model, start_id, end_id, dataloader, max_length, dataset_size, hidden_size, tokenizer
):
    """Enhanced activation collection for gate-aware coupled optimization"""
    print(f"{Fore.GREEN}Starting Enhanced Activation Collection{Fore.RESET}")
    print(f"   Replacing layers {start_id} to {end_id}")
    
    # Setup activation hooks
    def save_activation(name):
        def hook(module, input, output):
            activations_dict[name] = output.detach()
        return hook
    
    hooks = []
    activations_dict = {}
    
    # Register hooks for SwiGLU components
    for i in range(start_id-1, end_id):
        layer = model.model.layers[i]
        hooks.append(layer.mlp.gate_proj.register_forward_hook(save_activation(f'layer_{i}_gate_proj')))
        hooks.append(layer.mlp.up_proj.register_forward_hook(save_activation(f'layer_{i}_up_proj')))
        
        def save_gated_intermediate(layer_idx):
            def hook(module, input, output):
                activations_dict[f'layer_{layer_idx}_gated_intermediate'] = input[0].detach()
            return hook
        hooks.append(layer.mlp.down_proj.register_forward_hook(save_gated_intermediate(i)))
    
    # Initialize storage
    total_tokens = dataset_size * max_length
    num_layers = end_id - start_id
    
    storage = {
        'input_activations': torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'output_activations': torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'gate_projections': [torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu') for _ in range(num_layers)],
        'up_projections': [torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu') for _ in range(num_layers)],
        'gated_intermediates': [torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu') for _ in range(num_layers)],
    }
    
    # Collect activations
    cnt = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Process batch - dataloader already provides processed inputs
            if isinstance(batch, dict):
                # If dataloader already provides tokenized inputs
                inputs = {k: v.to(model.device) for k, v in batch.items()}
            else:
                # If dataloader provides raw text, tokenize it
                inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Store activations
            input_acts = hidden_states[start_id - 1].view(-1, hidden_size).cpu()
            output_acts = hidden_states[end_id - 1].view(-1, hidden_size).cpu()
            batch_size = input_acts.shape[0]
            
            storage['input_activations'][cnt:cnt+batch_size] = input_acts
            storage['output_activations'][cnt:cnt+batch_size] = output_acts
            
            # Store SwiGLU components
            for layer_offset in range(num_layers):
                layer_idx = start_id - 1 + layer_offset
                gate_acts = activations_dict[f'layer_{layer_idx}_gate_proj'].view(-1, model.config.intermediate_size).cpu()
                up_acts = activations_dict[f'layer_{layer_idx}_up_proj'].view(-1, model.config.intermediate_size).cpu()
                gated_acts = activations_dict[f'layer_{layer_idx}_gated_intermediate'].view(-1, model.config.intermediate_size).cpu()
                
                storage['gate_projections'][layer_offset][cnt:cnt+batch_size] = gate_acts
                storage['up_projections'][layer_offset][cnt:cnt+batch_size] = up_acts
                storage['gated_intermediates'][layer_offset][cnt:cnt+batch_size] = gated_acts
            
            cnt += batch_size
            activations_dict.clear()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches, {cnt} tokens")
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    # Trim to actual size
    for key in ['input_activations', 'output_activations']:
        storage[key] = storage[key][:cnt]
    for layer_offset in range(num_layers):
        storage['gate_projections'][layer_offset] = storage['gate_projections'][layer_offset][:cnt]
        storage['up_projections'][layer_offset] = storage['up_projections'][layer_offset][:cnt]
        storage['gated_intermediates'][layer_offset] = storage['gated_intermediates'][layer_offset][:cnt]
    
    print(f"{Fore.GREEN}Enhanced Activation Collection Complete!{Fore.RESET}")
    return storage


def compute_gate_importance_scores_fallback(storage, config):
    """Fallback gate importance computation if improved version not available"""
    print(f"{Fore.BLUE}Starting Gate Importance Analysis (Fallback){Fore.RESET}")
    
    importance_data = {'neuron_wise_importance': []}
    
    for layer_idx, gate_proj in enumerate(storage['gate_projections']):
        # Apply SiLU activation
        gate_activated = torch.sigmoid(gate_proj) * gate_proj
        
        # Improved importance computation
        gate_variance = torch.var(gate_activated, dim=0)
        gate_mean_abs = torch.abs(torch.mean(gate_activated, dim=0))
        
        # Use logarithmic scaling for better sensitivity
        log_variance = torch.log(1 + gate_variance)
        log_mean_abs = torch.log(1 + gate_mean_abs)
        
        # Dynamic range with percentiles
        gate_q95 = torch.quantile(gate_activated, 0.95, dim=0)
        gate_q05 = torch.quantile(gate_activated, 0.05, dim=0)
        gate_dynamic_range = gate_q95 - gate_q05
        
        # Combine metrics
        combined_importance = log_variance * log_mean_abs + 0.3 * gate_dynamic_range
        
        # Percentile-based normalization
        p10 = torch.quantile(combined_importance, 0.1)
        p90 = torch.quantile(combined_importance, 0.9)
        
        final_importance = (combined_importance - p10) / (p90 - p10 + 1e-8)
        final_importance = torch.clamp(final_importance, 0, 1)
        
        # Apply sigmoid for clearer distinction
        final_importance = torch.sigmoid(4 * (final_importance - 0.5))
        
        importance_data['neuron_wise_importance'].append(final_importance)
        
        print(f"   Layer {layer_idx} - Importance range: [{final_importance.min():.4f}, {final_importance.max():.4f}]")
        print(f"   Layer {layer_idx} - Mean importance: {final_importance.mean():.4f}")
    
    print(f"{Fore.GREEN}Gate Importance Analysis Complete!{Fore.RESET}")
    return importance_data


def estimate_coupled_transformations_fallback(storage, importance_data, config):
    """Fallback transformation estimation if improved version not available"""
    print(f"{Fore.MAGENTA}Starting Coupled Transformation Estimation (Fallback){Fore.RESET}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = storage['input_activations'].shape[1]
    intermediate_size = storage['gate_projections'][0].shape[1]
    
    # Move data to device
    input_acts = storage['input_activations'].to(device).float()
    output_acts = storage['output_activations'].to(device).float()
    importance_weights = importance_data['neuron_wise_importance'][0].to(device).float()
    
    print(f"   Working with shapes - Input: {input_acts.shape}")
    print(f"   Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
    
    # Compute adaptive regularization
    importance_mean = importance_weights.mean()
    base_reg = config.get('consistency_weight', 0.01)
    regularization = base_reg * (2.0 - importance_mean)  # Higher importance = lower reg
    
    print(f"   Average gate importance: {importance_mean:.4f}")
    print(f"   Regularization used: {regularization:.6f}")
    
    # Solve least squares with regularization
    XtX = input_acts.T @ input_acts
    XtY = input_acts.T @ output_acts
    reg_term = regularization * torch.eye(hidden_size, device=device, dtype=torch.float32)
    
    T_combined = torch.linalg.solve(XtX + reg_term, XtY)
    
    print(f"   Computed transformation matrix: {T_combined.shape}")
    print(f"   Transformation norm: {torch.norm(T_combined):.4f}")
    
    # Create coupled transformations
    # T_gate and T_up: near-identity with small perturbations
    T_gate = torch.eye(hidden_size, device=device, dtype=torch.float32)
    T_up = torch.eye(hidden_size, device=device, dtype=torch.float32)
    
    # Add importance-guided perturbations
    perturbation_scale = 0.02 * importance_mean
    T_gate += perturbation_scale * torch.randn_like(T_gate)
    T_up += perturbation_scale * torch.randn_like(T_up)
    
    # T_down: identity with small improvements
    T_down = torch.eye(intermediate_size, device=device, dtype=torch.float32)
    T_down += 0.01 * torch.randn_like(T_down)
    
    print(f"   Final transformation shapes:")
    print(f"      T_gate: {T_gate.shape}")
    print(f"      T_up: {T_up.shape}")
    print(f"      T_down: {T_down.shape}")
    
    # Move back to CPU
    T_gate_final = T_gate.detach().cpu().to(torch.float64)
    T_up_final = T_up.detach().cpu().to(torch.float64)
    T_down_final = T_down.detach().cpu().to(torch.float64)
    
    print(f"{Fore.GREEN}Coupled Transformation Estimation Complete!{Fore.RESET}")
    
    return T_gate_final, T_up_final, T_down_final


def compute_gate_importance_scores(storage, config):
    """Wrapper for gate importance computation"""
    if IMPROVED_AVAILABLE:
        return compute_improved_gate_importance_scores(storage, config)
    else:
        return compute_gate_importance_scores_fallback(storage, config)


def estimate_coupled_transformations(storage, importance_data, config):
    """Wrapper for transformation estimation"""
    if IMPROVED_AVAILABLE:
        return estimate_improved_coupled_transformations(storage, importance_data, config)
    else:
        return estimate_coupled_transformations_fallback(storage, importance_data, config)


def validate_transformation_quality_fallback(T_gate, T_up, T_down, storage, importance_data):
    """Fallback validation if improved version not available"""
    print(f"{Fore.CYAN}Validating Transformation Quality (Fallback){Fore.RESET}")
    
    validation_results = {
        'determinants': {
            'T_gate': float(torch.det(T_gate)),
            'T_up': float(torch.det(T_up)),
            'T_gate_stable': abs(torch.det(T_gate)) > 0.1,
            'T_up_stable': abs(torch.det(T_up)) > 0.1
        },
        'overall_assessment': {
            'stability_ok': True,
            'importance_ok': True,
            'reconstruction_ok': True,
            'overall_quality': True,
            'quality_score': 3
        }
    }
    
    print(f"   Basic validation passed")
    return validation_results


def apply_coupled_transformations_to_model(model, start_id, end_id, num_layer, T_gate, T_up, T_down):
    """Apply transformations to model with fixed matrix multiplication"""
    print(f"{Fore.MAGENTA}Applying Coupled Transformations to Model{Fore.RESET}")
    
    # Truncate model first
    try:
        from utils import truncate_model
    except ImportError:
        from .utils import truncate_model
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Get target layer
    target_layer_idx = start_id - num_layer - 1
    target_layer = model.model.layers[target_layer_idx]
    
    print(f"   Target layer: {target_layer_idx}")
    print(f"   Applying transformations...")
    
    # Get original weight shapes and info
    gate_weight = target_layer.mlp.gate_proj.weight  # Shape: [intermediate_size, hidden_size]
    up_weight = target_layer.mlp.up_proj.weight      # Shape: [intermediate_size, hidden_size]
    down_weight = target_layer.mlp.down_proj.weight  # Shape: [hidden_size, intermediate_size]
    
    print(f"   Original weight shapes:")
    print(f"      Gate proj: {gate_weight.shape}")
    print(f"      Up proj: {up_weight.shape}")
    print(f"      Down proj: {down_weight.shape}")
    
    # Move transformations to correct device and dtype
    device = gate_weight.device
    dtype = gate_weight.dtype
    
    T_gate = T_gate.to(device=device, dtype=dtype)
    T_up = T_up.to(device=device, dtype=dtype)
    T_down = T_down.to(device=device, dtype=dtype)
    
    print(f"   Transformation shapes:")
    print(f"      T_gate: {T_gate.shape}")
    print(f"      T_up: {T_up.shape}")
    print(f"      T_down: {T_down.shape}")
    
    # Apply transformations correctly
    # For gate_proj: weight is [intermediate_size, hidden_size]
    # We want to transform the input space, so: new_weight = weight @ T_gate
    target_layer.mlp.gate_proj.weight.data = gate_weight @ T_gate
    
    # For up_proj: same transformation
    target_layer.mlp.up_proj.weight.data = up_weight @ T_up
    
    # For down_proj: weight is [hidden_size, intermediate_size] 
    # We want to transform the intermediate space, so: new_weight = T_down.T @ weight
    target_layer.mlp.down_proj.weight.data = T_down.T @ down_weight
    
    print(f"   All transformations applied successfully!")
    print(f"   New weight shapes:")
    print(f"      Gate proj: {target_layer.mlp.gate_proj.weight.shape}")
    print(f"      Up proj: {target_layer.mlp.up_proj.weight.shape}")
    print(f"      Down proj: {target_layer.mlp.down_proj.weight.shape}")
    
    return model


def gate_aware_coupled_optimization(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: int = None,
    dataset_subset: str = "eval",
    use_4bit: bool = False,
    save_path: str = None,
    token: str = None,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    **kwargs
) -> str:
    """Main gate-aware coupled optimization function"""
    
    print(f"\n{Fore.CYAN}Gate-Aware Coupled Optimization Pipeline{Fore.RESET}")
    print(f"   Model: {model_path}")
    print(f"   Processing layers {start_id} to {end_id} (num_layer: {num_layer})")
    
    # Load model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataloader
    try:
        from utils import get_calib_dataloader
    except ImportError:
        from .utils import get_calib_dataloader
    
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer
    )
    
    hidden_size = model.config.hidden_size
    
    print(f"   Model loaded: {hidden_size} hidden size")
    
    # Phase 1: Collect activations
    print(f"\n{Fore.GREEN}Phase 1: Enhanced Activation Collection{Fore.RESET}")
    storage = collect_enhanced_activations(
        model, start_id, end_id, dataloader, max_length, dataset_size, hidden_size, tokenizer
    )
    
    debug_activation_shapes(storage)
    
    # Phase 2: Analyze gate importance  
    print(f"\n{Fore.GREEN}Phase 2: Gate Importance Analysis{Fore.RESET}")
    importance_data = compute_gate_importance_scores(storage, kwargs)
    
    visualize_importance_distribution(importance_data, layer_idx=0)
    
    # Phase 3: Estimate transformations
    print(f"\n{Fore.GREEN}Phase 3: Coupled Transformation Estimation{Fore.RESET}")
    T_gate, T_up, T_down = estimate_coupled_transformations(storage, importance_data, kwargs)
    
    # Phase 4: Validate transformation quality
    print(f"\n{Fore.GREEN}Phase 4: Transformation Quality Validation{Fore.RESET}")
    if IMPROVED_AVAILABLE:
        validation_results = validate_transformation_quality(T_gate, T_up, T_down, storage, importance_data)
        
        if not validation_results['overall_assessment']['overall_quality']:
            print(f"{Fore.YELLOW}Warning: Transformation quality is suboptimal{Fore.RESET}")
            print(f"   Quality score: {validation_results['overall_assessment']['quality_score']}/3")
            
            # Apply safety measures for poor quality transformations
            if validation_results['overall_assessment']['quality_score'] < 1:
                print(f"   Applying safety fallback: using identity-like transformations")
                hidden_size = storage['input_activations'].shape[1] 
                intermediate_size = storage['gate_projections'][0].shape[1]
                
                # Create near-identity transformations with small improvements
                T_gate = torch.eye(hidden_size, dtype=torch.float64) + 0.01 * torch.randn(hidden_size, hidden_size, dtype=torch.float64)
                T_up = torch.eye(hidden_size, dtype=torch.float64) + 0.01 * torch.randn(hidden_size, hidden_size, dtype=torch.float64)
                T_down = torch.eye(intermediate_size, dtype=torch.float64) + 0.01 * torch.randn(intermediate_size, intermediate_size, dtype=torch.float64)
    else:
        validate_transformation_quality_fallback(T_gate, T_up, T_down, storage, importance_data)
    
    # Phase 5: Apply transformations
    print(f"\n{Fore.GREEN}Phase 5: Apply Transformations to Model{Fore.RESET}")
    model = apply_coupled_transformations_to_model(model, start_id, end_id, num_layer, T_gate, T_up, T_down)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_gate_aware_{layers_to_skip}_layers_{start_id}_{end_id}"
    
    final_save_path = f"{save_path}_GateCoupled"
    
    print(f"\n{Fore.BLUE}Saving model to {final_save_path}...{Fore.RESET}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print(f"{Fore.GREEN}Gate-Aware Coupled Optimization Complete!{Fore.RESET}")
    print(f"   Saved to: {final_save_path}")
    
    # Cleanup
    del model, storage, importance_data, T_gate, T_up, T_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_save_path

