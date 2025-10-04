import argparse
import gc
import logging
import os
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all, select_non_overlapping_blocks

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def initialize_transform_smart(hidden_dim: int, device: str = 'cuda', init_scale: float = 0.01) -> torch.Tensor:
    """Initialize T as identity + small random perturbation."""
    
    print(f"Initializing transform with smart initialization (scale={init_scale})")
    
    # Start with identity
    T = torch.eye(hidden_dim, device=device, dtype=torch.float32)
    
    # Add small random perturbation
    perturbation = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32)
    perturbation = perturbation * init_scale
    
    T = T + perturbation
    
    # Check initialization quality
    identity_diff = (T - torch.eye(hidden_dim, device=device)).abs().mean().item()
    print(f"  Initial deviation from identity: {identity_diff:.6f}")
    
    return T


def combined_loss_function(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.9,
    norm_weight: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined loss with cosine similarity and norm preservation."""
    
    # Cosine similarity loss
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    cosine_loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
    
    # Norm preservation loss (normalized by target norm)
    pred_norms = pred.norm(dim=1)
    target_norms = target.norm(dim=1)
    norm_diff = (pred_norms - target_norms).pow(2)
    norm_loss = norm_diff.mean() / (target_norms.mean().pow(2) + 1e-8)
    
    # Combined loss
    total_loss = cosine_weight * cosine_loss + norm_weight * norm_loss
    
    # Return components for logging
    components = {
        'cosine': cosine_loss.item(),
        'norm': norm_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, components


def optimize_transform_improved(
    a1: torch.Tensor,  # Mi (MLP outputs)
    a2: torch.Tensor,  # Target (Li+n + Mi - Yi)
    hidden_dim: int,
    device: str = 'cuda',
    epochs: int = 10,
    batch_size: int = 1024,
    initial_lr: float = 5e-4,
    cosine_weight: float = 0.9,
    norm_weight: float = 0.1,
    init_scale: float = 0.01
) -> torch.Tensor:
    """Optimize T with improved techniques."""
    
    print(f"\n{Fore.CYAN}=== Improved ReplaceMe Optimization ==={Fore.RESET}")
    print(f"Data shape: {a1.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {initial_lr}")
    print(f"Loss weights - Cosine: {cosine_weight}, Norm: {norm_weight}")
    
    # Move data to device in chunks to avoid OOM
    n_samples = a1.shape[0]
    
    # Initialize T with smart initialization
    T = initialize_transform_smart(hidden_dim, device, init_scale)
    T.requires_grad_(True)
    
    # Optimizer with AdamW
    optimizer = torch.optim.AdamW(
        [T],
        lr=initial_lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Calculate steps for scheduler
    num_batches = (n_samples + batch_size - 1) // batch_size
    total_steps = epochs * num_batches
    
    # OneCycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        total_steps=total_steps,
        pct_start=0.3,  # 30% warm-up
        anneal_strategy='cos',
        final_div_factor=100  # End lr = initial_lr / 100
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warm-up steps: {int(total_steps * 0.3)}")
    
    # Training variables
    best_loss = float('inf')
    best_T = T.clone().detach()
    patience = 0
    max_patience = 3
    
    # Adaptive batch size schedule (optional)
    batch_size_schedule = [512, 768, 1024, 1024, 1536, 1536, 2048, 2048, 2048, 2048]
    
    for epoch in range(epochs):
        # Get batch size for this epoch
        current_batch_size = batch_size_schedule[epoch] if epoch < len(batch_size_schedule) else batch_size
        current_num_batches = (n_samples + current_batch_size - 1) // current_batch_size
        
        epoch_loss = 0.0
        epoch_cosine = 0.0
        epoch_norm = 0.0
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        
        # Progress bar for epoch
        pbar = tqdm(range(current_num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx in pbar:
            start_idx = batch_idx * current_batch_size
            end_idx = min(start_idx + current_batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data and move to device
            a1_batch = a1[batch_indices].to(device).to(torch.float32)
            a2_batch = a2[batch_indices].to(device).to(torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = a1_batch @ T
            
            # Compute combined loss
            loss, loss_components = combined_loss_function(
                pred, a2_batch,
                cosine_weight=cosine_weight,
                norm_weight=norm_weight
            )
            
            # Add small L2 regularization on deviation from identity
            identity = torch.eye(hidden_dim, device=device)
            reg_loss = 1e-6 * (T - identity).pow(2).mean()
            loss = loss + reg_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss_components['total']
            epoch_cosine += loss_components['cosine']
            epoch_norm += loss_components['norm']
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss_components['total']:.4f}",
                'lr': f"{current_lr:.1e}",
                'grad': f"{grad_norm:.3f}"
            })
            
            # Clear batch from GPU
            del a1_batch, a2_batch, pred
            torch.cuda.empty_cache()
        
        # Epoch statistics
        avg_loss = epoch_loss / current_num_batches
        avg_cosine = epoch_cosine / current_num_batches
        avg_norm = epoch_norm / current_num_batches
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.6f}, Cosine={avg_cosine:.6f}, "
              f"Norm={avg_norm:.6f}, BatchSize={current_batch_size}")
        
        # Early stopping check
        if avg_loss < best_loss - 1e-5:  # Significant improvement threshold
            best_loss = avg_loss
            best_T = T.clone().detach()
            patience = 0
            print(f"    {Fore.GREEN}New best! Loss improved by {best_loss - avg_loss:.6f}{Fore.RESET}")
        else:
            patience += 1
            print(f"    Patience: {patience}/{max_patience}")
            if patience >= max_patience and epoch >= 5:  # Don't stop too early
                print(f"    {Fore.YELLOW}Early stopping triggered{Fore.RESET}")
                break
    
    # Final evaluation
    print(f"\n{Fore.CYAN}Final Evaluation:{Fore.RESET}")
    
    with torch.no_grad():
        # Sample evaluation
        eval_samples = min(10000, n_samples)
        eval_indices = torch.randperm(n_samples)[:eval_samples]
        
        # Process in batches to avoid OOM
        total_cosine = 0.0
        total_norm_diff = 0.0
        
        for i in range(0, eval_samples, 1000):
            batch_indices = eval_indices[i:i+1000]
            a1_eval = a1[batch_indices].to(device).to(torch.float32)
            a2_eval = a2[batch_indices].to(device).to(torch.float32)
            
            # With learned T
            pred_T = a1_eval @ best_T
            pred_norm = pred_T / (pred_T.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = a2_eval / (a2_eval.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean().item()
            
            norm_diff = (pred_T.norm(dim=1) - a2_eval.norm(dim=1)).abs().mean().item()
            
            total_cosine += cosine_sim
            total_norm_diff += norm_diff
            
            del a1_eval, a2_eval, pred_T
            torch.cuda.empty_cache()
        
        avg_cosine = total_cosine / (eval_samples // 1000)
        avg_norm_diff = total_norm_diff / (eval_samples // 1000)
        
        print(f"  Cosine similarity: {avg_cosine:.6f}")
        print(f"  Average norm difference: {avg_norm_diff:.3f}")
        
        # Check T statistics
        identity = torch.eye(hidden_dim, device=device)
        T_diff = (best_T - identity).abs()
        print(f"  T deviation from identity - Mean: {T_diff.mean():.6f}, Max: {T_diff.max():.6f}")
        
        # Check T eigenvalues (stability check)
        try:
            eigenvalues = torch.linalg.eigvals(best_T)
            eigen_real = eigenvalues.real
            print(f"  T eigenvalues - Min: {eigen_real.min():.3f}, Max: {eigen_real.max():.3f}")
        except:
            print("  Could not compute eigenvalues")
    
    return best_T.cpu()


def improved_cosine_dist(
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
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    improved_lr: float = 5e-4,
    improved_cosine_weight: float = 0.9,
    improved_norm_weight: float = 0.1,
    improved_init_scale: float = 0.01,
    improved_epochs: int = 10,
    **kwargs
) -> str:
    """Improved ReplaceMe with better optimization techniques."""
    
    print(f"\n{Fore.CYAN}=== Improved ReplaceMe ==={Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    print(f"Dataset: {dataset}, Size: {dataset_size}")
    print(f"Improvements: OneCycle LR, Combined Loss, Smart Init")
    
    # Load model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"\n{Fore.YELLOW}Loading model...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    hidden_size = model.config.hidden_size
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Setup hooks for MLP activations
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
    # Collect activations (keep on CPU for memory efficiency)
    print(f"\n{Fore.YELLOW}Gathering activations...{Fore.RESET}")
    
    a1 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    cnt = 0
    for batch in tqdm(dataloader, desc="Gathering Activations", colour="blue"):
        inputs = tokenizer(
            batch, return_tensors="pt", padding="longest",
            max_length=max_length, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] 
            for i in range(model.config.num_hidden_layers)
        ]
        
        # Get relevant activations  
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape and compute target (ReplaceMe formulation)
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)
        hidden_states_i = hidden_states_i.view(-1, hidden_size)
        hidden_states_n = hidden_states_n.view(-1, hidden_size)
        
        batch_size_actual = hidden_states_mlp.shape[0]
        
        # Store on CPU
        a1[cnt:cnt+batch_size_actual] = hidden_states_mlp.cpu().to(torch.bfloat16)
        a2[cnt:cnt+batch_size_actual] = (hidden_states_n + hidden_states_mlp - hidden_states_i).cpu().to(torch.bfloat16)
        
        cnt += batch_size_actual
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n, outputs
        torch.cuda.empty_cache()
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    
    print(f"Collected {cnt} activation samples")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Clear model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Optimize transform with improvements
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = optimize_transform_improved(
        a1=a1,
        a2=a2,
        hidden_dim=hidden_size,
        device=device,
        epochs=improved_epochs,
        batch_size=1024,
        initial_lr=improved_lr,
        cosine_weight=improved_cosine_weight,
        norm_weight=improved_norm_weight,
        init_scale=improved_init_scale
    )
    
    # Clean up activations
    del a1, a2
    gc.collect()
    
    # Apply transformation and save
    print(f"\n{Fore.YELLOW}Applying transformation and saving model...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation
    layer_idx = start_id - num_layer - 1
    original_weight = model.model.layers[layer_idx].mlp.down_proj.weight
    
    print(f"Applying T to layer {layer_idx} down_proj")
    new_weight = (transform.T.to(torch.float64) @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[layer_idx].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Save
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path}_{layers_to_skip}_layers_{start_id}_{end_id}_improved".replace("/", "_")
    
    full_save_path = f"{save_path}_ReplaceMe_improved"
    model.save_pretrained(full_save_path)
    tokenizer.save_pretrained(full_save_path)
    
    print(f"{Fore.GREEN}Model saved to: {full_save_path}{Fore.RESET}")
    
    # Save metadata
    torch.save({
        'transform': transform,
        'method': 'improved_replaceme',
        'improvements': {
            'learning_rate': improved_lr,
            'cosine_weight': improved_cosine_weight,
            'norm_weight': improved_norm_weight,
            'init_scale': improved_init_scale
        }
    }, f"{full_save_path}_transform_data.pt")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return full_save_path