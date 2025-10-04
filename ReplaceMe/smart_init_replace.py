import argparse
import gc
import logging
import os
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

from .utils import get_calib_dataloader, truncate_model, seed_all, select_non_overlapping_blocks

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def compute_smart_initialization(
    Mi: torch.Tensor,  # MLP output (CPU)
    Li_n: torch.Tensor,  # Target output (CPU)
    Yi: torch.Tensor,  # Attention input (CPU)
    init_method: str = 'least_squares',
    device: str = 'cuda',
    batch_size: int = 10000
) -> torch.Tensor:
    """Compute smart initialization for transformation matrix."""
    
    print(f"\n{Fore.YELLOW}Computing smart initialization using {init_method}{Fore.RESET}")
    print(f"Data shape: Mi={Mi.shape}, Li_n={Li_n.shape}")
    
    if init_method == 'least_squares':
        T_init = compute_least_squares_init(Mi, Li_n, Yi, device, batch_size)
    elif init_method == 'pca_guided':
        T_init = compute_pca_guided_init(Mi, Li_n, Yi, device, batch_size)
    elif init_method == 'residual_aware':
        T_init = compute_residual_aware_init(Mi, Li_n, Yi, device, batch_size)
    else:
        print(f"{Fore.RED}Unknown init method {init_method}, using identity{Fore.RESET}")
        T_init = torch.eye(Mi.shape[1], dtype=torch.float32)
    
    # Validate initialization quality
    init_error = compute_init_quality(Mi, Li_n, Yi, T_init, device, batch_size)
    print(f"Initialization error: {init_error:.6f}")
    
    return T_init


def compute_least_squares_init(
    Mi: torch.Tensor,
    Li_n: torch.Tensor,
    Yi: torch.Tensor,
    device: str,
    batch_size: int
) -> torch.Tensor:
    """Compute least squares initialization in batches."""
    
    print("  Computing least squares initialization...")
    
    # For memory efficiency, accumulate in batches
    n_samples = Mi.shape[0]
    hidden_dim = Mi.shape[1]
    
    # Initialize accumulators on device
    A = torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32)
    B = torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32)
    
    # Process in batches
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="    Accumulating", leave=False):
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Move batch to GPU
        Mi_batch = Mi[start_idx:end_idx].to(device).to(torch.float32)
        Li_n_batch = Li_n[start_idx:end_idx].to(device).to(torch.float32)
        Yi_batch = Yi[start_idx:end_idx].to(device).to(torch.float32)
        
        # Target is Li_n - Yi (what MLP needs to produce)
        target_batch = Li_n_batch - Yi_batch
        
        # Accumulate A = Mi.T @ Mi and B = Mi.T @ target
        A += Mi_batch.T @ Mi_batch
        B += Mi_batch.T @ target_batch
        
        # Clear batch
        del Mi_batch, Li_n_batch, Yi_batch, target_batch
        torch.cuda.empty_cache()
    
    # Solve with regularization for numerical stability
    lambda_reg = 1e-4
    A_reg = A + lambda_reg * torch.eye(hidden_dim, device=device)
    
    try:
        # Try Cholesky decomposition (faster and more stable)
        L = torch.linalg.cholesky(A_reg)
        T_init = torch.cholesky_solve(B.T, L).T
        print("    Used Cholesky decomposition")
    except:
        # Fallback to pseudo-inverse
        T_init = torch.linalg.pinv(A_reg) @ B
        print("    Used pseudo-inverse")
    
    return T_init.cpu()


def compute_pca_guided_init(
    Mi: torch.Tensor,
    Li_n: torch.Tensor,
    Yi: torch.Tensor,
    device: str,
    batch_size: int,
    n_components: int = 1000
) -> torch.Tensor:
    """Compute PCA-guided initialization."""
    
    print("  Computing PCA-guided initialization...")
    
    # Sample subset for PCA (to avoid memory issues)
    n_samples = min(50000, Mi.shape[0])
    indices = torch.randperm(Mi.shape[0])[:n_samples]
    
    Mi_sample = Mi[indices].to(device).to(torch.float32)
    Li_n_sample = Li_n[indices].to(device).to(torch.float32)
    Yi_sample = Yi[indices].to(device).to(torch.float32)
    
    target_sample = Li_n_sample - Yi_sample
    
    # Compute PCA for Mi
    print("    Computing PCA for input...")
    Mi_centered = Mi_sample - Mi_sample.mean(dim=0, keepdim=True)
    U_mi, S_mi, V_mi = torch.svd_lowrank(Mi_centered, q=n_components)
    
    # Compute PCA for target
    print("    Computing PCA for target...")
    target_centered = target_sample - target_sample.mean(dim=0, keepdim=True)
    U_target, S_target, V_target = torch.svd_lowrank(target_centered, q=n_components)
    
    # Project data to PCA space
    Mi_proj = Mi_centered @ V_mi
    target_proj = target_centered @ V_target
    
    # Learn mapping in PCA space (much smaller problem)
    A_pca = Mi_proj.T @ Mi_proj + 1e-4 * torch.eye(n_components, device=device)
    B_pca = Mi_proj.T @ target_proj
    
    try:
        mapping_pca = torch.linalg.solve(A_pca, B_pca)
    except:
        mapping_pca = torch.linalg.pinv(A_pca) @ B_pca
    
    # Convert back to original space
    T_init = V_mi @ mapping_pca @ V_target.T
    
    # Add identity for unmodeled components
    hidden_dim = Mi.shape[1]
    T_full = torch.eye(hidden_dim, device=device, dtype=torch.float32)
    T_full[:T_init.shape[0], :T_init.shape[1]] += 0.1 * T_init  # Blend with identity
    
    # Cleanup
    del Mi_sample, Li_n_sample, Yi_sample, target_sample
    torch.cuda.empty_cache()
    
    return T_full.cpu()


def compute_residual_aware_init(
    Mi: torch.Tensor,
    Li_n: torch.Tensor,
    Yi: torch.Tensor,
    device: str,
    batch_size: int
) -> torch.Tensor:
    """Compute initialization that explicitly models attention residual."""
    
    print("  Computing residual-aware initialization...")
    
    # The key insight: MLP needs to predict Li_n - Yi
    # But also needs to preserve some of its original structure
    
    n_samples = Mi.shape[0]
    hidden_dim = Mi.shape[1]
    
    # Weight matrix for combining identity and learned transform
    alpha = 0.7  # How much to trust learned transform vs identity
    
    # Compute least squares solution for residual prediction
    A = torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32)
    B = torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32)
    
    # Also track correlation between Mi and residual
    correlation_accum = torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32)
    
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="    Computing residual mapping", leave=False):
        end_idx = min(start_idx + batch_size, n_samples)
        
        Mi_batch = Mi[start_idx:end_idx].to(device).to(torch.float32)
        Li_n_batch = Li_n[start_idx:end_idx].to(device).to(torch.float32)
        Yi_batch = Yi[start_idx:end_idx].to(device).to(torch.float32)
        
        residual = Li_n_batch - Yi_batch - Mi_batch  # What needs to be added to Mi
        
        # Accumulate
        A += Mi_batch.T @ Mi_batch
        B += Mi_batch.T @ (Li_n_batch - Yi_batch)
        correlation_accum += torch.abs(Mi_batch.T @ residual)
        
        del Mi_batch, Li_n_batch, Yi_batch, residual
        torch.cuda.empty_cache()
    
    # Normalize correlation
    correlation = correlation_accum / n_samples
    
    # Solve for transform
    A_reg = A + 1e-4 * torch.eye(hidden_dim, device=device)
    T_learned = torch.linalg.pinv(A_reg) @ B
    
    # Blend with identity based on correlation
    # High correlation means Mi already contains info about residual
    correlation_weight = torch.sigmoid(correlation.diagonal() - correlation.mean())
    
    T_init = torch.eye(hidden_dim, device=device, dtype=torch.float32)
    T_init = (1 - alpha) * T_init + alpha * T_learned
    
    # Apply correlation-based adjustment
    T_init = T_init * (1 + 0.1 * correlation_weight.unsqueeze(0))
    
    print(f"    Blend factor alpha={alpha:.2f}, mean correlation={correlation.mean():.4f}")
    
    return T_init.cpu()


def compute_init_quality(
    Mi: torch.Tensor,
    Li_n: torch.Tensor,
    Yi: torch.Tensor,
    T: torch.Tensor,
    device: str,
    batch_size: int
) -> float:
    """Evaluate initialization quality."""
    
    total_error = 0.0
    n_samples = min(10000, Mi.shape[0])  # Evaluate on subset
    indices = torch.randperm(Mi.shape[0])[:n_samples]
    
    T_gpu = T.to(device)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        Mi_batch = Mi[batch_indices].to(device).to(torch.float32)
        Li_n_batch = Li_n[batch_indices].to(device).to(torch.float32)
        Yi_batch = Yi[batch_indices].to(device).to(torch.float32)
        
        pred = Mi_batch @ T_gpu + Yi_batch
        
        # Cosine distance
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = Li_n_batch / (Li_n_batch.norm(dim=1, keepdim=True) + 1e-8)
        
        error = (1 - (pred_norm * target_norm).sum(dim=1).mean()).item()
        total_error += error * (end_idx - start_idx)
        
        del Mi_batch, Li_n_batch, Yi_batch, pred
        torch.cuda.empty_cache()
    
    del T_gpu
    torch.cuda.empty_cache()
    
    return total_error / n_samples


def optimize_with_smart_init(
    Mi: torch.Tensor,  # CPU tensor
    Li_n: torch.Tensor,  # CPU tensor
    Yi: torch.Tensor,  # CPU tensor
    T_init: torch.Tensor,  # Initial transformation
    device: str = 'cuda',
    epochs: int = 5,
    batch_size: int = 1024,
    loss_type: str = 'combined'
) -> torch.Tensor:
    """Fine-tune transformation starting from smart initialization."""
    
    print(f"\n{Fore.GREEN}Fine-tuning from smart initialization{Fore.RESET}")
    print(f"  Initial transform shape: {T_init.shape}")
    print(f"  Loss type: {loss_type}")
    
    # Move initialization to GPU and set up for optimization
    T = T_init.to(device).to(torch.float32).requires_grad_(True)
    
    # Learning rate schedule - start small to preserve good init
    lr_schedule = [1e-5, 5e-5, 1e-4, 5e-5, 1e-5]
    if epochs > len(lr_schedule):
        lr_schedule.extend([1e-5] * (epochs - len(lr_schedule)))
    
    optimizer = torch.optim.Adam([T], lr=lr_schedule[0])
    
    n_samples = Mi.shape[0]
    best_loss = float('inf')
    best_T = T.clone().detach()
    
    for epoch in range(epochs):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]
        
        epoch_loss = 0.0
        epoch_cosine_loss = 0.0
        epoch_mse_loss = 0.0
        
        indices = torch.randperm(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"  Epoch {epoch+1}/{epochs}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Move batch to GPU
            Mi_batch = Mi[batch_indices].to(device).to(torch.float32)
            Li_n_batch = Li_n[batch_indices].to(device).to(torch.float32)
            Yi_batch = Yi[batch_indices].to(device).to(torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = Mi_batch @ T + Yi_batch
            
            if loss_type == 'cosine':
                # Pure cosine loss
                pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
                target_norm = Li_n_batch / (Li_n_batch.norm(dim=1, keepdim=True) + 1e-8)
                loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
                
            elif loss_type == 'mse':
                # Pure MSE loss
                loss = ((pred - Li_n_batch) ** 2).mean()
                
            elif loss_type == 'combined':
                # Combined loss (weighted)
                pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
                target_norm = Li_n_batch / (Li_n_batch.norm(dim=1, keepdim=True) + 1e-8)
                cosine_loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
                mse_loss = ((pred - Li_n_batch) ** 2).mean()
                
                # Normalize and combine
                loss = 0.7 * cosine_loss + 0.3 * (mse_loss / 1000.0)  # Scale MSE
                epoch_cosine_loss += cosine_loss.item()
                epoch_mse_loss += mse_loss.item()
            
            # Add small L2 regularization to prevent overfitting
            reg_loss = 1e-5 * (T - T_init.to(device)).pow(2).mean()
            loss = loss + reg_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Clear GPU memory
            del Mi_batch, Li_n_batch, Yi_batch, pred
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if loss_type == 'combined':
            avg_cosine = epoch_cosine_loss / num_batches
            avg_mse = epoch_mse_loss / num_batches
            print(f"    Epoch {epoch+1}: Loss={avg_loss:.6f}, Cosine={avg_cosine:.6f}, MSE={avg_mse:.2f}, LR={lr_schedule[epoch]:.1e}")
        else:
            print(f"    Epoch {epoch+1}: Loss={avg_loss:.6f}, LR={lr_schedule[epoch]:.1e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T.clone().detach()
            print(f"      {Fore.GREEN}New best!{Fore.RESET}")
    
    return best_T.cpu()


def smart_init_cosine_dist(
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
    accurate: bool = False,
    init_method: str = 'least_squares',  # least_squares, pca_guided, residual_aware
    loss_type: str = 'combined',  # cosine, mse, combined
    fine_tune_epochs: int = 5,
    **kwargs
) -> str:
    """Main function for smart initialization ReplaceMe."""
    
    print(f"\n{Fore.CYAN}=== Smart Initialization ReplaceMe ==={Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    print(f"Initialization method: {init_method}")
    print(f"Loss type: {loss_type}")
    
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
    
    # Set up hooks
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
    
    # Gather activations (store on CPU)
    print(f"\n{Fore.YELLOW}Gathering activations...{Fore.RESET}")
    
    Mi = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu', pin_memory=True)
    Li_n = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu', pin_memory=True)
    Yi = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu', pin_memory=True)
    
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
        
        # Get activations
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape and store on CPU
        batch_size_actual = hidden_states_mlp.view(-1, hidden_size).shape[0]
        
        Mi[cnt:cnt+batch_size_actual] = hidden_states_mlp.view(-1, hidden_size).cpu().to(torch.bfloat16)
        Li_n[cnt:cnt+batch_size_actual] = hidden_states_n.view(-1, hidden_size).cpu().to(torch.bfloat16)
        Yi[cnt:cnt+batch_size_actual] = hidden_states_i.view(-1, hidden_size).cpu().to(torch.bfloat16)
        
        cnt += batch_size_actual
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        torch.cuda.empty_cache()
    
    # Trim to actual size
    Mi = Mi[:cnt]
    Li_n = Li_n[:cnt]
    Yi = Yi[:cnt]
    
    print(f"Collected {cnt} activation samples")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Clear model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compute smart initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T_init = compute_smart_initialization(
        Mi, Li_n, Yi,
        init_method=init_method,
        device=device,
        batch_size=10000
    )
    
    # Fine-tune from initialization
    transform = optimize_with_smart_init(
        Mi, Li_n, Yi,
        T_init=T_init,
        device=device,
        epochs=fine_tune_epochs,
        batch_size=1024,
        loss_type=loss_type
    )
    
    # Compare with identity baseline
    identity_error = compute_init_quality(
        Mi, Li_n, Yi,
        torch.eye(hidden_size, dtype=torch.float32),
        device, 10000
    )
    final_error = compute_init_quality(
        Mi, Li_n, Yi,
        transform,
        device, 10000
    )
    
    print(f"\n{Fore.CYAN}Transformation Summary:{Fore.RESET}")
    print(f"  Identity baseline error: {identity_error:.6f}")
    print(f"  Smart init error: {compute_init_quality(Mi, Li_n, Yi, T_init, device, 10000):.6f}")
    print(f"  Final error: {final_error:.6f}")
    print(f"  Improvement: {((identity_error - final_error) / identity_error * 100):.1f}%")
    
    # Clean up activations
    del Mi, Li_n, Yi
    gc.collect()
    
    # Apply transformation and save model
    print(f"\n{Fore.YELLOW}Applying transformation and saving model...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Prepare save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path}_{layers_to_skip}_layers_{start_id}_{end_id}_smart_{init_method}".replace("/", "_")
    
    # Apply transformation
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (transform.T.cpu().to(torch.float64) @ original_weight.cpu().to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Save
    full_save_path = f"{save_path}_ReplaceMe_smart"
    model.save_pretrained(full_save_path)
    tokenizer.save_pretrained(full_save_path)
    
    print(f"{Fore.GREEN}Model saved to: {full_save_path}{Fore.RESET}")
    
    # Save transform for analysis
    torch.save({
        'transform': transform,
        'init_transform': T_init,
        'init_method': init_method,
        'loss_type': loss_type,
        'identity_error': identity_error,
        'final_error': final_error
    }, f"{full_save_path}_transform_data.pt")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return full_save_path