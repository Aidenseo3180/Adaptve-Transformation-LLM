import gc
import logging
import math
import os
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from .utils import get_calib_dataloader, seed_all, truncate_model, select_non_overlapping_blocks

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed_all()


class TaskSpecificDataset(Dataset):
    """Dataset for task-specific optimization."""
    def __init__(self, a1, a2, a3=None, indices=None):
        if indices is not None:
            self.a1 = a1[indices]
            self.a2 = a2[indices]
            self.a3 = a3[indices] if a3 is not None else None
        else:
            self.a1 = a1
            self.a2 = a2
            self.a3 = a3
    
    def __len__(self):
        return len(self.a1)
    
    def __getitem__(self, idx):
        if self.a3 is not None:
            return self.a1[idx], self.a2[idx], self.a3[idx]
        return self.a1[idx], self.a2[idx], torch.zeros_like(self.a1[idx])


def compute_generation_loss(pred, target, pred_norm=None, target_norm=None):
    """Loss optimized for text generation tasks."""
    if pred_norm is None:
        pred_norm = F.normalize(pred, dim=-1)
    if target_norm is None:
        target_norm = F.normalize(target, dim=-1)
    
    # Primary: cosine similarity for fluency
    cosine_loss = 1 - (pred_norm * target_norm).sum(-1).mean()
    
    # Secondary: preserve local patterns (adjacent token relationships)
    if pred.shape[0] > 100:
        # Sample pairs of adjacent positions
        idx1 = torch.randperm(pred.shape[0] - 1)[:100]
        idx2 = idx1 + 1
        
        # Local coherence loss
        pred_diff = pred[idx2] - pred[idx1]
        target_diff = target[idx2] - target[idx1]
        local_loss = F.mse_loss(pred_diff, target_diff)
        
        total_loss = cosine_loss + 0.05 * local_loss
    else:
        total_loss = cosine_loss
    
    return total_loss, cosine_loss.item()


def compute_reasoning_loss(pred, target, pred_norm=None, target_norm=None):
    """Loss optimized for reasoning/QA tasks."""
    if pred_norm is None:
        pred_norm = F.normalize(pred, dim=-1)
    if target_norm is None:
        target_norm = F.normalize(target, dim=-1)
    
    # Primary: exact matching for factual accuracy
    cosine_loss = 1 - (pred_norm * target_norm).sum(-1).mean()
    
    # Secondary: preserve discriminative features
    # Maximize variance to keep features distinct
    pred_var = pred.var(dim=0).mean()
    target_var = target.var(dim=0).mean()
    var_loss = F.mse_loss(pred_var, target_var)
    
    # Tertiary: preserve top eigenvalues (important features)
    if pred.shape[0] > 1000:
        # Sample for efficiency
        sample_idx = torch.randperm(pred.shape[0])[:1000]
        pred_sample = pred[sample_idx]
        target_sample = target[sample_idx]
        
        # Compute covariance
        pred_cov = pred_sample.T @ pred_sample / pred_sample.shape[0]
        target_cov = target_sample.T @ target_sample / target_sample.shape[0]
        
        # Top eigenvalue preservation
        try:
            pred_eigvals = torch.linalg.eigvalsh(pred_cov)[-10:]
            target_eigvals = torch.linalg.eigvalsh(target_cov)[-10:]
            eigen_loss = F.mse_loss(pred_eigvals, target_eigvals)
        except:
            eigen_loss = torch.tensor(0.0, device=pred.device)
    else:
        eigen_loss = torch.tensor(0.0, device=pred.device)
    
    total_loss = cosine_loss + 0.1 * var_loss + 0.01 * eigen_loss
    
    return total_loss, cosine_loss.item()


def learn_task_specific_transform(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: Optional[torch.Tensor] = None,
    hidden_dim: int = 4096,
    mode: str = "balanced",
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = 'cuda',
    debug: bool = True
) -> torch.Tensor:
    """
    Learn transformation matrix optimized for specific task.
    
    Args:
        mode: 'generation', 'reasoning', or 'balanced'
    """
    print(f"\n{Fore.CYAN}Learning Task-Specific Transform{Fore.RESET}")
    print(f"  Mode: {Fore.YELLOW}{mode}{Fore.RESET}")
    print(f"  Input shape: {a1.shape}")
    print(f"  Device: {device}")
    
    # Mode-specific hyperparameters
    if mode == "generation":
        spectral_weight = 0.001  # Low regularization for fluency
        condition_weight = 0.0001
        orth_weight = 0.00001
        print(f"  Optimizing for {Fore.GREEN}text generation{Fore.RESET}")
        print(f"  Focus: Perplexity minimization, fluency preservation")
    elif mode == "reasoning":
        spectral_weight = 0.02  # High regularization for accuracy
        condition_weight = 0.002
        orth_weight = 0.0002
        print(f"  Optimizing for {Fore.BLUE}reasoning/QA{Fore.RESET}")
        print(f"  Focus: Accuracy maximization, feature discrimination")
    else:  # balanced
        spectral_weight = 0.01
        condition_weight = 0.001
        orth_weight = 0.0001
        print(f"  Using {Fore.MAGENTA}balanced{Fore.RESET} optimization")
    
    # Keep data on CPU
    a1 = a1.cpu().float()
    a2 = a2.cpu().float()
    if a3 is not None:
        a3 = a3.cpu().float()
    
    # Smart initialization based on mode
    print(f"\n{Fore.YELLOW}Initializing transformation matrix...{Fore.RESET}")
    try:
        subset_size = min(20000, a1.shape[0])
        indices = torch.randperm(a1.shape[0])[:subset_size]
        
        if mode == "generation":
            # For generation: preserve more of original structure
            a1_subset = a1[indices].to(device)
            a2_subset = a2[indices].to(device)
            U, S, V = torch.svd(a1_subset.T @ a2_subset)
            # Softer transformation
            S_soft = torch.sqrt(S)
            W = U @ torch.diag(S_soft / S_soft.max()) @ V.T
            print(f"  Generation-specific initialization (softer transform)")
            
        elif mode == "reasoning":
            # For reasoning: more aggressive transformation
            a1_subset = a1[indices].to(device)
            a2_subset = a2[indices].to(device)
            U, S, V = torch.svd(a1_subset.T @ a2_subset)
            W = U @ V.T
            print(f"  Reasoning-specific initialization (orthogonal transform)")
            
        else:
            # Balanced: standard SVD
            a1_subset = a1[indices].to(device)
            a2_subset = a2[indices].to(device)
            U, S, V = torch.svd(a1_subset.T @ a2_subset)
            W = U @ V.T
            print(f"  Balanced initialization")
        
        del a1_subset, a2_subset
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"{Fore.RED}  Initialization failed: {e}, using identity{Fore.RESET}")
        W = torch.eye(hidden_dim, device=device)
    
    W.requires_grad_(True)
    
    # Optimizer setup
    optimizer = torch.optim.Adam([W], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=debug
    )
    
    # Create dataset and dataloader
    dataset = TaskSpecificDataset(a1, a2, a3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Tracking
    best_W = W.clone().detach()
    best_loss = float('inf')
    
    # Training loop
    print(f"\n{Fore.CYAN}Starting optimization ({epochs} epochs)...{Fore.RESET}")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_primary_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, colour="green")
        for batch_idx, batch_data in enumerate(pbar):
            X, Y, Z = batch_data
            
            # Move to device
            X = X.to(device).float()
            Y = Y.to(device).float()
            Z = Z.to(device).float() if a3 is not None else torch.zeros_like(X)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = X @ W
            if a3 is not None and Z.sum() != 0:
                pred = pred + Z
            
            # Compute task-specific loss
            pred_norm = F.normalize(pred, dim=-1)
            Y_norm = F.normalize(Y, dim=-1)
            
            if mode == "generation":
                task_loss, primary_loss = compute_generation_loss(pred, Y, pred_norm, Y_norm)
            elif mode == "reasoning":
                task_loss, primary_loss = compute_reasoning_loss(pred, Y, pred_norm, Y_norm)
            else:  # balanced
                gen_loss, gen_primary = compute_generation_loss(pred, Y, pred_norm, Y_norm)
                rea_loss, rea_primary = compute_reasoning_loss(pred, Y, pred_norm, Y_norm)
                task_loss = 0.5 * gen_loss + 0.5 * rea_loss
                primary_loss = 0.5 * gen_primary + 0.5 * rea_primary
            
            # Regularization (computed less frequently)
            if batch_idx % 10 == 0:
                try:
                    U_w, S_w, V_w = torch.svd(W)
                    spectral_loss = torch.norm(S_w - 1.0, p=2) / math.sqrt(hidden_dim)
                    condition_number = S_w.max() / (S_w.min() + 1e-8)
                    condition_loss = torch.log(condition_number + 1.0)
                    orth_loss = torch.norm(W @ W.T - torch.eye(hidden_dim, device=device), 'fro')
                    
                    total_loss = (
                        task_loss + 
                        spectral_weight * spectral_loss +
                        condition_weight * condition_loss +
                        orth_weight * orth_loss
                    )
                    
                    if debug and batch_idx == 0 and epoch % 20 == 0:
                        print(f"\n  Regularization - Spectral: {spectral_loss.item():.4f}, "
                              f"Condition: {condition_loss.item():.4f}, Orth: {orth_loss.item():.4f}")
                        
                except:
                    total_loss = task_loss
            else:
                total_loss = task_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            epoch_primary_losses.append(primary_loss)
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'primary': f'{primary_loss:.4f}'
            })
            
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_primary = np.mean(epoch_primary_losses)
        scheduler.step(avg_primary)
        
        if avg_primary < best_loss:
            best_loss = avg_primary
            best_W = W.clone().detach()
        
        if debug and epoch % 10 == 0:
            print(f"\n{Fore.GREEN}Epoch {epoch+1}/{epochs}:{Fore.RESET}")
            print(f"  Mode: {mode}")
            print(f"  Average total loss: {avg_loss:.6f}")
            print(f"  Average primary loss: {avg_primary:.6f}")
            print(f"  Best primary loss: {best_loss:.6f}")
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n{Fore.GREEN}Optimization complete!{Fore.RESET}")
    print(f"  Final best loss ({mode} mode): {best_loss:.6f}")
    
    return best_W.cpu().to(torch.float64)


def apply_task_specific_transform(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    save_path: Optional[str] = None,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    token: Optional[str] = None,
    task_mode: str = "balanced",
    epochs: int = 100,
    optim_batch_size: int = 1024,
    learning_rate: float = 1e-3,
    use_accurate: bool = False,
    num_A: int = 1,
    merge_consecutive: bool = False,
    distances_path: str = "./distances.pth",
    **kwargs
) -> str:
    """
    Apply task-specific optimized transformation to model.
    
    Args:
        task_mode: 'generation', 'reasoning', or 'balanced'
    """
    print(f"\n{Fore.CYAN}{'='*70}{Fore.RESET}")
    print(f"{Fore.CYAN}Task-Specific Transform - Mode: {task_mode.upper()}{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*70}{Fore.RESET}")
    
    if task_mode == "generation":
        print(f"{Fore.GREEN}Optimizing for text generation tasks{Fore.RESET}")
        print(f"  Expected: Lower perplexity, better fluency")
        print(f"  Trade-off: Potentially lower QA accuracy")
    elif task_mode == "reasoning":
        print(f"{Fore.BLUE}Optimizing for reasoning/QA tasks{Fore.RESET}")
        print(f"  Expected: Higher accuracy on factual tasks")
        print(f"  Trade-off: Potentially higher perplexity")
    else:
        print(f"{Fore.MAGENTA}Using balanced optimization{Fore.RESET}")
        print(f"  Expected: Balanced performance across tasks")
    
    print(f"\nModel: {model_path}")
    print(f"Dataset: {dataset}, size: {dataset_size}")
    
    # Load model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    # Get calibration dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Setup hooks
    print(f"\n{Fore.YELLOW}Setting up activation collection...{Fore.RESET}")
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(
            save_mlp_activation(f'layer_{i}_mlp')
        ))
    
    # Allocate CPU tensors
    print(f"\n{Fore.YELLOW}Allocating memory for activations...{Fore.RESET}")
    estimated_tokens = dataset_size * max_length
    a1 = torch.empty((estimated_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((estimated_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a3 = torch.empty((estimated_tokens, hidden_size), dtype=torch.bfloat16, device='cpu') if use_accurate else None
    
    # Collect activations
    print(f"\n{Fore.YELLOW}Collecting activations...{Fore.RESET}")
    cnt = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering Activations", colour="cyan")):
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}: collected {cnt} tokens")
        
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
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
        
        # Extract and move to CPU
        h_mlp = hidden_states_mlp.view(-1, hidden_size).cpu().to(torch.bfloat16)
        h_in = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).cpu().to(torch.bfloat16)
        h_out = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).cpu().to(torch.bfloat16)
        
        batch_size_actual = h_mlp.shape[0]
        if cnt + batch_size_actual > estimated_tokens:
            print(f"{Fore.YELLOW}  Reached token limit{Fore.RESET}")
            break
        
        a1[cnt:cnt+batch_size_actual] = h_mlp
        if use_accurate:
            a2[cnt:cnt+batch_size_actual] = h_out
            a3[cnt:cnt+batch_size_actual] = h_in - h_mlp
        else:
            a2[cnt:cnt+batch_size_actual] = h_out + h_mlp - h_in
        
        cnt += batch_size_actual
        
        # Clear cache
        del hidden_states_mlp, h_in, h_out
        torch.cuda.empty_cache()
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if use_accurate and a3 is not None:
        a3 = a3[:cnt]
    
    print(f"\nCollected {cnt} activation vectors")
    
    # Subsample if too large
    max_tokens = 3000000
    if cnt > max_tokens:
        print(f"{Fore.YELLOW}Subsampling from {cnt} to {max_tokens} tokens{Fore.RESET}")
        indices = torch.randperm(cnt)[:max_tokens]
        a1 = a1[indices]
        a2 = a2[indices]
        if a3 is not None:
            a3 = a3[indices]
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Learn transformation
    print(f"\n{Fore.CYAN}Learning task-specific transformation...{Fore.RESET}")
    
    optim_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transform = learn_task_specific_transform(
        a1, a2, a3,
        hidden_dim=hidden_size,
        mode=task_mode,
        epochs=epochs,
        batch_size=optim_batch_size,
        lr=learning_rate,
        device=optim_device,
        debug=True
    )
    
    # Clean up
    del a1, a2
    if a3 is not None:
        del a3
    gc.collect()
    
    # Apply transformation
    print(f"\n{Fore.YELLOW}Applying transformation to model...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (transform.T @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    # Check for NaN/Inf
    if torch.isnan(new_weight).any() or torch.isinf(new_weight).any():
        print(f"{Fore.RED}Warning: NaN/Inf detected, using original weights{Fore.RESET}")
        new_weight = original_weight
    else:
        print(f"{Fore.GREEN}Transformation applied successfully{Fore.RESET}")
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight = nn.Parameter(new_weight)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_task_{task_mode}_{start_id}_{end_id}"
    
    save_path = f"{save_path}_task_{task_mode}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n{Fore.GREEN}Model saved to: {save_path}{Fore.RESET}")
    print(f"{Fore.GREEN}Task mode: {task_mode}{Fore.RESET}")
    
    # Final cleanup
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path