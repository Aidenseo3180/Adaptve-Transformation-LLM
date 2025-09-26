import gc
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from .utils import get_calib_dataloader, seed_all, truncate_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class ActivationDataset(Dataset):
    """Dataset for batch processing of activations."""
    def __init__(self, a1, a2, a3=None):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def __len__(self):
        return len(self.a1)
    
    def __getitem__(self, idx):
        if self.a3 is not None:
            return self.a1[idx], self.a2[idx], self.a3[idx]
        return self.a1[idx], self.a2[idx], torch.zeros_like(self.a1[idx])


def spectral_regularized_transform_batched(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: Optional[torch.Tensor] = None,
    hidden_dim: int = 4096,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 1024,
    spectral_weight: float = 0.01,
    condition_weight: float = 0.001,
    orth_weight: float = 0.0001,
    use_residual: bool = False,
    device: str = 'cuda',
    debug: bool = True
) -> torch.Tensor:
    """
    Learn transformation matrix with spectral regularization using batch processing.
    
    All data stays on CPU, only small batches move to GPU for optimization.
    """
    if debug:
        print(f"{Fore.CYAN}Starting spectral regularized transform optimization (Batched){Fore.RESET}")
        print(f"  Input shape: {a1.shape}")
        print(f"  Target shape: {a2.shape}")
        print(f"  Using residual: {a3 is not None}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
    
    # Keep data on CPU
    a1 = a1.cpu().float()
    a2 = a2.cpu().float()
    if a3 is not None:
        a3 = a3.cpu().float()
    
    # Smart initialization using subset of data
    print(f"{Fore.YELLOW}Initializing transformation matrix using SVD on subset...{Fore.RESET}")
    try:
        # Use only first 10000 samples for SVD to save memory
        subset_size = min(10000, a1.shape[0])
        indices = torch.randperm(a1.shape[0])[:subset_size]
        a1_subset = a1[indices].to(device)
        a2_subset = a2[indices].to(device)
        
        U, S, V = torch.svd(a1_subset.T @ a2_subset)
        W = (U @ V.T).contiguous()
        print(f"  SVD initialization successful on {subset_size} samples")
        
        # Clean up
        del a1_subset, a2_subset, U, S, V
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"{Fore.RED}  SVD failed: {e}, using identity initialization{Fore.RESET}")
        W = torch.eye(hidden_dim, device=device)
    
    W.requires_grad_(True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam([W], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=debug
    )
    
    # Create dataset and dataloader
    dataset = ActivationDataset(a1, a2, a3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Best model tracking
    best_W = W.clone().detach()
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                X, Y, Z = batch_data
            else:
                X, Y = batch_data
                Z = torch.zeros_like(X)
            
            # Move batch to device
            X = X.to(device).float()
            Y = Y.to(device).float()
            Z = Z.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = X @ W
            if a3 is not None and Z.sum() != 0:
                pred = pred + Z
            
            # 1. Main reconstruction loss (cosine similarity)
            pred_norm = F.normalize(pred, dim=-1)
            Y_norm = F.normalize(Y, dim=-1)
            recon_loss = 1 - (pred_norm * Y_norm).sum(-1).mean()
            
            # Calculate regularization losses only every 10 batches to save compute
            if batch_idx % 10 == 0:
                # 2. Spectral regularization
                try:
                    U_w, S_w, V_w = torch.svd(W)
                    spectral_loss = torch.norm(S_w - 1.0, p=2) / math.sqrt(hidden_dim)
                    condition_number = S_w.max() / (S_w.min() + 1e-8)
                    condition_loss = torch.log(condition_number + 1.0)
                except:
                    spectral_loss = torch.tensor(0.0, device=device)
                    condition_loss = torch.tensor(0.0, device=device)
                
                # 3. Orthogonality regularization
                orth_loss = torch.norm(W @ W.T - torch.eye(hidden_dim, device=device), 'fro')
                
                # Combined loss with regularization
                total_loss = (
                    recon_loss + 
                    spectral_weight * spectral_loss +
                    condition_weight * condition_loss +
                    orth_weight * orth_loss
                )
            else:
                # Just reconstruction loss for other batches
                total_loss = recon_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(recon_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{recon_loss.item():.4f}',
                'avg': f'{np.mean(epoch_losses):.4f}'
            })
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_W = W.clone().detach()
        
        # Debug prints
        if debug and epoch % 10 == 0:
            print(f"\n{Fore.GREEN}Epoch {epoch+1}/{epochs}:{Fore.RESET}")
            print(f"  Average loss: {avg_loss:.6f}")
            print(f"  Best loss: {best_loss:.6f}")
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n{Fore.GREEN}Optimization complete! Best loss: {best_loss:.6f}{Fore.RESET}")
    
    return best_W.cpu().to(torch.float64)


def apply_spectral_transform(
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
    spectral_weight: float = 0.01,
    condition_weight: float = 0.001,
    orth_weight: float = 0.0001,
    use_residual: bool = False,
    epochs: int = 100,
    optim_batch_size: int = 1024,
    use_accurate: bool = False,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """
    Apply spectral regularized transform to model layers.
    Memory-efficient version that stores activations on CPU.
    """
    print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}Applying Spectral Regularized Transform (Memory Efficient){Fore.RESET}")
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    print(f"Dataset size: {dataset_size}")
    
    # Load model for activation collection
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
    
    # Setup hooks to collect MLP outputs
    print(f"\n{Fore.YELLOW}Setting up activation hooks...{Fore.RESET}")
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
    
    # Pre-allocate CPU tensors (like ReplaceMe does)
    print(f"\n{Fore.YELLOW}Allocating CPU memory for activations...{Fore.RESET}")
    estimated_tokens = dataset_size * max_length
    a1 = torch.empty((estimated_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((estimated_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    if use_accurate:
        a3 = torch.empty((estimated_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    else:
        a3 = None
    
    # Collect activations
    print(f"\n{Fore.YELLOW}Collecting activations...{Fore.RESET}")
    cnt = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering Activations", colour="green")):
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx}/{len(dataloader)}, collected {cnt} tokens so far")
        
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
        
        # Get relevant hidden states and move to CPU immediately
        h_mlp = hidden_states_mlp.view(-1, hidden_size).cpu().to(torch.bfloat16)
        h_in = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).cpu().to(torch.bfloat16)
        h_out = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).cpu().to(torch.bfloat16)
        
        # Store in pre-allocated tensors
        batch_size_actual = h_mlp.shape[0]
        if cnt + batch_size_actual > estimated_tokens:
            print(f"{Fore.YELLOW}Warning: Exceeding estimated tokens, stopping collection{Fore.RESET}")
            break
        
        a1[cnt:cnt+batch_size_actual] = h_mlp
        if use_accurate:
            a2[cnt:cnt+batch_size_actual] = h_out
            a3[cnt:cnt+batch_size_actual] = h_in - h_mlp
        else:
            a2[cnt:cnt+batch_size_actual] = h_out + h_mlp - h_in
        
        cnt += batch_size_actual
        
        # Clear GPU cache
        del hidden_states_mlp, h_in, h_out
        torch.cuda.empty_cache()
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if use_accurate and a3 is not None:
        a3 = a3[:cnt]
    
    print(f"\nCollected {cnt} activation vectors")

    # 여기에 샘플링 코드 추가
    max_tokens = 3000000  # 300만 토큰 제한
    if cnt > max_tokens:
        print(f"{Fore.YELLOW}Sampling {max_tokens} from {cnt} tokens...{Fore.RESET}")
        # 랜덤 인덱스 생성
        indices = torch.randperm(cnt)[:max_tokens]
        # 샘플링
        a1 = a1[indices]
        a2 = a2[indices]
        if use_accurate and a3 is not None:
            a3 = a3[indices]
        print(f"Sampled down to {a1.shape[0]} tokens")
    else:
        print(f"Using all {cnt} tokens (under {max_tokens} limit)")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Learn transformation
    print(f"\n{Fore.CYAN}Learning spectral regularized transform...{Fore.RESET}")
    
    # Determine device for optimization
    optim_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if optim_device == 'cpu':
        print(f"{Fore.YELLOW}Warning: Using CPU for optimization, this will be slow{Fore.RESET}")
    
    transform = spectral_regularized_transform_batched(
        a1, a2, a3,
        hidden_dim=hidden_size,
        epochs=epochs,
        batch_size=optim_batch_size,
        spectral_weight=spectral_weight,
        condition_weight=condition_weight,
        orth_weight=orth_weight,
        use_residual=use_residual,
        device=optim_device,
        debug=True
    )
    
    # Clean up activations
    del a1, a2
    if a3 is not None:
        del a3
    gc.collect()
    
    # Reload model and apply transformation
    print(f"\n{Fore.YELLOW}Applying transformation to model...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (transform.T @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    # Stability check
    if torch.isnan(new_weight).any() or torch.isinf(new_weight).any():
        print(f"{Fore.RED}Warning: NaN/Inf detected in new weights, using original{Fore.RESET}")
        new_weight = original_weight
    else:
        print(f"{Fore.GREEN}Transformation applied successfully{Fore.RESET}")
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight = nn.Parameter(new_weight)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_spectral_{start_id}_{end_id}_{dataset}_{dataset_size}"
    
    save_path = save_path + "_spectral"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n{Fore.GREEN}Model saved to: {save_path}{Fore.RESET}")
    
    # Final cleanup
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path