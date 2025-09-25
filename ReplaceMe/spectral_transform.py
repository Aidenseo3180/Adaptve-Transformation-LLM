import gc
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_calib_dataloader, seed_all, truncate_model

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def spectral_regularized_transform(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: Optional[torch.Tensor] = None,
    hidden_dim: int = 4096,
    epochs: int = 100,
    lr: float = 1e-3,
    spectral_weight: float = 0.01,
    condition_weight: float = 0.001,
    orth_weight: float = 0.0001,
    use_residual: bool = False,
    debug: bool = True
) -> torch.Tensor:
    """
    Learn transformation matrix with spectral regularization.
    
    Args:
        a1: Input activations [N x hidden_dim]
        a2: Target activations [N x hidden_dim]
        a3: Optional residual correction [N x hidden_dim]
        hidden_dim: Dimension of hidden states
        epochs: Number of optimization epochs
        lr: Learning rate
        spectral_weight: Weight for spectral regularization
        condition_weight: Weight for condition number regularization
        orth_weight: Weight for orthogonality regularization
        use_residual: Whether to add residual connection
        debug: Whether to print debug information
    
    Returns:
        Optimized transformation matrix
    """
    if debug:
        print(f"{Fore.CYAN}Starting spectral regularized transform optimization{Fore.RESET}")
        print(f"  Input shape: {a1.shape}")
        print(f"  Target shape: {a2.shape}")
        print(f"  Using residual: {a3 is not None}")
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a1 = a1.to(device).float()
    a2 = a2.to(device).float()
    if a3 is not None:
        a3 = a3.to(device).float()
    
    # Smart initialization using SVD
    print(f"{Fore.YELLOW}Initializing transformation matrix using SVD...{Fore.RESET}")
    try:
        U, S, V = torch.svd(a1.T @ a2)
        W = (U @ V.T).contiguous()
        print(f"  SVD initialization successful")
    except:
        print(f"{Fore.RED}  SVD failed, using identity initialization{Fore.RESET}")
        W = torch.eye(hidden_dim, device=device)
    
    W.requires_grad_(True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam([W], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=debug
    )
    
    # Best model tracking
    best_W = W.clone().detach()
    best_loss = float('inf')
    
    # Training loop
    pbar = tqdm(range(epochs), desc="Optimizing Transform", colour="blue")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        pred = a1 @ W
        if a3 is not None:
            pred = pred + a3
        
        # 1. Main reconstruction loss (cosine similarity)
        pred_norm = F.normalize(pred, dim=-1)
        a2_norm = F.normalize(a2, dim=-1)
        recon_loss = 1 - (pred_norm * a2_norm).sum(-1).mean()
        
        # 2. Spectral regularization
        try:
            U_w, S_w, V_w = torch.svd(W)
            # Encourage singular values to be close to 1
            spectral_loss = torch.norm(S_w - 1.0, p=2) / math.sqrt(hidden_dim)
        except:
            # SVD convergence issue
            spectral_loss = torch.tensor(0.0, device=device)
            if debug and epoch % 10 == 0:
                print(f"\n{Fore.YELLOW}Warning: SVD failed at epoch {epoch}{Fore.RESET}")
        
        # 3. Condition number regularization
        if spectral_loss > 0:
            condition_number = S_w.max() / (S_w.min() + 1e-8)
            condition_loss = torch.log(condition_number + 1.0)
        else:
            condition_loss = torch.tensor(0.0, device=device)
        
        # 4. Orthogonality regularization
        orth_loss = torch.norm(W @ W.T - torch.eye(hidden_dim, device=device), 'fro')
        
        # 5. Optional: Residual connection regularization
        if use_residual:
            residual_loss = 0.001 * torch.norm(W - torch.eye(hidden_dim, device=device), 'fro')
        else:
            residual_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = (
            recon_loss + 
            spectral_weight * spectral_loss +
            condition_weight * condition_loss +
            orth_weight * orth_loss +
            residual_loss
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
        
        optimizer.step()
        scheduler.step(recon_loss)
        
        # Track best model
        if recon_loss < best_loss:
            best_loss = recon_loss
            best_W = W.clone().detach()
        
        # Update progress bar
        pbar.set_postfix({
            'recon': f'{recon_loss.item():.4f}',
            'spectral': f'{spectral_loss.item():.4f}',
            'best': f'{best_loss:.4f}'
        })
        
        # Debug prints
        if debug and epoch % 20 == 0:
            print(f"\n{Fore.GREEN}Epoch {epoch}/{epochs}:{Fore.RESET}")
            print(f"  Reconstruction loss: {recon_loss.item():.6f}")
            print(f"  Spectral loss: {spectral_loss.item():.6f}")
            print(f"  Condition loss: {condition_loss.item():.6f}")
            print(f"  Orthogonality loss: {orth_loss.item():.6f}")
            if spectral_loss > 0:
                print(f"  Singular values range: [{S_w.min().item():.3f}, {S_w.max().item():.3f}]")
    
    print(f"\n{Fore.GREEN}Optimization complete! Best loss: {best_loss:.6f}{Fore.RESET}")
    
    return best_W.to(torch.float64)


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
    **kwargs
) -> str:
    """
    Apply spectral regularized transform to model layers.
    """
    print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}Applying Spectral Regularized Transform{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    
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
    
    # Collect activations
    print(f"\n{Fore.YELLOW}Collecting activations...{Fore.RESET}")
    a1_list = []
    a2_list = []
    a3_list = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering Activations")):
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx}/{len(dataloader)}")
        
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
        
        # Get relevant hidden states
        h_mlp = hidden_states_mlp.view(-1, hidden_size)
        h_in = hidden_states[start_id - num_layer - 1].view(-1, hidden_size)
        h_out = hidden_states[end_id - num_layer - 1].view(-1, hidden_size)
        
        a1_list.append(h_mlp.cpu())
        a2_list.append(h_out.cpu())
        a3_list.append((h_in - h_mlp).cpu())
    
    # Concatenate all batches
    a1 = torch.cat(a1_list, dim=0)
    a2 = torch.cat(a2_list, dim=0)
    a3 = torch.cat(a3_list, dim=0)
    
    print(f"\nCollected activations shape: {a1.shape}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Learn transformation
    print(f"\n{Fore.CYAN}Learning spectral regularized transform...{Fore.RESET}")
    transform = spectral_regularized_transform(
        a1, a2, a3,
        hidden_dim=hidden_size,
        epochs=epochs,
        spectral_weight=spectral_weight,
        condition_weight=condition_weight,
        orth_weight=orth_weight,
        use_residual=use_residual,
        debug=True
    )
    
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
    new_weight = (transform.T.cpu() @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    # Stability check
    if torch.isnan(new_weight).any() or torch.isinf(new_weight).any():
        print(f"{Fore.RED}Warning: NaN/Inf detected in new weights, using original{Fore.RESET}")
        new_weight = original_weight
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight = nn.Parameter(new_weight)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_spectral_{start_id}_{end_id}"
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n{Fore.GREEN}Model saved to: {save_path}{Fore.RESET}")
    
    # Clean up
    del model, a1, a2, a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path