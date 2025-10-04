import argparse
import gc
import logging
import os
from typing import Optional, Tuple
import torch
import yaml
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


def compute_bidirectional_transform(
    a1: torch.Tensor,  # Mi (MLP output)
    a2: torch.Tensor,  # Li+n - Yi  
    a3: Optional[torch.Tensor] = None,  # Yi - Mi (attention residual) if accurate mode
    hidden_dim: int = 4096,
    rank_ratio: float = 0.1,
    num_iterations: int = 3,
    device: str = 'cuda'
) -> torch.Tensor:
    """Compute bidirectional transformation with iterative refinement."""
    
    print(f"{Fore.YELLOW}Starting Bidirectional Transformation Computation{Fore.RESET}")
    print(f"Data shape: {a1.shape}, Hidden dim: {hidden_dim}, Device: {device}")
    
    # Move data to correct device and dtype
    a1 = a1.to(device).to(torch.float32)
    a2 = a2.to(device).to(torch.float32)
    if a3 is not None:
        a3 = a3.to(device).to(torch.float32)
    
    # Step 1: Forward transformation (Mi -> Li+n - Yi)
    print(f"\n{Fore.GREEN}Step 1: Computing forward transformation{Fore.RESET}")
    T_forward = compute_single_direction_transform(a1, a2, a3, direction='forward', device=device)
    forward_error = compute_reconstruction_error(a1, a2, T_forward, a3, direction='forward')
    print(f"Forward reconstruction error: {forward_error:.6f}")
    
    # Step 2: Backward transformation (Li+n - Yi -> Mi)
    print(f"\n{Fore.GREEN}Step 2: Computing backward transformation{Fore.RESET}")
    T_backward = compute_single_direction_transform(a2, a1, -a3 if a3 is not None else None, 
                                                   direction='backward', device=device)
    backward_error = compute_reconstruction_error(a2, a1, T_backward, -a3 if a3 is not None else None, 
                                                 direction='backward')
    print(f"Backward reconstruction error: {backward_error:.6f}")
    
    # Step 3: Combine based on confidence
    print(f"\n{Fore.GREEN}Step 3: Combining transformations{Fore.RESET}")
    forward_weight = backward_error / (forward_error + backward_error + 1e-8)
    backward_weight = forward_error / (forward_error + backward_error + 1e-8)
    
    # Need to invert T_backward for combination
    try:
        T_backward_inv = torch.linalg.pinv(T_backward)
    except:
        print(f"{Fore.RED}Warning: Pseudo-inverse failed, using transpose{Fore.RESET}")
        T_backward_inv = T_backward.T
    
    T_combined = forward_weight * T_forward + backward_weight * T_backward_inv
    print(f"Weights - Forward: {forward_weight:.3f}, Backward: {backward_weight:.3f}")
    
    # Step 4: Iterative refinement with low-rank corrections
    print(f"\n{Fore.GREEN}Step 4: Iterative refinement{Fore.RESET}")
    T_refined = iterative_refinement(a1, a2, T_combined, a3, 
                                    rank_ratio=rank_ratio, 
                                    num_iterations=num_iterations,
                                    device=device)
    
    return T_refined.to(torch.float64)


def compute_single_direction_transform(
    X: torch.Tensor, 
    Y: torch.Tensor,
    Z: Optional[torch.Tensor],
    direction: str,
    device: str = 'cuda',
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 1024
) -> torch.Tensor:
    """Compute transformation in a single direction using Adam optimizer."""
    
    print(f"  Computing {direction} transform...")
    
    # Initialize transformation as identity
    T = torch.eye(X.shape[1], device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=lr)
    
    # Create batches
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    best_T = T.clone().detach()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle indices
        indices = torch.randperm(num_samples, device=device)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            
            optimizer.zero_grad()
            
            # Compute prediction
            pred = X_batch @ T
            if Z is not None:
                Z_batch = Z[batch_indices]
                pred = pred + Z_batch
            
            # Cosine loss
            pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            Y_norm = Y_batch / (Y_batch.norm(dim=1, keepdim=True) + 1e-8)
            loss = 1 - (pred_norm * Y_norm).sum(dim=1).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T.clone().detach()
        
        if epoch % 2 == 0:
            print(f"    Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    return best_T


def compute_reconstruction_error(
    X: torch.Tensor,
    Y: torch.Tensor, 
    T: torch.Tensor,
    Z: Optional[torch.Tensor],
    direction: str
) -> float:
    """Compute reconstruction error for a transformation."""
    
    with torch.no_grad():
        pred = X @ T
        if Z is not None:
            pred = pred + Z
        
        # Cosine distance
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * Y_norm).sum(dim=1).mean()
        
        return (1 - cosine_sim).item()


def iterative_refinement(
    X: torch.Tensor,
    Y: torch.Tensor,
    T_initial: torch.Tensor,
    Z: Optional[torch.Tensor],
    rank_ratio: float = 0.1,
    num_iterations: int = 3,
    device: str = 'cuda'
) -> torch.Tensor:
    """Iteratively refine transformation with low-rank corrections."""
    
    T_current = T_initial.clone()
    hidden_dim = T_initial.shape[0]
    
    for iteration in range(num_iterations):
        print(f"\n  Iteration {iteration + 1}/{num_iterations}:")
        
        # Compute residual
        with torch.no_grad():
            pred = X @ T_current
            if Z is not None:
                pred = pred + Z
            residual = Y - pred
        
        # Compute residual statistics
        residual_norm = residual.norm(dim=1).mean().item()
        print(f"    Residual norm: {residual_norm:.6f}")
        
        # Early stopping if residual is small
        if residual_norm < 0.01:
            print(f"    Converged early!")
            break
        
        # Compute low-rank correction
        rank = max(1, int(hidden_dim * rank_ratio / (2 ** iteration)))
        print(f"    Using rank {rank} for correction")
        
        # Use SVD on residual patterns to find correction
        U, S, V = torch.svd_lowrank(residual.T @ X, q=rank)
        
        # Scale correction based on iteration
        scale = 0.1 / (iteration + 1)
        correction = scale * (V @ U.T)
        
        # Update transformation
        T_current = T_current + correction
        
        # Compute new error
        new_error = compute_reconstruction_error(X, Y, T_current, Z, direction='refined')
        print(f"    Error after correction: {new_error:.6f}")
    
    return T_current


def bidirectional_cosine_dist(
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
    rank_ratio: float = 0.1,
    num_iterations: int = 3,
    **kwargs  # Catch any extra args from config
) -> str:
    """Main function for bidirectional transformation with iterative refinement."""
    
    print(f"\n{Fore.CYAN}=== Starting Bidirectional ReplaceMe ==={Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    print(f"Dataset: {dataset}, Size: {dataset_size}")
    
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
    
    # Set up hooks for MLP activations
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
    
    # Prepare activation storage
    print(f"\n{Fore.YELLOW}Gathering activations...{Fore.RESET}")
    a1 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    a3 = None
    if accurate:
        print("ACCURATE MODE: Storing attention residuals separately")
        a3 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    
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
        
        # Reshape and store
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states_i.view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states_n.view(-1, hidden_size).to(torch.float64)
        
        batch_size_actual = hidden_states_mlp.shape[0]
        
        a1[cnt:cnt+batch_size_actual] = hidden_states_mlp.to(torch.bfloat16).cpu()
        
        if accurate:
            a2[cnt:cnt+batch_size_actual] = hidden_states_n.to(torch.bfloat16).cpu()
            a3[cnt:cnt+batch_size_actual] = (hidden_states_i - hidden_states_mlp).to(torch.bfloat16).cpu()
        else:
            a2[cnt:cnt+batch_size_actual] = (hidden_states_n + hidden_states_mlp - hidden_states_i).to(torch.bfloat16).cpu()
        
        cnt += batch_size_actual
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
    
    print(f"\nCollected {cnt} activation samples")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute bidirectional transformation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{Fore.CYAN}Computing bidirectional transformation on {device}...{Fore.RESET}")
    
    transform = compute_bidirectional_transform(
        a1, a2, a3,
        hidden_dim=hidden_size,
        rank_ratio=rank_ratio,
        num_iterations=num_iterations,
        device=device
    )
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
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
        save_path = f"output_models/{model_path}_{layers_to_skip}_layers_{start_id}_{end_id}_bidirectional".replace("/", "_")
    
    # Apply transformation to down_proj
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    print(f"Original weight shape: {original_weight.shape}, dtype: {original_weight.dtype}")
    print(f"Transform shape: {transform.shape}, dtype: {transform.dtype}")
    
    # Ensure correct dtype and device
    new_weight = (transform.T.cpu().to(torch.float64) @ original_weight.cpu().to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Save model
    full_save_path = f"{save_path}_ReplaceMe_bidirectional"
    model.save_pretrained(full_save_path)
    tokenizer.save_pretrained(full_save_path)
    
    print(f"{Fore.GREEN}Model saved to: {full_save_path}{Fore.RESET}")
    
    # Also save transform for analysis
    torch.save(transform, f"{full_save_path}_transform.pt")
    
    # Clean up
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return full_save_path

