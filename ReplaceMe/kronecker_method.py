import argparse
import gc
import logging
import os
from typing import Optional, Tuple
import torch
import yaml
import numpy as np
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def kronecker_factorization(
    a1: torch.Tensor, 
    a2: torch.Tensor, 
    a3: torch.Tensor = None,
    rank_ratio: float = 0.25,
    max_iterations: int = 50,
    lr: float = 1e-3,
    loss_type: str = "cosine"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose transformation matrix T into Kronecker product T1 ⊗ T2
    
    Args:
        a1: Input activations (MLP output)
        a2: Target activations 
        a3: Optional residual activations
        rank_ratio: Ratio to determine Kronecker factors size
        max_iterations: Maximum optimization iterations
        lr: Learning rate
        loss_type: Loss function type
    
    Returns:
        T1, T2: Kronecker factors
    """
    d = a1.shape[1]
    print(f"{Fore.GREEN}[Kronecker] Input dimension: {d}{Fore.RESET}")
    
    # Determine Kronecker factors dimensions
    # For d x d matrix, we want T1: k1 x k1, T2: k2 x k2 where k1 * k2 = d
    target_params = int(d * rank_ratio)
    k1 = int(np.sqrt(target_params))
    k2 = d // k1
    
    # Adjust k1, k2 to ensure k1 * k2 <= d
    while k1 * k2 > d:
        k1 -= 1
        k2 = d // k1
    
    print(f"{Fore.GREEN}[Kronecker] Factorization: {d}x{d} -> {k1}x{k1} ⊗ {k2}x{k2}{Fore.RESET}")
    print(f"{Fore.GREEN}[Kronecker] Parameter reduction: {d*d} -> {k1*k1 + k2*k2} ({(k1*k1 + k2*k2)/(d*d)*100:.1f}%){Fore.RESET}")
    
    # Initialize factors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T1 = torch.randn(k1, k1, device=device, dtype=torch.float32, requires_grad=True)
    T2 = torch.randn(k2, k2, device=device, dtype=torch.float32, requires_grad=True)
    
    # Initialize as identity-like matrices
    with torch.no_grad():
        T1.fill_(0.0)
        T1.fill_diagonal_(1.0)
        T2.fill_(0.0) 
        T2.fill_diagonal_(1.0)
    
    # Move data to device
    a1 = a1.float().to(device)
    a2 = a2.float().to(device)
    if a3 is not None:
        a3 = a3.float().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam([T1, T2], lr=lr)
    
    # Loss function
    def compute_loss(pred, target):
        if loss_type == "cosine":
            pred_norm = pred / pred.norm(dim=1, keepdim=True)
            target_norm = target / target.norm(dim=1, keepdim=True)
            return 1 - (pred_norm * target_norm).sum(dim=1).mean()
        elif loss_type == "mse":
            return torch.nn.MSELoss()(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    print(f"{Fore.YELLOW}[Kronecker] Starting optimization with {loss_type} loss...{Fore.RESET}")
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for iteration in tqdm(range(max_iterations), desc="Kronecker Optimization"):
        optimizer.zero_grad()
        
        # Construct full transformation via Kronecker product
        # Reshape input for Kronecker product application
        batch_size = a1.shape[0]
        
        # Reshape a1 to (batch_size, k2, k1) for Kronecker product
        try:
            a1_reshaped = a1.view(batch_size, k2, k1)
            # Apply T1 ⊗ T2: (a1_reshaped @ T1.T).transpose(-2, -1) @ T2.T
            temp = torch.matmul(a1_reshaped, T1.T)  # (batch, k2, k1)
            pred = torch.matmul(temp.transpose(-2, -1), T2.T)  # (batch, k1, k2)
            pred = pred.transpose(-2, -1).contiguous().view(batch_size, -1)  # (batch, k2*k1)
            
            # Pad or truncate to match target dimension
            if pred.shape[1] < d:
                padding = torch.zeros(batch_size, d - pred.shape[1], device=device)
                pred = torch.cat([pred, padding], dim=1)
            elif pred.shape[1] > d:
                pred = pred[:, :d]
                
        except RuntimeError as e:
            print(f"{Fore.RED}[Kronecker] Reshape error: {e}{Fore.RESET}")
            # Fallback: direct matrix multiplication with reshaped factors
            T_full = torch.kron(T1, T2)
            if T_full.shape[0] != d or T_full.shape[1] != d:
                # Pad or truncate T_full to d x d
                T_padded = torch.zeros(d, d, device=device)
                min_dim = min(T_full.shape[0], d)
                T_padded[:min_dim, :min_dim] = T_full[:min_dim, :min_dim]
                T_full = T_padded
            pred = torch.matmul(a1, T_full.T)
        
        # Add residual if provided
        if a3 is not None:
            pred = pred + a3
            
        # Compute loss
        loss = compute_loss(pred, a2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"{Fore.GREEN}[Kronecker] Early stopping at iteration {iteration}{Fore.RESET}")
            break
            
        # Log progress
        if iteration % 10 == 0:
            print(f"{Fore.CYAN}[Kronecker] Iter {iteration}: Loss = {loss.item():.6f}{Fore.RESET}")
    
    print(f"{Fore.GREEN}[Kronecker] Optimization completed. Final loss: {best_loss:.6f}{Fore.RESET}")
    
    return T1.detach(), T2.detach()


def reconstruct_full_matrix(T1: torch.Tensor, T2: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Reconstruct full transformation matrix from Kronecker factors
    """
    print(f"{Fore.YELLOW}[Kronecker] Reconstructing {target_dim}x{target_dim} matrix from factors{Fore.RESET}")
    
    # Compute Kronecker product
    T_full = torch.kron(T1, T2)
    
    # Ensure correct dimensions
    if T_full.shape[0] != target_dim or T_full.shape[1] != target_dim:
        T_padded = torch.zeros(target_dim, target_dim, device=T_full.device, dtype=T_full.dtype)
        min_dim_0 = min(T_full.shape[0], target_dim)
        min_dim_1 = min(T_full.shape[1], target_dim)
        T_padded[:min_dim_0, :min_dim_1] = T_full[:min_dim_0, :min_dim_1]
        T_full = T_padded
        
    print(f"{Fore.GREEN}[Kronecker] Matrix reconstruction complete{Fore.RESET}")
    return T_full


def kronecker_dist(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "train",
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
    rank_ratio: float = 0.25,
    loss: str = "cosine",
    max_iterations: int = 50
) -> str:
    """
    Apply Kronecker factorization-based compression to transformer blocks.
    
    Args:
        rank_ratio: Ratio to determine Kronecker factors size (0.1-0.5)
        max_iterations: Maximum optimization iterations
        Other args: Same as cosine_dist function
    
    Returns:
        Path where transformed model is saved
    """
    print(f"{Fore.MAGENTA}=== Starting Kronecker Factorization Method ==={Fore.RESET}")
    print(f"{Fore.MAGENTA}Rank ratio: {rank_ratio}, Max iterations: {max_iterations}{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
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

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
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
    
    print(f"{Fore.CYAN}[Kronecker] Model loaded. Hidden size: {hidden_size}{Fore.RESET}")
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    mlp_activations = {}
    
    # Allocate memory for activations
    print(f"{Fore.YELLOW}[Kronecker] Allocating memory for activations...{Fore.RESET}")
    total_tokens = dataset_size * max_length
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    if accurate:
        print(f"{Fore.YELLOW}[Kronecker] ACCURATE MODE: Allocating additional memory{Fore.RESET}")
        a3 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    cnt = 0
    print(f"{Fore.RED}[Kronecker] Gathering activations from layers {start_id-num_layer-1} to {end_id-num_layer-1}...{Fore.RESET}")
    
    for batch in tqdm(dataloader, desc=Fore.RED + "Gathering Activations" + Fore.RESET, dynamic_ncols=True, colour="red"):
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
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        
        # Memory management
        if cnt % (batch_size * max_length * 5) == 0:  # Every 5 batches
            torch.cuda.empty_cache()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
        print(f"{Fore.GREEN}[Kronecker] Gathered {cnt} tokens with residual connections{Fore.RESET}")
    else:
        a3 = None
        print(f"{Fore.GREEN}[Kronecker] Gathered {cnt} tokens{Fore.RESET}")
    
    # Apply Kronecker factorization
    print(f"{Fore.MAGENTA}[Kronecker] Starting factorization...{Fore.RESET}")
    T1, T2 = kronecker_factorization(
        a1, a2, a3, 
        rank_ratio=rank_ratio, 
        max_iterations=max_iterations,
        loss_type=loss
    )
    
    # Reconstruct full transformation matrix
    transform = reconstruct_full_matrix(T1, T2, hidden_size)
    
    print(f"{Fore.GREEN}[Kronecker] Transformation matrix reconstructed: {transform.shape}{Fore.RESET}")
    
    # Clean up
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    print(f"{Fore.YELLOW}[Kronecker] Reloading model for transformation...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply transformation
    print(f"{Fore.YELLOW}[Kronecker] Applying transformation to down_proj layer...{Fore.RESET}")
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)
    transformed_weight = (transform.T.cpu() @ original_weight).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": transformed_weight
    })
    
    final_save_path = f"{save_path}_Kronecker_{loss}_r{rank_ratio}"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    # Save Kronecker factors separately for analysis
    factors_path = f"{final_save_path}_factors.pth"
    torch.save({
        'T1': T1.cpu(),
        'T2': T2.cpu(),
        'rank_ratio': rank_ratio,
        'original_dim': hidden_size,
        'compression_ratio': (T1.numel() + T2.numel()) / (hidden_size * hidden_size)
    }, factors_path)
    
    print(f"{Fore.GREEN}[Kronecker] Model saved to: {final_save_path}{Fore.RESET}")
    print(f"{Fore.GREEN}[Kronecker] Factors saved to: {factors_path}{Fore.RESET}")
    
    # Final cleanup
    del model, transform, T1, T2
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_save_path


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the Kronecker factorization from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run Kronecker factorization for linear transform estimation based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    average_distances = torch.load(config['distances_path'])
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        config['layers_to_skip'],
        num_blocks=config['num_A'],
        merge_consecutive=config['merge_consecutive']
    )
    
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    for i in range(len(selected_blocks)):
        path = kronecker_dist(**config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
        config["model_path"] = path