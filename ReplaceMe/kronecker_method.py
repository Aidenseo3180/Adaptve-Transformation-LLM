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


def kronecker_factorization_streaming(
    dataloader,
    tokenizer,
    model_hooks,
    start_id: int,
    end_id: int,
    num_layer: int,
    max_length: int,
    hidden_size: int,
    accurate: bool = False,
    rank_ratio: float = 0.25,
    max_iterations: int = 50,
    lr: float = 1e-3,
    loss_type: str = "cosine",
    mini_batch_size: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient Kronecker factorization using streaming mini-batches
    """
    print(f"{Fore.GREEN}[Kronecker] Input dimension: {hidden_size}{Fore.RESET}")
    
    # Determine Kronecker factors dimensions
    target_params = int(hidden_size * rank_ratio)
    k1 = int(np.sqrt(target_params))
    k2 = hidden_size // k1
    
    # Adjust k1, k2 to ensure k1 * k2 <= hidden_size
    while k1 * k2 > hidden_size:
        k1 -= 1
        k2 = hidden_size // k1
    
    print(f"{Fore.GREEN}[Kronecker] Factorization: {hidden_size}x{hidden_size} -> {k1}x{k1} ⊗ {k2}x{k2}{Fore.RESET}")
    print(f"{Fore.GREEN}[Kronecker] Parameter reduction: {hidden_size*hidden_size} -> {k1*k1 + k2*k2} ({(k1*k1 + k2*k2)/(hidden_size*hidden_size)*100:.1f}%){Fore.RESET}")
    print(f"{Fore.YELLOW}[Kronecker] Using mini-batch size: {mini_batch_size}{Fore.RESET}")
    
    # Initialize factors on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T1 = torch.randn(k1, k1, device=device, dtype=torch.float32, requires_grad=True)
    T2 = torch.randn(k2, k2, device=device, dtype=torch.float32, requires_grad=True)
    
    # Initialize as identity-like matrices
    with torch.no_grad():
        T1.fill_(0.0)
        T1.fill_diagonal_(1.0)
        T2.fill_(0.0)
        T2.fill_diagonal_(1.0)
    
    # Optimizer
    optimizer = torch.optim.Adam([T1, T2], lr=lr)
    
    # Loss function
    def compute_loss(pred, target):
        if loss_type == "cosine":
            # Add small epsilon for numerical stability
            pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
            return 1 - (pred_norm * target_norm).sum(dim=1).mean()
        elif loss_type == "mse":
            return torch.nn.MSELoss()(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    print(f"{Fore.YELLOW}[Kronecker] Starting streaming optimization with {loss_type} loss...{Fore.RESET}")
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    total_batches_processed = 0
    
    # Multiple epochs over the data
    for epoch in range(max_iterations):
        epoch_loss = 0.0
        num_mini_batches = 0
        
        print(f"{Fore.CYAN}[Kronecker] Epoch {epoch+1}/{max_iterations}{Fore.RESET}")
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} Processing", leave=False):
            # Process this batch
            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model_hooks['model'](**inputs)
                
                # Extract activations
                hidden_states = outputs.hidden_states[1:]
                hidden_states_mlp_list = [
                    model_hooks['mlp_activations'][f'layer_{i}_mlp'] 
                    for i in range(model_hooks['num_layers'])
                ]
                
                hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
                hidden_states_i = hidden_states[start_id - num_layer - 1]
                hidden_states_n = hidden_states[end_id - num_layer - 1]
                
                # Reshape and prepare data
                batch_tokens = hidden_states_mlp.shape[0] * hidden_states_mlp.shape[1]
                a1_batch = hidden_states_mlp.view(-1, hidden_size).float()
                hidden_states_i_flat = hidden_states_i.view(-1, hidden_size).float()
                hidden_states_n_flat = hidden_states_n.view(-1, hidden_size).float()
                
                if accurate:
                    a2_batch = hidden_states_n_flat
                    a3_batch = hidden_states_i_flat - a1_batch
                else:
                    a2_batch = hidden_states_n_flat + a1_batch - hidden_states_i_flat
                    a3_batch = None
                
                # Process in mini-batches to save memory
                for start_idx in range(0, batch_tokens, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, batch_tokens)
                    
                    mini_a1 = a1_batch[start_idx:end_idx].to(device)
                    mini_a2 = a2_batch[start_idx:end_idx].to(device)
                    mini_a3 = a3_batch[start_idx:end_idx].to(device) if a3_batch is not None else None
                    
                    optimizer.zero_grad()
                    
                    # Apply Kronecker transformation efficiently
                    mini_batch_size_actual = mini_a1.shape[0]
                    
                    # Simple approach: use small subsets for Kronecker application
                    if k1 * k2 == hidden_size:
                        # Perfect factorization
                        try:
                            mini_a1_reshaped = mini_a1.view(mini_batch_size_actual, k2, k1)
                            temp = torch.matmul(mini_a1_reshaped, T1.T)
                            pred = torch.matmul(temp.transpose(-2, -1), T2.T)
                            pred = pred.transpose(-2, -1).contiguous().view(mini_batch_size_actual, -1)
                        except:
                            # Fallback to approximation
                            T_approx = torch.kron(T1[:k1//2, :k1//2], T2[:k2//2, :k2//2])
                            if T_approx.shape[0] > hidden_size:
                                T_approx = T_approx[:hidden_size, :hidden_size]
                            elif T_approx.shape[0] < hidden_size:
                                T_padded = torch.zeros(hidden_size, hidden_size, device=device)
                                T_padded[:T_approx.shape[0], :T_approx.shape[1]] = T_approx
                                T_approx = T_padded
                            pred = torch.matmul(mini_a1, T_approx.T)
                    else:
                        # Approximation approach
                        effective_k1 = min(k1, int(np.sqrt(hidden_size)))
                        effective_k2 = min(k2, hidden_size // effective_k1)
                        
                        T_approx = torch.kron(T1[:effective_k1, :effective_k1], T2[:effective_k2, :effective_k2])
                        if T_approx.shape[0] < hidden_size:
                            T_padded = torch.zeros(hidden_size, hidden_size, device=device)
                            T_padded[:T_approx.shape[0], :T_approx.shape[1]] = T_approx
                            T_approx = T_padded
                        pred = torch.matmul(mini_a1, T_approx[:hidden_size, :hidden_size].T)
                    
                    # Add residual if provided
                    if mini_a3 is not None:
                        pred = pred + mini_a3
                    
                    # Compute loss
                    loss = compute_loss(pred, mini_a2)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_mini_batches += 1
                    total_batches_processed += 1
                    
                    # Clear mini-batch memory immediately
                    del mini_a1, mini_a2, mini_a3, pred
                    torch.cuda.empty_cache()
                
                # Clear batch memory
                del a1_batch, a2_batch, a3_batch, hidden_states_mlp, hidden_states_i, hidden_states_n
                del hidden_states, outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"{Fore.RED}[Kronecker] Error processing batch: {e}{Fore.RESET}")
                torch.cuda.empty_cache()
                continue
        
        # Calculate average loss for this epoch
        if num_mini_batches > 0:
            avg_loss = epoch_loss / num_mini_batches
            print(f"{Fore.CYAN}[Kronecker] Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, Mini-batches: {num_mini_batches}{Fore.RESET}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"{Fore.GREEN}[Kronecker] Early stopping at epoch {epoch+1}{Fore.RESET}")
                break
        else:
            print(f"{Fore.YELLOW}[Kronecker] No mini-batches processed in epoch {epoch+1}{Fore.RESET}")
    
    print(f"{Fore.GREEN}[Kronecker] Streaming optimization completed. Best loss: {best_loss:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}[Kronecker] Total mini-batches processed: {total_batches_processed}{Fore.RESET}")
    
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
    
    # Prepare model hooks for streaming
    model_hooks = {
        'model': model,
        'mlp_activations': mlp_activations,
        'num_layers': model.config.num_hidden_layers
    }
    
    print(f"{Fore.YELLOW}[Kronecker] Using streaming approach - no large memory allocation{Fore.RESET}")
    print(f"{Fore.RED}[Kronecker] Processing layers {start_id-num_layer-1} to {end_id-num_layer-1}...{Fore.RESET}")
    
    # Apply streaming Kronecker factorization
    print(f"{Fore.MAGENTA}[Kronecker] Starting streaming factorization...{Fore.RESET}")
    T1, T2 = kronecker_factorization_streaming(
        dataloader=dataloader,
        tokenizer=tokenizer,
        model_hooks=model_hooks,
        start_id=start_id,
        end_id=end_id,
        num_layer=num_layer,
        max_length=max_length,
        hidden_size=hidden_size,
        accurate=accurate,
        rank_ratio=rank_ratio,
        max_iterations=max_iterations,
        loss_type=loss,
        mini_batch_size=256  # Reduced mini-batch size for memory safety
    )
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Reconstruct full transformation matrix
    transform = reconstruct_full_matrix(T1, T2, hidden_size)
    
    print(f"{Fore.GREEN}[Kronecker] Transformation matrix reconstructed: {transform.shape}{Fore.RESET}")
    
    # Clean up model memory
    del model
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
    
    # Ensure transform and original_weight have the same dtype
    transform_cpu = transform.T.cpu().to(torch.float64)
    transformed_weight = (transform_cpu @ original_weight).to(torch.bfloat16)
    
    print(f"{Fore.GREEN}[Kronecker] Transform dtype: {transform_cpu.dtype}, Weight dtype: {original_weight.dtype}{Fore.RESET}")
    
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