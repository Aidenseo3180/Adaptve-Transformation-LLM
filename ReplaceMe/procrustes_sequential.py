# procrustes_sequential.py
import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from scipy.linalg import orthogonal_procrustes
import numpy as np

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

def procrustes_sequential_optimization(
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
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    sequential_constraint_weight: float = 0.1,  # New parameter for sequential dependency
    orthogonal_regularization: float = 0.01,    # Orthogonality constraint strength
) -> str:
    """
    ProcrustesGPT-based sequential optimization for multiple linear transforms.
    
    Key innovation: Each subsequent T matrix is optimized considering the 
    transformation space established by previous T matrices.
    
    Args:
        sequential_constraint_weight: Weight for sequential dependency loss
        orthogonal_regularization: Strength of orthogonality constraint
        Other args: Same as original ReplaceMe
    
    Returns:
        Path where transformed model is saved
    """
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
    
    # Get block selections for sequential optimization
    average_distances = torch.load(distances_path, weights_only=False)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    logging.info(f"{Fore.GREEN}Selected blocks for sequential optimization: {selected_blocks}{Fore.RESET}")
    
    # Collect activations for all selected blocks
    all_activations = collect_sequential_activations(
        model, dataloader, tokenizer, selected_blocks, max_length, dataset_size
    )
    
    # Sequential optimization of transformation matrices
    transforms = sequential_procrustes_optimization(
        all_activations, 
        selected_blocks,
        sequential_constraint_weight,
        orthogonal_regularization
    )
    
    # Apply transformations to model
    modified_model_path = apply_sequential_transforms(
        model_path, transforms, selected_blocks, save_path, token
    )
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return modified_model_path


def collect_sequential_activations(
    model, dataloader, tokenizer, selected_blocks, max_length, dataset_size
):
    """
    Collect activations for all selected blocks in a single forward pass.
    This is more efficient than multiple passes and ensures consistency.
    """
    logging.info(f"{Fore.BLUE}Collecting activations for sequential optimization...{Fore.RESET}")
    
    hidden_size = model.config.hidden_size
    num_blocks = len(selected_blocks)
    
    # Initialize storage for all activations
    activations = {}
    for i, (start_id, end_id) in enumerate(selected_blocks):
        activations[f'block_{i}'] = {
            'input': torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu'),
            'output': torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu'),
            'mlp_output': torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu'),
        }
    
    # Hook registration for MLP outputs
    def save_mlp_activation(block_idx, name):
        def hook(module, input, output):
            activations[f'block_{block_idx}'][name] = output.detach()
        return hook
    
    hooks = []
    for i, (start_id, end_id) in enumerate(selected_blocks):
        # Register hook for MLP output of the block before cut
        if 'falcon' in model.config.model_type.lower():
            layer = model.transformer.h[start_id - 1]
        else:
            layer = model.model.layers[start_id - 1]
        hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(i, 'mlp_temp')))
    
    cnt = 0
    for batch in tqdm(dataloader, desc=f"{Fore.BLUE}Collecting Sequential Activations{Fore.RESET}"):
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
        
        hidden_states = outputs.hidden_states[1:]  # Skip input embeddings
        
        # Store activations for each block
        for i, (start_id, end_id) in enumerate(selected_blocks):
            # Input to the block sequence (before start_id)
            input_activation = hidden_states[start_id - 1].view(-1, hidden_size).to(torch.float64)
            # Output after the block sequence (after end_id)
            output_activation = hidden_states[end_id - 1].view(-1, hidden_size).to(torch.float64)
            # MLP output from the hook
            mlp_activation = activations[f'block_{i}']['mlp_temp'].view(-1, hidden_size).to(torch.float64)
            
            batch_size = input_activation.shape[0]
            activations[f'block_{i}']['input'][cnt:cnt+batch_size] = input_activation
            activations[f'block_{i}']['output'][cnt:cnt+batch_size] = output_activation
            activations[f'block_{i}']['mlp_output'][cnt:cnt+batch_size] = mlp_activation
        
        cnt += batch_size
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Trim to actual size
    for i in range(num_blocks):
        for key in ['input', 'output', 'mlp_output']:
            activations[f'block_{i}'][key] = activations[f'block_{i}'][key][:cnt]
    
    return activations


def sequential_procrustes_optimization(
    activations, 
    selected_blocks, 
    sequential_constraint_weight=0.1,
    orthogonal_regularization=0.01,
    max_iterations=50
):
    """
    Core ProcrustesGPT sequential optimization.
    
    Key insight: Each T_i is optimized considering the cumulative effect of T_1, ..., T_{i-1}
    This models the sequential dependency in transformer information flow.
    """
    logging.info(f"{Fore.GREEN}Starting ProcrustesGPT sequential optimization...{Fore.RESET}")
    
    num_blocks = len(selected_blocks)
    transforms = {}
    cumulative_transform = None
    
    for i in range(num_blocks):
        logging.info(f"{Fore.YELLOW}Optimizing transform {i+1}/{num_blocks}...{Fore.RESET}")
        
        # Get activations for current block
        X = activations[f'block_{i}']['mlp_output']  # Input to transform
        Y_target = activations[f'block_{i}']['output'] - activations[f'block_{i}']['input']  # Target residual
        
        # Apply cumulative transform from previous steps
        if cumulative_transform is not None:
            X_transformed = X @ cumulative_transform.T
            # Add sequential constraint: current transform should be coherent with previous ones
            sequential_constraint = sequential_constraint_weight * torch.norm(
                X_transformed - X, dim=1, keepdim=True
            ).mean()
        else:
            X_transformed = X
            sequential_constraint = 0.0
        
        # Solve Procrustes problem: find orthogonal T that minimizes ||XT - Y||_F
        # with additional sequential and orthogonality constraints
        T_i = solve_constrained_procrustes(
            X_transformed, 
            Y_target, 
            sequential_constraint,
            orthogonal_regularization,
            max_iterations
        )
        
        transforms[f'T_{i}'] = T_i
        
        # Update cumulative transform for next iteration
        if cumulative_transform is None:
            cumulative_transform = T_i.clone()
        else:
            # Chain transformations: cumulative effect of T_1, T_2, ..., T_i
            cumulative_transform = cumulative_transform @ T_i
        
        # Log progress
        reconstruction_error = torch.norm(X_transformed @ T_i - Y_target, 'fro').item()
        orthogonality_error = torch.norm(T_i @ T_i.T - torch.eye(T_i.shape[0]), 'fro').item()
        
        logging.info(
            f"{Fore.GREEN}Block {i+1}: Reconstruction error = {reconstruction_error:.4f}, "
            f"Orthogonality error = {orthogonality_error:.4f}{Fore.RESET}"
        )
    
    return transforms


def solve_constrained_procrustes(
    X, Y, sequential_constraint, orthogonal_reg, max_iterations
):
    """
    Solve constrained Procrustes problem with sequential dependency and orthogonality constraints.
    
    Minimize: ||XT - Y||_F + λ₁ * sequential_constraint + λ₂ * ||TT^T - I||_F
    Subject to: T is orthogonal (approximately)
    """
    # Standard Procrustes solution as initialization
    U, _, Vt = torch.svd(X.T @ Y)
    T = U @ Vt
    
    # Refinement with alternating optimization
    for iteration in range(max_iterations):
        # Update T with gradient descent on the combined objective
        grad_reconstruction = X.T @ (X @ T - Y)
        grad_orthogonal = orthogonal_reg * (T @ T.T @ T - T)
        
        # Combined gradient
        total_grad = grad_reconstruction + grad_orthogonal
        
        # Gradient step
        T_new = T - 0.01 * total_grad
        
        # Project back to approximately orthogonal matrices using Procrustes
        U_new, _, Vt_new = torch.svd(T_new)
        T = U_new @ Vt_new
        
        # Check convergence
        if iteration > 0 and torch.norm(T - T_prev, 'fro').item() < 1e-6:
            break
        T_prev = T.clone()
    
    return T


def apply_sequential_transforms(model_path, transforms, selected_blocks, save_path, token):
    """
    Apply the sequential transforms to the model.
    """
    logging.info(f"{Fore.BLUE}Applying sequential transforms to model...{Fore.RESET}")
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Apply each transform and truncate corresponding blocks
    total_removed_layers = 0
    for i, (start_id, end_id) in enumerate(selected_blocks):
        T_i = transforms[f'T_{i}']
        
        # Adjust indices for previously removed layers
        adjusted_start = start_id - total_removed_layers
        adjusted_end = end_id - total_removed_layers
        
        # Apply transform to the MLP layer before the cut
        target_layer_idx = adjusted_start - 1
        if 'falcon' in model.config.model_type.lower():
            target_layer = model.transformer.h[target_layer_idx]
        else:
            target_layer = model.model.layers[target_layer_idx]
        
        # Merge transform with down_proj weight
        original_weight = target_layer.mlp.down_proj.weight.to(torch.float64)
        new_weight = (T_i.T @ original_weight.T).T.to(torch.bfloat16)
        target_layer.mlp.down_proj.load_state_dict({"weight": new_weight})
        
        # Remove the blocks
        model = truncate_model(model, adjusted_start, adjusted_end)
        total_removed_layers += (end_id - start_id)
        
        logging.info(f"{Fore.GREEN}Applied transform {i+1} and removed layers {adjusted_start}-{adjusted_end}{Fore.RESET}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_ProcrustesSequential"
    
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    tokenizer.save_pretrained(save_path)
    
    logging.info(f"{Fore.GREEN}Model saved to {save_path}{Fore.RESET}")
    return save_path

