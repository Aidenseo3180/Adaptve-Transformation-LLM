import argparse
import gc
import logging
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (select_non_overlapping_blocks, truncate_model, 
                    seed_all, get_calib_dataloader)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

def compute_hierarchical_transform(
    a1: torch.Tensor, 
    a2: torch.Tensor, 
    rank: int = 1024,
    sparsity_ratio: float = 0.05,
    max_iterations: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hierarchical transformation: T = T_coarse + T_fine
    
    Args:
        a1: Input activations (MLP output) [N, d]
        a2: Target activations (Li+n - Yi) [N, d] 
        rank: Rank for low-rank decomposition of T_coarse
        sparsity_ratio: Sparsity ratio for T_fine (0.05 = 5% non-zero)
        max_iterations: Maximum iterations for optimization
        
    Returns:
        U: Left factor of T_coarse [d, rank]
        V: Right factor of T_coarse [rank, d] 
        T_fine: Sparse fine-grained correction matrix [d, d]
    """
    device = a1.device
    d = a1.shape[1]
    
    print(f"{Fore.GREEN}[DEBUG] Starting hierarchical transform computation{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Input shape: {a1.shape}, Target shape: {a2.shape}{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Rank: {rank}, Sparsity ratio: {sparsity_ratio}{Fore.RESET}")
    
    # Step 1: Compute optimal full transformation using least squares
    print(f"{Fore.YELLOW}[DEBUG] Computing optimal full transformation (LS solution){Fore.RESET}")
    
    # Solve: a1 @ T_optimal = a2 => T_optimal = (a1^T @ a1)^-1 @ a1^T @ a2
    try:
        AtA = a1.T @ a1
        AtA_inv = torch.linalg.pinv(AtA)  # Use pseudo-inverse for numerical stability
        T_optimal = AtA_inv @ a1.T @ a2
        print(f"{Fore.GREEN}[DEBUG] Optimal transform computed successfully, shape: {T_optimal.shape}{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to compute optimal transform: {e}{Fore.RESET}")
        raise
    
    # Step 2: Low-rank approximation for T_coarse
    print(f"{Fore.YELLOW}[DEBUG] Computing low-rank approximation (rank={rank}){Fore.RESET}")
    
    try:
        # SVD decomposition: T_optimal = U_svd @ S @ V_svd^T
        U_svd, S, Vt_svd = torch.linalg.svd(T_optimal, full_matrices=False)
        
        # Keep only top-rank components
        rank = min(rank, S.shape[0])  # Ensure rank doesn't exceed matrix rank
        print(f"{Fore.CYAN}[DEBUG] Actual rank used: {rank} (max possible: {S.shape[0]}){Fore.RESET}")
        
        # Compute low-rank factors
        U = U_svd[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))  # [d, rank]
        V = torch.diag(torch.sqrt(S[:rank])) @ Vt_svd[:rank, :]  # [rank, d]
        
        T_coarse = U @ V
        print(f"{Fore.GREEN}[DEBUG] Low-rank decomposition completed{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG] U shape: {U.shape}, V shape: {V.shape}{Fore.RESET}")
        
        # Compute reconstruction error
        coarse_error = torch.norm(T_optimal - T_coarse).item()
        print(f"{Fore.YELLOW}[DEBUG] Coarse approximation error: {coarse_error:.6f}{Fore.RESET}")
        
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to compute low-rank approximation: {e}{Fore.RESET}")
        raise
    
    # Step 3: Compute sparse residual for T_fine
    print(f"{Fore.YELLOW}[DEBUG] Computing sparse residual correction{Fore.RESET}")
    
    try:
        # Compute residual
        residual = T_optimal - T_coarse
        residual_norm = torch.norm(residual).item()
        print(f"{Fore.CYAN}[DEBUG] Residual norm: {residual_norm:.6f}{Fore.RESET}")
        
        # Create sparse mask based on magnitude
        residual_flat = residual.flatten()
        threshold_idx = int(len(residual_flat) * (1 - sparsity_ratio))
        threshold_val = torch.kthvalue(torch.abs(residual_flat), threshold_idx).values
        
        sparse_mask = (torch.abs(residual) >= threshold_val)
        T_fine = residual * sparse_mask
        
        sparsity_actual = (T_fine == 0).float().mean().item()
        print(f"{Fore.CYAN}[DEBUG] Actual sparsity: {sparsity_actual:.3f} (target: {1-sparsity_ratio:.3f}){Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG] Non-zero elements in T_fine: {(T_fine != 0).sum().item()}{Fore.RESET}")
        
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to compute sparse residual: {e}{Fore.RESET}")
        raise
    
    # Step 4: Verify reconstruction quality
    print(f"{Fore.YELLOW}[DEBUG] Verifying reconstruction quality{Fore.RESET}")
    
    T_reconstructed = T_coarse + T_fine
    reconstruction_error = torch.norm(T_optimal - T_reconstructed).item()
    relative_error = reconstruction_error / torch.norm(T_optimal).item()
    
    print(f"{Fore.GREEN}[DEBUG] Final reconstruction error: {reconstruction_error:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}[DEBUG] Relative error: {relative_error:.6f}{Fore.RESET}")
    
    # Step 5: Test on actual data
    print(f"{Fore.YELLOW}[DEBUG] Testing transformation on calibration data{Fore.RESET}")
    
    try:
        # Original transformation
        output_original = a1 @ T_optimal
        
        # Hierarchical transformation  
        output_coarse = a1 @ U @ V
        output_fine = a1 @ T_fine
        output_hierarchical = output_coarse + output_fine
        
        # Compute errors
        data_error = torch.norm(output_original - output_hierarchical).item()
        target_error_original = torch.norm(output_original - a2).item()
        target_error_hierarchical = torch.norm(output_hierarchical - a2).item()
        
        print(f"{Fore.CYAN}[DEBUG] Data transformation error: {data_error:.6f}{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG] Target error (original): {target_error_original:.6f}{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG] Target error (hierarchical): {target_error_hierarchical:.6f}{Fore.RESET}")
        
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to test transformation: {e}{Fore.RESET}")
        raise
    
    print(f"{Fore.GREEN}[DEBUG] Hierarchical transform computation completed successfully!{Fore.RESET}")
    
    return U.to(torch.float64), V.to(torch.float64), T_fine.to(torch.float64)


def hierarchical_transform(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    rank: int = 1024,
    sparsity_ratio: float = 0.05
) -> str:
    """
    Apply hierarchical multi-scale linear transformation to replace transformer blocks.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of start layer for cut
        token: Authentication token
        save_transform_only: Whether to only save the transform
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
        rank: Rank for low-rank decomposition
        sparsity_ratio: Sparsity ratio for fine-grained correction
    
    Returns:
        Path where transformed model is saved
    """
    
    print(f"{Fore.MAGENTA}=== Starting Hierarchical Multi-Scale Transformation ==={Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Model: {model_path}{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Layers to replace: {start_id} to {end_id}{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Rank: {rank}, Sparsity: {sparsity_ratio}{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        print(f"{Fore.YELLOW}[DEBUG] Using 4-bit quantization{Fore.RESET}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model
    print(f"{Fore.YELLOW}[DEBUG] Loading model...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
    print(f"{Fore.CYAN}[DEBUG] Model loaded. Hidden size: {hidden_size}{Fore.RESET}")
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get calibration data
    print(f"{Fore.YELLOW}[DEBUG] Loading calibration data...{Fore.RESET}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    # Register hooks
    print(f"{Fore.YELLOW}[DEBUG] Registering activation hooks...{Fore.RESET}")
    hooks = []
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Collect activations
    print(f"{Fore.YELLOW}[DEBUG] Collecting activations for hierarchical transform estimation...{Fore.RESET}")
    
    mlp_activations = {}
    a1 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations for Hierarchical Transform" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
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
        a2_batch = hidden_states_n - hidden_states_i  # Target: Li+n - Yi
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    
    print(f"{Fore.GREEN}[DEBUG] Collected {cnt} activation samples{Fore.RESET}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute hierarchical transformation
    print(f"{Fore.MAGENTA}[DEBUG] Computing hierarchical transformation...{Fore.RESET}")
    U, V, T_fine = compute_hierarchical_transform(
        a1.to(torch.float64), 
        a2.to(torch.float64), 
        rank=rank,
        sparsity_ratio=sparsity_ratio
    )
    
    # Clean up activations
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clean up model and reload for transformation
    print(f"{Fore.YELLOW}[DEBUG] Reloading model for transformation application...{Fore.RESET}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Prepare save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply hierarchical transformation to down_proj
    print(f"{Fore.YELLOW}[DEBUG] Applying hierarchical transformation to model...{Fore.RESET}")
    
    try:
        target_layer = model.model.layers[start_id - num_layer - 1]
        original_weight = target_layer.mlp.down_proj.weight.to(torch.float64)
        
        print(f"{Fore.CYAN}[DEBUG] Original down_proj weight shape: {original_weight.shape}{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG] U shape: {U.shape}, V shape: {V.shape}, T_fine shape: {T_fine.shape}{Fore.RESET}")
        
        # Compute the combined transformation: T_total = U @ V + T_fine
        T_coarse = U @ V
        T_total = T_coarse + T_fine
        
        # Apply to down_proj: new_weight = T_total.T @ original_weight
        new_weight = (T_total.T.cpu() @ original_weight).to(torch.bfloat16)
        
        target_layer.mlp.down_proj.load_state_dict({
            "weight": new_weight
        })
        
        print(f"{Fore.GREEN}[DEBUG] Successfully applied hierarchical transformation to down_proj{Fore.RESET}")
        
        # Compute transformation statistics
        coarse_norm = torch.norm(T_coarse).item()
        fine_norm = torch.norm(T_fine).item()
        total_norm = torch.norm(T_total).item()
        sparsity_actual = (T_fine == 0).float().mean().item()
        
        print(f"{Fore.CYAN}[DEBUG] Transformation stats:{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG]   Coarse norm: {coarse_norm:.6f}{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG]   Fine norm: {fine_norm:.6f}{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG]   Total norm: {total_norm:.6f}{Fore.RESET}")
        print(f"{Fore.CYAN}[DEBUG]   Fine sparsity: {sparsity_actual:.3f}{Fore.RESET}")
        
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to apply transformation: {e}{Fore.RESET}")
        raise
    
    # Save model
    save_dir = f"{save_path}_Hierarchical_r{rank}_s{int(sparsity_ratio*100)}"
    print(f"{Fore.YELLOW}[DEBUG] Saving model to: {save_dir}{Fore.RESET}")
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    if save_transform_only:
        print(f"{Fore.YELLOW}[DEBUG] Saving transformation components...{Fore.RESET}")
        torch.save({
            'U': U,
            'V': V, 
            'T_fine': T_fine,
            'rank': rank,
            'sparsity_ratio': sparsity_ratio,
            'T_coarse': T_coarse,
            'T_total': T_total
        }, f"{save_dir}_transform_components")
    
    # Final cleanup
    del model, U, V, T_fine
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.MAGENTA}=== Hierarchical Multi-Scale Transformation Complete ==={Fore.RESET}")
    print(f"{Fore.GREEN}[DEBUG] Model saved to: {save_dir}{Fore.RESET}")
    
    return save_dir


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the hierarchical transformation from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run hierarchical multi-scale linear transformation based on a configuration file."
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
        path = hierarchical_transform(**config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
        config["model_path"] = path