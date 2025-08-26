import argparse
import gc
import logging
import os
from typing import Optional, Tuple
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_non_overlapping_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def activation_based_attribution(activations: torch.Tensor, target_activations: torch.Tensor) -> torch.Tensor:
    """
    GRASP-inspired activation-based attribution for singular value selection.
    Uses activation patterns instead of gradients for training-free approach.
    
    Args:
        activations: Input activations (N, d)
        target_activations: Target activations to approximate (N, d)
        
    Returns:
        attribution_scores: Importance scores for each dimension
    """
    print(f"[DEBUG] Computing activation-based attribution")
    print(f"[DEBUG] Activations shape: {activations.shape}")
    print(f"[DEBUG] Target activations shape: {target_activations.shape}")
    
    # Compute activation differences (proxy for gradient information)
    activation_diff = target_activations - activations
    
    # Compute variance-based importance (high variance = more important)
    activation_variance = torch.var(activations, dim=0)
    target_variance = torch.var(target_activations, dim=0)
    diff_variance = torch.var(activation_diff, dim=0)
    
    # Compute correlation-based importance
    correlation = torch.sum(activations * target_activations, dim=0) / (
        torch.norm(activations, dim=0) * torch.norm(target_activations, dim=0) + 1e-8
    )
    
    # Combine multiple attribution signals
    attribution_scores = (
        0.4 * activation_variance + 
        0.4 * target_variance + 
        0.2 * diff_variance
    ) * torch.abs(correlation)
    
    print(f"[DEBUG] Attribution scores range: [{attribution_scores.min():.6f}, {attribution_scores.max():.6f}]")
    print(f"[DEBUG] Attribution scores mean: {attribution_scores.mean():.6f}")
    
    return attribution_scores


def z_pruner_normalization(weight_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply Z-Pruner style normalization using statistical properties.
    
    Args:
        weight_matrix: Weight matrix to normalize
        
    Returns:
        normalized_weight: Statistically normalized weight matrix
    """
    print(f"[DEBUG] Applying Z-Pruner normalization")
    print(f"[DEBUG] Original weight norm: {torch.norm(weight_matrix):.6f}")
    
    # Compute Z-scores across weight matrices
    weight_mean = torch.mean(weight_matrix, dim=1, keepdim=True)
    weight_std = torch.std(weight_matrix, dim=1, keepdim=True) + 1e-8
    
    # Normalize using Z-scores
    normalized_weight = (weight_matrix - weight_mean) / weight_std
    
    # Apply activation-aware scaling
    scale_factor = torch.norm(weight_matrix) / torch.norm(normalized_weight)
    normalized_weight = normalized_weight * scale_factor
    
    print(f"[DEBUG] Normalized weight norm: {torch.norm(normalized_weight):.6f}")
    print(f"[DEBUG] Scale factor: {scale_factor:.6f}")
    
    return normalized_weight


def grasp_inspired_factorization(T: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, 
                                rank: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRASP-inspired factorization using activation-based attribution for singular value selection.
    
    Args:
        T: Original transformation matrix (d x d)
        a1: Input activations for attribution (N x d)
        a2: Target activations for attribution (N x d)
        rank: Target rank for low-rank approximation
        
    Returns:
        U: Left factor matrix (d x rank)
        V: Right factor matrix (d x rank)
    """
    print(f"[DEBUG] Starting GRASP-inspired factorization")
    print(f"[DEBUG] Transformation matrix T shape: {T.shape}")
    print(f"[DEBUG] Target rank: {rank}")
    
    # Move to CPU for SVD computation
    T_cpu = T.cpu().float()
    a1_sample = a1[:1000].cpu().float()  # Use sample for efficiency
    a2_sample = a2[:1000].cpu().float()
    
    # Perform SVD decomposition
    U, S, Vt = torch.svd(T_cpu)
    print(f"[DEBUG] SVD completed - singular values shape: {S.shape}")
    print(f"[DEBUG] Top 10 singular values: {S[:10].tolist()}")
    
    # Compute activation-based attribution for singular value selection
    # Project activations through singular vectors to get importance
    a1_projected = a1_sample @ U  # (N, d)
    a2_projected = a2_sample @ U  # (N, d)
    
    # Compute attribution scores for each singular direction
    attribution_scores = activation_based_attribution(a1_projected, a2_projected)
    
    # Combine singular values with attribution scores
    # Higher attribution + higher singular value = more important
    importance_scores = S * (attribution_scores + 1e-8)
    
    # Select top-k most important directions (not just top-k singular values)
    _, important_indices = torch.topk(importance_scores, rank)
    important_indices = important_indices.sort()[0]
    
    print(f"[DEBUG] Selected indices range: [{important_indices.min()}, {important_indices.max()}]")
    print(f"[DEBUG] Traditional top-{rank} vs GRASP-inspired difference: {torch.sum(important_indices != torch.arange(rank))}")
    
    # Create low-rank factors using selected directions
    U_selected = U[:, important_indices].contiguous()
    S_selected = S[important_indices]
    V_selected = Vt[important_indices, :].T.contiguous()
    
    # Scale by selected singular values
    U_lr = U_selected @ torch.diag(torch.sqrt(S_selected))
    V_lr = V_selected @ torch.diag(torch.sqrt(S_selected))
    
    # Apply Z-Pruner normalization
    U_lr = z_pruner_normalization(U_lr)
    V_lr = z_pruner_normalization(V_lr)
    
    # Verify reconstruction quality
    T_reconstructed = U_lr @ V_lr.T
    reconstruction_error = torch.norm(T_cpu - T_reconstructed) / torch.norm(T_cpu)
    
    print(f"[DEBUG] Reconstruction error: {reconstruction_error:.6f}")
    print(f"[DEBUG] Selected singular values mean: {S_selected.mean():.6f}")
    print(f"[DEBUG] Standard approach would select mean: {S[:rank].mean():.6f}")
    
    result_U = U_lr.to(T.dtype)
    result_V = V_lr.to(T.dtype)
    
    print(f"[DEBUG] GRASP-inspired factorization completed")
    
    return result_U, result_V


def apply_enhanced_transform(model, layer_idx: int, U: torch.Tensor, V: torch.Tensor):
    """
    Apply enhanced transformation with stabilized layer normalization.
    
    Args:
        model: The transformer model
        layer_idx: Index of the layer to modify
        U: Left factor matrix
        V: Right factor matrix
    """
    print(f"[DEBUG] Applying enhanced transform to layer {layer_idx}")
    
    # Get the down_proj layer
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    original_weight = down_proj.weight.data
    
    print(f"[DEBUG] Original down_proj weight shape: {original_weight.shape}")
    print(f"[DEBUG] U shape: {U.shape}, V shape: {V.shape}")
    
    # Store original weight norm for comparison
    original_norm = torch.norm(original_weight).item()
    print(f"[DEBUG] Original weight norm: {original_norm:.6f}")
    
    # Ensure all tensors are on the same device
    device = original_weight.device
    dtype = original_weight.dtype
    
    U = U.to(device=device, dtype=torch.float64)
    V = V.to(device=device, dtype=torch.float64)
    original_weight_fp64 = original_weight.to(torch.float64)
    
    # Create transformation: (UV^T)^T @ W
    transform_matrix = U @ V.T
    transformed_weight = transform_matrix.T @ original_weight_fp64
    
    print(f"[DEBUG] Transform matrix shape: {transform_matrix.shape}")
    print(f"[DEBUG] Transform matrix norm: {torch.norm(transform_matrix):.6f}")
    
    # Apply conservative scaling to preserve weight norms
    weight_scale = original_norm / (torch.norm(transformed_weight) + 1e-8)
    conservative_scale = min(weight_scale, 1.2)  # Limit scaling
    final_weight = transformed_weight * conservative_scale
    
    print(f"[DEBUG] Weight scale factor: {weight_scale:.6f}")
    print(f"[DEBUG] Conservative scale applied: {conservative_scale:.6f}")
    
    # Verify transformation
    final_norm = torch.norm(final_weight).item()
    print(f"[DEBUG] Final weight norm: {final_norm:.6f}")
    print(f"[DEBUG] Weight norm ratio (new/old): {final_norm/original_norm:.6f}")
    
    # Update the weight
    down_proj.weight.data = final_weight.to(dtype)
    
    # Apply SLNP (Stabilized LayerNorm Pruning) if RMSNorm exists
    if hasattr(model.model.layers[layer_idx], 'input_layernorm'):
        layernorm = model.model.layers[layer_idx].input_layernorm
        if hasattr(layernorm, 'weight'):
            original_gamma_norm = torch.norm(layernorm.weight.data)
            # Re-scale based on transformation impact
            rescale_factor = final_norm / original_norm
            layernorm.weight.data = layernorm.weight.data * rescale_factor
            print(f"[DEBUG] Applied SLNP rescaling: {rescale_factor:.6f}")
    
    # Verify the update was applied
    new_weight = down_proj.weight.data
    new_norm = torch.norm(new_weight).item()
    print(f"[DEBUG] After assignment - new weight norm: {new_norm:.6f}")
    
    print(f"[DEBUG] Successfully applied enhanced transform to layer {layer_idx}")
    logging.info(f"{Fore.GREEN}Applied GRASP-inspired transform to layer {layer_idx}{Fore.RESET}")


def grasp_inspired_replace(
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
    diag: bool = False,
    loss: str = "cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    low_rank: int = 1024
) -> str:
    """GRASP-inspired replacement with activation-based attribution and enhanced normalization.
    
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
        diag: Whether to use diagonal matrix
        loss: Loss function type
        solver: Optimization solver type
        thri: Whether to use three vectors
        two_vectors: Whether to use two vectors
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
        accurate: Whether to use accurate mode
        low_rank: Rank for low-rank factorization
    
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
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
        def hook(module, input, output):
            # Detach to avoid keeping computation history
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
    if accurate:
        print("ACCURATE MODE IS ON (MORE MEMORY IS NEEDED)")
        a3 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations for GRASP-inspired Method" + Fore.RESET,
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
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]    
    
    # Compute full-rank transformation first
    print(f"[DEBUG] Computing full-rank transformation matrix T")
    print(f"[DEBUG] Calibration data processed: {cnt} tokens")
    
    if solver == "adam":
        print(f"[DEBUG] Using Adam solver with loss: {loss}")
        transform = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        print(f"[DEBUG] Using optimization method with solver: {solver}")
        transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    print(f"[DEBUG] Full-rank transformation T computed")
    print(f"[DEBUG] T shape: {transform.shape}")
    print(f"[DEBUG] T norm: {torch.norm(transform):.6f}")
    
    # Apply GRASP-inspired factorization using activation-based attribution
    logging.info(f"{Fore.GREEN}Applying GRASP-inspired factorization with rank {low_rank}{Fore.RESET}")
    print(f"[DEBUG] Starting GRASP-inspired factorization process")
    U, V = grasp_inspired_factorization(transform, a1, a2, rank=low_rank)
    
    # Clean up
    print(f"[DEBUG] Cleaning up GPU memory and reloading model")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    print(f"[DEBUG] Reloading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    print(f"[DEBUG] Model reloaded successfully")
    
    print(f"[DEBUG] Truncating model - removing layers {start_id - num_layer} to {end_id - num_layer}")
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    print(f"[DEBUG] Model truncated - new layer count: {len(model.model.layers)}")
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_grasp_inspired_r{low_rank}"
        ).replace("/", "_")
    
    print(f"[DEBUG] Save path: {save_path}_GRASPInspired_{loss}_{solver}")
    
    # Apply GRASP-inspired transformation
    target_layer = start_id - num_layer - 1
    print(f"[DEBUG] Applying GRASP-inspired transformation to layer index: {target_layer}")
    apply_enhanced_transform(model, target_layer, U, V)
    
    final_save_path = f"{save_path}_GRASPInspired_{loss}_{solver}_r{low_rank}"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"[DEBUG] Model saved to: {final_save_path}")
    
    if save_transform_only:
        transform_save_path = f"{final_save_path}_factors"
        torch.save({
            'U': U,
            'V': V,
            'rank': low_rank,
            'original_transform': transform
        }, transform_save_path)
        print(f"[DEBUG] Transform factors saved to: {transform_save_path}")
    
    # Final cleanup
    print(f"[DEBUG] Final cleanup")
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"[DEBUG] GRASP-inspired replacement completed successfully")
    print(f"[DEBUG] Final model path: {final_save_path}")
    
    return final_save_path

