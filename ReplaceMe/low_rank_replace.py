import argparse
import gc
import logging
import os
from typing import Optional
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


def low_rank_factorization(T: torch.Tensor, rank: int = 1024) -> tuple:
    """
    Decompose transformation matrix T into low-rank factors U and V.
    
    Args:
        T: Original transformation matrix (d x d)
        rank: Target rank for decomposition
        
    Returns:
        U: Left factor matrix (d x rank)
        V: Right factor matrix (d x rank)
    """
    print(f"[DEBUG] Starting low-rank factorization")
    print(f"[DEBUG] Original T shape: {T.shape}")
    print(f"[DEBUG] Original T device: {T.device}")
    print(f"[DEBUG] Target rank: {rank}")
    print(f"[DEBUG] Parameter reduction: {T.numel()} -> {2 * T.shape[0] * rank} ({(1 - 2*rank/T.shape[0])*100:.1f}% reduction)")
    
    logging.info(f"{Fore.GREEN}Performing SVD decomposition with rank {rank}{Fore.RESET}")
    
    # Move to CPU for SVD computation to avoid memory issues
    T_cpu = T.cpu().float()
    print(f"[DEBUG] Moved T to CPU for SVD computation")
    
    # Perform SVD decomposition
    U, S, Vt = torch.svd(T_cpu)
    print(f"[DEBUG] SVD completed - U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
    print(f"[DEBUG] Top 10 singular values: {S[:10].tolist()}")
    
    # Take top-k singular values and vectors
    U_lr = U[:, :rank].contiguous()
    S_lr = S[:rank]
    V_lr = Vt[:rank, :].T.contiguous()
    print(f"[DEBUG] After rank reduction - U_lr: {U_lr.shape}, V_lr: {V_lr.shape}")
    
    # Scale by singular values
    U_lr = U_lr @ torch.diag(torch.sqrt(S_lr))
    V_lr = V_lr @ torch.diag(torch.sqrt(S_lr))
    print(f"[DEBUG] Applied singular value scaling")
    
    # Verify reconstruction
    T_reconstructed = U_lr @ V_lr.T
    reconstruction_error = torch.norm(T_cpu - T_reconstructed) / torch.norm(T_cpu)
    
    print(f"[DEBUG] Reconstruction verification:")
    print(f"[DEBUG] - Reconstructed T shape: {T_reconstructed.shape}")
    print(f"[DEBUG] - Reconstruction error: {reconstruction_error:.6f}")
    print(f"[DEBUG] - Original T norm: {torch.norm(T_cpu):.6f}")
    print(f"[DEBUG] - Reconstructed T norm: {torch.norm(T_reconstructed):.6f}")
    
    logging.info(f"{Fore.GREEN}Reconstruction error: {reconstruction_error:.6f}{Fore.RESET}")
    
    # Keep on CPU initially - will be moved to correct device in apply_low_rank_transform
    result_U = U_lr.to(T.dtype)
    result_V = V_lr.to(T.dtype)
    print(f"[DEBUG] Final U shape: {result_U.shape}, V shape: {result_V.shape}")
    print(f"[DEBUG] Low-rank factorization completed successfully")
    
    return result_U, result_V


def apply_low_rank_transform(model, layer_idx: int, U: torch.Tensor, V: torch.Tensor):
    """
    Apply low-rank transformation to model by modifying down_proj weights.
    
    Args:
        model: The transformer model
        layer_idx: Index of the layer to modify
        U: Left factor matrix
        V: Right factor matrix
    """
    print(f"[DEBUG] Applying low-rank transform to layer {layer_idx}")
    
    # Get the down_proj layer
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    original_weight = down_proj.weight.data
    
    print(f"[DEBUG] Original down_proj weight shape: {original_weight.shape}")
    print(f"[DEBUG] Original down_proj weight device: {original_weight.device}")
    print(f"[DEBUG] Original down_proj weight dtype: {original_weight.dtype}")
    print(f"[DEBUG] Input U shape: {U.shape}, V shape: {V.shape}")
    print(f"[DEBUG] Input U device: {U.device}, V device: {V.device}")
    
    # Store original weight norm for comparison
    original_norm = torch.norm(original_weight).item()
    print(f"[DEBUG] Original weight norm: {original_norm:.6f}")
    
    # Ensure all tensors are on the same device as the original weight
    device = original_weight.device
    dtype = original_weight.dtype
    
    U = U.to(device=device, dtype=torch.float64)
    V = V.to(device=device, dtype=torch.float64)
    original_weight_fp64 = original_weight.to(torch.float64)
    
    print(f"[DEBUG] After device transfer - U device: {U.device}, V device: {V.device}")
    print(f"[DEBUG] Computing transformation: V @ U.T @ original_weight")
    
    # Apply low-rank transformation: W_new = V^T @ U^T @ W_original
    # Since T = U @ V^T, the transformation becomes (U @ V^T)^T @ W = V @ U^T @ W
    intermediate = U.T @ original_weight_fp64
    print(f"[DEBUG] Intermediate result shape (U.T @ W): {intermediate.shape}")
    
    transformed_weight = V @ intermediate
    print(f"[DEBUG] Final transformed weight shape: {transformed_weight.shape}")
    
    # Verify transformation
    transformed_norm = torch.norm(transformed_weight).item()
    print(f"[DEBUG] Transformed weight norm: {transformed_norm:.6f}")
    print(f"[DEBUG] Weight norm ratio (new/old): {transformed_norm/original_norm:.6f}")
    
    # Update the weight
    down_proj.weight.data = transformed_weight.to(dtype)
    
    # Verify the update was applied
    new_weight = down_proj.weight.data
    new_norm = torch.norm(new_weight).item()
    print(f"[DEBUG] After assignment - new weight norm: {new_norm:.6f}")
    print(f"[DEBUG] Assignment verification: {torch.allclose(new_weight.float(), transformed_weight.float(), atol=1e-5)}")
    
    print(f"[DEBUG] Successfully applied low-rank transform to layer {layer_idx}")
    logging.info(f"{Fore.GREEN}Applied low-rank transform to layer {layer_idx}{Fore.RESET}")


def low_rank_replace(
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
    """Calculate cosine distance between model layers and apply low-rank transformations.
    
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
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
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
    print(f"[DEBUG] Input activation shapes - a1: {a1.shape}, a2: {a2.shape}")
    if accurate:
        print(f"[DEBUG] a3 shape (accurate mode): {a3.shape}")
    
    if solver == "adam":
        print(f"[DEBUG] Using Adam solver with loss: {loss}")
        transform = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        print(f"[DEBUG] Using optimization method with solver: {solver}")
        transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    print(f"[DEBUG] Full-rank transformation T computed")
    print(f"[DEBUG] T shape: {transform.shape}")
    print(f"[DEBUG] T device: {transform.device}")
    print(f"[DEBUG] T dtype: {transform.dtype}")
    print(f"[DEBUG] T norm: {torch.norm(transform):.6f}")
    
    # Apply low-rank factorization
    logging.info(f"{Fore.GREEN}Applying low-rank factorization with rank {low_rank}{Fore.RESET}")
    print(f"[DEBUG] Starting low-rank factorization process")
    U, V = low_rank_factorization(transform, rank=low_rank)
    
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
            f"{end_id}_{dataset}_{dataset_size}_lowrank_{low_rank}"
        ).replace("/", "_")
    
    print(f"[DEBUG] Save path: {save_path}_LowRank_{loss}_{solver}_r{low_rank}")
    
    # Apply low-rank transformation
    target_layer = start_id - num_layer - 1
    print(f"[DEBUG] Applying low-rank transformation to layer index: {target_layer}")
    apply_low_rank_transform(model, target_layer, U, V)
    
    model.save_pretrained(f"{save_path}_LowRank_{loss}_{solver}_r{low_rank}")
    tokenizer.save_pretrained(f"{save_path}_LowRank_{loss}_{solver}_r{low_rank}")
    print(f"[DEBUG] Model saved to: {save_path}_LowRank_{loss}_{solver}_r{low_rank}")
    
    if save_transform_only:
        transform_save_path = f"{save_path}_LowRank_{loss}_{solver}_r{low_rank}_factors"
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
    
    final_path = f"{save_path}_LowRank_{loss}_{solver}_r{low_rank}"
    print(f"[DEBUG] Low-rank replacement completed successfully")
    print(f"[DEBUG] Final model path: {final_path}")
    
    return final_path

