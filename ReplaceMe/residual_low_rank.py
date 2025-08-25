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


def residual_low_rank_factorization(T: torch.Tensor, rank: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose transformation matrix T into residual low-rank factors U and V.
    T_residual = I + α(U @ V^T)
    
    Args:
        T: Original transformation matrix (d x d)
        rank: Target rank for decomposition
        
    Returns:
        U: Left factor matrix (d x rank)
        V: Right factor matrix (d x rank)
    """
    print(f"[DEBUG] Starting residual low-rank factorization")
    print(f"[DEBUG] Original T shape: {T.shape}")
    print(f"[DEBUG] Original T device: {T.device}")
    print(f"[DEBUG] Target rank: {rank}")
    
    # Compute residual matrix: T_residual = T - I
    I = torch.eye(T.shape[0], dtype=T.dtype, device=T.device)
    T_residual = T - I
    print(f"[DEBUG] Computed T_residual = T - I")
    print(f"[DEBUG] T_residual norm: {torch.norm(T_residual):.6f}")
    print(f"[DEBUG] Original T norm: {torch.norm(T):.6f}")
    print(f"[DEBUG] Identity preservation ratio: {torch.norm(I) / torch.norm(T):.6f}")
    
    logging.info(f"{Fore.GREEN}Performing SVD decomposition on residual matrix with rank {rank}{Fore.RESET}")
    
    # Move to CPU for SVD computation to avoid memory issues
    T_residual_cpu = T_residual.cpu().float()
    print(f"[DEBUG] Moved T_residual to CPU for SVD computation")
    
    # Perform SVD decomposition on residual
    U, S, Vt = torch.svd(T_residual_cpu)
    print(f"[DEBUG] SVD completed - U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
    print(f"[DEBUG] Top 10 singular values of residual: {S[:10].tolist()}")
    
    # Take top-k singular values and vectors
    U_lr = U[:, :rank].contiguous()
    S_lr = S[:rank]
    V_lr = Vt[:rank, :].T.contiguous()
    print(f"[DEBUG] After rank reduction - U_lr: {U_lr.shape}, V_lr: {V_lr.shape}")
    
    # Scale by singular values
    U_lr = U_lr @ torch.diag(torch.sqrt(S_lr))
    V_lr = V_lr @ torch.diag(torch.sqrt(S_lr))
    print(f"[DEBUG] Applied singular value scaling")
    
    # Verify reconstruction of residual
    T_residual_reconstructed = U_lr @ V_lr.T
    reconstruction_error = torch.norm(T_residual_cpu - T_residual_reconstructed) / torch.norm(T_residual_cpu)
    
    print(f"[DEBUG] Residual reconstruction verification:")
    print(f"[DEBUG] - Reconstructed T_residual shape: {T_residual_reconstructed.shape}")
    print(f"[DEBUG] - Reconstruction error: {reconstruction_error:.6f}")
    print(f"[DEBUG] - Original T_residual norm: {torch.norm(T_residual_cpu):.6f}")
    print(f"[DEBUG] - Reconstructed T_residual norm: {torch.norm(T_residual_reconstructed):.6f}")
    
    logging.info(f"{Fore.GREEN}Residual reconstruction error: {reconstruction_error:.6f}{Fore.RESET}")
    
    # Keep on CPU initially - will be moved to correct device in apply function
    result_U = U_lr.to(T.dtype)
    result_V = V_lr.to(T.dtype)
    print(f"[DEBUG] Final U shape: {result_U.shape}, V shape: {result_V.shape}")
    print(f"[DEBUG] Residual low-rank factorization completed successfully")
    
    return result_U, result_V


def find_optimal_alpha(model_path: str, tokenizer, U: torch.Tensor, V: torch.Tensor, 
                      target_layer: int, original_weight: torch.Tensor,
                      calib_data_sample: list, alpha_candidates: list = [0.1, 0.2, 0.3, 0.4, 0.5]) -> float:
    """
    Find optimal alpha value by testing different candidates on calibration data.
    
    Args:
        model_path: Path to the model
        tokenizer: Tokenizer for the model  
        U, V: Low-rank factors
        target_layer: Layer index to modify
        original_weight: Original down_proj weight
        calib_data_sample: Small sample of calibration data for evaluation
        alpha_candidates: List of alpha values to test
        
    Returns:
        optimal_alpha: Best alpha value
    """
    print(f"[DEBUG] Finding optimal alpha value")
    print(f"[DEBUG] Alpha candidates: {alpha_candidates}")
    print(f"[DEBUG] Calibration sample size: {len(calib_data_sample)}")
    
    best_alpha = alpha_candidates[0]
    best_perplexity = float('inf')
    
    device = original_weight.device
    dtype = original_weight.dtype
    
    print(f"[DEBUG] Original weight shape: {original_weight.shape}")
    print(f"[DEBUG] U shape: {U.shape}, V shape: {V.shape}")
    
    # Move U, V to correct device
    U = U.to(device=device, dtype=torch.float64)
    V = V.to(device=device, dtype=torch.float64)
    original_weight_fp64 = original_weight.to(torch.float64)
    
    # The correct dimension for identity should match the transformation dimensions
    # down_proj weight is (hidden_dim, intermediate_dim)
    # So transformation should be (intermediate_dim, intermediate_dim)
    transform_dim = original_weight.shape[1]  # intermediate_dim
    print(f"[DEBUG] Transform dimension should be: {transform_dim}")
    print(f"[DEBUG] U @ V.T will have shape: ({U.shape[0]}, {V.shape[0]})")
    
    for alpha in alpha_candidates:
        print(f"[DEBUG] Testing alpha = {alpha}")
        
        # Create residual transformation: I + α(UV^T)
        # The identity should match the dimension of U @ V.T
        identity_transform = torch.eye(U.shape[0], device=device, dtype=torch.float64)
        residual_component = alpha * (U @ V.T)
        residual_transform = identity_transform + residual_component
        
        print(f"[DEBUG] Identity transform shape: {identity_transform.shape}")
        print(f"[DEBUG] Residual component shape: {residual_component.shape}")
        print(f"[DEBUG] Final residual transform shape: {residual_transform.shape}")
        
        # Apply transformation: W_new = T^T @ W_original
        transformed_weight = residual_transform.T @ original_weight_fp64
        
        print(f"[DEBUG] Transformed weight shape: {transformed_weight.shape}")
        
        # Load a fresh model for testing
        test_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cpu',
            torch_dtype=torch.bfloat16
        )
        
        # Apply transformation to test model
        test_model.model.layers[target_layer].mlp.down_proj.weight.data = transformed_weight.to(dtype)
        test_model = test_model.to(device)
        test_model.eval()
        
        # Compute perplexity on calibration sample
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in calib_data_sample[:10]:  # Use small sample for speed
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = test_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        print(f"[DEBUG] Alpha {alpha}: Perplexity = {perplexity:.6f}")
        
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_alpha = alpha
        
        # Clean up
        del test_model
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"[DEBUG] Optimal alpha found: {best_alpha} (Perplexity: {best_perplexity:.6f})")
    return best_alpha


def apply_residual_low_rank_transform(model, layer_idx: int, U: torch.Tensor, V: torch.Tensor, alpha: float):
    """
    Apply residual low-rank transformation to model: W_new = (I + α(UV^T))^T @ W_original
    
    Args:
        model: The transformer model
        layer_idx: Index of the layer to modify
        U: Left factor matrix
        V: Right factor matrix  
        alpha: Scaling factor for residual component
    """
    print(f"[DEBUG] Applying residual low-rank transform to layer {layer_idx}")
    print(f"[DEBUG] Using alpha = {alpha}")
    
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
    
    # Create residual transformation: I + α(UV^T)
    # The identity should match the dimension of the transformation matrix
    identity_transform = torch.eye(U.shape[0], device=device, dtype=torch.float64)
    residual_component = alpha * (U @ V.T)
    residual_transform = identity_transform + residual_component
    
    print(f"[DEBUG] Identity transform shape: {identity_transform.shape}")
    print(f"[DEBUG] Residual component shape: {residual_component.shape}")
    print(f"[DEBUG] Residual component norm: {torch.norm(residual_component):.6f}")
    print(f"[DEBUG] Final residual transform shape: {residual_transform.shape}")
    print(f"[DEBUG] Final residual transform norm: {torch.norm(residual_transform):.6f}")
    
    # Apply transformation: W_new = T^T @ W_original = (I + α(UV^T))^T @ W
    transformed_weight = residual_transform.T @ original_weight_fp64
    print(f"[DEBUG] Final transformed weight shape: {transformed_weight.shape}")
    
    # Verify transformation
    transformed_norm = torch.norm(transformed_weight).item()
    print(f"[DEBUG] Transformed weight norm: {transformed_norm:.6f}")
    print(f"[DEBUG] Weight norm ratio (new/old): {transformed_norm/original_norm:.6f}")
    
    # Check how much change was applied
    weight_change = transformed_weight - original_weight_fp64
    change_norm = torch.norm(weight_change).item()
    print(f"[DEBUG] Weight change norm: {change_norm:.6f}")
    print(f"[DEBUG] Weight change ratio (change/original): {change_norm/original_norm:.6f}")
    
    # Update the weight
    down_proj.weight.data = transformed_weight.to(dtype)
    
    # Verify the update was applied
    new_weight = down_proj.weight.data
    new_norm = torch.norm(new_weight).item()
    print(f"[DEBUG] After assignment - new weight norm: {new_norm:.6f}")
    print(f"[DEBUG] Assignment verification: {torch.allclose(new_weight.float(), transformed_weight.float(), atol=1e-5)}")
    
    print(f"[DEBUG] Successfully applied residual low-rank transform to layer {layer_idx}")
    logging.info(f"{Fore.GREEN}Applied residual low-rank transform to layer {layer_idx} with alpha={alpha}{Fore.RESET}")


def residual_low_rank_replace(
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
    low_rank: int = 1024,
    alpha_candidates: list = [0.1, 0.2, 0.3, 0.4, 0.5],
    auto_tune_alpha: bool = True
) -> str:
    """Calculate transformation and apply residual low-rank approximation.
    
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
        alpha_candidates: List of alpha values to test
        auto_tune_alpha: Whether to automatically tune alpha
    
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
    
    # Store calibration data for alpha tuning
    calib_texts = []
    
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
        # Store some texts for alpha tuning
        if auto_tune_alpha and len(calib_texts) < 20:
            calib_texts.extend(batch[:min(5, len(batch))])
        
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
    
    # Apply residual low-rank factorization
    logging.info(f"{Fore.GREEN}Applying residual low-rank factorization with rank {low_rank}{Fore.RESET}")
    print(f"[DEBUG] Starting residual low-rank factorization process")
    U, V = residual_low_rank_factorization(transform, rank=low_rank)
    
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
    
    # Find optimal alpha if auto-tuning is enabled
    target_layer = start_id - num_layer - 1
    optimal_alpha = alpha_candidates[0]  # Default
    
    if auto_tune_alpha and calib_texts:
        print(f"[DEBUG] Auto-tuning alpha parameter")
        original_weight = model.model.layers[target_layer].mlp.down_proj.weight.data
        optimal_alpha = find_optimal_alpha(
            model_path, tokenizer, U, V, target_layer, 
            original_weight, calib_texts, alpha_candidates
        )
    else:
        print(f"[DEBUG] Using default alpha: {optimal_alpha}")
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_residual_lowrank_{low_rank}_alpha_{optimal_alpha}"
        ).replace("/", "_")
    
    print(f"[DEBUG] Save path: {save_path}_ResidualLowRank_{loss}_{solver}_r{low_rank}_a{optimal_alpha}")
    
    # Apply residual low-rank transformation
    print(f"[DEBUG] Applying residual low-rank transformation to layer index: {target_layer}")
    apply_residual_low_rank_transform(model, target_layer, U, V, optimal_alpha)
    
    model.save_pretrained(f"{save_path}_ResidualLowRank_{loss}_{solver}_r{low_rank}_a{optimal_alpha}")
    tokenizer.save_pretrained(f"{save_path}_ResidualLowRank_{loss}_{solver}_r{low_rank}_a{optimal_alpha}")
    print(f"[DEBUG] Model saved to: {save_path}_ResidualLowRank_{loss}_{solver}_r{low_rank}_a{optimal_alpha}")
    
    if save_transform_only:
        transform_save_path = f"{save_path}_ResidualLowRank_{loss}_{solver}_r{low_rank}_a{optimal_alpha}_factors"
        torch.save({
            'U': U,
            'V': V,
            'rank': low_rank,
            'alpha': optimal_alpha,
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
    
    final_path = f"{save_path}_ResidualLowRank_{loss}_{solver}_r{low_rank}_a{optimal_alpha}"
    print(f"[DEBUG] Residual low-rank replacement completed successfully")
    print(f"[DEBUG] Final model path: {final_path}")
    print(f"[DEBUG] Optimal alpha used: {optimal_alpha}")
    
    return final_path

