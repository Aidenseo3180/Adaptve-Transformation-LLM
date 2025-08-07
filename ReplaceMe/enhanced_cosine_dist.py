import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import gc
import logging
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all

def compute_cosine_distance_loss(transformed, target):
    """
    Compute cosine distance loss between transformed and target activations
    """
    # Normalize along the last dimension
    transformed_norm = transformed / (torch.norm(transformed, dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    
    # Cosine similarity
    cosine_sim = (transformed_norm * target_norm).sum(dim=-1)
    
    # Cosine distance (1 - cosine similarity)
    cosine_dist = 1 - cosine_sim.mean()
    
    return cosine_dist

def find_optimal_rank_with_performance(
    M_i: torch.Tensor, 
    Y_i: torch.Tensor, 
    L_i_n: torch.Tensor, 
    variance_threshold: float = 0.95, 
    max_rank: int = 512,
    alpha_range: list = [0.0, 0.05, 0.1, 0.15, 0.2]
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performance-based optimal rank selection with residual enhancement
    
    Args:
        M_i: MLP output activations [N, d]
        Y_i: Attention output activations [N, d] 
        L_i_n: Target activations [N, d]
        variance_threshold: SVD variance threshold
        max_rank: Maximum allowed rank
        alpha_range: Residual weight candidates
        
    Returns:
        optimal_rank, U_optimal, S_optimal, Vt_optimal, optimal_alpha
    """
    device = M_i.device
    
    # Step 1: Initial transformation matrix calculation with regularization
    A = M_i.T @ M_i + 1e-6 * torch.eye(M_i.shape[1], device=device)
    b = M_i.T @ (L_i_n - Y_i)
    
    try:
        T_init = torch.linalg.solve(A, b)
    except:
        # Fallback to pseudo-inverse if solve fails
        T_init = torch.pinverse(A) @ b
    
    # Step 2: SVD decomposition
    U, S, Vt = torch.svd(T_init)
    
    # Step 3: Variance-based initial rank selection
    total_variance = torch.sum(S**2)
    cumsum_variance = torch.cumsum(S**2, dim=0) / total_variance
    
    # Find initial rank based on variance threshold
    rank_candidates = torch.where(cumsum_variance >= variance_threshold)[0]
    if len(rank_candidates) > 0:
        initial_rank = min(rank_candidates[0].item() + 1, max_rank)
    else:
        initial_rank = min(max_rank, len(S))
    
    # Step 4: Performance-based rank refinement
    best_rank = initial_rank
    best_alpha = 0.1
    best_loss = float('inf')
    
    # Search range: ±20% of initial rank
    search_range = max(1, int(0.2 * initial_rank))
    ranks_to_test = range(
        max(1, initial_rank - search_range),
        min(len(S), initial_rank + search_range + 1)
    )
    
    print(f"Testing ranks from {min(ranks_to_test)} to {max(ranks_to_test)} (initial: {initial_rank})")
    
    for r in ranks_to_test:
        # Low-rank approximation
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        T_r = U_r @ torch.diag(S_r) @ Vt_r
        
        # Test different alpha values
        for alpha in alpha_range:
            # Compute transformed activations with residual connection
            transformed = M_i @ T_r + alpha * M_i + Y_i
            
            # Compute cosine distance loss
            loss = compute_cosine_distance_loss(transformed, L_i_n)
            
            if loss < best_loss:
                best_loss = loss
                best_rank = r
                best_alpha = alpha
    
    print(f"Optimal rank: {best_rank}, Optimal alpha: {best_alpha:.3f}, Best loss: {best_loss:.6f}")
    
    # Return optimal low-rank factors
    U_optimal = U[:, :best_rank]
    S_optimal = S[:best_rank]
    Vt_optimal = Vt[:best_rank, :]
    
    return best_rank, U_optimal, S_optimal, Vt_optimal, best_alpha

def enhanced_adam_method(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor = None,
    loss: str = "cosine",
    max_rank: int = 512,
    variance_threshold: float = 0.95,
    lr: float = 1e-4,
    max_epochs: int = 10
) -> torch.Tensor:
    """
    Enhanced Adam optimization with rank-adaptive and residual connections
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a1 = a1.to(device).float()
    a2 = a2.to(device).float()
    if a3 is not None:
        a3 = a3.to(device).float()
    
    print(f"Enhanced Adam Method - Input shapes: a1={a1.shape}, a2={a2.shape}")
    
    # Find optimal rank and residual weight
    optimal_rank, U_opt, S_opt, Vt_opt, optimal_alpha = find_optimal_rank_with_performance(
        a1, a1, a2, variance_threshold, max_rank
    )
    
    print(f"Using rank {optimal_rank} out of {min(a1.shape[1], max_rank)} (compression: {optimal_rank/a1.shape[1]*100:.1f}%)")
    
    # Initialize low-rank factors
    U = nn.Parameter(U_opt.clone())
    S = nn.Parameter(S_opt.clone()) 
    V = nn.Parameter(Vt_opt.T.clone())  # Transpose Vt to get V
    
    # Residual weight (learnable or fixed)
    alpha = nn.Parameter(torch.tensor(optimal_alpha, device=device))
    
    # Optimizer
    optimizer = torch.optim.Adam([U, S, V, alpha], lr=lr)
    
    # Loss function
    def cosine_loss(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XA_norm = XA / (XA.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()
    
    # Training loop
    batch_size = 1024
    num_samples = a1.shape[0]
    
    with tqdm(range(max_epochs), desc="Enhanced Optimization") as pbar:
        for epoch in pbar:
            epoch_loss = 0
            num_batches = 0
            
            # Batch processing
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_a1 = a1[i:end_idx]
                batch_a2 = a2[i:end_idx]
                batch_a3 = a3[i:end_idx] if a3 is not None else None
                
                optimizer.zero_grad()
                
                # Reconstruct transformation matrix from low-rank factors
                T_reconstructed = U @ torch.diag(S) @ V.T
                
                # Apply transformation with residual connection
                XA = batch_a1 @ T_reconstructed + alpha * batch_a1
                if batch_a3 is not None:
                    XA += batch_a3
                
                # Compute loss
                if loss == "cosine":
                    loss_val = cosine_loss(XA, batch_a2)
                elif loss == "mse":
                    loss_val = nn.MSELoss()(XA, batch_a2)
                else:
                    raise ValueError(f"Unsupported loss: {loss}")
                
                loss_val.backward()
                optimizer.step()
                
                # Clamp alpha to reasonable range
                with torch.no_grad():
                    alpha.data = torch.clamp(alpha.data, 0.0, 0.5)
                
                epoch_loss += loss_val.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            pbar.set_postfix({
                f'{loss} Loss': f'{avg_loss:.6f}',
                'Rank': optimal_rank,
                'Alpha': f'{alpha.item():.3f}'
            })
    
    # Reconstruct final transformation matrix
    with torch.no_grad():
        final_T = U @ torch.diag(S) @ V.T
        print(f"Final residual weight: {alpha.item():.4f}")
        print(f"Final transformation matrix rank: {torch.matrix_rank(final_T).item()}")
    
    return final_T.cpu().to(torch.float64)

def enhanced_cosine_dist(
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
    loss: str = "cosine",
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    max_rank: int = 512,
    variance_threshold: float = 0.95,
    **kwargs
) -> str:
    """
    Enhanced cosine distance method with rank-adaptive and residual connections
    """
    print(f"=== Enhanced Cosine Distance Method ===")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id} (skipping {layers_to_skip} layers)")
    print(f"Max rank: {max_rank}, Variance threshold: {variance_threshold}")
    
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
    
    # Setup activation hooks (same as original)
    def save_mlp_activation(name):
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
    
    # Collect activations (same as original but store in float for better precision)
    a1 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.float32, device='cpu')
    a2 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.float32, device='cpu')
    
    cnt = 0
    for batch in tqdm(dataloader, desc="Gathering Activations for Enhanced Method"):
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
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).float()
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).float()
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).float()
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch.cpu()
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch.cpu()
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    
    print(f"Collected {cnt} activation samples")
    
    # Apply enhanced optimization
    transform = enhanced_adam_method(
        a1, a2, 
        loss=loss, 
        max_rank=max_rank,
        variance_threshold=variance_threshold
    )
    
    # Clean up activations
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clean up model and reload for transformation
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
    
    # Set save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply transformation to down_proj
    target_layer_idx = start_id - num_layer - 1
    current_weight = model.model.layers[target_layer_idx].mlp.down_proj.weight.to(torch.float64)
    new_weight = (transform.T @ current_weight).to(torch.bfloat16)
    
    model.model.layers[target_layer_idx].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Save model
    output_path = f"{save_path}_EnhancedCosine_{loss}"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    if save_transform_only:
        torch.save(transform, f"{output_path}_transform")
    
    print(f"Enhanced model saved to: {output_path}")
    
    # Final cleanup
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path