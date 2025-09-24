# # fixed_adaptive_replaceme_improved.py
# import gc
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional
# from tqdm import tqdm
# from colorama import Fore, init
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from torch.optim.lr_scheduler import CosineAnnealingLR

# from .utils import get_calib_dataloader, truncate_model, seed_all

# init(autoreset=True)
# seed_all()


# def fixed_adaptive_replaceme(
#     model_path: str,
#     dataset: str,
#     dataset_column: str,
#     batch_size: int,
#     max_length: int,
#     layers_to_skip: int,
#     dataset_size: Optional[int] = None,
#     dataset_subset: Optional[str] = "eval",
#     use_4bit: bool = False,
#     save_path: Optional[str] = None,
#     token: Optional[str] = None,
#     start_id: int = 0,
#     end_id: int = 0,
#     num_layer: int = 0,
#     adaptive_weight: float = 0.6,
#     distances_path: str = "./distances.pth",
#     num_A: int = 1,
#     merge_consecutive: bool = True,
#     max_tokens_to_collect: int = 500000,
#     **kwargs
# ) -> str:
    
#     print(f"\n[Fixed AR v2] Processing layers {start_id}-{end_id}")
#     print(f"[Fixed AR v2] Skip size: {end_id - start_id} layers")
#     print(f"[Fixed AR v2] Max tokens to collect: {max_tokens_to_collect}")
    
#     # Load model for gathering activations
#     device_map = "auto" if torch.cuda.is_available() else "cpu"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map=device_map,
#         output_hidden_states=True,
#         token=token,
#         torch_dtype=torch.float32
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     hidden_size = model.config.hidden_size
#     model.eval()
    
#     # Gather calibration data
#     dataloader = get_calib_dataloader(
#         dataset, dataset_subset, dataset_column,
#         dataset_size, batch_size, tokenizer
#     )
    
#     print(f"[Fixed AR v2] Gathering calibration data...")
    
#     # Hook to capture MLP outputs
#     mlp_activations = {}
#     def save_mlp_activation(name):
#         def hook(module, input, output):
#             mlp_activations[name] = output.detach()
#         return hook
    
#     # Register hooks
#     hooks = []
#     for i, layer in enumerate(model.model.layers):
#         hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
#     # Collect activations with diversity scoring
#     a1_list = []
#     a2_list = []
#     diversity_scores = []
#     total_tokens_collected = 0
    
#     for batch in tqdm(dataloader, desc="Calibration"):
#         inputs = tokenizer(
#             batch,
#             return_tensors="pt",
#             padding="longest",
#             max_length=max_length,
#             truncation=True
#         )
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = model(**inputs) 
        
#         hidden_states = outputs.hidden_states[1:]  # Skip embeddings
#         attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        
#         # Get MLP output and hidden states at key positions
#         mlp_out = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
#         h_before = hidden_states[start_id - num_layer - 1]
#         h_after = hidden_states[end_id - num_layer - 1]
        
#         # ===== IMPROVEMENT 1: Better target computation =====
#         # Target: what the MLP output should become to match the final state
#         # Original: a2 = (h_after + mlp_out - h_before)
#         # Improved: Direct residual mapping
#         a1 = mlp_out.view(-1, hidden_size)
#         a2 = (h_after - h_before).view(-1, hidden_size)  # More direct target
        
#         # ===== IMPROVEMENT 2: Diversity-based sampling =====
#         # Compute attention entropy for diversity scoring
#         if attentions is not None and len(diversity_scores) < 100:
#             # Use middle layer attention for diversity
#             mid_layer_attn = attentions[len(attentions)//2]
#             # Compute entropy across attention heads
#             entropy = -torch.sum(mid_layer_attn * torch.log(mid_layer_attn + 1e-8), dim=-1)
#             avg_entropy = entropy.mean().item()
#             diversity_scores.append(avg_entropy)
        
#         # Adaptive sampling based on diversity
#         if len(diversity_scores) > 10:
#             # Only collect high-diversity samples after initial collection
#             current_entropy = diversity_scores[-1] if diversity_scores else 0
#             avg_entropy = sum(diversity_scores) / len(diversity_scores)
#             if current_entropy < 0.8 * avg_entropy:  # Skip low-diversity samples
#                 continue
        
#         # Check token limit
#         if total_tokens_collected + a1.shape[0] > max_tokens_to_collect:
#             tokens_needed = max_tokens_to_collect - total_tokens_collected
#             if tokens_needed > 0:
#                 a1 = a1[:tokens_needed]
#                 a2 = a2[:tokens_needed]
#                 a1_list.append(a1.cpu())
#                 a2_list.append(a2.cpu())
#                 total_tokens_collected += tokens_needed
#             print(f"[Fixed AR v2] Reached token limit: {total_tokens_collected} tokens")
#             break
#         else:
#             a1_list.append(a1.cpu())
#             a2_list.append(a2.cpu())
#             total_tokens_collected += a1.shape[0]
    
#     # Remove hooks
#     for hook in hooks:
#         hook.remove()
    
#     # Concatenate
#     a1 = torch.cat(a1_list, dim=0).to(torch.float64)
#     a2 = torch.cat(a2_list, dim=0).to(torch.float64)
    
#     print(f"[Fixed AR v2] Collected {a1.shape[0]} samples")
    
#     # Analyze relationship
#     residual_norm = torch.norm(a2 - a1) / torch.norm(a1)
#     print(f"[Fixed AR v2] Residual norm ratio: {residual_norm:.3f}")
    
#     # ===== IMPROVEMENT 3: Enhanced training with multi-objective loss =====
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Better initialization - closer to identity with small perturbation
#     W = torch.eye(hidden_size, dtype=torch.float64, device=device)
#     W = W + 0.01 * torch.randn_like(W)  # Small random perturbation
#     W.requires_grad_(True)
    
#     # Two-phase training
#     print(f"[Fixed AR v2] Phase 1: Coarse optimization...")
    
#     # Phase 1: Higher learning rate for direction finding
#     optimizer_coarse = torch.optim.Adam([W], lr=5e-3)
    
#     for epoch in range(5):
#         total_loss = 0
#         indices = torch.randperm(a1.shape[0])
#         chunk_size = min(50000, a1.shape[0])
        
#         for i in range(0, len(indices), chunk_size):
#             batch_idx = indices[i:min(i+chunk_size, len(indices))]
#             batch_a1 = a1[batch_idx].to(device).float()
#             batch_a2 = a2[batch_idx].to(device).float()
            
#             optimizer_coarse.zero_grad()
#             pred = batch_a1 @ W.float()
            
#             # Simple cosine loss for coarse optimization
#             pred_norm = F.normalize(pred, dim=-1)
#             a2_norm = F.normalize(batch_a2, dim=-1)
#             loss = 1 - (pred_norm * a2_norm).sum(-1).mean()
            
#             loss.backward()
#             optimizer_coarse.step()
            
#             total_loss += loss.item()
#             del batch_a1, batch_a2
#             torch.cuda.empty_cache()
    
#     print(f"[Fixed AR v2] Phase 2: Fine-tuning with multi-objective loss...")
    
#     # Phase 2: Fine-tuning with lower learning rate and regularization
#     optimizer = torch.optim.AdamW([W], lr=1e-4, weight_decay=1e-5)
#     scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
#     best_W = W.clone()
#     best_loss = float('inf')
#     patience_counter = 0
    
#     for epoch in range(30):  # Increased from 20
#         total_loss = 0
#         num_batches = 0
        
#         indices = torch.randperm(a1.shape[0])
#         chunk_size = min(50000, a1.shape[0])
        
#         for i in range(0, len(indices), chunk_size):
#             batch_idx = indices[i:min(i+chunk_size, len(indices))]
#             batch_a1 = a1[batch_idx].to(device).float()
#             batch_a2 = a2[batch_idx].to(device).float()
            
#             mini_batch_size = 1024
#             mini_batch_loss = 0
#             mini_batch_count = 0
            
#             for j in range(0, len(batch_a1), mini_batch_size):
#                 mini_a1 = batch_a1[j:min(j+mini_batch_size, len(batch_a1))]
#                 mini_a2 = batch_a2[j:min(j+mini_batch_size, len(batch_a2))]
                
#                 optimizer.zero_grad()
                
#                 # Compute transformation
#                 pred = mini_a1 @ W.float()
                
#                 # ===== IMPROVEMENT 4: Multi-objective loss =====
#                 # 1. Cosine similarity loss (direction)
#                 pred_norm = F.normalize(pred, dim=-1)
#                 a2_norm = F.normalize(mini_a2, dim=-1)
#                 cosine_loss = 1 - (pred_norm * a2_norm).sum(-1).mean()
                
#                 # 2. MSE loss (magnitude)
#                 mse_loss = F.mse_loss(pred, mini_a2)
                
#                 # 3. Orthogonality regularization (numerical stability)
#                 ortho_loss = torch.norm(W @ W.T - W.T @ W, 'fro')
                
#                 # 4. Sparsity regularization (keep close to identity)
#                 identity = torch.eye(hidden_size, device=device, dtype=W.dtype)
#                 sparse_loss = torch.norm(W - identity, p=1)
                
#                 # Combined loss with weights
#                 loss = cosine_loss + 0.1 * mse_loss + 0.01 * ortho_loss + 0.001 * sparse_loss
                
#                 loss.backward()
                
#                 # Gradient clipping for stability
#                 torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
                
#                 optimizer.step()
                
#                 mini_batch_loss += loss.item()
#                 mini_batch_count += 1
            
#             del batch_a1, batch_a2
#             torch.cuda.empty_cache()
            
#             total_loss += mini_batch_loss / mini_batch_count
#             num_batches += 1
        
#         scheduler.step()
#         avg_loss = total_loss / num_batches
        
#         # Early stopping with patience
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             best_W = W.clone()
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter > 5:
#                 print(f"[Fixed AR v2] Early stopping at epoch {epoch}")
#                 break
        
#         if epoch % 5 == 0:
#             print(f"[Fixed AR v2] Epoch {epoch}: Loss={avg_loss:.4f}")
    
#     print(f"[Fixed AR v2] Training done. Best loss: {best_loss:.4f}")
    
#     # ===== IMPROVEMENT 5: Residual-aware adaptive blending =====
#     # Instead of simple weighted average, use residual-aware blending
#     identity = torch.eye(hidden_size, dtype=torch.float64, device=device)
    
#     # Extract residual component
#     W_residual = best_W - identity
    
#     # Apply adaptive weight to residual only
#     W_final = identity + adaptive_weight * W_residual
    
#     print(f"[Fixed AR v2] Applied residual-aware adaptive blending (weight={adaptive_weight})")
    
#     # ===== IMPROVEMENT 6: SVD-based denoising (optional) =====
#     if residual_norm > 0.5:  # Only if transformation is significant
#         U, S, V = torch.svd(W_final - identity)
#         # Keep 95% of energy
#         energy = torch.cumsum(S, 0) / S.sum()
#         k = (energy < 0.95).sum() + 1
#         # Reconstruct with top-k singular values
#         W_final = identity + U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].T
#         print(f"[Fixed AR v2] Applied SVD denoising (kept {k}/{hidden_size} components)")
    
#     # Clean up
#     del model, a1, a2
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     # Reload and modify model
#     print(f"[Fixed AR v2] Modifying model...")
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map='cpu',
#         torch_dtype=torch.bfloat16,
#         token=token
#     )
    
#     # Truncate layers
#     model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
#     # Apply transformation to down_proj
#     target_layer = model.model.layers[start_id - num_layer - 1]
#     original_weight = target_layer.mlp.down_proj.weight.to(torch.float64)
    
#     # Apply: new_weight = W^T @ original_weight
#     new_weight = W_final.T.cpu() @ original_weight
#     target_layer.mlp.down_proj.weight = nn.Parameter(new_weight.to(torch.bfloat16))
    
#     print(f"[Fixed AR v2] Updated down_proj weight")
    
#     # Save
#     if save_path is None:
#         save_path = f"output_models/FixedAR_v2_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
#     model.save_pretrained(save_path)
#     tokenizer.save_pretrained(save_path)
    
#     print(f"[Fixed AR v2] Model saved to {save_path}")
    
#     # Save transformation info with additional metrics
#     torch.save({
#         'W': W_final.cpu(),
#         'W_residual_norm': torch.norm(W_final - identity).item(),
#         'adaptive_weight': adaptive_weight,
#         'best_loss': best_loss,
#         'total_tokens_used': total_tokens_collected,
#         'diversity_scores': diversity_scores,
#         'residual_norm_ratio': residual_norm.item()
#     }, f"{save_path}/transform_info.pt")
    
#     return save_path


# fixed_adaptive_replaceme_improved.py
import gc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
seed_all()


def fixed_adaptive_replaceme(
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
    adaptive_weight: float = 0.6,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    max_tokens_to_collect: int = 500000,
    num_epochs: int = 30,  # Increased from 20
    use_scheduler: str = "cosine_warmup",  # "cosine_warmup", "onecycle", "plateau"
    **kwargs
) -> str:
    
    print(f"\n[Fixed AR+LR] Processing layers {start_id}-{end_id}")
    print(f"[Fixed AR+LR] Skip size: {end_id - start_id} layers")
    print(f"[Fixed AR+LR] Max tokens to collect: {max_tokens_to_collect}")
    print(f"[Fixed AR+LR] Using scheduler: {use_scheduler}")
    
    # Load model for gathering activations
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.float16,  # Use float16 for memory
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    # Gather calibration data
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    print(f"[Fixed AR+LR] Gathering calibration data...")
    
    # Hook to capture MLP outputs
    mlp_activations = {}
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
    # Collect activations
    a1_list = []
    a2_list = []
    total_tokens_collected = 0
    
    for batch in tqdm(dataloader, desc="Calibration"):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Process in smaller batches if needed to avoid OOM
        batch_size_actual = inputs['input_ids'].shape[0]
        max_batch_size = 4
        
        if batch_size_actual > max_batch_size:
            for start_idx in range(0, batch_size_actual, max_batch_size):
                end_idx = min(start_idx + max_batch_size, batch_size_actual)
                mini_inputs = {k: v[start_idx:end_idx] for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**mini_inputs)
                
                hidden_states = outputs.hidden_states[1:]
                
                mlp_out = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
                h_before = hidden_states[start_id - num_layer - 1]
                h_after = hidden_states[end_id - num_layer - 1]
                
                # ORIGINAL TARGET FORMULA (proven to work)
                a1 = mlp_out.view(-1, hidden_size)
                a2 = (h_after + mlp_out - h_before).view(-1, hidden_size)
                
                if total_tokens_collected + a1.shape[0] > max_tokens_to_collect:
                    tokens_needed = max_tokens_to_collect - total_tokens_collected
                    if tokens_needed > 0:
                        a1 = a1[:tokens_needed]
                        a2 = a2[:tokens_needed]
                        a1_list.append(a1.cpu())
                        a2_list.append(a2.cpu())
                        total_tokens_collected += tokens_needed
                    break
                else:
                    a1_list.append(a1.cpu())
                    a2_list.append(a2.cpu())
                    total_tokens_collected += a1.shape[0]
                
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                outputs = model(**inputs)
            
            hidden_states = outputs.hidden_states[1:]
            
            mlp_out = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
            h_before = hidden_states[start_id - num_layer - 1]
            h_after = hidden_states[end_id - num_layer - 1]
            
            # ORIGINAL TARGET FORMULA
            a1 = mlp_out.view(-1, hidden_size)
            a2 = (h_after + mlp_out - h_before).view(-1, hidden_size)
            
            if total_tokens_collected + a1.shape[0] > max_tokens_to_collect:
                tokens_needed = max_tokens_to_collect - total_tokens_collected
                if tokens_needed > 0:
                    a1 = a1[:tokens_needed]
                    a2 = a2[:tokens_needed]
                    a1_list.append(a1.cpu())
                    a2_list.append(a2.cpu())
                    total_tokens_collected += tokens_needed
                print(f"[Fixed AR+LR] Reached token limit: {total_tokens_collected} tokens")
                break
            else:
                a1_list.append(a1.cpu())
                a2_list.append(a2.cpu())
                total_tokens_collected += a1.shape[0]
        
        if total_tokens_collected >= max_tokens_to_collect:
            break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate
    a1 = torch.cat(a1_list, dim=0).to(torch.float64)
    a2 = torch.cat(a2_list, dim=0).to(torch.float64)
    
    print(f"[Fixed AR+LR] Collected {a1.shape[0]} samples")
    
    # Analyze relationship
    residual_norm = torch.norm(a2 - a1) / torch.norm(a1)
    print(f"[Fixed AR+LR] Residual norm ratio: {residual_norm:.3f}")
    
    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== IMPROVED TRAINING WITH LR SCHEDULING =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize closer to identity
    W = torch.eye(hidden_size, dtype=torch.float64, device=device)
    # Better initialization with truncated normal
    init_std = 0.02
    noise = torch.randn_like(W) * init_std
    noise = torch.clamp(noise, -2*init_std, 2*init_std)  # Truncate
    W = W + noise
    W.requires_grad_(True)
    
    # Use AdamW for better regularization
    optimizer = torch.optim.AdamW([W], lr=2e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Calculate total steps for scheduler
    chunk_size = min(50000, a1.shape[0])
    steps_per_epoch = max(1, a1.shape[0] // chunk_size)
    total_steps = num_epochs * steps_per_epoch
    
    # Setup Learning Rate Scheduler
    if use_scheduler == "cosine_warmup":
        # Cosine Annealing with Warm Restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # First restart after 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-5  # Minimum learning rate
        )
        print(f"[Fixed AR+LR] Using Cosine Annealing with Warm Restarts")
    
    elif use_scheduler == "onecycle":
        # One Cycle Learning Rate
        scheduler = OneCycleLR(
            optimizer,
            max_lr=5e-3,  # Peak learning rate
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=10,  # Initial lr = max_lr/10
            final_div_factor=100  # Final lr = max_lr/100
        )
        print(f"[Fixed AR+LR] Using OneCycle scheduler")
    
    elif use_scheduler == "plateau":
        # Reduce on Plateau (needs validation loss)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
        print(f"[Fixed AR+LR] Using ReduceLROnPlateau scheduler")
    
    else:
        # No scheduler, use constant learning rate
        scheduler = None
        print(f"[Fixed AR+LR] No scheduler, using constant LR")
    
    print(f"[Fixed AR+LR] Starting training for {num_epochs} epochs...")
    
    best_W = W.clone()
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle data each epoch
        indices = torch.randperm(a1.shape[0])
        
        # Process in chunks
        for i in range(0, len(indices), chunk_size):
            batch_idx = indices[i:min(i+chunk_size, len(indices))]
            
            # Move batch to device
            batch_a1 = a1[batch_idx].to(device).float()
            batch_a2 = a2[batch_idx].to(device).float()
            
            # Further split into mini-batches for gradient accumulation
            mini_batch_size = 1024
            accumulation_steps = max(1, len(batch_a1) // mini_batch_size)
            accumulated_loss = 0
            
            optimizer.zero_grad()
            
            for j in range(0, len(batch_a1), mini_batch_size):
                mini_a1 = batch_a1[j:min(j+mini_batch_size, len(batch_a1))]
                mini_a2 = batch_a2[j:min(j+mini_batch_size, len(batch_a2))]
                
                # Compute transformation
                pred = mini_a1 @ W.float()
                
                # Cosine similarity loss (original, proven to work)
                pred_norm = F.normalize(pred, dim=-1)
                a2_norm = F.normalize(mini_a2, dim=-1)
                cosine_loss = 1 - (pred_norm * a2_norm).sum(-1).mean()
                
                # Optional: Add small L2 regularization
                l2_reg = 0.001 * torch.norm(W - torch.eye(hidden_size, device=device), 'fro')
                
                loss = cosine_loss + l2_reg
                
                # Gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                accumulated_loss += loss.item()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update scheduler (for OneCycle)
            if scheduler and use_scheduler == "onecycle":
                scheduler.step()
            
            total_loss += accumulated_loss * accumulation_steps
            num_batches += 1
            
            # Clear GPU memory
            del batch_a1, batch_a2
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        
        # Update scheduler (for Cosine and Plateau)
        if scheduler:
            if use_scheduler == "cosine_warmup":
                scheduler.step()
            elif use_scheduler == "plateau":
                scheduler.step(avg_loss)
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_W = W.clone()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter > 7:  # Increased patience
            print(f"[Fixed AR+LR] Early stopping at epoch {epoch}")
            break
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"[Fixed AR+LR] Epoch {epoch}: Loss={avg_loss:.4f}, LR={current_lr:.6f}, Best={best_loss:.4f}")
    
    print(f"[Fixed AR+LR] Training done. Best loss: {best_loss:.4f}")
    
    # Optional: Stochastic Weight Averaging (SWA) for last few checkpoints
    if len(loss_history) > 5:
        # Average last 3 epochs if they're good
        if all(loss < best_loss * 1.1 for loss in loss_history[-3:]):
            print(f"[Fixed AR+LR] Applying SWA on last 3 checkpoints")
            W_final = (best_W + W) / 2  # Simple average with current
        else:
            W_final = best_W
    else:
        W_final = best_W
    
    # Apply adaptive blending with identity (original formula)
    identity = torch.eye(hidden_size, dtype=torch.float64, device=device)
    W_final = (1 - adaptive_weight) * W_final + adaptive_weight * identity
    
    print(f"[Fixed AR+LR] Applied adaptive blending (weight={adaptive_weight})")
    
    # Optional: Spectral regularization
    if residual_norm > 0.3:
        # Control singular values for numerical stability
        U, S, V = torch.svd(W_final)
        S_clipped = torch.clamp(S, min=0.5, max=2.0)  # Limit singular values
        W_final = U @ torch.diag(S_clipped) @ V.T
        print(f"[Fixed AR+LR] Applied spectral regularization")
    
    # Clean up
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload and modify model
    print(f"[Fixed AR+LR] Modifying model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate layers
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj (ORIGINAL METHOD)
    target_layer = model.model.layers[start_id - num_layer - 1]
    original_weight = target_layer.mlp.down_proj.weight.to(torch.float64)
    
    # Apply: new_weight = W^T @ original_weight
    new_weight = W_final.T.cpu() @ original_weight
    target_layer.mlp.down_proj.weight = nn.Parameter(new_weight.to(torch.bfloat16))
    
    print(f"[Fixed AR+LR] Updated down_proj weight")
    
    # Save
    if save_path is None:
        save_path = f"output_models/FixedAR_LR_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[Fixed AR+LR] Model saved to {save_path}")
    
    # Save detailed transformation info
    torch.save({
        'W': W_final.cpu(),
        'adaptive_weight': adaptive_weight,
        'best_loss': best_loss,
        'total_tokens_used': total_tokens_collected,
        'loss_history': loss_history,
        'scheduler_type': use_scheduler,
        'num_epochs_trained': len(loss_history),
        'residual_norm': residual_norm.item()
    }, f"{save_path}/transform_info.pt")
    
    return save_path