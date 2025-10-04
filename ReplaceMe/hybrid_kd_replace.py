import argparse
import gc
import logging
import os
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

from .utils import get_calib_dataloader, truncate_model, seed_all, select_non_overlapping_blocks

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def collect_teacher_activations_all_tokens(
    model,
    dataloader,
    tokenizer,
    start_id: int,
    end_id: int,
    num_layer: int,
    max_length: int,
    dataset_size: int,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Collect teacher outputs for ALL tokens, not just last token."""
    
    print(f"\n{Fore.YELLOW}Collecting teacher activations (ALL TOKENS)...{Fore.RESET}")
    print(f"  Collecting from layer {start_id-num_layer-1} to {end_id-num_layer-1}")
    
    # Estimate total tokens (approximate)
    estimated_tokens = dataset_size * max_length // 2  # Assume 50% padding on average
    
    # Pre-allocate storage
    all_mlp_outputs = []
    all_layer_i_outputs = []
    all_layer_n_outputs = []
    all_attention_masks = []
    
    # Setup hooks
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    if hasattr(model, 'model'):  # Llama style
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
    model.eval()
    token_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting teacher data", colour="green"):
            if token_count >= estimated_tokens:
                break
                
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            
            # Get relevant layers
            hidden_states_mlp = mlp_activations[f'layer_{start_id-num_layer-1}_mlp']
            hidden_states_i = hidden_states[start_id-num_layer-1]
            hidden_states_n = hidden_states[end_id-num_layer-1]
            
            # Store ALL tokens (not just last)
            batch_size, seq_len = inputs['attention_mask'].shape
            
            for b in range(batch_size):
                # Get actual sequence length (excluding padding)
                valid_len = inputs['attention_mask'][b].sum().item()
                
                if valid_len > 0:
                    # Store only valid tokens
                    all_mlp_outputs.append(hidden_states_mlp[b, :valid_len].cpu())
                    all_layer_i_outputs.append(hidden_states_i[b, :valid_len].cpu())
                    all_layer_n_outputs.append(hidden_states_n[b, :valid_len].cpu())
                    all_attention_masks.append(inputs['attention_mask'][b, :valid_len].cpu())
                    
                    token_count += valid_len
            
            # Clear GPU memory
            del outputs, hidden_states_mlp, hidden_states_i, hidden_states_n
            torch.cuda.empty_cache()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all tokens
    print(f"\nConcatenating {len(all_mlp_outputs)} sequences...")
    
    mlp_flat = torch.cat(all_mlp_outputs, dim=0).to(torch.bfloat16)
    layer_i_flat = torch.cat(all_layer_i_outputs, dim=0).to(torch.bfloat16)
    layer_n_flat = torch.cat(all_layer_n_outputs, dim=0).to(torch.bfloat16)
    
    print(f"Total tokens collected: {mlp_flat.shape[0]}")
    print(f"  MLP outputs shape: {mlp_flat.shape}")
    print(f"  Layer {start_id} outputs shape: {layer_i_flat.shape}")
    print(f"  Layer {end_id} outputs shape: {layer_n_flat.shape}")
    
    # Compute targets (following ReplaceMe's formulation)
    # a2 = hidden_states_n + hidden_states_mlp - hidden_states_i
    target_flat = layer_n_flat + mlp_flat - layer_i_flat
    
    return {
        'mlp_outputs': mlp_flat,
        'layer_i_outputs': layer_i_flat,
        'layer_n_outputs': layer_n_flat,
        'targets': target_flat,
        'token_count': token_count
    }


def optimize_hybrid_transform(
    teacher_data: Dict[str, torch.Tensor],
    hidden_size: int,
    device: str = 'cuda',
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 5e-5,
    alpha: float = 0.6,  # Weight for teacher supervision
    weight_decay: float = 1e-4
) -> torch.Tensor:
    """Optimize T using hybrid approach: Teacher supervision + ReplaceMe objective."""
    
    print(f"\n{Fore.CYAN}Optimizing T with Hybrid Approach{Fore.RESET}")
    print(f"  Total tokens: {teacher_data['token_count']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Alpha (teacher weight): {alpha}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    
    # Extract data
    mlp_outputs = teacher_data['mlp_outputs'].to(torch.float32)
    layer_n_outputs = teacher_data['layer_n_outputs'].to(torch.float32)
    targets = teacher_data['targets'].to(torch.float32)  # ReplaceMe target
    
    n_samples = mlp_outputs.shape[0]
    
    # Initialize T as identity with small random noise
    T = torch.eye(hidden_size, device=device, dtype=torch.float32)
    T += 0.01 * torch.randn_like(T)  # Small perturbation
    T.requires_grad_(True)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW([T], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    best_T = T.clone().detach()
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_teacher_loss = 0.0
        epoch_cosine_loss = 0.0
        epoch_reg_loss = 0.0
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch
            mlp_batch = mlp_outputs[batch_indices].to(device)
            targets_batch = targets[batch_indices].to(device)
            layer_n_batch = layer_n_outputs[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            # Prediction
            pred = mlp_batch @ T
            
            # Loss 1: Teacher supervision (match the expected output)
            teacher_loss = F.mse_loss(pred, targets_batch)
            
            # Loss 2: Cosine similarity (ReplaceMe's original objective)
            pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = targets_batch / (targets_batch.norm(dim=1, keepdim=True) + 1e-8)
            cosine_loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
            
            # Loss 3: Identity regularization (keep T close to identity)
            identity = torch.eye(hidden_size, device=device)
            reg_loss = 0.001 * (T - identity).pow(2).mean()
            
            # Combined loss
            loss = alpha * teacher_loss + (1 - alpha) * cosine_loss + reg_loss
            
            # Optional: Add gradient penalty for stability
            grad_norm = torch.autograd.grad(
                outputs=pred.sum(), 
                inputs=T, 
                create_graph=True,
                only_inputs=True
            )[0].norm()
            
            if grad_norm > 10:
                loss = loss + 0.01 * grad_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_teacher_loss += teacher_loss.item()
            epoch_cosine_loss += cosine_loss.item()
            epoch_reg_loss += reg_loss.item()
            
            # Clear memory
            del mlp_batch, targets_batch, pred
            torch.cuda.empty_cache()
        
        scheduler.step()
        
        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_teacher = epoch_teacher_loss / num_batches
        avg_cosine = epoch_cosine_loss / num_batches
        avg_reg = epoch_reg_loss / num_batches
        
        print(f"  Epoch {epoch+1}: Total={avg_loss:.6f}, Teacher={avg_teacher:.6f}, "
              f"Cosine={avg_cosine:.6f}, Reg={avg_reg:.6f}, LR={scheduler.get_last_lr()[0]:.1e}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T.clone().detach()
            patience_counter = 0
            print(f"    {Fore.GREEN}New best! Resetting patience.{Fore.RESET}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    {Fore.YELLOW}Early stopping triggered.{Fore.RESET}")
                break
    
    # Final evaluation
    print(f"\n{Fore.CYAN}Final Evaluation:{Fore.RESET}")
    with torch.no_grad():
        # Sample evaluation
        eval_size = min(10000, n_samples)
        eval_indices = torch.randperm(n_samples)[:eval_size]
        
        mlp_eval = mlp_outputs[eval_indices].to(device)
        targets_eval = targets[eval_indices].to(device)
        
        # With learned T
        pred_T = mlp_eval @ best_T
        mse_T = F.mse_loss(pred_T, targets_eval).item()
        
        pred_norm = pred_T / (pred_T.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = targets_eval / (targets_eval.norm(dim=1, keepdim=True) + 1e-8)
        cosine_T = (pred_norm * target_norm).sum(dim=1).mean().item()
        
        # With identity
        pred_I = mlp_eval
        mse_I = F.mse_loss(pred_I, targets_eval).item()
        
        pred_norm = pred_I / (pred_I.norm(dim=1, keepdim=True) + 1e-8)
        cosine_I = (pred_norm * target_norm).sum(dim=1).mean().item()
        
        print(f"  Learned T - MSE: {mse_T:.6f}, Cosine: {cosine_T:.6f}")
        print(f"  Identity  - MSE: {mse_I:.6f}, Cosine: {cosine_I:.6f}")
        print(f"  MSE Improvement: {((mse_I - mse_T) / mse_I * 100):.1f}%")
        print(f"  Cosine Improvement: {((cosine_T - cosine_I) / (1 - cosine_I) * 100):.1f}%")
        
        # Check T deviation from identity
        identity = torch.eye(hidden_size, device=device)
        deviation = (best_T - identity).abs().mean().item()
        max_deviation = (best_T - identity).abs().max().item()
        print(f"  T deviation from identity - Mean: {deviation:.6f}, Max: {max_deviation:.6f}")
    
    return best_T.cpu()


def hybrid_kd_replace(
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
    hybrid_epochs: int = 10,
    hybrid_lr: float = 5e-5,
    hybrid_alpha: float = 0.6,
    hybrid_batch_size: int = 4096,
    **kwargs
) -> str:
    """Hybrid KD+ReplaceMe approach using all tokens."""
    
    print(f"\n{Fore.CYAN}=== Hybrid KD-ReplaceMe (All Tokens) ==={Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    print(f"Dataset: {dataset}, Size: {dataset_size}")
    print(f"Using ALL tokens for better generalization")
    
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
    
    # Get dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Collect teacher activations for ALL tokens
    teacher_data = collect_teacher_activations_all_tokens(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        start_id=start_id,
        end_id=end_id,
        num_layer=num_layer,
        max_length=max_length,
        dataset_size=dataset_size,
        device=device_map
    )
    
    # Clear model to save memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Optimize T using hybrid approach
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = optimize_hybrid_transform(
        teacher_data=teacher_data,
        hidden_size=hidden_size,
        device=device,
        epochs=hybrid_epochs,
        batch_size=hybrid_batch_size,
        lr=hybrid_lr,
        alpha=hybrid_alpha,
        weight_decay=1e-4
    )
    
    # Clean up teacher data
    del teacher_data
    gc.collect()
    torch.cuda.empty_cache()
    
    # Apply transformation and save
    print(f"\n{Fore.YELLOW}Applying transformation and saving model...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply T to down_proj
    layer_idx = start_id - num_layer - 1
    original_weight = model.model.layers[layer_idx].mlp.down_proj.weight
    
    print(f"Applying T to layer {layer_idx} down_proj")
    print(f"  Original weight shape: {original_weight.shape}")
    print(f"  Transform shape: {transform.shape}")
    
    new_weight = (transform.T.to(torch.float64) @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[layer_idx].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Save
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path}_{layers_to_skip}_layers_{start_id}_{end_id}_hybrid".replace("/", "_")
    
    full_save_path = f"{save_path}_ReplaceMe_Hybrid"
    model.save_pretrained(full_save_path)
    tokenizer.save_pretrained(full_save_path)
    
    print(f"{Fore.GREEN}Model saved to: {full_save_path}{Fore.RESET}")
    
    # Save metadata
    torch.save({
        'transform': transform,
        'method': 'hybrid_kd_replaceme',
        'start_id': start_id,
        'end_id': end_id,
        'alpha': hybrid_alpha,
        'used_all_tokens': True
    }, f"{full_save_path}_transform_data.pt")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return full_save_path