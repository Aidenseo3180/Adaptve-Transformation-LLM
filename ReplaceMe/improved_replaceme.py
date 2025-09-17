# improved_replaceme.py
import torch
import torch.nn as nn
import gc
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import numpy as np

def improved_replaceme(
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
    save_transform_only: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",  # Added parameter
    **kwargs
) -> str:
    """
    Improved ReplaceMe with adaptive residual scaling and layer norm compensation.
    """
    print(f"[DEBUG] Starting Improved ReplaceMe")
    print(f"[DEBUG] Processing layers {start_id} to {end_id}")
    
    from .utils import get_calib_dataloader, truncate_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Using device: {device}")
    
    # Load model with attention implementation specified
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        attn_implementation="eager",  # Added to avoid SDPA warning
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    print(f"[DEBUG] Model hidden size: {hidden_size}")
    
    model.eval()
    
    # Get calibration dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, 
        dataset_size, batch_size, tokenizer
    )
    
    # Setup hooks to capture MLP activations and layer norm stats
    def save_activations(name, storage_dict):
        def hook(module, input, output):
            storage_dict[name] = output.detach()
        return hook
    
    def save_ln_stats(name, stats_dict):
        def hook(module, input, output):
            # Capture layer norm statistics (RMSNorm compatible)
            inp = input[0].detach()
            mean = inp.mean(dim=-1, keepdim=True)
            var = inp.var(dim=-1, keepdim=True, unbiased=False)
            # RMSNorm only has weight, no bias
            stats_dict[name] = {
                'mean': mean, 
                'var': var, 
                'weight': module.weight.detach() if hasattr(module, 'weight') else None
            }
        return hook
    
    mlp_activations = {}
    ln_stats = {}
    attn_patterns = {}
    hooks = []
    
    print("[DEBUG] Setting up hooks for activation capture")
    
    # Register hooks
    for i, layer in enumerate(model.model.layers):
        # MLP activation hook
        hooks.append(layer.mlp.register_forward_hook(
            save_activations(f'layer_{i}_mlp', mlp_activations)))
        
        # Layer norm stats hooks for pruned layers
        if start_id - num_layer <= i < end_id - num_layer:
            hooks.append(layer.post_attention_layernorm.register_forward_hook(
                save_ln_stats(f'layer_{i}_ln', ln_stats)))
            
            # Attention pattern hook
            def save_attn_pattern(name):
                def hook(module, input, output):
                    # Simplified: just save attention weights if available
                    if hasattr(output, 'attentions') and output.attentions is not None:
                        attn_patterns[name] = output.attentions.detach().mean(dim=1)  # Average over heads
                return hook
            
            hooks.append(layer.self_attn.register_forward_hook(
                save_attn_pattern(f'layer_{i}_attn')))
    
    # Collect activations
    print("[DEBUG] Collecting activations and statistics")
    
    all_Mi = []
    all_Yi = []
    all_Li_n = []
    all_ln_means = []
    all_ln_vars = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering Enhanced Activations")):
        # if batch_idx >= 10:  # Limit for testing
        #     break
            
        inputs = tokenizer(
            batch, return_tensors="pt", padding="longest",
            max_length=max_length, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
        
        # Get MLP output before pruned blocks
        Mi = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
        # Get residual before pruned blocks  
        Yi = hidden_states[start_id - num_layer - 1]
        # Get output after pruned blocks
        Li_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape and store
        batch_size_actual = Mi.shape[0] * Mi.shape[1]
        all_Mi.append(Mi.view(-1, hidden_size).cpu())
        all_Yi.append(Yi.view(-1, hidden_size).cpu())
        all_Li_n.append(Li_n.view(-1, hidden_size).cpu())
        
        # Collect LN stats from pruned blocks
        if ln_stats:
            first_pruned_ln = ln_stats.get(f'layer_{start_id - num_layer}_ln', None)
            if first_pruned_ln:
                all_ln_means.append(first_pruned_ln['mean'].view(-1, 1).cpu())
                all_ln_vars.append(first_pruned_ln['var'].view(-1, 1).cpu())
    
    print(f"[DEBUG] Collected {len(all_Mi)} batches of activations")
    
    # Concatenate all activations
    Mi_all = torch.cat(all_Mi, dim=0).to(torch.float64)
    Yi_all = torch.cat(all_Yi, dim=0).to(torch.float64)
    Li_n_all = torch.cat(all_Li_n, dim=0).to(torch.float64)
    
    print(f"[DEBUG] Activation shapes - Mi: {Mi_all.shape}, Yi: {Yi_all.shape}, Li_n: {Li_n_all.shape}")
    
    # Step 1: Compute basic linear transformation T (least squares)
    print("[DEBUG] Computing basic linear transformation T")
    try:
        # T* = (Mi^T Mi)^-1 Mi^T (Li_n - Yi)
        MtM = Mi_all.T @ Mi_all
        MtM_reg = MtM + 1e-6 * torch.eye(MtM.shape[0])  # Regularization for stability
        T_basic = torch.linalg.solve(MtM_reg, Mi_all.T @ (Li_n_all - Yi_all))
        print(f"[DEBUG] Basic T computed, shape: {T_basic.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to compute basic T: {e}")
        # Fallback to identity
        T_basic = torch.eye(hidden_size, dtype=torch.float64)
    
    # Step 2: Compute feature importance diagonal matrix D
    print("[DEBUG] Computing feature importance matrix D")
    feature_importance = torch.abs(Mi_all).mean(dim=0) + 1e-6
    feature_importance = feature_importance / feature_importance.mean()
    D = torch.diag(feature_importance)
    
    # Step 3: Compute attention pattern influence (simplified)
    print("[DEBUG] Computing attention pattern influence")
    if attn_patterns:
        # Use average attention pattern to create a mixing matrix
        avg_attn = torch.zeros((hidden_size, hidden_size), dtype=torch.float64)
        # Simplified: just add small influence based on pruned layers count
        lambda_attn = 0.01 * (end_id - start_id)  
        P_influence = torch.eye(hidden_size, dtype=torch.float64) * (1 + lambda_attn)
    else:
        P_influence = torch.eye(hidden_size, dtype=torch.float64)
    
    # Step 4: Compute adaptive residual weights (α, β, γ)
    print("[DEBUG] Computing adaptive residual weights")
    
    # Enhanced T with feature importance and attention influence
    T_enhanced = D @ T_basic @ P_influence
    
    # Now solve for optimal α, β, γ
    # We want: Mi * T_enhanced * α + Yi * β + γ ≈ Li_n
    # This is a linear system in [α, β, γ]
    
    # Prepare design matrix for residual weights
    n_samples = Mi_all.shape[0]
    Mi_T = (Mi_all @ T_enhanced).mean(dim=1, keepdim=True)  # Scalar projection
    Yi_mean = Yi_all.mean(dim=1, keepdim=True)
    ones = torch.ones((n_samples, 1), dtype=torch.float64)
    
    # Design matrix: [Mi*T_enhanced, Yi, 1]
    X_residual = torch.cat([Mi_T, Yi_mean, ones], dim=1)
    y_residual = Li_n_all.mean(dim=1, keepdim=True)
    
    # Solve for [α, β, γ]
    try:
        XtX = X_residual.T @ X_residual
        XtX_reg = XtX + 1e-6 * torch.eye(3)
        residual_weights = torch.linalg.solve(XtX_reg, X_residual.T @ y_residual)
        alpha = float(residual_weights[0].item())
        beta = float(residual_weights[1].item())
        gamma = float(residual_weights[2].item())
    except Exception as e:
        print(f"[WARNING] Failed to compute residual weights: {e}")
        alpha, beta, gamma = 1.0, 1.0, 0.0
    
    print(f"[DEBUG] Residual weights - α: {alpha:.4f}, β: {beta:.4f}, γ: {gamma:.4f}")
    
    # Step 5: Layer norm compensation
    if all_ln_means and all_ln_vars:
        print("[DEBUG] Computing layer norm compensation")
        ln_scale = torch.cat(all_ln_vars).mean().sqrt()
        ln_bias = torch.cat(all_ln_means).mean()
        print(f"[DEBUG] LN compensation - scale: {ln_scale:.4f}, bias: {ln_bias:.4f}")
    else:
        ln_scale = 1.0
        ln_bias = 0.0
    
    # Step 6: Create final transformation matrix
    print("[DEBUG] Creating final transformation matrix")
    
    # Incorporate all improvements
    T_final = T_enhanced * alpha * ln_scale
    
    # Add bias compensation to the transformation
    bias_adjustment = gamma + ln_bias * beta
    
    print(f"[DEBUG] Final transformation computed, max value: {T_final.max():.4f}, min value: {T_final.min():.4f}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for modification
    print("[DEBUG] Reloading model for transformation application")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj
    print("[DEBUG] Applying transformation to model")
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (T_final.T @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.data = new_weight    
    # Note: Llama models don't have bias in down_proj, so we skip bias adjustment
    
    # Save model
    import os
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_{layers_to_skip}_layers_{start_id}_{end_id}_improved"
    
    print(f"[DEBUG] Saving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    if save_transform_only:
        torch.save({
            'T_final': T_final,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'ln_scale': ln_scale,
            'ln_bias': ln_bias
        }, f"{save_path}_transform_data")
    
    print("[DEBUG] Improved ReplaceMe completed successfully")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path