"""Advanced ReplaceMe: Residual-Aware and Critical Dimension Selection
This module implements an enhanced version of ReplaceMe that considers:
1. Residual connections in transformers
2. Critical dimension selection based on multiple metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device handling for Google Colab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActivationDataset(Dataset):
    """Dataset for handling activations during optimization"""
    def __init__(self, a1: torch.Tensor, a2: torch.Tensor, yi: torch.Tensor, critical_dims: torch.Tensor = None):
        self.a1 = a1
        self.a2 = a2  
        self.yi = yi
        self.critical_dims = critical_dims
        
    def __len__(self) -> int:
        return len(self.a1)
        
    def __getitem__(self, idx: int) -> Tuple:
        if self.critical_dims is not None:
            return self.a1[idx], self.a2[idx], self.yi[idx], self.critical_dims
        return self.a1[idx], self.a2[idx], self.yi[idx]


def collect_attention_patterns(
    model: nn.Module,
    dataloader: DataLoader,
    start_id: int,
    end_id: int,
    tokenizer,
    max_length: int = 1024
) -> Dict[str, torch.Tensor]:
    """Collect attention weights and patterns from specified layers"""
    
    model.eval()
    attention_patterns = {f'layer_{i}': [] for i in range(start_id, end_id + 1)}
    
    def save_attention_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_patterns[name].append(output.attentions.detach().cpu())
        return hook
    
    # Register hooks
    hooks = []
    for i in range(start_id, min(end_id + 1, len(model.model.layers))):
        layer_name = f'layer_{i}'
        hook = model.model.layers[i].self_attn.register_forward_hook(save_attention_hook(layer_name))
        hooks.append(hook)
    
    # Collect patterns
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting attention patterns"):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Need to set output_attentions=True
            outputs = model(**inputs, output_attentions=True)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Aggregate patterns
    aggregated = {}
    for layer_name, patterns in attention_patterns.items():
        if patterns:
            aggregated[layer_name] = torch.cat(patterns, dim=0).mean(dim=0)  # Average over batch
    
    return aggregated


def compute_mlp_attention_interaction(
    model: nn.Module,
    dataloader: DataLoader,
    start_id: int,
    end_id: int,
    tokenizer,
    hidden_size: int,
    max_length: int = 1024,
    dataset_size: int = 1000
) -> torch.Tensor:
    """Compute how much each MLP dimension affects attention in subsequent layers"""
    
    model.eval()
    interaction_scores = torch.zeros(hidden_size, device='cpu')
    sample_count = 0
    
    # Hook for capturing MLP outputs
    mlp_outputs = {}
    def save_mlp_hook(name):
        def hook(module, input, output):
            mlp_outputs[name] = output.detach()
        return hook
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing MLP-Attention interaction")):
        if sample_count >= dataset_size:
            break
            
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get baseline attention scores
            outputs_baseline = model(**inputs, output_attentions=True, output_hidden_states=True)
            
            # For each layer in range
            for layer_idx in range(start_id, min(end_id, len(model.model.layers) - 1)):
                # Get MLP output and attention of next layer
                mlp_out = model.model.layers[layer_idx].mlp(
                    model.model.layers[layer_idx].post_attention_layernorm(
                        outputs_baseline.hidden_states[layer_idx + 1]
                    )
                )
                
                # Compute sensitivity for each dimension
                batch_size, seq_len, _ = mlp_out.shape
                
                # Sample dimensions to test (testing all would be too expensive)
                n_sample_dims = min(64, hidden_size)
                sample_dims = torch.randperm(hidden_size)[:n_sample_dims]
                
                for dim in sample_dims:
                    # Perturb this dimension
                    mlp_out_perturbed = mlp_out.clone()
                    mlp_out_perturbed[:, :, dim] *= 0.9  # 10% reduction
                    
                    # Compute how this affects next layer's attention
                    # This is a simplified approximation
                    perturbation_effect = torch.abs(mlp_out[:, :, dim]).mean()
                    interaction_scores[dim] += perturbation_effect.cpu()
        
        sample_count += len(batch)
        
        # Clear GPU memory
        del outputs_baseline
        torch.cuda.empty_cache()
    
    # Normalize
    interaction_scores = interaction_scores / sample_count
    return interaction_scores


def identify_critical_dimensions(
    a1: torch.Tensor,
    a2: torch.Tensor, 
    yi: torch.Tensor,
    interaction_scores: torch.Tensor,
    hidden_size: int,
    critical_ratio: float = 0.3
) -> torch.Tensor:
    """Identify critical dimensions using multiple metrics"""
    
    device_local = a1.device
    
    # 1. Activation frequency (how often dimension is significantly active)
    activation_freq = (torch.abs(a1) > torch.abs(a1).mean()).float().mean(dim=0)
    
    # 2. Variance across samples (high variance = more information)
    activation_variance = torch.var(a1, dim=0)
    
    # 3. Contribution to output change
    output_change = torch.abs(a2 - yi).mean(dim=0)
    mlp_contribution = torch.abs(a1).mean(dim=0)
    contribution_ratio = output_change / (mlp_contribution + 1e-8)
    
    # 4. Cross-correlation between input and output
    # Compute correlation for manageable batch size
    batch_size = min(1000, a1.shape[0])
    a1_sample = a1[:batch_size]
    a2_sample = a2[:batch_size]
    
    cross_correlation = torch.zeros(hidden_size, device=device_local)
    for i in range(hidden_size):
        if a1_sample[:, i].std() > 1e-8 and a2_sample[:, i].std() > 1e-8:
            corr = torch.corrcoef(torch.stack([a1_sample[:, i], a2_sample[:, i]]))[0, 1]
            cross_correlation[i] = torch.abs(corr) if not torch.isnan(corr) else 0.0
    
    # Normalize all metrics to [0, 1]
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 1e-8:
            return (x - x_min) / (x_max - x_min)
        return torch.zeros_like(x)
    
    activation_freq_norm = normalize(activation_freq)
    activation_variance_norm = normalize(activation_variance)
    contribution_ratio_norm = normalize(contribution_ratio)
    cross_correlation_norm = normalize(cross_correlation)
    
    # Include MLP-attention interaction scores if available
    if interaction_scores is not None and len(interaction_scores) == hidden_size:
        interaction_scores_norm = normalize(interaction_scores.to(device_local))
        
        # Comprehensive importance score
        importance_score = (
            0.2 * activation_freq_norm +
            0.2 * activation_variance_norm +
            0.2 * contribution_ratio_norm +
            0.2 * cross_correlation_norm +
            0.2 * interaction_scores_norm
        )
    else:
        importance_score = (
            0.25 * activation_freq_norm +
            0.25 * activation_variance_norm +
            0.25 * contribution_ratio_norm +
            0.25 * cross_correlation_norm
        )
    
    # Select top-k critical dimensions
    k = int(hidden_size * critical_ratio)
    critical_indices = torch.topk(importance_score, k).indices
    
    # Create mask
    critical_mask = torch.zeros(hidden_size, dtype=torch.bool, device=device_local)
    critical_mask[critical_indices] = True
    
    return critical_mask


def residual_aware_transform_estimation(
    a1_mlp: torch.Tensor,
    a2_full: torch.Tensor,
    yi: torch.Tensor,
    critical_dims: torch.Tensor,
    loss_type: str = "cosine",
    num_epochs: int = 15
) -> torch.Tensor:
    """Estimate transform considering residual connections"""
    
    hidden_size = a1_mlp.shape[1]
    device_local = a1_mlp.device
    
    # Initialize transforms
    T_critical = torch.eye(hidden_size, device=device_local, dtype=torch.float32)
    T_non_critical = torch.eye(hidden_size, device=device_local, dtype=torch.float32)
    
    # Compute residual contribution
    # a2_full ≈ yi + transformed_mlp
    # So we want: a1_mlp @ T ≈ a2_full - yi
    target = a2_full - yi
    
    # Critical dimensions: careful optimization
    if critical_dims.sum() > 0:
        critical_indices = torch.where(critical_dims)[0]
        a1_critical = a1_mlp[:, critical_dims].float()
        target_critical = target[:, critical_dims].float()
        
        # Create learnable parameters for critical dimensions
        T_critical_subset = nn.Parameter(
            torch.eye(critical_dims.sum(), device=device_local, dtype=torch.float32)
        )
        
        optimizer = torch.optim.Adam([T_critical_subset], lr=1e-4)
        
        dataset = ActivationDataset(a1_critical, target_critical, yi[:, critical_dims])
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        
        # Optimize critical dimensions
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_a1, batch_target, batch_yi in loader:
                optimizer.zero_grad()
                
                # Transform
                transformed = batch_a1 @ T_critical_subset
                
                if loss_type == "cosine":
                    # Cosine similarity loss
                    cos_sim = F.cosine_similarity(transformed, batch_target, dim=-1)
                    loss = 1 - cos_sim.mean()
                else:
                    # MSE loss
                    loss = F.mse_loss(transformed, batch_target)
                
                # Add regularization to prevent extreme values
                reg_loss = 0.01 * torch.norm(T_critical_subset - torch.eye(critical_dims.sum(), device=device_local))
                loss = loss + reg_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([T_critical_subset], max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
        
        # Update T_critical with optimized values
        for i, idx in enumerate(critical_indices):
            for j, jdx in enumerate(critical_indices):
                T_critical[idx, jdx] = T_critical_subset[i, j].detach()
    
    # Non-critical dimensions: fast least squares
    if (~critical_dims).sum() > 0:
        a1_non_critical = a1_mlp[:, ~critical_dims].double()
        target_non_critical = target[:, ~critical_dims].double()
        
        if a1_non_critical.shape[0] > 0 and a1_non_critical.shape[1] > 0:
            # Least squares solution
            try:
                # Add regularization for numerical stability
                reg = 1e-5 * torch.eye(a1_non_critical.shape[1], device=device_local, dtype=torch.float64)
                T_non_critical_subset = torch.linalg.solve(
                    a1_non_critical.T @ a1_non_critical + reg,
                    a1_non_critical.T @ target_non_critical
                ).float()
                
                # Update T_non_critical
                non_critical_indices = torch.where(~critical_dims)[0]
                for i, idx in enumerate(non_critical_indices):
                    for j, jdx in enumerate(non_critical_indices):
                        T_non_critical[idx, jdx] = T_non_critical_subset[i, j]
            except:
                # If least squares fails, keep identity for non-critical
                pass
    
    # Combine transforms with smooth blending
    alpha = 0.8  # Weight for critical dimensions
    T_combined = torch.zeros_like(T_critical)
    
    for i in range(hidden_size):
        for j in range(hidden_size):
            if critical_dims[i] and critical_dims[j]:
                T_combined[i, j] = T_critical[i, j]
            elif not critical_dims[i] and not critical_dims[j]:
                T_combined[i, j] = T_non_critical[i, j]
            else:
                # Mixed dimensions: weighted average
                T_combined[i, j] = alpha * T_critical[i, j] + (1 - alpha) * T_non_critical[i, j]
    
    return T_combined.to(torch.float64)


def advanced_cosine_dist(
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
    critical_ratio: float = 0.3,
    **kwargs
) -> str:
    """Advanced version of cosine_dist with residual awareness and critical dimension selection"""
    
    print(f"Starting Advanced ReplaceMe: layers {start_id} to {end_id}")
    
    # Import required functions from utils
    from .utils import get_calib_dataloader, truncate_model
    
    # Load model and tokenizer
    from transformers import BitsAndBytesConfig
    
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
        output_attentions=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
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
    
    # Collect MLP-Attention interactions
    print("Computing MLP-Attention interactions...")
    interaction_scores = compute_mlp_attention_interaction(
        model, dataloader, start_id, end_id, tokenizer, hidden_size, max_length, 
        min(dataset_size or 1000, 1000)
    )
    
    # Setup hooks for MLP activations
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
    # Collect activations
    print("Collecting activations...")
    a1 = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    yi = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    
    cnt = 0
    for batch in tqdm(dataloader, desc="Gathering Activations"):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        
        # Get relevant activations
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)
        hidden_states_i = hidden_states_i.view(-1, hidden_size)
        hidden_states_n = hidden_states_n.view(-1, hidden_size)
        
        # Store: a1 is MLP output, yi is attention output, a2 is final output
        a1_batch = hidden_states_mlp
        yi_batch = hidden_states_i  # This includes residual up to this point
        a2_batch = hidden_states_n  # This is the full output we want to match
        
        batch_size_actual = a1_batch.shape[0]
        a1[cnt:cnt+batch_size_actual] = a1_batch.cpu().bfloat16()
        yi[cnt:cnt+batch_size_actual] = yi_batch.cpu().bfloat16()
        a2[cnt:cnt+batch_size_actual] = a2_batch.cpu().bfloat16()
        
        cnt += batch_size_actual
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        torch.cuda.empty_cache()
    
    # Trim to actual size
    a1 = a1[:cnt].to(torch.float32)
    yi = yi[:cnt].to(torch.float32)
    a2 = a2[:cnt].to(torch.float32)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Move to GPU for computation
    if torch.cuda.is_available():
        a1 = a1.cuda()
        yi = yi.cuda()
        a2 = a2.cuda()
        interaction_scores = interaction_scores.cuda()
    
    # Identify critical dimensions
    print("Identifying critical dimensions...")
    critical_dims = identify_critical_dimensions(
        a1, a2, yi, interaction_scores, hidden_size, critical_ratio
    )
    
    print(f"Selected {critical_dims.sum().item()}/{hidden_size} critical dimensions")
    
    # Estimate transform with residual awareness
    print("Estimating residual-aware transform...")
    transform = residual_aware_transform_estimation(
        a1, a2, yi, critical_dims, loss_type="cosine", num_epochs=15
    )
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    print("Applying transformation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Prepare save path
    import os
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply transformation
    down_proj_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (transform.T.cpu() @ down_proj_weight.to(torch.float64)).to(torch.bfloat16)
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.data = new_weight
    
    # Save model
    final_save_path = f"{save_path}_AdvancedReplace"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print(f"Model saved to: {final_save_path}")
    
    # Final cleanup
    del model, a1, a2, yi, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_save_path