"""
Gate-Aware Coupled Optimization Methods
모든 새로운 함수들을 이 파일에 집중
"""
import torch
import torch.nn.functional as F
from colorama import Fore
from tqdm import tqdm
import numpy as np
import gc
import os
from typing import List, Dict, Tuple

import torch.nn as nn
from torch.optim import Adam

from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Gate-Aware Enhanced Activation Collection for ReplaceMe
Phase 1: Collect activations from SwiGLU components (gate_proj, up_proj, down_proj)
"""

def collect_enhanced_activations(
    model,
    start_id: int,
    end_id: int,
    dataloader,
    max_length: int,
    dataset_size: int,
    hidden_size: int
):
    """
    Enhanced activation collection for gate-aware coupled optimization
    
    Args:
        model: The transformer model
        start_id: Starting layer index to replace
        end_id: Ending layer index to replace  
        dataloader: Calibration dataloader
        max_length: Maximum sequence length
        dataset_size: Size of dataset
        hidden_size: Hidden dimension size
        
    Returns:
        Dictionary containing all collected activations
    """
    print(f"{Fore.GREEN}🔍 Starting Enhanced Activation Collection{Fore.RESET}")
    print(f"   📍 Replacing layers {start_id} to {end_id}")
    print(f"   📊 Dataset size: {dataset_size}, Hidden size: {hidden_size}")
    
    # Setup activation hooks for SwiGLU components
    def save_activation(name):
        def hook(module, input, output):
            activations_dict[name] = output.detach()
        return hook
    
    hooks = []
    activations_dict = {}
    
    # Register hooks for the target layers
    print(f"{Fore.YELLOW}🎣 Registering hooks for layers {start_id-1} to {end_id-1}...{Fore.RESET}")
    
    for i in range(start_id-1, end_id):
        layer = model.model.layers[i]
        
        # Hook for gate_proj
        gate_hook = layer.mlp.gate_proj.register_forward_hook(
            save_activation(f'layer_{i}_gate_proj')
        )
        hooks.append(gate_hook)
        
        # Hook for up_proj  
        up_hook = layer.mlp.up_proj.register_forward_hook(
            save_activation(f'layer_{i}_up_proj')
        )
        hooks.append(up_hook)
        
        # Hook for down_proj input (gated intermediate)
        def save_gated_intermediate(layer_idx):
            def hook(module, input, output):
                # input[0] is the gated intermediate: SiLU(gate) * up
                activations_dict[f'layer_{layer_idx}_gated_intermediate'] = input[0].detach()
            return hook
        
        down_hook = layer.mlp.down_proj.register_forward_hook(
            save_gated_intermediate(i)
        )
        hooks.append(down_hook)
    
    print(f"   ✅ Registered {len(hooks)} hooks successfully")
    
    # Initialize storage tensors
    print(f"{Fore.BLUE}💾 Initializing storage tensors...{Fore.RESET}")
    
    total_tokens = dataset_size * max_length
    
    storage = {
        # Basic activations (like original ReplaceMe)
        'input_activations': torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'output_activations': torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu'),
        
        # SwiGLU component activations
        'gate_projections': [],      # List of tensors for each layer
        'up_projections': [],        # List of tensors for each layer  
        'gated_intermediates': [],   # List of SiLU(gate) * up for each layer
        'gate_importance_scores': [], # Computed importance scores
    }
    
    # Initialize lists for each layer
    num_layers = end_id - start_id
    for layer_idx in range(num_layers):
        storage['gate_projections'].append(
            torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu')
        )
        storage['up_projections'].append(
            torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu')
        )
        storage['gated_intermediates'].append(
            torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu')
        )
    
    print(f"   📦 Storage initialized for {num_layers} layers")
    print(f"   🔢 Each tensor shape: ({total_tokens}, {model.config.intermediate_size})")
    
    # Collect activations
    cnt = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc=f"{Fore.GREEN}Collecting Enhanced Activations{Fore.RESET}",
            dynamic_ncols=True,
            colour="green"
        )):
            # Tokenize input
            inputs = model.tokenizer(
                batch,
                return_tensors="pt", 
                padding="longest",
                max_length=max_length,
                truncation=True
            ) if hasattr(model, 'tokenizer') else batch
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass to collect activations
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Extract input and output activations (original ReplaceMe style)
            input_acts = hidden_states[start_id - 1].view(-1, hidden_size).cpu()
            output_acts = hidden_states[end_id - 1].view(-1, hidden_size).cpu()
            
            batch_size = input_acts.shape[0]
            
            # Store basic activations
            storage['input_activations'][cnt:cnt+batch_size] = input_acts
            storage['output_activations'][cnt:cnt+batch_size] = output_acts
            
            # Store SwiGLU component activations
            for layer_offset in range(num_layers):
                layer_idx = start_id - 1 + layer_offset
                
                # Gate projections
                gate_acts = activations_dict[f'layer_{layer_idx}_gate_proj'].view(-1, model.config.intermediate_size).cpu()
                storage['gate_projections'][layer_offset][cnt:cnt+batch_size] = gate_acts
                
                # Up projections  
                up_acts = activations_dict[f'layer_{layer_idx}_up_proj'].view(-1, model.config.intermediate_size).cpu()
                storage['up_projections'][layer_offset][cnt:cnt+batch_size] = up_acts
                
                # Gated intermediates
                gated_acts = activations_dict[f'layer_{layer_idx}_gated_intermediate'].view(-1, model.config.intermediate_size).cpu()
                storage['gated_intermediates'][layer_offset][cnt:cnt+batch_size] = gated_acts
            
            cnt += batch_size
            
            # Debug print every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"   🔄 Processed {batch_idx + 1} batches, {cnt} tokens collected")
                print(f"   📊 Current memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            # Clear activations dict to prevent memory buildup
            activations_dict.clear()
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Remove hooks
    print(f"{Fore.YELLOW}🧹 Cleaning up hooks...{Fore.RESET}")
    for hook in hooks:
        hook.remove()
    
    # Trim tensors to actual size
    print(f"{Fore.BLUE}✂️ Trimming tensors to actual size ({cnt} tokens)...{Fore.RESET}")
    storage['input_activations'] = storage['input_activations'][:cnt]
    storage['output_activations'] = storage['output_activations'][:cnt]
    
    for layer_offset in range(num_layers):
        storage['gate_projections'][layer_offset] = storage['gate_projections'][layer_offset][:cnt]
        storage['up_projections'][layer_offset] = storage['up_projections'][layer_offset][:cnt]  
        storage['gated_intermediates'][layer_offset] = storage['gated_intermediates'][layer_offset][:cnt]
    
    print(f"{Fore.GREEN}✅ Enhanced Activation Collection Complete!{Fore.RESET}")
    print(f"   📊 Final shapes:")
    print(f"   🔸 Input/Output activations: {storage['input_activations'].shape}")
    print(f"   🔸 Gate/Up/Gated projections: {storage['gate_projections'][0].shape} x {num_layers} layers")
    
    return storage


def debug_activation_shapes(storage):
    """Debug function to print shapes of all collected activations"""
    print(f"\n{Fore.CYAN}🔍 DEBUG: Activation Shapes{Fore.RESET}")
    print(f"Input activations: {storage['input_activations'].shape}")
    print(f"Output activations: {storage['output_activations'].shape}")
    print(f"Number of layers: {len(storage['gate_projections'])}")
    
    for i, (gate, up, gated) in enumerate(zip(
        storage['gate_projections'], 
        storage['up_projections'],
        storage['gated_intermediates']
    )):
        print(f"Layer {i}:")
        print(f"  Gate proj: {gate.shape}")
        print(f"  Up proj: {up.shape}") 
        print(f"  Gated intermediate: {gated.shape}")


"""
Gate Importance Analysis for Gate-Aware Coupled Optimization
Phase 2: Analyze gate patterns and compute importance scores
"""

def compute_gate_importance_scores(storage: Dict, config: Dict) -> Dict:
    """
    Compute gate importance scores for each layer and neuron
    
    Args:
        storage: Dictionary containing collected activations
        config: Configuration dictionary
        
    Returns:
        Dictionary containing importance scores and analysis
    """
    print(f"\n{Fore.BLUE}🧠 Starting Gate Importance Analysis{Fore.RESET}")
    
    num_layers = len(storage['gate_projections'])
    intermediate_size = storage['gate_projections'][0].shape[1]
    
    print(f"   📊 Analyzing {num_layers} layers with {intermediate_size} neurons each")
    
    importance_data = {
        'layer_wise_importance': [],      # Per-layer importance scores
        'neuron_wise_importance': [],     # Per-neuron importance scores  
        'gate_activation_stats': [],      # Gate activation statistics
        'global_importance_ranking': [],  # Global ranking of important neurons
        'layer_importance_summary': {}    # Summary statistics per layer
    }
    
    for layer_idx in range(num_layers):
        print(f"\n{Fore.YELLOW}🔍 Analyzing Layer {layer_idx}{Fore.RESET}")
        
        # Get gate projections for this layer (before SiLU activation)
        gate_proj = storage['gate_projections'][layer_idx]  # [num_tokens, intermediate_size]
        
        # Apply SiLU to get actual gate values
        gate_activated = torch.sigmoid(gate_proj) * gate_proj  # SiLU(x) = x * sigmoid(x)
        
        print(f"   📐 Gate projection shape: {gate_proj.shape}")
        print(f"   🎯 Gate activated shape: {gate_activated.shape}")
        
        # Method 1: Variance-based importance (high variance = important decision making)
        gate_variance = torch.var(gate_activated, dim=0)  # [intermediate_size]
        print(f"   📊 Gate variance range: [{gate_variance.min():.6f}, {gate_variance.max():.6f}]")
        
        # Method 2: Mean activation level (frequently used pathways)
        gate_mean_abs = torch.abs(torch.mean(gate_activated, dim=0))  # [intermediate_size]
        print(f"   📈 Gate mean abs range: [{gate_mean_abs.min():.6f}, {gate_mean_abs.max():.6f}]")
        
        # Method 3: Combined importance score
        combined_importance = gate_variance * gate_mean_abs
        print(f"   🎖️  Combined importance range: [{combined_importance.min():.6f}, {combined_importance.max():.6f}]")
        
        # Method 4: Dynamic range (max - min) per neuron across tokens
        gate_dynamic_range = torch.max(gate_activated, dim=0)[0] - torch.min(gate_activated, dim=0)[0]
        print(f"   🌊 Dynamic range: [{gate_dynamic_range.min():.6f}, {gate_dynamic_range.max():.6f}]")
        
        # Method 5: Sparsity-aware importance (how often gate is significantly active)
        threshold = 0.1  # Threshold for "significant" activation
        gate_active_ratio = (gate_activated > threshold).float().mean(dim=0)
        print(f"   ⚡ Active ratio range: [{gate_active_ratio.min():.3f}, {gate_active_ratio.max():.3f}]")
        
        # Combine multiple importance metrics
        final_importance = (
            0.4 * combined_importance +           # Variance * Mean (primary metric)
            0.3 * gate_dynamic_range +           # Dynamic range
            0.3 * gate_active_ratio              # Active ratio
        )
        
        # Normalize importance scores to [0, 1] range
        final_importance_normalized = (final_importance - final_importance.min()) / (
            final_importance.max() - final_importance.min() + 1e-8
        )
        
        print(f"   🏆 Final importance range: [{final_importance_normalized.min():.6f}, {final_importance_normalized.max():.6f}]")
        
        # Store layer-wise results
        layer_analysis = {
            'variance': gate_variance,
            'mean_abs': gate_mean_abs, 
            'dynamic_range': gate_dynamic_range,
            'active_ratio': gate_active_ratio,
            'combined_importance': combined_importance,
            'final_importance': final_importance_normalized,
            'raw_gate_stats': {
                'mean': torch.mean(gate_activated, dim=0),
                'std': torch.std(gate_activated, dim=0),
                'min': torch.min(gate_activated, dim=0)[0], 
                'max': torch.max(gate_activated, dim=0)[0]
            }
        }
        
        importance_data['layer_wise_importance'].append(layer_analysis)
        importance_data['neuron_wise_importance'].append(final_importance_normalized)
        
        # Compute summary statistics for this layer
        summary = {
            'mean_importance': float(final_importance_normalized.mean()),
            'std_importance': float(final_importance_normalized.std()),
            'top_10_percent_threshold': float(torch.quantile(final_importance_normalized, 0.9)),
            'bottom_10_percent_threshold': float(torch.quantile(final_importance_normalized, 0.1)),
            'num_highly_important': int((final_importance_normalized > 0.8).sum()),
            'num_low_important': int((final_importance_normalized < 0.2).sum())
        }
        
        importance_data['layer_importance_summary'][f'layer_{layer_idx}'] = summary
        
        print(f"   📋 Layer {layer_idx} Summary:")
        print(f"      Mean importance: {summary['mean_importance']:.4f}")
        print(f"      Highly important neurons (>0.8): {summary['num_highly_important']}")
        print(f"      Low important neurons (<0.2): {summary['num_low_important']}")
    
    # Compute global importance ranking across all layers
    print(f"\n{Fore.MAGENTA}🌐 Computing Global Importance Ranking{Fore.RESET}")
    
    all_importance_scores = torch.cat(importance_data['neuron_wise_importance'], dim=0)
    global_ranking = torch.argsort(all_importance_scores, descending=True)
    
    importance_data['global_importance_ranking'] = global_ranking
    
    # Convert global indices back to (layer, neuron) pairs
    top_k = 100  # Top 100 most important neurons globally
    top_neurons = []
    
    for i in range(min(top_k, len(global_ranking))):
        global_idx = global_ranking[i]
        layer_idx = global_idx // intermediate_size
        neuron_idx = global_idx % intermediate_size
        importance_score = all_importance_scores[global_idx]
        
        top_neurons.append({
            'layer': int(layer_idx),
            'neuron': int(neuron_idx), 
            'importance': float(importance_score),
            'global_rank': i + 1
        })
    
    importance_data['top_neurons_global'] = top_neurons
    
    print(f"   🏆 Top 10 Most Important Neurons Globally:")
    for i in range(min(10, len(top_neurons))):
        neuron = top_neurons[i]
        print(f"      #{neuron['global_rank']}: Layer {neuron['layer']}, Neuron {neuron['neuron']} (score: {neuron['importance']:.4f})")
    
    # Compute cross-layer importance correlation
    if num_layers > 1:
        print(f"\n{Fore.CYAN}🔗 Computing Cross-Layer Importance Correlations{Fore.RESET}")
        
        correlations = torch.zeros((num_layers, num_layers))
        for i in range(num_layers):
            for j in range(num_layers):
                corr = torch.corrcoef(torch.stack([
                    importance_data['neuron_wise_importance'][i],
                    importance_data['neuron_wise_importance'][j] 
                ]))[0, 1]
                correlations[i, j] = corr if not torch.isnan(corr) else 0.0
        
        importance_data['layer_correlations'] = correlations
        
        print(f"   📊 Layer correlation matrix shape: {correlations.shape}")
        print(f"   📈 Average cross-layer correlation: {correlations.mean():.4f}")
    
    print(f"\n{Fore.GREEN}✅ Gate Importance Analysis Complete!{Fore.RESET}")
    print(f"   📊 Analyzed {num_layers} layers with {intermediate_size} neurons each")
    print(f"   🎯 Total neurons analyzed: {num_layers * intermediate_size}")
    
    return importance_data


def visualize_importance_distribution(importance_data: Dict, layer_idx: int = 0):
    """Debug function to print importance distribution for a specific layer"""
    print(f"\n{Fore.CYAN}📈 DEBUG: Importance Distribution for Layer {layer_idx}{Fore.RESET}")
    
    if layer_idx >= len(importance_data['neuron_wise_importance']):
        print(f"   ❌ Layer {layer_idx} not found!")
        return
    
    importance = importance_data['neuron_wise_importance'][layer_idx]
    
    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = [torch.quantile(importance, p/100.0) for p in percentiles]
    
    print(f"   📊 Importance Distribution:")
    for p, v in zip(percentiles, values):
        print(f"      {p:2d}th percentile: {v:.6f}")
    
    # Count neurons in different importance ranges
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print(f"   🎯 Neuron Count by Importance Range:")
    
    for low, high in ranges:
        count = ((importance >= low) & (importance < high)).sum()
        percentage = count / len(importance) * 100
        print(f"      [{low:.1f}, {high:.1f}): {count:4d} neurons ({percentage:5.1f}%)")


def debug_gate_patterns(storage: Dict, importance_data: Dict, layer_idx: int = 0, num_samples: int = 5):
    """Debug function to examine gate activation patterns for specific neurons"""
    print(f"\n{Fore.YELLOW}🔍 DEBUG: Gate Patterns for Layer {layer_idx}{Fore.RESET}")
    
    if layer_idx >= len(storage['gate_projections']):
        print(f"   ❌ Layer {layer_idx} not found!")
        return
    
    gate_proj = storage['gate_projections'][layer_idx]
    importance = importance_data['neuron_wise_importance'][layer_idx]
    
    # Get top important neurons
    top_indices = torch.argsort(importance, descending=True)[:num_samples]
    bottom_indices = torch.argsort(importance, descending=False)[:num_samples]
    
    print(f"   🏆 Top {num_samples} Important Neurons:")
    for i, neuron_idx in enumerate(top_indices):
        neuron_activations = gate_proj[:, neuron_idx]
        print(f"      Neuron {neuron_idx:4d}: importance={importance[neuron_idx]:.4f}, "
              f"mean={neuron_activations.mean():.4f}, std={neuron_activations.std():.4f}")
    
    print(f"   📉 Bottom {num_samples} Important Neurons:")
    for i, neuron_idx in enumerate(bottom_indices):
        neuron_activations = gate_proj[:, neuron_idx]
        print(f"      Neuron {neuron_idx:4d}: importance={importance[neuron_idx]:.4f}, "
              f"mean={neuron_activations.mean():.4f}, std={neuron_activations.std():.4f}")



"""
Coupled Transformation Estimation for Gate-Aware Optimization
Phase 3: Estimate T_gate, T_up, T_down with joint optimization
"""

class CoupledTransformationEstimator:
    """Estimates coupled transformations for gate-aware pruning"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, device: str = 'cuda'):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device
        
        # Initialize transformation matrices
        self.T_gate = nn.Parameter(
            torch.eye(hidden_size, device=device, dtype=torch.float32),
            requires_grad=True
        )
        self.T_up = nn.Parameter(
            torch.eye(hidden_size, device=device, dtype=torch.float32), 
            requires_grad=True
        )
        self.T_down = nn.Parameter(
            torch.eye(intermediate_size, device=device, dtype=torch.float32),
            requires_grad=True
        )
        
        print(f"{Fore.BLUE}🏗️  Initialized transformation matrices:{Fore.RESET}")
        print(f"   T_gate: {self.T_gate.shape}")
        print(f"   T_up: {self.T_up.shape}")
        print(f"   T_down: {self.T_down.shape}")

def estimate_coupled_transformations(
    storage: Dict,
    importance_data: Dict,
    config: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate coupled transformations T_gate, T_up, T_down
    
    Args:
        storage: Collected activations
        importance_data: Gate importance analysis results
        config: Configuration parameters
        
    Returns:
        Tuple of (T_gate, T_up, T_down) transformation matrices
    """
    print(f"\n{Fore.MAGENTA}🔄 Starting Coupled Transformation Estimation{Fore.RESET}")
    
    hidden_size = storage['input_activations'].shape[1]
    intermediate_size = storage['gate_projections'][0].shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"   📐 Hidden size: {hidden_size}")
    print(f"   📐 Intermediate size: {intermediate_size}")
    print(f"   🖥️  Device: {device}")
    
    # Move data to device
    print(f"{Fore.YELLOW}📦 Moving data to {device}...{Fore.RESET}")
    
    input_acts = storage['input_activations'].to(device).float()
    output_acts = storage['output_activations'].to(device).float()
    
    # For simplicity, we'll work with the first layer's projections
    # In practice, you might want to aggregate across layers
    gate_proj = storage['gate_projections'][0].to(device).float()
    up_proj = storage['up_projections'][0].to(device).float()
    gated_intermediate = storage['gated_intermediates'][0].to(device).float()
    
    # Get importance weights
    importance_weights = importance_data['neuron_wise_importance'][0].to(device).float()
    
    print(f"   ✅ Data moved to device successfully")
    print(f"   📊 Input activations: {input_acts.shape}")
    print(f"   📊 Gate projections: {gate_proj.shape}")
    print(f"   📊 Importance weights: {importance_weights.shape}")
    
    # Initialize transformation matrices
    T_gate = torch.eye(hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    T_up = torch.eye(hidden_size, device=device, dtype=torch.float32, requires_grad=True)  
    T_down = torch.eye(intermediate_size, device=device, dtype=torch.float32, requires_grad=True)
    
    # Setup optimizer
    optimizer = Adam([T_gate, T_up, T_down], lr=config.get('lr', 1e-4))
    
    # Training parameters
    num_epochs = config.get('coupled_epochs', 20)
    batch_size = config.get('batch_size', 1024)
    importance_weight = config.get('importance_weight', 1.0)
    consistency_weight = config.get('consistency_weight', 0.1)
    
    print(f"{Fore.GREEN}🎯 Training Parameters:{Fore.RESET}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Importance weight: {importance_weight}")
    print(f"   Consistency weight: {consistency_weight}")
    
    # Training loop
    num_samples = input_acts.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\n{Fore.CYAN}🚀 Starting Joint Optimization...{Fore.RESET}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Shuffle data
        indices = torch.randperm(num_samples, device=device)
        
        progress_bar = tqdm(
            range(num_batches),
            desc=f"{Fore.GREEN}Epoch {epoch+1}/{num_epochs}{Fore.RESET}",
            leave=False
        )
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_input = input_acts[batch_indices]  # [batch_size, hidden_size]
            batch_output = output_acts[batch_indices]  # [batch_size, hidden_size]
            batch_gate = gate_proj[batch_indices]  # [batch_size, intermediate_size]
            batch_up = up_proj[batch_indices]  # [batch_size, intermediate_size]
            batch_gated = gated_intermediate[batch_indices]  # [batch_size, intermediate_size]
            
            optimizer.zero_grad()
            
            # Forward pass through coupled transformations
            # Simulate the SwiGLU operation with transformations
            
            # 1. Transform input for gate and up paths
            transformed_input_gate = batch_input @ T_gate  # [batch_size, hidden_size]
            transformed_input_up = batch_input @ T_up      # [batch_size, hidden_size]
            
            # 2. Project to intermediate space (simulate gate_proj and up_proj)
            # We need to simulate: new_gate_proj = T_gate.T @ original_gate_proj
            # But we work with activations, so we approximate the transformation
            
            # Compute target transformations
            pred_gate = transformed_input_gate @ torch.randn(hidden_size, intermediate_size, device=device)
            pred_up = transformed_input_up @ torch.randn(hidden_size, intermediate_size, device=device)
            
            # Apply SwiGLU mechanism
            pred_gated = F.silu(pred_gate) * pred_up  # [batch_size, intermediate_size]
            
            # 3. Apply down transformation
            pred_intermediate_transformed = pred_gated @ T_down  # [batch_size, intermediate_size]
            
            # For simplicity, let's focus on the core transformations
            # Loss 1: Gate reconstruction loss (with importance weighting)
            gate_loss = torch.mean(
                importance_weights.unsqueeze(0) * (pred_gate - batch_gate) ** 2
            )
            
            # Loss 2: Up projection reconstruction loss
            up_loss = torch.mean((pred_up - batch_up) ** 2)
            
            # Loss 3: Gated intermediate reconstruction loss  
            gated_loss = torch.mean((pred_gated - batch_gated) ** 2)
            
            # Loss 4: Final output reconstruction loss
            # This is a simplified version - in practice, you'd need the full forward pass
            output_loss = torch.mean((transformed_input_gate - batch_output) ** 2)
            
            # Loss 5: Consistency regularization
            # Ensure transformations are consistent with each other
            consistency_loss = (
                torch.norm(T_gate @ T_gate.T - torch.eye(hidden_size, device=device)) +
                torch.norm(T_up @ T_up.T - torch.eye(hidden_size, device=device)) +
                torch.norm(T_down @ T_down.T - torch.eye(intermediate_size, device=device))
            )
            
            # Total loss
            total_loss = (
                gate_loss + 
                up_loss + 
                gated_loss +
                0.5 * output_loss +
                consistency_weight * consistency_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Gate': f'{gate_loss.item():.4f}',
                    'Up': f'{up_loss.item():.4f}',
                    'Gated': f'{gated_loss.item():.4f}',
                    'Consistency': f'{consistency_loss.item():.4f}'
                })
        
        avg_loss = np.mean(epoch_losses)
        print(f"   📊 Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if epoch > 5 and avg_loss < 1e-6:
            print(f"   🎯 Early stopping - loss converged!")
            break
    
    print(f"\n{Fore.GREEN}✅ Coupled Transformation Estimation Complete!{Fore.RESET}")
    
    # Move results back to CPU
    T_gate_final = T_gate.detach().cpu()
    T_up_final = T_up.detach().cpu() 
    T_down_final = T_down.detach().cpu()
    
    # Debug: Print transformation properties
    print(f"{Fore.CYAN}🔍 Final Transformation Properties:{Fore.RESET}")
    print(f"   T_gate - determinant: {torch.det(T_gate_final):.4f}")
    print(f"   T_up - determinant: {torch.det(T_up_final):.4f}")
    print(f"   T_down - determinant: {torch.det(T_down_final):.4f}")
    
    print(f"   T_gate - Frobenius norm: {torch.norm(T_gate_final, 'fro'):.4f}")
    print(f"   T_up - Frobenius norm: {torch.norm(T_up_final, 'fro'):.4f}")
    print(f"   T_down - Frobenius norm: {torch.norm(T_down_final, 'fro'):.4f}")
    
    return T_gate_final, T_up_final, T_down_final


def debug_transformation_quality(
    T_gate: torch.Tensor, 
    T_up: torch.Tensor, 
    T_down: torch.Tensor,
    storage: Dict
):
    """Debug function to evaluate transformation quality"""
    print(f"\n{Fore.YELLOW}🔍 DEBUG: Transformation Quality Analysis{Fore.RESET}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move to device for computation
    T_gate = T_gate.to(device)
    T_up = T_up.to(device)
    T_down = T_down.to(device)
    
    input_acts = storage['input_activations'][:1000].to(device).float()  # Sample for speed
    gate_proj = storage['gate_projections'][0][:1000].to(device).float()
    up_proj = storage['up_projections'][0][:1000].to(device).float()
    
    # Test transformation application
    transformed_input_gate = input_acts @ T_gate
    transformed_input_up = input_acts @ T_up
    
    # Compute reconstruction errors
    gate_error = torch.mean((transformed_input_gate - input_acts) ** 2)
    up_error = torch.mean((transformed_input_up - input_acts) ** 2)
    
    print(f"   📊 Reconstruction Errors:")
    print(f"      Gate transformation error: {gate_error:.6f}")
    print(f"      Up transformation error: {up_error:.6f}")
    
    # Compute condition numbers (stability measure)
    try:
        gate_cond = torch.linalg.cond(T_gate)
        up_cond = torch.linalg.cond(T_up) 
        down_cond = torch.linalg.cond(T_down)
        
        print(f"   📏 Condition Numbers (lower is better):")
        print(f"      T_gate condition: {gate_cond:.2f}")
        print(f"      T_up condition: {up_cond:.2f}")
        print(f"      T_down condition: {down_cond:.2f}")
        
    except Exception as e:
        print(f"   ⚠️  Could not compute condition numbers: {e}")
    
    # Check orthogonality (for well-conditioned transformations)
    gate_ortho = torch.norm(T_gate @ T_gate.T - torch.eye(T_gate.shape[0], device=device))
    up_ortho = torch.norm(T_up @ T_up.T - torch.eye(T_up.shape[0], device=device))
    
    print(f"   🔄 Orthogonality Measures (lower is better):")
    print(f"      T_gate orthogonality deviation: {gate_ortho:.6f}")
    print(f"      T_up orthogonality deviation: {up_ortho:.6f}")


def save_transformations(
    T_gate: torch.Tensor,
    T_up: torch.Tensor, 
    T_down: torch.Tensor,
    save_path: str
):
    """Save transformation matrices to disk"""
    print(f"\n{Fore.BLUE}💾 Saving transformation matrices to {save_path}...{Fore.RESET}")
    
    torch.save({
        'T_gate': T_gate,
        'T_up': T_up,
        'T_down': T_down,
        'metadata': {
            'T_gate_shape': T_gate.shape,
            'T_up_shape': T_up.shape, 
            'T_down_shape': T_down.shape,
            'method': 'gate_aware_coupled_optimization'
        }
    }, save_path)
    
    print(f"   ✅ Transformations saved successfully!")


"""
Model Application and Complete Gate-Aware Pipeline Integration
Phase 4: Apply transformations to model + Complete pipeline
"""

# Import our custom modules (these would be the previous artifacts)
# from gate_aware_collection import collect_enhanced_activations, debug_activation_shapes
# from gate_importance_analysis import compute_gate_importance_scores, visualize_importance_distribution
# from coupled_transformation import estimate_coupled_transformations, debug_transformation_quality

def apply_coupled_transformations_to_model(
    model,
    start_id: int,
    end_id: int,
    num_layer: int,
    T_gate: torch.Tensor,
    T_up: torch.Tensor, 
    T_down: torch.Tensor
) -> nn.Module:
    """
    Apply the estimated transformations to the model
    
    Args:
        model: The transformer model
        start_id: Starting layer index 
        end_id: Ending layer index
        num_layer: Number of layers being replaced
        T_gate, T_up, T_down: Estimated transformation matrices
        
    Returns:
        Modified model with transformations applied
    """
    print(f"\n{Fore.MAGENTA}🔧 Applying Coupled Transformations to Model{Fore.RESET}")
    print(f"   📍 Target layer: {start_id - num_layer - 1}")
    print(f"   📊 Replacing layers {start_id - num_layer} to {end_id - num_layer}")
    
    # First truncate the model (remove the blocks we're replacing)
    from utils import truncate_model  # Import from existing utils
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    print(f"   ✂️  Model truncated successfully")
    print(f"   📏 New model size: {model.config.num_hidden_layers} layers")
    
    # Get the target layer where we'll apply transformations
    target_layer_idx = start_id - num_layer - 1
    target_layer = model.model.layers[target_layer_idx]
    
    print(f"   🎯 Target layer MLP components:")
    print(f"      Gate proj: {target_layer.mlp.gate_proj.weight.shape}")
    print(f"      Up proj: {target_layer.mlp.up_proj.weight.shape}")  
    print(f"      Down proj: {target_layer.mlp.down_proj.weight.shape}")
    
    # Apply transformations to each component
    original_device = target_layer.mlp.gate_proj.weight.device
    original_dtype = target_layer.mlp.gate_proj.weight.dtype
    
    # Move transformations to correct device and dtype
    T_gate = T_gate.to(device=original_device, dtype=torch.float64)
    T_up = T_up.to(device=original_device, dtype=torch.float64) 
    T_down = T_down.to(device=original_device, dtype=torch.float64)
    
    print(f"   📦 Transformations moved to {original_device} with dtype {torch.float64}")
    
    # Apply transformations (this is the key difference from original ReplaceMe)
    
    # 1. Transform gate projection 
    # Original: gate_out = input @ W_gate
    # New: gate_out = input @ T_gate @ W_gate  
    # So: W_gate_new = T_gate.T @ W_gate_original
    original_gate_weight = target_layer.mlp.gate_proj.weight.data.to(torch.float64)
    new_gate_weight = (T_gate.T @ original_gate_weight.T).T
    target_layer.mlp.gate_proj.weight.data = new_gate_weight.to(original_dtype)
    
    print(f"   ✅ Gate projection transformed: {original_gate_weight.shape} -> {new_gate_weight.shape}")
    
    # 2. Transform up projection
    original_up_weight = target_layer.mlp.up_proj.weight.data.to(torch.float64)
    new_up_weight = (T_up.T @ original_up_weight.T).T  
    target_layer.mlp.up_proj.weight.data = new_up_weight.to(original_dtype)
    
    print(f"   ✅ Up projection transformed: {original_up_weight.shape} -> {new_up_weight.shape}")
    
    # 3. Transform down projection
    # Original: output = intermediate @ W_down
    # New: output = intermediate @ T_down @ W_down
    # So: W_down_new = T_down @ W_down_original
    original_down_weight = target_layer.mlp.down_proj.weight.data.to(torch.float64)
    new_down_weight = (T_down @ original_down_weight)
    target_layer.mlp.down_proj.weight.data = new_down_weight.to(original_dtype)
    
    print(f"   ✅ Down projection transformed: {original_down_weight.shape} -> {new_down_weight.shape}")
    
    print(f"\n{Fore.GREEN}🎉 All transformations applied successfully!{Fore.RESET}")
    print(f"   🧮 Total parameters modified: {new_gate_weight.numel() + new_up_weight.numel() + new_down_weight.numel()}")
    
    return model


def gate_aware_coupled_method(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: int = None,
    dataset_subset: str = "eval",
    use_4bit: bool = False,
    save_path: str = None,
    token: str = None,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """
    Complete Gate-Aware Coupled Optimization Pipeline
    
    This is the main function that will be called from ReplaceMe_pipeline.py
    """
    print(f"\n{Fore.CYAN}🚀 Starting Gate-Aware Coupled Optimization Pipeline{Fore.RESET}")
    print(f"   📂 Model: {model_path}")
    print(f"   📊 Dataset: {dataset} ({dataset_size} samples)")
    print(f"   🔢 Layers to skip: {layers_to_skip}")
    
    # Device setup
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(f"   🔧 4-bit quantization enabled")
    
    # Load model and tokenizer
    print(f"\n{Fore.YELLOW}📥 Loading model and tokenizer...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    hidden_size = model.config.hidden_size
    
    print(f"   ✅ Model loaded: {model.config.num_hidden_layers} layers, {hidden_size} hidden size")
    
    # Load dataloader
    from utils import get_calib_dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer
    )
    
    print(f"   ✅ Dataloader ready: {len(dataloader)} batches")
    
    # Load distances and select blocks
    print(f"\n{Fore.BLUE}📏 Loading distances and selecting blocks...{Fore.RESET}")
    
    average_distances = torch.load(distances_path)
    from utils import select_non_overlapping_blocks
    
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers_cumulative = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    print(f"   ✅ Selected blocks: {selected_blocks}")
    print(f"   📊 Start IDs: {start_ids}")
    print(f"   📊 End IDs: {end_ids}")
    
    # Process each block
    for i in range(len(selected_blocks)):
        start_id = start_ids[i]
        end_id = end_ids[i] 
        num_layer = num_layers_cumulative[i]
        
        print(f"\n{Fore.MAGENTA}🔄 Processing Block {i+1}/{len(selected_blocks)}{Fore.RESET}")
        print(f"   📍 Start: {start_id}, End: {end_id}, Num layer: {num_layer}")
        
        # Phase 1: Enhanced Activation Collection
        print(f"\n{Fore.GREEN}Phase 1: Enhanced Activation Collection{Fore.RESET}")
        storage = collect_enhanced_activations(
            model=model,
            start_id=start_id, 
            end_id=end_id,
            dataloader=dataloader,
            max_length=max_length,
            dataset_size=dataset_size,
            hidden_size=hidden_size
        )
        
        # Debug activation shapes
        debug_activation_shapes(storage)
        
        # Phase 2: Gate Importance Analysis
        print(f"\n{Fore.GREEN}Phase 2: Gate Importance Analysis{Fore.RESET}")
        importance_data = compute_gate_importance_scores(storage, config=kwargs)
        
        # Debug importance distribution
        visualize_importance_distribution(importance_data, layer_idx=0)
        
        # Phase 3: Coupled Transformation Estimation
        print(f"\n{Fore.GREEN}Phase 3: Coupled Transformation Estimation{Fore.RESET}")
        T_gate, T_up, T_down = estimate_coupled_transformations(
            storage=storage,
            importance_data=importance_data, 
            config=kwargs
        )
        
        # Debug transformation quality
        debug_transformation_quality(T_gate, T_up, T_down, storage)
        
        # Phase 4: Apply Transformations
        print(f"\n{Fore.GREEN}Phase 4: Apply Transformations to Model{Fore.RESET}")
        model = apply_coupled_transformations_to_model(
            model=model,
            start_id=start_id,
            end_id=end_id,
            num_layer=num_layer,
            T_gate=T_gate,
            T_up=T_up,
            T_down=T_down
        )
        
        print(f"   🎯 Block {i+1} processing complete!")
        
        # Cleanup
        del storage, importance_data, T_gate, T_up, T_down
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save the final model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_gate_aware_coupled_{layers_to_skip}_layers"
    
    final_save_path = f"{save_path}_GateCoupled_{num_A}"
    
    print(f"\n{Fore.BLUE}💾 Saving final model...{Fore.RESET}")
    print(f"   📁 Save path: {final_save_path}")
    
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print(f"\n{Fore.GREEN}🎉 Gate-Aware Coupled Optimization Complete!{Fore.RESET}")
    print(f"   ✅ Model saved to: {final_save_path}")
    
    # Final cleanup
    del model
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_save_path


"""
Integration code for ReplaceMe_pipeline.py
Add the gate-aware coupled optimization method
"""

# Add these imports to the top of ReplaceMe_pipeline.py
"""
Additional imports needed for gate-aware method:
"""

def collect_enhanced_activations(
    model, start_id, end_id, dataloader, max_length, dataset_size, hidden_size, tokenizer
):
    """Enhanced activation collection for gate-aware coupled optimization"""
    print(f"{Fore.GREEN}🔍 Starting Enhanced Activation Collection{Fore.RESET}")
    print(f"   📍 Replacing layers {start_id} to {end_id}")
    
    # Setup activation hooks
    def save_activation(name):
        def hook(module, input, output):
            activations_dict[name] = output.detach()
        return hook
    
    hooks = []
    activations_dict = {}
    
    # Register hooks for SwiGLU components
    for i in range(start_id-1, end_id):
        layer = model.model.layers[i]
        hooks.append(layer.mlp.gate_proj.register_forward_hook(save_activation(f'layer_{i}_gate_proj')))
        hooks.append(layer.mlp.up_proj.register_forward_hook(save_activation(f'layer_{i}_up_proj')))
        
        def save_gated_intermediate(layer_idx):
            def hook(module, input, output):
                activations_dict[f'layer_{layer_idx}_gated_intermediate'] = input[0].detach()
            return hook
        hooks.append(layer.mlp.down_proj.register_forward_hook(save_gated_intermediate(i)))
    
    # Initialize storage
    total_tokens = dataset_size * max_length
    num_layers = end_id - start_id
    
    storage = {
        'input_activations': torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'output_activations': torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'gate_projections': [torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu') for _ in range(num_layers)],
        'up_projections': [torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu') for _ in range(num_layers)],
        'gated_intermediates': [torch.empty((total_tokens, model.config.intermediate_size), dtype=torch.bfloat16, device='cpu') for _ in range(num_layers)],
    }
    
    # Collect activations
    cnt = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Process batch - dataloader already provides processed inputs
            if isinstance(batch, dict):
                # If dataloader already provides tokenized inputs
                inputs = {k: v.to(model.device) for k, v in batch.items()}
            else:
                # If dataloader provides raw text, tokenize it
                inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Store activations
            input_acts = hidden_states[start_id - 1].view(-1, hidden_size).cpu()
            output_acts = hidden_states[end_id - 1].view(-1, hidden_size).cpu()
            batch_size = input_acts.shape[0]
            
            storage['input_activations'][cnt:cnt+batch_size] = input_acts
            storage['output_activations'][cnt:cnt+batch_size] = output_acts
            
            # Store SwiGLU components
            for layer_offset in range(num_layers):
                layer_idx = start_id - 1 + layer_offset
                gate_acts = activations_dict[f'layer_{layer_idx}_gate_proj'].view(-1, model.config.intermediate_size).cpu()
                up_acts = activations_dict[f'layer_{layer_idx}_up_proj'].view(-1, model.config.intermediate_size).cpu()
                gated_acts = activations_dict[f'layer_{layer_idx}_gated_intermediate'].view(-1, model.config.intermediate_size).cpu()
                
                storage['gate_projections'][layer_offset][cnt:cnt+batch_size] = gate_acts
                storage['up_projections'][layer_offset][cnt:cnt+batch_size] = up_acts
                storage['gated_intermediates'][layer_offset][cnt:cnt+batch_size] = gated_acts
            
            cnt += batch_size
            activations_dict.clear()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   🔄 Processed {batch_idx + 1} batches, {cnt} tokens")

            if batch_idx >= len(dataloader) * 0.5:  # 50% 배치 처리하면 중단
                break
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    # Trim to actual size
    for key in ['input_activations', 'output_activations']:
        storage[key] = storage[key][:cnt]
    for layer_offset in range(num_layers):
        storage['gate_projections'][layer_offset] = storage['gate_projections'][layer_offset][:cnt]
        storage['up_projections'][layer_offset] = storage['up_projections'][layer_offset][:cnt]
        storage['gated_intermediates'][layer_offset] = storage['gated_intermediates'][layer_offset][:cnt]
    
    print(f"{Fore.GREEN}✅ Enhanced Activation Collection Complete!{Fore.RESET}")
    return storage


def compute_gate_importance_scores(storage, config):
    """Compute gate importance scores"""
    print(f"{Fore.BLUE}🧠 Starting Gate Importance Analysis{Fore.RESET}")
    
    importance_data = {'neuron_wise_importance': []}
    
    for layer_idx, gate_proj in enumerate(storage['gate_projections']):
        # Apply SiLU activation
        gate_activated = torch.sigmoid(gate_proj) * gate_proj
        
        # Compute importance metrics
        gate_variance = torch.var(gate_activated, dim=0)
        gate_mean_abs = torch.abs(torch.mean(gate_activated, dim=0))
        gate_dynamic_range = torch.max(gate_activated, dim=0)[0] - torch.min(gate_activated, dim=0)[0]
        
        # Combine metrics
        combined_importance = gate_variance * gate_mean_abs
        final_importance = (0.6 * combined_importance + 0.4 * gate_dynamic_range)
        
        # Normalize
        final_importance_normalized = (final_importance - final_importance.min()) / (
            final_importance.max() - final_importance.min() + 1e-8
        )
        
        importance_data['neuron_wise_importance'].append(final_importance_normalized)
        
        print(f"   📊 Layer {layer_idx} - Importance range: [{final_importance_normalized.min():.4f}, {final_importance_normalized.max():.4f}]")
    
    print(f"{Fore.GREEN}✅ Gate Importance Analysis Complete!{Fore.RESET}")
    return importance_data


def estimate_coupled_transformations(storage, importance_data, config):
    """Estimate coupled transformations with simplified approach"""
    print(f"{Fore.MAGENTA}🔄 Starting Coupled Transformation Estimation{Fore.RESET}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = storage['input_activations'].shape[1]
    intermediate_size = storage['gate_projections'][0].shape[1]
    
    # Move data to device
    input_acts = storage['input_activations'].to(device).float()
    output_acts = storage['output_activations'].to(device).float()
    gate_proj = storage['gate_projections'][0].to(device).float()
    importance_weights = importance_data['neuron_wise_importance'][0].to(device).float()
    
    print(f"   📊 Working with shapes - Input: {input_acts.shape}, Gate: {gate_proj.shape}")
    
    # Simplified transformation estimation using least squares with importance weighting
    
    # Method 1: Gate-aware linear transformation (similar to original ReplaceMe but with gate weighting)
    # We want: input @ T ≈ output, but weighted by gate importance
    
    # Compute importance-weighted transformation
    # Weight the loss by gate importance (averaged across intermediate dimension)
    gate_importance_avg = importance_weights.mean()  # Scalar weight for this layer
    
    print(f"   🎯 Average gate importance: {gate_importance_avg:.4f}")
    
    # Use original ReplaceMe-style least squares but with importance weighting
    # T* = (X^T X + λI)^(-1) X^T Y, where λ is regularization based on importance
    
    regularization = config.get('consistency_weight', 0.1) * (1.0 - gate_importance_avg)
    
    # Compute gram matrix with regularization
    XtX = input_acts.T @ input_acts
    XtY = input_acts.T @ output_acts
    reg_term = regularization * torch.eye(hidden_size, device=device, dtype=torch.float32)
    
    # Solve for transformation matrix
    T_combined = torch.linalg.solve(XtX + reg_term, XtY)
    
    print(f"   📐 Computed transformation matrix: {T_combined.shape}")
    print(f"   🔍 Transformation norm: {torch.norm(T_combined):.4f}")
    print(f"   🔍 Regularization used: {regularization:.6f}")
    
    # For coupled approach, we decompose this into three transformations
    # This is a simplified version - in practice, you might want more sophisticated decomposition
    
    # T_gate and T_up are initialized as identity with small perturbations based on importance
    T_gate = torch.eye(hidden_size, device=device, dtype=torch.float32)
    T_up = torch.eye(hidden_size, device=device, dtype=torch.float32)
    
    # Add importance-guided perturbations
    gate_perturbation = 0.01 * gate_importance_avg * torch.randn_like(T_gate)
    T_gate += gate_perturbation
    T_up += 0.01 * (1.0 - gate_importance_avg) * torch.randn_like(T_up)
    
    # T_down is the main transformation matrix (projected to intermediate space)
    T_down = T_combined[:intermediate_size, :].T  # Take first intermediate_size rows and transpose
    
    # Ensure proper dimensions
    if T_down.shape[0] != intermediate_size:
        T_down = torch.eye(intermediate_size, device=device, dtype=torch.float32)
        print(f"   ⚠️  Dimension mismatch, using identity for T_down")
    
    print(f"   ✅ Final transformation shapes:")
    print(f"      T_gate: {T_gate.shape}")
    print(f"      T_up: {T_up.shape}")
    print(f"      T_down: {T_down.shape}")
    
    # Move back to CPU
    T_gate_final = T_gate.detach().cpu()
    T_up_final = T_up.detach().cpu()
    T_down_final = T_down.detach().cpu()
    
    print(f"{Fore.GREEN}✅ Coupled Transformation Estimation Complete!{Fore.RESET}")
    
    return T_gate_final, T_up_final, T_down_final


def apply_coupled_transformations_to_model(model, start_id, end_id, num_layer, T_gate, T_up, T_down):
    """Apply transformations to model"""
    print(f"{Fore.MAGENTA}🔧 Applying Coupled Transformations to Model{Fore.RESET}")
    
    # Truncate model first
    from .utils import truncate_model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Get target layer
    target_layer_idx = start_id - num_layer - 1
    target_layer = model.model.layers[target_layer_idx]
    
    # Move transformations to correct device/dtype
    device = target_layer.mlp.gate_proj.weight.device
    dtype = torch.float64  # Use higher precision for transformations
    
    T_gate = T_gate.to(device=device, dtype=dtype)
    T_up = T_up.to(device=device, dtype=dtype)
    T_down = T_down.to(device=device, dtype=dtype)
    
    print(f"   📍 Target layer: {target_layer_idx}")
    print(f"   🔧 Applying transformations...")
    
    # Apply transformations to weights
    # Gate projection: W_new = T_gate.T @ W_original.T (transposed back)
    original_gate = target_layer.mlp.gate_proj.weight.data.to(dtype)
    new_gate_weight = (T_gate.T @ original_gate.T).T
    target_layer.mlp.gate_proj.weight.data = new_gate_weight.to(target_layer.mlp.gate_proj.weight.dtype)
    
    # Up projection  
    original_up = target_layer.mlp.up_proj.weight.data.to(dtype)
    new_up_weight = (T_up.T @ original_up.T).T
    target_layer.mlp.up_proj.weight.data = new_up_weight.to(target_layer.mlp.up_proj.weight.dtype)
    
    # Down projection
    original_down = target_layer.mlp.down_proj.weight.data.to(dtype)
    new_down_weight = T_down @ original_down
    target_layer.mlp.down_proj.weight.data = new_down_weight.to(target_layer.mlp.down_proj.weight.dtype)
    
    print(f"   ✅ All transformations applied successfully!")
    
    return model


def gate_aware_coupled_optimization(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: int = None,
    dataset_subset: str = "eval",
    use_4bit: bool = False,
    save_path: str = None,
    token: str = None,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    **kwargs
) -> str:
    """Main gate-aware coupled optimization function"""
    
    print(f"\n{Fore.CYAN}🚀 Gate-Aware Coupled Optimization Pipeline{Fore.RESET}")
    print(f"   📂 Model: {model_path}")
    print(f"   📍 Processing layers {start_id} to {end_id} (num_layer: {num_layer})")
    
    # Load model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataloader
    from .utils import get_calib_dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer
    )
    
    hidden_size = model.config.hidden_size
    
    print(f"   ✅ Model loaded: {hidden_size} hidden size")
    
    # Phase 1: Collect activations
    storage = collect_enhanced_activations(
        model, start_id, end_id, dataloader, max_length, dataset_size, hidden_size, tokenizer
    )
    
    # Phase 2: Analyze gate importance
    importance_data = compute_gate_importance_scores(storage, kwargs)
    
    # Phase 3: Estimate transformations
    T_gate, T_up, T_down = estimate_coupled_transformations(storage, importance_data, kwargs)
    
    # Phase 4: Apply transformations
    model = apply_coupled_transformations_to_model(model, start_id, end_id, num_layer, T_gate, T_up, T_down)
    
    # Save model
    if save_path is None:
        import os
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_gate_aware_{layers_to_skip}_layers_{start_id}_{end_id}"
    
    final_save_path = f"{save_path}_GateCoupled"
    
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print(f"{Fore.GREEN}🎉 Gate-Aware Coupled Optimization Complete!{Fore.RESET}")
    print(f"   💾 Saved to: {final_save_path}")
    
    # Cleanup
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_save_path

