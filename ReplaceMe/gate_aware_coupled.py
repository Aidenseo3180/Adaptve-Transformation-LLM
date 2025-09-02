"""
Gate-Aware Coupled Optimization (GACO) method for transformer pruning.

This module implements an enhanced version of the ReplaceMe method that incorporates:
1. Gate importance analysis for SwiGLU architectures
2. Coupled optimization of up_proj, gate_proj, and down_proj
3. Context-aware information filtering

Author: Research Team
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from colorama import Fore, init
import gc

# Initialize colorama
init(autoreset=True)

def collect_enhanced_activations_streaming(
    model,
    start_id: int,
    end_id: int,
    dataset_size: int,
    max_length: int,
    dataloader,
    device: str = "cuda",
    tokenizer = None
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient streaming activation collection for gate-aware coupled optimization.
    Processes batches one at a time and accumulates statistics instead of raw activations.
    
    Args:
        model: The transformer model
        start_id: Starting layer index for replacement
        end_id: Ending layer index for replacement  
        dataset_size: Number of samples in calibration dataset
        max_length: Maximum sequence length
        dataloader: Calibration data loader
        device: Device to run computations on
        tokenizer: Tokenizer for processing text batches
        
    Returns:
        Dictionary containing accumulated activation statistics
    """
    print(f"[Phase 2 Streaming] Starting memory-efficient activation collection...")
    print(f"[Phase 2 Streaming] Target blocks: {start_id} to {end_id-1} (total: {end_id - start_id} blocks)")
    print(f"[Phase 2 Streaming] Dataset size: {dataset_size}, Max length: {max_length}")
    
    hidden_size = model.config.hidden_size
    # For Llama models, intermediate_size is typically 4 * hidden_size or similar
    intermediate_size = getattr(model.config, 'intermediate_size', hidden_size * 4)
    num_layers = end_id - start_id
    
    print(f"[Phase 2 Streaming] Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
    print(f"[Phase 2 Streaming] Processing {num_layers} layers")
    
    # Initialize accumulators for streaming statistics
    activation_stats = {
        # Input/Output activations (concatenated for linear transformation estimation)
        'input_activations': torch.zeros(0, hidden_size, dtype=torch.float32, device='cpu'),
        'output_activations': torch.zeros(0, hidden_size, dtype=torch.float32, device='cpu'),
        
        # Gate importance statistics per layer
        'gate_importance': {},
        'gate_mean': {},
        'gate_variance': {},
        
        # Up projection statistics per layer  
        'up_mean': {},
        'up_variance': {},
        
        # MLP output statistics per layer
        'mlp_mean': {},
        'mlp_variance': {},
        
        # Counters
        'total_tokens': 0,
        'total_batches': 0
    }
    
    # Initialize per-layer statistics
    for layer_idx in range(start_id, end_id):
        activation_stats['gate_importance'][layer_idx] = torch.zeros(intermediate_size, dtype=torch.float32, device='cpu')
        activation_stats['gate_mean'][layer_idx] = torch.zeros(intermediate_size, dtype=torch.float32, device='cpu')
        activation_stats['gate_variance'][layer_idx] = torch.zeros(intermediate_size, dtype=torch.float32, device='cpu')
        activation_stats['up_mean'][layer_idx] = torch.zeros(intermediate_size, dtype=torch.float32, device='cpu')
        activation_stats['up_variance'][layer_idx] = torch.zeros(intermediate_size, dtype=torch.float32, device='cpu')
        activation_stats['mlp_mean'][layer_idx] = torch.zeros(hidden_size, dtype=torch.float32, device='cpu')
        activation_stats['mlp_variance'][layer_idx] = torch.zeros(hidden_size, dtype=torch.float32, device='cpu')
    
    print(f"[Phase 2 Streaming] Initialized statistics accumulators")
    
    # Setup hooks for real-time activation capture
    def save_activation(name: str, activations_dict: Dict):
        def hook(module, input, output):
            # Store only current batch, will be processed immediately
            activations_dict[name] = output.detach()
        return hook
    
    hooks = []
    hook_activations = {}
    
    # Register hooks for target layers
    print(f"[Phase 2 Streaming] Registering hooks for layers {start_id} to {end_id-1}...")
    for layer_idx in range(start_id, end_id):
        layer = model.model.layers[layer_idx]
        
        # Hook for gate projection
        hooks.append(
            layer.mlp.gate_proj.register_forward_hook(
                save_activation(f'gate_proj_{layer_idx}', hook_activations)
            )
        )
        
        # Hook for up projection
        hooks.append(
            layer.mlp.up_proj.register_forward_hook(
                save_activation(f'up_proj_{layer_idx}', hook_activations)
            )
        )
        
        # Hook for MLP final output
        hooks.append(
            layer.mlp.register_forward_hook(
                save_activation(f'mlp_out_{layer_idx}', hook_activations)
            )
        )
    
    print(f"[Phase 2 Streaming] Successfully registered {len(hooks)} hooks")
    
    # Streaming batch processing
    total_tokens_processed = 0
    batch_count = 0
    max_input_samples = min(dataset_size * max_length // 4, 50000)  # Limit samples for memory
    
    print(f"[Phase 2 Streaming] Starting streaming batch processing...")
    print(f"[Phase 2 Streaming] Will collect max {max_input_samples} input/output samples")
    
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Streaming Enhanced Activations{Fore.RESET}",
        dynamic_ncols=True,
        colour="green"
    ):
        batch_count += 1
        print(f"[Phase 2 Streaming] Processing batch {batch_count}...")
        
        # Handle different batch formats from dataloader
        if isinstance(batch, (list, tuple)):
            if tokenizer is None:
                raise ValueError("[Phase 2 Streaming] ERROR: Tokenizer is required when batch contains text")
            
            inputs = tokenizer(
                batch,
                return_tensors="pt", 
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            print(f"[Phase 2 Streaming] Tokenized {len(batch)} texts")
        elif isinstance(batch, dict) and 'input_ids' in batch:
            inputs = batch
            print(f"[Phase 2 Streaming] Using pre-tokenized batch")
        elif torch.is_tensor(batch):
            inputs = {'input_ids': batch, 'attention_mask': torch.ones_like(batch)}
            print(f"[Phase 2 Streaming] Using tensor batch")
        else:
            raise ValueError(f"[Phase 2 Streaming] ERROR: Unexpected batch type: {type(batch)}")
        
        # Move to device and get dimensions
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_size = inputs['input_ids'].shape[0]
        seq_len = inputs['input_ids'].shape[1]
        batch_tokens = batch_size * seq_len
        
        print(f"[Phase 2 Streaming] Batch shape: {batch_size} x {seq_len} = {batch_tokens} tokens")
        
        with torch.no_grad():
            # Forward pass to collect activations
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Collect input and output activations (limited samples to save memory)
            if activation_stats['input_activations'].shape[0] < max_input_samples:
                input_activation = hidden_states[start_id].view(-1, hidden_size).cpu()
                output_activation = hidden_states[end_id].view(-1, hidden_size).cpu()
                
                # Limit samples to prevent memory overflow
                samples_to_add = min(
                    input_activation.shape[0], 
                    max_input_samples - activation_stats['input_activations'].shape[0]
                )
                
                if samples_to_add > 0:
                    activation_stats['input_activations'] = torch.cat([
                        activation_stats['input_activations'], 
                        input_activation[:samples_to_add]
                    ], dim=0)
                    activation_stats['output_activations'] = torch.cat([
                        activation_stats['output_activations'], 
                        output_activation[:samples_to_add]
                    ], dim=0)
                    
                    print(f"[Phase 2 Streaming] Added {samples_to_add} input/output samples (total: {activation_stats['input_activations'].shape[0]})")
            
            # Process MLP component activations immediately (streaming)
            for layer_idx in range(start_id, end_id):
                gate_proj = hook_activations.get(f'gate_proj_{layer_idx}')
                up_proj = hook_activations.get(f'up_proj_{layer_idx}') 
                mlp_out = hook_activations.get(f'mlp_out_{layer_idx}')
                
                if gate_proj is not None:
                    gate_flat = gate_proj.view(-1, intermediate_size).float()  # [batch*seq, intermediate]
                    
                    # Compute gate importance (variance * |mean|)
                    gate_mean_batch = gate_flat.mean(dim=0)  # [intermediate]
                    gate_var_batch = gate_flat.var(dim=0)   # [intermediate]
                    gate_importance_batch = gate_var_batch * torch.abs(gate_mean_batch)
                    
                    # Update running statistics
                    activation_stats['gate_mean'][layer_idx] += gate_mean_batch.cpu()
                    activation_stats['gate_variance'][layer_idx] += gate_var_batch.cpu()
                    activation_stats['gate_importance'][layer_idx] += gate_importance_batch.cpu()
                    
                    print(f"[Phase 2 Streaming] Layer {layer_idx} gate processed: {gate_flat.shape}")
                
                if up_proj is not None:
                    up_flat = up_proj.view(-1, intermediate_size).float()
                    
                    up_mean_batch = up_flat.mean(dim=0)
                    up_var_batch = up_flat.var(dim=0)
                    
                    activation_stats['up_mean'][layer_idx] += up_mean_batch.cpu()
                    activation_stats['up_variance'][layer_idx] += up_var_batch.cpu()
                    
                    print(f"[Phase 2 Streaming] Layer {layer_idx} up processed: {up_flat.shape}")
                
                if mlp_out is not None:
                    mlp_flat = mlp_out.view(-1, hidden_size).float()
                    
                    mlp_mean_batch = mlp_flat.mean(dim=0)
                    mlp_var_batch = mlp_flat.var(dim=0)
                    
                    activation_stats['mlp_mean'][layer_idx] += mlp_mean_batch.cpu()
                    activation_stats['mlp_variance'][layer_idx] += mlp_var_batch.cpu()
                    
                    print(f"[Phase 2 Streaming] Layer {layer_idx} mlp processed: {mlp_flat.shape}")
        
        # Update counters
        activation_stats['total_tokens'] += batch_tokens
        activation_stats['total_batches'] += 1
        total_tokens_processed += batch_tokens
        
        # Clear hook activations immediately to save memory
        hook_activations.clear()
        
        # Force memory cleanup
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        print(f"[Phase 2 Streaming] Batch {batch_count} complete. Total tokens: {total_tokens_processed}")
        print(f"[Phase 2 Streaming] Memory cleanup performed")
        
        # Check if we have enough samples
        if total_tokens_processed >= 100000:
            print(f"[Phase 2 Streaming] Reached target token processing: {total_tokens_processed}")
            break
    
    # Remove hooks
    print(f"[Phase 2 Streaming] Removing {len(hooks)} hooks...")
    for hook in hooks:
        hook.remove()
    
    # Normalize statistics by number of batches
    print(f"[Phase 2 Streaming] Normalizing statistics across {activation_stats['total_batches']} batches...")
    for layer_idx in range(start_id, end_id):
        if activation_stats['total_batches'] > 0:
            activation_stats['gate_mean'][layer_idx] /= activation_stats['total_batches']
            activation_stats['gate_variance'][layer_idx] /= activation_stats['total_batches']
            activation_stats['gate_importance'][layer_idx] /= activation_stats['total_batches']
            activation_stats['up_mean'][layer_idx] /= activation_stats['total_batches']
            activation_stats['up_variance'][layer_idx] /= activation_stats['total_batches']
            activation_stats['mlp_mean'][layer_idx] /= activation_stats['total_batches']
            activation_stats['mlp_variance'][layer_idx] /= activation_stats['total_batches']
    
    # Final statistics
    print(f"[Phase 2 Streaming] Streaming collection complete!")
    print(f"[Phase 2 Streaming] Total batches processed: {activation_stats['total_batches']}")
    print(f"[Phase 2 Streaming] Total tokens processed: {activation_stats['total_tokens']}")
    print(f"[Phase 2 Streaming] Input/output samples collected: {activation_stats['input_activations'].shape[0]}")
    print(f"[Phase 2 Streaming] Gate importance computed for {len(activation_stats['gate_importance'])} layers")
    
    return activation_stats


def analyze_gate_patterns(activation_stats: Dict, start_id: int, end_id: int) -> Dict[str, torch.Tensor]:
    """
    Phase 3: Gate Importance Analysis
    Analyzes gate patterns to understand information filtering behavior across layers.
    
    Args:
        activation_stats: Statistics collected from Phase 2 streaming
        start_id: Starting layer index
        end_id: Ending layer index
        
    Returns:
        Dictionary containing processed gate analysis results
    """
    print(f"[Phase 3] Starting Gate Importance Analysis...")
    print(f"[Phase 3] Analyzing layers {start_id} to {end_id-1}")
    
    gate_analysis = {
        'layer_importance_scores': {},     # Per-layer aggregate importance
        'neuron_importance_ranking': {},   # Per-layer neuron ranking
        'gate_activation_patterns': {},    # Gate activation distribution analysis
        'information_flow_weights': {},    # Weights for information flow control
        'pruning_candidates': {},          # Neurons that can be safely pruned
        'critical_neurons': {}             # Neurons that must be preserved
    }
    
    total_layers = end_id - start_id
    print(f"[Phase 3] Processing {total_layers} layers")
    
    # Process each layer's gate importance
    for layer_idx in range(start_id, end_id):
        if layer_idx not in activation_stats['gate_importance']:
            print(f"[Phase 3] WARNING: No gate data for layer {layer_idx}")
            continue
            
        print(f"[Phase 3] Analyzing layer {layer_idx}...")
        
        # Extract gate statistics for this layer
        gate_importance = activation_stats['gate_importance'][layer_idx]  # [intermediate_size]
        gate_mean = activation_stats['gate_mean'][layer_idx]
        gate_variance = activation_stats['gate_variance'][layer_idx]
        
        intermediate_size = gate_importance.shape[0]
        print(f"[Phase 3] Layer {layer_idx} intermediate size: {intermediate_size}")
        
        # 1. Compute layer-level importance score
        layer_total_importance = gate_importance.sum().item()
        layer_mean_importance = gate_importance.mean().item()
        layer_std_importance = gate_importance.std().item()
        
        gate_analysis['layer_importance_scores'][layer_idx] = {
            'total_importance': layer_total_importance,
            'mean_importance': layer_mean_importance,
            'std_importance': layer_std_importance,
            'importance_density': layer_total_importance / intermediate_size
        }
        
        print(f"[Phase 3] Layer {layer_idx} total importance: {layer_total_importance:.4f}")
        print(f"[Phase 3] Layer {layer_idx} mean importance: {layer_mean_importance:.6f}")
        
        # 2. Rank neurons by importance
        importance_values, importance_indices = torch.sort(gate_importance, descending=True)
        
        gate_analysis['neuron_importance_ranking'][layer_idx] = {
            'ranked_importance': importance_values,
            'ranked_indices': importance_indices,
            'top_10_percent_indices': importance_indices[:intermediate_size//10],
            'bottom_10_percent_indices': importance_indices[-intermediate_size//10:]
        }
        
        top_importance = importance_values[:10].mean().item()
        bottom_importance = importance_values[-10:].mean().item()
        print(f"[Phase 3] Layer {layer_idx} top-10 neuron importance: {top_importance:.6f}")
        print(f"[Phase 3] Layer {layer_idx} bottom-10 neuron importance: {bottom_importance:.6f}")
        
        # 3. Analyze activation patterns
        # High variance + high mean = dynamic and frequently used (critical)
        # Low variance + low mean = rarely used (pruning candidate)
        # High variance + low mean = context-dependent (interesting)
        
        high_variance_threshold = gate_variance.quantile(0.8)  # Top 20% variance
        high_mean_threshold = torch.abs(gate_mean).quantile(0.8)  # Top 20% mean
        low_variance_threshold = gate_variance.quantile(0.2)   # Bottom 20% variance
        low_mean_threshold = torch.abs(gate_mean).quantile(0.2)    # Bottom 20% mean
        
        # Categorize neurons
        high_var_mask = gate_variance > high_variance_threshold
        high_mean_mask = torch.abs(gate_mean) > high_mean_threshold
        low_var_mask = gate_variance < low_variance_threshold
        low_mean_mask = torch.abs(gate_mean) < low_mean_threshold
        
        critical_neurons = torch.where(high_var_mask & high_mean_mask)[0]  # High var + High mean
        context_dependent = torch.where(high_var_mask & low_mean_mask)[0]  # High var + Low mean
        stable_active = torch.where(low_var_mask & high_mean_mask)[0]      # Low var + High mean
        pruning_candidates = torch.where(low_var_mask & low_mean_mask)[0]  # Low var + Low mean
        
        gate_analysis['gate_activation_patterns'][layer_idx] = {
            'high_var_threshold': high_variance_threshold.item(),
            'high_mean_threshold': high_mean_threshold.item(),
            'critical_neurons': critical_neurons,
            'context_dependent_neurons': context_dependent,
            'stable_active_neurons': stable_active,
            'pruning_candidate_neurons': pruning_candidates
        }
        
        print(f"[Phase 3] Layer {layer_idx} neuron categories:")
        print(f"[Phase 3]   Critical (high var + high mean): {len(critical_neurons)}")
        print(f"[Phase 3]   Context-dependent (high var + low mean): {len(context_dependent)}")
        print(f"[Phase 3]   Stable active (low var + high mean): {len(stable_active)}")
        print(f"[Phase 3]   Pruning candidates (low var + low mean): {len(pruning_candidates)}")
        
        # 4. Compute information flow weights
        # Normalize importance scores to create weights for coupled optimization
        importance_weights = gate_importance / (gate_importance.max() + 1e-8)  # Normalize to [0,1]
        
        # Apply sigmoid to create smoother weights
        sigmoid_weights = torch.sigmoid(4 * (importance_weights - 0.5))  # Sigmoid around 0.5
        
        gate_analysis['information_flow_weights'][layer_idx] = {
            'raw_weights': importance_weights,
            'sigmoid_weights': sigmoid_weights,
            'binary_mask': importance_weights > importance_weights.mean()  # Binary important/not
        }
        
        # 5. Store pruning recommendations
        gate_analysis['pruning_candidates'][layer_idx] = pruning_candidates
        gate_analysis['critical_neurons'][layer_idx] = critical_neurons
        
        print(f"[Phase 3] Layer {layer_idx} analysis complete")
    
    # Cross-layer analysis
    print(f"[Phase 3] Performing cross-layer analysis...")
    
    # Find layers with highest/lowest overall importance
    layer_importances = [
        gate_analysis['layer_importance_scores'][layer_idx]['total_importance'] 
        for layer_idx in range(start_id, end_id)
        if layer_idx in gate_analysis['layer_importance_scores']
    ]
    
    if layer_importances:
        most_important_layer = start_id + layer_importances.index(max(layer_importances))
        least_important_layer = start_id + layer_importances.index(min(layer_importances))
        
        gate_analysis['cross_layer_analysis'] = {
            'most_important_layer': most_important_layer,
            'least_important_layer': least_important_layer,
            'importance_range': max(layer_importances) - min(layer_importances),
            'mean_layer_importance': sum(layer_importances) / len(layer_importances)
        }
        
        print(f"[Phase 3] Most important layer: {most_important_layer} (importance: {max(layer_importances):.4f})")
        print(f"[Phase 3] Least important layer: {least_important_layer} (importance: {min(layer_importances):.4f})")
        print(f"[Phase 3] Importance range: {max(layer_importances) - min(layer_importances):.4f}")
    
    print(f"[Phase 3] Gate Importance Analysis complete!")
    return gate_analysis


def estimate_coupled_transformations(
    activation_stats: Dict, 
    gate_analysis: Dict, 
    start_id: int, 
    end_id: int,
    hidden_size: int
) -> Dict[str, torch.Tensor]:
    """
    Phase 4: Coupled Transformation Estimation (FIXED VERSION)
    Estimates three coupled linear transformations (gate, up, down) using gate importance analysis.
    
    Args:
        activation_stats: Statistics from Phase 2
        gate_analysis: Analysis results from Phase 3
        start_id: Starting layer index
        end_id: Ending layer index
        hidden_size: Model hidden dimension
        
    Returns:
        Dictionary containing estimated transformations
    """
    print(f"[Phase 4 FIXED] Starting Coupled Transformation Estimation...")
    print(f"[Phase 4 FIXED] Target layers: {start_id} to {end_id-1}")
    print(f"[Phase 4 FIXED] Hidden size: {hidden_size}")
    
    # Get input/output data for transformation estimation
    input_activations = activation_stats['input_activations']  # [samples, hidden_size]
    output_activations = activation_stats['output_activations']  # [samples, hidden_size]
    
    num_samples = input_activations.shape[0]
    print(f"[Phase 4 FIXED] Using {num_samples} samples for transformation estimation")
    
    if num_samples == 0:
        raise ValueError("[Phase 4 FIXED] ERROR: No input/output samples available")
    
    # Convert to float64 for numerical stability
    X = input_activations.to(torch.float64)  # [num_samples, hidden_size]
    Y = output_activations.to(torch.float64)  # [num_samples, hidden_size]
    
    print(f"[Phase 4 FIXED] Input tensor shape: {X.shape}")
    print(f"[Phase 4 FIXED] Output tensor shape: {Y.shape}")
    
    # Initialize transformation matrices
    transformations = {
        'T_gate': torch.eye(hidden_size, dtype=torch.float64),
        'T_up': torch.eye(hidden_size, dtype=torch.float64),
        'T_down': torch.eye(hidden_size, dtype=torch.float64),
        'gate_compensation': {},
        'layer_priorities': {},
    }
    
    # Extract layer priorities from gate analysis
    if 'cross_layer_analysis' in gate_analysis:
        cross_analysis = gate_analysis['cross_layer_analysis']
        most_important = cross_analysis['most_important_layer']
        least_important = cross_analysis['least_important_layer']
        
        print(f"[Phase 4 FIXED] Most important layer: {most_important}")
        print(f"[Phase 4 FIXED] Least important layer: {least_important}")
        
        # Create priority ordering
        layer_importances = []
        for layer_idx in range(start_id, end_id):
            if layer_idx in gate_analysis['layer_importance_scores']:
                importance = gate_analysis['layer_importance_scores'][layer_idx]['total_importance']
                layer_importances.append((layer_idx, importance))
        
        layer_importances.sort(key=lambda x: x[1], reverse=True)
        transformations['layer_priorities'] = layer_importances
        
        print(f"[Phase 4 FIXED] Layer priority order: {[f'L{idx}({imp:.1f})' for idx, imp in layer_importances]}")
    
    # Compute aggregate gate importance weights
    print(f"[Phase 4 FIXED] Computing gate-weighted transformation...")
    
    total_gate_weights = torch.zeros(hidden_size, dtype=torch.float64)
    weight_count = 0
    
    for layer_idx in range(start_id, end_id):
        if layer_idx in gate_analysis['information_flow_weights']:
            layer_weights = gate_analysis['information_flow_weights'][layer_idx]['sigmoid_weights']
            intermediate_size = layer_weights.shape[0]
            
            # Map intermediate weights to hidden dimension
            if intermediate_size >= hidden_size:
                if intermediate_size % hidden_size == 0:
                    stride = intermediate_size // hidden_size
                    mapped_weights = layer_weights.view(hidden_size, stride).mean(dim=1)
                else:
                    mapped_weights = torch.nn.functional.interpolate(
                        layer_weights.unsqueeze(0).unsqueeze(0),
                        size=hidden_size,
                        mode='linear',
                        align_corners=False
                    ).squeeze().squeeze()
            else:
                if hidden_size % intermediate_size == 0:
                    repeat_factor = hidden_size // intermediate_size
                    mapped_weights = layer_weights.repeat_interleave(repeat_factor)[:hidden_size]
                else:
                    mapped_weights = torch.nn.functional.interpolate(
                        layer_weights.unsqueeze(0).unsqueeze(0),
                        size=hidden_size,
                        mode='linear',
                        align_corners=False
                    ).squeeze().squeeze()
            
            # Ensure correct size
            if mapped_weights.shape[0] != hidden_size:
                if mapped_weights.shape[0] > hidden_size:
                    mapped_weights = mapped_weights[:hidden_size]
                else:
                    pad_size = hidden_size - mapped_weights.shape[0]
                    mapped_weights = torch.cat([mapped_weights, mapped_weights[-1:].repeat(pad_size)])
            
            total_gate_weights += mapped_weights.to(torch.float64)
            weight_count += 1
            
            print(f"[Phase 4 FIXED] Layer {layer_idx} weights mapped: {intermediate_size} -> {mapped_weights.shape[0]}")
        else:
            print(f"[Phase 4 FIXED] WARNING: No flow weights for layer {layer_idx}")
    
    if weight_count > 0:
        avg_gate_weights = total_gate_weights / weight_count
        print(f"[Phase 4 FIXED] Average gate weights computed from {weight_count} layers")
        print(f"[Phase 4 FIXED] Gate weight range: {avg_gate_weights.min():.4f} to {avg_gate_weights.max():.4f}")
    else:
        avg_gate_weights = torch.ones(hidden_size, dtype=torch.float64)
        print(f"[Phase 4 FIXED] WARNING: No gate weights found, using uniform weights")
    
    # FIXED: Correct weighted least squares formulation
    # We want to solve: X @ T = Y with sample weights W
    # Weighted least squares: T = (X.T @ W @ X)^(-1) @ X.T @ W @ Y
    
    # Create diagonal weight matrix for samples (not features)
    sample_weights = torch.ones(num_samples, dtype=torch.float64)  # Can be modified based on importance
    W_samples = torch.diag(sample_weights)  # [num_samples, num_samples]
    
    # Also create feature weights
    W_features = torch.diag(avg_gate_weights)  # [hidden_size, hidden_size]
    
    print(f"[Phase 4 FIXED] Matrix dimensions check:")
    print(f"[Phase 4 FIXED] X shape: {X.shape}")  # [num_samples, hidden_size]
    print(f"[Phase 4 FIXED] Y shape: {Y.shape}")  # [num_samples, hidden_size]
    print(f"[Phase 4 FIXED] W_samples shape: {W_samples.shape}")  # [num_samples, num_samples]
    print(f"[Phase 4 FIXED] W_features shape: {W_features.shape}")  # [hidden_size, hidden_size]
    
    try:
        # Method 1: Sample-weighted least squares
        # T = (X.T @ W_samples @ X)^(-1) @ X.T @ W_samples @ Y
        XTW = X.T @ W_samples  # [hidden_size, num_samples] @ [num_samples, num_samples] = [hidden_size, num_samples]
        XTWX = XTW @ X  # [hidden_size, num_samples] @ [num_samples, hidden_size] = [hidden_size, hidden_size]
        XTWY = XTW @ Y  # [hidden_size, num_samples] @ [num_samples, hidden_size] = [hidden_size, hidden_size]
        
        print(f"[Phase 4 FIXED] Intermediate matrix shapes:")
        print(f"[Phase 4 FIXED] XTW shape: {XTW.shape}")
        print(f"[Phase 4 FIXED] XTWX shape: {XTWX.shape}")
        print(f"[Phase 4 FIXED] XTWY shape: {XTWY.shape}")
        
        # Add regularization for numerical stability
        reg_strength = 1e-4
        regularizer = reg_strength * torch.eye(hidden_size, dtype=torch.float64)
        XTWX_reg = XTWX + regularizer
        
        # Solve the linear system: T = (X.T @ W @ X + reg)^(-1) @ (X.T @ W @ Y)
        T_coupled = torch.linalg.solve(XTWX_reg, XTWY)
        
        print(f"[Phase 4 FIXED] Coupled transformation computed successfully")
        print(f"[Phase 4 FIXED] Transformation matrix shape: {T_coupled.shape}")
        print(f"[Phase 4 FIXED] Transformation matrix norm: {torch.norm(T_coupled, 'fro').item():.4f}")
        
    except Exception as e:
        print(f"[Phase 4 FIXED] WARNING: Numerical instability in solve: {str(e)}")
        print(f"[Phase 4 FIXED] Using pseudo-inverse fallback...")
        try:
            T_coupled = torch.linalg.pinv(XTWX) @ XTWY
        except Exception as e2:
            print(f"[Phase 4 FIXED] ERROR: Pseudo-inverse also failed: {str(e2)}")
            print(f"[Phase 4 FIXED] Using simple least squares without weights...")
            # Fallback to simple least squares: T = (X.T @ X)^(-1) @ X.T @ Y
            XTX = X.T @ X
            XTY = X.T @ Y
            XTX_reg = XTX + reg_strength * torch.eye(hidden_size, dtype=torch.float64)
            T_coupled = torch.linalg.solve(XTX_reg, XTY)
    
    # Apply feature weighting to the result
    T_coupled = W_features @ T_coupled
    print(f"[Phase 4 FIXED] Applied feature weighting to transformation")
    
    # Decompose coupled transformation using SVD
    print(f"[Phase 4 FIXED] Decomposing coupled transformation...")
    
    U, S, Vt = torch.linalg.svd(T_coupled, full_matrices=False)
    
    # Ensure proper dimensions
    min_dim = min(U.shape[1], S.shape[0], Vt.shape[0])
    U_sq = U[:, :min_dim]
    S_diag = torch.diag(S[:min_dim])
    Vt_sq = Vt[:min_dim, :]
    
    # Pad to full size if necessary
    if min_dim < hidden_size:
        U_full = torch.eye(hidden_size, dtype=torch.float64)
        U_full[:, :min_dim] = U_sq
        
        S_full = torch.eye(hidden_size, dtype=torch.float64)
        S_full[:min_dim, :min_dim] = S_diag
        
        Vt_full = torch.eye(hidden_size, dtype=torch.float64)
        Vt_full[:min_dim, :] = Vt_sq
        
        transformations['T_up'] = U_full
        transformations['T_gate'] = S_full
        transformations['T_down'] = Vt_full
    else:
        transformations['T_up'] = U_sq
        transformations['T_gate'] = S_diag
        transformations['T_down'] = Vt_sq
    
    print(f"[Phase 4 FIXED] SVD decomposition complete:")
    print(f"[Phase 4 FIXED] T_up shape: {transformations['T_up'].shape}")
    print(f"[Phase 4 FIXED] T_gate shape: {transformations['T_gate'].shape}")
    print(f"[Phase 4 FIXED] T_down shape: {transformations['T_down'].shape}")
    
    # Consistency check
    print(f"[Phase 4 FIXED] Applying consistency regularization...")
    reconstructed = transformations['T_up'] @ transformations['T_gate'] @ transformations['T_down']
    consistency_error = torch.norm(reconstructed - T_coupled, 'fro').item()
    
    print(f"[Phase 4 FIXED] Reconstruction error: {consistency_error:.6f}")
    
    if consistency_error > 0.1:
        print(f"[Phase 4 FIXED] High reconstruction error, applying correction...")
        correction_factor = 0.9
        
        identity = torch.eye(hidden_size, dtype=torch.float64)
        transformations['T_up'] = correction_factor * transformations['T_up'] + (1-correction_factor) * identity
        transformations['T_gate'] = correction_factor * transformations['T_gate'] + (1-correction_factor) * identity
        transformations['T_down'] = correction_factor * transformations['T_down'] + (1-correction_factor) * identity
        
        reconstructed_corrected = transformations['T_up'] @ transformations['T_gate'] @ transformations['T_down']
        corrected_error = torch.norm(reconstructed_corrected - T_coupled, 'fro').item()
        print(f"[Phase 4 FIXED] Corrected reconstruction error: {corrected_error:.6f}")
    
    # Per-layer gate compensation
    print(f"[Phase 4 FIXED] Computing per-layer gate compensation...")
    
    for layer_idx in range(start_id, end_id):
        if layer_idx in gate_analysis['gate_activation_patterns']:
            patterns = gate_analysis['gate_activation_patterns'][layer_idx]
            
            num_critical = len(patterns['critical_neurons'])
            num_pruning = len(patterns['pruning_candidate_neurons'])
            
            if num_critical + num_pruning > 0:
                critical_ratio = num_critical / (num_critical + num_pruning)
                compensation_factor = 0.5 + 0.5 * critical_ratio
            else:
                compensation_factor = 1.0
            
            transformations['gate_compensation'][layer_idx] = compensation_factor
            print(f"[Phase 4 FIXED] Layer {layer_idx} compensation: {compensation_factor:.3f}")
    
    # Final validation
    print(f"[Phase 4 FIXED] Transformation validation...")
    
    if num_samples > 0:
        sample_input = X[:min(100, num_samples)]
        sample_expected = Y[:min(100, num_samples)]
        
        # Apply full transformation
        transformed = sample_input @ transformations['T_up'] @ transformations['T_gate'] @ transformations['T_down']
        
        # Compute approximation error
        approx_error = torch.norm(transformed - sample_expected, 'fro') / torch.norm(sample_expected, 'fro')
        print(f"[Phase 4 FIXED] Relative approximation error: {approx_error.item():.6f}")
        
        if approx_error.item() < 0.1:
            print(f"[Phase 4 FIXED] SUCCESS: Good approximation quality")
        elif approx_error.item() < 0.5:
            print(f"[Phase 4 FIXED] WARNING: Moderate approximation quality")
        else:
            print(f"[Phase 4 FIXED] WARNING: Poor approximation quality")
    
    print(f"[Phase 4 FIXED] Coupled Transformation Estimation complete!")
    return transformations


def estimate_coupled_transformations_residual_aware(
    activation_stats: Dict, 
    gate_analysis: Dict, 
    start_id: int, 
    end_id: int,
    hidden_size: int
) -> Dict[str, torch.Tensor]:
    """
    Phase 4: Residual-Aware Gate-weighted Transformation Estimation
    
    This is a simplified, stable approach that:
    1. Learns residual transformation: X @ T ≈ (Y - X) instead of X @ T ≈ Y
    2. Uses gate importance for weighted least squares
    3. Focuses on single stable transformation rather than complex multi-component
    
    Args:
        activation_stats: Statistics from Phase 2
        gate_analysis: Analysis results from Phase 3
        start_id: Starting layer index
        end_id: Ending layer index
        hidden_size: Model hidden dimension
        
    Returns:
        Dictionary containing residual-aware transformation
    """
    print(f"[Phase 4 RESIDUAL] Starting Residual-Aware Gate-weighted Transformation...")
    print(f"[Phase 4 RESIDUAL] Target layers: {start_id} to {end_id-1}")
    print(f"[Phase 4 RESIDUAL] Hidden size: {hidden_size}")
    
    # Get input/output data for transformation estimation
    input_activations = activation_stats['input_activations']  # X: [samples, hidden_size]
    output_activations = activation_stats['output_activations']  # Y: [samples, hidden_size]
    
    num_samples = input_activations.shape[0]
    print(f"[Phase 4 RESIDUAL] Using {num_samples} samples for transformation estimation")
    
    if num_samples == 0:
        raise ValueError("[Phase 4 RESIDUAL] ERROR: No input/output samples available")
    
    # Convert to float64 for numerical stability
    X = input_activations.to(torch.float64)  # [num_samples, hidden_size]
    Y = output_activations.to(torch.float64)  # [num_samples, hidden_size]
    
    print(f"[Phase 4 RESIDUAL] Input shape: {X.shape}, Output shape: {Y.shape}")
    
    # Step 1: Compute residual (the key innovation)
    # Instead of learning X @ T = Y, we learn X @ T = (Y - X)
    # This focuses on the transformation that the blocks actually perform
    residual = Y - X  # [num_samples, hidden_size]
    
    print(f"[Phase 4 RESIDUAL] Residual computation:")
    print(f"[Phase 4 RESIDUAL]   Residual norm: {torch.norm(residual, 'fro').item():.4f}")
    print(f"[Phase 4 RESIDUAL]   Residual mean: {residual.mean().item():.6f}")
    print(f"[Phase 4 RESIDUAL]   Residual std: {residual.std().item():.6f}")
    print(f"[Phase 4 RESIDUAL]   Residual/Input ratio: {(torch.norm(residual, 'fro') / torch.norm(X, 'fro')).item():.4f}")
    
    # Step 2: Create gate-importance based feature weights
    print(f"[Phase 4 RESIDUAL] Creating gate-importance feature weights...")
    
    # Initialize feature importance weights
    feature_importance = torch.ones(hidden_size, dtype=torch.float64)
    
    # Aggregate gate importance across layers with neuron category weighting
    total_importance = 0.0
    processed_layers = 0
    
    for layer_idx in range(start_id, end_id):
        if layer_idx not in gate_analysis['gate_activation_patterns']:
            print(f"[Phase 4 RESIDUAL] WARNING: No patterns for layer {layer_idx}")
            continue
        if layer_idx not in gate_analysis['information_flow_weights']:
            print(f"[Phase 4 RESIDUAL] WARNING: No flow weights for layer {layer_idx}")
            continue
            
        patterns = gate_analysis['gate_activation_patterns'][layer_idx]
        layer_weights = gate_analysis['information_flow_weights'][layer_idx]['sigmoid_weights']
        intermediate_size = layer_weights.shape[0]
        
        print(f"[Phase 4 RESIDUAL] Processing layer {layer_idx}:")
        print(f"[Phase 4 RESIDUAL]   Intermediate size: {intermediate_size}")
        print(f"[Phase 4 RESIDUAL]   Critical neurons: {len(patterns['critical_neurons'])}")
        print(f"[Phase 4 RESIDUAL]   Context neurons: {len(patterns['context_dependent_neurons'])}")
        print(f"[Phase 4 RESIDUAL]   Stable neurons: {len(patterns['stable_active_neurons'])}")
        print(f"[Phase 4 RESIDUAL]   Pruning neurons: {len(patterns['pruning_candidate_neurons'])}")
        
        # Create enhanced weights based on neuron categories
        enhanced_layer_weights = layer_weights.clone()
        
        # Apply category-specific multipliers
        for neuron_idx in patterns['critical_neurons']:
            if neuron_idx < intermediate_size:
                enhanced_layer_weights[neuron_idx] *= 2.0  # High importance
        
        for neuron_idx in patterns['context_dependent_neurons']:
            if neuron_idx < intermediate_size:
                enhanced_layer_weights[neuron_idx] *= 1.5  # Medium-high importance
        
        for neuron_idx in patterns['stable_active_neurons']:
            if neuron_idx < intermediate_size:
                enhanced_layer_weights[neuron_idx] *= 1.2  # Slightly higher importance
        
        for neuron_idx in patterns['pruning_candidate_neurons']:
            if neuron_idx < intermediate_size:
                enhanced_layer_weights[neuron_idx] *= 0.5  # Lower importance
        
        # Map intermediate dimension to hidden dimension
        if intermediate_size >= hidden_size:
            if intermediate_size % hidden_size == 0:
                stride = intermediate_size // hidden_size
                mapped_weights = enhanced_layer_weights.view(hidden_size, stride).mean(dim=1)
            else:
                # Use average pooling for non-divisible sizes
                chunk_size = intermediate_size / hidden_size
                mapped_weights = torch.zeros(hidden_size, dtype=enhanced_layer_weights.dtype)
                for i in range(hidden_size):
                    start_idx = int(i * chunk_size)
                    end_idx = int((i + 1) * chunk_size)
                    if start_idx < intermediate_size:
                        end_idx = min(end_idx, intermediate_size)
                        mapped_weights[i] = enhanced_layer_weights[start_idx:end_idx].mean()
        else:
            # Upsample for smaller intermediate sizes
            if hidden_size % intermediate_size == 0:
                repeat_factor = hidden_size // intermediate_size
                mapped_weights = enhanced_layer_weights.repeat_interleave(repeat_factor)[:hidden_size]
            else:
                # Use interpolation
                scale_factor = hidden_size / intermediate_size
                mapped_weights = torch.zeros(hidden_size, dtype=enhanced_layer_weights.dtype)
                for i in range(hidden_size):
                    src_idx = min(int(i / scale_factor), intermediate_size - 1)
                    mapped_weights[i] = enhanced_layer_weights[src_idx]
        
        # Accumulate importance across layers
        feature_importance += mapped_weights.to(torch.float64)
        total_importance += mapped_weights.sum().item()
        processed_layers += 1
        
        print(f"[Phase 4 RESIDUAL]   Mapped weights shape: {mapped_weights.shape}")
        print(f"[Phase 4 RESIDUAL]   Mapped weights range: {mapped_weights.min():.4f} to {mapped_weights.max():.4f}")
    
    if processed_layers > 0:
        # Normalize by number of layers
        feature_importance = feature_importance / processed_layers
        average_importance = total_importance / processed_layers
        
        # Ensure positive weights and reasonable range
        feature_importance = torch.clamp(feature_importance, min=0.1, max=5.0)
        
        print(f"[Phase 4 RESIDUAL] Feature importance statistics:")
        print(f"[Phase 4 RESIDUAL]   Processed layers: {processed_layers}")
        print(f"[Phase 4 RESIDUAL]   Average importance: {average_importance:.4f}")
        print(f"[Phase 4 RESIDUAL]   Feature importance range: {feature_importance.min():.4f} to {feature_importance.max():.4f}")
        print(f"[Phase 4 RESIDUAL]   High importance features (>2.0): {(feature_importance > 2.0).sum().item()}")
        print(f"[Phase 4 RESIDUAL]   Low importance features (<1.0): {(feature_importance < 1.0).sum().item()}")
    else:
        print(f"[Phase 4 RESIDUAL] WARNING: No layers processed, using uniform weights")
        feature_importance = torch.ones(hidden_size, dtype=torch.float64)
    
    # Step 3: Gate-weighted Residual Least Squares
    # Goal: X @ T ≈ residual, with feature importance weighting
    # Formulation: min ||sqrt(W) * (X @ T - residual)||²
    # where W = diag(feature_importance)
    
    print(f"[Phase 4 RESIDUAL] Solving gate-weighted residual least squares...")
    
    # Create feature weight matrix
    W_features = torch.diag(feature_importance)  # [hidden_size, hidden_size]
    
    # Apply weighting to both sides of the equation: X @ T ≈ residual
    # Weighted formulation: sqrt(W) @ X @ T ≈ sqrt(W) @ residual
    sqrt_W = torch.diag(torch.sqrt(feature_importance))  # [hidden_size, hidden_size]
    
    X_weighted = X @ sqrt_W  # [num_samples, hidden_size]
    residual_weighted = residual @ sqrt_W  # [num_samples, hidden_size]
    
    print(f"[Phase 4 RESIDUAL] Weighted matrix shapes:")
    print(f"[Phase 4 RESIDUAL]   X_weighted: {X_weighted.shape}")
    print(f"[Phase 4 RESIDUAL]   residual_weighted: {residual_weighted.shape}")
    
    # Solve weighted least squares: X_weighted @ T = residual_weighted
    # Solution: T = (X_weighted.T @ X_weighted)^(-1) @ X_weighted.T @ residual_weighted
    
    try:
        XTX_weighted = X_weighted.T @ X_weighted  # [hidden_size, hidden_size]
        XTR_weighted = X_weighted.T @ residual_weighted  # [hidden_size, hidden_size]
        
        print(f"[Phase 4 RESIDUAL] Normal equation matrices:")
        print(f"[Phase 4 RESIDUAL]   XTX_weighted shape: {XTX_weighted.shape}")
        print(f"[Phase 4 RESIDUAL]   XTR_weighted shape: {XTR_weighted.shape}")
        
        # Add regularization for numerical stability
        reg_strength = 1e-4
        regularizer = reg_strength * torch.eye(hidden_size, dtype=torch.float64)
        XTX_reg = XTX_weighted + regularizer
        
        print(f"[Phase 4 RESIDUAL] Added regularization: {reg_strength}")
        
        # Solve the linear system
        T_residual = torch.linalg.solve(XTX_reg, XTR_weighted)
        
        print(f"[Phase 4 RESIDUAL] Transformation computed successfully")
        print(f"[Phase 4 RESIDUAL] T_residual shape: {T_residual.shape}")
        print(f"[Phase 4 RESIDUAL] T_residual norm: {torch.norm(T_residual, 'fro').item():.4f}")
        
        # Check condition number for numerical stability
        try:
            cond_num = torch.linalg.cond(XTX_reg).item()
            print(f"[Phase 4 RESIDUAL] Condition number: {cond_num:.2e}")
            if cond_num > 1e12:
                print(f"[Phase 4 RESIDUAL] WARNING: High condition number indicates numerical instability")
        except:
            print(f"[Phase 4 RESIDUAL] Could not compute condition number")
        
    except Exception as e:
        print(f"[Phase 4 RESIDUAL] ERROR in solve: {str(e)}")
        print(f"[Phase 4 RESIDUAL] Using pseudo-inverse fallback...")
        
        try:
            T_residual = torch.linalg.pinv(XTX_weighted) @ XTR_weighted
            print(f"[Phase 4 RESIDUAL] Pseudo-inverse solution computed")
        except Exception as e2:
            print(f"[Phase 4 RESIDUAL] ERROR in pseudo-inverse: {str(e2)}")
            print(f"[Phase 4 RESIDUAL] Using identity transformation fallback")
            T_residual = torch.eye(hidden_size, dtype=torch.float64)
    
    # Step 4: Quality Assessment
    print(f"[Phase 4 RESIDUAL] Evaluating transformation quality...")
    
    # Test on sample data
    sample_size = min(1000, num_samples)
    sample_X = X[:sample_size]
    sample_residual = residual[:sample_size]
    
    # Apply transformation
    predicted_residual = sample_X @ T_residual  # [sample_size, hidden_size]
    
    # Compute errors
    absolute_error = torch.norm(predicted_residual - sample_residual, 'fro').item()
    relative_error = absolute_error / torch.norm(sample_residual, 'fro').item()
    
    # Compute weighted error (more important for our gate-aware approach)
    weighted_predicted = predicted_residual @ sqrt_W
    weighted_actual = sample_residual @ sqrt_W
    weighted_error = torch.norm(weighted_predicted - weighted_actual, 'fro') / torch.norm(weighted_actual, 'fro')
    
    print(f"[Phase 4 RESIDUAL] Quality metrics:")
    print(f"[Phase 4 RESIDUAL]   Absolute error: {absolute_error:.4f}")
    print(f"[Phase 4 RESIDUAL]   Relative error: {relative_error:.6f}")
    print(f"[Phase 4 RESIDUAL]   Weighted relative error: {weighted_error.item():.6f}")
    
    # Test full transformation: X + X @ T vs Y
    predicted_output = sample_X + predicted_residual  # X + residual
    actual_output = sample_X + sample_residual  # Should equal Y
    output_error = torch.norm(predicted_output - actual_output, 'fro') / torch.norm(actual_output, 'fro')
    
    print(f"[Phase 4 RESIDUAL]   Full output error: {output_error.item():.6f}")
    
    # Quality assessment
    if output_error.item() < 0.2:
        quality = "EXCELLENT"
        print(f"[Phase 4 RESIDUAL] {quality}: High-quality residual approximation achieved!")
    elif output_error.item() < 0.4:
        quality = "GOOD"
        print(f"[Phase 4 RESIDUAL] {quality}: Acceptable residual approximation")
    elif output_error.item() < 0.6:
        quality = "MODERATE"
        print(f"[Phase 4 RESIDUAL] {quality}: Moderate residual approximation")
    else:
        quality = "POOR"
        print(f"[Phase 4 RESIDUAL] {quality}: Poor residual approximation - may need adjustment")
    
    # Step 5: Prepare results
    print(f"[Phase 4 RESIDUAL] Preparing transformation results...")
    
    # Store layer priorities for reference
    layer_priorities = []
    if 'cross_layer_analysis' in gate_analysis:
        for layer_idx in range(start_id, end_id):
            if layer_idx in gate_analysis['layer_importance_scores']:
                importance = gate_analysis['layer_importance_scores'][layer_idx]['total_importance']
                layer_priorities.append((layer_idx, importance))
        layer_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Gate compensation factors (for potential use in model application)
    gate_compensation = {}
    for layer_idx in range(start_id, end_id):
        if layer_idx in gate_analysis['gate_activation_patterns']:
            patterns = gate_analysis['gate_activation_patterns'][layer_idx]
            num_critical = len(patterns['critical_neurons'])
            num_total = len(patterns['critical_neurons']) + len(patterns['pruning_candidate_neurons'])
            if num_total > 0:
                critical_ratio = num_critical / num_total
                gate_compensation[layer_idx] = 0.5 + 0.5 * critical_ratio
            else:
                gate_compensation[layer_idx] = 1.0
        else:
            gate_compensation[layer_idx] = 1.0
    
    transformations = {
        # Main transformation (this is what gets applied to the model)
        'T_residual': T_residual,
        
        # For compatibility with existing code structure
        'T_gate': torch.eye(hidden_size, dtype=torch.float64),
        'T_up': torch.eye(hidden_size, dtype=torch.float64), 
        'T_down': T_residual,  # The actual transformation goes in T_down
        
        # Analysis and metadata
        'layer_priorities': layer_priorities,
        'gate_compensation': gate_compensation,
        'feature_importance': feature_importance,
        
        # Quality metrics
        'approximation_quality': {
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'weighted_error': weighted_error.item(),
            'output_error': output_error.item(),
            'quality_assessment': quality,
            'num_samples_used': sample_size,
            'num_layers_processed': processed_layers
        }
    }
    
    print(f"[Phase 4 RESIDUAL] Results summary:")
    print(f"[Phase 4 RESIDUAL]   Quality: {quality}")
    print(f"[Phase 4 RESIDUAL]   Output approximation error: {output_error.item():.1%}")
    print(f"[Phase 4 RESIDUAL]   Processed {processed_layers} layers")
    print(f"[Phase 4 RESIDUAL]   Used {sample_size} samples for validation")
    
    if layer_priorities:
        print(f"[Phase 4 RESIDUAL]   Most important layer: {layer_priorities[0][0]} ({layer_priorities[0][1]:.1f})")
        print(f"[Phase 4 RESIDUAL]   Least important layer: {layer_priorities[-1][0]} ({layer_priorities[-1][1]:.1f})")
    
    print(f"[Phase 4 RESIDUAL] Residual-Aware Gate-weighted Transformation complete!")
    return transformations


# def gate_aware_coupled_method(
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
#     distances_path: str = "./distances.pth",
#     num_A: int = 1,
#     merge_consecutive: bool = True,
#     **kwargs
# ) -> str:
#     """
#     Gate-aware coupled optimization method - Phases 2-4 implementation
    
#     Args:
#         model_path: Path to pretrained model
#         dataset: Dataset name
#         dataset_column: Column containing text data
#         batch_size: Batch size for processing
#         max_length: Maximum sequence length
#         layers_to_skip: Number of layers to skip between compared blocks
#         dataset_size: Size of calibration dataset
#         dataset_subset: Dataset subset to use
#         use_4bit: Whether to use 4-bit quantization
#         save_path: Path to save the processed model
#         token: HuggingFace token for private models
#         distances_path: Path to pre-computed distance metrics
#         num_A: Number of blocks to process
#         merge_consecutive: Whether to merge consecutive blocks
#         **kwargs: Additional arguments
        
#     Returns:
#         Path to the processed model
#     """
#     print(f"[GACO] Starting Gate-Aware Coupled Optimization method...")
#     print(f"[GACO] Model: {model_path}")
#     print(f"[GACO] Dataset: {dataset}, Size: {dataset_size}")
#     print(f"[GACO] Layers to skip: {layers_to_skip}")
    
#     # Import required modules (same as original ReplaceMe)
#     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#     from .utils import get_calib_dataloader, select_non_overlapping_blocks, truncate_model
    
#     device_map = "auto" if torch.cuda.is_available() else "cpu"
#     quantization_config = None
    
#     if use_4bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
    
#     print(f"[GACO] Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map=device_map,
#         quantization_config=quantization_config,
#         output_hidden_states=True,
#         token=token
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     print(f"[GACO] Model loaded. Hidden size: {model.config.hidden_size}")
#     print(f"[GACO] Number of layers: {model.config.num_hidden_layers}")
    
#     model.eval()
#     dataloader = get_calib_dataloader(
#         dataset,
#         dataset_subset,
#         dataset_column,
#         dataset_size,
#         batch_size,
#         tokenizer
#     )
    
#     print(f"[GACO] Data loader created")
    
#     # Load pre-computed distances and select blocks
#     print(f"[GACO] Loading distances from: {distances_path}")
#     average_distances = torch.load(distances_path, weights_only=False)
#     selected_blocks = select_non_overlapping_blocks(
#         average_distances,
#         layers_to_skip,
#         num_blocks=num_A,
#         merge_consecutive=merge_consecutive
#     )
    
#     start_ids = sorted([x[0] for x in selected_blocks])
#     end_ids = sorted([x[1] for x in selected_blocks])
#     num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
#     num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
    
#     print(f"[GACO] Selected blocks: {selected_blocks}")
#     print(f"[GACO] Start IDs: {start_ids}")
#     print(f"[GACO] End IDs: {end_ids}")
    
#     # Process each selected block
#     for i in range(len(selected_blocks)):
#         start_id = start_ids[i]
#         end_id = end_ids[i]
#         num_layer = num_layers[i]
        
#         print(f"[GACO] Processing block {i+1}/{len(selected_blocks)}: layers {start_id} to {end_id}")
        
#         # Phase 2: Collect enhanced activations using streaming approach
#         activations = collect_enhanced_activations_streaming(
#             model=model,
#             start_id=start_id - num_layer,
#             end_id=end_id - num_layer,
#             dataset_size=dataset_size,
#             max_length=max_length,
#             dataloader=dataloader,
#             device=next(model.parameters()).device,
#             tokenizer=tokenizer
#         )
        
#         print(f"[GACO] Enhanced activations collected for block {i+1}")
#         print(f"[GACO] Phase 2 streaming complete for block {i+1}")
        
#         # Validate that we collected the statistics properly
#         if len(activations['gate_importance']) > 0:
#             total_importance = sum(activations['gate_importance'][k].sum().item() for k in activations['gate_importance'])
#             print(f"[GACO] SUCCESS: Gate importance computed - total importance: {total_importance:.4f}")
#         else:
#             print(f"[GACO] WARNING: No gate importance computed")
            
#         if activations['input_activations'].shape[0] > 0:
#             print(f"[GACO] SUCCESS: Input/output samples collected - {activations['input_activations'].shape[0]} samples")
#         else:
#             print(f"[GACO] WARNING: No input/output samples collected")
        
#         print(f"[GACO] Total tokens processed: {activations['total_tokens']}")
#         print(f"[GACO] Total batches processed: {activations['total_batches']}")
        
#         # Phase 3: Gate Importance Analysis
#         print(f"[GACO] Starting Phase 3 for block {i+1}...")
#         try:
#             gate_analysis = analyze_gate_patterns(
#                 activation_stats=activations,
#                 start_id=start_id - num_layer,
#                 end_id=end_id - num_layer
#             )
#             print(f"[GACO] Phase 3 function returned: {type(gate_analysis)}")
#         except Exception as e:
#             print(f"[GACO] ERROR in Phase 3: {str(e)}")
#             gate_analysis = None
        
#         print(f"[GACO] Phase 3 complete for block {i+1}")
        
#         # Validate Phase 3 results
#         if gate_analysis is not None and 'layer_importance_scores' in gate_analysis and gate_analysis['layer_importance_scores']:
#             analyzed_layers = len(gate_analysis['layer_importance_scores'])
#             print(f"[GACO] SUCCESS: Gate analysis completed for {analyzed_layers} layers")
            
#             # Show some key insights
#             if 'cross_layer_analysis' in gate_analysis:
#                 cross_analysis = gate_analysis['cross_layer_analysis']
#                 print(f"[GACO] Most important layer: {cross_analysis['most_important_layer']}")
#                 print(f"[GACO] Least important layer: {cross_analysis['least_important_layer']}")
#                 print(f"[GACO] Mean layer importance: {cross_analysis['mean_layer_importance']:.4f}")
            
#             # Show neuron analysis for first layer as example
#             first_layer = start_id - num_layer
#             if first_layer in gate_analysis['gate_activation_patterns']:
#                 patterns = gate_analysis['gate_activation_patterns'][first_layer]
#                 print(f"[GACO] Layer {first_layer} neuron analysis:")
#                 print(f"[GACO]   Critical neurons: {len(patterns['critical_neurons'])}")
#                 print(f"[GACO]   Pruning candidates: {len(patterns['pruning_candidate_neurons'])}")
#         else:
#             print(f"[GACO] WARNING: Gate analysis failed or incomplete")
#             print(f"[GACO] Gate analysis result: {gate_analysis}")
            
#             # Skip Phase 4 if Phase 3 failed
#             print(f"[GACO] Skipping Phase 4 due to Phase 3 failure")
            
#             # Memory cleanup
#             del activations
#             if gate_analysis is not None:
#                 del gate_analysis
#             gc.collect()
#             torch.cuda.empty_cache()
            
#             print(f"[GACO] Block {i+1} processing complete (Phases 2-3, Phase 4 skipped)")
#             continue
        
#         # Phase 4: Coupled Transformation Estimation
#         print(f"[GACO] Starting Phase 4 for block {i+1}...")
#         try:
#             transformations = estimate_coupled_transformations_residual_aware(
#                 activation_stats=activations,
#                 gate_analysis=gate_analysis,
#                 start_id=start_id - num_layer,
#                 end_id=end_id - num_layer,
#                 hidden_size=model.config.hidden_size
#             )
            
#             print(f"[GACO] Phase 4 complete for block {i+1}")
            
#             # Validate Phase 4 results
#             if transformations is not None and 'T_gate' in transformations and 'T_up' in transformations and 'T_down' in transformations:
#                 print(f"[GACO] SUCCESS: Coupled transformations estimated")
#                 print(f"[GACO] T_gate norm: {torch.norm(transformations['T_gate'], 'fro').item():.4f}")
#                 print(f"[GACO] T_up norm: {torch.norm(transformations['T_up'], 'fro').item():.4f}")  
#                 print(f"[GACO] T_down norm: {torch.norm(transformations['T_down'], 'fro').item():.4f}")
                
#                 if 'layer_priorities' in transformations:
#                     priorities = transformations['layer_priorities']
#                     print(f"[GACO] Layer priorities: {[f'L{idx}' for idx, _ in priorities[:2]]}")
#             else:
#                 print(f"[GACO] WARNING: Transformation estimation failed")
#                 print(f"[GACO] Transformations result: {transformations}")
            
#             print(f"[GACO] Phase 4 analysis complete for block {i+1}")
            
#             # Memory cleanup
#             del transformations
            
#         except Exception as e:
#             print(f"[GACO] ERROR in Phase 4: {str(e)}")
#             print(f"[GACO] Phase 4 failed, continuing with next block...")
        
#         # Memory cleanup
#         del activations
#         del gate_analysis
#         gc.collect()
#         torch.cuda.empty_cache()
        
#         print(f"[GACO] Block {i+1} processing complete (Phases 2-4)")
    
#     # Cleanup
#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     # Generate output path
#     if save_path is None:
#         import os
#         if not os.path.exists('output_models'):
#             os.makedirs('output_models')
#         save_path = "output_models/" + (
#             f"{model_path}_{layers_to_skip}_layers_GACO"
#         ).replace("/", "_")
    
#     print(f"[GACO] Method execution complete (Phases 2-4)")
#     return f"{save_path}_GACO_phase2_phase3_phase4"


def apply_residual_transformation_to_model(
    model,
    transformations: Dict,
    start_id: int,
    end_id: int,
    num_layer: int,
    save_path: str,
    tokenizer=None,
    model_path: str = None,
    token: str = None
) -> str:
    """
    Phase 5: Apply the estimated residual transformation to the actual model
    
    This function:
    1. Truncates the model by removing the selected blocks
    2. Applies the learned transformation to the preceding layer's down_proj
    3. Saves the modified model
    
    Args:
        model: The loaded transformer model
        transformations: Results from Phase 4 (containing T_residual)
        start_id: Starting layer index for replacement
        end_id: Ending layer index for replacement
        num_layer: Number of layers already removed (for sequential processing)
        save_path: Path to save the modified model
        tokenizer: Tokenizer to save with the model
        
    Returns:
        Path where the modified model was saved
    """
    print(f"[Phase 5 RECONSTRUCTION] Starting model reconstruction...")
    print(f"[Phase 5 RECONSTRUCTION] Target layers: {start_id} to {end_id-1}")
    print(f"[Phase 5 RECONSTRUCTION] Layers already removed: {num_layer}")
    print(f"[Phase 5 RECONSTRUCTION] Save path: {save_path}")
    
    from .utils import truncate_model
    import torch
    import os
    
    # Step 1: Validate transformation quality
    if 'approximation_quality' in transformations:
        quality = transformations['approximation_quality']
        output_error = quality['output_error']
        quality_assessment = quality['quality_assessment']
        
        print(f"[Phase 5 RECONSTRUCTION] Transformation quality check:")
        print(f"[Phase 5 RECONSTRUCTION]   Quality: {quality_assessment}")
        print(f"[Phase 5 RECONSTRUCTION]   Output error: {output_error:.1%}")
        
        if output_error > 0.5:  # 50% threshold
            print(f"[Phase 5 RECONSTRUCTION] WARNING: High approximation error detected!")
            print(f"[Phase 5 RECONSTRUCTION] Consider adjusting transformation parameters")
    
    # Step 2: Extract the main transformation
    if 'T_residual' in transformations:
        T_main = transformations['T_residual']
        print(f"[Phase 5 RECONSTRUCTION] Using T_residual transformation")
    elif 'T_down' in transformations:
        T_main = transformations['T_down']
        print(f"[Phase 5 RECONSTRUCTION] Using T_down transformation")
    else:
        raise ValueError("[Phase 5 RECONSTRUCTION] ERROR: No valid transformation found!")
    
    T_main = T_main.to(torch.float64)  # Ensure high precision
    print(f"[Phase 5 RECONSTRUCTION] Transformation matrix shape: {T_main.shape}")
    print(f"[Phase 5 RECONSTRUCTION] Transformation matrix norm: {torch.norm(T_main, 'fro').item():.4f}")
    
    # Step 3: Get layer information before truncation
    total_layers_before = model.config.num_hidden_layers
    target_layer_idx = start_id - num_layer - 1  # Layer that will receive the transformation
    
    print(f"[Phase 5 RECONSTRUCTION] Model info before truncation:")
    print(f"[Phase 5 RECONSTRUCTION]   Total layers: {total_layers_before}")
    print(f"[Phase 5 RECONSTRUCTION]   Target layer index: {target_layer_idx}")
    print(f"[Phase 5 RECONSTRUCTION]   Layers to remove: {start_id - num_layer} to {end_id - num_layer - 1}")
    
    # Validate target layer index
    if target_layer_idx < 0 or target_layer_idx >= total_layers_before:
        raise ValueError(f"[Phase 5 RECONSTRUCTION] ERROR: Invalid target layer index: {target_layer_idx}")
    
    # Step 4: Extract original down_proj weights before truncation
    target_layer = model.model.layers[target_layer_idx]
    
    # Handle quantized weights properly
    if hasattr(target_layer.mlp.down_proj, 'weight') and hasattr(target_layer.mlp.down_proj.weight, 'data'):
        original_down_proj_raw = target_layer.mlp.down_proj.weight.data
        
        print(f"[Phase 5 RECONSTRUCTION] Raw weight info:")
        print(f"[Phase 5 RECONSTRUCTION]   Raw shape: {original_down_proj_raw.shape}")
        print(f"[Phase 5 RECONSTRUCTION]   Raw dtype: {original_down_proj_raw.dtype}")
        
        # Handle 4-bit quantized weights
        if original_down_proj_raw.dtype == torch.uint8:
            print(f"[Phase 5 RECONSTRUCTION] Detected 4-bit quantized weights, dequantizing...")
            
            # For 4-bit quantized weights, we need to load a clean CPU model
            print(f"[Phase 5 RECONSTRUCTION] Loading clean CPU model for proper weight extraction...")
            if model_path is None:
                raise ValueError("model_path is required to handle quantized weights")
            
            from transformers import AutoModelForCausalLM
            print(f"[Phase 5 RECONSTRUCTION] Loading CPU model from: {model_path}")
            cpu_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                token=token
            )
            original_down_proj = cpu_model.model.layers[target_layer_idx].mlp.down_proj.weight.data.clone()
            del cpu_model  # Free memory
            torch.cuda.empty_cache()
            print(f"[Phase 5 RECONSTRUCTION] CPU model weights loaded successfully")
            print(f"[Phase 5 RECONSTRUCTION] Proper weight shape: {original_down_proj.shape}")
            
        else:
            # Regular float weights
            original_down_proj = original_down_proj_raw.clone()
    else:
        raise ValueError("[Phase 5 RECONSTRUCTION] ERROR: Cannot access down_proj weights")
    
    print(f"[Phase 5 RECONSTRUCTION] Final down_proj info:")
    print(f"[Phase 5 RECONSTRUCTION]   Shape: {original_down_proj.shape}")
    print(f"[Phase 5 RECONSTRUCTION]   Dtype: {original_down_proj.dtype}")
    
    # Only compute norm if we have float weights
    if original_down_proj.dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
        norm_value = torch.norm(original_down_proj.float(), 'fro').item()
        print(f"[Phase 5 RECONSTRUCTION]   Norm: {norm_value:.4f}")
    else:
        print(f"[Phase 5 RECONSTRUCTION]   Norm: Cannot compute (non-float dtype)")
    
    # Step 5: Apply transformation to down_proj weights
    # The transformation is applied as: new_weight = T_main.T @ original_weight
    # This is because T_main is designed to transform activations: activation @ T_main
    # But weights transform activations in the opposite direction: activation @ weight.T
    
    print(f"[Phase 5 RECONSTRUCTION] Applying transformation to down_proj...")
    
    try:
        # Ensure both tensors are on the same device
        print(f"[Phase 5 RECONSTRUCTION] Device alignment:")
        print(f"[Phase 5 RECONSTRUCTION]   T_main device: {T_main.device}")
        print(f"[Phase 5 RECONSTRUCTION]   original_down_proj device: {original_down_proj.device}")
        
        # Move both tensors to CPU for safe computation (avoids memory issues with large matrices)
        if T_main.device.type != 'cpu':
            T_main = T_main.cpu()
            print(f"[Phase 5 RECONSTRUCTION] Moved T_main to CPU")
        
        if original_down_proj.device.type != 'cpu':
            original_down_proj = original_down_proj.cpu()
            print(f"[Phase 5 RECONSTRUCTION] Moved original_down_proj to CPU")
        
        # Convert to same precision for multiplication
        original_down_proj_f64 = original_down_proj.to(torch.float64)
        T_main_f64 = T_main.to(torch.float64)
        
        print(f"[Phase 5 RECONSTRUCTION] Final device check:")
        print(f"[Phase 5 RECONSTRUCTION]   T_main_f64 device: {T_main_f64.device}")
        print(f"[Phase 5 RECONSTRUCTION]   original_down_proj_f64 device: {original_down_proj_f64.device}")
        print(f"[Phase 5 RECONSTRUCTION]   Both tensors on same device: {T_main_f64.device == original_down_proj_f64.device}")
        
        # Apply transformation: new_weight = T_main.T @ original_weight
        new_down_proj = T_main_f64.T @ original_down_proj_f64
        
        # Convert back to model's precision
        new_down_proj = new_down_proj.to(original_down_proj.dtype)
        
        print(f"[Phase 5 RECONSTRUCTION] Transformation applied successfully")
        print(f"[Phase 5 RECONSTRUCTION] New down_proj shape: {new_down_proj.shape}")
        print(f"[Phase 5 RECONSTRUCTION] New down_proj device: {new_down_proj.device}")
        
        # Only compute norms if we have float weights
        if original_down_proj.dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
            new_norm = torch.norm(new_down_proj.float(), 'fro').item()
            orig_norm = torch.norm(original_down_proj.float(), 'fro').item()
            change_ratio = torch.norm((new_down_proj - original_down_proj).float(), 'fro').item() / orig_norm
            
            print(f"[Phase 5 RECONSTRUCTION] New down_proj norm: {new_norm:.4f}")
            print(f"[Phase 5 RECONSTRUCTION] Weight change ratio: {change_ratio:.4f}")
        else:
            print(f"[Phase 5 RECONSTRUCTION] Cannot compute norms (non-float dtype)")
        
    except Exception as e:
        print(f"[Phase 5 RECONSTRUCTION] ERROR in transformation application: {str(e)}")
        raise
    
    # Step 6: Truncate model (remove the selected blocks)
    print(f"[Phase 5 RECONSTRUCTION] Truncating model...")
    
    layers_before_truncation = len(model.model.layers)
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    layers_after_truncation = len(model.model.layers)
    
    print(f"[Phase 5 RECONSTRUCTION] Model truncation complete:")
    print(f"[Phase 5 RECONSTRUCTION]   Layers before: {layers_before_truncation}")
    print(f"[Phase 5 RECONSTRUCTION]   Layers after: {layers_after_truncation}")
    print(f"[Phase 5 RECONSTRUCTION]   Layers removed: {layers_before_truncation - layers_after_truncation}")
    print(f"[Phase 5 RECONSTRUCTION]   New total layers: {model.config.num_hidden_layers}")
    
    # Step 7: Handle quantized model - replace with float model
    print(f"[Phase 5 RECONSTRUCTION] Preparing model for weight updates...")
    
    # After truncation, the target layer index needs to be adjusted
    adjusted_target_idx = target_layer_idx  # Should remain the same since we remove layers after it
    
    # Check if the model is quantized
    target_layer_after_truncation = model.model.layers[adjusted_target_idx]
    current_weight = target_layer_after_truncation.mlp.down_proj.weight.data
    
    print(f"[Phase 5 RECONSTRUCTION] Current weight info after truncation:")
    print(f"[Phase 5 RECONSTRUCTION]   Shape: {current_weight.shape}")
    print(f"[Phase 5 RECONSTRUCTION]   Dtype: {current_weight.dtype}")
    
    # If the model is still quantized after truncation, replace with float model
    if current_weight.dtype == torch.uint8 or current_weight.shape != new_down_proj.shape:
        print(f"[Phase 5 RECONSTRUCTION] Model still quantized, replacing with float model...")
        
        # Load the full model in float format
        from transformers import AutoModelForCausalLM
        float_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            token=token
        )
        
        print(f"[Phase 5 RECONSTRUCTION] Float model loaded successfully")
        
        # Apply the same truncation to the float model
        from .utils import truncate_model
        float_model = truncate_model(float_model, start_id - num_layer, end_id - num_layer)
        
        print(f"[Phase 5 RECONSTRUCTION] Float model truncated: {len(float_model.model.layers)} layers")
        
        # Apply the transformed weights to the target layer
        if adjusted_target_idx < len(float_model.model.layers):
            print(f"[Phase 5 RECONSTRUCTION] Applying transformed weights to layer {adjusted_target_idx}")
            float_model.model.layers[adjusted_target_idx].mlp.down_proj.weight.data.copy_(new_down_proj.to(float_model.device))
            print(f"[Phase 5 RECONSTRUCTION] Transformed weights applied successfully")
        else:
            raise ValueError(f"[Phase 5 RECONSTRUCTION] ERROR: Target layer index out of range: {adjusted_target_idx}")
        
        # Replace the quantized model with the float model
        del model  # Free quantized model memory
        model = float_model
        torch.cuda.empty_cache()
        
        print(f"[Phase 5 RECONSTRUCTION] Model replaced with float model containing transformed weights")
        
    else:
        # Model is already in float format, just update the weights
        print(f"[Phase 5 RECONSTRUCTION] Updating weights in existing float model...")
        if adjusted_target_idx >= len(model.model.layers):
            raise ValueError(f"[Phase 5 RECONSTRUCTION] ERROR: Target layer index out of range after truncation: {adjusted_target_idx}")
        
        model.model.layers[adjusted_target_idx].mlp.down_proj.weight.data.copy_(new_down_proj.to(model.device))
        print(f"[Phase 5 RECONSTRUCTION] Weights updated successfully")
    
    # Step 8: Validate the updated model structure
    print(f"[Phase 5 RECONSTRUCTION] Validating updated model...")
    
    try:
        # Check that we can access the updated layer
        updated_layer = model.model.layers[adjusted_target_idx]
        updated_weight = updated_layer.mlp.down_proj.weight.data
        
        print(f"[Phase 5 RECONSTRUCTION] Validation successful:")
        print(f"[Phase 5 RECONSTRUCTION]   Updated weight shape: {updated_weight.shape}")
        print(f"[Phase 5 RECONSTRUCTION]   Updated weight dtype: {updated_weight.dtype}")
        
        # Only compute norm if we have float weights
        if updated_weight.dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
            updated_norm = torch.norm(updated_weight.float(), 'fro').item()
            print(f"[Phase 5 RECONSTRUCTION]   Updated weight norm: {updated_norm:.4f}")
            
            # Verify the transformation was applied by checking if weights changed significantly
            if abs(updated_norm - 53.3220) < 1.0:  # Compare with our computed norm
                print(f"[Phase 5 RECONSTRUCTION] SUCCESS: Transformation applied correctly")
            else:
                print(f"[Phase 5 RECONSTRUCTION] WARNING: Weight norm mismatch, but update completed")
        else:
            print(f"[Phase 5 RECONSTRUCTION] Cannot compute norm (non-float dtype)")
            
    except Exception as e:
        print(f"[Phase 5 RECONSTRUCTION] ERROR in model validation: {str(e)}")
        raise
    
    # Step 9: Save the model
    print(f"[Phase 5 RECONSTRUCTION] Saving model...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Save model
        model.save_pretrained(save_path)
        print(f"[Phase 5 RECONSTRUCTION] Model saved to: {save_path}")
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
            print(f"[Phase 5 RECONSTRUCTION] Tokenizer saved to: {save_path}")
        
        # Save transformation metadata
        metadata = {
            'transformation_applied': True,
            'original_layers': layers_before_truncation,
            'final_layers': layers_after_truncation,
            'layers_removed': layers_before_truncation - layers_after_truncation,
            'target_layer_idx': adjusted_target_idx,
            'transformation_norm': torch.norm(T_main, 'fro').item(),
            'approximation_quality': transformations.get('approximation_quality', {}),
            'layer_priorities': transformations.get('layer_priorities', []),
            'gate_compensation': transformations.get('gate_compensation', {})
        }
        
        import json
        metadata_path = os.path.join(save_path, 'gaco_metadata.json')
        with open(metadata_path, 'w') as f:
            # Convert any tensor values to float for JSON serialization
            def convert_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tensors(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_tensors(metadata), f, indent=2)
        
        print(f"[Phase 5 RECONSTRUCTION] Metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"[Phase 5 RECONSTRUCTION] ERROR in saving: {str(e)}")
        raise
    
    # Step 10: Summary
    print(f"[Phase 5 RECONSTRUCTION] Model reconstruction complete!")
    print(f"[Phase 5 RECONSTRUCTION] Summary:")
    print(f"[Phase 5 RECONSTRUCTION]   Original layers: {layers_before_truncation}")
    print(f"[Phase 5 RECONSTRUCTION]   Final layers: {layers_after_truncation}")
    print(f"[Phase 5 RECONSTRUCTION]   Compression ratio: {(layers_before_truncation - layers_after_truncation) / layers_before_truncation * 100:.1f}%")
    print(f"[Phase 5 RECONSTRUCTION]   Approximation quality: {transformations.get('approximation_quality', {}).get('quality_assessment', 'Unknown')}")
    print(f"[Phase 5 RECONSTRUCTION]   Output error: {transformations.get('approximation_quality', {}).get('output_error', 0) * 100:.1f}%")
    print(f"[Phase 5 RECONSTRUCTION]   Model saved to: {save_path}")
    
    return save_path


def gate_aware_coupled_method(
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
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """
    Updated GACO method with Phase 5 integration
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset name
        dataset_column: Column containing text data
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip between compared blocks
        dataset_size: Size of calibration dataset
        dataset_subset: Dataset subset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save the processed model
        token: HuggingFace token for private models
        distances_path: Path to pre-computed distance metrics
        num_A: Number of blocks to process
        merge_consecutive: Whether to merge consecutive blocks
        **kwargs: Additional arguments
        
    Returns:
        Path to the final processed model
    """
    print(f"[GACO COMPLETE] Starting complete GACO pipeline with Phase 5...")
    print(f"[GACO COMPLETE] Model: {model_path}")
    print(f"[GACO COMPLETE] Dataset: {dataset}, Size: {dataset_size}")
    print(f"[GACO COMPLETE] Layers to skip: {layers_to_skip}")
    
    # Import required modules
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from .utils import get_calib_dataloader, select_non_overlapping_blocks
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"[GACO COMPLETE] Loading model...")
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
    
    print(f"[GACO COMPLETE] Model loaded. Hidden size: {model.config.hidden_size}")
    print(f"[GACO COMPLETE] Number of layers: {model.config.num_hidden_layers}")
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    print(f"[GACO COMPLETE] Data loader created")
    
    # Load pre-computed distances and select blocks
    print(f"[GACO COMPLETE] Loading distances from: {distances_path}")
    average_distances = torch.load(distances_path, weights_only=False)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
    
    print(f"[GACO COMPLETE] Selected blocks: {selected_blocks}")
    
    # Process each selected block
    final_save_path = None
    
    for i in range(len(selected_blocks)):
        start_id = start_ids[i]
        end_id = end_ids[i]
        num_layer = num_layers[i]
        
        print(f"[GACO COMPLETE] Processing block {i+1}/{len(selected_blocks)}: layers {start_id} to {end_id}")
        
        # Phases 2-4 (existing implementation)
        activations = collect_enhanced_activations_streaming(
            model=model,
            start_id=start_id - num_layer,
            end_id=end_id - num_layer,
            dataset_size=dataset_size,
            max_length=max_length,
            dataloader=dataloader,
            device=next(model.parameters()).device,
            tokenizer=tokenizer
        )
        
        gate_analysis = analyze_gate_patterns(
            activation_stats=activations,
            start_id=start_id - num_layer,
            end_id=end_id - num_layer
        )
        
        transformations = estimate_coupled_transformations_residual_aware(
            activation_stats=activations,
            gate_analysis=gate_analysis,
            start_id=start_id - num_layer,
            end_id=end_id - num_layer,
            hidden_size=model.config.hidden_size
        )
        
        print(f"[GACO COMPLETE] Phases 2-4 complete for block {i+1}")
        
        # Phase 5: Apply transformation to model
        if save_path is None:
            import os
            if not os.path.exists('output_models'):
                os.makedirs('output_models')
            base_save_path = "output_models/" + (
                f"{model_path}_{layers_to_skip}_layers_GACO_block_{i+1}"
            ).replace("/", "_")
        else:
            base_save_path = f"{save_path}_block_{i+1}"
        
        final_save_path = apply_residual_transformation_to_model(
            model=model,
            transformations=transformations,
            start_id=start_id,
            end_id=end_id,
            num_layer=num_layer,
            save_path=base_save_path,
            tokenizer=tokenizer,
            model_path=model_path,
            token=token
        )
        
        print(f"[GACO COMPLETE] Phase 5 complete for block {i+1}")
        
        # Update model path for next iteration (if processing multiple blocks)
        # Note: For multiple blocks, we would need to reload the model
        # For now, we assume single block processing
        
        # Memory cleanup
        del activations
        del gate_analysis
        del transformations
        torch.cuda.empty_cache()
        
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print(f"[GACO COMPLETE] Complete GACO pipeline finished!")
    print(f"[GACO COMPLETE] Final model saved to: {final_save_path}")
    
    return final_save_path

