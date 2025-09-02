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
    
def estimate_coupled_transformations(
    activation_stats: Dict, 
    gate_analysis: Dict, 
    start_id: int, 
    end_id: int,
    hidden_size: int
) -> Dict[str, torch.Tensor]:
    """
    Phase 4: Coupled Transformation Estimation
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
    print(f"[Phase 4] Starting Coupled Transformation Estimation...")
    print(f"[Phase 4] Target layers: {start_id} to {end_id-1}")
    print(f"[Phase 4] Hidden size: {hidden_size}")
    
    # Get input/output data for transformation estimation
    input_activations = activation_stats['input_activations']  # [samples, hidden_size]
    output_activations = activation_stats['output_activations']  # [samples, hidden_size]
    
    num_samples = input_activations.shape[0]
    print(f"[Phase 4] Using {num_samples} samples for transformation estimation")
    
    if num_samples == 0:
        raise ValueError("[Phase 4] ERROR: No input/output samples available")
    
    # Convert to float64 for numerical stability
    X = input_activations.to(torch.float64)  # Input to blocks
    Y = output_activations.to(torch.float64)  # Expected output
    
    print(f"[Phase 4] Input tensor shape: {X.shape}")
    print(f"[Phase 4] Output tensor shape: {Y.shape}")
    
    # Initialize transformation matrices
    transformations = {
        'T_gate': torch.eye(hidden_size, dtype=torch.float64),  # Gate transformation
        'T_up': torch.eye(hidden_size, dtype=torch.float64),    # Up projection transformation  
        'T_down': torch.eye(hidden_size, dtype=torch.float64),  # Down projection transformation
        'gate_compensation': {},  # Per-layer gate compensation factors
        'layer_priorities': {},   # Layer importance for sequential optimization
    }
    
    # Extract layer priorities from gate analysis
    if 'cross_layer_analysis' in gate_analysis:
        cross_analysis = gate_analysis['cross_layer_analysis']
        most_important = cross_analysis['most_important_layer']
        least_important = cross_analysis['least_important_layer']
        
        print(f"[Phase 4] Most important layer: {most_important}")
        print(f"[Phase 4] Least important layer: {least_important}")
        
        # Create priority ordering (most important first)
        layer_importances = []
        for layer_idx in range(start_id, end_id):
            if layer_idx in gate_analysis['layer_importance_scores']:
                importance = gate_analysis['layer_importance_scores'][layer_idx]['total_importance']
                layer_importances.append((layer_idx, importance))
        
        # Sort by importance (descending)
        layer_importances.sort(key=lambda x: x[1], reverse=True)
        transformations['layer_priorities'] = layer_importances
        
        print(f"[Phase 4] Layer priority order: {[f'L{idx}({imp:.1f})' for idx, imp in layer_importances]}")
    
    # Method 1: Gate-Weighted Least Squares (Primary approach)
    print(f"[Phase 4] Computing gate-weighted transformation...")
    
    # Compute aggregate gate importance weights across all layers
    total_gate_weights = torch.zeros(hidden_size, dtype=torch.float64)
    weight_count = 0
    
    for layer_idx in range(start_id, end_id):
        if layer_idx in gate_analysis['information_flow_weights']:
            # Get sigmoid weights for this layer
            layer_weights = gate_analysis['information_flow_weights'][layer_idx]['sigmoid_weights']
            
            # For gate transformation, we need to map intermediate_size to hidden_size
            # Use simple aggregation: take mean of intermediate weights
            intermediate_size = layer_weights.shape[0]
            
            # Map intermediate weights to hidden dimension
            if intermediate_size >= hidden_size:
                # Downsample intermediate weights
                stride = intermediate_size // hidden_size
                mapped_weights = layer_weights.view(-1, stride).mean(dim=1)[:hidden_size]
            else:
                # Upsample intermediate weights  
                repeat_factor = hidden_size // intermediate_size
                mapped_weights = layer_weights.repeat(repeat_factor)[:hidden_size]
            
            total_gate_weights += mapped_weights.to(torch.float64)
            weight_count += 1
            
            print(f"[Phase 4] Layer {layer_idx} weights mapped: {intermediate_size} -> {mapped_weights.shape[0]}")
    
    if weight_count > 0:
        # Average weights across layers
        avg_gate_weights = total_gate_weights / weight_count
        print(f"[Phase 4] Average gate weights computed from {weight_count} layers")
        print(f"[Phase 4] Gate weight range: {avg_gate_weights.min():.4f} to {avg_gate_weights.max():.4f}")
    else:
        # Fallback to uniform weights
        avg_gate_weights = torch.ones(hidden_size, dtype=torch.float64)
        print(f"[Phase 4] WARNING: No gate weights found, using uniform weights")
    
    # Create weight matrix for optimization
    W = torch.diag(avg_gate_weights)  # [hidden_size, hidden_size]
    
    # Gate-aware weighted least squares: min ||W(XT - Y)||²
    # Solution: T = (X'WX)^(-1) X'WY
    XTW = X.T @ W  # [hidden_size, hidden_size]
    XTWX = XTW @ X  # [hidden_size, hidden_size]
    XTWY = XTW @ Y  # [hidden_size, hidden_size]
    
    print(f"[Phase 4] Computing weighted least squares solution...")
    
    try:
        # Add regularization for numerical stability
        reg_strength = 1e-4
        regularizer = reg_strength * torch.eye(hidden_size, dtype=torch.float64)
        XTWX_reg = XTWX + regularizer
        
        # Solve the linear system
        T_coupled = torch.linalg.solve(XTWX_reg, XTWY)
        print(f"[Phase 4] Coupled transformation computed successfully")
        print(f"[Phase 4] Transformation matrix shape: {T_coupled.shape}")
        
    except Exception as e:
        print(f"[Phase 4] WARNING: Numerical instability in solve, using pseudo-inverse")
        T_coupled = torch.linalg.pinv(XTWX) @ XTWY
    
    # Decompose coupled transformation into gate/up/down components
    print(f"[Phase 4] Decomposing coupled transformation...")
    
    # Method: Use SVD to decompose T_coupled ≈ U @ S @ V.T
    # Assign: T_up ← U, T_gate ← S (diagonal), T_down ← V.T
    U, S, Vt = torch.linalg.svd(T_coupled, full_matrices=False)
    
    # Ensure square matrices
    min_dim = min(U.shape[1], S.shape[0], Vt.shape[0])
    U_sq = U[:, :min_dim]  # [hidden_size, min_dim]
    S_sq = torch.diag(S[:min_dim])  # [min_dim, min_dim] 
    Vt_sq = Vt[:min_dim, :]  # [min_dim, hidden_size]
    
    # Pad to full size if necessary
    if min_dim < hidden_size:
        # Pad with identity
        U_full = torch.eye(hidden_size, dtype=torch.float64)
        U_full[:, :min_dim] = U_sq
        
        S_full = torch.eye(hidden_size, dtype=torch.float64) 
        S_full[:min_dim, :min_dim] = S_sq
        
        Vt_full = torch.eye(hidden_size, dtype=torch.float64)
        Vt_full[:min_dim, :] = Vt_sq
        
        transformations['T_up'] = U_full
        transformations['T_gate'] = S_full  
        transformations['T_down'] = Vt_full
    else:
        transformations['T_up'] = U_sq
        transformations['T_gate'] = S_sq
        transformations['T_down'] = Vt_sq
    
    print(f"[Phase 4] SVD decomposition complete:")
    print(f"[Phase 4] T_up shape: {transformations['T_up'].shape}")
    print(f"[Phase 4] T_gate shape: {transformations['T_gate'].shape}")  
    print(f"[Phase 4] T_down shape: {transformations['T_down'].shape}")
    
    # Method 2: Consistency Regularization
    print(f"[Phase 4] Applying consistency regularization...")
    
    # Ensure T_up @ T_gate @ T_down ≈ T_coupled
    reconstructed = transformations['T_up'] @ transformations['T_gate'] @ transformations['T_down']
    consistency_error = torch.norm(reconstructed - T_coupled, 'fro').item()
    
    print(f"[Phase 4] Reconstruction error: {consistency_error:.6f}")
    
    if consistency_error > 0.1:  # If error too large, apply correction
        print(f"[Phase 4] High reconstruction error, applying correction...")
        correction_factor = 0.9  # Blend factor
        
        # Blend with identity to reduce error
        transformations['T_up'] = correction_factor * transformations['T_up'] + (1-correction_factor) * torch.eye(hidden_size, dtype=torch.float64)
        transformations['T_gate'] = correction_factor * transformations['T_gate'] + (1-correction_factor) * torch.eye(hidden_size, dtype=torch.float64)
        transformations['T_down'] = correction_factor * transformations['T_down'] + (1-correction_factor) * torch.eye(hidden_size, dtype=torch.float64)
        
        # Re-check consistency
        reconstructed_corrected = transformations['T_up'] @ transformations['T_gate'] @ transformations['T_down']
        corrected_error = torch.norm(reconstructed_corrected - T_coupled, 'fro').item()
        print(f"[Phase 4] Corrected reconstruction error: {corrected_error:.6f}")
    
    # Method 3: Per-layer gate compensation
    print(f"[Phase 4] Computing per-layer gate compensation...")
    
    for layer_idx in range(start_id, end_id):
        if layer_idx in gate_analysis['gate_activation_patterns']:
            patterns = gate_analysis['gate_activation_patterns'][layer_idx]
            
            # Compensation based on critical vs pruning candidates
            num_critical = len(patterns['critical_neurons'])
            num_pruning = len(patterns['pruning_candidate_neurons'])
            
            # Higher compensation for layers with more critical neurons
            if num_critical + num_pruning > 0:
                critical_ratio = num_critical / (num_critical + num_pruning)
                compensation_factor = 0.5 + 0.5 * critical_ratio  # Range [0.5, 1.0]
            else:
                compensation_factor = 1.0
            
            transformations['gate_compensation'][layer_idx] = compensation_factor
            print(f"[Phase 4] Layer {layer_idx} compensation: {compensation_factor:.3f} (critical: {num_critical}, pruning: {num_pruning})")
    
    # Final validation
    print(f"[Phase 4] Transformation validation...")
    
    # Test transformation on sample data
    if num_samples > 0:
        sample_input = X[:min(100, num_samples)]  # Use up to 100 samples
        sample_expected = Y[:min(100, num_samples)]
        
        # Apply full transformation chain
        transformed = sample_input @ transformations['T_up'] @ transformations['T_gate'] @ transformations['T_down']
        
        # Compute approximation error
        approx_error = torch.norm(transformed - sample_expected, 'fro') / torch.norm(sample_expected, 'fro')
        print(f"[Phase 4] Relative approximation error: {approx_error.item():.6f}")
        
        if approx_error.item() < 0.1:
            print(f"[Phase 4] SUCCESS: Good approximation quality")
        elif approx_error.item() < 0.5:
            print(f"[Phase 4] WARNING: Moderate approximation quality")
        else:
            print(f"[Phase 4] WARNING: Poor approximation quality")
    
    print(f"[Phase 4] Coupled Transformation Estimation complete!")
    return transformations


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
    Gate-aware coupled optimization method - Phases 2-4 implementation
    
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
        Path to the processed model
    """
    print(f"[GACO] Starting Gate-Aware Coupled Optimization method...")
    print(f"[GACO] Model: {model_path}")
    print(f"[GACO] Dataset: {dataset}, Size: {dataset_size}")
    print(f"[GACO] Layers to skip: {layers_to_skip}")
    
    # Import required modules (same as original ReplaceMe)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from .utils import get_calib_dataloader, select_non_overlapping_blocks, truncate_model
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"[GACO] Loading model...")
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
    
    print(f"[GACO] Model loaded. Hidden size: {model.config.hidden_size}")
    print(f"[GACO] Number of layers: {model.config.num_hidden_layers}")
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    print(f"[GACO] Data loader created")
    
    # Load pre-computed distances and select blocks
    print(f"[GACO] Loading distances from: {distances_path}")
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
    
    print(f"[GACO] Selected blocks: {selected_blocks}")
    print(f"[GACO] Start IDs: {start_ids}")
    print(f"[GACO] End IDs: {end_ids}")
    
    # Process each selected block
    for i in range(len(selected_blocks)):
        start_id = start_ids[i]
        end_id = end_ids[i]
        num_layer = num_layers[i]
        
        print(f"[GACO] Processing block {i+1}/{len(selected_blocks)}: layers {start_id} to {end_id}")
        
        # Phase 2: Collect enhanced activations using streaming approach
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
        
        print(f"[GACO] Enhanced activations collected for block {i+1}")
        print(f"[GACO] Phase 2 streaming complete for block {i+1}")
        
        # Validate that we collected the statistics properly
        if len(activations['gate_importance']) > 0:
            total_importance = sum(activations['gate_importance'][k].sum().item() for k in activations['gate_importance'])
            print(f"[GACO] SUCCESS: Gate importance computed - total importance: {total_importance:.4f}")
        else:
            print(f"[GACO] WARNING: No gate importance computed")
            
        if activations['input_activations'].shape[0] > 0:
            print(f"[GACO] SUCCESS: Input/output samples collected - {activations['input_activations'].shape[0]} samples")
        else:
            print(f"[GACO] WARNING: No input/output samples collected")
        
        print(f"[GACO] Total tokens processed: {activations['total_tokens']}")
        print(f"[GACO] Total batches processed: {activations['total_batches']}")
        
        # Phase 3: Gate Importance Analysis
        print(f"[GACO] Starting Phase 3 for block {i+1}...")
        try:
            gate_analysis = analyze_gate_patterns(
                activation_stats=activations,
                start_id=start_id - num_layer,
                end_id=end_id - num_layer
            )
            print(f"[GACO] Phase 3 function returned: {type(gate_analysis)}")
        except Exception as e:
            print(f"[GACO] ERROR in Phase 3: {str(e)}")
            gate_analysis = None
        
        print(f"[GACO] Phase 3 complete for block {i+1}")
        
        # Validate Phase 3 results
        if gate_analysis is not None and 'layer_importance_scores' in gate_analysis and gate_analysis['layer_importance_scores']:
            analyzed_layers = len(gate_analysis['layer_importance_scores'])
            print(f"[GACO] SUCCESS: Gate analysis completed for {analyzed_layers} layers")
            
            # Show some key insights
            if 'cross_layer_analysis' in gate_analysis:
                cross_analysis = gate_analysis['cross_layer_analysis']
                print(f"[GACO] Most important layer: {cross_analysis['most_important_layer']}")
                print(f"[GACO] Least important layer: {cross_analysis['least_important_layer']}")
                print(f"[GACO] Mean layer importance: {cross_analysis['mean_layer_importance']:.4f}")
            
            # Show neuron analysis for first layer as example
            first_layer = start_id - num_layer
            if first_layer in gate_analysis['gate_activation_patterns']:
                patterns = gate_analysis['gate_activation_patterns'][first_layer]
                print(f"[GACO] Layer {first_layer} neuron analysis:")
                print(f"[GACO]   Critical neurons: {len(patterns['critical_neurons'])}")
                print(f"[GACO]   Pruning candidates: {len(patterns['pruning_candidate_neurons'])}")
        else:
            print(f"[GACO] WARNING: Gate analysis failed or incomplete")
            print(f"[GACO] Gate analysis result: {gate_analysis}")
            
            # Skip Phase 4 if Phase 3 failed
            print(f"[GACO] Skipping Phase 4 due to Phase 3 failure")
            
            # Memory cleanup
            del activations
            if gate_analysis is not None:
                del gate_analysis
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"[GACO] Block {i+1} processing complete (Phases 2-3, Phase 4 skipped)")
            continue
        
        # Phase 4: Coupled Transformation Estimation
        print(f"[GACO] Starting Phase 4 for block {i+1}...")
        try:
            transformations = estimate_coupled_transformations(
                activation_stats=activations,
                gate_analysis=gate_analysis,
                start_id=start_id - num_layer,
                end_id=end_id - num_layer,
                hidden_size=model.config.hidden_size
            )
            
            print(f"[GACO] Phase 4 complete for block {i+1}")
            
            # Validate Phase 4 results
            if transformations is not None and 'T_gate' in transformations and 'T_up' in transformations and 'T_down' in transformations:
                print(f"[GACO] SUCCESS: Coupled transformations estimated")
                print(f"[GACO] T_gate norm: {torch.norm(transformations['T_gate'], 'fro').item():.4f}")
                print(f"[GACO] T_up norm: {torch.norm(transformations['T_up'], 'fro').item():.4f}")  
                print(f"[GACO] T_down norm: {torch.norm(transformations['T_down'], 'fro').item():.4f}")
                
                if 'layer_priorities' in transformations:
                    priorities = transformations['layer_priorities']
                    print(f"[GACO] Layer priorities: {[f'L{idx}' for idx, _ in priorities[:2]]}")
            else:
                print(f"[GACO] WARNING: Transformation estimation failed")
                print(f"[GACO] Transformations result: {transformations}")
            
            print(f"[GACO] Phase 4 analysis complete for block {i+1}")
            
            # Memory cleanup
            del transformations
            
        except Exception as e:
            print(f"[GACO] ERROR in Phase 4: {str(e)}")
            print(f"[GACO] Phase 4 failed, continuing with next block...")
        
        # Memory cleanup
        del activations
        del gate_analysis
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[GACO] Block {i+1} processing complete (Phases 2-4)")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Generate output path
    if save_path is None:
        import os
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_GACO"
        ).replace("/", "_")
    
    print(f"[GACO] Method execution complete (Phases 2-4)")
    return f"{save_path}_GACO_phase2_phase3_phase4"