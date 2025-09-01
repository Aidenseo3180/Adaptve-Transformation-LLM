import argparse
import gc
import logging
import os
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def analyze_gate_importance(
    model,
    dataloader,
    tokenizer,
    max_length: int,
    start_layer: int,
    end_layer: int
) -> torch.Tensor:
    """Analyze gate projection importance patterns across calibration data.
    
    Args:
        model: The transformer model
        dataloader: Calibration data loader
        tokenizer: Model tokenizer
        max_length: Maximum sequence length
        start_layer: Starting layer index for analysis
        end_layer: Ending layer index for analysis
        
    Returns:
        gate_importance: Tensor of shape [hidden_size] with importance scores
    """
    print("Starting gate importance analysis...")
    
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    
    # Storage for gate activations
    gate_activations = []
    
    def gate_hook(module, input, output):
        """Hook to capture gate projection outputs"""
        gate_activations.append(output.detach().to(torch.bfloat16).cpu())  # Convert to bfloat16
    
    # Register hooks for gate projections in target layers
    hooks = []
    target_layers = list(range(start_layer, end_layer))
    print(f"Analyzing gate importance for layers {start_layer} to {end_layer-1}")
    
    for layer_idx in target_layers:
        if hasattr(model.model.layers[layer_idx].mlp, 'gate_proj'):
            hook = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(gate_hook)
            hooks.append(hook)
        else:
            print(f"Warning: Layer {layer_idx} does not have gate_proj")
    
    # Collect gate activations
    all_gate_activations = []  # Store all activations for proper analysis
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting gate activations")):
            if batch_idx >= 10:  # Limit samples for efficiency
                break
                
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Clear previous activations
            gate_activations.clear()
            
            # Forward pass to trigger hooks
            _ = model(**inputs)
            
            # Process collected activations
            if gate_activations:
                # Stack activations from all layers and flatten
                batch_activations = torch.stack(gate_activations)  # [num_layers, batch_size, seq_len, hidden_size]
                batch_flat = batch_activations.view(-1, batch_activations.shape[-1])  # [total_tokens, hidden_size]
                all_gate_activations.append(batch_flat.cpu())
                print(f"Batch {batch_idx}: collected {batch_flat.shape[0]} tokens")
    
    # Concatenate all activations and compute statistics
    if all_gate_activations:
        all_gates = torch.cat(all_gate_activations, dim=0)  # [total_tokens, hidden_size]
        print(f"Total gate activations collected: {all_gates.shape}")
        
        # Calculate importance scores across all tokens
        mean_activation = all_gates.mean(dim=0)  # [hidden_size]
        variance = all_gates.var(dim=0)  # [hidden_size]
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
        # Importance = variance * absolute mean (captures both diversity and magnitude)
        gate_importance = variance * torch.abs(mean_activation)
        gate_importance = gate_importance / gate_importance.max()  # Normalize
    else:
        print("Warning: No gate activations collected, using uniform importance")
        gate_importance = torch.ones(hidden_size)
    
    print(f"Gate importance analysis complete. Max importance: {gate_importance.max():.4f}")
    print(f"Mean importance: {gate_importance.mean():.4f}")
    
    return gate_importance


def coupled_optimization(
    up_activations: torch.Tensor,
    down_activations: torch.Tensor, 
    targets: torch.Tensor,
    gate_importance: torch.Tensor,
    max_epochs: int = 10,
    lr: float = 1e-4,
    chunk_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform coupled optimization of up_proj and down_proj transformations.
    
    Args:
        up_activations: Up projection activations [N, hidden_size]
        down_activations: Down projection activations [N, hidden_size] 
        targets: Target activations [N, hidden_size]
        gate_importance: Gate importance scores [hidden_size]
        max_epochs: Maximum optimization epochs
        lr: Learning rate
        chunk_idx: Chunk index for debugging
        
    Returns:
        T_up: Up projection transformation matrix
        T_down: Down projection transformation matrix
    """
    print(f"Starting coupled optimization for chunk {chunk_idx}...")
    print(f"Input shapes - Up: {up_activations.shape}, Down: {down_activations.shape}, Targets: {targets.shape}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = up_activations.shape[-1]
    
    # Initialize transformation matrices in bfloat16
    T_up = torch.eye(hidden_size, requires_grad=True, device=device, dtype=torch.bfloat16)
    T_down = torch.eye(hidden_size, requires_grad=True, device=device, dtype=torch.bfloat16)
    
    # Move data to device and convert to bfloat16
    up_activations = up_activations.to(device).to(torch.bfloat16)
    down_activations = down_activations.to(device).to(torch.bfloat16)
    targets = targets.to(device).to(torch.bfloat16)
    gate_importance = gate_importance.to(device).to(torch.bfloat16)
    
    print(f"Moved tensors to {device}, all using bfloat16")
    
    # Optimizers
    optimizer_up = torch.optim.Adam([T_up], lr=lr)
    optimizer_down = torch.optim.Adam([T_down], lr=lr)
    optimizer_joint = torch.optim.Adam([T_up, T_down], lr=lr/2)
    
    def cosine_loss_weighted(pred, target, weights):
        """Weighted cosine loss using gate importance - chunk independent"""
        print(f"[LOSS DEBUG] Pred shape: {pred.shape}, Target shape: {target.shape}")
        
        # Ensure pred and target have same shape
        if pred.shape[0] != target.shape[0]:
            min_size = min(pred.shape[0], target.shape[0])
            pred = pred[:min_size]
            target = target[:min_size]
            print(f"[LOSS DEBUG] Trimmed to common size: {min_size}")
        
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        
        # Use scalar weight (mean of gate importance) instead of per-token weighting
        weight_scalar = weights.mean()  # Convert to scalar
        weighted_loss = (1 - cosine_sim) * weight_scalar
        
        loss_value = weighted_loss.mean()
        print(f"[LOSS DEBUG] Loss computed: {loss_value.item():.6f}")
        return loss_value
    
    def consistency_loss(T_up, T_down):
        """Consistency regularization between transformations"""
        # Encourage T_up and T_down to be approximately inverse-related
        product = T_up.T @ T_down
        identity = torch.eye(hidden_size, device=device, dtype=torch.bfloat16)
        return torch.norm(product - identity, p='fro') ** 2
    
    print("Phase 1: Alternating optimization")
    # Phase 1: Alternating optimization
    for epoch in range(max_epochs // 2):
        # Optimize T_up with fixed T_down
        optimizer_up.zero_grad()
        up_pred = up_activations @ T_up
        loss_up = cosine_loss_weighted(up_pred, targets, gate_importance)
        loss_up.backward()
        optimizer_up.step()
        
        # Optimize T_down with fixed T_up  
        optimizer_down.zero_grad()
        down_pred = down_activations @ T_down
        loss_down = cosine_loss_weighted(down_pred, targets, gate_importance)
        loss_down.backward()
        optimizer_down.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Up loss: {loss_up.item():.4f}, Down loss: {loss_down.item():.4f}")
    
    print("Phase 2: Joint optimization with consistency regularization")
    # Phase 2: Joint optimization with consistency
    for epoch in range(max_epochs // 2):
        optimizer_joint.zero_grad()
        
        up_pred = up_activations @ T_up
        down_pred = down_activations @ T_down
        
        loss_up = cosine_loss_weighted(up_pred, targets, gate_importance)
        loss_down = cosine_loss_weighted(down_pred, targets, gate_importance)
        loss_consistency = consistency_loss(T_up, T_down)
        
        # Combined loss with consistency regularization
        total_loss = loss_up + loss_down + 0.1 * loss_consistency
        total_loss.backward()
        optimizer_joint.step()
        
        if epoch % 2 == 0:
            print(f"Joint epoch {epoch}: Total loss: {total_loss.item():.4f}, "
                  f"Consistency: {loss_consistency.item():.4f}")
    
    print("Coupled optimization complete")
    return T_up.detach(), T_down.detach()


def malt_method(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "train",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """Multi-Component Adaptive Linear Transformation method.
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset name for calibration
        dataset_column: Column containing text data
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Size of calibration dataset
        dataset_subset: Dataset subset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save compressed model
        min_distance_layer: Minimum distance layer index
        token: Authentication token
        save_transform_only: Whether to save only transformations
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to distance metrics
        num_A: Number of transformation matrices
        merge_consecutive: Whether to merge consecutive blocks
        
    Returns:
        save_path: Path where the compressed model is saved
    """
    print("=== Starting MALT (Multi-Component Adaptive Linear Transformation) ===")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        print("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model and tokenizer
    print(f"Loading model: {model_path}")
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
    
    # Load calibration data
    print(f"Loading calibration dataset: {dataset}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    hidden_size = model.config.hidden_size
    print(f"Model hidden size: {hidden_size}")
    print(f"Target layers for replacement: {start_id} to {end_id-1}")
    
    # Step 1: Analyze gate importance
    gate_importance = analyze_gate_importance(
        model, dataloader, tokenizer, max_length, start_id, end_id
    )
    
    # Step 2: Collect activations for transformation learning
    print("Collecting MLP component activations...")
    
    def save_mlp_components(name, component_type):
        """Hook to save different MLP component outputs"""
        def hook(module, input, output):
            mlp_components[f'{name}_{component_type}'] = output.detach().to(torch.bfloat16)  # Convert to bfloat16
        return hook
    
    # Register hooks for all MLP components
    hooks = []
    mlp_components = {}
    
    target_layer = model.model.layers[start_id - num_layer - 1]
    if hasattr(target_layer.mlp, 'gate_proj'):
        hooks.append(target_layer.mlp.gate_proj.register_forward_hook(
            save_mlp_components(f'layer_{start_id}', 'gate')))
    if hasattr(target_layer.mlp, 'up_proj'):
        hooks.append(target_layer.mlp.up_proj.register_forward_hook(
            save_mlp_components(f'layer_{start_id}', 'up')))
    if hasattr(target_layer.mlp, 'down_proj'):
        hooks.append(target_layer.mlp.down_proj.register_forward_hook(
            save_mlp_components(f'layer_{start_id}', 'down')))
    
    # Storage for collected activations
    gate_activations = []
    up_activations = []
    down_input_activations = []  # Input to down_proj (= up_proj output after activation)
    target_activations = []
    
    total_tokens = 0
    max_tokens = dataset_size * max_length if dataset_size else 100000
    batch_size_limit = 50  # Process in chunks of 50 batches
    
    print(f"Collecting activations in chunks of {batch_size_limit} batches...")
    
    # Initialize transformation storage
    all_transformations = {'up': [], 'down': [], 'single': []}
    
    with torch.no_grad():
        batch_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting MALT activations")):
            inputs = tokenizer(
                batch,
                return_tensors="pt", 
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Clear previous activations
            mlp_components.clear()
            
            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Get target activation (what we want to approximate)
            target_hidden = hidden_states[end_id - num_layer - 1]
            target_reshaped = target_hidden.view(-1, hidden_size).to(torch.bfloat16)
            
            # Get source activations
            if f'layer_{start_id}_gate' in mlp_components:
                gate_reshaped = mlp_components[f'layer_{start_id}_gate'].view(-1, hidden_size)
                gate_activations.append(gate_reshaped.cpu())
            
            if f'layer_{start_id}_up' in mlp_components:
                up_reshaped = mlp_components[f'layer_{start_id}_up'].view(-1, hidden_size)
                up_activations.append(up_reshaped.cpu())
            
            if f'layer_{start_id}_down' in mlp_components:
                # For down_proj, we need the input (which is up_proj output after SiLU)
                source_hidden = hidden_states[start_id - num_layer - 1]
                # Approximate down_proj input by applying gate and up transformations
                gate_out = mlp_components[f'layer_{start_id}_gate'].view(-1, hidden_size)
                up_out = mlp_components[f'layer_{start_id}_up'].view(-1, hidden_size)
                # SwiGLU: gate * SiLU(up)
                down_input = gate_out * torch.nn.functional.silu(up_out)
                down_input_activations.append(down_input.cpu())
            
            target_activations.append(target_reshaped.cpu())
            
            total_tokens += target_reshaped.shape[0]
            batch_count += 1
            
            # Process when we reach batch limit or end of data
            if batch_count >= batch_size_limit or batch_idx >= len(dataloader) - 1 or total_tokens >= max_tokens:
                print(f"\nProcessing chunk with {batch_count} batches, {total_tokens} tokens")
                
                # Concatenate collected activations
                chunk_transformations = {}
                
                if gate_activations:
                    gate_tensor = torch.cat(gate_activations, dim=0)
                    print(f"Gate activations shape: {gate_tensor.shape}")
                else:
                    gate_tensor = None
                
                if up_activations:
                    up_tensor = torch.cat(up_activations, dim=0)
                    print(f"Up activations shape: {up_tensor.shape}")
                else:
                    up_tensor = None
                    
                if down_input_activations:
                    down_input_tensor = torch.cat(down_input_activations, dim=0)
                    print(f"Down input activations shape: {down_input_tensor.shape}")
                else:
                    down_input_tensor = None
                
                target_tensor = torch.cat(target_activations, dim=0)
                print(f"Target activations shape: {target_tensor.shape}")
                
                # Perform optimization for this chunk
                if up_tensor is not None and down_input_tensor is not None:
                    print(f"Performing coupled optimization for chunk {len(all_transformations['up']) + 1}...")
                    print(f"Chunk data shapes - Up: {up_tensor.shape}, Down: {down_input_tensor.shape}, Target: {target_tensor.shape}")
                    T_up, T_down = coupled_optimization(
                        up_tensor, down_input_tensor, target_tensor, gate_importance,
                        chunk_idx=len(all_transformations['up']) + 1  # Pass chunk index
                    )
                    chunk_transformations['up'] = T_up
                    chunk_transformations['down'] = T_down
                    all_transformations['up'].append(T_up)
                    all_transformations['down'].append(T_down)
                else:
                    print(f"Performing single transformation for chunk {len(all_transformations['single']) + 1}...")
                    if down_input_tensor is not None:
                        source_tensor = down_input_tensor
                    elif up_tensor is not None:
                        source_tensor = up_tensor
                    else:
                        source_tensor = gate_tensor
                    
                    print(f"Single transform data shapes - Source: {source_tensor.shape}, Target: {target_tensor.shape}")
                    T_single = learn_single_transformation(
                        source_tensor, target_tensor, gate_importance,
                        chunk_idx=len(all_transformations['single']) + 1  # Pass chunk index
                    )
                    chunk_transformations['single'] = T_single
                    all_transformations['single'].append(T_single)
                
                # Clear activations to free memory
                gate_activations.clear()
                up_activations.clear()
                down_input_activations.clear()
                target_activations.clear()
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                print(f"Chunk processing complete, memory cleared")
                
                # Reset counters
                batch_count = 0
                
                # Break if we've collected enough tokens
                if total_tokens >= max_tokens:
                    print(f"Collected sufficient tokens ({total_tokens}), stopping")
                    break
    
    print(f"\nCollected {len(all_transformations.get('up', [])) + len(all_transformations.get('single', []))} transformation chunks")
    
    # Average transformations across chunks
    final_transformations = {}
    
    if all_transformations['up']:
        print("Averaging coupled transformations across chunks...")
        avg_T_up = torch.stack(all_transformations['up']).mean(dim=0)
        avg_T_down = torch.stack(all_transformations['down']).mean(dim=0)
        final_transformations['up'] = avg_T_up
        final_transformations['down'] = avg_T_down
        print(f"Averaged {len(all_transformations['up'])} coupled transformations")
    
    if all_transformations['single']:
        print("Averaging single transformations across chunks...")
        avg_T_single = torch.stack(all_transformations['single']).mean(dim=0)
        final_transformations['single'] = avg_T_single
        print(f"Averaged {len(all_transformations['single'])} single transformations")
    
    # Step 4: Apply transformations to model
    print("Applying transformations to model...")
    
    # Clean up memory before model manipulation
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation application
    print("Reloading model for transformation application...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformations
    target_layer = model.model.layers[start_id - num_layer - 1]
    
    if 'up' in transformations and 'down' in transformations:
        print("Applying coupled transformations...")
        # Apply up transformation to up_proj
        if hasattr(target_layer.mlp, 'up_proj'):
            original_up_weight = target_layer.mlp.up_proj.weight.data.to(torch.float64)
            new_up_weight = (transformations['up'].T.cpu().to(torch.float64) @ original_up_weight.cpu()).to(torch.bfloat16)
            target_layer.mlp.up_proj.weight.data = new_up_weight
            print("Applied transformation to up_proj")
        
        # Apply down transformation to down_proj
        if hasattr(target_layer.mlp, 'down_proj'):
            original_down_weight = target_layer.mlp.down_proj.weight.data.to(torch.float64)
            new_down_weight = (original_down_weight.cpu().to(torch.float64) @ transformations['down'].cpu().to(torch.float64)).to(torch.bfloat16)
            target_layer.mlp.down_proj.weight.data = new_down_weight
            print("Applied transformation to down_proj")
    
    elif 'single' in transformations:
        print("Applying single transformation to down_proj...")
        if hasattr(target_layer.mlp, 'down_proj'):
            original_weight = target_layer.mlp.down_proj.weight.data.to(torch.float64)
            new_weight = (original_weight.cpu().to(torch.float64) @ transformations['single'].cpu().to(torch.float64)).to(torch.bfloat16)
            target_layer.mlp.down_proj.weight.data = new_weight
            print("Applied single transformation to down_proj")
    
    # Step 5: Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_save_path = f"{save_path}_MALT"
    print(f"Saving MALT model to: {final_save_path}")
    
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    if save_transform_only:
        torch.save(transformations, f"{final_save_path}_transformations.pth")
        print("Saved transformation matrices")
    
    # Final cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("=== MALT compression complete ===")
    return final_save_path


def learn_single_transformation(
    source_activations: torch.Tensor,
    target_activations: torch.Tensor, 
    gate_importance: torch.Tensor,
    max_epochs: int = 10,
    lr: float = 1e-4,
    chunk_idx: int = 0
) -> torch.Tensor:
    """Learn single transformation matrix with gate importance weighting."""
    print(f"Learning single transformation with gate importance weighting for chunk {chunk_idx}...")
    print(f"Input shapes - Source: {source_activations.shape}, Target: {target_activations.shape}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = source_activations.shape[-1]
    
    T = torch.eye(hidden_size, requires_grad=True, device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.Adam([T], lr=lr)
    
    source_activations = source_activations.to(device).to(torch.bfloat16)
    target_activations = target_activations.to(device).to(torch.bfloat16)
    gate_importance = gate_importance.to(device).to(torch.bfloat16)
    
    def weighted_cosine_loss(pred, target, weights):
        # Ensure pred and target have same shape
        if pred.shape[0] != target.shape[0]:
            min_size = min(pred.shape[0], target.shape[0])
            pred = pred[:min_size]
            target = target[:min_size]
            print(f"[SINGLE LOSS DEBUG] Trimmed to common size: {min_size}")
        
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        # Use scalar weight instead of per-token weighting
        weight_scalar = weights.mean()  # Convert to scalar
        return ((1 - cosine_sim) * weight_scalar).mean()
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred = source_activations @ T
        loss = weighted_cosine_loss(pred, target_activations, gate_importance)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Single transform epoch {epoch}: Loss: {loss.item():.4f}")
    
    print("Single transformation learning complete")
    return T.detach()

