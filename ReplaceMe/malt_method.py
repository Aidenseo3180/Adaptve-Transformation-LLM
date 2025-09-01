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
        gate_activations.append(output.detach().cpu())
    
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
                # Average across layers and tokens
                batch_gate_avg = torch.stack(gate_activations).mean(dim=[0, 1])
                if batch_idx == 0:
                    accumulated_gates = batch_gate_avg
                    accumulated_variance = batch_gate_avg ** 2
                else:
                    accumulated_gates += batch_gate_avg
                    accumulated_variance += batch_gate_avg ** 2
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate importance scores
    mean_activation = accumulated_gates / (batch_idx + 1)
    variance = (accumulated_variance / (batch_idx + 1)) - mean_activation ** 2
    
    # Importance = variance * absolute mean (captures both diversity and magnitude)
    gate_importance = variance * torch.abs(mean_activation)
    gate_importance = gate_importance / gate_importance.max()  # Normalize
    
    print(f"Gate importance analysis complete. Max importance: {gate_importance.max():.4f}")
    print(f"Mean importance: {gate_importance.mean():.4f}")
    
    return gate_importance


def coupled_optimization(
    up_activations: torch.Tensor,
    down_activations: torch.Tensor, 
    targets: torch.Tensor,
    gate_importance: torch.Tensor,
    max_epochs: int = 10,
    lr: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform coupled optimization of up_proj and down_proj transformations.
    
    Args:
        up_activations: Up projection activations [N, hidden_size]
        down_activations: Down projection activations [N, hidden_size] 
        targets: Target activations [N, hidden_size]
        gate_importance: Gate importance scores [hidden_size]
        max_epochs: Maximum optimization epochs
        lr: Learning rate
        
    Returns:
        T_up: Up projection transformation matrix
        T_down: Down projection transformation matrix
    """
    print("Starting coupled optimization...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = up_activations.shape[-1]
    
    # Initialize transformation matrices
    T_up = torch.eye(hidden_size, requires_grad=True, device=device, dtype=torch.float32)
    T_down = torch.eye(hidden_size, requires_grad=True, device=device, dtype=torch.float32)
    
    # Move data to device
    up_activations = up_activations.to(device).float()
    down_activations = down_activations.to(device).float()
    targets = targets.to(device).float()
    gate_importance = gate_importance.to(device).float()
    
    # Optimizers
    optimizer_up = torch.optim.Adam([T_up], lr=lr)
    optimizer_down = torch.optim.Adam([T_down], lr=lr)
    optimizer_joint = torch.optim.Adam([T_up, T_down], lr=lr/2)
    
    def cosine_loss_weighted(pred, target, weights):
        """Weighted cosine loss using gate importance"""
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        # Weight by gate importance (broadcast across batch)
        weighted_loss = (1 - cosine_sim) * weights.mean()  # Simple weighting for now
        return weighted_loss.mean()
    
    def consistency_loss(T_up, T_down):
        """Consistency regularization between transformations"""
        # Encourage T_up and T_down to be approximately inverse-related
        product = T_up.T @ T_down
        identity = torch.eye(hidden_size, device=device)
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
            mlp_components[f'{name}_{component_type}'] = output.detach()
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
    
    print(f"Collecting activations for up to {max_tokens} tokens...")
    
    with torch.no_grad():
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
            target_reshaped = target_hidden.view(-1, hidden_size)
            
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
            if total_tokens >= max_tokens:
                print(f"Collected {total_tokens} tokens, stopping collection")
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"Collected {len(target_activations)} batches of activations")
    
    # Concatenate all collected activations
    if gate_activations:
        gate_tensor = torch.cat(gate_activations, dim=0)
        print(f"Gate activations shape: {gate_tensor.shape}")
    else:
        gate_tensor = None
        print("No gate activations collected")
    
    if up_activations:
        up_tensor = torch.cat(up_activations, dim=0)
        print(f"Up activations shape: {up_tensor.shape}")
    else:
        up_tensor = None
        print("No up activations collected")
        
    if down_input_activations:
        down_input_tensor = torch.cat(down_input_activations, dim=0)
        print(f"Down input activations shape: {down_input_tensor.shape}")
    else:
        down_input_tensor = None
        print("No down input activations collected")
    
    target_tensor = torch.cat(target_activations, dim=0)
    print(f"Target activations shape: {target_tensor.shape}")
    
    # Step 3: Coupled optimization
    transformations = {}
    
    if up_tensor is not None and down_input_tensor is not None:
        print("Performing coupled up/down optimization...")
        T_up, T_down = coupled_optimization(
            up_tensor, down_input_tensor, target_tensor, gate_importance
        )
        transformations['up'] = T_up
        transformations['down'] = T_down
    else:
        print("Insufficient activations for coupled optimization, falling back to single transformation")
        # Fallback: use available activations for single transformation
        if down_input_tensor is not None:
            source_tensor = down_input_tensor
        elif up_tensor is not None:
            source_tensor = up_tensor
        else:
            source_tensor = gate_tensor
            
        T_single = learn_single_transformation(source_tensor, target_tensor, gate_importance)
        transformations['single'] = T_single
    
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
    lr: float = 1e-4
) -> torch.Tensor:
    """Learn single transformation matrix with gate importance weighting."""
    print("Learning single transformation with gate importance weighting...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = source_activations.shape[-1]
    
    T = torch.eye(hidden_size, requires_grad=True, device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam([T], lr=lr)
    
    source_activations = source_activations.to(device).float()
    target_activations = target_activations.to(device).float()
    gate_importance = gate_importance.to(device).float()
    
    def weighted_cosine_loss(pred, target, weights):
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        return ((1 - cosine_sim) * weights.mean()).mean()
    
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

