"""Distributed Down-Projection Compensation (DDPC) for layer pruning.

This module implements a novel approach that distributes compensation across
multiple down_proj layers to better preserve model performance after pruning.
"""

import argparse
import gc
import logging
import os
from typing import Dict, List, Optional, Tuple
import json

import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, seed_all, 
                    select_non_overlapping_blocks, truncate_model)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def ddpc(
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
    num_compensation_points: int = 3,
    compensation_epochs: int = 10,
    compensation_lr: float = 1e-4,
    **kwargs
) -> str:
    """Distributed Down-Projection Compensation for layer pruning.
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset name for calibration
        dataset_column: Column containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip/prune
        dataset_size: Size of calibration dataset
        dataset_subset: Dataset subset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save pruned model
        token: HuggingFace token
        distances_path: Path to layer distances file
        num_A: Number of blocks to prune
        merge_consecutive: Whether to merge consecutive blocks
        num_compensation_points: Number of layers to distribute compensation
        compensation_epochs: Training epochs for each compensation
        compensation_lr: Learning rate for compensation training
    
    Returns:
        Path to saved model
    """
    
    print(f"\n{Fore.GREEN}=== Starting DDPC (Distributed Down-Projection Compensation) ==={Fore.RESET}")
    print(f"{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  - Model: {model_path}")
    print(f"  - Layers to skip: {layers_to_skip}")
    print(f"  - Compensation points: {num_compensation_points}")
    print(f"  - Training epochs: {compensation_epochs}")
    
    # Load model with appropriate configuration
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Phase 1: Load model and select blocks to prune
    print(f"\n{Fore.YELLOW}Phase 1: Loading model and selecting pruning blocks{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Load distances and select blocks
    average_distances = torch.load(distances_path, weights_only=False)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    print(f"{Fore.GREEN}Selected blocks to prune: {selected_blocks}{Fore.RESET}")
    
    # Process each block
    all_compensations = []
    
    for block_idx, (start_id, end_id) in enumerate(selected_blocks):
        print(f"\n{Fore.CYAN}Processing block {block_idx+1}/{len(selected_blocks)}: "
              f"Layers {start_id} to {end_id}{Fore.RESET}")
        
        # Phase 2: Profile activations before pruning
        print(f"{Fore.YELLOW}Phase 2: Profiling activations (before pruning){Fore.RESET}")
        
        original_activations = profile_model_activations(
            model, tokenizer, dataset, dataset_subset, dataset_column,
            dataset_size, batch_size, max_length, start_id, end_id
        )
        
        # Phase 3: Compute residual impacts
        print(f"{Fore.YELLOW}Phase 3: Computing residual impacts{Fore.RESET}")
        
        residual_impacts = compute_residual_impacts(
            model, tokenizer, dataset, dataset_subset, dataset_column,
            dataset_size, batch_size, max_length, start_id, end_id,
            original_activations
        )
        
        # Phase 4: Select compensation points
        print(f"{Fore.YELLOW}Phase 4: Selecting compensation points{Fore.RESET}")
        
        compensation_points = select_compensation_points(
            start_id, end_id, model.config.num_hidden_layers,
            num_compensation_points, residual_impacts
        )
        
        print(f"{Fore.GREEN}Compensation points selected: {[p['layer'] for p in compensation_points]}{Fore.RESET}")
        
        # Phase 5: Train compensation transforms
        print(f"{Fore.YELLOW}Phase 5: Training compensation transforms{Fore.RESET}")
        
        compensations = train_compensations(
            model, tokenizer, dataset, dataset_subset, dataset_column,
            dataset_size, batch_size, max_length,
            start_id, end_id, compensation_points,
            residual_impacts, compensation_epochs, compensation_lr
        )
        
        all_compensations.append({
            'block': (start_id, end_id),
            'compensations': compensations
        })
    
    # Phase 6: Apply all compensations and prune
    print(f"\n{Fore.YELLOW}Phase 6: Applying compensations and pruning{Fore.RESET}")
    
    # Reload model in CPU for modification
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    # Apply compensations
    for comp_data in all_compensations:
        start_id, end_id = comp_data['block']
        
        for comp in comp_data['compensations']:
            layer_idx = comp['layer']
            transform = comp['transform']
            
            print(f"Applying compensation to layer {layer_idx}")
            
            # Apply transform to down_proj
            original_weight = model.model.layers[layer_idx].mlp.down_proj.weight.to(torch.float64)
            new_weight = (transform.T.cpu() @ original_weight)
            model.model.layers[layer_idx].mlp.down_proj.weight.data = new_weight.to(torch.bfloat16)
        
        # Prune the block
        print(f"Pruning layers {start_id} to {end_id}")
        model = truncate_model(model, start_id, end_id)
    
    # Save model
    if save_path is None:
        save_path = f"output_models/{model_path.replace('/', '_')}_DDPC_{layers_to_skip}layers"
    
    print(f"\n{Fore.GREEN}Saving pruned model to {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save compensation info for debugging
    comp_info_path = f"{save_path}/compensation_info.json"
    with open(comp_info_path, 'w') as f:
        json.dump({
            'method': 'DDPC',
            'pruned_blocks': selected_blocks,
            'compensation_points': num_compensation_points,
            'epochs': compensation_epochs,
            'learning_rate': compensation_lr
        }, f, indent=2)
    
    print(f"{Fore.GREEN}=== DDPC Complete ==={Fore.RESET}")
    
    return save_path


def profile_model_activations(
    model, tokenizer, dataset, dataset_subset, dataset_column,
    dataset_size, batch_size, max_length, start_id, end_id
) -> Dict[int, torch.Tensor]:
    """Profile model activations for specified layers."""
    
    print(f"  Collecting activations for layers {start_id} to {end_id}")
    
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    activations = {}
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    
    # Initialize storage
    for layer_idx in range(num_layers):
        activations[layer_idx] = []
    
    # Collect activations
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Profiling", colour="blue")):
        if batch_idx * batch_size >= dataset_size:
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
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            
            for layer_idx, hidden_state in enumerate(hidden_states):
                # Average pooling over sequence length
                pooled = hidden_state.mean(dim=1)  # [batch, hidden_size]
                activations[layer_idx].append(pooled.cpu())
    
    # Concatenate all batches
    for layer_idx in activations:
        activations[layer_idx] = torch.cat(activations[layer_idx], dim=0)
    
    print(f"  Collected activations shape: {activations[0].shape}")
    
    return activations


def compute_residual_impacts(
    model, tokenizer, dataset, dataset_subset, dataset_column,
    dataset_size, batch_size, max_length, start_id, end_id,
    original_activations
) -> Dict[int, torch.Tensor]:
    """Compute the impact of removing layers on residual stream."""
    
    print(f"  Computing residual impacts after removing layers {start_id}-{end_id}")
    
    # Temporarily remove layers
    model_copy = model
    original_layers = model_copy.model.layers[start_id:end_id]
    model_copy.model.layers = nn.ModuleList([
        layer for idx, layer in enumerate(model_copy.model.layers)
        if idx < start_id or idx >= end_id
    ])
    
    # Collect new activations
    pruned_activations = profile_model_activations(
        model_copy, tokenizer, dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, max_length, start_id, end_id
    )
    
    # Restore layers
    model_copy.model.layers = nn.ModuleList(
        list(model_copy.model.layers[:start_id]) +
        list(original_layers) +
        list(model_copy.model.layers[start_id:])
    )
    
    # Compute impacts
    impacts = {}
    for layer_idx in range(model.config.num_hidden_layers):
        if layer_idx in original_activations and layer_idx in pruned_activations:
            impact = original_activations[layer_idx] - pruned_activations[layer_idx]
            impacts[layer_idx] = impact
            
            # Debug info
            impact_norm = impact.norm(dim=1).mean().item()
            print(f"    Layer {layer_idx} impact norm: {impact_norm:.4f}")
    
    return impacts


def select_compensation_points(
    start_id: int,
    end_id: int,
    num_layers: int,
    num_points: int,
    residual_impacts: Dict[int, torch.Tensor]
) -> List[Dict]:
    """Select optimal layers for distributing compensation."""
    
    compensation_points = []
    
    # Strategy: Use layers before, after, and at intervals
    candidates = []
    
    # 1. Layer right before pruned block (highest weight)
    if start_id > 0:
        candidates.append({
            'layer': start_id - 1,
            'weight': 0.5,
            'type': 'before'
        })
    
    # 2. Layer right after pruned block
    if end_id < num_layers:
        candidates.append({
            'layer': end_id,
            'weight': 0.3,
            'type': 'after'
        })
    
    # 3. Additional layers further away
    if end_id + 3 < num_layers:
        candidates.append({
            'layer': end_id + 3,
            'weight': 0.2,
            'type': 'distant'
        })
    
    # Normalize weights
    total_weight = sum(c['weight'] for c in candidates)
    for c in candidates:
        c['weight'] /= total_weight
    
    # Take up to num_points
    compensation_points = candidates[:num_points]
    
    return compensation_points


def train_compensations(
    model, tokenizer, dataset, dataset_subset, dataset_column,
    dataset_size, batch_size, max_length,
    start_id, end_id, compensation_points,
    residual_impacts, epochs, learning_rate
) -> List[Dict]:
    """Train compensation transforms for each point."""
    
    print(f"  Training {len(compensation_points)} compensation transforms")
    
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        min(dataset_size, 1000),  # Use subset for training
        batch_size, tokenizer
    )
    
    trained_compensations = []
    hidden_size = model.config.hidden_size
    
    for point in compensation_points:
        layer_idx = point['layer']
        weight = point['weight']
        
        print(f"\n  Training compensation for layer {layer_idx} (weight: {weight:.2f})")
        
        # Initialize transform as identity
        transform = torch.eye(hidden_size, dtype=torch.float32, requires_grad=True)
        if torch.cuda.is_available():
            transform = transform.cuda()
        
        optimizer = torch.optim.Adam([transform], lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
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
                    
                    # Get activation at compensation point
                    if layer_idx < len(hidden_states):
                        activation = hidden_states[layer_idx].mean(dim=1)  # Pool over sequence
                        
                        # Target is weighted portion of residual impact
                        if layer_idx in residual_impacts:
                            target = weight * residual_impacts[layer_idx][:activation.shape[0]]
                            target = target.to(activation.device).to(torch.float32)
                            
                            # Compute transformed activation
                            activation = activation.to(torch.float32)
                            transformed = activation @ transform
                            
                            # Cosine similarity loss
                            cos_sim = nn.functional.cosine_similarity(
                                transformed,
                                activation + target,
                                dim=1
                            )
                            loss = 1 - cos_sim.mean()
                            
                            # Backward pass
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                            num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            if epoch % 2 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Store trained compensation
        trained_compensations.append({
            'layer': layer_idx,
            'weight': weight,
            'transform': transform.detach().cpu()
        })
    
    return trained_compensations

