"""Magnitude-Aware Progressive Pruning (MAPP) Module

Implements intelligent layer pruning with residual redistribution and cascade calibration.
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import numpy as np
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from copy import deepcopy

from .utils import get_calib_dataloader, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class ResidualRedistribution:
    """Handles residual redistribution when removing layers."""
    
    def __init__(self, model, calibration_loader):
        self.model = model
        self.calibration_loader = calibration_loader
        self.residual_stats = {}
        
    def measure_residual(self, layer_idx):
        """Measure the residual contribution of a specific layer."""
        
        print(f"{Fore.YELLOW}[MAPP] Measuring residual for layer {layer_idx}...{Fore.RESET}")
        
        residuals = []
        layer = self.model.model.layers[layer_idx]
        
        with torch.no_grad():
            for batch_idx, batch_text in enumerate(self.calibration_loader):
                if batch_idx >= 10:  # Use first 10 batches
                    break
                
                # Get inputs
                inputs = self.model.tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                # Forward pass up to this layer
                hidden = self.model.model.embed_tokens(inputs['input_ids'])
                for i in range(layer_idx):
                    hidden = self.model.model.layers[i](hidden)[0]
                
                # Get layer output
                layer_output = layer(hidden)[0]
                
                # Calculate residual (what this layer adds)
                residual = layer_output - hidden
                
                # Store statistics
                residuals.append({
                    'mean': residual.mean(dim=[0, 1]).cpu(),
                    'std': residual.std(dim=[0, 1]).cpu(),
                    'norm': residual.norm(dim=-1).mean().item()
                })
        
        # Aggregate statistics
        self.residual_stats[layer_idx] = {
            'mean_vector': torch.stack([r['mean'] for r in residuals]).mean(dim=0),
            'std_vector': torch.stack([r['std'] for r in residuals]).mean(dim=0),
            'avg_norm': np.mean([r['norm'] for r in residuals])
        }
        
        print(f"[MAPP] Layer {layer_idx} avg residual norm: {self.residual_stats[layer_idx]['avg_norm']:.4f}")
        
        return self.residual_stats[layer_idx]
    
    def redistribute(self, layer_to_remove):
        """Redistribute the residual of layer_to_remove to adjacent layers."""
        
        print(f"{Fore.CYAN}[MAPP] Redistributing residual from layer {layer_to_remove}...{Fore.RESET}")
        
        if layer_to_remove not in self.residual_stats:
            self.measure_residual(layer_to_remove)
        
        stats = self.residual_stats[layer_to_remove]
        mean_residual = stats['mean_vector'].to(self.model.device)
        
        # Add compensation to previous layer
        if layer_to_remove > 0:
            prev_layer = self.model.model.layers[layer_to_remove - 1]
            
            # Create a compensation module
            class CompensationModule(nn.Module):
                def __init__(self, compensation_vector):
                    super().__init__()
                    self.register_buffer('compensation', compensation_vector * 0.5)
                
                def forward(self, hidden_states):
                    return hidden_states + self.compensation
            
            # Wrap the previous layer
            original_forward = prev_layer.forward
            compensation_module = CompensationModule(mean_residual).to(self.model.device)
            
            def compensated_forward(hidden_states, **kwargs):
                output = original_forward(hidden_states, **kwargs)
                if isinstance(output, tuple):
                    return (compensation_module(output[0]),) + output[1:]
                return compensation_module(output)
            
            prev_layer.forward = compensated_forward
            print(f"[MAPP] Added compensation to layer {layer_to_remove - 1}")
        
        # Add compensation to next layer input
        if layer_to_remove < len(self.model.model.layers) - 1:
            next_layer = self.model.model.layers[layer_to_remove + 1]
            
            # Create input compensation
            input_compensation = CompensationModule(mean_residual).to(self.model.device)
            original_forward_next = next_layer.forward
            
            def compensated_forward_next(hidden_states, **kwargs):
                hidden_states = input_compensation(hidden_states)
                return original_forward_next(hidden_states, **kwargs)
            
            next_layer.forward = compensated_forward_next
            print(f"[MAPP] Added input compensation to layer {layer_to_remove + 1}")


class LayerImportanceAnalyzer:
    """Analyzes and ranks layer importance."""
    
    def __init__(self, model, calibration_loader):
        self.model = model
        self.calibration_loader = calibration_loader
        self.importance_scores = {}
        
    def compute_importance(self, layer_idx):
        """Compute importance score for a specific layer."""
        
        print(f"[MAPP] Computing importance for layer {layer_idx}...")
        
        layer = self.model.model.layers[layer_idx]
        
        # Store original forward
        original_forward = layer.forward
        
        # Create identity forward
        def identity_forward(hidden_states, **kwargs):
            return (hidden_states,) if kwargs.get('use_cache', False) else hidden_states
        
        total_change = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch_text in enumerate(self.calibration_loader):
                if batch_idx >= 5:  # Use first 5 batches for speed
                    break
                
                inputs = self.model.tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.model.device)
                
                # Forward with layer
                outputs_with = self.model(**inputs, output_hidden_states=True)
                logits_with = outputs_with.logits
                
                # Forward without layer (identity)
                layer.forward = identity_forward
                outputs_without = self.model(**inputs, output_hidden_states=True)
                logits_without = outputs_without.logits
                
                # Restore original forward
                layer.forward = original_forward
                
                # Calculate change
                change = (logits_with - logits_without).abs().mean().item()
                total_change += change
                num_samples += 1
        
        importance = total_change / max(num_samples, 1)
        self.importance_scores[layer_idx] = importance
        
        print(f"[MAPP] Layer {layer_idx} importance: {importance:.6f}")
        
        return importance
    
    def analyze_all_layers(self):
        """Analyze importance of all layers."""
        
        print(f"{Fore.CYAN}[MAPP] Analyzing all layers...{Fore.RESET}")
        
        for layer_idx in tqdm(range(len(self.model.model.layers)), desc="Layer analysis"):
            if layer_idx not in self.importance_scores:
                self.compute_importance(layer_idx)
        
        return self.importance_scores
    
    def get_removal_candidates(self, pruning_ratio):
        """Get layers to remove based on importance scores."""
        
        if not self.importance_scores:
            self.analyze_all_layers()
        
        # Sort by importance (ascending - less important first)
        sorted_layers = sorted(self.importance_scores.items(), key=lambda x: x[1])
        
        # Calculate how many to remove
        total_layers = len(self.model.model.layers)
        num_to_remove = int(total_layers * pruning_ratio)
        
        # Select candidates (avoid first and last 2 layers)
        candidates = []
        for layer_idx, importance in sorted_layers:
            if 2 <= layer_idx <= total_layers - 3:  # Keep first 2 and last 2 layers
                candidates.append(layer_idx)
                if len(candidates) >= num_to_remove:
                    break
        
        print(f"{Fore.GREEN}[MAPP] Selected {len(candidates)} layers for removal: {candidates}{Fore.RESET}")
        
        return candidates


class SafeLayerRemover:
    """Safely removes layers with performance monitoring."""
    
    def __init__(self, model, calibration_loader, max_perplexity_increase=1.2):
        self.model = model
        self.calibration_loader = calibration_loader
        self.max_perplexity_increase = max_perplexity_increase
        self.original_perplexity = None
        
    def compute_perplexity(self):
        """Compute model perplexity on calibration data."""
        
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_text in enumerate(self.calibration_loader):
                if batch_idx >= 10:
                    break
                
                inputs = self.model.tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.model.device)
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        return perplexity
    
    def remove_layer(self, layer_idx):
        """Remove a single layer and adjust model."""
        
        print(f"{Fore.YELLOW}[MAPP] Removing layer {layer_idx}...{Fore.RESET}")
        
        # Create new layer list without the target layer
        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.model.model.layers):
            if i != layer_idx:
                new_layers.append(layer)
        
        # Update model
        self.model.model.layers = new_layers
        self.model.config.num_hidden_layers = len(new_layers)
        
        print(f"[MAPP] Layer {layer_idx} removed. Model now has {len(new_layers)} layers.")
    
    def try_remove(self, layer_idx, redistributor):
        """Try to remove a layer and check if performance is acceptable."""
        
        print(f"{Fore.CYAN}[MAPP] Attempting to remove layer {layer_idx}...{Fore.RESET}")
        
        # Compute baseline perplexity if not done
        if self.original_perplexity is None:
            self.original_perplexity = self.compute_perplexity()
            print(f"[MAPP] Baseline perplexity: {self.original_perplexity:.2f}")
        
        # Apply redistribution
        redistributor.redistribute(layer_idx)
        
        # Create backup
        original_layers = deepcopy(self.model.model.layers)
        original_config = deepcopy(self.model.config)
        
        # Remove layer
        self.remove_layer(layer_idx)
        
        # Test new perplexity
        new_perplexity = self.compute_perplexity()
        perplexity_ratio = new_perplexity / self.original_perplexity
        
        print(f"[MAPP] New perplexity: {new_perplexity:.2f} (ratio: {perplexity_ratio:.3f})")
        
        if perplexity_ratio <= self.max_perplexity_increase:
            print(f"{Fore.GREEN}[MAPP] Layer {layer_idx} successfully removed!{Fore.RESET}")
            self.original_perplexity = new_perplexity  # Update baseline
            return True
        else:
            # Restore model
            print(f"{Fore.RED}[MAPP] Performance drop too large. Restoring layer {layer_idx}...{Fore.RESET}")
            self.model.model.layers = original_layers
            self.model.config = original_config
            return False


def mapp_transform(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    pruning_ratio: float = 0.25,
    max_perplexity_increase: float = 1.2,
    **kwargs
) -> str:
    """Main MAPP transformation function."""
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"[MAPP] Magnitude-Aware Progressive Pruning")
    print(f"[MAPP] Target pruning ratio: {pruning_ratio:.1%}")
    print(f"[MAPP] Max perplexity increase: {max_perplexity_increase:.1f}x")
    print(f"{'='*60}{Fore.RESET}")
    
    # Load model
    print(f"[MAPP] Loading model: {model_path}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.tokenizer = tokenizer
    model.eval()
    
    # Get calibration data
    print(f"[MAPP] Loading calibration dataset")
    calibration_loader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size or 1000,
        batch_size,
        tokenizer
    )
    
    # Initialize components
    analyzer = LayerImportanceAnalyzer(model, calibration_loader)
    redistributor = ResidualRedistribution(model, calibration_loader)
    remover = SafeLayerRemover(model, calibration_loader, max_perplexity_increase)
    
    # Analyze layers
    print(f"\n{Fore.CYAN}[MAPP] Phase 1: Layer Analysis{Fore.RESET}")
    importance_scores = analyzer.analyze_all_layers()
    
    # Get removal candidates
    candidates = analyzer.get_removal_candidates(pruning_ratio)
    
    # Progressive removal
    print(f"\n{Fore.CYAN}[MAPP] Phase 2: Progressive Removal{Fore.RESET}")
    removed_layers = []
    failed_attempts = []
    
    for layer_idx in candidates:
        # Measure residual before removal
        redistributor.measure_residual(layer_idx)
        
        # Try to remove
        if remover.try_remove(layer_idx, redistributor):
            removed_layers.append(layer_idx)
        else:
            failed_attempts.append(layer_idx)
    
    # Report results
    print(f"\n{Fore.GREEN}[MAPP] Removal Summary:")
    print(f"  Successfully removed: {len(removed_layers)} layers")
    print(f"  Failed attempts: {len(failed_attempts)} layers")
    print(f"  Final layer count: {len(model.model.layers)}")
    print(f"  Actual pruning ratio: {len(removed_layers)/32:.1%}{Fore.RESET}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        model_name = model_path.split('/')[-1]
        save_path = f"output_models/{model_name}_MAPP_p{int(pruning_ratio*100)}"
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n[MAPP] Saving pruned model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save MAPP configuration
    mapp_config = {
        'method': 'mapp',
        'original_layers': 32,  # Assuming Llama-3.2-3B
        'removed_layers': removed_layers,
        'failed_attempts': failed_attempts,
        'final_layers': len(model.model.layers),
        'pruning_ratio': len(removed_layers) / 32,
        'importance_scores': {str(k): float(v) for k, v in importance_scores.items()},
        'residual_stats': {
            str(k): {
                'avg_norm': float(v['avg_norm'])
            } for k, v in redistributor.residual_stats.items()
        }
    }
    
    with open(f"{save_path}/mapp_config.json", 'w') as f:
        json.dump(mapp_config, f, indent=2)
    
    print(f"{Fore.GREEN}[MAPP] Pruning complete!{Fore.RESET}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path