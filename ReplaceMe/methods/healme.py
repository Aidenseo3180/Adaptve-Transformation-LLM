"""Self-healing ReplaceMe module for transformer model optimization.

This module implements a self-healing version of ReplaceMe that can detect performance
degradation in real-time and apply micro-corrections during inference.
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import numpy as np

from ..utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_non_overlapping_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class HealableLinear(nn.Module):
    """Self-healing wrapper for Linear layers that can apply micro-corrections."""
    
    def __init__(self, original_layer: nn.Module, healing_config: dict):
        super().__init__()
        print(f"[HealableLinear] Initializing healable layer wrapper")
        
        # Store original layer
        self.original_layer = original_layer
        
        # Healing configuration
        self.healing_enabled = healing_config.get('enabled', True)
        self.perplexity_threshold = healing_config.get('threshold', 15.0)
        self.history_length = healing_config.get('history_length', 10)
        self.prediction_window = healing_config.get('prediction_window', 3)
        
        # State tracking
        self.perplexity_history = []
        self.correction_matrix = None
        self.healing_count = 0
        self.total_forward_calls = 0
        
        # Store original transformation for reference
        self.original_transform = healing_config.get('transform', None)
        
        print(f"[HealableLinear] Healing threshold: {self.perplexity_threshold}")
        print(f"[HealableLinear] History length: {self.history_length}")
        print(f"[HealableLinear] Prediction window: {self.prediction_window}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional healing correction."""
        self.total_forward_calls += 1
        
        # Always compute original output first
        output = self.original_layer(x)
        
        # Apply healing correction if needed and enabled
        if self.healing_enabled and self.needs_healing():
            print(f"[HealableLinear] Applying healing correction (call #{self.total_forward_calls})")
            
            if self.correction_matrix is None:
                print("[HealableLinear] Computing initial correction matrix")
                self._compute_correction_matrix(x, output)
            
            # Apply rank-1 correction
            if self.correction_matrix is not None:
                correction = x @ self.correction_matrix
                output = output + correction
                self.healing_count += 1
                print(f"[HealableLinear] Applied correction #{self.healing_count}")
        
        return output
    
    def needs_healing(self) -> bool:
        """Predict if healing will be needed based on perplexity trend."""
        if len(self.perplexity_history) < self.prediction_window:
            return False
        
        # Get recent perplexity values
        recent_perplexities = self.perplexity_history[-self.prediction_window:]
        current_avg = np.mean(recent_perplexities)
        
        # Simple trend analysis: if recent average exceeds threshold
        needs_healing = current_avg > self.perplexity_threshold
        
        if needs_healing:
            print(f"[HealableLinear] Healing needed - avg perplexity: {current_avg:.3f} > {self.perplexity_threshold}")
            
            # Compute trend slope for additional context
            if len(recent_perplexities) >= 2:
                x_vals = np.arange(len(recent_perplexities))
                slope = np.polyfit(x_vals, recent_perplexities, 1)[0]
                print(f"[HealableLinear] Perplexity trend slope: {slope:.3f}")
        
        return needs_healing
    
    def update_perplexity(self, perplexity: float):
        """Update perplexity history for trend analysis."""
        self.perplexity_history.append(perplexity)
        
        # Keep history within specified length
        if len(self.perplexity_history) > self.history_length:
            self.perplexity_history.pop(0)
        
        print(f"[HealableLinear] Updated perplexity: {perplexity:.3f} "
              f"(history: {len(self.perplexity_history)})")
    
    def _compute_correction_matrix(self, input_activation: torch.Tensor, 
                                 current_output: torch.Tensor):
        """Compute rank-1 correction matrix based on current state."""
        print("[HealableLinear] Computing correction matrix")
        
        if self.original_transform is None:
            print("[HealableLinear] No original transform available, using identity correction")
            # Use small identity-based correction as fallback
            correction_scale = 0.1
            self.correction_matrix = correction_scale * torch.eye(
                input_activation.shape[-1], 
                device=input_activation.device,
                dtype=input_activation.dtype
            )
            return
        
        try:
            # Compute what the output should be with original transform
            target_output = input_activation @ self.original_transform.to(
                input_activation.device).to(input_activation.dtype)
            
            # Compute difference
            output_diff = target_output - current_output
            
            # Use SVD to get rank-1 approximation of the correction
            # Reshape for SVD if needed
            if output_diff.dim() > 2:
                batch_size = output_diff.shape[0]
                seq_len = output_diff.shape[1]
                hidden_size = output_diff.shape[2]
                diff_2d = output_diff.view(-1, hidden_size)
            else:
                diff_2d = output_diff
            
            # Compute SVD
            U, S, V = torch.svd(diff_2d.float())
            
            # Take only the first component for rank-1 approximation
            if S.numel() > 0:
                correction_strength = 0.1  # Scale down the correction
                u1 = U[:, 0:1]  # First left singular vector
                s1 = S[0] * correction_strength  # First singular value (scaled)
                v1 = V[:, 0:1]  # First right singular vector
                
                # Create rank-1 correction matrix: input_dim x output_dim
                self.correction_matrix = s1 * v1 @ u1.t()
                self.correction_matrix = self.correction_matrix.to(input_activation.dtype)
                
                print(f"[HealableLinear] Computed rank-1 correction with strength {s1:.6f}")
            else:
                print("[HealableLinear] SVD returned empty singular values, using zero correction")
                self.correction_matrix = torch.zeros(
                    input_activation.shape[-1], 
                    current_output.shape[-1],
                    device=input_activation.device,
                    dtype=input_activation.dtype
                )
                
        except Exception as e:
            print(f"[HealableLinear] Error computing correction matrix: {e}")
            # Fallback to zero correction
            self.correction_matrix = torch.zeros(
                input_activation.shape[-1], 
                current_output.shape[-1] if current_output.dim() >= 2 else input_activation.shape[-1],
                device=input_activation.device,
                dtype=input_activation.dtype
            )
    
    def get_healing_stats(self) -> dict:
        """Return statistics about healing performance."""
        return {
            'total_forward_calls': self.total_forward_calls,
            'healing_applications': self.healing_count,
            'healing_rate': self.healing_count / max(1, self.total_forward_calls),
            'current_perplexity_history': self.perplexity_history.copy(),
            'avg_recent_perplexity': np.mean(self.perplexity_history[-3:]) if len(self.perplexity_history) >= 3 else None
        }


def healme(
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
    healing_threshold: float = 15.0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    save_transform_only: bool = False,
    solver: str = "adam",
    loss: str = "cosine",
    diag: bool = False,
    two_vectors: bool = False,
    thri: bool = False,
    accurate: bool = False
) -> str:
    """Apply self-healing ReplaceMe optimization to transformer model.
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset name for calibration
        dataset_column: Column containing text data
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip in each block
        dataset_size: Size of calibration dataset
        dataset_subset: Dataset subset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save the model
        token: HuggingFace token
        healing_threshold: Perplexity threshold for triggering healing
        distances_path: Path to precomputed distances
        num_A: Number of blocks to process
        merge_consecutive: Whether to merge consecutive blocks
        save_transform_only: Whether to save only transformations
        solver: Optimization solver to use
        loss: Loss function type
        diag: Whether to use diagonal matrix
        two_vectors: Whether to use two-vector factorization
        thri: Whether to use triangular matrix
        accurate: Whether to use accurate mode
        
    Returns:
        Path where the healed model is saved
    """
    print(f"[healme] Starting self-healing ReplaceMe optimization")
    print(f"[healme] Model: {model_path}")
    print(f"[healme] Healing threshold: {healing_threshold}")
    print(f"[healme] Dataset: {dataset} ({dataset_size} samples)")
    print(f"[healme] Layers to skip: {layers_to_skip}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        print("[healme] Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model and tokenizer
    print("[healme] Loading model and tokenizer")
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
        print("[healme] Set pad_token to eos_token")
    
    print(f"[healme] Model loaded - Hidden size: {hidden_size}")
    print(f"[healme] Number of layers: {model.config.num_hidden_layers}")
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    print(f"[healme] Calibration dataloader prepared")

    # Load distances and select blocks
    print(f"[healme] Loading distances from {distances_path}")
    average_distances = torch.load(distances_path)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    print(f"[healme] Selected {len(selected_blocks)} blocks for processing: {selected_blocks}")
    
    # Setup activation hooks
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    if 'falcon' in model_path.lower():
        print("[healme] Using Falcon model structure")
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        print("[healme] Using standard model structure")
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Process each selected block
    transforms = []
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]

    for block_idx, (start_id, end_id) in enumerate(selected_blocks):
        print(f"[healme] Processing block {block_idx+1}/{len(selected_blocks)}: layers {start_id}-{end_id}")
        
        mlp_activations = {}
        a1 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
        a2 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
        if accurate:
            print("[healme] Using accurate mode - additional memory needed")
            a3 = torch.empty(
                (dataset_size * max_length, model.config.hidden_size),
                dtype=torch.bfloat16,
                device='cpu'
            )
        
        cnt = 0
        for batch in tqdm(
            dataloader,
            desc=f"{Fore.BLUE}Gathering Activations (Block {block_idx+1}){Fore.RESET}",
            dynamic_ncols=True,
            colour="blue"
        ):
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
            hidden_states_mlp_list = [
                mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
            ]
            
            # Get relevant activations for this block
            hidden_states_mlp = hidden_states_mlp_list[start_id - num_layers[block_idx] - 1]
            hidden_states_i = hidden_states[start_id - num_layers[block_idx] - 1]
            hidden_states_n = hidden_states[end_id - num_layers[block_idx] - 1]

            # Reshape activations
            hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
            hidden_states_i = hidden_states_i.view(-1, hidden_size).to(torch.float64)
            hidden_states_n = hidden_states_n.view(-1, hidden_size).to(torch.float64)
            
            a1_batch = hidden_states_mlp
            if accurate:
                a2_batch = hidden_states_n 
                a3_batch = hidden_states_i - hidden_states_mlp 
                a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
            else:
                a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
                
            a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
            a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
            
            cnt += a2_batch.shape[0]
            
            del hidden_states_mlp, hidden_states_i, hidden_states_n
        
        # Compute transformation for this block
        a1 = a1[:cnt]
        a2 = a2[:cnt]
        if accurate:
            a3 = a3[:cnt]
        
        print(f"[healme] Computing transformation using {solver} solver with {loss} loss")
        if solver == "adam":
            transform = adam_method(a1, a2, a3=a3 if accurate else None, 
                                  loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
        else:
            transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
        
        transforms.append(transform)
        print(f"[healme] Block {block_idx+1} transformation computed - shape: {transform.shape}")
        
        # Clean up for next block
        del a1, a2
        if accurate:
            del a3

    # Remove hooks
    for hook in hooks:
        hook.remove()
    print("[healme] Removed activation hooks")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[healme] Cleaned up original model from memory")
    
    # Reload model for transformation
    print("[healme] Reloading model for transformation")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Apply transformations and healing wrappers
    for i, (start_id, end_id) in enumerate(selected_blocks):
        print(f"[healme] Applying transformation and healing wrapper to block {i+1}")
        
        # Truncate model (remove layers)
        model = truncate_model(model, start_id - num_layers[i], end_id - num_layers[i])
        print(f"[healme] Truncated layers {start_id - num_layers[i]} to {end_id - num_layers[i]}")
        
        # Get the layer to modify
        target_layer_idx = start_id - num_layers[i] - 1
        original_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
        
        # Apply the transformation to the weights
        transformed_weight = (transforms[i].T.cpu() @ original_down_proj.weight.to(torch.float64)).to(torch.bfloat16)
        original_down_proj.weight.data = transformed_weight
        print(f"[healme] Applied linear transformation to layer {target_layer_idx}")
        
        # Wrap with HealableLinear
        healing_config = {
            'enabled': True,
            'threshold': healing_threshold,
            'history_length': 10,
            'prediction_window': 3,
            'transform': transforms[i]
        }
        
        healable_layer = HealableLinear(original_down_proj, healing_config)
        model.model.layers[target_layer_idx].mlp.down_proj = healable_layer
        print(f"[healme] Wrapped layer {target_layer_idx} with HealableLinear")

    # Set up save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        layer_indices_for_name = '__'.join([f"{start_ids[i]}_{end_ids[i]}" for i in range(len(selected_blocks))])
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{layer_indices_for_name}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_save_path = f"{save_path}_ReplaceMe_healme_{solver}_{loss}"
    
    # Save model
    print(f"[healme] Saving healed model to {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    if save_transform_only:
        transform_path = f"{final_save_path}_transforms"
        torch.save(transforms, transform_path)
        print(f"[healme] Saved transformations to {transform_path}")
    
    # Final cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"[healme] Self-healing ReplaceMe optimization completed")
    print(f"[healme] Model saved at: {final_save_path}")
    
    return final_save_path


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the healme method from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run self-healing ReplaceMe optimization based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    healme(**config)