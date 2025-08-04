import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class AttentionPatternAnalyzer:
    """Analyze attention patterns to determine optimal replacement strategy"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    def analyze_attention_complexity(self, attention_weights: torch.Tensor) -> float:
        """
        Compute attention complexity score
        attention_weights: [batch, heads, seq_len, seq_len]
        Returns: complexity score (lower = better for replacement)
        """
        if attention_weights is None or attention_weights.numel() == 0:
            return 1.0  # High complexity if no attention data
        
        # Ensure we have the right dimensions
        if len(attention_weights.shape) != 4:
            return 1.0
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 1. Entropy of attention distribution
        attention_flat = attention_weights.view(-1, seq_len)
        entropy = -torch.sum(attention_flat * torch.log(attention_flat + 1e-9), dim=-1)
        avg_entropy = entropy.mean().item()
        
        # 2. Attention sparsity (how concentrated is attention?)
        max_attention = torch.max(attention_weights, dim=-1)[0]
        sparsity = max_attention.mean().item()
        
        # 3. Pattern regularity (how similar are attention patterns across heads?)
        if num_heads > 1:
            attention_reshaped = attention_weights.mean(0)  # Average over batch
            attention_2d = attention_reshaped.view(num_heads, -1)
            if attention_2d.shape[1] > 1:
                correlation_matrix = torch.corrcoef(attention_2d)
                # Get upper triangular part (excluding diagonal)
                mask = torch.triu(torch.ones_like(correlation_matrix), diagonal=1).bool()
                if mask.sum() > 0:
                    regularity = correlation_matrix[mask].mean().item()
                else:
                    regularity = 0.0
            else:
                regularity = 0.0
        else:
            regularity = 0.0
        
        # Combined complexity score (lower = better for replacement)
        # Normalize entropy by log(seq_len) to make it scale-invariant
        normalized_entropy = avg_entropy / (np.log(seq_len) + 1e-9)
        complexity = normalized_entropy * 0.4 + (1 - sparsity) * 0.3 + (1 - regularity) * 0.3
        
        return complexity


class MultiScaleLinearTransform:
    """Different transforms for different sequence positions and patterns"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    def compute_position_aware_transform(
        self, 
        a1: torch.Tensor, 
        a2: torch.Tensor,
        positions: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute different transforms for different sequence positions
        """
        if positions is None:
            # Create position indices
            positions = torch.arange(a1.shape[0], device=a1.device)
        
        seq_len = len(positions)
        if seq_len == 0:
            return {"default": torch.eye(self.hidden_size, dtype=a1.dtype, device=a1.device)}
        
        # Divide sequence into early, middle, late
        early_threshold = int(seq_len * 0.3)
        late_threshold = int(seq_len * 0.7)
        
        early_mask = positions < early_threshold
        middle_mask = (positions >= early_threshold) & (positions < late_threshold)
        late_mask = positions >= late_threshold
        
        transforms = {}
        
        for name, mask in [("early", early_mask), ("middle", middle_mask), ("late", late_mask)]:
            if mask.sum() > 0:
                a1_subset = a1[mask]
                a2_subset = a2[mask]
                
                # Compute transform for this subset
                if len(a1_subset) >= min(32, self.hidden_size // 16):  # Minimum samples needed
                    transform = self._compute_ls_transform(a1_subset, a2_subset)
                    transforms[name] = transform
        
        # If no transforms computed, use default
        if not transforms:
            transforms["default"] = self._compute_ls_transform(a1, a2)
        
        return transforms
    
    def _compute_ls_transform(self, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        """Standard least squares solution like original ReplaceMe"""
        try:
            # Convert to float32 for numerical stability
            a1_f32 = a1.float()
            a2_f32 = a2.float()
            
            # T* = (a1^T a1)^-1 a1^T a2
            a1_t = a1_f32.t()
            gram_matrix = a1_t @ a1_f32
            
            # Add regularization for numerical stability
            reg_strength = 1e-6 * torch.trace(gram_matrix) / gram_matrix.shape[0]
            reg_term = reg_strength * torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            gram_regularized = gram_matrix + reg_term
            
            # Use pseudoinverse for better numerical stability
            gram_inv = torch.linalg.pinv(gram_regularized)
            transform = gram_inv @ a1_t @ a2_f32
            
            # Convert back to original dtype
            return transform.t().to(a1.dtype)  # Return as row transform
            
        except Exception as e:
            logging.warning(f"Failed to compute LS transform: {e}. Using identity.")
            return torch.eye(self.hidden_size, dtype=a1.dtype, device=a1.device)


class ResidualAwareTransform:
    """Explicitly model residual connections when computing transforms"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    def compute_residual_aware_transform(
        self,
        mlp_input: torch.Tensor,      # Input to MLP (after attention)
        mlp_output: torch.Tensor,     # Output of MLP
        target_output: torch.Tensor   # Target output after skipped blocks
    ) -> torch.Tensor:
        """
        Model the residual structure:
        We want: mlp_input + T * mlp_output ≈ target_output
        So: T * mlp_output ≈ target_output - mlp_input
        """
        
        # Compute residual target
        residual_target = target_output - mlp_input
        
        # Solve for T: mlp_output @ T.T = residual_target
        return self._solve_transform(mlp_output, residual_target)
    
    def _solve_transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Solve T such that X @ T.T = Y"""
        try:
            # Convert to float32 for numerical stability
            X_f32 = X.float()
            Y_f32 = Y.float()
            
            # Least squares: T = (X^T X)^-1 X^T Y
            X_t = X_f32.t()
            gram_matrix = X_t @ X_f32
            
            # Regularization
            reg_strength = 1e-6 * torch.trace(gram_matrix) / gram_matrix.shape[0]
            reg_term = reg_strength * torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            gram_regularized = gram_matrix + reg_term
            
            gram_inv = torch.linalg.pinv(gram_regularized)
            transform = gram_inv @ X_t @ Y_f32
            
            return transform.t().to(X.dtype)
        except Exception as e:
            logging.warning(f"Failed to compute residual transform: {e}. Using identity.")
            return torch.eye(self.hidden_size, dtype=X.dtype, device=X.device)


class TemperatureAdaptiveSelector:
    """Select number of blocks to replace based on SoC temperature"""
    
    def __init__(self):
        # Temperature thresholds (Celsius) -> compression ratios
        self.temp_thresholds = {
            65: 0.10,   # Light compression
            70: 0.15,   # Medium compression  
            75: 0.25,   # Heavy compression
            80: 0.35,   # Aggressive compression
        }
    
    def select_compression_ratio(self, temperature: float) -> float:
        """Select compression ratio based on current temperature"""
        for temp_threshold in sorted(self.temp_thresholds.keys()):
            if temperature <= temp_threshold:
                return self.temp_thresholds[temp_threshold]
        
        # If temperature is very high, use maximum compression
        return 0.40
    
    def get_blocks_to_replace(
        self, 
        temperature: float, 
        total_layers: int,
        layer_complexities: List[float],
        layers_to_skip: int = 8
    ) -> List[Tuple[int, int]]:
        """Get specific blocks to replace based on temperature and complexity"""
        
        compression_ratio = self.select_compression_ratio(temperature)
        num_layers_to_remove = int(total_layers * compression_ratio)
        
        logging.info(f"🌡️  Temperature: {temperature}°C → {compression_ratio:.1%} compression")
        logging.info(f"📊 Removing {num_layers_to_remove} out of {total_layers} layers")
        
        if num_layers_to_remove == 0:
            return []
        
        # Find the best contiguous blocks to remove based on complexity
        best_blocks = []
        remaining_to_remove = num_layers_to_remove
        
        # Create candidates for contiguous blocks
        block_candidates = []
        for start_idx in range(total_layers - layers_to_skip):
            end_idx = min(start_idx + layers_to_skip, total_layers)
            
            # Calculate average complexity for this block
            block_complexity = np.mean(layer_complexities[start_idx:end_idx])
            block_candidates.append((start_idx, end_idx, block_complexity))
        
        # Sort by complexity (ascending - lowest complexity first)
        block_candidates.sort(key=lambda x: x[2])
        
        # Select non-overlapping blocks
        used_layers = set()
        for start_idx, end_idx, complexity in block_candidates:
            if remaining_to_remove <= 0:
                break
                
            # Check if this block overlaps with already used layers
            block_layers = set(range(start_idx, end_idx))
            if not block_layers & used_layers:
                best_blocks.append((start_idx, end_idx))
                used_layers.update(block_layers)
                layers_removed = end_idx - start_idx
                remaining_to_remove -= layers_removed
                
                logging.info(f"🎯 Selected block {start_idx}-{end_idx} (complexity: {complexity:.3f})")
        
        return best_blocks


def arm(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    soc_temperature: float = 70.0,
    use_attention_gating: bool = True,
    use_multi_scale: bool = True,
    use_residual_aware: bool = True,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    # Existing ReplaceMe parameters for compatibility
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    **kwargs
) -> str:
    """
    Adaptive ReplaceMe (ARM) implementation
    
    Args:
        soc_temperature: Current SoC temperature in Celsius
        use_attention_gating: Whether to use attention pattern analysis
        use_multi_scale: Whether to use position-aware transforms
        use_residual_aware: Whether to model residuals explicitly
        ... (other args same as cosine_dist)
    
    Returns:
        Path where transformed model is saved
    """
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        output_attentions=use_attention_gating,  # Need attention for gating
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    
    # Initialize ARM components
    hidden_size = model.config.hidden_size
    attention_analyzer = AttentionPatternAnalyzer(hidden_size)
    multi_scale_transform = MultiScaleLinearTransform(hidden_size)
    residual_transform = ResidualAwareTransform(hidden_size)
    temp_selector = TemperatureAdaptiveSelector()
    
    # Collect activations and attention patterns
    logging.info(f"{Fore.GREEN}🔍 Analyzing model with ARM method{Fore.RESET}")
    
    def save_activations(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    def save_attention(name):
        def hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_patterns[name] = output.attentions.detach()
            elif isinstance(output, tuple) and len(output) > 1:
                attention_patterns[name] = output[1].detach()
        return hook

    hooks = []
    activations = {}
    attention_patterns = {}
    
    # Register hooks for all layers
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            # MLP activations
            hooks.append(layer.mlp.register_forward_hook(save_activations(f'layer_{i}_mlp')))
            # Attention patterns
            if use_attention_gating:
                hooks.append(layer.self_attention.register_forward_hook(save_attention(f'layer_{i}_attn')))
    else:
        for i, layer in enumerate(model.model.layers):
            # MLP activations  
            hooks.append(layer.mlp.register_forward_hook(save_activations(f'layer_{i}_mlp')))
            # Block input/output
            hooks.append(layer.register_forward_hook(save_activations(f'layer_{i}_block')))
            # Attention patterns
            if use_attention_gating:
                hooks.append(layer.self_attn.register_forward_hook(save_attention(f'layer_{i}_attn')))

    # Collect data
    all_activations = []
    all_attention_patterns = []
    
    for batch in tqdm(dataloader, desc=f"{Fore.RED}Gathering ARM Data{Fore.RESET}"):
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
        
        # Store this batch's data
        batch_activations = {k: v.clone() for k, v in activations.items()}
        batch_attention = {k: v.clone() for k, v in attention_patterns.items()}
        
        all_activations.append(batch_activations)
        all_attention_patterns.append(batch_attention)
        
        # Clear for next batch
        activations.clear()
        attention_patterns.clear()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze attention complexity for each layer
    num_layers = model.config.num_hidden_layers
    layer_complexities = []
    
    if use_attention_gating and all_attention_patterns:
        logging.info(f"{Fore.GREEN}🧠 Analyzing attention patterns{Fore.RESET}")
        for layer_idx in range(num_layers):
            layer_attentions = []
            for batch_attention in all_attention_patterns:
                attn_key = f'layer_{layer_idx}_attn'
                if attn_key in batch_attention:
                    layer_attentions.append(batch_attention[attn_key])
            
            if layer_attentions:
                # Concatenate attention weights from all batches
                combined_attention = torch.cat(layer_attentions, dim=0)
                complexity = attention_analyzer.analyze_attention_complexity(combined_attention)
            else:
                complexity = 1.0  # High complexity if no attention data
            
            layer_complexities.append(complexity)
    else:
        # Fallback to uniform complexity if no attention analysis
        layer_complexities = [0.5] * num_layers
    
    # Temperature-adaptive block selection
    blocks_to_replace = temp_selector.get_blocks_to_replace(
        soc_temperature, 
        num_layers,
        layer_complexities,
        layers_to_skip
    )
    
    if not blocks_to_replace:
        logging.info(f"{Fore.YELLOW}🌡️  Temperature {soc_temperature}°C too low, no compression needed{Fore.RESET}")
        return model_path  # Return original model path
    
    # Clean up before model modification
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    # Apply ARM transforms to each selected block
    for block_idx, (start_idx, end_idx) in enumerate(blocks_to_replace):
        logging.info(f"🔧 Processing block {start_idx}-{end_idx}")
        
        # Get activations for this block from collected data
        mlp_inputs = []
        mlp_outputs = []
        target_outputs = []
        
        for batch_activations in all_activations:
            if f'layer_{start_idx}_mlp' in batch_activations and f'layer_{end_idx-1}_block' in batch_activations:
                mlp_out = batch_activations[f'layer_{start_idx}_mlp']
                target_out = batch_activations[f'layer_{end_idx-1}_block']
                
                # Reshape to (batch*seq, hidden_size)
                mlp_out_flat = mlp_out.view(-1, hidden_size).to(torch.float16)
                target_out_flat = target_out.view(-1, hidden_size).to(torch.float16)
                
                mlp_outputs.append(mlp_out_flat)
                target_outputs.append(target_out_flat)
        
        if not mlp_outputs:
            logging.warning(f"No activations found for block {start_idx}-{end_idx}, skipping")
            continue
        
        # Concatenate all batches
        mlp_output = torch.cat(mlp_outputs, dim=0)
        target_output = torch.cat(target_outputs, dim=0)
        
        # Compute transform based on selected method
        if use_residual_aware:
            # For residual-aware, we need mlp input as well
            mlp_input = target_output - mlp_output  # Approximate
            transform = residual_transform.compute_residual_aware_transform(
                mlp_input, mlp_output, target_output
            )
        elif use_multi_scale:
            # Use multi-scale transform
            positions = torch.arange(mlp_output.size(0))
            transform_dict = multi_scale_transform.compute_position_aware_transform(
                mlp_output, target_output, positions
            )
            # Use weighted average of transforms
            if len(transform_dict) == 1:
                transform = list(transform_dict.values())[0]
            else:
                transform = torch.zeros_like(list(transform_dict.values())[0])
                for t in transform_dict.values():
                    transform += t
                transform /= len(transform_dict)
        else:
            # Use standard transform
            transform = multi_scale_transform._compute_ls_transform(mlp_output, target_output)
        
        # Apply transformation to the model
        model = truncate_model(model, start_idx, end_idx)
        
        # Merge transform with down_proj layer
        target_layer = model.model.layers[start_idx].mlp.down_proj
        original_weight = target_layer.weight.to(torch.float64)
        transform_64 = transform.to(torch.float64)
        
        # Apply transform: new_weight = transform @ original_weight
        new_weight = (transform_64 @ original_weight).to(torch.bfloat16)
        target_layer.weight = nn.Parameter(new_weight)
        
        logging.info(f"✅ Applied ARM transform to block {start_idx}-{end_idx}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{len(blocks_to_replace)}_blocks_{int(soc_temperature)}C"
        ).replace("/", "_")
    
    final_path = f"{save_path}_ARM_{soc_temperature}C"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    if save_transform_only:
        # Save ARM analysis results
        arm_results = {
            'layer_complexities': layer_complexities,
            'blocks_replaced': blocks_to_replace,
            'compression_ratio': temp_selector.select_compression_ratio(soc_temperature),
            'temperature': soc_temperature
        }
        torch.save(arm_results, f"{final_path}_arm_analysis.pth")
    
    logging.info(f"{Fore.GREEN}🎉 ARM model saved to {final_path}{Fore.RESET}")
    
    # Final cleanup
    del model, all_activations, all_attention_patterns
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run ARM from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run Adaptive ReplaceMe (ARM) based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    # For ARM, we don't need the block selection from distances
    # since we do our own temperature-adaptive selection
    path = arm(**config)
    print(f"ARM completed. Model saved to: {path}")