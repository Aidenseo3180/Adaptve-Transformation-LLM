import argparse
import gc
import logging
import os
import psutil
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


def log_memory_usage(stage=""):
    """Log current memory usage"""
    try:
        ram_percent = psutil.virtual_memory().percent
        ram_gb = psutil.virtual_memory().used / (1024**3)
        
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        
        logging.info(f"🔍 {stage} - RAM: {ram_gb:.1f}GB ({ram_percent:.1f}%), GPU: {gpu_mem:.1f}GB")
    except:
        pass


def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Force Python memory cleanup
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass


class StreamingAttentionAnalyzer:
    """Streaming attention pattern analyzer - processes data immediately"""
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # Only store accumulated statistics, not raw data
        self.complexity_accumulator = [[] for _ in range(num_layers)]
        self.sample_counts = [0] * num_layers
    
    def process_attention_batch(self, attention_outputs, layer_idx: int):
        """Process attention patterns immediately and accumulate statistics"""
        if attention_outputs is None:
            return
        
        try:
            # Extract attention weights from transformer outputs
            if hasattr(attention_outputs, 'attentions') and attention_outputs.attentions is not None:
                attn_weights = attention_outputs.attentions[layer_idx]
            elif isinstance(attention_outputs, tuple) and len(attention_outputs) > 1:
                attn_weights = attention_outputs[1]
            else:
                return
            
            if attn_weights is None or attn_weights.numel() == 0:
                return
            
            # Compute complexity immediately
            complexity = self._compute_complexity_fast(attn_weights)
            self.complexity_accumulator[layer_idx].append(complexity)
            self.sample_counts[layer_idx] += 1
            
            # Immediately delete attention weights to free memory
            del attn_weights
            
        except Exception as e:
            logging.warning(f"Failed to process attention for layer {layer_idx}: {e}")
    
    def _compute_complexity_fast(self, attention_weights: torch.Tensor) -> float:
        """Fast complexity computation with minimal memory usage"""
        try:
            if len(attention_weights.shape) != 4:
                return 0.5
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Subsample for memory efficiency
            if seq_len > 512:
                # Randomly sample positions to reduce computation
                indices = torch.randperm(seq_len)[:512]
                attention_weights = attention_weights[:, :, indices][:, :, :, indices]
            
            # 1. Fast entropy computation
            attention_flat = attention_weights.view(-1, attention_weights.size(-1))
            # Use log2 for faster computation
            entropy = -torch.sum(attention_flat * torch.log2(attention_flat + 1e-9), dim=-1)
            avg_entropy = entropy.mean().item()
            
            # 2. Fast sparsity
            max_attention = torch.max(attention_weights, dim=-1)[0]
            sparsity = max_attention.mean().item()
            
            # 3. Simplified regularity (just variance across heads)
            if num_heads > 1:
                head_means = attention_weights.mean(dim=(0, 2, 3))  # [num_heads]
                regularity = 1.0 - head_means.var().item()
            else:
                regularity = 0.5
            
            # Combined score (normalize and combine)
            normalized_entropy = avg_entropy / 9.0  # log2(512) ≈ 9
            complexity = normalized_entropy * 0.4 + (1 - sparsity) * 0.3 + (1 - regularity) * 0.3
            
            return float(complexity)
            
        except Exception as e:
            logging.warning(f"Complexity computation failed: {e}")
            return 0.5
    
    def get_final_complexities(self) -> List[float]:
        """Get final average complexities for all layers"""
        final_complexities = []
        for layer_idx in range(self.num_layers):
            if self.complexity_accumulator[layer_idx]:
                avg_complexity = np.mean(self.complexity_accumulator[layer_idx])
            else:
                avg_complexity = 0.5  # Default
            final_complexities.append(avg_complexity)
        
        print(f" Processed attention patterns: {self.sample_counts}")
        return final_complexities


class StreamingActivationProcessor:
    """Process activations in streaming fashion"""
    
    def __init__(self, hidden_size: int, target_blocks: List[Tuple[int, int]]):
        self.hidden_size = hidden_size
        self.target_blocks = target_blocks
        
        # Accumulators for each target block - only store statistics
        self.block_processors = {}
        for start_idx, end_idx in target_blocks:
            self.block_processors[f"{start_idx}_{end_idx}"] = {
                'mlp_sum': torch.zeros(hidden_size, hidden_size, dtype=torch.float64),
                'mlp_target_sum': torch.zeros(hidden_size, hidden_size, dtype=torch.float64),
                'mlp_gram': torch.zeros(hidden_size, hidden_size, dtype=torch.float64),
                'sample_count': 0,
                'positions': []
            }
    
    def process_activation_batch(self, activations: Dict[str, torch.Tensor]):
        """Process one batch of activations immediately"""
        try:
            for start_idx, end_idx in self.target_blocks:
                block_key = f"{start_idx}_{end_idx}"
                
                mlp_key = f'layer_{start_idx}_mlp'
                target_key = f'layer_{end_idx-1}_block'
                
                if mlp_key in activations and target_key in activations:
                    mlp_out = activations[mlp_key].detach()
                    target_out = activations[target_key].detach()
                    
                    # Reshape and convert to float64 for numerical stability
                    batch_size, seq_len = mlp_out.shape[:2]
                    mlp_flat = mlp_out.view(-1, self.hidden_size).to(torch.float64)
                    target_flat = target_out.view(-1, self.hidden_size).to(torch.float64)
                    
                    # Update accumulators using Welford's online algorithm for numerical stability
                    self._update_online_statistics(block_key, mlp_flat, target_flat)
                    
                    # Store position information for multi-scale
                    positions = torch.arange(seq_len).repeat(batch_size)
                    self.block_processors[block_key]['positions'].extend(positions.tolist())
                    
                    # Immediately delete tensors
                    del mlp_out, target_out, mlp_flat, target_flat, positions
                    
        except Exception as e:
            logging.warning(f"Failed to process activations: {e}")
        
        # Clean up activations dict
        for key in list(activations.keys()):
            del activations[key]
        activations.clear()
    
    def _update_online_statistics(self, block_key: str, mlp_data: torch.Tensor, target_data: torch.Tensor):
        """Update statistics using online/streaming algorithms"""
        processor = self.block_processors[block_key]
        
        # Online covariance computation for least squares
        # We need to compute: (X^T X)^(-1) X^T Y
        # Accumulate X^T X and X^T Y incrementally
        
        mlp_t = mlp_data.t()  # [hidden, samples]
        
        # Update Gram matrix: X^T X
        processor['mlp_gram'] += mlp_t @ mlp_data
        
        # Update cross-correlation: X^T Y  
        processor['mlp_target_sum'] += mlp_t @ target_data
        
        processor['sample_count'] += mlp_data.shape[0]
        
        # Clean up
        del mlp_t
    
    def compute_final_transforms(self) -> Dict[str, torch.Tensor]:
        """Compute final transforms from accumulated statistics"""
        transforms = {}
        
        for start_idx, end_idx in self.target_blocks:
            block_key = f"{start_idx}_{end_idx}"
            processor = self.block_processors[block_key]
            
            if processor['sample_count'] > self.hidden_size:  # Ensure we have enough samples
                try:
                    # Solve: X^T X * T = X^T Y  =>  T = (X^T X)^(-1) X^T Y
                    gram_matrix = processor['mlp_gram']
                    cross_corr = processor['mlp_target_sum']
                    
                    # Add regularization for numerical stability
                    reg_strength = 1e-6 * torch.trace(gram_matrix) / gram_matrix.shape[0]
                    regularized_gram = gram_matrix + reg_strength * torch.eye(gram_matrix.shape[0])
                    
                    # Use pseudoinverse for better numerical stability
                    gram_inv = torch.linalg.pinv(regularized_gram)
                    transform = gram_inv @ cross_corr
                    
                    transforms[block_key] = transform.t().to(torch.float16)  # Convert back to float16
                    
                    print(f" Computed transform for block {start_idx}-{end_idx} "
                               f"({processor['sample_count']} samples)")
                    
                except Exception as e:
                    print(f"Failed to compute transform for {block_key}: {e}")
                    transforms[block_key] = torch.eye(self.hidden_size, dtype=torch.float16)
            else:
                print(f"Not enough samples for block {block_key}, using identity")
                transforms[block_key] = torch.eye(self.hidden_size, dtype=torch.float16)
        
        return transforms


class TemperatureAdaptiveSelector:
    """Temperature-based block selection (unchanged from original)"""
    
    def __init__(self):
        self.temp_thresholds = {
            65: 0.10, 70: 0.15, 75: 0.25, 80: 0.35
        }
    
    def select_compression_ratio(self, temperature: float) -> float:
        for temp_threshold in sorted(self.temp_thresholds.keys()):
            if temperature <= temp_threshold:
                return self.temp_thresholds[temp_threshold]
        return 0.40
    
    def get_blocks_to_replace(
        self, 
        temperature: float, 
        total_layers: int,
        layer_complexities: List[float],
        layers_to_skip: int = 8
    ) -> List[Tuple[int, int]]:
        compression_ratio = self.select_compression_ratio(temperature)
        num_layers_to_remove = int(total_layers * compression_ratio)
        
        print(f" Temperature: {temperature}°C → {compression_ratio:.1%} compression")
        print(f" Removing {num_layers_to_remove} out of {total_layers} layers")
        
        if num_layers_to_remove == 0:
            return []
        
        # Create block candidates
        block_candidates = []
        for start_idx in range(total_layers - layers_to_skip):
            end_idx = min(start_idx + layers_to_skip, total_layers)
            block_complexity = np.mean(layer_complexities[start_idx:end_idx])
            block_candidates.append((start_idx, end_idx, block_complexity))
        
        # Sort by complexity and select non-overlapping blocks
        block_candidates.sort(key=lambda x: x[2])
        
        best_blocks = []
        used_layers = set()
        remaining_to_remove = num_layers_to_remove
        
        for start_idx, end_idx, complexity in block_candidates:
            if remaining_to_remove <= 0:
                break
                
            block_layers = set(range(start_idx, end_idx))
            if not block_layers & used_layers:
                best_blocks.append((start_idx, end_idx))
                used_layers.update(block_layers)
                remaining_to_remove -= (end_idx - start_idx)
                print(f" Selected block {start_idx}-{end_idx} (complexity: {complexity:.3f})")
        
        return best_blocks


def arm_streaming(
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
    # ReplaceMe compatibility parameters - for cosine distance based block selection
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    # Additional compatibility
    **kwargs
) -> str:
    """
    Memory-efficient streaming ARM implementation
    Now works with ReplaceMe's cosine distance block selection!
    
    Args:
        start_id: Starting layer index (from ReplaceMe block selection)
        end_id: Ending layer index (from ReplaceMe block selection)  
        num_layer: Number of previous layers removed (from ReplaceMe)
        ... (other args same as before)
    """
    
    log_memory_usage("ARM Initial")
    
    # If start_id and end_id are provided (from ReplaceMe pipeline), use them directly
    if start_id > 0 and end_id > start_id:
        target_blocks = [(start_id, end_id)]
        print(f" ARM using ReplaceMe-selected block: {start_id}-{end_id}")
    else:
        # Fallback to temperature-based selection (old behavior)
        print("⚠️  No ReplaceMe block selection found, using temperature-based selection")
        
        # Initialize temperature selector for fallback
        temp_selector = TemperatureAdaptiveSelector()
        initial_complexities = [0.5] * 32  # Assume 32 layers for fallback
        target_blocks = temp_selector.get_blocks_to_replace(
            soc_temperature, 32, initial_complexities, layers_to_skip
        )
        
        if not target_blocks:
            print(f". Temperature {soc_temperature}°C too low, no compression needed")
            return model_path
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        output_attentions=use_attention_gating,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    log_memory_usage("Model loaded")
    
    # Get model dimensions
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    
    # Initialize streaming processors
    attention_analyzer = StreamingAttentionAnalyzer(num_layers) if use_attention_gating else None
    activation_processor = StreamingActivationProcessor(hidden_size, target_blocks)
    
    # Setup hooks for streaming data collection
    def create_activation_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[layer_name] = output[0].detach().cpu().to(torch.float16)
            else:
                activations[layer_name] = output.detach().cpu().to(torch.float16)
        return hook
    
    hooks = []
    activations = {}
    
    # Register hooks only for layers we need
    layers_needed = set()
    for start_idx, end_idx in target_blocks:
        layers_needed.add(start_idx)
        layers_needed.add(end_idx - 1)
    
    if 'falcon' in model_path.lower():
        for i in layers_needed:
            if i < len(model.transformer.h):
                hooks.append(model.transformer.h[i].mlp.register_forward_hook(
                    create_activation_hook(f'layer_{i}_mlp')))
                hooks.append(model.transformer.h[i].register_forward_hook(
                    create_activation_hook(f'layer_{i}_block')))
    else:
        for i in layers_needed:
            if i < len(model.model.layers):
                hooks.append(model.model.layers[i].mlp.register_forward_hook(
                    create_activation_hook(f'layer_{i}_mlp')))
                hooks.append(model.model.layers[i].register_forward_hook(
                    create_activation_hook(f'layer_{i}_block')))
    
    # Load data
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer
    )
    
    log_memory_usage("Hooks registered")
    
    # Streaming data processing
    batch_count = 0
    max_batches = min(50, len(dataloader))  # Limit for memory safety
    
    print(f" Starting streaming processing of {max_batches} batches")
    
    for batch in tqdm(dataloader, desc=f"{Fore.GREEN}Streaming ARM Processing{Fore.RESET}"):
        if batch_count >= max_batches:
            break
        
        # Prepare inputs
        inputs = tokenizer(
            batch, return_tensors="pt", padding="longest", 
            max_length=max_length, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process attention patterns immediately (if enabled)
        if use_attention_gating and attention_analyzer:
            for layer_idx in range(num_layers):
                attention_analyzer.process_attention_batch(outputs, layer_idx)
        
        # Process activations immediately
        activation_processor.process_activation_batch(activations)
        
        # Aggressive cleanup
        del inputs, outputs
        activations.clear()
        
        batch_count += 1
        
        # Memory management every few batches
        if batch_count % 5 == 0:
            force_cleanup()
            log_memory_usage(f"Batch {batch_count}")
    
    # Remove hooks immediately
    for hook in hooks:
        hook.remove()
    hooks.clear()
    
    log_memory_usage("Data processing complete")
    
    # Get final results from streaming processors
    if use_attention_gating and attention_analyzer:
        final_complexities = attention_analyzer.get_final_complexities()
        # Re-select blocks based on actual attention complexity (optional refinement)
        # target_blocks = temp_selector.get_blocks_to_replace(
        #     soc_temperature, num_layers, final_complexities, layers_to_skip
        # )
    
    # Compute transforms from accumulated statistics
    transforms = activation_processor.compute_final_transforms()
    
    # Clean up processors
    del activation_processor
    if attention_analyzer:
        del attention_analyzer
    force_cleanup()
    log_memory_usage("Transforms computed")
    
    # Apply transforms to model
    del model
    force_cleanup()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='cpu', torch_dtype=torch.bfloat16
    )
    
    log_memory_usage("Model reloaded for transformation")
    
    # Apply transformations
    for start_idx, end_idx in target_blocks:
        block_key = f"{start_idx}_{end_idx}"
        if block_key in transforms:
            # Truncate model
            model = truncate_model(model, start_idx, end_idx)
            
            # Apply transform
            target_layer = model.model.layers[start_idx].mlp.down_proj
            original_weight = target_layer.weight.to(torch.float64)
            transform = transforms[block_key].to(torch.float64)
            
            new_weight = (transform @ original_weight).to(torch.bfloat16)
            target_layer.weight = nn.Parameter(new_weight)
            
            print(f" Applied streaming transform to block {start_idx}-{end_idx}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_ARM_streaming_{int(soc_temperature)}C"
    
    final_path = f"{save_path}_streaming"
    model.save_pretrained(final_path)
    
    # Reload tokenizer for saving (in case it was modified)
    tokenizer_for_save = AutoTokenizer.from_pretrained(model_path)
    tokenizer_for_save.save_pretrained(final_path)
    
    if save_transform_only:
        # Save minimal results only
        temp_selector = TemperatureAdaptiveSelector()  # Reinitialize for compression ratio
        results = {
            'temperature': soc_temperature,
            'blocks_replaced': target_blocks,
            'compression_ratio': temp_selector.select_compression_ratio(soc_temperature)
        }
        torch.save(results, f"{final_path}_results.pth")
    
    print(f" Streaming ARM completed! Model saved to: {final_path}")
    log_memory_usage("Final")
    
    # Final cleanup
    del model, transforms
    force_cleanup()
    
    return final_path


# Alias for compatibility
arm = arm_streaming


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run streaming ARM from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run Memory-Efficient Streaming ARM."
    )
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to the configuration file.")
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    path = arm_streaming(**config)
    print(f"Streaming ARM completed. Model saved to: {path}")