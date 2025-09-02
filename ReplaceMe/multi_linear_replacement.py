"""
Multi-Linear Block Replacement (MLBR) method for transformer compression.

This module implements replacement of multiple transformer blocks with 
sequential linear layers, addressing the "averaging problem" of single 
linear approximations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from colorama import Fore, init
import gc
import os

# Initialize colorama
init(autoreset=True)

class MultiLinearBlock(nn.Module):
    """
    Sequential linear layers with residual connections to replace transformer blocks.
    """
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Create sequential linear layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each linear layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        print(f"[MLBR] Created MultiLinearBlock with {num_layers} layers, hidden_size={hidden_size}")
    
    def forward(self, x):
        """Forward pass through sequential linear layers with residuals."""
        for i, (linear, norm) in enumerate(zip(self.linear_layers, self.layer_norms)):
            # Apply linear transformation with residual connection
            residual = x
            x = linear(x)
            x = x + residual  # Residual connection
            x = norm(x)       # Layer normalization
        return x


def collect_block_wise_activations_streaming(
    model,
    start_id: int,
    end_id: int,
    dataset_size: int,
    max_length: int,
    dataloader,
    device: str = "cuda",
    tokenizer=None
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient streaming collection of input/output pairs for each transformer block.
    
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
        Dictionary containing normal equation components for each block
    """
    print(f"[MLBR Streaming] Starting block-wise activation collection...")
    print(f"[MLBR Streaming] Target blocks: {start_id} to {end_id-1} (total: {end_id - start_id} blocks)")
    
    hidden_size = model.config.hidden_size
    num_blocks = end_id - start_id
    
    # Initialize accumulators for normal equations: A^T*A and A^T*b for each block
    # For block i: minimize ||W_i @ input_i - output_i||^2
    # Normal equation: (input_i^T @ input_i) @ W_i = input_i^T @ output_i
    accumulators = {}
    for block_idx in range(start_id, end_id):
        accumulators[block_idx] = {
            'input_T_input': torch.zeros(hidden_size, hidden_size, dtype=torch.float64, device='cpu'),
            'input_T_output': torch.zeros(hidden_size, hidden_size, dtype=torch.float64, device='cpu'),
            'num_samples': 0,
            'total_input_norm': 0.0,
            'total_output_norm': 0.0,
            'total_error_before_learning': 0.0
        }
    
    print(f"[MLBR Streaming] Initialized accumulators for {num_blocks} blocks")
    
    batch_count = 0
    total_tokens_processed = 0
    
    print(f"[MLBR Streaming] Starting streaming batch processing...")
    
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Streaming Block-wise Data{Fore.RESET}",
        dynamic_ncols=True,
        colour="green"
    ):
        batch_count += 1
        
        # Handle different batch formats from dataloader
        if isinstance(batch, (list, tuple)):
            if tokenizer is None:
                raise ValueError("[MLBR Streaming] ERROR: Tokenizer is required when batch contains text")
            
            inputs = tokenizer(
                batch,
                return_tensors="pt", 
                padding="longest",
                max_length=max_length,
                truncation=True
            )
        elif isinstance(batch, dict) and 'input_ids' in batch:
            inputs = batch
        elif torch.is_tensor(batch):
            inputs = {'input_ids': batch, 'attention_mask': torch.ones_like(batch)}
        else:
            raise ValueError(f"[MLBR Streaming] ERROR: Unexpected batch type: {type(batch)}")
        
        # Move to device and get dimensions
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_size = inputs['input_ids'].shape[0]
        seq_len = inputs['input_ids'].shape[1]
        batch_tokens = batch_size * seq_len
        
        with torch.no_grad():
            # Forward pass to collect hidden states
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Process each block independently
            for block_idx in range(start_id, end_id):
                # Get input and output for this specific block
                block_input = hidden_states[block_idx].view(-1, hidden_size).to(torch.float64)  # Before block
                block_output = hidden_states[block_idx + 1].view(-1, hidden_size).to(torch.float64)  # After block
                
                # Move to CPU immediately to save GPU memory
                block_input_cpu = block_input.cpu()
                block_output_cpu = block_output.cpu()
                
                # Update normal equation components
                input_T_input_batch = block_input_cpu.T @ block_input_cpu
                input_T_output_batch = block_input_cpu.T @ block_output_cpu
                
                accumulators[block_idx]['input_T_input'] += input_T_input_batch
                accumulators[block_idx]['input_T_output'] += input_T_output_batch
                accumulators[block_idx]['num_samples'] += block_input_cpu.shape[0]
                
                # Compute statistics for monitoring
                input_norm = torch.norm(block_input_cpu, 'fro').item()
                output_norm = torch.norm(block_output_cpu, 'fro').item()
                identity_error = torch.norm(block_output_cpu - block_input_cpu, 'fro').item()
                
                accumulators[block_idx]['total_input_norm'] += input_norm
                accumulators[block_idx]['total_output_norm'] += output_norm
                accumulators[block_idx]['total_error_before_learning'] += identity_error
                
                # Print progress every 10 batches
                if batch_count % 10 == 0:
                    avg_input_norm = accumulators[block_idx]['total_input_norm'] / batch_count
                    avg_output_norm = accumulators[block_idx]['total_output_norm'] / batch_count
                    avg_identity_error = accumulators[block_idx]['total_error_before_learning'] / batch_count
                    linearity_ratio = avg_identity_error / (avg_output_norm + 1e-8)
                    
                    print(f"[MLBR Streaming] Block {block_idx} (batch {batch_count}):")
                    print(f"[MLBR Streaming]   Samples: {accumulators[block_idx]['num_samples']}")
                    print(f"[MLBR Streaming]   Avg Input norm: {avg_input_norm:.4f}")
                    print(f"[MLBR Streaming]   Avg Output norm: {avg_output_norm:.4f}")
                    print(f"[MLBR Streaming]   Avg Identity error: {avg_identity_error:.4f}")
                    print(f"[MLBR Streaming]   Linearity ratio: {linearity_ratio:.4f}")
        
        total_tokens_processed += batch_tokens
        
        # Force memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print overall progress
        if batch_count % 10 == 0:
            print(f"[MLBR Streaming] Batch {batch_count} complete. Total tokens: {total_tokens_processed}")
        
        # Check if we have enough samples
        if total_tokens_processed >= 200000:  # Reasonable amount for stable learning
            print(f"[MLBR Streaming] Reached target token processing: {total_tokens_processed}")
            break
    
    print(f"[MLBR Streaming] Streaming collection complete!")
    print(f"[MLBR Streaming] Total batches processed: {batch_count}")
    print(f"[MLBR Streaming] Total tokens processed: {total_tokens_processed}")
    
    return accumulators


def solve_linear_layers_from_accumulators(
    accumulators: Dict,
    start_id: int,
    end_id: int,
    hidden_size: int,
    regularization: float = 1e-6
) -> Dict[int, torch.Tensor]:
    """
    Solve individual linear transformations for each block using accumulated normal equations.
    
    Args:
        accumulators: Accumulated normal equation components
        start_id: Starting layer index
        end_id: Ending layer index
        hidden_size: Hidden dimension size
        regularization: Ridge regularization strength
        
    Returns:
        Dictionary mapping block indices to learned transformation matrices
    """
    print(f"[MLBR Solver] Starting linear layer solving...")
    print(f"[MLBR Solver] Blocks to solve: {start_id} to {end_id-1}")
    print(f"[MLBR Solver] Regularization strength: {regularization}")
    
    learned_transformations = {}
    
    for block_idx in range(start_id, end_id):
        print(f"[MLBR Solver] Solving block {block_idx}...")
        
        acc = accumulators[block_idx]
        input_T_input = acc['input_T_input']
        input_T_output = acc['input_T_output']
        num_samples = acc['num_samples']
        
        print(f"[MLBR Solver]   Samples used: {num_samples}")
        
        # Check condition number before regularization
        try:
            cond_num = torch.linalg.cond(input_T_input).item()
            print(f"[MLBR Solver]   Condition number: {cond_num:.2e}")
        except:
            cond_num = float('inf')
            print(f"[MLBR Solver]   Condition number: inf (singular)")
        
        # Apply regularization
        regularizer = regularization * torch.eye(hidden_size, dtype=torch.float64)
        input_T_input_reg = input_T_input + regularizer
        
        try:
            # Solve normal equation: (X^T X + λI) W = X^T Y
            # This gives us W such that X @ W ≈ Y
            transformation = torch.linalg.solve(input_T_input_reg, input_T_output)
            
            # Check final condition number
            final_cond = torch.linalg.cond(input_T_input_reg).item()
            print(f"[MLBR Solver]   Final condition number: {final_cond:.2e}")
            
            # Verify solution quality by computing residual
            residual_norm = torch.norm(input_T_input_reg @ transformation - input_T_output, 'fro').item()
            solution_norm = torch.norm(transformation, 'fro').item()
            relative_residual = residual_norm / (torch.norm(input_T_output, 'fro').item() + 1e-8)
            
            print(f"[MLBR Solver]   Solution norm: {solution_norm:.4f}")
            print(f"[MLBR Solver]   Relative residual: {relative_residual:.6f}")
            
            if relative_residual < 0.1:
                print(f"[MLBR Solver]   ✓ GOOD solution quality")
            elif relative_residual < 0.3:
                print(f"[MLBR Solver]   ⚠ MODERATE solution quality")
            else:
                print(f"[MLBR Solver]   ✗ POOR solution quality")
            
            learned_transformations[block_idx] = transformation.to(torch.float32)  # Convert back to float32
            
        except Exception as e:
            print(f"[MLBR Solver]   ERROR solving block {block_idx}: {str(e)}")
            print(f"[MLBR Solver]   Using identity matrix as fallback")
            learned_transformations[block_idx] = torch.eye(hidden_size, dtype=torch.float32)
    
    print(f"[MLBR Solver] Linear layer solving complete!")
    return learned_transformations


def apply_multi_linear_replacement_to_model(
    model,
    transformations: Dict[int, torch.Tensor],
    start_id: int,
    end_id: int,
    save_path: str,
    tokenizer=None,
    model_path: str = None,
    token: str = None
) -> str:
    """
    Apply multi-linear replacement to the model by replacing transformer blocks
    with sequential linear layers.
    
    Args:
        model: The loaded transformer model
        transformations: Learned transformation matrices
        start_id: Starting layer index for replacement
        end_id: Ending layer index for replacement
        save_path: Path to save the modified model
        tokenizer: Tokenizer to save with the model
        model_path: Original model path for reloading if needed
        token: HuggingFace token
        
    Returns:
        Path where the modified model was saved
    """
    print(f"[MLBR Model] Starting multi-linear model reconstruction...")
    print(f"[MLBR Model] Target layers: {start_id} to {end_id-1}")
    print(f"[MLBR Model] Save path: {save_path}")
    
    # Get model info before modification
    total_layers_before = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_blocks_to_replace = end_id - start_id
    
    print(f"[MLBR Model] Model info:")
    print(f"[MLBR Model]   Total layers before: {total_layers_before}")
    print(f"[MLBR Model]   Hidden size: {hidden_size}")
    print(f"[MLBR Model]   Blocks to replace: {num_blocks_to_replace}")
    
    # Load a clean model if the current one is quantized
    if hasattr(model.model.layers[0].mlp.down_proj.weight, 'data'):
        current_weight = model.model.layers[0].mlp.down_proj.weight.data
        if current_weight.dtype == torch.uint8:
            print(f"[MLBR Model] Detected quantized model, loading clean version...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                token=token
            )
            print(f"[MLBR Model] Clean model loaded successfully")
    
    # Create the multi-linear block
    multi_linear_block = MultiLinearBlock(hidden_size, num_blocks_to_replace)
    
    # Load the learned transformations into the multi-linear block
    for i, block_idx in enumerate(range(start_id, end_id)):
        if block_idx in transformations:
            transformation = transformations[block_idx]
            multi_linear_block.linear_layers[i].weight.data = transformation.T  # Transpose for nn.Linear
            print(f"[MLBR Model] Loaded transformation for block {block_idx} -> linear layer {i}")
            print(f"[MLBR Model]   Weight shape: {multi_linear_block.linear_layers[i].weight.shape}")
            print(f"[MLBR Model]   Weight norm: {torch.norm(multi_linear_block.linear_layers[i].weight).item():.4f}")
        else:
            print(f"[MLBR Model] WARNING: No transformation found for block {block_idx}, using identity")
    
    # Replace the transformer blocks with our multi-linear block
    # We need to modify the model architecture directly
    
    # Create new layer list without the replaced blocks
    new_layers = []
    
    # Add layers before replacement
    for i in range(start_id):
        new_layers.append(model.model.layers[i])
    
    # Add our multi-linear block as a special layer
    # For simplicity, we'll create a wrapper that mimics transformer layer interface
    class MultiLinearWrapper(nn.Module):
        def __init__(self, multi_linear_block):
            super().__init__()
            self.multi_linear = multi_linear_block
        
        def forward(self, hidden_states, attention_mask=None, **kwargs):
            # Just pass through the multi-linear block
            return (self.multi_linear(hidden_states),)  # Return tuple to match transformer layer output
    
    multi_linear_wrapper = MultiLinearWrapper(multi_linear_block)
    new_layers.append(multi_linear_wrapper)
    
    # Add layers after replacement
    for i in range(end_id, total_layers_before):
        new_layers.append(model.model.layers[i])
    
    # Replace the model's layers
    model.model.layers = nn.ModuleList(new_layers)
    
    # Update model config
    model.config.num_hidden_layers = len(new_layers)
    
    print(f"[MLBR Model] Model reconstruction complete:")
    print(f"[MLBR Model]   Layers before: {total_layers_before}")
    print(f"[MLBR Model]   Layers after: {len(new_layers)}")
    print(f"[MLBR Model]   Blocks replaced: {num_blocks_to_replace}")
    print(f"[MLBR Model]   New architecture: {len(new_layers)} layers total")
    
    # Save the model
    print(f"[MLBR Model] Saving model...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Save model
        model.save_pretrained(save_path)
        print(f"[MLBR Model] Model saved to: {save_path}")
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
            print(f"[MLBR Model] Tokenizer saved to: {save_path}")
        
        # Save metadata
        metadata = {
            'method': 'Multi-Linear Block Replacement (MLBR)',
            'original_layers': total_layers_before,
            'final_layers': len(new_layers),
            'blocks_replaced': num_blocks_to_replace,
            'replacement_range': [start_id, end_id-1],
            'compression_ratio': (total_layers_before - len(new_layers)) / total_layers_before,
            'transformations_used': list(transformations.keys())
        }
        
        import json
        metadata_path = os.path.join(save_path, 'mlbr_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[MLBR Model] Metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"[MLBR Model] ERROR in saving: {str(e)}")
        raise
    
    print(f"[MLBR Model] Multi-linear replacement complete!")
    print(f"[MLBR Model] Summary:")
    print(f"[MLBR Model]   Compression ratio: {((total_layers_before - len(new_layers)) / total_layers_before * 100):.1f}%")
    print(f"[MLBR Model]   Model saved to: {save_path}")
    
    return save_path


def multi_linear_block_replacement(
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
    regularization: float = 1e-6,
    **kwargs
) -> str:
    """
    Complete Multi-Linear Block Replacement pipeline.
    
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
        regularization: Regularization strength for linear layer learning
        **kwargs: Additional arguments
        
    Returns:
        Path to the final processed model
    """
    print(f"[MLBR Pipeline] Starting Multi-Linear Block Replacement pipeline...")
    print(f"[MLBR Pipeline] Model: {model_path}")
    print(f"[MLBR Pipeline] Dataset: {dataset}, Size: {dataset_size}")
    print(f"[MLBR Pipeline] Regularization: {regularization}")
    
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
    
    print(f"[MLBR Pipeline] Loading model...")
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
    
    print(f"[MLBR Pipeline] Model loaded. Hidden size: {model.config.hidden_size}")
    print(f"[MLBR Pipeline] Number of layers: {model.config.num_hidden_layers}")
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Load pre-computed distances and select blocks
    print(f"[MLBR Pipeline] Loading distances from: {distances_path}")
    average_distances = torch.load(distances_path, weights_only=False)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    print(f"[MLBR Pipeline] Selected blocks: {selected_blocks}")
    
    # For now, process only the first selected block
    if selected_blocks:
        start_id, end_id = selected_blocks[0]
        print(f"[MLBR Pipeline] Processing block: layers {start_id} to {end_id}")
        
        # Step 1: Collect block-wise activations
        accumulators = collect_block_wise_activations_streaming(
            model=model,
            start_id=start_id,
            end_id=end_id,
            dataset_size=dataset_size,
            max_length=max_length,
            dataloader=dataloader,
            device=next(model.parameters()).device,
            tokenizer=tokenizer
        )
        
        # Step 2: Solve individual linear transformations
        transformations = solve_linear_layers_from_accumulators(
            accumulators=accumulators,
            start_id=start_id,
            end_id=end_id,
            hidden_size=model.config.hidden_size,
            regularization=regularization
        )
        
        # Step 3: Apply multi-linear replacement to model
        if save_path is None:
            os.makedirs('output_models', exist_ok=True)
            save_path = f"output_models/{model_path.replace('/', '_')}_MLBR_layers_{start_id}_{end_id}"
        
        final_save_path = apply_multi_linear_replacement_to_model(
            model=model,
            transformations=transformations,
            start_id=start_id,
            end_id=end_id,
            save_path=save_path,
            tokenizer=tokenizer,
            model_path=model_path,
            token=token
        )
        
        # Cleanup
        del model
        del accumulators
        del transformations
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"[MLBR Pipeline] Multi-Linear Block Replacement complete!")
        print(f"[MLBR Pipeline] Final model saved to: {final_save_path}")
        
        return final_save_path
    
    else:
        raise ValueError("[MLBR Pipeline] No blocks selected for replacement!")