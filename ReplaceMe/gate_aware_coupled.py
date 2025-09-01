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

def collect_enhanced_activations(
    model,
    start_id: int,
    end_id: int,
    dataset_size: int,
    max_length: int,
    dataloader,
    device: str = "cuda",
    tokenizer = None
) -> Dict[str, List]:
    """
    Enhanced activation collection for gate-aware coupled optimization.
    Collects activations from input, output, and all MLP components within the target blocks.
    
    Args:
        model: The transformer model
        start_id: Starting layer index for replacement
        end_id: Ending layer index for replacement  
        dataset_size: Number of samples in calibration dataset
        max_length: Maximum sequence length
        dataloader: Calibration data loader
        device: Device to run computations on
        
    Returns:
        Dictionary containing all collected activations
    """
    print(f"[Phase 2] Starting enhanced activation collection...")
    print(f"[Phase 2] Target blocks: {start_id} to {end_id-1} (total: {end_id - start_id} blocks)")
    print(f"[Phase 2] Dataset size: {dataset_size}, Max length: {max_length}")
    
    # Initialize storage for activations
    activations = {
        'input_to_blocks': [],      # X_i (input to first target block)
        'output_from_blocks': [],   # X_{i+n} (output from last target block)
        'gate_activations': [],     # Gate projections from each target block
        'up_activations': [],       # Up projections from each target block  
        'mlp_outputs': [],          # Final MLP outputs from each target block
        'attention_outputs': [],    # Attention outputs (for residual consideration)
    }
    
    # Setup hooks for activation collection
    def save_activation(name: str, activations_dict: Dict):
        def hook(module, input, output):
            activations_dict[name] = output.detach()
        return hook
    
    hooks = []
    hook_activations = {}
    
    # Register hooks for target layers
    print(f"[Phase 2] Registering hooks for layers {start_id} to {end_id-1}...")
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
    
    print(f"[Phase 2] Successfully registered {len(hooks)} hooks")
    
    # Collect activations batch by batch
    total_tokens_collected = 0
    batch_count = 0
    
    print(f"[Phase 2] Starting batch processing...")
    
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Collecting Enhanced Activations{Fore.RESET}",
        dynamic_ncols=True,
        colour="green"
    ):
        batch_count += 1
        print(f"[Phase 2] Processing batch {batch_count}...")
        
        # Handle different batch formats from dataloader
        if isinstance(batch, (list, tuple)):
            # Dataloader returns list of texts, need to tokenize
            if tokenizer is None:
                raise ValueError("[Phase 2] ERROR: Tokenizer is required when batch contains text")
            
            inputs = tokenizer(
                batch,
                return_tensors="pt", 
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            print(f"[Phase 2] Tokenized {len(batch)} texts")
        elif isinstance(batch, dict) and 'input_ids' in batch:
            # Batch is already tokenized dictionary
            inputs = batch
            print(f"[Phase 2] Using pre-tokenized batch")
        elif torch.is_tensor(batch):
            # Batch is tensor
            inputs = {'input_ids': batch, 'attention_mask': torch.ones_like(batch)}
            print(f"[Phase 2] Using tensor batch")
        else:
            raise ValueError(f"[Phase 2] ERROR: Unexpected batch type: {type(batch)}, content: {batch}")
        
        # Move to device and get dimensions
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_size = inputs['input_ids'].shape[0]
        seq_len = inputs['input_ids'].shape[1]
        
        
        print(f"[Phase 2] Batch shape: {batch_size} x {seq_len}")
        
        with torch.no_grad():
            # Forward pass to collect activations
            outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            hidden_states = outputs.hidden_states
            
            # Collect input and output activations (same as original ReplaceMe)
            input_activation = hidden_states[start_id].view(-1, model.config.hidden_size).cpu()
            output_activation = hidden_states[end_id].view(-1, model.config.hidden_size).cpu() 
            
            activations['input_to_blocks'].append(input_activation)
            activations['output_from_blocks'].append(output_activation)
            
            print(f"[Phase 2] Input activation shape: {input_activation.shape}")
            print(f"[Phase 2] Output activation shape: {output_activation.shape}")
            
            # Collect MLP component activations
            batch_gate_acts = []
            batch_up_acts = []
            batch_mlp_outs = []
            
            for layer_idx in range(start_id, end_id):
                # Extract from hooks
                gate_proj = hook_activations.get(f'gate_proj_{layer_idx}')
                up_proj = hook_activations.get(f'up_proj_{layer_idx}') 
                mlp_out = hook_activations.get(f'mlp_out_{layer_idx}')
                
                if gate_proj is not None:
                    gate_reshaped = gate_proj.view(-1, gate_proj.shape[-1]).cpu()
                    batch_gate_acts.append(gate_reshaped)
                    print(f"[Phase 2] Layer {layer_idx} gate projection shape: {gate_reshaped.shape}")
                else:
                    print(f"[Phase 2] WARNING: Gate projection for layer {layer_idx} not found")
                
                if up_proj is not None:
                    up_reshaped = up_proj.view(-1, up_proj.shape[-1]).cpu()
                    batch_up_acts.append(up_reshaped)
                    print(f"[Phase 2] Layer {layer_idx} up projection shape: {up_reshaped.shape}")
                else:
                    print(f"[Phase 2] WARNING: Up projection for layer {layer_idx} not found")
                
                if mlp_out is not None:
                    mlp_reshaped = mlp_out.view(-1, mlp_out.shape[-1]).cpu()
                    batch_mlp_outs.append(mlp_reshaped)
                    print(f"[Phase 2] Layer {layer_idx} MLP output shape: {mlp_reshaped.shape}")
                else:
                    print(f"[Phase 2] WARNING: MLP output for layer {layer_idx} not found")
            
            # Store batch activations
            activations['gate_activations'].extend(batch_gate_acts)
            activations['up_activations'].extend(batch_up_acts)
            activations['mlp_outputs'].extend(batch_mlp_outs)
            
            total_tokens_collected += batch_size * seq_len
            print(f"[Phase 2] Total tokens collected so far: {total_tokens_collected}")

        
        # Clear hook activations for next batch
        hook_activations.clear()
        
        # Memory cleanup
        torch.cuda.empty_cache()
        
        if total_tokens_collected >= 100000:
            print(f"[Phase 2] Reached target token collection: {total_tokens_collected}")
            break
    
    # Remove hooks
    print(f"[Phase 2] Removing {len(hooks)} hooks...")
    for hook in hooks:
        hook.remove()
    
    # Final statistics
    print(f"[Phase 2] Collection complete!")
    print(f"[Phase 2] Total batches processed: {batch_count}")
    print(f"[Phase 2] Total tokens collected: {total_tokens_collected}")
    print(f"[Phase 2] Input activations: {len(activations['input_to_blocks'])} batches")
    print(f"[Phase 2] Output activations: {len(activations['output_from_blocks'])} batches")
    print(f"[Phase 2] Gate activations: {len(activations['gate_activations'])} layer batches")
    print(f"[Phase 2] Up activations: {len(activations['up_activations'])} layer batches")
    print(f"[Phase 2] MLP outputs: {len(activations['mlp_outputs'])} layer batches")
    
    return activations


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
    Gate-aware coupled optimization method - Phase 2 implementation
    
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
        
        # Phase 2: Collect enhanced activations
        activations = collect_enhanced_activations(
            model=model,
            start_id=start_id - num_layer,
            end_id=end_id - num_layer,
            dataset_size=dataset_size,
            max_length=max_length,
            dataloader=dataloader,
            device=next(model.parameters()).device,
            tokenizer=tokenizer  # Pass tokenizer explicitly
        )
        
        print(f"[GACO] Enhanced activations collected for block {i+1}")
        print(f"[GACO] Phase 2 complete for block {i+1}")
        
        # TODO: Phase 3 & 4 will be implemented in next steps
        # For now, just validate that we collected the activations properly
        
        if len(activations['gate_activations']) > 0:
            print(f"[GACO] SUCCESS: Gate activations collected - {len(activations['gate_activations'])} samples")
        else:
            print(f"[GACO] WARNING: No gate activations collected")
            
        if len(activations['up_activations']) > 0:
            print(f"[GACO] SUCCESS: Up activations collected - {len(activations['up_activations'])} samples")
        else:
            print(f"[GACO] WARNING: No up activations collected")
        
        # Memory cleanup
        del activations
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[GACO] Block {i+1} processing complete")
    
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
    
    print(f"[GACO] Method execution complete (Phase 2 only)")
    return f"{save_path}_GACO_phase2"