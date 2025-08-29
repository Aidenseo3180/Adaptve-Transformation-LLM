import argparse
import gc
import logging
import os
from typing import Optional, List
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_sparse_linear_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

seed_all()


def sparse_linear_replacement(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,  # Number of blocks to replace sparsely
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,
    loss: str = "cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False
    
) -> str:
    """Replace sparse linear transformer blocks with matrix transformations.
    
    Args:
        layers_to_skip: Number of most linear blocks to replace sparsely (avoiding adjacent blocks)
        Other args same as cosine_dist
    
    Returns:
        Path where transformed model is saved
    """
    print(f"=== Sparse Linear Block Replacement ===")
    print(f"Target: Replace {layers_to_skip} most linear blocks (avoiding adjacent blocks)")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
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
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden_size={hidden_size}")
    
    # Step 1: Analyze linearity of all blocks
    print("\n=== Step 1: Analyzing Block Linearity ===")
    
    all_hidden_states = []
    sample_count = 0
    
    for batch in tqdm(
        dataloader,
        desc="Collecting Hidden States for Linearity Analysis",
        dynamic_ncols=True
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
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # Extract last non-padded tokens for linearity analysis
        attention_mask = inputs["attention_mask"]
        batch_hidden_states = []
        
        for layer_states in hidden_states:
            last_tokens = []
            for batch_idx, mask in enumerate(attention_mask):
                last_non_pad_idx = mask.nonzero()[-1].item()
                last_tokens.append(layer_states[batch_idx, last_non_pad_idx, :])
            batch_hidden_states.append(torch.stack(last_tokens))
        
        all_hidden_states.append(batch_hidden_states)
        sample_count += len(batch)
        
        if sample_count >= 200:  # Small sample for linearity analysis
            break
    
    print(f"Collected {sample_count} samples for linearity analysis")
    
    # Concatenate all samples
    concatenated_states = []
    for layer_idx in range(len(all_hidden_states[0])):
        layer_states = torch.cat([batch_states[layer_idx] for batch_states in all_hidden_states], dim=0)
        concatenated_states.append(layer_states)
    
    # Select sparse linear blocks
    selected_block_indices = select_sparse_linear_blocks(
        concatenated_states, 
        layers_to_skip
    )
    
    # Step 2: Collect activations for selected blocks
    print(f"\n=== Step 2: Collecting Activations for Selected Blocks ===")
    print(f"Selected blocks: {selected_block_indices}")
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    
    # Register hooks for selected blocks
    if 'falcon' in model_path.lower():
        for block_idx in selected_block_indices:
            hooks.append(
                model.transformer.h[block_idx].mlp.register_forward_hook(
                    save_mlp_activation(f'layer_{block_idx}_mlp')
                )
            )
    else:
        for block_idx in selected_block_indices:
            hooks.append(
                model.model.layers[block_idx].mlp.register_forward_hook(
                    save_mlp_activation(f'layer_{block_idx}_mlp')
                )
            )
    
    # Prepare storage for each selected block
    max_samples = min(dataset_size * max_length, 100000)  # Limit samples
    print(f"Limiting activation collection to {max_samples} samples per block")
    
    block_data = {}
    for block_idx in selected_block_indices:
        block_data[block_idx] = {
            'a1': torch.empty((max_samples, hidden_size), dtype=torch.float32, device='cpu'),
            'a2': torch.empty((max_samples, hidden_size), dtype=torch.float32, device='cpu'),
            'cnt': 0
        }
        if accurate:
            block_data[block_idx]['a3'] = torch.empty((max_samples, hidden_size), dtype=torch.float32, device='cpu')
    
    # Collect activations
    for batch in tqdm(
        dataloader,
        desc="Collecting Activations for Matrix T Estimation",
        dynamic_ncols=True
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
        
        hidden_states = outputs.hidden_states[1:]  # Skip input embeddings
        
        # Process each selected block
        for block_idx in selected_block_indices:
            if block_data[block_idx]['cnt'] >= max_samples:
                continue
                
            # Get MLP output for this block
            mlp_output = mlp_activations[f'layer_{block_idx}_mlp']
            
            # Get correct input and output for this block
            input_states = hidden_states[block_idx].view(-1, hidden_size).to(torch.float32)
            output_states = hidden_states[block_idx + 1].view(-1, hidden_size).to(torch.float32) if block_idx + 1 < len(hidden_states) else hidden_states[block_idx].view(-1, hidden_size).to(torch.float32)
            
            a1_batch = mlp_output.view(-1, hidden_size).to(torch.float32)
            a2_batch = output_states + a1_batch - input_states
            
            # Calculate samples to add
            current_cnt = block_data[block_idx]['cnt']
            available_space = max_samples - current_cnt
            samples_to_add = min(a1_batch.shape[0], available_space)
            
            if samples_to_add <= 0:
                continue
            
            if accurate:
                a2_batch = output_states[:samples_to_add]
                a3_batch = input_states[:samples_to_add] - a1_batch[:samples_to_add]
                block_data[block_idx]['a3'][current_cnt:current_cnt+samples_to_add] = a3_batch
            
            block_data[block_idx]['a1'][current_cnt:current_cnt+samples_to_add] = a1_batch[:samples_to_add]
            block_data[block_idx]['a2'][current_cnt:current_cnt+samples_to_add] = a2_batch[:samples_to_add]
            block_data[block_idx]['cnt'] += samples_to_add
    
    # Step 3: Estimate transformations
    print(f"\n=== Step 3: Estimating Matrix Transformations ===")
    
    transforms = {}
    
    for block_idx in selected_block_indices:
        print(f"\nEstimating Matrix T for Block {block_idx}...")
        
        data = block_data[block_idx]
        a1 = data['a1'][:data['cnt']]
        a2 = data['a2'][:data['cnt']]
        a3 = data['a3'][:data['cnt']] if accurate else None
        
        print(f"Block {block_idx}: Using {data['cnt']} samples")
        
        # Debug: Check data properties
        print(f"  Data stats:")
        print(f"    - a1 norm: {a1.norm():.6f}, mean: {a1.mean():.6f}")
        print(f"    - a2 norm: {a2.norm():.6f}, mean: {a2.mean():.6f}")
        
        similarity = torch.nn.functional.cosine_similarity(a1.flatten(), a2.flatten(), dim=0)
        print(f"    - Cosine similarity between a1 and a2: {similarity:.6f}")
        
        if similarity > 0.99:
            print(f"    ⚠️  WARNING: a1 and a2 are very similar for Block {block_idx}!")
        
        # Estimate transformation
        if solver == "adam":
            transform = adam_method(
                a1, a2,
                a3=a3,
                loss=loss,
                diag=diag,
                two_vectors=two_vectors,
                thri=thri
            )
        else:
            transform = optimizing_method(a1, a2, a3=a3, solver=solver)
        
        transforms[block_idx] = transform
        print(f"Block {block_idx}: Matrix T estimated, shape={transform.shape}")
        
        # Debug transform properties
        identity = torch.eye(transform.shape[0], dtype=transform.dtype, device=transform.device)
        identity_distance = (transform - identity).norm()
        print(f"  Transform stats:")
        print(f"    - Matrix norm: {transform.norm():.6f}")
        print(f"    - Distance from identity: {identity_distance:.6f}")
        
        if identity_distance < 0.01:
            print(f"    ⚠️  WARNING: Transform very close to identity!")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 4: Apply transformations and remove blocks
    print(f"\n=== Step 4: Applying Transformations ===")
    
    # Reload model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    # Apply transformations to previous blocks and remove selected blocks
    blocks_to_remove = []
    
    for block_idx in selected_block_indices:
        print(f"\nProcessing Block {block_idx}:")
        
        # Apply matrix T to the PREVIOUS block's down_proj
        prev_block_idx = block_idx - 1
        if prev_block_idx >= 0:
            transform = transforms[block_idx]
            
            print(f"  Applying Matrix T to Block {prev_block_idx}'s down_proj")
            
            # Get original weight
            original_weight = model.model.layers[prev_block_idx].mlp.down_proj.weight.to(torch.float64)
            print(f"    Original weight shape: {original_weight.shape}")
            print(f"    Original weight norm: {original_weight.norm():.6f}")
            
            # Apply transformation
            new_weight = (transform.T.cpu() @ original_weight).to(torch.bfloat16)
            print(f"    New weight norm: {new_weight.norm():.6f}")
            print(f"    Weight change ratio: {(new_weight.norm() / original_weight.norm()):.6f}")
            
            # Update weight
            model.model.layers[prev_block_idx].mlp.down_proj.load_state_dict({
                "weight": new_weight
            })
            
            # Verify update
            updated_weight = model.model.layers[prev_block_idx].mlp.down_proj.weight
            verification_diff = (updated_weight.to(torch.float64) - original_weight).norm()
            print(f"    Verification - Weight changed by: {verification_diff:.6f}")
            
            if verification_diff < 1e-6:
                print(f"    ⚠️  WARNING: Weight barely changed!")
            else:
                print(f"    ✓ Matrix T successfully applied to Block {prev_block_idx}")
        
        # Mark block for removal
        blocks_to_remove.append(block_idx)
        print(f"  Block {block_idx} marked for removal")
    
    # Remove selected blocks
    print(f"\nRemoving blocks: {sorted(blocks_to_remove, reverse=True)}")
    
    # Remove in reverse order to maintain indices
    for block_idx in sorted(blocks_to_remove, reverse=True):
        print(f"  Removing Block {block_idx}")
        model.model.layers = nn.ModuleList([
            layer for i, layer in enumerate(model.model.layers) if i != block_idx
        ])
    
    # Update model config
    model.config.num_hidden_layers -= len(blocks_to_remove)
    print(f"Model layers reduced from {model.config.num_hidden_layers + len(blocks_to_remove)} to {model.config.num_hidden_layers}")
    
    # Step 5: Save model
    print(f"\n=== Step 5: Saving Model ===")
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_sparse_linear_{len(selected_block_indices)}_blocks_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_save_path = f"{save_path}_SparseLinear_{loss}_{solver}"
    
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    if save_transform_only:
        torch.save(transforms, f"{final_save_path}_transforms")
    
    print(f"Model saved to: {final_save_path}")
    print(f"Removed blocks: {sorted(blocks_to_remove)}")
    print(f"Applied Matrix T to blocks: {[idx-1 for idx in selected_block_indices if idx > 0]}")
    print(f"Final model has {model.config.num_hidden_layers} layers")
    
    # Final cleanup
    del model
    for data in block_data.values():
        del data['a1'], data['a2']
        if 'a3' in data:
            del data['a3']
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_save_path