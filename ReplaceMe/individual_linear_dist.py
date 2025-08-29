import argparse
import gc
import logging
import os
from typing import Optional, List
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_most_linear_individual_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

seed_all()


def individual_linear_dist(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,  # 이제 이것은 "교체할 block 개수"를 의미
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
    """Replace individual linear transformer blocks with matrix transformations.
    
    Args:
        layers_to_skip: Number of most linear blocks to replace (not skip distance)
        Other args same as cosine_dist (beta parameter removed)
    
    Returns:
        Path where transformed model is saved
    """
    print(f"Starting individual linear block replacement...")
    print(f"Target: Replace {layers_to_skip} most linear blocks")
    
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
    
    # Collect hidden states for linearity analysis
    print("Collecting hidden states for linearity analysis...")
    
    all_hidden_states = []
    sample_count = 0
    
    for batch in tqdm(
        dataloader,
        desc="Gathering Hidden States for Analysis",
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
        hidden_states = outputs.hidden_states  # [layer_0, layer_1, ..., layer_N]
        
        # Get last non-padded tokens for linearity analysis
        attention_mask = inputs["attention_mask"]
        
        # Extract last non-padded token from each layer
        batch_hidden_states = []
        for layer_states in hidden_states:
            last_tokens = []
            for batch_idx, mask in enumerate(attention_mask):
                last_non_pad_idx = mask.nonzero()[-1].item()
                last_tokens.append(layer_states[batch_idx, last_non_pad_idx, :])
            batch_hidden_states.append(torch.stack(last_tokens))
        
        all_hidden_states.append(batch_hidden_states)
        sample_count += len(batch)
        
        if sample_count >= 200:  # Reduced sample count for linearity analysis
            break
    
    # Concatenate all samples
    print(f"Collected {sample_count} samples for linearity analysis")
    concatenated_states = []
    for layer_idx in range(len(all_hidden_states[0])):
        layer_states = torch.cat([batch_states[layer_idx] for batch_states in all_hidden_states], dim=0)
        concatenated_states.append(layer_states)
    
    # Select most linear blocks
    selected_block_indices = select_most_linear_individual_blocks(
        concatenated_states, 
        layers_to_skip
    )
    
    # Now collect activations for selected blocks
    print(f"\nCollecting activations for selected blocks: {selected_block_indices}")
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    
    # Register hooks only for selected blocks
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
    
    # Prepare storage for each selected block - reduced memory usage
    max_samples = min(dataset_size * max_length, 100000)  # Limit to 100k samples max
    print(f"Limiting activation collection to {max_samples} samples per block")
    
    block_data = {}
    for block_idx in selected_block_indices:
        block_data[block_idx] = {
            'a1': torch.empty((max_samples, hidden_size), dtype=torch.float32, device='cpu'),  # Changed to float32
            'a2': torch.empty((max_samples, hidden_size), dtype=torch.float32, device='cpu'),  # Changed to float32
            'cnt': 0
        }
        if accurate:
            block_data[block_idx]['a3'] = torch.empty((max_samples, hidden_size), dtype=torch.float32, device='cpu')  # Changed to float32
    
    # Collect activations for transformation estimation
    print("Collecting activations for transformation estimation...")
    
    for batch in tqdm(
        dataloader,
        desc="Gathering Activations for Transform",
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
            # Check if we have enough samples already
            if block_data[block_idx]['cnt'] >= max_samples:
                continue
                
            # Get MLP output for this block
            mlp_output = mlp_activations[f'layer_{block_idx}_mlp']
            
            # FIXED: Get correct input and output hidden states
            input_states = hidden_states[block_idx].view(-1, hidden_size).to(torch.float32)  # Block N input, changed to float32
            output_states = hidden_states[block_idx + 1].view(-1, hidden_size).to(torch.float32) if block_idx + 1 < len(hidden_states) else hidden_states[block_idx].view(-1, hidden_size).to(torch.float32)  # Block N output, changed to float32
            
            a1_batch = mlp_output.view(-1, hidden_size).to(torch.float32)  # Changed to float32
            a2_batch = output_states + a1_batch - input_states
            
            # Calculate how many samples we can add
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
    
    # Estimate transformations for each selected block
    print("Estimating transformations for selected blocks...")
    transforms = {}
    
    for block_idx in selected_block_indices:
        print(f"\nProcessing Block {block_idx}...")
        
        data = block_data[block_idx]
        a1 = data['a1'][:data['cnt']]
        a2 = data['a2'][:data['cnt']]
        a3 = data['a3'][:data['cnt']] if accurate else None
        
        print(f"Block {block_idx}: Using {data['cnt']} samples")
        
        # Debug: Check input data properties
        print(f"  Input data stats:")
        print(f"    - a1 (MLP output) norm: {a1.norm():.6f}")
        print(f"    - a2 (target) norm: {a2.norm():.6f}")
        print(f"    - a1 mean: {a1.mean():.6f}, std: {a1.std():.6f}")
        print(f"    - a2 mean: {a2.mean():.6f}, std: {a2.std():.6f}")
        
        # Check if a1 and a2 are too similar (which caused the original problem)
        similarity = torch.nn.functional.cosine_similarity(a1.flatten(), a2.flatten(), dim=0)
        print(f"    - Cosine similarity between a1 and a2: {similarity:.6f}")
        
        if similarity > 0.99:
            print(f"    ⚠️  WARNING: a1 and a2 are too similar for Block {block_idx}!")
        
        if data['cnt'] < 1000:
            print(f"    ⚠️  WARNING: Very few samples ({data['cnt']}) for Block {block_idx}!")
        
        if solver == "adam":
            # Removed beta parameter
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
        print(f"Block {block_idx}: Transform estimated, shape={transform.shape}")
        
        # Debug: Check transform properties
        print(f"  Transform matrix stats:")
        print(f"    - Matrix norm: {transform.norm():.6f}")
        print(f"    - Matrix determinant: {torch.det(transform):.6f}")
        print(f"    - Min eigenvalue: {torch.linalg.eigvals(transform).real.min():.6f}")
        print(f"    - Max eigenvalue: {torch.linalg.eigvals(transform).real.max():.6f}")
        
        # Check if transform is close to identity
        identity = torch.eye(transform.shape[0], dtype=transform.dtype, device=transform.device)
        identity_distance = (transform - identity).norm()
        print(f"    - Distance from identity: {identity_distance:.6f}")
        
        if identity_distance < 0.01:
            print(f"    ⚠️  WARNING: Transform for Block {block_idx} is very close to identity!")
        print()
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model and apply transformations
    print("Reloading model and applying transformations...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    # Apply transformations to selected blocks and remove them
    print(f"Selected blocks for replacement: {selected_block_indices}")
    
    # Sort block indices in ascending order for proper matrix multiplication order
    sorted_blocks = sorted(selected_block_indices)
    print(f"Processing blocks in order: {sorted_blocks}")
    
    # Find the block before the first selected block (where we'll apply combined transform)
    first_block_idx = sorted_blocks[0]
    target_block_idx = first_block_idx - 1
    
    if target_block_idx < 0:
        print(f"⚠️ ERROR: Cannot apply transform before Block 0. First selected block is {first_block_idx}")
        return None
    
    print(f"Will apply combined transform to Block {target_block_idx}'s down_proj")
    print(f"Will remove blocks: {sorted_blocks}")
    
    # Combine transforms using matrix multiplication
    print(f"\nCombining transforms using matrix multiplication...")
    
    # Start with identity and multiply transforms in sequence
    combined_transform = torch.eye(4096, dtype=torch.float64)  # Assuming hidden_size = 4096
    
    for i, block_idx in enumerate(sorted_blocks):
        transform = transforms[block_idx]
        print(f"  Step {i+1}: Multiplying with Transform for Block {block_idx}")
        print(f"    Current combined norm: {combined_transform.norm():.6f}")
        print(f"    Block {block_idx} transform norm: {transform.norm():.6f}")
        
        # Apply transform: combined = T_new @ combined
        combined_transform = transform.cpu().to(torch.float64) @ combined_transform
        print(f"    After multiplication norm: {combined_transform.norm():.6f}")
        
        # Check condition number to detect potential numerical issues
        try:
            cond_num = torch.linalg.cond(combined_transform)
            print(f"    Condition number: {cond_num:.2e}")
            if cond_num > 1e12:
                print(f"    ⚠️  WARNING: High condition number detected!")
        except:
            print(f"    Could not compute condition number")
    
    print(f"\nFinal combined transform:")
    print(f"  Shape: {combined_transform.shape}")
    print(f"  Norm: {combined_transform.norm():.6f}")
    print(f"  Determinant: {torch.det(combined_transform):.6f}")
    
    # Check if combined transform is close to identity
    identity = torch.eye(combined_transform.shape[0], dtype=torch.float64)
    identity_diff = (combined_transform - identity).norm()
    print(f"  Distance from identity: {identity_diff:.6f}")
    
    if identity_diff < 0.01:
        print(f"  ⚠️  WARNING: Combined transform is very close to identity!")
    
    # Apply combined transformation to target block
    print(f"\nApplying combined transform to Block {target_block_idx}...")
    
    original_weight = model.model.layers[target_block_idx].mlp.down_proj.weight.to(torch.float64)
    print(f"  Original weight shape: {original_weight.shape}")
    print(f"  Original weight norm: {original_weight.norm():.6f}")
    
    # Apply combined transformation
    new_weight = (combined_transform.T @ original_weight).to(torch.bfloat16)
    print(f"  New weight norm: {new_weight.norm():.6f}")
    print(f"  Weight change ratio: {(new_weight.norm() / original_weight.norm()):.6f}")
    
    # Store original for verification
    original_weight_backup = original_weight.clone()
    
    # Update the target block's down_proj
    model.model.layers[target_block_idx].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Verify the weight was actually updated
    updated_weight = model.model.layers[target_block_idx].mlp.down_proj.weight
    verification_diff = (updated_weight.to(torch.float64) - original_weight_backup).norm()
    print(f"  Verification - Weight changed by: {verification_diff:.6f}")
    
    if verification_diff < 1e-6:
        print(f"  ⚠️  WARNING: Weight barely changed!")
    else:
        print(f"  ✓ Combined transform successfully applied!")
    
    # Remove the selected blocks (in reverse order to maintain indices)
    print(f"\nRemoving selected blocks: {sorted(sorted_blocks, reverse=True)}")
    original_num_layers = len(model.model.layers)
    
    layers_list = list(model.model.layers)
    for block_idx in sorted(sorted_blocks, reverse=True):
        print(f"  Removing Block {block_idx}...")
        if 0 <= block_idx < len(layers_list):
            layers_list.pop(block_idx)
            print(f"  ✓ Block {block_idx} removed successfully")
        else:
            print(f"  ⚠️  WARNING: Block {block_idx} index out of range")
    
    # Update the model with new layer list
    model.model.layers = torch.nn.ModuleList(layers_list)
    
    # Update model config to reflect new number of layers
    model.config.num_hidden_layers = len(model.model.layers)
    
    print(f"\nModel compression completed:")
    print(f"  Original layers: {original_num_layers}")
    print(f"  New layers: {len(model.model.layers)}")
    print(f"  Removed layers: {original_num_layers - len(model.model.layers)}")
    print(f"  Compression ratio: {(1 - len(model.model.layers) / original_num_layers) * 100:.1f}%")
    print(f"  Combined transform applied to Block {target_block_idx}")
    print(f"  Blocks {sorted_blocks} → Block {target_block_idx} → remaining blocks")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_individual_linear_{len(selected_block_indices)}_blocks_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_save_path = f"{save_path}_IndividualLinear_{loss}_{solver}"
    
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    if save_transform_only:
        torch.save(transforms, f"{final_save_path}_transforms")
    
    print(f"Model saved to: {final_save_path}")
    print(f"Applied transformations to blocks: {selected_block_indices}")
    
    # Final cleanup
    del model
    for data in block_data.values():
        del data['a1'], data['a2']
        if 'a3' in data:
            del data['a3']
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_save_path