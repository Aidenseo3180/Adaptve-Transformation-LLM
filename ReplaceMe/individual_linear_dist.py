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
    accurate: bool = False,
    beta: float = 0.1
    
) -> str:
    """Replace individual linear transformer blocks with matrix transformations.
    
    Args:
        layers_to_skip: Number of most linear blocks to replace (not skip distance)
        Other args same as cosine_dist
    
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
            outputs = model(**outputs)
        
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
        
        if sample_count >= 100:  # Use smaller sample for linearity analysis
            break
    
    # Concatenate all samples
    print(f"Collected {sample_count} samples for analysis")
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
    
    # Prepare storage for each selected block
    block_data = {}
    for block_idx in selected_block_indices:
        block_data[block_idx] = {
            'a1': torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu'),
            'a2': torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu'),
            'cnt': 0
        }
        if accurate:
            block_data[block_idx]['a3'] = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    
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
            # Get MLP output for this block
            mlp_output = mlp_activations[f'layer_{block_idx}_mlp']
            
            # Get input and output hidden states for this block
            input_states = hidden_states[block_idx].view(-1, hidden_size).to(torch.float64)
            output_states = hidden_states[block_idx].view(-1, hidden_size).to(torch.float64)  # Same layer since we're replacing individual blocks
            
            a1_batch = mlp_output.view(-1, hidden_size).to(torch.float64)
            a2_batch = output_states + a1_batch - input_states
            
            if accurate:
                a2_batch = output_states
                a3_batch = input_states - a1_batch
                block_data[block_idx]['a3'][block_data[block_idx]['cnt']:block_data[block_idx]['cnt']+a3_batch.shape[0]] = a3_batch
            
            block_data[block_idx]['a1'][block_data[block_idx]['cnt']:block_data[block_idx]['cnt']+a1_batch.shape[0]] = a1_batch
            block_data[block_idx]['a2'][block_data[block_idx]['cnt']:block_data[block_idx]['cnt']+a2_batch.shape[0]] = a2_batch
            block_data[block_idx]['cnt'] += a1_batch.shape[0]
    
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
        
        if solver == "adam":
            transform = adam_method(
                a1, a2,
                a3=a3,
                loss=loss,
                diag=diag,
                two_vectors=two_vectors,
                thri=thri,
                beta=beta
            )
        else:
            transform = optimizing_method(a1, a2, a3=a3, solver=solver)
        
        transforms[block_idx] = transform
        print(f"Block {block_idx}: Transform estimated, shape={transform.shape}")
    
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
    
    # Apply transformations to selected blocks
    for block_idx in selected_block_indices:
        print(f"Applying transformation to Block {block_idx}")
        
        transform = transforms[block_idx]
        
        # Apply transformation to down_proj of selected block
        original_weight = model.model.layers[block_idx].mlp.down_proj.weight.to(torch.float64)
        new_weight = (transform.T.cpu() @ original_weight).to(torch.bfloat16)
        
        model.model.layers[block_idx].mlp.down_proj.load_state_dict({
            "weight": new_weight
        })
    
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