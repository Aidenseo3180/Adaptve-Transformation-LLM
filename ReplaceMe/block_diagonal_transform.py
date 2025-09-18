# block_diagonal_transform.py
import argparse
import gc
import logging
import os
from typing import Optional
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

def block_diagonal_transform(
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
    save_transform_only: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    block_size: int = 128,  # Size of each diagonal block
    **kwargs
) -> str:
    """
    Block Diagonal Transform: Split 4096 dims into smaller blocks
    """
    print(f"[DEBUG] Using Block Diagonal Transform with block_size={block_size}")
    
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
    num_blocks = hidden_size // block_size
    
    print(f"[DEBUG] Hidden size: {hidden_size}, Num blocks: {num_blocks}")
    
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
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Collect activations
    all_a1 = []
    all_a2 = []
    
    print("[DEBUG] Gathering activations...")
    for batch in tqdm(dataloader, desc="Gathering Activations", colour="red"):
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
        
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size)
        
        # Mi
        a1_batch = hidden_states_mlp.view(-1, hidden_size)
        # Target: Li+n - Yi
        a2_batch = hidden_states_n - hidden_states_i
        
        all_a1.append(a1_batch.cpu().to(torch.float64))
        all_a2.append(a2_batch.cpu().to(torch.float64))
    
    # Concatenate
    a1 = torch.cat(all_a1, dim=0)
    a2 = torch.cat(all_a2, dim=0)
    
    print(f"[DEBUG] Collected {a1.shape[0]} samples")
    
    # Compute block diagonal transform
    print("[DEBUG] Computing block diagonal transforms...")
    block_transforms = []
    
    for b in range(num_blocks):
        start_dim = b * block_size
        end_dim = min((b + 1) * block_size, hidden_size)
        actual_block_size = end_dim - start_dim
        
        # Extract block
        a1_block = a1[:, start_dim:end_dim]
        a2_block = a2[:, start_dim:end_dim]
        
        # Solve for this block using least squares with regularization
        MtM = a1_block.T @ a1_block
        MtM_reg = MtM + 1e-4 * torch.eye(actual_block_size, dtype=torch.float64)
        
        try:
            T_block = torch.linalg.solve(MtM_reg, a1_block.T @ a2_block)
        except:
            print(f"[WARNING] Block {b} failed, using identity")
            T_block = torch.eye(actual_block_size, dtype=torch.float64)
        
        block_transforms.append(T_block)
        
        # Print statistics for this block
        max_val = T_block.abs().max().item()
        mean_val = T_block.abs().mean().item()
        print(f"[DEBUG] Block {b}: size={actual_block_size}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    # Construct full block diagonal matrix
    print("[DEBUG] Constructing full block diagonal matrix...")
    T_full = torch.zeros((hidden_size, hidden_size), dtype=torch.float64)
    
    for b, T_block in enumerate(block_transforms):
        start_dim = b * block_size
        end_dim = start_dim + T_block.shape[0]
        T_full[start_dim:end_dim, start_dim:end_dim] = T_block
    
    print(f"[DEBUG] Final transform shape: {T_full.shape}")
    print(f"[DEBUG] Transform sparsity: {(T_full == 0).sum().item() / T_full.numel():.2%}")
    
    # Verify transformation quality
    with torch.no_grad():
        pred = a1 @ T_full
        error = (pred - a2).norm() / a2.norm()
        print(f"[DEBUG] Relative reconstruction error: {error:.4f}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model
    print("[DEBUG] Reloading model for transformation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_{layers_to_skip}layers_blockdiag{block_size}"
    
    # Apply transformation
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (T_full.T @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.data = new_weight
    
    model.save_pretrained(f"{save_path}_BlockDiagonal")
    tokenizer.save_pretrained(f"{save_path}_BlockDiagonal")
    
    if save_transform_only:
        torch.save({
            'T_full': T_full,
            'block_transforms': block_transforms,
            'block_size': block_size
        }, f"{save_path}_BlockDiagonal_transform")
    
    # Cleanup
    del model, a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_BlockDiagonal"