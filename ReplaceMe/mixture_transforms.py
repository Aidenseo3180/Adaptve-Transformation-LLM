# mixture_transforms.py
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

from .utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_non_overlapping_blocks, truncate_model, seed_all)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

def mixture_transforms(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
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
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    num_mixtures: int = 4,  # Number of transform mixtures
    **kwargs
) -> str:
    """
    Mixture of Transforms: Learn K different transforms and combine them
    """
    print(f"[DEBUG] Using Mixture of Transforms with K={num_mixtures}")
    
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
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    mlp_activations = {}
    
    # Collect all activations
    all_a1 = []
    all_a2 = []
    all_a3 = [] if accurate else None
    
    print("[DEBUG] Gathering activations...")
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
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
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        a1_batch = hidden_states_mlp
        if accurate:
            a2_batch = hidden_states_n
            a3_batch = hidden_states_i - hidden_states_mlp
            all_a3.append(a3_batch.cpu())
        else:
            a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
            
        all_a1.append(a1_batch.cpu())
        all_a2.append(a2_batch.cpu())
    
    # Concatenate all batches
    a1 = torch.cat(all_a1, dim=0)
    a2 = torch.cat(all_a2, dim=0)
    if accurate:
        a3 = torch.cat(all_a3, dim=0)
    else:
        a3 = None
    
    print(f"[DEBUG] Total samples: {a1.shape[0]}")
    
    # Learn K different transforms
    transforms = []
    weights = []
    
    print(f"[DEBUG] Learning {num_mixtures} different transforms...")
    for k in range(num_mixtures):
        # Use different subset for each transform
        subset_size = len(a1) // num_mixtures
        start_idx = k * subset_size
        end_idx = (k + 1) * subset_size if k < num_mixtures - 1 else len(a1)
        
        print(f"[DEBUG] Transform {k+1}/{num_mixtures}: using samples {start_idx} to {end_idx}")
        
        a1_subset = a1[start_idx:end_idx]
        a2_subset = a2[start_idx:end_idx]
        a3_subset = a3[start_idx:end_idx] if accurate else None
        
        # Learn transform for this subset
        if solver == "adam":
            T_k = adam_method(
                a1_subset, a2_subset, 
                a3=a3_subset,
                loss=loss, diag=diag, 
                two_vectors=two_vectors, thri=thri
            )
        else:
            T_k = optimizing_method(
                a1_subset, a2_subset,
                a3=a3_subset,
                solver=solver
            )

        # Ensure T_k is on CPU
        T_k = T_k.cpu() if hasattr(T_k, 'cpu') else T_k
        transforms.append(T_k)

        # Calculate importance weight based on how well this transform works
        with torch.no_grad():
            pred = a1 @ T_k  # Both on CPU now
            if accurate and a3 is not None:
                pred = pred + a3
                
            # Cosine similarity as weight
            pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
            a2_norm = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
            cosine_sim = (pred_norm * a2_norm).sum(dim=-1).mean()
            weight = cosine_sim.item()
            
        weights.append(weight)
        print(f"[DEBUG] Transform {k+1} weight: {weight:.4f}")
    
    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float64)
    weights = torch.softmax(weights * 2.0, dim=0)  # Temperature scaling
    print(f"[DEBUG] Normalized weights: {weights}")
    
    # Combine transforms
    print("[DEBUG] Combining transforms with weighted average...")
    T_final = torch.zeros_like(transforms[0], dtype=torch.float64)
    for w, T in zip(weights, transforms):
        T_final = T_final + w * T
    
    print(f"[DEBUG] Final transform - Max: {T_final.max():.4f}, Min: {T_final.min():.4f}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
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
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_mixture{num_mixtures}"
        ).replace("/", "_")
    
    # Apply transformation
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (T_final.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    model.save_pretrained(f"{save_path}_MixtureTransforms_{loss}_{solver}")
    tokenizer.save_pretrained(f"{save_path}_MixtureTransforms_{loss}_{solver}")
    
    if save_transform_only:
        torch.save({
            'T_final': T_final,
            'transforms': transforms,
            'weights': weights,
            'num_mixtures': num_mixtures
        }, f"{save_path}_MixtureTransforms_{loss}_{solver}_transform")
    
    # Final cleanup
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_MixtureTransforms_{loss}_{solver}"