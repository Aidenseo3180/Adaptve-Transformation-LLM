# conditional_linear_transform.py
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
from sklearn.cluster import KMeans
import numpy as np

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

def conditional_linear_transform(
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
    n_clusters: int = 4,  # Number of conditional clusters
    **kwargs
) -> str:
    """
    Conditional Linear Transform: Different transforms for different activation patterns
    """
    print(f"[DEBUG] Using Conditional Linear Transform with {n_clusters} clusters")
    print(f"[DEBUG] Processing layers {start_id} to {end_id}")
    
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
    mlp_activations = {}
    
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Collect activations
    all_Mi = []
    all_Yi = []
    all_Li_n = []
    
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
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape and store
        Mi = hidden_states_mlp.view(-1, hidden_size)
        Yi = hidden_states_i.view(-1, hidden_size)
        Li_n = hidden_states_n.view(-1, hidden_size)
        
        all_Mi.append(Mi.cpu().to(torch.float32))
        all_Yi.append(Yi.cpu().to(torch.float32))
        all_Li_n.append(Li_n.cpu().to(torch.float32))
    
    # Concatenate all batches
    Mi_all = torch.cat(all_Mi, dim=0)
    Yi_all = torch.cat(all_Yi, dim=0)
    Li_n_all = torch.cat(all_Li_n, dim=0)
    
    print(f"[DEBUG] Total samples: {Mi_all.shape[0]}")
    
    # ========== STEP 1: Cluster activations ==========
    print(f"[DEBUG] Clustering activations into {n_clusters} groups...")
    
    # Use a subset for clustering to save memory
    cluster_sample_size = min(10000, Mi_all.shape[0])
    indices = torch.randperm(Mi_all.shape[0])[:cluster_sample_size]
    Mi_sample = Mi_all[indices].numpy()
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_sample = kmeans.fit_predict(Mi_sample)
    
    # Assign all data points to clusters
    print("[DEBUG] Assigning all samples to clusters...")
    Mi_numpy = Mi_all.numpy()
    cluster_labels = kmeans.predict(Mi_numpy)
    
    # Get cluster centers
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    # ========== STEP 2: Compute transform for each cluster ==========
    print("[DEBUG] Computing linear transform for each cluster...")
    
    transforms = []
    cluster_sizes = []
    
    for c in range(n_clusters):
        mask = (cluster_labels == c)
        cluster_size = mask.sum()
        cluster_sizes.append(cluster_size)
        
        if cluster_size < 100:  # Too few samples
            print(f"[WARNING] Cluster {c} has only {cluster_size} samples, using identity")
            T_c = torch.eye(hidden_size, dtype=torch.float32)
        else:
            # Get cluster samples
            Mi_c = Mi_all[mask]
            Li_n_c = Li_n_all[mask]
            Yi_c = Yi_all[mask]
            
            # Target: Li_n - Yi (as in original ReplaceMe)
            target_c = Li_n_c - Yi_c
            
            # Solve least squares for this cluster
            MtM = Mi_c.T @ Mi_c
            MtM_reg = MtM + 1e-4 * torch.eye(hidden_size, dtype=torch.float32)
            
            try:
                T_c = torch.linalg.solve(MtM_reg, Mi_c.T @ target_c)
                print(f"[DEBUG] Cluster {c}: {cluster_size} samples, transform computed")
            except:
                print(f"[WARNING] Failed to compute transform for cluster {c}, using identity")
                T_c = torch.eye(hidden_size, dtype=torch.float32)
        
        transforms.append(T_c)
    
    print(f"[DEBUG] Cluster sizes: {cluster_sizes}")
    
    # ========== STEP 3: Compute average transform for runtime efficiency ==========
    # Since we can't dynamically route at inference, we'll create a weighted average
    # based on cluster frequencies
    
    cluster_weights = torch.tensor(cluster_sizes, dtype=torch.float32)
    cluster_weights = cluster_weights / cluster_weights.sum()
    
    print(f"[DEBUG] Cluster weights: {cluster_weights}")
    
    # Weighted average of transforms
    T_final = torch.zeros((hidden_size, hidden_size), dtype=torch.float32)
    for i, (T, w) in enumerate(zip(transforms, cluster_weights)):
        T_final += w * T
    
    print(f"[DEBUG] Final weighted transform computed")
    
    # Optional: Verify improvement over single transform
    with torch.no_grad():
        # Single transform error
        MtM_single = Mi_all.T @ Mi_all
        MtM_single_reg = MtM_single + 1e-4 * torch.eye(hidden_size, dtype=torch.float32)
        T_single = torch.linalg.solve(MtM_single_reg, Mi_all.T @ (Li_n_all - Yi_all))
        error_single = ((Mi_all @ T_single + Yi_all - Li_n_all) ** 2).mean().sqrt()
        
        # Conditional transform error (approximate)
        error_conditional = ((Mi_all @ T_final + Yi_all - Li_n_all) ** 2).mean().sqrt()
        
        print(f"[DEBUG] Single transform error: {error_single:.4f}")
        print(f"[DEBUG] Conditional transform error: {error_conditional:.4f}")
        
        if error_conditional > error_single:
            print("[WARNING] Conditional transform has higher error, falling back to single transform")
            T_final = T_single
    
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
        save_path = f"output_models/{model_path.replace('/', '_')}_{layers_to_skip}layers_conditional{n_clusters}"
    
    # Apply transformation (convert to float64 for compatibility)
    T_final_64 = T_final.to(torch.float64)
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (T_final_64.T @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.data = new_weight
    
    model.save_pretrained(f"{save_path}_ConditionalLinear")
    tokenizer.save_pretrained(f"{save_path}_ConditionalLinear")
    
    if save_transform_only:
        torch.save({
            'T_final': T_final,
            'transforms': transforms,
            'cluster_centers': cluster_centers,
            'cluster_weights': cluster_weights,
            'n_clusters': n_clusters
        }, f"{save_path}_ConditionalLinear_transform")
    
    print("[DEBUG] Conditional Linear Transform completed successfully")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_ConditionalLinear"