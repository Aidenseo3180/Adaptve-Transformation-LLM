# fixed_adaptive_replaceme.py
import gc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
seed_all()


def fixed_adaptive_replaceme(
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
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    adaptive_weight: float = 0.1,  # How much to blend with identity
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    
    print(f"\n[Fixed AR] Processing layers {start_id}-{end_id}")
    print(f"[Fixed AR] Skip size: {end_id - start_id} layers")
    
    # Load model for gathering activations
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.float32  # Use float32 for accuracy
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    # Gather calibration data - JUST LIKE ORIGINAL REPLACEME
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    print(f"[Fixed AR] Gathering calibration data...")
    
    # Hook to capture MLP outputs
    mlp_activations = {}
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
    # Collect activations
    a1_list = []
    a2_list = []
    
    for batch in tqdm(dataloader, desc="Calibration"):
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
        
        hidden_states = outputs.hidden_states[1:]  # Skip embeddings
        
        # Get MLP output and hidden states at key positions
        mlp_out = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
        h_before = hidden_states[start_id - num_layer - 1]
        h_after = hidden_states[end_id - num_layer - 1]
        
        # Compute what we need to match
        # a1 = MLP output, a2 = target (h_after + mlp_out - h_before)
        a1 = mlp_out.view(-1, hidden_size)
        a2 = (h_after + mlp_out - h_before).view(-1, hidden_size)
        
        a1_list.append(a1.cpu())
        a2_list.append(a2.cpu())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate
    a1 = torch.cat(a1_list, dim=0).to(torch.float64)
    a2 = torch.cat(a2_list, dim=0).to(torch.float64)
    
    print(f"[Fixed AR] Collected {a1.shape[0]} samples")
    
    # Analyze relationship
    residual_norm = torch.norm(a2 - a1) / torch.norm(a1)
    print(f"[Fixed AR] Residual norm ratio: {residual_norm:.3f}")
    
    # Train transformation matrix with adaptive component
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize closer to identity
    W = torch.eye(hidden_size, dtype=torch.float64, device=device)
    W.requires_grad_(True)
    
    optimizer = torch.optim.Adam([W], lr=1e-3)
    
    print(f"[Fixed AR] Training adaptive transformation...")
    
    a1 = a1.to(device).float()
    a2 = a2.to(device).float()
    
    best_W = W.clone()
    best_loss = float('inf')
    
    for epoch in range(20):
        total_loss = 0
        num_batches = 0
        
        indices = torch.randperm(a1.shape[0])
        
        for i in range(0, len(indices), 1024):
            batch_idx = indices[i:min(i+1024, len(indices))]
            batch_a1 = a1[batch_idx]
            batch_a2 = a2[batch_idx]
            
            optimizer.zero_grad()
            
            # Compute transformation with adaptive blending
            pred = batch_a1 @ W.float()
            
            # Cosine loss
            pred_norm = F.normalize(pred, dim=-1)
            a2_norm = F.normalize(batch_a2, dim=-1)
            loss = 1 - (pred_norm * a2_norm).sum(-1).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_W = W.clone()
        
        if epoch % 3 == 0:
            print(f"[Fixed AR] Epoch {epoch}: Loss={avg_loss:.4f}")
    
    print(f"[Fixed AR] Training done. Best loss: {best_loss:.4f}")
    
    # Apply adaptive blending with identity
    identity = torch.eye(hidden_size, dtype=torch.float64, device=device)
    W_final = (1 - adaptive_weight) * best_W + adaptive_weight * identity
    
    print(f"[Fixed AR] Applied adaptive blending (weight={adaptive_weight})")
    
    # Clean up
    del model, a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload and modify model
    print(f"[Fixed AR] Modifying model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate layers
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj (LIKE ORIGINAL REPLACEME)
    target_layer = model.model.layers[start_id - num_layer - 1]
    original_weight = target_layer.mlp.down_proj.weight.to(torch.float64)
    
    # Apply: new_weight = W^T @ original_weight
    new_weight = W_final.T.cpu() @ original_weight
    target_layer.mlp.down_proj.weight = nn.Parameter(new_weight.to(torch.bfloat16))
    
    print(f"[Fixed AR] Updated down_proj weight")
    
    # Save
    if save_path is None:
        save_path = f"output_models/FixedAR_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[Fixed AR] Model saved to {save_path}")
    
    # Save transformation info
    torch.save({
        'W': W_final.cpu(),
        'adaptive_weight': adaptive_weight,
        'best_loss': best_loss
    }, f"{save_path}/transform_info.pt")
    
    return save_path