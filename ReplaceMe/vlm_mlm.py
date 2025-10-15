# ============================================================
# vlm_mlm.py - Multi-Layer Matching
# Teacher-student inspired multi-layer supervision for VLM pruning
# ============================================================

import gc
import logging
import os
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from colorama import Fore, init
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

from .utils import (
    get_vlm_calib_dataloader, 
    setup_vlm_processor,
    get_vlm_layers,
    truncate_vlm_model,
    apply_vlm_transform,
    seed_all
)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def compute_layer_weights(
    num_layers: int,
    strategy: str = 'linear_increasing'
) -> torch.Tensor:
    """
    Compute weights for multi-layer matching.
    
    Args:
        num_layers: Number of layers to match
        strategy: Weighting strategy
            - 'linear_increasing': Far layers more important
            - 'exponential': Exponential decay to final layer
            - 'uniform': All layers equal
            
    Returns:
        weights: [num_layers] normalized weights
    """
    print(f"\n{Fore.CYAN}Computing layer weights:{Fore.RESET}")
    print(f"  Num layers: {num_layers}")
    print(f"  Strategy: {strategy}\n")
    
    if strategy == 'linear_increasing':
        # Li+1: low weight, Li+n: high weight
        weights = torch.linspace(0.1, 1.0, num_layers)
        
    elif strategy == 'exponential':
        # Exponential increase toward final layer
        weights = torch.tensor([0.5 ** (num_layers - k - 1) for k in range(num_layers)])
        
    elif strategy == 'uniform':
        # All equal
        weights = torch.ones(num_layers)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Normalize
    weights = weights / weights.sum()
    
    print(f"{Fore.GREEN}Layer weights:{Fore.RESET}")
    for k in range(num_layers):
        print(f"  Layer i+{k+1}: {weights[k].item():.4f}")
    print(f"  Sum: {weights.sum().item():.4f}\n")
    
    return weights


def multi_layer_loss(
    pred: torch.Tensor,
    target_list: List[torch.Tensor],
    weights: torch.Tensor,
    debug: bool = False
) -> Tuple[torch.Tensor, dict]:
    """
    Compute weighted multi-layer matching loss.
    
    Args:
        pred: Predicted activations [N, d]
        target_list: List of target activations [[N, d], ...]
        weights: Layer weights [num_layers]
        debug: Print detailed breakdown
        
    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Individual losses per layer
    """
    # Ensure float64
    pred = pred.to(torch.float64)
    
    num_layers = len(target_list)
    losses = []
    
    for k, target in enumerate(target_list):
        target = target.to(torch.float64)
        
        # Cosine distance
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        loss_k = (1.0 - cosine_sim).mean()
        
        losses.append(loss_k)
    
    # Weighted combination
    losses_tensor = torch.stack(losses)  # [num_layers]
    weights = weights.to(device=losses_tensor.device, dtype=losses_tensor.dtype)
    total_loss = (losses_tensor * weights).sum()
    
    # Loss dict
    loss_dict = {
        'total': total_loss.item(),
    }
    for k in range(num_layers):
        loss_dict[f'layer_{k+1}'] = losses[k].item()
        loss_dict[f'weighted_{k+1}'] = (losses[k] * weights[k]).item()
    
    if debug:
        print(f"\n{Fore.YELLOW}[Multi-Layer Loss Breakdown]{Fore.RESET}")
        for k in range(num_layers):
            print(f"  Layer i+{k+1}: {losses[k].item():.6f} × {weights[k].item():.4f} = {loss_dict[f'weighted_{k+1}']:.6f}")
        print(f"  {Fore.GREEN}TOTAL: {total_loss.item():.6f}{Fore.RESET}")
    
    return total_loss, loss_dict


def estimate_mlm_transform(
    a1: torch.Tensor,
    a2_list: List[torch.Tensor],
    layer_weights: torch.Tensor,
    lr: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Estimate transformation using Multi-Layer Matching (MLM).
    
    Args:
        a1: Input activations [N, d]
        a2_list: List of target activations for each layer
        layer_weights: Weights for each layer [num_layers]
        lr: Learning rate
        num_epochs: Number of epochs
        batch_size: Batch size
        device: Device
        
    Returns:
        transform: Optimized transformation [d, d]
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}MLM: MULTI-LAYER MATCHING OPTIMIZATION{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    hidden_size = a1.shape[1]
    num_samples = a1.shape[0]
    num_layers = len(a2_list)
    
    print(f"{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total samples: {num_samples:,}")
    print(f"  Number of teacher layers: {num_layers}")
    print(f"  Input dtype: {a1.dtype} (CPU)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}\n")
    
    # Initialize transform
    print(f"{Fore.GREEN}Initializing transformation matrix...{Fore.RESET}")
    transform = torch.eye(hidden_size, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform], lr=lr)
    
    print(f"  Shape: {transform.shape}")
    print(f"  Dtype: {transform.dtype}")
    print(f"  Device: {transform.device}\n")
    
    # Optimization
    print(f"{Fore.GREEN}Starting multi-layer optimization...{Fore.RESET}\n")
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = {
            'total': 0.0,
        }
        for k in range(num_layers):
            epoch_losses[f'layer_{k+1}'] = 0.0
        
        # Shuffle
        indices = torch.randperm(num_samples)
        
        pbar = tqdm(
            range(num_batches),
            desc=f"{Fore.CYAN}Epoch {epoch+1}/{num_epochs}{Fore.RESET}",
            leave=True,
            dynamic_ncols=True
        )
        
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch (convert to float64 on GPU)
            a1_batch = a1[batch_indices].to(device=device, dtype=torch.float64)
            
            # Get all target batches
            target_batches = []
            for a2 in a2_list:
                target_batches.append(
                    a2[batch_indices].to(device=device, dtype=torch.float64)
                )
            
            if batch_idx == 0 and epoch == 0:
                print(f"\n{Fore.YELLOW}[DEBUG] First batch:{Fore.RESET}")
                print(f"  a1_batch: {a1_batch.shape}, {a1_batch.dtype}, {a1_batch.device}")
                print(f"  Target layers: {len(target_batches)}")
                for k, tb in enumerate(target_batches):
                    print(f"    Layer {k+1}: {tb.shape}, {tb.dtype}")
                print()
            
            # Forward
            pred = a1_batch @ transform
            
            # Multi-layer loss
            loss, loss_dict = multi_layer_loss(
                pred, target_batches,
                layer_weights,
                debug=(batch_idx == 0 and epoch == 0)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            
            # Update progress
            postfix = {'Loss': f"{loss_dict['total']:.6f}"}
            for k in range(min(3, num_layers)):  # Show first 3 layers
                postfix[f'L{k+1}'] = f"{loss_dict[f'layer_{k+1}']:.4f}"
            postfix['Best'] = f"{best_loss:.6f}"
            pbar.set_postfix(postfix)
            
            # Cleanup
            del a1_batch, target_batches, pred
            torch.cuda.empty_cache()
        
        # Epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            print(f"\n{Fore.GREEN}  ✓ New best: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Epoch {epoch+1}/{num_epochs}:{Fore.RESET}")
        print(f"  Total: {avg_losses['total']:.6f}")
        for k in range(num_layers):
            print(f"  Layer i+{k+1}: {avg_losses[f'layer_{k+1}']:.6f}")
        print(f"  Best: {best_loss:.6f}\n")
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}OPTIMIZATION COMPLETED{Fore.RESET}")
    print(f"{Fore.GREEN}Final Loss: {best_loss:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}\n")
    
    return transform.detach()


def vlm_mlm(
    model_path: str,
    image_dir: str = "train2014",
    batch_size: int = 4,
    max_length: int = 512,
    layers_to_skip: int = 8,
    dataset_size: Optional[int] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    save_transform_only: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    weight_strategy: str = 'linear_increasing',
    lr: float = 1e-4,
    num_epochs: int = 10,
    opt_batch_size: int = 1024,
    token: Optional[str] = None
) -> str:
    """
    VLM pruning with Multi-Layer Matching (MLM).
    
    Uses multiple intermediate layers as teacher signals.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}VLM-MLM: MULTI-LAYER MATCHING{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{Fore.CYAN}System:{Fore.RESET}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}\n")
    
    # Quantization
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    num_intermediate_layers = end_id - start_id
    
    print(f"{Fore.CYAN}MLM Configuration:{Fore.RESET}")
    print(f"  Model: {model_path}")
    print(f"  Layers to skip: {layers_to_skip}")
    print(f"  Intermediate layers to use: {num_intermediate_layers}")
    print(f"  Weight strategy: {weight_strategy}\n")
    
    # Load model
    print(f"{Fore.GREEN}[1/8] Loading model...{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    processor = setup_vlm_processor(model_path)
    layers, num_hidden_layers = get_vlm_layers(model)
    hidden_size = model.config.text_config.hidden_size
    
    print(f"  Layers: {num_hidden_layers}, Hidden: {hidden_size}\n")
    model.eval()
    
    # Load data
    print(f"{Fore.GREEN}[2/8] Loading data...{Fore.RESET}")
    dataloader = get_vlm_calib_dataloader(image_dir, dataset_size, batch_size, processor)
    print(f"  Batches: {len(dataloader)}\n")
    
    # Setup hooks
    print(f"{Fore.GREEN}[3/8] Setting up hooks...{Fore.RESET}")
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    
    for i, layer in enumerate(layers):
        hooks.append(
            layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        )
    
    target_layer_before = start_id - num_layer - 1
    target_layer_after = end_id - num_layer - 1
    
    print(f"  Hooks: {len(hooks)}")
    print(f"  Target: layer {target_layer_before} → {target_layer_after}")
    print(f"  Will collect {num_intermediate_layers} intermediate layers\n")
    
    # Pre-allocate
    print(f"{Fore.GREEN}[4/8] Pre-allocating memory...{Fore.RESET}")
    
    total_samples = dataset_size if dataset_size else len(dataloader.dataset)
    total_tokens = int(total_samples * 1000 * 1.5)
    
    # Allocate for input and all intermediate targets
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2_list = [
        torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
        for _ in range(num_intermediate_layers)
    ]
    
    print(f"  Allocated: {total_tokens:,} tokens")
    print(f"  Number of target layers: {num_intermediate_layers}\n")
    
    cnt = 0
    
    # Gather activations
    print(f"{Fore.GREEN}[5/8] Gathering activations (with intermediate layers)...{Fore.RESET}\n")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering", colour="red")):
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]  # Skip embedding
        mlp_key = f'layer_{target_layer_before}_mlp'
        
        if mlp_key not in mlp_activations:
            raise KeyError(f"MLP not found: {mlp_key}")
        
        hidden_states_mlp = mlp_activations[mlp_key]
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_i = hidden_states[target_layer_before].view(-1, hidden_size).to(torch.bfloat16)
        
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] First batch:{Fore.RESET}")
            print(f"  MLP shape: {hidden_states_mlp.shape}")
            print(f"  Hidden i shape: {hidden_states_i.shape}")
            print(f"  Collecting intermediate layers: {target_layer_before+1} to {target_layer_after}")
        
        # Input: Mi
        a1_batch = hidden_states_mlp
        
        # Targets: Li+1, Li+2, ..., Li+n
        a2_batches = []
        for k in range(num_intermediate_layers):
            layer_idx = target_layer_before + k + 1  # Li+1, Li+2, ...
            hidden_states_k = hidden_states[layer_idx].view(-1, hidden_size).to(torch.bfloat16)
            
            # Target: Lk + Mi - Yi (ReplaceMe formula)
            a2_k = hidden_states_k + hidden_states_mlp - hidden_states_i
            a2_batches.append(a2_k)
            
            if batch_idx == 0:
                print(f"  Layer i+{k+1} (layer {layer_idx}): {a2_k.shape}")
        
        batch_size_tokens = a1_batch.shape[0]
        
        if cnt + batch_size_tokens > total_tokens:
            print(f"\n{Fore.RED}Buffer full{Fore.RESET}\n")
            break
        
        # Write to buffers
        a1[cnt:cnt+batch_size_tokens] = a1_batch.cpu()
        for k, a2_k in enumerate(a2_batches):
            a2_list[k][cnt:cnt+batch_size_tokens] = a2_k.cpu()
        
        cnt += batch_size_tokens
        
        # Cleanup
        del hidden_states_mlp, hidden_states_i, a1_batch, a2_batches
        torch.cuda.empty_cache()
    
    # Slice to actual size
    a1 = a1[:cnt]
    for k in range(num_intermediate_layers):
        a2_list[k] = a2_list[k][:cnt]
    
    print(f"\n{Fore.GREEN}Collection complete: {cnt:,} tokens{Fore.RESET}\n")
    
    # Compute layer weights
    layer_weights = compute_layer_weights(num_intermediate_layers, weight_strategy)
    
    # MLM optimization
    print(f"{Fore.GREEN}[6/8] MLM optimization...{Fore.RESET}\n")
    
    transform = estimate_mlm_transform(
        a1, a2_list,
        layer_weights,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=opt_batch_size,
        device=device
    )
    
    print(f"{Fore.GREEN}Transform ready{Fore.RESET}\n")
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    del model, a1, a2_list
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload
    print(f"{Fore.GREEN}[7/8] Reloading and truncating...{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    model = truncate_vlm_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/vlm_{os.path.basename(model_path)}_{layers_to_skip}layers"
    
    apply_vlm_transform(model, transform, start_id - num_layer - 1)
    
    # Save
    print(f"\n{Fore.GREEN}[8/8] Saving...{Fore.RESET}")
    
    final_path = f"{save_path}_MLM"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}✓ MODEL SAVED{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.CYAN}Path: {final_path}{Fore.RESET}\n")
    
    if save_transform_only:
        torch.save(transform, f"{final_path}_transform.pt")
    
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path