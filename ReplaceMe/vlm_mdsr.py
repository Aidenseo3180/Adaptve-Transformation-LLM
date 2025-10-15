# ============================================================
# vlm_mdsr.py - Modality-Decomposed Spectral Regularization
# Training-free VLM Pruning with Component-wise Regularization
# ============================================================

import gc
import logging
import os
from typing import Optional, Tuple
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


def identify_vision_tokens_robust(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    image_token_index: int = 32000,
    num_vision_tokens: int = 576,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robustly identify vision and text tokens.
    (Reusing from vlm_cmapt.py)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    if debug:
        print(f"\n{Fore.CYAN}[identify_vision_tokens_robust]{Fore.RESET}")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  hidden_states shape: {hidden_states.shape}")
    
    vision_mask_2d = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for b in range(batch_size):
        image_positions = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
        
        if len(image_positions) == 0:
            if debug:
                print(f"  {Fore.YELLOW}[Sample {b}] No image token found!{Fore.RESET}")
            continue
        
        img_pos = image_positions[0].item()
        vision_start = img_pos + 1
        vision_end = min(vision_start + num_vision_tokens, seq_len)
        
        if debug and b == 0:
            print(f"  [Sample {b}] Image token at: {img_pos}")
            print(f"  [Sample {b}] Vision tokens: [{vision_start}:{vision_end}]")
        
        vision_mask_2d[b, vision_start:vision_end] = True
    
    vision_mask = vision_mask_2d.reshape(-1)
    text_mask = ~vision_mask
    
    num_vision = vision_mask.sum().item()
    num_text = text_mask.sum().item()
    total = num_vision + num_text
    
    print(f"\n{Fore.GREEN}Vision Token Identification:{Fore.RESET}")
    print(f"  Total: {total:,}")
    print(f"  Vision: {num_vision:,} ({num_vision/total*100:.1f}%)")
    print(f"  Text: {num_text:,} ({num_text/total*100:.1f}%)\n")
    
    if num_vision == 0:
        raise ValueError(f"{Fore.RED}ERROR: No vision tokens detected!{Fore.RESET}")
    
    return vision_mask, text_mask


def compute_component_attribution(
    target_activations: torch.Tensor,
    vision_mask: torch.Tensor,
    text_mask: torch.Tensor,
    variance_threshold: float = 2.0,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute per-component modality attribution using variance analysis.
    
    Args:
        target_activations: [N, d] target activations
        vision_mask: [N] boolean mask for vision tokens
        text_mask: [N] boolean mask for text tokens
        variance_threshold: Ratio threshold for attribution
        device: Device for computation
        
    Returns:
        components: [d, d] principal components (V from SVD)
        lambdas: [d] regularization strengths per component
        stats: Dictionary with attribution statistics
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}COMPONENT ATTRIBUTION ANALYSIS{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    # Move to GPU and ensure float64 for SVD
    target_gpu = target_activations.to(device=device, dtype=torch.float64)
    vision_mask_gpu = vision_mask.to(device=device)
    text_mask_gpu = text_mask.to(device=device)
    
    d = target_gpu.shape[1]
    
    print(f"{Fore.CYAN}Computing SVD of target activations...{Fore.RESET}")
    print(f"  Shape: {target_gpu.shape}")
    print(f"  Device: {target_gpu.device}")
    print(f"  Dtype: {target_gpu.dtype}")
    
    # SVD
    try:
        U, S, Vt = torch.linalg.svd(target_gpu, full_matrices=False)
        components = Vt.T  # [d, d] - columns are principal components
        print(f"{Fore.GREEN}  ✓ SVD completed{Fore.RESET}")
        print(f"  Components shape: {components.shape}")
        print(f"  Singular values range: [{S.min():.4f}, {S.max():.4f}]\n")
    except Exception as e:
        print(f"{Fore.RED}SVD failed: {e}{Fore.RESET}")
        raise
    
    # Component attribution
    print(f"{Fore.CYAN}Analyzing component attribution...{Fore.RESET}")
    
    vision_tokens = target_gpu[vision_mask_gpu]
    text_tokens = target_gpu[text_mask_gpu]
    
    print(f"  Vision tokens: {vision_tokens.shape[0]:,}")
    print(f"  Text tokens: {text_tokens.shape[0]:,}\n")
    
    lambdas = torch.zeros(d, device=device, dtype=torch.float64)
    component_types = []
    
    num_vision_dom = 0
    num_text_dom = 0
    num_shared = 0
    
    # Sample for efficiency (analyze top components + sample of rest)
    important_components = min(512, d)  # Analyze top 512
    
    for i in tqdm(range(important_components), desc="Attributing components", colour="cyan"):
        v_i = components[:, i]  # [d]
        
        # Project tokens onto component
        vision_proj = vision_tokens @ v_i  # [num_vision]
        text_proj = text_tokens @ v_i      # [num_text]
        
        # Compute variances
        var_vision = vision_proj.var().item()
        var_text = text_proj.var().item()
        
        # Avoid division by zero
        if var_text < 1e-10:
            var_text = 1e-10
        if var_vision < 1e-10:
            var_vision = 1e-10
        
        ratio = var_vision / var_text
        
        # Attribution based on variance ratio
        if ratio > variance_threshold:
            # Vision-dominant
            lambdas[i] = 0.01  # Low reg = preserve
            component_types.append('vision')
            num_vision_dom += 1
        elif ratio < (1.0 / variance_threshold):
            # Text-dominant
            lambdas[i] = 0.1   # High reg = compress
            component_types.append('text')
            num_text_dom += 1
        else:
            # Shared
            lambdas[i] = 0.05  # Medium reg
            component_types.append('shared')
            num_shared += 1
    
    # For remaining components (if any), use medium regularization
    if important_components < d:
        lambdas[important_components:] = 0.05
        num_shared += (d - important_components)
    
    # Statistics
    stats = {
        'num_vision_dominant': num_vision_dom,
        'num_text_dominant': num_text_dom,
        'num_shared': num_shared,
        'total_analyzed': important_components,
        'vision_ratio': num_vision_dom / important_components,
        'text_ratio': num_text_dom / important_components,
        'shared_ratio': num_shared / important_components
    }
    
    print(f"\n{Fore.GREEN}Component Attribution Results:{Fore.RESET}")
    print(f"  Vision-dominant: {num_vision_dom} ({stats['vision_ratio']*100:.1f}%)")
    print(f"  Text-dominant: {num_text_dom} ({stats['text_ratio']*100:.1f}%)")
    print(f"  Shared: {num_shared} ({stats['shared_ratio']*100:.1f}%)")
    print(f"  Total analyzed: {important_components} / {d}")
    print(f"\n{Fore.CYAN}Regularization strengths:{Fore.RESET}")
    print(f"  Vision (λ=0.01): Preserve complex components")
    print(f"  Text (λ=0.1): Compress simple components")
    print(f"  Shared (λ=0.05): Moderate compression\n")
    
    return components, lambdas, stats


def estimate_mdsr_transform(
    a1: torch.Tensor,
    a2: torch.Tensor,
    vision_masks: torch.Tensor,
    text_masks: torch.Tensor,
    variance_threshold: float = 2.0,
    lr: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, dict]:
    """
    Estimate transformation using MDSR (Modality-Decomposed Spectral Regularization).
    
    Key: Component-wise regularization based on modality attribution.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}MDSR: MODALITY-DECOMPOSED SPECTRAL REGULARIZATION{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    hidden_size = a1.shape[1]
    num_samples = a1.shape[0]
    
    print(f"{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total samples: {num_samples:,}")
    print(f"  Input dtype: {a1.dtype} (CPU)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Variance threshold: {variance_threshold}")
    print(f"  Device: {device}\n")
    
    # Step 1: Component Attribution
    print(f"{Fore.GREEN}[Step 1/3] Component Attribution Analysis{Fore.RESET}")
    
    # Sample for SVD (more efficient)
    sample_size = min(50000, num_samples)
    sample_indices = torch.randperm(num_samples)[:sample_size]
    
    print(f"Using {sample_size:,} samples for component analysis\n")
    
    a2_sample = a2[sample_indices]
    vision_mask_sample = vision_masks[sample_indices]
    text_mask_sample = text_masks[sample_indices]
    
    components, lambdas, stats = compute_component_attribution(
        a2_sample,
        vision_mask_sample,
        text_mask_sample,
        variance_threshold=variance_threshold,
        device=device
    )
    
    # Clean up
    del a2_sample, vision_mask_sample, text_mask_sample
    torch.cuda.empty_cache()
    
    # Step 2: Initialize Transform
    print(f"{Fore.GREEN}[Step 2/3] Initializing transformation matrix{Fore.RESET}")
    
    transform = torch.eye(hidden_size, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform], lr=lr)
    
    print(f"  Shape: {transform.shape}")
    print(f"  Dtype: {transform.dtype}")
    print(f"  Device: {transform.device}\n")
    
    # Step 3: Optimization with Spectral Regularization
    print(f"{Fore.GREEN}[Step 3/3] Optimizing with spectral regularization{Fore.RESET}\n")
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = {
            'total': 0.0,
            'output': 0.0,
            'spectral': 0.0
        }
        
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
            a2_batch = a2[batch_indices].to(device=device, dtype=torch.float64)
            
            if batch_idx == 0 and epoch == 0:
                print(f"\n{Fore.YELLOW}[DEBUG] First batch:{Fore.RESET}")
                print(f"  a1: {a1_batch.shape}, {a1_batch.dtype}, {a1_batch.device}")
                print(f"  a2: {a2_batch.shape}, {a2_batch.dtype}, {a2_batch.device}")
                print(f"  transform: {transform.shape}, {transform.dtype}\n")
            
            # Forward
            pred = a1_batch @ transform
            
            # Loss 1: Output matching
            pred_norm = pred / pred.norm(dim=1, keepdim=True)
            target_norm = a2_batch / a2_batch.norm(dim=1, keepdim=True)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)
            output_loss = (1.0 - cosine_sim).mean()
            
            # Loss 2: Component-wise spectral regularization
            # SVD of current transform
            try:
                _, S_t, _ = torch.linalg.svd(transform, full_matrices=False)
                
                # Weighted regularization (component-wise!)
                spectral_loss = (lambdas * S_t.pow(2)).sum() / hidden_size
                
            except Exception as e:
                if batch_idx == 0:
                    print(f"{Fore.YELLOW}Warning: SVD failed, skipping spectral reg: {e}{Fore.RESET}")
                spectral_loss = torch.tensor(0.0, device=device)
            
            # Combined
            total_loss = output_loss + spectral_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate
            epoch_losses['total'] += total_loss.item()
            epoch_losses['output'] += output_loss.item()
            epoch_losses['spectral'] += spectral_loss.item() if isinstance(spectral_loss, torch.Tensor) else 0.0
            
            # Update progress
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.6f}",
                'Output': f"{output_loss.item():.4f}",
                'Spectral': f"{spectral_loss.item():.4f}" if isinstance(spectral_loss, torch.Tensor) else "0",
                'Best': f"{best_loss:.6f}"
            })
            
            # Cleanup
            del a1_batch, a2_batch, pred
            torch.cuda.empty_cache()
        
        # Epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            print(f"\n{Fore.GREEN}  ✓ New best loss: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Epoch {epoch+1}/{num_epochs}:{Fore.RESET}")
        print(f"  Total: {avg_losses['total']:.6f}")
        print(f"  Output: {avg_losses['output']:.6f}")
        print(f"  Spectral: {avg_losses['spectral']:.6f}")
        print(f"  Best: {best_loss:.6f}\n")
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}MDSR OPTIMIZATION COMPLETED{Fore.RESET}")
    print(f"{Fore.GREEN}Final Loss: {best_loss:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}\n")
    
    return transform.detach(), stats


def vlm_mdsr(
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
    variance_threshold: float = 2.0,
    lr: float = 1e-4,
    num_epochs: int = 10,
    opt_batch_size: int = 1024,
    token: Optional[str] = None
) -> str:
    """
    VLM pruning with Modality-Decomposed Spectral Regularization (MDSR).
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}VLM-MDSR: MODALITY-DECOMPOSED SPECTRAL REGULARIZATION{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{Fore.CYAN}System Info:{Fore.RESET}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device: {device}\n")
    
    # Quantization
    quantization_config = None
    if use_4bit:
        print(f"{Fore.YELLOW}Using 4-bit quantization{Fore.RESET}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"{Fore.CYAN}MDSR Configuration:{Fore.RESET}")
    print(f"  Model: {model_path}")
    print(f"  Layers to skip: {layers_to_skip}")
    print(f"  Variance threshold: {variance_threshold}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}\n")
    
    # Load model
    print(f"{Fore.GREEN}[Step 1/8] Loading model...{Fore.RESET}")
    
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
    image_token_index = model.config.image_token_index
    
    print(f"\n{Fore.GREEN}Model loaded:{Fore.RESET}")
    print(f"  Layers: {num_hidden_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Image token index: {image_token_index}\n")
    
    model.eval()
    
    # Load data
    print(f"{Fore.GREEN}[Step 2/8] Loading calibration data...{Fore.RESET}")
    dataloader = get_vlm_calib_dataloader(image_dir, dataset_size, batch_size, processor)
    print(f"  Batches: {len(dataloader)}\n")
    
    # Setup hooks
    print(f"{Fore.GREEN}[Step 3/8] Setting up hooks...{Fore.RESET}")
    
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
    
    print(f"  Registered {len(hooks)} hooks\n")
    
    target_layer_before = start_id - num_layer - 1
    target_layer_after = end_id - num_layer - 1
    
    print(f"{Fore.CYAN}Target layers:{Fore.RESET}")
    print(f"  MLP layer: {target_layer_before}")
    print(f"  Output layer: {target_layer_after}\n")
    
    # Pre-allocate
    print(f"{Fore.GREEN}[Step 4/8] Pre-allocating memory...{Fore.RESET}")
    
    total_samples = dataset_size if dataset_size else len(dataloader.dataset)
    estimated_tokens_per_image = 1000
    total_tokens = int(total_samples * estimated_tokens_per_image * 1.5)
    
    print(f"  Estimated tokens: {total_tokens:,}\n")
    
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    vision_masks = torch.empty((total_tokens,), dtype=torch.bool, device='cpu')
    text_masks = torch.empty((total_tokens,), dtype=torch.bool, device='cpu')
    
    cnt = 0
    
    # Gather activations
    print(f"{Fore.GREEN}[Step 5/8] Gathering activations...{Fore.RESET}\n")
    
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.RED}Gathering{Fore.RESET}",
        colour="red"
    )):
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] First batch:{Fore.RESET}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, {v.dtype}")
        
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]
        mlp_key = f'layer_{target_layer_before}_mlp'
        
        if mlp_key not in mlp_activations:
            raise KeyError(f"MLP not found: {mlp_key}")
        
        hidden_states_mlp = mlp_activations[mlp_key]
        
        # Reshape
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_i = hidden_states[target_layer_before].view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_n = hidden_states[target_layer_after].view(-1, hidden_size).to(torch.bfloat16)
        
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] Shapes:{Fore.RESET}")
            print(f"  MLP: {hidden_states_mlp.shape}")
            print(f"  Hidden i: {hidden_states_i.shape}")
            print(f"  Hidden n: {hidden_states_n.shape}\n")
        
        # Identify tokens
        if batch_idx == 0:
            vision_mask, text_mask = identify_vision_tokens_robust(
                input_ids,
                hidden_states[target_layer_after],
                image_token_index,
                576,
                debug=True
            )
        else:
            vision_mask, text_mask = identify_vision_tokens_robust(
                input_ids,
                hidden_states[target_layer_after],
                image_token_index,
                576,
                debug=False
            )
        
        # Compute
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        batch_size_tokens = a1_batch.shape[0]
        
        if cnt + batch_size_tokens > total_tokens:
            print(f"\n{Fore.RED}Buffer overflow!{Fore.RESET}")
            print(f"  Stopping at {cnt:,} tokens\n")
            break
        
        # Write
        a1[cnt:cnt+batch_size_tokens] = a1_batch.cpu()
        a2[cnt:cnt+batch_size_tokens] = a2_batch.cpu()
        vision_masks[cnt:cnt+batch_size_tokens] = vision_mask.cpu()
        text_masks[cnt:cnt+batch_size_tokens] = text_mask.cpu()
        
        cnt += batch_size_tokens
        
        if (batch_idx + 1) % 100 == 0:
            avg_tokens = cnt / ((batch_idx + 1) * batch_size)
            print(f"\n  [{batch_idx+1}] Avg tokens/img: {avg_tokens:.1f}")
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n, a1_batch, a2_batch
        torch.cuda.empty_cache()
    
    # Slice
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    vision_masks = vision_masks[:cnt]
    text_masks = text_masks[:cnt]
    
    print(f"\n{Fore.GREEN}Collection complete: {cnt:,} tokens{Fore.RESET}\n")
    
    # Estimate transform
    print(f"{Fore.GREEN}[Step 6/8] Estimating MDSR transform...{Fore.RESET}\n")
    
    transform, stats = estimate_mdsr_transform(
        a1, a2,
        vision_masks, text_masks,
        variance_threshold=variance_threshold,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=opt_batch_size,
        device=device
    )
    
    print(f"{Fore.GREEN}Transform estimated: {transform.shape}, {transform.dtype}{Fore.RESET}\n")
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    del model, a1, a2, vision_masks, text_masks
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload and truncate
    print(f"{Fore.GREEN}[Step 7/8] Reloading and truncating...{Fore.RESET}")
    
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
    print(f"\n{Fore.GREEN}[Step 8/8] Saving model...{Fore.RESET}")
    
    final_path = f"{save_path}_MDSR"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    # Save stats
    import json
    stats_path = f"{final_path}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}✓ MODEL SAVED{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.CYAN}Path: {final_path}{Fore.RESET}")
    print(f"{Fore.CYAN}Stats: {stats_path}{Fore.RESET}\n")
    
    if save_transform_only:
        torch.save(transform, f"{final_path}_transform.pt")
        print(f"{Fore.GREEN}Transform saved{Fore.RESET}\n")
    
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path