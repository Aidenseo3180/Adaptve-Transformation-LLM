# ============================================================
# vlm_modality_cosine_dist.py
# Modality-Aware Weighted Cosine Loss Implementation
# WITH EXTENSIVE DEBUGGING
# ============================================================

import gc
import logging
import os
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
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


def identify_modality_tokens(
    input_ids: torch.Tensor,
    image_token_index: int = 32000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify vision and text token positions in the sequence.
    
    Args:
        input_ids: Token IDs [batch_size, seq_len] or [batch*seq]
        image_token_index: Special token ID for image patches
        
    Returns:
        vision_mask: Boolean mask for vision tokens
        text_mask: Boolean mask for text tokens
    """
    print(f"\n{Fore.CYAN}[DEBUG identify_modality_tokens]{Fore.RESET}")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  image_token_index: {image_token_index}")
    
    # Vision tokens are marked with image_token_index
    vision_mask = (input_ids == image_token_index)
    text_mask = ~vision_mask
    
    num_vision = vision_mask.sum().item()
    num_text = text_mask.sum().item()
    
    print(f"  Vision tokens: {num_vision}")
    print(f"  Text tokens: {num_text}")
    print(f"  Vision ratio: {num_vision / (num_vision + num_text) * 100:.1f}%")
    
    if num_vision == 0:
        print(f"{Fore.RED}  WARNING: No vision tokens found! Check image processing.{Fore.RESET}")
    
    return vision_mask, text_mask


def modality_aware_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vision_mask: torch.Tensor,
    text_mask: torch.Tensor,
    lambda_vision: float = 2.0,
    lambda_text: float = 1.0,
    eps: float = 1e-8,
    debug: bool = False
) -> torch.Tensor:
    """
    Compute modality-aware weighted cosine distance loss.
    
    Args:
        pred: Predicted activations [N, hidden_size]
        target: Target activations [N, hidden_size]
        vision_mask: Vision token mask [N]
        text_mask: Text token mask [N]
        lambda_vision: Weight for vision tokens
        lambda_text: Weight for text tokens
        eps: Small value for numerical stability
        debug: Print detailed debug info
        
    Returns:
        Weighted cosine distance loss (scalar)
    """
    # Ensure same device and type
    pred = pred.to(torch.float64)
    target = target.to(torch.float64)
    vision_mask = vision_mask.to(pred.device)
    text_mask = text_mask.to(pred.device)
    
    # Compute cosine similarity per token
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)  # [N]
    
    # Cosine distance = 1 - cosine_similarity
    cosine_dist = 1.0 - cosine_sim
    
    # Apply modality-specific weights
    weighted_dist = torch.zeros_like(cosine_dist)
    
    if vision_mask.sum() > 0:
        weighted_dist[vision_mask] = lambda_vision * cosine_dist[vision_mask]
    
    if text_mask.sum() > 0:
        weighted_dist[text_mask] = lambda_text * cosine_dist[text_mask]
    
    total_loss = weighted_dist.mean()
    
    if debug:
        print(f"\n{Fore.YELLOW}[DEBUG modality_aware_cosine_loss]{Fore.RESET}")
        print(f"  Total tokens: {len(cosine_dist)}")
        if vision_mask.sum() > 0:
            vision_loss = cosine_dist[vision_mask].mean().item()
            print(f"  Vision loss (unweighted): {vision_loss:.6f}")
            print(f"  Vision loss (weighted): {vision_loss * lambda_vision:.6f}")
        if text_mask.sum() > 0:
            text_loss = cosine_dist[text_mask].mean().item()
            print(f"  Text loss (unweighted): {text_loss:.6f}")
            print(f"  Text loss (weighted): {text_loss * lambda_text:.6f}")
        print(f"  Combined loss: {total_loss.item():.6f}")
    
    return total_loss


def estimate_transform_modality_aware(
    a1: torch.Tensor,
    a2: torch.Tensor,
    vision_masks: torch.Tensor,
    text_masks: torch.Tensor,
    lambda_vision: float = 2.0,
    lambda_text: float = 1.0,
    lr: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Estimate transformation matrix using modality-aware cosine loss.
    """
    print(f"\n{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}STARTING MODALITY-AWARE TRANSFORM ESTIMATION{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    hidden_size = a1.shape[1]
    num_samples = a1.shape[0]
    
    print(f"\n{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Lambda vision: {lambda_vision:.4f}")
    print(f"  Lambda text: {lambda_text:.4f}")
    print(f"  Device: {device}")
    
    # Initialize transform as identity
    transform = torch.eye(hidden_size, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform], lr=lr)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\n{Fore.GREEN}Starting optimization...{Fore.RESET}")
    print(f"  Batches per epoch: {num_batches}")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        
        pbar = tqdm(range(num_batches), desc=f"{Fore.CYAN}Epoch {epoch+1}/{num_epochs}{Fore.RESET}", leave=True)
        
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch
            a1_batch = a1[batch_indices].to(device)
            a2_batch = a2[batch_indices].to(device)
            vision_mask_batch = vision_masks[batch_indices].to(device)
            text_mask_batch = text_masks[batch_indices].to(device)
            
            # Forward pass
            pred = a1_batch @ transform
            
            # Compute modality-aware loss
            loss = modality_aware_cosine_loss(
                pred, a2_batch,
                vision_mask_batch, text_mask_batch,
                lambda_vision, lambda_text,
                debug=(i == 0 and epoch == 0)  # Debug first batch of first epoch
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Best': f'{best_loss:.6f}'
            })
            
            # Memory cleanup
            del a1_batch, a2_batch, pred, vision_mask_batch, text_mask_batch
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\n{Fore.GREEN}  ✓ New best loss: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f} - Best: {best_loss:.6f}{Fore.RESET}\n")
    
    print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}OPTIMIZATION COMPLETED{Fore.RESET}")
    print(f"{Fore.GREEN}Final Loss: {best_loss:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*60}{Fore.RESET}\n")
    
    return transform.detach()


def vlm_modality_cosine_dist(
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
    lambda_vision: float = 2.0,
    lambda_text: float = 1.0,
    auto_lambda: bool = True,
    lr: float = 1e-4,
    num_epochs: int = 10,
    opt_batch_size: int = 1024,
    accurate: bool = False,
    token: Optional[str] = None
) -> str:
    """
    VLM pruning with modality-aware weighted cosine distance.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}VLM MODALITY-AWARE PRUNING PIPELINE{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{Fore.CYAN}System Info:{Fore.RESET}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device map: {device_map}")
    print(f"  Compute device: {device}\n")
    
    quantization_config = None
    if use_4bit:
        print(f"{Fore.YELLOW}Using 4-bit quantization{Fore.RESET}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"{Fore.CYAN}Pruning Configuration:{Fore.RESET}")
    print(f"  Model: {model_path}")
    print(f"  Image dir: {image_dir}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Layers to skip: {layers_to_skip}")
    print(f"  Start ID: {start_id}, End ID: {end_id}, Num layer: {num_layer}")
    print(f"  Auto lambda: {auto_lambda}")
    if not auto_lambda:
        print(f"  Manual lambda_vision: {lambda_vision}, lambda_text: {lambda_text}")
    print(f"  Optimization: lr={lr}, epochs={num_epochs}, batch_size={opt_batch_size}\n")
    
    # Load model
    print(f"{Fore.GREEN}[Step 1/7] Loading VLM model...{Fore.RESET}")
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
    
    # Get image token index
    image_token_index = model.config.image_token_index
    
    print(f"\n{Fore.GREEN}Model loaded successfully:{Fore.RESET}")
    print(f"  Language model layers: {num_hidden_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Image token index: {image_token_index}")
    print(f"  Model device: {model.device}\n")
    
    model.eval()
    
    # Get dataloader
    print(f"{Fore.GREEN}[Step 2/7] Loading calibration data...{Fore.RESET}")
    dataloader = get_vlm_calib_dataloader(
        image_dir,
        dataset_size,
        batch_size,
        processor
    )
    print(f"  Dataloader batches: {len(dataloader)}\n")
    
    # Setup hooks for MLP activations
    print(f"{Fore.GREEN}[Step 3/7] Setting up activation hooks...{Fore.RESET}")
    
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
    
    print(f"  Registered {len(hooks)} hooks on language model layers\n")
    
    # Storage for activations and masks
    a1_list = []
    a2_list = []
    if accurate:
        a3_list = []
    vision_mask_list = []
    text_mask_list = []
    
    target_layer_before = start_id - num_layer - 1
    target_layer_after = end_id - num_layer - 1
    
    print(f"{Fore.CYAN}Target layers for activation capture:{Fore.RESET}")
    print(f"  Layer before pruned block (MLP): {target_layer_before}")
    print(f"  Layer after pruned block: {target_layer_after}")
    print(f"  Accurate mode: {accurate}\n")
    
    # Gather activations
    print(f"{Fore.GREEN}[Step 4/7] Gathering activations...{Fore.RESET}")
    
    batch_count = 0
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.RED}Gathering Activations{Fore.RESET}",
        dynamic_ncols=True,
        colour="red"
    )):
        batch_count += 1
        
        # Debug first batch
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] First batch info:{Fore.RESET}")
            print(f"  Batch keys: {batch.keys()}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get hidden states
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
        
        # Check if MLP activation was captured
        mlp_key = f'layer_{target_layer_before}_mlp'
        if mlp_key not in mlp_activations:
            print(f"\n{Fore.RED}ERROR: MLP activation not captured for {mlp_key}{Fore.RESET}")
            print(f"Available keys: {list(mlp_activations.keys())[:5]}...")
            raise KeyError(f"MLP activation not found: {mlp_key}")
        
        # Get MLP output
        hidden_states_mlp = mlp_activations[mlp_key]
        
        # Reshape to [batch*seq, hidden_size]
        batch_size_actual = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[target_layer_before].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[target_layer_after].view(-1, hidden_size).to(torch.float64)
        
        # Debug shapes
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] Activation shapes:{Fore.RESET}")
            print(f"  MLP output: {hidden_states_mlp.shape}")
            print(f"  Hidden state i: {hidden_states_i.shape}")
            print(f"  Hidden state n: {hidden_states_n.shape}")
        
        # Identify modality tokens
        input_ids_flat = input_ids.view(-1)  # [batch*seq]
        
        if batch_idx == 0:
            vision_mask, text_mask = identify_modality_tokens(input_ids_flat, image_token_index)
        else:
            vision_mask = (input_ids_flat == image_token_index)
            text_mask = ~vision_mask
        
        # Store batch
        a1_batch = hidden_states_mlp
        
        if accurate:
            a2_batch = hidden_states_n
            a3_batch = hidden_states_i - hidden_states_mlp
        else:
            a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        a1_list.append(a1_batch.cpu())
        a2_list.append(a2_batch.cpu())
        if accurate:
            a3_list.append(a3_batch.cpu())
        vision_mask_list.append(vision_mask.cpu())
        text_mask_list.append(text_mask.cpu())
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        torch.cuda.empty_cache()
    
    print(f"\n{Fore.GREEN}Processed {batch_count} batches{Fore.RESET}\n")
    
    # Concatenate all batches
    print(f"{Fore.GREEN}[Step 5/7] Concatenating activations...{Fore.RESET}")
    a1 = torch.cat(a1_list, dim=0)
    a2 = torch.cat(a2_list, dim=0)
    if accurate:
        a3 = torch.cat(a3_list, dim=0)
    vision_masks = torch.cat(vision_mask_list, dim=0)
    text_masks = torch.cat(text_mask_list, dim=0)
    
    total_tokens = a1.shape[0]
    num_vision = vision_masks.sum().item()
    num_text = text_masks.sum().item()
    
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Vision tokens: {num_vision:,} ({num_vision/total_tokens*100:.1f}%)")
    print(f"  Text tokens: {num_text:,} ({num_text/total_tokens*100:.1f}%)")
    
    if num_vision == 0:
        print(f"\n{Fore.RED}ERROR: No vision tokens detected!{Fore.RESET}")
        print(f"Check if images are being processed correctly.")
        raise ValueError("No vision tokens found")
    
    # Auto-balance lambda if requested
    if auto_lambda:
        print(f"\n{Fore.GREEN}[Step 6/7] Auto-balancing lambda values...{Fore.RESET}")
        
        # Sample for efficiency
        sample_size = min(10000, a1.shape[0])
        indices = torch.randperm(a1.shape[0])[:sample_size]
        
        with torch.no_grad():
            a1_sample = a1[indices].to(device)
            a2_sample = a2[indices].to(device)
            vision_mask_sample = vision_masks[indices].to(device)
            text_mask_sample = text_masks[indices].to(device)
            
            # Initial identity transform
            pred_sample = a1_sample
            
            # Compute per-modality losses
            pred_norm = F.normalize(pred_sample, p=2, dim=1)
            target_norm = F.normalize(a2_sample, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)
            cosine_dist = 1.0 - cosine_sim
            
            if vision_mask_sample.sum() > 0:
                loss_vision_init = cosine_dist[vision_mask_sample].mean().item()
            else:
                loss_vision_init = 1.0
            
            if text_mask_sample.sum() > 0:
                loss_text_init = cosine_dist[text_mask_sample].mean().item()
            else:
                loss_text_init = 1.0
            
            # Balance: make vision loss contribution equal to text
            if loss_vision_init > 0:
                lambda_vision = loss_text_init / loss_vision_init
                lambda_text = 1.0
            else:
                lambda_vision = 1.0
                lambda_text = 1.0
            
            print(f"\n{Fore.CYAN}Auto-lambda results:{Fore.RESET}")
            print(f"  Initial vision loss: {loss_vision_init:.6f}")
            print(f"  Initial text loss: {loss_text_init:.6f}")
            print(f"  Computed λ_vision: {lambda_vision:.4f}")
            print(f"  Computed λ_text: {lambda_text:.4f}")
            print(f"  Ratio: {lambda_vision/lambda_text:.2f}x more weight on vision\n")
            
            del a1_sample, a2_sample, vision_mask_sample, text_mask_sample, pred_sample
            torch.cuda.empty_cache()
    else:
        print(f"\n{Fore.YELLOW}Using manual lambda values:{Fore.RESET}")
        print(f"  λ_vision: {lambda_vision}")
        print(f"  λ_text: {lambda_text}\n")
    
    # Estimate transformation
    print(f"{Fore.GREEN}[Step 6/7] Estimating transformation matrix...{Fore.RESET}")
    
    if accurate:
        print(f"{Fore.YELLOW}Note: Accurate mode with a3 not fully optimized for modality-aware loss{Fore.RESET}")
        print(f"{Fore.YELLOW}Using standard mode (a2 = target){Fore.RESET}\n")
    
    transform = estimate_transform_modality_aware(
        a1, a2,
        vision_masks, text_masks,
        lambda_vision, lambda_text,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=opt_batch_size,
        device=device
    )
    
    print(f"{Fore.GREEN}Transform estimated: {transform.shape}{Fore.RESET}")
    print(f"  Transform dtype: {transform.dtype}")
    print(f"  Transform device: {transform.device}\n")
    
    # Remove hooks and clean memory
    print(f"{Fore.GREEN}[Step 7/7] Applying transform and saving model...{Fore.RESET}")
    
    for hook in hooks:
        hook.remove()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model and truncate
    print(f"  Reloading model for truncation...")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    print(f"  Truncating layers {start_id - num_layer} to {end_id - num_layer}...")
    model = truncate_vlm_model(model, start_id - num_layer, end_id - num_layer)
    
    # Setup save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/vlm_{os.path.basename(model_path)}_{layers_to_skip}layers"
    
    # Apply transform
    print(f"  Applying transform to layer {start_id - num_layer - 1}...")
    apply_vlm_transform(model, transform, start_id - num_layer - 1)
    
    # Save model
    final_path = f"{save_path}_ReplaceMe_modality_aware"
    print(f"  Saving model to: {final_path}")
    
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}✓ MODEL SAVED SUCCESSFULLY{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.CYAN}Output path: {final_path}{Fore.RESET}\n")
    
    if save_transform_only:
        transform_path = f"{final_path}_transform.pt"
        torch.save(transform, transform_path)
        print(f"{Fore.GREEN}Transform matrix saved to: {transform_path}{Fore.RESET}\n")
    
    # Final cleanup
    del model, a1, a2, vision_masks, text_masks
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}Memory cleaned up. Pipeline completed.{Fore.RESET}\n")
    
    return final_path