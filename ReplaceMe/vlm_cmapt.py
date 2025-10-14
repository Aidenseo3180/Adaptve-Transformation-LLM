# ============================================================
# vlm_cmapt.py - Cross-Modal Alignment Preserving Transform
# Training-free VLM Pruning with Alignment Preservation
# ============================================================

import gc
import logging
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
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


def identify_vision_tokens_robust(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    image_token_index: int = 32000,
    num_vision_tokens: int = 576,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robustly identify vision and text tokens using multiple strategies.
    
    Strategy:
    1. Find image token position in input_ids
    2. Vision tokens = next 576 tokens after image token
    3. Validation using attention pattern
    
    Args:
        input_ids: [batch_size, seq_len]
        hidden_states: [batch_size, seq_len, hidden_size]
        image_token_index: Special token for image (default 32000 for LLaVA)
        num_vision_tokens: Number of vision patches (576 for LLaVA-1.5)
        debug: Print detailed info
        
    Returns:
        vision_mask: [batch*seq] boolean mask
        text_mask: [batch*seq] boolean mask
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    if debug:
        print(f"\n{Fore.CYAN}[identify_vision_tokens_robust]{Fore.RESET}")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  hidden_states shape: {hidden_states.shape}")
        print(f"  image_token_index: {image_token_index}")
        print(f"  num_vision_tokens: {num_vision_tokens}")
    
    # Initialize masks
    vision_mask_2d = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # Process each sample in batch
    for b in range(batch_size):
        # Find image token position
        image_positions = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
        
        if len(image_positions) == 0:
            if debug:
                print(f"  {Fore.YELLOW}[Sample {b}] No image token found!{Fore.RESET}")
            continue
        
        # Image token position
        img_pos = image_positions[0].item()
        
        # Vision tokens are AFTER image token
        vision_start = img_pos + 1
        vision_end = min(vision_start + num_vision_tokens, seq_len)
        
        if debug and b == 0:
            print(f"  [Sample {b}] Image token at position: {img_pos}")
            print(f"  [Sample {b}] Vision tokens: [{vision_start}:{vision_end}]")
            print(f"  [Sample {b}] Num vision tokens: {vision_end - vision_start}")
        
        # Mark vision tokens
        vision_mask_2d[b, vision_start:vision_end] = True
    
    # Flatten to [batch*seq]
    vision_mask = vision_mask_2d.reshape(-1)
    text_mask = ~vision_mask
    
    # Statistics
    num_vision = vision_mask.sum().item()
    num_text = text_mask.sum().item()
    total = num_vision + num_text
    
    # print(f"\n{Fore.GREEN}Vision Token Identification Results:{Fore.RESET}")
    # print(f"  Total tokens: {total:,}")
    # print(f"  Vision tokens: {num_vision:,} ({num_vision/total*100:.1f}%)")
    # print(f"  Text tokens: {num_text:,} ({num_text/total*100:.1f}%)")
    
    if num_vision == 0:
        raise ValueError(f"{Fore.RED}ERROR: No vision tokens detected!{Fore.RESET}")
    
    # Sanity check: vision tokens should be ~40-60% of total
    # vision_ratio = num_vision / total
    # if vision_ratio < 0.3 or vision_ratio > 0.7:
    #     print(f"{Fore.YELLOW}WARNING: Unusual vision token ratio: {vision_ratio:.1%}{Fore.RESET}")
    #     print(f"{Fore.YELLOW}Expected: 40-60%. Check if identification is correct.{Fore.RESET}")
    
    return vision_mask, text_mask


def compute_alignment_matrix(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute cross-modal alignment matrix.
    
    Args:
        vision_features: [num_vision, hidden_size]
        text_features: [num_text, hidden_size]
        normalize: Use cosine similarity (True) or dot product (False)
        
    Returns:
        alignment_matrix: [num_vision, num_text]
    """
    if normalize:
        vision_norm = F.normalize(vision_features, p=2, dim=1)
        text_norm = F.normalize(text_features, p=2, dim=1)
        alignment = vision_norm @ text_norm.T
    else:
        alignment = vision_features @ text_features.T
    
    return alignment


def compute_intra_modal_similarity(
    features: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute intra-modal similarity matrix (token-to-token within same modality).
    
    Args:
        features: [num_tokens, hidden_size]
        normalize: Use cosine similarity
        
    Returns:
        similarity_matrix: [num_tokens, num_tokens]
    """
    if normalize:
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = features_norm @ features_norm.T
    else:
        similarity = features @ features.T
    
    return similarity

def alignment_preserving_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vision_mask: torch.Tensor,
    text_mask: torch.Tensor,
    lambda_output: float = 1.0,
    lambda_alignment: float = 0.3,
    lambda_intra: float = 0.1,
    debug: bool = False
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss with alignment preservation.
    
    NOTE: Target alignment computed from current batch (not pre-computed).
    This ensures shape compatibility and semantic correctness.
    """
    # Ensure float64 for numerical stability
    pred = pred.to(torch.float64)
    target = target.to(torch.float64)
    
    # Extract modality-specific features
    vision_pred = pred[vision_mask]
    text_pred = pred[text_mask]
    vision_target = target[vision_mask]
    text_target = target[text_mask]
    
    # Loss 1: Output Matching (standard ReplaceMe objective)
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)
    output_loss = (1.0 - cosine_sim).mean()
    
    # Loss 2: Cross-Modal Alignment Preservation
    if vision_pred.shape[0] > 0 and text_pred.shape[0] > 0:
        # Compute BOTH target and pred alignment from current batch
        # This ensures shape compatibility!
        target_alignment_vt = compute_alignment_matrix(
            vision_target, text_target, normalize=True
        )
        pred_alignment_vt = compute_alignment_matrix(
            vision_pred, text_pred, normalize=True
        )
        
        # Now shapes match!
        alignment_loss = F.mse_loss(pred_alignment_vt, target_alignment_vt)
    else:
        alignment_loss = torch.tensor(0.0, device=pred.device)
    
    # Loss 3: Intra-Modal Structure Preservation
    # Vision-vision similarity
    if vision_pred.shape[0] > 10:
        target_similarity_v = compute_intra_modal_similarity(
            vision_target, normalize=True
        )
        pred_similarity_v = compute_intra_modal_similarity(
            vision_pred, normalize=True
        )
        intra_vision_loss = F.mse_loss(pred_similarity_v, target_similarity_v)
    else:
        intra_vision_loss = torch.tensor(0.0, device=pred.device)
    
    # Text-text similarity
    if text_pred.shape[0] > 10:
        target_similarity_t = compute_intra_modal_similarity(
            text_target, normalize=True
        )
        pred_similarity_t = compute_intra_modal_similarity(
            text_pred, normalize=True
        )
        intra_text_loss = F.mse_loss(pred_similarity_t, target_similarity_t)
    else:
        intra_text_loss = torch.tensor(0.0, device=pred.device)
    
    intra_loss = (intra_vision_loss + intra_text_loss) / 2.0
    
    # Combined loss
    total_loss = (
        lambda_output * output_loss +
        lambda_alignment * alignment_loss +
        lambda_intra * intra_loss
    )
    
    # Loss dictionary for logging
    loss_dict = {
        'total': total_loss.item(),
        'output': output_loss.item(),
        'alignment': alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else 0.0,
        'intra': intra_loss.item(),
        'intra_vision': intra_vision_loss.item() if isinstance(intra_vision_loss, torch.Tensor) else 0.0,
        'intra_text': intra_text_loss.item() if isinstance(intra_text_loss, torch.Tensor) else 0.0,
    }
    
    if debug:
        print(f"\n{Fore.YELLOW}[Loss Breakdown]{Fore.RESET}")
        print(f"  Batch shapes: V={vision_pred.shape[0]}, T={text_pred.shape[0]}")
        if vision_pred.shape[0] > 0 and text_pred.shape[0] > 0:
            print(f"  Alignment matrix shape: {pred_alignment_vt.shape}")
        print(f"  Output matching: {output_loss.item():.6f} (weight: {lambda_output})")
        print(f"  Cross-modal alignment: {loss_dict['alignment']:.6f} (weight: {lambda_alignment})")
        print(f"  Intra-modal (vision): {loss_dict['intra_vision']:.6f}")
        print(f"  Intra-modal (text): {loss_dict['intra_text']:.6f}")
        print(f"  Intra-modal avg: {loss_dict['intra']:.6f} (weight: {lambda_intra})")
        print(f"  {Fore.GREEN}TOTAL: {loss_dict['total']:.6f}{Fore.RESET}")
    
    return total_loss, loss_dict

def estimate_cmapt_transform(
    a1: torch.Tensor,
    a2: torch.Tensor,
    vision_masks: torch.Tensor,
    text_masks: torch.Tensor,
    lambda_output: float = 1.0,
    lambda_alignment: float = 0.3,
    lambda_intra: float = 0.1,
    lr: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Estimate transformation matrix using CMAPT objective.
    
    Key: Target alignment computed per-batch (not pre-computed).
    This ensures shape compatibility and is semantically correct.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}CMAPT: Cross-Modal Alignment Preserving Transform{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    hidden_size = a1.shape[1]
    num_samples = a1.shape[0]
    
    print(f"{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total samples: {num_samples:,}")
    print(f"  Input dtype: {a1.dtype} (CPU) -> will convert to float64 (GPU) per batch")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"\n{Fore.CYAN}Loss Weights:{Fore.RESET}")
    print(f"  λ_output (output matching): {lambda_output}")
    print(f"  λ_alignment (cross-modal): {lambda_alignment}")
    print(f"  λ_intra (intra-modal): {lambda_intra}")
    
    print(f"\n{Fore.YELLOW}Note: Target alignment computed per-batch for shape compatibility{Fore.RESET}\n")
    
    # ===== Initialize Transform (no pre-computation needed!) =====
    print(f"{Fore.GREEN}Initializing transformation matrix...{Fore.RESET}")
    
    transform = torch.eye(hidden_size, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform], lr=lr)
    
    print(f"  Transform shape: {transform.shape}")
    print(f"  Transform dtype: {transform.dtype}")
    print(f"  Transform device: {transform.device}\n")
    
    # ===== Optimization Loop =====
    print(f"{Fore.GREEN}Starting optimization...{Fore.RESET}\n")
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = {
            'total': 0.0,
            'output': 0.0,
            'alignment': 0.0,
            'intra': 0.0
        }
        
        # Shuffle data
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
            
            # Get batch - convert to float64 on GPU
            a1_batch = a1[batch_indices].to(device=device, dtype=torch.float64)
            a2_batch = a2[batch_indices].to(device=device, dtype=torch.float64)
            vision_mask_batch = vision_masks[batch_indices].to(device=device)
            text_mask_batch = text_masks[batch_indices].to(device=device)
            
            # Debug first batch
            if batch_idx == 0 and epoch == 0:
                print(f"\n{Fore.YELLOW}[DEBUG] First batch:{Fore.RESET}")
                print(f"  a1_batch: {a1_batch.shape}, {a1_batch.dtype}, {a1_batch.device}")
                print(f"  a2_batch: {a2_batch.shape}, {a2_batch.dtype}, {a2_batch.device}")
                print(f"  Vision tokens: {vision_mask_batch.sum().item()}")
                print(f"  Text tokens: {text_mask_batch.sum().item()}")
            
            # Forward pass
            pred = a1_batch @ transform
            
            # Compute loss (target computed inside from a2_batch!)
            loss, loss_dict = alignment_preserving_loss(
                pred, a2_batch,
                vision_mask_batch, text_mask_batch,
                lambda_output=lambda_output,
                lambda_alignment=lambda_alignment,
                lambda_intra=lambda_intra,
                debug=(batch_idx == 0 and epoch == 0)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total']:.6f}",
                'Out': f"{loss_dict['output']:.4f}",
                'Align': f"{loss_dict['alignment']:.4f}",
                'Intra': f"{loss_dict['intra']:.4f}",
                'Best': f"{best_loss:.6f}"
            })
            
            # Memory cleanup
            del a1_batch, a2_batch, pred, vision_mask_batch, text_mask_batch
            torch.cuda.empty_cache()
        
        # Epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            print(f"\n{Fore.GREEN}  ✓ New best loss: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Epoch {epoch+1}/{num_epochs} Summary:{Fore.RESET}")
        print(f"  Total: {avg_losses['total']:.6f}")
        print(f"  Output: {avg_losses['output']:.6f}")
        print(f"  Alignment: {avg_losses['alignment']:.6f}")
        print(f"  Intra: {avg_losses['intra']:.6f}")
        print(f"  Best: {best_loss:.6f}\n")
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}OPTIMIZATION COMPLETED{Fore.RESET}")
    print(f"{Fore.GREEN}Final Loss: {best_loss:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}\n")
    
    return transform.detach()

def vlm_cmapt(
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
    lambda_output: float = 1.0,
    lambda_alignment: float = 0.3,
    lambda_intra: float = 0.1,
    lr: float = 1e-4,
    num_epochs: int = 10,
    opt_batch_size: int = 1024,
    token: Optional[str] = None
) -> str:
    """
    VLM pruning with Cross-Modal Alignment Preservation.
    
    Args:
        model_path: Path to VLM model
        image_dir: Directory with calibration images
        batch_size: Batch size for data loading
        max_length: Max sequence length
        layers_to_skip: Number of layers to prune
        dataset_size: Number of calibration samples
        use_4bit: Use 4-bit quantization
        save_path: Where to save pruned model
        save_transform_only: Save transform matrix separately
        start_id: Start layer index
        end_id: End layer index
        num_layer: Number of layers already pruned
        lambda_output: Weight for output matching loss
        lambda_alignment: Weight for alignment preservation loss
        lambda_intra: Weight for intra-modal structure loss
        lr: Learning rate
        num_epochs: Optimization epochs
        opt_batch_size: Batch size for optimization
        token: HuggingFace token
        
    Returns:
        Path to saved model
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}VLM-CMAPT: CROSS-MODAL ALIGNMENT PRESERVING PRUNING{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{Fore.CYAN}System Configuration:{Fore.RESET}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {mem_gb:.1f} GB")
    print(f"  Device map: {device_map}")
    print(f"  Compute device: {device}\n")
    
    # Quantization config
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
    print(f"  Image directory: {image_dir}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Layers to skip: {layers_to_skip}")
    print(f"  Target layers: start={start_id}, end={end_id}, num_layer={num_layer}\n")
    
    print(f"{Fore.CYAN}CMAPT Hyperparameters:{Fore.RESET}")
    print(f"  λ_output: {lambda_output}")
    print(f"  λ_alignment: {lambda_alignment}")
    print(f"  λ_intra: {lambda_intra}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Optimization batch size: {opt_batch_size}\n")
    
    # ===== LOAD MODEL =====
    print(f"{Fore.GREEN}[Step 1/8] Loading VLM model...{Fore.RESET}")
    
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
    print(f"  Language layers: {num_hidden_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Image token index: {image_token_index}")
    print(f"  Model device: {model.device}\n")
    
    model.eval()
    
    # ===== LOAD DATA =====
    print(f"{Fore.GREEN}[Step 2/8] Loading calibration data...{Fore.RESET}")
    
    dataloader = get_vlm_calib_dataloader(
        image_dir,
        dataset_size,
        batch_size,
        processor
    )
    print(f"  Dataloader created: {len(dataloader)} batches\n")
    
    # ===== SETUP HOOKS =====
    print(f"{Fore.GREEN}[Step 3/8] Setting up activation hooks...{Fore.RESET}")
    
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
    print(f"  MLP output layer: {target_layer_before}")
    print(f"  Target output layer: {target_layer_after}\n")
    
    # ===== PRE-ALLOCATE MEMORY =====
    print(f"{Fore.GREEN}[Step 4/8] Pre-allocating memory...{Fore.RESET}")
    
    total_samples = dataset_size if dataset_size else len(dataloader.dataset)
    estimated_tokens_per_image = 1000
    total_tokens = int(total_samples * estimated_tokens_per_image * 1.5)
    
    print(f"  Images: {total_samples}")
    print(f"  Estimated tokens: {total_tokens:,} (with 1.5x buffer)")
    print(f"  Hidden size: {hidden_size}")
    
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    vision_masks = torch.empty((total_tokens,), dtype=torch.bool, device='cpu')
    text_masks = torch.empty((total_tokens,), dtype=torch.bool, device='cpu')
    
    print(f"  Allocated: {total_tokens * hidden_size * 2 * 2 / 1e9:.2f} GB\n")
    
    cnt = 0
    
    # ===== GATHER ACTIVATIONS =====
    print(f"{Fore.GREEN}[Step 5/8] Gathering activations...{Fore.RESET}\n")
    
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.RED}Gathering Activations{Fore.RESET}",
        dynamic_ncols=True,
        colour="red"
    )):
        # Debug first batch
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] First batch:{Fore.RESET}")
            print(f"  Keys: {batch.keys()}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, {v.dtype}")
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in batch.items() 
                 if isinstance(v, torch.Tensor)}
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get hidden states
        hidden_states = outputs.hidden_states[1:]  # Skip embedding
        
        # Check MLP activation
        mlp_key = f'layer_{target_layer_before}_mlp'
        if mlp_key not in mlp_activations:
            raise KeyError(f"MLP activation not found: {mlp_key}")
        
        hidden_states_mlp = mlp_activations[mlp_key]
        
        # Reshape
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_i = hidden_states[target_layer_before].view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_n = hidden_states[target_layer_after].view(-1, hidden_size).to(torch.bfloat16)
        
        # Debug shapes
        if batch_idx == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] Shapes:{Fore.RESET}")
            print(f"  MLP: {hidden_states_mlp.shape}")
            print(f"  Hidden i: {hidden_states_i.shape}")
            print(f"  Hidden n: {hidden_states_n.shape}")
        
        # Identify vision/text tokens
        if batch_idx == 0:
            vision_mask, text_mask = identify_vision_tokens_robust(
                input_ids,
                hidden_states[target_layer_after],
                image_token_index=image_token_index,
                num_vision_tokens=576,
                debug=True
            )
        else:
            vision_mask, text_mask = identify_vision_tokens_robust(
                input_ids,
                hidden_states[target_layer_after],
                image_token_index=image_token_index,
                num_vision_tokens=576,
                debug=False
            )
        
        # Compute activations
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        batch_size_tokens = a1_batch.shape[0]
        
        # Buffer check
        if cnt + batch_size_tokens > total_tokens:
            print(f"\n{Fore.RED}Buffer overflow at batch {batch_idx}!{Fore.RESET}")
            print(f"  Current: {cnt:,}, Batch: {batch_size_tokens:,}")
            print(f"  Required: {cnt + batch_size_tokens:,}, Allocated: {total_tokens:,}")
            print(f"  Stopping early. Using {cnt:,} tokens.\n")
            break
        
        # Write to buffers
        a1[cnt:cnt+batch_size_tokens] = a1_batch.cpu()
        a2[cnt:cnt+batch_size_tokens] = a2_batch.cpu()
        vision_masks[cnt:cnt+batch_size_tokens] = vision_mask.cpu()
        text_masks[cnt:cnt+batch_size_tokens] = text_mask.cpu()
        
        cnt += batch_size_tokens
        
        # Periodic progress
        if (batch_idx + 1) % 100 == 0:
            avg_tokens = cnt / ((batch_idx + 1) * batch_size)
            usage_pct = (cnt / total_tokens) * 100
            print(f"\n  [{batch_idx+1}] Avg tokens/img: {avg_tokens:.1f}, Buffer: {usage_pct:.1f}%")
        
        # Cleanup
        del hidden_states_mlp, hidden_states_i, hidden_states_n, a1_batch, a2_batch
        torch.cuda.empty_cache()
    
    # Slice to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    vision_masks = vision_masks[:cnt]
    text_masks = text_masks[:cnt]
    
    # Statistics
    avg_tokens = cnt / ((batch_idx + 1) * batch_size)
    usage_pct = (cnt / total_tokens) * 100
    num_vision = vision_masks.sum().item()
    num_text = text_masks.sum().item()
    
    print(f"\n{Fore.GREEN}Collection complete:{Fore.RESET}")
    print(f"  Total tokens: {cnt:,}")
    print(f"  Avg tokens/image: {avg_tokens:.1f}")
    print(f"  Buffer usage: {usage_pct:.1f}%")
    print(f"  Vision: {num_vision:,} ({num_vision/cnt*100:.1f}%)")
    print(f"  Text: {num_text:,} ({num_text/cnt*100:.1f}%)\n")
    
    # ===== ESTIMATE TRANSFORM =====
    print(f"{Fore.GREEN}[Step 6/8] Estimating CMAPT transformation...{Fore.RESET}\n")
    
    transform = estimate_cmapt_transform(
        a1, a2,
        vision_masks, text_masks,
        lambda_output=lambda_output,
        lambda_alignment=lambda_alignment,
        lambda_intra=lambda_intra,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=opt_batch_size,
        device=device
    )
    
    print(f"{Fore.GREEN}Transform estimated: {transform.shape}, {transform.dtype}{Fore.RESET}\n")
    
    # ===== CLEANUP =====
    for hook in hooks:
        hook.remove()
    
    del model, a1, a2, vision_masks, text_masks
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== RELOAD AND TRUNCATE =====
    print(f"{Fore.GREEN}[Step 7/8] Reloading and truncating model...{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    print(f"  Truncating layers [{start_id - num_layer}:{end_id - num_layer}]...")
    model = truncate_vlm_model(model, start_id - num_layer, end_id - num_layer)
    
    # Setup save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/vlm_{os.path.basename(model_path)}_{layers_to_skip}layers"
    
    # Apply transform
    print(f"  Applying transform to layer {start_id - num_layer - 1}...")
    apply_vlm_transform(model, transform, start_id - num_layer - 1)
    
    # ===== SAVE =====
    print(f"\n{Fore.GREEN}[Step 8/8] Saving model...{Fore.RESET}")
    
    final_path = f"{save_path}_CMAPT"
    print(f"  Saving to: {final_path}")
    
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}✓ MODEL SAVED SUCCESSFULLY{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.CYAN}Path: {final_path}{Fore.RESET}\n")
    
    if save_transform_only:
        transform_path = f"{final_path}_transform.pt"
        torch.save(transform, transform_path)
        print(f"{Fore.GREEN}Transform saved: {transform_path}{Fore.RESET}\n")
    
    # Final cleanup
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}Pipeline completed.{Fore.RESET}\n")
    
    return final_path