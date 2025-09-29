import gc
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def collect_activations_minimal(
    model,
    tokenizer,
    dataloader,
    start_id: int,
    end_id: int,
    num_layer: int,
    max_length: int,
    dataset_size: int,
    is_teacher: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect minimal activations following ReplaceMe pattern."""
    
    model_type = "Teacher" if is_teacher else "Pruned"
    print(f"{Fore.GREEN}[DEBUG] Collecting {model_type} model activations{Fore.RESET}")
    print(f"{Fore.YELLOW}[DEBUG] Layers: {start_id} to {end_id}, num_layer offset: {num_layer}{Fore.RESET}")
    
    hidden_size = model.config.hidden_size
    device = next(model.parameters()).device
    
    # Pre-allocate CPU tensors like ReplaceMe
    total_samples = dataset_size * max_length
    a1 = torch.empty((total_samples, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_samples, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    # Setup MLP hooks
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    if hasattr(model, 'transformer'):  # Falcon
        layers = model.transformer.h
    else:  # LLaMA
        layers = model.model.layers
    
    for i, layer in enumerate(layers):
        hook = layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        hooks.append(hook)
    
    cnt = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc=f"{Fore.BLUE}{model_type} Forward Pass{Fore.RESET}",
            dynamic_ncols=True
        )):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding
            
            # Get the right layer indices
            if is_teacher:
                # For teacher: use original indices
                mlp_idx = start_id - 1
                hidden_i_idx = start_id - 1
                hidden_n_idx = end_id - 1
            else:
                # For pruned: adjust for removed layers
                mlp_idx = start_id - num_layer - 1
                hidden_i_idx = start_id - num_layer - 1
                hidden_n_idx = end_id - num_layer - 1
            
            # Extract activations
            hidden_states_mlp = mlp_activations[f'layer_{mlp_idx}_mlp']
            hidden_states_i = hidden_states[hidden_i_idx]
            hidden_states_n = hidden_states[hidden_n_idx]
            
            # Reshape to (batch*seq_len, hidden_size)
            batch_size, seq_len = inputs['input_ids'].shape
            actual_samples = batch_size * seq_len
            
            hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)
            hidden_states_i = hidden_states_i.view(-1, hidden_size)
            hidden_states_n = hidden_states_n.view(-1, hidden_size)
            
            # Store like ReplaceMe does
            a1_batch = hidden_states_mlp
            
            if is_teacher:
                # For teacher: store the actual transformation
                a2_batch = hidden_states_n
            else:
                # For pruned: store the target (like ReplaceMe)
                a2_batch = hidden_states_n - hidden_states_i + hidden_states_mlp
            
            # Move to CPU and store
            a1[cnt:cnt+actual_samples] = a1_batch.cpu().to(torch.bfloat16)
            a2[cnt:cnt+actual_samples] = a2_batch.cpu().to(torch.bfloat16)
            
            cnt += actual_samples
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                print(f"{Fore.YELLOW}[DEBUG] Processed {cnt}/{total_samples} samples{Fore.RESET}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    
    print(f"{Fore.GREEN}[DEBUG] {model_type} collection complete: {cnt} samples{Fore.RESET}")
    print(f"{Fore.YELLOW}[DEBUG] Shapes: a1={a1.shape}, a2={a2.shape}{Fore.RESET}")
    
    return a1, a2


def teacher_guided_adam_optimization(
    pruned_a1: torch.Tensor,
    pruned_a2: torch.Tensor,
    teacher_a1: torch.Tensor,
    teacher_a2: torch.Tensor,
    hidden_size: int,
    teacher_weight: float = 0.3,
    lr: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 1024
) -> torch.Tensor:
    """Optimize T using both teacher and pruned activations."""
    
    print(f"{Fore.CYAN}[DEBUG] Starting optimization with teacher weight={teacher_weight}{Fore.RESET}")
    
    # Initialize T as identity
    T = torch.eye(hidden_size, dtype=torch.float32, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([T], lr=lr)
    
    # Use minimum size to handle mismatch
    num_samples = min(pruned_a1.shape[0], teacher_a1.shape[0])
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"{Fore.YELLOW}[DEBUG] Training samples: {num_samples}, Batches: {num_batches}{Fore.RESET}")
    
    best_loss = float('inf')
    best_T = T.clone()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_teacher_loss = 0
        epoch_pruned_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(num_samples)
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data and move to GPU
            p_a1 = pruned_a1[batch_indices].to('cuda', dtype=torch.float32)
            p_a2 = pruned_a2[batch_indices].to('cuda', dtype=torch.float32)
            t_a1 = teacher_a1[batch_indices].to('cuda', dtype=torch.float32)
            t_a2 = teacher_a2[batch_indices].to('cuda', dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Compute predictions
            pruned_pred = p_a1 @ T
            
            # For teacher, we need to compute the difference
            # Teacher a2 contains the full hidden state at layer n
            # We need to extract what the transformation should be
            teacher_pred = t_a1 @ T
            
            # Cosine similarity loss for pruned model
            pruned_pred_norm = F.normalize(pruned_pred, p=2, dim=1)
            p_a2_norm = F.normalize(p_a2, p=2, dim=1)
            pruned_loss = 1 - (pruned_pred_norm * p_a2_norm).sum(dim=1).mean()
            
            # Cosine similarity loss for teacher model
            teacher_pred_norm = F.normalize(teacher_pred, p=2, dim=1)
            t_a2_norm = F.normalize(t_a2, p=2, dim=1)
            teacher_loss = 1 - (teacher_pred_norm * t_a2_norm).sum(dim=1).mean()
            
            # Combined loss
            total_loss = (1 - teacher_weight) * pruned_loss + teacher_weight * teacher_loss
            
            # Add small regularization to keep T close to identity
            reg_loss = 0.001 * torch.norm(T - torch.eye(hidden_size, device='cuda'), p='fro')
            total_loss = total_loss + reg_loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_teacher_loss += teacher_loss.item()
            epoch_pruned_loss += pruned_loss.item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Teacher': f'{teacher_loss.item():.4f}',
                'Pruned': f'{pruned_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / num_batches
        print(f"{Fore.GREEN}[DEBUG] Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}{Fore.RESET}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T.clone()
            print(f"{Fore.YELLOW}[DEBUG] New best loss: {best_loss:.4f}{Fore.RESET}")
    
    return best_T.detach().cpu().to(torch.float64)


def teacher_guided_replaceme(
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
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    teacher_weight: float = 0.3,
    lr: float = 1e-4,
    epochs: int = 5,
    token: Optional[str] = None,
    **kwargs
) -> str:
    """Teacher-guided ReplaceMe with sequential processing."""
    
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Teacher-Guided ReplaceMe (Sequential Mode){Fore.RESET}")
    print(f"{Fore.MAGENTA}Model: {model_path}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Skip layers: {start_id} to {end_id}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Teacher weight: {teacher_weight}{Fore.RESET}")
    print(f"{Fore.MAGENTA}4-bit mode: {use_4bit}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    device_map = 'auto' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get dataloader once
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # ========== PHASE 1: Teacher Model ==========
    print(f"{Fore.CYAN}[PHASE 1] Processing Teacher Model{Fore.RESET}")
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        token=token
    )
    
    hidden_size = teacher_model.config.hidden_size
    print(f"{Fore.YELLOW}[DEBUG] Model loaded, hidden_size={hidden_size}{Fore.RESET}")
    
    # Collect teacher activations
    teacher_a1, teacher_a2 = collect_activations_minimal(
        teacher_model, tokenizer, dataloader,
        start_id, end_id, num_layer, max_length, dataset_size,
        is_teacher=True
    )
    
    # Clear teacher model
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{Fore.GREEN}[PHASE 1] Complete - Teacher model cleared{Fore.RESET}")
    
    # ========== PHASE 2: Pruned Model ==========
    print(f"{Fore.CYAN}[PHASE 2] Processing Pruned Model{Fore.RESET}")
    
    pruned_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        token=token
    )
    
    # Truncate model
    pruned_model = truncate_model(pruned_model, start_id, end_id)
    print(f"{Fore.YELLOW}[DEBUG] Model truncated: removed layers {start_id} to {end_id-1}{Fore.RESET}")
    
    # Collect pruned activations
    pruned_a1, pruned_a2 = collect_activations_minimal(
        pruned_model, tokenizer, dataloader,
        start_id, end_id, num_layer, max_length, dataset_size,
        is_teacher=False
    )
    
    # Clear pruned model
    del pruned_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{Fore.GREEN}[PHASE 2] Complete - Pruned model cleared{Fore.RESET}")
    
    # ========== PHASE 3: Optimization ==========
    print(f"{Fore.CYAN}[PHASE 3] Optimizing Transformation Matrix{Fore.RESET}")
    
    T_optimal = teacher_guided_adam_optimization(
        pruned_a1, pruned_a2,
        teacher_a1, teacher_a2,
        hidden_size, teacher_weight, lr, epochs, batch_size
    )
    
    print(f"{Fore.GREEN}[PHASE 3] Complete - T matrix optimized{Fore.RESET}")
    
    # Clear activation tensors
    del teacher_a1, teacher_a2, pruned_a1, pruned_a2
    gc.collect()
    
    # ========== PHASE 4: Apply Transformation ==========
    print(f"{Fore.CYAN}[PHASE 4] Applying Transformation to Final Model{Fore.RESET}")
    
    final_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    final_model = truncate_model(final_model, start_id, end_id)
    
    # Apply transformation
    if hasattr(final_model, 'model'):  # LLaMA
        layer = final_model.model.layers[start_id - num_layer - 1]
    else:  # Falcon
        layer = final_model.transformer.h[start_id - num_layer - 1]
    
    old_weight = layer.mlp.down_proj.weight.data.to(torch.float64)
    new_weight = (T_optimal.T @ old_weight).to(torch.bfloat16)
    layer.mlp.down_proj.weight.data = new_weight
    
    print(f"{Fore.YELLOW}[DEBUG] Transformation applied to layer {start_id - num_layer - 1}{Fore.RESET}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_teacher_guided_{layers_to_skip}layers_{start_id}_{end_id}"
    
    save_path = f"{save_path}_tw{teacher_weight}"
    final_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save transformation matrix
    torch.save({
        'transform': T_optimal,
        'teacher_weight': teacher_weight,
        'start_id': start_id,
        'end_id': end_id
    }, f"{save_path}/transform_info.pt")
    
    print(f"{Fore.GREEN}[SUCCESS] Model saved to {save_path}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    # Final cleanup
    del final_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path