import gc
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def collect_teacher_activations(
    model,
    tokenizer,
    dataloader,
    start_id: int,
    end_id: int,
    max_length: int,
    dataset_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect activations from teacher model."""
    
    print(f"{Fore.GREEN}[DEBUG] Collecting teacher activations from layers {start_id} to {end_id}{Fore.RESET}")
    
    hidden_size = model.config.hidden_size
    num_samples = dataset_size * max_length
    
    # Initialize storage tensors on CPU to save GPU memory
    teacher_mlp = torch.zeros((num_samples, hidden_size), dtype=torch.bfloat16, device='cpu')
    teacher_attn = torch.zeros((num_samples, hidden_size), dtype=torch.bfloat16, device='cpu')
    teacher_target = torch.zeros((num_samples, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    # Hook to capture MLP outputs
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    if hasattr(model, 'transformer'):  # Falcon-style
        layers = model.transformer.h
    else:  # LLaMA-style
        layers = model.model.layers
    
    for i in range(len(layers)):
        hook = layers[i].mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        hooks.append(hook)
    
    cnt = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc=f"{Fore.BLUE}Teacher Forward Pass{Fore.RESET}",
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
            
            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            
            # Get activations for the specific layers
            batch_size = inputs['input_ids'].shape[0]
            seq_len = inputs['input_ids'].shape[1]
            actual_samples = batch_size * seq_len
            
            # Extract MLP output at layer (start_id - 1)
            mlp_out = mlp_activations[f'layer_{start_id-1}_mlp'].view(-1, hidden_size)
            
            # Extract attention output (hidden_state - mlp_output) at layer (start_id - 1)
            hidden_i = hidden_states[start_id-1].view(-1, hidden_size)
            attn_out = hidden_i - mlp_out
            
            # Extract target: hidden state at layer (end_id - 1)
            hidden_n = hidden_states[end_id-1].view(-1, hidden_size)
            
            # Store in CPU tensors
            teacher_mlp[cnt:cnt+actual_samples] = mlp_out.cpu().to(torch.bfloat16)
            teacher_attn[cnt:cnt+actual_samples] = attn_out.cpu().to(torch.bfloat16)
            teacher_target[cnt:cnt+actual_samples] = hidden_n.cpu().to(torch.bfloat16)
            
            cnt += actual_samples
            
            if batch_idx % 10 == 0:
                print(f"{Fore.YELLOW}[DEBUG] Processed {cnt}/{num_samples} samples{Fore.RESET}")
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Trim to actual size
    teacher_mlp = teacher_mlp[:cnt]
    teacher_attn = teacher_attn[:cnt]
    teacher_target = teacher_target[:cnt]
    
    print(f"{Fore.GREEN}[DEBUG] Collected {cnt} teacher samples{Fore.RESET}")
    print(f"{Fore.GREEN}[DEBUG] Teacher tensors shapes: mlp={teacher_mlp.shape}, attn={teacher_attn.shape}, target={teacher_target.shape}{Fore.RESET}")
    
    return teacher_mlp, teacher_attn, teacher_target


def teacher_guided_adam_optimization(
    pruned_mlp: torch.Tensor,
    pruned_target: torch.Tensor,
    teacher_mlp: torch.Tensor,
    teacher_attn: torch.Tensor,
    teacher_target: torch.Tensor,
    hidden_size: int,
    teacher_weight: float = 0.3,
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 1024
) -> torch.Tensor:
    """Optimize T matrix using teacher guidance."""
    
    print(f"{Fore.CYAN}[DEBUG] Starting teacher-guided optimization{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Teacher weight: {teacher_weight}, LR: {lr}, Epochs: {epochs}{Fore.RESET}")
    
    # Initialize T as identity matrix
    T = torch.eye(hidden_size, dtype=torch.float32, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([T], lr=lr)
    
    # Create batches
    num_samples = min(pruned_mlp.shape[0], teacher_mlp.shape[0])
    num_batches = (num_samples + batch_size - 1) // batch_size
    
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
            
            # Get batch data - ensure all on same device
            p_mlp = pruned_mlp[batch_indices].to('cuda', dtype=torch.float32)
            p_target = pruned_target[batch_indices].to('cuda', dtype=torch.float32)
            
            t_mlp = teacher_mlp[batch_indices].to('cuda', dtype=torch.float32)
            t_attn = teacher_attn[batch_indices].to('cuda', dtype=torch.float32)
            t_target = teacher_target[batch_indices].to('cuda', dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Compute predictions
            pruned_pred = p_mlp @ T
            teacher_pred = t_mlp @ T + t_attn  # Add attention for teacher
            
            # Compute cosine similarity losses
            # Pruned model loss
            pruned_pred_norm = F.normalize(pruned_pred, p=2, dim=1)
            p_target_norm = F.normalize(p_target, p=2, dim=1)
            pruned_loss = 1 - (pruned_pred_norm * p_target_norm).sum(dim=1).mean()
            
            # Teacher model loss
            teacher_pred_norm = F.normalize(teacher_pred, p=2, dim=1)
            t_target_norm = F.normalize(t_target, p=2, dim=1)
            teacher_loss = 1 - (teacher_pred_norm * t_target_norm).sum(dim=1).mean()
            
            # Combined loss
            total_loss = (1 - teacher_weight) * pruned_loss + teacher_weight * teacher_loss
            
            # Add L2 regularization
            reg_loss = 0.001 * torch.norm(T - torch.eye(hidden_size, device='cuda'), p='fro')
            total_loss = total_loss + reg_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_teacher_loss += teacher_loss.item()
            epoch_pruned_loss += pruned_loss.item()
            
            pbar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Teacher': f'{teacher_loss.item():.4f}',
                'Pruned': f'{pruned_loss.item():.4f}'
            })
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"{Fore.GREEN}[DEBUG] Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, "
              f"Teacher: {epoch_teacher_loss/num_batches:.4f}, "
              f"Pruned: {epoch_pruned_loss/num_batches:.4f}{Fore.RESET}")
        
        # Save best model
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
    epochs: int = 10,
    token: Optional[str] = None,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """Teacher-guided ReplaceMe implementation."""
    
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Starting Teacher-Guided ReplaceMe{Fore.RESET}")
    print(f"{Fore.MAGENTA}Model: {model_path}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Layers to skip: {start_id} -> {end_id}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Teacher weight: {teacher_weight}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{Fore.YELLOW}[DEBUG] Using device: {device}{Fore.RESET}")
    
    # Quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(f"{Fore.YELLOW}[DEBUG] Using 4-bit quantization{Fore.RESET}")
    
    # Load teacher model (original)
    print(f"{Fore.CYAN}[STEP 1] Loading teacher model...{Fore.RESET}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto' if device == 'cuda' else 'cpu',
        quantization_config=quantization_config,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        token=token
    )
    teacher_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = teacher_model.config.hidden_size
    print(f"{Fore.YELLOW}[DEBUG] Hidden size: {hidden_size}{Fore.RESET}")
    
    # Get dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Collect teacher activations
    print(f"{Fore.CYAN}[STEP 2] Collecting teacher activations...{Fore.RESET}")
    teacher_mlp, teacher_attn, teacher_target = collect_teacher_activations(
        teacher_model, tokenizer, dataloader,
        start_id, end_id, max_length, dataset_size, device
    )
    
    # Clear teacher model from memory
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{Fore.YELLOW}[DEBUG] Teacher model cleared from memory{Fore.RESET}")
    
    # Load pruned model
    print(f"{Fore.CYAN}[STEP 3] Loading pruned model...{Fore.RESET}")
    pruned_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto' if device == 'cuda' else 'cpu',
        quantization_config=quantization_config,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    pruned_model = truncate_model(pruned_model, start_id, end_id)
    pruned_model.eval()
    print(f"{Fore.YELLOW}[DEBUG] Model truncated: removed layers {start_id} to {end_id-1}{Fore.RESET}")
    
    # Collect pruned model activations
    print(f"{Fore.CYAN}[STEP 4] Collecting pruned model activations...{Fore.RESET}")
    pruned_mlp, _, pruned_target = collect_teacher_activations(
        pruned_model, tokenizer, dataloader,
        start_id - num_layer, end_id - num_layer, max_length, dataset_size, device
    )
    
    # Clear pruned model for optimization
    del pruned_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Optimize T matrix
    print(f"{Fore.CYAN}[STEP 5] Optimizing transformation matrix...{Fore.RESET}")
    T_optimal = teacher_guided_adam_optimization(
        pruned_mlp, pruned_target,
        teacher_mlp, teacher_attn, teacher_target,
        hidden_size, teacher_weight, lr, epochs, batch_size
    )
    
    # Load model again for final transformation
    print(f"{Fore.CYAN}[STEP 6] Applying transformation to final model...{Fore.RESET}")
    final_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    final_model = truncate_model(final_model, start_id, end_id)
    
    # Apply transformation
    if hasattr(final_model, 'model'):  # LLaMA-style
        layer = final_model.model.layers[start_id - num_layer - 1]
    else:  # Falcon-style
        layer = final_model.transformer.h[start_id - num_layer - 1]
    
    # Update MLP down projection
    old_weight = layer.mlp.down_proj.weight.data.to(torch.float64)
    new_weight = (T_optimal.T @ old_weight).to(torch.bfloat16)
    layer.mlp.down_proj.weight.data = new_weight
    
    print(f"{Fore.YELLOW}[DEBUG] Transformation applied to layer {start_id - num_layer - 1}{Fore.RESET}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_teacher_guided_{layers_to_skip}layers"
    
    save_path = f"{save_path}_w{teacher_weight}"
    final_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save transformation matrix
    torch.save(T_optimal, f"{save_path}/transform.pt")
    
    print(f"{Fore.GREEN}[SUCCESS] Model saved to {save_path}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    # Final cleanup
    del final_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path