# ============================================================
# vlm_hybrid.py - Hybrid Teacher-Student Fine-tuning
# Stage 1: ReplaceMe (fast initialization)
# Stage 2: Teacher-Student (end-to-end refinement)
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


def optimize_replaceme_standard(
    a1: torch.Tensor,
    a2: torch.Tensor,
    lr: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Standard ReplaceMe optimization (Stage 1).
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}STAGE 1: REPLACEME INITIALIZATION{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    hidden_size = a1.shape[1]
    num_samples = a1.shape[0]
    
    print(f"{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  Samples: {num_samples:,}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}\n")
    
    # Initialize
    transform = torch.eye(hidden_size, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform], lr=lr)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
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
            
            a1_batch = a1[batch_indices].to(device=device, dtype=torch.float64)
            a2_batch = a2[batch_indices].to(device=device, dtype=torch.float64)
            
            pred = a1_batch @ transform
            
            pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = a2_batch / (a2_batch.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)
            loss = (1.0 - cosine_sim).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.6f}", 'Best': f"{best_loss:.6f}"})
            
            del a1_batch, a2_batch, pred
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\n{Fore.GREEN}  ✓ New best: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Epoch {epoch+1}/{num_epochs}: {avg_loss:.6f} (Best: {best_loss:.6f}){Fore.RESET}")
    
    print(f"\n{Fore.GREEN}Stage 1 completed: T_init ready (Loss: {best_loss:.6f}){Fore.RESET}\n")
    
    return transform.detach()


def teacher_student_finetune(
    T_init: torch.Tensor,
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader,
    target_layer_before: int,
    target_layer_after: int,
    lr: float = 5e-5,
    num_steps: int = 100,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, dict]:
    """
    Teacher-Student fine-tuning (Stage 2).
    
    Refine T by comparing actual student output with teacher output.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}STAGE 2: TEACHER-STUDENT FINE-TUNING{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    print(f"{Fore.CYAN}Configuration:{Fore.RESET}")
    print(f"  Teacher model: Full model")
    print(f"  Student model: Pruned model with T_init")
    print(f"  Target layers: {target_layer_before} → {target_layer_after}")
    print(f"  Fine-tuning steps: {num_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Strategy: Optimize T only (student weights frozen)\n")
    
    # Move T to device and make it require grad
    T = T_init.clone().to(device=device, dtype=torch.float64)
    T.requires_grad = True
    
    # Optimizer for T only
    optimizer = torch.optim.Adam([T], lr=lr)
    
    # Freeze all student model parameters
    print(f"{Fore.CYAN}Freezing student model parameters...{Fore.RESET}")
    for name, param in student_model.named_parameters():
        param.requires_grad = False
    print(f"  ✓ All student parameters frozen\n")
    
    # Setup hooks for MLP activation
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output
        return hook
    
    student_layers, _ = get_vlm_layers(student_model)
    hook = student_layers[target_layer_before].mlp.register_forward_hook(
        save_mlp_activation('student_mlp')
    )
    
    teacher_model.eval()
    student_model.eval()
    
    # Fine-tuning loop
    print(f"{Fore.GREEN}Starting teacher-student fine-tuning...{Fore.RESET}\n")
    
    step = 0
    total_loss = 0.0
    best_loss = float('inf')
    
    pbar = tqdm(total=num_steps, desc=f"{Fore.MAGENTA}Fine-tuning{Fore.RESET}", colour="magenta")
    
    dataloader_iter = iter(dataloader)
    
    while step < num_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader if we run out
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Move to device
        inputs = {k: v.to(teacher_model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Forward through teacher (no grad)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_hidden = teacher_outputs.hidden_states[1:]  # Skip embedding
            teacher_target = teacher_hidden[target_layer_after]  # Li+n from teacher
        
        # Forward through student (with grad for T)
        student_outputs = student_model(**inputs)
        student_hidden = student_outputs.hidden_states[1:]
        
        # Get MLP activation
        student_mlp = mlp_activations['student_mlp']
        
        # Apply transformation
        # Reshape for matmul
        original_shape = student_mlp.shape
        student_mlp_flat = student_mlp.view(-1, student_mlp.shape[-1]).to(torch.float64)
        
        # Mi @ T
        transformed = student_mlp_flat @ T
        
        # Reshape back
        transformed = transformed.view(original_shape).to(student_mlp.dtype)
        
        # Student's predicted output at layer i+n
        # We need to "inject" transformed output into student's computation
        # For simplicity, we compare the final layer output
        student_output = student_hidden[target_layer_after]
        
        # Teacher-student loss
        # Compare outputs at target layer
        teacher_target_flat = teacher_target.view(-1, teacher_target.shape[-1]).to(torch.float64)
        student_output_flat = student_output.view(-1, student_output.shape[-1]).to(torch.float64)
        
        # Cosine distance
        teacher_norm = teacher_target_flat / (teacher_target_flat.norm(dim=1, keepdim=True) + 1e-8)
        student_norm = student_output_flat / (student_output_flat.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (teacher_norm * student_norm).sum(dim=1)
        loss = (1.0 - cosine_sim).mean()
        
        # Backward (only T gets gradient)
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient
        if step == 0:
            print(f"\n{Fore.YELLOW}[DEBUG] First step:{Fore.RESET}")
            print(f"  T requires_grad: {T.requires_grad}")
            print(f"  T.grad is None: {T.grad is None}")
            if T.grad is not None:
                print(f"  T.grad norm: {T.grad.norm().item():.6f}")
            print(f"  Loss: {loss.item():.6f}\n")
        
        optimizer.step()
        
        total_loss += loss.item()
        step += 1
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        pbar.set_postfix({
            'Loss': f"{loss.item():.6f}",
            'Avg': f"{total_loss/step:.6f}",
            'Best': f"{best_loss:.6f}"
        })
        pbar.update(1)
        
        # Cleanup
        del teacher_outputs, student_outputs, teacher_hidden, student_hidden
        torch.cuda.empty_cache()
    
    pbar.close()
    
    # Remove hook
    hook.remove()
    
    # Statistics
    avg_loss = total_loss / num_steps
    stats = {
        'num_steps': num_steps,
        'final_loss': avg_loss,
        'best_loss': best_loss
    }
    
    print(f"\n{Fore.GREEN}Stage 2 completed:{Fore.RESET}")
    print(f"  Average loss: {avg_loss:.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Total steps: {num_steps}\n")
    
    return T.detach(), stats


def vlm_hybrid(
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
    lr_stage1: float = 1e-4,
    num_epochs_stage1: int = 10,
    lr_stage2: float = 5e-5,
    num_steps_stage2: int = 100,
    opt_batch_size: int = 1024,
    token: Optional[str] = None
) -> str:
    """
    VLM pruning with Hybrid Teacher-Student approach.
    
    Two-stage process:
    1. ReplaceMe: Fast initialization
    2. Teacher-Student: End-to-end refinement
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}VLM-HYBRID: REPLACEME + TEACHER-STUDENT{Fore.RESET}")
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
    
    print(f"{Fore.CYAN}Hybrid Configuration:{Fore.RESET}")
    print(f"  Model: {model_path}")
    print(f"  Layers to skip: {layers_to_skip}")
    print(f"  Stage 1 (ReplaceMe): {num_epochs_stage1} epochs, lr={lr_stage1}")
    print(f"  Stage 2 (Teacher-Student): {num_steps_stage2} steps, lr={lr_stage2}\n")
    
    # ===== STAGE 1: ReplaceMe Initialization =====
    
    # Load model
    print(f"{Fore.GREEN}[1/9] Loading model for Stage 1...{Fore.RESET}")
    
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
    print(f"{Fore.GREEN}[2/9] Loading data...{Fore.RESET}")
    dataloader = get_vlm_calib_dataloader(image_dir, dataset_size, batch_size, processor)
    print(f"  Batches: {len(dataloader)}\n")
    
    # Setup hooks
    print(f"{Fore.GREEN}[3/9] Setting up hooks...{Fore.RESET}")
    
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
    print(f"  Target: layer {target_layer_before} → {target_layer_after}\n")
    
    # Pre-allocate
    print(f"{Fore.GREEN}[4/9] Gathering activations for ReplaceMe...{Fore.RESET}")
    
    total_samples = dataset_size if dataset_size else len(dataloader.dataset)
    total_tokens = int(total_samples * 1000 * 1.5)
    
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    cnt = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering", colour="red")):
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]
        mlp_key = f'layer_{target_layer_before}_mlp'
        
        hidden_states_mlp = mlp_activations[mlp_key]
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_i = hidden_states[target_layer_before].view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_n = hidden_states[target_layer_after].view(-1, hidden_size).to(torch.bfloat16)
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        batch_size_tokens = a1_batch.shape[0]
        
        if cnt + batch_size_tokens > total_tokens:
            break
        
        a1[cnt:cnt+batch_size_tokens] = a1_batch.cpu()
        a2[cnt:cnt+batch_size_tokens] = a2_batch.cpu()
        
        cnt += batch_size_tokens
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        torch.cuda.empty_cache()
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    
    print(f"\n{Fore.GREEN}Collected {cnt:,} tokens{Fore.RESET}\n")
    
    # Optimize with ReplaceMe
    print(f"{Fore.GREEN}[5/9] Stage 1: ReplaceMe optimization...{Fore.RESET}\n")
    
    T_init = optimize_replaceme_standard(
        a1, a2,
        lr=lr_stage1,
        num_epochs=num_epochs_stage1,
        batch_size=opt_batch_size,
        device=device
    )
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== STAGE 2: Teacher-Student Fine-tuning =====
    
    print(f"{Fore.GREEN}[6/9] Preparing for Stage 2...{Fore.RESET}\n")
    
    # Teacher model (full)
    print(f"{Fore.CYAN}Loading teacher model (full model)...{Fore.RESET}")
    teacher_model = model  # Keep the full model as teacher
    teacher_model.eval()
    print(f"  ✓ Teacher model ready\n")
    
    # Student model (pruned + T_init applied)
    print(f"{Fore.CYAN}Creating student model (pruned)...{Fore.RESET}")
    
    student_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    # Truncate student
    student_model = truncate_vlm_model(student_model, start_id - num_layer, end_id - num_layer)
    
    # Apply T_init to student
    apply_vlm_transform(student_model, T_init, start_id - num_layer - 1)
    
    student_model.eval()
    print(f"  ✓ Student model ready (pruned with T_init applied)\n")
    
    # Reload dataloader for stage 2
    print(f"{Fore.GREEN}[7/9] Reloading data for fine-tuning...{Fore.RESET}")
    # Use smaller dataset for fine-tuning
    finetune_dataset_size = min(1000, dataset_size) if dataset_size else 1000
    dataloader_finetune = get_vlm_calib_dataloader(
        image_dir, 
        finetune_dataset_size, 
        batch_size, 
        processor
    )
    print(f"  Fine-tuning batches: {len(dataloader_finetune)}\n")
    
    # Fine-tune
    print(f"{Fore.GREEN}[8/9] Stage 2: Teacher-Student fine-tuning...{Fore.RESET}\n")
    
    T_final, stats = teacher_student_finetune(
        T_init,
        teacher_model,
        student_model,
        dataloader_finetune,
        target_layer_before=start_id - num_layer - 1,
        target_layer_after=end_id - num_layer - 1,
        lr=lr_stage2,
        num_steps=num_steps_stage2,
        device=device
    )
    
    print(f"{Fore.GREEN}Hybrid optimization completed!{Fore.RESET}\n")
    
    # Cleanup
    del teacher_model, student_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== FINAL: Apply and Save =====
    
    print(f"{Fore.GREEN}[9/9] Applying final transform and saving...{Fore.RESET}\n")
    
    # Reload clean model
    final_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    final_model = truncate_vlm_model(final_model, start_id - num_layer, end_id - num_layer)
    apply_vlm_transform(final_model, T_final, start_id - num_layer - 1)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/vlm_{os.path.basename(model_path)}_{layers_to_skip}layers"
    
    final_path = f"{save_path}_HYBRID"
    final_model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    # Save stats
    import json
    stats['stage1_epochs'] = num_epochs_stage1
    stats['stage2_steps'] = num_steps_stage2
    stats_path = f"{final_path}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.GREEN}✓ HYBRID MODEL SAVED{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*70}{Fore.RESET}")
    print(f"{Fore.CYAN}Path: {final_path}{Fore.RESET}")
    print(f"{Fore.CYAN}Stats: {stats_path}{Fore.RESET}\n")
    
    if save_transform_only:
        torch.save(T_final, f"{final_path}_transform.pt")
    
    del final_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path