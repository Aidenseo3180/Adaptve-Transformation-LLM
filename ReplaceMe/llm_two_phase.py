# ============================================================
# two_phase.py - LLM Two-Phase Teacher-Student Optimization
# Phase 1: ReplaceMe baseline (MLP-level)
# Phase 2: Full model guidance (Layer-level)
# ============================================================

import gc
import logging
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import (
    get_calib_dataloader,
    truncate_model,
    seed_all
)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def optimize_phase1_replaceme(
    a1: torch.Tensor,
    a2: torch.Tensor,
    lr: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Phase 1: Standard ReplaceMe optimization.
    MLP-level matching: M_i @ T + Y_i ≈ L_{i+n}
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}PHASE 1: REPLACEME BASELINE (MLP-LEVEL){Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    hidden_size = a1.shape[1]
    num_samples = a1.shape[0]
    
    print(f"{Fore.CYAN}Phase 1 Configuration:{Fore.RESET}")
    print(f"  Objective: M_i @ T + Y_i ≈ L_{{i+n}}")
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
        
        # Shuffle
        indices = torch.randperm(num_samples)
        
        pbar = tqdm(
            range(num_batches),
            desc=f"{Fore.CYAN}Phase 1 Epoch {epoch+1}/{num_epochs}{Fore.RESET}",
            leave=True,
            dynamic_ncols=True
        )
        
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch
            a1_batch = a1[batch_indices].to(device=device, dtype=torch.float64)
            a2_batch = a2[batch_indices].to(device=device, dtype=torch.float64)
            
            # Forward
            pred = a1_batch @ transform
            
            # Cosine distance loss
            pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = a2_batch / (a2_batch.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)
            loss = (1.0 - cosine_sim).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Best': f"{best_loss:.6f}"
            })
            
            del a1_batch, a2_batch, pred
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\n{Fore.GREEN}  ✓ New best: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Phase 1 Epoch {epoch+1}/{num_epochs}: {avg_loss:.6f} (Best: {best_loss:.6f}){Fore.RESET}")
    
    print(f"\n{Fore.GREEN}✓ Phase 1 completed: T_base ready (Loss: {best_loss:.6f}){Fore.RESET}\n")
    
    return transform.detach()


def collect_layer_outputs(
    model: AutoModelForCausalLM,
    dataloader,
    tokenizer,
    layer_before: int,
    layer_after: int,
    max_length: int,
    max_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect layer outputs from full teacher model.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}COLLECTING TEACHER LAYER OUTPUTS{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    print(f"{Fore.CYAN}Teacher Model:{Fore.RESET}")
    print(f"  Layer before (input): {layer_before}")
    print(f"  Layer after (target): {layer_after}")
    print(f"  Skip distance: {layer_after - layer_before} layers")
    print(f"  Max samples: {max_samples if max_samples else 'unlimited'}\n")
    
    hidden_size = model.config.hidden_size
    
    L_before_list = []
    L_after_list = []
    
    total_samples = 0
    
    model.eval()
    
    print(f"{Fore.GREEN}Collecting layer outputs from teacher model...{Fore.RESET}\n")
    
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.MAGENTA}Teacher Forward{Fore.RESET}",
        colour="magenta"
    )):
        if max_samples and total_samples >= max_samples:
            print(f"\n{Fore.YELLOW}Reached max samples limit ({max_samples}), stopping{Fore.RESET}\n")
            break
        
        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract specific layers
        # hidden_states[0] = embeddings, hidden_states[1] = layer 0, ...
        L_before_batch = outputs.hidden_states[layer_before + 1]
        L_after_batch = outputs.hidden_states[layer_after + 1]
        
        # Flatten [batch, seq, hidden] → [batch*seq, hidden]
        L_before_batch = L_before_batch.view(-1, hidden_size).cpu().to(torch.bfloat16)
        L_after_batch = L_after_batch.view(-1, hidden_size).cpu().to(torch.bfloat16)
        
        L_before_list.append(L_before_batch)
        L_after_list.append(L_after_batch)
        
        total_samples += inputs['input_ids'].shape[0]
        
        # Memory cleanup
        del outputs
        torch.cuda.empty_cache()
    
    # Concatenate
    print(f"\n{Fore.GREEN}Concatenating layer outputs...{Fore.RESET}")
    L_before = torch.cat(L_before_list, dim=0)
    L_after = torch.cat(L_after_list, dim=0)
    
    print(f"  L_before: {L_before.shape}, {L_before.dtype}")
    print(f"  L_after: {L_after.shape}, {L_after.dtype}")
    print(f"  Total samples: {total_samples}\n")
    
    print(f"{Fore.GREEN}✓ Teacher layer outputs collected{Fore.RESET}\n")
    
    return L_before, L_after


def optimize_phase2_teacher_guided(
    L_before: torch.Tensor,
    L_after: torch.Tensor,
    T_base: torch.Tensor,
    lr: float = 1e-5,
    num_epochs: int = 5,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Phase 2: Fine-tune T using full model layer outputs as teacher.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}PHASE 2: TEACHER-GUIDED REFINEMENT (LAYER-LEVEL){Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    hidden_size = L_before.shape[1]
    num_samples = L_before.shape[0]
    
    print(f"{Fore.CYAN}Phase 2 Configuration:{Fore.RESET}")
    print(f"  Objective: L_before @ T ≈ L_after (teacher guidance)")
    print(f"  Samples: {num_samples:,}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr} (lower for fine-tuning)\n")
    
    # Initialize with T_base
    transform = T_base.clone().to(device=device, dtype=torch.float64)
    transform.requires_grad = True
    
    optimizer = torch.optim.Adam([transform], lr=lr)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle
        indices = torch.randperm(num_samples)
        
        pbar = tqdm(
            range(num_batches),
            desc=f"{Fore.MAGENTA}Phase 2 Epoch {epoch+1}/{num_epochs}{Fore.RESET}",
            leave=True,
            dynamic_ncols=True
        )
        
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch
            L_before_batch = L_before[batch_indices].to(device=device, dtype=torch.float64)
            L_after_batch = L_after[batch_indices].to(device=device, dtype=torch.float64)
            
            # Forward
            pred = L_before_batch @ transform
            
            # Cosine distance loss
            pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = L_after_batch / (L_after_batch.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)
            loss = (1.0 - cosine_sim).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Best': f"{best_loss:.6f}"
            })
            
            del L_before_batch, L_after_batch, pred
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\n{Fore.GREEN}  ✓ New best: {best_loss:.6f}{Fore.RESET}")
        
        print(f"{Fore.YELLOW}Phase 2 Epoch {epoch+1}/{num_epochs}: {avg_loss:.6f} (Best: {best_loss:.6f}){Fore.RESET}")
    
    print(f"\n{Fore.GREEN}✓ Phase 2 completed: T_refined ready (Loss: {best_loss:.6f}){Fore.RESET}\n")
    
    return transform.detach()


def ensemble_transforms(
    T_base: torch.Tensor,
    T_refined: torch.Tensor,
    alpha: float = 0.5
) -> Tuple[torch.Tensor, dict]:
    """
    Ensemble two transforms.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}ENSEMBLE: COMBINING PHASE 1 & PHASE 2{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    print(f"{Fore.CYAN}Ensemble Strategy:{Fore.RESET}")
    print(f"  T_final = {alpha} × T_base + {1-alpha} × T_refined\n")
    
    T_final = alpha * T_base + (1 - alpha) * T_refined
    
    # Statistics
    stats = {
        'alpha': alpha,
        'T_base_norm': T_base.norm().item(),
        'T_refined_norm': T_refined.norm().item(),
        'T_final_norm': T_final.norm().item(),
        'difference_norm': (T_refined - T_base).norm().item()
    }
    
    print(f"{Fore.CYAN}Transform Statistics:{Fore.RESET}")
    print(f"  T_base norm: {stats['T_base_norm']:.4f}")
    print(f"  T_refined norm: {stats['T_refined_norm']:.4f}")
    print(f"  T_final norm: {stats['T_final_norm']:.4f}")
    print(f"  ||T_refined - T_base||: {stats['difference_norm']:.4f}\n")
    
    print(f"{Fore.GREEN}✓ Ensemble completed{Fore.RESET}\n")
    
    return T_final, stats


def two_phase(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: str = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    save_transform_only: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    ensemble_alpha: float = 0.5,
    lr_phase1: float = 1e-4,
    lr_phase2: float = 1e-5,
    num_epochs_phase1: int = 10,
    num_epochs_phase2: int = 5,
    opt_batch_size: int = 1024,
    token: Optional[str] = None,
    **kwargs
) -> str:
    """
    LLM pruning with Two-Phase Teacher-Student optimization.
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Fore.RESET}")
    print(f"{Fore.MAGENTA}LLM TWO-PHASE: TEACHER-STUDENT OPTIMIZATION{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*70}{Fore.RESET}\n")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"{Fore.CYAN}Two-Phase Configuration:{Fore.RESET}")
    print(f"  Layers to skip: {layers_to_skip}")
    print(f"  Start layer ID: {start_id}")
    print(f"  End layer ID: {end_id}")
    print(f"  Phase 1 epochs: {num_epochs_phase1}")
    print(f"  Phase 2 epochs: {num_epochs_phase2}")
    print(f"  Ensemble alpha: {ensemble_alpha}\n")
    
    target_layer_before = start_id - num_layer - 1
    target_layer_after = end_id - num_layer - 1
    
    print(f"{Fore.CYAN}Layer Mapping:{Fore.RESET}")
    print(f"  Pruning layers: {start_id - num_layer} to {end_id - num_layer}")
    print(f"  Target layer (before): {target_layer_before}")
    print(f"  Target layer (after): {target_layer_after}\n")
    
    # ===== LOAD MODEL =====
    print(f"{Fore.GREEN}[Step 1/9] Loading model...{Fore.RESET}")
    
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
    
    print(f"  Layers: {model.config.num_hidden_layers}, Hidden: {hidden_size}\n")
    model.eval()
    
    # ===== LOAD DATA =====
    print(f"{Fore.GREEN}[Step 2/9] Loading data...{Fore.RESET}")
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    print(f"  Batches: {len(dataloader)}\n")
    
    # ===== PHASE 1: COLLECT ACTIVATIONS =====
    print(f"{Fore.GREEN}[Step 3/9] Phase 1 - Collecting MLP activations...{Fore.RESET}")
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    
    for i, layer in enumerate(model.model.layers):
        hooks.append(
            layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        )
    
    print(f"  Hooks registered: {len(hooks)}\n")
    
    # Pre-allocate
    total_tokens = dataset_size * max_length
    
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    cnt = 0
    
    print(f"{Fore.GREEN}[Step 4/9] Gathering Phase 1 activations...{Fore.RESET}\n")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gathering", colour="cyan")):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
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
    
    print(f"\n{Fore.GREEN}Collected {cnt:,} tokens for Phase 1{Fore.RESET}\n")
    
    # Cleanup Phase 1
    for hook in hooks:
        hook.remove()
    
    # ===== PHASE 1: OPTIMIZE =====
    print(f"{Fore.GREEN}[Step 5/9] Phase 1 optimization...{Fore.RESET}\n")
    
    T_base = optimize_phase1_replaceme(
        a1, a2,
        lr=lr_phase1,
        num_epochs=num_epochs_phase1,
        batch_size=opt_batch_size,
        device=device
    )
    
    # Cleanup
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== PHASE 2: COLLECT TEACHER OUTPUTS =====
    print(f"{Fore.GREEN}[Step 6/9] Phase 2 - Collecting teacher layer outputs...{Fore.RESET}")
    
    L_before, L_after = collect_layer_outputs(
        model,
        dataloader,
        tokenizer,
        layer_before=target_layer_before,
        layer_after=target_layer_after,
        max_length=max_length,
        max_samples=dataset_size
    )
    
    # Cleanup model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== PHASE 2: OPTIMIZE =====
    print(f"{Fore.GREEN}[Step 7/9] Phase 2 optimization...{Fore.RESET}\n")
    
    T_refined = optimize_phase2_teacher_guided(
        L_before, L_after,
        T_base,
        lr=lr_phase2,
        num_epochs=num_epochs_phase2,
        batch_size=opt_batch_size,
        device=device
    )
    
    # Cleanup
    del L_before, L_after
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== ENSEMBLE =====
    T_final, stats = ensemble_transforms(T_base, T_refined, alpha=ensemble_alpha)
    
    # ===== RELOAD AND APPLY =====
    print(f"{Fore.GREEN}[Step 8/9] Reloading model and applying transform...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path}_{layers_to_skip}layers".replace("/", "_")
    
    # Apply transformation
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (T_final.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    # ===== SAVE =====
    print(f"\n{Fore.GREEN}[Step 9/9] Saving model...{Fore.RESET}")
    
    final_path = f"{save_path}_TwoPhase"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
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
        torch.save(T_final, f"{final_path}_transform.pt")
    
    del model, T_base, T_refined, T_final
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path