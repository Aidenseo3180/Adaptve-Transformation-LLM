"""Optimized Progressive Layer Distillation with Learnable Shortcuts (Optimized-PLDS)

Improved version combining ReplaceMe's simplicity with adaptive components.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all,
                    select_non_overlapping_blocks, adam_method)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class OptimizedMetaBlock(nn.Module):
    """Optimized meta-block combining ReplaceMe's efficiency with adaptive components."""
    
    def __init__(self, 
                 hidden_dim: int, 
                 num_blocks_compressed: int,
                 dtype=torch.bfloat16,
                 use_correction: bool = True,
                 correction_dim_reduction: int = 16):
        super().__init__()
        
        print(f"[DEBUG] Initializing OptimizedMetaBlock:")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - num_blocks_compressed: {num_blocks_compressed}")
        print(f"  - dtype: {dtype}")
        print(f"  - use_correction: {use_correction}")
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks_compressed
        self.dtype = dtype
        
        # 1. Core transform (ReplaceMe style)
        self.core_transform = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        # Initialize as identity
        with torch.no_grad():
            self.core_transform.weight.copy_(torch.eye(hidden_dim, dtype=dtype))
        print(f"[DEBUG] Core transform initialized as identity matrix")
        
        # 2. Residual modeling - critical for performance
        self.residual_scales = nn.Parameter(
            torch.ones(num_blocks_compressed, dtype=dtype) * 0.1
        )
        print(f"[DEBUG] Residual scales initialized: {self.residual_scales.shape}")
        
        # 3. Optional correction network (only for many blocks)
        self.use_correction = use_correction and (num_blocks_compressed > 3)
        if self.use_correction:
            correction_hidden = hidden_dim // correction_dim_reduction
            self.correction = nn.Sequential(
                nn.Linear(hidden_dim, correction_hidden, dtype=dtype),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(correction_hidden, hidden_dim, dtype=dtype)
            )
            # Start with very small correction
            self.correction_weight = nn.Parameter(torch.tensor(0.01, dtype=dtype))
            print(f"[DEBUG] Correction network initialized with hidden dim {correction_hidden}")
        else:
            print(f"[DEBUG] No correction network (blocks <= 3)")
        
        # 4. Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        
        # 5. Gating mechanism for residual paths
        self.residual_gate = nn.Parameter(torch.tensor(0.9, dtype=dtype))
        
        print(f"[DEBUG] OptimizedMetaBlock initialization complete")
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        """Forward pass with optional intermediate outputs for better training."""
        
        x = x.to(self.dtype)
        batch_size, seq_len, hidden_dim = x.shape
        
        # Store input for residual connections
        identity = x.clone()
        
        # Main transformation
        output = self.core_transform(x)
        
        intermediates = []
        
        # Progressive residual addition (simulating multiple layers)
        for i, scale in enumerate(self.residual_scales):
            # Each virtual layer adds scaled residual
            output = output + identity * scale
            output = self.layer_norm(output)
            
            if return_intermediates:
                intermediates.append(output.clone())
        
        # Apply gated residual
        output = self.residual_gate * output + (1 - self.residual_gate) * identity
        
        # Optional correction
        if self.use_correction:
            correction = self.correction(identity)
            output = output + correction * self.correction_weight
        
        if return_intermediates:
            return output, intermediates
        return output
    
    def initialize_from_linear(self, linear_transform: torch.Tensor):
        """Initialize core transform from pre-computed linear transformation."""
        print(f"[DEBUG] Initializing from pre-computed linear transform: {linear_transform.shape}")
        with torch.no_grad():
            self.core_transform.weight.copy_(linear_transform.T.to(self.dtype))
        print(f"[DEBUG] Core transform updated with pre-computed weights")


def compute_initial_transform(
    model,
    tokenizer,
    dataloader,
    start_id: int,
    end_id: int,
    num_layer: int,
    hidden_size: int,
    max_length: int,
    device: str
) -> torch.Tensor:
    """Compute optimal linear transform using ReplaceMe's approach."""
    
    print(f"\n[DEBUG] Computing initial transform for layers {start_id} to {end_id}")
    
    # Setup hooks to capture activations
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    
    # Register hooks
    for i, layer in enumerate(model.model.layers):
        hook = layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        hooks.append(hook)
    
    # Collect activations - Handle variable sequence lengths
    all_mlp_acts = []
    all_layer_inputs = []
    all_layer_outputs = []
    
    print(f"[DEBUG] Collecting activations from dataloader...")
    for batch_idx, batch_text in enumerate(tqdm(dataloader, desc="Collecting activations")):
        if batch_idx >= 10:  # Limit for speed
            break
            
        inputs = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
        
        # Get relevant activations
        mlp_act = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
        layer_input = hidden_states[start_id - num_layer]
        layer_output = hidden_states[end_id - num_layer]
        
        # Flatten batch and sequence dimensions properly
        batch_size, seq_len, hidden_dim = mlp_act.shape
        
        # Reshape to [batch*seq, hidden] for each tensor
        mlp_act_flat = mlp_act.reshape(-1, hidden_dim).cpu()
        layer_input_flat = layer_input.reshape(-1, hidden_dim).cpu()
        layer_output_flat = layer_output.reshape(-1, hidden_dim).cpu()
        
        all_mlp_acts.append(mlp_act_flat)
        all_layer_inputs.append(layer_input_flat)
        all_layer_outputs.append(layer_output_flat)
        
        print(f"[DEBUG] Batch {batch_idx}: shape {mlp_act.shape} -> flattened {mlp_act_flat.shape}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all activations
    print(f"[DEBUG] Concatenating activations...")
    a1 = torch.cat(all_mlp_acts, dim=0)
    a2_input = torch.cat(all_layer_inputs, dim=0)
    a2_output = torch.cat(all_layer_outputs, dim=0)
    
    # Compute target (output - input + mlp)
    a2 = a2_output - a2_input + a1
    
    print(f"[DEBUG] Activation shapes: a1={a1.shape}, a2={a2.shape}")
    
    # Subsample if too large
    max_samples = 50000
    if a1.shape[0] > max_samples:
        print(f"[DEBUG] Subsampling from {a1.shape[0]} to {max_samples} samples")
        indices = torch.randperm(a1.shape[0])[:max_samples]
        a1 = a1[indices]
        a2 = a2[indices]
    
    print(f"[DEBUG] Computing optimal transform using adam method...")
    
    # Use ReplaceMe's adam method
    transform = adam_method(
        a1.to(torch.bfloat16),
        a2.to(torch.bfloat16),
        loss="cosine",
        diag=False
    )
    
    print(f"[DEBUG] Transform computed: {transform.shape}")
    return transform


def optimized_plds_compress(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "train",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    # Optimized PLDS specific
    use_correction: bool = True,
    correction_dim_reduction: int = 16,
    distillation_epochs: int = 15,
    learning_rate: float = 5e-5,
    warmup_epochs: int = 2,
    alpha_mse: float = 0.7,
    alpha_cos: float = 0.2,
    alpha_intermediate: float = 0.1,
    use_pretrained_init: bool = True,
    **kwargs
) -> str:
    """Optimized PLDS compression combining ReplaceMe's approach with adaptive components."""
    
    print(f"\n{'='*60}")
    print(f"Starting Optimized PLDS Compression")
    print(f"Layers {start_id} to {end_id} ({end_id - start_id} blocks)")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logging.info(f"{Fore.GREEN}Loading model for Optimized PLDS...{Fore.RESET}")
    
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
        device_map="auto",
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    num_blocks_to_compress = end_id - start_id
    
    print(f"[DEBUG] Model loaded: hidden_size={hidden_size}")
    
    # Create dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Initialize meta-block
    logging.info(f"{Fore.YELLOW}Creating optimized meta-block...{Fore.RESET}")
    
    meta_block = OptimizedMetaBlock(
        hidden_dim=hidden_size,
        num_blocks_compressed=num_blocks_to_compress,
        dtype=torch.bfloat16,
        use_correction=use_correction,
        correction_dim_reduction=correction_dim_reduction
    ).to(device)
    
    # Pre-compute optimal linear transform if requested
    if use_pretrained_init:
        logging.info(f"{Fore.CYAN}Computing initial transform (ReplaceMe style)...{Fore.RESET}")
        initial_transform = compute_initial_transform(
            model, tokenizer, dataloader,
            start_id, end_id, num_layer,
            hidden_size, max_length, device
        )
        meta_block.initialize_from_linear(initial_transform)
        print(f"[DEBUG] Meta-block initialized with pre-computed transform")
    
    # Setup optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': meta_block.core_transform.parameters(), 'lr': learning_rate},
        {'params': meta_block.residual_scales, 'lr': learning_rate * 2},
        {'params': [meta_block.residual_gate], 'lr': learning_rate * 0.5},
        {'params': meta_block.correction.parameters() if meta_block.use_correction else [], 'lr': learning_rate * 0.5},
        {'params': [meta_block.correction_weight] if meta_block.use_correction else [], 'lr': learning_rate * 0.1},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=distillation_epochs
    )
    
    # Training loop
    logging.info(f"{Fore.CYAN}Starting distillation training...{Fore.RESET}")
    
    model.eval()
    best_loss = float('inf')
    loss_history = []
    best_state = meta_block.state_dict().copy()
    
    for epoch in range(distillation_epochs):
        epoch_losses = {
            'total': 0, 'mse': 0, 'cos': 0, 'inter': 0
        }
        num_batches = 0
        
        # Warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{distillation_epochs}", colour="blue") as pbar:
            for batch_idx, batch_text in enumerate(pbar):
                if batch_idx >= 50:  # Limit batches
                    break
                
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Skip short batches
                if inputs['input_ids'].shape[1] < 10:
                    continue
                
                # Get teacher outputs
                with torch.no_grad():
                    teacher_outputs = model(**inputs)
                    hidden_states = teacher_outputs.hidden_states
                
                # Extract block input/output
                block_input = hidden_states[start_id - num_layer].detach().to(torch.bfloat16)
                block_target = hidden_states[end_id - num_layer].detach().to(torch.bfloat16)
                
                # Check shapes
                if block_input.shape != block_target.shape:
                    continue
                
                # Get intermediate targets
                intermediate_targets = []
                for i in range(num_blocks_to_compress):
                    alpha = (i + 1) / num_blocks_to_compress
                    intermediate = (1 - alpha) * block_input + alpha * block_target
                    intermediate_targets.append(intermediate)
                
                # Forward with intermediates
                optimizer.zero_grad()
                meta_output, intermediates = meta_block(
                    block_input,
                    return_intermediates=True
                )
                
                # Compute losses
                mse_loss = F.mse_loss(meta_output, block_target)
                
                # Cosine similarity loss
                batch_size, seq_len, hidden_dim = meta_output.shape
                meta_norm = F.normalize(meta_output.reshape(-1, hidden_dim), p=2, dim=-1)
                target_norm = F.normalize(block_target.reshape(-1, hidden_dim), p=2, dim=-1)
                cos_loss = 1 - (meta_norm * target_norm).sum(dim=-1).mean()
                
                # Intermediate supervision
                inter_loss = 0
                if len(intermediates) > 0 and len(intermediate_targets) > 0:
                    for inter, target in zip(intermediates, intermediate_targets):
                        inter_loss += F.mse_loss(inter, target)
                    inter_loss /= len(intermediates)
                
                # Total loss
                loss = alpha_mse * mse_loss + alpha_cos * cos_loss + alpha_intermediate * inter_loss
                
                # Check for NaN
                if torch.isnan(loss):
                    continue
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_block.parameters(), 1.0)
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['mse'] += mse_loss.item()
                epoch_losses['cos'] += cos_loss.item()
                epoch_losses['inter'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss
                num_batches += 1
                
                # Update progress
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MSE': f'{mse_loss.item():.4f}',
                    'Cos': f'{cos_loss.item():.4f}'
                })
        
        # Epoch summary
        if num_batches > 0:
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            loss_history.append(avg_losses)
            
            logging.info(
                f"{Fore.GREEN}Epoch {epoch+1} - "
                f"Total: {avg_losses['total']:.4f}, "
                f"MSE: {avg_losses['mse']:.4f}, "
                f"Cos: {avg_losses['cos']:.4f}, "
                f"Inter: {avg_losses['inter']:.4f}{Fore.RESET}"
            )
            
            # Save best model
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                best_state = meta_block.state_dict().copy()
                print(f"[DEBUG] New best loss: {best_loss:.4f}")
        
        scheduler.step()
    
    # Restore best weights
    meta_block.load_state_dict(best_state)
    print(f"[DEBUG] Restored best model with loss: {best_loss:.4f}")
    
    # Print final parameter values
    print(f"\n[DEBUG] Final parameter values:")
    print(f"  - Residual scales: {meta_block.residual_scales.data}")
    print(f"  - Residual gate: {meta_block.residual_gate.data}")
    if meta_block.use_correction:
        print(f"  - Correction weight: {meta_block.correction_weight.data}")
    
    # Clean up teacher
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload for modification
    logging.info(f"{Fore.YELLOW}Loading model for ReplaceMe-style integration...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # ReplaceMe 스타일 통합
    print(f"[DEBUG] Applying ReplaceMe-style integration...")
    
    # 메타블록의 변환을 단일 선형 변환으로 통합
    with torch.no_grad():
        # Core transform weight
        transform_matrix = meta_block.core_transform.weight.data.cpu()
        
        # Residual 효과 통합
        identity = torch.eye(hidden_size, dtype=torch.bfloat16)
        avg_residual_scale = meta_block.residual_scales.mean().item()
        gate_value = meta_block.residual_gate.item()
        
        # 최종 변환 = gated transform + residuals
        final_transform = (gate_value * transform_matrix + 
                          (1 - gate_value) * identity +
                          avg_residual_scale * num_blocks_to_compress * identity)
        
        print(f"[DEBUG] Final transform computed:")
        print(f"  - Gate value: {gate_value:.4f}")
        print(f"  - Avg residual scale: {avg_residual_scale:.4f}")
        print(f"  - Transform shape: {final_transform.shape}")
        
        # Correction 효과 추가 (있는 경우)
        if meta_block.use_correction and meta_block.correction_weight.item() > 0.01:
            correction_weight = meta_block.correction_weight.item()
            print(f"[DEBUG] Adding correction effect: {correction_weight:.4f}")
            # Correction을 작은 perturbation으로 근사
            final_transform = final_transform * (1 + correction_weight)
    
    # 이전 레이어의 down_proj 수정 (ReplaceMe 방식)
    prev_layer_idx = start_id - num_layer - 1
    prev_layer = model.model.layers[prev_layer_idx]
    
    print(f"[DEBUG] Modifying layer {prev_layer_idx} down_proj...")
    print(f"  - Original weight shape: {prev_layer.mlp.down_proj.weight.shape}")
    
    original_weight = prev_layer.mlp.down_proj.weight.data
    
    # Transform 적용: new = T^T @ original
    new_weight = torch.matmul(
        final_transform.T.to(torch.float32),
        original_weight.to(torch.float32)
    ).to(torch.bfloat16)
    
    prev_layer.mlp.down_proj.weight.data = new_weight
    print(f"[DEBUG] down_proj weight updated")
    
    # Truncate model (중간 레이어 제거)
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    print(f"[DEBUG] Model truncated: removed layers {start_id - num_layer} to {end_id - num_layer}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_OptimizedPLDS_ReplaceMe"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}")
    tokenizer.save_pretrained(f"{save_path}")
    
    # Save training info
    torch.save({
        'meta_block': meta_block.state_dict(),
        'loss_history': loss_history,
        'config': {
            'start_id': start_id,
            'end_id': end_id,
            'num_blocks': num_blocks_to_compress,
            'best_loss': best_loss,
            'final_transform': final_transform
        }
    }, f"{save_path}/training_info.pth")
    
    logging.info(f"{Fore.GREEN}Model saved to {save_path}{Fore.RESET}")
    print(f"[DEBUG] Training complete. Best loss: {best_loss:.4f}")
    
    # Cleanup
    del model, meta_block
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path