"""LinearPatch implementation for enhanced LLM compression.

This module implements the LinearPatch method that significantly improves
perplexity preservation in transformer block replacement compression.
"""

import gc
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply Walsh-Hadamard transform for outlier suppression.
    
    Args:
        x: Input tensor of shape (..., d) where d must be power of 2
        
    Returns:
        Transformed tensor with same shape
    """
    # Ensure input dimension is power of 2
    d = x.shape[-1]
    if d & (d - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 1 << (d - 1).bit_length()
        x_padded = torch.zeros(*x.shape[:-1], next_pow2, device=x.device, dtype=x.dtype)
        x_padded[..., :d] = x
        x = x_padded
        d = next_pow2
    
    # Apply Fast Walsh-Hadamard Transform
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                u = x[..., j]
                v = x[..., j + h]
                x[..., j] = u + v
                x[..., j + h] = u - v
        h *= 2
    
    # Normalize
    x = x / torch.sqrt(torch.tensor(d, dtype=x.dtype, device=x.device))
    
    # Remove padding if it was added
    if x.shape[-1] != x.shape[-1]:
        x = x[..., :x.shape[-1]]
    
    return x


def compute_channel_scaling(
    activations_before: torch.Tensor,
    activations_after: torch.Tensor
) -> torch.Tensor:
    """Compute channel-wise scaling factors to preserve activation magnitudes.
    
    Args:
        activations_before: Activations before pruning [N, d]
        activations_after: Activations after pruning [N, d]
        
    Returns:
        Scaling factors [d] in float32 dtype
    """
    # Convert to float32 for numerical stability
    before = activations_before.to(dtype=torch.float32)
    after = activations_after.to(dtype=torch.float32)
    
    # Compute channel-wise mean magnitudes
    mag_before = torch.mean(torch.abs(before), dim=0)
    mag_after = torch.mean(torch.abs(after), dim=0)
    
    # Avoid division by zero
    epsilon = 1e-8
    scaling = mag_before / (mag_after + epsilon)
    
    # Clamp to reasonable range
    scaling = torch.clamp(scaling, min=0.8, max=1.2)
    
    return scaling


def create_linearpatch_matrix(
    hidden_size: int,
    channel_scaling: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Create LinearPatch transformation matrix.
    
    Args:
        hidden_size: Hidden dimension size
        channel_scaling: Channel-wise scaling factors [d]
        device: Target device
        
    Returns:
        LinearPatch matrix [d, d]
    """
    # Ensure consistent dtype (float32 for numerical stability)
    dtype = torch.float32
    
    # Create Hadamard matrix
    if hidden_size & (hidden_size - 1) == 0:
        # Power of 2 - can use efficient Hadamard
        H = torch.ones(1, 1, device=device, dtype=dtype)
        while H.shape[0] < hidden_size:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1)
            ], dim=0)
    else:
        # Use identity matrix if not power of 2
        H = torch.eye(hidden_size, device=device, dtype=dtype)
    
    # Create diagonal scaling matrix with consistent dtype
    channel_scaling = channel_scaling.to(device=device, dtype=dtype)
    D = torch.diag(channel_scaling)
    
    # Combine: P = H^T @ D @ H
    if hidden_size & (hidden_size - 1) == 0:
        P = H.T @ D @ H / hidden_size
    else:
        P = D  # Fallback to just scaling
    
    return P


def linearpatch_compression(
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
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    knowledge_distillation: bool = True,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.3,
    debug_mode: bool = True,
    **kwargs
) -> str:
    """Apply LinearPatch compression method.
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset name for calibration
        dataset_column: Column containing text data
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip (compatibility)
        dataset_size: Size of calibration dataset
        dataset_subset: Subset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save compressed model
        token: Authentication token
        start_id: Starting layer index for compression
        end_id: Ending layer index for compression
        num_layer: Number of layers processed so far
        distances_path: Path to distances file (compatibility)
        num_A: Number of compression blocks
        merge_consecutive: Whether to merge consecutive blocks
        knowledge_distillation: Whether to apply KD during compression
        kd_temperature: Temperature for knowledge distillation
        kd_alpha: Weight for KD loss
        
    Returns:
        Path to saved compressed model
    """
    
    if debug_mode:
        print(f"{Fore.MAGENTA}[DEBUG] LinearPatch starting with parameters:")
        print(f"  - Model: {model_path}")
        print(f"  - Blocks to replace: {start_id-num_layer}-{end_id-num_layer} (originally {start_id}-{end_id})")
        print(f"  - Dataset: {dataset} (size: {dataset_size})")
        print(f"  - KD enabled: {knowledge_distillation}")
        print(f"  - Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "  - Using CPU")
        print(f"{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        if debug_mode:
            print(f"{Fore.YELLOW}[DEBUG] 4-bit quantization enabled{Fore.RESET}")

    # Load model and tokenizer
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Loading model...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    if debug_mode:
        model_size = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{Fore.GREEN}[DEBUG] Model loaded: {model_size:.1f}M parameters{Fore.RESET}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Memory-optimized calibration dataset
    calib_size = min(128, dataset_size) if dataset_size else 128  # Reduced for memory
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Using {calib_size} calibration samples (reduced for memory efficiency){Fore.RESET}")
    
    # Use smaller batch size for memory efficiency
    memory_efficient_batch_size = min(batch_size, 4)
    if debug_mode and memory_efficient_batch_size < batch_size:
        print(f"{Fore.YELLOW}[DEBUG] Reducing batch size from {batch_size} to {memory_efficient_batch_size} for memory{Fore.RESET}")
    
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        calib_size,
        memory_efficient_batch_size,
        tokenizer
    )
    
    logging.info(f"{Fore.GREEN}Collecting activations for LinearPatch analysis{Fore.RESET}")
    
    # Memory-efficient activation collection
    activations_before = []
    activations_after = []
    max_samples_in_memory = 32  # Limit memory usage
    
    def save_activation_before(name):
        def hook(module, input, output):
            if len(activations_before) < max_samples_in_memory:
                # Handle both single tensor and tuple outputs
                if isinstance(output, tuple):
                    # Take the first element if it's a tuple (usually the main output)
                    activation = output[0].detach().cpu()
                else:
                    activation = output.detach().cpu()
                activations_before.append(activation)
        return hook
    
    def save_activation_after(name):
        def hook(module, input, output):
            if len(activations_after) < max_samples_in_memory:
                # Handle both single tensor and tuple outputs  
                if isinstance(output, tuple):
                    # Take the first element if it's a tuple (usually the main output)
                    activation = output[0].detach().cpu()
                else:
                    activation = output.detach().cpu()
                activations_after.append(activation)
        return hook
    
    # Register hooks
    hooks = []
    target_before_idx = start_id - num_layer - 1
    target_after_idx = end_id - num_layer - 1
    
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Target layer indices: before={target_before_idx}, after={target_after_idx}{Fore.RESET}")
        print(f"{Fore.YELLOW}[DEBUG] Total layers in model: {model.config.num_hidden_layers}{Fore.RESET}")
    
    try:
        if 'falcon' in model_path.lower():
            if debug_mode:
                print(f"{Fore.YELLOW}[DEBUG] Using Falcon model architecture{Fore.RESET}")
            before_layer = model.transformer.h[target_before_idx]
            after_layer = model.transformer.h[target_after_idx]
            hooks.append(before_layer.register_forward_hook(save_activation_before('before')))
            hooks.append(after_layer.register_forward_hook(save_activation_after('after')))
        else:
            if debug_mode:
                print(f"{Fore.YELLOW}[DEBUG] Using LLaMA model architecture{Fore.RESET}")
            before_layer = model.model.layers[target_before_idx]
            after_layer = model.model.layers[target_after_idx]
            hooks.append(before_layer.register_forward_hook(save_activation_before('before')))
            hooks.append(after_layer.register_forward_hook(save_activation_after('after')))
            
        if debug_mode:
            print(f"{Fore.GREEN}[DEBUG] Successfully registered hooks on layers {target_before_idx} and {target_after_idx}{Fore.RESET}")
            
    except IndexError as e:
        raise RuntimeError(f"Layer index out of bounds. Model has {model.config.num_hidden_layers} layers, "
                          f"but trying to access layers {target_before_idx} and {target_after_idx}. "
                          f"Check start_id ({start_id}), end_id ({end_id}), num_layer ({num_layer})") from e
    
    # Process calibration data with memory management
    processed_batches = 0
    for batch in tqdm(dataloader, desc=f"{Fore.BLUE}LinearPatch Calibration{Fore.RESET}"):
        # if processed_batches >= 16:  # Limit total batches for memory
        #     if debug_mode:
        #         print(f"{Fore.YELLOW}[DEBUG] Stopping at {processed_batches} batches to preserve memory{Fore.RESET}")
        #     break
            
        # Fixed tokenization to ensure consistent sequence lengths
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",  # Changed from "longest" to "max_length"
            max_length=min(max_length, 512),  # Shorter sequences for memory
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Verify input shape consistency
        if debug_mode and processed_batches < 3:
            print(f"{Fore.CYAN}[DEBUG] Batch {processed_batches} input shape: {inputs['input_ids'].shape}{Fore.RESET}")
        
        with torch.no_grad():
            _ = model(**inputs)
        
        processed_batches += 1
        
        # Clear cache periodically
        if processed_batches % 4 == 0:
            torch.cuda.empty_cache()
            if debug_mode:
                print(f"{Fore.GREEN}[DEBUG] Processed {processed_batches} batches, cleared cache{Fore.RESET}")
    
    # Remove hooks immediately
    for hook in hooks:
        hook.remove()
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] Collected {len(activations_before)} before activations, {len(activations_after)} after activations{Fore.RESET}")
        if len(activations_before) > 0:
            print(f"{Fore.GREEN}[DEBUG] Sample activation shapes:")
            for i, (before, after) in enumerate(zip(activations_before[:3], activations_after[:3])):
                print(f"  Batch {i}: before={before.shape}, after={after.shape}")
            print(f"{Fore.RESET}")
    
    # Check if we have enough data
    if len(activations_before) == 0 or len(activations_after) == 0:
        raise RuntimeError(f"No activations collected! "
                          f"Before: {len(activations_before)}, After: {len(activations_after)}. "
                          f"Check layer indices and model architecture.")
    
    if len(activations_before) != len(activations_after):
        min_len = min(len(activations_before), len(activations_after))
        if debug_mode:
            print(f"{Fore.YELLOW}[DEBUG] Mismatched activation counts, truncating to {min_len}{Fore.RESET}")
        activations_before = activations_before[:min_len]
        activations_after = activations_after[:min_len]
    
    # Concatenate activations efficiently with shape validation
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Concatenating {len(activations_before)} activation tensors...{Fore.RESET}")
    
    try:
        # Validate shapes before concatenation
        expected_seq_len = activations_before[0].shape[1]
        expected_hidden_size = activations_before[0].shape[2]
        
        # Filter out tensors with different sequence lengths
        valid_before = []
        valid_after = []
        
        for i, (before, after) in enumerate(zip(activations_before, activations_after)):
            if before.shape[1] == expected_seq_len and after.shape[1] == expected_seq_len:
                valid_before.append(before)
                valid_after.append(after)
            elif debug_mode:
                print(f"{Fore.YELLOW}[DEBUG] Skipping batch {i} due to shape mismatch: "
                     f"before={before.shape}, after={after.shape} (expected seq_len={expected_seq_len}){Fore.RESET}")
        
        if len(valid_before) == 0:
            raise RuntimeError("No valid activations after shape filtering!")
        
        if debug_mode:
            print(f"{Fore.GREEN}[DEBUG] Using {len(valid_before)}/{len(activations_before)} batches after shape filtering{Fore.RESET}")
        
        acts_before = torch.cat(valid_before, dim=0).view(-1, expected_hidden_size)
        acts_after = torch.cat(valid_after, dim=0).view(-1, expected_hidden_size)
        
    except RuntimeError as e:
        if debug_mode:
            print(f"{Fore.RED}[DEBUG] Detailed error information:")
            print(f"  Expected sequence length: {expected_seq_len}")
            print(f"  Expected hidden size: {expected_hidden_size}")
            print(f"  All before shapes: {[act.shape for act in activations_before]}")
            print(f"  All after shapes: {[act.shape for act in activations_after]}")
            print(f"  Model hidden size: {model.config.hidden_size}{Fore.RESET}")
        raise RuntimeError(f"Failed to concatenate activations. Shape inconsistency detected.") from e
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] Activation shapes: before={acts_before.shape}, after={acts_after.shape}{Fore.RESET}")
    
    # Clear activation lists to free memory
    del activations_before, activations_after
    gc.collect()
    
    logging.info(f"{Fore.GREEN}Computing LinearPatch transformation{Fore.RESET}")
    
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Computing channel scaling...{Fore.RESET}")
    
    # Compute channel scaling
    channel_scaling = compute_channel_scaling(acts_before, acts_after)
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] Channel scaling stats:")
        print(f"  - Mean scaling: {channel_scaling.mean().item():.3f}")
        print(f"  - Min scaling: {channel_scaling.min().item():.3f}")
        print(f"  - Max scaling: {channel_scaling.max().item():.3f}")
        print(f"  - Std scaling: {channel_scaling.std().item():.3f}{Fore.RESET}")
    
    # Create LinearPatch matrix
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Creating LinearPatch matrix (size: {model.config.hidden_size}x{model.config.hidden_size})...{Fore.RESET}")
    
    linearpatch_matrix = create_linearpatch_matrix(
        model.config.hidden_size,
        channel_scaling,
        torch.device('cpu')  # Create on CPU to save GPU memory
    )
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] LinearPatch matrix created successfully")
        print(f"  - Matrix shape: {linearpatch_matrix.shape}")
        print(f"  - Matrix norm: {torch.norm(linearpatch_matrix).item():.3f}")
        print(f"  - Is finite: {torch.isfinite(linearpatch_matrix).all().item()}{Fore.RESET}")
    
    # Clear activation tensors early to free memory
    del acts_before, acts_after
    gc.collect()
    torch.cuda.empty_cache()
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] Freed activation memory{Fore.RESET}")
    
    # Apply knowledge distillation if enabled (with memory optimization)
    if knowledge_distillation:
        if debug_mode:
            print(f"{Fore.YELLOW}[DEBUG] Starting knowledge distillation...")
            print(f"  - Temperature: {kd_temperature}")
            print(f"  - Alpha: {kd_alpha}{Fore.RESET}")
        
        linearpatch_matrix = apply_knowledge_distillation_memory_efficient(
            model, linearpatch_matrix, dataloader, tokenizer, 
            start_id, end_id, num_layer, kd_temperature, kd_alpha, debug_mode
        )
        
        if debug_mode:
            print(f"{Fore.GREEN}[DEBUG] Knowledge distillation completed{Fore.RESET}")
    
    # Clean up and reload model for modification
    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] Reloading model for modification...{Fore.RESET}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation  
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model (remove target blocks)
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Find the optimal target layer - CORRECTED VERSION
    target_layer_idx = start_id - num_layer - 1
    
    if debug_mode:
        print(f"{Fore.CYAN}[DEBUG] Analyzing model architecture for layer {target_layer_idx}:")
    
    if 'falcon' in model_path.lower():
        layer = model.transformer.h[target_layer_idx]
        target_layer = layer.mlp.dense_4h_to_h
        target_layer_name = "dense_4h_to_h"
        if debug_mode:
            print(f"  - Falcon architecture: using {target_layer_name}")
    else:
        layer = model.model.layers[target_layer_idx]
        mlp = layer.mlp
        
        if debug_mode:
            print(f"  - LLaMA layer structure:")
            
        # Priority order: gate_proj > up_proj > down_proj  
        target_layer = None
        target_layer_name = None
        
        mlp_components_priority = [
            ('gate_proj', 'Gate projection (4096 → intermediate)'),
            ('up_proj', 'Up projection (4096 → intermediate)'), 
            ('down_proj', 'Down projection (intermediate → 4096)')
        ]
        
        for comp_name, comp_desc in mlp_components_priority:
            if hasattr(mlp, comp_name):
                comp = getattr(mlp, comp_name)
                if hasattr(comp, 'weight'):
                    comp_shape = comp.weight.shape
                    if debug_mode:
                        print(f"    - {comp_name}: {comp_shape} ({comp.weight.dtype}) - {comp_desc}")
                    
                    # Look for layer with input dimension = hidden_size (4096)
                    if comp_shape[1] == model.config.hidden_size:  # [xxx, 4096] - perfect match
                        target_layer = comp
                        target_layer_name = comp_name
                        if debug_mode:
                            print(f"  ✓ Selected {comp_name}: input dimension matches LinearPatch ({comp_shape[1]})")
                        break
                else:
                    if debug_mode:
                        print(f"    - {comp_name}: No weight attribute")
            else:
                if debug_mode:
                    print(f"    - {comp_name}: Not found")
        
        # Fallback if no perfect match found
        if target_layer is None:
            if debug_mode:
                print(f"  ! No perfect match found, falling back to down_proj")
            target_layer = mlp.down_proj
            target_layer_name = "down_proj"

    if debug_mode:
        print(f"{Fore.RED}[VERIFICATION] Detailed weight analysis:")
        
        # 실제 target layer 구조 확인
        print(f"  - target_layer object: {type(target_layer)}")
        print(f"  - target_layer.weight shape: {target_layer.weight.shape}")
        print(f"  - target_layer.weight.data shape: {target_layer.weight.data.shape}")
        
        # 실제 forward pass 테스트
        test_input = torch.randn(1, 4096, device=target_layer.weight.device, dtype=target_layer.weight.dtype)
        test_output = target_layer(test_input)
        print(f"  - Forward pass: {test_input.shape} -> {test_output.shape}")
        
        # LinearPatch 변환 전후 비교
        original_weight = target_layer.weight.data.clone()
        print(f"  - Original weight shape: {original_weight.shape}")
        print(f"  - Original weight norm: {torch.norm(original_weight):.3f}")
        
        # Matrix multiplication 호환성 체크
        print(f"  - LinearPatch matrix: {linearpatch_matrix.shape}")
        print(f"  - Can multiply LP @ weight? {linearpatch_matrix.shape[1] == original_weight.shape[0]}")
        print(f"  - Can multiply weight @ LP? {original_weight.shape[1] == linearpatch_matrix.shape[0]}")
        
        print(f"{Fore.RESET}")
    
    # CORRECTED 변환 로직 - 실제 target_layer.weight 사용
    original_weight_cpu = target_layer.weight.data.cpu().to(torch.float32)  # 실제 target layer weight 사용
    linearpatch_cpu = linearpatch_matrix.cpu().to(torch.float32)

    if debug_mode:
        print(f"{Fore.YELLOW}[DEBUG] CORRECTED Matrix multiplication setup:")
        print(f"  - Actual weight shape: {original_weight_cpu.shape}")
        print(f"  - LinearPatch shape: {linearpatch_cpu.shape}")
        print(f"  - Weight norm: {torch.norm(original_weight_cpu):.3f}{Fore.RESET}")

    # 올바른 변환 방식 선택
    if original_weight_cpu.shape[1] == linearpatch_cpu.shape[0]:
        # Case: weight @ LP (입력 변환) - 이것이 올바른 방식
        enhanced_weight = original_weight_cpu @ linearpatch_cpu
        print(f"{Fore.GREEN}[DEBUG] Using CORRECT input transformation: weight @ LP")
        print(f"  Result shape: {enhanced_weight.shape}{Fore.RESET}")
        
    elif linearpatch_cpu.shape[1] == original_weight_cpu.shape[0]:
        # Case: LP @ weight (출력 변환)
        enhanced_weight = linearpatch_cpu @ original_weight_cpu
        print(f"{Fore.GREEN}[DEBUG] Using output transformation: LP @ weight")
        print(f"  Result shape: {enhanced_weight.shape}{Fore.RESET}")
        
    else:
        print(f"{Fore.RED}[ERROR] Matrix dimensions incompatible!")
        print(f"  LP: {linearpatch_cpu.shape}, weight: {original_weight_cpu.shape}{Fore.RESET}")
        enhanced_weight = original_weight_cpu  # Fallback

    # Norm 보존
    original_norm = torch.norm(original_weight_cpu)
    enhanced_norm = torch.norm(enhanced_weight)
    if enhanced_norm > 0:
        enhanced_weight = enhanced_weight * (original_norm / enhanced_norm)

    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] Norm preserved: {original_norm:.3f} -> {torch.norm(enhanced_weight):.3f}")
        print(f"  Final enhanced shape: {enhanced_weight.shape}{Fore.RESET}")

    # dtype 복원하여 적용
    target_layer.weight.data = enhanced_weight.to(target_layer.weight.dtype)
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG] Weight transformation completed")
        print(f"  - Enhanced weight shape: {target_layer.weight.data.shape}")
        print(f"  - Enhanced weight dtype: {target_layer.weight.dtype}")
        print(f"  - Weight norm ratio: {torch.norm(target_layer.weight.data) / original_norm:.3f}{Fore.RESET}")
    
    # Save compressed model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_path = f"{save_path}_LinearPatch"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save LinearPatch matrix for analysis
    torch.save(linearpatch_matrix, f"{final_path}_linearpatch_matrix.pth")
    
    logging.info(f"{Fore.GREEN}LinearPatch compression completed: {final_path}{Fore.RESET}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path


def apply_knowledge_distillation_memory_efficient(
    model: nn.Module,
    linearpatch_matrix: torch.Tensor,
    dataloader,
    tokenizer,
    start_id: int,
    end_id: int,
    num_layer: int,
    temperature: float = 4.0,
    alpha: float = 0.3,
    debug_mode: bool = True,
    num_epochs: int = 2  # Reduced epochs for memory
) -> torch.Tensor:
    """Memory-efficient knowledge distillation for LinearPatch improvement.
    
    Args:
        model: The transformer model
        linearpatch_matrix: Initial LinearPatch matrix  
        dataloader: Calibration data loader
        tokenizer: Model tokenizer
        start_id: Start layer index
        end_id: End layer index
        num_layer: Number of layers processed
        temperature: KD temperature
        alpha: KD loss weight
        debug_mode: Enable debug printing
        num_epochs: Number of fine-tuning epochs
        
    Returns:
        Improved LinearPatch matrix
    """
    
    if debug_mode:
        print(f"{Fore.CYAN}[DEBUG KD] Starting memory-efficient KD with {num_epochs} epochs{Fore.RESET}")
    
    # Create learnable parameter from LinearPatch matrix (keep on CPU initially)
    patch_param = nn.Parameter(linearpatch_matrix.clone().requires_grad_(True))
    optimizer = torch.optim.AdamW([patch_param], lr=5e-5, weight_decay=0.01)  # Lower LR for stability
    
    # Store original layer for teacher outputs
    if 'falcon' in str(model.config.architectures):
        original_layer = model.transformer.h[start_id - num_layer - 1].mlp.dense_4h_to_h
    else:
        original_layer = model.model.layers[start_id - num_layer - 1].mlp.down_proj
    
    original_weight = original_layer.weight.data.clone()
    
    if debug_mode:
        print(f"{Fore.CYAN}[DEBUG KD] Target layer: {start_id - num_layer - 1}")
        print(f"  - Original weight shape: {original_weight.shape}")
        print(f"  - Original weight dtype: {original_weight.dtype}")
        print(f"  - LinearPatch matrix shape: {linearpatch_matrix.shape}")
        
        # Check if weight shape is reasonable
        if len(original_weight.shape) != 2:
            print(f"{Fore.RED}[DEBUG KD] WARNING: Weight has unexpected shape! Expected 2D matrix.{Fore.RESET}")
        
        expected_shapes = [
            (4096, 4096),    # Standard down_proj for LLaMA
            (14336, 4096),   # Alternate down_proj size
            (11008, 4096)    # Another common size
        ]
        
        if original_weight.shape not in expected_shapes:
            print(f"{Fore.RED}[DEBUG KD] WARNING: Unexpected weight shape {original_weight.shape}")
            print(f"Expected one of: {expected_shapes}{Fore.RESET}")
            
            # Handle flattened or quantized weights
            if len(original_weight.shape) == 2 and original_weight.shape[1] == 1:
                total_params = original_weight.shape[0]
                print(f"{Fore.YELLOW}[DEBUG KD] Detected flattened weight with {total_params} parameters{Fore.RESET}")
                
                # Standard LLaMA-3-8B down_proj sizes
                possible_shapes = [
                    (11008, 4096),   # LLaMA-2/3 down_proj: 11008 -> 4096
                    (14336, 4096),   # Some variants
                    (4096, 11008),   # Transpose case
                    (4096, 14336)    # Transpose variant
                ]
                
                reshaped = False
                for out_dim, in_dim in possible_shapes:
                    if out_dim * in_dim == total_params:
                        print(f"{Fore.YELLOW}[DEBUG KD] Attempting reshape to {out_dim}x{in_dim}{Fore.RESET}")
                        try:
                            original_weight = original_weight.view(out_dim, in_dim)
                            reshaped = True
                            break
                        except:
                            continue
                
                if not reshaped:
                    # Last resort: try square matrix
                    import math
                    sqrt_params = int(math.sqrt(total_params))
                    if sqrt_params * sqrt_params == total_params:
                        print(f"{Fore.YELLOW}[DEBUG KD] Attempting square reshape to {sqrt_params}x{sqrt_params}{Fore.RESET}")
                        original_weight = original_weight.view(sqrt_params, sqrt_params)
                        reshaped = True
                
                if not reshaped:
                    raise RuntimeError(f"Cannot reshape weight with {total_params} parameters. "
                                     f"Not compatible with standard LLaMA architectures.")
            
            # Handle quantized weights (uint8)
            if original_weight.dtype == torch.uint8:
                print(f"{Fore.YELLOW}[DEBUG KD] Detected quantized weights, converting to float32{Fore.RESET}")
                # Convert uint8 to float32 (simple scaling)
                original_weight = original_weight.to(torch.float32) / 255.0 * 2.0 - 1.0  # Scale to [-1, 1]
        
        print(f"{Fore.GREEN}[DEBUG KD] Final weight shape after processing: {original_weight.shape}")
        print(f"[DEBUG KD] Final weight dtype: {original_weight.dtype}{Fore.RESET}")
        
        print(f"{Fore.GREEN}[DEBUG KD] Final weight shape: {original_weight.shape}{Fore.RESET}")
    
    # Verify that LinearPatch matrix is compatible
    patch_out_dim, patch_in_dim = linearpatch_matrix.shape
    weight_out_dim, weight_in_dim = original_weight.shape
    
    if patch_in_dim != weight_in_dim:
        if debug_mode:
            print(f"{Fore.RED}[DEBUG KD] Dimension mismatch!")
            print(f"  LinearPatch matrix: {linearpatch_matrix.shape}")
            print(f"  Weight matrix: {original_weight.shape}")
            print(f"  Cannot multiply {patch_out_dim}x{patch_in_dim} @ {weight_out_dim}x{weight_in_dim}{Fore.RESET}")
        
        # Try to adjust LinearPatch matrix if possible
        if patch_out_dim == weight_in_dim and patch_in_dim == weight_in_dim:
            # LinearPatch is square and matches input dimension - this is correct
            pass
        else:
            raise RuntimeError(f"LinearPatch matrix {linearpatch_matrix.shape} is incompatible with "
                             f"weight matrix {original_weight.shape}. Expected LinearPatch to be "
                             f"{weight_in_dim}x{weight_in_dim} to transform the input dimension.")
    
    model.eval()  # Keep in eval mode for KD
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Limit batches per epoch for memory
        max_batches_per_epoch = 8
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"KD Epoch {epoch+1}/{num_epochs}")):
            if batch_idx >= max_batches_per_epoch:
                if debug_mode:
                    print(f"{Fore.CYAN}[DEBUG KD] Limiting to {max_batches_per_epoch} batches per epoch{Fore.RESET}")
                break
                
            # Use shorter sequences and smaller batches for memory
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest", 
                max_length=256,  # Much shorter for KD
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Reduce batch size if needed
            if inputs['input_ids'].shape[0] > 2:
                inputs = {k: v[:2] for k, v in inputs.items()}  # Keep only 2 samples
            
            optimizer.zero_grad()
            
            try:
                # Teacher outputs (original model) 
                with torch.no_grad():
                    teacher_outputs = model(**inputs, output_hidden_states=False)  # Don't store hidden states
                    teacher_logits = teacher_outputs.logits.detach()
                
                # Student outputs (with current LinearPatch)
                patch_param_device = patch_param.to(model.device)
                original_weight_device = original_weight.to(model.device)
                
                # Apply LinearPatch transformation - need to handle different weight shapes
                patch_shape = patch_param_device.shape  # Should be [4096, 4096]
                weight_shape = original_weight_device.shape  # Could be [11008, 4096] or other
                
                if debug_mode and batch_idx == 0:
                    print(f"{Fore.CYAN}[DEBUG KD] Matrix multiplication setup:")
                    print(f"  - Patch matrix: {patch_shape}")
                    print(f"  - Weight matrix: {weight_shape}{Fore.RESET}")
                
                # Apply transformation based on weight shape
                if weight_shape[1] == patch_shape[0]:  # weight: [out, in], patch: [in, in]
                    # Transform input dimension: weight @ patch^T
                    enhanced_weight = original_weight_device @ patch_param_device.T
                    if debug_mode and batch_idx == 0:
                        print(f"{Fore.CYAN}[DEBUG KD] Using input transformation: weight @ patch.T{Fore.RESET}")
                        
                elif weight_shape[0] == patch_shape[1]:  # weight: [in, out], patch: [in, in] 
                    # Transform output dimension: patch @ weight
                    enhanced_weight = patch_param_device @ original_weight_device
                    if debug_mode and batch_idx == 0:
                        print(f"{Fore.CYAN}[DEBUG KD] Using output transformation: patch @ weight{Fore.RESET}")
                        
                else:
                    # Incompatible shapes - skip this batch
                    if debug_mode:
                        print(f"{Fore.RED}[DEBUG KD] Incompatible shapes, skipping batch")
                        print(f"  Cannot multiply {patch_shape} with {weight_shape}{Fore.RESET}")
                    continue
                
                if debug_mode and batch_idx == 0:
                    print(f"{Fore.GREEN}[DEBUG KD] Enhanced weight shape: {enhanced_weight.shape}{Fore.RESET}")
                
                original_layer.weight.data = enhanced_weight.to(original_layer.weight.dtype)
                
                student_outputs = model(**inputs, output_hidden_states=False)
                student_logits = student_outputs.logits
                
                # Memory-efficient KD loss - use only small subset of logits
                k = 50  # Reduced from 100
                vocab_size = teacher_logits.shape[-1]
                
                # Sample random subset of vocabulary for efficiency
                if vocab_size > k:
                    random_indices = torch.randperm(vocab_size)[:k].to(teacher_logits.device)
                    teacher_subset = teacher_logits[:, :, random_indices]
                    student_subset = student_logits[:, :, random_indices] 
                else:
                    teacher_subset = teacher_logits
                    student_subset = student_logits
                
                # Compute KD loss
                teacher_soft = torch.softmax(teacher_subset / temperature, dim=-1)
                student_soft = torch.log_softmax(student_subset / temperature, dim=-1)
                
                kd_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft)
                kd_loss *= (temperature ** 2)
                
                # Lighter regularization
                reg_loss = 0.01 * torch.norm(patch_param_device - torch.eye(patch_param_device.shape[0], device=patch_param_device.device), p='fro')
                
                # Total loss
                loss = alpha * kd_loss + (1 - alpha) * reg_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([patch_param], 0.5)  # Smaller clip value
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if debug_mode and batch_idx % 2 == 0:
                    print(f"{Fore.CYAN}[DEBUG KD] Batch {batch_idx}: loss={loss.item():.4f}, kd_loss={kd_loss.item():.4f}{Fore.RESET}")
                
                # Clear cache frequently
                del teacher_logits, student_logits, teacher_subset, student_subset
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if debug_mode:
                        print(f"{Fore.RED}[DEBUG KD] OOM error, skipping batch{Fore.RESET}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            if debug_mode:
                print(f"{Fore.GREEN}[DEBUG KD] Epoch {epoch+1} completed, avg loss: {avg_loss:.4f}{Fore.RESET}")
        
        # Clear cache between epochs
        torch.cuda.empty_cache()
    
    # Restore original weight
    original_layer.weight.data = original_weight
    
    if debug_mode:
        print(f"{Fore.GREEN}[DEBUG KD] Knowledge distillation completed successfully{Fore.RESET}")
    
    return patch_param.detach().cpu()  # Return on CPU


def compute_gradient_importance(
    model: nn.Module,
    dataloader,
    tokenizer,
    layer_indices: list
) -> torch.Tensor:
    """Compute gradient-based importance scores for layers.
    
    Args:
        model: The transformer model
        dataloader: Calibration data
        tokenizer: Model tokenizer
        layer_indices: List of layer indices to evaluate
        
    Returns:
        Importance scores for each layer
    """
    importance_scores = torch.zeros(len(layer_indices))
    
    model.eval()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing gradient importance")):
        if batch_idx >= 10:  # Limit to 10 batches for efficiency
            break
            
        inputs = tokenizer(
            batch,
            return_tensors="pt", 
            padding="longest",
            max_length=512,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass with gradients
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        
        # Accumulate gradient norms for target layers
        for i, layer_idx in enumerate(layer_indices):
            if 'falcon' in str(model.config.architectures):
                layer = model.transformer.h[layer_idx]
            else:
                layer = model.model.layers[layer_idx]
            
            grad_norm = 0.0
            for param in layer.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            
            importance_scores[i] += grad_norm ** 0.5
    
    # Normalize scores
    importance_scores = importance_scores / importance_scores.max()
    
    return importance_scores


def enhanced_layer_selection(
    model: nn.Module,
    dataloader,
    tokenizer,
    layers_to_skip: int,
    cosine_distances: torch.Tensor
) -> Tuple[int, int]:
    """Enhanced layer selection combining multiple metrics.
    
    Args:
        model: The transformer model
        dataloader: Calibration data
        tokenizer: Model tokenizer  
        layers_to_skip: Number of consecutive layers to select
        cosine_distances: Pre-computed cosine distances
        
    Returns:
        (start_idx, end_idx) of optimal layer range
    """
    
    num_layers = model.config.num_hidden_layers
    possible_starts = list(range(num_layers - layers_to_skip))
    
    # Get gradient importance for all layers
    gradient_importance = compute_gradient_importance(
        model, dataloader, tokenizer, possible_starts
    )
    
    # Combine metrics (weights: cosine 50%, gradient 30%, position 20%)
    combined_scores = []
    
    for i, start_idx in enumerate(possible_starts):
        end_idx = start_idx + layers_to_skip
        
        # Cosine distance score (lower is better)
        cosine_score = 1.0 - cosine_distances[start_idx]  # Invert so higher is better
        
        # Gradient importance (lower is better for pruning)  
        grad_score = 1.0 - gradient_importance[i]
        
        # Position score (middle layers often safer to prune)
        position_score = 1.0 - abs(start_idx - num_layers//2) / (num_layers//2)
        
        # Combined score
        combined = 0.5 * cosine_score + 0.3 * grad_score + 0.2 * position_score
        combined_scores.append((start_idx, end_idx, combined))
    
    # Select best scoring range
    best_start, best_end, best_score = max(combined_scores, key=lambda x: x[2])
    
    logging.info(f"{Fore.GREEN}Enhanced selection: layers {best_start}-{best_end} "
                f"(score: {best_score:.3f}){Fore.RESET}")
    
    return best_start, best_end


# Configuration for LinearPatch method
LINEARPATCH_DEFAULT_CONFIG = {
    "knowledge_distillation": True,
    "kd_temperature": 4.0, 
    "kd_alpha": 0.3,
    "enhanced_selection": True,
    "calibration_size_override": 256
}