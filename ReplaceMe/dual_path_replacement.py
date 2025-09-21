# dual_path_replacement.py
"""Dual-Path Adaptive Replacement: Combining linear and non-linear transformations."""

import gc
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tqdm import tqdm
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class DualPathTransform(nn.Module):
    """Dual-path transformation combining linear and non-linear components."""
    
    def __init__(self, hidden_size: int, reduction_factor: int = 4):
        super().__init__()
        
        # Linear transformation path
        self.W = nn.Parameter(torch.eye(hidden_size, dtype=torch.float32))
        print(f"[DualPath] Initialized linear path: {hidden_size}x{hidden_size}")
        
        # Non-linear corrector path (bottleneck)
        reduced_dim = hidden_size // reduction_factor
        self.corrector = nn.Sequential(
            nn.Linear(hidden_size, reduced_dim, bias=False),
            nn.GELU(),
            nn.Linear(reduced_dim, hidden_size, bias=False)
        )
        
        # Initialize corrector to near-zero output
        with torch.no_grad():
            nn.init.xavier_uniform_(self.corrector[0].weight, gain=0.01)
            nn.init.xavier_uniform_(self.corrector[2].weight, gain=0.01)
        
        print(f"[DualPath] Initialized corrector: {hidden_size} -> {reduced_dim} -> {hidden_size}")
        
        # Learnable mixing parameter (starts mostly linear)
        self.alpha = nn.Parameter(torch.tensor(2.0))  # sigmoid(2.0) ≈ 0.88
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-path transformation."""
        # Ensure correct dtype
        x = x.float()
        
        # Linear path
        linear_out = torch.matmul(x, self.W)
        
        # Non-linear correction path
        correction = self.corrector(x)
        
        # Adaptive mixing
        alpha = torch.sigmoid(self.alpha)
        output = alpha * linear_out + (1 - alpha) * correction
        
        return output
    
    def get_effective_matrix(self) -> torch.Tensor:
        """Get the effective linear transformation matrix for model integration."""
        # For integration into down_proj, we use mainly the linear part
        # with a small correction bias
        alpha = torch.sigmoid(self.alpha).item()
        print(f"[DualPath] Effective alpha for final transform: {alpha:.3f}")
        return self.W.detach()


def gather_calibration_data(
    model: nn.Module,
    tokenizer,
    dataloader,
    start_id: int,
    end_id: int,
    num_layer: int,
    hidden_size: int,
    max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather calibration data with MLP outputs."""
    
    print(f"[DualPath] Gathering calibration data from layers {start_id} to {end_id}")
    
    # Setup hooks for MLP outputs
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        hooks.append(hook)
    
    a1_list = []
    a2_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Gathering activations"):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]
            
            # Get MLP output and target states
            mlp_out = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
            h_before = hidden_states[start_id - num_layer - 1]
            h_after = hidden_states[end_id - num_layer - 1]
            
            # Compute calibration pairs
            a1 = mlp_out.view(-1, hidden_size)
            a2 = (h_after + mlp_out - h_before).view(-1, hidden_size)
            
            a1_list.append(a1.cpu())
            a2_list.append(a2.cpu())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all data
    a1 = torch.cat(a1_list, dim=0)
    a2 = torch.cat(a2_list, dim=0)
    
    print(f"[DualPath] Collected {a1.shape[0]} calibration samples")
    
    return a1, a2


def train_dual_path(
    dual_path: DualPathTransform,
    a1: torch.Tensor,
    a2: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 1024,
    lr: float = 5e-4,
    weight_decay: float = 1e-5
) -> DualPathTransform:
    """Train the dual-path transformation."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dual_path = dual_path.to(device)
    
    # Move data to device
    a1 = a1.to(device).float()
    a2 = a2.to(device).float()
    
    # Setup optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': [dual_path.W], 'lr': lr},
        {'params': dual_path.corrector.parameters(), 'lr': lr * 0.5},
        {'params': [dual_path.alpha], 'lr': lr * 0.1}
    ], weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"[DualPath] Training for {epochs} epochs...")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_cosine = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(a1.shape[0])
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:min(i+batch_size, len(indices))]
            batch_a1 = a1[batch_idx]
            batch_a2 = a2[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = dual_path(batch_a1)
            
            # MSE loss
            mse_loss = F.mse_loss(pred, batch_a2)
            
            # Cosine similarity loss
            pred_norm = F.normalize(pred, dim=-1)
            a2_norm = F.normalize(batch_a2, dim=-1)
            cosine_loss = 1 - (pred_norm * a2_norm).sum(-1).mean()
            
            # Regularization: keep W close to identity
            reg_loss = 0.01 * ((dual_path.W - torch.eye(dual_path.W.shape[0], device=device)) ** 2).mean()
            
            # Combined loss
            loss = 0.7 * mse_loss + 0.3 * cosine_loss + reg_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_path.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_cosine += cosine_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_cosine = epoch_cosine / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = dual_path.state_dict().copy()
        
        if epoch % 3 == 0 or epoch == epochs - 1:
            alpha = torch.sigmoid(dual_path.alpha).item()
            print(f"[DualPath] Epoch {epoch}: Loss={avg_loss:.4f}, MSE={avg_mse:.4f}, "
                  f"Cosine={avg_cosine:.4f}, Alpha={alpha:.3f}")
    
    # Load best state
    if best_state is not None:
        dual_path.load_state_dict(best_state)
    
    print(f"[DualPath] Training complete. Best loss: {best_loss:.4f}")
    
    return dual_path


def dual_path_replacement(
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
    reduction_factor: int = 4,
    epochs: int = 15,
    lr: float = 5e-4,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """Main dual-path replacement function."""
    
    print(f"\n{'='*60}")
    print(f"[DualPath] Starting Dual-Path Replacement")
    print(f"[DualPath] Layers {start_id} to {end_id} (skip {end_id - start_id} layers)")
    print(f"{'='*60}\n")
    
    # Device configuration
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("[DualPath] Using 4-bit quantization")
    
    # Load model for calibration
    print("[DualPath] Loading model for calibration...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    # Get calibration dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Gather calibration data
    a1, a2 = gather_calibration_data(
        model, tokenizer, dataloader,
        start_id, end_id, num_layer,
        hidden_size, max_length
    )
    
    # Analyze calibration data
    with torch.no_grad():
        residual = a2 - a1
        residual_ratio = torch.norm(residual) / torch.norm(a1)
        print(f"[DualPath] Residual ratio: {residual_ratio:.3f}")
    
    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create and train dual-path transform
    dual_path = DualPathTransform(hidden_size, reduction_factor)
    dual_path = train_dual_path(dual_path, a1, a2, epochs=epochs, lr=lr)
    
    # Clean up calibration data
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model for modification
    print("\n[DualPath] Loading model for modification...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model (remove skipped layers)
    print(f"[DualPath] Truncating layers {start_id - num_layer} to {end_id - num_layer}")
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj
    target_layer_idx = start_id - num_layer - 1
    target_layer = model.model.layers[target_layer_idx]
    
    print(f"[DualPath] Applying transformation to layer {target_layer_idx}")
    
    # Get effective transformation matrix
    W_effective = dual_path.get_effective_matrix().cpu()
    
    # Apply to down_proj weight
    original_weight = target_layer.mlp.down_proj.weight.to(torch.float32)
    new_weight = torch.matmul(W_effective.T, original_weight)
    target_layer.mlp.down_proj.weight = nn.Parameter(new_weight.to(torch.bfloat16))
    
    print(f"[DualPath] Modified down_proj weight shape: {new_weight.shape}")
    
    # Verify model structure
    print(f"[DualPath] Final model has {len(model.model.layers)} layers")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/DualPath_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
    print(f"[DualPath] Saving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save transformation details
    torch.save({
        'dual_path_state': dual_path.state_dict(),
        'config': {
            'start_id': start_id,
            'end_id': end_id,
            'hidden_size': hidden_size,
            'reduction_factor': reduction_factor,
            'final_alpha': torch.sigmoid(dual_path.alpha).item()
        }
    }, f"{save_path}/dual_path_info.pt")
    
    print(f"[DualPath] Model saved successfully!")
    print(f"{'='*60}\n")
    
    return save_path