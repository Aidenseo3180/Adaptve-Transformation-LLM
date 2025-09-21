# adaptive_replaceme.py
"""Adaptive Residual ReplaceMe: Context-aware transformation with residual preservation."""

import gc
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class AdaptiveResidualTransform(nn.Module):
    """Adaptive transformation that preserves important residual information."""
    
    def __init__(self, hidden_size: int, calibration_stats: dict):
        super().__init__()
        
        # Base transformation matrix (initialized as identity)
        self.W = nn.Parameter(torch.eye(hidden_size, dtype=torch.float32))
        
        # Residual preservation gate (learnable)
        initial_ratio = calibration_stats.get('residual_ratio', 0.7)
        self.residual_gate = nn.Parameter(torch.tensor(initial_ratio, dtype=torch.float32))
        
        # Confidence estimator (small network for adaptive mixing)
        self.confidence_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        print(f"[AR-ReplaceMe] Initialized with residual_ratio={initial_ratio:.3f}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive transformation with residual preservation."""
        # Ensure correct device and dtype
        device = x.device
        dtype = x.dtype
        
        # Move parameters to correct device if needed
        if self.W.device != device:
            self.W = self.W.to(device)
            self.residual_gate = self.residual_gate.to(device)
            self.confidence_scale = self.confidence_scale.to(device)
        
        # Convert to float32 for computation
        x_float = x.float()
        
        # Standard transformation
        transformed = torch.matmul(x_float, self.W.T)
        
        # Compute input confidence (higher variance = more confident)
        x_std = x_float.std(dim=-1, keepdim=True)
        x_mean_std = x_std.mean()
        
        # Avoid division by zero
        if x_mean_std > 1e-6:
            confidence = torch.sigmoid((x_std - x_mean_std) / (x_mean_std + 1e-6))
        else:
            confidence = torch.ones_like(x_std) * 0.5
        
        # Adaptive mixing: confident inputs preserve more residual
        gate_value = torch.sigmoid(self.residual_gate) * confidence
        
        # Mix original and transformed
        output = x_float * gate_value + transformed * (1 - gate_value)
        
        # Convert back to original dtype
        return output.to(dtype)


def analyze_calibration_data(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    residuals: Optional[torch.Tensor] = None
) -> dict:
    """Analyze calibration data to extract statistics for adaptive transformation."""
    
    print("[AR-ReplaceMe] Analyzing calibration data...")
    
    # Compute residual vs transformation ratio
    residual_component = outputs - inputs
    
    # Avoid division by zero
    input_norm = torch.norm(inputs) + 1e-8
    residual_norm = torch.norm(residual_component) + 1e-8
    
    residual_ratio = input_norm / (input_norm + residual_norm)
    
    # Compute per-token variance
    per_token_variance = residual_component.std(dim=1).mean().item()
    
    # Position-wise patterns (if sequence dimension exists)
    if len(inputs.shape) > 2:
        position_patterns = residual_component.mean(dim=0).std().item()
    else:
        position_patterns = 0.0
    
    stats = {
        'residual_ratio': residual_ratio.item(),
        'token_variance': per_token_variance,
        'position_bias': position_patterns,
        'input_norm': input_norm.item(),
        'output_norm': torch.norm(outputs).item()
    }
    
    print(f"[AR-ReplaceMe] Calibration stats:")
    print(f"  - Residual ratio: {stats['residual_ratio']:.3f}")
    print(f"  - Token variance: {stats['token_variance']:.3f}")
    print(f"  - Position bias: {stats['position_bias']:.3f}")
    
    return stats


def train_adaptive_transform(
    transform: AdaptiveResidualTransform,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    residuals: Optional[torch.Tensor] = None,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3
) -> AdaptiveResidualTransform:
    """Train the adaptive transformation using calibration data."""
    
    class CalibrationDataset(Dataset):
        def __init__(self, inputs, outputs, residuals=None):
            self.inputs = inputs
            self.outputs = outputs
            self.residuals = residuals
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            if self.residuals is not None:
                return self.inputs[idx], self.outputs[idx], self.residuals[idx]
            return self.inputs[idx], self.outputs[idx], torch.zeros_like(self.inputs[idx])
    
    # Move transform to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transform.to(device)
    
    # Create dataset and loader
    dataset = CalibrationDataset(inputs, outputs, residuals)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)
    
    # Loss function (combination of cosine and MSE)
    def combined_loss(pred, target):
        # Cosine similarity loss
        pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_loss = 1 - (pred_norm * target_norm).sum(dim=-1).mean()
        
        # MSE loss
        mse_loss = nn.MSELoss()(pred, target)
        
        # Weighted combination
        return 0.7 * cosine_loss + 0.3 * mse_loss
    
    print(f"[AR-ReplaceMe] Training adaptive transform for {epochs} epochs...")
    
    with tqdm(range(epochs), desc="Training AR-Transform") as pbar:
        for epoch in pbar:
            total_loss = 0
            num_batches = 0
            
            for batch_inputs, batch_outputs, batch_residuals in loader:
                # Move to device
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                batch_residuals = batch_residuals.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                pred = transform(batch_inputs)
                
                # Add residuals if available
                if residuals is not None and batch_residuals.sum() != 0:
                    pred = pred + batch_residuals
                
                # Compute loss
                loss = combined_loss(pred, batch_outputs)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Clamp residual gate to [0, 1] range
                transform.residual_gate.data = transform.residual_gate.data.clamp(0, 1)
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 
                             'Gate': f'{torch.sigmoid(transform.residual_gate).item():.3f}'})
    
    print(f"[AR-ReplaceMe] Training completed. Final gate value: {torch.sigmoid(transform.residual_gate).item():.3f}")
    
    return transform


def adaptive_replaceme(
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
    epochs: int = 10,
    lr: float = 1e-3,
    **kwargs
) -> str:
    """Main function for Adaptive Residual ReplaceMe."""
    
    print(f"[AR-ReplaceMe] Starting adaptive transformation for layers {start_id}-{end_id}")
    
    # Load model for activation gathering
    device_map = "auto" if torch.cuda.is_available() else "cpu"
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
    
    # Gather activations
    print(f"[AR-ReplaceMe] Gathering activations from {dataset_size} samples...")
    
    all_inputs = []
    all_outputs = []
    all_residuals = []
    
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
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            
            # Get relevant hidden states
            h_start = hidden_states[start_id - num_layer - 1]
            h_end = hidden_states[end_id - num_layer - 1]
            
            # Flatten batch and sequence dimensions
            h_start_flat = h_start.view(-1, hidden_size).cpu()
            h_end_flat = h_end.view(-1, hidden_size).cpu()
            
            all_inputs.append(h_start_flat)
            all_outputs.append(h_end_flat)
            
            # Optional: gather attention residuals if needed
            if len(hidden_states) > start_id - num_layer:
                h_mid = hidden_states[start_id - num_layer]
                residual = (h_mid - h_start).view(-1, hidden_size).cpu()
                all_residuals.append(residual)
    
    # Concatenate all activations
    all_inputs = torch.cat(all_inputs, dim=0).to(torch.float32)
    all_outputs = torch.cat(all_outputs, dim=0).to(torch.float32)
    
    if all_residuals:
        all_residuals = torch.cat(all_residuals, dim=0).to(torch.float32)
    else:
        all_residuals = None
    
    print(f"[AR-ReplaceMe] Collected {all_inputs.shape[0]} activation pairs")
    
    # Analyze calibration data
    calibration_stats = analyze_calibration_data(all_inputs, all_outputs, all_residuals)
    
    # Create and train adaptive transform
    transform = AdaptiveResidualTransform(hidden_size, calibration_stats)
    transform = train_adaptive_transform(
        transform, all_inputs, all_outputs, all_residuals,
        epochs=epochs, batch_size=batch_size, lr=lr
    )
    
    # Clean up before reloading
    del model, all_inputs, all_outputs
    if all_residuals is not None:
        del all_residuals
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for modification
    print("[AR-ReplaceMe] Applying transformation to model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Extract the learned transformation matrix
    W_learned = transform.W.detach().cpu()
    gate_value = torch.sigmoid(transform.residual_gate.detach().cpu()).item()
    
    print(f"[AR-ReplaceMe] Applying transformation with gate={gate_value:.3f}")
    
    # Create effective transformation (mixing identity and learned)
    identity = torch.eye(hidden_size, dtype=torch.float32)
    W_effective = gate_value * identity + (1 - gate_value) * W_learned
    
    # Apply to down_proj layer
    target_layer = model.model.layers[start_id - num_layer - 1]
    original_weight = target_layer.mlp.down_proj.weight.to(torch.float32)
    
    # Apply transformation
    new_weight = torch.matmul(W_effective.T, original_weight)
    target_layer.mlp.down_proj.weight = nn.Parameter(new_weight.to(torch.bfloat16))
    
    # Save model
    if save_path is None:
        save_path = f"output_models/AR_ReplaceMe_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[AR-ReplaceMe] Model saved to {save_path}")
    
    # Save transformation details
    torch.save({
        'W': W_effective,
        'gate': gate_value,
        'stats': calibration_stats
    }, f"{save_path}/transform_details.pt")
    
    return save_path