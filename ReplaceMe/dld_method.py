"""Differential Layer Distillation (DLD) - Novel layer compression via delta learning

Learns lightweight networks to approximate layer deltas rather than full transformations.
"""

import gc
import logging
import os
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import json
from dataclasses import dataclass

from .utils import (get_calib_dataloader, seed_all, truncate_model)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


@dataclass
class LayerDeltaStats:
    """Statistics for layer delta patterns"""
    mean_magnitude: float
    std_magnitude: float
    cosine_similarity: float
    compression_ratio: float
    layer_type: str  # 'attention' or 'ffn'


class DeltaNetwork(nn.Module):
    """Lightweight network to learn layer deltas"""
    
    def __init__(self, 
                 hidden_dim: int,
                 bottleneck_dim: int,
                 use_attention: bool = False,
                 dropout: float = 0.1,
                 dtype=torch.bfloat16):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_attention = use_attention
        self.dtype = dtype
        
        if use_attention:
            # Lightweight attention mechanism
            self.query = nn.Linear(hidden_dim, bottleneck_dim, bias=False, dtype=dtype)
            self.key = nn.Linear(hidden_dim, bottleneck_dim, bias=False, dtype=dtype)
            self.value = nn.Linear(hidden_dim, bottleneck_dim, bias=False, dtype=dtype)
            self.output = nn.Linear(bottleneck_dim, hidden_dim, bias=False, dtype=dtype)
            self.scale = bottleneck_dim ** -0.5
            
            print(f"[DLD] Created attention-based delta network (bottleneck: {bottleneck_dim})")
        else:
            # Simple MLP
            self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False, dtype=dtype)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False, dtype=dtype)
            
            print(f"[DLD] Created MLP-based delta network (bottleneck: {bottleneck_dim})")
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with small values for delta learning"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute delta to add to input"""
        original_dtype = x.dtype
        x = x.to(self.dtype)
        
        if self.use_attention:
            batch_size, seq_len, _ = x.shape
            
            # Compute attention
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            delta = self.output(attn_output)
        else:
            # MLP path
            delta = self.down_proj(x)
            delta = self.activation(delta)
            delta = self.dropout(delta)
            delta = self.up_proj(delta)
        
        return delta.to(original_dtype)


class DLDLayer(nn.Module):
    """Layer replacement using learned delta"""
    
    def __init__(self,
                 delta_network: DeltaNetwork,
                 layer_idx: int,
                 preserve_norm: bool = True,
                 scale_factor: float = 1.0):
        super().__init__()
        
        self.delta_network = delta_network
        self.layer_idx = layer_idx
        self.scale_factor = scale_factor
        self.preserve_norm = preserve_norm
        
        # Optional: Keep normalization layers from original
        self.input_layernorm = None
        self.post_attention_layernorm = None
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, ...]:
        
        # Apply layer norm if preserved
        if self.preserve_norm and self.input_layernorm is not None:
            normed_hidden = self.input_layernorm(hidden_states)
        else:
            normed_hidden = hidden_states
        
        # Compute delta
        delta = self.delta_network(normed_hidden)
        
        # Scale and add to input (residual connection)
        output = hidden_states + self.scale_factor * delta
        
        # Return in expected format
        outputs = (output,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)
        
        return outputs


def collect_layer_deltas(
    model: nn.Module,
    layer_idx: int,
    dataloader: List,
    tokenizer: Any,
    max_samples: int = 1000,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, LayerDeltaStats]:
    """Collect input-output pairs and compute deltas for a layer"""
    
    print(f"[DLD] Collecting deltas for layer {layer_idx}")
    
    all_inputs = []
    all_deltas = []
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_text in tqdm(dataloader, desc=f"Layer {layer_idx} deltas"):
            if sample_count >= max_samples:
                break
            
            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get input and output for this layer
            layer_input = hidden_states[layer_idx]
            layer_output = hidden_states[layer_idx + 1]
            
            # Compute delta
            delta = layer_output - layer_input
            
            # Store
            all_inputs.append(layer_input.cpu())
            all_deltas.append(delta.cpu())
            
            sample_count += inputs['input_ids'].shape[0]
    
    # Concatenate all samples
    all_inputs = torch.cat(all_inputs, dim=0)
    all_deltas = torch.cat(all_deltas, dim=0)
    
    # Reshape to [num_tokens, hidden_dim]
    num_samples = all_inputs.shape[0] * all_inputs.shape[1]
    all_inputs = all_inputs.reshape(num_samples, -1)
    all_deltas = all_deltas.reshape(num_samples, -1)
    
    # Compute statistics
    with torch.no_grad():
        mean_magnitude = all_deltas.norm(dim=-1).mean().item()
        std_magnitude = all_deltas.norm(dim=-1).std().item()
        
        # Cosine similarity between input and output
        input_norm = F.normalize(all_inputs, p=2, dim=-1)
        output_norm = F.normalize(all_inputs + all_deltas, p=2, dim=-1)
        cosine_sim = (input_norm * output_norm).sum(dim=-1).mean().item()
        
        # Estimate compression ratio based on delta magnitude
        if mean_magnitude < 0.1:
            compression_ratio = 0.05  # 5% of original
        elif mean_magnitude < 0.5:
            compression_ratio = 0.1   # 10% of original
        elif mean_magnitude < 1.0:
            compression_ratio = 0.2   # 20% of original
        else:
            compression_ratio = 0.3   # 30% of original
        
        # Determine layer type (simplified)
        layer_type = "ffn" if layer_idx % 2 == 0 else "attention"
    
    stats = LayerDeltaStats(
        mean_magnitude=mean_magnitude,
        std_magnitude=std_magnitude,
        cosine_similarity=cosine_sim,
        compression_ratio=compression_ratio,
        layer_type=layer_type
    )
    
    print(f"[DLD] Layer {layer_idx} stats: mag={mean_magnitude:.4f}, cos_sim={cosine_sim:.4f}, compression={compression_ratio}")
    
    return all_inputs, all_deltas, stats


def train_delta_network(
    inputs: torch.Tensor,
    deltas: torch.Tensor,
    hidden_dim: int,
    compression_ratio: float,
    use_attention: bool = False,
    epochs: int = 50,
    batch_size: int = 256,
    device: str = "cuda",
    dtype=torch.bfloat16
) -> DeltaNetwork:
    """Train a delta network to approximate layer deltas"""
    
    print(f"[DLD] Training delta network (compression: {compression_ratio}, attention: {use_attention})")
    
    # Create delta network
    bottleneck_dim = max(16, int(hidden_dim * compression_ratio))
    delta_net = DeltaNetwork(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        use_attention=use_attention,
        dtype=dtype
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(delta_net.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(inputs, deltas)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Training loop
    delta_net.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_inputs, batch_deltas in dataloader:
            batch_inputs = batch_inputs.to(device).to(dtype)
            batch_deltas = batch_deltas.to(device).to(dtype)
            
            # Reshape to add sequence dimension
            batch_inputs = batch_inputs.unsqueeze(1)  # [B, 1, H]
            batch_deltas = batch_deltas.unsqueeze(1)  # [B, 1, H]
            
            # Forward pass
            predicted_delta = delta_net(batch_inputs)
            
            # Multi-objective loss
            mse_loss = F.mse_loss(predicted_delta, batch_deltas)
            
            # Cosine similarity loss
            pred_norm = F.normalize(predicted_delta.squeeze(1), p=2, dim=-1)
            delta_norm = F.normalize(batch_deltas.squeeze(1), p=2, dim=-1)
            cosine_loss = 1 - (pred_norm * delta_norm).sum(dim=-1).mean()
            
            # Magnitude preservation loss
            pred_mag = predicted_delta.norm(dim=-1)
            delta_mag = batch_deltas.norm(dim=-1)
            mag_loss = F.mse_loss(pred_mag, delta_mag)
            
            # Combined loss
            loss = 0.5 * mse_loss + 0.3 * cosine_loss + 0.2 * mag_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(delta_net.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 10 == 0:
            print(f"[DLD] Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    print(f"[DLD] Training complete. Best loss: {best_loss:.6f}")
    
    delta_net.eval()
    return delta_net


def apply_dld_transformation(
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
    # DLD specific parameters
    compression_threshold: float = 0.95,  # Cosine similarity threshold
    min_compression_ratio: float = 0.05,
    max_compression_ratio: float = 0.3,
    use_attention_for_complex: bool = True,
    training_epochs: int = 50,
    preserve_critical_layers: bool = True,
    healing_epochs: int = 5,
    healing_lr: float = 1e-5,
    **kwargs
) -> str:
    """Apply Differential Layer Distillation transformation"""
    
    print(f"\n{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Starting Differential Layer Distillation (DLD){Fore.RESET}")
    print(f"{Fore.MAGENTA}Target layers: {start_id} to {end_id}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[DLD] Using device: {device}")
    
    # Load model
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
        )
    
    print(f"[DLD] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=dtype
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    hidden_size = model.config.hidden_size
    
    # Get calibration data
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size if dataset_size else 1000,
        batch_size,
        tokenizer
    )
    dataloader_list = list(dataloader)
    
    # Determine target layers
    if end_id == 0:
        end_id = model.config.num_hidden_layers
    
    target_layers = list(range(start_id - 1, min(end_id - 1, model.config.num_hidden_layers - 1)))
    
    # Protect critical layers
    if preserve_critical_layers:
        # Keep first 3 and last 3 layers
        critical_layers = list(range(3)) + list(range(model.config.num_hidden_layers - 3, model.config.num_hidden_layers))
        target_layers = [l for l in target_layers if l not in critical_layers]
    
    print(f"[DLD] Target layers for transformation: {target_layers}")
    
    # Phase 1: Collect deltas and train networks
    layer_delta_networks = {}
    layer_stats = {}
    
    for layer_idx in target_layers:
        print(f"\n[DLD] Processing layer {layer_idx + 1}/{model.config.num_hidden_layers}")
        
        # Collect deltas
        inputs, deltas, stats = collect_layer_deltas(
            model, layer_idx, dataloader_list[:100], 
            tokenizer, max_samples=1000, device=device
        )
        
        layer_stats[layer_idx] = stats
        
        # Skip if layer is already very similar to input
        if stats.cosine_similarity > compression_threshold:
            print(f"[DLD] Layer {layer_idx + 1} has high similarity ({stats.cosine_similarity:.4f}), using minimal delta")
            compression_ratio = min_compression_ratio
        else:
            compression_ratio = min(max_compression_ratio, stats.compression_ratio)
        
        # Decide whether to use attention based on complexity
        use_attention = use_attention_for_complex and stats.mean_magnitude > 0.5
        
        # Train delta network
        delta_net = train_delta_network(
            inputs, deltas,
            hidden_dim=hidden_size,
            compression_ratio=compression_ratio,
            use_attention=use_attention,
            epochs=training_epochs,
            device=device,
            dtype=dtype
        )
        
        layer_delta_networks[layer_idx] = delta_net
    
    # Phase 2: Replace layers
    print(f"\n{Fore.CYAN}[DLD] Replacing layers with delta networks...{Fore.RESET}")
    
    for layer_idx in target_layers:
        if layer_idx in layer_delta_networks:
            original_layer = model.model.layers[layer_idx]
            delta_net = layer_delta_networks[layer_idx]
            
            # Create DLD layer
            dld_layer = DLDLayer(
                delta_network=delta_net,
                layer_idx=layer_idx,
                preserve_norm=True,
                scale_factor=1.0
            )
            
            # Preserve normalization layers
            if hasattr(original_layer, 'input_layernorm'):
                dld_layer.input_layernorm = original_layer.input_layernorm
            if hasattr(original_layer, 'post_attention_layernorm'):
                dld_layer.post_attention_layernorm = original_layer.post_attention_layernorm
            
            # Replace layer
            model.model.layers[layer_idx] = dld_layer
            
            print(f"[DLD] Layer {layer_idx + 1} replaced with delta network")
    
    # Phase 3: Optional healing/fine-tuning
    if healing_epochs > 0:
        print(f"\n{Fore.YELLOW}[DLD] Starting healing phase...{Fore.RESET}")
        
        # Prepare for fine-tuning
        model.train()
        
        # Only optimize delta networks and layer norms
        params_to_optimize = []
        for layer_idx in target_layers:
            if layer_idx in layer_delta_networks:
                dld_layer = model.model.layers[layer_idx]
                params_to_optimize.extend(dld_layer.delta_network.parameters())
                if dld_layer.input_layernorm is not None:
                    params_to_optimize.extend(dld_layer.input_layernorm.parameters())
        
        if params_to_optimize:
            optimizer = torch.optim.AdamW(params_to_optimize, lr=healing_lr)
            
            for epoch in range(healing_epochs):
                total_loss = 0
                num_batches = min(50, len(dataloader_list))
                
                for batch_idx, batch_text in enumerate(dataloader_list[:num_batches]):
                    inputs = tokenizer(
                        batch_text,
                        return_tensors="pt",
                        padding="longest",
                        max_length=max_length,
                        truncation=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / num_batches
                print(f"[DLD] Healing epoch {epoch + 1}/{healing_epochs}, Loss: {avg_loss:.4f}")
        
        model.eval()
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_DLD_{start_id}_{end_id}"
    
    print(f"\n{Fore.GREEN}[DLD] Saving model to {save_path}{Fore.RESET}")
    
    # Move to CPU for saving
    model = model.to('cpu')
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save DLD metadata
    dld_metadata = {
        'method': 'DLD',
        'layers_replaced': target_layers,
        'layer_stats': {
            f'layer_{idx}': {
                'mean_magnitude': stats.mean_magnitude,
                'std_magnitude': stats.std_magnitude,
                'cosine_similarity': stats.cosine_similarity,
                'compression_ratio': stats.compression_ratio,
                'layer_type': stats.layer_type
            } for idx, stats in layer_stats.items()
        },
        'parameters': {
            'compression_threshold': compression_threshold,
            'min_compression_ratio': min_compression_ratio,
            'max_compression_ratio': max_compression_ratio,
            'use_attention_for_complex': use_attention_for_complex,
            'training_epochs': training_epochs,
            'healing_epochs': healing_epochs
        }
    }
    
    with open(f"{save_path}/dld_metadata.json", 'w') as f:
        json.dump(dld_metadata, f, indent=2)
    
    print(f"[DLD] Metadata saved to {save_path}/dld_metadata.json")
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}DLD Transformation Complete!{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*60}{Fore.RESET}")
    
    total_original_params = sum(p.numel() for p in model.parameters())
    delta_params = sum(
        sum(p.numel() for p in layer_delta_networks[idx].parameters())
        for idx in target_layers if idx in layer_delta_networks
    )
    
    print(f"Layers replaced: {len(target_layers)}")
    print(f"Parameter reduction: {(1 - delta_params/total_original_params)*100:.1f}%")
    print(f"Average compression ratio: {np.mean([s.compression_ratio for s in layer_stats.values()]):.3f}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path