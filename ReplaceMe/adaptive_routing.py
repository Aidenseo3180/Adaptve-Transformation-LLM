"""Adaptive Layer Routing with Learnable Bypass Gates

This module implements dynamic layer routing that adaptively skips or replaces
transformer layers based on input complexity, without requiring healing.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all,
                    select_non_overlapping_blocks)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class AdaptiveRouter(nn.Module):
    """Lightweight router for deciding layer bypass vs full computation"""
    
    def __init__(self, hidden_size: int, temperature: float = 1.0):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # [full_path_prob, bypass_prob]
        )
        self.temperature = temperature
        
        # Initialize with slight bias towards full path
        with torch.no_grad():
            self.router[-1].bias.data = torch.tensor([0.5, -0.5])
        
        print(f"[DEBUG] Initialized AdaptiveRouter with hidden_size={hidden_size}")
    
    def forward(self, hidden_states: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            training: whether in training mode
        
        Returns:
            routing_weights: [batch_size, 2] probabilities for [full, bypass]
            stats: dictionary with routing statistics
        """
        # Pool across sequence dimension - using attention-weighted pooling
        attention_weights = torch.softmax(hidden_states.mean(dim=-1), dim=-1)  # [B, seq_len]
        pooled = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)  # [B, hidden_size]
        
        # Get routing logits
        routing_logits = self.router(pooled)  # [B, 2]
        
        # Apply temperature and get probabilities
        if training:
            # Gumbel softmax for differentiable discrete decisions
            routing_weights = F.gumbel_softmax(routing_logits, tau=self.temperature, hard=False)
        else:
            # Hard routing for inference
            routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)
        
        # Compute statistics for monitoring
        stats = {
            'full_prob': routing_weights[:, 0].mean().item(),
            'bypass_prob': routing_weights[:, 1].mean().item(),
            'entropy': -(routing_weights * (routing_weights + 1e-8).log()).sum(dim=-1).mean().item()
        }
        
        return routing_weights, stats


class AdaptiveBypass(nn.Module):
    """Lightweight bypass transformation"""
    
    def __init__(self, hidden_size: int, use_bias: bool = False):
        super().__init__()
        # Simple linear transformation
        self.bypass_linear = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Initialize as near-identity
        with torch.no_grad():
            self.bypass_linear.weight.data = torch.eye(hidden_size) + 0.01 * torch.randn(hidden_size, hidden_size)
            if use_bias:
                self.bypass_linear.bias.data.zero_()
        
        print(f"[DEBUG] Initialized AdaptiveBypass with hidden_size={hidden_size}, use_bias={use_bias}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply lightweight transformation"""
        return hidden_states + 0.1 * self.bypass_linear(hidden_states)  # Residual connection


class AdaptiveTransformerLayer(nn.Module):
    """Transformer layer with adaptive routing between full and bypass paths"""
    
    def __init__(self, original_layer: nn.Module, hidden_size: int, 
                 router_temp: float = 1.0, use_bypass_bias: bool = False):
        super().__init__()
        self.original_layer = original_layer
        self.router = AdaptiveRouter(hidden_size, temperature=router_temp)
        self.bypass = AdaptiveBypass(hidden_size, use_bias=use_bypass_bias)
        
        # Complexity scoring parameters
        self.complexity_alpha = nn.Parameter(torch.tensor(0.5))
        
        print(f"[DEBUG] Created AdaptiveTransformerLayer wrapper")
    
    def compute_complexity_score(self, hidden_states: torch.Tensor, 
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute input complexity score for adaptive routing threshold"""
        # Token entropy (diversity of representations)
        token_std = hidden_states.std(dim=1).mean(dim=-1)  # [batch_size]
        
        # Attention dispersion (if mask available)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).float()
            dispersion = seq_lengths / hidden_states.size(1)  # Ratio of actual tokens
        else:
            dispersion = torch.ones_like(token_std)
        
        # Combined complexity score
        complexity = token_std * dispersion
        return complexity
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with adaptive routing"""
        
        # Get routing decision
        routing_weights, routing_stats = self.router(hidden_states, training=self.training)
        
        # Compute complexity-adjusted routing
        complexity = self.compute_complexity_score(hidden_states, attention_mask)
        complexity_factor = 1 + self.complexity_alpha * complexity.unsqueeze(-1)
        adjusted_weights = routing_weights * complexity_factor
        adjusted_weights = F.softmax(adjusted_weights, dim=-1)
        
        # Debug logging
        if torch.rand(1).item() < 0.01:  # Log 1% of the time to avoid spam
            print(f"[DEBUG] Routing stats: full={routing_stats['full_prob']:.3f}, "
                  f"bypass={routing_stats['bypass_prob']:.3f}, "
                  f"complexity={complexity.mean().item():.3f}")
        
        # Execute dual paths
        full_output = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Handle different output formats
        if isinstance(full_output, tuple):
            full_hidden = full_output[0]
            other_outputs = full_output[1:]
        else:
            full_hidden = full_output
            other_outputs = ()
        
        bypass_output = self.bypass(hidden_states)
        
        # Weighted combination based on routing
        gate = adjusted_weights[:, 0:1].unsqueeze(-1)  # [B, 1, 1] for broadcasting
        combined_hidden = gate * full_hidden + (1 - gate) * bypass_output
        
        # Return in same format as original
        if other_outputs:
            return (combined_hidden,) + other_outputs
        else:
            return combined_hidden


def convert_to_adaptive_model(
    model: nn.Module,
    layer_indices: List[int],
    hidden_size: int,
    router_temp: float = 1.0,
    use_bypass_bias: bool = False
) -> nn.Module:
    """Convert specified layers to adaptive routing layers"""
    
    print(f"\n[DEBUG] Converting {len(layer_indices)} layers to adaptive routing")
    print(f"[DEBUG] Layer indices to convert: {layer_indices}")
    
    # Get the layers container (handle different model architectures)
    if hasattr(model, 'transformer'):  # GPT-style
        layers = model.transformer.h
        layer_attr = 'h'
        parent = model.transformer
    elif hasattr(model, 'model'):  # LLaMA-style
        layers = model.model.layers
        layer_attr = 'layers'
        parent = model.model
    else:
        raise ValueError("Unknown model architecture")
    
    # Replace specified layers
    new_layers = []
    for i, layer in enumerate(layers):
        if i in layer_indices:
            print(f"[DEBUG] Converting layer {i} to adaptive")
            adaptive_layer = AdaptiveTransformerLayer(
                layer, hidden_size, router_temp, use_bypass_bias
            )
            new_layers.append(adaptive_layer)
        else:
            new_layers.append(layer)
    
    # Set the new layers
    setattr(parent, layer_attr, nn.ModuleList(new_layers))
    
    return model


def train_adaptive_routing(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer,
    max_length: int,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    sparsity_lambda: float = 0.01,
    sparsity_increase_rate: float = 1.5
) -> nn.Module:
    """Train the adaptive routing components"""
    
    print(f"\n[INFO] Starting adaptive routing training")
    print(f"[INFO] Epochs: {num_epochs}, LR: {learning_rate}, Sparsity λ: {sparsity_lambda}")
    
    device = next(model.parameters()).device
    
    # Collect router and bypass parameters
    router_params = []
    bypass_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveRouter):
            router_params.extend(module.parameters())
        elif isinstance(module, AdaptiveBypass):
            bypass_params.extend(module.parameters())
    
    print(f"[DEBUG] Found {len(router_params)} router params, {len(bypass_params)} bypass params")
    
    # Three-stage training
    all_params = router_params + bypass_params
    
    # Stage 1: Router pretraining (1 epoch)
    print("\n[Stage 1] Router Pretraining")
    optimizer = torch.optim.Adam(router_params, lr=learning_rate * 2)
    
    for epoch in range(1):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/1", colour="blue")
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**inputs)
            
            # Simple language modeling loss
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 50 == 0:
                print(f"[DEBUG] Batch {batch_idx}: Loss={loss.item():.4f}")
    
    # Stage 2: Joint training (main epochs)
    print("\n[Stage 2] Joint Training")
    optimizer = torch.optim.Adam(all_params, lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_sparsity = 0
        
        # Curriculum learning for sparsity
        current_lambda = sparsity_lambda * (sparsity_increase_rate ** epoch)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", colour="green")
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            optimizer.zero_grad()
            
            # Forward pass and collect routing decisions
            routing_probs = []
            
            def hook_fn(module, input, output):
                if isinstance(module, AdaptiveRouter):
                    probs, _ = output
                    routing_probs.append(probs[:, 1])  # Bypass probability
            
            hooks = []
            for module in model.modules():
                if isinstance(module, AdaptiveRouter):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            outputs = model(**inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Compute losses
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Sparsity loss (encourage bypass usage)
            if routing_probs:
                sparsity_loss = current_lambda * torch.stack(routing_probs).mean()
            else:
                sparsity_loss = 0
            
            total_loss_val = task_loss + sparsity_loss
            total_loss_val.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            optimizer.step()
            
            total_loss += task_loss.item()
            if isinstance(sparsity_loss, torch.Tensor):
                total_sparsity += sparsity_loss.item()
            
            progress_bar.set_postfix({
                'task_loss': f'{task_loss.item():.4f}',
                'sparse_loss': f'{sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else 0:.4f}'
            })
    
    # Stage 3: Bypass refinement (1 epoch)
    print("\n[Stage 3] Bypass Path Refinement")
    optimizer = torch.optim.Adam(bypass_params, lr=learning_rate * 0.5)
    
    for epoch in range(1):
        model.train()
        progress_bar = tqdm(dataloader, desc="Refinement", colour="yellow")
        
        for batch in progress_bar:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    print("[INFO] Training complete!")
    model.eval()
    return model


def adaptive_routing(
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
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    router_temp: float = 1.0,
    use_bypass_bias: bool = False,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    sparsity_lambda: float = 0.01,
    **kwargs
) -> str:
    """Main adaptive routing function"""
    
    print(f"\n{Fore.GREEN}[ADAPTIVE ROUTING] Starting adaptive layer routing{Fore.RESET}")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Dataset: {dataset}, Size: {dataset_size}")
    print(f"[INFO] Layers to process: {layers_to_skip}, Num blocks: {num_A}")
    
    # Load model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        print("[INFO] Using 4-bit quantization")
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
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    
    # Load distance information and select blocks
    print(f"[DEBUG] Loading distances from {distances_path}")
    average_distances = torch.load(distances_path, weights_only=True)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive
    )
    
    # Get layer indices to convert
    layer_indices = []
    for start, end in selected_blocks:
        layer_indices.extend(range(start - 1, end - 1))  # Convert to 0-indexed
    
    print(f"[INFO] Converting layers: {layer_indices}")
    
    # Convert model
    model = convert_to_adaptive_model(
        model,
        layer_indices,
        hidden_size,
        router_temp=router_temp,
        use_bypass_bias=use_bypass_bias
    )
    
    # Prepare training data
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Train adaptive components
    model = train_adaptive_routing(
        model,
        dataloader,
        tokenizer,
        max_length,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        sparsity_lambda=sparsity_lambda
    )
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_adaptive_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    print(f"[INFO] Saving model to {save_path}_AdaptiveRouting")
    model.save_pretrained(f"{save_path}_AdaptiveRouting")
    tokenizer.save_pretrained(f"{save_path}_AdaptiveRouting")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_AdaptiveRouting"