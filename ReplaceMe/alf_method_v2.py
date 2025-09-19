"""Asymmetric Layer Fusion (ALF) v2 - Improved version with better regularization and simpler structure"""

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
                    select_non_overlapping_blocks)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class SimplifiedFFNBank(nn.Module):
    """Simplified FFN bank with only 2 experts for better stability"""
    
    def __init__(self, 
                 hidden_dim: int,
                 intermediate_dim: int,
                 num_experts: int = 2,  # Reduced from 4
                 dropout: float = 0.05,  # Reduced dropout
                 dtype=torch.bfloat16):
        super().__init__()
        
        print(f"[DEBUG] Initializing Simplified FFN Bank:")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - intermediate_dim: {intermediate_dim}")
        print(f"  - num_experts: {num_experts}")
        
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Only 2 experts with different initialization
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim, dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim, dtype=dtype)
            )
            
            # Different initialization for each expert
            with torch.no_grad():
                # Expert 0: Standard initialization
                if i == 0:
                    for param in expert.parameters():
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param)
                # Expert 1: Smaller initialization (for fine details)
                elif i == 1:
                    for param in expert.parameters():
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param)
                            param.data *= 0.5
            
            self.experts.append(expert)
        
        # Simpler router - just a single linear layer
        self.router = nn.Linear(hidden_dim, num_experts, dtype=dtype)
        
        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        
        print(f"[DEBUG] FFN Bank initialized with {num_experts} diversified experts")
    
    def forward(self, x: torch.Tensor, return_router_info: bool = False):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute router probabilities with temperature
        x_pooled = x.mean(dim=1)  # [batch, hidden]
        router_logits = self.router(x_pooled) / self.temperature  # [batch, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(router_probs) * 0.01
            router_probs = F.softmax(router_logits + noise, dim=-1)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            # Expand router prob for this expert
            weight = router_probs[:, i].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            output = output + weight * expert_out
        
        if return_router_info:
            # Return entropy for diversity loss
            entropy = -(router_probs * (router_probs + 1e-10).log()).sum(dim=-1).mean()
            load_balance = router_probs.mean(dim=0).std()  # Standard deviation across experts
            return output, entropy, load_balance
        
        return output


class ImprovedAsymmetricLayerFusion(nn.Module):
    """Improved ALF with better regularization and stability"""
    
    def __init__(self,
                 hidden_dim: int,
                 intermediate_dim: int,
                 num_blocks_compressed: int,
                 num_experts: int = 2,  # Reduced
                 use_layer_adapters: bool = True,
                 adapter_reduction: int = 32,  # Smaller adapters
                 dtype=torch.bfloat16):
        super().__init__()
        
        print(f"[DEBUG] Initializing Improved ALF:")
        print(f"  - num_blocks_compressed: {num_blocks_compressed}")
        print(f"  - num_experts: {num_experts}")
        print(f"  - adapter_reduction: {adapter_reduction}")
        
        self.num_blocks = num_blocks_compressed
        self.hidden_dim = hidden_dim
        
        # Simplified FFN bank
        self.ffn_bank = SimplifiedFFNBank(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            dtype=dtype
        )
        
        # Lighter weight adapters (bottleneck style)
        if use_layer_adapters:
            adapter_dim = hidden_dim // adapter_reduction
            self.layer_adapters = nn.ModuleList()
            
            for i in range(num_blocks_compressed):
                adapter = nn.Sequential(
                    nn.Linear(hidden_dim, adapter_dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(adapter_dim, hidden_dim, dtype=dtype)
                )
                # Initialize near zero
                with torch.no_grad():
                    adapter[0].weight.data *= 0.01
                    adapter[2].weight.data *= 0.01
                    if adapter[0].bias is not None:
                        adapter[0].bias.data.zero_()
                        adapter[2].bias.data.zero_()
                
                self.layer_adapters.append(adapter)
        else:
            self.layer_adapters = None
        
        # Progressive blending weights (learnable but regularized)
        self.blend_weights = nn.Parameter(
            torch.ones(num_blocks_compressed, dtype=dtype) * 0.1
        )
        
        # Higher skip connection (preserve more original)
        self.skip_gate = nn.Parameter(torch.tensor(0.95, dtype=dtype))
        
        print(f"[DEBUG] Improved ALF initialization complete")
    
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None):
        identity = x
        
        # Process through FFN bank with router info
        if self.training:
            ffn_out, entropy, load_balance = self.ffn_bank(x, return_router_info=True)
            # Store for loss computation
            self.last_entropy = entropy
            self.last_load_balance = load_balance
        else:
            ffn_out = self.ffn_bank(x, return_router_info=False)
        
        # Apply layer-specific adaptation
        if self.layer_adapters is not None and layer_idx is not None:
            if 0 <= layer_idx < len(self.layer_adapters):
                adapter = self.layer_adapters[layer_idx]
                adapted = adapter(x)
                ffn_out = ffn_out + adapted * self.blend_weights[layer_idx]
        
        # High skip ratio to preserve original information
        output = self.skip_gate * identity + (1 - self.skip_gate) * ffn_out
        
        return output


def alf_compress_v2(
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
    # ALF v2 specific parameters
    num_experts: int = 2,  # Reduced from 4
    use_layer_adapters: bool = True,
    adapter_reduction: int = 32,
    training_epochs: int = 15,  # More epochs
    learning_rate: float = 1e-5,  # Lower learning rate
    warmup_ratio: float = 0.1,
    diversity_weight: float = 0.1,  # Increased
    load_balance_weight: float = 0.05,  # New
    adapter_reg_weight: float = 0.01,  # New
    **kwargs
) -> str:
    """Improved Asymmetric Layer Fusion compression"""
    
    print(f"\n{'='*60}")
    print(f"Starting Improved ALF v2")
    print(f"Compressing layers {start_id} to {end_id}")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logging.info(f"{Fore.GREEN}Loading model for ALF v2 compression...{Fore.RESET}")
    
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
    intermediate_size = model.config.intermediate_size
    num_blocks_to_compress = end_id - start_id
    
    # Create dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Initialize improved ALF module
    logging.info(f"{Fore.YELLOW}Creating improved ALF module...{Fore.RESET}")
    
    alf_module = ImprovedAsymmetricLayerFusion(
        hidden_dim=hidden_size,
        intermediate_dim=intermediate_size,
        num_blocks_compressed=num_blocks_to_compress,
        num_experts=num_experts,
        use_layer_adapters=use_layer_adapters,
        adapter_reduction=adapter_reduction,
        dtype=torch.bfloat16
    ).to(device)
    
    # Setup optimizer with different learning rates
    param_groups = [
        {'params': alf_module.ffn_bank.parameters(), 'lr': learning_rate},
        {'params': [alf_module.skip_gate], 'lr': learning_rate * 0.1},  # Slower for skip gate
        {'params': [alf_module.blend_weights], 'lr': learning_rate * 0.5},
    ]
    
    if alf_module.layer_adapters is not None:
        param_groups.append({
            'params': [p for adapter in alf_module.layer_adapters for p in adapter.parameters()],
            'lr': learning_rate * 2  # Faster for adapters
        })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    
    # Scheduler
    total_steps = training_epochs * min(100, len(dataloader))  # More steps per epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy='cos'
    )
    
    # Training loop
    logging.info(f"{Fore.CYAN}Starting ALF v2 training...{Fore.RESET}")
    
    model.eval()
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(training_epochs):
        epoch_losses = {'total': 0, 'mse': 0, 'cos': 0, 'div': 0, 'lb': 0, 'reg': 0}
        num_batches = 0
        
        # Shuffle data
        dataloader_list = list(dataloader)
        import random
        random.shuffle(dataloader_list)
        
        with tqdm(dataloader_list[:100], desc=f"Epoch {epoch+1}/{training_epochs}") as pbar:
            for batch_text in pbar:
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Skip very short sequences
                if inputs['input_ids'].shape[1] < 20:
                    continue
                
                # Get teacher outputs
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                
                # Get input and target
                block_input = hidden_states[start_id - num_layer].detach()
                block_target = hidden_states[end_id - num_layer].detach()
                
                # Forward through ALF (progressive application)
                current = block_input
                for virtual_layer in range(num_blocks_to_compress):
                    current = alf_module(current, layer_idx=virtual_layer)
                
                # MSE loss
                mse_loss = F.mse_loss(current, block_target)
                
                # Cosine similarity loss
                cos_sim = F.cosine_similarity(
                    current.reshape(-1, hidden_size),
                    block_target.reshape(-1, hidden_size),
                    dim=-1
                ).mean()
                cos_loss = 1 - cos_sim
                
                # Diversity loss (encourage using different experts)
                diversity_loss = -alf_module.last_entropy * diversity_weight if hasattr(alf_module, 'last_entropy') else 0
                
                # Load balance loss (encourage equal expert usage)
                lb_loss = alf_module.last_load_balance * load_balance_weight if hasattr(alf_module, 'last_load_balance') else 0
                
                # Adapter regularization
                reg_loss = 0
                if alf_module.layer_adapters is not None:
                    for adapter in alf_module.layer_adapters:
                        # L2 regularization on adapter weights
                        for param in adapter.parameters():
                            reg_loss += param.pow(2).mean() * adapter_reg_weight
                
                # Total loss with balanced weights
                loss = 0.6 * mse_loss + 0.4 * cos_loss + diversity_loss + lb_loss + reg_loss
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"[DEBUG] NaN detected, skipping batch")
                    continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(alf_module.parameters(), 0.5)
                
                optimizer.step()
                scheduler.step()
                
                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['mse'] += mse_loss.item()
                epoch_losses['cos'] += cos_loss.item()
                epoch_losses['div'] += diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
                epoch_losses['lb'] += lb_loss.item() if isinstance(lb_loss, torch.Tensor) else lb_loss
                epoch_losses['reg'] += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
                num_batches += 1
                
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
                f"Div: {avg_losses['div']:.4f}, "
                f"LB: {avg_losses['lb']:.4f}{Fore.RESET}"
            )
            
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                best_state = alf_module.state_dict().copy()
                print(f"[DEBUG] New best loss: {best_loss:.4f}")
    
    # Restore best weights
    alf_module.load_state_dict(best_state)
    print(f"[DEBUG] Training complete. Best loss: {best_loss:.4f}")
    
    # Print final parameters
    print(f"[DEBUG] Final parameters:")
    print(f"  - Skip gate: {alf_module.skip_gate.item():.4f}")
    print(f"  - Blend weights: {alf_module.blend_weights.data}")
    print(f"  - Temperature: {alf_module.ffn_bank.temperature.item():.4f}")
    
    # Clean up and integrate (same as before)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload and integrate
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Custom integration layer
    class ALFv2IntegratedLayer(nn.Module):
        def __init__(self, original_layer, alf_module, layer_offset):
            super().__init__()
            self.self_attn = original_layer.self_attn
            self.input_layernorm = original_layer.input_layernorm
            self.post_attention_layernorm = original_layer.post_attention_layernorm
            self.alf_module = alf_module
            self.layer_offset = layer_offset
            
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                   past_key_value=None, output_attentions=False, use_cache=False,
                   cache_position=None, **kwargs):
            # Self-attention (preserved)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            
            attn_outputs = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position
            )
            
            hidden_states = attn_outputs[0]
            hidden_states = residual + hidden_states
            
            # FFN replaced by ALF v2
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.alf_module(hidden_states, layer_idx=self.layer_offset)
            hidden_states = residual + hidden_states
            
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (attn_outputs[1],)
            if use_cache:
                outputs += (attn_outputs[2] if len(attn_outputs) > 2 else None,)
            
            return outputs
    
    # Replace and truncate
    model.model.layers[start_id - num_layer - 1] = ALFv2IntegratedLayer(
        model.model.layers[start_id - num_layer - 1],
        alf_module.cpu(),
        0
    )
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Save
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_ALFv2"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}")
    tokenizer.save_pretrained(f"{save_path}")
    
    torch.save({
        'alf_state': alf_module.state_dict(),
        'loss_history': loss_history,
        'config': {
            'num_experts': num_experts,
            'best_loss': best_loss
        }
    }, f"{save_path}/alf_v2_info.pth")
    
    logging.info(f"{Fore.GREEN}Model saved to {save_path}{Fore.RESET}")
    
    del model, alf_module
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path