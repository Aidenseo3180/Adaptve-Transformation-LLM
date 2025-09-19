"""Asymmetric Layer Fusion (ALF) - Novel layer compression method

Inspired by FinerCut's discovery that consecutive attention layers can be removed
while multiple FFNs process one attention output.
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
                    select_non_overlapping_blocks)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class FFNBank(nn.Module):
    """Shared FFN bank that replaces multiple FFN layers"""
    
    def __init__(self, 
                 hidden_dim: int,
                 intermediate_dim: int,
                 num_experts: int = 4,
                 dropout: float = 0.1,
                 dtype=torch.bfloat16):
        super().__init__()
        
        print(f"[DEBUG] Initializing FFN Bank:")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - intermediate_dim: {intermediate_dim}")
        print(f"  - num_experts: {num_experts}")
        
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Expert FFNs (like MoE but at component level)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim, dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim, dtype=dtype)
            ) for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, num_experts, dtype=dtype)
        )
        
        print(f"[DEBUG] FFN Bank initialized with {num_experts} experts")
    
    def forward(self, x: torch.Tensor, return_router_probs: bool = False):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute router probabilities
        # Use pooled representation for routing decision
        x_pooled = x.mean(dim=1)  # [batch, hidden]
        router_logits = self.router(x_pooled)  # [batch, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch, seq, hidden]
            expert_outputs.append(expert_out)
        
        # Stack and weight by router probabilities
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, batch, seq, hidden]
        
        # Reshape router probs for broadcasting
        router_probs_expanded = router_probs.unsqueeze(1).unsqueeze(3)  # [batch, 1, num_experts, 1]
        router_probs_expanded = router_probs_expanded.permute(2, 0, 1, 3)  # [num_experts, batch, 1, 1]
        
        # Weighted sum
        output = (expert_outputs * router_probs_expanded).sum(dim=0)
        
        if return_router_probs:
            return output, router_probs
        return output


class AsymmetricLayerFusion(nn.Module):
    """ALF module that separates attention preservation from FFN compression"""
    
    def __init__(self,
                 hidden_dim: int,
                 intermediate_dim: int,
                 num_blocks_compressed: int,
                 num_experts: int = 4,
                 use_layer_adapters: bool = True,
                 dtype=torch.bfloat16):
        super().__init__()
        
        print(f"[DEBUG] Initializing AsymmetricLayerFusion:")
        print(f"  - num_blocks_compressed: {num_blocks_compressed}")
        print(f"  - use_layer_adapters: {use_layer_adapters}")
        
        self.num_blocks = num_blocks_compressed
        self.hidden_dim = hidden_dim
        
        # Shared FFN bank
        self.ffn_bank = FFNBank(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            dtype=dtype
        )
        
        # Layer-specific lightweight adapters
        if use_layer_adapters:
            self.layer_adapters = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
                for _ in range(num_blocks_compressed)
            ])
            
            # Initialize adapters near identity
            for adapter in self.layer_adapters:
                with torch.no_grad():
                    adapter.weight.copy_(torch.eye(hidden_dim, dtype=dtype) * 0.1)
        else:
            self.layer_adapters = None
        
        # Progressive blending weights
        self.blend_weights = nn.Parameter(
            torch.ones(num_blocks_compressed, dtype=dtype) / num_blocks_compressed
        )
        
        # Skip connection gate
        self.skip_gate = nn.Parameter(torch.tensor(0.8, dtype=dtype))
        
        print(f"[DEBUG] ALF initialization complete")
    
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None):
        """
        Forward pass through ALF module
        Args:
            x: Input tensor [batch, seq, hidden]
            layer_idx: Which virtual layer is being processed (for adapters)
        """
        identity = x
        
        # Process through FFN bank
        ffn_out = self.ffn_bank(x)
        
        # Apply layer-specific adaptation if specified
        if self.layer_adapters is not None and layer_idx is not None:
            if 0 <= layer_idx < len(self.layer_adapters):
                adapter = self.layer_adapters[layer_idx]
                adapted = adapter(x)
                ffn_out = ffn_out + adapted * self.blend_weights[layer_idx]
        
        # Gated skip connection
        output = self.skip_gate * ffn_out + (1 - self.skip_gate) * identity
        
        return output


def analyze_layer_components(model, tokenizer, dataloader, start_id, end_id, device):
    """Analyze attention vs FFN importance in target layers"""
    
    print(f"[DEBUG] Analyzing layer components {start_id} to {end_id}")
    
    attention_impacts = []
    ffn_impacts = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_text in enumerate(tqdm(dataloader, desc="Analyzing components")):
            if batch_idx >= 5:  # Sample a few batches
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
            
            for layer_idx in range(start_id, end_id):
                # Measure change from attention
                pre_attn = hidden_states[layer_idx]
                post_attn = hidden_states[layer_idx + 1]  # After attn+ffn
                
                # Approximate impact (simplified - in practice would need hooks)
                attn_change = (post_attn - pre_attn).abs().mean().item()
                attention_impacts.append(attn_change)
    
    avg_attn_impact = sum(attention_impacts) / len(attention_impacts) if attention_impacts else 0
    print(f"[DEBUG] Average attention impact: {avg_attn_impact:.4f}")
    
    return avg_attn_impact


def alf_compress(
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
    # ALF specific parameters
    num_experts: int = 4,
    use_layer_adapters: bool = True,
    training_epochs: int = 10,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    **kwargs
) -> str:
    """Asymmetric Layer Fusion compression"""
    
    print(f"\n{'='*60}")
    print(f"Starting Asymmetric Layer Fusion (ALF)")
    print(f"Compressing layers {start_id} to {end_id}")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logging.info(f"{Fore.GREEN}Loading model for ALF compression...{Fore.RESET}")
    
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
    
    # Analyze components (optional)
    avg_impact = analyze_layer_components(
        model, tokenizer, dataloader, 
        start_id - num_layer, end_id - num_layer, device
    )
    
    # Initialize ALF module
    logging.info(f"{Fore.YELLOW}Creating ALF module...{Fore.RESET}")
    
    alf_module = AsymmetricLayerFusion(
        hidden_dim=hidden_size,
        intermediate_dim=intermediate_size,
        num_blocks_compressed=num_blocks_to_compress,
        num_experts=num_experts,
        use_layer_adapters=use_layer_adapters,
        dtype=torch.bfloat16
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(alf_module.parameters(), lr=learning_rate)
    
    # Cosine annealing with warmup
    total_steps = training_epochs * min(50, len(dataloader))  # Limit steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps
    )
    
    # Training loop
    logging.info(f"{Fore.CYAN}Starting ALF training...{Fore.RESET}")
    
    model.eval()
    best_loss = float('inf')
    
    for epoch in range(training_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data each epoch
        dataloader_list = list(dataloader)
        import random
        random.shuffle(dataloader_list)
        
        with tqdm(dataloader_list[:50], desc=f"Epoch {epoch+1}/{training_epochs}") as pbar:
            for batch_text in pbar:
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get teacher outputs
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                
                # Get input and target
                block_input = hidden_states[start_id - num_layer].detach()
                block_target = hidden_states[end_id - num_layer].detach()
                
                # Forward through ALF (simulate multiple layers)
                current = block_input
                for virtual_layer in range(num_blocks_to_compress):
                    current = alf_module(current, layer_idx=virtual_layer)
                
                # Compute loss
                mse_loss = F.mse_loss(current, block_target)
                
                # Cosine similarity loss
                cos_sim = F.cosine_similarity(
                    current.reshape(-1, hidden_size),
                    block_target.reshape(-1, hidden_size),
                    dim=-1
                ).mean()
                cos_loss = 1 - cos_sim
                
                # Router diversity loss (encourage using different experts)
                _, router_probs = alf_module.ffn_bank(block_input, return_router_probs=True)
                entropy = -(router_probs * router_probs.log()).sum(dim=-1).mean()
                diversity_loss = -entropy * 0.01  # Small weight
                
                # Total loss
                loss = 0.7 * mse_loss + 0.3 * cos_loss + diversity_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(alf_module.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MSE': f'{mse_loss.item():.4f}',
                    'Cos': f'{cos_loss.item():.4f}'
                })
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logging.info(f"{Fore.GREEN}Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}{Fore.RESET}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = alf_module.state_dict().copy()
    
    # Restore best weights
    alf_module.load_state_dict(best_state)
    print(f"[DEBUG] Training complete. Best loss: {best_loss:.4f}")
    
    # Print learned parameters
    print(f"[DEBUG] Final parameters:")
    print(f"  - Skip gate: {alf_module.skip_gate.item():.4f}")
    print(f"  - Blend weights: {alf_module.blend_weights.data}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for modification
    logging.info(f"{Fore.YELLOW}Integrating ALF module into model...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Integration strategy: Replace FFN layers with ALF module
    print(f"[DEBUG] Replacing layers {start_id - num_layer} to {end_id - num_layer}")
    
    # Custom layer wrapper
    class ALFIntegratedLayer(nn.Module):
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
            # Self-attention
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
            
            # FFN replaced by ALF
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
    
    # Replace first layer with ALF-integrated version
    model.model.layers[start_id - num_layer - 1] = ALFIntegratedLayer(
        model.model.layers[start_id - num_layer - 1],
        alf_module.cpu(),
        0
    )
    
    # Remove intermediate layers
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_ALF"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}")
    tokenizer.save_pretrained(f"{save_path}")
    
    # Save ALF module separately
    torch.save({
        'alf_state': alf_module.state_dict(),
        'config': {
            'num_experts': num_experts,
            'best_loss': best_loss
        }
    }, f"{save_path}/alf_module.pth")
    
    logging.info(f"{Fore.GREEN}Model saved to {save_path}{Fore.RESET}")
    
    # Cleanup
    del model, alf_module
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path