"""Progressive Layer Distillation with Learnable Shortcuts (PLDS)

This module implements an advanced layer compression technique that goes beyond
simple linear transformations by using compressed meta-blocks with adaptive routing.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Tuple, List
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


class CompressedMetaBlock(nn.Module):
    """Compressed meta-block that replaces multiple transformer layers."""
    
    def __init__(self, hidden_dim: int, num_blocks_compressed: int, 
                 use_attention: bool = True, num_heads_compressed: int = 8,
                 dtype=torch.bfloat16):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks_compressed
        self.use_attention = use_attention
        self.dtype = dtype
        self.num_heads = num_heads_compressed  # Store num_heads
        
        # Path 1: Direct bypass
        self.bypass_scale = nn.Parameter(torch.ones(1, dtype=dtype))
        
        # Path 2: Fast linear transformation
        self.fast_transform = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        with torch.no_grad():
            self.fast_transform.weight.copy_(torch.eye(hidden_dim, dtype=dtype))
        
        # Path 3: Slow deep processing
        if use_attention:
            self.compressed_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads=num_heads_compressed,
                dropout=0.0,
                batch_first=True
            )
            self.compressed_attn = self.compressed_attn.to(dtype=dtype)
        
        self.slow_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, dtype=dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim, dtype=dtype)
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, 256, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3, dtype=dtype),
            nn.Softmax(dim=-1)
        )
        
        self.residual_scales = nn.Parameter(torch.ones(num_blocks_compressed, dtype=dtype))
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        
        self.shortcut_transform = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        with torch.no_grad():
            self.shortcut_transform.weight.copy_(torch.eye(hidden_dim, dtype=dtype))
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through compressed meta-block."""
        # Ensure input is in correct dtype
        x = x.to(self.dtype)
        
        batch_size, seq_len, _ = x.shape
        identity = x.clone()
        
        # Compute gating weights
        x_pooled = x.mean(dim=1)
        gate_weights = self.gate_network(x_pooled)
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(2)
        
        # Path 1: Direct bypass
        path1 = identity * self.bypass_scale
        
        # Path 2: Fast linear transformation
        path2 = self.fast_transform(x)
        
        # Path 3: Slow deep processing
        if self.use_attention:
            # Fix attention mask for MultiheadAttention
            if attention_mask is not None:
                # Convert [batch, seq_len] -> proper attention mask
                # MultiheadAttention expects shape [batch*num_heads, tgt_seq, src_seq] OR [batch, tgt_seq, src_seq]
                # We'll use the simpler [seq_len, seq_len] broadcast format
                
                # Create causal mask if needed
                attn_mask = torch.zeros(seq_len, seq_len, dtype=self.dtype, device=x.device)
                
                # Apply padding mask if provided
                if attention_mask is not None:
                    # attention_mask is [batch, seq_len] with 1s for valid positions, 0s for padding
                    # We need to convert this to an additive mask
                    padding_mask = (1.0 - attention_mask.float()) * -10000.0
                    # For simplicity, we'll just use key_padding_mask instead
                    key_padding_mask = (attention_mask == 0)  # True for positions to ignore
                else:
                    key_padding_mask = None
                
                # Apply attention without attn_mask, only key_padding_mask
                attn_out, _ = self.compressed_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=False
                )
            else:
                # No mask needed
                attn_out, _ = self.compressed_attn(
                    x, x, x,
                    need_weights=False
                )
            
            path3 = self.slow_ffn(self.layer_norm(attn_out))
        else:
            path3 = self.slow_ffn(self.layer_norm(x))
        
        # Weighted combination of paths
        output = (gate_weights[..., 0] * path1 + 
                 gate_weights[..., 1] * path2 + 
                 gate_weights[..., 2] * path3)
        
        # Apply progressive residual connections
        for i in range(self.num_blocks):
            scale = self.residual_scales[i]
            output = output + identity * scale * (1.0 / self.num_blocks)
        
        # Final shortcut connection
        output = output + self.shortcut_transform(identity)
        
        return output


def plds_compress(
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
    # PLDS specific parameters
    use_attention_in_meta: bool = True,
    num_compressed_heads: int = 8,
    distillation_epochs: int = 10,
    learning_rate: float = 1e-4,
    temperature: float = 4.0,
    alpha_mse: float = 0.5,
    alpha_cos: float = 0.3,
    alpha_attn: float = 0.2,
    **kwargs
) -> str:
    """Progressive Layer Distillation with Learnable Shortcuts compression.
    
    This method compresses multiple transformer blocks into a single meta-block
    with multiple processing paths and adaptive routing.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model for analysis
    logging.info(f"{Fore.GREEN}Loading model for PLDS compression...{Fore.RESET}")
    
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
        output_attentions=True,  # Need attention for distillation
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    num_blocks_to_compress = end_id - start_id
    
    # Create calibration dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Initialize compressed meta-block with correct dtype
    logging.info(f"{Fore.YELLOW}Creating compressed meta-block for layers {start_id} to {end_id}{Fore.RESET}")
    
    meta_block = CompressedMetaBlock(
        hidden_dim=hidden_size,
        num_blocks_compressed=num_blocks_to_compress,
        use_attention=use_attention_in_meta,
        num_heads_compressed=num_compressed_heads,
        dtype=torch.bfloat16  # Explicitly set dtype
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(meta_block.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=distillation_epochs
    )
    
    # Training loop - Progressive Distillation
    logging.info(f"{Fore.CYAN}Starting progressive distillation...{Fore.RESET}")
    
    model.eval()
    for epoch in range(distillation_epochs):
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{distillation_epochs}", colour="blue") as pbar:
            for batch_text in pbar:
                # Tokenize input
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get original model outputs with teacher forcing
                with torch.no_grad():
                    teacher_outputs = model(**inputs)
                    hidden_states = teacher_outputs.hidden_states
                    attentions = teacher_outputs.attentions if hasattr(teacher_outputs, 'attentions') else None
                
                # Extract inputs and targets for compressed block
                # Convert to bfloat16 explicitly
                block_input = hidden_states[start_id - num_layer].detach().to(torch.bfloat16)
                block_target = hidden_states[end_id - num_layer].detach().to(torch.bfloat16)
                
                # Forward through meta-block
                optimizer.zero_grad()
                meta_output = meta_block(
                    block_input,
                    attention_mask=inputs.get('attention_mask', None)
                )
                
                # Compute multi-objective loss
                loss = 0
                
                # 1. MSE Loss (direct output matching)
                mse_loss = F.mse_loss(meta_output, block_target)
                loss += alpha_mse * mse_loss
                
                # 2. Cosine similarity loss (directional matching)
                meta_norm = F.normalize(meta_output, p=2, dim=-1)
                target_norm = F.normalize(block_target, p=2, dim=-1)
                cos_loss = 1 - (meta_norm * target_norm).sum(dim=-1).mean()
                loss += alpha_cos * cos_loss
                
                # 3. Attention transfer loss (if applicable)
                # Skip for now as it requires more complex implementation
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_block.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MSE': f'{mse_loss.item():.4f}',
                    'Cos': f'{cos_loss.item():.4f}'
                })
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        logging.info(f"{Fore.GREEN}Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}{Fore.RESET}")
    
    # Clean up teacher model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model again for modification
    logging.info(f"{Fore.YELLOW}Loading model for modification...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Insert compressed meta-block
    # Create a more compatible wrapper
    class MetaBlockWrapper(nn.Module):
        def __init__(self, meta_block, original_layer):
            super().__init__()
            self.meta_block = meta_block
            # Keep original layer attributes for compatibility
            if hasattr(original_layer, 'self_attn'):
                self.self_attn = original_layer.self_attn
            if hasattr(original_layer, 'mlp'):
                self.mlp = original_layer.mlp
            if hasattr(original_layer, 'input_layernorm'):
                self.input_layernorm = original_layer.input_layernorm
            if hasattr(original_layer, 'post_attention_layernorm'):
                self.post_attention_layernorm = original_layer.post_attention_layernorm
            
        def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                   past_key_value=None, output_attentions=False, use_cache=False, 
                   cache_position=None, **kwargs):
            # Apply meta-block transformation
            hidden_states = self.meta_block(hidden_states, attention_mask=attention_mask)
            
            # Return in expected format (tuple)
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)  # No attention weights
            if use_cache:
                outputs += (None,)  # No cache
            
            return outputs
    
    # Replace the layer before the truncated section with our wrapper
    wrapper = MetaBlockWrapper(meta_block.cpu(), model.model.layers[start_id - num_layer - 1])
    model.model.layers[start_id - num_layer - 1] = wrapper
    
    # Save the compressed model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_PLDS"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}")
    tokenizer.save_pretrained(f"{save_path}")
    
    # Save meta-block separately for analysis
    torch.save(meta_block.state_dict(), f"{save_path}/meta_block.pth")
    
    logging.info(f"{Fore.GREEN}Model compressed and saved to {save_path}{Fore.RESET}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path