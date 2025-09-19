"""Simplified PLDS with Direct Integration

A practical approach that combines PLDS learning with ReplaceMe-style integration.
"""

import gc
import logging
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class SimplifiedMetaBlock(nn.Module):
    """Simplified meta-block focusing on learnable linear transformation."""
    
    def __init__(self, hidden_dim: int, num_blocks_compressed: int, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks_compressed
        self.dtype = dtype
        
        # Primary transformation (like ReplaceMe but learnable)
        self.main_transform = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
        nn.init.eye_(self.main_transform.weight)
        
        # Small correction network (minimal parameters)
        compression_dim = min(128, hidden_dim // 16)  # Very small bottleneck
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim, compression_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(compression_dim, hidden_dim, dtype=dtype)
        )
        
        # Initialize correction to near-zero to start close to identity
        with torch.no_grad():
            for layer in self.correction:
                if isinstance(layer, nn.Linear):
                    layer.weight.data *= 0.01
        
        # Learnable mixing weight (start with mostly main transform)
        self.mix_weight = nn.Parameter(torch.tensor(0.95, dtype=dtype))
        
        # Residual scaling for simulating multiple layers
        self.residual_scale = nn.Parameter(torch.ones(1, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with focus on extractable linear transformation."""
        x = x.to(self.dtype)
        
        # Main linear path
        main_out = self.main_transform(x)
        
        # Small correction
        correction_out = self.correction(x)
        
        # Mix with learnable weight
        output = self.mix_weight * main_out + (1 - self.mix_weight) * correction_out
        
        # Add scaled residual (simulates skip connections across removed layers)
        output = output + x * self.residual_scale * self.num_blocks
        
        return output
    
    def get_effective_transform(self) -> torch.Tensor:
        """Extract the effective linear transformation matrix."""
        with torch.no_grad():
            # For a linear transform, we can probe with identity
            eye = torch.eye(self.hidden_dim, dtype=self.dtype, device=self.main_transform.weight.device)
            
            # Get the effective transformation
            # This captures the main transform + average correction effect
            effective = self.forward(eye)
            
            return effective.T  # Transpose for weight matrix format


def simplified_plds_compress(
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
    # Simplified PLDS parameters
    learning_epochs: int = 5,  # Fewer epochs needed
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    **kwargs
) -> str:
    """Simplified PLDS with direct integration into model weights."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logging.info(f"{Fore.GREEN}Loading model for simplified PLDS...{Fore.RESET}")
    
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
    
    # Create dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Initialize simplified meta-block
    logging.info(f"{Fore.YELLOW}Creating simplified meta-block for layers {start_id}-{end_id}{Fore.RESET}")
    
    meta_block = SimplifiedMetaBlock(
        hidden_dim=hidden_size,
        num_blocks_compressed=num_blocks_to_compress,
        dtype=torch.bfloat16
    ).to(device)
    
    # Setup optimizer with warmup
    optimizer = torch.optim.AdamW(meta_block.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(dataloader) * learning_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    logging.info(f"{Fore.CYAN}Training simplified meta-block...{Fore.RESET}")
    
    model.eval()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(learning_epochs):
        epoch_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{learning_epochs}", colour="blue") as pbar:
            for batch_text in pbar:
                # Prepare inputs
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get original model states
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.hidden_states
                    
                    # Input to compressed section
                    block_input = hidden_states[start_id - num_layer].detach()
                    
                    # Target output after compressed section
                    block_target = hidden_states[end_id - num_layer].detach()
                    
                    # Also get intermediate states for auxiliary loss
                    if num_blocks_to_compress > 2:
                        mid_point = (start_id + end_id) // 2 - num_layer
                        block_middle = hidden_states[mid_point].detach()
                
                # Forward through meta-block
                optimizer.zero_grad()
                meta_output = meta_block(block_input)
                
                # Multi-scale loss
                # 1. Final output matching
                final_loss = F.mse_loss(meta_output, block_target)
                
                # 2. Cosine similarity for direction
                cos_sim = F.cosine_similarity(
                    meta_output.view(-1, hidden_size),
                    block_target.view(-1, hidden_size),
                    dim=1
                ).mean()
                cos_loss = 1 - cos_sim
                
                # 3. Intermediate supervision if available
                if num_blocks_to_compress > 2 and 'block_middle' in locals():
                    # Meta-block should partially transform towards middle
                    partial_output = meta_block.main_transform(block_input) * 0.5 + block_input * 0.5
                    mid_loss = F.mse_loss(partial_output, block_middle) * 0.3
                else:
                    mid_loss = 0
                
                # Combined loss
                loss = final_loss + 0.2 * cos_loss + mid_loss
                
                # Regularization: keep transform close to achievable linear transform
                if global_step > warmup_steps:
                    # Penalize excessive correction network activation
                    correction_penalty = (1 - meta_block.mix_weight).abs() * 0.1
                    loss = loss + correction_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_block.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Cos': f'{cos_sim.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
        
        avg_loss = epoch_loss / num_batches
        logging.info(f"{Fore.GREEN}Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}{Fore.RESET}")
    
    # Extract effective transformation
    logging.info(f"{Fore.YELLOW}Extracting effective transformation...{Fore.RESET}")
    
    with torch.no_grad():
        effective_transform = meta_block.get_effective_transform()
        
        # Optionally save the transform for analysis
        transform_stats = {
            'singular_values': torch.svd(effective_transform).S.cpu(),
            'condition_number': torch.linalg.cond(effective_transform).item(),
            'frobenius_norm': torch.norm(effective_transform, 'fro').item(),
            'mix_weight': meta_block.mix_weight.item(),
            'residual_scale': meta_block.residual_scale.item()
        }
        
        logging.info(f"Transform condition number: {transform_stats['condition_number']:.2f}")
        logging.info(f"Mix weight (main vs correction): {transform_stats['mix_weight']:.3f}")
    
    # Clean up teacher model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model for modification
    logging.info(f"{Fore.YELLOW}Applying transformation to model...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Apply transformation ReplaceMe-style
    prev_layer_idx = start_id - num_layer - 1
    prev_layer = model.model.layers[prev_layer_idx]
    
    # Get original down_proj weight
    original_weight = prev_layer.mlp.down_proj.weight.data.to(torch.float32)
    
    # Apply effective transformation
    new_weight = (effective_transform.T.cpu().to(torch.float32) @ original_weight).to(torch.bfloat16)
    
    # Update weight
    prev_layer.mlp.down_proj.weight.data = new_weight
    
    # Truncate model (remove compressed layers)
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_SimplifiedPLDS"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}")
    tokenizer.save_pretrained(f"{save_path}")
    
    # Save transform statistics
    torch.save(transform_stats, f"{save_path}/transform_stats.pt")
    
    logging.info(f"{Fore.GREEN}Model compressed and saved to {save_path}{Fore.RESET}")
    
    # Cleanup
    del model, meta_block
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path