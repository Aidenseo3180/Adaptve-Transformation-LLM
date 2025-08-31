import argparse
import gc
import logging
import os
from typing import Optional
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class StreamingActivationDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that processes activations on-the-fly"""
    def __init__(self, dataloader, model, tokenizer, max_length, start_id, end_id, num_layer):
        self.dataloader = dataloader
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_id = start_id
        self.end_id = end_id
        self.num_layer = num_layer
        self.device = next(model.parameters()).device
        
        # Calculate total length
        self.total_length = 0
        for batch in dataloader:
            inputs = tokenizer(
                batch, return_tensors="pt", padding="longest", 
                max_length=max_length, truncation=True
            )
            self.total_length += inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1]
        
        print(f"{Fore.GREEN}[STREAMING] Dataset initialized with {self.total_length} total tokens{Fore.RESET}")
    
    def __len__(self):
        return self.total_length
    
    def process_batch_streaming(self, batch_idx, batch_size=256):
        """Process a specific batch and return the activations"""
        batch_data = []
        current_idx = 0
        target_start = batch_idx * batch_size
        target_end = (batch_idx + 1) * batch_size
        
        # Set up hooks for activation collection
        mlp_activations = {}
        def save_mlp_activation(name):
            def hook(module, input, output):
                mlp_activations[name] = output.detach()
            return hook

        hooks = []
        if 'falcon' in str(self.model).lower():
            for i, layer in enumerate(self.model.transformer.h):
                hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
        else:
            for i, layer in enumerate(self.model.model.layers):
                hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
        
        try:
            for batch in self.dataloader:
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding="longest",
                    max_length=self.max_length, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Extract activations
                hidden_states = outputs.hidden_states[1:]
                hidden_states_mlp_list = [
                    mlp_activations[f'layer_{i}_mlp'] for i in range(self.model.config.num_hidden_layers)
                ]
                
                # Get relevant activations
                hidden_states_mlp = hidden_states_mlp_list[self.start_id - self.num_layer - 1]
                hidden_states_i = hidden_states[self.start_id - self.num_layer - 1]
                hidden_states_n = hidden_states[self.end_id - self.num_layer - 1]
                
                # Flatten and convert to bfloat16 for memory efficiency
                batch_tokens = hidden_states_mlp.shape[0] * hidden_states_mlp.shape[1]
                hidden_size = hidden_states_mlp.shape[-1]
                
                mlp_flat = hidden_states_mlp.view(-1, hidden_size).to(torch.bfloat16)
                i_flat = hidden_states_i.view(-1, hidden_size).to(torch.bfloat16)
                n_flat = hidden_states_n.view(-1, hidden_size).to(torch.bfloat16)
                
                # Calculate what we need
                a1_batch = mlp_flat
                a2_batch = n_flat + mlp_flat - i_flat
                attention_weights = torch.abs(i_flat - mlp_flat)
                
                # Extract the specific batch we need
                batch_end = current_idx + batch_tokens
                if target_start < batch_end and current_idx < target_end:
                    # Calculate overlap
                    start_in_batch = max(0, target_start - current_idx)
                    end_in_batch = min(batch_tokens, target_end - current_idx)
                    
                    batch_data.append({
                        'a1': a1_batch[start_in_batch:end_in_batch].cpu(),
                        'a2': a2_batch[start_in_batch:end_in_batch].cpu(),
                        'attention': attention_weights[start_in_batch:end_in_batch].cpu()
                    })
                
                current_idx = batch_end
                
                # Clear GPU memory immediately
                del hidden_states_mlp, hidden_states_i, hidden_states_n
                del mlp_flat, i_flat, n_flat, a1_batch, a2_batch, attention_weights
                torch.cuda.empty_cache()
                
                if current_idx >= target_end:
                    break
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Combine batch data
        if batch_data:
            combined = {
                'a1': torch.cat([d['a1'] for d in batch_data], dim=0),
                'a2': torch.cat([d['a2'] for d in batch_data], dim=0),
                'attention': torch.cat([d['attention'] for d in batch_data], dim=0)
            }
            return combined
        return None


def residual_linear_approximation_method(
    streaming_dataset,
    loss: str = "cosine",
    num_residual_layers: int = 3,
    device: str = "cuda"
) -> list:
    """
    Memory-efficient residual linear approximation learning
    """
    print(f"{Fore.GREEN}[DEBUG] Starting memory-efficient residual learning with {num_residual_layers} layers{Fore.RESET}")
    
    transformations = []
    batch_size = 256  # Reduced batch size for memory efficiency
    num_batches = (streaming_dataset.total_length + batch_size - 1) // batch_size
    
    # Get a sample to determine hidden size
    sample_batch = streaming_dataset.process_batch_streaming(0, batch_size=min(100, streaming_dataset.total_length))
    if sample_batch is None:
        raise ValueError("Could not get sample batch")
    
    hidden_size = sample_batch['a1'].shape[1]
    print(f"{Fore.GREEN}[DEBUG] Hidden size: {hidden_size}, Total batches: {num_batches}{Fore.RESET}")
    
    # Initialize running estimates for current_output
    current_output_estimate = None
    
    for layer_idx in range(num_residual_layers):
        print(f"{Fore.BLUE}[DEBUG] Training residual layer {layer_idx + 1}/{num_residual_layers}{Fore.RESET}")
        
        # Initialize transformation matrix with better initialization
        transform = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        torch.nn.init.normal_(transform.weight, mean=0.0, std=0.02)
        
        # Use smaller learning rate and add weight decay
        optimizer = torch.optim.Adam(transform.parameters(), lr=5e-5, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        
        # Training loop with streaming
        for epoch in range(6):  # Reduced epochs to save time
            epoch_loss = 0.0
            valid_batches = 0
            
            for batch_idx in range(0, num_batches, 4):  # Process every 4th batch to save memory
                # Get batch data
                batch_data = streaming_dataset.process_batch_streaming(batch_idx, batch_size)
                if batch_data is None:
                    continue
                
                # Convert to float32 only when needed for computation
                a1_batch = batch_data['a1'].to(device, dtype=torch.float32)
                a2_batch = batch_data['a2'].to(device, dtype=torch.float32)
                attention_batch = batch_data['attention'].to(device, dtype=torch.float32)
                
                # Estimate current output if this is the first layer
                if current_output_estimate is None:
                    current_output_estimate = a1_batch.clone()
                else:
                    # Use a running average for memory efficiency
                    current_output_estimate = 0.9 * current_output_estimate + 0.1 * a1_batch
                
                # Calculate residual target
                residual_target = a2_batch - current_output_estimate
                
                # Determine input for this layer
                if layer_idx == 0:
                    input_data = a1_batch
                elif layer_idx == 1:
                    input_data = a1_batch * (attention_batch + 1e-8)  # Add small epsilon
                else:
                    input_data = current_output_estimate
                
                # Check for problematic values
                if torch.isnan(input_data).any() or torch.isnan(residual_target).any():
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                pred_residual = transform(input_data)
                
                # Compute loss
                if loss == "cosine":
                    loss_val = cosine_loss_stable(pred_residual, residual_target)
                else:
                    loss_val = torch.nn.MSELoss()(pred_residual, residual_target)
                
                if torch.isnan(loss_val):
                    continue
                
                # Backward pass with gradient clipping
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(transform.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_loss += loss_val.item()
                valid_batches += 1
                
                # Update current output estimate
                with torch.no_grad():
                    current_output_estimate = current_output_estimate + pred_residual.detach()
                
                # Immediately free GPU memory
                del a1_batch, a2_batch, attention_batch, residual_target, pred_residual
                torch.cuda.empty_cache()
            
            scheduler.step()
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                print(f"{Fore.YELLOW}[DEBUG] Layer {layer_idx+1}, Epoch {epoch}, Loss: {avg_loss:.6f}, Batches: {valid_batches}{Fore.RESET}")
        
        # Save transformation (convert to bfloat16 to save memory)
        transformations.append(transform.weight.data.clone().cpu().to(torch.bfloat16))
        
        # Clean up
        del transform, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"{Fore.GREEN}[DEBUG] Memory-efficient residual learning completed{Fore.RESET}")
    return transformations


def cosine_loss_stable(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically stable cosine distance loss"""
    eps = 1e-8
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_norm = target / (target.norm(dim=1, keepdim=True) + eps)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)
    cosine_sim = torch.clamp(cosine_sim, -1 + eps, 1 - eps)
    return 1 - cosine_sim.mean()


def residual_linear_compress(
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
    loss: str = "cosine",
    solver: str = "adam",
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    num_residual_layers: int = 3,
    save_transform_only: bool = False
) -> str:
    """
    Memory-efficient residual linear approximation compression.
    """
    print(f"{Fore.CYAN}[DEBUG] Starting memory-efficient residual linear compression...{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB{Fore.RESET}")
    
    # Reduce batch size for memory efficiency
    batch_size = min(batch_size, 2)  # Much smaller batch size
    dataset_size = min(dataset_size or 1000, 1000)  # Limit dataset size
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model with memory optimization
    print(f"{Fore.GREEN}[DEBUG] Loading model with memory optimization...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16  # Use bfloat16 for memory efficiency
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"{Fore.GREEN}[DEBUG] Model loaded. GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB{Fore.RESET}")
    
    # Get calibration data
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Create streaming dataset
    streaming_dataset = StreamingActivationDataset(
        dataloader, model, tokenizer, max_length, start_id, end_id, num_layer
    )
    
    print(f"{Fore.BLUE}[DEBUG] Starting residual transformation learning...{Fore.RESET}")
    # Learn residual transformations with streaming
    transformations = residual_linear_approximation_method(
        streaming_dataset,
        loss=loss,
        num_residual_layers=num_residual_layers,
        device=model.device
    )
    
    # Clean up model and reload for modification
    del model, streaming_dataset, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}[DEBUG] Memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB{Fore.RESET}")
    print(f"{Fore.GREEN}[DEBUG] Reloading model for transformation application...{Fore.RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformations (convert back to float for computation)
    print(f"{Fore.BLUE}[DEBUG] Applying {len(transformations)} residual transformations...{Fore.RESET}")
    
    # Combine transformations efficiently
    combined_transform = transformations[0].to(torch.float32)
    for i in range(1, len(transformations)):
        combined_transform = combined_transform + transformations[i].to(torch.float32)
    
    # Apply to model
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (combined_transform.T @ original_weight.to(torch.float32)).to(torch.bfloat16)
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.data = new_weight
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_path = f"{save_path}_ResidualLinear_{loss}_{num_residual_layers}layers"
    print(f"{Fore.GREEN}[DEBUG] Saving model to {final_path}...{Fore.RESET}")
    
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    if save_transform_only:
        torch.save(transformations, f"{final_path}_transforms.pt")
    
    # Final cleanup
    del model, transformations, combined_transform
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}[DEBUG] Residual linear compression completed successfully!{Fore.RESET}")
    print(f"{Fore.GREEN}[DEBUG] Final GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB{Fore.RESET}")
    return final_path

