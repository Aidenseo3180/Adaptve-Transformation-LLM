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


class StreamingActivationDataset:
    """Memory-efficient dataset that processes activations on-the-fly"""
    def __init__(self, dataloader, model, tokenizer, max_length, start_id, end_id, num_layer, dataset_size):
        self.dataloader = dataloader
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_id = start_id
        self.end_id = end_id
        self.num_layer = num_layer
        self.device = next(model.parameters()).device
        
        # Estimate total length without processing all data
        self.estimated_total_length = dataset_size * max_length
        
        print(f"{Fore.GREEN}[STREAMING] Dataset initialized with estimated {self.estimated_total_length} total tokens{Fore.RESET}")
    
    def process_batch_streaming_with_progress(self, batch_idx, batch_size=256):
        """Process a specific batch and return the activations with progress tracking"""
        print(f"{Fore.CYAN}[PROGRESS] Processing batch {batch_idx}, target tokens: {batch_size}{Fore.RESET}")
        
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
            print(f"{Fore.BLUE}[MODEL] Detected Falcon architecture{Fore.RESET}")
            for i, layer in enumerate(self.model.transformer.h):
                hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
        else:
            print(f"{Fore.BLUE}[MODEL] Detected standard transformer architecture{Fore.RESET}")
            for i, layer in enumerate(self.model.model.layers):
                hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
        
        print(f"{Fore.YELLOW}[HOOKS] Registered {len(hooks)} activation hooks{Fore.RESET}")
        
        try:
            batch_count = 0
            for batch in tqdm(self.dataloader, desc=f"Processing data batch {batch_idx}", leave=False):
                batch_count += 1
                print(f"{Fore.CYAN}[BATCH] Processing dataloader batch {batch_count}, current_idx: {current_idx}{Fore.RESET}")
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding="longest",
                    max_length=self.max_length, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                print(f"{Fore.YELLOW}[TOKENIZER] Input shape: {inputs['input_ids'].shape}{Fore.RESET}")
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                print(f"{Fore.GREEN}[MODEL] Forward pass completed{Fore.RESET}")
                
                # Extract activations
                hidden_states = outputs.hidden_states[1:]
                hidden_states_mlp_list = [
                    mlp_activations[f'layer_{i}_mlp'] for i in range(self.model.config.num_hidden_layers)
                ]
                
                # Get relevant activations
                try:
                    hidden_states_mlp = hidden_states_mlp_list[self.start_id - self.num_layer - 1]
                    hidden_states_i = hidden_states[self.start_id - self.num_layer - 1]
                    hidden_states_n = hidden_states[self.end_id - self.num_layer - 1]
                    print(f"{Fore.GREEN}[ACTIVATIONS] Extracted activations successfully{Fore.RESET}")
                except IndexError as e:
                    print(f"{Fore.RED}[ERROR] IndexError in activation extraction: {e}{Fore.RESET}")
                    print(f"start_id: {self.start_id}, end_id: {self.end_id}, num_layer: {self.num_layer}")
                    print(f"Available hidden states: {len(hidden_states)}, MLP activations: {len(hidden_states_mlp_list)}")
                    continue
                
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
                
                print(f"{Fore.BLUE}[DATA] Processed {batch_tokens} tokens, shapes: a1={a1_batch.shape}, a2={a2_batch.shape}{Fore.RESET}")
                
                # Extract the specific batch we need
                batch_end = current_idx + batch_tokens
                if target_start < batch_end and current_idx < target_end:
                    # Calculate overlap
                    start_in_batch = max(0, target_start - current_idx)
                    end_in_batch = min(batch_tokens, target_end - current_idx)
                    
                    extracted_size = end_in_batch - start_in_batch
                    print(f"{Fore.GREEN}[EXTRACT] Extracting {extracted_size} tokens from positions {start_in_batch} to {end_in_batch}{Fore.RESET}")
                    
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
                    print(f"{Fore.GREEN}[COMPLETE] Reached target end {target_end}, stopping{Fore.RESET}")
                    break
                    
                # Progress update
                if batch_count % 5 == 0:
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"{Fore.CYAN}[PROGRESS] Processed {batch_count} batches, current_idx: {current_idx}, GPU memory: {memory_used:.2f}GB{Fore.RESET}")
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
            print(f"{Fore.YELLOW}[CLEANUP] Removed {len(hooks)} hooks{Fore.RESET}")
        
        # Combine batch data
        if batch_data:
            combined = {
                'a1': torch.cat([d['a1'] for d in batch_data], dim=0),
                'a2': torch.cat([d['a2'] for d in batch_data], dim=0),
                'attention': torch.cat([d['attention'] for d in batch_data], dim=0)
            }
            print(f"{Fore.GREEN}[RESULT] Combined data shapes: a1={combined['a1'].shape}, a2={combined['a2'].shape}{Fore.RESET}")
            return combined
            
        print(f"{Fore.RED}[WARNING] No batch data collected{Fore.RESET}")
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
    batch_size = 128  # Smaller batch size
    num_batches = max(10, (streaming_dataset.estimated_total_length + batch_size - 1) // batch_size // 10)  # Process fewer batches
    
    print(f"{Fore.GREEN}[DEBUG] Will process {num_batches} batches with batch_size {batch_size}{Fore.RESET}")
    
    # Get a sample to determine hidden size
    print(f"{Fore.BLUE}[DEBUG] Getting sample batch to determine hidden size...{Fore.RESET}")
    sample_batch = streaming_dataset.process_batch_streaming_with_progress(0, batch_size=min(50, batch_size))
    if sample_batch is None:
        raise ValueError("Could not get sample batch")
    
    hidden_size = sample_batch['a1'].shape[1]
    print(f"{Fore.GREEN}[DEBUG] Hidden size: {hidden_size}, Total batches to process: {num_batches}{Fore.RESET}")
    
    # Initialize running estimates for current_output
    current_output_estimate = None
    
    for layer_idx in range(num_residual_layers):
        print(f"{Fore.BLUE}[DEBUG] Training residual layer {layer_idx + 1}/{num_residual_layers}{Fore.RESET}")
        
        # Initialize transformation matrix with better initialization
        transform = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        torch.nn.init.normal_(transform.weight, mean=0.0, std=0.02)
        
        # Use smaller learning rate and add weight decay
        optimizer = torch.optim.Adam(transform.parameters(), lr=5e-5, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        
        # Training loop with streaming
        for epoch in range(3):  # Further reduced epochs
            print(f"{Fore.YELLOW}[EPOCH] Layer {layer_idx+1}, Epoch {epoch+1}/3{Fore.RESET}")
            epoch_loss = 0.0
            valid_batches = 0
            
            # Process fewer batches for efficiency
            for batch_idx in range(0, min(num_batches, 20), 5):  # Process max 4 batches per epoch
                print(f"{Fore.CYAN}[BATCH] Processing batch {batch_idx+1}/{min(num_batches, 20)} for epoch {epoch+1}{Fore.RESET}")
                
                # Get batch data
                batch_data = streaming_dataset.process_batch_streaming_with_progress(batch_idx, batch_size)
                if batch_data is None:
                    print(f"{Fore.RED}[WARNING] Batch {batch_idx} returned None, skipping{Fore.RESET}")
                    continue
                
                # Convert to float32 only when needed for computation
                print(f"{Fore.BLUE}[CONVERT] Converting batch data to GPU...{Fore.RESET}")
                a1_batch = batch_data['a1'].to(device, dtype=torch.float32)
                a2_batch = batch_data['a2'].to(device, dtype=torch.float32)
                attention_batch = batch_data['attention'].to(device, dtype=torch.float32)
                
                # Estimate current output if this is the first layer
                if current_output_estimate is None:
                    current_output_estimate = a1_batch.clone()
                    print(f"{Fore.GREEN}[INIT] Initialized current_output_estimate with shape {current_output_estimate.shape}{Fore.RESET}")
                else:
                    # Use a running average for memory efficiency
                    current_output_estimate = 0.9 * current_output_estimate + 0.1 * a1_batch
                    print(f"{Fore.BLUE}[UPDATE] Updated current_output_estimate{Fore.RESET}")
                
                # Calculate residual target
                residual_target = a2_batch - current_output_estimate
                
                # Determine input for this layer
                if layer_idx == 0:
                    input_data = a1_batch
                    print(f"{Fore.GREEN}[LAYER] Layer 1: Using original MLP output{Fore.RESET}")
                elif layer_idx == 1:
                    input_data = a1_batch * (attention_batch + 1e-8)  # Add small epsilon
                    print(f"{Fore.GREEN}[LAYER] Layer 2: Using attention-weighted input{Fore.RESET}")
                else:
                    input_data = current_output_estimate
                    print(f"{Fore.GREEN}[LAYER] Layer 3+: Using current estimate{Fore.RESET}")
                
                # Check for problematic values
                if torch.isnan(input_data).any() or torch.isnan(residual_target).any():
                    print(f"{Fore.RED}[WARNING] NaN detected in input_data or residual_target, skipping{Fore.RESET}")
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                pred_residual = transform(input_data)
                
                # Compute loss
                if loss == "cosine":
                    loss_val = cosine_loss_stable(pred_residual, residual_target)
                    print(f"{Fore.BLUE}[LOSS] Using cosine loss: {loss_val.item():.6f}{Fore.RESET}")
                else:
                    loss_val = torch.nn.MSELoss()(pred_residual, residual_target)
                    print(f"{Fore.BLUE}[LOSS] Using MSE loss: {loss_val.item():.6f}{Fore.RESET}")
                
                if torch.isnan(loss_val):
                    print(f"{Fore.RED}[WARNING] Loss is NaN, skipping{Fore.RESET}")
                    continue
                
                # Backward pass with gradient clipping
                loss_val.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(transform.parameters(), max_norm=0.5)
                optimizer.step()
                
                print(f"{Fore.GREEN}[GRAD] Gradient norm: {grad_norm:.6f}{Fore.RESET}")
                
                epoch_loss += loss_val.item()
                valid_batches += 1
                
                # Update current output estimate
                with torch.no_grad():
                    current_output_estimate = current_output_estimate + pred_residual.detach()
                    print(f"{Fore.BLUE}[UPDATE] Updated current_output_estimate with residual{Fore.RESET}")
                
                # Immediately free GPU memory
                del a1_batch, a2_batch, attention_batch, residual_target, pred_residual
                torch.cuda.empty_cache()
                
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"{Fore.CYAN}[MEMORY] GPU memory after batch: {memory_used:.2f}GB{Fore.RESET}")
            
            scheduler.step()
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                print(f"{Fore.YELLOW}[RESULT] Layer {layer_idx+1}, Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}, Valid Batches: {valid_batches}{Fore.RESET}")
            else:
                print(f"{Fore.RED}[WARNING] Layer {layer_idx+1}, Epoch {epoch+1}, No valid batches processed!{Fore.RESET}")
        
        # Save transformation (convert to bfloat16 to save memory)
        transformations.append(transform.weight.data.clone().cpu().to(torch.bfloat16))
        print(f"{Fore.GREEN}[SAVE] Saved transformation {layer_idx+1} to CPU{Fore.RESET}")
        
        # Clean up
        del transform, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
        print(f"{Fore.GREEN}[CLEANUP] Cleaned up layer {layer_idx+1} resources{Fore.RESET}")
    
    print(f"{Fore.GREEN}[DEBUG] Memory-efficient residual learning completed with {len(transformations)} transformations{Fore.RESET}")
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
    print(f"{Fore.BLUE}[DEBUG] Creating streaming dataset...{Fore.RESET}")
    streaming_dataset = StreamingActivationDataset(
        dataloader, model, tokenizer, max_length, start_id, end_id, num_layer, dataset_size
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

