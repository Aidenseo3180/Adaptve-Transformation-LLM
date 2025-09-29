"""
Teacher-Guided Transform for ReplaceMe
Memory-efficient implementation that uses original model as teacher
to guide the transform learning process.
"""

import gc
import logging
import os
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from colorama import Fore, init

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{Fore.GREEN}Using device: {device}{Fore.RESET}")


class TeacherGuidedDataset(Dataset):
    """Dataset for teacher-guided training"""
    def __init__(self, mlp_outputs, teacher_inputs, teacher_outputs):
        self.mlp_outputs = mlp_outputs
        self.teacher_inputs = teacher_inputs
        self.teacher_outputs = teacher_outputs
        
    def __len__(self):
        return len(self.mlp_outputs)
    
    def __getitem__(self, idx):
        return (
            self.mlp_outputs[idx],
            self.teacher_inputs[idx],
            self.teacher_outputs[idx]
        )


def collect_teacher_signals(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    start_id: int,
    end_id: int,
    max_length: int = 1024,
    dataset_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect teacher signals from the original model.
    
    Returns:
        mlp_outputs: MLP output from layer (start_id - 1)
        teacher_inputs: Input to layer start_id
        teacher_outputs: Output from layer end_id
    """
    print(f"\n{Fore.CYAN}Collecting teacher signals from layers {start_id} to {end_id}{Fore.RESET}")
    
    model.eval()
    hidden_size = model.config.hidden_size
    
    # Setup hooks to capture MLP outputs
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    print(f"{Fore.YELLOW}Registering hooks for MLP layers...{Fore.RESET}")
    for i in range(len(model.model.layers)):
        hook = model.model.layers[i].mlp.register_forward_hook(
            save_mlp_activation(f'layer_{i}_mlp')
        )
        hooks.append(hook)
    
    # Storage lists
    mlp_outputs_list = []
    teacher_inputs_list = []
    teacher_outputs_list = []
    
    processed_tokens = 0
    target_tokens = dataset_size * max_length if dataset_size else float('inf')
    
    print(f"{Fore.GREEN}Processing batches to collect activations...{Fore.RESET}")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting teacher signals")):
        if processed_tokens >= target_tokens:
            print(f"{Fore.YELLOW}Reached target token count: {processed_tokens}{Fore.RESET}")
            break
            
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            # Forward pass through original model
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List of hidden states from each layer
            
            # Get MLP output from layer before start_id
            mlp_out = mlp_activations[f'layer_{start_id - 1}_mlp']
            
            # Get input to start_id layer (includes attention residual)
            teacher_input = hidden_states[start_id]
            
            # Get output from end_id layer
            teacher_output = hidden_states[end_id + 1]
            
            # Apply attention mask to ignore padding tokens
            b, s, h = mlp_out.shape
            mask = attention_mask.view(-1).bool()
            
            # Flatten and mask
            mlp_out_flat = mlp_out.view(-1, h)[mask]
            teacher_input_flat = teacher_input.view(-1, h)[mask]
            teacher_output_flat = teacher_output.view(-1, h)[mask]
            
            # Move to CPU to save GPU memory
            mlp_outputs_list.append(mlp_out_flat.cpu().float())
            teacher_inputs_list.append(teacher_input_flat.cpu().float())
            teacher_outputs_list.append(teacher_output_flat.cpu().float())
            
            processed_tokens += mask.sum().item()
        
        # Clear GPU cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            print(f"  Processed {processed_tokens} tokens so far...")
    
    # Remove hooks
    print(f"{Fore.YELLOW}Removing hooks...{Fore.RESET}")
    for hook in hooks:
        hook.remove()
    
    # Concatenate all collected data
    print(f"{Fore.GREEN}Concatenating collected data...{Fore.RESET}")
    mlp_outputs = torch.cat(mlp_outputs_list, dim=0)
    teacher_inputs = torch.cat(teacher_inputs_list, dim=0)
    teacher_outputs = torch.cat(teacher_outputs_list, dim=0)
    
    print(f"{Fore.CYAN}Collected {mlp_outputs.shape[0]} token activations{Fore.RESET}")
    print(f"  MLP outputs shape: {mlp_outputs.shape}")
    print(f"  Teacher inputs shape: {teacher_inputs.shape}")
    print(f"  Teacher outputs shape: {teacher_outputs.shape}")
    
    # Sample data if too large
    max_tokens = 2000000  # 2M tokens max for memory
    if mlp_outputs.shape[0] > max_tokens:
        print(f"{Fore.YELLOW}Sampling {max_tokens} from {mlp_outputs.shape[0]} tokens...{Fore.RESET}")
        indices = torch.randperm(mlp_outputs.shape[0])[:max_tokens]
        mlp_outputs = mlp_outputs[indices]
        teacher_inputs = teacher_inputs[indices]
        teacher_outputs = teacher_outputs[indices]
    
    return mlp_outputs, teacher_inputs, teacher_outputs


def learn_teacher_guided_transform(
    mlp_outputs: torch.Tensor,
    teacher_inputs: torch.Tensor,
    teacher_outputs: torch.Tensor,
    hidden_size: int,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 5e-5
) -> torch.Tensor:
    """
    Learn transform T that makes layer predictions match teacher outputs.
    """
    print(f"\n{Fore.CYAN}Learning teacher-guided transform{Fore.RESET}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    # Move to appropriate device
    compute_device = device if torch.cuda.is_available() else torch.device('cpu')
    
    # Initialize transform with least squares solution
    print(f"{Fore.YELLOW}Computing initial transform with least squares...{Fore.RESET}")
    try:
        # Target: what layers 20-23 add to the representation
        residual_target = teacher_outputs - teacher_inputs
        
        # Least squares initialization
        XtX = mlp_outputs.T @ mlp_outputs
        XtY = mlp_outputs.T @ residual_target
        reg = 1e-4 * torch.trace(XtX) / hidden_size * torch.eye(hidden_size)
        T_init = torch.linalg.solve(XtX + reg, XtY)
        print(f"{Fore.GREEN}Least squares initialization successful{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}Least squares failed: {e}, using identity init{Fore.RESET}")
        T_init = torch.eye(hidden_size)
    
    # Move to device and enable gradients
    T = T_init.to(compute_device).requires_grad_(True)
    
    # Optimizer and scheduler
    optimizer = Adam([T], lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create dataset and dataloader
    dataset = TeacherGuidedDataset(mlp_outputs, teacher_inputs, teacher_outputs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if compute_device.type == 'cuda' else False
    )
    
    # Training variables
    best_loss = float('inf')
    best_T = T.clone().detach()
    
    print(f"\n{Fore.GREEN}Starting optimization...{Fore.RESET}")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_mlp, batch_input, batch_output in pbar:
            # Move to device
            batch_mlp = batch_mlp.to(compute_device)
            batch_input = batch_input.to(compute_device)
            batch_output = batch_output.to(compute_device)
            
            optimizer.zero_grad()
            
            # Apply transform
            transformed = batch_mlp @ T
            
            # Predict output after layers
            predicted_output = batch_input + transformed
            
            # Loss 1: Output matching (main objective)
            output_norm = F.normalize(predicted_output, dim=-1)
            teacher_norm = F.normalize(batch_output, dim=-1)
            loss_output = 1 - (output_norm * teacher_norm).sum(-1).mean()
            
            # Loss 2: Residual matching
            actual_residual = batch_output - batch_input
            residual_norm = F.normalize(transformed, dim=-1)
            actual_residual_norm = F.normalize(actual_residual, dim=-1)
            loss_residual = 1 - (residual_norm * actual_residual_norm).sum(-1).mean()
            
            # Loss 3: Magnitude preservation (softer constraint)
            pred_magnitude = torch.norm(predicted_output, dim=-1)
            teacher_magnitude = torch.norm(batch_output, dim=-1)
            magnitude_ratio = pred_magnitude / (teacher_magnitude + 1e-8)
            loss_magnitude = (magnitude_ratio - 1).abs().mean()
            
            # Combined loss with weights
            total_loss = (
                0.5 * loss_output +     # Primary objective
                0.3 * loss_residual +   # Residual accuracy
                0.2 * loss_magnitude * 0.1  # Soft magnitude constraint
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Track loss
            epoch_losses.append(total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # Learning rate scheduling
        scheduler.step()
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T.clone().detach()
            print(f"{Fore.GREEN}  New best loss: {best_loss:.6f}{Fore.RESET}")
        
        # Periodic logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, Best = {best_loss:.6f}, LR = {current_lr:.2e}")
    
    print(f"\n{Fore.GREEN}Optimization complete! Final best loss: {best_loss:.6f}{Fore.RESET}")
    
    return best_T.cpu().to(torch.float64)


def teacher_guided_cosine_dist(
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
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    **kwargs
) -> str:
    """
    Main function for teacher-guided transform estimation and model pruning.
    """
    print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}Teacher-Guided Transform for Layer Pruning{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Pruning layers {start_id} to {end_id}")
    print(f"Dataset: {dataset} (size: {dataset_size})")
    
    # Import required utilities
    from .utils import get_calib_dataloader, truncate_model
    
    # Setup quantization if needed
    quantization_config = None
    if use_4bit:
        print(f"{Fore.YELLOW}Using 4-bit quantization{Fore.RESET}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    print(f"\n{Fore.YELLOW}Loading model...{Fore.RESET}")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden size {hidden_size}")
    
    # Get calibration dataloader
    print(f"\n{Fore.YELLOW}Preparing calibration data...{Fore.RESET}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Collect teacher signals
    mlp_outputs, teacher_inputs, teacher_outputs = collect_teacher_signals(
        model,
        dataloader,
        tokenizer,
        start_id,
        end_id,
        max_length,
        dataset_size
    )
    
    # Clean up model to free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Learn transform
    transform = learn_teacher_guided_transform(
        mlp_outputs,
        teacher_inputs,
        teacher_outputs,
        hidden_size,
        epochs=20,
        batch_size=1024,
        lr=5e-5
    )
    
    # Clean up training data
    del mlp_outputs, teacher_inputs, teacher_outputs
    gc.collect()
    
    # Reload model for modification
    print(f"\n{Fore.YELLOW}Reloading model for pruning...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model (remove layers)
    print(f"{Fore.YELLOW}Removing layers {start_id} to {end_id}...{Fore.RESET}")
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transform to down_proj
    print(f"{Fore.YELLOW}Applying transform to layer {start_id - num_layer - 1} down_proj...{Fore.RESET}")
    
    try:
        down_proj_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
        new_weight = (transform.T @ down_proj_weight.to(torch.float64)).to(torch.bfloat16)
        
        # Stability check
        if torch.isnan(new_weight).any() or torch.isinf(new_weight).any():
            print(f"{Fore.RED}Warning: Transform produced NaN/Inf, using original weights{Fore.RESET}")
            new_weight = down_proj_weight
        else:
            print(f"{Fore.GREEN}Transform applied successfully{Fore.RESET}")
        
        model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight = nn.Parameter(new_weight)
    except Exception as e:
        print(f"{Fore.RED}Error applying transform: {e}{Fore.RESET}")
        raise
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_teacher_guided_{start_id}_{end_id}_{dataset}_{dataset_size}"
    
    save_path = f"{save_path}_TeacherGuided"
    
    print(f"\n{Fore.YELLOW}Saving model to {save_path}...{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"{Fore.GREEN}Model saved successfully!{Fore.RESET}")
    
    # Final cleanup
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{Fore.CYAN}Teacher-guided pruning complete!{Fore.RESET}")
    
    return save_path


def validate_teacher_guided_transform(
    model_path: str,
    pruned_model_path: str,
    dataloader: DataLoader,
    tokenizer,
    start_id: int,
    end_id: int,
    max_samples: int = 100
):
    """
    Validate how well the teacher-guided transform approximates the original model.
    """
    print(f"\n{Fore.CYAN}Validating teacher-guided transform...{Fore.RESET}")
    
    # Load both models
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    pruned_model = AutoModelForCausalLM.from_pretrained(
        pruned_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    original_model.eval()
    pruned_model.eval()
    
    total_cosine_sim = 0
    count = 0
    
    for idx, batch in enumerate(dataloader):
        if idx >= max_samples:
            break
            
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            max_length=1024,
            truncation=True
        )
        inputs = {k: v.to(original_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Original model output
            orig_outputs = original_model(**inputs)
            orig_logits = orig_outputs.logits
            
            # Pruned model output
            pruned_outputs = pruned_model(**inputs)
            pruned_logits = pruned_outputs.logits
            
            # Compare final predictions
            cos_sim = F.cosine_similarity(
                orig_logits.view(-1, orig_logits.shape[-1]),
                pruned_logits.view(-1, pruned_logits.shape[-1]),
                dim=-1
            ).mean()
            
            total_cosine_sim += cos_sim.item()
            count += 1
    
    avg_similarity = total_cosine_sim / count
    print(f"{Fore.GREEN}Average cosine similarity: {avg_similarity:.4f}{Fore.RESET}")
    print(f"Loss: {1 - avg_similarity:.4f}")
    
    return avg_similarity