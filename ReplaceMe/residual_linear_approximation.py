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


def residual_linear_approximation_method(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor = None,
    loss: str = "cosine",
    num_residual_layers: int = 3,
    attention_weights: torch.Tensor = None
) -> list:
    """
    Learn multiple residual linear transformations that build upon each other.
    
    Args:
        a1: MLP activations before pruning
        a2: Target activations after pruning blocks  
        a3: Optional attention residual information
        loss: Loss function type
        num_residual_layers: Number of residual transformation layers
        attention_weights: Optional attention pattern information
    
    Returns:
        List of transformation matrices [T1, T2, T3, ...]
    """
    print(f"{Fore.GREEN}[DEBUG] Starting residual linear approximation with {num_residual_layers} layers{Fore.RESET}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformations = []
    
    # Start with identity as base
    current_output = a1.clone()
    target = a2.clone()
    
    for layer_idx in range(num_residual_layers):
        print(f"{Fore.BLUE}[DEBUG] Training residual layer {layer_idx + 1}/{num_residual_layers}{Fore.RESET}")
        
        # Calculate residual to learn
        residual_target = target - current_output
        
        # Initialize transformation matrix
        hidden_size = a1.shape[1]
        transform = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        torch.nn.init.zeros_(transform.weight)  # Start from zero transformation
        
        # Optimizer for this layer
        optimizer = torch.optim.Adam(transform.parameters(), lr=1e-4)
        
        # Training data for this residual layer
        if layer_idx == 0:
            # First layer: basic transformation
            input_data = a1
        elif layer_idx == 1 and attention_weights is not None:
            # Second layer: attention-aware transformation
            input_data = a1 * attention_weights
        else:
            # Additional layers: refined transformations
            input_data = current_output
        
        # Train this residual layer
        dataset = ResidualDataset(input_data, residual_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        
        for epoch in range(10):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                pred_residual = transform(batch_x)
                
                if loss == "cosine":
                    loss_val = cosine_loss(pred_residual, batch_y)
                else:
                    loss_val = torch.nn.MSELoss()(pred_residual, batch_y)
                
                loss_val.backward()
                optimizer.step()
                epoch_loss += loss_val.item()
            
            if epoch % 2 == 0:
                print(f"{Fore.YELLOW}[DEBUG] Layer {layer_idx+1}, Epoch {epoch}, Loss: {epoch_loss/len(loader):.6f}{Fore.RESET}")
        
        # Update current output with this layer's contribution
        with torch.no_grad():
            residual_contribution = transform(input_data)
            current_output = current_output + residual_contribution
        
        transformations.append(transform.weight.data.clone().cpu())
        print(f"{Fore.GREEN}[DEBUG] Layer {layer_idx + 1} completed. Current MSE: {torch.nn.MSELoss()(current_output, target).item():.6f}{Fore.RESET}")
        
        # Clear GPU memory
        del transform, optimizer
        torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}[DEBUG] Residual linear approximation completed with {len(transformations)} layers{Fore.RESET}")
    return transformations


class ResidualDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance loss"""
    pred_norm = pred / pred.norm(dim=1, keepdim=True)
    target_norm = target / target.norm(dim=1, keepdim=True)
    return 1 - (pred_norm * target_norm).sum(dim=1).mean()


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
    Main function for residual linear approximation compression.
    """
    print(f"{Fore.CYAN}[DEBUG] Starting residual linear compression...{Fore.RESET}")
    print(f"{Fore.CYAN}[DEBUG] Model: {model_path}, Layers to skip: {layers_to_skip}{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        print(f"{Fore.YELLOW}[DEBUG] Using 4-bit quantization{Fore.RESET}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model
    print(f"{Fore.GREEN}[DEBUG] Loading model from {model_path}...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"{Fore.GREEN}[DEBUG] Model loaded successfully. Hidden size: {hidden_size}{Fore.RESET}")
    
    # Get calibration data
    print(f"{Fore.GREEN}[DEBUG] Loading calibration data...{Fore.RESET}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Set up hooks for activation collection
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    
    # Determine model architecture and set hooks
    if 'falcon' in model_path.lower():
        print(f"{Fore.BLUE}[DEBUG] Detected Falcon architecture{Fore.RESET}")
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        print(f"{Fore.BLUE}[DEBUG] Detected standard transformer architecture{Fore.RESET}")
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Collect activations
    print(f"{Fore.GREEN}[DEBUG] Collecting activations for residual learning...{Fore.RESET}")
    
    a1 = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    attention_weights = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    
    cnt = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{Fore.RED}Gathering Activations{Fore.RESET}")):
        if batch_idx % 10 == 0:
            print(f"{Fore.YELLOW}[DEBUG] Processing batch {batch_idx}, collected {cnt} samples so far{Fore.RESET}")
            
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        
        # Get activations for the blocks we want to replace
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states_i.view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states_n.view(-1, hidden_size).to(torch.float64)
        
        # Calculate attention influence (simplified)
        attention_influence = torch.abs(hidden_states_i - hidden_states_mlp)
        attention_weights[cnt:cnt+attention_influence.shape[0]] = attention_influence
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        # Clean up batch-specific variables
        del hidden_states_mlp, hidden_states_i, hidden_states_n, attention_influence
        torch.cuda.empty_cache()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"{Fore.GREEN}[DEBUG] Collected {cnt} activation samples{Fore.RESET}")
    
    # Trim tensors to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    attention_weights = attention_weights[:cnt]
    
    print(f"{Fore.BLUE}[DEBUG] Starting residual transformation learning...{Fore.RESET}")
    # Learn residual transformations
    transformations = residual_linear_approximation_method(
        a1, a2, 
        attention_weights=attention_weights,
        loss=loss,
        num_residual_layers=num_residual_layers
    )
    
    # Clean up activations
    del a1, a2, attention_weights
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{Fore.GREEN}[DEBUG] Memory cleaned up after transformation learning{Fore.RESET}")
    
    # Clean up original model and reload for modification
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}[DEBUG] Reloading model for transformation application...{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformations to model
    print(f"{Fore.BLUE}[DEBUG] Applying {len(transformations)} residual transformations to model...{Fore.RESET}")
    
    # For simplicity, we'll combine all transformations into a single effective transformation
    # In practice, you might want to modify the model architecture to support multiple residual layers
    combined_transform = transformations[0]
    for i in range(1, len(transformations)):
        combined_transform = combined_transform + transformations[i]
    
    # Apply to the model
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (combined_transform.T @ original_weight.to(torch.float64)).to(torch.bfloat16)
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
        print(f"{Fore.GREEN}[DEBUG] Saved transformations to {final_path}_transforms.pt{Fore.RESET}")
    
    # Final cleanup
    del model, transformations, combined_transform
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}[DEBUG] Residual linear compression completed successfully!{Fore.RESET}")
    return final_path

