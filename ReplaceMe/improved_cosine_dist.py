import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .improved_replaceme import improved_adam_optimizer
from .utils import get_calib_dataloader, truncate_model, seed_all

seed_all()


def improved_cosine_dist(
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
    # New parameters for improved optimization
    optimizer_epochs: int = 20,
    optimizer_lr: float = 5e-4,
    optimizer_batch_size: int = 1024,
    loss_type: str = "mixed",
    early_stopping_patience: int = 3,
    **kwargs
) -> str:
    """
    Improved version of cosine_dist with better optimization.
    
    Additional parameters:
        optimizer_epochs: Number of epochs for optimization
        optimizer_lr: Learning rate for optimizer
        optimizer_batch_size: Batch size for optimizer
        loss_type: Type of loss function ("cosine", "mixed", "mse")
        early_stopping_patience: Patience for early stopping
    """
    
    print(f"\n[Improved ReplaceMe] Processing layers {start_id}-{end_id}")
    print(f"[Improved ReplaceMe] Using {loss_type} loss with {optimizer_epochs} epochs")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    # Load model for activation collection
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.float16  # Use float16 for memory efficiency
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    # Get calibration data
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Register hooks for MLP activations
    mlp_activations = {}
    
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    
    # Collect activations
    print(f"[Improved ReplaceMe] Gathering activations...")
    
    a1_list = []
    a2_list = []
    
    for batch in tqdm(dataloader, desc="Collecting Activations"):
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
        
        # Get relevant hidden states and MLP outputs
        hidden_states_mlp = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        # Reshape and prepare data
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states_i.view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states_n.view(-1, hidden_size).to(torch.float64)
        
        # Original ReplaceMe formula
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        a1_list.append(a1_batch.cpu())
        a2_list.append(a2_batch.cpu())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all batches
    a1 = torch.cat(a1_list, dim=0)
    a2 = torch.cat(a2_list, dim=0)
    
    print(f"[Improved ReplaceMe] Collected {a1.shape[0]} samples")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Optimize transformation matrix
    print(f"[Improved ReplaceMe] Starting optimization...")
    
    transform = improved_adam_optimizer(
        a1=a1,
        a2=a2,
        a3=None,
        loss_type=loss_type,
        epochs=optimizer_epochs,
        lr=optimizer_lr,
        batch_size=optimizer_batch_size,
        weight_decay=1e-4,
        patience=early_stopping_patience,
        verbose=True
    )
    
    print(f"[Improved ReplaceMe] Optimization complete")
    
    # Reload model and apply transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj
    target_layer = model.model.layers[start_id - num_layer - 1]
    original_weight = target_layer.mlp.down_proj.weight.to(torch.float64)
    new_weight = transform.T.cpu() @ original_weight
    target_layer.mlp.down_proj.weight = nn.Parameter(new_weight.to(torch.bfloat16))
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/ImprovedReplaceMe_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save transform for analysis
    torch.save({
        'transform': transform,
        'loss_type': loss_type,
        'epochs': optimizer_epochs,
        'lr': optimizer_lr
    }, f"{save_path}/transform_info.pt")
    
    print(f"[Improved ReplaceMe] Model saved to {save_path}")
    
    # Final cleanup
    del model, a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path