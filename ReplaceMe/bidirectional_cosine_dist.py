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

from .utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_non_overlapping_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def bidirectional_cosine_dist(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,
    loss: str = "cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    bidirectional_alpha: float = 0.5  # Weight for up_proj compensation
    
) -> str:
    """Calculate cosine distance between model layers and apply bidirectional transformations.
    
    This is an enhanced version of the original cosine_dist that applies compensation
    to both down_proj (before removed blocks) and up_proj (after removed blocks).
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of start layer for cut
        token: Authentication token
        save_transform_only: Whether to only save the transform
        diag: Whether to use diagonal matrix
        loss: Loss function type
        solver: Optimization solver type
        thri: Whether to use three vectors
        two_vectors: Whether to use two vectors
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
        accurate: Whether to use accurate mode
        bidirectional_alpha: Weight for up_proj compensation (0.0 = only down_proj, 1.0 = only up_proj)
    
    Returns:
        Path where transformed model is saved
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
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
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
        def hook(module, input, output):
            # Detach to avoid keeping computation history
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    mlp_activations = {}
    a1 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    if accurate:
        print("ACCURATE MODE IS ON (MORE MEMORY IS NEEDED)")
        a3 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
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
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]    
    
    # Estimate the main transformation matrix (for down_proj)
    if solver == "adam":
        transform_down = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        transform_down = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    # Estimate complementary transformation for up_proj
    # This compensates for the input distribution change to the next block
    print(f"{Fore.GREEN}Estimating up_proj compensation matrix{Fore.RESET}")
    
    # For up_proj compensation, we want to adapt the next layer to the changed input
    # We use the residual between original output and transformed output
    # Ensure all tensors are on the same device (CPU) and same dtype
    a1_cpu = a1.to('cpu').to(torch.float64)
    a2_cpu = a2.to('cpu').to(torch.float64)
    transform_down_cpu = transform_down.to('cpu').to(torch.float64)
    
    residual_activations = a2_cpu - (a1_cpu @ transform_down_cpu.T)  # What we're missing
    
    if solver == "adam":
        transform_up = adam_method(residual_activations, a2_cpu, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        transform_up = optimizing_method(residual_activations, a2_cpu, solver=solver)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply bidirectional transformations
    print(f"{Fore.GREEN}Applying bidirectional transformations{Fore.RESET}")
    
    # Apply down_proj transformation (main compensation)
    down_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)
    new_down_weight = (transform_down.T.cpu() @ down_weight).to(torch.bfloat16)
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": new_down_weight
    })
    
    # Apply up_proj transformation (complementary compensation)
    if end_id - num_layer < len(model.model.layers):  # Check if next layer exists
        up_weight = model.model.layers[end_id - num_layer].mlp.up_proj.weight.to(torch.float64)
        
        # Weighted combination: balance between original and compensated
        compensation_weight = bidirectional_alpha * (up_weight @ transform_up.T.cpu()).to(torch.bfloat16)
        original_weight = (1 - bidirectional_alpha) * up_weight.to(torch.bfloat16)
        new_up_weight = compensation_weight + original_weight
        
        model.model.layers[end_id - num_layer].mlp.up_proj.load_state_dict({
            "weight": new_up_weight
        })
        
        print(f"{Fore.GREEN}Applied up_proj compensation with alpha={bidirectional_alpha}{Fore.RESET}")
    else:
        print(f"{Fore.YELLOW}No next layer found, skipping up_proj compensation{Fore.RESET}")
    
    # Save model
    model_save_path = f"{save_path}_BiReplaceME_{loss}_{solver}_alpha{bidirectional_alpha}"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    if save_transform_only:
        torch.save({
            'transform_down': transform_down,
            'transform_up': transform_up,
            'bidirectional_alpha': bidirectional_alpha
        }, f"{model_save_path}_transforms.pth")
    
    # Final cleanup
    del model, a1, a2, a1_cpu, a2_cpu, transform_down_cpu, residual_activations
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{Fore.GREEN}Bidirectional ReplaceMe completed successfully{Fore.RESET}")
    return model_save_path

