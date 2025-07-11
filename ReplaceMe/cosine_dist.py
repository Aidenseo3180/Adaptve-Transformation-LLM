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

# Set random seed for reproducibility
seed_all()


def cosine_dist(
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
    accurate: bool= False
    
) -> str:
    """Calculate cosine distance between model layers and apply transformations.
    
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
    
    Returns:
        Path where transformed model is saved
    """
    # Determine device mapping based on CUDA availability
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    # Setup 4-bit quantization configuration if requested
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load the transformer model with specified configurations
    # output_hidden_states=True is crucial for capturing intermediate layer activations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    # Load tokenizer and get model's hidden dimension size
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
    # Set padding token if not available (needed for batch processing)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    
    # Create data loader for calibration dataset
    # This dataset will be used to collect activations for transformation estimation
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'.
        
        Forward hooks are used to capture intermediate activations during model forward pass.
        This is essential for collecting MLP outputs at specific layers.
        """
        def hook(module, input, output):
            # Detach to avoid keeping computation history and reduce memory usage
            mlp_activations[name] = output.detach()
        return hook

    # Register forward hooks to capture MLP activations from all layers
    # These hooks will fire during forward pass and store MLP outputs
    hooks = []
    
    # Handle different model architectures (Falcon vs standard transformer)
    if 'falcon' in model_path.lower():
        # Falcon models have different layer structure
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        # Standard transformer models (LLaMA, etc.)
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Dictionary to store captured MLP activations
    mlp_activations = {}
    
    # Pre-allocate tensors to store activations for all samples
    # a1: MLP activations at the start layer (layer i)
    # a2: Target activations that we want to approximate
    # These are stored on CPU to manage GPU memory efficiently
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
    
    # If accurate mode is enabled, allocate additional tensor for more precise estimation
    if accurate:
        print("ACCURATE MODE IS ON (MORE MEMORY IS NEEDED")
        a3 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    # Counter to track how many activations have been collected
    cnt = 0
    
    # Main data collection loop: process each batch to collect activations
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
        # Tokenize the batch with padding and truncation
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        # Move tokenized inputs to model's device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass through the model (no gradient computation needed)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract hidden states from all layers (excluding input embeddings)
        hidden_states = outputs.hidden_states[1:]
        
        # Get MLP activations that were captured by the hooks
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        
        # Extract the specific MLP activation we need (at the start of the block to be pruned)
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        # Reshape activations from [batch_size, seq_len, hidden_size] to [batch_size * seq_len, hidden_size]
        # This flattens the batch and sequence dimensions for easier processing
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        # Prepare the input and target activations for transformation learning
        # a1_batch: MLP output at start layer (what we'll transform)
        a1_batch = hidden_states_mlp
        
        # a2_batch: Target output we want to achieve after transformation
        # This represents: output_at_end + mlp_at_start - input_at_start
        # The intuition: we want mlp_at_start * T to approximate what the removed blocks would produce
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        # In accurate mode, use different formulation for more precise estimation
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            # Store the additional activation tensor
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
            
        # Store the collected activations in pre-allocated tensors
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        # Update counter for next batch
        cnt += a2_batch.shape[0]
        
        # Clean up intermediate tensors to free memory
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    # Trim the pre-allocated tensors to actual size (remove unused portions)
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
        
    # Estimate the linear transformation matrix T using the collected activations
    # Two approaches: Adam-based optimization or analytical methods
    if solver == "adam":
        # Use Adam optimizer with specified loss function (cosine, MSE, etc.)
        transform = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        # Use analytical or other numerical optimization methods
        transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    # Clean up the original model from memory before loading a fresh copy
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation (on CPU to avoid device memory issues)
    # This fresh model will be modified with the estimated transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    # Remove the layers that will be replaced by the linear transformation
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Generate save path if not provided
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply the learned transformation by fusing it with the MLP down-projection layer
    # This is the key step: instead of adding new parameters, we modify existing ones
    # The transformation T is applied by matrix multiplication: new_weight = T^T @ old_weight
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (transform.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    # Save the modified model and tokenizer to disk
    model.save_pretrained(f"{save_path}_ReplaceMe_{loss}_{solver}")
    tokenizer.save_pretrained(f"{save_path}_ReplaceMe_{loss}_{solver}")
    
    # Optionally save only the transformation matrix for analysis
    if save_transform_only:
        torch.save(transform, f"{save_path}_ReplaceMe_{loss}_{solver}_transform")
    
    # Final cleanup to free memory
    del model, a1, a2
    if accurate:
      del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return the path where the compressed model was saved
    return f"{save_path}_ReplaceMe_{loss}_{solver}"


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the cosine distance calculation from a configuration file.
    
    This is the main entry point that:
    1. Reads configuration from YAML file
    2. Loads precomputed layer distances
    3. Selects optimal blocks for pruning
    4. Applies cosine_dist to each selected block
    """
    parser = argparse.ArgumentParser(
        description="Run numerical solvers for linear transform estimation based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    # Load precomputed distances between layers (from distance profiling step)
    average_distances = torch.load(config['distances_path'], weights_only=False)
    
    # Select non-overlapping blocks of layers to prune based on distance analysis
    # This identifies which consecutive layers can be safely removed
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        config['layers_to_skip'],
        num_blocks=config['num_A'],
        merge_consecutive=config['merge_consecutive']
    )
    
    # Extract start and end indices for each block to be pruned
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    
    # Calculate cumulative layer counts for proper indexing after each pruning step
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    # Apply cosine_dist to each selected block sequentially
    # Each iteration removes a block and estimates its replacement transformation
    for i in range(len(selected_blocks)):
        path = cosine_dist(**config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
        # Update the model path for the next iteration (chaining transformations)
        config["model_path"] = path