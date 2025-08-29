"""
Identity-Convergence Guided Block Replacement Method
Integration with existing ReplaceMe pipeline
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple
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

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def compute_identity_convergence_score(model, layer_idx: int) -> float:
    """
    Compute identity convergence score for a transformer layer.
    
    Args:
        model: Transformer model
        layer_idx: Index of the layer to analyze
    
    Returns:
        Identity convergence score (lower = more identity-like)
    """
    try:
        # Try to access layer - this might vary by architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style models (e.g., GPT-2, Falcon)
            layer = model.transformer.h[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LLaMA-style models
            layer = model.model.layers[layer_idx]
        else:
            logging.warning("Unknown model architecture for identity convergence analysis")
            return float('inf')  # Return high score to avoid selection
        
        # Access attention weights - architecture-dependent
        identity_distances = []
        
        # Try to find value and projection weights
        attention_module = None
        if hasattr(layer, 'attention') or hasattr(layer, 'attn'):
            attention_module = getattr(layer, 'attention', None) or getattr(layer, 'attn', None)
        elif hasattr(layer, 'self_attn'):
            attention_module = layer.self_attn
            
        if attention_module is None:
            logging.warning(f"Cannot find attention module in layer {layer_idx}")
            return float('inf')
        
        # Try different possible names for value and projection weights
        value_weight = None
        proj_weight = None
        
        # Common attribute names for value projection
        for attr_name in ['v_proj', 'value', 'w_v', 'V']:
            if hasattr(attention_module, attr_name):
                value_weight = getattr(attention_module, attr_name).weight
                break
                
        # Common attribute names for output projection  
        for attr_name in ['o_proj', 'out_proj', 'proj', 'w_o', 'dense']:
            if hasattr(attention_module, attr_name):
                proj_weight = getattr(attention_module, attr_name).weight
                break
        
        # Compute identity distances
        if value_weight is not None:
            # Handle rectangular matrices by using minimum dimension
            min_dim = min(value_weight.size(0), value_weight.size(1))
            if value_weight.size(0) == value_weight.size(1):
                # Square matrix case
                identity = torch.eye(value_weight.size(0), device=value_weight.device, dtype=value_weight.dtype)
                identity_dist_v = torch.norm(value_weight - identity).item()
            else:
                # Rectangular matrix case - compare to identity-like structure
                # For rectangular matrices, we measure how close the matrix is to having
                # identity-like properties in its square submatrix
                square_part = value_weight[:min_dim, :min_dim]
                identity = torch.eye(min_dim, device=value_weight.device, dtype=value_weight.dtype)
                identity_dist_v = torch.norm(square_part - identity).item()
            identity_distances.append(identity_dist_v)
            
        if proj_weight is not None:
            min_dim = min(proj_weight.size(0), proj_weight.size(1))
            if proj_weight.size(0) == proj_weight.size(1):
                identity = torch.eye(proj_weight.size(0), device=proj_weight.device, dtype=proj_weight.dtype)
                identity_dist_p = torch.norm(proj_weight - identity).item()
            else:
                square_part = proj_weight[:min_dim, :min_dim]
                identity = torch.eye(min_dim, device=proj_weight.device, dtype=proj_weight.dtype)
                identity_dist_p = torch.norm(square_part - identity).item()
            identity_distances.append(identity_dist_p)
        
        if not identity_distances:
            logging.warning(f"Could not compute identity convergence for layer {layer_idx}")
            return float('inf')
            
        # Return average identity distance
        return sum(identity_distances) / len(identity_distances)
        
    except Exception as e:
        logging.warning(f"Error computing identity convergence for layer {layer_idx}: {e}")
        return float('inf')


def compute_identity_convergence_distances(model, layers_to_skip: int) -> List[float]:
    """
    Compute identity convergence scores for consecutive layer blocks.
    
    Args:
        model: Transformer model
        layers_to_skip: Number of layers to skip (block size)
    
    Returns:
        List of identity convergence scores for each possible block
    """
    num_layers = model.config.num_hidden_layers
    convergence_distances = []
    
    for i in range(num_layers - layers_to_skip):
        # Compute average identity convergence for the block
        block_scores = []
        for j in range(layers_to_skip + 1):  # Include start and end layers
            if i + j < num_layers:
                score = compute_identity_convergence_score(model, i + j)
                block_scores.append(score)
        
        if block_scores:
            avg_convergence = sum(block_scores) / len(block_scores)
            convergence_distances.append(avg_convergence)
            logging.info(f"Block {i}-{i+layers_to_skip}: Identity convergence score = {avg_convergence:.4f}")
        else:
            convergence_distances.append(float('inf'))
    
    return convergence_distances


def identity_convergence_method(
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
    accurate: bool = False
) -> str:
    """
    Identity-convergence guided block replacement method.
    
    This method selects blocks for replacement based on how close their 
    value/projection parameters are to identity matrices, rather than 
    using cosine distance between activations.
    
    Args:
        Same as cosine_dist function
    
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

    # Load model for identity convergence analysis
    logging.info(f"{Fore.GREEN}Loading model for identity convergence analysis...{Fore.RESET}")
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
    
    # Step 1: Compute identity convergence distances instead of activation distances
    logging.info(f"{Fore.GREEN}Computing identity convergence scores...{Fore.RESET}")
    identity_distances = compute_identity_convergence_distances(model, layers_to_skip)
    
    # Save identity distances for analysis
    torch.save(identity_distances, distances_path.replace('.pth', '_identity.pth'))
    logging.info(f"Identity convergence scores saved to {distances_path.replace('.pth', '_identity.pth')}")
    
    # Step 2: Select best block based on identity convergence
    min_distance = min(identity_distances)
    optimal_start_idx = identity_distances.index(min_distance) + 1  # 1-based indexing
    optimal_end_idx = optimal_start_idx + layers_to_skip
    
    logging.info(f"{Fore.GREEN}Selected block based on identity convergence: "
                f"layers {optimal_start_idx}-{optimal_end_idx} "
                f"(convergence score: {min_distance:.4f}){Fore.RESET}")
    
    # Override start_id and end_id with identity-convergence guided selection
    start_id = optimal_start_idx
    end_id = optimal_end_idx
    
    # Step 3: Proceed with standard ReplaceMe pipeline for activation gathering
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
        a3 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    logging.info(f"{Fore.GREEN}Gathering activations for identity-selected block "
                f"{start_id}-{end_id}...{Fore.RESET}")
    
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations (Identity-Guided)" + Fore.RESET,
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
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
        
    # Step 4: Estimate transformation using standard methods
    logging.info(f"{Fore.GREEN}Estimating linear transformation...{Fore.RESET}")
    if solver == "adam":
        transform = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 5: Apply transformation and save model
    logging.info(f"{Fore.GREEN}Applying transformation and saving model...{Fore.RESET}")
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
    
    # Apply transformation
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (transform.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    final_save_path = f"{save_path}_IdentityGuided_{loss}_{solver}"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    if save_transform_only:
        torch.save(transform, f"{final_save_path}_transform")
    
    # Final cleanup
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    logging.info(f"{Fore.GREEN}Identity-guided compression completed. "
                f"Model saved to {final_save_path}{Fore.RESET}")
    
    return final_save_path


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the identity-convergence guided method from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run identity-convergence guided block replacement."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    # Use identity distances if available, otherwise compute them
    if 'distances_path' not in config or config['distances_path'] is None:
        config['distances_path'] = './distances.pth'
    
    # For multiple block replacement, use select_non_overlapping_blocks
    if config.get('num_A', 1) > 1:
        # Load or compute identity distances
        identity_distances_path = config['distances_path'].replace('.pth', '_identity.pth')
        if os.path.exists(identity_distances_path):
            identity_distances = torch.load(identity_distances_path)
        else:
            # Need to compute distances first
            logging.info("Computing identity convergence distances...")
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                config['model_path'],
                device_map=device_map,
                output_hidden_states=True,
                token=config.get('token')
            )
            identity_distances = compute_identity_convergence_distances(model, config['layers_to_skip'])
            torch.save(identity_distances, identity_distances_path)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        # Select non-overlapping blocks based on identity convergence
        selected_blocks = select_non_overlapping_blocks(
            identity_distances,
            config['layers_to_skip'],
            num_blocks=config['num_A'],
            merge_consecutive=config['merge_consecutive']
        )
        
        start_ids = sorted([x[0] for x in selected_blocks])
        end_ids = sorted([x[1] for x in selected_blocks])
        num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
        num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
        
        for i in range(len(selected_blocks)):
            path = identity_convergence_method(
                **config, 
                start_id=start_ids[i], 
                end_id=end_ids[i], 
                num_layer=num_layers[i]
            )
            config["model_path"] = path
    else:
        # Single block replacement
        identity_convergence_method(**config)