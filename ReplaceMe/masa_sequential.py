import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
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


class SharedAtomDictionary(nn.Module):
    """Shared dictionary of atoms for multiple linear transformations."""
    
    def __init__(self, hidden_size: int, num_atoms: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_atoms = num_atoms
        
        # Initialize shared atoms with Xavier initialization
        self.atoms = nn.Parameter(torch.randn(num_atoms, hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.atoms)
        
    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coefficients: (num_atoms,) tensor of mixing coefficients
        Returns:
            Linear transformation matrix of shape (hidden_size, hidden_size)
        """
        return torch.einsum('a,ahd->hd', coefficients, self.atoms)


def learn_shared_atoms(activations_list: List[Tuple[torch.Tensor, torch.Tensor]], 
                      hidden_size: int, 
                      num_atoms: int,
                      device: str = "cuda") -> SharedAtomDictionary:
    """
    Learn shared dictionary atoms from all activation pairs.
    
    Args:
        activations_list: List of (input_activations, target_activations) pairs
        hidden_size: Hidden dimension size
        num_atoms: Number of atoms in dictionary
        device: Device to run on
    
    Returns:
        Trained SharedAtomDictionary
    """
    dictionary = SharedAtomDictionary(hidden_size, num_atoms).to(device)
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=1e-3)
    
    logging.info(f"{Fore.GREEN}Learning shared atoms dictionary with {num_atoms} atoms{Fore.RESET}")
    
    # Prepare all data and convert to float32 for computation
    all_inputs = torch.cat([pair[0].to(torch.float32) for pair in activations_list], dim=0).to(device)
    all_targets = torch.cat([pair[1].to(torch.float32) for pair in activations_list], dim=0).to(device)
    
    # Training loop for atom dictionary
    for epoch in tqdm(range(50), desc="Learning Shared Atoms"):
        optimizer.zero_grad()
        
        # Sample random coefficients for this epoch
        batch_size = min(1024, all_inputs.shape[0])
        indices = torch.randperm(all_inputs.shape[0])[:batch_size]
        
        batch_inputs = all_inputs[indices]
        batch_targets = all_targets[indices]
        
        # Learn optimal coefficients for this batch
        coefficients = torch.randn(num_atoms, requires_grad=True, device=device)
        coeff_optimizer = torch.optim.Adam([coefficients], lr=1e-2)
        
        # Inner loop to optimize coefficients
        for _ in range(10):
            coeff_optimizer.zero_grad()
            transform_matrix = dictionary(torch.softmax(coefficients, dim=0))
            transformed = batch_inputs @ transform_matrix.T
            
            # Cosine distance loss
            transformed_norm = transformed / transformed.norm(dim=1, keepdim=True)
            target_norm = batch_targets / batch_targets.norm(dim=1, keepdim=True)
            cosine_loss = 1 - (transformed_norm * target_norm).sum(dim=1).mean()
            
            cosine_loss.backward(retain_graph=True)
            coeff_optimizer.step()
        
        # Update atoms based on optimal coefficients
        transform_matrix = dictionary(torch.softmax(coefficients.detach(), dim=0))
        transformed = batch_inputs @ transform_matrix.T
        
        transformed_norm = transformed / transformed.norm(dim=1, keepdim=True)
        target_norm = batch_targets / batch_targets.norm(dim=1, keepdim=True)
        atom_loss = 1 - (transformed_norm * target_norm).sum(dim=1).mean()
        
        atom_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, Atom Loss: {atom_loss.item():.4f}")
    
    return dictionary


def sequential_optimization(activations_list: List[Tuple[torch.Tensor, torch.Tensor]],
                          shared_atoms: SharedAtomDictionary,
                          sequential_alpha: float = 0.8,
                          device: str = "cuda") -> List[torch.Tensor]:
    """
    Sequentially optimize coefficients for each transformation.
    
    Args:
        activations_list: List of (input_activations, target_activations) pairs
        shared_atoms: Pre-learned shared atom dictionary
        sequential_alpha: Weight for sequential dependency
        device: Device to run on
    
    Returns:
        List of coefficient vectors for each transformation
    """
    coefficients_list = []
    previous_output = None
    
    logging.info(f"{Fore.GREEN}Starting sequential optimization with alpha={sequential_alpha}{Fore.RESET}")
    
    for i, (input_activations, target_activations) in enumerate(activations_list):
        logging.info(f"Optimizing transformation {i+1}/{len(activations_list)}")
        
        input_activations = input_activations.to(device)
        target_activations = target_activations.to(device)
        
        # Initialize coefficients
        coefficients = torch.randn(shared_atoms.num_atoms, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([coefficients], lr=1e-3)
        
        # If this is not the first transformation, consider previous output
        if previous_output is not None and sequential_alpha > 0:
            # Adjust input based on previous transformation output
            adjusted_input = sequential_alpha * input_activations + (1 - sequential_alpha) * previous_output
        else:
            adjusted_input = input_activations
        
        # Optimize coefficients for this transformation
        for epoch in tqdm(range(100), desc=f"Transform {i+1} Optimization"):
            optimizer.zero_grad()
            
            # Create transformation matrix from coefficients
            coeff_softmax = torch.softmax(coefficients, dim=0)
            transform_matrix = shared_atoms(coeff_softmax)
            
            # Apply transformation
            transformed = adjusted_input @ transform_matrix.T
            
            # Cosine distance loss
            transformed_norm = transformed / (transformed.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = target_activations / (target_activations.norm(dim=1, keepdim=True) + 1e-8)
            cosine_loss = 1 - (transformed_norm * target_norm).sum(dim=1).mean()
            
            # L2 regularization on coefficients
            l2_reg = 0.01 * (coefficients ** 2).sum()
            
            total_loss = cosine_loss + l2_reg
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logging.info(f"Transform {i+1}, Epoch {epoch}, Loss: {total_loss.item():.4f}")
        
        # Store optimized coefficients
        final_coefficients = torch.softmax(coefficients.detach(), dim=0)
        coefficients_list.append(final_coefficients)
        
        # Compute output for next iteration
        final_transform = shared_atoms(final_coefficients)
        previous_output = adjusted_input @ final_transform.T
        
        # Clean up
        del coefficients, optimizer
        torch.cuda.empty_cache()
    
    return coefficients_list


def masa_sequential(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = 50000,
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
    num_shared_atoms: int = 32,
    sequential_alpha: float = 0.8,
) -> str:
    """
    MASA + Sequential Optimization method for transformer block replacement.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Size limit for dataset (default: 50000)
        dataset_subset: Subset of dataset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        token: Authentication token
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to distance metrics
        num_A: Number of transformations
        merge_consecutive: Whether to merge consecutive blocks
        num_shared_atoms: Number of shared atoms in dictionary
        sequential_alpha: Weight for sequential dependency
    
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
    
    # Register hooks for MLP activations
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Collect activations for all blocks to be replaced
    logging.info(f"{Fore.GREEN}Collecting activations for MASA Sequential method{Fore.RESET}")
    
    activations_data = []
    total_samples = dataset_size * max_length
    
    # Preallocate tensors for different transformation blocks
    block_activations = {
        'input': torch.empty((total_samples, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'target': torch.empty((total_samples, hidden_size), dtype=torch.bfloat16, device='cpu'),
        'mlp': torch.empty((total_samples, hidden_size), dtype=torch.bfloat16, device='cpu')
    }
    
    cnt = 0
    for batch in tqdm(dataloader, desc=Fore.RED + "Gathering Activations" + Fore.RESET):
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
        
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
        mlp_states = [mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)]
        
        # Get activations for the specific blocks
        input_activations = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        target_activations = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        mlp_activations_tensor = mlp_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        
        batch_size_actual = input_activations.shape[0]
        
        block_activations['input'][cnt:cnt+batch_size_actual] = input_activations.cpu()
        block_activations['target'][cnt:cnt+batch_size_actual] = target_activations.cpu()
        block_activations['mlp'][cnt:cnt+batch_size_actual] = mlp_activations_tensor.cpu()
        
        cnt += batch_size_actual
        
        if cnt >= total_samples:
            break
    
    # Trim to actual size
    for key in block_activations:
        block_activations[key] = block_activations[key][:cnt]
    
    # Prepare activation pairs for MASA
    # For sequential optimization, we need multiple transformation pairs
    activation_pairs = [
        (block_activations['mlp'], block_activations['target'] - block_activations['input'])
    ]
    
    logging.info(f"{Fore.GREEN}Learning shared atoms dictionary{Fore.RESET}")
    
    # Learn shared atoms dictionary
    shared_atoms = learn_shared_atoms(
        activation_pairs, 
        hidden_size, 
        num_shared_atoms,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logging.info(f"{Fore.GREEN}Performing sequential optimization{Fore.RESET}")
    
    # Perform sequential optimization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coefficients_list = sequential_optimization(
        activation_pairs,
        shared_atoms,
        sequential_alpha,
        device
    )
    
    # Create final transformation matrix
    final_coefficients = coefficients_list[0]  # For single block replacement
    final_transform = shared_atoms(final_coefficients).cpu().to(torch.float64)
    
    # Clean up
    for hook in hooks:
        hook.remove()
    del model, shared_atoms
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_masa_seq"
        ).replace("/", "_")
    
    # Apply transformation to down_proj layer
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (final_transform.T @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    final_save_path = f"{save_path}_MASA_Sequential"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    # Save shared atoms and coefficients for analysis
    torch.save({
        'shared_atoms': shared_atoms.cpu() if hasattr(shared_atoms, 'cpu') else shared_atoms,
        'coefficients': final_coefficients.cpu(),
        'num_atoms': num_shared_atoms,
        'sequential_alpha': sequential_alpha
    }, f"{final_save_path}_masa_data.pth")
    
    logging.info(f"{Fore.GREEN}MASA Sequential optimization completed. Model saved to {final_save_path}{Fore.RESET}")
    
    # Final cleanup
    del model, block_activations
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_save_path

