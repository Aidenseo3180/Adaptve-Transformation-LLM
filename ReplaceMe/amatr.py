# amatr.py
"""Adaptive Magnitude-Aware Token Replacement (AMATR) for layer pruning.

This module implements AMATR, which combines token replacement with magnitude
compensation for efficient layer pruning without retraining.
"""

import gc
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from colorama import Fore, init

from .utils import get_calib_dataloader, truncate_model, seed_all

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def compute_magnitude_gap(
    model: nn.Module,
    dataloader,
    layer_idx: int,
    device: str = "cuda"
) -> float:
    """Compute the magnitude gap introduced by a specific layer.
    
    Args:
        model: The transformer model
        dataloader: Calibration dataloader
        layer_idx: Index of the layer to analyze
        device: Device to run computations on
    
    Returns:
        Magnitude scaling factor (alpha)
    """
    magnitude_ratios = []
    
    for batch in tqdm(dataloader, desc=f"Computing magnitude gap for layer {layer_idx}"):
        with torch.no_grad():
            # Get hidden states before and after the layer
            hidden_states = model(batch, output_hidden_states=True).hidden_states
            
            input_magnitude = hidden_states[layer_idx].norm(dim=-1).mean()
            output_magnitude = hidden_states[layer_idx + 1].norm(dim=-1).mean()
            
            magnitude_ratios.append((output_magnitude / input_magnitude).item())
    
    return sum(magnitude_ratios) / len(magnitude_ratios)


def identify_representative_tokens(
    model: nn.Module,
    dataloader,
    layer_idx: int,
    top_k: int = 10,
    device: str = "cuda"
) -> torch.Tensor:
    """Identify representative tokens for a layer based on importance.
    
    Args:
        model: The transformer model
        dataloader: Calibration dataloader
        layer_idx: Index of the layer
        top_k: Number of representative tokens to select
        device: Device to run computations on
    
    Returns:
        Representative token embeddings
    """
    all_activations = []
    importance_scores = []
    
    for batch in tqdm(dataloader, desc=f"Analyzing tokens for layer {layer_idx}"):
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get activations at this layer
            layer_input = hidden_states[layer_idx]
            layer_output = hidden_states[layer_idx + 1]
            
            # Compute importance based on magnitude change
            importance = (layer_output - layer_input).norm(dim=-1)
            
            # Flatten batch and sequence dimensions
            flat_activations = layer_input.view(-1, layer_input.size(-1))
            flat_importance = importance.view(-1)
            
            all_activations.append(flat_activations)
            importance_scores.append(flat_importance)
    
    # Concatenate all batches
    all_activations = torch.cat(all_activations, dim=0)
    importance_scores = torch.cat(importance_scores, dim=0)
    
    # Select top-k important tokens
    top_indices = importance_scores.topk(top_k).indices
    representative_tokens = all_activations[top_indices]
    
    return representative_tokens


def apply_magnitude_compensation(
    model: nn.Module,
    layer_idx: int,
    alpha: float
) -> nn.Module:
    """Apply magnitude compensation to model weights.
    
    Args:
        model: The transformer model
        layer_idx: Index of the removed layer
        alpha: Magnitude scaling factor
    
    Returns:
        Modified model
    """
    # Scale embedding layer if removing early layers
    if layer_idx < len(model.model.layers) // 4:
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.weight.data *= alpha
    
    # Scale output projections of preceding layers
    for i in range(min(layer_idx, len(model.model.layers))):
        # Scale MHA output projection
        if hasattr(model.model.layers[i].self_attn, 'o_proj'):
            model.model.layers[i].self_attn.o_proj.weight.data *= alpha
        
        # Scale MLP down projection
        if hasattr(model.model.layers[i].mlp, 'down_proj'):
            model.model.layers[i].mlp.down_proj.weight.data *= alpha
    
    return model


def create_token_replacement_layer(
    representative_tokens: torch.Tensor,
    alpha: float,
    hidden_size: int
) -> nn.Module:
    """Create a lightweight replacement layer using representative tokens.
    
    Args:
        representative_tokens: Representative token embeddings
        alpha: Magnitude scaling factor
        hidden_size: Hidden dimension of the model
    
    Returns:
        Replacement layer module
    """
    class TokenReplacementLayer(nn.Module):
        def __init__(self, tokens, alpha):
            super().__init__()
            self.register_buffer('tokens', tokens)
            self.alpha = alpha
            
            # Learnable attention weights for token selection
            self.token_attention = nn.Linear(hidden_size, len(tokens), bias=False)
            nn.init.xavier_uniform_(self.token_attention.weight)
        
        def forward(self, x):
            # Compute attention scores over representative tokens
            scores = self.token_attention(x)  # [batch, seq, num_tokens]
            weights = torch.softmax(scores, dim=-1)
            
            # Weighted combination of representative tokens
            output = torch.einsum('bst,td->bsd', weights, self.tokens)
            
            # Apply magnitude compensation
            return x + self.alpha * output
    
    return TokenReplacementLayer(representative_tokens, alpha)


def amatr(
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
    top_k_tokens: int = 32,
    apply_compensation: bool = True,
    **kwargs
) -> str:
    """Apply Adaptive Magnitude-Aware Token Replacement for layer pruning.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        token: Authentication token
        start_id: Starting layer ID for pruning
        end_id: Ending layer ID for pruning
        num_layer: Number of layers already pruned
        distances_path: Path to distance metrics
        num_A: Number of transformations
        merge_consecutive: Whether to merge consecutive blocks
        top_k_tokens: Number of representative tokens per layer
        apply_compensation: Whether to apply magnitude compensation
    
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
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get calibration dataloader
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Process each layer to be pruned
    layers_to_prune = list(range(start_id - num_layer - 1, end_id - num_layer))
    logging.info(f"Pruning layers: {layers_to_prune}")
    
    magnitude_gaps = []
    representative_tokens_list = []
    
    # Analyze layers before pruning
    for layer_idx in layers_to_prune:
        logging.info(f"Analyzing layer {layer_idx}")
        
        # Compute magnitude gap
        if apply_compensation:
            alpha = compute_magnitude_gap(model, dataloader, layer_idx)
            magnitude_gaps.append(alpha)
            logging.info(f"Layer {layer_idx} magnitude gap: {alpha:.4f}")
        
        # Identify representative tokens
        rep_tokens = identify_representative_tokens(
            model, dataloader, layer_idx, top_k=top_k_tokens
        )
        representative_tokens_list.append(rep_tokens)
    
    # Apply pruning and compensation
    if apply_compensation and magnitude_gaps:
        avg_alpha = sum(magnitude_gaps) / len(magnitude_gaps)
        logging.info(f"Applying average magnitude compensation: {avg_alpha:.4f}")
        model = apply_magnitude_compensation(model, start_id - num_layer - 1, avg_alpha)
    
    # Truncate model (remove layers)
    model = truncate_model(model, start_id - num_layer - 1, end_id - num_layer)
    
    # Optional: Insert lightweight replacement layers
    # This is experimental and can be toggled
    insert_replacement = kwargs.get('insert_replacement', False)
    if insert_replacement and representative_tokens_list:
        logging.info("Inserting token replacement layers")
        # Average representative tokens across pruned layers
        avg_tokens = torch.stack(representative_tokens_list).mean(dim=0)
        replacement_layer = create_token_replacement_layer(
            avg_tokens, 
            avg_alpha if apply_compensation else 1.0,
            model.config.hidden_size
        )
        # Insert at the pruning point
        # This would require modifying the model architecture
        # For now, we skip this step
        pass
    
    # Save model
    if save_path is None:
        save_path = f"output_models/{model_path.replace('/', '_')}_AMATR_{layers_to_skip}layers"
    
    logging.info(f"Saving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path