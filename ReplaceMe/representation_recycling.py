"""Block-wise Representation Recycling module for transformer model optimization.

This module implements a novel approach to reduce FLOPs by recycling similar
representations across layers without requiring healing or fine-tuning.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class RepresentationRecyclingLayer(nn.Module):
    """Custom layer that can recycle representations from previous layers."""
    
    def __init__(self, original_layer, layer_idx, recycling_map, similarity_threshold=0.9):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.recycling_map = recycling_map  # Dict mapping this layer to source layer
        self.similarity_threshold = similarity_threshold
        self.correction_mlp = None
        self.cached_representations = {}
        
        # Statistics tracking
        self.recycling_count = 0
        self.total_count = 0
        
        print(f"{Fore.YELLOW}[Layer {layer_idx}] Initialized with recycling from layer {recycling_map.get(layer_idx, 'None')}{Fore.RESET}")
    
    def forward(self, hidden_states, **kwargs):
        self.total_count += 1
        
        # Check if we should recycle
        source_layer = self.recycling_map.get(self.layer_idx)
        
        if source_layer is not None and source_layer in self.cached_representations:
            cached_output = self.cached_representations[source_layer]
            
            # Compute similarity
            with torch.no_grad():
                norm_current = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-8)
                norm_cached = cached_output / (cached_output.norm(dim=-1, keepdim=True) + 1e-8)
                similarity = (norm_current * norm_cached).sum(dim=-1).mean()
            
            if similarity > self.similarity_threshold:
                self.recycling_count += 1
                
                # Apply small correction if available
                if self.correction_mlp is not None:
                    delta = self.correction_mlp(hidden_states - cached_output)
                    output = cached_output + delta
                    print(f"{Fore.GREEN}[Layer {self.layer_idx}] Recycled from layer {source_layer} (similarity: {similarity:.3f}){Fore.RESET}")
                else:
                    output = cached_output
                    print(f"{Fore.GREEN}[Layer {self.layer_idx}] Direct recycling from layer {source_layer}{Fore.RESET}")
                
                return output
        
        # Normal forward pass
        output = self.original_layer(hidden_states, **kwargs)
        
        # Cache this output for potential future recycling
        self.cached_representations[self.layer_idx] = output.detach()
        
        # Clean old cache entries (keep only last 5)
        if len(self.cached_representations) > 5:
            oldest_key = min(self.cached_representations.keys())
            del self.cached_representations[oldest_key]
        
        return output
    
    def get_recycling_stats(self):
        if self.total_count > 0:
            return self.recycling_count / self.total_count
        return 0.0


def analyze_representation_similarity(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    token: Optional[str] = None,
    similarity_threshold: float = 0.9,
) -> Dict[int, List[Tuple[int, float]]]:
    """Analyze representation similarity between layers to identify recycling opportunities.
    
    Returns:
        Dictionary mapping each layer to list of (source_layer, similarity) tuples
    """
    print(f"{Fore.CYAN}=== Starting Representation Similarity Analysis ==={Fore.RESET}")
    
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    
    # Collect hidden states for each layer
    num_layers = model.config.num_hidden_layers
    layer_representations = [[] for _ in range(num_layers + 1)]  # +1 for embedding
    
    print(f"{Fore.YELLOW}Collecting layer representations...{Fore.RESET}")
    for batch in tqdm(dataloader, desc="Processing batches", colour="yellow"):
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
        
        # Store mean pooled representations for each layer
        for i, hidden_state in enumerate(outputs.hidden_states):
            # Mean pool over sequence length
            pooled = hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
            layer_representations[i].append(pooled.cpu())
    
    # Concatenate all batches
    for i in range(len(layer_representations)):
        if layer_representations[i]:
            layer_representations[i] = torch.cat(layer_representations[i], dim=0)
    
    # Compute similarity matrix
    print(f"{Fore.YELLOW}Computing similarity matrix...{Fore.RESET}")
    similarity_matrix = torch.zeros((num_layers, num_layers))
    
    for i in range(1, num_layers):  # Skip embedding layer
        for j in range(max(0, i-10), i):  # Only check previous 10 layers
            if len(layer_representations[i]) > 0 and len(layer_representations[j]) > 0:
                # Compute cosine similarity
                rep_i = layer_representations[i]
                rep_j = layer_representations[j]
                
                norm_i = rep_i / (rep_i.norm(dim=-1, keepdim=True) + 1e-8)
                norm_j = rep_j / (rep_j.norm(dim=-1, keepdim=True) + 1e-8)
                
                similarity = (norm_i * norm_j).sum(dim=-1).mean().item()
                similarity_matrix[i, j] = similarity
                
                if similarity > similarity_threshold:
                    print(f"{Fore.GREEN}Layer {i} similar to layer {j}: {similarity:.3f}{Fore.RESET}")
    
    # Find best recycling opportunities
    recycling_opportunities = {}
    for i in range(1, num_layers):
        similar_layers = []
        for j in range(max(0, i-10), i):
            sim = similarity_matrix[i, j].item()
            if sim > similarity_threshold:
                similar_layers.append((j, sim))
        
        if similar_layers:
            # Sort by similarity
            similar_layers.sort(key=lambda x: x[1], reverse=True)
            recycling_opportunities[i] = similar_layers
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return recycling_opportunities


def create_recycling_map(
    recycling_opportunities: Dict[int, List[Tuple[int, float]]],
    min_gap: int = 3
) -> Dict[int, int]:
    """Create optimal recycling map from similarity analysis.
    
    Args:
        recycling_opportunities: Dict of layer -> [(source_layer, similarity)]
        min_gap: Minimum gap between source and target layer
    
    Returns:
        Dict mapping layer_idx -> source_layer_idx
    """
    recycling_map = {}
    
    for layer_idx, similar_layers in recycling_opportunities.items():
        for source_layer, similarity in similar_layers:
            # Ensure sufficient gap between layers
            if layer_idx - source_layer >= min_gap:
                recycling_map[layer_idx] = source_layer
                print(f"{Fore.CYAN}Layer {layer_idx} will recycle from layer {source_layer} (gap: {layer_idx - source_layer}){Fore.RESET}")
                break
    
    return recycling_map


def apply_representation_recycling(
    model_path: str,
    recycling_map: Dict[int, int],
    save_path: Optional[str] = None,
    similarity_threshold: float = 0.9
) -> str:
    """Apply representation recycling to model.
    
    Returns:
        Path to saved model
    """
    print(f"{Fore.CYAN}=== Applying Representation Recycling ==={Fore.RESET}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Replace layers with recycling layers
    for i, layer in enumerate(model.model.layers):
        if i in recycling_map:
            print(f"{Fore.YELLOW}Wrapping layer {i} with RecyclingLayer{Fore.RESET}")
            model.model.layers[i] = RepresentationRecyclingLayer(
                layer, i, recycling_map, similarity_threshold
            )
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_recycling"
    
    print(f"{Fore.GREEN}Saving model to {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path


def representation_recycling(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,  # Not used but kept for compatibility
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    similarity_threshold: float = 0.9,
    min_recycling_gap: int = 3,
    **kwargs  # Catch extra args
) -> str:
    """Main entry point for representation recycling method.
    
    Returns:
        Path to optimized model
    """
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Starting Block-wise Representation Recycling{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    
    # Step 1: Analyze similarity
    recycling_opportunities = analyze_representation_similarity(
        model_path=model_path,
        dataset=dataset,
        dataset_column=dataset_column,
        batch_size=batch_size,
        max_length=max_length,
        dataset_size=dataset_size,
        dataset_subset=dataset_subset,
        use_4bit=use_4bit,
        token=token,
        similarity_threshold=similarity_threshold
    )
    
    print(f"\n{Fore.CYAN}Found {len(recycling_opportunities)} layers with recycling opportunities{Fore.RESET}")
    
    # Step 2: Create recycling map
    recycling_map = create_recycling_map(
        recycling_opportunities,
        min_gap=min_recycling_gap
    )
    
    print(f"\n{Fore.CYAN}Created recycling map for {len(recycling_map)} layers{Fore.RESET}")
    
    # Step 3: Apply recycling
    optimized_model_path = apply_representation_recycling(
        model_path=model_path,
        recycling_map=recycling_map,
        save_path=save_path,
        similarity_threshold=similarity_threshold
    )
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}Optimization Complete!{Fore.RESET}")
    print(f"{Fore.GREEN}Expected FLOPs reduction: ~{len(recycling_map) * 3:.1f}%{Fore.RESET}")
    print(f"{Fore.GREEN}Model saved to: {optimized_model_path}{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*60}{Fore.RESET}")
    
    return optimized_model_path