"""Distance profiling module for analyzing transformer model layer distances.

This module provides functionality to compute and analyze distances between
transformer model layers to identify potential optimization opportunities.
"""

import argparse
import csv
import gc
import logging
from typing import Optional

import numpy as np
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (compute_block_distances, get_calib_dataloader,
                    get_last_non_padded_tokens, seed_all)


from huggingface_hub import login

# Token of the authorized huggingface account
login(token = '')


# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=(
        f"{Fore.CYAN}%(asctime)s "
        f"{Fore.YELLOW}[%(levelname)s] "
        f"{Fore.RESET}%(message)s"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

seed_all()

def profile_distances(
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
) -> None:
    """Profile distances between transformer model layers.

    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use for profiling
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers between compared blocks
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations (unused)
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save results (unused)
        min_distance_layer: index of the layer to start cut
        token: Authentication token for private models
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer,
    )

    # Special handling for individual layer analysis
    if layers_to_skip == 1:
        logging.info(f"{Fore.MAGENTA}Running individual layer importance analysis{Fore.RESET}")
        # For individual layers, we need num_hidden_layers entries
        all_distances = [[] for _ in range(model.config.num_hidden_layers)]
    else:
        # Original: for blocks spanning multiple layers
        all_distances = [[] for _ in range(model.config.num_hidden_layers - layers_to_skip)]

    # Process batches
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing Layer Importance{Fore.RESET}" if layers_to_skip == 1 
             else f"{Fore.GREEN}Computing Distances{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states  # [embedding, layer1, ..., layer32]
        
        # For individual layer analysis
        if layers_to_skip == 1:
            # Skip embedding layer (index 0), start from actual layers
            for i in range(1, len(hidden_states)):
                if i == 1:
                    # First layer: compare with embedding
                    h_in = hidden_states[0]
                    h_out = hidden_states[1]
                else:
                    # Other layers: compare with previous layer
                    h_in = hidden_states[i-1]
                    h_out = hidden_states[i]
                
                # Mask padded tokens
                mask = attention_mask.unsqueeze(-1)
                h_in_masked = h_in * mask
                h_out_masked = h_out * mask
                
                # Flatten and compute cosine similarity
                h_in_flat = h_in_masked.view(-1, h_in.size(-1))
                h_out_flat = h_out_masked.view(-1, h_out.size(-1))
                
                # Filter out padded tokens (all zeros)
                non_zero_mask = h_in_flat.sum(dim=1) != 0
                if non_zero_mask.sum() > 0:
                    h_in_valid = h_in_flat[non_zero_mask]
                    h_out_valid = h_out_flat[non_zero_mask]
                    
                    # Compute cosine distance (1 - similarity)
                    cos_sim = torch.nn.functional.cosine_similarity(h_in_valid, h_out_valid, dim=1)
                    cos_dist = 1.0 - cos_sim.mean().item()
                else:
                    cos_dist = 0.0
                
                # Store in correct index (layer i-1 since we skip embedding)
                all_distances[i-1].append(cos_dist)
        else:
            # Original block-wise analysis
            last_non_padded_hidden_states = get_last_non_padded_tokens(
                hidden_states, attention_mask
            )
            last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
            
            distances = compute_block_distances(
                last_non_padded_hidden_states, layers_to_skip
            )
            
            for i, distance in enumerate(distances):
                all_distances[i].append(distance)

        gc.collect()
        torch.cuda.empty_cache()

    # Calculate average distances/importance
    average_distances = [np.mean(block_distances) for block_distances in all_distances]
    
    # Special logging for individual layers
    if layers_to_skip == 1:
        logging.info(f"{Fore.CYAN}Layer Importance Analysis Complete{Fore.RESET}")
        logging.info(f"Total layers analyzed: {len(average_distances)}")
        
        # Find most and least important layers
        sorted_indices = sorted(range(len(average_distances)), 
                              key=lambda i: average_distances[i])
        
        logging.info(f"Most important layers (lowest distance): {sorted_indices[:5]}")
        logging.info(f"Least important layers (highest distance): {sorted_indices[-5:]}")
        
        # Save detailed analysis
        with open("layer_importance_analysis.csv", "w", newline="") as csvfile:
            fieldnames = ["layer_index", "importance_score", "redundancy_level"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, score in enumerate(average_distances):
                redundancy = "High" if score > 0.1 else "Medium" if score > 0.05 else "Low"
                writer.writerow({
                    "layer_index": i,
                    "importance_score": score,
                    "redundancy_level": redundancy
                })

    else:
        # Original block-wise reporting
        min_distance = float("inf")
        min_distance_layer = 0

        # Write results to CSV
        with open("layer_distances.csv", "w", newline="") as csvfile:
            fieldnames = ["block_start", "block_end", "average_distance"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, avg_dist in enumerate(average_distances):
                writer.writerow({
                    "block_start": i + 1,  # 1-based indexing
                    "block_end": i + 1 + layers_to_skip,
                    "average_distance": avg_dist,
                })

                if avg_dist < min_distance:
                    min_distance = avg_dist
                    min_distance_layer = i + 1

        logging.info(
            f"{Fore.GREEN}Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} "
            f"has the minimum average distance of {min_distance}. Consider examining "
            f"this layer more closely for potential optimization or removal.{Fore.RESET}"
        )
        logging.info(
            f"{Fore.GREEN}Layer distances written to layer_distances.csv{Fore.RESET}"
        )

    # Save distances and cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    torch.save(average_distances, "distances.pth")
    logging.info(f"{Fore.BLUE}Distances saved to distances.pth{Fore.RESET}")

def profile_distances_llava(
    model_path: str,
    dataset: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    use_4bit: bool = False,
    token: Optional[str] = None,
) -> None:
    """Profile distances for LLaVA model."""
    print(f"[DEBUG] Starting LLaVA distance profiling")
    print(f"[DEBUG] Model: {model_path}")
    print(f"[DEBUG] Dataset: {dataset}, Size: {dataset_size}")
    print(f"[DEBUG] Layers to skip: {layers_to_skip}")
    
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Using device_map: {device_map}")
    
    quantization_config = None
    if use_4bit:
        print("[DEBUG] Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    print("[DEBUG] Loading LLaVA model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    print(f"[DEBUG] Model loaded. Number of language layers: {model.config.text_config.num_hidden_layers}")
    
    # Load processor
    print("[DEBUG] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Get dataloader
    print("[DEBUG] Loading calibration data...")
    dataloader = get_calib_dataloader_llava(dataset, dataset_size, batch_size, processor)
    
    model.eval()
    
    # Initialize distance tracking
    num_layers = model.config.text_config.num_hidden_layers
    all_distances = [[] for _ in range(num_layers - layers_to_skip)]
    print(f"[DEBUG] Tracking distances for {len(all_distances)} layer pairs")
    
    # Process batches
    batch_count = 0
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing LLaVA Distances{Fore.RESET}",
        dynamic_ncols=True,
        colour="green"
    ):
        batch_count += 1
        print(f"\n[DEBUG] Processing batch {batch_count}")
        
        # Prepare inputs
        try:
            inputs = processor(
                text=[item['text'] for item in batch],
                images=[item['image'] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            print(f"[DEBUG] Input shape - input_ids: {inputs['input_ids'].shape}")
            print(f"[DEBUG] Input device: {inputs['input_ids'].device}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process batch {batch_count}: {e}")
            continue
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"[DEBUG] Got {len(outputs.hidden_states)} hidden states")
        
        # Get hidden states (language model states)
        hidden_states = outputs.hidden_states
        attention_mask = inputs["attention_mask"]
        
        # Get last non-padded tokens
        last_non_padded = get_last_non_padded_tokens(hidden_states, attention_mask)
        print(f"[DEBUG] Last non-padded tokens shape: {last_non_padded[0].shape}")
        
        # Compute distances
        distances = compute_block_distances(last_non_padded, layers_to_skip)
        print(f"[DEBUG] Computed {len(distances)} distances")
        
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)
        
        # Cleanup
        del outputs, hidden_states, inputs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate average distances
    print("\n[DEBUG] Computing average distances...")
    average_distances = [np.mean(block_distances) for block_distances in all_distances]
    
    min_distance = float("inf")
    min_distance_layer = 0
    
    # Write to CSV
    print("[DEBUG] Writing results to CSV...")
    with open("layer_distances_llava.csv", "w", newline="") as csvfile:
        fieldnames = ["block_start", "block_end", "average_distance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, avg_dist in enumerate(average_distances):
            writer.writerow({
                "block_start": i + 1,
                "block_end": i + 1 + layers_to_skip,
                "average_distance": avg_dist,
            })
            
            if avg_dist < min_distance:
                min_distance = avg_dist
                min_distance_layer = i + 1
    
    # Save distances
    torch.save(average_distances, "distances_llava.pth")
    print(f"[DEBUG] Distances saved to distances_llava.pth")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{Fore.GREEN}[RESULT] Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} "
          f"has minimum distance of {min_distance:.6f}{Fore.RESET}")
    print(f"{Fore.GREEN}Results written to layer_distances_llava.csv{Fore.RESET}")