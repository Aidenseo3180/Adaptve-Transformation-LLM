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
        tokenizer,
    )

    # Initialize distance tracking
    all_distances = [
        [] for _ in range(model.config.num_hidden_layers - layers_to_skip)
    ]

    # Process batches
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing Distances{Fore.RESET}",
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

        # Get non-padded tokens and compute distances
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states
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

    # Calculate and save average distances
    average_distances = [np.mean(block_distances) for block_distances in all_distances]
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

    # Save distances and log results
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    torch.save(average_distances, "distances.pth")
    logging.info(
        f"{Fore.GREEN}Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} "
        f"has the minimum average distance of {min_distance}. Consider examining "
        f"this layer more closely for potential optimization or removal.{Fore.RESET}"
    )
    logging.info(
        f"{Fore.GREEN}Layer distances written to layer_distances.csv{Fore.RESET}"
    )


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run distance profiling from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run distance analysis based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    profile_distances(**config)