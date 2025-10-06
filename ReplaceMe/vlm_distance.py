# ============================================================
# vlm_distance.py (새 파일 생성)
# ============================================================

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
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

from .utils import (
    compute_block_distances,
    get_vlm_calib_dataloader,
    setup_vlm_processor,
    get_vlm_layers,
    get_last_non_padded_tokens,
    seed_all
)

init(autoreset=True)

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


def vlm_profile_distances(
    model_path: str,
    image_dir: str = "train2014",
    batch_size: int = 4,
    max_length: int = 512,
    layers_to_skip: int = 8,
    dataset_size: Optional[int] = None,
    use_4bit: bool = False,
    token: Optional[str] = None,
) -> None:
    """Profile distances between VLM language model layers.
    
    Args:
        model_path: Path to VLM model
        image_dir: Image directory path
        batch_size: Batch size
        max_length: Max sequence length
        layers_to_skip: Layers to skip
        dataset_size: Dataset size limit
        use_4bit: Use 4-bit quantization
        token: HuggingFace token
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

    print(f"{Fore.GREEN}Loading VLM for distance profiling{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )

    processor = setup_vlm_processor(model_path)
    layers, num_hidden_layers = get_vlm_layers(model)
    
    print(f"{Fore.GREEN}VLM loaded - {num_hidden_layers} language model layers{Fore.RESET}")

    model.eval()
    dataloader = get_vlm_calib_dataloader(
        image_dir,
        dataset_size,
        batch_size,
        processor
    )

    # Distance 추적
    all_distances = [
        [] for _ in range(num_hidden_layers - layers_to_skip)
    ]

    # 배치 처리
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing VLM Distances{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = model(**inputs)

        # Attention mask와 hidden states
        attention_mask = batch["attention_mask"].to(model.device)
        hidden_states = outputs.hidden_states
        
        # Last non-padded tokens
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

    # 평균 distance 계산
    average_distances = [np.mean(block_distances) for block_distances in all_distances]
    min_distance = float("inf")
    min_distance_layer = 0

    # CSV 저장
    with open("vlm_layer_distances.csv", "w", newline="") as csvfile:
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

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    torch.save(average_distances, "vlm_distances.pth")
    
    print(
        f"{Fore.GREEN}VLM Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} "
        f"has minimum distance of {min_distance}{Fore.RESET}"
    )
    print(
        f"{Fore.GREEN}Distances saved to vlm_layer_distances.csv and vlm_distances.pth{Fore.RESET}"
    )