# ============================================================
# vlm_distance.py (새 파일 생성) -> NEW CODE
# ============================================================
# """
# Enhanced VLM Distance Profiling with Token-Type Separation
# Integrates compute_token_type_distances into vlm_profile_distances
# """

# import argparse
# import csv
# import gc
# import logging
# from typing import Optional, Dict, List, Tuple

# import numpy as np
# import torch
# import yaml
# from colorama import Fore, init
# from tqdm import tqdm
# from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

# from .utils import (
#     get_vlm_calib_dataloader,
#     setup_vlm_processor,
#     get_vlm_layers,
#     seed_all
# )

# init(autoreset=True)

# logging.basicConfig(
#     format=(
#         f"{Fore.CYAN}%(asctime)s "
#         f"{Fore.YELLOW}[%(levelname)s] "
#         f"{Fore.RESET}%(message)s"
#     ),
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# seed_all()


# def angular_distance(x_l: torch.Tensor, x_l_plus_n: torch.Tensor) -> torch.Tensor:
#     """Compute angular distance between normalized layer outputs."""
#     x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
#     x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
#     cosine_sim = (x_l_norm * x_l_plus_n_norm).sum(-1).clamp(min=-1, max=1)
#     return torch.acos(cosine_sim) / torch.pi


# def compute_token_type_distances(
#     hidden_states_list: List[torch.Tensor],
#     layers_to_skip: int,
#     num_vision_tokens: int,
#     attention_mask: torch.Tensor
# ) -> Dict[str, List[float]]:
#     """
#     Compute cosine distances separately for vision and text tokens.
    
#     Args:
#         hidden_states_list: List of hidden states from all layers
#         layers_to_skip: Number of layers to skip
#         num_vision_tokens: Number of vision tokens (e.g., 576 for LLaVA 1.5)
#         attention_mask: Attention mask to identify valid tokens
        
#     Returns:
#         Dictionary with 'vision', 'text', and 'combined' distances
#     """
#     # Initialize distance lists
#     vision_distances = []
#     text_distances = []
#     combined_distances = []
    
#     # Iterate through layer pairs
#     for layer_idx in range(len(hidden_states_list) - layers_to_skip):
#         h_start = hidden_states_list[layer_idx]  # [batch, seq_len, hidden_dim]
#         h_end = hidden_states_list[layer_idx + layers_to_skip]
        
#         # Separate vision and text tokens
#         vision_start = h_start[:, :num_vision_tokens, :]  # [batch, 576, hidden_dim]
#         vision_end = h_end[:, :num_vision_tokens, :]
        
#         text_start = h_start[:, num_vision_tokens:, :]  # [batch, text_len, hidden_dim]
#         text_end = h_end[:, num_vision_tokens:, :]
        
#         # Get text mask (vision tokens are always valid)
#         text_mask = attention_mask[:, num_vision_tokens:]  # [batch, text_len]
        
#         # === Vision Token Distance ===
#         # Vision tokens are always valid, so simple average
#         vision_dist = angular_distance(
#             vision_start.reshape(-1, h_start.shape[-1]),
#             vision_end.reshape(-1, h_end.shape[-1])
#         )
#         vision_distances.append(vision_dist.mean().item())
        
#         # === Text Token Distance ===
#         # Only compute for non-padded text tokens
#         text_mask_expanded = text_mask.unsqueeze(-1).expand_as(text_start)
        
#         # Mask out padded positions
#         text_start_masked = text_start * text_mask_expanded
#         text_end_masked = text_end * text_mask_expanded
        
#         # Flatten and filter valid tokens
#         text_start_flat = text_start_masked.reshape(-1, h_start.shape[-1])
#         text_end_flat = text_end_masked.reshape(-1, h_end.shape[-1])
#         text_mask_flat = text_mask.reshape(-1)
        
#         # Only keep non-padded tokens
#         valid_text_start = text_start_flat[text_mask_flat.bool()]
#         valid_text_end = text_end_flat[text_mask_flat.bool()]
        
#         if valid_text_start.shape[0] > 0:
#             text_dist = angular_distance(valid_text_start, valid_text_end)
#             text_distances.append(text_dist.mean().item())
#         else:
#             text_distances.append(0.0)
        
#         # === Combined Distance (Original) ===
#         # For comparison with original method
#         combined_mask = attention_mask.unsqueeze(-1).expand_as(h_start)
#         h_start_masked = h_start * combined_mask
#         h_end_masked = h_end * combined_mask
        
#         h_start_flat = h_start_masked.reshape(-1, h_start.shape[-1])
#         h_end_flat = h_end_masked.reshape(-1, h_end.shape[-1])
#         combined_mask_flat = attention_mask.reshape(-1)
        
#         valid_start = h_start_flat[combined_mask_flat.bool()]
#         valid_end = h_end_flat[combined_mask_flat.bool()]
        
#         if valid_start.shape[0] > 0:
#             combined_dist = angular_distance(valid_start, valid_end)
#             combined_distances.append(combined_dist.mean().item())
#         else:
#             combined_distances.append(0.0)
    
#     return {
#         'vision': vision_distances,
#         'text': text_distances,
#         'combined': combined_distances
#     }


# def vlm_profile_distances(
#     model_path: str,
#     image_dir: str = "train2014",
#     batch_size: int = 4,
#     max_length: int = 512,
#     layers_to_skip: int = 8,
#     num_vision_tokens: int = 576,  # LLaVA 1.5 default
#     dataset_size: Optional[int] = None,
#     use_4bit: bool = False,
#     token: Optional[str] = None,
#     output_prefix: str = "vlm",
# ) -> Dict[str, List[float]]:
#     """
#     Profile distances between VLM language model layers with token-type separation.
    
#     Args:
#         model_path: Path to VLM model (e.g., liuhaotian/llava-v1.5-7b)
#         image_dir: Image directory path (e.g., train2014)
#         batch_size: Batch size for processing
#         max_length: Maximum sequence length
#         layers_to_skip: Number of layers to skip
#         num_vision_tokens: Number of vision tokens (576 for LLaVA 1.5)
#         dataset_size: Dataset size limit
#         use_4bit: Use 4-bit quantization
#         token: HuggingFace authentication token
#         output_prefix: Prefix for output files
        
#     Returns:
#         Dictionary with vision, text, and combined distances
#     """
#     device_map = "auto" if torch.cuda.is_available() else "cpu"
#     quantization_config = None

#     if use_4bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#     logging.info(f"{Fore.CYAN}Loading VLM: {model_path}{Fore.RESET}")
    
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_path,
#         device_map=device_map,
#         quantization_config=quantization_config,
#         output_hidden_states=True,
#         token=token,
#         torch_dtype=torch.bfloat16
#     )

#     processor = setup_vlm_processor(model_path)
#     layers, num_hidden_layers = get_vlm_layers(model)
    
#     logging.info(f"{Fore.GREEN}VLM loaded - {num_hidden_layers} language model layers{Fore.RESET}")
#     logging.info(f"{Fore.MAGENTA}Starting token-type separated distance profiling{Fore.RESET}")
#     logging.info(f"Vision tokens: {num_vision_tokens}, Layers to skip: {layers_to_skip}")

#     model.eval()
#     dataloader = get_vlm_calib_dataloader(
#         image_dir,
#         dataset_size,
#         batch_size,
#         processor
#     )

#     # Initialize distance tracking (separated by token type)
#     num_blocks = num_hidden_layers - layers_to_skip
#     all_vision_distances = [[] for _ in range(num_blocks)]
#     all_text_distances = [[] for _ in range(num_blocks)]
#     all_combined_distances = [[] for _ in range(num_blocks)]

#     # Process batches
#     for batch_idx, batch in enumerate(tqdm(
#         dataloader,
#         desc=f"{Fore.GREEN}Computing VLM Token-Type Distances{Fore.RESET}",
#         dynamic_ncols=True,
#         colour="green",
#     )):
#         # Move inputs to device
#         inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

#         with torch.no_grad():
#             outputs = model(**inputs)

#         # Get attention mask and hidden states
#         attention_mask = batch["attention_mask"].to(model.device)
#         hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]
        
#         # Skip embedding layer (index 0)
#         hidden_states_list = list(hidden_states[1:])
        
#         # Compute distances for this batch (separated by token type)
#         batch_distances = compute_token_type_distances(
#             hidden_states_list,
#             layers_to_skip,
#             num_vision_tokens,
#             attention_mask
#         )
        
#         # Accumulate distances by token type
#         for i in range(num_blocks):
#             all_vision_distances[i].append(batch_distances['vision'][i])
#             all_text_distances[i].append(batch_distances['text'][i])
#             all_combined_distances[i].append(batch_distances['combined'][i])

#         # Memory cleanup
#         del outputs, hidden_states, hidden_states_list
#         gc.collect()
#         torch.cuda.empty_cache()

#     # Calculate average distances
#     avg_vision = [np.mean(dists) for dists in all_vision_distances]
#     avg_text = [np.mean(dists) for dists in all_text_distances]
#     avg_combined = [np.mean(dists) for dists in all_combined_distances]

#     # Find minimum distances
#     min_vision_idx = np.argmin(avg_vision)
#     min_text_idx = np.argmin(avg_text)
#     min_combined_idx = np.argmin(avg_combined)

#     # Log results
#     logging.info(f"\n{Fore.CYAN}{'='*70}{Fore.RESET}")
#     logging.info(f"{Fore.YELLOW}RESULTS: VLM Token-Type Distance Analysis{Fore.RESET}")
#     logging.info(f"{Fore.CYAN}{'='*70}{Fore.RESET}")
    
#     logging.info(f"\n{Fore.GREEN}Vision Tokens:{Fore.RESET}")
#     logging.info(f"  Most linear block: Layer {min_vision_idx+1} → {min_vision_idx+1+layers_to_skip}")
#     logging.info(f"  Distance: {avg_vision[min_vision_idx]:.4f}")
#     logging.info(f"  Average distance: {np.mean(avg_vision):.4f}")
    
#     logging.info(f"\n{Fore.GREEN}Text Tokens:{Fore.RESET}")
#     logging.info(f"  Most linear block: Layer {min_text_idx+1} → {min_text_idx+1+layers_to_skip}")
#     logging.info(f"  Distance: {avg_text[min_text_idx]:.4f}")
#     logging.info(f"  Average distance: {np.mean(avg_text):.4f}")
    
#     logging.info(f"\n{Fore.GREEN}Combined (Original Method):{Fore.RESET}")
#     logging.info(f"  Most linear block: Layer {min_combined_idx+1} → {min_combined_idx+1+layers_to_skip}")
#     logging.info(f"  Distance: {avg_combined[min_combined_idx]:.4f}")
#     logging.info(f"  Average distance: {np.mean(avg_combined):.4f}")

#     # Save to CSV files
#     csv_files = {
#         'vision': f"{output_prefix}_vision_distances_{layers_to_skip}layers.csv",
#         'text': f"{output_prefix}_text_distances_{layers_to_skip}layers.csv",
#         'combined': f"{output_prefix}_combined_distances_{layers_to_skip}layers.csv",
#         'comparison': f"{output_prefix}_comparison_{layers_to_skip}layers.csv"
#     }

#     # Save individual CSV files
#     for token_type, distances in [('vision', avg_vision), 
#                                    ('text', avg_text), 
#                                    ('combined', avg_combined)]:
#         with open(csv_files[token_type], "w", newline="") as csvfile:
#             fieldnames = ["block_start", "block_end", "average_distance"]
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()

#             for i, avg_dist in enumerate(distances):
#                 writer.writerow({
#                     "block_start": i + 1,
#                     "block_end": i + 1 + layers_to_skip,
#                     "average_distance": avg_dist,
#                 })
        
#         logging.info(f"{Fore.BLUE}Saved: {csv_files[token_type]}{Fore.RESET}")

#     # Save comparison CSV
#     with open(csv_files['comparison'], "w", newline="") as csvfile:
#         fieldnames = ["block_start", "block_end", 
#                       "vision_distance", "text_distance", "combined_distance",
#                       "vision_vs_text_ratio", "difference"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         for i in range(len(avg_vision)):
#             ratio = avg_vision[i] / avg_text[i] if avg_text[i] > 0 else 0
#             difference = avg_text[i] - avg_vision[i]
#             writer.writerow({
#                 "block_start": i + 1,
#                 "block_end": i + 1 + layers_to_skip,
#                 "vision_distance": avg_vision[i],
#                 "text_distance": avg_text[i],
#                 "combined_distance": avg_combined[i],
#                 "vision_vs_text_ratio": ratio,
#                 "difference": difference,
#             })
    
#     logging.info(f"{Fore.BLUE}Saved: {csv_files['comparison']}{Fore.RESET}")
    
#     # Save to PyTorch file
#     torch.save({
#         'vision': avg_vision,
#         'text': avg_text,
#         'combined': avg_combined,
#         'config': {
#             'model_path': model_path,
#             'layers_to_skip': layers_to_skip,
#             'num_vision_tokens': num_vision_tokens,
#             'num_hidden_layers': num_hidden_layers,
#         }
#     }, f"{output_prefix}_distances_{layers_to_skip}layers.pth")
    
#     logging.info(f"{Fore.BLUE}Saved: {output_prefix}_distances_{layers_to_skip}layers.pth{Fore.RESET}")

#     # Cleanup
#     del model
#     gc.collect()
#     torch.cuda.empty_cache()

#     logging.info(f"\n{Fore.GREEN}✅ VLM token-type distance profiling complete!{Fore.RESET}\n")

#     return {
#         'vision': avg_vision,
#         'text': avg_text,
#         'combined': avg_combined
#     }



# ============================================================
# vlm_distance.py (새 파일 생성) -> OLD CODE
# ============================================================

# import argparse
# import csv
# import gc
# import logging
# from typing import Optional

# import numpy as np
# import torch
# import yaml
# from colorama import Fore, init
# from tqdm import tqdm
# from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

# from .utils import (
#     compute_block_distances,
#     get_vlm_calib_dataloader,
#     setup_vlm_processor,
#     get_vlm_layers,
#     get_last_non_padded_tokens,
#     seed_all
# )

# init(autoreset=True)

# logging.basicConfig(
#     format=(
#         f"{Fore.CYAN}%(asctime)s "
#         f"{Fore.YELLOW}[%(levelname)s] "
#         f"{Fore.RESET}%(message)s"
#     ),
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# seed_all()


# def vlm_profile_distances(
#     model_path: str,
#     image_dir: str = "train2014",
#     batch_size: int = 4,
#     max_length: int = 512,
#     layers_to_skip: int = 8,
#     dataset_size: Optional[int] = None,
#     use_4bit: bool = False,
#     token: Optional[str] = None,
# ) -> None:
#     """Profile distances between VLM language model layers.
    
#     Args:
#         model_path: Path to VLM model
#         image_dir: Image directory path
#         batch_size: Batch size
#         max_length: Max sequence length
#         layers_to_skip: Layers to skip
#         dataset_size: Dataset size limit
#         use_4bit: Use 4-bit quantization
#         token: HuggingFace token
#     """
#     device_map = "auto" if torch.cuda.is_available() else "cpu"
#     quantization_config = None

#     if use_4bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#     print(f"{Fore.GREEN}Loading VLM for distance profiling{Fore.RESET}")
    
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_path,
#         device_map=device_map,
#         quantization_config=quantization_config,
#         output_hidden_states=True,
#         token=token,
#         torch_dtype=torch.bfloat16
#     )

#     processor = setup_vlm_processor(model_path)
#     layers, num_hidden_layers = get_vlm_layers(model)
    
#     print(f"{Fore.GREEN}VLM loaded - {num_hidden_layers} language model layers{Fore.RESET}")

#     model.eval()
#     dataloader = get_vlm_calib_dataloader(
#         image_dir,
#         dataset_size,
#         batch_size,
#         processor
#     )

#     # Distance 추적
#     all_distances = [
#         [] for _ in range(num_hidden_layers - layers_to_skip)
#     ]

#     # 배치 처리
#     for batch in tqdm(
#         dataloader,
#         desc=f"{Fore.GREEN}Computing VLM Distances{Fore.RESET}",
#         dynamic_ncols=True,
#         colour="green",
#     ):
#         # ===== 1. Processor 출력 확인 =====
#         # print(f"\n[DEBUG] Batch keys: {batch.keys()}")
#         # for k, v in batch.items():
#         #     if isinstance(v, torch.Tensor):
#         #         print(f"[DEBUG] {k}: shape={v.shape}, dtype={v.dtype}")

#         inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#         # print(f"[DEBUG] Inputs keys after device move: {inputs.keys()}")
#         # print(f"[DEBUG] Has pixel_values: {'pixel_values' in inputs}")

#         # # ===== 3. Model 입력 직전 확인 =====
#         # print(f"[DEBUG] Final input shapes:")
#         # for k, v in inputs.items():
#         #     print(f"[DEBUG]   {k}: {v.shape}")

#         with torch.no_grad():
#             outputs = model(**inputs)

#         # Attention mask와 hidden states
#         attention_mask = batch["attention_mask"].to(model.device)
#         hidden_states = outputs.hidden_states
        
#         # Last non-padded tokens
#         last_non_padded_hidden_states = get_last_non_padded_tokens(
#             hidden_states, attention_mask
#         )
#         last_non_padded_hidden_states = last_non_padded_hidden_states[1:]

#         distances = compute_block_distances(
#             last_non_padded_hidden_states, layers_to_skip
#         )

#         for i, distance in enumerate(distances):
#             all_distances[i].append(distance)

#         gc.collect()
#         torch.cuda.empty_cache()

#     # 평균 distance 계산
#     average_distances = [np.mean(block_distances) for block_distances in all_distances]
#     min_distance = float("inf")
#     min_distance_layer = 0

#     # CSV 저장
#     with open("vlm_layer_distances.csv", "w", newline="") as csvfile:
#         fieldnames = ["block_start", "block_end", "average_distance"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         for i, avg_dist in enumerate(average_distances):
#             writer.writerow({
#                 "block_start": i + 1,
#                 "block_end": i + 1 + layers_to_skip,
#                 "average_distance": avg_dist,
#             })

#             if avg_dist < min_distance:
#                 min_distance = avg_dist
#                 min_distance_layer = i + 1

#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     torch.save(average_distances, "vlm_distances.pth")
    
#     print(
#         f"{Fore.GREEN}VLM Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} "
#         f"has minimum distance of {min_distance}{Fore.RESET}"
#     )
#     print(
#         f"{Fore.GREEN}Distances saved to vlm_layer_distances.csv and vlm_distances.pth{Fore.RESET}"
#     )


# TODO:
# TODO:
# TODO:

import argparse
import csv
import gc
from typing import Optional, List

import numpy as np
import torch
from colorama import Fore, init
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

from .utils import (
    get_vlm_calib_dataloader,
    setup_vlm_processor,
    get_vlm_layers,
    seed_all
)

init(autoreset=True)
seed_all()


def angular_distance(x_l: torch.Tensor, x_l_plus_n: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between normalized layer outputs."""
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_sim = (x_l_norm * x_l_plus_n_norm).sum(-1).clamp(min=-1, max=1)
    return torch.acos(cosine_sim) / torch.pi


def compute_combined_distances(
    hidden_states_list: List[torch.Tensor],
    layers_to_skip: int,
    attention_mask: torch.Tensor
) -> List[float]:
    """
    Compute cosine distances for combined (all) tokens.
    
    Args:
        hidden_states_list: List of hidden states from all layers
        layers_to_skip: Number of layers to skip
        attention_mask: Attention mask to identify valid tokens
        
    Returns:
        List of average distances for each layer pair
    """
    distances = []
    
    # Iterate through layer pairs
    for layer_idx in range(len(hidden_states_list) - layers_to_skip):
        h_start = hidden_states_list[layer_idx]  # [batch, seq_len, hidden_dim]
        h_end = hidden_states_list[layer_idx + layers_to_skip]
        
        # Apply attention mask to get only valid tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(h_start)
        
        # Mask out padded positions
        h_start_masked = h_start * mask_expanded
        h_end_masked = h_end * mask_expanded
        
        # Flatten and filter valid tokens
        h_start_flat = h_start_masked.reshape(-1, h_start.shape[-1])
        h_end_flat = h_end_masked.reshape(-1, h_end.shape[-1])
        mask_flat = attention_mask.reshape(-1)
        
        # Only keep non-padded tokens
        valid_start = h_start_flat[mask_flat.bool()]
        valid_end = h_end_flat[mask_flat.bool()]
        
        if valid_start.shape[0] > 0:
            dist = angular_distance(valid_start, valid_end)
            distances.append(dist.mean().item())
        else:
            distances.append(0.0)
    
    return distances


def vlm_profile_distances(
    model_path: str,
    image_dir: str = "train2014",
    batch_size: int = 4,
    max_length: int = 512,
    layers_to_skip: int = 8,
    dataset_size: Optional[int] = None,
    use_4bit: bool = False,
    token: Optional[str] = None,
) -> List[float]:
    """
    Profile distances between VLM language model layers (combined tokens).
    
    Args:
        model_path: Path to VLM model (e.g., liuhaotian/llava-v1.5-7b)
        image_dir: Image directory path (e.g., train2014)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Dataset size limit
        use_4bit: Use 4-bit quantization
        token: HuggingFace authentication token
        
    Returns:
        List of average distances for each layer block
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

    print(f"{Fore.GREEN}Loading VLM for distance profiling: {model_path}{Fore.RESET}")
    
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
    print(f"{Fore.MAGENTA}Skipping {layers_to_skip} layers at a time{Fore.RESET}")

    model.eval()
    dataloader = get_vlm_calib_dataloader(
        image_dir,
        dataset_size,
        batch_size,
        processor
    )

    # Initialize distance tracking
    num_blocks = num_hidden_layers - layers_to_skip
    all_distances = [[] for _ in range(num_blocks)]

    # Process batches
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing VLM Distances{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = model(**inputs)

        # Get attention mask and hidden states
        attention_mask = batch["attention_mask"].to(model.device)
        hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]
        
        # Skip embedding layer (index 0)
        hidden_states_list = list(hidden_states[1:])
        
        # Compute distances for this batch
        batch_distances = compute_combined_distances(
            hidden_states_list,
            layers_to_skip,
            attention_mask
        )
        
        # Accumulate distances
        for i, dist in enumerate(batch_distances):
            all_distances[i].append(dist)

        # Memory cleanup
        del outputs, hidden_states, hidden_states_list
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate average distances
    average_distances = [np.mean(dists) for dists in all_distances]

    # Find minimum distance (skip first 10 blocks)
    skip_first_n_blocks = 8
    
    if len(average_distances) > skip_first_n_blocks:
        print("Disregarding the first 8 layers")
        # Search for minimum in remaining blocks
        min_distance = min(average_distances[skip_first_n_blocks:])
        # Get the actual index (accounting for skipped blocks)
        min_distance_layer = average_distances[skip_first_n_blocks:].index(min_distance) + skip_first_n_blocks + 1
    else:
        # Fallback if total blocks <= 8
        print(f"{Fore.YELLOW}Warning: Only {len(average_distances)} blocks, using all for minimum{Fore.RESET}")
        min_distance = min(average_distances)
        min_distance_layer = average_distances.index(min_distance) + 1

    # Save to CSV
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

    # Save to PyTorch file
    torch.save(average_distances, "vlm_distances.pth")
    
    # Print results
    print(f"\n{Fore.CYAN}{'='*70}{Fore.RESET}")
    print(f"{Fore.YELLOW}RESULTS: VLM Distance Analysis{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*70}{Fore.RESET}")
    print(
        f"{Fore.GREEN}Most linear block: Layer {min_distance_layer} → "
        f"{min_distance_layer + layers_to_skip}{Fore.RESET}"
    )
    print(f"{Fore.GREEN}Minimum distance: {min_distance:.4f}{Fore.RESET}")
    print(f"{Fore.GREEN}Average distance: {np.mean(average_distances):.4f}{Fore.RESET}")
    print(f"{Fore.GREEN}Distances saved to vlm_layer_distances.csv and vlm_distances.pth{Fore.RESET}\n")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return average_distances





