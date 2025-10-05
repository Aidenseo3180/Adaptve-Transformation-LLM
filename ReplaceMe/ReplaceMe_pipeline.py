import argparse
import inspect
import logging

import torch
import yaml
from colorama import Fore, Style, init

# Import local modules
from .cosine_dist import cosine_dist
from .distance import profile_distances
from .evaluator import evaluator

from .utils import seed_all, select_non_overlapping_blocks

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages with timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set random seeds for reproducibility
seed_all()

# def ReplaceMe_pipeline(config):
#     # Extract the relevant parameters based on function signatures
#     signature = inspect.signature(profile_distances)
#     filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
#     # if config['distances_path'] is None:
#     #     # Profile distances using filtered configuration
#     #     profile_distances(**filtered_config)
#     #     config['distances_path'] = "./distances.pth"

#     if config["method"] == "cosine":  # Original cosine/adam methods
#         signature = inspect.signature(cosine_dist)
#         filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
#         # Load average distances and select non-overlapping blocks
#         average_distances = torch.load(filtered_config['distances_path'], weights_only=False)  
#         selected_blocks = select_non_overlapping_blocks(
#             average_distances, 
#             filtered_config['layers_to_skip'], 
#             num_blocks=filtered_config['num_A'], 
#             merge_consecutive=filtered_config['merge_consecutive']
#         )
        
#         # Calculate start and end IDs, and number of layers
#         start_ids = sorted([x[0] for x in selected_blocks])
#         end_ids = sorted([x[1] for x in selected_blocks])
#         num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
#         num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
        
#         # Iterate over each selected block
#         for i in range(len(selected_blocks)):
#             path = cosine_dist(**filtered_config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
#             filtered_config["model_path"] = path

#     elif config["method"] == "layer_quantization":
#         from .layer_quantization import apply_layer_quantization

#         # Profile distances if not already done
#         if config['distances_path'] is None:
#             signature = inspect.signature(profile_distances)
#             filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
#             profile_distances(**filtered_config)
#             config['distances_path'] = "./distances.pth"
        
#         # Load layer importance scores
#         logging.info(f"{Fore.CYAN}Loading layer importance scores...{Fore.RESET}")
#         layer_distances = torch.load(config['distances_path'], weights_only=False)
        
#         # For layers_to_skip=1, each score represents individual layer importance
#         # Higher distance = less important (more similar input/output)
#         layer_indices_sorted = sorted(
#             range(len(layer_distances)), 
#             key=lambda i: layer_distances[i], 
#             reverse=True  # Most redundant first
#         )
        
#         # Select top N least important layers for quantization
#         num_layers_to_quantize = config.get('num_layers_to_quantize', 16)
#         layers_to_quantize = layer_indices_sorted[:num_layers_to_quantize]
        
#         logging.info(f"{Fore.YELLOW}Selected {num_layers_to_quantize} least important layers:{Fore.RESET}")
#         logging.info(f"Layers: {sorted(layers_to_quantize)}")
#         logging.info(f"Importance scores: {[layer_distances[i] for i in layers_to_quantize[:5]]}...")
        
#         # Apply quantization
#         quantization_bits = config.get('quantization_bits', 8)
#         path = apply_layer_quantization(
#             model_path=config['model_path'],
#             layers_to_quantize=layers_to_quantize,
#             quantization_bits=quantization_bits,
#             save_path=config.get('save_path'),
#             token=config.get('token')
#         )
        
#         # Update config for evaluation
#         config["model_path"] = path

#     elif config["method"] == "spectral":
#         from .spectral_transform import apply_spectral_transform
        
#         print(f"\n{Fore.MAGENTA}Running Spectral Regularized Transform method{Fore.RESET}")
        
#         signature = inspect.signature(apply_spectral_transform)
#         filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
#         # Load distances and select blocks
#         average_distances = torch.load(filtered_config['distances_path'], weights_only=False)
#         selected_blocks = select_non_overlapping_blocks(
#             average_distances,
#             filtered_config['layers_to_skip'],
#             num_blocks=filtered_config.get('num_A', 1),
#             merge_consecutive=filtered_config.get('merge_consecutive', False)
#         )
        
#         print(f"Selected blocks for spectral transform: {selected_blocks}")
        
#         # Process each block
#         start_ids = sorted([x[0] for x in selected_blocks])
#         end_ids = sorted([x[1] for x in selected_blocks])
#         num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
#         num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
        
#         for i in range(len(selected_blocks)):
#             print(f"\n{Fore.CYAN}Processing block {i+1}/{len(selected_blocks)}{Fore.RESET}")
#             path = apply_spectral_transform(
#                 **filtered_config,
#                 start_id=start_ids[i],
#                 end_id=end_ids[i],
#                 num_layer=num_layers[i]
#             )
#             filtered_config["model_path"] = path

#     else:
#         raise ValueError(f"Unknown method: {config['method']}")


#     # Evaluate using the updated configuration
#     signature = inspect.signature(evaluator)
#     filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
#     filtered_config["model_path"] = path
#     evaluator(**filtered_config)

def ReplaceMe_pipeline(config):
    is_llava = config.get('is_llava', False)
    
    print(f"\n{'='*70}")
    print(f"Starting ReplaceMe Pipeline")
    print(f"{'='*70}")
    print(f"Model: {config['model_path']}")
    print(f"Method: {config['method']}")
    print(f"Is LLaVA: {is_llava}")
    print(f"{'='*70}\n")
    
    # Distance profiling
    if config['distances_path'] is None:
        print("[STEP 1] Distance Profiling...")
        
        if is_llava:
            from .distance import profile_distances_llava
            signature = inspect.signature(profile_distances_llava)
            filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
            profile_distances_llava(**filtered_config)
            config['distances_path'] = "./distances_llava.pth"
        else:
            signature = inspect.signature(profile_distances)
            filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
            profile_distances(**filtered_config)
            config['distances_path'] = "./distances.pth"
        
        print(f"[STEP 1 COMPLETE] Distances saved to {config['distances_path']}\n")
    else:
        print(f"[STEP 1 SKIPPED] Using existing distances from {config['distances_path']}\n")
    
    # Transformation estimation
    print("[STEP 2] Transformation Estimation...")
    
    if config["method"] == "lstsq":
        print("[METHOD] Using Least Squares")
        signature = inspect.signature(lstsq)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        path = lstsq(**filtered_config)
        
    elif config["method"] == "cosine":
        if is_llava:
            print("[METHOD] Using Cosine Distance for LLaVA")
            from .cosine_dist import cosine_dist_llava
            signature = inspect.signature(cosine_dist_llava)
            filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
            
            # Load distances and select blocks
            average_distances = torch.load(filtered_config['distances_path'], weights_only=False)
            print(f"[DEBUG] Loaded {len(average_distances)} distance values")
            
            selected_blocks = select_non_overlapping_blocks(
                average_distances,
                filtered_config['layers_to_skip'],
                num_blocks=filtered_config['num_A'],
                merge_consecutive=filtered_config['merge_consecutive']
            )
            
            start_ids = sorted([x[0] for x in selected_blocks])
            end_ids = sorted([x[1] for x in selected_blocks])
            num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
            num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
            
            print(f"[DEBUG] Selected blocks: {selected_blocks}")
            print(f"[DEBUG] Start IDs: {start_ids}")
            print(f"[DEBUG] End IDs: {end_ids}")
            print(f"[DEBUG] Num layers: {num_layers}")
            
            # Process each block
            for i in range(len(selected_blocks)):
                print(f"\n[BLOCK {i+1}/{len(selected_blocks)}] Processing layers {start_ids[i]} to {end_ids[i]}")
                path = cosine_dist_llava(
                    **filtered_config,
                    start_id=start_ids[i],
                    end_id=end_ids[i],
                    num_layer=num_layers[i]
                )
                filtered_config["model_path"] = path
                print(f"[BLOCK {i+1} COMPLETE] Model saved to {path}")
        else:
            print("[METHOD] Using Cosine Distance for standard LLM")
            signature = inspect.signature(cosine_dist)
            filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
            
            average_distances = torch.load(filtered_config['distances_path'], weights_only=False)
            selected_blocks = select_non_overlapping_blocks(
                average_distances,
                filtered_config['layers_to_skip'],
                num_blocks=filtered_config['num_A'],
                merge_consecutive=filtered_config['merge_consecutive']
            )
            
            start_ids = sorted([x[0] for x in selected_blocks])
            end_ids = sorted([x[1] for x in selected_blocks])
            num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
            num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
            
            for i in range(len(selected_blocks)):
                path = cosine_dist(
                    **filtered_config,
                    start_id=start_ids[i],
                    end_id=end_ids[i],
                    num_layer=num_layers[i]
                )
                filtered_config["model_path"] = path
    
    print(f"\n[STEP 2 COMPLETE] Final model path: {path}\n")
    
    # Evaluation
    print("[STEP 3] Model Evaluation...")
    
    if is_llava:
        from .utils import eval_llava
        print("[EVALUATION] Running LLaVA-specific evaluation")
        results = eval_llava(path)
    else:
        print("[EVALUATION] Running standard LLM evaluation")
        signature = inspect.signature(evaluator)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        filtered_config["model_path"] = path
        results = evaluator(**filtered_config)
    
    print(f"\n[STEP 3 COMPLETE] Evaluation finished")
    
    print(f"\n{'='*70}")
    print(f"ReplaceMe Pipeline Complete!")
    print(f"Final model: {path}")
    print(f"{'='*70}\n")
    
    return results

def read_config(config_path: str):
    # Read and return the YAML configuration
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_from_config():
    # Parse command-line arguments for configuration file path
    parser = argparse.ArgumentParser(
        description="Run compression methods for linear transform estimation based on a configuration file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    # Execute pipeline based on parsed configuration
    args = parser.parse_args()
    config = read_config(args.config)
    ReplaceMe_pipeline(config)