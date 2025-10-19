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

def ReplaceMe_pipeline(config):
    # Extract the relevant parameters based on function signatures
    # signature = inspect.signature(profile_distances)
    # filtered_config = {k: v for k, v in config.items() if k in signature.parameters}

    if config["method"] == "vlm_cosine":
        from .vlm_cosine_dist import vlm_cosine_dist
        from .vlm_distance import vlm_profile_distances
        import os

        print(f"{Fore.MAGENTA}=== VLM COSINE METHOD ==={Fore.RESET}")
        
        signature = inspect.signature(vlm_profile_distances)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        if config.get('distances_path') is None or not os.path.exists(config.get('distances_path', '')):
            print(f"{Fore.CYAN}Profiling VLM distances...{Fore.RESET}")
            vlm_profile_distances(**filtered_config)
            config['distances_path'] = "./vlm_distances.pth"
        
        signature = inspect.signature(vlm_cosine_dist)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        average_distances = torch.load(config['distances_path'], weights_only=False)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            config['layers_to_skip'],
            num_blocks=config.get('num_A', 1),
            merge_consecutive=config.get('merge_consecutive', True)
        )
        
        start_ids = sorted([x[0] for x in selected_blocks])
        end_ids = sorted([x[1] for x in selected_blocks])
        num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
        num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]

        print(f"[DEBUG] Selected blocks: {selected_blocks}")
        print(f"[DEBUG] start_ids: {start_ids}")
        print(f"[DEBUG] end_ids: {end_ids}")
        print(f"[DEBUG] num_layers: {num_layers}")
        
        for i in range(len(selected_blocks)):
            print(f"{Fore.YELLOW}Processing block {i+1}/{len(selected_blocks)}{Fore.RESET}")
            path = vlm_cosine_dist(
                **filtered_config,
                start_id=start_ids[i],
                end_id=end_ids[i],
                num_layer=num_layers[i]
            )
            filtered_config["model_path"] = path
        
        return

    elif config["method"] == "vlm_two_phase":
        from .vlm_two_phase import vlm_two_phase
        from .vlm_distance import vlm_profile_distances
        import os

        print(f"{Fore.MAGENTA}=== VLM TWO-PHASE METHOD ==={Fore.RESET}")
        
        signature = inspect.signature(vlm_profile_distances)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        if config.get('distances_path') is None or not os.path.exists(config.get('distances_path', '')):
            print(f"{Fore.CYAN}Profiling distances...{Fore.RESET}")
            vlm_profile_distances(**filtered_config)
            config['distances_path'] = "./vlm_distances.pth"
        
        signature = inspect.signature(vlm_two_phase)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        average_distances = torch.load(config['distances_path'], weights_only=False)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            config['layers_to_skip'],
            num_blocks=config.get('num_A', 1),
            merge_consecutive=config.get('merge_consecutive', True)
        )
        
        start_ids = sorted([x[0] for x in selected_blocks])
        end_ids = sorted([x[1] for x in selected_blocks])
        num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
        num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
        
        for i in range(len(selected_blocks)):
            print(f"{Fore.YELLOW}Block {i+1}/{len(selected_blocks)}{Fore.RESET}")
            path = vlm_two_phase(
                **filtered_config,
                start_id=start_ids[i],
                end_id=end_ids[i],
                num_layer=num_layers[i]
            )
            filtered_config["model_path"] = path
        
        print(f"{Fore.GREEN}✓ Two-Phase pruning completed{Fore.RESET}")
        return
    
    elif config["method"] == "two_phase":
        from .llm_two_phase import two_phase
        from .distance import profile_distances
        import os

        print(f"{Fore.MAGENTA}=== LLM TWO-PHASE METHOD ==={Fore.RESET}")
        
        signature = inspect.signature(profile_distances)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        if config.get('distances_path') is None or not os.path.exists(config.get('distances_path', '')):
            print(f"{Fore.CYAN}Profiling distances...{Fore.RESET}")
            profile_distances(**filtered_config)
            config['distances_path'] = "./distances.pth"
        
        signature = inspect.signature(two_phase)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        average_distances = torch.load(config['distances_path'], weights_only=False)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            config['layers_to_skip'],
            num_blocks=config.get('num_A', 1),
            merge_consecutive=config.get('merge_consecutive', True)
        )
        
        start_ids = sorted([x[0] for x in selected_blocks])
        end_ids = sorted([x[1] for x in selected_blocks])
        num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
        num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
        
        for i in range(len(selected_blocks)):
            print(f"{Fore.YELLOW}Block {i+1}/{len(selected_blocks)}{Fore.RESET}")
            path = two_phase(
                **filtered_config,
                start_id=start_ids[i],
                end_id=end_ids[i],
                num_layer=num_layers[i]
            )
            filtered_config["model_path"] = path
        
        print(f"{Fore.GREEN}✓ Two-Phase pruning completed{Fore.RESET}")

    else:
        raise ValueError(f"Unknown method: {config['method']}")


    # Evaluate using the updated configuration
    signature = inspect.signature(evaluator)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    filtered_config["model_path"] = path
    evaluator(**filtered_config)


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