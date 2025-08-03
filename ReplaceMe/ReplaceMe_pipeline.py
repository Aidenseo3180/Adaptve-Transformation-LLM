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
from .lstsq import lstsq
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
    signature = inspect.signature(profile_distances)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    if config['distances_path'] is None:
        # Profile distances using filtered configuration
        profile_distances(**filtered_config)
        config['distances_path'] = "./distances.pth"
    # Determine method and apply configurations accordingly
    if config["method"] == "lstsq":
        signature = inspect.signature(lstsq)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        path = lstsq(**filtered_config)

    elif config["method"] == "iclt":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from iclt_fit import fit_iclt
        from ICLTSkipModel import ICLTSkipModel
        from distance import profile_distances
        from utils import select_non_overlapping_blocks
        from cosine_dist import cosine_dist
        from evaluator import evaluator
        import torch
        import os

        # 하드코딩된 디렉토리
        iclt_save_dir = "./ICLT_weights/"
        save_path = "./outputs/iclt_model"
        os.makedirs(iclt_save_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        # 1. 모델 로딩
        print("[ICLT] Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 2. cosine distance 기반 block pair 선택
        if isinstance(config["layer_to_skip"], int):
            print("[ICLT] Profiling distances to select best block pair...")
            profile_distances(config)
            distances = torch.load(config["distances_path"])
            selected_pairs = select_non_overlapping_blocks(distances, num_blocks=config["layer_to_skip"])
            start_idx, end_idx = selected_pairs[0]
            print(f"[ICLT] Selected skip block: {start_idx} → {end_idx}")
        else:
            start_idx, end_idx = config["layer_to_skip"]

        # 3. hidden state 없을 시 자동 생성
        if not os.path.exists(config["a1_path"]) or not os.path.exists(config["a2_path"]):
            print("[ICLT] a1/a2 hidden states not found. Extracting from model...")
            cosine_dist(config)

        # 4. ICLT 학습
        print("[ICLT] Fitting ICLT transforms...")
        fit_iclt(
            x_path=config["a1_path"],
            y_path=config["a2_path"],
            save_dir=iclt_save_dir,
            K=config["k"],
            rank=config.get("rank", 256)
        )

        # 5. 모델에 adapter 삽입
        model = ICLTSkipModel(
            base_model=model,
            adapter_dir=iclt_save_dir,
            K=config["k"],
            rank=config.get("rank", 256),
            start_idx=start_idx,
            end_idx=end_idx
        )

        # 6. 저장 후 evaluator 호출
        print(f"[ICLT] Saving model to {save_path}...")
        model.save_pretrained(save_path)

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(save_path)

        print("[ICLT] Running evaluator...")
        results = evaluator(
            model_path=save_path,
            tasks=config.get("tasks", "default"),
            **config
        )

        print("[ICLT] Evaluation Results:")
        print(results)
        return

    else:
        signature = inspect.signature(cosine_dist)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        # Load average distances and select non-overlapping blocks
        average_distances = torch.load(filtered_config['distances_path'], weights_only=False)  
        selected_blocks = select_non_overlapping_blocks(
            average_distances, 
            filtered_config['layers_to_skip'], 
            num_blocks=filtered_config['num_A'], 
            merge_consecutive=filtered_config['merge_consecutive']
        )
        
        # Calculate start and end IDs, and number of layers
        start_ids = sorted([x[0] for x in selected_blocks])
        end_ids = sorted([x[1] for x in selected_blocks])
        num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
        num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
        
        # Iterate over each selected block
        for i in range(len(selected_blocks)):
            path = cosine_dist(**filtered_config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
            filtered_config["model_path"] = path
    
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
        description="Run LSTSQ for linear transform estimation based on a configuration file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    # Execute pipeline based on parsed configuration
    args = parser.parse_args()
    config = read_config(args.config)
    ReplaceMe_pipeline(config)
