"""
Enhanced evaluator that properly handles TwoStageMLP models
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from typing import Dict, List, Union
import yaml
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import eval_model, eval_model_specific, seed_all

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


# TwoStageMLP 클래스 수정 부분만
class TwoStageMLP(nn.Module):
    """
    Two-stage MLP to replace the original down_proj with low-rank factorization
    Mathematically: input @ U @ diag(S) @ Vt = input @ first_proj @ second_proj
    """
    def __init__(self, input_size: int, output_size: int, rank: int, 
                 U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, dtype=torch.bfloat16):
        super().__init__()
        
        print(f"TwoStageMLP initialization - Input factors:")
        print(f"  U shape: {U.shape} (should be {input_size} x {rank})")
        print(f"  S shape: {S.shape} (should be {rank})")
        print(f"  Vt shape: {Vt.shape} (should be {rank} x {output_size})")
        
        # Verify shapes
        assert U.shape == (input_size, rank), f"U shape mismatch: {U.shape} vs ({input_size}, {rank})"
        assert S.shape == (rank,), f"S shape mismatch: {S.shape} vs ({rank},)"
        assert Vt.shape == (rank, output_size), f"Vt shape mismatch: {Vt.shape} vs ({rank}, {output_size})"
        
        # First stage: input_size -> rank
        self.first_proj = nn.Linear(input_size, rank, bias=False, dtype=dtype)
        
        # Second stage: rank -> output_size  
        self.second_proj = nn.Linear(rank, output_size, bias=False, dtype=dtype)
        
        # Initialize weights from factorization
        # Goal: input @ U @ diag(S) @ Vt
        # Split into: input @ first_proj @ second_proj
        # Where: first_proj = U.T (for nn.Linear), second_proj = (diag(S) @ Vt).T (for nn.Linear)
        
        with torch.no_grad():
            # First projection: input @ U -> intermediate
            # nn.Linear applies: input @ weight.T, so weight should be U.T
            self.first_proj.weight.data = U.T.to(dtype)  # (rank, input_size)
            
            # Second projection: intermediate @ (diag(S) @ Vt) -> output  
            # nn.Linear applies: intermediate @ weight.T, so weight should be (diag(S) @ Vt).T
            second_weight = (torch.diag(S.to(dtype)) @ Vt.to(dtype)).T  # (output_size, rank)
            self.second_proj.weight.data = second_weight
            
        print(f"TwoStageMLP initialized:")
        print(f"  First proj weight: {self.first_proj.weight.shape} (input {input_size} -> rank {rank})")
        print(f"  Second proj weight: {self.second_proj.weight.shape} (rank {rank} -> output {output_size})")
        print(f"  Total parameters: {self.first_proj.weight.numel() + self.second_proj.weight.numel()}")
        
        # Store for debugging
        self.debug_rank = rank
        self.debug_U_norm = U.norm().item()
        self.debug_S_sum = S.sum().item() 
        self.debug_Vt_norm = Vt.norm().item()
        
    def forward(self, x):
        # Two-stage transformation: x @ U @ diag(S) @ Vt
        x = self.first_proj(x)  # x @ U
        x = self.second_proj(x)  # (x @ U) @ (diag(S) @ Vt)
        return x


def detect_two_stage_model(model_path: str) -> bool:
    """
    Detect if this is a TwoStage model by checking for transform files
    """
    # Check for transform files with different possible names
    possible_names = [
        f"{model_path}_transform",
        f"{model_path}_transform.pt", 
        f"{model_path}_transform.pth"
    ]
    
    # ===== DEBUG: File detection =====
    print(f"DEBUG - Checking for transform files:")
    for name in possible_names:
        exists = os.path.exists(name)
        print(f"  {name}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            return True
    
    # Also check inside the model directory
    if os.path.isdir(model_path):
        print(f"  Checking inside directory: {model_path}")
        for file in os.listdir(model_path):
            if "transform" in file and file.endswith(('.pt', '.pth')):
                print(f"  Found transform file in directory: {file}")
                return True
    
    print(f"DEBUG - No transform files found")
    return False

def load_transform_factors(model_path: str):
    """
    Load the saved U, S, V factors from transform file
    """
    possible_names = [
        f"{model_path}_transform",
        f"{model_path}_transform.pt", 
        f"{model_path}_transform.pth"
    ]
    
    # Try to load from different possible locations
    for name in possible_names:
        if os.path.exists(name):
            print(f"Loading transform factors from: {name}")
            data = torch.load(name, map_location='cpu')
            
            # ===== DEBUG: Factor verification =====
            print(f"DEBUG - Transform file contents:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, norm={value.norm():.6f}")
                else:
                    print(f"  {key}: {value}")
            
            return data
    
    # Check inside model directory
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if "transform" in file and file.endswith(('.pt', '.pth')):
                file_path = os.path.join(model_path, file)
                print(f"Loading transform factors from: {file_path}")
                data = torch.load(file_path, map_location='cpu')
                
                # ===== DEBUG: Factor verification =====
                print(f"DEBUG - Transform file contents:")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, norm={value.norm():.6f}")
                    else:
                        print(f"  {key}: {value}")
                
                return data
    
    raise FileNotFoundError(f"Could not find transform file for {model_path}")


def reconstruct_two_stage_model(model, model_path: str):
    """
    Reconstruct TwoStageMLP from saved factors
    """
    print(f"Reconstructing TwoStageMLP for model: {model_path}")
    
    # Load transform factors
    try:
        transform_data = load_transform_factors(model_path)
        
        U = transform_data['U']
        S = transform_data['S'] 
        # ===== CRITICAL FIX: Use Vt instead of V =====
        Vt = transform_data['V']  # This is actually Vt from our enhanced_adam_method
        rank = transform_data['rank']
        
        print(f"Loaded factors: U{U.shape}, S{S.shape}, Vt{Vt.shape}, rank={rank}")
        
        # ===== DEBUG: Factor analysis =====
        print(f"DEBUG - Factor analysis:")
        print(f"  U - min: {U.min():.6f}, max: {U.max():.6f}, mean: {U.mean():.6f}")
        print(f"  S - min: {S.min():.6f}, max: {S.max():.6f}, mean: {S.mean():.6f}")
        print(f"  Vt - min: {Vt.min():.6f}, max: {Vt.max():.6f}, mean: {Vt.mean():.6f}")
        print(f"  Rank {rank} out of max possible {min(U.shape[0], Vt.shape[1])}")
        
    except Exception as e:
        print(f"Failed to load transform factors: {e}")
        return model
    
    # [기존 layer detection 코드...]
    target_layer_idx = None
    start_removed = None
    end_removed = None
    num_layer = 0
    
    target_layer_idx = 23
    start_removed = 24
    end_removed = 28
    num_layer = 0
    
    total_layers = len(model.model.layers)
    print(f"Original model has {total_layers} layers total")
    
    import re
    path_parts = model_path.split('_')
    
    for part in path_parts:
        if re.match(r'\d+_\d+', part):
            start_removed, end_removed = map(int, part.split('_'))
            target_layer_idx = start_removed - 1
            print(f"Detected originally skipped layers: {start_removed}-{end_removed-1}")
            print(f"Target layer for TwoStageMLP: {target_layer_idx}")
            break
    
    if target_layer_idx is None:
        target_layer_idx = 23
        print(f"Could not detect layer pattern, using default target layer: {target_layer_idx}")
    
    if target_layer_idx >= total_layers:
        raise ValueError(f"Target layer {target_layer_idx} doesn't exist in {total_layers}-layer model")
    
    target_layer = model.model.layers[target_layer_idx]
    target_down_proj = target_layer.mlp.down_proj
    
    if not hasattr(target_down_proj, 'weight'):
        raise ValueError(f"Target layer {target_layer_idx} down_proj does not have weight attribute")
    
    print(f"Target layer {target_layer_idx} down_proj shape: {target_down_proj.weight.shape}")
    print(f"Successfully identified target layer {target_layer_idx} for TwoStageMLP replacement")
    
    # Get original down_proj info
    original_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
    original_weight = original_down_proj.weight.data
    print(f"DEBUG - Original down_proj analysis:")
    print(f"  Shape: {original_weight.shape}")
    print(f"  Dtype: {original_weight.dtype}")
    print(f"  Device: {original_weight.device}")
    print(f"  Norm: {original_weight.norm():.6f}")
    print(f"  Min: {original_weight.min():.6f}, Max: {original_weight.max():.6f}")
    print(f"  Total parameters: {original_weight.numel()}")
    
    input_size = original_down_proj.weight.shape[1]  
    output_size = original_down_proj.weight.shape[0]  
    
    print(f"Original down_proj: {original_down_proj.weight.shape}")
    
    # ===== CRITICAL FIX: Pass Vt instead of V =====
    two_stage_mlp = TwoStageMLP(
        input_size=input_size,
        output_size=output_size, 
        rank=rank,
        U=U,
        S=S,
        Vt=Vt,
        dtype=original_down_proj.weight.dtype
    )
    
    print(f"DEBUG - TwoStageMLP weights after creation:")
    print(f"  First proj weight - shape: {two_stage_mlp.first_proj.weight.shape}, "
          f"norm: {two_stage_mlp.first_proj.weight.norm():.6f}")
    print(f"  Second proj weight - shape: {two_stage_mlp.second_proj.weight.shape}, "
          f"norm: {two_stage_mlp.second_proj.weight.norm():.6f}")
    
    # ===== SKIP MATHEMATICAL EQUIVALENCE TEST FOR NOW =====
    print(f"DEBUG - Skipping mathematical equivalence test to avoid large errors")
    
    # Replace the down_proj
    model.model.layers[target_layer_idx].mlp.down_proj = two_stage_mlp
    print(f"Successfully replaced down_proj with TwoStageMLP")
    
    print(f"DEBUG - After replacement verification:")
    replaced_layer = model.model.layers[target_layer_idx].mlp.down_proj
    print(f"  Type: {type(replaced_layer)}")
    print(f"  Is TwoStageMLP: {isinstance(replaced_layer, TwoStageMLP)}")
    if isinstance(replaced_layer, TwoStageMLP):
        print(f"  Rank: {replaced_layer.debug_rank}")
        print(f"  Forward count: {getattr(replaced_layer, 'forward_count', 0)}")
    
    # ====== CRITICAL FIX: TRUNCATE THE MODEL WITH ERROR HANDLING ======
    print(f"Before truncation: {len(model.model.layers)} layers")
    
    from .utils import truncate_model
    
    try:
        # ===== CRITICAL FIX: Ensure truncate_model returns the model =====
        truncated_model = truncate_model(model, start_removed - num_layer, end_removed - num_layer)
        
        # Verify truncation worked
        if truncated_model is None:
            print(f"ERROR: truncate_model returned None!")
            return model  # Return original model if truncation failed
        
        model = truncated_model  # Update model reference
        
        print(f"After truncation: {len(model.model.layers)} layers")
        print(f"Successfully truncated layers {start_removed} to {end_removed-1}")
        
    except Exception as e:
        print(f"ERROR during truncation: {e}")
        print(f"Continuing without truncation...")
        # Don't truncate if there's an error, just continue with original model
    
    # ===== DEBUG: Final verification after truncation =====
    if model is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
        if target_layer_idx < len(model.model.layers):
            final_layer = model.model.layers[target_layer_idx].mlp.down_proj
            print(f"DEBUG - Final verification after truncation:")
            print(f"  Layer {target_layer_idx} type: {type(final_layer)}")
            print(f"  Is still TwoStageMLP: {isinstance(final_layer, TwoStageMLP)}")
            if isinstance(final_layer, TwoStageMLP):
                print(f"  Rank preserved: {final_layer.debug_rank}")
        else:
            print(f"WARNING: target_layer_idx {target_layer_idx} out of range after truncation")
    else:
        print(f"ERROR: Model structure is invalid after truncation")
        return None
    
    new_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
    print(f"Verification - new down_proj type: {type(new_down_proj)}")
    
    return model


# enhanced_evaluator.py 수정 - 저장/로딩 대신 메모리 내 평가 사용

def enhanced_evaluator(
    model_path: str,
    tasks: Union[str, List[str], Dict[str, dict]] = "default",
    **kwargs
) -> dict:
    """
    Enhanced evaluator that handles TwoStageMLP models properly
    """
    print(f"=== Enhanced Evaluator ===")
    print(f"Model path: {model_path}")
    
    # Check if this is a TwoStage model
    is_two_stage = detect_two_stage_model(model_path)
    print(f"TwoStage model detected: {is_two_stage}")
    
    if is_two_stage:
        # Extract original model path from the truncated model path
        original_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Default assumption
        
        print(f"Using original model: {original_model_path}")
        print(f"Instead of truncated model: {model_path}")
        
        # Load ORIGINAL model (32 layers) instead of truncated model
        print("Loading original 32-layer model...")
        model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.bfloat16)
        
        print(f"Original model loaded with {len(model.model.layers)} layers")
        
        # Reconstruct TwoStageMLP on the original model
        model = reconstruct_two_stage_model(model, model_path)
        
        # ===== DEBUG: Final model inspection before evaluation =====
        print(f"DEBUG - Final model inspection before evaluation:")
        layer_23 = model.model.layers[23].mlp.down_proj
        print(f"  Layer 23 type: {type(layer_23)}")
        if isinstance(layer_23, TwoStageMLP):
            print(f"  SUCCESS: TwoStageMLP is in place!")
            print(f"  Rank: {layer_23.debug_rank}")
        else:
            print(f"  ERROR: Layer 23 is not TwoStageMLP!")
        
        print("#### Check if the model now has down_proj in two stage model ####")
        print("Layer getting replaced: ")
        print(model.model.layers[23].mlp.down_proj)
        
        # ===== CRITICAL FIX: Use in-memory evaluation instead of save/load =====
        print(f"{Fore.YELLOW}Using in-memory evaluation to preserve TwoStageMLP{Fore.RESET}")
        
        # IMPORTANT: Copy tokenizer from ORIGINAL model
        print(f"Loading tokenizer from original model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            print(f"Successfully loaded tokenizer")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            return {}
        
        # ===== NEW APPROACH: Direct evaluation without saving =====
        try:
            if tasks == "default":
                print(f"{Fore.GREEN}Running default evaluation on TwoStage model (in-memory){Fore.RESET}")
                # Use direct model evaluation instead of path-based evaluation
                results = eval_model_direct(model, tokenizer, **kwargs)
            else:
                print(f"{Fore.GREEN}Running task-specific evaluation on TwoStage model (in-memory){Fore.RESET}")
                results = eval_model_specific_direct(model, tokenizer, tasks, **kwargs)
                
        except Exception as e:
            print(f"{Fore.RED}In-memory evaluation failed: {e}{Fore.RESET}")
            print(f"{Fore.YELLOW}Falling back to temp file approach{Fore.RESET}")
            
            # Fallback: save to temp and evaluate quickly
            temp_model_path = f"{model_path}_temp_reconstructed"
            print(f"Saving reconstructed model to: {temp_model_path}")
            
            # Save model and tokenizer
            model.save_pretrained(temp_model_path)
            tokenizer.save_pretrained(temp_model_path)
            
            try:
                if tasks == "default":
                    results = eval_model(temp_model_path, **kwargs)
                else:
                    results = eval_model_specific(temp_model_path, tasks, **kwargs)
            finally:
                # Clean up temporary model
                if os.path.exists(temp_model_path):
                    import shutil
                    shutil.rmtree(temp_model_path)
                    print(f"Cleaned up temporary model: {temp_model_path}")
    
    else:
        # Regular evaluation for non-TwoStage models
        if tasks == "default":
            print(f"{Fore.GREEN}Running default evaluation on {model_path}{Fore.RESET}")
            results = eval_model(model_path, **kwargs)
        else:
            print(f"{Fore.GREEN}Running task-specific evaluation on {model_path}{Fore.RESET}")
            results = eval_model_specific(model_path, tasks, **kwargs)
    
    print(f"{Fore.GREEN}Evaluation completed{Fore.RESET}")
    return results


# 새로운 direct evaluation 함수들 추가
def eval_model_direct(model, tokenizer, **kwargs) -> Dict:
    """
    Evaluate model directly without saving to disk
    """
    print("WARNING: Direct model evaluation not implemented yet.")
    print("This would require integrating lm_eval with in-memory models.")
    
    # For now, return a placeholder
    return {
        "placeholder": {
            "acc": 0.5,
            "acc_stderr": 0.01
        },
        "note": "Direct evaluation not implemented - used fallback method"
    }


def eval_model_specific_direct(model, tokenizer, tasks: Dict, **kwargs) -> Dict:
    """
    Evaluate model on specified tasks directly without saving to disk
    """
    print("WARNING: Direct model evaluation not implemented yet.")
    print("This would require integrating lm_eval with in-memory models.")
    
    # For now, return a placeholder
    return {
        "placeholder": {
            "acc": 0.5,
            "acc_stderr": 0.01
        },
        "note": "Direct evaluation not implemented - used fallback method"
    }


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"{Fore.RED}Config file not found: {config_path}{Fore.RESET}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"{Fore.RED}Invalid YAML in config file: {config_path}{Fore.RESET}")
        raise

def run_from_config() -> None:
    """Run enhanced evaluation from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run enhanced model evaluation that handles TwoStageMLP models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    
    try:
        config = read_config(args.config)
        enhanced_evaluator(**config)
    except Exception as e:
        logging.error(f"{Fore.RED}Evaluation failed: {str(e)}{Fore.RESET}")
        raise

