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


# reconstruct_two_stage_model 함수에서 factors 로딩 부분 수정
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
    
    # [기존 layer detection 코드는 그대로 유지...]
    # Find which layer to replace (adapted for truncated models)
    target_layer_idx = None
    start_removed = None
    end_removed = None
    num_layer = 0  # This should be extracted from the path as well
    
    target_layer_idx = 23  # Default assumption
    start_removed = 24
    end_removed = 28
    num_layer = 0
    
    # Find which layer to replace (now working with original 32-layer model)
    total_layers = len(model.model.layers)
    print(f"Original model has {total_layers} layers total")
    
    # Extract layer information from the truncated model path
    # Pattern: "24_28" means layers 24-27 were originally skipped
    target_layer_idx = None
    
    import re
    path_parts = model_path.split('_')
    
    for part in path_parts:
        if re.match(r'\d+_\d+', part):  # Find pattern like "24_28"
            start_removed, end_removed = map(int, part.split('_'))
            # Target should be the layer just before the removed section
            target_layer_idx = start_removed - 1  # Layer 23 for "24_28"
            print(f"Detected originally skipped layers: {start_removed}-{end_removed-1}")
            print(f"Target layer for TwoStageMLP: {target_layer_idx}")
            break
    
    # Fallback if pattern not found
    if target_layer_idx is None:
        target_layer_idx = 23  # Default assumption
        print(f"Could not detect layer pattern, using default target layer: {target_layer_idx}")
    
    # Verify target layer exists and has correct structure
    if target_layer_idx >= total_layers:
        raise ValueError(f"Target layer {target_layer_idx} doesn't exist in {total_layers}-layer model")
    
    target_layer = model.model.layers[target_layer_idx]
    target_down_proj = target_layer.mlp.down_proj
    
    if not hasattr(target_down_proj, 'weight'):
        raise ValueError(f"Target layer {target_layer_idx} down_proj does not have weight attribute")
    
    print(f"Target layer {target_layer_idx} down_proj shape: {target_down_proj.weight.shape}")
    print(f"Successfully identified target layer {target_layer_idx} for TwoStageMLP replacement")
    
    # ===== DEBUG: Original layer analysis =====
    original_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
    original_weight = original_down_proj.weight.data
    print(f"DEBUG - Original down_proj analysis:")
    print(f"  Shape: {original_weight.shape}")
    print(f"  Dtype: {original_weight.dtype}")
    print(f"  Device: {original_weight.device}")
    print(f"  Norm: {original_weight.norm():.6f}")
    print(f"  Min: {original_weight.min():.6f}, Max: {original_weight.max():.6f}")
    print(f"  Total parameters: {original_weight.numel()}")
    
    # Get original down_proj info
    input_size = original_down_proj.weight.shape[1]  
    output_size = original_down_proj.weight.shape[0]  
    
    print(f"Original down_proj: {original_down_proj.weight.shape}")
    
    # ===== CRITICAL FIX: Pass Vt instead of V =====
    # Create and replace with TwoStageMLP
    two_stage_mlp = TwoStageMLP(
        input_size=input_size,
        output_size=output_size, 
        rank=rank,
        U=U,
        S=S,
        Vt=Vt,  # Pass Vt, not V
        dtype=original_down_proj.weight.dtype
    )
    
    # [나머지 코드는 그대로 유지...]
    # ===== DEBUG: Verify TwoStageMLP weights after creation =====
    print(f"DEBUG - TwoStageMLP weights after creation:")
    print(f"  First proj weight - shape: {two_stage_mlp.first_proj.weight.shape}, "
          f"norm: {two_stage_mlp.first_proj.weight.norm():.6f}")
    print(f"  Second proj weight - shape: {two_stage_mlp.second_proj.weight.shape}, "
          f"norm: {two_stage_mlp.second_proj.weight.norm():.6f}")
    
    # Test mathematical equivalence
    print(f"DEBUG - Testing mathematical equivalence:")
    with torch.no_grad():
        # Check factor shapes first
        print(f"  U shape: {U.shape}, Vt shape: {Vt.shape}, S shape: {S.shape}")
        print(f"  Expected final matrix shape: ({input_size}, {output_size})")
        
        # Now shapes should match for proper reconstruction test
        if U.shape[0] == input_size and Vt.shape[1] == output_size and U.shape[1] == Vt.shape[0]:
            # ===== CRITICAL FIX: Ensure consistent dtype =====
            # Convert all factors to the same dtype as the original model
            target_dtype = original_down_proj.weight.dtype
            U_test = U.to(target_dtype)
            S_test = S.to(target_dtype) 
            Vt_test = Vt.to(target_dtype)
            
            # Reconstruct full matrix from factors
            reconstructed_T = U_test @ torch.diag(S_test) @ Vt_test
            print(f"  Reconstructed T shape: {reconstructed_T.shape}, norm: {reconstructed_T.norm():.6f}")
            print(f"  Reconstructed T dtype: {reconstructed_T.dtype}")
            
            # Test on random input with matching dtype
            test_input = torch.randn(2, input_size, dtype=target_dtype)
            print(f"  Test input dtype: {test_input.dtype}")
            
            # Original transformation output
            original_output = test_input @ reconstructed_T
            
            # TwoStageMLP output  
            two_stage_output = two_stage_mlp(test_input)
            
            # Check difference
            diff = (original_output - two_stage_output).norm()
            print(f"  Reconstruction error on test input: {diff:.8f}")
            if diff > 1e-3:
                print(f"  WARNING: Large reconstruction error!")
            else:
                print(f"  SUCCESS: Mathematical equivalence verified!")
        else:
            print(f"  WARNING: Shape mismatch - U: {U.shape}, Vt: {Vt.shape}")
            print(f"  Expected U: ({input_size}, rank), Vt: (rank, {output_size})")
    
    # Replace the down_proj
    model.model.layers[target_layer_idx].mlp.down_proj = two_stage_mlp
    print(f"Successfully replaced down_proj with TwoStageMLP")

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
        # Pattern: "meta-llama_Meta-Llama-3-8B-Instruct_4_layers_24_28_..."
        # We want: "meta-llama/Meta-Llama-3-8B-Instruct"
        
        path_parts = model_path.split('/')[-1].split('_')  # Get filename and split
        
        # Reconstruct original model path
        # if path_parts[0] == 'meta-llama' and len(path_parts) >= 2:
        #     original_model_path = f"{path_parts[0]}/{'-'.join(path_parts[1:3])}"  # meta-llama/Meta-Llama-3-8B-Instruct
        # else:
        # Fallback: try to extract from the path
        original_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Default assumption
        
        print(f"Using original model: {original_model_path}")
        print(f"Instead of truncated model: {model_path}")
        
        # Load ORIGINAL model (32 layers) instead of truncated model
        print("Loading original 32-layer model...")
        model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.bfloat16)
        
        print(f"Original model loaded with {len(model.model.layers)} layers")
        
        # Reconstruct TwoStageMLP on the original model
        model = reconstruct_two_stage_model(model, model_path)
        
        # ===== DEBUG: Final model inspection before saving =====
        print(f"DEBUG - Final model inspection before saving:")
        layer_23 = model.model.layers[23].mlp.down_proj
        print(f"  Layer 23 type: {type(layer_23)}")
        if isinstance(layer_23, TwoStageMLP):
            print(f"  SUCCESS: TwoStageMLP is in place!")
            print(f"  Rank: {layer_23.debug_rank}")
        else:
            print(f"  ERROR: Layer 23 is not TwoStageMLP!")
        
        # Save the reconstructed model temporarily for evaluation
        temp_model_path = f"{model_path}_temp_reconstructed"
        print(f"Saving reconstructed model to: {temp_model_path}")
        model.save_pretrained(temp_model_path)

        print("#### Check if the model now has down_proj in two stage model ####")
        print("Entire Model: ")
        print(model)
        print("Layer getting replaced: ")
        print(model.model.layers[23].mlp.down_proj)
        
        # ===== DEBUG: Verify saved model =====
        print(f"DEBUG - Verifying saved model:")
        try:
            saved_model = AutoModelForCausalLM.from_pretrained(temp_model_path, torch_dtype=torch.bfloat16)
            saved_layer_23 = saved_model.model.layers[23].mlp.down_proj
            print(f"  Saved model layer 23 type: {type(saved_layer_23)}")
            if not isinstance(saved_layer_23, TwoStageMLP):
                print(f"  WARNING: TwoStageMLP was not preserved in saving!")
                print(f"  This might be why rank changes don't affect results!")
            del saved_model  # Clean up
        except Exception as e:
            print(f"  ERROR loading saved model for verification: {e}")
        
        # IMPORTANT: Copy tokenizer from ORIGINAL model (not truncated)
        print(f"Copying tokenizer from original model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(original_model_path)  # Use original path!
            tokenizer.save_pretrained(temp_model_path)
            print(f"Successfully copied tokenizer to: {temp_model_path}")
        except Exception as e:
            print(f"Warning: Could not copy tokenizer: {e}")
            print("Evaluation might fail due to missing tokenizer files")
        
        # Now evaluate using the reconstructed model
        try:
            if tasks == "default":
                print(f"{Fore.GREEN}Running default evaluation on reconstructed TwoStage model{Fore.RESET}")
                print(f"{Fore.YELLOW}Using temp model path: {temp_model_path}{Fore.RESET}")
                results = eval_model(temp_model_path, **kwargs)
            else:
                print(f"{Fore.GREEN}Running task-specific evaluation on reconstructed TwoStage model{Fore.RESET}")
                print(f"{Fore.YELLOW}Using temp model path: {temp_model_path}{Fore.RESET}")
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



# """
# Enhanced evaluator that properly handles TwoStageMLP models
# """

# import argparse
# import logging
# import os
# import torch
# import torch.nn as nn
# from typing import Dict, List, Union
# import yaml
# from colorama import Fore, init
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from .utils import eval_model, eval_model_specific, seed_all

# # Initialize colorama for Windows compatibility
# init(autoreset=True)

# # Configure logging to display colored messages and timestamps
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

# class TwoStageMLP(nn.Module):
#     """
#     Two-stage MLP to replace the original down_proj with low-rank factorization
#     """
#     def __init__(self, input_size: int, output_size: int, rank: int, 
#                  U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, dtype=torch.bfloat16):
#         super().__init__()
        
#         # First stage: input_size -> rank
#         self.first_proj = nn.Linear(input_size, rank, bias=False, dtype=dtype)
        
#         # Second stage: rank -> output_size  
#         self.second_proj = nn.Linear(rank, output_size, bias=False, dtype=dtype)
        
#         # Initialize weights from factorization
#         with torch.no_grad():
#             # First projection: compress input to rank dimensions
#             first_weight = V.to(dtype) @ torch.diag(S.to(dtype))  # [4096, rank]
#             self.first_proj.weight.data = first_weight.T  # [rank, 4096] for nn.Linear
            
#             # Second projection: map from rank back to output
#             self.second_proj.weight.data = U.T.to(dtype)  # [output_size, rank]
            
#         print(f"TwoStageMLP reconstructed:")
#         print(f"  First proj: {self.first_proj.weight.shape} (input {input_size} -> rank {rank})")
#         print(f"  Second proj: {self.second_proj.weight.shape} (rank {rank} -> output {output_size})")
#         print(f"  Total parameters: {self.first_proj.weight.numel() + self.second_proj.weight.numel()}")
        
#     def forward(self, x):
#         # Two-stage transformation
#         x = self.first_proj(x)
#         x = self.second_proj(x)
#         return x

# def detect_two_stage_model(model_path: str) -> bool:
#     """
#     Detect if this is a TwoStage model by checking for transform files
#     """
#     # Check for transform files with different possible names
#     possible_names = [
#         f"{model_path}_transform",
#         f"{model_path}_transform.pt", 
#         f"{model_path}_transform.pth"
#     ]
    
#     for name in possible_names:
#         if os.path.exists(name):
#             return True
    
#     # Also check inside the model directory
#     if os.path.isdir(model_path):
#         for file in os.listdir(model_path):
#             if "transform" in file and file.endswith(('.pt', '.pth')):
#                 return True
                
#     return False

# def load_transform_factors(model_path: str):
#     """
#     Load the saved U, S, V factors from transform file
#     """
#     possible_names = [
#         f"{model_path}_transform",
#         f"{model_path}_transform.pt", 
#         f"{model_path}_transform.pth"
#     ]
    
#     # Try to load from different possible locations
#     for name in possible_names:
#         if os.path.exists(name):
#             print(f"Loading transform factors from: {name}")
#             return torch.load(name, map_location='cpu')
    
#     # Check inside model directory
#     if os.path.isdir(model_path):
#         for file in os.listdir(model_path):
#             if "transform" in file and file.endswith(('.pt', '.pth')):
#                 file_path = os.path.join(model_path, file)
#                 print(f"Loading transform factors from: {file_path}")
#                 return torch.load(file_path, map_location='cpu')
    
#     raise FileNotFoundError(f"Could not find transform file for {model_path}")

# def reconstruct_two_stage_model(model, model_path: str):
#     """
#     Reconstruct TwoStageMLP from saved factors
#     """
#     print(f"Reconstructing TwoStageMLP for model: {model_path}")
    
#     # Load transform factors
#     try:
#         transform_data = load_transform_factors(model_path)
        
#         U = transform_data['U']
#         S = transform_data['S'] 
#         V = transform_data['V']
#         rank = transform_data['rank']
        
#         print(f"Loaded factors: U{U.shape}, S{S.shape}, V{V.shape}, rank={rank}")
        
#     except Exception as e:
#         print(f"Failed to load transform factors: {e}")
#         return model
    
#     # Find which layer to replace (adapted for truncated models)
#     target_layer_idx = None
#     start_removed = None
#     end_removed = None
#     num_layer = 0  # This should be extracted from the path as well
    
#     target_layer_idx = 23  # Default assumption
#     start_removed = 24
#     end_removed = 28
#     num_layer = 0
    
#     # Find which layer to replace (now working with original 32-layer model)
#     total_layers = len(model.model.layers)
#     print(f"Original model has {total_layers} layers total")
    
#     # Extract layer information from the truncated model path
#     # Pattern: "24_28" means layers 24-27 were originally skipped
#     target_layer_idx = None
    
#     import re
#     path_parts = model_path.split('_')
    
#     for part in path_parts:
#         if re.match(r'\d+_\d+', part):  # Find pattern like "24_28"
#             start_removed, end_removed = map(int, part.split('_'))
#             # Target should be the layer just before the removed section
#             target_layer_idx = start_removed - 1  # Layer 23 for "24_28"
#             print(f"Detected originally skipped layers: {start_removed}-{end_removed-1}")
#             print(f"Target layer for TwoStageMLP: {target_layer_idx}")
#             break
    
#     # Fallback if pattern not found
#     if target_layer_idx is None:
#         target_layer_idx = 23  # Default assumption
#         print(f"Could not detect layer pattern, using default target layer: {target_layer_idx}")
    
#     # Verify target layer exists and has correct structure
#     if target_layer_idx >= total_layers:
#         raise ValueError(f"Target layer {target_layer_idx} doesn't exist in {total_layers}-layer model")
    
#     target_layer = model.model.layers[target_layer_idx]
#     target_down_proj = target_layer.mlp.down_proj
    
#     if not hasattr(target_down_proj, 'weight'):
#         raise ValueError(f"Target layer {target_layer_idx} down_proj does not have weight attribute")
    
#     print(f"Target layer {target_layer_idx} down_proj shape: {target_down_proj.weight.shape}")
#     print(f"Successfully identified target layer {target_layer_idx} for TwoStageMLP replacement")
    
#     # Get original down_proj info
#     original_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
#     input_size = original_down_proj.weight.shape[1]  
#     output_size = original_down_proj.weight.shape[0]  
    
#     print(f"Original down_proj: {original_down_proj.weight.shape}")
    
#     # Create and replace with TwoStageMLP
#     two_stage_mlp = TwoStageMLP(
#         input_size=input_size,
#         output_size=output_size, 
#         rank=rank,
#         U=U,
#         S=S,
#         V=V,
#         dtype=original_down_proj.weight.dtype
#     )
    
#     # Replace the down_proj
#     model.model.layers[target_layer_idx].mlp.down_proj = two_stage_mlp
#     print(f"Successfully replaced down_proj with TwoStageMLP")
    
#     # ====== CRITICAL FIX: TRUNCATE THE MODEL ======
#     # This is the missing step! We need to remove layers 24-27 (start_removed to end_removed-1)
#     print(f"Before truncation: {len(model.model.layers)} layers")
    
#     # Import the truncate_model function
#     from .utils import truncate_model
    
#     # Truncate the model to remove the replaced layers
#     # Note: We use start_removed-num_layer and end_removed-num_layer just like in cosine_dist.py
#     model = truncate_model(model, start_removed - num_layer, end_removed - num_layer)
    
#     print(f"After truncation: {len(model.model.layers)} layers")
#     print(f"Successfully truncated layers {start_removed} to {end_removed-1}")
    
#     # Verify the replacement
#     new_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
#     print(f"Verification - new down_proj type: {type(new_down_proj)}")
    
#     return model

# def enhanced_evaluator(
#     model_path: str,
#     tasks: Union[str, List[str], Dict[str, dict]] = "default",
#     **kwargs
# ) -> dict:
#     """
#     Enhanced evaluator that handles TwoStageMLP models properly
#     """
#     print(f"=== Enhanced Evaluator ===")
#     print(f"Model path: {model_path}")
    
#     # Check if this is a TwoStage model
#     is_two_stage = detect_two_stage_model(model_path)
#     print(f"TwoStage model detected: {is_two_stage}")
    
#     if is_two_stage:
#         # Extract original model path from the truncated model path
#         # Pattern: "meta-llama_Meta-Llama-3-8B-Instruct_4_layers_24_28_..."
#         # We want: "meta-llama/Meta-Llama-3-8B-Instruct"
        
#         path_parts = model_path.split('/')[-1].split('_')  # Get filename and split
        
#         # Reconstruct original model path
#         # if path_parts[0] == 'meta-llama' and len(path_parts) >= 2:
#         #     original_model_path = f"{path_parts[0]}/{'-'.join(path_parts[1:3])}"  # meta-llama/Meta-Llama-3-8B-Instruct
#         # else:
#         # Fallback: try to extract from the path
#         original_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Default assumption
        
#         print(f"Using original model: {original_model_path}")
#         print(f"Instead of truncated model: {model_path}")
        
#         # Load ORIGINAL model (32 layers) instead of truncated model
#         print("Loading original 32-layer model...")
#         model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.bfloat16)
        
#         print(f"Original model loaded with {len(model.model.layers)} layers")
        
#         # Reconstruct TwoStageMLP on the original model
#         model = reconstruct_two_stage_model(model, model_path)
        
#         # Save the reconstructed model temporarily for evaluation
#         temp_model_path = f"{model_path}_temp_reconstructed"
#         print(f"Saving reconstructed model to: {temp_model_path}")
#         model.save_pretrained(temp_model_path)

#         print("#### Check if the model now has down_proj in two stage model ####")
#         print("Entire Model: ")
#         print(model)
#         print("Layer getting replaced: ")
#         print(model.model.layers[23].mlp.down_proj)
        
#         # IMPORTANT: Copy tokenizer from ORIGINAL model (not truncated)
#         print(f"Copying tokenizer from original model...")
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(original_model_path)  # Use original path!
#             tokenizer.save_pretrained(temp_model_path)
#             print(f"Successfully copied tokenizer to: {temp_model_path}")
#         except Exception as e:
#             print(f"Warning: Could not copy tokenizer: {e}")
#             print("Evaluation might fail due to missing tokenizer files")
        
#         # Now evaluate using the reconstructed model
#         try:
#             if tasks == "default":
#                 print(f"{Fore.GREEN}Running default evaluation on reconstructed TwoStage model{Fore.RESET}")
#                 print(f"{Fore.YELLOW}Using temp model path: {temp_model_path}{Fore.RESET}")
#                 results = eval_model(temp_model_path, **kwargs)
#             else:
#                 print(f"{Fore.GREEN}Running task-specific evaluation on reconstructed TwoStage model{Fore.RESET}")
#                 print(f"{Fore.YELLOW}Using temp model path: {temp_model_path}{Fore.RESET}")
#                 results = eval_model_specific(temp_model_path, tasks, **kwargs)
#         finally:
#             # Clean up temporary model
#             if os.path.exists(temp_model_path):
#                 import shutil
#                 shutil.rmtree(temp_model_path)
#                 print(f"Cleaned up temporary model: {temp_model_path}")
    
#     else:
#         # Regular evaluation for non-TwoStage models
#         if tasks == "default":
#             print(f"{Fore.GREEN}Running default evaluation on {model_path}{Fore.RESET}")
#             results = eval_model(model_path, **kwargs)
#         else:
#             print(f"{Fore.GREEN}Running task-specific evaluation on {model_path}{Fore.RESET}")
#             results = eval_model_specific(model_path, tasks, **kwargs)
    
#     print(f"{Fore.GREEN}Evaluation completed{Fore.RESET}")
#     return results

# def read_config(config_path: str) -> dict:
#     """Read and parse YAML configuration file."""
#     try:
#         with open(config_path, "r") as f:
#             return yaml.safe_load(f)
#     except FileNotFoundError as e:
#         logging.error(f"{Fore.RED}Config file not found: {config_path}{Fore.RESET}")
#         raise
#     except yaml.YAMLError as e:
#         logging.error(f"{Fore.RED}Invalid YAML in config file: {config_path}{Fore.RESET}")
#         raise

# def run_from_config() -> None:
#     """Run enhanced evaluation from configuration file."""
#     parser = argparse.ArgumentParser(
#         description="Run enhanced model evaluation that handles TwoStageMLP models."
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#         help="Path to the configuration YAML file.",
#     )
#     args = parser.parse_args()
    
#     try:
#         config = read_config(args.config)
#         enhanced_evaluator(**config)
#     except Exception as e:
#         logging.error(f"{Fore.RED}Evaluation failed: {str(e)}{Fore.RESET}")
#         raise