import argparse
import gc
import logging
import os
from typing import Optional
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (adam_method, get_calib_dataloader, optimizing_method,
                    select_non_overlapping_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def cosine_dist(
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
    save_transform_only: bool = False,
    diag: bool = False,
    loss: str = "cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool= False
    
) -> str:
    """Calculate cosine distance between model layers and apply transformations.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of start layer for cut
        token: Authentication token
        save_transform_only: Whether to only save the transform
        diag: Whether to use diagonal matrix
        loss: Loss function type
        solver: Optimization solver type
        thri: Whether to use three vectors
        two_vectors: Whether to use two vectors
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
    
    Returns:
        Path where transformed model is saved
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    
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
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
        def hook(module, input, output):
            # Detach to avoid keeping computation history
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    if 'falcon' in model_path.lower():
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    mlp_activations = {}
    a1 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    if accurate:
        print("ACCURATE MODE IS ON (MORE MEMORY IS NEEDED")
        a3 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
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
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]    
    if solver == "adam":
        transform = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Apply transformation
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (transform.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    model.save_pretrained(f"{save_path}_ReplaceMe_{loss}_{solver}")
    tokenizer.save_pretrained(f"{save_path}_ReplaceMe_{loss}_{solver}")
    
    if save_transform_only:
        torch.save(transform, f"{save_path}_ReplaceMe_{loss}_{solver}_transform")
    
    # Final cleanup
    del model, a1, a2
    if accurate:
      del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_ReplaceMe_{loss}_{solver}"


def cosine_dist_llava(
    model_path: str,
    dataset: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,
    loss: str = "cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False
) -> str:
    """Calculate cosine distance for LLaVA and apply transformations."""
    print(f"\n{'='*60}")
    print(f"[DEBUG] Starting LLaVA Transformation Estimation")
    print(f"{'='*60}")
    print(f"[DEBUG] Model: {model_path}")
    print(f"[DEBUG] Method: {loss} loss with {solver} solver")
    print(f"[DEBUG] Pruning layers {start_id} to {end_id}")
    print(f"[DEBUG] Dataset: {dataset}, Size: {dataset_size}")
    
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
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
    print("[DEBUG] Loading LLaVA model for activation collection...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    num_layers = model.config.text_config.num_hidden_layers
    hidden_size = model.config.text_config.hidden_size
    print(f"[DEBUG] Model config - Layers: {num_layers}, Hidden size: {hidden_size}")
    
    # Load processor
    print("[DEBUG] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    model.eval()
    
    # Get dataloader
    print("[DEBUG] Loading calibration dataloader...")
    dataloader = get_calib_dataloader_llava(dataset, dataset_size, batch_size, processor)
    
    # Setup MLP hooks
    print("[DEBUG] Setting up MLP hooks...")
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    for i, layer in enumerate(model.language_model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    print(f"[DEBUG] Registered {len(hooks)} MLP hooks")
    
    # Initialize activation storage
    mlp_activations = {}
    print(f"[DEBUG] Allocating activation storage...")
    print(f"[DEBUG] Expected total tokens: {dataset_size * max_length}")
    
    a1 = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    
    if accurate:
        print("[DEBUG] ACCURATE MODE: Allocating a3 tensor")
        a3 = torch.empty(
            (dataset_size * max_length, hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    # Collect activations
    cnt = 0
    batch_count = 0
    
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering LLaVA Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
        batch_count += 1
        print(f"\n[DEBUG] Batch {batch_count}/{len(dataloader)}")
        
        try:
            # Process inputs
            inputs = processor(
                text=[item['text'] for item in batch],
                images=[item['image'] for item in batch],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}, Device: {inputs['input_ids'].device}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process batch: {e}")
            continue
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get activations
        hidden_states = outputs.hidden_states
        print(f"[DEBUG] Got {len(hidden_states)} hidden states")
        
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(num_layers)
        ]
        
        # Extract relevant layers
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]
        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]
        
        print(f"[DEBUG] MLP activation shape: {hidden_states_mlp.shape}")
        print(f"[DEBUG] Hidden state i shape: {hidden_states_i.shape}")
        print(f"[DEBUG] Hidden state n shape: {hidden_states_n.shape}")
        
        # Reshape and convert to float64
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64).cpu()
        hidden_states_i = hidden_states_i.view(-1, hidden_size).to(torch.float64).cpu()
        hidden_states_n = hidden_states_n.view(-1, hidden_size).to(torch.float64).cpu()
        
        # Prepare batch tensors
        a1_batch = hidden_states_mlp
        
        if accurate:
            a2_batch = hidden_states_n
            a3_batch = hidden_states_i - hidden_states_mlp
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch.to(torch.bfloat16)
        else:
            a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        # Store activations
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch.to(torch.bfloat16)
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch.to(torch.bfloat16)
        
        print(f"[DEBUG] Stored activations at indices {cnt}:{cnt+a1_batch.shape[0]}")
        cnt += a1_batch.shape[0]
        
        # Cleanup
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        del outputs, inputs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Truncate to actual size
    print(f"\n[DEBUG] Total activations collected: {cnt}")
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
    
    print(f"[DEBUG] Final activation shapes - a1: {a1.shape}, a2: {a2.shape}")
    
    # Estimate transformation
    print(f"\n[DEBUG] Estimating transformation using {solver} solver...")
    if solver == "adam":
        transform = adam_method(
            a1, a2, 
            a3=a3 if accurate else None,
            loss=loss, 
            diag=diag, 
            two_vectors=two_vectors, 
            thri=thri
        )
    else:
        transform = optimizing_method(
            a1, a2,
            a3=a3 if accurate else None,
            solver=solver
        )
    
    print(f"[DEBUG] Transform shape: {transform.shape}, dtype: {transform.dtype}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    print("[DEBUG] Removed all MLP hooks")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    print("\n[DEBUG] Reloading model for transformation application...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    
    print(f"[DEBUG] Original number of layers: {len(model.language_model.model.layers)}")
    
    # Truncate language model
    print(f"[DEBUG] Truncating layers from {start_id - num_layer} to {end_id - num_layer}")
    model.language_model.config.num_hidden_layers -= (end_id - num_layer - start_id + num_layer)
    model.language_model.model.layers = nn.ModuleList([
        layer for idx, layer in enumerate(model.language_model.model.layers)
        if idx < start_id - num_layer or idx >= end_id - num_layer
    ])
    
    print(f"[DEBUG] New number of layers: {len(model.language_model.model.layers)}")
    
    # Apply transformation
    target_layer_idx = start_id - num_layer - 1
    print(f"\n[DEBUG] Applying transformation to layer {target_layer_idx}")
    
    target_layer = model.language_model.model.layers[target_layer_idx]
    original_weight = target_layer.mlp.down_proj.weight
    print(f"[DEBUG] Original down_proj weight shape: {original_weight.shape}, dtype: {original_weight.dtype}")
    
    # Compute transformed weight
    transform_cpu = transform.T.cpu().to(torch.float64)
    weight_float64 = original_weight.to(torch.float64)
    
    print(f"[DEBUG] Transform shape: {transform_cpu.shape}")
    print(f"[DEBUG] Weight shape: {weight_float64.shape}")
    
    new_weight = (transform_cpu @ weight_float64).to(torch.bfloat16)
    print(f"[DEBUG] New weight shape: {new_weight.shape}, dtype: {new_weight.dtype}")
    
    target_layer.mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    print("[DEBUG] Transformation applied successfully")
    
    # Prepare save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}_llava"
        ).replace("/", "_")
    
    final_path = f"{save_path}_ReplaceMe_{loss}_{solver}"
    
    # Save model
    print(f"\n[DEBUG] Saving model to {final_path}")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print("[DEBUG] Model and processor saved")
    
    # Save transform if requested
    if save_transform_only:
        transform_path = f"{final_path}_transform.pth"
        torch.save(transform, transform_path)
        print(f"[DEBUG] Transform saved to {transform_path}")
    
    # Final cleanup
    del model, a1, a2, transform
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] LLaVA transformation complete!")
    print(f"[SUCCESS] Model saved to: {final_path}")
    print(f"{'='*60}\n")
    
    return final_path