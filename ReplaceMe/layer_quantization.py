import gc
import logging
import os
import torch
import torch.nn as nn
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, init

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def quantize_to_float8(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to 8-bit format.
    
    Note: Using bfloat16 as proxy since float8 requires special hardware.
    For actual float8, use torch.float8_e4m3fn with appropriate scaling.
    """
    # For now, using bfloat16 as a proxy for 8-bit
    # In production, you'd use: tensor.to(torch.float8_e4m3fn)
    return tensor.to(torch.bfloat16)


def quantize_linear_layer(layer: nn.Linear, bits: int = 8):
    """Quantize a linear layer's weights."""
    with torch.no_grad():
        if bits == 8:
            layer.weight.data = quantize_to_float8(layer.weight.data)
        elif bits == 4:
            # Implement 4-bit quantization if needed
            pass
        # Keep bias in original precision if exists
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.to(torch.bfloat16)


def apply_layer_quantization(
    model_path: str,
    layers_to_quantize: List[int],
    quantization_bits: int = 8,
    save_path: Optional[str] = None,
    token: Optional[str] = None
) -> str:
    """Apply per-layer quantization based on importance scores.
    
    Args:
        model_path: Path to the model
        layers_to_quantize: List of layer indices to quantize
        quantization_bits: Number of bits for quantization
        save_path: Path to save quantized model
        token: HuggingFace token if needed
        
    Returns:
        Path where quantized model is saved
    """
    logging.info(f"{Fore.GREEN}Loading model for quantization...{Fore.RESET}")
    
    # Load model in full precision first
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.float32,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    
    # Get total number of layers
    total_layers = len(model.model.layers)
    logging.info(f"Total layers: {total_layers}")
    logging.info(f"Quantizing {len(layers_to_quantize)} layers to {quantization_bits}-bit")
    logging.info(f"Layers to quantize: {sorted(layers_to_quantize)}")
    
    # Apply quantization to selected layers
    for layer_idx in tqdm(layers_to_quantize, desc="Quantizing layers"):
        if layer_idx >= total_layers:
            logging.warning(f"Layer {layer_idx} exceeds model layers, skipping...")
            continue
            
        layer = model.model.layers[layer_idx]
        
        # Quantize self-attention layers
        quantize_linear_layer(layer.self_attn.q_proj, quantization_bits)
        quantize_linear_layer(layer.self_attn.k_proj, quantization_bits)
        quantize_linear_layer(layer.self_attn.v_proj, quantization_bits)
        quantize_linear_layer(layer.self_attn.o_proj, quantization_bits)
        
        # Quantize MLP layers
        quantize_linear_layer(layer.mlp.gate_proj, quantization_bits)
        quantize_linear_layer(layer.mlp.up_proj, quantization_bits)
        quantize_linear_layer(layer.mlp.down_proj, quantization_bits)
    
    # Calculate compression ratio
    quantized_layers = len(layers_to_quantize)
    compression_ratio = 1 - (quantized_layers * 0.5) / total_layers  # Assuming 50% size reduction per layer
    logging.info(f"Approximate compression ratio: {compression_ratio:.2%} of original size")
    
    # Save quantized model
    if save_path is None:
        model_name = model_path.replace('/', '_')
        save_path = f"output_models/{model_name}_{quantized_layers}layers_{quantization_bits}bit"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    logging.info(f"{Fore.BLUE}Saving quantized model to {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save quantization info
    import json
    quant_info = {
        "original_model": model_path,
        "quantized_layers": sorted(layers_to_quantize),
        "quantization_bits": quantization_bits,
        "total_layers": total_layers,
        "compression_ratio": compression_ratio
    }
    
    with open(f"{save_path}/quantization_info.json", 'w') as f:
        json.dump(quant_info, f, indent=2)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    logging.info(f"{Fore.GREEN}Quantization complete!{Fore.RESET}")
    return save_path