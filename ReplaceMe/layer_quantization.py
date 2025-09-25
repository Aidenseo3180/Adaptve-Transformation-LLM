import gc
import logging
import os
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, init
import json

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def quantize_to_int8_symmetric(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric INT8 quantization with scale factor."""
    if tensor.numel() == 0:
        return tensor, 1.0
    
    # Calculate scale factor
    max_val = tensor.abs().max()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    
    # Quantize to INT8 range
    quantized = torch.round(tensor / scale).clamp(-128, 127)
    
    # Dequantize for storage (PyTorch operations need float)
    dequantized = quantized * scale
    
    return dequantized.to(tensor.dtype), scale


def quantize_linear_layer(layer: nn.Linear, bits: int = 8) -> Dict:
    """Quantize a linear layer's weights and return statistics."""
    stats = {
        'original_dtype': str(layer.weight.dtype),
        'original_mean': layer.weight.data.mean().item(),
        'original_std': layer.weight.data.std().item(),
        'original_min': layer.weight.data.min().item(),
        'original_max': layer.weight.data.max().item(),
    }
    
    with torch.no_grad():
        if bits == 8:
            # Quantize weight
            quantized_weight, weight_scale = quantize_to_int8_symmetric(layer.weight.data)
            
            # Replace weight with quantized version
            layer.weight.data = quantized_weight
            
            # Store quantization metadata as buffers
            if not hasattr(layer, 'weight_scale'):
                layer.register_buffer('weight_scale', torch.tensor(weight_scale))
                layer.register_buffer('weight_bits', torch.tensor(bits))
            
            # Keep bias at higher precision (bfloat16)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(torch.bfloat16)
            
            stats.update({
                'quantized_mean': layer.weight.data.mean().item(),
                'quantized_std': layer.weight.data.std().item(),
                'quantized_min': layer.weight.data.min().item(),
                'quantized_max': layer.weight.data.max().item(),
                'scale': weight_scale,
                'bits': bits
            })
        else:
            raise NotImplementedError(f"{bits}-bit quantization not implemented")
    
    return stats


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
        quantization_bits: Number of bits for quantization (currently only 8)
        save_path: Path to save quantized model
        token: HuggingFace token if needed
        
    Returns:
        Path where quantized model is saved
    """
    logging.info(f"{Fore.GREEN}Starting layer-wise quantization...{Fore.RESET}")
    logging.info(f"Model: {model_path}")
    logging.info(f"Layers to quantize: {len(layers_to_quantize)} layers")
    logging.info(f"Target bits: {quantization_bits}")
    
    # Load model in full precision
    logging.info("Loading model in full precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.float32,  # Load in FP32 for quantization
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get model info
    total_layers = len(model.model.layers)
    logging.info(f"Model has {total_layers} layers total")
    
    # Validate layer indices
    layers_to_quantize = [l for l in layers_to_quantize if l < total_layers]
    logging.info(f"Will quantize layers: {sorted(layers_to_quantize)}")
    
    # Quantization statistics
    all_stats = {}
    total_original_params = 0
    total_quantized_params = 0
    
    # Apply quantization to selected layers
    for layer_idx in tqdm(layers_to_quantize, desc="Quantizing layers"):
        layer = model.model.layers[layer_idx]
        layer_stats = {}
        
        # Count original parameters
        layer_params = sum(p.numel() for p in layer.parameters())
        total_original_params += layer_params
        
        # Quantize all linear layers in this transformer layer
        linear_modules = [
            ('self_attn.q_proj', layer.self_attn.q_proj),
            ('self_attn.k_proj', layer.self_attn.k_proj),
            ('self_attn.v_proj', layer.self_attn.v_proj),
            ('self_attn.o_proj', layer.self_attn.o_proj),
            ('mlp.gate_proj', layer.mlp.gate_proj),
            ('mlp.up_proj', layer.mlp.up_proj),
            ('mlp.down_proj', layer.mlp.down_proj),
        ]
        
        for module_name, module in linear_modules:
            if isinstance(module, nn.Linear):
                stats = quantize_linear_layer(module, quantization_bits)
                layer_stats[module_name] = stats
        
        # Count quantized parameters (approximation)
        total_quantized_params += layer_params * (quantization_bits / 32.0)
        all_stats[f'layer_{layer_idx}'] = layer_stats
    
    # Calculate compression statistics
    compression_ratio = total_quantized_params / max(total_original_params, 1)
    memory_saved = (1 - compression_ratio) * 100
    
    logging.info(f"{Fore.CYAN}Quantization Statistics:{Fore.RESET}")
    logging.info(f"Layers quantized: {len(layers_to_quantize)}/{total_layers}")
    logging.info(f"Approximate compression: {compression_ratio:.2%}")
    logging.info(f"Memory saved: {memory_saved:.1f}%")
    
    # Verify quantization by checking weight ranges
    sample_layer_idx = layers_to_quantize[0] if layers_to_quantize else 0
    sample_layer = model.model.layers[sample_layer_idx]
    sample_weight = sample_layer.self_attn.q_proj.weight.data
    logging.info(f"Sample quantized weight stats:")
    logging.info(f"  - Mean: {sample_weight.mean().item():.6f}")
    logging.info(f"  - Std: {sample_weight.std().item():.6f}")
    logging.info(f"  - Min: {sample_weight.min().item():.6f}")
    logging.info(f"  - Max: {sample_weight.max().item():.6f}")
    
    # Generate save path if not provided
    if save_path is None:
        model_name = model_path.replace('/', '_')
        save_path = f"output_models/{model_name}_{len(layers_to_quantize)}layers_{quantization_bits}bit"
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and tokenizer
    logging.info(f"{Fore.BLUE}Saving quantized model to {save_path}{Fore.RESET}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save quantization metadata
    metadata = {
        "original_model": model_path,
        "total_layers": total_layers,
        "quantized_layers": sorted(layers_to_quantize),
        "num_quantized_layers": len(layers_to_quantize),
        "quantization_bits": quantization_bits,
        "compression_ratio": compression_ratio,
        "memory_saved_percent": memory_saved,
        "quantization_stats": all_stats
    }
    
    with open(f"{save_path}/quantization_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"{Fore.GREEN}✓ Quantization complete!{Fore.RESET}")
    logging.info(f"{Fore.GREEN}✓ Model saved to: {save_path}{Fore.RESET}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path