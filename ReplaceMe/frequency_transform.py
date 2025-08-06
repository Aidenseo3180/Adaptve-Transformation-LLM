import argparse
import gc
import logging
import os
from typing import Optional
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class FrequencyDomainTransform:
    """Frequency Domain Transform for efficient linear approximation"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        
    def compute_frequency_transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute transform in frequency domain using chunked processing for memory efficiency
        
        Args:
            X: Input activations [samples, hidden_size]
            Y: Target activations [samples, hidden_size]
            
        Returns:
            T_spatial: Transform matrix [hidden_size, hidden_size] in spatial domain
        """
        
        num_samples, hidden_size = X.shape
        print(f"Computing Chunked Frequency Domain Transform for {num_samples} samples")
        
        # Calculate optimal chunk size based on available memory
        chunk_size = self._calculate_optimal_chunk_size(num_samples, hidden_size)
        total_chunks = (num_samples + chunk_size - 1) // chunk_size
        
        print(f"Processing in {total_chunks} chunks of size {chunk_size}")
        
        # Initialize accumulators for frequency-wise statistics
        gram_accumulator = torch.zeros(hidden_size, dtype=torch.complex64, device='cpu')
        cross_accumulator = torch.zeros(hidden_size, dtype=torch.complex64, device='cpu')
        
        # Process in chunks to avoid memory overflow
        for chunk_idx in tqdm(range(total_chunks), desc="Chunked Frequency Transform"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            
            # Extract chunk (keep small!)
            X_chunk = X[start_idx:end_idx].clone()  # [chunk_size, hidden_size]
            Y_chunk = Y[start_idx:end_idx].clone()
            
            # Process this chunk
            gram_chunk, cross_chunk = self._process_frequency_chunk(X_chunk, Y_chunk)
            
            # Accumulate statistics
            gram_accumulator += gram_chunk.cpu()
            cross_accumulator += cross_chunk.cpu()
            
            # Aggressive cleanup
            del X_chunk, Y_chunk, gram_chunk, cross_chunk
            
            # Memory cleanup every few chunks
            if chunk_idx % 5 == 0:
                torch.cuda.empty_cache()
                self._log_memory_usage(f"Chunk {chunk_idx + 1}/{total_chunks}")
        
        print(f"Completed chunked processing of {total_chunks} chunks")
        
        # Solve frequency-wise transforms from accumulated statistics
        T_freq = self._solve_from_accumulated_stats(gram_accumulator, cross_accumulator)
        
        # Convert back to spatial domain
        T_spatial = self._frequency_to_spatial_transform(T_freq)
        
        # Validation with a small subset (to avoid memory issues)
        self._validate_transform_chunked(X, Y, T_spatial, chunk_size=min(10000, num_samples))
        
        return T_spatial.to(X.dtype)
    
    def _calculate_optimal_chunk_size(self, num_samples: int, hidden_size: int) -> int:
        """Calculate optimal chunk size based on available memory"""
        
        # Estimate memory per sample in MB
        # Each sample: hidden_size * (2 complex64 for FFT) * 8 bytes ≈ hidden_size * 16 bytes
        memory_per_sample_mb = hidden_size * 16 / (1024 * 1024)
        
        # Target: Use max 2GB for chunk processing
        target_memory_gb = 2.0
        target_memory_mb = target_memory_gb * 1024
        
        optimal_chunk_size = int(target_memory_mb / memory_per_sample_mb)
        
        # Reasonable bounds
        min_chunk_size = 1000
        max_chunk_size = min(100000, num_samples)
        
        chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
        
        print(f"Calculated chunk size: {chunk_size} (~{chunk_size * memory_per_sample_mb:.1f}MB per chunk)")
        
        return chunk_size
    
    def _process_frequency_chunk(self, X_chunk: torch.Tensor, Y_chunk: torch.Tensor) -> tuple:
        """Process a single chunk in frequency domain"""
        
        # Convert to float32 for FFT (good balance of precision and memory)
        X_f32 = X_chunk.to(torch.float32)
        Y_f32 = Y_chunk.to(torch.float32)
        
        # FFT transform
        X_freq = torch.fft.fft(X_f32, dim=-1)  # [chunk_size, hidden_size] complex64
        Y_freq = torch.fft.fft(Y_f32, dim=-1)
        
        # Clean up float32 tensors immediately
        del X_f32, Y_f32
        
        chunk_size, hidden_size = X_freq.shape
        
        # Compute gram and cross-correlation for this chunk
        gram_chunk = torch.zeros(hidden_size, dtype=torch.complex64, device=X_freq.device)
        cross_chunk = torch.zeros(hidden_size, dtype=torch.complex64, device=X_freq.device)
        
        # Frequency-wise computation (vectorized for efficiency)
        for k in range(hidden_size):
            x_k = X_freq[:, k]  # [chunk_size] complex
            y_k = Y_freq[:, k]  # [chunk_size] complex
            
            # Accumulate: gram_k += x_k^H @ x_k, cross_k += x_k^H @ y_k
            gram_chunk[k] = torch.conj(x_k) @ x_k
            cross_chunk[k] = torch.conj(x_k) @ y_k
        
        # Clean up frequency domain tensors
        del X_freq, Y_freq
        
        return gram_chunk, cross_chunk
    
    def _solve_from_accumulated_stats(self, gram_accum: torch.Tensor, cross_accum: torch.Tensor) -> torch.Tensor:
        """Solve frequency transforms from accumulated statistics"""
        
        hidden_size = len(gram_accum)
        T_freq = torch.zeros(hidden_size, dtype=torch.complex64)
        
        print(f"Solving frequency-wise transforms from accumulated statistics")
        
        # Solve for each frequency independently
        for k in range(hidden_size):
            gram_k = gram_accum[k]
            cross_k = cross_accum[k]
            
            # Regularization for numerical stability
            gram_magnitude = torch.abs(gram_k)
            reg_strength = 1e-8 * gram_magnitude
            gram_reg = gram_k + reg_strength
            
            # Solve: T_k = cross_k / gram_k
            if torch.abs(gram_reg) > 1e-12:
                T_freq[k] = cross_k / gram_reg
            else:
                T_freq[k] = torch.complex(torch.tensor(0.0), torch.tensor(0.0))
                if k < 10:  # Only log first few warnings
                    print(f"Warning: Frequency bin {k} has near-zero gram matrix")
        
        # Log frequency domain statistics
        magnitude_stats = torch.abs(T_freq)
        print(f"Frequency transform stats:")
        print(f"   Mean magnitude: {magnitude_stats.mean():.4f}")
        print(f"   Std magnitude: {magnitude_stats.std():.4f}")
        print(f"   Max magnitude: {magnitude_stats.max():.4f}")
        print(f"   Num zeros: {(magnitude_stats < 1e-10).sum()}/{len(magnitude_stats)}")
        
        return T_freq
    
    def _validate_transform_chunked(self, X: torch.Tensor, Y: torch.Tensor, 
                                  T_spatial: torch.Tensor, chunk_size: int = 10000):
        """Validate transform using a subset to avoid memory issues"""
        
        # Use only a subset for validation
        subset_size = min(chunk_size, len(X))
        indices = torch.randperm(len(X))[:subset_size]
        
        X_subset = X[indices]
        Y_subset = Y[indices]
        
        # Test reconstruction
        Y_pred = X_subset @ T_spatial.t()
        
        # Compute errors
        mse_error = torch.mean((Y_pred - Y_subset) ** 2).item()
        relative_error = (torch.norm(Y_pred - Y_subset) / torch.norm(Y_subset)).item()
        
        # Compute correlation
        Y_flat = Y_subset.flatten()
        Y_pred_flat = Y_pred.flatten()
        
        if len(Y_flat) > 1:
            correlation = torch.corrcoef(torch.stack([Y_flat, Y_pred_flat]))[0, 1].item()
        else:
            correlation = 1.0
        
        print(f"Transform validation (subset of {subset_size} samples):")
        print(f"   MSE Error: {mse_error:.6f}")
        print(f"   Relative Error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        print(f"   Correlation: {correlation:.4f}")
        
        # Quality assessment
        if relative_error < 0.1:
            print("Excellent transform quality!")
        elif relative_error < 0.2:
            print("Good transform quality")
        else:
            print("Warning: Transform quality may be suboptimal")
        
        # Clean up
        del X_subset, Y_subset, Y_pred
    
    def _log_memory_usage(self, stage: str = ""):
        """Log current memory usage"""
        try:
            import psutil
            
            # RAM usage
            ram_info = psutil.virtual_memory()
            ram_used_gb = ram_info.used / (1024**3)
            ram_percent = ram_info.percent
            
            # GPU usage
            gpu_used_gb = 0
            if torch.cuda.is_available():
                gpu_used_gb = torch.cuda.memory_allocated() / (1024**3)
            
            print(f"Memory usage - {stage} - RAM: {ram_used_gb:.1f}GB ({ram_percent:.1f}%), GPU: {gpu_used_gb:.1f}GB")
            
        except Exception as e:
            print(f"Could not log memory usage: {e}")
    
    def _frequency_to_spatial_transform(self, T_freq: torch.Tensor) -> torch.Tensor:
        """
        Convert frequency domain transform to spatial domain transform matrix
        
        The key insight: A frequency domain diagonal transform corresponds to 
        a circulant matrix in spatial domain.
        
        Args:
            T_freq: [hidden_size] complex tensor
            
        Returns:
            T_spatial: [hidden_size, hidden_size] real tensor
        """
        
        logging.info(f"Converting frequency transform to spatial domain")
        
        # Method: Use the fact that multiplication in frequency domain 
        # corresponds to circular convolution in spatial domain
        
        # 1. Inverse FFT to get impulse response
        impulse_response = torch.fft.ifft(T_freq).real  # [hidden_size] real
        
        # 2. Construct circulant matrix from impulse response
        hidden_size = len(T_freq)
        T_spatial = torch.zeros(hidden_size, hidden_size, dtype=torch.float64, device=T_freq.device)
        
        for i in range(hidden_size):
            for j in range(hidden_size):
                # Circulant property: T[i,j] = impulse_response[(i-j) % hidden_size]
                shift = (i - j) % hidden_size
                T_spatial[i, j] = impulse_response[shift]
        
        logging.info(f"Spatial transform matrix constructed: {T_spatial.shape}")
        
        # Log spatial domain statistics
        logging.info(f"Spatial transform stats - Mean: {T_spatial.mean():.4f}, "
                    f"Std: {T_spatial.std():.4f}, Frobenius norm: {torch.norm(T_spatial):.4f}")
        
        return T_spatial


def frequency_transform(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    # ReplaceMe compatibility parameters - for cosine distance based block selection
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    # Additional compatibility
    **kwargs
) -> str:
    """
    Frequency Domain Transform method for ReplaceMe
    Uses ReplaceMe's cosine distance block selection + Frequency Domain Transform
    
    Args:
        start_id: Starting layer index (from ReplaceMe block selection)
        end_id: Ending layer index (from ReplaceMe block selection)  
        num_layer: Number of previous layers removed (from ReplaceMe)
        ... (other args same as ReplaceMe)
    
    Returns:
        Path where transformed model is saved
    """
    
    logging.info(f"Starting Frequency Domain Transform")
    logging.info(f"Target block: {start_id}-{end_id} (selected by cosine distance)")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    hidden_size = model.config.hidden_size
    
    # Initialize Frequency Domain Transform
    freq_transform = FrequencyDomainTransform(hidden_size)
    
    # Collect activations using hooks (similar to original ReplaceMe)
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    
    # Register hooks for the specific layers we need
    if 'falcon' in model_path.lower():
        hooks.append(model.transformer.h[start_id].mlp.register_forward_hook(
            save_mlp_activation(f'layer_{start_id}_mlp')))
    else:
        hooks.append(model.model.layers[start_id].mlp.register_forward_hook(
            save_mlp_activation(f'layer_{start_id}_mlp')))
    
    # Load calibration data
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer
    )
    
    # Collect activations
    a1 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    cnt = 0
    logging.info(f"Collecting activations for frequency domain analysis...")
    
    for batch in tqdm(dataloader, desc="Gathering Activations"):
        inputs = tokenizer(
            batch, return_tensors="pt", padding="longest", 
            max_length=max_length, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get hidden states and MLP activations
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
        mlp_output = mlp_activations[f'layer_{start_id}_mlp']
        
        # Get input and target for the block
        hidden_states_i = hidden_states[start_id - num_layer - 1]  # Input to block
        hidden_states_n = hidden_states[end_id - num_layer - 1]    # Output after block
        
        # Reshape activations
        batch_size_actual = mlp_output.shape[0]
        seq_len_actual = mlp_output.shape[1]
        
        mlp_flat = mlp_output.view(-1, hidden_size).cpu().to(torch.bfloat16)
        target_flat = (hidden_states_n - hidden_states_i).view(-1, hidden_size).cpu().to(torch.bfloat16)
        
        # Store activations
        end_idx = cnt + mlp_flat.shape[0]
        a1[cnt:end_idx] = mlp_flat
        a2[cnt:end_idx] = target_flat
        cnt = end_idx
        
        # Clear activations
        mlp_activations.clear()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Trim to actual size
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    
    print(f"Collected {cnt} activation samples")
    
    # Compute Frequency Domain Transform
    print(f"Computing Frequency Domain Transform...")
    transform = freq_transform.compute_frequency_transform(a1.float(), a2.float())
    
    # Clean up activations
    del model, a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    print(f"Reloading model for transformation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='cpu', torch_dtype=torch.bfloat16
    )
    
    # Truncate model (remove the target layers)
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply frequency domain transform to the down_proj layer
    target_layer = model.model.layers[start_id - num_layer].mlp.down_proj
    original_weight = target_layer.weight.to(torch.float64)
    transform_64 = transform.to(torch.float64)
    
    # Apply transform: new_weight = transform @ original_weight
    new_weight = (transform_64 @ original_weight).to(torch.bfloat16)
    target_layer.weight = nn.Parameter(new_weight)
    
    print(f"Applied Frequency Domain Transform to layer {start_id}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    final_path = f"{save_path}_FreqTransform"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    if save_transform_only:
        # Save the frequency domain transform
        torch.save({
            'transform_spatial': transform,
            'method': 'frequency_domain',
            'block_range': (start_id, end_id),
            'hidden_size': hidden_size
        }, f"{final_path}_freq_transform.pth")
    
    print(f"Frequency Domain Transform completed!")
    print(f"Model saved to: {final_path}")
    
    # Final cleanup
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run Frequency Domain Transform from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run Frequency Domain Transform based on a configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, 
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    path = frequency_transform(**config)
    print(f"Frequency Domain Transform completed. Model saved to: {path}")