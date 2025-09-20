"""Adaptive Block Routing (ABR) module for dynamic transformer block skipping.

This module implements a novel approach to dynamically route tokens through
transformer blocks based on their complexity, enabling significant FLOPs reduction
without healing/fine-tuning.
"""

import argparse
import gc
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, seed_all

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class TokenComplexityEstimator:
    """Estimates token complexity based on attention patterns and hidden state dynamics."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.complexity_thresholds = {
            'simple': 0.3,
            'medium': 0.6,
            'complex': 1.0
        }
        print(f"{Fore.GREEN}Initialized TokenComplexityEstimator with hidden_dim={hidden_dim}, num_heads={num_heads}{Fore.RESET}")
    
    def estimate_complexity(self, 
                           hidden_states: torch.Tensor,
                           attention_weights: Optional[torch.Tensor] = None,
                           layer_idx: int = 0) -> torch.Tensor:
        """
        Estimate complexity score for each token in the sequence.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_weights: Optional attention weights from previous layer
            layer_idx: Current layer index
            
        Returns:
            complexity_scores: [batch_size, seq_len] with scores in [0, 1]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Compute entropy of hidden states (normalized)
        hidden_probs = F.softmax(hidden_states, dim=-1)
        entropy = -torch.sum(hidden_probs * torch.log(hidden_probs + 1e-10), dim=-1)
        entropy_normalized = entropy / np.log(self.hidden_dim)  # Normalize to [0, 1]
        
        # 2. Compute variance across hidden dimensions
        variance = torch.var(hidden_states, dim=-1)
        variance_normalized = torch.sigmoid(variance)  # Normalize using sigmoid
        
        # 3. If attention weights available, compute attention sparsity
        if attention_weights is not None:
            # attention_weights: [batch_size, num_heads, seq_len, seq_len]
            # Compute average attention entropy across heads
            attn_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
            attn_entropy = torch.mean(attn_entropy, dim=1)  # Average across heads
            attn_sparsity = 1.0 - (attn_entropy / np.log(seq_len))  # High sparsity = low complexity
        else:
            attn_sparsity = torch.zeros_like(entropy_normalized)
        
        # 4. Layer-dependent weighting (deeper layers might need more computation)
        layer_weight = 0.5 + 0.5 * (layer_idx / 32)  # Assuming ~32 layers
        
        # Combine metrics
        complexity_score = (
            0.3 * entropy_normalized + 
            0.3 * variance_normalized + 
            0.2 * (1.0 - attn_sparsity) +
            0.2 * layer_weight
        )
        
        return complexity_score
    
    def classify_tokens(self, complexity_scores: torch.Tensor) -> torch.Tensor:
        """
        Classify tokens into simple/medium/complex categories.
        
        Returns:
            classifications: [batch_size, seq_len] with values 0 (simple), 1 (medium), 2 (complex)
        """
        classifications = torch.zeros_like(complexity_scores, dtype=torch.long)
        classifications[complexity_scores >= self.complexity_thresholds['simple']] = 1
        classifications[complexity_scores >= self.complexity_thresholds['medium']] = 2
        
        return classifications


class AdaptiveRouter:
    """Routes tokens through different computational paths based on complexity."""
    
    def __init__(self, model_config, layer_skip_map: Dict[int, List[int]]):
        """
        Args:
            model_config: Model configuration object
            layer_skip_map: Dictionary mapping layer indices to skip patterns
                            e.g., {9: [10, 11], 15: [16, 17, 18]} means at layer 9, 
                            we can skip to 12 for simple tokens
        """
        self.num_layers = model_config.num_hidden_layers
        self.hidden_dim = model_config.hidden_size
        self.layer_skip_map = layer_skip_map
        
        # Pre-computed linear approximations for each skippable block
        self.linear_approximations = {}
        
        print(f"{Fore.CYAN}AdaptiveRouter initialized with skip map: {layer_skip_map}{Fore.RESET}")
    
    def compute_linear_approximations(self, 
                                     model: nn.Module,
                                     calibration_data: torch.Tensor) -> None:
        """
        Pre-compute linear approximations for skippable layer blocks.
        
        Args:
            model: The transformer model
            calibration_data: Sample data for computing approximations
        """
        print(f"{Fore.YELLOW}Computing linear approximations for skippable blocks...{Fore.RESET}")
        
        for start_layer, skip_layers in tqdm(self.layer_skip_map.items(), 
                                            desc="Computing approximations"):
            end_layer = skip_layers[-1] + 1
            
            # Get input/output pairs for this block
            with torch.no_grad():
                # Run through layers before the block
                hidden = calibration_data
                for i in range(start_layer):
                    hidden = model.model.layers[i](hidden)[0]
                
                input_hidden = hidden.clone()
                
                # Run through the block
                for i in range(start_layer, end_layer):
                    hidden = model.model.layers[i](hidden)[0]
                
                output_hidden = hidden
            
            # Compute least squares solution: W = (X^T X)^-1 X^T Y
            X = input_hidden.view(-1, self.hidden_dim).double()
            Y = output_hidden.view(-1, self.hidden_dim).double()
            
            # Add small regularization for stability
            XtX = X.T @ X + 1e-6 * torch.eye(self.hidden_dim, device=X.device, dtype=torch.float64)
            XtY = X.T @ Y
            
            # Solve for transformation matrix
            W = torch.linalg.solve(XtX, XtY)
            
            self.linear_approximations[start_layer] = W.float()
            
            # Compute approximation error for debugging
            approx_output = X @ W
            error = torch.mean((approx_output - Y) ** 2).item()
            print(f"  Layer {start_layer}-{end_layer}: Approximation MSE = {error:.6f}")
    
    def route_forward(self,
                     hidden_states: torch.Tensor,
                     layer_idx: int,
                     token_complexity: torch.Tensor,
                     model_layers: nn.ModuleList) -> Tuple[torch.Tensor, int]:
        """
        Route tokens through appropriate computational path.
        
        Args:
            hidden_states: Current hidden states
            layer_idx: Current layer index
            token_complexity: Complexity classification for each token
            model_layers: Model's transformer layers
            
        Returns:
            output_states: Processed hidden states
            next_layer_idx: Next layer to process
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Check if current layer has skip option
        if layer_idx not in self.layer_skip_map:
            # No skip available, use full computation
            output = model_layers[layer_idx](hidden_states)[0]
            return output, layer_idx + 1
        
        # Get skip pattern for this layer
        skip_layers = self.layer_skip_map[layer_idx]
        end_layer = skip_layers[-1] + 1
        
        # Separate tokens by complexity
        simple_mask = (token_complexity == 0)
        medium_mask = (token_complexity == 1)
        complex_mask = (token_complexity == 2)
        
        # Count tokens in each category for debugging
        num_simple = simple_mask.sum().item()
        num_medium = medium_mask.sum().item()
        num_complex = complex_mask.sum().item()
        
        if num_simple > 0 or num_medium > 0:
            print(f"  Layer {layer_idx}: Routing {num_simple} simple, {num_medium} medium, {num_complex} complex tokens")
        
        output_states = torch.zeros_like(hidden_states)
        
        # Route simple tokens: Use linear approximation
        if num_simple > 0:
            W = self.linear_approximations[layer_idx]
            simple_hidden = hidden_states[simple_mask]
            simple_output = simple_hidden @ W.T
            output_states[simple_mask] = simple_output
        
        # Route medium tokens: Use selective attention (skip FFN)
        if num_medium > 0:
            medium_hidden = hidden_states[medium_mask]
            # Only compute attention, skip FFN for medium complexity
            for i in range(layer_idx, min(layer_idx + 2, end_layer)):
                # Simplified computation - only attention
                medium_hidden = model_layers[i].self_attn(medium_hidden)[0] + medium_hidden
                medium_hidden = model_layers[i].post_attention_layernorm(medium_hidden)
            output_states[medium_mask] = medium_hidden
        
        # Route complex tokens: Full computation
        if num_complex > 0:
            complex_hidden = hidden_states[complex_mask]
            for i in range(layer_idx, end_layer):
                complex_hidden = model_layers[i](complex_hidden)[0]
            output_states[complex_mask] = complex_hidden
        
        return output_states, end_layer


def analyze_layer_patterns(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    dataset_size: int,
    use_4bit: bool = False,
    token: Optional[str] = None
) -> Dict[int, float]:
    """
    Analyze attention patterns and hidden state dynamics across layers.
    
    Returns:
        pattern_similarity: Dictionary mapping layer pairs to similarity scores
    """
    print(f"{Fore.CYAN}Analyzing layer patterns for {model_path}{Fore.RESET}")
    
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
        output_attentions=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    dataloader = get_calib_dataloader(
        dataset, "train", dataset_column, 
        min(dataset_size, 100), batch_size, tokenizer  # Use small sample for analysis
    )
    
    layer_patterns = {i: [] for i in range(model.config.num_hidden_layers)}
    
    print(f"{Fore.YELLOW}Collecting attention patterns...{Fore.RESET}")
    for batch in tqdm(dataloader, desc="Analyzing patterns"):
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
        
        # Analyze attention patterns
        attentions = outputs.attentions  # List of attention weights per layer
        
        for layer_idx, attn in enumerate(attentions):
            # Compute attention pattern statistics
            # attn shape: [batch_size, num_heads, seq_len, seq_len]
            attn_entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
            avg_entropy = attn_entropy.mean().item()
            layer_patterns[layer_idx].append(avg_entropy)
    
    # Compute pattern similarity between consecutive layer blocks
    pattern_similarity = {}
    block_size = 4  # Consider blocks of 4 layers
    
    for i in range(0, model.config.num_hidden_layers - block_size):
        block1_pattern = np.mean(layer_patterns[i:i+block_size])
        block2_pattern = np.mean(layer_patterns[i+block_size:i+2*block_size])
        
        # Simple similarity metric (could be more sophisticated)
        similarity = 1.0 - abs(block1_pattern - block2_pattern) / max(block1_pattern, block2_pattern)
        pattern_similarity[(i, i+block_size)] = similarity
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return pattern_similarity


def adaptive_routing(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: int,
    dataset_subset: str = "train",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    skip_threshold: float = 0.7,
    **kwargs
) -> str:
    """
    Main function for Adaptive Block Routing.
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset name for calibration
        dataset_column: Text column in dataset
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to potentially skip
        dataset_size: Size of calibration dataset
        dataset_subset: Dataset split to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save modified model
        token: HuggingFace token
        skip_threshold: Similarity threshold for determining skippable blocks
        
    Returns:
        save_path: Path where the modified model is saved
    """
    print(f"{Fore.GREEN}Starting Adaptive Block Routing for {model_path}{Fore.RESET}")
    
    # Step 1: Analyze layer patterns
    pattern_similarity = analyze_layer_patterns(
        model_path, dataset, dataset_column, batch_size,
        max_length, dataset_size, use_4bit, token
    )
    
    # Step 2: Identify skippable blocks based on pattern similarity
    layer_skip_map = {}
    for (start, end), similarity in pattern_similarity.items():
        if similarity > skip_threshold:
            # This block can be skipped for simple tokens
            layer_skip_map[start] = list(range(start + 1, end))
            print(f"  Layers {start}-{end} are skippable (similarity: {similarity:.3f})")
    
    if not layer_skip_map:
        print(f"{Fore.YELLOW}No highly similar blocks found. Adjusting threshold...{Fore.RESET}")
        # Find top 3 most similar blocks
        sorted_blocks = sorted(pattern_similarity.items(), key=lambda x: x[1], reverse=True)[:3]
        for (start, end), similarity in sorted_blocks:
            layer_skip_map[start] = list(range(start + 1, end))
            print(f"  Selected layers {start}-{end} for routing (similarity: {similarity:.3f})")
    
    # Step 3: Load model and setup routing
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        output_attentions=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize components
    estimator = TokenComplexityEstimator(
        model.config.hidden_size,
        model.config.num_attention_heads
    )
    
    router = AdaptiveRouter(model.config, layer_skip_map)
    
    # Step 4: Compute linear approximations using calibration data
    model.eval()
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        min(dataset_size, 500), batch_size, tokenizer
    )
    
    print(f"{Fore.YELLOW}Computing linear approximations...{Fore.RESET}")
    calibration_samples = []
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Use first 10 batches for calibration
            break
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        input_ids = inputs["input_ids"].to(model.device)
        
        # Get embeddings
        if hasattr(model, 'model'):
            embeddings = model.model.embed_tokens(input_ids)
        else:
            embeddings = model.transformer.wte(input_ids)
        
        calibration_samples.append(embeddings)
    
    calibration_data = torch.cat(calibration_samples, dim=0)
    router.compute_linear_approximations(model, calibration_data)
    
    # Step 5: Create modified forward function
    print(f"{Fore.CYAN}Creating adaptive forward pass...{Fore.RESET}")
    
    original_forward = model.forward
    
    def adaptive_forward(input_ids, attention_mask=None, **kwargs):
        # Get embeddings
        if hasattr(model, 'model'):
            hidden_states = model.model.embed_tokens(input_ids)
            layers = model.model.layers
        else:
            hidden_states = model.transformer.wte(input_ids)
            layers = model.transformer.h
        
        batch_size, seq_len = input_ids.shape
        
        # Process through layers with adaptive routing
        layer_idx = 0
        total_skipped = 0
        
        while layer_idx < len(layers):
            # Estimate token complexity
            with torch.no_grad():
                complexity_scores = estimator.estimate_complexity(
                    hidden_states, None, layer_idx
                )
                token_complexity = estimator.classify_tokens(complexity_scores)
            
            # Route through appropriate path
            hidden_states, next_layer = router.route_forward(
                hidden_states, layer_idx, token_complexity, layers
            )
            
            skipped = next_layer - layer_idx - 1
            if skipped > 0:
                total_skipped += skipped
                print(f"  Skipped {skipped} layers at position {layer_idx}")
            
            layer_idx = next_layer
        
        print(f"{Fore.GREEN}Total layers skipped: {total_skipped}/{len(layers)} ({total_skipped/len(layers)*100:.1f}%){Fore.RESET}")
        
        # Final layer norm and LM head
        if hasattr(model, 'model'):
            hidden_states = model.model.norm(hidden_states)
            logits = model.lm_head(hidden_states)
        else:
            hidden_states = model.transformer.ln_f(hidden_states)
            logits = model.lm_head(hidden_states)
        
        return type('', (), {'logits': logits})()
    
    # Replace forward method
    model.forward = adaptive_forward
    
    # Step 6: Save the modified model configuration
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_ABR_{len(layer_skip_map)}_blocks"
    
    # Save routing configuration
    import json
    routing_config = {
        'layer_skip_map': layer_skip_map,
        'linear_approximations': {k: v.cpu().numpy().tolist() 
                                 for k, v in router.linear_approximations.items()},
        'skip_threshold': skip_threshold,
        'pattern_similarity': {f"{k[0]}-{k[1]}": v for k, v in pattern_similarity.items()}
    }
    
    with open(f"{save_path}_routing_config.json", 'w') as f:
        json.dump(routing_config, f, indent=2)
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"{Fore.GREEN}Adaptive routing model saved to {save_path}{Fore.RESET}")
    print(f"{Fore.CYAN}Routing configuration saved to {save_path}_routing_config.json{Fore.RESET}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path