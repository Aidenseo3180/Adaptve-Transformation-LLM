"""Dynamic Resolution Transformer (DRT) Module - Fixed Version"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, seed_all)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class TokenMergeStrategy:
    """Manages token merging decisions based on attention patterns."""
    
    def __init__(self, merge_threshold: float = 0.7, min_tokens_ratio: float = 0.25):
        self.merge_threshold = merge_threshold
        self.min_tokens_ratio = min_tokens_ratio
        self.merge_history = {}
        
        print(f"{Fore.GREEN}[DRT] Initialized TokenMergeStrategy:")
        print(f"  - Merge threshold: {merge_threshold}")
        print(f"  - Min tokens ratio: {min_tokens_ratio}{Fore.RESET}")
    
    def compute_token_similarity(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute similarity between tokens using hidden states and optionally attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_weights: Optional [batch, heads, seq_len, seq_len]
            
        Returns:
            Similarity matrix [seq_len, seq_len]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute cosine similarity between all token pairs
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)  # [batch, seq_len, hidden_dim]
        similarity = torch.bmm(hidden_norm, hidden_norm.transpose(1, 2))  # [batch, seq_len, seq_len]
        
        # Average across batch
        similarity = similarity.mean(dim=0)  # [seq_len, seq_len]
        
        # If attention weights provided, combine with similarity
        if attention_weights is not None and attention_weights.numel() > 0:
            # Average attention across heads and batch
            avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
            
            # Combine: 70% similarity, 30% attention
            similarity = 0.7 * similarity + 0.3 * avg_attention
            
            print(f"{Fore.BLUE}[DRT Debug] Using combined similarity (hidden + attention){Fore.RESET}")
        else:
            print(f"{Fore.YELLOW}[DRT Debug] Using only hidden state similarity (no attention){Fore.RESET}")
        
        return similarity
    
    def find_merge_candidates(
        self,
        similarity: torch.Tensor,
        layer_idx: int,
        depth_ratio: float
    ) -> List[Tuple[int, int]]:
        """
        Find token pairs to merge based on similarity.
        
        Args:
            similarity: [seq_len, seq_len] similarity matrix
            layer_idx: Current layer index
            depth_ratio: Depth in network (0=shallow, 1=deep)
            
        Returns:
            List of token pairs to merge
        """
        seq_len = similarity.shape[0]
        
        # Adaptive threshold based on depth
        adaptive_threshold = self.merge_threshold - (0.2 * depth_ratio)  # More aggressive deeper
        
        print(f"{Fore.CYAN}[DRT Debug] Layer {layer_idx}: adaptive_threshold={adaptive_threshold:.3f}{Fore.RESET}")
        
        # Set diagonal to -1 to avoid self-merging
        similarity = similarity.clone()
        similarity.fill_diagonal_(-1)
        
        # Find high similarity pairs
        merge_candidates = []
        used_tokens = set()
        
        # Get top similarity pairs
        values, indices = similarity.flatten().sort(descending=True)
        
        num_considered = 0
        for idx in range(len(values)):
            if values[idx] < adaptive_threshold:
                break
                
            num_considered += 1
            
            # Get token pair
            i = (indices[idx] // seq_len).item()
            j = (indices[idx] % seq_len).item()
            
            # Skip if already used
            if i in used_tokens or j in used_tokens:
                continue
            
            # Skip special positions (keep first/last tokens)
            if i <= 1 or j <= 1 or i >= seq_len-2 or j >= seq_len-2:
                continue
            
            # Add to candidates
            merge_candidates.append((i, j))
            used_tokens.add(i)
            used_tokens.add(j)
            
            # Limit merging based on depth
            max_merge_ratio = 0.3 + 0.3 * depth_ratio  # 30% → 60% as we go deeper
            if len(used_tokens) >= seq_len * max_merge_ratio:
                break
        
        print(f"{Fore.GREEN}[DRT] Layer {layer_idx}: Found {len(merge_candidates)} merge pairs "
              f"(considered {num_considered} pairs, max similarity: {values[0]:.3f}){Fore.RESET}")
        
        return merge_candidates
    
    def create_merge_groups(
        self,
        merge_candidates: List[Tuple[int, int]],
        seq_len: int
    ) -> Dict[int, List[int]]:
        """Create groups of tokens to merge."""
        groups = {}
        token_to_group = {}
        next_group_id = 0
        
        # Build merge groups
        for i, j in merge_candidates:
            if i in token_to_group and j in token_to_group:
                # Merge two groups
                group_i = token_to_group[i]
                group_j = token_to_group[j]
                if group_i != group_j:
                    # Merge smaller into larger
                    if len(groups[group_i]) < len(groups[group_j]):
                        group_i, group_j = group_j, group_i
                    groups[group_i].extend(groups[group_j])
                    for token in groups[group_j]:
                        token_to_group[token] = group_i
                    del groups[group_j]
            elif i in token_to_group:
                group_id = token_to_group[i]
                groups[group_id].append(j)
                token_to_group[j] = group_id
            elif j in token_to_group:
                group_id = token_to_group[j]
                groups[group_id].append(i)
                token_to_group[i] = group_id
            else:
                # Create new group
                groups[next_group_id] = [i, j]
                token_to_group[i] = next_group_id
                token_to_group[j] = next_group_id
                next_group_id += 1
        
        # Add unmerged tokens
        for idx in range(seq_len):
            if idx not in token_to_group:
                groups[next_group_id] = [idx]
                next_group_id += 1
        
        return groups


def drt_transform(
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
    merge_threshold: float = 0.65,  # Lowered default
    min_tokens_ratio: float = 0.25,
    start_merge_layer: int = 10,
    **kwargs
) -> str:
    """Apply Dynamic Resolution Transformer to model."""
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"[DRT] Starting Dynamic Resolution Transformer")
    print(f"{'='*60}{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model with eager attention for proper attention weights
    print(f"{Fore.YELLOW}[DRT] Loading model: {model_path}{Fore.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",  # Force eager attention for attention weights
        token=token,
        torch_dtype=torch.bfloat16 if not use_4bit else None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get calibration data
    print(f"{Fore.YELLOW}[DRT] Loading calibration dataset{Fore.RESET}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Initialize merge strategy
    merge_strategy = TokenMergeStrategy(
        merge_threshold=merge_threshold,
        min_tokens_ratio=min_tokens_ratio
    )
    
    # Collect merge statistics
    layer_merge_stats = {i: [] for i in range(start_merge_layer, model.config.num_hidden_layers)}
    num_layers = model.config.num_hidden_layers
    
    print(f"{Fore.CYAN}[DRT] Analyzing patterns on {min(10, len(dataloader))} batches...{Fore.RESET}")
    
    # Analyze patterns
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calibration", colour="cyan")):
        if batch_idx >= 10:  # Use first 10 batches
            break
        
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
        
        hidden_states = outputs.hidden_states if outputs.hidden_states else []
        attentions = outputs.attentions if outputs.attentions else []
        
        print(f"{Fore.BLUE}[DRT Debug] Batch {batch_idx}: "
              f"hidden_states={len(hidden_states)}, attentions={len(attentions)}{Fore.RESET}")
        
        # Analyze each layer
        for layer_idx in range(start_merge_layer, min(len(hidden_states)-1, num_layers)):
            depth_ratio = (layer_idx - start_merge_layer) / (num_layers - start_merge_layer)
            
            # Get hidden states for this layer
            hidden = hidden_states[layer_idx]
            
            # Get attention if available
            attn = attentions[layer_idx] if layer_idx < len(attentions) else None
            
            # Compute similarity
            similarity = merge_strategy.compute_token_similarity(hidden, attn)
            
            # Find merge candidates
            merge_candidates = merge_strategy.find_merge_candidates(
                similarity, layer_idx, depth_ratio
            )
            
            layer_merge_stats[layer_idx].append(len(merge_candidates))
    
    # Print statistics
    print(f"\n{Fore.GREEN}[DRT] Merge Statistics:{Fore.RESET}")
    total_merges = 0
    for layer_idx in sorted(layer_merge_stats.keys()):
        if layer_merge_stats[layer_idx]:
            avg_merges = sum(layer_merge_stats[layer_idx]) / len(layer_merge_stats[layer_idx])
            total_merges += avg_merges
            print(f"  Layer {layer_idx}: avg {avg_merges:.1f} merge pairs")
    
    if total_merges == 0:
        print(f"{Fore.RED}[DRT WARNING] No merge candidates found! Consider:")
        print(f"  - Lowering merge_threshold (current: {merge_threshold})")
        print(f"  - Using a different dataset")
        print(f"  - Checking if model outputs hidden states properly{Fore.RESET}")
    
    # Create wrapper class for DRT-enabled model
    print(f"\n{Fore.CYAN}[DRT] Creating DRT model wrapper...{Fore.RESET}")
    
    # Save the model with DRT configuration
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        model_name = model_path.split('/')[-1]
        save_path = f"output_models/{model_name}_DRT_thr{merge_threshold}_start{start_merge_layer}"
    
    print(f"{Fore.GREEN}[DRT] Saving model to: {save_path}{Fore.RESET}")
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # DRT 설정 생성
    drt_config = {
        'method': 'drt',
        'merge_threshold': merge_threshold,
        'min_tokens_ratio': min_tokens_ratio,
        'start_merge_layer': start_merge_layer,
        'layer_merge_stats': {k: v for k, v in layer_merge_stats.items() if v},
        'total_avg_merges': total_merges / len([v for v in layer_merge_stats.values() if v]) if total_merges > 0 else 0
    }

    from .drt_fixed import apply_drt_and_save
    
    # 실제로 DRT 적용하여 저장
    drt_model = apply_drt_and_save(
        model_path=model_path,
        save_path=save_path,
        drt_config=drt_config
    )
    
    print(f"{Fore.GREEN}[DRT] Model with active DRT saved to: {save_path}{Fore.RESET}")
    
    # Cleanup
    del drt_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path