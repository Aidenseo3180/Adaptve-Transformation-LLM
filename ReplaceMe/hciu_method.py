"""Hybrid Caching with Incremental Updates (HCIU) - Advanced layer optimization method

Combines intelligent caching with adaptive low-rank updates for optimal performance/efficiency trade-off.
"""

import gc
import logging
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import json
import time

from .utils import (get_calib_dataloader, seed_all)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


@dataclass
class CacheEntry:
    """Single cache entry for pattern matching"""
    key: torch.Tensor  # Compressed representation of input
    output_delta: torch.Tensor  # Output - Input
    frequency: int = 1
    last_access: float = 0.0
    hit_rate: float = 0.0


class PatternCache:
    """Intelligent caching system for transformer layers"""
    
    def __init__(self, 
                 hidden_dim: int,
                 max_entries: int = 1000,
                 key_dim: int = 32,
                 similarity_threshold: float = 0.95,
                 device: str = "cuda"):
        
        self.hidden_dim = hidden_dim
        self.max_entries = max_entries
        self.key_dim = key_dim
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Cache storage
        self.cache = OrderedDict()
        
        # Key projection matrix (for dimensionality reduction)
        self.key_projection = nn.Linear(hidden_dim, key_dim, bias=False).to(device)
        nn.init.orthogonal_(self.key_projection.weight)
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"[HCIU Cache] Initialized with max_entries={max_entries}, key_dim={key_dim}, threshold={similarity_threshold}")
    
    def _compute_key(self, x: torch.Tensor) -> torch.Tensor:
        """Compute compressed key from input"""
        # Average pooling across sequence dimension
        x_pooled = x.mean(dim=1) if x.dim() == 3 else x.mean(dim=0)
        
        # Project to lower dimension
        with torch.no_grad():
            key = self.key_projection(x_pooled)
            key = F.normalize(key, p=2, dim=-1)
        
        return key
    
    def query(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], bool]:
        """Query cache for similar pattern"""
        self.total_queries += 1
        
        # Compute query key
        query_key = self._compute_key(x)
        
        # Search for similar entries
        best_similarity = -1
        best_entry = None
        
        for cache_key_str, entry in self.cache.items():
            similarity = F.cosine_similarity(query_key.flatten(), entry.key.flatten(), dim=0)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold and best_entry is not None:
            self.cache_hits += 1
            best_entry.frequency += 1
            best_entry.last_access = time.time()
            
            # Move to end (LRU)
            self.cache.move_to_end(str(best_entry.key.cpu().numpy().tobytes()))
            
            print(f"[HCIU Cache] Hit! Similarity: {best_similarity:.4f}, Frequency: {best_entry.frequency}")
            return best_entry.output_delta, True
        
        self.cache_misses += 1
        return None, False
    
    def update(self, x: torch.Tensor, output: torch.Tensor):
        """Add new pattern to cache"""
        key = self._compute_key(x)
        delta = output - x
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            output_delta=delta.detach(),
            last_access=time.time()
        )
        
        # Add to cache
        key_str = str(key.cpu().numpy().tobytes())
        
        if len(self.cache) >= self.max_entries:
            # Evict least recently used
            self.cache.popitem(last=False)
            print(f"[HCIU Cache] Evicted LRU entry. Cache size: {len(self.cache)}")
        
        self.cache[key_str] = entry
    
    def get_statistics(self) -> Dict[str, float]:
        """Return cache statistics"""
        hit_rate = self.cache_hits / max(self.total_queries, 1)
        return {
            'hit_rate': hit_rate,
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.cache)
        }


class AdaptiveLowRankUpdate(nn.Module):
    """Adaptive low-rank update module"""
    
    def __init__(self,
                 hidden_dim: int,
                 min_rank: int = 4,
                 max_rank: int = 128,
                 num_rank_options: int = 4,
                 device: str = "cuda"):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.device = device
        
        # Pre-define rank options
        self.rank_options = np.logspace(
            np.log10(min_rank),
            np.log10(max_rank),
            num_rank_options,
            dtype=int
        )
        
        # Create update matrices for each rank
        self.update_matrices = nn.ModuleDict()
        for rank in self.rank_options:
            self.update_matrices[str(rank)] = nn.ModuleDict({
                'U': nn.Linear(hidden_dim, rank, bias=False),
                'V': nn.Linear(rank, hidden_dim, bias=False)
            })
            
            # Initialize with small values
            nn.init.normal_(self.update_matrices[str(rank)]['U'].weight, std=0.02)
            nn.init.normal_(self.update_matrices[str(rank)]['V'].weight, std=0.02)
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_rank_options)
        )
        
        print(f"[HCIU LowRank] Initialized with ranks: {self.rank_options}")
    
    def estimate_complexity(self, x: torch.Tensor) -> int:
        """Estimate required rank based on input complexity"""
        with torch.no_grad():
            # Pool across sequence
            x_pooled = x.mean(dim=1) if x.dim() == 3 else x.mean(dim=0)
            
            # Get complexity scores
            scores = self.complexity_estimator(x_pooled)
            rank_idx = torch.argmax(scores).item()
            
            selected_rank = self.rank_options[rank_idx]
            
        return selected_rank
    
    def forward(self, x: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
        """Compute low-rank update"""
        if rank is None:
            rank = self.estimate_complexity(x)
        
        # Ensure rank is valid
        if rank not in self.rank_options:
            rank = min(self.rank_options, key=lambda r: abs(r - rank))
        
        # Apply low-rank update
        U = self.update_matrices[str(rank)]['U']
        V = self.update_matrices[str(rank)]['V']
        
        # x -> U -> V -> output
        update = V(U(x))
        
        return update


class TokenImportanceScorer(nn.Module):
    """Score token importance for routing decisions"""
    
    def __init__(self, hidden_dim: int, device: str = "cuda"):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Importance scoring network
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Position embedding for importance
        self.position_importance = nn.Parameter(torch.ones(1024))  # Max seq length
        
        print(f"[HCIU Scorer] Initialized token importance scorer")
    
    def forward(self, x: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score token importance
        
        Args:
            x: Hidden states [batch, seq, hidden]
            attention_weights: Optional attention weights for additional signal
        
        Returns:
            importance scores [batch, seq]
        """
        batch_size, seq_len, _ = x.shape
        
        # Content-based importance
        content_scores = self.scorer(x).squeeze(-1)  # [batch, seq]
        
        # Position-based importance
        position_scores = self.position_importance[:seq_len].unsqueeze(0).expand(batch_size, -1)
        
        # Combine scores
        importance = torch.sigmoid(content_scores + position_scores * 0.1)
        
        # Boost first and last tokens
        importance[:, 0] *= 2.0
        importance[:, -1] *= 2.0
        
        # Use attention weights if available
        if attention_weights is not None:
            # Average attention received by each token
            attn_importance = attention_weights.mean(dim=1).mean(dim=1)  # [batch, seq]
            importance = importance * 0.7 + attn_importance * 0.3
        
        return importance


class HCIULayer(nn.Module):
    """HCIU-enhanced transformer layer"""
    
    def __init__(self,
                 original_layer: nn.Module,
                 layer_idx: int,
                 hidden_dim: int,
                 cache_config: Dict[str, Any],
                 update_config: Dict[str, Any],
                 device: str = "cuda"):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Keep original components
        self.original_layer = original_layer
        
        # Initialize cache
        self.cache = PatternCache(
            hidden_dim=hidden_dim,
            max_entries=cache_config.get('max_entries', 1000),
            key_dim=cache_config.get('key_dim', 32),
            similarity_threshold=cache_config.get('similarity_threshold', 0.95),
            device=device
        )
        
        # Initialize low-rank updater
        self.low_rank_update = AdaptiveLowRankUpdate(
            hidden_dim=hidden_dim,
            min_rank=update_config.get('min_rank', 4),
            max_rank=update_config.get('max_rank', 128),
            device=device
        )
        
        # Initialize importance scorer
        self.importance_scorer = TokenImportanceScorer(hidden_dim, device)
        
        # Thresholds
        self.critical_threshold = 0.8
        self.simple_threshold = 0.3
        
        # Statistics
        self.stats = defaultdict(int)
        
        print(f"[HCIU Layer {layer_idx}] Initialized with cache and adaptive updates")
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Score token importance
        importance_scores = self.importance_scorer(hidden_states)
        
        # Classify tokens
        critical_mask = importance_scores > self.critical_threshold
        simple_mask = importance_scores < self.simple_threshold
        normal_mask = ~(critical_mask | simple_mask)
        
        output = torch.zeros_like(hidden_states)
        
        # Process critical tokens with full computation
        if critical_mask.any():
            self.stats['critical_tokens'] += critical_mask.sum().item()
            critical_indices = critical_mask.nonzero(as_tuple=True)
            
            # Full computation for critical tokens
            critical_input = hidden_states.clone()
            critical_input[~critical_mask] = 0
            
            critical_output = self.original_layer(
                critical_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )[0]
            
            output[critical_mask] = critical_output[critical_mask]
        
        # Process simple tokens with cache
        if simple_mask.any():
            self.stats['simple_tokens'] += simple_mask.sum().item()
            
            # Try cache first
            cached_delta, cache_hit = self.cache.query(hidden_states)
            
            if cache_hit and cached_delta is not None:
                self.stats['cache_hits'] += 1
                # Apply cached transformation
                output[simple_mask] = (hidden_states + cached_delta)[simple_mask]
            else:
                self.stats['cache_misses'] += 1
                # Compute with minimal rank
                simple_update = self.low_rank_update(hidden_states, rank=4)
                output[simple_mask] = (hidden_states + simple_update)[simple_mask]
                
                # Update cache
                self.cache.update(hidden_states, hidden_states + simple_update)
        
        # Process normal tokens with adaptive low-rank
        if normal_mask.any():
            self.stats['normal_tokens'] += normal_mask.sum().item()
            
            # Estimate complexity and apply appropriate rank
            complexity_rank = self.low_rank_update.estimate_complexity(hidden_states)
            normal_update = self.low_rank_update(hidden_states, rank=complexity_rank)
            
            output[normal_mask] = (hidden_states + normal_update)[normal_mask]
            
            # Selectively cache if update is small
            update_magnitude = normal_update.norm(dim=-1).mean()
            if update_magnitude < 0.1:
                self.cache.update(hidden_states, hidden_states + normal_update)
        
        # Return in original format
        outputs = (output,)
        if output_attentions:
            outputs += (None,)  # Placeholder for attention weights
        if use_cache:
            outputs += (None,)  # Placeholder for cache
        
        return outputs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get layer statistics"""
        stats = dict(self.stats)
        stats.update(self.cache.get_statistics())
        stats['low_rank_usage'] = {
            str(rank): self.stats.get(f'rank_{rank}_usage', 0)
            for rank in self.low_rank_update.rank_options
        }
        return stats


def calibrate_hciu_thresholds(
    model: nn.Module,
    tokenizer: Any,
    dataloader: List[str],
    target_layers: List[int],
    device: str,
    max_samples: int = 100
) -> Dict[str, float]:
    """Calibrate HCIU thresholds using calibration data"""
    
    print(f"\n{Fore.YELLOW}[HCIU] Calibrating thresholds...{Fore.RESET}")
    
    all_importance_scores = []
    all_complexities = []
    all_cache_similarities = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_text in enumerate(tqdm(dataloader, desc="Calibration")):
            if batch_idx >= max_samples:
                break
            
            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            for layer_idx in target_layers:
                if layer_idx < len(hidden_states):
                    h = hidden_states[layer_idx]
                    
                    # Measure importance variance
                    importance_var = h.std(dim=-1).mean().item()
                    all_importance_scores.append(importance_var)
                    
                    # Measure complexity (entropy)
                    h_normalized = F.softmax(h.abs().mean(dim=-1), dim=-1)
                    entropy = -(h_normalized * h_normalized.log()).sum(dim=-1).mean().item()
                    all_complexities.append(entropy)
    
    # Calculate percentile-based thresholds
    thresholds = {
        'critical_threshold': np.percentile(all_importance_scores, 90),
        'simple_threshold': np.percentile(all_importance_scores, 30),
        'cache_similarity_threshold': 0.95,  # Can be adjusted based on cache hit rate
        'complexity_thresholds': {
            'low': np.percentile(all_complexities, 25),
            'medium': np.percentile(all_complexities, 50),
            'high': np.percentile(all_complexities, 75)
        }
    }
    
    print(f"[HCIU] Calibrated thresholds:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    
    return thresholds


def apply_hciu_transformation(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "train",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    # HCIU specific parameters
    cache_max_entries: int = 1000,
    cache_key_dim: int = 32,
    cache_similarity_threshold: float = 0.95,
    update_min_rank: int = 4,
    update_max_rank: int = 128,
    calibration_samples: int = 100,
    warmup_samples: int = 50,
    **kwargs
) -> str:
    """Apply Hybrid Caching with Incremental Updates transformation"""
    
    print(f"\n{Fore.MAGENTA}{'='*60}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Starting Hybrid Caching + Incremental Updates (HCIU){Fore.RESET}")
    print(f"{Fore.MAGENTA}Target layers: {start_id} to {end_id}{Fore.RESET}")
    print(f"{Fore.MAGENTA}{'='*60}{Fore.RESET}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HCIU] Using device: {device}")
    
    # Load model
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"[HCIU] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get calibration data
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        min(dataset_size, calibration_samples),
        batch_size,
        tokenizer
    )
    
    # Determine target layers
    if end_id == 0:
        end_id = model.config.num_hidden_layers
    
    target_layers = list(range(start_id - 1, end_id - 1))
    print(f"[HCIU] Target layers for transformation: {target_layers}")
    
    # Calibrate thresholds
    dataloader_list = list(dataloader)
    thresholds = calibrate_hciu_thresholds(
        model, tokenizer, dataloader_list,
        target_layers, device, calibration_samples
    )
    
    # Configuration for HCIU components
    cache_config = {
        'max_entries': cache_max_entries,
        'key_dim': cache_key_dim,
        'similarity_threshold': thresholds.get('cache_similarity_threshold', cache_similarity_threshold)
    }
    
    update_config = {
        'min_rank': update_min_rank,
        'max_rank': update_max_rank
    }
    
    # Replace layers with HCIU-enhanced versions
    print(f"\n{Fore.CYAN}[HCIU] Replacing layers with HCIU-enhanced versions...{Fore.RESET}")
    
    for layer_idx in target_layers:
        original_layer = model.model.layers[layer_idx]
        
        hciu_layer = HCIULayer(
            original_layer=original_layer,
            layer_idx=layer_idx,
            hidden_dim=model.config.hidden_size,
            cache_config=cache_config,
            update_config=update_config,
            device=device
        )
        
        # Update thresholds from calibration
        hciu_layer.critical_threshold = thresholds['critical_threshold']
        hciu_layer.simple_threshold = thresholds['simple_threshold']
        
        model.model.layers[layer_idx] = hciu_layer
        
        print(f"[HCIU] Layer {layer_idx + 1} enhanced with HCIU")
    
    # Warm up caches with calibration data
    if warmup_samples > 0:
        print(f"\n{Fore.YELLOW}[HCIU] Warming up caches...{Fore.RESET}")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_text in enumerate(tqdm(dataloader_list[:warmup_samples], desc="Cache warmup")):
                inputs = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass to populate caches
                _ = model(**inputs)
        
        # Print cache statistics
        print(f"\n[HCIU] Cache warmup complete. Statistics:")
        for layer_idx in target_layers:
            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'get_statistics'):
                stats = layer.get_statistics()
                print(f"  Layer {layer_idx + 1}: Hit rate: {stats.get('hit_rate', 0):.2%}, "
                      f"Cache size: {stats.get('cache_size', 0)}")
    
    # Save model
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = f"output_models/{model_path.replace('/', '_')}_HCIU_{start_id}_{end_id}"
    
    print(f"\n{Fore.GREEN}[HCIU] Saving enhanced model to {save_path}{Fore.RESET}")
    
    # Move to CPU for saving
    model = model.to('cpu')
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save HCIU configuration and statistics
    hciu_metadata = {
        'method': 'HCIU',
        'layers_transformed': target_layers,
        'cache_config': cache_config,
        'update_config': update_config,
        'calibrated_thresholds': thresholds,
        'layer_statistics': {}
    }
    
    # Collect final statistics
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'get_statistics'):
            hciu_metadata['layer_statistics'][f'layer_{layer_idx}'] = layer.get_statistics()
    
    with open(f"{save_path}/hciu_metadata.json", 'w') as f:
        json.dump(hciu_metadata, f, indent=2, default=str)
    
    print(f"[HCIU] Metadata saved to {save_path}/hciu_metadata.json")
    
    # Print final summary
    print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}HCIU Transformation Complete!{Fore.RESET}")
    print(f"{Fore.GREEN}{'='*60}{Fore.RESET}")
    
    total_cache_hits = sum(
        hciu_metadata['layer_statistics'].get(f'layer_{idx}', {}).get('cache_hits', 0)
        for idx in target_layers
    )
    total_queries = sum(
        hciu_metadata['layer_statistics'].get(f'layer_{idx}', {}).get('total_queries', 0)
        for idx in target_layers
    )
    
    if total_queries > 0:
        overall_hit_rate = total_cache_hits / total_queries
        print(f"Overall cache hit rate: {overall_hit_rate:.2%}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path