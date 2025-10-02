import argparse
import gc
import logging
import os
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class AdaptiveKernelBlock(nn.Module):
    """
    Kernel-based replacement for multiple transformer layers
    Uses Random Fourier Features for efficient kernel approximation
    """
    def __init__(self, hidden_size: int, num_replaced_layers: int, compression_ratio: int = 4):
        super().__init__()
        print(f"[DEBUG] Initializing AdaptiveKernelBlock: hidden_size={hidden_size}, "
              f"num_replaced={num_replaced_layers}, compression_ratio={compression_ratio}")
        
        self.hidden_size = hidden_size
        self.num_replaced = num_replaced_layers
        
        # Random Fourier Features for kernel approximation
        self.num_features = hidden_size // compression_ratio
        print(f"[DEBUG] Number of random features: {self.num_features}")
        
        # Initialize with proper device and dtype from the start
        self.register_buffer('W', torch.randn(hidden_size, self.num_features) * 0.1)
        self.register_buffer('b', torch.rand(self.num_features) * 2 * np.pi)
        
        # Bottleneck for efficiency
        self.compress = nn.Linear(hidden_size, self.num_features, bias=False)
        self.expand = nn.Linear(self.num_features, hidden_size, bias=False)
        
        # Non-linearity
        self.activation = nn.GELU()
        
        # Learnable gate for smooth integration
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        
        # Variance correction
        self.register_buffer('variance_scale', torch.ones(1))
        
        print(f"[DEBUG] AdaptiveKernelBlock initialized successfully")
    
    def gaussian_kernel_features(self, x):
        """Apply Random Fourier Features for Gaussian kernel approximation"""
        # x shape: [batch, seq_len, hidden_size]
        device = x.device
        dtype = x.dtype
        
        # Ensure W and b are on the correct device
        if self.W.device != device:
            self.W = self.W.to(device)
            self.b = self.b.to(device)
        
        # Project to random features
        proj = torch.matmul(x, self.W.to(dtype)) + self.b.to(dtype)
        
        # Concatenate cos and sin features
        features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return features / np.sqrt(self.num_features)
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Forward pass with kernel approximation
        """
        # Store input for residual connection
        residual = hidden_states
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        # Debug shapes
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:
                print(f"[DEBUG] Forward pass {self._debug_counter}: input shape {hidden_states.shape}")
        else:
            self._debug_counter = 0
        
        # Apply kernel features
        kernel_features = self.gaussian_kernel_features(hidden_states)
        
        # Bottleneck transformation with non-linearity
        compressed = self.compress(hidden_states.to(self.compress.weight.dtype))
        activated = self.activation(compressed)
        expanded = self.expand(activated)
        
        # Ensure correct dtype for output
        expanded = expanded.to(dtype)
        
        # Apply variance correction
        output = residual + self.gate * self.variance_scale * expanded
        
        return output


def identify_critical_paths(
    model,
    dataloader,
    num_layers: int,
    device: str,
    max_samples: int = 100
) -> List[float]:
    """
    Identify critical computation paths in the model
    Returns importance scores for each layer
    """
    print(f"\n[DEBUG] Starting critical path identification with {max_samples} samples")
    
    # Initialize scores
    attention_entropy_sum = torch.zeros(num_layers)
    representation_shift_sum = torch.zeros(num_layers)
    sample_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing layers", total=max_samples)):
            if batch_idx >= max_samples:
                break
                
            inputs = model.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=512,  # Reduced for efficiency
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            
            # Compute metrics for each layer
            for layer_idx in range(min(len(hidden_states) - 1, num_layers)):
                # Compute representation shift
                curr_hidden = hidden_states[layer_idx].mean(dim=1)  # Average over sequence
                next_hidden = hidden_states[layer_idx + 1].mean(dim=1)
                
                # Cosine distance as representation shift
                cosine_sim = torch.nn.functional.cosine_similarity(curr_hidden, next_hidden, dim=-1)
                rep_shift = (1 - cosine_sim).mean()
                
                representation_shift_sum[layer_idx] += rep_shift.cpu()
            
            sample_count += 1
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Compute average scores
    critical_scores = representation_shift_sum / sample_count
    
    print(f"[DEBUG] Critical scores computed. Min: {critical_scores.min():.4f}, "
          f"Max: {critical_scores.max():.4f}, Mean: {critical_scores.mean():.4f}")
    
    return critical_scores.tolist()


def initialize_kernel_block(
    kernel_block: AdaptiveKernelBlock,
    original_model,
    start_layer: int,
    end_layer: int,
    calibration_dataloader,
    device: str,
    max_samples: int = 50
):
    """
    Initialize kernel block to approximate original layers
    """
    print(f"\n[DEBUG] Initializing kernel block to replace layers {start_layer}-{end_layer}")
    
    original_model.eval()
    kernel_block.eval()
    
    # Collect input-output pairs from original layers
    original_inputs = []
    original_outputs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_dataloader):
            if batch_idx >= max_samples:
                break
            
            inputs = original_model.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=256,  # Smaller for initialization
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get activations before and after the replaced layers
            outputs = original_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get input to the block (output of layer before start)
            block_input = hidden_states[start_layer]
            # Get output of the block (output of last replaced layer)
            block_output = hidden_states[end_layer]
            
            original_inputs.append(block_input)
            original_outputs.append(block_output)
    
    # Stack all samples
    all_inputs = torch.cat(original_inputs, dim=0)
    all_outputs = torch.cat(original_outputs, dim=0)
    
    print(f"[DEBUG] Collected {all_inputs.shape[0]} samples for initialization")
    
    # Initialize using SVD-based approximation
    try:
        # Compute difference (what the layers actually do)
        diff = all_outputs - all_inputs
        diff_flat = diff.view(-1, diff.size(-1))
        
        # Use a subset for SVD (for memory efficiency)
        subset_size = min(1000, diff_flat.size(0))
        diff_subset = diff_flat[:subset_size]
        
        # SVD to find principal components
        U, S, V = torch.svd(diff_subset.T.float())
        
        # Initialize compress/expand with principal components
        k = kernel_block.num_features
        kernel_block.compress.weight.data = V[:, :k].T.to(kernel_block.compress.weight.dtype)
        kernel_block.expand.weight.data = (U[:, :k] @ torch.diag(S[:k])).to(kernel_block.expand.weight.dtype)
        
        # Initialize gate based on magnitude of change
        avg_change = diff.abs().mean()
        avg_input = all_inputs.abs().mean()
        kernel_block.gate.data = (avg_change / avg_input).clamp(0.01, 0.5)
        
        # Compute variance scale
        output_var = all_outputs.var()
        input_var = all_inputs.var()
        kernel_block.variance_scale = torch.sqrt(output_var / input_var).clamp(0.5, 2.0)
        
        print(f"[DEBUG] Initialization complete. Gate: {kernel_block.gate.item():.4f}, "
              f"Variance scale: {kernel_block.variance_scale.item():.4f}")
        
    except Exception as e:
        print(f"[WARNING] SVD initialization failed: {e}. Using random initialization.")
        # Fallback to identity-like initialization
        kernel_block.gate.data = torch.tensor([0.1])
        kernel_block.variance_scale = torch.tensor([1.0])

def select_blocks_by_criticality(
    critical_scores: List[float],
    layers_to_skip: int,
    num_blocks: int = 1,
    merge_consecutive: bool = True
) -> List[Tuple[int, int]]:
    """
    Critical score 기반으로 블록 선택 (낮은 score = 덜 중요 = 교체 대상)
    
    Args:
        critical_scores: 각 레이어의 중요도 점수
        layers_to_skip: 한 블록당 레이어 수
        num_blocks: 선택할 블록 수
        merge_consecutive: 연속된 블록 병합 여부
    
    Returns:
        선택된 블록들 [(start, end), ...]
    """
    print(f"\n[DEBUG] Selecting blocks based on criticality scores")
    print(f"[DEBUG] Score stats - Min: {min(critical_scores):.4f}, "
          f"Max: {max(critical_scores):.4f}, Mean: {np.mean(critical_scores):.4f}")
    
    num_layers = len(critical_scores)
    
    # 가능한 모든 블록과 평균 critical score 계산
    possible_blocks = []
    for start in range(num_layers - layers_to_skip + 1):
        end = start + layers_to_skip
        # 블록의 평균 critical score (낮을수록 덜 중요)
        avg_score = np.mean(critical_scores[start:end])
        possible_blocks.append((start, end, avg_score))
    
    # Critical score가 낮은 순으로 정렬 (덜 중요한 블록부터)
    possible_blocks.sort(key=lambda x: x[2])
    
    print(f"[DEBUG] Found {len(possible_blocks)} possible blocks")
    print(f"[DEBUG] Least critical block: layers {possible_blocks[0][0]}-{possible_blocks[0][1]} "
          f"with score {possible_blocks[0][2]:.4f}")
    
    # Non-overlapping 블록 선택
    selected = []
    used_layers = set()
    
    for start, end, score in possible_blocks:
        block_layers = set(range(start, end))
        
        # 겹치지 않는 블록만 선택
        if not block_layers & used_layers:
            selected.append((start, end, score))
            used_layers.update(block_layers)
            
            print(f"[DEBUG] Selected block: layers {start}-{end} (score: {score:.4f})")
            
            if len(selected) >= num_blocks:
                break
    
    # 연속된 블록 병합 (선택사항)
    if merge_consecutive and len(selected) > 1:
        selected.sort(key=lambda x: x[0])  # start 기준 정렬
        merged = []
        current_start, current_end, current_score = selected[0]
        
        for start, end, score in selected[1:]:
            if start == current_end:  # 연속된 블록
                current_end = end
                # 병합된 블록의 score는 평균
                current_score = (current_score + score) / 2
                print(f"[DEBUG] Merging blocks: {current_start}-{start} + {start}-{end}")
            else:
                merged.append((current_start, current_end))
                current_start, current_end, current_score = start, end, score
        
        merged.append((current_start, current_end))
        print(f"[DEBUG] After merging: {merged}")
        return merged
    
    # score 제거하고 (start, end) 튜플만 반환
    return [(start, end) for start, end, _ in selected]


def compute_critical_scores(
    model,
    dataloader,
    device: str,
    max_samples: int = 100
) -> List[float]:
    """
    각 레이어의 critical score 계산
    높은 score = 중요한 레이어 (교체하면 안됨)
    낮은 score = 덜 중요한 레이어 (교체 가능)
    """
    print(f"\n[DEBUG] Computing critical scores for {model.config.num_hidden_layers} layers")
    
    num_layers = model.config.num_hidden_layers
    
    # 각 레이어의 중요도 메트릭 초기화
    attention_diversity = torch.zeros(num_layers)
    representation_change = torch.zeros(num_layers)
    information_flow = torch.zeros(num_layers)
    sample_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing critical scores")):
            if batch_idx >= max_samples:
                break
            
            inputs = model.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass with attention weights
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
            
            for layer_idx in range(min(len(hidden_states) - 1, num_layers)):
                # 1. Representation change (높을수록 중요)
                curr = hidden_states[layer_idx]
                next = hidden_states[layer_idx + 1] if layer_idx + 1 < len(hidden_states) else curr
                
                # Layer 전후 변화량
                diff = (next - curr).norm(dim=-1).mean()
                representation_change[layer_idx] += diff.cpu()
                
                # 2. Attention diversity (높을수록 중요)
                if attentions and layer_idx < len(attentions):
                    attn = attentions[layer_idx]  # [batch, heads, seq, seq]
                    # Entropy of attention distribution
                    entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean()
                    attention_diversity[layer_idx] += entropy.cpu()
                
                # 3. Information flow (gradient norm proxy)
                # 출력과의 correlation (높을수록 중요)
                if layer_idx < len(hidden_states) - 1:
                    final_hidden = hidden_states[-1].mean(dim=1)  # [batch, hidden]
                    curr_hidden = hidden_states[layer_idx].mean(dim=1)
                    correlation = torch.nn.functional.cosine_similarity(
                        final_hidden, curr_hidden, dim=-1
                    ).abs().mean()
                    information_flow[layer_idx] += correlation.cpu()
            
            sample_count += 1
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # 평균 계산 및 정규화
    attention_diversity = attention_diversity / sample_count
    representation_change = representation_change / sample_count
    information_flow = information_flow / sample_count
    
    # 각 메트릭 정규화 (0-1 범위)
    def normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            return (tensor - min_val) / (max_val - min_val)
        return tensor
    
    attention_diversity = normalize(attention_diversity)
    representation_change = normalize(representation_change)
    information_flow = normalize(information_flow)
    
    # 종합 critical score (높을수록 중요)
    critical_scores = (
        0.3 * attention_diversity +
        0.4 * representation_change +
        0.3 * information_flow
    )
    
    # 디버그 출력
    for i in range(min(5, num_layers)):
        print(f"[DEBUG] Layer {i}: attention={attention_diversity[i]:.3f}, "
              f"change={representation_change[i]:.3f}, flow={information_flow[i]:.3f}, "
              f"critical={critical_scores[i]:.3f}")
    
    return critical_scores.tolist()


# adaptive_kernel_replacement 함수 내부 수정
def adaptive_kernel_replacement(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    compression_ratio: int = 4,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    num_blocks: int = 1,
    merge_consecutive: bool = True,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    **kwargs  # 나머지 파라미터 무시
) -> str:
    """
    Adaptive Kernel replacement main function
    
    이 함수는 distances_path를 사용하지 않고,
    자체적인 critical path 분석을 수행합니다.
    """
    print(f"\n{'='*60}")
    print(f"Adaptive Kernel Network (AKN) Replacement")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset}")
    print(f"Layers per block: {layers_to_skip}")
    print(f"Number of blocks: {num_blocks}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"{'='*60}\n")
    
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    # 4-bit quantization 설정
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("[INFO] 4-bit quantization enabled")
    
    # 모델 로드
    print(f"\n[INFO] Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device if device == "cuda" else "cpu",
        quantization_config=quantization_config,
        output_hidden_states=True,
        output_attentions=True,  # Attention weights도 필요
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer
    
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    
    print(f"[INFO] Model loaded: {num_layers} layers, hidden_size={hidden_size}")
    
    # Calibration 데이터 로드
    print("\n[INFO] Loading calibration data")
    cal_size = min(dataset_size, 1000) if dataset_size else 1000
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        cal_size,
        batch_size,
        tokenizer
    )
    
    # Critical scores 계산
    print("\n[INFO] Computing critical scores")
    critical_scores = compute_critical_scores(
        model, dataloader, device, max_samples=100
    )
    
    # 블록 선택 (우리의 기준으로)
    selected_blocks = select_blocks_by_criticality(
        critical_scores,
        layers_to_skip,
        num_blocks,
        merge_consecutive
    )
    
    print(f"\n[INFO] Selected blocks for replacement: {selected_blocks}")
    
    # 메모리 정리 후 모델 재로드 (CPU에서 수정)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n[INFO] Reloading model on CPU for modification")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        token=token
    )
    model.tokenizer = tokenizer
    
    # 각 블록 교체
    total_removed = 0
    for block_idx, (start_layer, end_layer) in enumerate(selected_blocks):
        print(f"\n[INFO] Replacing block {block_idx+1}/{len(selected_blocks)}: "
              f"layers {start_layer}-{end_layer-1}")
        
        # Kernel block 생성
        kernel_block = AdaptiveKernelBlock(
            hidden_size=hidden_size,
            num_replaced_layers=end_layer - start_layer,
            compression_ratio=compression_ratio
        )
        
        # 초기화를 위해 임시로 GPU로 이동
        model = model.to(device)
        kernel_block = kernel_block.to(device)
        
        # Kernel block 초기화
        initialize_kernel_block(
            kernel_block, model, start_layer, end_layer,
            dataloader, device, max_samples=50
        )
        
        # CPU로 되돌리기
        model = model.cpu()
        kernel_block = kernel_block.cpu()
        
        # 레이어 교체
        adjusted_start = start_layer - total_removed
        adjusted_end = end_layer - total_removed
        
        new_layers = nn.ModuleList()
        for i, layer in enumerate(model.model.layers):
            if i == adjusted_start:
                new_layers.append(kernel_block)
                print(f"[INFO] Inserted kernel block at position {i}")
            elif i < adjusted_start or i >= adjusted_end:
                new_layers.append(layer)
        
        model.model.layers = new_layers
        total_removed += (end_layer - start_layer - 1)
    
    # 모델 설정 업데이트
    model.config.num_hidden_layers = len(model.model.layers)
    print(f"\n[INFO] Final model has {len(model.model.layers)} layers")
    
    # 모델 저장
    if save_path is None:
        os.makedirs('output_models', exist_ok=True)
        save_path = f"output_models/{model_path.replace('/', '_')}_AKN_{layers_to_skip}L_{num_blocks}B"
    
    print(f"\n[INFO] Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[SUCCESS] Model saved successfully!")
    
    # 검증
    try:
        test_model = AutoModelForCausalLM.from_pretrained(save_path, device_map="cpu")
        print(f"[SUCCESS] Verification passed. Model has {len(test_model.model.layers)} layers")
        del test_model
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    
    # 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path

