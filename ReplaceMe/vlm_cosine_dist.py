# ============================================================
# vlm_cosine_dist.py - Fixed Version
# ============================================================

import argparse
import gc
import logging
import os
from typing import Optional
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

from .utils import (
    adam_method, 
    get_vlm_calib_dataloader, 
    setup_vlm_processor,
    get_vlm_layers,
    truncate_vlm_model,
    apply_vlm_transform,
    optimizing_method,
    seed_all
)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def vlm_cosine_dist(
    model_path: str,
    image_dir: str = "train2014",
    batch_size: int = 4,
    max_length: int = 512,
    layers_to_skip: int = 8,
    dataset_size: Optional[int] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
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
    accurate: bool = False,
    token: Optional[str] = None
) -> str:
    """VLM-specific cosine distance pruning with visual calibration data.
    
    Args:
        model_path: Path to VLM model
        image_dir: Directory containing calibration images
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Number of samples to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save model
        save_transform_only: Save only transformation matrix
        diag: Use diagonal matrix
        loss: Loss function (cosine/mse)
        solver: Optimizer (adam/cg/l-bfgs)
        thri: Use triangular matrix
        two_vectors: Use two-vector decomposition
        start_id: Start layer index
        end_id: End layer index
        num_layer: Number of layers processed
        num_A: Number of transformations
        merge_consecutive: Merge consecutive blocks
        accurate: Use accurate mode (more memory)
        token: HuggingFace token
        
    Returns:
        Path to saved model
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
    
    print(f"{Fore.GREEN}Loading VLM model: {model_path}{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    processor = setup_vlm_processor(model_path)
    layers, num_hidden_layers = get_vlm_layers(model)
    hidden_size = model.config.text_config.hidden_size
    
    print(f"{Fore.GREEN}Model loaded - Layers: {num_hidden_layers}, Hidden: {hidden_size}{Fore.RESET}")
    
    model.eval()
    
    dataloader = get_vlm_calib_dataloader(
        image_dir,
        dataset_size,
        batch_size,
        processor
    )
    
    # MLP activation hook
    def save_mlp_activation(name):
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    
    for i, layer in enumerate(layers):
        hooks.append(
            layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        )
    
    print(f"{Fore.GREEN}Registered {len(hooks)} hooks{Fore.RESET}")
    
    # ===== FIXED: VLM용 토큰 수 추정 =====
    total_samples = dataset_size if dataset_size else len(dataloader.dataset)
    
    # LLaVA: visual(576) + text(평균 100) + 여유 = 1000 tokens/image
    # 안전하게 1.5배 버퍼 추가
    estimated_tokens_per_image = 1000
    total_tokens = int(total_samples * estimated_tokens_per_image * 1.5)
    
    print(f"{Fore.CYAN}Pre-allocating for {total_samples} images{Fore.RESET}")
    print(f"{Fore.CYAN}Estimated: {total_tokens:,} tokens (1.5x buffer){Fore.RESET}")
    
    a1 = torch.empty(
        (total_tokens, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (total_tokens, hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    if accurate:
        print(f"{Fore.YELLOW}ACCURATE MODE (using more memory){Fore.RESET}")
        a3 = torch.empty(
            (total_tokens, hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    
    # Activation 수집
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=Fore.RED + "Gathering VLM Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    )):
        inputs = {k: v.to(model.device) for k, v in batch.items() 
                 if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp = mlp_activations[f'layer_{start_id - num_layer - 1}_mlp']
        
        # Reshape
        h_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.bfloat16)
        h_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        h_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        
        # Compute activations
        a1_batch = h_mlp
        
        if accurate:
            a2_batch = h_n
            a3_batch = h_i - h_mlp
        else:
            a2_batch = h_n + h_mlp - h_i
        
        batch_size_actual = a1_batch.shape[0]
        
        # ===== FIXED: 버퍼 체크 및 안전장치 =====
        if cnt + batch_size_actual > total_tokens:
            print(f"\n{Fore.RED}ERROR: Buffer overflow at batch {batch_idx}!{Fore.RESET}")
            print(f"  Current position: {cnt:,}")
            print(f"  Batch size: {batch_size_actual:,}")
            print(f"  Required: {cnt + batch_size_actual:,}")
            print(f"  Allocated: {total_tokens:,}")
            print(f"  Average tokens/image so far: {cnt / (batch_idx * batch_size):.1f}")
            print(f"{Fore.YELLOW}Stopping collection early. Using {cnt:,} tokens.{Fore.RESET}")
            break
        
        # 쓰기
        a1[cnt:cnt+batch_size_actual] = a1_batch.cpu()
        a2[cnt:cnt+batch_size_actual] = a2_batch.cpu()
        if accurate:
            a3[cnt:cnt+batch_size_actual] = a3_batch.cpu()
        
        cnt += batch_size_actual
        
        # ===== FIXED: 주기적 진행 상황 출력 =====
        if (batch_idx + 1) % 100 == 0:
            avg_tokens = cnt / ((batch_idx + 1) * batch_size)
            usage_pct = (cnt / total_tokens) * 100
            print(f"\n  [{batch_idx+1} batches] Avg tokens/image: {avg_tokens:.1f}, Buffer usage: {usage_pct:.1f}%")
        
        # 메모리 정리
        del hidden_states_mlp, h_i, h_n, h_mlp, a1_batch, a2_batch
        if accurate:
            del a3_batch
        torch.cuda.empty_cache()
    
    # 실제 사용된 부분만 slice
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
    
    # ===== FIXED: 최종 통계 출력 =====
    avg_tokens_per_image = cnt / ((batch_idx + 1) * batch_size)
    usage_pct = (cnt / total_tokens) * 100
    
    print(f"\n{Fore.GREEN}Collection complete:{Fore.RESET}")
    print(f"  Total tokens collected: {cnt:,}")
    print(f"  Avg tokens/image: {avg_tokens_per_image:.1f}")
    print(f"  Buffer usage: {usage_pct:.1f}% ({cnt:,} / {total_tokens:,})")
    
    # Transform 추정
    print(f"{Fore.CYAN}Estimating transformation - Solver: {solver}, Loss: {loss}{Fore.RESET}")
    print(f"{Fore.YELLOW}Using bfloat16 (adam_method will convert internally){Fore.RESET}")
    
    # ===== bfloat16 그대로 전달 (메모리 절약) =====
    # adam_method 내부에서 배치마다 float로 변환하므로 문제없음
    if solver == "adam":
        transform = adam_method(
            a1,  # bfloat16 그대로!
            a2, 
            a3=a3 if accurate else None, 
            loss=loss, 
            diag=diag, 
            two_vectors=two_vectors, 
            thri=thri
        )
    else:
        # optimizing_method는 float64 필요하므로 변환
        print(f"{Fore.YELLOW}Converting to float64 for {solver}...{Fore.RESET}")
        transform = optimizing_method(
            a1.to(torch.float64), 
            a2.to(torch.float64), 
            a3=a3.to(torch.float64) if accurate else None, 
            solver=solver
        )
    
    print(f"{Fore.GREEN}Transform estimated: {transform.shape}{Fore.RESET}")
    
    # 메모리 정리
    for hook in hooks:
        hook.remove()
    
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    # 모델 재로드 및 truncate
    print(f"{Fore.CYAN}Reloading model for truncation{Fore.RESET}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    model = truncate_vlm_model(model, start_id - num_layer, end_id - num_layer)
    
    # Save path 설정
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/vlm_{os.path.basename(model_path)}_{layers_to_skip}layers"
    
    # Transform 적용
    apply_vlm_transform(model, transform, start_id - num_layer - 1)
    
    # 모델 저장
    final_path = f"{save_path}_ReplaceMe_{loss}_{solver}"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    print(f"{Fore.GREEN}Model saved to: {final_path}{Fore.RESET}")
    
    if save_transform_only:
        torch.save(transform, f"{final_path}_transform.pt")
        print(f"{Fore.GREEN}Transform saved separately{Fore.RESET}")
    
    # 최종 정리
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path



