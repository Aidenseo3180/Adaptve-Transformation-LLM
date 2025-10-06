# ============================================================
# vlm_cosine_dist.py (새 파일 생성)
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
from transformers import AutoModelForVision2Seq

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
    from transformers import BitsAndBytesConfig
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    logging.info(f"{Fore.GREEN}Loading VLM model: {model_path}{Fore.RESET}")
    
    # VLM 모델 로드
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
        torch_dtype=torch.bfloat16
    )
    
    # Processor 설정
    processor = setup_vlm_processor(model_path)
    
    # Language model layers 가져오기
    layers, num_hidden_layers = get_vlm_layers(model)
    hidden_size = model.config.text_config.hidden_size
    
    logging.info(f"{Fore.GREEN}Model loaded - Layers: {num_hidden_layers}, Hidden: {hidden_size}{Fore.RESET}")
    
    model.eval()
    
    # VLM calibration dataloader
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
    
    # Language model layers에 hook 등록
    for i, layer in enumerate(layers):
        hooks.append(
            layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp'))
        )
    
    logging.info(f"{Fore.GREEN}Registered {len(hooks)} hooks{Fore.RESET}")
    
    # Activation 저장 버퍼
    total_tokens = dataset_size * max_length if dataset_size else len(dataloader.dataset) * max_length
    
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
        logging.info(f"{Fore.YELLOW}ACCURATE MODE - More memory required{Fore.RESET}")
        a3 = torch.empty(
            (total_tokens, hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    
    # Activation 수집
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering VLM Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
        # Batch를 모델 device로 이동
        inputs = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Hidden states 추출 (vision + text 포함)
        hidden_states = outputs.hidden_states[1:]  # 첫 번째는 embedding
        
        # MLP outputs
        hidden_states_mlp = mlp_activations[f'layer_{start_id - num_layer - 1}']
        
        # Reshape
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.float64)
        
        # 배치 저장
        a1_batch = hidden_states_mlp
        
        if accurate:
            a2_batch = hidden_states_n
            a3_batch = hidden_states_i - hidden_states_mlp
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch.cpu()
        else:
            a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch.cpu()
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch.cpu()
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
        torch.cuda.empty_cache()
    
    # Activation 크기 조정
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
    
    logging.info(f"{Fore.GREEN}Collected {cnt} activation samples{Fore.RESET}")
    
    # Transform 추정
    logging.info(f"{Fore.CYAN}Estimating transformation - Solver: {solver}, Loss: {loss}{Fore.RESET}")
    
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
    
    logging.info(f"{Fore.GREEN}Transform estimated: {transform.shape}{Fore.RESET}")
    
    # 메모리 정리
    for hook in hooks:
        hook.remove()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 모델 재로드 및 truncate
    logging.info(f"{Fore.CYAN}Reloading model for truncation{Fore.RESET}")
    
    model = AutoModelForVision2Seq.from_pretrained(
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
    
    logging.info(f"{Fore.GREEN}Model saved to: {final_path}{Fore.RESET}")
    
    if save_transform_only:
        torch.save(transform, f"{final_path}_transform.pt")
        logging.info(f"{Fore.GREEN}Transform saved separately{Fore.RESET}")
    
    # 최종 정리
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path