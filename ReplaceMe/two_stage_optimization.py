# two_stage_optimization.py
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

from .utils import (adam_method, get_calib_dataloader,
                    select_non_overlapping_blocks, truncate_model, seed_all)

init(autoreset=True)

logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

def two_stage_optimization(
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
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False,
    stage1_epochs: int = 5,  # Epochs for stage 1 (MSE)
    stage2_epochs: int = 10,  # Epochs for stage 2 (Cosine)
    **kwargs
) -> str:
    """
    Two-stage optimization:
    Stage 1: Coarse approximation with MSE loss (analytical)
    Stage 2: Fine-tuning with Cosine loss (iterative)
    """
    print(f"[DEBUG] Using Two-Stage Optimization")
    print(f"[DEBUG] Stage 1: MSE loss (analytical)")
    print(f"[DEBUG] Stage 2: Cosine loss ({stage2_epochs} epochs)")
    
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
        def hook(module, input, output):
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
    all_a1 = []
    all_a2 = []
    all_a3 = [] if accurate else None
    
    cnt = 0
    print("[DEBUG] Gathering activations...")
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
        if accurate:
            a2_batch = hidden_states_n
            a3_batch = hidden_states_i - hidden_states_mlp
            all_a3.append(a3_batch.cpu())
        else:
            a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
            
        all_a1.append(a1_batch.cpu())
        all_a2.append(a2_batch.cpu())
    
    # Concatenate all batches
    a1 = torch.cat(all_a1, dim=0)
    a2 = torch.cat(all_a2, dim=0)
    if accurate:
        a3 = torch.cat(all_a3, dim=0)
    else:
        a3 = None
    
    print(f"[DEBUG] Collected {a1.shape[0]} samples")
    
    # ============ STAGE 1: MSE Loss (Analytical) ============
    print("[DEBUG] Stage 1: Computing initial transform with MSE loss")
    
    # Analytical least squares solution
    MtM = a1.T @ a1
    MtM_reg = MtM + 1e-6 * torch.eye(MtM.shape[0], dtype=torch.float64)
    
    try:
        if accurate and a3 is not None:
            T_stage1 = torch.linalg.solve(MtM_reg, a1.T @ (a2 - a3))
        else:
            T_stage1 = torch.linalg.solve(MtM_reg, a1.T @ a2)
        print(f"[DEBUG] Stage 1 complete. Transform stats - Max: {T_stage1.max():.4f}, Min: {T_stage1.min():.4f}")
    except Exception as e:
        print(f"[ERROR] Stage 1 failed: {e}. Using identity matrix.")
        T_stage1 = torch.eye(hidden_size, dtype=torch.float64)
    
    # Evaluate Stage 1 performance
    with torch.no_grad():
        pred_stage1 = a1 @ T_stage1
        if accurate and a3 is not None:
            pred_stage1 = pred_stage1 + a3
        mse_loss = ((pred_stage1 - a2) ** 2).mean()
        print(f"[DEBUG] Stage 1 MSE Loss: {mse_loss:.6f}")
    
    # ============ STAGE 2: Cosine Loss (Fine-tuning) ============
    print(f"[DEBUG] Stage 2: Fine-tuning with cosine loss for {stage2_epochs} epochs")
    
    # Custom adam method for stage 2 with initialized transform
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    
    class ActivationDataset(Dataset):
        def __init__(self, a1, a2, a3=None):
            self.a1, self.a2, self.a3 = a1, a2, a3
        def __len__(self):
            return len(self.a1)
        def __getitem__(self, idx):
            attn = torch.tensor([-1]) if self.a3 is None else self.a3[idx]
            return self.a1[idx], self.a2[idx], attn
    
    # Create dataset and loader
    dataset = ActivationDataset(a1, a2, a3)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Initialize with Stage 1 result
    T_stage2 = nn.Parameter(T_stage1.clone().cuda())
    optimizer = torch.optim.Adam([T_stage2], lr=1e-5)  # Lower LR for fine-tuning
    
    # Cosine loss function
    def cosine_loss(XA, Y):
        XA_norm = XA / (XA.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()
    
    # Fine-tuning loop
    best_loss = float('inf')
    best_T = T_stage2.clone()
    
    with tqdm(range(stage2_epochs), desc="Stage 2 Fine-tuning") as pbar:
        for epoch in pbar:
            epoch_loss = 0.0
            num_batches = 0
            
            for X, Y, Z in loader:
                optimizer.zero_grad()
                
                X = X.float().cuda()
                Y = Y.float().cuda()
                
                XA = X @ T_stage2
                if len(Z) != 1 and Z[0].item() != -1:
                    XA = XA + Z.float().cuda()
                
                loss = cosine_loss(XA, Y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([T_stage2], max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            pbar.set_postfix({'Cosine Loss': f'{avg_loss:.4f}'})
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_T = T_stage2.clone()
    
    T_final = best_T.cpu().to(torch.float64)
    print(f"[DEBUG] Stage 2 complete. Final cosine loss: {best_loss:.4f}")
    print(f"[DEBUG] Final transform stats - Max: {T_final.max():.4f}, Min: {T_final.min():.4f}")
    
    # Compare Stage 1 vs Stage 2
    print(f"[DEBUG] Transform difference (Stage2 - Stage1) norm: {(T_final - T_stage1).norm():.4f}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    print("[DEBUG] Applying final transform to model...")
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
            f"{end_id}_{dataset}_{dataset_size}_twostage"
        ).replace("/", "_")
    
    # Apply transformation
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": (T_final.T.cpu() @ model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight.to(torch.float64)).to(torch.bfloat16)
    })
    
    model.save_pretrained(f"{save_path}_TwoStage")
    tokenizer.save_pretrained(f"{save_path}_TwoStage")
    
    if save_transform_only:
        torch.save({
            'T_stage1': T_stage1,
            'T_final': T_final,
            'stage1_mse_loss': mse_loss.item(),
            'stage2_cosine_loss': best_loss
        }, f"{save_path}_TwoStage_transform")
    
    # Final cleanup
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_TwoStage"