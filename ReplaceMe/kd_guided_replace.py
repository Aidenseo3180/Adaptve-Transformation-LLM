import argparse
import gc
import logging
import os
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

from .utils import get_calib_dataloader, truncate_model, seed_all, select_non_overlapping_blocks

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class TeacherStudentDistiller:
    """Handles teacher-student distillation for finding optimal T matrix."""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        start_id: int,
        end_id: int,
        hidden_size: int,
        device: str = 'cuda'
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.start_id = start_id
        self.end_id = end_id
        self.hidden_size = hidden_size
        self.device = device
        
        # Layer indices (0-based internally)
        self.mlp_layer_idx = start_id - 1  # Layer 19 if start_id=20
        self.target_layer_idx = end_id - 1  # Layer 27 if end_id=28
        
        print(f"{Fore.GREEN}Distiller initialized:{Fore.RESET}")
        print(f"  MLP layer (gets T): {self.mlp_layer_idx}")
        print(f"  Target layer output: {self.target_layer_idx}")
        print(f"  Layers to remove: {start_id} to {end_id-1}")
    
    def collect_teacher_outputs(
        self,
        dataloader,
        tokenizer,
        max_length: int,
        total_samples: int
    ) -> Dict[str, torch.Tensor]:
        """Collect teacher model outputs for distillation."""
        
        print(f"\n{Fore.YELLOW}Collecting teacher outputs...{Fore.RESET}")
        
        # Storage for teacher outputs (CPU to save memory)
        teacher_data = {
            'mlp_outputs': torch.empty((total_samples, self.hidden_size), dtype=torch.bfloat16, device='cpu'),
            'layer_19_outputs': torch.empty((total_samples, self.hidden_size), dtype=torch.bfloat16, device='cpu'),
            'target_outputs': torch.empty((total_samples, self.hidden_size), dtype=torch.bfloat16, device='cpu'),
            'final_logits': []  # Store separately as they can be large
        }
        
        # Hooks to capture intermediate outputs
        mlp_output = {}
        layer_outputs = {}
        
        def save_mlp_hook(name):
            def hook(module, input, output):
                mlp_output[name] = output.detach()
            return hook
        
        def save_layer_hook(name):
            def hook(module, input, output):
                # output[0] is hidden states
                layer_outputs[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
            return hook
        
        # Register hooks
        hooks = []
        if hasattr(self.teacher, 'model'):  # Llama style
            # Hook for MLP output
            hooks.append(
                self.teacher.model.layers[self.mlp_layer_idx].mlp.register_forward_hook(
                    save_mlp_hook('mlp')
                )
            )
            # Hook for layer outputs
            hooks.append(
                self.teacher.model.layers[self.mlp_layer_idx].register_forward_hook(
                    save_layer_hook('layer_19')
                )
            )
            hooks.append(
                self.teacher.model.layers[self.target_layer_idx].register_forward_hook(
                    save_layer_hook('target')
                )
            )
        
        # Collect outputs
        cnt = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Teacher forward pass", colour="green"):
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True
                )
                inputs = {k: v.to(self.teacher.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.teacher(**inputs)
                
                # Get last token outputs (for causal LM)
                batch_size = inputs['input_ids'].shape[0]
                seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
                
                for i in range(batch_size):
                    if cnt >= total_samples:
                        break
                    
                    seq_len = seq_lengths[i].item()
                    
                    # Extract last token representations
                    teacher_data['mlp_outputs'][cnt] = mlp_output['mlp'][i, seq_len].cpu().to(torch.bfloat16)
                    teacher_data['layer_19_outputs'][cnt] = layer_outputs['layer_19'][i, seq_len].cpu().to(torch.bfloat16)
                    teacher_data['target_outputs'][cnt] = layer_outputs['target'][i, seq_len].cpu().to(torch.bfloat16)
                    
                    # Store logits (last token)
                    teacher_data['final_logits'].append(
                        outputs.logits[i, seq_len].cpu().to(torch.bfloat16)
                    )
                    
                    cnt += 1
                
                # Clear GPU memory
                del outputs
                torch.cuda.empty_cache()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Stack logits
        teacher_data['final_logits'] = torch.stack(teacher_data['final_logits'][:cnt])
        
        # Trim to actual size
        for key in ['mlp_outputs', 'layer_19_outputs', 'target_outputs']:
            teacher_data[key] = teacher_data[key][:cnt]
        
        print(f"Collected {cnt} teacher samples")
        print(f"  MLP output shape: {teacher_data['mlp_outputs'].shape}")
        print(f"  Target output shape: {teacher_data['target_outputs'].shape}")
        
        return teacher_data
    
    def compute_residual_path(
        self,
        teacher_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the residual (attention) path that bypasses MLP."""
        
        # Layer output = MLP output + residual (attention + input)
        # So residual = Layer output - MLP output
        
        mlp_out = teacher_data['mlp_outputs']
        layer_out = teacher_data['layer_19_outputs']
        
        # This is what gets added to MLP output
        residual = layer_out - mlp_out
        
        print(f"Computed residual path: shape={residual.shape}")
        print(f"  Residual norm mean: {residual.norm(dim=1).mean():.3f}")
        print(f"  MLP norm mean: {mlp_out.norm(dim=1).mean():.3f}")
        
        return residual


def optimize_transform_with_distillation(
    teacher_data: Dict[str, torch.Tensor],
    hidden_size: int,
    device: str = 'cuda',
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    alpha: float = 0.8  # Weight for intermediate loss
) -> torch.Tensor:
    """Optimize T matrix using knowledge distillation."""
    
    print(f"\n{Fore.CYAN}Optimizing T with Knowledge Distillation{Fore.RESET}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Alpha (intermediate weight): {alpha}")
    
    # Initialize T as identity
    T = torch.eye(hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=lr)
    
    # Prepare data
    mlp_outputs = teacher_data['mlp_outputs'].to(torch.float32)
    target_outputs = teacher_data['target_outputs'].to(torch.float32)
    residual = teacher_data['layer_19_outputs'].to(torch.float32) - mlp_outputs
    final_logits = teacher_data['final_logits'].to(torch.float32)
    
    n_samples = mlp_outputs.shape[0]
    
    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    best_T = T.clone().detach()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_intermediate_loss = 0.0
        epoch_final_loss = 0.0
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            mlp_batch = mlp_outputs[batch_indices].to(device)
            target_batch = target_outputs[batch_indices].to(device)
            residual_batch = residual[batch_indices].to(device)
            logits_batch = final_logits[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            # Student prediction: MLP @ T + residual should match target
            student_pred = mlp_batch @ T + residual_batch
            
            # Intermediate loss: Match Layer 27 output
            intermediate_loss = F.mse_loss(student_pred, target_batch)
            
            # For final loss, we'd need to run through remaining layers
            # Here we use a proxy: match the direction and magnitude
            student_norm = student_pred / (student_pred.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = target_batch / (target_batch.norm(dim=1, keepdim=True) + 1e-8)
            cosine_loss = 1 - (student_norm * target_norm).sum(dim=1).mean()
            
            # Combined loss
            loss = alpha * intermediate_loss + (1 - alpha) * cosine_loss
            
            # Add regularization to keep T close to identity
            reg_loss = 1e-4 * (T - torch.eye(hidden_size, device=device)).pow(2).mean()
            loss = loss + reg_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([T], max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_intermediate_loss += intermediate_loss.item()
            
            # Clear memory
            del mlp_batch, target_batch, residual_batch, student_pred
            torch.cuda.empty_cache()
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_intermediate = epoch_intermediate_loss / num_batches
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.6f}, Intermediate={avg_intermediate:.6f}, LR={scheduler.get_last_lr()[0]:.1e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T.clone().detach()
            print(f"    {Fore.GREEN}New best!{Fore.RESET}")
    
    # Evaluate final quality
    with torch.no_grad():
        mlp_gpu = mlp_outputs[:1000].to(device)
        target_gpu = target_outputs[:1000].to(device)
        residual_gpu = residual[:1000].to(device)
        
        pred = mlp_gpu @ best_T + residual_gpu
        
        mse = F.mse_loss(pred, target_gpu).item()
        
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target_gpu / (target_gpu.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=1).mean().item()
        
        print(f"\n{Fore.CYAN}Final T evaluation:{Fore.RESET}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")
        
        # Compare with identity
        identity_pred = mlp_gpu + residual_gpu
        identity_mse = F.mse_loss(identity_pred, target_gpu).item()
        print(f"  Identity baseline MSE: {identity_mse:.6f}")
        print(f"  Improvement: {((identity_mse - mse) / identity_mse * 100):.1f}%")
    
    return best_T.cpu()


def kd_guided_replace(
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
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    kd_epochs: int = 10,
    kd_lr: float = 1e-4,
    kd_alpha: float = 0.8,
    **kwargs
) -> str:
    """Main function for Knowledge Distillation Guided ReplaceMe."""
    
    print(f"\n{Fore.CYAN}=== Knowledge Distillation Guided ReplaceMe ==={Fore.RESET}")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id}")
    print(f"Dataset: {dataset}, Size: {dataset_size}")
    
    # Load teacher model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"\n{Fore.YELLOW}Loading teacher model...{Fore.RESET}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    hidden_size = teacher_model.config.hidden_size
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    teacher_model.eval()
    
    # Create student (will be created after collecting teacher outputs)
    print(f"\n{Fore.YELLOW}Creating student model (copy of teacher for now)...{Fore.RESET}")
    student_model = teacher_model  # We'll modify this later
    
    # Create distiller
    distiller = TeacherStudentDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        start_id=start_id,
        end_id=end_id,
        hidden_size=hidden_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Get dataloader
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    # Collect teacher outputs
    total_samples = min(dataset_size, 50000)  # Limit to prevent OOM
    teacher_data = distiller.collect_teacher_outputs(
        dataloader=dataloader,
        tokenizer=tokenizer,
        max_length=max_length,
        total_samples=total_samples
    )
    
    # Clear teacher from GPU to save memory
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Optimize T matrix using distillation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = optimize_transform_with_distillation(
        teacher_data=teacher_data,
        hidden_size=hidden_size,
        device=device,
        epochs=kd_epochs,
        batch_size=256,
        lr=kd_lr,
        alpha=kd_alpha
    )
    
    print(f"\n{Fore.YELLOW}Applying transformation and saving model...{Fore.RESET}")
    
    # Load model again (CPU only for modification)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Remove layers
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Apply transformation to down_proj of layer before removed section
    layer_idx = start_id - num_layer - 1
    original_weight = model.model.layers[layer_idx].mlp.down_proj.weight
    
    print(f"Applying T to layer {layer_idx} down_proj")
    print(f"  Original weight shape: {original_weight.shape}")
    print(f"  Transform shape: {transform.shape}")
    
    # Apply: new_weight = T.T @ original_weight
    new_weight = (transform.T.to(torch.float64) @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[layer_idx].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    # Prepare save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = f"output_models/{model_path}_{layers_to_skip}_layers_{start_id}_{end_id}_kd_guided".replace("/", "_")
    
    # Save model and tokenizer
    full_save_path = f"{save_path}_ReplaceMe_KD"
    model.save_pretrained(full_save_path)
    tokenizer.save_pretrained(full_save_path)
    
    print(f"{Fore.GREEN}Model saved to: {full_save_path}{Fore.RESET}")
    
    # Save additional data for analysis
    torch.save({
        'transform': transform,
        'start_id': start_id,
        'end_id': end_id,
        'method': 'knowledge_distillation',
        'kd_alpha': kd_alpha,
        'kd_epochs': kd_epochs
    }, f"{full_save_path}_transform_data.pt")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return full_save_path