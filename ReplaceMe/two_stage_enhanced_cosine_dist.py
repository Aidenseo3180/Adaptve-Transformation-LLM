import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import gc
import logging
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, truncate_model, seed_all

def compute_cosine_distance_loss(transformed, target):
    """
    Compute cosine distance loss between transformed and target activations
    """
    # Normalize along the last dimension
    transformed_norm = transformed / (torch.norm(transformed, dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    
    # Cosine similarity
    cosine_sim = (transformed_norm * target_norm).sum(dim=-1)
    
    # Cosine distance (1 - cosine similarity)
    cosine_dist = 1 - cosine_sim.mean()
    
    return cosine_dist


def find_optimal_rank_with_performance(
    M_i: torch.Tensor, 
    Y_i: torch.Tensor, 
    L_i_n: torch.Tensor, 
    variance_threshold: float = 0.95, 
    max_rank: int = 256,
    alpha_range: list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    sample_ratio: float = 0.05
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    수정된 버전: M_i는 (N, 14336), L_i_n은 (N, 4096)
    올바른 down_proj transformation 학습
    """
    device = M_i.device
    original_dtype = M_i.dtype
    
    print(f"Input shapes: M_i={M_i.shape}, L_i_n={L_i_n.shape}, dtype={original_dtype}")
    
    # Step 1: Smart sampling for efficiency
    total_samples = M_i.shape[0]
    sample_size = min(int(total_samples * sample_ratio), 500000)  # Max 500k samples
    
    print(f"Sampling {sample_size} out of {total_samples} samples ({sample_ratio*100:.1f}%)")
    
    # Random sampling for representative data
    indices = torch.randperm(total_samples)[:sample_size]
    M_sample = M_i[indices]
    L_sample = L_i_n[indices]
    
    print(f"Sampled data shapes: M={M_sample.shape}, L={L_sample.shape}")
    
    # Step 2: Mixed precision computation (Float32 for numerical stability)
    print("Converting to Float32 for numerical stability...")
    M_f32 = M_sample.float()
    L_f32 = L_sample.float()
    
    # Step 3: Solve linear system M @ T ≈ L with better conditioning
    # 이제 T는 (14336, 4096) 형태가 되어야 함
    target = L_f32  # 더 이상 Y를 빼지 않음 (down_proj 직접 변환)
    
    # Use batched least squares for better numerical stability
    batch_size = min(8192, M_f32.shape[0])
    
    if M_f32.shape[0] <= batch_size:
        # Small enough to solve directly
        try:
            # Add ridge regression for stability
            ridge_alpha = 1e-4
            AtA = M_f32.T @ M_f32 + ridge_alpha * torch.eye(M_f32.shape[1], device=device)
            AtB = M_f32.T @ target
            
            print(f"Condition number: {torch.linalg.cond(AtA).item():.2e}")
            
            T_init = torch.linalg.solve(AtA, AtB)
            print(f"Successfully solved linear system, T_init shape: {T_init.shape}")
            
        except Exception as e:
            print(f"Direct solve failed: {e}, using SVD-based pseudo-inverse")
            T_init, _ = torch.linalg.lstsq(M_f32, target, rcond=1e-4)
            print(f"SVD-based solution successful, T_init shape: {T_init.shape}")
    else:
        # Use iterative method for large data
        print("Using iterative least squares for large data...")
        T_init = torch.zeros(M_f32.shape[1], target.shape[1], device=device)  # (14336, 4096)
        weight_sum = 0
        
        for i in range(0, M_f32.shape[0], batch_size):
            end_idx = min(i + batch_size, M_f32.shape[0])
            M_batch = M_f32[i:end_idx]
            target_batch = target[i:end_idx]
            
            try:
                ridge_alpha = 1e-4
                AtA = M_batch.T @ M_batch + ridge_alpha * torch.eye(M_batch.shape[1], device=device)
                AtB = M_batch.T @ target_batch
                T_batch = torch.linalg.solve(AtA, AtB)
                
                batch_weight = M_batch.shape[0]
                T_init += batch_weight * T_batch
                weight_sum += batch_weight
                
            except:
                continue
        
        if weight_sum > 0:
            T_init /= weight_sum
            print(f"Iterative solution successful, T_init shape: {T_init.shape}")
        else:
            print("All batches failed, using random initialization")
            T_init = torch.randn(M_f32.shape[1], target.shape[1], device=device) * 0.01
    
    # Step 4: SVD decomposition
    try:
        U, S, Vt = torch.svd(T_init)
        print(f"SVD successful, U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
        print(f"Singular values range: [{S.min():.2e}, {S.max():.2e}]")
        
        # Remove very small singular values for stability
        valid_mask = S > S.max() * 1e-6
        S = S[valid_mask]
        U = U[:, valid_mask]
        Vt = Vt[valid_mask, :]
        print(f"After filtering small singular values: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
        
    except Exception as e:
        print(f"SVD failed: {e}, using random initialization")
        min_dim = min(M_f32.shape[1], target.shape[1])
        U = torch.randn(M_f32.shape[1], min_dim, device=device)
        S = torch.ones(min_dim, device=device)
        Vt = torch.randn(min_dim, target.shape[1], device=device)
    
    # Step 5: Smart rank selection
    total_variance = torch.sum(S**2)
    cumsum_variance = torch.cumsum(S**2, dim=0) / total_variance
    
    # Find initial rank based on variance threshold
    rank_candidates = torch.where(cumsum_variance >= variance_threshold)[0]
    if len(rank_candidates) > 0:
        initial_rank = min(rank_candidates[0].item() + 1, max_rank, len(S))
    else:
        initial_rank = min(max_rank, len(S))

    # TODO: hardcoded for now
    initial_rank = 256
    
    print(f"Initial rank from variance threshold: {initial_rank}")
    
    # Step 6: Performance-based rank optimization
    def evaluate_rank_performance(rank, alpha):
        """Evaluate performance for given rank and alpha"""
        if rank <= 0 or rank > len(S):
            return float('inf')
            
        # Low-rank approximation
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        T_r = U_r @ torch.diag(S_r) @ Vt_r
        
        # Apply transformation
        transformed = M_f32 @ T_r
        
        # Compute cosine distance loss
        return compute_cosine_distance_loss(transformed, L_f32).item()
    
    # Smart rank search: test only promising candidates
    rank_candidates = [initial_rank]
    rank_candidates = sorted(list(set(rank_candidates)))
    
    print(f"Testing rank candidates: {rank_candidates}")
    
    best_rank = initial_rank
    best_alpha = 0.1
    best_loss = float('inf')
    
    for rank in rank_candidates:
        for alpha in alpha_range:
            loss = evaluate_rank_performance(rank, alpha)
            
            print(f"Current rank: {rank}, alpha: {alpha}, loss: {loss}")
            if loss < best_loss:
                best_loss = loss
                best_rank = rank
                best_alpha = alpha
    
    print(f"Optimal rank: {best_rank}, Optimal alpha: {best_alpha:.3f}, Best loss: {best_loss:.6f}")
    
    # Return optimal low-rank factors (convert back to original dtype)
    U_optimal = U[:, :best_rank].to(original_dtype)      # (14336, rank)
    S_optimal = S[:best_rank].to(original_dtype)         # (rank,)
    Vt_optimal = Vt[:best_rank, :].to(original_dtype)    # (rank, 4096)
    
    print(f"Final factor shapes: U: {U_optimal.shape}, S: {S_optimal.shape}, Vt: {Vt_optimal.shape}")
    
    # Clean up
    del M_f32, L_f32, M_sample, L_sample
    torch.cuda.empty_cache()
    
    return best_rank, U_optimal, S_optimal, Vt_optimal, best_alpha



# enhanced_adam_method 함수 수정 부분만
def enhanced_adam_method(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor = None,
    loss: str = "cosine",
    max_rank: int = 256,
    variance_threshold: float = 0.95,
    lr: float = 1e-3,
    max_epochs: int = 0,
    sample_ratio: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enhanced Adam optimization returning separate U, S, V factors for two-stage implementation
    """
    original_device = a1.device
    original_dtype = a1.dtype
    
    print(f"Enhanced Adam Method - Input shapes: a1={a1.shape}, a2={a2.shape}, dtype={original_dtype}")
    print(f"Using sample ratio: {sample_ratio*100:.1f}% for rank selection")
    
    # Find optimal rank using sampled data for efficiency
    optimal_rank, U_opt, S_opt, Vt_opt, optimal_alpha = find_optimal_rank_with_performance(
        a1, a1, a2, variance_threshold, max_rank, sample_ratio=sample_ratio
    )
    
    print(f"Using rank {optimal_rank} out of {a1.shape[1]} (compression: {optimal_rank/a1.shape[1]*100:.1f}%)")
    
    # ===== CRITICAL FIX: Correct factor interpretation =====
    # SVD gives us: T (14336×4096) = U (14336×rank) @ diag(S) @ Vt (rank×4096)
    # For TwoStageMLP: input @ W1 @ W2 where W1: (14336×rank), W2: (rank×4096)
    # So: W1 = U, W2 = diag(S) @ Vt
    
    print(f"SVD factors - U: {U_opt.shape}, S: {S_opt.shape}, Vt: {Vt_opt.shape}")
    
    # Initialize low-rank factors with proper interpretation
    U = nn.Parameter(U_opt.clone().detach().requires_grad_(True))  # (14336, rank)
    S = nn.Parameter(S_opt.clone().detach().requires_grad_(True))  # (rank,)
    Vt = nn.Parameter(Vt_opt.clone().detach().requires_grad_(True))  # (rank, 4096)
    alpha = nn.Parameter(torch.tensor(optimal_alpha, device=original_device, dtype=original_dtype).requires_grad_(True))
    
    # Optimizer with higher learning rate
    optimizer = torch.optim.Adam([U, S, Vt, alpha], lr=lr, weight_decay=1e-5)
    
    # Loss function
    def cosine_loss_batch(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XA_norm = XA / (XA.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()
    
    # Training with larger batch processing for efficiency
    batch_size = 4096
    num_samples = a1.shape[0]
    
    print(f"Training with batch size {batch_size} for {max_epochs} epochs")
    
    with tqdm(range(max_epochs), desc="Enhanced Optimization") as pbar:
        for epoch in pbar:
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle indices for better training
            indices = torch.randperm(num_samples)
            
            # Batch processing
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_indices = indices[i:end_idx]
                
                # Get batch (keep in original dtype)
                batch_a1 = a1[batch_indices]
                batch_a2 = a2[batch_indices]
                batch_a3 = a3[batch_indices] if a3 is not None else None
                
                optimizer.zero_grad()
                
                # Reconstruct transformation matrix from low-rank factors
                # T = U @ diag(S) @ Vt  (14336×4096)
                T_reconstructed = U @ torch.diag(S) @ Vt
                
                # Apply transformation with residual connection
                XA = batch_a1 @ T_reconstructed + alpha * batch_a1
                if batch_a3 is not None:
                    XA += batch_a3
                
                # Compute loss
                if loss == "cosine":
                    loss_val = cosine_loss_batch(XA, batch_a2)
                elif loss == "mse":
                    loss_val = nn.MSELoss()(XA, batch_a2)
                else:
                    raise ValueError(f"Unsupported loss: {loss}")
                
                loss_val.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([U, S, Vt, alpha], max_norm=1.0)
                
                optimizer.step()
                
                # Clamp alpha to reasonable range
                with torch.no_grad():
                    alpha.data = torch.clamp(alpha.data, 0.0, 0.3)
                    # Ensure S stays positive
                    S.data = torch.clamp(S.data, min=1e-8)
                
                epoch_loss += loss_val.item()
                num_batches += 1
                
                # Clear batch from memory
                del batch_a1, batch_a2, XA
                if batch_a3 is not None:
                    del batch_a3
            
            avg_loss = epoch_loss / num_batches
            pbar.set_postfix({
                f'{loss} Loss': f'{avg_loss:.6f}',
                'Rank': optimal_rank,
                'Alpha': f'{alpha.item():.3f}',
                'S_range': f'[{S.min().item():.2e}, {S.max().item():.2e}]'
            })
            
            # Early stopping if loss is very low
            if avg_loss < 1e-6:
                print(f"Early stopping at epoch {epoch+1} due to low loss")
                break
                
            # Clear cache every epoch
            torch.cuda.empty_cache()
    
    # Return separate factors for two-stage implementation
    with torch.no_grad():
        final_U = U.clone().detach()
        final_S = S.clone().detach()
        final_Vt = Vt.clone().detach()  # Keep as Vt, not V
        final_alpha = alpha.item()
        
        print(f"Final residual weight: {final_alpha:.4f}")
        print(f"Final rank: {optimal_rank}")
        print(f"Compression ratio: {optimal_rank}/{a1.shape[1]} = {optimal_rank/a1.shape[1]*100:.1f}%")
        
        # Verify reconstruction
        reconstructed_T = final_U @ torch.diag(final_S) @ final_Vt
        reconstruction_rank = torch.linalg.matrix_rank(reconstructed_T.float()).item()
        print(f"Reconstructed matrix rank: {reconstruction_rank}")
        print(f"Final factor shapes: U: {final_U.shape}, S: {final_S.shape}, Vt: {final_Vt.shape}")
    
    return final_U.cpu().to(torch.float64), final_S.cpu().to(torch.float64), final_Vt.cpu().to(torch.float64)


# TwoStageMLP 클래스 완전 수정
class TwoStageMLP(nn.Module):
    """
    Two-stage MLP to replace the original down_proj with low-rank factorization
    Mathematically: input @ U @ diag(S) @ Vt = input @ first_proj @ second_proj
    """
    def __init__(self, input_size: int, output_size: int, rank: int, 
                 U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, dtype=torch.bfloat16):
        super().__init__()
        
        print(f"TwoStageMLP initialization - Input factors:")
        print(f"  U shape: {U.shape} (should be {input_size} x {rank})")
        print(f"  S shape: {S.shape} (should be {rank})")
        print(f"  Vt shape: {Vt.shape} (should be {rank} x {output_size})")
        
        # Verify shapes
        assert U.shape == (input_size, rank), f"U shape mismatch: {U.shape} vs ({input_size}, {rank})"
        assert S.shape == (rank,), f"S shape mismatch: {S.shape} vs ({rank},)"
        assert Vt.shape == (rank, output_size), f"Vt shape mismatch: {Vt.shape} vs ({rank}, {output_size})"
        
        # First stage: input_size -> rank
        self.first_proj = nn.Linear(input_size, rank, bias=False, dtype=dtype)
        
        # Second stage: rank -> output_size  
        self.second_proj = nn.Linear(rank, output_size, bias=False, dtype=dtype)
        
        # Initialize weights from factorization
        # Goal: input @ U @ diag(S) @ Vt
        # Split into: input @ first_proj @ second_proj
        # Where: first_proj = U.T (for nn.Linear), second_proj = (diag(S) @ Vt).T (for nn.Linear)
        
        with torch.no_grad():
            # First projection: input @ U -> intermediate
            # nn.Linear applies: input @ weight.T, so weight should be U.T
            self.first_proj.weight.data = U.T.to(dtype)  # (rank, input_size)
            
            # Second projection: intermediate @ (diag(S) @ Vt) -> output  
            # nn.Linear applies: intermediate @ weight.T, so weight should be (diag(S) @ Vt).T
            second_weight = (torch.diag(S.to(dtype)) @ Vt.to(dtype)).T  # (output_size, rank)
            self.second_proj.weight.data = second_weight
            
        print(f"TwoStageMLP initialized:")
        print(f"  First proj weight: {self.first_proj.weight.shape} (input {input_size} -> rank {rank})")
        print(f"  Second proj weight: {self.second_proj.weight.shape} (rank {rank} -> output {output_size})")
        print(f"  Total parameters: {self.first_proj.weight.numel() + self.second_proj.weight.numel()}")
        
        # Store for debugging
        self.debug_rank = rank
        self.debug_U_norm = U.norm().item()
        self.debug_S_sum = S.sum().item() 
        self.debug_Vt_norm = Vt.norm().item()
        
    def forward(self, x):
        # Two-stage transformation: x @ U @ diag(S) @ Vt
        x = self.first_proj(x)  # x @ U
        x = self.second_proj(x)  # (x @ U) @ (diag(S) @ Vt)
        return x


def two_stage_enhanced_cosine_dist(
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
    loss: str = "cosine",
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    max_rank: int = 256,
    variance_threshold: float = 0.95,
    **kwargs
) -> str:
    """
    Enhanced cosine distance method with two-stage MLP implementation for true low-rank inference
    """
    print(f"=== Two-Stage Enhanced Cosine Distance Method ===")
    print(f"Model: {model_path}")
    print(f"Layers to replace: {start_id} to {end_id} (skipping {layers_to_skip} layers)")
    print(f"Max rank: {max_rank}, Variance threshold: {variance_threshold}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size  # 4096
    intermediate_size = model.config.intermediate_size  # 14336
    
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
    
    # ========== 수정된 활성화 수집: down_proj 입력/출력 직접 캡처 ==========
    target_layer_idx = start_id - num_layer - 1  # 23번째 layer
    
    # down_proj 입력을 수집하기 위한 hook
    def save_down_proj_input(name):
        def hook(module, input, output):
            mlp_activations[f'{name}_input'] = input[0].detach()  # down_proj 입력
            mlp_activations[f'{name}_output'] = output.detach()   # down_proj 출력
        return hook
    
    hooks = []
    if 'falcon' in model_path.lower():
        down_proj = model.transformer.h[target_layer_idx].mlp.dense_4h_to_h
    else:
        down_proj = model.model.layers[target_layer_idx].mlp.down_proj
    
    hooks.append(down_proj.register_forward_hook(save_down_proj_input('target_down')))

    mlp_activations = {}
    
    # 올바른 차원으로 활성화 텐서 생성
    a1 = torch.empty((dataset_size * max_length, intermediate_size), dtype=torch.bfloat16, device='cpu')  # down_proj 입력 (14336)
    a2 = torch.empty((dataset_size * max_length, hidden_size), dtype=torch.bfloat16, device='cpu')        # 최종 목표 출력 (4096)
    
    cnt = 0
    for batch in tqdm(dataloader, desc="Gathering down_proj I/O Activations"):
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
        
        # down_proj의 실제 입력 수집 (intermediate_size = 14336)
        down_proj_input = mlp_activations['target_down_input'].view(-1, intermediate_size).to(torch.bfloat16)
        
        # 목표 출력: 원래 ReplaceMe와 동일한 방식으로 계산
        hidden_states = outputs.hidden_states[1:]
        hidden_states_i = hidden_states[start_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        hidden_states_n = hidden_states[end_id - num_layer - 1].view(-1, hidden_size).to(torch.bfloat16)
        
        # 현재 down_proj 출력
        current_down_output = mlp_activations['target_down_output'].view(-1, hidden_size).to(torch.bfloat16)
        
        a1_batch = down_proj_input                                        # (N, 14336) - down_proj 입력
        a2_batch = hidden_states_n + current_down_output - hidden_states_i  # (N, 4096) - 목표 출력
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch.cpu()
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch.cpu()
        
        cnt += a1_batch.shape[0]
        
        del down_proj_input, hidden_states_i, hidden_states_n, current_down_output
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    a1 = a1[:cnt]  # (N, 14336)
    a2 = a2[:cnt]  # (N, 4096)
    
    print(f"Collected {cnt} activation samples")
    print(f"down_proj input activations shape: {a1.shape} (should be N x {intermediate_size})")
    print(f"Target output activations shape: {a2.shape} (should be N x {hidden_size})")
    
    # Apply enhanced optimization to get U, S, V factors
    U_factor, S_factor, Vt_factor = enhanced_adam_method(  # V_factor -> Vt_factor로 변경
        a1, a2, 
        loss=loss, 
        max_rank=max_rank,
        variance_threshold=variance_threshold
    )
    
    print(f"Low-rank factors obtained:")
    print(f"  U: {U_factor.shape}")
    print(f"  S: {S_factor.shape}") 
    print(f"  Vt: {Vt_factor.shape}")  # V -> Vt로 변경
    
    # Clean up activations
    del a1, a2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clean up model and reload for transformation
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Get original down_proj info
    target_layer_idx = start_id - num_layer - 1
    original_down_proj = model.model.layers[target_layer_idx].mlp.down_proj
    
    print(f"Original down_proj shape: {original_down_proj.weight.shape}")
    print(f"Original parameters: {original_down_proj.weight.numel()}")
    
    # Create two-stage MLP replacement
    input_size = original_down_proj.weight.shape[1]  # Usually intermediate_size (e.g., 14336)
    output_size = original_down_proj.weight.shape[0]  # Usually hidden_size (e.g., 4096)
    rank = S_factor.shape[0]
    
    # ===== CRITICAL FIX: Pass Vt instead of V =====
    two_stage_mlp = TwoStageMLP(
        input_size=input_size,
        output_size=output_size, 
        rank=rank,
        U=U_factor,
        S=S_factor,
        Vt=Vt_factor,  # V=V_factor -> Vt=Vt_factor로 변경
        dtype=torch.bfloat16
    )
    
    # Replace the down_proj with two-stage MLP
    model.model.layers[target_layer_idx].mlp.down_proj = two_stage_mlp
    
    print(f"Replaced down_proj with TwoStageMLP")
    print(f"Memory reduction: {original_down_proj.weight.numel()} -> {two_stage_mlp.first_proj.weight.numel() + two_stage_mlp.second_proj.weight.numel()}")
    memory_reduction = 1 - (two_stage_mlp.first_proj.weight.numel() + two_stage_mlp.second_proj.weight.numel()) / original_down_proj.weight.numel()
    print(f"Memory reduction percentage: {memory_reduction*100:.1f}%")
    
    # Set save path
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    # Save model
    output_path = f"{save_path}_TwoStageEnhanced_{loss}_rank{rank}"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    if save_transform_only:
        # ===== CRITICAL FIX: Save Vt instead of V =====
        torch.save({
            'U': U_factor,
            'S': S_factor, 
            'V': Vt_factor,  # 주의: 저장할 때는 'V' 키로 저장하지만 실제로는 Vt 값
            'rank': rank
        }, f"{output_path}_transform")
    
    print(f"Two-stage enhanced model saved to: {output_path}")
    
    # Final cleanup
    # Final cleanup - 변수명 수정
    del model, U_factor, S_factor, Vt_factor  # V_factor -> Vt_factor로 변경
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path