"""HyperReplace: Adaptive Linear Transformation Networks for Layer Replacement

This module implements input-adaptive layer replacement using HyperNetworks
that generate conditional linear transformations based on input characteristics.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from torch.cuda.amp import autocast, GradScaler

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()

# ============================
# HyperNetwork Architecture
# ============================

class HyperNetwork(nn.Module):
    """HyperNetwork that generates input-adaptive linear transformations.
    
    Takes input statistics and generates low-rank transformation matrices
    that adapt to the complexity and characteristics of the input.
    """
    
    def __init__(self, hidden_size: int, rank: int = 16, 
                 feature_dim: int = 3, dropout: float = 0.1):
        """Initialize HyperNetwork.
        
        Args:
            hidden_size: Model's hidden dimension
            rank: Rank for low-rank decomposition
            feature_dim: Dimension of input features (mean, std, norm)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        print(f"[DEBUG] Initializing HyperNetwork: hidden_size={hidden_size}, rank={rank}")
        
        # Feature extraction network
        input_dim = hidden_size * feature_dim  # mean, std, norm only
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Low-rank matrix generators
        self.u_generator = nn.Linear(256, hidden_size * rank)
        self.v_generator = nn.Linear(256, hidden_size * rank)
        
        # Scaling factors for adaptive adjustment
        self.scale_u = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_v = nn.Parameter(torch.ones(1) * 0.1)
        
        # Base transformation (starts from identity)
        self.register_buffer('base_transform', torch.eye(hidden_size))
        
        # Higher residual weight for stability (0.95 instead of 0.9)
        self.residual_weight = nn.Parameter(torch.tensor(0.95))
        
        # Complexity threshold for selective adaptation
        self.complexity_threshold = 0.5
        
        self.hidden_size = hidden_size
        self.rank = rank
        self.feature_dim = feature_dim
        
        # Initialize weights carefully
        self._initialize_weights()
        
        print(f"[DEBUG] HyperNetwork initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _initialize_weights(self):
        """Careful initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Very small initial values for U and V generators
        nn.init.normal_(self.u_generator.weight, std=0.001)
        nn.init.normal_(self.v_generator.weight, std=0.001)
    
    def extract_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            features: [batch_size, hidden_size * feature_dim]
        """
        # Use only 3 basic statistics to reduce memory
        batch_mean = hidden_states.mean(dim=1)
        batch_std = hidden_states.std(dim=1)
        batch_norm = hidden_states.norm(dim=2).mean(dim=1, keepdim=True).expand(-1, self.hidden_size)
        
        features = torch.cat([batch_mean, batch_std, batch_norm], dim=-1)
        return features
    
    def get_complexity_score(self, hidden_states: torch.Tensor) -> float:
        """Estimate input complexity for adaptive block selection."""
        features = self.extract_features(hidden_states)
        
        # Use feature statistics as complexity indicators
        feature_std = features.std(dim=-1).mean()
        feature_norm = features.norm(dim=-1).mean()
        
        complexity = (feature_std + feature_norm) / 2
        return complexity.item()
    
    def forward(self, hidden_states: torch.Tensor, 
                return_components: bool = False) -> torch.Tensor:
        """Generate and apply adaptive transformation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            return_components: Whether to return U and V separately
        
        Returns:
            transformed: [batch_size, seq_len, hidden_size]
            or (transformed, U, V) if return_components=True
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Extract features
        features = self.extract_features(hidden_states)
        
        # Check complexity for selective adaptation
        complexity = features.std(dim=-1).mean()
        
        if complexity < self.complexity_threshold:
            # Low complexity: use mostly base transformation
            transform = self.base_transform.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Apply transformation efficiently using einsum
            transformed = torch.einsum('bsh,bhd->bsd', hidden_states, transform)
            
            if return_components:
                # Return dummy U and V for compatibility
                U = torch.zeros(batch_size, hidden_size, self.rank, device=hidden_states.device)
                V = torch.zeros(batch_size, self.rank, hidden_size, device=hidden_states.device)
                return transformed, U, V
            return transformed
        
        # High complexity: use adaptive transformation
        feat_encoded = self.feature_extractor(features)
        
        # Generate U and V matrices
        u_params = self.u_generator(feat_encoded)
        v_params = self.v_generator(feat_encoded)
        
        # Reshape to matrices with smaller scaling
        U = u_params.view(batch_size, hidden_size, self.rank) * self.scale_u
        V = v_params.view(batch_size, self.rank, hidden_size) * self.scale_v
        
        # Compute low-rank transformation: U @ V
        adaptive_transform = torch.bmm(U, V)
        
        # Combine with base transformation using high residual weight
        transform = (self.residual_weight * self.base_transform.unsqueeze(0) + 
                    (1 - self.residual_weight) * adaptive_transform)
        
        # Apply transformation efficiently using einsum (no repeat_interleave!)
        transformed = torch.einsum('bsh,bhd->bsd', hidden_states, transform)
        
        if return_components:
            return transformed, U, V
        return transformed


# ============================
# Loss Functions
# ============================

def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss."""
    pred_norm = F.normalize(pred.reshape(-1, pred.shape[-1]), p=2, dim=-1)
    target_norm = F.normalize(target.reshape(-1, target.shape[-1]), p=2, dim=-1)
    return 1 - (pred_norm * target_norm).sum(dim=-1).mean()

def kl_divergence_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence loss for better language modeling preservation."""
    # Treat hidden states as distributions
    pred_log_probs = F.log_softmax(pred.view(-1, pred.size(-1)), dim=-1)
    target_probs = F.softmax(target.view(-1, target.size(-1)), dim=-1)
    return F.kl_div(pred_log_probs, target_probs, reduction='batchmean')

def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                 stage: int = 0, use_kl: bool = True) -> torch.Tensor:
    """Combined loss with progressive weighting."""
    mse = F.mse_loss(pred, target)
    cosine = cosine_similarity_loss(pred, target)
    
    if use_kl:
        kl = kl_divergence_loss(pred, target)
    else:
        kl = 0
    
    if stage == 0:
        # Stage 1: Focus on reconstruction
        loss = mse
    elif stage == 1:
        # Stage 2: Balance reconstruction and direction
        loss = 0.5 * mse + 0.3 * cosine + 0.2 * kl if use_kl else 0.7 * mse + 0.3 * cosine
    else:
        # Stage 3: Focus on direction with KL for language modeling
        loss = 0.2 * mse + 0.3 * cosine + 0.5 * kl if use_kl else 0.3 * mse + 0.7 * cosine
    
    return loss


# ============================
# Main HyperReplace Function
# ============================

def hyper_replace(
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
    # HyperReplace specific parameters
    hyper_rank: int = 16,
    hyper_lr: float = 5e-5,
    hyper_epochs: int = 20,
    adaptive_selection: bool = True,
    complexity_threshold: List[float] = [0.3, 0.7],
    progressive_stages: bool = True,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    use_kl_loss: bool = True,
    use_mixed_precision: bool = True,
    **kwargs
) -> str:
    """Main HyperReplace function for adaptive layer replacement."""
    
    # Type conversion to ensure correct types
    hyper_lr = float(hyper_lr) if isinstance(hyper_lr, str) else hyper_lr
    hyper_rank = int(hyper_rank) if isinstance(hyper_rank, str) else hyper_rank
    hyper_epochs = int(hyper_epochs) if isinstance(hyper_epochs, str) else hyper_epochs
    batch_size = int(batch_size) if isinstance(batch_size, str) else batch_size
    max_length = int(max_length) if isinstance(max_length, str) else max_length
    layers_to_skip = int(layers_to_skip) if isinstance(layers_to_skip, str) else layers_to_skip
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"Starting HyperReplace Pipeline")
    print(f"{'='*60}{Fore.RESET}\n")
    
    print(f"[DEBUG] Configuration:")
    print(f"  - Model: {model_path}")
    print(f"  - Dataset: {dataset} (size: {dataset_size})")
    print(f"  - Layers to skip: {layers_to_skip}")
    print(f"  - HyperNetwork rank: {hyper_rank}")
    print(f"  - Learning rate: {hyper_lr}")
    print(f"  - Epochs: {hyper_epochs}")
    print(f"  - Use KL loss: {use_kl_loss}")
    print(f"  - Use mixed precision: {use_mixed_precision}")
    
    # ============================
    # Step 1: Load Model and Data
    # ============================
    
    print(f"\n{Fore.CYAN}Step 1: Loading model and preparing data...{Fore.RESET}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("[DEBUG] Using 4-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    hidden_size = model.config.hidden_size
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print(f"[DEBUG] Model loaded: {model.config.num_hidden_layers} layers, hidden_size={hidden_size}")
    
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    print(f"[DEBUG] DataLoader prepared with {len(dataloader)} batches")
    
    # ============================
    # Step 2: Collect Activations
    # ============================
    
    print(f"\n{Fore.CYAN}Step 2: Collecting activations from target layers...{Fore.RESET}")
    
    if start_id == 0 or end_id == 0:
        average_distances = torch.load(distances_path)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            layers_to_skip,
            num_blocks=num_A,
            merge_consecutive=merge_consecutive
        )
        start_id, end_id = selected_blocks[0]
        print(f"[DEBUG] Selected block: layers {start_id} to {end_id}")
    
    collected_inputs = []
    collected_outputs = []
    
    print(f"[DEBUG] Collecting activations for layers {start_id-1} to {end_id-1}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting Activations")):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            
            if start_id > 0:
                input_act = hidden_states[start_id]
            else:
                input_act = hidden_states[1]
            
            output_act = hidden_states[end_id]
            
            if batch_idx == 0:
                print(f"[DEBUG] First batch shape: input={input_act.shape}, output={output_act.shape}")
            
            collected_inputs.append(input_act.cpu())
            collected_outputs.append(output_act.cpu())
    
    print(f"[DEBUG] Collected {len(collected_inputs)} batches")
    
    input_activations = torch.cat(collected_inputs, dim=0)
    output_activations = torch.cat(collected_outputs, dim=0)
    
    print(f"[DEBUG] Total activations shape: input={input_activations.shape}, output={output_activations.shape}")
    
    # ============================
    # Step 3: Estimate Complexity
    # ============================
    
    if adaptive_selection:
        print(f"\n{Fore.CYAN}Step 3: Estimating input complexity...{Fore.RESET}")
        
        temp_hypernet = HyperNetwork(hidden_size, rank=hyper_rank, feature_dim=3)
        
        sample_complexities = []
        num_samples = min(10, input_activations.shape[0])
        
        for i in range(num_samples):
            complexity = temp_hypernet.get_complexity_score(input_activations[i:i+1])
            sample_complexities.append(complexity)
        
        avg_complexity = sum(sample_complexities) / len(sample_complexities)
        print(f"[DEBUG] Average complexity score: {avg_complexity:.4f}")
        
        if avg_complexity < complexity_threshold[0]:
            print(f"[INFO] Low complexity detected - using conservative adaptation")
            temp_hypernet.complexity_threshold = 0.7
        elif avg_complexity > complexity_threshold[1]:
            print(f"[INFO] High complexity detected - using aggressive adaptation")
            temp_hypernet.complexity_threshold = 0.3
        
        del temp_hypernet
    
    # ============================
    # Step 4: Train HyperNetwork
    # ============================
    
    print(f"\n{Fore.CYAN}Step 4: Training HyperNetwork...{Fore.RESET}")
    
    hypernet = HyperNetwork(
        hidden_size=hidden_size,
        rank=hyper_rank,
        feature_dim=3
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hypernet = hypernet.to(device)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=hyper_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_epochs)
    
    # Mixed precision training setup
    if use_mixed_precision and device == "cuda":
        scaler = GradScaler()
        print("[DEBUG] Using mixed precision training")
    else:
        scaler = None
    
    # Create index dataloader for training
    num_samples = input_activations.shape[0]
    indices = torch.arange(num_samples)
    index_dataloader = torch.utils.data.DataLoader(
        indices,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    for epoch in range(hyper_epochs):
        print(f"\n{Fore.GREEN}=== Epoch {epoch+1}/{hyper_epochs} ==={Fore.RESET}")
        
        if progressive_stages:
            if epoch < hyper_epochs // 3:
                stage = 0
            elif epoch < 2 * hyper_epochs // 3:
                stage = 1
            else:
                stage = 2
        else:
            stage = 2
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(index_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_indices in progress_bar:
            batch_input = input_activations[batch_indices].to(device)
            batch_target = output_activations[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    transformed, U, V = hypernet(batch_input, return_components=True)
                    loss = combined_loss(transformed, batch_target, stage, use_kl=use_kl_loss)
                    
                    # Add regularization for U and V matrices
                    if stage >= 2:
                        u_reg = torch.norm(U, p=1) * 1e-6
                        v_reg = torch.norm(V, p=1) * 1e-6
                        loss = loss + u_reg + v_reg
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                transformed, U, V = hypernet(batch_input, return_components=True)
                loss = combined_loss(transformed, batch_target, stage, use_kl=use_kl_loss)
                
                if stage >= 2:
                    u_reg = torch.norm(U, p=1) * 1e-6
                    v_reg = torch.norm(V, p=1) * 1e-6
                    loss = loss + u_reg + v_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.5)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                'stage': stage
            })
        
        scheduler.step()
        
        avg_epoch_loss = total_loss / num_batches
        print(f"{Fore.CYAN}Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.6f}{Fore.RESET}")
        
        # Early stopping with higher threshold
        if avg_epoch_loss < 1e-4:
            print(f"{Fore.YELLOW}Early stopping: Loss below threshold{Fore.RESET}")
            break
    
    print(f"{Fore.GREEN}Training completed!{Fore.RESET}")
    
    # ============================
    # Step 5: Apply to Model
    # ============================
    
    print(f"\n{Fore.CYAN}Step 5: Applying HyperNetwork to model...{Fore.RESET}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    print(f"[DEBUG] Truncating model: removing layers {start_id} to {end_id}")
    model = truncate_model(model, start_id, end_id)
    
    # ============================
    # Step 6: Save Model
    # ============================
    
    print(f"\n{Fore.CYAN}Step 6: Saving transformed model...{Fore.RESET}")
    
    if save_path is None:
        if not os.path.exists('output_models'):
            os.makedirs('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}_HyperReplace_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}_model")
    tokenizer.save_pretrained(f"{save_path}_model")
    
    torch.save({
        'hypernet_state': hypernet.state_dict(),
        'hypernet_config': {
            'hidden_size': hidden_size,
            'rank': hyper_rank,
            'feature_dim': 3
        },
        'replaced_layers': {
            'start': start_id,
            'end': end_id
        }
    }, f"{save_path}_hypernet.pth")
    
    print(f"{Fore.GREEN}Model saved to: {save_path}_model")
    print(f"HyperNetwork saved to: {save_path}_hypernet.pth{Fore.RESET}")
    
    del model, hypernet, input_activations, output_activations
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"HyperReplace Pipeline Completed Successfully!")
    print(f"{'='*60}{Fore.RESET}\n")
    
    return f"{save_path}_model"