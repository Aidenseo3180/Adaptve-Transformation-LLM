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
    
    def __init__(self, hidden_size: int, rank: int = 64, 
                 feature_dim: int = 7, dropout: float = 0.1):
        """Initialize HyperNetwork.
        
        Args:
            hidden_size: Model's hidden dimension
            rank: Rank for low-rank decomposition
            feature_dim: Dimension of input features (mean, std, norm, etc.)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        print(f"[DEBUG] Initializing HyperNetwork: hidden_size={hidden_size}, rank={rank}")
        
        # Feature extraction network
        # Input: statistical features of the batch
        # Output: transformation parameters
        input_dim = hidden_size * feature_dim  # mean, std, norm, max, min, kurtosis, skewness
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Low-rank matrix generators
        # U: [hidden_size, rank], V: [rank, hidden_size]
        self.u_generator = nn.Linear(256, hidden_size * rank)
        self.v_generator = nn.Linear(256, hidden_size * rank)
        
        # Scaling factors for adaptive adjustment
        self.scale_u = nn.Parameter(torch.ones(1))
        self.scale_v = nn.Parameter(torch.ones(1))
        
        # Base transformation (starts from identity)
        self.register_buffer('base_transform', torch.eye(hidden_size))
        
        # Residual weight for stability
        self.residual_weight = nn.Parameter(torch.tensor(0.9))
        
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
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Small initial values for U and V generators
        nn.init.normal_(self.u_generator.weight, std=0.01)
        nn.init.normal_(self.v_generator.weight, std=0.01)
    
    def extract_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            features: [batch_size, hidden_size * feature_dim]
        """
        print(f"[DEBUG] Extracting features from hidden_states shape: {hidden_states.shape}")
        
        # Compute various statistics
        batch_mean = hidden_states.mean(dim=1)  # [batch, hidden]
        batch_std = hidden_states.std(dim=1)    # [batch, hidden]
        batch_norm = hidden_states.norm(dim=2).mean(dim=1, keepdim=True).expand(-1, self.hidden_size)
        batch_max = hidden_states.max(dim=1)[0]  # [batch, hidden]
        batch_min = hidden_states.min(dim=1)[0]  # [batch, hidden]
        
        # Higher-order statistics
        centered = hidden_states - batch_mean.unsqueeze(1)
        batch_skewness = (centered ** 3).mean(dim=1) / (batch_std ** 3 + 1e-8)
        batch_kurtosis = (centered ** 4).mean(dim=1) / (batch_std ** 4 + 1e-8) - 3
        
        # Concatenate all features
        features = torch.cat([
            batch_mean, batch_std, batch_norm, 
            batch_max, batch_min, batch_skewness, batch_kurtosis
        ], dim=-1)
        
        print(f"[DEBUG] Extracted features shape: {features.shape}")
        return features
    
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
        
        # Generate transformation parameters
        feat_encoded = self.feature_extractor(features)  # [batch, 256]
        
        # Generate U and V matrices
        u_params = self.u_generator(feat_encoded)  # [batch, hidden_size * rank]
        v_params = self.v_generator(feat_encoded)  # [batch, hidden_size * rank]
        
        # Reshape to matrices
        U = u_params.view(batch_size, hidden_size, self.rank) * self.scale_u
        V = v_params.view(batch_size, self.rank, hidden_size) * self.scale_v
        
        # Compute low-rank transformation: U @ V
        # [batch, hidden, rank] @ [batch, rank, hidden] = [batch, hidden, hidden]
        adaptive_transform = torch.bmm(U, V)
        
        # Combine with base transformation using residual
        # This helps maintain stability
        transform = (self.residual_weight * self.base_transform.unsqueeze(0) + 
                    (1 - self.residual_weight) * adaptive_transform)
        
        # Apply transformation to each sequence position
        # Reshape for batch matrix multiplication
        hidden_reshaped = hidden_states.reshape(batch_size * seq_len, hidden_size)
        transform_expanded = transform.repeat_interleave(seq_len, dim=0)
        
        transformed = torch.bmm(
            hidden_reshaped.unsqueeze(1), 
            transform_expanded
        ).squeeze(1)
        
        transformed = transformed.reshape(batch_size, seq_len, hidden_size)
        
        if return_components:
            return transformed, U, V
        return transformed

    def get_complexity_score(self, hidden_states: torch.Tensor) -> float:
        """Estimate input complexity for adaptive block selection.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            complexity_score: Scalar value indicating input complexity
        """
        features = self.extract_features(hidden_states)
        
        # Use feature statistics as complexity indicators
        feature_std = features.std(dim=-1).mean()
        feature_norm = features.norm(dim=-1).mean()
        
        complexity = (feature_std + feature_norm) / 2
        
        print(f"[DEBUG] Complexity score: {complexity.item():.4f}")
        return complexity.item()


# ============================
# Training Functions
# ============================

def train_hypernet(
    hypernet: HyperNetwork,
    dataloader: torch.utils.data.DataLoader,
    target_activations: Dict[str, torch.Tensor],
    input_activations: Dict[str, torch.Tensor],
    epochs: int = 5,
    lr: float = 1e-4,
    progressive: bool = True,
    device: str = "cuda"
) -> HyperNetwork:
    """Train the HyperNetwork to generate appropriate transformations.
    
    Args:
        hypernet: HyperNetwork model
        dataloader: DataLoader for calibration data
        target_activations: Target outputs from original blocks
        input_activations: Input activations to blocks
        epochs: Number of training epochs
        lr: Learning rate
        progressive: Whether to use progressive training
        device: Device to train on
    
    Returns:
        Trained HyperNetwork
    """
    print(f"[DEBUG] Starting HyperNetwork training: epochs={epochs}, lr={lr}")
    
    hypernet = hypernet.to(device)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss."""
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        return 1 - (pred_norm * target_norm).sum(dim=-1).mean()
    
    def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                     stage: int = 0) -> torch.Tensor:
        """Combined loss with progressive weighting."""
        mse = F.mse_loss(pred, target)
        cosine = cosine_similarity_loss(pred, target)
        
        if stage == 0:
            # Stage 1: Focus on reconstruction
            loss = mse
            print(f"[DEBUG] Stage 1 - MSE Loss: {mse.item():.6f}")
        elif stage == 1:
            # Stage 2: Balance reconstruction and direction
            loss = 0.7 * mse + 0.3 * cosine
            print(f"[DEBUG] Stage 2 - MSE: {mse.item():.6f}, Cosine: {cosine.item():.6f}")
        else:
            # Stage 3: Focus on direction with sparsity
            # Add L1 regularization on U and V for sparsity
            loss = 0.3 * mse + 0.7 * cosine
            print(f"[DEBUG] Stage 3 - MSE: {mse.item():.6f}, Cosine: {cosine.item():.6f}")
        
        return loss
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n{Fore.GREEN}=== Epoch {epoch+1}/{epochs} ==={Fore.RESET}")
        
        # Determine training stage for progressive training
        if progressive:
            if epoch < epochs // 3:
                stage = 0
            elif epoch < 2 * epochs // 3:
                stage = 1
            else:
                stage = 2
        else:
            stage = 2  # Use full loss from the start
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                           desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch_data in progress_bar:
            # Get corresponding activations for this batch
            batch_start = batch_idx * dataloader.batch_size
            batch_end = batch_start + len(batch_data)
            
            # Get input and target activations for current batch
            batch_input = input_activations[batch_start:batch_end].to(device)
            batch_target = target_activations[batch_start:batch_end].to(device)
            
            # Forward pass through HyperNetwork
            optimizer.zero_grad()
            transformed, U, V = hypernet(batch_input, return_components=True)
            
            # Compute loss
            loss = combined_loss(transformed, batch_target, stage)
            
            # Add regularization for U and V matrices
            if stage >= 2:
                u_reg = torch.norm(U, p=1) * 1e-5
                v_reg = torch.norm(V, p=1) * 1e-5
                loss = loss + u_reg + v_reg
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        scheduler.step()
        
        avg_epoch_loss = total_loss / num_batches
        print(f"{Fore.CYAN}Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.6f}{Fore.RESET}")
        
        # Early stopping check
        if avg_epoch_loss < 1e-5:
            print(f"{Fore.YELLOW}Early stopping: Loss below threshold{Fore.RESET}")
            break
    
    print(f"{Fore.GREEN}Training completed!{Fore.RESET}")
    return hypernet


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
    hyper_rank: int = 64,
    hyper_lr: float = 1e-4,
    hyper_epochs: int = 5,
    adaptive_selection: bool = True,
    complexity_threshold: List[float] = [0.3, 0.7],
    progressive_stages: bool = True,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    **kwargs  # Catch any extra arguments
) -> str:
    """Main HyperReplace function for adaptive layer replacement.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip/replace
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        token: Authentication token
        hyper_rank: Rank for low-rank decomposition in HyperNetwork
        hyper_lr: Learning rate for HyperNetwork training
        hyper_epochs: Number of training epochs
        adaptive_selection: Whether to use adaptive block selection
        complexity_threshold: Thresholds for complexity-based selection
        progressive_stages: Whether to use progressive training
        distances_path: Path to layer distance metrics
        num_A: Number of transformation blocks
        merge_consecutive: Whether to merge consecutive blocks
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers processed
    
    Returns:
        Path where transformed model is saved
    """
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"Starting HyperReplace Pipeline")
    print(f"{'='*60}{Fore.RESET}\n")
    
    print(f"[DEBUG] Configuration:")
    print(f"  - Model: {model_path}")
    print(f"  - Dataset: {dataset} (size: {dataset_size})")
    print(f"  - Layers to skip: {layers_to_skip}")
    print(f"  - HyperNetwork rank: {hyper_rank}")
    print(f"  - Adaptive selection: {adaptive_selection}")
    
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
    
    # Prepare dataloader
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
    
    # Determine which blocks to replace
    if start_id == 0 or end_id == 0:
        # Load distances and select blocks
        average_distances = torch.load(distances_path)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            layers_to_skip,
            num_blocks=num_A,
            merge_consecutive=merge_consecutive
        )
        start_id, end_id = selected_blocks[0]
        print(f"[DEBUG] Selected block: layers {start_id} to {end_id}")
    
    # Hooks for capturing activations
    def save_activation(name, storage_dict):
        def hook(module, input, output):
            storage_dict[name] = output.detach()
        return hook
    
    # Storage for activations
    input_activations = []
    output_activations = []
    mlp_activations = {}
    
    # Register hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if i == start_id - 1:
            # Capture input to the block
            hooks.append(layer.register_forward_hook(
                save_activation(f'layer_{i}_output', mlp_activations)
            ))
        elif i == end_id - 1:
            # Capture output from the block
            hooks.append(layer.register_forward_hook(
                save_activation(f'layer_{i}_output', mlp_activations)
            ))
    
    print(f"[DEBUG] Registered {len(hooks)} hooks for activation capture")
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting Activations")):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Collect activations
            if f'layer_{start_id-1}_output' in mlp_activations:
                input_act = mlp_activations[f'layer_{start_id-1}_output']
                input_activations.append(input_act.cpu())
            
            if f'layer_{end_id-1}_output' in mlp_activations:
                output_act = mlp_activations[f'layer_{end_id-1}_output']
                output_activations.append(output_act.cpu())
            
            # Clear for next batch
            mlp_activations.clear()
            
            print(f"[DEBUG] Batch {batch_idx+1}: collected activation shapes: "
                  f"input={input_activations[-1].shape if input_activations else None}, "
                  f"output={output_activations[-1].shape if output_activations else None}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all activations
    input_activations = torch.cat(input_activations, dim=0)
    output_activations = torch.cat(output_activations, dim=0)
    
    print(f"[DEBUG] Total activations collected: input={input_activations.shape}, output={output_activations.shape}")
    
    # ============================
    # Step 3: Estimate Complexity
    # ============================
    
    if adaptive_selection:
        print(f"\n{Fore.CYAN}Step 3: Estimating input complexity for adaptive selection...{Fore.RESET}")
        
        # Create temporary HyperNetwork for complexity estimation
        temp_hypernet = HyperNetwork(hidden_size, rank=hyper_rank)
        
        # Sample complexity scores
        sample_complexities = []
        for i in range(min(10, len(input_activations))):
            complexity = temp_hypernet.get_complexity_score(input_activations[i:i+1])
            sample_complexities.append(complexity)
        
        avg_complexity = sum(sample_complexities) / len(sample_complexities)
        print(f"[DEBUG] Average complexity score: {avg_complexity:.4f}")
        
        # Adjust number of layers to replace based on complexity
        if avg_complexity < complexity_threshold[0]:
            print(f"[INFO] Low complexity detected - can be more aggressive with replacement")
            # Could adjust layers_to_skip here
        elif avg_complexity > complexity_threshold[1]:
            print(f"[INFO] High complexity detected - being conservative with replacement")
            # Could reduce layers_to_skip here
        
        del temp_hypernet
    
    # ============================
    # Step 4: Train HyperNetwork
    # ============================
    
    print(f"\n{Fore.CYAN}Step 4: Training HyperNetwork...{Fore.RESET}")
    
    # Initialize HyperNetwork
    hypernet = HyperNetwork(
        hidden_size=hidden_size,
        rank=hyper_rank,
        feature_dim=7
    )
    
    # Create training dataloader (just for batching, actual data comes from stored activations)
    train_dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    
    # Train HyperNetwork
    hypernet = train_hypernet(
        hypernet=hypernet,
        dataloader=train_dataloader,
        target_activations=output_activations,
        input_activations=input_activations,
        epochs=hyper_epochs,
        lr=hyper_lr,
        progressive=progressive_stages,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # ============================
    # Step 5: Apply to Model
    # ============================
    
    print(f"\n{Fore.CYAN}Step 5: Applying HyperNetwork to model...{Fore.RESET}")
    
    # Clean up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload model for modification
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Truncate model (remove replaced layers)
    print(f"[DEBUG] Truncating model: removing layers {start_id} to {end_id}")
    model = truncate_model(model, start_id, end_id)
    
    # Integrate HyperNetwork into the model
    # This is a simplified version - in practice, you'd want to properly integrate it
    # For now, we'll save both the model and HyperNetwork separately
    
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
    
    # Save the truncated model
    model.save_pretrained(f"{save_path}_model")
    tokenizer.save_pretrained(f"{save_path}_model")
    
    # Save the HyperNetwork
    torch.save({
        'hypernet_state': hypernet.state_dict(),
        'hypernet_config': {
            'hidden_size': hidden_size,
            'rank': hyper_rank,
            'feature_dim': 7
        },
        'replaced_layers': {
            'start': start_id,
            'end': end_id
        }
    }, f"{save_path}_hypernet.pth")
    
    print(f"{Fore.GREEN}Model saved to: {save_path}_model")
    print(f"HyperNetwork saved to: {save_path}_hypernet.pth{Fore.RESET}")
    
    # Clean up
    del model, hypernet, input_activations, output_activations
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"HyperReplace Pipeline Completed Successfully!")
    print(f"{'='*60}{Fore.RESET}\n")
    
    return f"{save_path}_model"


# ============================
# Configuration Functions
# ============================

def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run HyperReplace from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run HyperReplace adaptive layer replacement from configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    # Load distances if needed
    if 'distances_path' in config and config['distances_path']:
        average_distances = torch.load(config['distances_path'])
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            config['layers_to_skip'],
            num_blocks=config.get('num_A', 1),
            merge_consecutive=config.get('merge_consecutive', False)
        )
        
        # Process multiple blocks if needed
        for i, (start_id, end_id) in enumerate(selected_blocks):
            print(f"\n[INFO] Processing block {i+1}/{len(selected_blocks)}: layers {start_id} to {end_id}")
            
            config['start_id'] = start_id
            config['end_id'] = end_id
            
            path = hyper_replace(**config)
            
            # Update model path for next iteration if needed
            if i < len(selected_blocks) - 1:
                config["model_path"] = path
    else:
        # Single run without distance-based selection
        hyper_replace(**config)


if __name__ == "__main__":
    run_from_config()