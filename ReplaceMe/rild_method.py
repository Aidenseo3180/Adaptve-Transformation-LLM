import argparse
import gc
import logging
import os
from typing import Optional
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, 
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class ResidualLowRankTransform(nn.Module):
    """Residual-Informed Low-Rank Decomposition Transform.
    
    T = I + U @ V^T where U, V are low-rank matrices.
    This preserves identity mapping while learning a low-rank perturbation.
    """
    
    def __init__(self, hidden_size: int, rank: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        
        print(f"DEBUG: Initializing ResidualLowRankTransform with hidden_size={hidden_size}, rank={rank}")
        
        # Low-rank factors: U @ V^T represents the perturbation from identity
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)  # Small initialization
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)  # Small initialization
        
        # Calculate FLOP reduction
        original_flops = hidden_size * hidden_size
        new_flops = 2 * hidden_size * rank  # U @ V^T computation
        flop_reduction = 1.0 - (new_flops / original_flops)
        
        print(f"DEBUG: Low-rank factors initialized - U: {self.U.shape}, V: {self.V.shape}")
        print(f"DEBUG: FLOP reduction: {flop_reduction:.2%} (from {original_flops} to {new_flops})")
        print(f"DEBUG: Rank compression ratio: {rank}/{hidden_size} = {rank/hidden_size:.3f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x @ (I + U @ V^T)^T = x + x @ (V @ U^T)"""
        # Identity component
        identity_output = x
        
        # Low-rank perturbation component: x @ (U @ V^T)^T = x @ (V @ U^T)
        perturbation = x @ (self.V @ self.U.T)
        
        return identity_output + perturbation
    
    def get_full_matrix(self) -> torch.Tensor:
        """Get the full transformation matrix I + U @ V^T"""
        identity = torch.eye(self.hidden_size, device=self.U.device, dtype=self.U.dtype)
        perturbation = self.U @ self.V.T
        return identity + perturbation
    
    def get_flop_reduction(self) -> float:
        """Calculate FLOP reduction compared to dense matrix."""
        original_flops = self.hidden_size * self.hidden_size
        new_flops = 2 * self.hidden_size * self.rank
        return 1.0 - (new_flops / original_flops)


def rild_optimization(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor = None,
    rank: int = 32,
    loss: str = "cosine",
    num_epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 1024,
    weight_decay: float = 1e-5
) -> torch.Tensor:
    """Optimize residual low-rank transformation using Adam optimizer."""
    
    print(f"DEBUG: Starting RILD optimization with rank={rank}")
    print(f"DEBUG: Input shapes - a1: {a1.shape}, a2: {a2.shape}")
    if a3 is not None:
        print(f"DEBUG: a3 shape: {a3.shape}")
    
    class ActivationDataset:
        def __init__(self, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor):
            self.a1, self.a2, self.a3 = a1, a2, a3            
        
        def __len__(self) -> int:
            return len(self.a1)
        
        def __getitem__(self, idx: int):
            attn = torch.tensor([-1]) if self.a3 is None else self.a3[idx]
            return self.a1[idx], self.a2[idx], attn

    # Initialize RILD model
    rild_model = ResidualLowRankTransform(a1.shape[1], rank=rank).to("cuda")
    
    # Use weight decay for regularization
    optimizer = torch.optim.Adam(rild_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Define loss functions
    def cosine_loss(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()

    def mse_loss(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(XA, Y)

    loss_fn = cosine_loss if loss == "cosine" else mse_loss
    
    # Training loop
    dataset = ActivationDataset(a1, a2, a3)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"DEBUG: Starting RILD training with {num_epochs} epochs, batch_size={batch_size}")
    print(f"DEBUG: Learning rate: {lr}, Weight decay: {weight_decay}")
    
    with tqdm(range(num_epochs), desc="Optimizing RILD Transformation") as pbar:
        for epoch in pbar:
            epoch_loss = 0.0
            num_batches = 0
            
            for X, Y, Z in loader:
                optimizer.zero_grad()
                
                # Forward pass through RILD transformation
                XA = rild_model(X.float().to("cuda"))
                
                # Add residual if available  
                if len(Z) > 1 and not torch.equal(Z, torch.tensor([-1])):
                    XA += Z.float().to("cuda")
                
                # Compute loss
                loss_val = loss_fn(XA, Y.float().to("cuda"))
                
                # Backward pass
                loss_val.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(rild_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss_val.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            pbar.set_postfix({f'{loss} Loss': f'{avg_loss:.4f}'})
            
            # Detailed debugging every few epochs
            if epoch % 3 == 0:
                with torch.no_grad():
                    # Check transformation quality
                    test_output = rild_model(a1[:1000].float().to("cuda"))
                    target_output = a2[:1000].float().to("cuda")
                    
                    cosine_sim = torch.cosine_similarity(test_output, target_output, dim=1).mean()
                    
                    # Check perturbation magnitude
                    perturbation_norm = (rild_model.U @ rild_model.V.T).norm().item()
                    u_norm = rild_model.U.norm().item()
                    v_norm = rild_model.V.norm().item()
                    
                    print(f"DEBUG: Epoch {epoch}, Average Loss: {avg_loss:.6f}")
                    print(f"DEBUG: Cosine similarity: {cosine_sim:.4f}")
                    print(f"DEBUG: Perturbation norm: {perturbation_norm:.4f}")
                    print(f"DEBUG: U norm: {u_norm:.4f}, V norm: {v_norm:.4f}")
    
    # Get the final transformation matrix
    with torch.no_grad():
        final_transform = rild_model.get_full_matrix()
    
    flop_reduction = rild_model.get_flop_reduction()
    print(f"DEBUG: Final FLOP reduction achieved: {flop_reduction:.2%}")
    print(f"DEBUG: Final perturbation magnitude: {(rild_model.U @ rild_model.V.T).norm().item():.4f}")
    
    return final_transform.to(torch.float64)


def rild_method(
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
    rank: int = 32,
    loss: str = "cosine",
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool = False
) -> str:
    """RILD method: Residual-Informed Low-Rank Decomposition for efficient LLM compression.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of start layer for cut
        token: Authentication token
        save_transform_only: Whether to only save the transform
        rank: Rank for low-rank decomposition
        loss: Loss function type
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
        accurate: Whether to use accurate mode
    
    Returns:
        Path where transformed model is saved
    """
    
    print(f"DEBUG: Starting RILD method")
    print(f"DEBUG: Processing layers {start_id} to {end_id} (total: {end_id - start_id} layers)")
    print(f"DEBUG: Low-rank settings - rank: {rank}, loss: {loss}")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    print("DEBUG: Loading model for activation extraction...")
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
    
    print(f"DEBUG: Model hidden size: {hidden_size}")
    print(f"DEBUG: Rank compression ratio: {rank}/{hidden_size} = {rank/hidden_size:.3f}")
    print(f"DEBUG: Calibration dataset loaded with batch_size: {batch_size}")
    
    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""
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
    
    # Prepare activation storage
    total_tokens = dataset_size * max_length
    a1 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    a2 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    if accurate:
        print("DEBUG: ACCURATE MODE IS ON (MORE MEMORY IS NEEDED)")
        a3 = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16, device='cpu')
    
    cnt = 0
    print("DEBUG: Starting activation gathering...")
    
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations for RILD" + Fore.RESET,
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
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        
        if accurate:
            a2_batch = hidden_states_n 
            a3_batch = hidden_states_i - hidden_states_mlp 
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]
    
    print(f"DEBUG: Collected {cnt} activation samples")
    print(f"DEBUG: Starting RILD optimization...")
    
    # Estimate RILD transformation
    transform = rild_optimization(
        a1, a2, 
        a3=a3 if accurate else None, 
        rank=rank,
        loss=loss
    )
    
    print(f"DEBUG: RILD transform estimated with shape: {transform.shape}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("DEBUG: Reloading model for transformation application...")
    
    # Reload model for transformation
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
            f"{end_id}_{dataset}_{dataset_size}"
        ).replace("/", "_")
    
    print(f"DEBUG: Applying RILD transformation to model...")
    
    # Apply RILD transformation
    original_weight = model.model.layers[start_id - num_layer - 1].mlp.down_proj.weight
    new_weight = (transform.T.cpu() @ original_weight.to(torch.float64)).to(torch.bfloat16)
    
    model.model.layers[start_id - num_layer - 1].mlp.down_proj.load_state_dict({
        "weight": new_weight
    })
    
    output_path = f"{save_path}_RILD_rank{rank}_{loss}"
    
    print(f"DEBUG: Saving model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    if save_transform_only:
        torch.save(transform, f"{output_path}_transform")
        print(f"DEBUG: RILD transform saved separately")
    
    # Final cleanup
    del model, a1, a2
    if accurate:
        del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"DEBUG: RILD method completed successfully")
    
    return output_path


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the RILD method from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run RILD for residual-informed low-rank decomposition based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    average_distances = torch.load(config['distances_path'])
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        config['layers_to_skip'],
        num_blocks=config['num_A'],
        merge_consecutive=config['merge_consecutive']
    )
    
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    for i in range(len(selected_blocks)):
        path = rild_method(**config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
        config["model_path"] = path