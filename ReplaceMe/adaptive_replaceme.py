# improved_adaptive_replaceme.py (fixed version)
import gc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import get_calib_dataloader, seed_all

init(autoreset=True)
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


class ImprovedAdaptiveTransform(nn.Module):
    """Improved adaptive transformation with better initialization and training."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Main transformation (initialized closer to identity)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        # Initialize with identity + small noise
        nn.init.eye_(self.W.weight)
        self.W.weight.data += torch.randn_like(self.W.weight) * 0.01
        
        # Dynamic gating based on input statistics
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize gate network to output ~0.5
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.constant_(self.gate_net[-2].bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if len(x.shape) == 3:
            B, T, D = x.shape
            x = x.view(-1, D)
        
        # Compute transformation
        transformed = self.W(x)
        
        # Compute adaptive gate per token
        gate = self.gate_net(x)  # [B*T, 1] or [N, 1]
        
        # Mix based on gate
        output = x * gate + transformed * (1 - gate)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            output = output.view(original_shape)
            
        return output


def improved_adaptive_replaceme(
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
    **kwargs
) -> str:
    
    print(f"\n[Improved AR] Processing layers {start_id}-{end_id}")
    print(f"[Improved AR] Layers to skip: {layers_to_skip}")
    
    # Load model
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_size = model.config.hidden_size
    model.eval()
    
    # Gather calibration data
    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer
    )
    
    print(f"[Improved AR] Gathering calibration data...")
    
    inputs_list = []
    outputs_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calibration", total=dataset_size//batch_size):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]
            
            h_in = hidden_states[start_id - num_layer - 1]
            h_out = hidden_states[end_id - num_layer - 1]
            
            # Flatten batch and sequence dimensions to avoid padding issues
            B, T, D = h_in.shape
            h_in_flat = h_in.view(-1, D)
            h_out_flat = h_out.view(-1, D)
            
            inputs_list.append(h_in_flat.cpu())
            outputs_list.append(h_out_flat.cpu())
    
    # Concatenate all flattened data
    all_inputs = torch.cat(inputs_list, dim=0).to(torch.float32)
    all_outputs = torch.cat(outputs_list, dim=0).to(torch.float32)
    
    print(f"[Improved AR] Data shape: {all_inputs.shape}")
    
    # Analyze data
    residual = all_outputs - all_inputs
    residual_ratio = torch.norm(all_inputs) / (torch.norm(residual) + 1e-8)
    print(f"[Improved AR] Residual ratio: {residual_ratio:.3f}")
    
    # Calculate average transformation magnitude
    transform_magnitude = torch.norm(residual) / torch.norm(all_inputs)
    print(f"[Improved AR] Transform magnitude: {transform_magnitude:.3f}")
    
    # Create and train transform
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = ImprovedAdaptiveTransform(hidden_size).to(device)
    
    # Use both MSE and cosine loss
    optimizer = torch.optim.AdamW(transform.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training
    print(f"[Improved AR] Training transformation...")
    
    all_inputs = all_inputs.to(device)
    all_outputs = all_outputs.to(device)
    
    dataset_size = all_inputs.shape[0]
    
    best_loss = float('inf')
    
    for epoch in range(20):
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        num_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, 1024):
            batch_idx = indices[i:min(i+1024, dataset_size)]
            batch_in = all_inputs[batch_idx]
            batch_out = all_outputs[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward
            pred = transform(batch_in)
            
            # Combined loss
            mse_loss = F.mse_loss(pred, batch_out)
            
            # Cosine loss
            pred_norm = F.normalize(pred, dim=-1)
            out_norm = F.normalize(batch_out, dim=-1)
            cosine_loss = 1 - (pred_norm * out_norm).sum(-1).mean()
            
            # Weight cosine loss less
            loss = mse_loss + 0.1 * cosine_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transform.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_cosine += cosine_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_cosine = total_cosine / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 5 == 0 or epoch == 19:
            # Check gate statistics
            with torch.no_grad():
                sample_size = min(1000, dataset_size)
                sample = all_inputs[:sample_size]
                gates = transform.gate_net(sample).mean().item()
                gate_std = transform.gate_net(sample).std().item()
                
            print(f"[Improved AR] Epoch {epoch}: Loss={avg_loss:.4f} (MSE={avg_mse:.4f}, Cos={avg_cosine:.4f})")
            print(f"              Gate: mean={gates:.3f}, std={gate_std:.3f}")
    
    print(f"[Improved AR] Training complete. Best loss: {best_loss:.4f}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model for modification
    print(f"[Improved AR] Modifying model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    # Create custom layer that wraps our transform
    class AdaptiveReplacementLayer(nn.Module):
        def __init__(self, transform_module, dtype=torch.bfloat16):
            super().__init__()
            self.transform = transform_module
            self.dtype = dtype
            
        def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            # Convert to float32 for computation
            h = hidden_states.to(torch.float32)
            
            # Apply transformation
            h = self.transform(h)
            
            # Convert back to model dtype
            h = h.to(self.dtype)
            
            return h
    
    # Create replacement
    replacement = AdaptiveReplacementLayer(transform.cpu(), model.dtype)
    
    # Modify layers - insert replacement at the right position
    new_layers = []
    
    for i in range(len(model.model.layers)):
        if i == start_id - num_layer:
            # Insert replacement layer here
            new_layers.append(replacement)
            print(f"[Improved AR] Inserted replacement at position {i}")
        
        if i < start_id - num_layer or i >= end_id - num_layer:
            # Keep this layer
            new_layers.append(model.model.layers[i])
        # else: skip this layer (it's being replaced)
    
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    
    print(f"[Improved AR] Final model has {len(new_layers)} layers")
    
    # Save
    if save_path is None:
        save_path = f"output_models/ImprovedAR_{model_path.replace('/', '_')}_{start_id}_{end_id}"
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[Improved AR] Model saved to {save_path}")
    
    # Save transform details
    torch.save({
        'transform_state': transform.state_dict(),
        'config': {
            'start_id': start_id,
            'end_id': end_id,
            'hidden_size': hidden_size,
            'best_loss': best_loss
        }
    }, f"{save_path}/transform.pt")
    
    return save_path