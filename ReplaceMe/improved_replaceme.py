import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional, Tuple
from colorama import Fore, init

init(autoreset=True)

class ActivationDataset(Dataset):
    """Dataset for activation pairs."""
    def __init__(self, a1: torch.Tensor, a2: torch.Tensor, a3: Optional[torch.Tensor] = None):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def __len__(self) -> int:
        return len(self.a1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = torch.tensor([-1.0])
        if self.a3 is not None:
            attn = self.a3[idx]
        return self.a1[idx], self.a2[idx], attn


def improved_adam_optimizer(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: Optional[torch.Tensor] = None,
    loss_type: str = "mixed",
    epochs: int = 20,
    lr: float = 5e-4,
    batch_size: int = 1024,
    weight_decay: float = 1e-4,
    patience: int = 3,
    verbose: bool = True
) -> torch.Tensor:
    """
    Improved optimization method with better hyperparameters and techniques.
    
    Args:
        a1: Input activations
        a2: Target activations
        a3: Optional residual activations
        loss_type: Type of loss ("cosine", "mixed", "mse")
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        weight_decay: Weight decay for AdamW
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Optimized transformation matrix
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = a1.shape[1]
    
    # Initialize transformation matrix
    transform = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
    nn.init.eye_(transform.weight)  # Initialize as identity
    
    # Add small perturbation for better optimization
    with torch.no_grad():
        transform.weight.data += torch.randn_like(transform.weight) * 0.01
    
    # Optimizer with weight decay (AdamW)
    optimizer = torch.optim.AdamW(
        transform.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Dataset and dataloader
    dataset = ActivationDataset(a1, a2, a3)
    
    # Early stopping variables
    best_loss = float('inf')
    best_weight = None
    patience_counter = 0
    
    # Training loop
    pbar = tqdm(range(epochs), desc="Optimizing Transform", disable=not verbose)
    
    for epoch in pbar:
        # Create new dataloader each epoch for better shuffling
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch)  # Deterministic but different each epoch
        )
        
        epoch_loss = 0.0
        num_batches = 0
        
        for X, Y, Z in loader:
            X = X.float().to(device)
            Y = Y.float().to(device)
            Z = Z.float().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            XA = transform(X)
            
            # Add residual if provided
            if Z.shape[0] > 1 and Z[0].item() != -1:
                XA = XA + Z
            
            # Compute loss based on type
            if loss_type == "cosine":
                loss_val = cosine_loss(XA, Y)
            elif loss_type == "mixed":
                loss_val = mixed_loss(XA, Y, transform.weight, device)
            elif loss_type == "mse":
                loss_val = F.mse_loss(XA, Y)
            else:
                loss_val = cosine_loss(XA, Y)  # Default to cosine
            
            # Backward pass with gradient clipping
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(transform.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            num_batches += 1
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'LR': f'{current_lr:.6f}',
            'Best': f'{best_loss:.4f}'
        })
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weight = transform.weight.T.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n{Fore.YELLOW}Early stopping at epoch {epoch+1}{Fore.RESET}")
                break
        
        # Periodic logging
        if verbose and epoch % 5 == 0:
            print(f"\n{Fore.GREEN}Epoch {epoch}: Loss={avg_loss:.4f}, LR={current_lr:.6f}{Fore.RESET}")
    
    return best_weight.to(torch.float64)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss."""
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return 1 - (pred_norm * target_norm).sum(dim=-1).mean()


def mixed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Mixed loss combining cosine, MSE, and regularization.
    
    Args:
        pred: Predicted values
        target: Target values
        weight: Transformation weight matrix
        device: Device to use
    
    Returns:
        Combined loss value
    """
    # Cosine similarity loss (main component)
    cos_loss = cosine_loss(pred, target)
    
    # Normalized MSE loss (for magnitude)
    target_norm = target.norm() + 1e-6
    mse_loss = F.mse_loss(pred, target) / target_norm
    
    # L2 regularization to keep close to identity
    identity = torch.eye(weight.shape[0], device=device)
    reg_loss = torch.norm(weight - identity, 'fro') / weight.shape[0]
    
    # Combine losses with weights
    total_loss = 0.9 * cos_loss + 0.08 * mse_loss + 0.02 * reg_loss
    
    return total_loss