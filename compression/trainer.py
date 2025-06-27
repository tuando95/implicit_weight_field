"""Training utilities for implicit weight fields."""

import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

from core.positional_encoding import generate_coordinate_grid


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for field training."""
    learning_rate: float = 1e-3
    max_steps: int = 2000
    convergence_threshold: float = 1e-6
    convergence_patience: int = 100
    batch_size: Optional[int] = None  # None means full batch
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    scheduler: str = "cosine"  # "cosine", "step", "none"
    warmup_steps: int = 100
    log_interval: int = 100


class FieldTrainer:
    """Trainer for implicit weight fields."""
    
    def __init__(
        self,
        field: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            field: Implicit weight field to train
            config: Training configuration
            device: Device to train on
        """
        self.field = field
        self.config = config
        self.device = device or next(field.parameters()).device
        
        # Move field to device
        self.field = self.field.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.field.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=1e-6
            )
        elif self.config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_steps // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def train(
        self,
        target_tensor: torch.Tensor,
        verbose: bool = True
    ) -> int:
        """
        Train the implicit field to represent the target tensor.
        
        Args:
            target_tensor: Target weight tensor to fit
            verbose: Whether to show progress
            
        Returns:
            Number of training steps taken
        """
        # Move target to device
        target_tensor = target_tensor.to(self.device)
        
        # Generate training data (all coordinates)
        indices, normalized_coords = generate_coordinate_grid(
            target_tensor.shape,
            device=self.device
        )
        
        # Flatten target tensor
        target_values = target_tensor.flatten()
        
        # Training loop
        self.field.train()
        
        progress_bar = tqdm(range(self.config.max_steps), disable=not verbose)
        
        for step in progress_bar:
            self.step = step
            
            # Forward pass
            if self.config.batch_size is not None:
                # Mini-batch training
                batch_idx = torch.randperm(len(normalized_coords))[:self.config.batch_size]
                batch_coords = normalized_coords[batch_idx]
                batch_targets = target_values[batch_idx]
            else:
                # Full batch
                batch_coords = normalized_coords
                batch_targets = target_values
            
            # Predict
            predictions = self.field(batch_coords)
            
            # Compute loss
            loss = nn.functional.mse_loss(predictions, batch_targets)
            
            # Add regularization if using explicit parameters
            if hasattr(self.field, 'explicit_weights'):
                loss = loss + self.config.weight_decay * torch.norm(self.field.explicit_weights)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.field.parameters(),
                    self.config.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update progress
            current_loss = loss.item()
            progress_bar.set_description(f"Loss: {current_loss:.6f}")
            
            # Check convergence
            if self._check_convergence(current_loss):
                logger.info(f"Converged at step {step}")
                break
            
            # Logging
            if step % self.config.log_interval == 0:
                self._log_metrics(step, current_loss)
        
        progress_bar.close()
        
        # Final validation
        self.field.eval()
        with torch.no_grad():
            final_predictions = self.field(normalized_coords)
            final_loss = nn.functional.mse_loss(final_predictions, target_values)
            max_error = torch.max(torch.abs(final_predictions - target_values))
            
        logger.info(f"Training complete. Final MSE: {final_loss:.6f}, Max error: {max_error:.6f}")
        
        return self.step + 1
    
    def _check_convergence(self, current_loss: float) -> bool:
        """Check if training has converged."""
        # Check if loss improved
        if current_loss < self.best_loss - self.config.convergence_threshold:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check patience
        return self.patience_counter >= self.config.convergence_patience
    
    def _log_metrics(self, step: int, loss: float):
        """Log training metrics."""
        metrics = {
            'step': step,
            'loss': loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Count parameters
        total_params = sum(p.numel() for p in self.field.parameters())
        metrics['params'] = total_params
        
        logger.debug(f"Step {step}: {metrics}")
    
    def evaluate(self, target_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate field reconstruction quality.
        
        Args:
            target_tensor: Original tensor
            
        Returns:
            Dictionary of metrics
        """
        self.field.eval()
        target_tensor = target_tensor.to(self.device)
        
        with torch.no_grad():
            # Reconstruct full tensor
            reconstructed = self.field.reconstruct_full_tensor()
            
            # Compute metrics
            mse = torch.mean((reconstructed - target_tensor) ** 2).item()
            rmse = torch.sqrt(torch.mean((reconstructed - target_tensor) ** 2)).item()
            max_error = torch.max(torch.abs(reconstructed - target_tensor)).item()
            
            # SNR calculation
            signal_power = torch.var(target_tensor).item()
            noise_power = torch.var(reconstructed - target_tensor).item()
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Relative error
            rel_error = torch.norm(reconstructed - target_tensor) / torch.norm(target_tensor)
            
        return {
            'mse': mse,
            'rmse': rmse,
            'max_error': max_error,
            'snr': snr,
            'relative_error': rel_error.item()
        }