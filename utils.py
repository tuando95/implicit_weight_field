"""Utility functions for the project."""

import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
from datetime import datetime


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    output_dir: Optional[str] = None,
    log_level: str = "INFO",
    verbose: bool = False
):
    """
    Setup logging configuration.
    
    Args:
        output_dir: Directory to save logs
        log_level: Logging level
        verbose: Whether to use verbose logging
    """
    if verbose:
        log_level = "DEBUG"
    
    handlers = [logging.StreamHandler()]
    
    if output_dir:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Path to save file
    """
    path = Path(path)
    
    if path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Trainable parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return size_mb


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_experiment_id() -> str:
    """
    Create unique experiment ID.
    
    Returns:
        Experiment ID string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    return f"{timestamp}_{random_suffix}"


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str = "meter"):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter.
        
        Args:
            val: Value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current score
            
        Returns:
            Whether to stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is improvement."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta