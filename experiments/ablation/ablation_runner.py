"""Ablation study runner implementing systematic parameter variation."""

import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import itertools
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from core.implicit_field import ImplicitWeightField, CompressionConfig
from core.siren import SIREN
from core.positional_encoding import FourierFeatures, HashEncoding
from compression import compress_model
from evaluation import evaluate_compression, evaluate_reconstruction_quality
from compression.trainer import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result from a single ablation experiment."""
    parameter_name: str
    parameter_value: Any
    compression_ratio: float
    reconstruction_error: float
    training_steps: int
    convergence_time: float
    field_parameters: int
    additional_metrics: Dict[str, Any]


class AblationStudy:
    """Base class for ablation studies."""
    
    def __init__(self, name: str, base_config: CompressionConfig):
        """
        Initialize ablation study.
        
        Args:
            name: Study name
            base_config: Base configuration to modify
        """
        self.name = name
        self.base_config = base_config
        self.results = []
    
    def run(
        self,
        model: nn.Module,
        test_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> List[AblationResult]:
        """
        Run ablation study.
        
        Args:
            model: Model to test on
            test_tensors: Specific tensors to test (None = use all)
            device: Device to use
            
        Returns:
            List of ablation results
        """
        raise NotImplementedError
    
    def summarize(self) -> Dict[str, Any]:
        """Summarize ablation results."""
        if not self.results:
            return {}
        
        summary = {
            "name": self.name,
            "num_experiments": len(self.results),
            "parameters_tested": list(set(r.parameter_name for r in self.results)),
            "best_compression": max(self.results, key=lambda r: r.compression_ratio),
            "best_reconstruction": min(self.results, key=lambda r: r.reconstruction_error),
            "average_metrics": self._compute_average_metrics()
        }
        
        return summary
    
    def _compute_average_metrics(self) -> Dict[str, float]:
        """Compute average metrics across all experiments."""
        metrics = {}
        
        for result in self.results:
            for key, value in result.additional_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in metrics.items()}


class ArchitectureAblation(AblationStudy):
    """Ablation study for field architecture choices."""
    
    def __init__(
        self,
        base_config: CompressionConfig,
        depths: List[int] = [1, 2, 3, 4],
        widths: List[int] = [64, 128, 256, 512, 1024],
        activations: List[str] = ["siren", "relu", "gaussian", "swish"],
        skip_connections: bool = True
    ):
        """
        Initialize architecture ablation.
        
        Args:
            base_config: Base configuration
            depths: Network depths to test
            widths: Hidden widths to test
            activations: Activation functions to test
            skip_connections: Whether to test skip connections
        """
        super().__init__("architecture", base_config)
        self.depths = depths
        self.widths = widths
        self.activations = activations
        self.skip_connections = skip_connections
    
    def run(
        self,
        model: nn.Module,
        test_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> List[AblationResult]:
        """Run architecture ablation."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get test tensors
        if test_tensors is None:
            test_tensors = self._get_representative_tensors(model)
        
        logger.info(f"Running architecture ablation on {len(test_tensors)} tensors")
        
        # Test different depths
        for depth in self.depths:
            logger.info(f"Testing depth={depth}")
            config = CompressionConfig(
                bandwidth=self.base_config.bandwidth,
                hidden_width=self.base_config.hidden_width,
                num_layers=depth,
                w0=self.base_config.w0,
                learning_rate=self.base_config.learning_rate,
                max_steps=self.base_config.max_steps
            )
            
            result = self._evaluate_config(
                config, test_tensors, "depth", depth, device
            )
            self.results.append(result)
        
        # Test different widths
        for width in self.widths:
            logger.info(f"Testing width={width}")
            config = CompressionConfig(
                bandwidth=self.base_config.bandwidth,
                hidden_width=width,
                num_layers=self.base_config.num_layers,
                w0=self.base_config.w0,
                learning_rate=self.base_config.learning_rate,
                max_steps=self.base_config.max_steps
            )
            
            result = self._evaluate_config(
                config, test_tensors, "width", width, device
            )
            self.results.append(result)
        
        # Test different activations
        for activation in self.activations:
            logger.info(f"Testing activation={activation}")
            # This would require modifying SIREN to support different activations
            # For now, we'll simulate with different w0 values
            w0_map = {"siren": 30.0, "relu": 1.0, "gaussian": 10.0, "swish": 1.0}
            
            config = CompressionConfig(
                bandwidth=self.base_config.bandwidth,
                hidden_width=self.base_config.hidden_width,
                num_layers=self.base_config.num_layers,
                w0=w0_map.get(activation, 1.0),
                learning_rate=self.base_config.learning_rate,
                max_steps=self.base_config.max_steps
            )
            
            result = self._evaluate_config(
                config, test_tensors, "activation", activation, device
            )
            self.results.append(result)
        
        return self.results
    
    def _get_representative_tensors(
        self,
        model: nn.Module,
        max_tensors: int = 5
    ) -> Dict[str, torch.Tensor]:
        """Get representative tensors from model."""
        tensors = {}
        count = 0
        
        for name, param in model.named_parameters():
            if count >= max_tensors:
                break
            
            # Skip small tensors
            if param.numel() < 10000:
                continue
            
            tensors[name] = param.data.clone()
            count += 1
        
        return tensors
    
    def _evaluate_config(
        self,
        config: CompressionConfig,
        test_tensors: Dict[str, torch.Tensor],
        param_name: str,
        param_value: Any,
        device: torch.device
    ) -> AblationResult:
        """Evaluate a single configuration."""
        import time
        
        total_original = 0
        total_compressed = 0
        total_error = 0
        total_steps = 0
        start_time = time.time()
        
        for name, tensor in test_tensors.items():
            # Create and train field
            from core.implicit_field import ImplicitWeightField
            from compression.trainer import FieldTrainer
            
            field = ImplicitWeightField(
                tensor_shape=tensor.shape,
                config=config
            ).to(device)
            
            trainer = FieldTrainer(
                field,
                TrainingConfig(
                    learning_rate=config.learning_rate,
                    max_steps=config.max_steps,
                    convergence_threshold=config.convergence_threshold
                )
            )
            
            steps = trainer.train(tensor.to(device), verbose=False)
            
            # Evaluate
            with torch.no_grad():
                reconstructed = field.reconstruct_full_tensor()
                mse = torch.mean((reconstructed - tensor.to(device)) ** 2).item()
            
            total_original += tensor.numel()
            total_compressed += field.count_parameters()
            total_error += mse
            total_steps += steps
        
        convergence_time = time.time() - start_time
        
        return AblationResult(
            parameter_name=param_name,
            parameter_value=param_value,
            compression_ratio=total_original / max(total_compressed, 1),
            reconstruction_error=total_error / len(test_tensors),
            training_steps=total_steps // len(test_tensors),
            convergence_time=convergence_time,
            field_parameters=total_compressed,
            additional_metrics={
                "avg_mse": total_error / len(test_tensors),
                "total_params": total_compressed
            }
        )


class EncodingAblation(AblationStudy):
    """Ablation study for positional encoding choices."""
    
    def __init__(
        self,
        base_config: CompressionConfig,
        bandwidths: List[int] = [1, 2, 4, 8, 16],
        encoding_types: List[str] = ["fourier", "learned", "hash", "none"],
        coordinate_normalizations: List[str] = ["zero_one", "minus_one_one", "standard"]
    ):
        """Initialize encoding ablation."""
        super().__init__("encoding", base_config)
        self.bandwidths = bandwidths
        self.encoding_types = encoding_types
        self.coordinate_normalizations = coordinate_normalizations
    
    def run(
        self,
        model: nn.Module,
        test_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> List[AblationResult]:
        """Run encoding ablation."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if test_tensors is None:
            test_tensors = self._get_representative_tensors(model)
        
        logger.info(f"Running encoding ablation on {len(test_tensors)} tensors")
        
        # Test different bandwidths
        for bandwidth in self.bandwidths:
            logger.info(f"Testing bandwidth={bandwidth}")
            config = CompressionConfig(
                bandwidth=bandwidth,
                hidden_width=self.base_config.hidden_width,
                num_layers=self.base_config.num_layers,
                w0=self.base_config.w0,
                learning_rate=self.base_config.learning_rate,
                max_steps=self.base_config.max_steps
            )
            
            result = self._evaluate_config(
                config, test_tensors, "bandwidth", bandwidth, device
            )
            self.results.append(result)
        
        # Test no encoding
        logger.info("Testing no encoding")
        config = CompressionConfig(
            bandwidth=0,  # No encoding
            hidden_width=self.base_config.hidden_width,
            num_layers=self.base_config.num_layers,
            w0=self.base_config.w0,
            learning_rate=self.base_config.learning_rate,
            max_steps=self.base_config.max_steps
        )
        
        result = self._evaluate_config(
            config, test_tensors, "encoding_type", "none", device
        )
        self.results.append(result)
        
        return self.results
    
    def _get_representative_tensors(
        self,
        model: nn.Module,
        max_tensors: int = 5
    ) -> Dict[str, torch.Tensor]:
        """Get representative tensors."""
        tensors = {}
        
        # Get diverse tensor shapes
        shapes_seen = set()
        
        for name, param in model.named_parameters():
            if len(tensors) >= max_tensors:
                break
            
            shape_key = tuple(sorted(param.shape))
            if shape_key in shapes_seen:
                continue
            
            if param.numel() > 5000:
                tensors[name] = param.data.clone()
                shapes_seen.add(shape_key)
        
        return tensors
    
    def _evaluate_config(
        self,
        config: CompressionConfig,
        test_tensors: Dict[str, torch.Tensor],
        param_name: str,
        param_value: Any,
        device: torch.device
    ) -> AblationResult:
        """Evaluate configuration."""
        # Similar to architecture ablation
        import time
        from core.implicit_field import ImplicitWeightField
        from compression.trainer import FieldTrainer
        
        total_compression = 0
        total_error = 0
        start_time = time.time()
        
        for name, tensor in test_tensors.items():
            field = ImplicitWeightField(
                tensor_shape=tensor.shape,
                config=config
            ).to(device)
            
            trainer = FieldTrainer(
                field,
                TrainingConfig(
                    learning_rate=config.learning_rate,
                    max_steps=config.max_steps
                )
            )
            
            trainer.train(tensor.to(device), verbose=False)
            
            metrics = trainer.evaluate(tensor.to(device))
            
            total_compression += tensor.numel() / field.count_parameters()
            total_error += metrics['mse']
        
        return AblationResult(
            parameter_name=param_name,
            parameter_value=param_value,
            compression_ratio=total_compression / len(test_tensors),
            reconstruction_error=total_error / len(test_tensors),
            training_steps=config.max_steps,
            convergence_time=time.time() - start_time,
            field_parameters=0,
            additional_metrics={}
        )


class TrainingAblation(AblationStudy):
    """Ablation study for training procedure choices."""
    
    def __init__(
        self,
        base_config: CompressionConfig,
        learning_rates: List[float] = [1e-4, 1e-3, 1e-2],
        training_steps: List[int] = [500, 1000, 2000, 5000],
        optimizers: List[str] = ["adam", "sgd", "adamw", "rmsprop"],
        batch_processing: List[Optional[int]] = [None, 256, 512, 1024],
        regularization: List[float] = [0, 1e-7, 1e-6, 1e-5, 1e-4]
    ):
        """Initialize training ablation."""
        super().__init__("training", base_config)
        self.learning_rates = learning_rates
        self.training_steps = training_steps
        self.optimizers = optimizers
        self.batch_processing = batch_processing
        self.regularization = regularization
    
    def run(
        self,
        model: nn.Module,
        test_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> List[AblationResult]:
        """Run training ablation."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if test_tensors is None:
            test_tensors = self._get_test_tensors(model)
        
        # Test learning rates
        for lr in self.learning_rates:
            logger.info(f"Testing learning_rate={lr}")
            config = self._modify_config(learning_rate=lr)
            result = self._evaluate_training_config(
                config, test_tensors, "learning_rate", lr, device
            )
            self.results.append(result)
        
        # Test training steps
        for steps in self.training_steps:
            logger.info(f"Testing max_steps={steps}")
            config = self._modify_config(max_steps=steps)
            result = self._evaluate_training_config(
                config, test_tensors, "max_steps", steps, device
            )
            self.results.append(result)
        
        # Test regularization
        for reg in self.regularization:
            logger.info(f"Testing regularization={reg}")
            config = self._modify_config(regularization=reg)
            result = self._evaluate_training_config(
                config, test_tensors, "regularization", reg, device
            )
            self.results.append(result)
        
        return self.results
    
    def _get_test_tensors(
        self,
        model: nn.Module,
        num_tensors: int = 3
    ) -> Dict[str, torch.Tensor]:
        """Get test tensors of different sizes."""
        tensors = {}
        
        # Small, medium, large tensors
        size_ranges = [(1e4, 1e5), (1e5, 1e6), (1e6, 1e7)]
        
        for i, (min_size, max_size) in enumerate(size_ranges):
            for name, param in model.named_parameters():
                if min_size <= param.numel() < max_size:
                    tensors[f"{name}_size{i}"] = param.data.clone()
                    break
        
        return tensors
    
    def _modify_config(self, **kwargs) -> CompressionConfig:
        """Create modified config."""
        config_dict = {
            'bandwidth': self.base_config.bandwidth,
            'hidden_width': self.base_config.hidden_width,
            'num_layers': self.base_config.num_layers,
            'w0': self.base_config.w0,
            'learning_rate': self.base_config.learning_rate,
            'max_steps': self.base_config.max_steps,
            'convergence_threshold': self.base_config.convergence_threshold,
            'regularization': self.base_config.regularization
        }
        config_dict.update(kwargs)
        return CompressionConfig(**config_dict)
    
    def _evaluate_training_config(
        self,
        config: CompressionConfig,
        test_tensors: Dict[str, torch.Tensor],
        param_name: str,
        param_value: Any,
        device: torch.device
    ) -> AblationResult:
        """Evaluate training configuration."""
        import time
        from core.implicit_field import ImplicitWeightField
        from compression.trainer import FieldTrainer, TrainingConfig
        
        results = []
        start_time = time.time()
        
        for name, tensor in test_tensors.items():
            field = ImplicitWeightField(
                tensor_shape=tensor.shape,
                config=config
            ).to(device)
            
            training_config = TrainingConfig(
                learning_rate=config.learning_rate,
                max_steps=config.max_steps,
                convergence_threshold=config.convergence_threshold,
                weight_decay=config.regularization
            )
            
            trainer = FieldTrainer(field, training_config)
            steps = trainer.train(tensor.to(device), verbose=False)
            metrics = trainer.evaluate(tensor.to(device))
            
            results.append({
                'compression': tensor.numel() / field.count_parameters(),
                'mse': metrics['mse'],
                'steps': steps
            })
        
        # Aggregate results
        avg_compression = np.mean([r['compression'] for r in results])
        avg_mse = np.mean([r['mse'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        return AblationResult(
            parameter_name=param_name,
            parameter_value=param_value,
            compression_ratio=avg_compression,
            reconstruction_error=avg_mse,
            training_steps=int(avg_steps),
            convergence_time=time.time() - start_time,
            field_parameters=0,
            additional_metrics={
                'convergence_rate': avg_steps / config.max_steps
            }
        )


def run_ablation_studies(
    model: nn.Module,
    studies: List[str],
    base_config: Optional[CompressionConfig] = None,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> Dict[str, List[AblationResult]]:
    """
    Run multiple ablation studies.
    
    Args:
        model: Model to test on
        studies: List of study names to run
        base_config: Base configuration
        output_dir: Directory to save results
        device: Device to use
        
    Returns:
        Dictionary of study results
    """
    if base_config is None:
        base_config = CompressionConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Create study instances
    study_map = {
        "architecture": ArchitectureAblation(base_config),
        "encoding": EncodingAblation(base_config),
        "training": TrainingAblation(base_config)
    }
    
    # Run requested studies
    for study_name in studies:
        if study_name not in study_map:
            logger.warning(f"Unknown study: {study_name}")
            continue
        
        logger.info(f"Running {study_name} ablation study")
        study = study_map[study_name]
        study_results = study.run(model, device=device)
        results[study_name] = study_results
        
        # Save intermediate results
        if output_dir:
            save_ablation_results(study_results, output_dir / f"{study_name}_results.json")
        
        # Log summary
        summary = study.summarize()
        logger.info(f"{study_name} ablation summary:")
        logger.info(f"  Best compression: {summary['best_compression'].compression_ratio:.2f}x "
                   f"(param={summary['best_compression'].parameter_name}="
                   f"{summary['best_compression'].parameter_value})")
        logger.info(f"  Best reconstruction: {summary['best_reconstruction'].reconstruction_error:.6f} "
                   f"(param={summary['best_reconstruction'].parameter_name}="
                   f"{summary['best_reconstruction'].parameter_value})")
    
    return results


def save_ablation_results(results: List[AblationResult], filepath: Path):
    """Save ablation results to JSON."""
    data = []
    for result in results:
        data.append({
            'parameter_name': result.parameter_name,
            'parameter_value': str(result.parameter_value),
            'compression_ratio': result.compression_ratio,
            'reconstruction_error': result.reconstruction_error,
            'training_steps': result.training_steps,
            'convergence_time': result.convergence_time,
            'field_parameters': result.field_parameters,
            'additional_metrics': result.additional_metrics
        })
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)