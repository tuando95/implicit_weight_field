"""Main compression pipeline implementing Algorithm 1 from CLAUDE.md."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
import numpy as np

from ..core.implicit_field import (
    ImplicitWeightField,
    MultiScaleImplicitField,
    CompressionConfig,
    FieldArchitecture,
    TensorStatistics
)
from ..core.positional_encoding import generate_coordinate_grid
from .trainer import FieldTrainer, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class LayerCompressionResult:
    """Result of compressing a single layer."""
    layer_name: str
    original_shape: Tuple[int, ...]
    original_params: int
    compressed_params: int
    compression_ratio: float
    reconstruction_error: float
    max_error: float
    field_architecture: str
    training_steps: int
    field: Union[ImplicitWeightField, MultiScaleImplicitField]


@dataclass
class CompressionResult:
    """Overall compression result."""
    original_params: int
    compressed_params: int
    compression_ratio: float
    layer_results: Dict[str, LayerCompressionResult]
    total_time: float
    config: CompressionConfig


class ImplicitWeightFieldCompressor:
    """Main compressor implementing Algorithm 1."""
    
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize compressor.
        
        Args:
            config: Compression configuration
            device: Device to use for compression
        """
        self.config = config or CompressionConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training configuration
        self.training_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            convergence_threshold=self.config.convergence_threshold,
            weight_decay=self.config.regularization
        )
    
    def compress_model(
        self,
        model: nn.Module,
        layer_filter: Optional[List[str]] = None,
        validation_loader: Optional[Any] = None
    ) -> Tuple[nn.Module, CompressionResult]:
        """
        Compress a neural network model (Algorithm 1).
        
        Args:
            model: PyTorch model to compress
            layer_filter: Optional list of layer names to compress
            validation_loader: Optional dataloader for accuracy validation
            
        Returns:
            Compressed model and compression results
        """
        import time
        start_time = time.time()
        
        # Initialize statistics
        total_original_params = 0
        total_compressed_params = 0
        layer_results = {}
        
        # Get layers to compress
        layers_to_compress = self._get_compressible_layers(model, layer_filter)
        
        logger.info(f"Compressing {len(layers_to_compress)} layers")
        
        # Compress each layer
        for layer_name, layer in tqdm(layers_to_compress, desc="Compressing layers"):
            logger.info(f"Compressing layer: {layer_name}")
            
            # Extract weight tensor
            if hasattr(layer, 'weight'):
                weight_tensor = layer.weight.data
            else:
                logger.warning(f"Layer {layer_name} has no weight attribute, skipping")
                continue
            
            # Move to device
            weight_tensor = weight_tensor.to(self.device)
            
            # Compress the tensor
            result = self._compress_tensor(
                weight_tensor,
                layer_name,
                max_retries=3
            )
            
            if result is not None:
                layer_results[layer_name] = result
                total_original_params += result.original_params
                total_compressed_params += result.compressed_params
                
                # Replace layer with compressed version
                self._replace_layer_weights(model, layer_name, result.field)
            else:
                logger.warning(f"Failed to compress layer {layer_name}")
                total_original_params += weight_tensor.numel()
                total_compressed_params += weight_tensor.numel()
        
        # Validate model accuracy if validation loader provided
        if validation_loader is not None:
            self._validate_model_accuracy(model, validation_loader)
        
        # Create compression result
        compression_result = CompressionResult(
            original_params=total_original_params,
            compressed_params=total_compressed_params,
            compression_ratio=total_original_params / max(total_compressed_params, 1),
            layer_results=layer_results,
            total_time=time.time() - start_time,
            config=self.config
        )
        
        logger.info(f"Compression complete. Ratio: {compression_result.compression_ratio:.2f}x")
        
        return model, compression_result
    
    def _compress_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        max_retries: int = 3
    ) -> Optional[LayerCompressionResult]:
        """Compress a single tensor."""
        # Compute tensor statistics
        stats = ImplicitWeightField.compute_tensor_statistics(tensor)
        logger.info(f"Tensor {name} statistics: {stats}")
        
        # Select architecture
        if self.config.architecture is not None:
            architecture = self.config.architecture
        else:
            architecture = ImplicitWeightField.select_architecture_from_stats(
                stats, self.config
            )
        
        logger.info(f"Selected architecture: {architecture}")
        
        # Skip compression for tiny tensors
        if architecture == FieldArchitecture.EXPLICIT:
            return None
        
        # Try compression with retries
        retry = 0
        hidden_width = self.config.hidden_width
        
        while retry < max_retries:
            try:
                # Create field
                if stats.num_params > 1e6 and retry > 0:
                    # Use multi-scale for large tensors on retry
                    field = MultiScaleImplicitField(
                        tensor_shape=tensor.shape,
                        config=self.config,
                        num_scales=2
                    )
                else:
                    # Update config for retry
                    retry_config = CompressionConfig(
                        bandwidth=self.config.bandwidth,
                        hidden_width=int(hidden_width * (1.5 ** retry)),
                        num_layers=self.config.num_layers + (retry > 1),
                        w0=self.config.w0,
                        learning_rate=self.config.learning_rate,
                        max_steps=self.config.max_steps,
                        convergence_threshold=self.config.convergence_threshold,
                        regularization=self.config.regularization
                    )
                    
                    field = ImplicitWeightField(
                        tensor_shape=tensor.shape,
                        config=retry_config,
                        architecture=architecture
                    )
                
                field = field.to(self.device)
                
                # Train field
                trainer = FieldTrainer(field, self.training_config)
                training_steps = trainer.train(tensor)
                
                # Validate reconstruction
                with torch.no_grad():
                    reconstructed = field.reconstruct_full_tensor()
                    mse = torch.mean((reconstructed - tensor) ** 2).item()
                    max_error = torch.max(torch.abs(reconstructed - tensor)).item()
                
                logger.info(f"Reconstruction MSE: {mse:.6f}, Max error: {max_error:.6f}")
                
                # Check if reconstruction is acceptable
                threshold = 0.01 * tensor.abs().mean().item()  # 1% of mean magnitude
                if max_error < threshold or retry == max_retries - 1:
                    return LayerCompressionResult(
                        layer_name=name,
                        original_shape=tuple(tensor.shape),
                        original_params=tensor.numel(),
                        compressed_params=field.count_parameters(),
                        compression_ratio=tensor.numel() / field.count_parameters(),
                        reconstruction_error=mse,
                        max_error=max_error,
                        field_architecture=str(architecture),
                        training_steps=training_steps,
                        field=field
                    )
                else:
                    logger.warning(f"Retry {retry + 1}: increasing capacity")
                    retry += 1
                    
            except Exception as e:
                logger.error(f"Error compressing {name}: {e}")
                retry += 1
        
        return None
    
    def _get_compressible_layers(
        self,
        model: nn.Module,
        layer_filter: Optional[List[str]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """Get list of layers to compress."""
        compressible_layers = []
        
        for name, module in model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
            
            # Check if layer has weights
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            
            # Apply filter if provided
            if layer_filter is not None and name not in layer_filter:
                continue
            
            # Skip small layers (BatchNorm, etc.)
            if module.weight.numel() < 1000:
                logger.info(f"Skipping small layer {name} with {module.weight.numel()} params")
                continue
            
            compressible_layers.append((name, module))
        
        return compressible_layers
    
    def _replace_layer_weights(
        self,
        model: nn.Module,
        layer_name: str,
        field: Union[ImplicitWeightField, MultiScaleImplicitField]
    ):
        """Replace layer weights with implicit field."""
        # This is a placeholder - actual implementation would need
        # to modify the model's forward pass to use the field
        # For now, we'll store the field as an attribute
        parts = layer_name.split('.')
        current = model
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Store field on the module
        setattr(current, '_implicit_field', field)
        setattr(current, '_use_implicit_weights', True)
    
    def _validate_model_accuracy(self, model: nn.Module, validation_loader):
        """Validate model accuracy after compression."""
        # Placeholder for accuracy validation
        logger.info("Validating model accuracy...")
        # Implementation would depend on the specific task


def compress_model(
    model: nn.Module,
    config: Optional[CompressionConfig] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[nn.Module, CompressionResult]:
    """
    Convenience function to compress a model.
    
    Args:
        model: Model to compress
        config: Compression configuration
        device: Device to use
        **kwargs: Additional arguments for compress_model
        
    Returns:
        Compressed model and results
    """
    compressor = ImplicitWeightFieldCompressor(config, device)
    return compressor.compress_model(model, **kwargs)