"""Implicit weight field implementation with adaptive architecture selection."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum

from .siren import SIREN
from .positional_encoding import (
    FourierFeatures,
    positional_encoding,
    normalize_coordinates,
    generate_coordinate_grid
)


class FieldArchitecture(Enum):
    """Architecture types for implicit fields."""
    EXPLICIT = "explicit"  # Store weights directly (tiny tensors)
    LINEAR_1L = "linear_1l"  # 1-layer linear field
    SIREN_1L = "siren_1l"  # 1-layer SIREN
    SIREN_2L = "siren_2l"  # 2-layer SIREN (standard)
    SIREN_3L = "siren_3l"  # 3-layer SIREN
    MULTISCALE = "multiscale"  # Multi-scale decomposition
    HYBRID = "hybrid"  # Hybrid sparse + implicit


@dataclass
class TensorStatistics:
    """Statistics about a weight tensor."""
    shape: Tuple[int, ...]
    num_params: int
    effective_rank: float
    sparsity: float
    smoothness: float
    value_range: Tuple[float, float]
    std_dev: float


@dataclass
class CompressionConfig:
    """Configuration for compression."""
    bandwidth: int = 4
    hidden_width: int = 256
    num_layers: int = 2
    w0: float = 30.0
    learning_rate: float = 1e-3
    max_steps: int = 2000
    convergence_threshold: float = 1e-6
    regularization: float = 1e-6
    architecture: Optional[FieldArchitecture] = None


class ImplicitWeightField(nn.Module):
    """Implicit representation of a weight tensor."""
    
    def __init__(
        self,
        tensor_shape: Tuple[int, ...],
        config: CompressionConfig,
        architecture: Optional[FieldArchitecture] = None
    ):
        """
        Initialize implicit weight field.
        
        Args:
            tensor_shape: Shape of the tensor to represent
            config: Compression configuration
            architecture: Specific architecture to use (overrides adaptive selection)
        """
        super().__init__()
        self.tensor_shape = tensor_shape
        self.config = config
        self.num_params = np.prod(tensor_shape)
        
        # Positional encoding
        self.encoder = FourierFeatures(
            input_dim=len(tensor_shape),
            bandwidth=config.bandwidth,
            include_input=False
        )
        
        # Select architecture
        if architecture is not None:
            self.architecture = architecture
        else:
            self.architecture = self._select_architecture()
        
        # Build field network
        self._build_field()
        
        # Cached reconstruction for preload mode
        self._cached_weights = None
    
    def _select_architecture(self) -> FieldArchitecture:
        """Select architecture based on tensor properties."""
        num_params = self.num_params
        
        # Tiny tensors: store explicitly
        if num_params < 1000:
            return FieldArchitecture.EXPLICIT
        
        # Small tensors: 1-layer linear
        elif num_params < 10000:
            return FieldArchitecture.LINEAR_1L
        
        # Medium tensors: 2-layer SIREN (standard)
        elif num_params < 1000000:
            return FieldArchitecture.SIREN_2L
        
        # Large tensors: 2-layer SIREN with larger width
        else:
            return FieldArchitecture.SIREN_2L
    
    def _build_field(self):
        """Build the field network based on selected architecture."""
        if self.architecture == FieldArchitecture.EXPLICIT:
            # Direct parameter storage
            self.explicit_weights = nn.Parameter(
                torch.zeros(self.tensor_shape)
            )
            self.field = None
            
        elif self.architecture == FieldArchitecture.LINEAR_1L:
            # Single linear layer
            in_features = self.encoder.output_dim
            self.field = nn.Linear(in_features, 1, bias=True)
            
        elif self.architecture == FieldArchitecture.SIREN_1L:
            # 1-layer SIREN
            self.field = SIREN(
                in_features=self.encoder.output_dim,
                hidden_features=self.config.hidden_width,
                out_features=1,
                num_layers=1,
                w0_initial=self.config.w0
            )
            
        elif self.architecture == FieldArchitecture.SIREN_2L:
            # 2-layer SIREN (standard)
            # Adjust hidden width based on tensor size
            hidden_width = self.config.hidden_width
            if self.num_params >= 1000000:
                hidden_width = 512
            
            self.field = SIREN(
                in_features=self.encoder.output_dim,
                hidden_features=hidden_width,
                out_features=1,
                num_layers=2,
                w0_initial=self.config.w0
            )
            
        elif self.architecture == FieldArchitecture.SIREN_3L:
            # 3-layer SIREN for complex patterns
            self.field = SIREN(
                in_features=self.encoder.output_dim,
                hidden_features=self.config.hidden_width,
                out_features=1,
                num_layers=3,
                w0_initial=self.config.w0
            )
            
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct weights at given coordinates.
        
        Args:
            coords: Tensor coordinates, shape (..., num_dims)
            
        Returns:
            Weight values at coordinates
        """
        if self.architecture == FieldArchitecture.EXPLICIT:
            # Direct indexing for explicit storage
            if coords.dtype == torch.long:
                # Integer indexing
                return self.explicit_weights[tuple(coords.T)]
            else:
                # Normalized coordinates - denormalize and index
                indices = (coords * torch.tensor(self.tensor_shape).to(coords.device)).long()
                return self.explicit_weights[tuple(indices.T)]
        
        # Normalize coordinates if needed
        if coords.dtype == torch.long:
            coords = normalize_coordinates(coords, self.tensor_shape)
        
        # Apply positional encoding
        encoded = self.encoder(coords)
        
        # Pass through field network
        weights = self.field(encoded).squeeze(-1)
        
        return weights
    
    def reconstruct_full_tensor(self) -> torch.Tensor:
        """Reconstruct the complete weight tensor."""
        if self._cached_weights is not None:
            return self._cached_weights
        
        if self.architecture == FieldArchitecture.EXPLICIT:
            return self.explicit_weights
        
        # Generate all coordinates
        indices, normalized_coords = generate_coordinate_grid(
            self.tensor_shape,
            device=next(self.parameters()).device
        )
        
        # Reconstruct all weights
        with torch.no_grad():
            weights = self(normalized_coords)
        
        # Reshape to original tensor shape
        return weights.reshape(self.tensor_shape)
    
    def cache_weights(self):
        """Cache reconstructed weights for preload mode."""
        self._cached_weights = self.reconstruct_full_tensor()
    
    def clear_cache(self):
        """Clear cached weights."""
        self._cached_weights = None
    
    def count_parameters(self) -> int:
        """Count total parameters in the field."""
        if self.architecture == FieldArchitecture.EXPLICIT:
            return self.num_params
        return sum(p.numel() for p in self.parameters())
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.num_params / self.count_parameters()
    
    @staticmethod
    def compute_tensor_statistics(tensor: torch.Tensor) -> TensorStatistics:
        """Compute statistics about a tensor."""
        # Flatten tensor for analysis
        flat_tensor = tensor.flatten()
        
        # Basic statistics
        num_params = tensor.numel()
        value_range = (tensor.min().item(), tensor.max().item())
        std_dev = tensor.std().item()
        
        # Sparsity
        sparsity = (torch.abs(flat_tensor) < 1e-6).float().mean().item()
        
        # Effective rank via SVD on matricized tensor
        if len(tensor.shape) > 1:
            matrix = tensor.reshape(tensor.shape[0], -1)
            try:
                s = torch.linalg.svdvals(matrix)
                # Effective rank: sum(s)^2 / sum(s^2)
                effective_rank = (s.sum() ** 2 / (s ** 2).sum()).item()
            except:
                effective_rank = min(matrix.shape)
        else:
            effective_rank = 1.0
        
        # Smoothness via finite differences
        if len(tensor.shape) > 1:
            # Compute gradients along first dimension
            grad = torch.diff(tensor, dim=0)
            smoothness = grad.abs().mean().item() / (std_dev + 1e-8)
        else:
            smoothness = 0.0
        
        return TensorStatistics(
            shape=tuple(tensor.shape),
            num_params=num_params,
            effective_rank=effective_rank,
            sparsity=sparsity,
            smoothness=smoothness,
            value_range=value_range,
            std_dev=std_dev
        )
    
    @staticmethod
    def select_architecture_from_stats(
        stats: TensorStatistics,
        config: CompressionConfig
    ) -> FieldArchitecture:
        """Select architecture based on tensor statistics."""
        # Tiny tensors
        if stats.num_params < 1000:
            return FieldArchitecture.EXPLICIT
        
        # High sparsity: consider hybrid approach
        if stats.sparsity > 0.9:
            # For now, use standard approach
            # TODO: Implement hybrid sparse + implicit
            return FieldArchitecture.SIREN_2L
        
        # Low-rank tensors
        if stats.effective_rank / min(stats.shape) < 0.1:
            return FieldArchitecture.SIREN_1L
        
        # Small tensors
        if stats.num_params < 10000:
            return FieldArchitecture.LINEAR_1L
        
        # Standard tensors
        if stats.num_params < 1000000:
            return FieldArchitecture.SIREN_2L
        
        # Large complex tensors
        return FieldArchitecture.SIREN_2L


class MultiScaleImplicitField(nn.Module):
    """Multi-scale implicit field for large tensors."""
    
    def __init__(
        self,
        tensor_shape: Tuple[int, ...],
        config: CompressionConfig,
        num_scales: int = 2
    ):
        """
        Initialize multi-scale field.
        
        Args:
            tensor_shape: Shape of tensor to represent
            config: Compression configuration
            num_scales: Number of scales
        """
        super().__init__()
        self.tensor_shape = tensor_shape
        self.config = config
        self.num_scales = num_scales
        
        # Base field captures low frequencies
        self.base_field = ImplicitWeightField(
            tensor_shape=tensor_shape,
            config=config,
            architecture=FieldArchitecture.SIREN_2L
        )
        
        # Detail fields capture progressively higher frequencies
        self.detail_fields = nn.ModuleList()
        self.scale_weights = nn.ParameterList()
        
        for s in range(1, num_scales):
            # Create detail field with same architecture
            detail_field = ImplicitWeightField(
                tensor_shape=tensor_shape,
                config=config,
                architecture=FieldArchitecture.SIREN_2L
            )
            self.detail_fields.append(detail_field)
            
            # Learnable scale weight, initialized as 2^(-s)
            scale_weight = nn.Parameter(torch.tensor(2.0 ** (-s)))
            self.scale_weights.append(scale_weight)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct weights using multi-scale decomposition.
        
        Args:
            coords: Tensor coordinates
            
        Returns:
            Weight values
        """
        # Base reconstruction
        weights = self.base_field(coords)
        
        # Add detail at each scale
        for s, (detail_field, scale_weight) in enumerate(zip(self.detail_fields, self.scale_weights)):
            # Scale coordinates for higher frequency
            scale_factor = 2 ** (s + 1)
            scaled_coords = coords * scale_factor
            
            # Wrap coordinates for tiling
            if coords.dtype != torch.long:
                scaled_coords = scaled_coords % 1.0
            
            # Add scaled detail
            detail = detail_field(scaled_coords)
            weights = weights + scale_weight * detail
        
        return weights
    
    def reconstruct_full_tensor(self) -> torch.Tensor:
        """Reconstruct complete tensor."""
        indices, normalized_coords = generate_coordinate_grid(
            self.tensor_shape,
            device=next(self.parameters()).device
        )
        
        with torch.no_grad():
            weights = self(normalized_coords)
        
        return weights.reshape(self.tensor_shape)
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        total = self.base_field.count_parameters()
        for field in self.detail_fields:
            total += field.count_parameters()
        total += len(self.scale_weights)
        return total
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return np.prod(self.tensor_shape) / self.count_parameters()