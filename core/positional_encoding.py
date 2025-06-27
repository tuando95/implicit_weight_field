"""Positional encoding utilities for implicit weight fields."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def positional_encoding(
    coords: torch.Tensor,
    bandwidth: int = 4,
    include_input: bool = False
) -> torch.Tensor:
    """
    Apply Fourier feature positional encoding to coordinates.
    
    As defined in CLAUDE.md:
    γ(c̃) = [sin(2^0πc̃), cos(2^0πc̃), sin(2^1πc̃), cos(2^1πc̃), ..., sin(2^(B-1)πc̃), cos(2^(B-1)πc̃)]
    
    Args:
        coords: Normalized coordinates in [0, 1], shape (..., coord_dim)
        bandwidth: Number of frequency bands B
        include_input: Whether to concatenate original coordinates
        
    Returns:
        Encoded coordinates with shape (..., 2 * bandwidth * coord_dim) or
        (..., (2 * bandwidth + 1) * coord_dim) if include_input=True
    """
    if bandwidth == 0:
        return coords
    
    # Generate frequency bands: 2^0, 2^1, ..., 2^(B-1)
    freqs = 2.0 ** torch.arange(bandwidth, dtype=coords.dtype, device=coords.device)
    
    # Reshape for broadcasting: (1, 1, bandwidth)
    freqs = freqs.view(1, 1, -1)
    
    # Expand coordinates for frequency multiplication: (..., coord_dim, 1)
    coords_expanded = coords.unsqueeze(-1)
    
    # Apply frequencies: (..., coord_dim, bandwidth)
    scaled_coords = np.pi * coords_expanded * freqs
    
    # Compute sin and cos: (..., coord_dim, bandwidth, 2)
    encoded = torch.stack([torch.sin(scaled_coords), torch.cos(scaled_coords)], dim=-1)
    
    # Flatten last dimensions: (..., coord_dim * bandwidth * 2)
    encoded = encoded.view(*coords.shape[:-1], -1)
    
    if include_input:
        # Concatenate original coordinates
        encoded = torch.cat([coords, encoded], dim=-1)
    
    return encoded


class FourierFeatures(nn.Module):
    """Fourier feature encoding module."""
    
    def __init__(
        self,
        input_dim: int,
        bandwidth: int = 4,
        include_input: bool = False,
        learnable: bool = False
    ):
        """
        Initialize Fourier feature encoder.
        
        Args:
            input_dim: Dimension of input coordinates
            bandwidth: Number of frequency bands
            include_input: Whether to concatenate original coordinates
            learnable: Whether to make frequencies learnable
        """
        super().__init__()
        self.input_dim = input_dim
        self.bandwidth = bandwidth
        self.include_input = include_input
        self.learnable = learnable
        
        if learnable and bandwidth > 0:
            # Initialize learnable frequencies
            init_freqs = 2.0 ** torch.arange(bandwidth, dtype=torch.float32)
            self.frequencies = nn.Parameter(init_freqs)
        else:
            self.register_buffer(
                'frequencies',
                2.0 ** torch.arange(bandwidth, dtype=torch.float32) if bandwidth > 0 else torch.tensor([])
            )
        
        # Calculate output dimension
        self.output_dim = input_dim * bandwidth * 2
        if include_input:
            self.output_dim += input_dim
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.
        
        Args:
            coords: Input coordinates (..., input_dim)
            
        Returns:
            Encoded features (..., output_dim)
        """
        if self.bandwidth == 0:
            return coords
        
        # Reshape frequencies for broadcasting
        freqs = self.frequencies.view(1, 1, -1)
        
        # Expand coordinates
        coords_expanded = coords.unsqueeze(-1)
        
        # Apply frequencies
        scaled_coords = np.pi * coords_expanded * freqs
        
        # Compute sin and cos
        encoded = torch.stack([torch.sin(scaled_coords), torch.cos(scaled_coords)], dim=-1)
        
        # Flatten
        encoded = encoded.view(*coords.shape[:-1], -1)
        
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        
        return encoded
    
    def extra_repr(self) -> str:
        """String representation of module."""
        return f'input_dim={self.input_dim}, bandwidth={self.bandwidth}, ' \
               f'include_input={self.include_input}, learnable={self.learnable}'


def normalize_coordinates(
    indices: torch.Tensor,
    tensor_shape: Tuple[int, ...],
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize discrete tensor indices to [0, 1] range.
    
    As defined in CLAUDE.md:
    c̃ = ((c_1 - 1) / max(d_1 - 1, 1), ..., (c_k - 1) / max(d_k - 1, 1))
    
    Args:
        indices: Tensor indices, shape (..., num_dims) with 0-based indexing
        tensor_shape: Shape of the original tensor
        eps: Small value to avoid division issues
        
    Returns:
        Normalized coordinates in [0, 1]^k
    """
    device = indices.device
    dtype = indices.dtype if indices.dtype.is_floating_point else torch.float32
    
    # Convert shape to tensor for computation
    shape_tensor = torch.tensor(tensor_shape, device=device, dtype=dtype)
    
    # Normalize: indices / (shape - 1), handling single-dimension case
    denominators = torch.maximum(shape_tensor - 1, torch.ones_like(shape_tensor))
    
    # Cast indices to float if needed
    if not indices.dtype.is_floating_point:
        indices = indices.float()
    
    normalized = indices / denominators.view(1, -1)
    
    # Clamp to [0, 1] to handle any numerical issues
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    return normalized


def generate_coordinate_grid(
    tensor_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate all coordinates for a tensor and their normalized versions.
    
    Args:
        tensor_shape: Shape of the tensor
        device: Device to place tensors on
        dtype: Data type for coordinates
        
    Returns:
        Tuple of (indices, normalized_coords) where:
        - indices: Integer indices of shape (num_elements, num_dims)
        - normalized_coords: Normalized coordinates in [0, 1]^k
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    
    # Generate coordinate grids for each dimension
    coord_grids = torch.meshgrid(
        *[torch.arange(dim, device=device) for dim in tensor_shape],
        indexing='ij'
    )
    
    # Stack and reshape to (num_elements, num_dims)
    indices = torch.stack(coord_grids, dim=-1).reshape(-1, len(tensor_shape))
    
    # Normalize coordinates
    normalized_coords = normalize_coordinates(indices, tensor_shape).to(dtype)
    
    return indices, normalized_coords


class HashEncoding(nn.Module):
    """Hash-based positional encoding for large coordinate spaces."""
    
    def __init__(
        self,
        input_dim: int,
        num_levels: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 512,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2
    ):
        """
        Initialize hash encoding.
        
        Args:
            input_dim: Dimension of input coordinates
            num_levels: Number of resolution levels
            base_resolution: Base resolution
            max_resolution: Maximum resolution
            log2_hashmap_size: Log2 of hash table size
            features_per_level: Features per resolution level
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Calculate resolutions for each level
        self.resolutions = []
        for i in range(num_levels):
            resolution = int(base_resolution * ((max_resolution / base_resolution) ** (i / (num_levels - 1))))
            self.resolutions.append(resolution)
        
        # Initialize hash tables
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, features_per_level)
            for _ in range(num_levels)
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
        
        self.output_dim = num_levels * features_per_level
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply hash encoding to coordinates.
        
        Args:
            coords: Normalized coordinates in [0, 1], shape (..., input_dim)
            
        Returns:
            Encoded features (..., output_dim)
        """
        encoded_features = []
        
        for level, resolution in enumerate(self.resolutions):
            # Scale coordinates to grid resolution
            scaled_coords = coords * resolution
            
            # Get integer grid coordinates
            grid_coords = torch.floor(scaled_coords).long()
            
            # Compute hash indices
            hash_indices = self._compute_hash(grid_coords) % self.hashmap_size
            
            # Look up features
            features = self.embeddings[level](hash_indices)
            encoded_features.append(features)
        
        # Concatenate all levels
        return torch.cat(encoded_features, dim=-1)
    
    def _compute_hash(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute spatial hash for coordinates."""
        primes = torch.tensor([1, 2654435761, 805459861, 3674653429],
                            device=coords.device, dtype=torch.long)
        
        result = torch.zeros(coords.shape[:-1], device=coords.device, dtype=torch.long)
        
        for i in range(min(coords.shape[-1], len(primes))):
            result ^= coords[..., i] * primes[i]
        
        return result