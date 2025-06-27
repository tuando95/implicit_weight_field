"""Inference modes for implicit weight fields."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from enum import Enum
from abc import ABC, abstractmethod
import logging

from .cache import LRUWeightCache, CacheConfig
from core.implicit_field import ImplicitWeightField, MultiScaleImplicitField


logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Available inference modes."""
    PRELOAD = "preload"  # Reconstruct all weights at initialization
    STREAMING = "streaming"  # Reconstruct weights on-demand with caching


class BaseInference(ABC):
    """Base class for inference modes."""
    
    @abstractmethod
    def get_weight(self, layer_id: str, field: Any) -> torch.Tensor:
        """Get weight tensor for a layer."""
        pass
    
    @abstractmethod
    def prepare(self, fields: Dict[str, Any]):
        """Prepare for inference with given fields."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        pass


class PreloadInference(BaseInference):
    """Preload mode: reconstruct all weights at initialization."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize preload inference.
        
        Args:
            device: Device to store reconstructed weights
        """
        self.device = device or torch.device('cpu')
        self.reconstructed_weights = {}
        self._prepared = False
    
    def prepare(self, fields: Dict[str, Any]):
        """
        Prepare by reconstructing all weights.
        
        Args:
            fields: Dictionary mapping layer IDs to implicit fields
        """
        logger.info(f"Preloading weights for {len(fields)} layers")
        
        for layer_id, field in fields.items():
            logger.debug(f"Reconstructing layer {layer_id}")
            
            # Reconstruct full tensor
            with torch.no_grad():
                if isinstance(field, (ImplicitWeightField, MultiScaleImplicitField)):
                    weights = field.reconstruct_full_tensor()
                else:
                    # Handle custom field types
                    weights = field.reconstruct_full_tensor()
                
                # Move to device
                self.reconstructed_weights[layer_id] = weights.to(self.device)
        
        self._prepared = True
        logger.info(f"Preloading complete. Memory usage: {self.get_memory_usage():.1f} MB")
    
    def get_weight(self, layer_id: str, field: Any = None) -> torch.Tensor:
        """
        Get reconstructed weight tensor.
        
        Args:
            layer_id: Layer identifier
            field: Not used in preload mode
            
        Returns:
            Weight tensor
        """
        if not self._prepared:
            raise RuntimeError("Inference not prepared. Call prepare() first.")
        
        if layer_id not in self.reconstructed_weights:
            raise KeyError(f"Layer {layer_id} not found in reconstructed weights")
        
        return self.reconstructed_weights[layer_id]
    
    def get_memory_usage(self) -> float:
        """Get memory usage of reconstructed weights in MB."""
        total_bytes = 0
        for weights in self.reconstructed_weights.values():
            total_bytes += weights.element_size() * weights.numel()
        return total_bytes / (1024 * 1024)
    
    def clear(self):
        """Clear reconstructed weights."""
        self.reconstructed_weights.clear()
        self._prepared = False


class StreamingInference(BaseInference):
    """Streaming mode: reconstruct weights on-demand with LRU caching."""
    
    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize streaming inference.
        
        Args:
            cache_config: Cache configuration
            device: Device for reconstruction
        """
        self.device = device or torch.device('cpu')
        self.cache_config = cache_config or CacheConfig(device=self.device)
        self.cache = LRUWeightCache(self.cache_config)
        self.fields = {}
        self._prepared = False
    
    def prepare(self, fields: Dict[str, Any]):
        """
        Prepare fields for streaming reconstruction.
        
        Args:
            fields: Dictionary mapping layer IDs to implicit fields
        """
        self.fields = fields
        self._prepared = True
        logger.info(f"Streaming inference ready for {len(fields)} layers")
    
    def get_weight(self, layer_id: str, field: Any = None) -> torch.Tensor:
        """
        Get weight tensor, reconstructing as needed.
        
        Args:
            layer_id: Layer identifier
            field: Optional field override
            
        Returns:
            Weight tensor
        """
        if not self._prepared:
            raise RuntimeError("Inference not prepared. Call prepare() first.")
        
        # Use provided field or lookup
        if field is None:
            if layer_id not in self.fields:
                raise KeyError(f"Layer {layer_id} not found")
            field = self.fields[layer_id]
        
        # For now, reconstruct full tensor
        # TODO: Implement partial reconstruction for specific operations
        with torch.no_grad():
            weights = field.reconstruct_full_tensor()
        
        return weights.to(self.device)
    
    def get_weight_at_coords(
        self,
        layer_id: str,
        coords: torch.Tensor,
        tensor_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Get specific weight values at coordinates.
        
        Args:
            layer_id: Layer identifier
            coords: Coordinates to access
            tensor_shape: Shape of the full tensor
            
        Returns:
            Weight values
        """
        if layer_id not in self.fields:
            raise KeyError(f"Layer {layer_id} not found")
        
        field = self.fields[layer_id]
        
        # Use cache for coordinate access
        if coords.shape[0] == 1:
            # Single coordinate
            coord_tuple = tuple(coords[0].tolist())
            return self.cache.get(layer_id, coord_tuple, field, tensor_shape)
        else:
            # Batch of coordinates
            return self.cache.get_batch(layer_id, coords, field, tensor_shape)
    
    def get_memory_usage(self) -> float:
        """Get current cache memory usage in MB."""
        return self.cache.get_stats()['current_size_mb']
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
    
    def clear(self):
        """Clear all state."""
        self.fields.clear()
        self.cache.clear()
        self._prepared = False


def create_inference_mode(
    mode: InferenceMode,
    device: Optional[torch.device] = None,
    cache_config: Optional[CacheConfig] = None
) -> BaseInference:
    """
    Factory function to create inference mode.
    
    Args:
        mode: Inference mode to use
        device: Device for inference
        cache_config: Cache configuration for streaming mode
        
    Returns:
        Inference mode instance
    """
    if mode == InferenceMode.PRELOAD:
        return PreloadInference(device=device)
    elif mode == InferenceMode.STREAMING:
        return StreamingInference(cache_config=cache_config, device=device)
    else:
        raise ValueError(f"Unknown inference mode: {mode}")