"""LRU cache implementation for streaming weight reconstruction."""

import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import OrderedDict
import logging


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for weight cache."""
    max_size_mb: float = 100.0  # Maximum cache size in MB
    prefetch_neighbors: bool = True  # Prefetch neighboring weights
    prefetch_radius: int = 1  # Radius for prefetching
    batch_size: int = 128  # Batch size for reconstruction
    device: Optional[torch.device] = None


class LRUWeightCache:
    """LRU cache for reconstructed weights (Algorithm 2 from CLAUDE.md)."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize LRU cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.device = config.device or torch.device('cpu')
        
        # Convert MB to number of float32 elements
        self.max_elements = int(config.max_size_mb * 1024 * 1024 / 4)
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[Tuple, torch.Tensor] = OrderedDict()
        self.current_size = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(
        self,
        layer_id: str,
        coord: Tuple[int, ...],
        field: Any,
        tensor_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Get weight value from cache or reconstruct.
        
        Args:
            layer_id: Layer identifier
            coord: Weight coordinate
            field: Implicit field for reconstruction
            tensor_shape: Shape of the full tensor
            
        Returns:
            Weight value
        """
        # Create cache key
        key = (layer_id, coord)
        
        # Check cache
        if key in self.cache:
            # Cache hit - move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        # Cache miss
        self.misses += 1
        
        # Reconstruct weight
        weight = self._reconstruct_weight(coord, field, tensor_shape)
        
        # Add to cache
        self._add_to_cache(key, weight)
        
        # Prefetch neighbors if enabled
        if self.config.prefetch_neighbors:
            self._prefetch_neighbors(layer_id, coord, field, tensor_shape)
        
        return weight
    
    def get_batch(
        self,
        layer_id: str,
        coords: torch.Tensor,
        field: Any,
        tensor_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Get batch of weights, using cache where possible.
        
        Args:
            layer_id: Layer identifier
            coords: Batch of coordinates (N, num_dims)
            field: Implicit field
            tensor_shape: Tensor shape
            
        Returns:
            Batch of weight values
        """
        weights = []
        uncached_indices = []
        uncached_coords = []
        
        # Check cache for each coordinate
        for i, coord in enumerate(coords):
            coord_tuple = tuple(coord.tolist())
            key = (layer_id, coord_tuple)
            
            if key in self.cache:
                self.cache.move_to_end(key)
                weights.append(self.cache[key])
                self.hits += 1
            else:
                weights.append(None)
                uncached_indices.append(i)
                uncached_coords.append(coord)
                self.misses += 1
        
        # Batch reconstruct uncached weights
        if uncached_coords:
            uncached_coords = torch.stack(uncached_coords)
            reconstructed = self._reconstruct_batch(
                uncached_coords, field, tensor_shape
            )
            
            # Fill in reconstructed weights and add to cache
            for idx, (i, coord) in enumerate(zip(uncached_indices, uncached_coords)):
                weight = reconstructed[idx]
                weights[i] = weight
                
                coord_tuple = tuple(coord.tolist())
                key = (layer_id, coord_tuple)
                self._add_to_cache(key, weight)
        
        return torch.stack([w for w in weights])
    
    def _reconstruct_weight(
        self,
        coord: Tuple[int, ...],
        field: Any,
        tensor_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Reconstruct single weight value."""
        # Convert to tensor
        coord_tensor = torch.tensor(coord, device=self.device).unsqueeze(0)
        
        # Normalize coordinates
        from ..core.positional_encoding import normalize_coordinates
        normalized = normalize_coordinates(coord_tensor, tensor_shape)
        
        # Reconstruct
        with torch.no_grad():
            weight = field(normalized).squeeze()
        
        return weight
    
    def _reconstruct_batch(
        self,
        coords: torch.Tensor,
        field: Any,
        tensor_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Reconstruct batch of weights."""
        # Normalize coordinates
        from ..core.positional_encoding import normalize_coordinates
        normalized = normalize_coordinates(coords, tensor_shape)
        
        # Reconstruct in batches if needed
        weights = []
        for i in range(0, len(normalized), self.config.batch_size):
            batch = normalized[i:i + self.config.batch_size]
            with torch.no_grad():
                batch_weights = field(batch)
            weights.append(batch_weights)
        
        return torch.cat(weights)
    
    def _add_to_cache(self, key: Tuple, value: torch.Tensor):
        """Add value to cache with LRU eviction."""
        # Check if key already exists
        if key in self.cache:
            # Update value and move to end
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        
        # Check capacity
        value_size = value.numel()
        
        # Evict if necessary
        while self.current_size + value_size > self.max_elements and self.cache:
            # Remove least recently used
            evicted_key, evicted_value = self.cache.popitem(last=False)
            self.current_size -= evicted_value.numel()
            self.evictions += 1
        
        # Add new value
        self.cache[key] = value
        self.current_size += value_size
    
    def _prefetch_neighbors(
        self,
        layer_id: str,
        coord: Tuple[int, ...],
        field: Any,
        tensor_shape: Tuple[int, ...]
    ):
        """Prefetch neighboring coordinates."""
        neighbors = []
        
        # Generate neighboring coordinates
        for dim in range(len(coord)):
            for offset in [-1, 1]:
                neighbor = list(coord)
                neighbor[dim] += offset * self.config.prefetch_radius
                
                # Check bounds
                if 0 <= neighbor[dim] < tensor_shape[dim]:
                    neighbor_tuple = tuple(neighbor)
                    key = (layer_id, neighbor_tuple)
                    
                    # Only prefetch if not in cache
                    if key not in self.cache:
                        neighbors.append(neighbor)
        
        # Batch reconstruct neighbors
        if neighbors:
            neighbors_tensor = torch.tensor(neighbors, device=self.device)
            reconstructed = self._reconstruct_batch(
                neighbors_tensor, field, tensor_shape
            )
            
            # Add to cache
            for coord, weight in zip(neighbors, reconstructed):
                key = (layer_id, tuple(coord))
                self._add_to_cache(key, weight)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'current_size_mb': self.current_size * 4 / (1024 * 1024),
            'max_size_mb': self.config.max_size_mb,
            'num_entries': len(self.cache)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return f"LRUWeightCache(hit_rate={stats['hit_rate']:.2%}, " \
               f"size={stats['current_size_mb']:.1f}/{stats['max_size_mb']:.1f}MB)"