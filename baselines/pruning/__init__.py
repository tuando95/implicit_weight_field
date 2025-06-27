"""Pruning baseline methods."""

from .prune import (
    magnitude_prune,
    gradual_magnitude_prune,
    structured_prune,
    PruningConfig
)

__all__ = [
    'magnitude_prune',
    'gradual_magnitude_prune', 
    'structured_prune',
    'PruningConfig'
]