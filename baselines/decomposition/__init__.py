"""Matrix/tensor decomposition baseline methods."""

from .decompose import (
    tensor_train_decomposition,
    low_rank_factorization,
    svd_decomposition,
    DecompositionConfig
)

__all__ = [
    'tensor_train_decomposition',
    'low_rank_factorization', 
    'svd_decomposition',
    'DecompositionConfig'
]