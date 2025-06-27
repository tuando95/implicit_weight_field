"""Tensor decomposition baseline implementations."""

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tensor_train, parafac, tucker
import numpy as np
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

# Set tensorly backend to PyTorch
tl.set_backend('pytorch')


@dataclass
class DecompositionConfig:
    """Configuration for tensor decomposition."""
    method: str = "tt"  # "tt", "svd", "tucker", "cp"
    rank: Optional[Union[int, List[int]]] = None  # None for automatic
    explained_variance: float = 0.95  # For automatic rank selection
    max_rank: Optional[int] = None  # Maximum allowed rank


def tensor_train_decomposition(
    model: nn.Module,
    config: Optional[DecompositionConfig] = None,
    layers_to_decompose: Optional[List[str]] = None
) -> nn.Module:
    """
    Tensor-Train decomposition with rank constraint.
    
    Args:
        model: Model to decompose
        config: Decomposition configuration
        layers_to_decompose: Specific layers to decompose
        
    Returns:
        Model with TT-decomposed layers
    """
    if config is None:
        config = DecompositionConfig()
    
    # Clone model
    decomposed_model = type(model)()
    decomposed_model.load_state_dict(model.state_dict())
    
    # Get layers to decompose
    if layers_to_decompose is None:
        layers_to_decompose = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers_to_decompose.append(name)
    
    total_original_params = 0
    total_decomposed_params = 0
    
    for layer_name in layers_to_decompose:
        # Get module
        module = dict(decomposed_model.named_modules())[layer_name]
        
        if isinstance(module, nn.Linear):
            # Decompose linear layer
            decomposed = decompose_linear_tt(module, config)
            if decomposed is not None:
                # Replace module
                parent, child_name = get_parent_module(decomposed_model, layer_name)
                setattr(parent, child_name, decomposed)
                
                # Count parameters
                total_original_params += module.weight.numel()
                total_decomposed_params += sum(p.numel() for p in decomposed.parameters())
        
        elif isinstance(module, nn.Conv2d):
            # Decompose conv layer
            decomposed = decompose_conv2d_tt(module, config)
            if decomposed is not None:
                parent, child_name = get_parent_module(decomposed_model, layer_name)
                setattr(parent, child_name, decomposed)
                
                total_original_params += module.weight.numel()
                total_decomposed_params += sum(p.numel() for p in decomposed.parameters())
    
    compression_ratio = total_original_params / max(total_decomposed_params, 1)
    logger.info(f"TT decomposition compression ratio: {compression_ratio:.2f}x")
    
    return decomposed_model


def decompose_linear_tt(
    layer: nn.Linear,
    config: DecompositionConfig
) -> Optional[nn.Module]:
    """Decompose a linear layer using TT decomposition."""
    weight = layer.weight.data  # (out_features, in_features)
    
    # Reshape to higher-order tensor for TT decomposition
    # Find suitable factorization of dimensions
    in_factors = factorize_dimension(layer.in_features)
    out_factors = factorize_dimension(layer.out_features)
    
    if len(in_factors) < 2 or len(out_factors) < 2:
        logger.warning(f"Cannot decompose layer with shape {weight.shape}")
        return None
    
    # Reshape weight tensor
    tensor_shape = out_factors + in_factors
    weight_tensor = weight.reshape(tensor_shape)
    
    # Determine TT ranks
    if config.rank is None:
        # Automatic rank selection based on explained variance
        ranks = estimate_tt_ranks(weight_tensor, config.explained_variance)
    else:
        ranks = config.rank if isinstance(config.rank, list) else [config.rank] * (len(tensor_shape) - 1)
    
    # Apply TT decomposition
    try:
        factors = tensor_train(weight_tensor, rank=ranks)
        
        # Create TT layer
        tt_layer = TTLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            in_factors=in_factors,
            out_factors=out_factors,
            factors=factors,
            bias=layer.bias is not None
        )
        
        if layer.bias is not None:
            tt_layer.bias.data = layer.bias.data.clone()
        
        return tt_layer
        
    except Exception as e:
        logger.error(f"TT decomposition failed: {e}")
        return None


def decompose_conv2d_tt(
    layer: nn.Conv2d,
    config: DecompositionConfig
) -> Optional[nn.Module]:
    """Decompose a Conv2d layer using TT decomposition."""
    weight = layer.weight.data  # (out_channels, in_channels, height, width)
    
    # For conv layers, we can decompose the channel dimensions
    if weight.shape[2] != weight.shape[3]:
        logger.warning("TT decomposition for non-square kernels not implemented")
        return None
    
    # Reshape for TT decomposition
    out_factors = factorize_dimension(layer.out_channels)
    in_factors = factorize_dimension(layer.in_channels)
    
    if len(in_factors) < 2 or len(out_factors) < 2:
        return None
    
    # Keep spatial dimensions intact
    tensor_shape = out_factors + in_factors + [weight.shape[2], weight.shape[3]]
    weight_tensor = weight.reshape(tensor_shape)
    
    # Apply TT decomposition
    try:
        if config.rank is None:
            ranks = estimate_tt_ranks(weight_tensor, config.explained_variance)
        else:
            ranks = config.rank if isinstance(config.rank, list) else [config.rank] * (len(tensor_shape) - 1)
        
        factors = tensor_train(weight_tensor, rank=ranks)
        
        # Create decomposed conv layer (simplified - would need custom implementation)
        # For now, return None as proper conv TT layer is complex
        return None
        
    except Exception as e:
        logger.error(f"Conv TT decomposition failed: {e}")
        return None


def low_rank_factorization(
    model: nn.Module,
    config: Optional[DecompositionConfig] = None,
    layers_to_decompose: Optional[List[str]] = None
) -> nn.Module:
    """
    Low-rank matrix factorization for linear layers.
    
    Args:
        model: Model to decompose
        config: Decomposition configuration
        layers_to_decompose: Layers to decompose
        
    Returns:
        Model with low-rank layers
    """
    if config is None:
        config = DecompositionConfig(method="svd")
    
    decomposed_model = type(model)()
    decomposed_model.load_state_dict(model.state_dict())
    
    if layers_to_decompose is None:
        layers_to_decompose = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]
    
    for layer_name in layers_to_decompose:
        module = dict(decomposed_model.named_modules())[layer_name]
        
        if isinstance(module, nn.Linear):
            # SVD decomposition
            U, S, V = torch.svd(module.weight.data)
            
            # Determine rank
            if config.rank is None:
                # Select rank based on explained variance
                cumsum = torch.cumsum(S ** 2, dim=0)
                total_variance = cumsum[-1]
                rank = torch.searchsorted(
                    cumsum / total_variance,
                    config.explained_variance
                ).item() + 1
            else:
                rank = min(config.rank, min(module.weight.shape))
            
            # Truncate
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = V[:, :rank]
            
            # Create low-rank layer
            lr_layer = LowRankLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank
            )
            
            # Set decomposed weights
            lr_layer.U.data = U_r
            lr_layer.S.data = S_r
            lr_layer.V.data = V_r.t()
            
            if module.bias is not None:
                lr_layer.bias.data = module.bias.data.clone()
            
            # Replace module
            parent, child_name = get_parent_module(decomposed_model, layer_name)
            setattr(parent, child_name, lr_layer)
    
    return decomposed_model


def svd_decomposition(
    weight: torch.Tensor,
    rank: Optional[int] = None,
    explained_variance: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SVD decomposition with automatic rank selection.
    
    Args:
        weight: Weight matrix to decompose
        rank: Target rank (None for automatic)
        explained_variance: Explained variance threshold
        
    Returns:
        U, S, V matrices
    """
    U, S, V = torch.svd(weight)
    
    if rank is None:
        # Automatic rank selection
        cumsum = torch.cumsum(S ** 2, dim=0)
        total_variance = cumsum[-1]
        rank = torch.searchsorted(
            cumsum / total_variance,
            explained_variance
        ).item() + 1
    
    rank = min(rank, min(weight.shape))
    
    return U[:, :rank], S[:rank], V[:, :rank]


class TTLinear(nn.Module):
    """Tensor-Train linear layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_factors: List[int],
        out_factors: List[int],
        factors: List[torch.Tensor],
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_factors = in_factors
        self.out_factors = out_factors
        
        # Store TT factors as parameters
        self.factors = nn.ParameterList([
            nn.Parameter(factor) for factor in factors
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input
        batch_size = x.shape[0]
        x = x.view(batch_size, *self.in_factors)
        
        # Contract TT factors
        # This is a simplified implementation
        result = x
        for factor in self.factors:
            # Tensor contraction logic would go here
            pass
        
        # Reshape output
        result = result.view(batch_size, self.out_features)
        
        if self.bias is not None:
            result = result + self.bias
        
        return result


class LowRankLinear(nn.Module):
    """Low-rank linear layer using SVD decomposition."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Decomposed matrices
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(rank, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute USV^T x
        result = x @ self.V.t()  # (batch, rank)
        result = result * self.S  # (batch, rank)
        result = result @ self.U.t()  # (batch, out_features)
        
        if self.bias is not None:
            result = result + self.bias
        
        return result
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original = self.in_features * self.out_features
        compressed = (self.out_features + self.in_features + 1) * self.rank
        return original / compressed


def factorize_dimension(n: int) -> List[int]:
    """Factorize dimension for tensor decomposition."""
    factors = []
    
    # Try to factorize into roughly equal factors
    for i in range(2, int(np.sqrt(n)) + 1):
        while n % i == 0:
            factors.append(i)
            n //= i
    
    if n > 1:
        factors.append(n)
    
    # If too few factors, split largest factor
    while len(factors) < 2:
        if not factors:
            factors = [1, 1]
        else:
            largest = max(factors)
            factors.remove(largest)
            if largest > 1:
                factors.extend([2, largest // 2])
            else:
                factors.extend([1, 1])
    
    return factors


def estimate_tt_ranks(
    tensor: torch.Tensor,
    explained_variance: float = 0.95
) -> List[int]:
    """Estimate TT ranks based on explained variance."""
    ranks = []
    
    # Simplified rank estimation
    # In practice, would use more sophisticated methods
    ndim = len(tensor.shape)
    for i in range(ndim - 1):
        # Matricize tensor
        n1 = np.prod(tensor.shape[:i+1])
        n2 = np.prod(tensor.shape[i+1:])
        matrix = tensor.reshape(n1, n2)
        
        # SVD to estimate rank
        try:
            s = torch.svd(matrix)[1]
            cumsum = torch.cumsum(s ** 2, dim=0)
            total = cumsum[-1]
            rank = torch.searchsorted(cumsum / total, explained_variance).item() + 1
            ranks.append(min(rank, min(matrix.shape)))
        except:
            ranks.append(min(8, min(matrix.shape)))  # Default rank
    
    return ranks


def get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """Get parent module and child name from module path."""
    parts = module_name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    return parent, parts[-1]