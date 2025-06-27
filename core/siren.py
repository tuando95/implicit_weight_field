"""SIREN (Sinusoidal Representation Networks) implementation for implicit weight fields."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SIRENLayer(nn.Module):
    """Single SIREN layer with sinusoidal activation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 1.0,
        is_first: bool = False,
        use_bias: bool = True
    ):
        """
        Initialize SIREN layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            w0: Frequency parameter for the sinusoidal activation
            is_first: Whether this is the first layer (affects initialization)
            use_bias: Whether to use bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to SIREN initialization scheme."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform initialization
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layers: uniform initialization scaled by w0
                bound = np.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            
            if self.linear.bias is not None:
                bound = 1 / np.sqrt(self.in_features) if self.is_first else np.sqrt(6 / self.in_features) / self.w0
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sinusoidal activation."""
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN network for implicit weight field representation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        w0_initial: float = 30.0,
        w0_hidden: float = 1.0,
        final_activation: bool = False,
        use_bias: bool = True
    ):
        """
        Initialize SIREN network.
        
        Args:
            in_features: Number of input features (coordinate dimension)
            hidden_features: Number of hidden units per layer
            out_features: Number of output features (typically 1 for weight value)
            num_layers: Total number of layers (including output)
            w0_initial: Frequency parameter for first layer
            w0_hidden: Frequency parameter for hidden layers
            final_activation: Whether to apply sin activation to final layer
            use_bias: Whether to use bias in layers
        """
        super().__init__()
        self.num_layers = num_layers
        self.final_activation = final_activation
        
        layers = []
        
        # First layer
        layers.append(SIRENLayer(
            in_features, 
            hidden_features, 
            w0=w0_initial, 
            is_first=True,
            use_bias=use_bias
        ))
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(SIRENLayer(
                hidden_features,
                hidden_features,
                w0=w0_hidden,
                is_first=False,
                use_bias=use_bias
            ))
        
        # Final layer
        if num_layers > 1:
            self.final_layer = nn.Linear(hidden_features, out_features, bias=use_bias)
            # Initialize final layer
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features)
                self.final_layer.weight.uniform_(-bound, bound)
                if use_bias and self.final_layer.bias is not None:
                    self.final_layer.bias.uniform_(-bound, bound)
        else:
            # Single layer network
            layers = []
            self.final_layer = nn.Linear(in_features, out_features, bias=use_bias)
            with torch.no_grad():
                self.final_layer.weight.uniform_(-1 / in_features, 1 / in_features)
                if use_bias and self.final_layer.bias is not None:
                    self.final_layer.bias.uniform_(-1 / in_features, 1 / in_features)
        
        self.layers = nn.Sequential(*layers) if layers else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SIREN network.
        
        Args:
            x: Input coordinates (batch_size, in_features)
            
        Returns:
            Output values (batch_size, out_features)
        """
        x = self.layers(x)
        x = self.final_layer(x)
        
        if self.final_activation:
            x = torch.sin(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @staticmethod
    def create_for_tensor(
        tensor_shape: Tuple[int, ...],
        encoding_dim: int,
        hidden_width: int = 256,
        num_layers: int = 2,
        w0: float = 30.0
    ) -> 'SIREN':
        """
        Factory method to create SIREN for specific tensor shape.
        
        Args:
            tensor_shape: Shape of the tensor to be compressed
            encoding_dim: Dimension of positional encoding
            hidden_width: Width of hidden layers
            num_layers: Number of layers
            w0: Frequency parameter
            
        Returns:
            SIREN network configured for the tensor
        """
        # Input dimension is encoding_dim * number of tensor dimensions
        in_features = encoding_dim * len(tensor_shape)
        
        return SIREN(
            in_features=in_features,
            hidden_features=hidden_width,
            out_features=1,  # Single weight value output
            num_layers=num_layers,
            w0_initial=w0,
            w0_hidden=1.0,
            final_activation=False
        )