"""Visualization tools for weight patterns and compression analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from mpl_toolkits.axes_grid1 import ImageGrid

logger = logging.getLogger(__name__)


class WeightVisualizer:
    """Visualize weight patterns and compression effects."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize weight visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def visualize_weight_comparison(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        layer_name: str = "layer",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize original vs reconstructed weights.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            layer_name: Name of the layer
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Ensure tensors are on CPU
        original = original.detach().cpu()
        reconstructed = reconstructed.detach().cpu()
        
        # Calculate error
        error = (original - reconstructed).abs()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Flatten tensors for 2D visualization if needed
        if original.dim() > 2:
            # For conv weights, show first output channel
            if original.dim() == 4:  # Conv2d
                original_2d = original[0].reshape(original.shape[1], -1)
                reconstructed_2d = reconstructed[0].reshape(reconstructed.shape[1], -1)
                error_2d = error[0].reshape(error.shape[1], -1)
            else:
                original_2d = original.flatten(1)
                reconstructed_2d = reconstructed.flatten(1)
                error_2d = error.flatten(1)
        else:
            original_2d = original
            reconstructed_2d = reconstructed
            error_2d = error
        
        # Plot original weights
        im1 = axes[0, 0].imshow(original_2d, cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title(f'Original Weights - {layer_name}')
        axes[0, 0].set_xlabel('Weight Index')
        axes[0, 0].set_ylabel('Neuron Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot reconstructed weights
        im2 = axes[0, 1].imshow(reconstructed_2d, cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('Reconstructed Weights')
        axes[0, 1].set_xlabel('Weight Index')
        axes[0, 1].set_ylabel('Neuron Index')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot error heatmap
        im3 = axes[0, 2].imshow(error_2d, cmap='hot', aspect='auto')
        axes[0, 2].set_title('Absolute Error')
        axes[0, 2].set_xlabel('Weight Index')
        axes[0, 2].set_ylabel('Neuron Index')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Weight distribution comparison
        axes[1, 0].hist(original.flatten().numpy(), bins=50, alpha=0.7, 
                       label='Original', density=True)
        axes[1, 0].hist(reconstructed.flatten().numpy(), bins=50, alpha=0.7,
                       label='Reconstructed', density=True)
        axes[1, 0].set_xlabel('Weight Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Weight Distribution')
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(error.flatten().numpy(), bins=50, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].axvline(error.mean().item(), color='black', linestyle='--',
                          label=f'Mean: {error.mean().item():.4f}')
        axes[1, 1].legend()
        
        # Scatter plot
        sample_indices = torch.randperm(original.numel())[:1000]
        axes[1, 2].scatter(original.flatten()[sample_indices].numpy(),
                          reconstructed.flatten()[sample_indices].numpy(),
                          alpha=0.5, s=1)
        axes[1, 2].plot([-3, 3], [-3, 3], 'r--', label='y=x')
        axes[1, 2].set_xlabel('Original Weight')
        axes[1, 2].set_ylabel('Reconstructed Weight')
        axes[1, 2].set_title('Weight Correspondence (1000 samples)')
        axes[1, 2].legend()
        axes[1, 2].set_xlim(original.min().item(), original.max().item())
        axes[1, 2].set_ylim(reconstructed.min().item(), reconstructed.max().item())
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_spectral_analysis(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        layer_name: str = "layer",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize frequency domain analysis.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            layer_name: Name of the layer
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        original = original.detach().cpu()
        reconstructed = reconstructed.detach().cpu()
        
        # Reshape to 2D if needed
        if original.dim() > 2:
            original_2d = original.flatten(1)
            reconstructed_2d = reconstructed.flatten(1)
        else:
            original_2d = original
            reconstructed_2d = reconstructed
        
        # Compute 2D FFT
        fft_original = np.abs(np.fft.fft2(original_2d.numpy()))
        fft_reconstructed = np.abs(np.fft.fft2(reconstructed_2d.numpy()))
        fft_error = np.abs(fft_original - fft_reconstructed)
        
        # Shift zero frequency to center
        fft_original = np.fft.fftshift(fft_original)
        fft_reconstructed = np.fft.fftshift(fft_reconstructed)
        fft_error = np.fft.fftshift(fft_error)
        
        # Log scale for better visualization
        fft_original_log = np.log1p(fft_original)
        fft_reconstructed_log = np.log1p(fft_reconstructed)
        fft_error_log = np.log1p(fft_error)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot original spectrum
        im1 = axes[0, 0].imshow(fft_original_log, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Original Spectrum - {layer_name}')
        axes[0, 0].set_xlabel('Frequency X')
        axes[0, 0].set_ylabel('Frequency Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot reconstructed spectrum
        im2 = axes[0, 1].imshow(fft_reconstructed_log, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Reconstructed Spectrum')
        axes[0, 1].set_xlabel('Frequency X')
        axes[0, 1].set_ylabel('Frequency Y')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot error spectrum
        im3 = axes[0, 2].imshow(fft_error_log, cmap='hot', aspect='auto')
        axes[0, 2].set_title('Spectrum Error')
        axes[0, 2].set_xlabel('Frequency X')
        axes[0, 2].set_ylabel('Frequency Y')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Radial frequency analysis
        center = (fft_original.shape[0] // 2, fft_original.shape[1] // 2)
        y, x = np.ogrid[:fft_original.shape[0], :fft_original.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Compute radial average
        radial_profile_orig = np.bincount(r.ravel(), fft_original.ravel()) / np.bincount(r.ravel())
        radial_profile_recon = np.bincount(r.ravel(), fft_reconstructed.ravel()) / np.bincount(r.ravel())
        
        axes[1, 0].plot(radial_profile_orig[:len(radial_profile_orig)//2], 
                       label='Original', linewidth=2)
        axes[1, 0].plot(radial_profile_recon[:len(radial_profile_recon)//2], 
                       label='Reconstructed', linewidth=2)
        axes[1, 0].set_xlabel('Radial Frequency')
        axes[1, 0].set_ylabel('Average Magnitude')
        axes[1, 0].set_title('Radial Frequency Profile')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Frequency preservation ratio
        freq_preservation = fft_reconstructed / (fft_original + 1e-10)
        im4 = axes[1, 1].imshow(freq_preservation, cmap='RdBu_r', 
                               vmin=0, vmax=2, aspect='auto')
        axes[1, 1].set_title('Frequency Preservation Ratio')
        axes[1, 1].set_xlabel('Frequency X')
        axes[1, 1].set_ylabel('Frequency Y')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Cumulative energy
        sorted_orig = np.sort(fft_original.flatten())[::-1]
        sorted_recon = np.sort(fft_reconstructed.flatten())[::-1]
        
        cumsum_orig = np.cumsum(sorted_orig) / np.sum(sorted_orig)
        cumsum_recon = np.cumsum(sorted_recon) / np.sum(sorted_recon)
        
        n_components = len(cumsum_orig)
        x_axis = np.arange(1, n_components + 1) / n_components * 100
        
        axes[1, 2].plot(x_axis[:1000], cumsum_orig[:1000], 
                       label='Original', linewidth=2)
        axes[1, 2].plot(x_axis[:1000], cumsum_recon[:1000], 
                       label='Reconstructed', linewidth=2)
        axes[1, 2].set_xlabel('Percentage of Frequencies')
        axes[1, 2].set_ylabel('Cumulative Energy')
        axes[1, 2].set_title('Energy Concentration')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_compression_summary(
        self,
        compression_results: Dict[str, Any],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize compression summary across layers.
        
        Args:
            compression_results: Dictionary with layer compression results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract data
        layer_names = []
        compression_ratios = []
        reconstruction_errors = []
        original_sizes = []
        
        for name, result in compression_results.items():
            layer_names.append(name)
            compression_ratios.append(result.get('compression_ratio', 0))
            reconstruction_errors.append(result.get('reconstruction_error', 0))
            original_sizes.append(result.get('original_size', 0))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Compression ratios by layer
        axes[0, 0].bar(range(len(layer_names)), compression_ratios)
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Compression Ratio')
        axes[0, 0].set_title('Compression Ratios by Layer')
        axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Reconstruction errors
        axes[0, 1].bar(range(len(layer_names)), reconstruction_errors, color='orange')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Reconstruction Error (MSE)')
        axes[0, 1].set_title('Reconstruction Errors by Layer')
        axes[0, 1].set_yscale('log')
        
        # Compression vs size scatter
        axes[1, 0].scatter(original_sizes, compression_ratios, s=50, alpha=0.7)
        axes[1, 0].set_xlabel('Original Layer Size (# parameters)')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].set_title('Compression Ratio vs Layer Size')
        axes[1, 0].set_xscale('log')
        
        # Add trend line
        if len(original_sizes) > 1:
            z = np.polyfit(np.log(original_sizes), compression_ratios, 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(min(original_sizes)), 
                                 np.log10(max(original_sizes)), 100)
            axes[1, 0].plot(x_trend, p(np.log(x_trend)), 'r--', alpha=0.5)
        
        # Error vs compression trade-off
        axes[1, 1].scatter(compression_ratios, reconstruction_errors, s=50, alpha=0.7)
        axes[1, 1].set_xlabel('Compression Ratio')
        axes[1, 1].set_ylabel('Reconstruction Error (MSE)')
        axes[1, 1].set_title('Error vs Compression Trade-off')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class FieldVisualization:
    """Visualize implicit field properties."""
    
    def visualize_field_interpolation(
        self,
        field,
        dimension: int = 0,
        n_samples: int = 100,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize field interpolation along one dimension.
        
        Args:
            field: Implicit weight field
            dimension: Which dimension to interpolate along
            n_samples: Number of interpolation samples
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Generate interpolation coordinates
        coords = torch.zeros(n_samples, len(field.tensor_shape))
        
        # Set middle values for other dimensions
        for i, size in enumerate(field.tensor_shape):
            if i != dimension:
                coords[:, i] = 0.5
        
        # Interpolate along specified dimension
        coords[:, dimension] = torch.linspace(0, 1, n_samples)
        
        # Get field values
        with torch.no_grad():
            encoded = field.encoder(coords)
            values = field.field(encoded).squeeze()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot interpolation
        ax.plot(coords[:, dimension].numpy(), values.numpy(), 'b-', linewidth=2)
        ax.set_xlabel(f'Normalized Position (Dimension {dimension})')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Field Interpolation Along Dimension {dimension}')
        ax.grid(True, alpha=0.3)
        
        # Add discrete points if tensor is small
        if field.tensor_shape[dimension] <= 20:
            discrete_positions = torch.linspace(0, 1, field.tensor_shape[dimension])
            discrete_coords = coords[:len(discrete_positions)].clone()
            discrete_coords[:, dimension] = discrete_positions
            
            with torch.no_grad():
                encoded = field.encoder(discrete_coords)
                discrete_values = field.field(encoded).squeeze()
            
            ax.scatter(discrete_positions.numpy(), discrete_values.numpy(), 
                      color='red', s=50, zorder=5, label='Discrete positions')
            ax.legend()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_2d_field_surface(
        self,
        field,
        resolution: int = 50,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize 2D field as a surface.
        
        Args:
            field: Implicit weight field (must be 2D)
            resolution: Grid resolution
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if len(field.tensor_shape) != 2:
            raise ValueError("Field must be 2D for surface visualization")
        
        # Create grid
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Flatten and create coordinates
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Get field values
        with torch.no_grad():
            encoded = field.encoder(coords)
            Z = field.field(encoded).reshape(resolution, resolution)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(),
                              cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_zlabel('Weight Value')
        ax.set_title('2D Weight Field Surface')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class TrainingVisualization:
    """Visualize training dynamics."""
    
    def visualize_loss_landscape(
        self,
        loss_history: List[float],
        additional_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize training loss and additional metrics.
        
        Args:
            loss_history: List of loss values
            additional_metrics: Dictionary of metric name to values
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_subplots = 1 + (len(additional_metrics) if additional_metrics else 0)
        fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots))
        
        if n_subplots == 1:
            axes = [axes]
        
        # Plot loss
        axes[0].plot(loss_history, 'b-', linewidth=2)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Plot additional metrics
        if additional_metrics:
            for i, (name, values) in enumerate(additional_metrics.items(), 1):
                axes[i].plot(values, linewidth=2)
                axes[i].set_xlabel('Training Step')
                axes[i].set_ylabel(name)
                axes[i].set_title(f'{name} During Training')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_gradient_flow(
        self,
        model,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize gradient flow through model layers.
        
        Args:
            model: PyTorch model
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Collect gradient information
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(layers))
        
        # Plot bars
        ax.bar(x_pos - 0.2, ave_grads, 0.4, label='Average gradient', alpha=0.7)
        ax.bar(x_pos + 0.2, max_grads, 0.4, label='Max gradient', alpha=0.7)
        
        ax.set_xlabel('Layers')
        ax.set_ylabel('Gradient magnitude')
        ax.set_title('Gradient Flow')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def create_compression_report(
    original_model,
    compressed_model,
    compression_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Create comprehensive compression visualization report.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        compression_results: Compression results dictionary
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = WeightVisualizer()
    
    # Visualize each layer
    for layer_name, result in compression_results.items():
        if 'original_weights' in result and 'reconstructed_weights' in result:
            # Weight comparison
            fig = visualizer.visualize_weight_comparison(
                result['original_weights'],
                result['reconstructed_weights'],
                layer_name,
                save_path=output_dir / f'{layer_name}_comparison.png'
            )
            plt.close(fig)
            
            # Spectral analysis
            fig = visualizer.visualize_spectral_analysis(
                result['original_weights'],
                result['reconstructed_weights'],
                layer_name,
                save_path=output_dir / f'{layer_name}_spectral.png'
            )
            plt.close(fig)
    
    # Overall compression summary
    fig = visualizer.visualize_compression_summary(
        compression_results,
        save_path=output_dir / 'compression_summary.png'
    )
    plt.close(fig)
    
    logger.info(f"Compression report saved to {output_dir}")