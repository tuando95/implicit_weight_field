"""Interpretability tools for understanding compression behavior."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

logger = logging.getLogger(__name__)


class CompressionAnalyzer:
    """Analyze compression patterns and behavior."""
    
    def __init__(self):
        """Initialize compression analyzer."""
        self.layer_stats = {}
        self.compression_patterns = {}
    
    def analyze_layer_compressibility(
        self,
        model: nn.Module,
        sample_ratio: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which layers are most compressible.
        
        Args:
            model: Neural network model
            sample_ratio: Ratio of weights to sample for analysis
            
        Returns:
            Dictionary of layer analysis results
        """
        results = {}
        
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            
            # Flatten weights
            weights = param.data.flatten()
            
            # Sample weights if too large
            if len(weights) > 10000:
                indices = torch.randperm(len(weights))[:int(len(weights) * sample_ratio)]
                weights = weights[indices]
            
            # Compute statistics
            stats = {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'sparsity': (weights.abs() < 1e-6).float().mean().item(),
                'kurtosis': self._compute_kurtosis(weights),
                'entropy': self._compute_entropy(weights),
                'effective_rank': self._compute_effective_rank(param.data),
                'spectral_decay': self._compute_spectral_decay(param.data),
                'quantization_error': self._estimate_quantization_error(param.data),
                'predicted_compressibility': 0.0  # Will be computed
            }
            
            # Predict compressibility score
            stats['predicted_compressibility'] = self._predict_compressibility(stats)
            
            results[name] = stats
        
        return results
    
    def _compute_kurtosis(self, weights: torch.Tensor) -> float:
        """Compute kurtosis of weight distribution."""
        mean = weights.mean()
        std = weights.std()
        if std < 1e-8:
            return 0.0
        normalized = (weights - mean) / std
        return (normalized ** 4).mean().item() - 3.0
    
    def _compute_entropy(self, weights: torch.Tensor, bins: int = 50) -> float:
        """Compute entropy of weight distribution."""
        hist, _ = np.histogram(weights.numpy(), bins=bins)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        return -np.sum(hist * np.log(hist))
    
    def _compute_effective_rank(self, weights: torch.Tensor) -> float:
        """Compute effective rank via SVD."""
        if weights.dim() > 2:
            weights = weights.flatten(0, -2)
        
        try:
            s = torch.linalg.svdvals(weights)
            s = s / s.sum()
            return torch.exp(-torch.sum(s * torch.log(s + 1e-10))).item()
        except:
            return min(weights.shape)
    
    def _compute_spectral_decay(self, weights: torch.Tensor) -> float:
        """Compute rate of spectral decay."""
        if weights.dim() > 2:
            weights = weights.flatten(0, -2)
        
        try:
            s = torch.linalg.svdvals(weights)
            s = s / s[0]  # Normalize by largest singular value
            
            # Fit exponential decay
            log_s = torch.log(s + 1e-10)
            x = torch.arange(len(s)).float()
            
            # Linear regression in log space
            A = torch.stack([torch.ones_like(x), x], dim=1)
            coeffs = torch.linalg.lstsq(A, log_s).solution
            
            return -coeffs[1].item()  # Decay rate
        except:
            return 0.0
    
    def _estimate_quantization_error(self, weights: torch.Tensor, bits: int = 8) -> float:
        """Estimate quantization error."""
        # Simulate quantization
        scale = weights.abs().max()
        if scale < 1e-8:
            return 0.0
        
        # Quantize
        n_levels = 2 ** bits
        quantized = torch.round(weights / scale * (n_levels / 2)) * scale / (n_levels / 2)
        
        # Compute error
        error = (weights - quantized).pow(2).mean().item()
        return error / weights.pow(2).mean().item()  # Relative error
    
    def _predict_compressibility(self, stats: Dict[str, float]) -> float:
        """Predict compressibility score based on statistics."""
        # Simple heuristic-based scoring
        score = 0.0
        
        # High sparsity is good for compression
        score += stats['sparsity'] * 2.0
        
        # Low effective rank is good
        score += 1.0 / (stats['effective_rank'] + 1)
        
        # Fast spectral decay is good
        score += min(stats['spectral_decay'], 1.0)
        
        # Low entropy suggests structure
        score += 1.0 / (stats['entropy'] + 1)
        
        # Low quantization error suggests discrete structure
        score += 1.0 - stats['quantization_error']
        
        # High kurtosis suggests heavy tails
        score += 0.1 * abs(stats['kurtosis'])
        
        return score / 6.0  # Normalize to [0, 1]
    
    def identify_compression_bottlenecks(
        self,
        compression_results: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[str]:
        """
        Identify layers that resist compression.
        
        Args:
            compression_results: Results from compression
            threshold: Compression ratio threshold
            
        Returns:
            List of problematic layer names
        """
        bottlenecks = []
        
        for layer_name, result in compression_results.items():
            compression_ratio = result.get('compression_ratio', 1.0)
            reconstruction_error = result.get('reconstruction_error', 0.0)
            
            # Check if layer is a bottleneck
            if compression_ratio < threshold or reconstruction_error > 0.01:
                bottlenecks.append(layer_name)
        
        return bottlenecks
    
    def analyze_weight_patterns(
        self,
        weights: torch.Tensor,
        n_components: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze patterns in weight tensors.
        
        Args:
            weights: Weight tensor
            n_components: Number of PCA components
            
        Returns:
            Pattern analysis results
        """
        # Reshape to 2D
        original_shape = weights.shape
        if weights.dim() > 2:
            weights_2d = weights.flatten(0, -2)
        else:
            weights_2d = weights
        
        results = {}
        
        # PCA analysis
        try:
            pca = PCA(n_components=min(n_components, min(weights_2d.shape)))
            weights_np = weights_2d.detach().cpu().numpy()
            transformed = pca.fit_transform(weights_np.T)
            
            results['pca_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            results['pca_components'] = pca.components_
            results['pca_transformed'] = transformed
        except:
            results['pca_variance_ratio'] = []
        
        # Detect block structure
        results['has_block_structure'] = self._detect_block_structure(weights_2d)
        
        # Detect repetitive patterns
        results['repetition_score'] = self._compute_repetition_score(weights_2d)
        
        # Symmetry detection
        results['symmetry_score'] = self._compute_symmetry_score(weights_2d)
        
        return results
    
    def _detect_block_structure(self, weights: torch.Tensor) -> bool:
        """Detect if weights have block structure."""
        if weights.shape[0] < 8 or weights.shape[1] < 8:
            return False
        
        # Compute block-wise variance
        block_size = 4
        h_blocks = weights.shape[0] // block_size
        w_blocks = weights.shape[1] // block_size
        
        block_vars = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = weights[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size]
                block_vars.append(block.var().item())
        
        # Check if variance is concentrated in few blocks
        block_vars = np.array(block_vars)
        sorted_vars = np.sort(block_vars)[::-1]
        
        # If top 20% blocks contain 80% variance -> block structure
        top_20_percent = int(len(sorted_vars) * 0.2)
        variance_ratio = sorted_vars[:top_20_percent].sum() / sorted_vars.sum()
        
        return variance_ratio > 0.8
    
    def _compute_repetition_score(self, weights: torch.Tensor) -> float:
        """Compute score for repetitive patterns."""
        # Use autocorrelation
        weights_flat = weights.flatten()
        if len(weights_flat) < 100:
            return 0.0
        
        # Compute autocorrelation for different lags
        autocorr = []
        for lag in range(1, min(50, len(weights_flat) // 2)):
            corr = torch.corrcoef(torch.stack([
                weights_flat[:-lag],
                weights_flat[lag:]
            ]))[0, 1]
            if not torch.isnan(corr):
                autocorr.append(abs(corr.item()))
        
        return max(autocorr) if autocorr else 0.0
    
    def _compute_symmetry_score(self, weights: torch.Tensor) -> float:
        """Compute symmetry score."""
        if weights.shape[0] != weights.shape[1]:
            return 0.0
        
        # Check different types of symmetry
        # Transpose symmetry
        transpose_diff = (weights - weights.T).abs().mean()
        
        # Flip symmetry
        flip_h_diff = (weights - weights.flip(0)).abs().mean()
        flip_v_diff = (weights - weights.flip(1)).abs().mean()
        
        # Overall symmetry score (lower is more symmetric)
        min_diff = min(transpose_diff, flip_h_diff, flip_v_diff)
        weight_scale = weights.abs().mean()
        
        if weight_scale < 1e-8:
            return 0.0
        
        return 1.0 - min(min_diff / weight_scale, 1.0)


class LayerImportanceAnalyzer:
    """Analyze layer importance for compression decisions."""
    
    def compute_fisher_importance(
        self,
        model: nn.Module,
        dataloader,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute Fisher information-based importance scores.
        
        Args:
            model: Neural network model
            dataloader: Data loader for importance computation
            n_samples: Number of samples to use
            
        Returns:
            Dictionary of layer importance scores
        """
        importance_scores = {}
        
        # Register hooks to compute gradients
        gradients = {}
        
        def save_grad(name):
            def hook(grad):
                gradients[name] = grad
                return grad
            return hook
        
        # Register hooks
        handles = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                h = param.register_hook(save_grad(name))
                handles.append(h)
        
        # Accumulate Fisher information
        fisher_info = {name: 0 for name, _ in model.named_parameters()}
        
        model.eval()
        samples_processed = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if samples_processed >= n_samples:
                break
            
            data = data.cuda() if torch.cuda.is_available() else data
            target = target.cuda() if torch.cuda.is_available() else target
            
            # Forward pass
            output = model(data)
            
            # Sample from output distribution
            if output.dim() == 2:  # Classification
                probs = torch.softmax(output, dim=1)
                samples = torch.multinomial(probs, 1).squeeze()
                loss = nn.CrossEntropyLoss()(output, samples)
            else:
                loss = output.pow(2).mean()
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in model.named_parameters():
                if name in gradients and gradients[name] is not None:
                    fisher_info[name] += gradients[name].pow(2).sum().item()
            
            samples_processed += len(data)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Normalize scores
        total_importance = sum(fisher_info.values())
        if total_importance > 0:
            importance_scores = {
                name: score / total_importance 
                for name, score in fisher_info.items()
            }
        else:
            importance_scores = {name: 1.0 / len(fisher_info) for name in fisher_info}
        
        return importance_scores
    
    def compute_magnitude_importance(
        self,
        model: nn.Module
    ) -> Dict[str, float]:
        """
        Compute magnitude-based importance scores.
        
        Args:
            model: Neural network model
            
        Returns:
            Dictionary of layer importance scores
        """
        importance_scores = {}
        total_magnitude = 0
        
        # Compute magnitude for each layer
        for name, param in model.named_parameters():
            magnitude = param.abs().sum().item()
            importance_scores[name] = magnitude
            total_magnitude += magnitude
        
        # Normalize
        if total_magnitude > 0:
            importance_scores = {
                name: score / total_magnitude
                for name, score in importance_scores.items()
            }
        
        return importance_scores


def visualize_layer_importance(
    importance_scores: Dict[str, float],
    title: str = "Layer Importance Scores",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize layer importance scores.
    
    Args:
        importance_scores: Dictionary of layer importance scores
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_layers]
    scores = [score for _, score in sorted_layers]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    bars = ax.bar(range(len(names)), scores)
    
    # Color by importance
    colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Importance Score')
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split('.')[-1] for n in names], rotation=45, ha='right')
    
    # Add cumulative line
    ax2 = ax.twinx()
    cumulative = np.cumsum(scores) / np.sum(scores)
    ax2.plot(range(len(names)), cumulative, 'k--', linewidth=2, label='Cumulative')
    ax2.set_ylabel('Cumulative Importance')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig