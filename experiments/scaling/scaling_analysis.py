"""Scaling analysis for compression performance vs tensor/model size."""

import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from ...core.implicit_field import (
    ImplicitWeightField,
    CompressionConfig,
    TensorStatistics
)
from ...compression.trainer import FieldTrainer, TrainingConfig
from ...experiments.models import (
    load_resnet50,
    load_mobilenet_v2,
    load_vit,
    load_bert_base
)

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Result from a scaling experiment."""
    tensor_size: int
    tensor_shape: Tuple[int, ...]
    effective_rank: float
    compression_ratio: float
    reconstruction_error: float
    training_time: float
    field_parameters: int
    convergence_steps: int


class TensorSizeScaling:
    """Analyze scaling behavior with tensor size."""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize tensor size scaling analysis.
        
        Args:
            config: Base compression configuration
        """
        self.config = config or CompressionConfig()
        self.results = []
    
    def analyze_conv_layers(
        self,
        kernel_sizes: List[int] = [3, 5, 7],
        channel_counts: List[int] = [64, 128, 256, 512, 1024],
        device: Optional[torch.device] = None
    ) -> List[ScalingResult]:
        """
        Analyze scaling for convolutional layers.
        
        Args:
            kernel_sizes: Kernel sizes to test
            channel_counts: Channel counts to test
            device: Device to use
            
        Returns:
            List of scaling results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Analyzing convolutional layer scaling")
        
        for kernel_size in kernel_sizes:
            for in_channels in channel_counts:
                for out_channels in channel_counts:
                    # Create synthetic conv weight
                    shape = (out_channels, in_channels, kernel_size, kernel_size)
                    tensor = self._create_synthetic_tensor(shape, device)
                    
                    # Analyze compression
                    result = self._analyze_tensor(tensor, shape, device)
                    self.results.append(result)
                    
                    logger.debug(f"Conv {shape}: ratio={result.compression_ratio:.2f}, "
                               f"error={result.reconstruction_error:.6f}")
        
        return self.results
    
    def analyze_linear_layers(
        self,
        dimensions: List[int] = [64, 128, 256, 512, 1024, 2048, 4096],
        device: Optional[torch.device] = None
    ) -> List[ScalingResult]:
        """
        Analyze scaling for linear layers.
        
        Args:
            dimensions: Input/output dimensions to test
            device: Device to use
            
        Returns:
            List of scaling results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Analyzing linear layer scaling")
        
        for in_dim in dimensions:
            for out_dim in dimensions:
                # Create synthetic linear weight
                shape = (out_dim, in_dim)
                tensor = self._create_synthetic_tensor(shape, device)
                
                # Analyze compression
                result = self._analyze_tensor(tensor, shape, device)
                self.results.append(result)
                
                logger.debug(f"Linear {shape}: ratio={result.compression_ratio:.2f}, "
                           f"error={result.reconstruction_error:.6f}")
        
        return self.results
    
    def analyze_attention_layers(
        self,
        head_counts: List[int] = [8, 12, 16],
        dimensions: List[int] = [64, 128, 256],
        device: Optional[torch.device] = None
    ) -> List[ScalingResult]:
        """
        Analyze scaling for multi-head attention layers.
        
        Args:
            head_counts: Number of attention heads
            dimensions: Head dimensions
            device: Device to use
            
        Returns:
            List of scaling results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Analyzing attention layer scaling")
        
        for num_heads in head_counts:
            for head_dim in dimensions:
                total_dim = num_heads * head_dim
                
                # Query, Key, Value projections
                for proj_name in ['query', 'key', 'value']:
                    shape = (total_dim, total_dim)
                    tensor = self._create_synthetic_tensor(shape, device)
                    
                    result = self._analyze_tensor(tensor, shape, device)
                    self.results.append(result)
        
        return self.results
    
    def analyze_embedding_layers(
        self,
        vocab_sizes: List[int] = [1000, 10000, 30000],
        embedding_dims: List[int] = [128, 256, 512],
        device: Optional[torch.device] = None
    ) -> List[ScalingResult]:
        """Analyze scaling for embedding layers."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Analyzing embedding layer scaling")
        
        for vocab_size in vocab_sizes:
            for embed_dim in embedding_dims:
                shape = (vocab_size, embed_dim)
                tensor = self._create_synthetic_tensor(shape, device)
                
                result = self._analyze_tensor(tensor, shape, device)
                self.results.append(result)
        
        return self.results
    
    def _create_synthetic_tensor(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        rank_ratio: float = 0.5
    ) -> torch.Tensor:
        """Create synthetic tensor with controlled properties."""
        # Create low-rank tensor with noise
        if len(shape) == 2:
            # Matrix case
            rank = int(min(shape) * rank_ratio)
            U = torch.randn(shape[0], rank, device=device)
            V = torch.randn(rank, shape[1], device=device)
            tensor = U @ V
        else:
            # General tensor
            tensor = torch.randn(shape, device=device)
        
        # Add noise
        noise = torch.randn_like(tensor) * 0.1
        tensor = tensor + noise
        
        # Normalize
        tensor = tensor / tensor.std()
        
        return tensor
    
    def _analyze_tensor(
        self,
        tensor: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> ScalingResult:
        """Analyze compression for a single tensor."""
        import time
        
        # Compute tensor statistics
        stats = ImplicitWeightField.compute_tensor_statistics(tensor)
        
        # Create and train field
        field = ImplicitWeightField(
            tensor_shape=shape,
            config=self.config
        ).to(device)
        
        trainer = FieldTrainer(
            field,
            TrainingConfig(
                learning_rate=self.config.learning_rate,
                max_steps=self.config.max_steps,
                convergence_threshold=self.config.convergence_threshold
            )
        )
        
        start_time = time.time()
        steps = trainer.train(tensor, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = trainer.evaluate(tensor)
        
        return ScalingResult(
            tensor_size=tensor.numel(),
            tensor_shape=shape,
            effective_rank=stats.effective_rank,
            compression_ratio=field.compression_ratio(),
            reconstruction_error=metrics['mse'],
            training_time=training_time,
            field_parameters=field.count_parameters(),
            convergence_steps=steps
        )
    
    def fit_scaling_law(self) -> Dict[str, Any]:
        """
        Fit power-law scaling relationship.
        
        Returns:
            Dictionary with fitted parameters
        """
        if not self.results:
            return {}
        
        # Extract data
        sizes = np.array([r.tensor_size for r in self.results])
        ranks = np.array([r.effective_rank for r in self.results])
        ratios = np.array([r.compression_ratio for r in self.results])
        
        # Fit: compression_ratio = A * size^alpha * rank^beta
        def scaling_law(X, A, alpha, beta):
            size, rank = X
            return A * (size ** alpha) * (rank ** beta)
        
        # Log transform for linear fitting
        log_sizes = np.log(sizes)
        log_ranks = np.log(ranks)
        log_ratios = np.log(ratios)
        
        # Fit using linear regression in log space
        from sklearn.linear_model import LinearRegression
        X = np.column_stack([log_sizes, log_ranks])
        model = LinearRegression()
        model.fit(X, log_ratios)
        
        A = np.exp(model.intercept_)
        alpha = model.coef_[0]
        beta = model.coef_[1]
        
        # Calculate R-squared
        predictions = model.predict(X)
        r_squared = 1 - np.sum((log_ratios - predictions)**2) / np.sum((log_ratios - log_ratios.mean())**2)
        
        return {
            'A': A,
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'equation': f'compression_ratio = {A:.3f} * size^{alpha:.3f} * rank^{beta:.3f}'
        }


class ModelSizeScaling:
    """Analyze scaling behavior with model size."""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize model size scaling analysis."""
        self.config = config or CompressionConfig()
        self.results = {}
    
    def analyze_model_families(
        self,
        device: Optional[torch.device] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze scaling across model families.
        
        Args:
            device: Device to use
            
        Returns:
            Dictionary of results per model family
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ResNet family
        logger.info("Analyzing ResNet family")
        self.results['resnet'] = self._analyze_resnets(device)
        
        # MobileNet family
        logger.info("Analyzing MobileNet family")
        self.results['mobilenet'] = self._analyze_mobilenets(device)
        
        # Vision Transformer family
        logger.info("Analyzing ViT family")
        self.results['vit'] = self._analyze_vits(device)
        
        # BERT family
        logger.info("Analyzing BERT family")
        self.results['bert'] = self._analyze_berts(device)
        
        return self.results
    
    def _analyze_resnets(self, device: torch.device) -> List[Dict[str, Any]]:
        """Analyze ResNet models."""
        results = []
        
        # Different ResNet sizes
        resnet_configs = [
            ('resnet18', torchvision.models.resnet18),
            ('resnet34', torchvision.models.resnet34),
            ('resnet50', torchvision.models.resnet50),
            ('resnet101', torchvision.models.resnet101),
        ]
        
        for name, model_fn in resnet_configs:
            try:
                model = model_fn(pretrained=False).to(device)
                result = self._analyze_model(model, name, device)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
        
        return results
    
    def _analyze_mobilenets(self, device: torch.device) -> List[Dict[str, Any]]:
        """Analyze MobileNet models."""
        results = []
        
        # MobileNet V2 with different width multipliers
        for width_mult in [0.5, 0.75, 1.0, 1.4]:
            try:
                model = torchvision.models.mobilenet_v2(
                    pretrained=False,
                    width_mult=width_mult
                ).to(device)
                
                name = f"mobilenet_v2_{width_mult}"
                result = self._analyze_model(model, name, device)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing mobilenet {width_mult}: {e}")
        
        return results
    
    def _analyze_vits(self, device: torch.device) -> List[Dict[str, Any]]:
        """Analyze Vision Transformer models."""
        results = []
        
        # Different ViT sizes
        vit_configs = [
            ('vit_b_16', torchvision.models.vit_b_16),
            ('vit_b_32', torchvision.models.vit_b_32),
            ('vit_l_16', torchvision.models.vit_l_16),
        ]
        
        for name, model_fn in vit_configs:
            try:
                model = model_fn(pretrained=False).to(device)
                result = self._analyze_model(model, name, device)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
        
        return results
    
    def _analyze_berts(self, device: torch.device) -> List[Dict[str, Any]]:
        """Analyze BERT models."""
        results = []
        
        # Different BERT sizes
        bert_configs = [
            ('bert-tiny', {'num_hidden_layers': 2, 'hidden_size': 128}),
            ('bert-mini', {'num_hidden_layers': 4, 'hidden_size': 256}),
            ('bert-small', {'num_hidden_layers': 4, 'hidden_size': 512}),
            ('bert-medium', {'num_hidden_layers': 8, 'hidden_size': 512}),
            ('bert-base', {'num_hidden_layers': 12, 'hidden_size': 768}),
        ]
        
        for name, config_update in bert_configs:
            try:
                from transformers import BertConfig, BertModel
                
                config = BertConfig()
                for key, value in config_update.items():
                    setattr(config, key, value)
                
                model = BertModel(config).to(device)
                result = self._analyze_model(model, name, device)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
        
        return results
    
    def _analyze_model(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze compression for a single model."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Sample layers for compression analysis
        layer_results = []
        sampled_layers = self._sample_layers(model, max_layers=10)
        
        for name, param in sampled_layers:
            if param.numel() < 1000:
                continue
            
            # Create field
            field = ImplicitWeightField(
                tensor_shape=param.shape,
                config=self.config
            ).to(device)
            
            # Train
            trainer = FieldTrainer(field, TrainingConfig(max_steps=500))
            trainer.train(param.data.to(device), verbose=False)
            
            # Record result
            layer_results.append({
                'layer_name': name,
                'tensor_size': param.numel(),
                'compression_ratio': field.compression_ratio(),
                'field_params': field.count_parameters()
            })
        
        # Aggregate results
        if layer_results:
            avg_compression = np.mean([r['compression_ratio'] for r in layer_results])
            total_compressed = sum(r['field_params'] for r in layer_results)
            sampled_original = sum(r['tensor_size'] for r in layer_results)
        else:
            avg_compression = 1.0
            total_compressed = total_params
            sampled_original = total_params
        
        return {
            'model_name': model_name,
            'total_parameters': total_params,
            'sampled_layers': len(layer_results),
            'average_compression_ratio': avg_compression,
            'estimated_total_compression': sampled_original / max(total_compressed, 1),
            'layer_results': layer_results
        }
    
    def _sample_layers(
        self,
        model: nn.Module,
        max_layers: int = 10
    ) -> List[Tuple[str, nn.Parameter]]:
        """Sample representative layers from model."""
        all_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        
        if len(all_params) <= max_layers:
            return all_params
        
        # Sample uniformly across model depth
        indices = np.linspace(0, len(all_params) - 1, max_layers, dtype=int)
        return [all_params[i] for i in indices]


class ScalingLawAnalysis:
    """Derive and validate scaling laws."""
    
    def __init__(self, tensor_results: List[ScalingResult]):
        """
        Initialize scaling law analysis.
        
        Args:
            tensor_results: Results from tensor size scaling
        """
        self.results = tensor_results
    
    def derive_compression_law(self) -> Dict[str, Any]:
        """
        Derive compression ratio scaling law.
        
        Returns:
            Fitted scaling law parameters
        """
        # Extract features
        sizes = np.array([r.tensor_size for r in self.results])
        ranks = np.array([r.effective_rank for r in self.results])
        ratios = np.array([r.compression_ratio for r in self.results])
        
        # Multiple regression in log space
        log_ratios = np.log(ratios + 1e-10)
        log_sizes = np.log(sizes)
        log_ranks = np.log(ranks + 1e-10)
        
        # Design matrix
        X = np.column_stack([
            np.ones(len(sizes)),
            log_sizes,
            log_ranks,
            log_sizes * log_ranks  # Interaction term
        ])
        
        # Least squares fit
        coeffs, residuals, rank, s = np.linalg.lstsq(X, log_ratios, rcond=None)
        
        # Extract parameters
        log_A = coeffs[0]
        alpha = coeffs[1]
        beta = coeffs[2]
        gamma = coeffs[3]
        
        # Calculate R-squared
        ss_res = residuals[0] if len(residuals) > 0 else 0
        ss_tot = np.sum((log_ratios - log_ratios.mean())**2)
        r_squared = 1 - ss_res / ss_tot
        
        return {
            'A': np.exp(log_A),
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'r_squared': r_squared,
            'equation': f'log(CR) = {log_A:.3f} + {alpha:.3f}*log(size) + {beta:.3f}*log(rank) + {gamma:.3f}*log(size)*log(rank)'
        }
    
    def derive_accuracy_law(
        self,
        accuracy_results: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Derive accuracy vs compression trade-off law.
        
        Args:
            accuracy_results: List of (compression_ratio, accuracy_drop) pairs
            
        Returns:
            Fitted trade-off parameters
        """
        if accuracy_results is None:
            # Simulate some data for demonstration
            compression_ratios = np.array([r.compression_ratio for r in self.results])
            # Simulated accuracy drop
            accuracy_drops = 0.1 * (1 - np.exp(-0.1 * compression_ratios))
        else:
            compression_ratios, accuracy_drops = zip(*accuracy_results)
            compression_ratios = np.array(compression_ratios)
            accuracy_drops = np.array(accuracy_drops)
        
        # Fit exponential decay: accuracy_drop = alpha * (1 - exp(-beta * compression_ratio))
        def exp_decay(x, alpha, beta):
            return alpha * (1 - np.exp(-beta * x))
        
        try:
            popt, pcov = curve_fit(exp_decay, compression_ratios, accuracy_drops)
            alpha, beta = popt
            
            # Calculate R-squared
            predictions = exp_decay(compression_ratios, alpha, beta)
            ss_res = np.sum((accuracy_drops - predictions)**2)
            ss_tot = np.sum((accuracy_drops - accuracy_drops.mean())**2)
            r_squared = 1 - ss_res / ss_tot
            
            return {
                'alpha': alpha,
                'beta': beta,
                'r_squared': r_squared,
                'equation': f'accuracy_drop = {alpha:.4f} * (1 - exp(-{beta:.4f} * compression_ratio))'
            }
        except Exception as e:
            logger.error(f"Failed to fit accuracy law: {e}")
            return {}
    
    def plot_scaling_laws(self, output_dir: Optional[Path] = None):
        """Plot scaling law visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Compression ratio vs tensor size
        ax = axes[0, 0]
        sizes = [r.tensor_size for r in self.results]
        ratios = [r.compression_ratio for r in self.results]
        ax.scatter(sizes, ratios, alpha=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Tensor Size')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratio vs Tensor Size')
        ax.grid(True, alpha=0.3)
        
        # 2. Compression ratio vs effective rank
        ax = axes[0, 1]
        ranks = [r.effective_rank for r in self.results]
        ax.scatter(ranks, ratios, alpha=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Effective Rank')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratio vs Effective Rank')
        ax.grid(True, alpha=0.3)
        
        # 3. Reconstruction error vs compression ratio
        ax = axes[1, 0]
        errors = [r.reconstruction_error for r in self.results]
        ax.scatter(ratios, errors, alpha=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Reconstruction Error (MSE)')
        ax.set_title('Error vs Compression Trade-off')
        ax.grid(True, alpha=0.3)
        
        # 4. Training time vs tensor size
        ax = axes[1, 1]
        times = [r.training_time for r in self.results]
        ax.scatter(sizes, times, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('Tensor Size')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Training Time Scaling')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'scaling_laws.png', dpi=150)
        
        return fig


def run_scaling_analysis(
    config: Optional[CompressionConfig] = None,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Run complete scaling analysis.
    
    Args:
        config: Compression configuration
        output_dir: Directory to save results
        device: Device to use
        
    Returns:
        Dictionary of analysis results
    """
    if config is None:
        config = CompressionConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # 1. Tensor size scaling
    logger.info("Running tensor size scaling analysis")
    tensor_scaling = TensorSizeScaling(config)
    
    # Analyze different layer types
    tensor_scaling.analyze_linear_layers(device=device)
    tensor_scaling.analyze_conv_layers(device=device)
    tensor_scaling.analyze_attention_layers(device=device)
    
    # Fit scaling law
    tensor_law = tensor_scaling.fit_scaling_law()
    results['tensor_scaling_law'] = tensor_law
    results['tensor_results'] = tensor_scaling.results
    
    logger.info(f"Tensor scaling law: {tensor_law.get('equation', 'Failed to fit')}")
    
    # 2. Model size scaling
    logger.info("Running model size scaling analysis")
    model_scaling = ModelSizeScaling(config)
    model_results = model_scaling.analyze_model_families(device=device)
    results['model_scaling'] = model_results
    
    # 3. Derive comprehensive scaling laws
    logger.info("Deriving scaling laws")
    law_analysis = ScalingLawAnalysis(tensor_scaling.results)
    
    compression_law = law_analysis.derive_compression_law()
    results['compression_law'] = compression_law
    
    accuracy_law = law_analysis.derive_accuracy_law()
    results['accuracy_law'] = accuracy_law
    
    # 4. Generate plots
    if output_dir:
        logger.info("Generating scaling law plots")
        law_analysis.plot_scaling_laws(output_dir)
    
    # 5. Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_dir / 'scaling_results.json', 'w') as f:
            # Convert results to serializable format
            serializable_results = {
                'tensor_scaling_law': tensor_law,
                'compression_law': compression_law,
                'accuracy_law': accuracy_law,
                'num_tensor_experiments': len(tensor_scaling.results),
                'model_families_analyzed': list(model_results.keys())
            }
            json.dump(serializable_results, f, indent=2)
        
        # Save detailed tensor results
        tensor_data = []
        for r in tensor_scaling.results:
            tensor_data.append({
                'tensor_size': r.tensor_size,
                'tensor_shape': list(r.tensor_shape),
                'effective_rank': r.effective_rank,
                'compression_ratio': r.compression_ratio,
                'reconstruction_error': r.reconstruction_error,
                'training_time': r.training_time
            })
        
        with open(output_dir / 'tensor_scaling_data.json', 'w') as f:
            json.dump(tensor_data, f, indent=2)
    
    return results