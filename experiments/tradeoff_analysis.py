"""Compression-accuracy trade-off analysis."""

import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import seaborn as sns

from core.implicit_field import CompressionConfig
from compression import compress_model
from evaluation import evaluate_model_accuracy, evaluate_compression
from baselines.quantization import quantize_model_int8
from baselines.pruning import magnitude_prune
from baselines.decomposition import tensor_train_decomposition

logger = logging.getLogger(__name__)


@dataclass
class TradeoffPoint:
    """Single point in compression-accuracy trade-off."""
    method: str
    compression_ratio: float
    accuracy: float
    accuracy_drop: float
    inference_latency: Optional[float] = None
    memory_usage: Optional[float] = None
    config: Optional[Dict[str, Any]] = None


class CompressionAccuracyTradeoff:
    """Analyze compression-accuracy trade-offs."""
    
    def __init__(self, baseline_accuracy: float):
        """
        Initialize trade-off analysis.
        
        Args:
            baseline_accuracy: Original model accuracy
        """
        self.baseline_accuracy = baseline_accuracy
        self.points = []
    
    def add_point(self, point: TradeoffPoint):
        """Add a trade-off point."""
        self.points.append(point)
    
    def analyze_inwf_tradeoff(
        self,
        model: nn.Module,
        validation_loader: Any,
        compression_configs: List[CompressionConfig],
        device: Optional[torch.device] = None
    ) -> List[TradeoffPoint]:
        """
        Analyze INWF trade-off with different configurations.
        
        Args:
            model: Model to compress
            validation_loader: Validation data
            compression_configs: List of compression configurations
            device: Device to use
            
        Returns:
            List of trade-off points
        """
        if device is None:
            device = next(model.parameters()).device
        
        inwf_points = []
        
        for i, config in enumerate(compression_configs):
            logger.info(f"Testing INWF config {i+1}/{len(compression_configs)}")
            
            # Compress model
            compressed_model, results = compress_model(model, config, device)
            
            # Evaluate accuracy
            accuracy_metrics = evaluate_model_accuracy(
                compressed_model, validation_loader, device=device
            )
            
            # Create trade-off point
            point = TradeoffPoint(
                method="INWF",
                compression_ratio=results.compression_ratio,
                accuracy=accuracy_metrics.top1_accuracy,
                accuracy_drop=self.baseline_accuracy - accuracy_metrics.top1_accuracy,
                config={
                    'bandwidth': config.bandwidth,
                    'hidden_width': config.hidden_width,
                    'num_layers': config.num_layers
                }
            )
            
            inwf_points.append(point)
            self.add_point(point)
        
        return inwf_points
    
    def analyze_baseline_tradeoffs(
        self,
        model: nn.Module,
        validation_loader: Any,
        device: Optional[torch.device] = None
    ) -> Dict[str, List[TradeoffPoint]]:
        """
        Analyze baseline method trade-offs.
        
        Args:
            model: Model to compress
            validation_loader: Validation data
            device: Device to use
            
        Returns:
            Dictionary of baseline trade-off points
        """
        if device is None:
            device = next(model.parameters()).device
        
        baseline_points = {
            'quantization': [],
            'pruning': [],
            'decomposition': []
        }
        
        # Quantization trade-offs
        logger.info("Analyzing quantization trade-offs")
        
        # INT8 quantization
        int8_model = quantize_model_int8(model)
        int8_accuracy = evaluate_model_accuracy(
            int8_model, validation_loader, device=device
        )
        
        baseline_points['quantization'].append(TradeoffPoint(
            method="INT8",
            compression_ratio=4.0,  # FP32 to INT8
            accuracy=int8_accuracy.top1_accuracy,
            accuracy_drop=self.baseline_accuracy - int8_accuracy.top1_accuracy
        ))
        
        # Pruning trade-offs
        logger.info("Analyzing pruning trade-offs")
        
        for sparsity in [0.5, 0.7, 0.9, 0.95]:
            pruned_model = magnitude_prune(model, sparsity)
            
            # Evaluate
            pruned_accuracy = evaluate_model_accuracy(
                pruned_model, validation_loader, device=device
            )
            
            # Effective compression (considering sparse storage)
            compression_ratio = 1 / (1 - sparsity)
            
            baseline_points['pruning'].append(TradeoffPoint(
                method=f"Pruning_{int(sparsity*100)}%",
                compression_ratio=compression_ratio,
                accuracy=pruned_accuracy.top1_accuracy,
                accuracy_drop=self.baseline_accuracy - pruned_accuracy.top1_accuracy,
                config={'sparsity': sparsity}
            ))
        
        # Decomposition trade-offs
        logger.info("Analyzing decomposition trade-offs")
        
        for rank in [8, 16, 32]:
            try:
                decomposed_model = tensor_train_decomposition(
                    model,
                    config={'rank': rank}
                )
                
                decomposed_accuracy = evaluate_model_accuracy(
                    decomposed_model, validation_loader, device=device
                )
                
                # Estimate compression ratio
                original_params = sum(p.numel() for p in model.parameters())
                decomposed_params = sum(p.numel() for p in decomposed_model.parameters())
                compression_ratio = original_params / decomposed_params
                
                baseline_points['decomposition'].append(TradeoffPoint(
                    method=f"TT_rank{rank}",
                    compression_ratio=compression_ratio,
                    accuracy=decomposed_accuracy.top1_accuracy,
                    accuracy_drop=self.baseline_accuracy - decomposed_accuracy.top1_accuracy,
                    config={'rank': rank}
                ))
            except Exception as e:
                logger.error(f"Decomposition with rank {rank} failed: {e}")
        
        # Add all baseline points to main collection
        for method_points in baseline_points.values():
            for point in method_points:
                self.add_point(point)
        
        return baseline_points
    
    def fit_pareto_frontier(self) -> List[TradeoffPoint]:
        """
        Find Pareto-optimal points.
        
        Returns:
            List of Pareto-optimal trade-off points
        """
        pareto_points = []
        
        for candidate in self.points:
            is_pareto = True
            
            for other in self.points:
                if other == candidate:
                    continue
                
                # Check if other dominates candidate
                if (other.compression_ratio >= candidate.compression_ratio and
                    other.accuracy >= candidate.accuracy and
                    (other.compression_ratio > candidate.compression_ratio or
                     other.accuracy > candidate.accuracy)):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_points.append(candidate)
        
        # Sort by compression ratio
        pareto_points.sort(key=lambda p: p.compression_ratio)
        
        return pareto_points
    
    def fit_tradeoff_curve(self) -> Dict[str, Any]:
        """
        Fit empirical trade-off curve.
        
        Returns:
            Fitted curve parameters
        """
        if len(self.points) < 3:
            return {}
        
        # Extract data
        compressions = np.array([p.compression_ratio for p in self.points])
        accuracy_drops = np.array([p.accuracy_drop for p in self.points])
        
        # Fit exponential model: drop = a * (1 - exp(-b * compression))
        def exp_model(x, a, b):
            return a * (1 - np.exp(-b * x))
        
        try:
            popt, pcov = curve_fit(
                exp_model,
                compressions,
                accuracy_drops,
                bounds=(0, [100, 10])
            )
            
            # Calculate R-squared
            residuals = accuracy_drops - exp_model(compressions, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((accuracy_drops - np.mean(accuracy_drops))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'model': 'exponential',
                'parameters': {'a': popt[0], 'b': popt[1]},
                'r_squared': r_squared,
                'equation': f'accuracy_drop = {popt[0]:.3f} * (1 - exp(-{popt[1]:.3f} * compression_ratio))'
            }
        except Exception as e:
            logger.error(f"Failed to fit trade-off curve: {e}")
            return {}
    
    def plot_tradeoff(
        self,
        output_path: Optional[Path] = None,
        include_pareto: bool = True
    ):
        """
        Plot compression-accuracy trade-off.
        
        Args:
            output_path: Path to save plot
            include_pareto: Whether to highlight Pareto frontier
        """
        plt.figure(figsize=(10, 6))
        
        # Group points by method
        method_groups = {}
        for point in self.points:
            method = point.method.split('_')[0]  # Extract base method
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(point)
        
        # Plot each method
        colors = plt.cm.tab10(np.linspace(0, 1, len(method_groups)))
        
        for (method, points), color in zip(method_groups.items(), colors):
            compressions = [p.compression_ratio for p in points]
            accuracies = [p.accuracy for p in points]
            
            plt.scatter(compressions, accuracies, label=method, 
                       color=color, s=100, alpha=0.7)
            
            # Connect points for same method
            if len(points) > 1:
                sorted_points = sorted(points, key=lambda p: p.compression_ratio)
                comp_sorted = [p.compression_ratio for p in sorted_points]
                acc_sorted = [p.accuracy for p in sorted_points]
                plt.plot(comp_sorted, acc_sorted, color=color, alpha=0.3)
        
        # Plot Pareto frontier
        if include_pareto:
            pareto_points = self.fit_pareto_frontier()
            if pareto_points:
                pareto_comp = [p.compression_ratio for p in pareto_points]
                pareto_acc = [p.accuracy for p in pareto_points]
                plt.plot(pareto_comp, pareto_acc, 'k--', 
                        linewidth=2, label='Pareto Frontier')
        
        # Add baseline accuracy line
        plt.axhline(y=self.baseline_accuracy, color='gray', 
                   linestyle=':', label='Baseline')
        
        plt.xlabel('Compression Ratio', fontsize=12)
        plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
        plt.title('Compression-Accuracy Trade-off', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive trade-off report."""
        report = {
            'baseline_accuracy': self.baseline_accuracy,
            'num_experiments': len(self.points),
            'methods_tested': list(set(p.method.split('_')[0] for p in self.points)),
            'compression_range': (
                min(p.compression_ratio for p in self.points),
                max(p.compression_ratio for p in self.points)
            ),
            'accuracy_range': (
                min(p.accuracy for p in self.points),
                max(p.accuracy for p in self.points)
            )
        }
        
        # Best points per method
        method_best = {}
        for point in self.points:
            method = point.method.split('_')[0]
            if method not in method_best:
                method_best[method] = point
            else:
                # Update if better (higher accuracy at similar compression)
                if (point.accuracy > method_best[method].accuracy and
                    point.compression_ratio >= method_best[method].compression_ratio * 0.9):
                    method_best[method] = point
        
        report['best_per_method'] = {
            method: {
                'compression_ratio': point.compression_ratio,
                'accuracy': point.accuracy,
                'accuracy_retention': (point.accuracy / self.baseline_accuracy) * 100
            }
            for method, point in method_best.items()
        }
        
        # Pareto-optimal points
        pareto_points = self.fit_pareto_frontier()
        report['pareto_points'] = [
            {
                'method': p.method,
                'compression_ratio': p.compression_ratio,
                'accuracy': p.accuracy
            }
            for p in pareto_points
        ]
        
        # Fitted curve
        curve_fit = self.fit_tradeoff_curve()
        report['fitted_curve'] = curve_fit
        
        return report


def analyze_layer_sensitivity(
    model: nn.Module,
    validation_loader: Any,
    compression_config: CompressionConfig,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Analyze sensitivity of different layers to compression.
    
    Args:
        model: Model to analyze
        validation_loader: Validation data
        compression_config: Compression configuration
        device: Device to use
        
    Returns:
        Layer sensitivity scores
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get baseline accuracy
    baseline_accuracy = evaluate_model_accuracy(
        model, validation_loader, device=device
    ).top1_accuracy
    
    layer_sensitivities = {}
    
    # Test compression impact per layer
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        
        if module.weight.numel() < 10000:  # Skip small layers
            continue
        
        logger.info(f"Testing sensitivity of layer: {name}")
        
        # Compress only this layer
        compressed_model, _ = compress_model(
            model,
            compression_config,
            layer_filter=[name],
            device=device
        )
        
        # Evaluate impact
        compressed_accuracy = evaluate_model_accuracy(
            compressed_model, validation_loader, device=device
        ).top1_accuracy
        
        sensitivity = baseline_accuracy - compressed_accuracy
        layer_sensitivities[name] = sensitivity
    
    # Normalize sensitivities
    max_sensitivity = max(layer_sensitivities.values())
    if max_sensitivity > 0:
        layer_sensitivities = {
            k: v / max_sensitivity for k, v in layer_sensitivities.items()
        }
    
    return layer_sensitivities


def find_optimal_compression(
    model: nn.Module,
    validation_loader: Any,
    target_accuracy_drop: float = 1.0,
    max_compression_ratio: float = 50.0,
    device: Optional[torch.device] = None
) -> Tuple[CompressionConfig, TradeoffPoint]:
    """
    Find optimal compression configuration for target accuracy.
    
    Args:
        model: Model to compress
        validation_loader: Validation data
        target_accuracy_drop: Maximum acceptable accuracy drop (%)
        max_compression_ratio: Maximum compression ratio to consider
        device: Device to use
        
    Returns:
        Optimal configuration and resulting trade-off point
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get baseline
    baseline_accuracy = evaluate_model_accuracy(
        model, validation_loader, device=device
    ).top1_accuracy
    
    # Binary search for optimal configuration
    min_width = 64
    max_width = 512
    best_config = None
    best_point = None
    
    while max_width - min_width > 32:
        mid_width = (min_width + max_width) // 2
        
        config = CompressionConfig(
            hidden_width=mid_width,
            bandwidth=4,
            num_layers=2
        )
        
        # Test configuration
        compressed_model, results = compress_model(model, config, device)
        accuracy = evaluate_model_accuracy(
            compressed_model, validation_loader, device=device
        ).top1_accuracy
        
        accuracy_drop = baseline_accuracy - accuracy
        
        if accuracy_drop <= target_accuracy_drop:
            # Acceptable drop, try smaller width
            best_config = config
            best_point = TradeoffPoint(
                method="INWF_optimal",
                compression_ratio=results.compression_ratio,
                accuracy=accuracy,
                accuracy_drop=accuracy_drop
            )
            max_width = mid_width
        else:
            # Too much drop, increase width
            min_width = mid_width
    
    return best_config, best_point