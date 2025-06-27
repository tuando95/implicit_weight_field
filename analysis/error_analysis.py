"""Error analysis and failure case detection for compression."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Represents an error pattern in compression."""
    layer_name: str
    error_type: str
    severity: float
    description: str
    metrics: Dict[str, float]
    suggested_action: str


@dataclass 
class FailureCase:
    """Represents a compression failure case."""
    layer_name: str
    failure_type: str
    error_metrics: Dict[str, float]
    recovery_attempted: bool
    recovery_successful: bool
    final_compression_ratio: float
    notes: str


class CompressionErrorAnalyzer:
    """Analyze compression errors and identify patterns."""
    
    def __init__(self, tolerance_mse: float = 1e-4):
        """
        Initialize error analyzer.
        
        Args:
            tolerance_mse: MSE tolerance for acceptable compression
        """
        self.tolerance_mse = tolerance_mse
        self.error_patterns = []
        self.failure_cases = []
    
    def analyze_reconstruction_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        layer_name: str
    ) -> Dict[str, Any]:
        """
        Comprehensive error analysis for a layer.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            layer_name: Name of the layer
            
        Returns:
            Error analysis results
        """
        # Basic error metrics
        error = original - reconstructed
        mse = error.pow(2).mean().item()
        rmse = np.sqrt(mse)
        mae = error.abs().mean().item()
        max_error = error.abs().max().item()
        
        # Relative errors
        rel_mse = mse / (original.pow(2).mean().item() + 1e-10)
        rel_mae = mae / (original.abs().mean().item() + 1e-10)
        
        # Error distribution analysis
        error_flat = error.flatten().cpu().numpy()
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'relative_mse': rel_mse,
            'relative_mae': rel_mae,
            'error_mean': error_flat.mean(),
            'error_std': error_flat.std(),
            'error_skewness': stats.skew(error_flat),
            'error_kurtosis': stats.kurtosis(error_flat),
            'error_percentiles': {
                '1%': np.percentile(error_flat, 1),
                '5%': np.percentile(error_flat, 5),
                '25%': np.percentile(error_flat, 25),
                '50%': np.percentile(error_flat, 50),
                '75%': np.percentile(error_flat, 75),
                '95%': np.percentile(error_flat, 95),
                '99%': np.percentile(error_flat, 99)
            }
        }
        
        # Spatial error analysis
        if original.dim() >= 2:
            results['spatial_error'] = self._analyze_spatial_error(error)
        
        # Frequency domain error
        results['frequency_error'] = self._analyze_frequency_error(original, reconstructed)
        
        # Error patterns
        patterns = self._detect_error_patterns(error, original, results)
        results['patterns'] = patterns
        
        # Check for failure
        if mse > self.tolerance_mse:
            self._record_failure(layer_name, results)
        
        return results
    
    def _analyze_spatial_error(self, error: torch.Tensor) -> Dict[str, float]:
        """Analyze spatial distribution of errors."""
        # Reshape to 2D if needed
        if error.dim() > 2:
            error_2d = error.flatten(0, -2)
        else:
            error_2d = error
        
        # Compute spatial statistics
        row_errors = error_2d.abs().mean(dim=1)
        col_errors = error_2d.abs().mean(dim=0)
        
        return {
            'row_error_std': row_errors.std().item(),
            'col_error_std': col_errors.std().item(),
            'spatial_clustering': self._compute_spatial_clustering(error_2d),
            'edge_vs_center_ratio': self._compute_edge_center_ratio(error_2d)
        }
    
    def _compute_spatial_clustering(self, error: torch.Tensor) -> float:
        """Compute spatial clustering of errors using Moran's I."""
        if error.shape[0] < 3 or error.shape[1] < 3:
            return 0.0
        
        # Simplified Moran's I
        error_np = error.abs().cpu().numpy()
        mean_error = error_np.mean()
        
        # Compute spatial weights (adjacent cells)
        n_rows, n_cols = error_np.shape
        spatial_corr = 0.0
        weight_sum = 0.0
        
        for i in range(1, n_rows - 1):
            for j in range(1, n_cols - 1):
                center = error_np[i, j] - mean_error
                
                # Check 4-connected neighbors
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = error_np[i + di, j + dj] - mean_error
                    spatial_corr += center * neighbor
                    weight_sum += 1
        
        if weight_sum > 0:
            variance = ((error_np - mean_error) ** 2).mean()
            moran_i = spatial_corr / (weight_sum * variance + 1e-10)
            return float(moran_i)
        
        return 0.0
    
    def _compute_edge_center_ratio(self, error: torch.Tensor) -> float:
        """Compute ratio of edge errors to center errors."""
        if error.shape[0] < 5 or error.shape[1] < 5:
            return 1.0
        
        # Define edge and center regions
        edge_mask = torch.zeros_like(error, dtype=torch.bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        
        edge_error = error[edge_mask].abs().mean()
        center_error = error[~edge_mask].abs().mean()
        
        if center_error > 1e-10:
            return (edge_error / center_error).item()
        return 1.0
    
    def _analyze_frequency_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze error in frequency domain."""
        # Reshape to 2D
        if original.dim() > 2:
            orig_2d = original.flatten(0, -2)
            recon_2d = reconstructed.flatten(0, -2)
        else:
            orig_2d = original
            recon_2d = reconstructed
        
        # Compute FFT
        fft_orig = torch.fft.fft2(orig_2d)
        fft_recon = torch.fft.fft2(recon_2d)
        
        # Compute magnitude spectrum
        mag_orig = fft_orig.abs()
        mag_recon = fft_recon.abs()
        
        # Low frequency error (center of spectrum)
        h, w = mag_orig.shape
        center_h, center_w = h // 4, w // 4
        low_freq_region = mag_orig[h//2-center_h:h//2+center_h,
                                   w//2-center_w:w//2+center_w]
        low_freq_error = (mag_orig[h//2-center_h:h//2+center_h,
                                  w//2-center_w:w//2+center_w] - 
                         mag_recon[h//2-center_h:h//2+center_h,
                                  w//2-center_w:w//2+center_w]).abs().mean()
        
        # High frequency error (edges of spectrum)
        high_freq_error = ((mag_orig - mag_recon).abs().sum() - 
                          low_freq_error * center_h * center_w * 4) / (h * w - center_h * center_w * 4)
        
        return {
            'low_freq_error': low_freq_error.item(),
            'high_freq_error': high_freq_error.item(),
            'freq_error_ratio': (high_freq_error / (low_freq_error + 1e-10)).item()
        }
    
    def _detect_error_patterns(
        self,
        error: torch.Tensor,
        original: torch.Tensor,
        metrics: Dict[str, Any]
    ) -> List[ErrorPattern]:
        """Detect specific error patterns."""
        patterns = []
        
        # High frequency loss
        if metrics['frequency_error']['freq_error_ratio'] > 10:
            patterns.append(ErrorPattern(
                layer_name="",
                error_type="high_frequency_loss",
                severity=min(metrics['frequency_error']['freq_error_ratio'] / 10, 1.0),
                description="Significant loss of high-frequency components",
                metrics={'freq_ratio': metrics['frequency_error']['freq_error_ratio']},
                suggested_action="Increase bandwidth or hidden width"
            ))
        
        # Spatial clustering
        if 'spatial_error' in metrics and metrics['spatial_error']['spatial_clustering'] > 0.5:
            patterns.append(ErrorPattern(
                layer_name="",
                error_type="spatial_clustering",
                severity=metrics['spatial_error']['spatial_clustering'],
                description="Errors are spatially clustered",
                metrics={'clustering': metrics['spatial_error']['spatial_clustering']},
                suggested_action="Consider multi-scale decomposition"
            ))
        
        # Outlier reconstruction
        if metrics['error_percentiles']['99%'] > 10 * metrics['error_percentiles']['50%']:
            patterns.append(ErrorPattern(
                layer_name="",
                error_type="outlier_errors", 
                severity=0.8,
                description="Large errors on outlier weights",
                metrics={'outlier_ratio': metrics['error_percentiles']['99%'] / metrics['error_percentiles']['50%']},
                suggested_action="Add regularization or use robust loss"
            ))
        
        # Systematic bias
        if abs(metrics['error_mean']) > 0.1 * metrics['error_std']:
            patterns.append(ErrorPattern(
                layer_name="",
                error_type="systematic_bias",
                severity=abs(metrics['error_mean']) / metrics['error_std'],
                description="Systematic bias in reconstruction",
                metrics={'bias': metrics['error_mean'], 'std': metrics['error_std']},
                suggested_action="Check field initialization or add bias correction"
            ))
        
        # Edge artifacts
        if 'spatial_error' in metrics and metrics['spatial_error']['edge_vs_center_ratio'] > 2.0:
            patterns.append(ErrorPattern(
                layer_name="",
                error_type="edge_artifacts",
                severity=min((metrics['spatial_error']['edge_vs_center_ratio'] - 1) / 3, 1.0),
                description="Higher errors at tensor edges",
                metrics={'edge_ratio': metrics['spatial_error']['edge_vs_center_ratio']},
                suggested_action="Use padding or boundary-aware encoding"
            ))
        
        return patterns
    
    def _record_failure(self, layer_name: str, error_metrics: Dict[str, Any]):
        """Record a compression failure case."""
        failure = FailureCase(
            layer_name=layer_name,
            failure_type="high_reconstruction_error",
            error_metrics={
                'mse': error_metrics['mse'],
                'relative_mse': error_metrics['relative_mse'],
                'max_error': error_metrics['max_error']
            },
            recovery_attempted=False,
            recovery_successful=False,
            final_compression_ratio=0.0,
            notes=f"MSE {error_metrics['mse']:.6f} exceeds tolerance {self.tolerance_mse}"
        )
        self.failure_cases.append(failure)
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across all failure cases."""
        if not self.failure_cases:
            return {"num_failures": 0}
        
        # Group failures by type
        failure_types = {}
        for failure in self.failure_cases:
            if failure.failure_type not in failure_types:
                failure_types[failure.failure_type] = []
            failure_types[failure.failure_type].append(failure)
        
        # Analyze each failure type
        analysis = {
            "num_failures": len(self.failure_cases),
            "failure_types": {},
            "recovery_success_rate": sum(f.recovery_successful for f in self.failure_cases) / len(self.failure_cases),
            "avg_mse": np.mean([f.error_metrics.get('mse', 0) for f in self.failure_cases])
        }
        
        for ftype, failures in failure_types.items():
            analysis["failure_types"][ftype] = {
                "count": len(failures),
                "layers": [f.layer_name for f in failures],
                "avg_severity": np.mean([f.error_metrics.get('mse', 0) for f in failures])
            }
        
        return analysis


class AdaptiveCompressionStrategy:
    """Adaptive strategies for handling compression failures."""
    
    def __init__(self):
        """Initialize adaptive strategy handler."""
        self.strategies = {
            'high_frequency_loss': self._handle_high_frequency_loss,
            'spatial_clustering': self._handle_spatial_clustering,
            'outlier_errors': self._handle_outlier_errors,
            'systematic_bias': self._handle_systematic_bias,
            'edge_artifacts': self._handle_edge_artifacts
        }
    
    def suggest_recovery_strategy(
        self,
        error_patterns: List[ErrorPattern],
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest recovery strategy based on error patterns.
        
        Args:
            error_patterns: Detected error patterns
            current_config: Current compression configuration
            
        Returns:
            Updated configuration
        """
        updated_config = current_config.copy()
        
        # Apply strategies for each pattern
        for pattern in error_patterns:
            if pattern.error_type in self.strategies:
                strategy_update = self.strategies[pattern.error_type](
                    pattern, updated_config
                )
                updated_config.update(strategy_update)
        
        return updated_config
    
    def _handle_high_frequency_loss(
        self,
        pattern: ErrorPattern,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle high frequency loss pattern."""
        updates = {}
        
        # Increase bandwidth
        current_bandwidth = config.get('bandwidth', 4)
        updates['bandwidth'] = min(current_bandwidth * 2, 32)
        
        # Increase hidden width slightly
        current_hidden = config.get('hidden_width', 256)
        updates['hidden_width'] = int(current_hidden * 1.25)
        
        # Increase frequency parameter for SIREN
        updates['w0'] = config.get('w0', 30.0) * 1.5
        
        return updates
    
    def _handle_spatial_clustering(
        self,
        pattern: ErrorPattern,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle spatial clustering pattern."""
        updates = {}
        
        # Enable multi-scale decomposition
        updates['use_multiscale'] = True
        updates['num_scales'] = 3
        
        # Increase model capacity
        current_hidden = config.get('hidden_width', 256)
        updates['hidden_width'] = int(current_hidden * 1.5)
        
        return updates
    
    def _handle_outlier_errors(
        self,
        pattern: ErrorPattern,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle outlier errors pattern."""
        updates = {}
        
        # Use robust loss
        updates['loss_type'] = 'huber'
        updates['huber_delta'] = 0.1
        
        # Increase regularization
        current_reg = config.get('regularization', 1e-6)
        updates['regularization'] = current_reg * 10
        
        # Clip gradients more aggressively
        updates['gradient_clip'] = 0.5
        
        return updates
    
    def _handle_systematic_bias(
        self,
        pattern: ErrorPattern,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle systematic bias pattern."""
        updates = {}
        
        # Add bias correction layer
        updates['use_bias_correction'] = True
        
        # Adjust learning rate schedule
        updates['use_lr_schedule'] = True
        updates['lr_schedule_type'] = 'cosine_restarts'
        
        return updates
    
    def _handle_edge_artifacts(
        self,
        pattern: ErrorPattern,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle edge artifacts pattern."""
        updates = {}
        
        # Use padded coordinates
        updates['coordinate_padding'] = 0.1
        
        # Increase bandwidth for better boundary representation
        current_bandwidth = config.get('bandwidth', 4)
        updates['bandwidth'] = current_bandwidth + 2
        
        return updates


class CompressionDiagnostics:
    """Comprehensive diagnostics for compression quality."""
    
    def __init__(self):
        """Initialize diagnostics."""
        self.diagnostic_results = {}
    
    def run_diagnostics(
        self,
        model: nn.Module,
        compressed_model: nn.Module,
        test_loader: Any,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics on compressed model.
        
        Args:
            model: Original model
            compressed_model: Compressed model  
            test_loader: Test data loader
            device: Device to use
            
        Returns:
            Diagnostic results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        results = {}
        
        # Weight fidelity
        results['weight_fidelity'] = self._check_weight_fidelity(model, compressed_model)
        
        # Output consistency
        results['output_consistency'] = self._check_output_consistency(
            model, compressed_model, test_loader, device
        )
        
        # Gradient flow
        results['gradient_flow'] = self._check_gradient_flow(
            compressed_model, test_loader, device
        )
        
        # Numerical stability
        results['numerical_stability'] = self._check_numerical_stability(
            compressed_model, test_loader, device
        )
        
        return results
    
    def _check_weight_fidelity(
        self,
        model: nn.Module,
        compressed_model: nn.Module
    ) -> Dict[str, float]:
        """Check weight reconstruction fidelity."""
        fidelity_scores = {}
        
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            compressed_model.named_parameters()
        ):
            if name1 == name2:
                # Compute correlation
                corr = torch.corrcoef(torch.stack([
                    param1.flatten(),
                    param2.flatten()
                ]))[0, 1].item()
                
                fidelity_scores[name1] = {
                    'correlation': corr,
                    'mse': (param1 - param2).pow(2).mean().item(),
                    'cosine_sim': torch.nn.functional.cosine_similarity(
                        param1.flatten().unsqueeze(0),
                        param2.flatten().unsqueeze(0)
                    ).item()
                }
        
        return fidelity_scores
    
    def _check_output_consistency(
        self,
        model: nn.Module,
        compressed_model: nn.Module,
        test_loader: Any,
        device: torch.device
    ) -> Dict[str, float]:
        """Check output consistency between models."""
        model.eval()
        compressed_model.eval()
        
        total_mse = 0
        total_cosine_sim = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                
                output_orig = model(data)
                output_comp = compressed_model(data)
                
                # MSE
                mse = (output_orig - output_comp).pow(2).mean()
                total_mse += mse.item() * len(data)
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    output_orig.view(len(data), -1),
                    output_comp.view(len(data), -1)
                ).mean()
                total_cosine_sim += cos_sim.item() * len(data)
                
                total_samples += len(data)
                
                if total_samples > 1000:  # Limit samples
                    break
        
        return {
            'avg_mse': total_mse / total_samples,
            'avg_cosine_similarity': total_cosine_sim / total_samples
        }
    
    def _check_gradient_flow(
        self,
        model: nn.Module,
        test_loader: Any,
        device: torch.device
    ) -> Dict[str, Any]:
        """Check gradient flow through compressed model."""
        model.train()
        
        # Get single batch
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        if output.shape[-1] > 1:
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            loss = nn.MSELoss()(output.squeeze(), target.float())
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Analyze gradients
        gradient_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                gradient_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item()
                }
        
        return gradient_stats
    
    def _check_numerical_stability(
        self,
        model: nn.Module,
        test_loader: Any,
        device: torch.device
    ) -> Dict[str, bool]:
        """Check numerical stability of compressed model."""
        model.eval()
        
        stability_checks = {
            'output_bounded': True,
            'no_nan_outputs': True,
            'no_inf_outputs': True,
            'activation_stable': True
        }
        
        # Test with normal and edge case inputs
        test_inputs = []
        
        # Normal input
        data, _ = next(iter(test_loader))
        test_inputs.append(('normal', data.to(device)))
        
        # Very small values
        test_inputs.append(('small', data.to(device) * 1e-8))
        
        # Large values
        test_inputs.append(('large', data.to(device) * 1e3))
        
        # Noisy input
        test_inputs.append(('noisy', data.to(device) + torch.randn_like(data) * 0.1))
        
        with torch.no_grad():
            for input_type, test_data in test_inputs:
                try:
                    output = model(test_data)
                    
                    # Check for NaN/Inf
                    if torch.isnan(output).any():
                        stability_checks['no_nan_outputs'] = False
                    if torch.isinf(output).any():
                        stability_checks['no_inf_outputs'] = False
                    
                    # Check if bounded
                    if output.abs().max() > 1e6:
                        stability_checks['output_bounded'] = False
                        
                except Exception as e:
                    stability_checks['activation_stable'] = False
                    logger.warning(f"Stability check failed for {input_type} input: {e}")
        
        return stability_checks


def create_error_analysis_report(
    error_analyzer: CompressionErrorAnalyzer,
    output_path: Path
) -> None:
    """
    Create comprehensive error analysis report.
    
    Args:
        error_analyzer: Error analyzer with results
        output_path: Path to save report
    """
    report = {
        'failure_analysis': error_analyzer.analyze_failure_patterns(),
        'error_patterns': [
            {
                'layer': p.layer_name,
                'type': p.error_type,
                'severity': p.severity,
                'description': p.description,
                'suggested_action': p.suggested_action
            }
            for p in error_analyzer.error_patterns
        ],
        'failure_cases': [
            {
                'layer': f.layer_name,
                'type': f.failure_type,
                'metrics': f.error_metrics,
                'recovery_attempted': f.recovery_attempted,
                'recovery_successful': f.recovery_successful,
                'notes': f.notes
            }
            for f in error_analyzer.failure_cases
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Error analysis report saved to {output_path}")