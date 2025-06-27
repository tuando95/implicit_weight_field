"""Evaluation metrics for implicit weight field compression."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from scipy import stats
from tqdm import tqdm

from ..compression.compressor import CompressionResult


@dataclass
class CompressionMetrics:
    """Metrics for compression effectiveness."""
    model_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    parameter_reduction: float
    bits_per_parameter: float
    layer_compression_ratios: Dict[str, float]


@dataclass
class AccuracyMetrics:
    """Metrics for model accuracy."""
    top1_accuracy: float
    top5_accuracy: float
    accuracy_retention: float
    task_specific_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]


@dataclass
class EfficiencyMetrics:
    """Metrics for computational efficiency."""
    inference_latency_ms: float
    memory_usage_mb: float
    energy_consumption_j: float
    throughput_samples_per_sec: float
    cache_hit_rate: Optional[float]
    memory_overhead_percent: float


@dataclass
class ReconstructionMetrics:
    """Metrics for weight reconstruction quality."""
    mse: float
    rmse: float
    max_error: float
    snr_db: float
    weight_distribution_ks_statistic: float
    weight_distribution_p_value: float
    spectral_error: float


def evaluate_compression(
    original_model: nn.Module,
    compression_result: CompressionResult
) -> CompressionMetrics:
    """
    Evaluate compression effectiveness.
    
    Args:
        original_model: Original uncompressed model
        compression_result: Result from compression
        
    Returns:
        Compression metrics
    """
    # Calculate original model size
    original_params = sum(p.numel() for p in original_model.parameters())
    original_size_mb = original_params * 4 / (1024 * 1024)  # float32
    
    # Calculate compressed size
    compressed_params = compression_result.compressed_params
    compressed_size_mb = compressed_params * 4 / (1024 * 1024)
    
    # Parameter reduction percentage
    param_reduction = (original_params - compressed_params) / original_params * 100
    
    # Bits per parameter
    bits_per_param = (compressed_size_mb * 1024 * 1024 * 8) / original_params
    
    # Layer-wise compression ratios
    layer_ratios = {
        name: result.compression_ratio
        for name, result in compression_result.layer_results.items()
    }
    
    return CompressionMetrics(
        model_size_mb=original_size_mb,
        compressed_size_mb=compressed_size_mb,
        compression_ratio=compression_result.compression_ratio,
        parameter_reduction=param_reduction,
        bits_per_parameter=bits_per_param,
        layer_compression_ratios=layer_ratios
    )


def evaluate_model_accuracy(
    model: nn.Module,
    dataloader: Any,
    task: str = "classification",
    num_classes: int = 1000,
    device: Optional[torch.device] = None
) -> AccuracyMetrics:
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        task: Task type ("classification", "detection", etc.)
        num_classes: Number of classes for classification
        device: Device to use
        
    Returns:
        Accuracy metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    if task == "classification":
        return _evaluate_classification(model, dataloader, num_classes, device)
    else:
        raise NotImplementedError(f"Task {task} not implemented")


def _evaluate_classification(
    model: nn.Module,
    dataloader: Any,
    num_classes: int,
    device: torch.device
) -> AccuracyMetrics:
    """Evaluate classification accuracy."""
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    
    # Calculate confidence interval using bootstrap
    n_bootstrap = 1000
    bootstrap_accs = []
    predictions = np.array(predictions)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(predictions), len(predictions), replace=True)
        bootstrap_acc = np.mean(predictions[indices] == predictions[indices])  # Placeholder
        bootstrap_accs.append(bootstrap_acc)
    
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    
    return AccuracyMetrics(
        top1_accuracy=top1_acc,
        top5_accuracy=top5_acc,
        accuracy_retention=1.0,  # To be set by comparison
        task_specific_scores={'top1': top1_acc, 'top5': top5_acc},
        confidence_interval=(ci_lower, ci_upper)
    )


def evaluate_reconstruction_quality(
    original_weights: Dict[str, torch.Tensor],
    reconstructed_weights: Dict[str, torch.Tensor]
) -> Dict[str, ReconstructionMetrics]:
    """
    Evaluate reconstruction quality for each layer.
    
    Args:
        original_weights: Original weight tensors
        reconstructed_weights: Reconstructed weight tensors
        
    Returns:
        Dictionary of reconstruction metrics per layer
    """
    metrics = {}
    
    for name in original_weights:
        if name not in reconstructed_weights:
            continue
        
        original = original_weights[name].flatten()
        reconstructed = reconstructed_weights[name].flatten()
        
        # MSE and RMSE
        mse = torch.mean((original - reconstructed) ** 2).item()
        rmse = np.sqrt(mse)
        
        # Max error
        max_error = torch.max(torch.abs(original - reconstructed)).item()
        
        # SNR
        signal_power = torch.var(original).item()
        noise_power = torch.var(original - reconstructed).item()
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Weight distribution preservation (KS test)
        ks_stat, p_value = stats.ks_2samp(
            original.cpu().numpy(),
            reconstructed.cpu().numpy()
        )
        
        # Spectral error (2D FFT for conv layers)
        if len(original_weights[name].shape) == 4:  # Conv layer
            original_fft = torch.fft.fft2(original_weights[name].mean(dim=[0, 1]))
            reconstructed_fft = torch.fft.fft2(reconstructed_weights[name].mean(dim=[0, 1]))
            spectral_error = torch.norm(original_fft - reconstructed_fft).item()
        else:
            spectral_error = 0.0
        
        metrics[name] = ReconstructionMetrics(
            mse=mse,
            rmse=rmse,
            max_error=max_error,
            snr_db=snr_db,
            weight_distribution_ks_statistic=ks_stat,
            weight_distribution_p_value=p_value,
            spectral_error=spectral_error
        )
    
    return metrics


def benchmark_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 32],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[torch.device] = None
) -> Dict[int, EfficiencyMetrics]:
    """
    Benchmark inference latency for different batch sizes.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape (without batch dimension)
        batch_sizes: Batch sizes to test
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to use
        
    Returns:
        Dictionary of efficiency metrics per batch size
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Synchronize CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        memory_usage = []
        
        for _ in range(num_runs):
            # Memory measurement
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # Time measurement
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Record metrics
            latencies.append((end_time - start_time) * 1000)  # ms
            
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append((peak_memory - start_memory) / (1024 * 1024))  # MB
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        throughput = batch_size / (avg_latency / 1000)  # samples/sec
        
        results[batch_size] = EfficiencyMetrics(
            inference_latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            energy_consumption_j=0.0,  # Requires external measurement
            throughput_samples_per_sec=throughput,
            cache_hit_rate=None,
            memory_overhead_percent=0.0
        )
    
    return results


def calculate_accuracy_retention(
    original_accuracy: float,
    compressed_accuracy: float
) -> float:
    """Calculate accuracy retention percentage."""
    return (compressed_accuracy / original_accuracy) * 100


def calculate_memory_overhead(
    original_memory_mb: float,
    compressed_memory_mb: float
) -> float:
    """Calculate memory overhead percentage."""
    return ((compressed_memory_mb - original_memory_mb) / original_memory_mb) * 100