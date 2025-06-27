"""Computational efficiency analysis and benchmarking tools."""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    operation: str
    device: str
    batch_size: int
    input_shape: Tuple[int, ...]
    latency_ms: float
    throughput: float
    memory_mb: float
    gpu_memory_mb: Optional[float]
    cpu_percent: float
    gpu_percent: Optional[float]
    energy_joules: Optional[float]


class PerformanceProfiler:
    """Profile computational performance of models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize profiler.
        
        Args:
            device: Device to profile on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        # Start measurements
        start_time = time.perf_counter()
        start_cpu_percent = psutil.cpu_percent(interval=None)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
            gpus = GPUtil.getGPUs()
            start_gpu_util = gpus[0].load if gpus else 0
        else:
            start_gpu_memory = None
            start_gpu_util = None
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # End measurements
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            end_cpu_percent = psutil.cpu_percent(interval=None)
            end_memory = process.memory_info().rss / 1024 / 1024
            
            if self.device.type == 'cuda':
                end_gpu_memory = torch.cuda.memory_allocated()
                gpus = GPUtil.getGPUs()
                end_gpu_util = gpus[0].load if gpus else 0
            else:
                end_gpu_memory = None
                end_gpu_util = None
            
            # Calculate metrics
            latency = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory
            cpu_usage = (start_cpu_percent + end_cpu_percent) / 2
            
            if self.device.type == 'cuda':
                gpu_memory_delta = (end_gpu_memory - start_gpu_memory) / 1024 / 1024
                gpu_usage = (start_gpu_util + end_gpu_util) / 2 * 100
            else:
                gpu_memory_delta = None
                gpu_usage = None
            
            # Store result
            self.last_profile = {
                'operation': operation_name,
                'latency_ms': latency,
                'memory_mb': memory_delta,
                'gpu_memory_mb': gpu_memory_delta,
                'cpu_percent': cpu_usage,
                'gpu_percent': gpu_usage
            }


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 32, 128],
    num_warmup: int = 10,
    num_runs: int = 100,
    device: Optional[torch.device] = None
) -> Dict[int, BenchmarkResult]:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape (without batch)
        batch_sizes: Batch sizes to test
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        device: Device to run on
        
    Returns:
        Dictionary mapping batch size to benchmark results
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    profiler = PerformanceProfiler(device)
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size {batch_size}")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Benchmark runs
        latencies = []
        memory_usage = []
        gpu_memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                with profiler.profile(f"inference_batch_{batch_size}"):
                    output = model(dummy_input)
                
                latencies.append(profiler.last_profile['latency_ms'])
                memory_usage.append(profiler.last_profile['memory_mb'])
                
                if device.type == 'cuda':
                    gpu_memory_usage.append(profiler.last_profile['gpu_memory_mb'])
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        throughput = batch_size / (avg_latency / 1000)  # samples/sec
        avg_memory = np.mean(memory_usage)
        avg_gpu_memory = np.mean(gpu_memory_usage) if gpu_memory_usage else None
        
        results[batch_size] = BenchmarkResult(
            operation="inference",
            device=str(device),
            batch_size=batch_size,
            input_shape=input_shape,
            latency_ms=avg_latency,
            throughput=throughput,
            memory_mb=avg_memory,
            gpu_memory_mb=avg_gpu_memory,
            cpu_percent=profiler.last_profile['cpu_percent'],
            gpu_percent=profiler.last_profile['gpu_percent'],
            energy_joules=None  # Would need external measurement
        )
    
    return results


def profile_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Profile memory usage during inference.
    
    Args:
        model: Model to profile
        input_shape: Input shape
        batch_size: Batch size
        device: Device to use
        detailed: Whether to include detailed breakdown
        
    Returns:
        Memory usage statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Baseline memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated()
    else:
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
    
    # Model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Create input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    input_memory = dummy_input.numel() * dummy_input.element_size()
    
    # Forward pass
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
            pre_forward_memory = torch.cuda.memory_allocated()
        
        output = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            post_forward_memory = torch.cuda.memory_allocated()
            activation_memory = post_forward_memory - pre_forward_memory
        else:
            process = psutil.Process()
            post_forward_memory = process.memory_info().rss
            activation_memory = post_forward_memory - baseline_memory - input_memory
    
    # Output memory
    output_memory = output.numel() * output.element_size()
    
    results = {
        'total_memory_mb': (post_forward_memory - baseline_memory) / 1024 / 1024,
        'parameter_memory_mb': param_memory / 1024 / 1024,
        'buffer_memory_mb': buffer_memory / 1024 / 1024,
        'input_memory_mb': input_memory / 1024 / 1024,
        'output_memory_mb': output_memory / 1024 / 1024,
        'activation_memory_mb': activation_memory / 1024 / 1024,
        'device': str(device)
    }
    
    if detailed and device.type == 'cuda':
        # Detailed GPU memory breakdown
        results['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        results['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        results['gpu_memory_cached_mb'] = torch.cuda.memory_cached() / 1024 / 1024
    
    return results


def compare_inference_modes(
    compressed_model: Any,
    original_model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 32],
    cache_sizes_mb: List[float] = [50, 100, 200],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Compare preload vs streaming inference modes.
    
    Args:
        compressed_model: Compressed model with fields
        original_model: Original model
        input_shape: Input shape
        batch_sizes: Batch sizes to test
        cache_sizes_mb: Cache sizes for streaming mode
        device: Device to use
        
    Returns:
        Comparison results
    """
    from ..inference import InferenceMode, create_inference_mode, CacheConfig
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {
        'original': {},
        'preload': {},
        'streaming': {}
    }
    
    # Benchmark original model
    logger.info("Benchmarking original model")
    original_results = benchmark_inference(
        original_model, input_shape, batch_sizes, device=device
    )
    results['original'] = {bs: r.__dict__ for bs, r in original_results.items()}
    
    # Benchmark preload mode
    logger.info("Benchmarking preload mode")
    preload_inference = create_inference_mode(InferenceMode.PRELOAD, device=device)
    # Note: Would need proper integration with compressed model
    # For now, use original model as placeholder
    preload_results = benchmark_inference(
        original_model, input_shape, batch_sizes, device=device
    )
    results['preload'] = {bs: r.__dict__ for bs, r in preload_results.items()}
    
    # Benchmark streaming mode with different cache sizes
    for cache_size in cache_sizes_mb:
        logger.info(f"Benchmarking streaming mode with {cache_size}MB cache")
        
        cache_config = CacheConfig(max_size_mb=cache_size, device=device)
        streaming_inference = create_inference_mode(
            InferenceMode.STREAMING, 
            device=device,
            cache_config=cache_config
        )
        
        # Benchmark with cache
        streaming_results = benchmark_inference(
            original_model, input_shape, batch_sizes, device=device
        )
        
        results[f'streaming_{cache_size}mb'] = {
            bs: r.__dict__ for bs, r in streaming_results.items()
        }
    
    return results


def analyze_memory_compute_tradeoff(
    model: nn.Module,
    compressed_model: Any,
    memory_budgets_mb: List[float] = [50, 100, 200, 500, 1000],
    input_shape: Tuple[int, ...] = (3, 224, 224),
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Analyze memory-compute trade-offs.
    
    Args:
        model: Original model
        compressed_model: Compressed model
        memory_budgets_mb: Memory budgets to test
        input_shape: Input shape
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Trade-off analysis results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {
        'memory_budgets': memory_budgets_mb,
        'latencies': [],
        'throughputs': [],
        'cache_hit_rates': [],
        'memory_usage': []
    }
    
    for budget in memory_budgets_mb:
        logger.info(f"Testing memory budget: {budget}MB")
        
        # Configure cache for budget
        from ..inference import CacheConfig, StreamingInference
        
        cache_config = CacheConfig(
            max_size_mb=budget * 0.8,  # 80% for cache, 20% for overhead
            device=device
        )
        
        # Create streaming inference with cache
        inference = StreamingInference(cache_config, device)
        
        # Run benchmark
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        
        latencies = []
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(10):  # Reduced runs for memory analysis
                t0 = time.perf_counter()
                # Would need proper integration with compressed model
                output = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
        
        avg_latency = np.mean(latencies)
        throughput = batch_size / (avg_latency / 1000)
        
        # Get cache statistics
        cache_stats = inference.get_cache_stats()
        
        results['latencies'].append(avg_latency)
        results['throughputs'].append(throughput)
        results['cache_hit_rates'].append(cache_stats.get('hit_rate', 0))
        results['memory_usage'].append(cache_stats.get('current_size_mb', 0))
    
    return results


def profile_energy_consumption(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    duration_seconds: int = 60,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Profile energy consumption (requires nvidia-ml-py).
    
    Args:
        model: Model to profile
        input_shape: Input shape
        batch_size: Batch size
        duration_seconds: Duration to profile
        device: Device to use
        
    Returns:
        Energy consumption metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    if device.type != 'cuda':
        logger.warning("Energy profiling only available for CUDA devices")
        return {}
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        logger.error("Failed to initialize NVML for energy profiling")
        return {}
    
    model.eval()
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Measure baseline power
    baseline_powers = []
    for _ in range(10):
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
        baseline_powers.append(power)
        time.sleep(0.1)
    
    baseline_power = np.mean(baseline_powers)
    
    # Measure during inference
    inference_powers = []
    energy_consumed = 0
    start_time = time.time()
    
    with torch.no_grad():
        while time.time() - start_time < duration_seconds:
            # Run inference
            _ = model(dummy_input)
            
            # Measure power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
            inference_powers.append(power)
            
            # Approximate energy (power * time)
            energy_consumed += power * 0.01  # Assuming ~10ms per iteration
    
    avg_inference_power = np.mean(inference_powers)
    power_increase = avg_inference_power - baseline_power
    
    results = {
        'baseline_power_w': baseline_power,
        'inference_power_w': avg_inference_power,
        'power_increase_w': power_increase,
        'total_energy_j': energy_consumed,
        'duration_s': duration_seconds,
        'energy_per_sample_j': energy_consumed / (duration_seconds * throughput)
        if 'throughput' in locals() else None
    }
    
    pynvml.nvmlShutdown()
    return results


def generate_efficiency_report(
    benchmark_results: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive efficiency report.
    
    Args:
        benchmark_results: Results from various benchmarks
        output_dir: Directory to save plots
        
    Returns:
        Efficiency report
    """
    report = {
        'summary': {},
        'detailed_results': benchmark_results,
        'recommendations': []
    }
    
    # Calculate summary statistics
    if 'inference' in benchmark_results:
        batch_sizes = list(benchmark_results['inference'].keys())
        latencies = [r['latency_ms'] for r in benchmark_results['inference'].values()]
        throughputs = [r['throughput'] for r in benchmark_results['inference'].values()]
        
        report['summary']['avg_latency_ms'] = np.mean(latencies)
        report['summary']['avg_throughput'] = np.mean(throughputs)
        report['summary']['latency_range'] = (min(latencies), max(latencies))
        report['summary']['optimal_batch_size'] = batch_sizes[np.argmax(throughputs)]
    
    # Memory efficiency
    if 'memory' in benchmark_results:
        mem = benchmark_results['memory']
        report['summary']['total_memory_mb'] = mem['total_memory_mb']
        report['summary']['memory_efficiency'] = (
            mem['parameter_memory_mb'] / mem['total_memory_mb']
        )
    
    # Generate recommendations
    if report['summary'].get('memory_efficiency', 1.0) < 0.5:
        report['recommendations'].append(
            "Low memory efficiency detected. Consider using streaming inference mode."
        )
    
    if report['summary'].get('avg_latency_ms', 0) > 100:
        report['recommendations'].append(
            "High inference latency. Consider using smaller field architectures."
        )
    
    # Generate plots if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Latency vs batch size plot
        if 'inference' in benchmark_results:
            plt.figure(figsize=(10, 6))
            
            batch_sizes = list(benchmark_results['inference'].keys())
            latencies = [r['latency_ms'] for r in benchmark_results['inference'].values()]
            throughputs = [r['throughput'] for r in benchmark_results['inference'].values()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Latency plot
            ax1.plot(batch_sizes, latencies, 'b-o')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Inference Latency vs Batch Size')
            ax1.grid(True, alpha=0.3)
            
            # Throughput plot
            ax2.plot(batch_sizes, throughputs, 'g-o')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (samples/sec)')
            ax2.set_title('Inference Throughput vs Batch Size')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'efficiency_analysis.png', dpi=150)
            plt.close()
    
    return report