"""Example: Streaming inference with memory constraints."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
import psutil
import argparse

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from compression import compress_model
from core.implicit_field import CompressionConfig
from inference import StreamingInference, PreloadInference


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class DemoModel(nn.Module):
    """Demo model with multiple large layers."""
    def __init__(self, width=1024):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width * 2)
        self.fc3 = nn.Linear(width * 2, width)
        self.fc4 = nn.Linear(width, width // 2)
        self.fc5 = nn.Linear(width // 2, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def create_streaming_model(compressed_fields, cache_size_mb=10):
    """Create a model that uses streaming inference."""
    
    class StreamingDemoModel(nn.Module):
        def __init__(self, inference_engine):
            super().__init__()
            self.inference = inference_engine
            
        def forward(self, x):
            # Weights are loaded on-demand
            weight1 = self.inference.get_weight('fc1.weight')
            bias1 = self.inference.get_weight('fc1.bias')
            x = F.linear(x, weight1, bias1)
            x = F.relu(x)
            
            weight2 = self.inference.get_weight('fc2.weight')
            bias2 = self.inference.get_weight('fc2.bias')
            x = F.linear(x, weight2, bias2)
            x = F.relu(x)
            
            weight3 = self.inference.get_weight('fc3.weight')
            bias3 = self.inference.get_weight('fc3.bias')
            x = F.linear(x, weight3, bias3)
            x = F.relu(x)
            
            weight4 = self.inference.get_weight('fc4.weight')
            bias4 = self.inference.get_weight('fc4.bias')
            x = F.linear(x, weight4, bias4)
            x = F.relu(x)
            
            weight5 = self.inference.get_weight('fc5.weight')
            bias5 = self.inference.get_weight('fc5.bias')
            x = F.linear(x, weight5, bias5)
            
            return x
    
    # Create streaming inference engine
    inference = StreamingInference(
        compressed_fields,
        cache_size_mb=cache_size_mb,
        enable_prefetch=True
    )
    
    return StreamingDemoModel(inference), inference


def main():
    parser = argparse.ArgumentParser(description='Streaming inference demo')
    parser.add_argument('--width', type=int, default=2048,
                       help='Model width')
    parser.add_argument('--cache-size', type=int, default=10,
                       help='Cache size in MB')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    args = parser.parse_args()
    
    print("=== Streaming Inference Demo ===")
    print(f"Model width: {args.width}")
    print(f"Cache size: {args.cache_size} MB")
    
    # Create original model
    print("\n1. Creating original model...")
    model = DemoModel(width=args.width).to(args.device)
    
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / 1024 / 1024
    print(f"Original parameters: {param_count:,}")
    print(f"Original size: {model_size_mb:.1f} MB")
    
    # Compress model
    print("\n2. Compressing model...")
    config = CompressionConfig(
        bandwidth=4,
        hidden_width=128,
        max_steps=500  # Quick compression for demo
    )
    
    compressed_model, results = compress_model(model, config, device=args.device)
    print(f"Compression ratio: {results.total_compression_ratio:.2f}x")
    
    # Extract compressed fields
    compressed_fields = {}
    for name, result in results.layer_results.items():
        compressed_fields[name] = result.field
    
    # Memory usage comparison
    print("\n3. Memory Usage Comparison")
    
    # Baseline memory
    baseline_mem = get_memory_usage()
    print(f"Baseline memory: {baseline_mem:.1f} MB")
    
    # Preload inference
    print("\n   a) Preload mode (all weights in memory):")
    preload_start_mem = get_memory_usage()
    preload_inference = PreloadInference(compressed_fields, device=args.device)
    preload_mem = get_memory_usage() - preload_start_mem
    print(f"      Additional memory: {preload_mem:.1f} MB")
    
    # Clear preload
    del preload_inference
    torch.cuda.empty_cache()
    
    # Streaming inference
    print(f"\n   b) Streaming mode ({args.cache_size} MB cache):")
    streaming_start_mem = get_memory_usage()
    streaming_model, streaming_inference = create_streaming_model(
        compressed_fields, 
        cache_size_mb=args.cache_size
    )
    streaming_mem = get_memory_usage() - streaming_start_mem
    print(f"      Additional memory: {streaming_mem:.1f} MB")
    print(f"      Memory savings: {(1 - streaming_mem/preload_mem)*100:.1f}%")
    
    # Inference performance
    print("\n4. Inference Performance")
    
    # Create test data
    test_input = torch.randn(args.batch_size, args.width).to(args.device)
    
    # Original model
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(test_input)
        
        # Time
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = model(test_input)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        original_time = (time.time() - start) / 100
    
    print(f"   Original model: {original_time*1000:.2f} ms/batch")
    
    # Streaming model - first run (cold cache)
    streaming_inference.weight_cache.clear()
    streaming_inference.coord_cache.clear()
    
    with torch.no_grad():
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = streaming_model(test_input)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        cold_time = time.time() - start
    
    print(f"   Streaming (cold cache): {cold_time*1000:.2f} ms/batch")
    
    # Streaming model - subsequent runs (warm cache)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = streaming_model(test_input)
        
        # Time
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = streaming_model(test_input)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        warm_time = (time.time() - start) / 100
    
    print(f"   Streaming (warm cache): {warm_time*1000:.2f} ms/batch")
    
    # Cache statistics
    print("\n5. Cache Performance")
    cache_stats = streaming_inference.get_cache_stats()
    print(f"   Weight cache:")
    print(f"     - Hits: {cache_stats['weight_cache']['hits']}")
    print(f"     - Misses: {cache_stats['weight_cache']['misses']}")
    print(f"     - Hit rate: {cache_stats['weight_cache']['hit_rate']:.2%}")
    print(f"     - Size: {cache_stats['weight_cache']['size']}")
    print(f"   Coordinate cache:")
    print(f"     - Hit rate: {cache_stats['coord_cache']['hit_rate']:.2%}")
    
    # Simulate memory-constrained scenario
    print("\n6. Memory-Constrained Scenario")
    print(f"   Original model requires: {model_size_mb:.1f} MB")
    print(f"   Compressed fields require: {results.compressed_parameters * 4 / 1024 / 1024:.1f} MB")
    print(f"   Streaming with {args.cache_size} MB cache uses: {streaming_mem:.1f} MB total")
    
    # Test different access patterns
    print("\n7. Access Pattern Analysis")
    
    # Sequential access
    streaming_inference.weight_cache.clear()
    layer_names = ['fc1.weight', 'fc2.weight', 'fc3.weight', 'fc4.weight', 'fc5.weight']
    
    print("   Sequential access:")
    for i in range(2):
        for name in layer_names:
            _ = streaming_inference.get_weight(name)
    
    seq_stats = streaming_inference.get_cache_stats()
    print(f"     - Hit rate: {seq_stats['weight_cache']['hit_rate']:.2%}")
    
    # Random access
    streaming_inference.weight_cache.clear()
    import random
    
    print("   Random access:")
    for _ in range(10):
        name = random.choice(layer_names)
        _ = streaming_inference.get_weight(name)
    
    rand_stats = streaming_inference.get_cache_stats()
    print(f"     - Hit rate: {rand_stats['weight_cache']['hit_rate']:.2%}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Model compression: {results.total_compression_ratio:.2f}x")
    print(f"Memory reduction with streaming: {(1 - streaming_mem/model_size_mb)*100:.1f}%")
    print(f"Inference overhead (warm cache): {warm_time/original_time:.2f}x")
    print(f"Cache efficiency: {cache_stats['weight_cache']['hit_rate']:.1%} hit rate")


if __name__ == '__main__':
    main()