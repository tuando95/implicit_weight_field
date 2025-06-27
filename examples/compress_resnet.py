"""Example: Compress ResNet-50 with Implicit Weight Fields."""

import argparse
import torch
import torchvision.models as models
from pathlib import Path
import time

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from compression import ModelCompressor
from core.implicit_field import CompressionConfig
from evaluation import evaluate_model_accuracy, evaluate_compression
from experiments.models import prepare_imagenet_loader
from visualization import create_compression_report


def parse_args():
    parser = argparse.ArgumentParser(description='Compress ResNet-50 example')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='./results/resnet50',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--eval-samples', type=int, default=50000,
                       help='Number of validation samples to use')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== ResNet-50 Compression Example ===")
    
    # Load pre-trained ResNet-50
    print("\n1. Loading pre-trained ResNet-50...")
    model = models.resnet50(pretrained=True)
    model.eval()
    model = model.to(args.device)
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")
    print(f"Original size: {original_params * 4 / 1024 / 1024:.1f} MB")
    
    # Prepare validation data
    print("\n2. Preparing validation data...")
    val_loader = prepare_imagenet_loader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        split='val'
    )
    
    # Evaluate original model
    print("\n3. Evaluating original model...")
    original_metrics = evaluate_model_accuracy(
        model, val_loader, 
        device=args.device,
        num_samples=args.eval_samples
    )
    print(f"Original Top-1 Accuracy: {original_metrics.top1_accuracy:.2f}%")
    print(f"Original Top-5 Accuracy: {original_metrics.top5_accuracy:.2f}%")
    
    # Configure compression
    print("\n4. Configuring compression...")
    config = CompressionConfig(
        bandwidth=4,
        hidden_width=256,
        num_layers=2,
        w0=30.0,
        learning_rate=1e-3,
        max_steps=2000,
        regularization=1e-6
    )
    
    print("Compression configuration:")
    print(f"  - Bandwidth: {config.bandwidth}")
    print(f"  - Hidden width: {config.hidden_width}")
    print(f"  - Num layers: {config.num_layers}")
    print(f"  - Max steps: {config.max_steps}")
    
    # Create compressor
    compressor = ModelCompressor(
        model,
        config=config,
        min_tensor_size=10000,  # Only compress tensors > 10k params
        device=args.device
    )
    
    # Compress model
    print("\n5. Compressing model (this may take a while)...")
    start_time = time.time()
    
    compression_result = compressor.compress()
    
    compression_time = time.time() - start_time
    print(f"Compression completed in {compression_time:.1f} seconds")
    
    # Print compression statistics
    print("\n6. Compression Results:")
    print(f"Total compression ratio: {compression_result.total_compression_ratio:.2f}x")
    print(f"Parameter reduction: {compression_result.parameter_reduction:.1f}%")
    print(f"Compressed parameters: {compression_result.compressed_parameters:,}")
    print(f"Compressed size: {compression_result.compressed_parameters * 4 / 1024 / 1024:.1f} MB")
    
    # Layer-wise statistics
    print("\nPer-layer compression:")
    for layer_name, layer_result in compression_result.layer_results.items():
        print(f"  {layer_name}: {layer_result.compression_ratio:.1f}x, "
              f"MSE: {layer_result.reconstruction_error:.6f}")
    
    # Get compressed model
    print("\n7. Creating compressed model...")
    compressed_model = compressor.get_compressed_model()
    compressed_model.eval()
    
    # Evaluate compressed model
    print("\n8. Evaluating compressed model...")
    compressed_metrics = evaluate_model_accuracy(
        compressed_model, val_loader,
        device=args.device,
        num_samples=args.eval_samples
    )
    print(f"Compressed Top-1 Accuracy: {compressed_metrics.top1_accuracy:.2f}%")
    print(f"Compressed Top-5 Accuracy: {compressed_metrics.top5_accuracy:.2f}%")
    
    # Calculate accuracy drop
    acc_drop_top1 = original_metrics.top1_accuracy - compressed_metrics.top1_accuracy
    acc_drop_top5 = original_metrics.top5_accuracy - compressed_metrics.top5_accuracy
    print(f"\nAccuracy drop:")
    print(f"  Top-1: {acc_drop_top1:.2f}%")
    print(f"  Top-5: {acc_drop_top5:.2f}%")
    
    # Save compressed model
    print("\n9. Saving compressed model...")
    save_path = output_dir / 'resnet50_compressed.pth'
    compressor.save_compressed_model(str(save_path))
    print(f"Compressed model saved to: {save_path}")
    
    # Create visualization report
    print("\n10. Creating visualization report...")
    create_compression_report(
        model,
        compressed_model,
        compression_result.layer_results,
        output_dir / 'visualizations'
    )
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Original size: {original_params * 4 / 1024 / 1024:.1f} MB")
    print(f"Compressed size: {compression_result.compressed_parameters * 4 / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {compression_result.total_compression_ratio:.2f}x")
    print(f"Top-1 accuracy: {original_metrics.top1_accuracy:.2f}% → {compressed_metrics.top1_accuracy:.2f}% "
          f"(drop: {acc_drop_top1:.2f}%)")
    print(f"Compression time: {compression_time:.1f} seconds")
    
    # Test inference speed
    print("\n=== Inference Speed Test ===")
    test_input = torch.randn(1, 3, 224, 224).to(args.device)
    
    # Original model
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(test_input)
        
        # Time
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = model(test_input)
        torch.cuda.synchronize()
        original_time = (time.time() - start) / 100
    
    # Compressed model
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = compressed_model(test_input)
        
        # Time
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = compressed_model(test_input)
        torch.cuda.synchronize()
        compressed_time = (time.time() - start) / 100
    
    print(f"Original inference time: {original_time*1000:.2f} ms")
    print(f"Compressed inference time: {compressed_time*1000:.2f} ms")
    print(f"Slowdown: {compressed_time/original_time:.2f}x")


if __name__ == '__main__':
    main()