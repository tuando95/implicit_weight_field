"""Training script for implicit weight field compression with hyperparameter management."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
import yaml
import torch
import wandb
from datetime import datetime
import json

from core.implicit_field import CompressionConfig
from compression import compress_model
from experiments.models import (
    load_resnet50, load_mobilenet_v2, load_vit,
    prepare_imagenet_loader, prepare_cifar_loader
)
from evaluation import (
    evaluate_compression, evaluate_model_accuracy,
    evaluate_reconstruction_quality
)
from utils import set_random_seeds, setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train implicit weight field compression")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'mobilenet_v2', 'vit'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='imagenet',
                       choices=['imagenet', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Compression arguments
    parser.add_argument('--bandwidth', type=int, default=4,
                       help='Fourier feature bandwidth')
    parser.add_argument('--hidden-width', type=int, default=256,
                       help='Hidden layer width')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of SIREN layers')
    parser.add_argument('--w0', type=float, default=30.0,
                       help='SIREN frequency parameter')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for field training')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum training steps per field')
    parser.add_argument('--regularization', type=float, default=1e-6,
                       help='L2 regularization weight')
    
    # Training arguments
    parser.add_argument('--layers-to-compress', nargs='+', default=None,
                       help='Specific layers to compress')
    parser.add_argument('--min-tensor-size', type=int, default=10000,
                       help='Minimum tensor size to compress')
    parser.add_argument('--target-compression', type=float, default=None,
                       help='Target compression ratio')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--save-model', action='store_true',
                       help='Save compressed model')
    parser.add_argument('--save-fields', action='store_true',
                       help='Save individual fields')
    
    # Experiment tracking
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Weights & Biases entity')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    return parser.parse_args()


def load_model_and_data(args):
    """Load model and dataset based on arguments."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if args.model == 'resnet50':
        model = load_resnet50(pretrained=args.pretrained, device=device)
    elif args.model == 'mobilenet_v2':
        model = load_mobilenet_v2(pretrained=args.pretrained, device=device)
    elif args.model == 'vit':
        model = load_vit(pretrained=args.pretrained, device=device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Load dataset
    if args.dataset == 'imagenet':
        val_loader = prepare_imagenet_loader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split='val'
        )
    elif args.dataset in ['cifar10', 'cifar100']:
        val_loader = prepare_cifar_loader(
            args.data_dir,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train=False
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return model, val_loader, device


def train_compression(args):
    """Main compression training function."""
    # Setup
    set_random_seeds(args.seed)
    setup_logging(args.output_dir, verbose=args.verbose)
    
    # Initialize wandb if requested
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name or f"{args.model}_compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Load model and data
    logger.info("Loading model and dataset...")
    model, val_loader, device = load_model_and_data(args)
    
    # Evaluate baseline
    logger.info("Evaluating baseline model...")
    baseline_metrics = evaluate_model_accuracy(model, val_loader, device=device)
    logger.info(f"Baseline accuracy: {baseline_metrics.top1_accuracy:.2f}%")
    
    # Create compression config
    compression_config = CompressionConfig(
        bandwidth=args.bandwidth,
        hidden_width=args.hidden_width,
        num_layers=args.num_layers,
        w0=args.w0,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        regularization=args.regularization
    )
    
    # Compress model
    logger.info("Starting compression...")
    compressed_model, compression_results = compress_model(
        model,
        config=compression_config,
        device=device,
        layer_filter=args.layers_to_compress
    )
    
    # Evaluate compressed model
    logger.info("Evaluating compressed model...")
    compressed_metrics = evaluate_model_accuracy(compressed_model, val_loader, device=device)
    
    # Calculate metrics
    compression_metrics = evaluate_compression(model, compression_results)
    
    # Extract weights for reconstruction quality
    original_weights = {name: p.data for name, p in model.named_parameters()}
    compressed_weights = {name: p.data for name, p in compressed_model.named_parameters()}
    reconstruction_metrics = evaluate_reconstruction_quality(original_weights, compressed_weights)
    
    # Prepare results
    results = {
        'args': vars(args),
        'baseline_accuracy': baseline_metrics.top1_accuracy,
        'compressed_accuracy': compressed_metrics.top1_accuracy,
        'accuracy_drop': baseline_metrics.top1_accuracy - compressed_metrics.top1_accuracy,
        'compression_ratio': compression_metrics.compression_ratio,
        'parameter_reduction': compression_metrics.parameter_reduction,
        'model_size_mb': compression_metrics.model_size_mb,
        'compressed_size_mb': compression_metrics.compressed_size_mb,
        'layer_results': {},
        'reconstruction_quality': {}
    }
    
    # Add layer-wise results
    for layer_name, layer_result in compression_results.layer_results.items():
        results['layer_results'][layer_name] = {
            'compression_ratio': layer_result.compression_ratio,
            'reconstruction_error': layer_result.reconstruction_error,
            'field_architecture': layer_result.field_architecture,
            'training_steps': layer_result.training_steps
        }
    
    # Add reconstruction metrics
    for layer_name, metrics in reconstruction_metrics.items():
        results['reconstruction_quality'][layer_name] = {
            'mse': metrics.mse,
            'rmse': metrics.rmse,
            'snr_db': metrics.snr_db
        }
    
    # Log to wandb
    if args.wandb_project:
        wandb.log({
            'baseline_accuracy': baseline_metrics.top1_accuracy,
            'compressed_accuracy': compressed_metrics.top1_accuracy,
            'accuracy_drop': results['accuracy_drop'],
            'compression_ratio': compression_metrics.compression_ratio,
            'parameter_reduction': compression_metrics.parameter_reduction
        })
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results JSON
    with open(output_dir / 'compression_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save compressed model if requested
    if args.save_model:
        model_path = output_dir / 'compressed_model.pth'
        torch.save({
            'state_dict': compressed_model.state_dict(),
            'compression_config': compression_config,
            'compression_results': compression_results,
            'baseline_accuracy': baseline_metrics.top1_accuracy,
            'compressed_accuracy': compressed_metrics.top1_accuracy
        }, model_path)
        logger.info(f"Saved compressed model to {model_path}")
    
    # Save individual fields if requested
    if args.save_fields:
        fields_dir = output_dir / 'fields'
        fields_dir.mkdir(exist_ok=True)
        
        for layer_name, layer_result in compression_results.layer_results.items():
            field_path = fields_dir / f"{layer_name.replace('.', '_')}_field.pth"
            torch.save({
                'field_state': layer_result.field.state_dict(),
                'field_config': layer_result.field.config,
                'tensor_shape': layer_result.original_shape,
                'compression_ratio': layer_result.compression_ratio
            }, field_path)
        
        logger.info(f"Saved individual fields to {fields_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("COMPRESSION SUMMARY")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Baseline Accuracy: {baseline_metrics.top1_accuracy:.2f}%")
    print(f"Compressed Accuracy: {compressed_metrics.top1_accuracy:.2f}%")
    print(f"Accuracy Drop: {results['accuracy_drop']:.2f}%")
    print(f"Compression Ratio: {compression_metrics.compression_ratio:.2f}x")
    print(f"Parameter Reduction: {compression_metrics.parameter_reduction:.1f}%")
    print(f"Model Size: {compression_metrics.model_size_mb:.1f} MB -> {compression_metrics.compressed_size_mb:.1f} MB")
    print("="*50)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.dataset}_b{args.bandwidth}_h{args.hidden_width}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Update output directory with experiment name
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    # Run compression
    results = train_compression(args)
    
    # Cleanup
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()