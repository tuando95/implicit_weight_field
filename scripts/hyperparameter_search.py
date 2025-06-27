"""Hyperparameter search for optimal compression configurations."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.trial import Trial
import json
from datetime import datetime

from core.implicit_field import CompressionConfig
from compression import compress_model
from experiments.models import load_resnet50, load_mobilenet_v2, prepare_imagenet_loader
from evaluation import evaluate_model_accuracy, evaluate_compression
from utils import set_random_seeds


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for compression")
    
    # Model and data
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'mobilenet_v2'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       help='Dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset')
    
    # Search space
    parser.add_argument('--bandwidth-min', type=int, default=1,
                       help='Minimum bandwidth')
    parser.add_argument('--bandwidth-max', type=int, default=16,
                       help='Maximum bandwidth')
    parser.add_argument('--hidden-width-choices', nargs='+', type=int,
                       default=[64, 128, 256, 512],
                       help='Hidden width choices')
    parser.add_argument('--num-layers-min', type=int, default=1,
                       help='Minimum number of layers')
    parser.add_argument('--num-layers-max', type=int, default=4,
                       help='Maximum number of layers')
    parser.add_argument('--lr-min', type=float, default=1e-4,
                       help='Minimum learning rate')
    parser.add_argument('--lr-max', type=float, default=1e-2,
                       help='Maximum learning rate')
    
    # Search configuration
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of trials')
    parser.add_argument('--target-accuracy-drop', type=float, default=1.0,
                       help='Maximum acceptable accuracy drop (%)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout per trial (seconds)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./hyperparam_search',
                       help='Output directory')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Optuna study name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of validation samples for fast evaluation')
    
    return parser.parse_args()


class CompressionObjective:
    """Objective function for hyperparameter optimization."""
    
    def __init__(self, args, model, val_loader, baseline_accuracy):
        """Initialize objective function."""
        self.args = args
        self.model = model
        self.val_loader = val_loader
        self.baseline_accuracy = baseline_accuracy
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    def __call__(self, trial: Trial) -> float:
        """
        Objective function to maximize compression ratio while maintaining accuracy.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Objective value (negative compression ratio for minimization)
        """
        # Sample hyperparameters
        bandwidth = trial.suggest_int('bandwidth', 
                                     self.args.bandwidth_min, 
                                     self.args.bandwidth_max)
        hidden_width = trial.suggest_categorical('hidden_width',
                                                self.args.hidden_width_choices)
        num_layers = trial.suggest_int('num_layers',
                                      self.args.num_layers_min,
                                      self.args.num_layers_max)
        w0 = trial.suggest_float('w0', 10.0, 50.0)
        learning_rate = trial.suggest_float('learning_rate',
                                          self.args.lr_min,
                                          self.args.lr_max,
                                          log=True)
        regularization = trial.suggest_float('regularization',
                                           1e-8, 1e-4,
                                           log=True)
        
        # Create compression config
        config = CompressionConfig(
            bandwidth=bandwidth,
            hidden_width=hidden_width,
            num_layers=num_layers,
            w0=w0,
            learning_rate=learning_rate,
            max_steps=1000,  # Reduced for faster search
            regularization=regularization
        )
        
        try:
            # Compress model (sample layers for speed)
            sample_layers = self._sample_layers()
            compressed_model, results = compress_model(
                self.model,
                config=config,
                device=self.device,
                layer_filter=sample_layers
            )
            
            # Evaluate accuracy on subset
            accuracy_metrics = self._fast_evaluate(compressed_model)
            accuracy_drop = self.baseline_accuracy - accuracy_metrics.top1_accuracy
            
            # Check constraint
            if accuracy_drop > self.args.target_accuracy_drop:
                # Penalize if accuracy drop is too high
                return -1.0
            
            # Calculate compression metrics
            compression_metrics = evaluate_compression(self.model, results)
            
            # Report intermediate values
            trial.report(compression_metrics.compression_ratio, step=0)
            
            # Objective: maximize compression ratio (minimize negative)
            return -compression_metrics.compression_ratio
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0  # Failed trials get worst score
    
    def _sample_layers(self) -> List[str]:
        """Sample representative layers for faster evaluation."""
        all_layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.numel() > 10000:  # Only large layers
                    all_layers.append(name)
        
        # Sample up to 5 layers
        n_sample = min(5, len(all_layers))
        indices = np.linspace(0, len(all_layers)-1, n_sample, dtype=int)
        return [all_layers[i] for i in indices]
    
    def _fast_evaluate(self, model):
        """Fast evaluation on a subset of validation data."""
        from torch.utils.data import Subset
        
        # Create subset
        indices = list(range(min(self.args.n_samples, len(self.val_loader.dataset))))
        subset = Subset(self.val_loader.dataset, indices)
        subset_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=self.val_loader.batch_size,
            num_workers=2
        )
        
        return evaluate_model_accuracy(model, subset_loader, device=self.device)


def run_hyperparameter_search(args):
    """Run hyperparameter search."""
    # Setup
    set_random_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'search.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load model and data
    logger.info("Loading model and dataset...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'resnet50':
        model = load_resnet50(pretrained=True, device=device)
    elif args.model == 'mobilenet_v2':
        model = load_mobilenet_v2(pretrained=True, device=device)
    
    val_loader = prepare_imagenet_loader(
        args.data_dir,
        batch_size=128,
        num_workers=4,
        split='val'
    )
    
    # Get baseline accuracy
    logger.info("Evaluating baseline accuracy...")
    baseline_metrics = evaluate_model_accuracy(model, val_loader, device=device)
    baseline_accuracy = baseline_metrics.top1_accuracy
    logger.info(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Create study
    study_name = args.study_name or f"{args.model}_compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=f'sqlite:///{output_dir}/optuna.db',
        load_if_exists=True
    )
    
    # Create objective
    objective = CompressionObjective(args, model, val_loader, baseline_accuracy)
    
    # Run optimization
    logger.info(f"Starting hyperparameter search with {args.n_trials} trials...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout * args.n_trials,
        n_jobs=1
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = -study.best_value  # Convert back to compression ratio
    
    logger.info(f"Best compression ratio: {best_value:.2f}x")
    logger.info(f"Best parameters: {best_params}")
    
    # Save results
    results = {
        'study_name': study_name,
        'model': args.model,
        'baseline_accuracy': baseline_accuracy,
        'target_accuracy_drop': args.target_accuracy_drop,
        'n_trials': len(study.trials),
        'best_compression_ratio': best_value,
        'best_params': best_params,
        'all_trials': []
    }
    
    # Add all trial results
    for trial in study.trials:
        if trial.value is not None:
            results['all_trials'].append({
                'number': trial.number,
                'params': trial.params,
                'compression_ratio': -trial.value,
                'state': str(trial.state)
            })
    
    # Save JSON
    with open(output_dir / 'search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create parameter importance plot
    try:
        import optuna.visualization as vis
        import plotly
        
        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importance.html'))
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / 'optimization_history.html'))
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
        
    except Exception as e:
        logger.warning(f"Failed to create visualization plots: {e}")
    
    return best_params, best_value


def create_final_config(best_params: Dict) -> CompressionConfig:
    """Create final compression config from best parameters."""
    return CompressionConfig(
        bandwidth=best_params['bandwidth'],
        hidden_width=best_params['hidden_width'],
        num_layers=best_params['num_layers'],
        w0=best_params['w0'],
        learning_rate=best_params['learning_rate'],
        max_steps=2000,  # Full training for final model
        regularization=best_params['regularization']
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run search
    best_params, best_compression = run_hyperparameter_search(args)
    
    # Create final configuration
    final_config = create_final_config(best_params)
    
    # Save final config
    output_dir = Path(args.output_dir)
    with open(output_dir / 'best_config.yaml', 'w') as f:
        yaml.dump({
            'compression': {
                'bandwidth': final_config.bandwidth,
                'hidden_width': final_config.hidden_width,
                'num_layers': final_config.num_layers,
                'w0': final_config.w0,
                'learning_rate': final_config.learning_rate,
                'max_steps': final_config.max_steps,
                'regularization': final_config.regularization
            },
            'expected_compression_ratio': best_compression,
            'target_accuracy_drop': args.target_accuracy_drop
        }, f)
    
    print("\n" + "="*50)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*50)
    print(f"Best compression ratio: {best_compression:.2f}x")
    print(f"Best configuration saved to: {output_dir / 'best_config.yaml'}")
    print("="*50)


if __name__ == "__main__":
    main()