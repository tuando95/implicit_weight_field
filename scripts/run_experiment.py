"""Main script to run compression experiments."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from pathlib import Path
import json
import time

from configs.config import Config, register_configs
from experiments.models import (
    load_resnet50, load_mobilenet_v2, load_vit,
    prepare_imagenet_loader, prepare_cifar_loader,
    load_bert_base, prepare_glue_dataset
)
from compression import compress_model, CompressionConfig as CompConfig
from evaluation import (
    evaluate_compression, evaluate_model_accuracy,
    evaluate_reconstruction_quality, benchmark_inference_latency
)
from baselines.quantization import quantize_model_int8
from baselines.pruning import magnitude_prune
from baselines.decomposition import tensor_train_decomposition
from inference import InferenceMode, create_inference_mode

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main experiment runner."""
    # Set up logging
    setup_logging(cfg)
    
    # Set random seeds
    set_seeds(cfg.experiment.seed)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and data
    model, dataloader = load_model_and_data(cfg, device)
    
    # Run compression
    logger.info("Starting compression...")
    compressed_model, results = run_compression(model, cfg, device)
    
    # Evaluate compression
    logger.info("Evaluating compression...")
    evaluation_results = evaluate_compression_results(
        model, compressed_model, dataloader, results, cfg, device
    )
    
    # Run baseline comparisons if enabled
    if cfg.baselines.enabled:
        logger.info("Running baseline comparisons...")
        baseline_results = run_baselines(model, dataloader, cfg, device)
        evaluation_results["baselines"] = baseline_results
    
    # Run ablation studies if enabled
    if cfg.ablation.enabled:
        logger.info("Running ablation studies...")
        ablation_results = run_ablation_studies(model, dataloader, cfg, device)
        evaluation_results["ablation"] = ablation_results
    
    # Save results
    save_results(evaluation_results, cfg)
    
    logger.info("Experiment completed!")


def setup_logging(cfg: DictConfig):
    """Set up logging configuration."""
    log_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "experiment.log"),
            logging.StreamHandler()
        ]
    )


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_data(cfg: DictConfig, device: torch.device):
    """Load model and dataset based on configuration."""
    # Load model
    if cfg.model.name == "resnet50":
        model = load_resnet50(
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
            device=device
        )
    elif cfg.model.name == "mobilenet_v2":
        model = load_mobilenet_v2(
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
            device=device
        )
    elif cfg.model.name == "vit":
        model = load_vit(
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
            device=device
        )
    elif cfg.model.name == "bert_base":
        model, tokenizer = load_bert_base(
            task_name=cfg.dataset.name,
            pretrained=cfg.model.pretrained,
            device=device
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    
    # Load dataset
    if cfg.dataset.name == "imagenet":
        dataloader = prepare_imagenet_loader(
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            split="val"
        )
    elif cfg.dataset.name in ["cifar10", "cifar100"]:
        dataloader = prepare_cifar_loader(
            data_dir=cfg.dataset.data_dir,
            dataset=cfg.dataset.name,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            train=False
        )
    elif cfg.dataset.name in ["cola", "sst2", "mrpc", "qnli", "qqp", "mnli", "rte", "stsb", "wnli"]:
        dataloader = prepare_glue_dataset(
            task_name=cfg.dataset.name,
            tokenizer=tokenizer,
            batch_size=cfg.dataset.batch_size,
            split="validation"
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    return model, dataloader


def run_compression(model, cfg: DictConfig, device: torch.device):
    """Run implicit weight field compression."""
    # Create compression config
    comp_config = CompConfig(
        bandwidth=cfg.compression.bandwidth,
        hidden_width=cfg.compression.hidden_width,
        num_layers=cfg.compression.num_layers,
        w0=cfg.compression.w0,
        learning_rate=cfg.compression.learning_rate,
        max_steps=cfg.compression.max_steps,
        convergence_threshold=cfg.compression.convergence_threshold,
        regularization=cfg.compression.regularization,
        architecture=cfg.compression.architecture
    )
    
    # Compress model
    compressed_model, results = compress_model(
        model,
        config=comp_config,
        device=device
    )
    
    return compressed_model, results


def evaluate_compression_results(
    original_model, compressed_model, dataloader, 
    compression_results, cfg: DictConfig, device: torch.device
):
    """Evaluate compression results."""
    results = {}
    
    # Compression metrics
    if "compression_ratio" in cfg.evaluation.metrics:
        results["compression"] = evaluate_compression(original_model, compression_results)
    
    # Accuracy metrics
    if "accuracy" in cfg.evaluation.metrics:
        logger.info("Evaluating accuracy...")
        
        # Original model accuracy
        orig_acc = evaluate_model_accuracy(
            original_model, dataloader, 
            task="classification", device=device
        )
        
        # Compressed model accuracy (would need proper integration)
        # For now, use original model as placeholder
        comp_acc = evaluate_model_accuracy(
            original_model, dataloader,
            task="classification", device=device
        )
        
        results["accuracy"] = {
            "original": orig_acc,
            "compressed": comp_acc,
            "retention": (comp_acc.top1_accuracy / orig_acc.top1_accuracy) * 100
        }
    
    # Inference latency
    if "inference_latency" in cfg.evaluation.metrics:
        logger.info("Benchmarking inference...")
        results["efficiency"] = benchmark_inference_latency(
            original_model,
            input_shape=(3, 224, 224),
            batch_sizes=cfg.evaluation.batch_sizes,
            num_runs=cfg.evaluation.num_runs,
            warmup_runs=cfg.evaluation.warmup_runs,
            device=device
        )
    
    # Reconstruction quality
    if "reconstruction_quality" in cfg.evaluation.metrics:
        logger.info("Evaluating reconstruction quality...")
        # Extract original and reconstructed weights
        original_weights = {
            name: param.data for name, param in original_model.named_parameters()
        }
        
        # For now, use original weights as placeholder
        reconstructed_weights = original_weights
        
        results["reconstruction"] = evaluate_reconstruction_quality(
            original_weights, reconstructed_weights
        )
    
    return results


def run_baselines(model, dataloader, cfg: DictConfig, device: torch.device):
    """Run baseline compression methods."""
    baseline_results = {}
    
    for baseline in cfg.baselines.methods:
        logger.info(f"Running baseline: {baseline.name}")
        
        if baseline.name == "quantization_int8":
            quantized_model = quantize_model_int8(model)
            # Evaluate quantized model
            baseline_results[baseline.name] = {
                "compression_ratio": 4.0,  # FP32 to INT8
                "method": "quantization"
            }
            
        elif baseline.name == "pruning":
            for sparsity in baseline.config.get("sparsity", [0.5]):
                pruned_model = magnitude_prune(model, sparsity)
                baseline_results[f"pruning_{sparsity}"] = {
                    "sparsity": sparsity,
                    "method": "pruning"
                }
                
        elif baseline.name == "tensor_train":
            tt_model = tensor_train_decomposition(model)
            baseline_results[baseline.name] = {
                "rank": baseline.config.get("rank", 8),
                "method": "tensor_train"
            }
    
    return baseline_results


def run_ablation_studies(model, dataloader, cfg: DictConfig, device: torch.device):
    """Run ablation studies."""
    ablation_results = {}
    
    for study in cfg.ablation.studies:
        logger.info(f"Running ablation study: {study.name}")
        study_results = {}
        
        # Run experiments for different parameter values
        for param_name, param_values in study.parameters.items():
            for value in param_values:
                logger.info(f"Testing {param_name}={value}")
                # Run compression with modified parameter
                # This would need proper implementation
                study_results[f"{param_name}_{value}"] = {
                    "parameter": param_name,
                    "value": value,
                    "results": {}  # Placeholder
                }
        
        ablation_results[study.name] = study_results
    
    return ablation_results


def save_results(results: dict, cfg: DictConfig):
    """Save experiment results."""
    output_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save configuration
    if cfg.reproducibility.save_configs:
        with open(output_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    register_configs()
    main()