"""
Main endpoint to run comprehensive experiments for Implicit Neural Weight Fields.
Implements the full experimental pipeline described in CLAUDE.md.
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from tqdm import tqdm

# Fix import issues by finding the project root
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# Go up directories until we find the project root (contains key directories)
project_root = current_dir
while project_root != os.path.dirname(project_root):  # Not at filesystem root
    if all(os.path.exists(os.path.join(project_root, d)) for d in ['experiments', 'compression', 'evaluation']):
        break
    project_root = os.path.dirname(project_root)
else:
    # If we didn't find it by going up, assume current directory is project root
    project_root = current_dir

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import Config, register_configs
from scripts.run_experiment import main as run_single_experiment
from utils import set_random_seed, setup_logging, create_results_dir

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Orchestrates multiple experiments following CLAUDE.md specifications."""
    
    def __init__(self, base_config_path: str = "./configs", output_dir: str = "./results"):
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.output_dir / f"full_experiments_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize experiments queue
        self.experiments_queue = []
        
    def setup_logging(self):
        """Configure logging for experiment orchestration."""
        log_file = self.results_dir / "orchestrator.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def create_experiment_configs(self) -> List[Dict[str, Any]]:
        """Create all experiment configurations based on CLAUDE.md."""
        experiments = []
        
        # 1. Main compression experiments (Section 4)
        experiments.extend(self._create_main_experiments())
        
        # 2. Ablation studies (Section 5.1)
        experiments.extend(self._create_ablation_experiments())
        
        # 3. Scaling analysis (Section 5.2)
        experiments.extend(self._create_scaling_experiments())
        
        # 4. Robustness evaluation (Section 5.4)
        experiments.extend(self._create_robustness_experiments())
        
        # 5. Efficiency analysis (Section 5.5)
        experiments.extend(self._create_efficiency_experiments())
        
        return experiments
    
    def _create_main_experiments(self) -> List[Dict[str, Any]]:
        """Create main compression experiments for different models."""
        experiments = []
        
        # Computer Vision models
        cv_models = [
            ("resnet50", "imagenet", {"num_classes": 1000}),
            ("mobilenet_v2", "imagenet", {"num_classes": 1000}),
            ("vit", "cifar100", {"num_classes": 100, "patch_size": 16})
        ]
        
        # NLP models
        nlp_models = [
            ("bert_base", "glue", {"tasks": ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]})
        ]
        
        # Create experiments for each model
        for model_name, dataset, model_config in cv_models:
            exp = {
                "name": f"main_{model_name}_{dataset}",
                "type": "main_compression",
                "model": {
                    "name": model_name,
                    "dataset": dataset,
                    **model_config
                },
                "compression": {
                    "bandwidth": 4,
                    "hidden_width": 256,
                    "num_layers": 2,
                    "w0": 30.0,
                    "learning_rate": 1e-3,
                    "max_steps": 2000
                },
                "evaluation": {
                    "metrics": ["accuracy", "compression_ratio", "inference_latency", 
                               "memory_usage", "reconstruction_quality"]
                },
                "baselines": {
                    "enabled": True,
                    "methods": ["quantization_int8", "pruning_0.5", "tensor_train"]
                }
            }
            experiments.append(exp)
        
        # Add BERT experiments
        for model_name, dataset, model_config in nlp_models:
            for task in model_config.get("tasks", ["cola"]):
                exp = {
                    "name": f"main_{model_name}_{task}",
                    "type": "main_compression",
                    "model": {
                        "name": model_name,
                        "dataset": task,
                        "task": task
                    },
                    "compression": {
                        "bandwidth": 4,
                        "hidden_width": 512,  # Larger for BERT
                        "num_layers": 2,
                        "w0": 30.0,
                        "learning_rate": 1e-3,
                        "max_steps": 2000
                    },
                    "evaluation": {
                        "metrics": ["accuracy", "compression_ratio", "inference_latency", 
                                   "memory_usage", "reconstruction_quality"]
                    },
                    "baselines": {
                        "enabled": True,
                        "methods": ["quantization_int8", "pruning_0.5", "tensor_train"]
                    }
                }
                experiments.append(exp)
        
        return experiments
    
    def _create_ablation_experiments(self) -> List[Dict[str, Any]]:
        """Create ablation study experiments."""
        experiments = []
        base_model = "resnet50"  # Use ResNet-50 for ablations
        
        # 1. Architecture ablation
        arch_ablation = {
            "depth": [1, 2, 3, 4],
            "width": [64, 128, 256, 512, 1024],
            "activation": ["siren", "relu", "gaussian", "swish"]
        }
        
        for param, values in arch_ablation.items():
            for value in values:
                exp = {
                    "name": f"ablation_architecture_{param}_{value}",
                    "type": "ablation",
                    "model": {"name": base_model, "dataset": "imagenet"},
                    "ablation_param": param,
                    "ablation_value": value,
                    "compression": self._get_ablation_compression_config(param, value)
                }
                experiments.append(exp)
        
        # 2. Positional encoding ablation
        encoding_ablation = {
            "bandwidth": [1, 2, 4, 8, 16],
            "encoding_type": ["fourier", "learned", "hash", "none"]
        }
        
        for param, values in encoding_ablation.items():
            for value in values:
                exp = {
                    "name": f"ablation_encoding_{param}_{value}",
                    "type": "ablation",
                    "model": {"name": base_model, "dataset": "imagenet"},
                    "ablation_param": param,
                    "ablation_value": value,
                    "compression": self._get_ablation_compression_config(param, value)
                }
                experiments.append(exp)
        
        # 3. Training procedure ablation
        training_ablation = {
            "optimizer": ["adam", "sgd", "adamw", "rmsprop"],
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "training_steps": [500, 1000, 2000, 5000, 10000],
            "regularization": [0, 1e-7, 1e-6, 1e-5, 1e-4]
        }
        
        for param, values in training_ablation.items():
            for value in values:
                exp = {
                    "name": f"ablation_training_{param}_{value}",
                    "type": "ablation",
                    "model": {"name": base_model, "dataset": "imagenet"},
                    "ablation_param": param,
                    "ablation_value": value,
                    "compression": self._get_ablation_compression_config(param, value)
                }
                experiments.append(exp)
        
        return experiments
    
    def _create_scaling_experiments(self) -> List[Dict[str, Any]]:
        """Create scaling analysis experiments."""
        experiments = []
        
        # 1. Tensor size scaling
        tensor_sizes = {
            "conv_kernels": [(3, 3), (5, 5), (7, 7)],
            "conv_channels": [64, 128, 256, 512, 1024],
            "linear_dims": [64, 128, 256, 512, 1024, 2048, 4096],
            "attention_heads": [8, 12, 16],
            "embedding_vocab": [1000, 10000, 30000]
        }
        
        # Create synthetic models for controlled scaling experiments
        for param_type, sizes in tensor_sizes.items():
            for size in sizes:
                exp = {
                    "name": f"scaling_tensor_{param_type}_{size}",
                    "type": "scaling",
                    "scaling_param": param_type,
                    "scaling_value": size,
                    "model": {
                        "name": "synthetic",
                        "type": param_type,
                        "size": size
                    }
                }
                experiments.append(exp)
        
        # 2. Model size scaling
        model_scales = [
            ("mobilenet_v1", "small", "1-10M params"),
            ("resnet50", "medium", "25M params"),
            ("resnet101", "large", "44M params"),
            ("bert_base", "medium", "110M params"),
            ("bert_large", "large", "340M params", False)  # Optional if resources allow
        ]
        
        for model_name, scale, description, *optional in model_scales:
            if optional and not optional[0]:
                continue
                
            exp = {
                "name": f"scaling_model_{model_name}_{scale}",
                "type": "scaling",
                "model": {"name": model_name},
                "scale": scale,
                "description": description
            }
            experiments.append(exp)
        
        return experiments
    
    def _create_robustness_experiments(self) -> List[Dict[str, Any]]:
        """Create robustness evaluation experiments."""
        experiments = []
        
        # 1. Downstream task transfer
        transfer_tasks = [
            ("resnet50", "imagenet", "imagenet_sketch"),
            ("resnet50", "imagenet", "imagenet_c"),
            ("bert_base", "glue", "biobert"),
            ("bert_base", "glue", "finbert")
        ]
        
        for base_model, base_dataset, transfer_dataset in transfer_tasks:
            exp = {
                "name": f"robustness_transfer_{base_model}_{transfer_dataset}",
                "type": "robustness",
                "subtype": "transfer",
                "model": {"name": base_model, "dataset": base_dataset},
                "transfer_dataset": transfer_dataset,
                "evaluation": {
                    "fine_tuning_stability": True,
                    "few_shot_performance": [0.01, 0.05, 0.1],  # Fraction of data
                    "catastrophic_forgetting": True
                }
            }
            experiments.append(exp)
        
        # 2. Adversarial robustness
        adversarial_configs = [
            ("fgsm", [2/255, 4/255, 8/255]),
            ("pgd", [2/255, 4/255, 8/255])
        ]
        
        for attack_type, epsilons in adversarial_configs:
            for epsilon in epsilons:
                exp = {
                    "name": f"robustness_adversarial_{attack_type}_{epsilon}",
                    "type": "robustness",
                    "subtype": "adversarial",
                    "model": {"name": "resnet50", "dataset": "imagenet"},
                    "attack": {
                        "type": attack_type,
                        "epsilon": epsilon,
                        "steps": 20 if attack_type == "pgd" else 1
                    }
                }
                experiments.append(exp)
        
        return experiments
    
    def _create_efficiency_experiments(self) -> List[Dict[str, Any]]:
        """Create computational efficiency experiments."""
        experiments = []
        
        # 1. Memory-compute trade-offs
        cache_configs = [
            {"size_mb": 50, "mode": "streaming"},
            {"size_mb": 100, "mode": "streaming"},
            {"size_mb": 200, "mode": "streaming"},
            {"size_mb": 500, "mode": "streaming"},
            {"size_mb": None, "mode": "preload"}
        ]
        
        for cache_config in cache_configs:
            exp = {
                "name": f"efficiency_cache_{cache_config['mode']}_{cache_config.get('size_mb', 'full')}",
                "type": "efficiency",
                "subtype": "memory_compute",
                "model": {"name": "resnet50", "dataset": "imagenet"},
                "inference": {
                    "mode": cache_config["mode"],
                    "cache_size_mb": cache_config.get("size_mb")
                },
                "evaluation": {
                    "batch_sizes": [1, 8, 32, 128],
                    "measure_latency": True,
                    "measure_memory": True,
                    "measure_energy": True
                }
            }
            experiments.append(exp)
        
        # 2. Hardware-specific performance
        hardware_configs = [
            {"device": "cuda", "optimization": "standard"},
            {"device": "cuda", "optimization": "cuda_kernels"},
            {"device": "cpu", "optimization": "standard"},
            {"device": "cpu", "optimization": "simd"}
        ]
        
        for hw_config in hardware_configs:
            exp = {
                "name": f"efficiency_hardware_{hw_config['device']}_{hw_config['optimization']}",
                "type": "efficiency",
                "subtype": "hardware",
                "model": {"name": "mobilenet_v2", "dataset": "imagenet"},
                "hardware": hw_config,
                "evaluation": {
                    "measure_throughput": True,
                    "measure_energy": True
                }
            }
            experiments.append(exp)
        
        return experiments
    
    def _get_ablation_compression_config(self, param: str, value: Any) -> Dict[str, Any]:
        """Get compression config for ablation experiments."""
        # Base configuration
        config = {
            "bandwidth": 4,
            "hidden_width": 256,
            "num_layers": 2,
            "w0": 30.0,
            "learning_rate": 1e-3,
            "max_steps": 2000,
            "convergence_threshold": 1e-6,
            "regularization": 1e-6
        }
        
        # Modify based on ablation parameter
        if param == "depth":
            config["num_layers"] = value
        elif param == "width":
            config["hidden_width"] = value
        elif param == "activation":
            config["activation"] = value
        elif param == "bandwidth":
            config["bandwidth"] = value
        elif param == "encoding_type":
            config["encoding_type"] = value
        elif param == "optimizer":
            config["optimizer"] = value
        elif param == "learning_rate":
            config["learning_rate"] = value
        elif param == "training_steps":
            config["max_steps"] = value
        elif param == "regularization":
            config["regularization"] = value
            
        return config
    
    def run_experiment(self, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with the given configuration."""
        logger.info(f"Running experiment: {exp_config['name']}")
        
        try:
            # Create experiment-specific directory
            exp_dir = self.results_dir / exp_config['name']
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save experiment configuration
            with open(exp_dir / "config.json", "w") as f:
                json.dump(exp_config, f, indent=2)
            
            # Convert to Hydra config format
            hydra_config = self._convert_to_hydra_config(exp_config)
            
            # Run experiment using the existing infrastructure
            with initialize_config_dir(config_dir=str(self.base_config_path)):
                cfg = compose(config_name="default", overrides=hydra_config)
                
                # Override output directory
                cfg.experiment.output_dir = str(exp_dir)
                cfg.experiment.name = exp_config['name']
                
                # Run the experiment
                start_time = time.time()
                run_single_experiment(cfg)
                duration = time.time() - start_time
                
                # Load and return results
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        results = json.load(f)
                    results["duration_seconds"] = duration
                    results["status"] = "completed"
                else:
                    results = {
                        "status": "failed",
                        "error": "No results file generated",
                        "duration_seconds": duration
                    }
                    
        except Exception as e:
            logger.error(f"Experiment {exp_config['name']} failed: {str(e)}")
            results = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
        return results
    
    def _convert_to_hydra_config(self, exp_config: Dict[str, Any]) -> List[str]:
        """Convert experiment config to Hydra override format."""
        overrides = []
        
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                if key in ["name", "type", "subtype", "description"]:
                    continue  # Skip metadata fields
                    
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    flatten_dict(value, full_key)
                elif isinstance(value, list):
                    overrides.append(f"{full_key}=[{','.join(map(str, value))}]")
                else:
                    overrides.append(f"{full_key}={value}")
        
        flatten_dict(exp_config)
        return overrides
    
    def run_parallel_experiments(self, experiments: List[Dict[str, Any]], 
                               max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Run multiple experiments in parallel."""
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, 4)
            
        logger.info(f"Running {len(experiments)} experiments with {max_workers} workers")
        
        results = {}
        failed_experiments = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(self.run_experiment, exp): exp 
                for exp in experiments
            }
            
            # Process completed experiments
            with tqdm(total=len(experiments), desc="Running experiments") as pbar:
                for future in as_completed(future_to_exp):
                    exp = future_to_exp[future]
                    try:
                        result = future.result()
                        results[exp['name']] = result
                        
                        if result.get("status") == "failed":
                            failed_experiments.append(exp['name'])
                            
                    except Exception as e:
                        logger.error(f"Exception in experiment {exp['name']}: {str(e)}")
                        results[exp['name']] = {
                            "status": "failed",
                            "error": str(e)
                        }
                        failed_experiments.append(exp['name'])
                        
                    pbar.update(1)
        
        # Summary statistics
        summary = {
            "total_experiments": len(experiments),
            "completed": len([r for r in results.values() if r.get("status") == "completed"]),
            "failed": len(failed_experiments),
            "failed_experiments": failed_experiments
        }
        
        return {"results": results, "summary": summary}
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results and generate summary statistics."""
        analysis = {}
        
        # Group results by experiment type
        by_type = {}
        for exp_name, result in results.items():
            exp_type = exp_name.split("_")[0]
            if exp_type not in by_type:
                by_type[exp_type] = []
            by_type[exp_type].append((exp_name, result))
        
        # Analyze each experiment type
        for exp_type, type_results in by_type.items():
            if exp_type == "main":
                analysis[exp_type] = self._analyze_main_results(type_results)
            elif exp_type == "ablation":
                analysis[exp_type] = self._analyze_ablation_results(type_results)
            elif exp_type == "scaling":
                analysis[exp_type] = self._analyze_scaling_results(type_results)
            elif exp_type == "robustness":
                analysis[exp_type] = self._analyze_robustness_results(type_results)
            elif exp_type == "efficiency":
                analysis[exp_type] = self._analyze_efficiency_results(type_results)
                
        return analysis
    
    def _analyze_main_results(self, results: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Analyze main compression experiment results."""
        analysis = {
            "compression_ratios": {},
            "accuracy_retention": {},
            "inference_overhead": {}
        }
        
        for exp_name, result in results:
            if result.get("status") != "completed":
                continue
                
            model_name = exp_name.split("_")[1]
            
            if "compression" in result:
                analysis["compression_ratios"][model_name] = result["compression"]["overall_ratio"]
                
            if "accuracy" in result:
                analysis["accuracy_retention"][model_name] = result["accuracy"]["retention"]
                
            if "efficiency" in result:
                analysis["inference_overhead"][model_name] = result["efficiency"]["overhead_percent"]
                
        return analysis
    
    def _analyze_ablation_results(self, results: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Analyze ablation study results."""
        # Group by ablation parameter
        by_param = {}
        for exp_name, result in results:
            if result.get("status") != "completed":
                continue
                
            parts = exp_name.split("_")
            param = parts[2]
            value = "_".join(parts[3:])
            
            if param not in by_param:
                by_param[param] = {}
            by_param[param][value] = result
            
        return by_param
    
    def _analyze_scaling_results(self, results: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Analyze scaling experiment results."""
        # Fit scaling laws
        tensor_sizes = []
        compression_ratios = []
        
        for exp_name, result in results:
            if result.get("status") != "completed":
                continue
                
            if "tensor_size" in result and "compression_ratio" in result:
                tensor_sizes.append(result["tensor_size"])
                compression_ratios.append(result["compression_ratio"])
                
        if tensor_sizes and compression_ratios:
            # Fit power law: compression_ratio = A * tensor_size^alpha
            log_sizes = np.log(tensor_sizes)
            log_ratios = np.log(compression_ratios)
            alpha, log_A = np.polyfit(log_sizes, log_ratios, 1)
            A = np.exp(log_A)
            
            return {
                "scaling_law": {
                    "A": A,
                    "alpha": alpha,
                    "equation": f"compression_ratio = {A:.2f} * tensor_size^{alpha:.2f}"
                },
                "data_points": list(zip(tensor_sizes, compression_ratios))
            }
        
        return {}
    
    def _analyze_robustness_results(self, results: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Analyze robustness experiment results."""
        transfer_results = {}
        adversarial_results = {}
        
        for exp_name, result in results:
            if result.get("status") != "completed":
                continue
                
            if "transfer" in exp_name:
                dataset = exp_name.split("_")[-1]
                transfer_results[dataset] = result.get("transfer_accuracy", {})
            elif "adversarial" in exp_name:
                attack_type = exp_name.split("_")[-2]
                epsilon = exp_name.split("_")[-1]
                if attack_type not in adversarial_results:
                    adversarial_results[attack_type] = {}
                adversarial_results[attack_type][epsilon] = result.get("robust_accuracy", {})
                
        return {
            "transfer_learning": transfer_results,
            "adversarial_robustness": adversarial_results
        }
    
    def _analyze_efficiency_results(self, results: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Analyze efficiency experiment results."""
        cache_performance = {}
        hardware_performance = {}
        
        for exp_name, result in results:
            if result.get("status") != "completed":
                continue
                
            if "cache" in exp_name:
                cache_size = exp_name.split("_")[-1]
                cache_performance[cache_size] = {
                    "hit_rate": result.get("cache_hit_rate", 0),
                    "latency": result.get("inference_latency", 0),
                    "memory": result.get("memory_usage", 0)
                }
            elif "hardware" in exp_name:
                device = exp_name.split("_")[-2]
                optimization = exp_name.split("_")[-1]
                key = f"{device}_{optimization}"
                hardware_performance[key] = {
                    "throughput": result.get("throughput", 0),
                    "energy": result.get("energy_consumption", 0)
                }
                
        return {
            "cache_analysis": cache_performance,
            "hardware_analysis": hardware_performance
        }
    
    def generate_report(self, all_results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Generate comprehensive experiment report."""
        report_path = self.results_dir / "experiment_report.md"
        
        with open(report_path, "w") as f:
            f.write("# Implicit Neural Weight Fields - Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            summary = all_results["summary"]
            f.write(f"- Total experiments: {summary['total_experiments']}\n")
            f.write(f"- Completed: {summary['completed']}\n")
            f.write(f"- Failed: {summary['failed']}\n\n")
            
            # Main results
            if "main" in analysis:
                f.write("## Main Compression Results\n\n")
                main_analysis = analysis["main"]
                
                f.write("### Compression Ratios\n\n")
                for model, ratio in main_analysis["compression_ratios"].items():
                    f.write(f"- {model}: {ratio:.2f}x\n")
                    
                f.write("\n### Accuracy Retention\n\n")
                for model, retention in main_analysis["accuracy_retention"].items():
                    f.write(f"- {model}: {retention:.1f}%\n")
                    
            # Ablation results
            if "ablation" in analysis:
                f.write("\n## Ablation Study Results\n\n")
                for param, param_results in analysis["ablation"].items():
                    f.write(f"### {param}\n\n")
                    # Add detailed ablation analysis
                    
            # Scaling results
            if "scaling" in analysis and "scaling_law" in analysis["scaling"]:
                f.write("\n## Scaling Analysis\n\n")
                scaling = analysis["scaling"]["scaling_law"]
                f.write(f"Scaling law: {scaling['equation']}\n\n")
                
            # Add more sections as needed
            
        logger.info(f"Report generated: {report_path}")
        
    def create_visualizations(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Create visualization plots for experiment results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            viz_dir = self.results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Compression vs Accuracy trade-off
            if "main" in analysis:
                self._plot_compression_accuracy_tradeoff(
                    analysis["main"], 
                    viz_dir / "compression_accuracy_tradeoff.png"
                )
                
            # Ablation heatmaps
            if "ablation" in analysis:
                self._plot_ablation_heatmaps(
                    analysis["ablation"],
                    viz_dir / "ablation_heatmaps.png"
                )
                
            # Scaling curves
            if "scaling" in analysis and "data_points" in analysis["scaling"]:
                self._plot_scaling_curves(
                    analysis["scaling"]["data_points"],
                    viz_dir / "scaling_curves.png"
                )
                
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")
            
    def _plot_compression_accuracy_tradeoff(self, main_results: Dict, output_path: Path):
        """Plot compression ratio vs accuracy retention."""
        import matplotlib.pyplot as plt
        
        models = list(main_results["compression_ratios"].keys())
        compression_ratios = [main_results["compression_ratios"][m] for m in models]
        accuracy_retentions = [main_results["accuracy_retention"][m] for m in models]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(compression_ratios, accuracy_retentions, s=100)
        
        for i, model in enumerate(models):
            plt.annotate(model, (compression_ratios[i], accuracy_retentions[i]),
                        xytext=(5, 5), textcoords='offset points')
                        
        plt.xlabel("Compression Ratio")
        plt.ylabel("Accuracy Retention (%)")
        plt.title("Compression-Accuracy Trade-off")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    def _plot_ablation_heatmaps(self, ablation_results: Dict, output_path: Path):
        """Plot ablation study results as heatmaps."""
        # Implementation for ablation heatmaps
        pass
        
    def _plot_scaling_curves(self, data_points: List[Tuple], output_path: Path):
        """Plot scaling analysis curves."""
        import matplotlib.pyplot as plt
        
        tensor_sizes, compression_ratios = zip(*data_points)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(tensor_sizes, compression_ratios, 'o-')
        plt.xlabel("Tensor Size")
        plt.ylabel("Compression Ratio")
        plt.title("Compression Scaling Analysis")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def main():
    """Main entry point for running all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run INWF experiments")
    parser.add_argument("--config-dir", type=str, default="./configs",
                       help="Path to configuration directory")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Path to output directory")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers")
    parser.add_argument("--experiment-types", nargs="+", 
                       default=["main", "ablation", "scaling", "robustness", "efficiency"],
                       help="Types of experiments to run")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate configs without running experiments")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ExperimentOrchestrator(
        base_config_path=args.config_dir,
        output_dir=args.output_dir
    )
    
    # Create experiment configurations
    all_experiments = orchestrator.create_experiment_configs()
    
    # Filter by experiment types
    filtered_experiments = [
        exp for exp in all_experiments 
        if exp.get("type", "").split("_")[0] in args.experiment_types
    ]
    
    logger.info(f"Created {len(filtered_experiments)} experiment configurations")
    
    if args.dry_run:
        # Save configurations without running
        config_path = orchestrator.results_dir / "all_experiments.json"
        with open(config_path, "w") as f:
            json.dump(filtered_experiments, f, indent=2)
        logger.info(f"Saved experiment configurations to {config_path}")
        return
    
    # Run experiments
    all_results = orchestrator.run_parallel_experiments(
        filtered_experiments,
        max_workers=args.max_workers
    )
    
    # Save raw results
    results_path = orchestrator.results_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Analyze results
    analysis = orchestrator.analyze_results(all_results["results"])
    
    # Save analysis
    analysis_path = orchestrator.results_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Generate report
    orchestrator.generate_report(all_results, analysis)
    
    # Create visualizations
    orchestrator.create_visualizations(all_results["results"], analysis)
    
    logger.info(f"All experiments completed. Results saved to {orchestrator.results_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total experiments: {all_results['summary']['total_experiments']}")
    print(f"Completed: {all_results['summary']['completed']}")
    print(f"Failed: {all_results['summary']['failed']}")
    print(f"Results directory: {orchestrator.results_dir}")
    print("="*50)


if __name__ == "__main__":
    register_configs()
    main()