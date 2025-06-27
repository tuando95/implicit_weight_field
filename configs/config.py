"""Configuration management using Hydra."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str = "default_experiment"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./results"
    save_checkpoints: bool = True
    

@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = False
    project: str = "implicit-weight-fields"
    entity: Optional[str] = None


@dataclass 
class CompressionConfig:
    """Compression configuration."""
    bandwidth: int = 4
    hidden_width: int = 256
    num_layers: int = 2
    w0: float = 30.0
    learning_rate: float = 1e-3
    max_steps: int = 2000
    convergence_threshold: float = 1e-6
    regularization: float = 1e-6
    architecture: Optional[str] = None


@dataclass
class CacheConfig:
    """Cache configuration for streaming inference."""
    max_size_mb: float = 100.0
    prefetch_neighbors: bool = True
    prefetch_radius: int = 1
    batch_size: int = 128


@dataclass
class InferenceConfig:
    """Inference configuration."""
    mode: str = "preload"  # "preload" or "streaming"
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 1000
    dataset: str = "imagenet"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "imagenet"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    augment: bool = False
    validation_split: float = 0.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "compression_ratio", "inference_latency", 
        "memory_usage", "reconstruction_quality"
    ])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    num_runs: int = 100
    warmup_runs: int = 10


@dataclass
class BaselineConfig:
    """Baseline comparison configuration."""
    name: str = MISSING
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselinesConfig:
    """All baseline configurations."""
    enabled: bool = True
    methods: List[BaselineConfig] = field(default_factory=list)


@dataclass
class AblationStudy:
    """Single ablation study configuration."""
    name: str = MISSING
    parameters: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class AblationConfig:
    """Ablation studies configuration."""
    enabled: bool = False
    studies: List[AblationStudy] = field(default_factory=list)


@dataclass
class AdversarialConfig:
    """Adversarial robustness configuration."""
    enabled: bool = True
    attacks: List[str] = field(default_factory=lambda: ["fgsm", "pgd"])
    epsilon: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.1])


@dataclass
class RobustnessConfig:
    """Robustness evaluation configuration."""
    enabled: bool = False
    downstream_tasks: List[str] = field(default_factory=lambda: [
        "fine_tuning", "transfer_learning", "few_shot"
    ])
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration."""
    deterministic: bool = True
    save_logs: bool = True
    save_configs: bool = True
    docker_image: str = "implicit-weight-field:latest"


@dataclass
class Config:
    """Main configuration combining all components."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)


def register_configs():
    """Register configurations with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    cs.store(group="compression", name="default", node=CompressionConfig)
    cs.store(group="model", name="resnet50", node=ModelConfig)
    cs.store(group="dataset", name="imagenet", node=DatasetConfig)