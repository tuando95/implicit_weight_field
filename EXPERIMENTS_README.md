# Running INWF Experiments

This document explains how to run experiments for Implicit Neural Weight Fields (INWF) as described in CLAUDE.md.

## Quick Start

### Run Key Demonstrations
To quickly see INWF capabilities, run the key experiments:

```bash
python run_key_experiments.py
```

This will:
- Compress popular models (ResNet-50, MobileNet-V2, BERT)
- Demonstrate >20x compression with <1% accuracy loss
- Show ablation study results
- Generate a summary report with key findings

### Run Full Experiments
To run the complete experimental suite from CLAUDE.md:

```bash
python run_all_experiments.py
```

Options:
- `--experiment-types main ablation scaling`: Select specific experiment types
- `--max-workers 4`: Control parallel execution
- `--dry-run`: Generate configs without running

## Experiment Types

### 1. Main Compression Experiments
Compresses standard models and compares against baselines:
- **Vision Models**: ResNet-50, MobileNet-V2, ViT on ImageNet/CIFAR
- **NLP Models**: BERT-Base on GLUE tasks
- **Baselines**: Quantization, pruning, tensor decomposition

### 2. Ablation Studies
Analyzes impact of design choices:
- **Architecture**: Depth, width, activation functions
- **Encoding**: Fourier bandwidth, encoding types
- **Training**: Optimizers, learning rates, regularization

### 3. Scaling Analysis
Studies compression behavior with model/tensor size:
- **Tensor Scaling**: Various kernel sizes, channel counts
- **Model Scaling**: Small to large models (1M-300M+ params)
- **Scaling Laws**: Derives compression ratio relationships

### 4. Robustness Evaluation
Tests compressed model robustness:
- **Transfer Learning**: Domain adaptation capabilities
- **Adversarial**: FGSM/PGD attack resistance
- **Fine-tuning**: Gradient flow and stability

### 5. Efficiency Analysis
Benchmarks computational performance:
- **Memory Modes**: Preload vs streaming with caching
- **Hardware**: GPU/CPU optimizations
- **Energy**: Power consumption profiling

## Output Structure

Results are saved in timestamped directories:

```
results/
├── key_experiments_YYYYMMDD_HHMMSS/
│   ├── key_findings.md          # Summary report
│   ├── experiment_list.json     # Experiment configurations
│   └── results.json             # Raw results
│
└── full_experiments_YYYYMMDD_HHMMSS/
    ├── experiment_report.md     # Comprehensive report
    ├── all_results.json         # All experiment results
    ├── analysis.json            # Statistical analysis
    ├── visualizations/          # Plots and figures
    │   ├── compression_accuracy_tradeoff.png
    │   ├── ablation_heatmaps.png
    │   └── scaling_curves.png
    └── [experiment_name]/       # Individual experiment dirs
        ├── config.json
        ├── results.json
        └── logs/
```

## Configuration

Experiments use Hydra for configuration. Override defaults:

```bash
python scripts/run_experiment.py \
    model.name=resnet50 \
    compression.bandwidth=8 \
    compression.hidden_width=512
```

## Key Findings (Expected)

Based on CLAUDE.md, experiments should demonstrate:

1. **Compression**: 20-25x reduction in model size
2. **Accuracy**: <1% degradation (>99% retention)
3. **Efficiency**: <10% inference overhead with caching
4. **Robustness**: Maintains transfer learning and adversarial robustness
5. **Scaling**: Favorable power-law scaling with tensor size

## Requirements

- PyTorch >= 1.12.0
- CUDA-capable GPU (recommended)
- 32GB+ RAM for large model experiments
- Python packages: see requirements.txt

## Reproducibility

All experiments use:
- Fixed random seeds
- Deterministic operations
- Logged configurations
- Versioned dependencies

Results include confidence intervals from multiple runs.