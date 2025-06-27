# Getting Started with Implicit Weight Field Compression

This guide will help you get started with using the Implicit Weight Field compression framework.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.13.0 or higher
- CUDA 11.6 or higher (for GPU support)
- 16GB RAM minimum (32GB recommended)
- GPU with at least 8GB VRAM (for large models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/implicit_weight_field.git
cd implicit_weight_field
```

### Step 2: Set Up Environment

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate implicit-weight-field
```

Using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Download Example Data

```bash
# Download CIFAR-10 for quick testing
python -c "import torchvision; torchvision.datasets.CIFAR10('./data/cifar', download=True)"
```

## Basic Example: Compress a Small Model

Let's start with a simple example of compressing a small CNN model:

```python
import torch
import torch.nn as nn
from compression import compress_model
from core.implicit_field import CompressionConfig

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and compress the model
model = SimpleCNN()

# Configure compression
config = CompressionConfig(
    bandwidth=4,
    hidden_width=128,
    max_steps=1000
)

# Compress
compressed_model, results = compress_model(model, config)

# Print results
print(f"Original parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Compressed parameters: {results.compressed_parameters}")
print(f"Compression ratio: {results.total_compression_ratio:.2f}x")
```

## Compressing Pre-trained Models

### ResNet-50 Example

```python
import torchvision.models as models
from compression import ModelCompressor
from core.implicit_field import CompressionConfig

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)
model.eval()

# Create compressor with custom configuration
config = CompressionConfig(
    bandwidth=4,
    hidden_width=256,
    num_layers=2,
    learning_rate=1e-3,
    max_steps=2000
)

compressor = ModelCompressor(
    model, 
    config=config,
    min_tensor_size=10000,  # Only compress large tensors
    device='cuda'
)

# Compress the model
result = compressor.compress()

# Get compressed model
compressed_model = compressor.get_compressed_model()

# Save compressed model
compressor.save_compressed_model('resnet50_compressed.pth')

print(f"Compression ratio: {result.total_compression_ratio:.2f}x")
print(f"Number of compressed layers: {len(result.layer_results)}")
```

### Evaluating Compressed Model

```python
from evaluation import evaluate_model_accuracy
from experiments.models import prepare_imagenet_loader

# Prepare data loader
val_loader = prepare_imagenet_loader(
    data_dir='/path/to/imagenet',
    batch_size=128,
    split='val'
)

# Evaluate original model
original_metrics = evaluate_model_accuracy(model, val_loader, device='cuda')
print(f"Original Top-1 Accuracy: {original_metrics.top1_accuracy:.2f}%")

# Evaluate compressed model
compressed_metrics = evaluate_model_accuracy(compressed_model, val_loader, device='cuda')
print(f"Compressed Top-1 Accuracy: {compressed_metrics.top1_accuracy:.2f}%")
print(f"Accuracy drop: {original_metrics.top1_accuracy - compressed_metrics.top1_accuracy:.2f}%")
```

## Understanding Compression Configuration

### Key Parameters

- **bandwidth**: Controls frequency representation (higher = more detail)
  - Small models: 2-4
  - Large models: 4-8
  
- **hidden_width**: Hidden layer size in SIREN networks
  - Small tensors: 64-128
  - Large tensors: 256-512
  
- **num_layers**: Depth of SIREN networks
  - Simple patterns: 2 layers
  - Complex patterns: 3-4 layers
  
- **learning_rate**: Training rate for implicit fields
  - Default: 1e-3
  - Decrease if training is unstable
  
- **max_steps**: Maximum training iterations per field
  - Quick compression: 500-1000
  - High quality: 2000-5000

### Adaptive Configuration

The framework automatically adjusts architecture based on tensor properties:

```python
# Let the framework choose optimal settings
config = CompressionConfig()  # Uses defaults

# Or provide hints
config = CompressionConfig(
    target_compression_ratio=20.0,  # Desired compression
    quality_priority=0.8  # 0=speed, 1=quality
)
```

## Memory-Efficient Inference

### Streaming Mode

For deployment on memory-constrained devices:

```python
from inference import StreamingInference

# Create streaming inference engine
inference = StreamingInference(
    compressed_fields,
    cache_size_mb=50,  # Only 50MB cache
    enable_prefetch=True
)

# Use in model
class StreamingModel(nn.Module):
    def __init__(self, inference_engine):
        super().__init__()
        self.inference = inference_engine
        
    def forward(self, x):
        # Weights are generated on-demand
        conv1_weight = self.inference.get_weight('conv1.weight')
        x = F.conv2d(x, conv1_weight)
        # ... rest of forward pass
        return x
```

### Preload Mode

For maximum inference speed:

```python
from inference import PreloadInference

# Preload all weights at initialization
inference = PreloadInference(compressed_fields, device='cuda')

# All weights are immediately available
weight = inference.get_weight('layer.weight')  # No generation needed
```

## Monitoring Compression

### Using Weights & Biases

```python
import wandb

# Initialize wandb
wandb.init(project="compression-experiments")

# Log compression results
wandb.log({
    "compression_ratio": result.total_compression_ratio,
    "parameter_reduction": result.parameter_reduction,
    "layers_compressed": len(result.layer_results)
})

# Log per-layer metrics
for layer_name, layer_result in result.layer_results.items():
    wandb.log({
        f"{layer_name}/compression_ratio": layer_result.compression_ratio,
        f"{layer_name}/reconstruction_error": layer_result.reconstruction_error
    })
```

### Visualization

```python
from visualization import WeightVisualizer

visualizer = WeightVisualizer()

# Compare original vs reconstructed weights
fig = visualizer.visualize_weight_comparison(
    original_weights,
    reconstructed_weights,
    layer_name='conv1.weight'
)
fig.savefig('weight_comparison.png')

# Analyze compression across layers
fig = visualizer.visualize_compression_summary(
    compression_results
)
fig.savefig('compression_summary.png')
```

## Common Issues and Solutions

### Issue: Low Compression Ratio

**Solution**: Adjust configuration for your model:
```python
config = CompressionConfig(
    bandwidth=8,  # Increase for more complex patterns
    hidden_width=512,  # Larger networks for complex weights
    num_layers=3  # Deeper networks
)
```

### Issue: High Reconstruction Error

**Solution**: Increase training steps or use multi-scale:
```python
config = CompressionConfig(
    max_steps=5000,  # More training
    use_multiscale=True,  # For large tensors
    num_scales=3
)
```

### Issue: Out of Memory

**Solution**: Process layers sequentially:
```python
compressor = ModelCompressor(model, config)

# Compress one layer at a time
for name, param in model.named_parameters():
    if param.numel() > 10000:  # Only large layers
        compressor.compress_layer(name)
```

## Next Steps

- [Advanced Configuration](advanced_configuration.md) - Fine-tune compression settings
- [Experiment Guide](experiment_guide.md) - Run full experiments
- [API Reference](api_reference.md) - Detailed API documentation
- [Examples](../examples/) - More code examples