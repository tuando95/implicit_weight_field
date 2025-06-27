#!/bin/bash
# Setup script for Implicit Weight Field compression experiments

echo "Setting up Implicit Weight Field compression environment..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data/imagenet
mkdir -p data/cifar
mkdir -p results
mkdir -p logs
mkdir -p checkpoints
mkdir -p notebooks

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
else
    echo "Running on host system"
    
    # Check for conda
    if command -v conda &> /dev/null; then
        echo "Creating conda environment..."
        conda env create -f environment.yml
        echo "Activate the environment with: conda activate implicit-weight-field"
    else
        echo "Conda not found. Installing dependencies with pip..."
        pip install -r requirements.txt
    fi
fi

# Download example data (CIFAR-10 for quick testing)
echo "Downloading CIFAR-10 dataset..."
python -c "
import torchvision
import os
data_dir = os.environ.get('DATA_DIR', './data')
torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar'), download=True)
print('CIFAR-10 downloaded successfully')
"

# Create example config
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env with your settings"
fi

# Run tests to verify installation
echo "Running basic tests..."
python -m pytest tests/test_siren.py::TestSIRENLayer::test_layer_creation -v

echo "Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the environment: conda activate implicit-weight-field"
echo "2. Edit .env with your settings" 
echo "3. Run experiments: python scripts/train_compression.py --help"