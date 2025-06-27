# Docker image for Implicit Weight Field Compression experiments
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data and results
RUN mkdir -p data results logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=8

# Default command
CMD ["/bin/bash"]