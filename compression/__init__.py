"""Compression pipeline for implicit weight fields."""

from .compressor import (
    ImplicitWeightFieldCompressor,
    CompressionResult,
    compress_model
)
from .trainer import FieldTrainer, TrainingConfig

__all__ = [
    'ImplicitWeightFieldCompressor',
    'CompressionResult',
    'compress_model',
    'FieldTrainer',
    'TrainingConfig'
]