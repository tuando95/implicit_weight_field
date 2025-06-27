"""Inference utilities for implicit weight fields."""

from .modes import InferenceMode, PreloadInference, StreamingInference
from .cache import LRUWeightCache, CacheConfig
#from .wrapper import ImplicitWeightModule, wrap_model_for_inference

__all__ = [
    'InferenceMode',
    'PreloadInference',
    'StreamingInference',
    'LRUWeightCache',
    'CacheConfig'
]