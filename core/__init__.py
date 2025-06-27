"""Core modules for Implicit Neural Weight Field compression."""

from .siren import SIREN, SIRENLayer
from .positional_encoding import FourierFeatures, positional_encoding
from .implicit_field import ImplicitWeightField, MultiScaleImplicitField

__all__ = [
    'SIREN',
    'SIRENLayer', 
    'FourierFeatures',
    'positional_encoding',
    'ImplicitWeightField',
    'MultiScaleImplicitField'
]