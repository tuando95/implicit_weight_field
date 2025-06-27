"""Analysis tools for compression errors and failures."""

from .error_analysis import (
    CompressionErrorAnalyzer,
    AdaptiveCompressionStrategy,
    CompressionDiagnostics,
    ErrorPattern,
    FailureCase,
    create_error_analysis_report
)

__all__ = [
    'CompressionErrorAnalyzer',
    'AdaptiveCompressionStrategy',
    'CompressionDiagnostics',
    'ErrorPattern',
    'FailureCase',
    'create_error_analysis_report'
]