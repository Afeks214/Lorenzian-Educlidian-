"""
Lorentzian Strategy Module

This module implements the mathematical core of the Lorentzian distance classification system
for financial time series analysis and trading strategy development.

Key Components:
- distance_metrics: Core Lorentzian distance implementation with performance optimization
- Mathematical validation and testing framework
- GPU acceleration support
- Production-ready error handling and monitoring

Mathematical Foundation:
The Lorentzian distance metric D_L(x,y) = Σᵢ ln(1 + |xᵢ - yᵢ|) provides superior
pattern recognition for financial time series compared to traditional Euclidean distance.

Author: Claude Code Assistant
Version: 1.0.0
Date: 2025-07-20
"""

from .distance_metrics import (
    # Main classes
    LorentzianDistanceCalculator,
    DistanceMetricsConfig,
    DistanceResult,
    PerformanceMonitor,
    DistanceCache,
    
    # Convenience functions
    lorentzian_distance,
    euclidean_distance,
    manhattan_distance,
    
    # Testing and validation
    run_comprehensive_tests,
)

# Version information
__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__email__ = "noreply@anthropic.com"

# Module metadata
__all__ = [
    # Main classes
    "LorentzianDistanceCalculator",
    "DistanceMetricsConfig", 
    "DistanceResult",
    "PerformanceMonitor",
    "DistanceCache",
    
    # Convenience functions
    "lorentzian_distance",
    "euclidean_distance", 
    "manhattan_distance",
    
    # Testing and validation
    "run_comprehensive_tests",
]

# Default configuration for easy access
DEFAULT_CONFIG = DistanceMetricsConfig()

# Quick access functions for common use cases
def quick_lorentzian_distance(x, y, epsilon=1e-12):
    """Quick calculation of Lorentzian distance with minimal overhead"""
    return lorentzian_distance(x, y, epsilon)

def create_optimized_calculator(**kwargs):
    """Create a LorentzianDistanceCalculator with optimized settings"""
    config = DistanceMetricsConfig(**kwargs)
    return LorentzianDistanceCalculator(config)

def create_production_calculator():
    """Create a LorentzianDistanceCalculator optimized for production use"""
    config = DistanceMetricsConfig(
        use_numba_jit=True,
        use_gpu_acceleration=True,
        enable_caching=True,
        cache_size=50000,
        validate_inputs=True,
        log_performance=True
    )
    return LorentzianDistanceCalculator(config)

# Module initialization
import logging
logger = logging.getLogger(__name__)
logger.info(f"Lorentzian Strategy Module v{__version__} initialized")