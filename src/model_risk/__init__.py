"""
Model Risk Management Module

This module provides comprehensive model risk management capabilities including
model validation, backtesting, performance monitoring, and statistical significance testing.
"""

__version__ = "1.0.0"
__author__ = "GrandModel MARL Team"

from .model_validator import ModelValidator, ValidationResult, ValidationRule
from .backtesting_engine import BacktestingEngine, BacktestResult, BacktestConfig
from .performance_monitor import ModelPerformanceMonitor, PerformanceMetrics
from .statistical_validator import StatisticalValidator, StatisticalTest
from .model_registry import ModelRegistry, ModelMetadata, ModelStatus

__all__ = [
    "ModelValidator",
    "ValidationResult",
    "ValidationRule",
    "BacktestingEngine", 
    "BacktestResult",
    "BacktestConfig",
    "ModelPerformanceMonitor",
    "PerformanceMetrics",
    "StatisticalValidator",
    "StatisticalTest",
    "ModelRegistry",
    "ModelMetadata",
    "ModelStatus"
]