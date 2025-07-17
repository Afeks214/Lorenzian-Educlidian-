"""
Unified Data Pipeline System for NQ Dataset Processing

This module provides a comprehensive data pipeline system for massive NQ dataset processing
that can be used by both execution engine and risk management notebooks.

Key Features:
- Unified data loading with chunked processing
- Memory optimization with shared pools
- Data flow coordination between notebooks
- Performance monitoring and benchmarking
- Scalability with multi-GPU and distributed processing
"""

__version__ = "1.0.0"
__author__ = "QuantNova GrandModel Team"

from .unified_data_loader import UnifiedDataLoader
from .memory_manager import MemoryManager, SharedMemoryPool
from .data_flow_coordinator import (
    DataFlowCoordinator,
    EnhancedDataFlowCoordinator,
    EnhancedDataStream,
    DataStreamPriority,
    AtomicCounter,
    ThreadSafeDict,
    DependencyGraph,
    LockFreeQueue,
    ConcurrencyMonitor,
    EnhancedCoordinatorConfig,
    create_enhanced_coordinator
)
from .performance_monitor import PerformanceMonitor, DataLoadingBenchmark
from .scalability_manager import ScalabilityManager, MultiGPUProcessor

__all__ = [
    'UnifiedDataLoader',
    'MemoryManager',
    'SharedMemoryPool',
    'DataFlowCoordinator',
    'EnhancedDataFlowCoordinator',
    'EnhancedDataStream',
    'DataStreamPriority',
    'AtomicCounter',
    'ThreadSafeDict',
    'DependencyGraph',
    'LockFreeQueue',
    'ConcurrencyMonitor',
    'EnhancedCoordinatorConfig',
    'create_enhanced_coordinator',
    'PerformanceMonitor',
    'DataLoadingBenchmark',
    'ScalabilityManager',
    'MultiGPUProcessor'
]