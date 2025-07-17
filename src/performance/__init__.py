"""
Performance Optimization Package for GrandModel

This package provides comprehensive performance optimization solutions:

Memory Optimization:
- Tensor pooling for PyTorch operations
- Garbage collection optimization
- Memory leak detection and cleanup
- Object pooling for frequently created objects

CPU Optimization:
- PyTorch JIT compilation for inference
- Vectorized mathematical operations using NumPy/Numba
- Thread pool optimization
- CPU affinity settings for critical processes

I/O Optimization:
- Async I/O operations for non-blocking operations
- Intelligent caching with LRU and time-based eviction
- Database connection pooling
- Batch processing for high-volume operations

Enhanced Monitoring:
- Real-time performance metrics collection
- ML-based performance regression detection
- Automated performance alerts
- Performance trend analysis and dashboards

Integration:
- Unified optimization control system
- Coordinated optimization strategies
- Performance validation framework
- Comprehensive testing suite

Usage Example:
    from src.performance import integrated_optimizer
    
    # Enable all optimizations
    await integrated_optimizer.enable_optimizations('real_time')
    
    # Auto-optimize based on workload
    strategy = await integrated_optimizer.auto_optimize()
    
    # Validate optimizations
    report = await integrated_optimizer.validate_optimizations()
    
    # Use optimization context
    async with integrated_optimizer.optimization_context('inference'):
        # Your optimized code here
        pass
"""

# Legacy imports
from .async_event_bus import AsyncEventBus
from .memory_manager import MemoryManager
from .connection_pool import ConnectionPool

# New optimization components
from .memory_optimizer import (
    MemoryOptimizer,
    TensorPool,
    ObjectPool,
    GCOptimizer,
    memory_optimizer,
    optimize_memory,
    memory_optimized_context
)

from .cpu_optimizer import (
    CPUOptimizer,
    JITCompiler,
    VectorizedOps,
    ThreadPoolOptimizer,
    CPUAffinityManager,
    cpu_optimizer,
    optimize_cpu
)

from .io_optimizer import (
    IOOptimizer,
    AsyncCache,
    AsyncFileManager,
    AsyncConnectionPool,
    BatchProcessor,
    io_optimizer,
    optimize_io
)

from .enhanced_monitoring import (
    EnhancedPerformanceMonitor,
    RealTimeMetricsCollector,
    PerformanceRegressionDetector,
    PerformanceAlertSystem,
    enhanced_monitor,
    monitor_performance
)

from .integrated_optimizer import (
    IntegratedOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    PerformanceValidator,
    integrated_optimizer,
    optimize_system,
    optimize_performance
)

from .test_optimization_suite import (
    PerformanceTestSuite
)

# Version information
__version__ = "1.0.0"
__author__ = "GrandModel Performance Team"
__description__ = "Comprehensive performance optimization system for GrandModel"

# Export all components
__all__ = [
    # Legacy components
    "AsyncEventBus", 
    "MemoryManager", 
    "ConnectionPool",
    
    # Memory optimization
    'MemoryOptimizer',
    'TensorPool',
    'ObjectPool',
    'GCOptimizer',
    'memory_optimizer',
    'optimize_memory',
    'memory_optimized_context',
    
    # CPU optimization
    'CPUOptimizer',
    'JITCompiler',
    'VectorizedOps',
    'ThreadPoolOptimizer',
    'CPUAffinityManager',
    'cpu_optimizer',
    'optimize_cpu',
    
    # I/O optimization
    'IOOptimizer',
    'AsyncCache',
    'AsyncFileManager',
    'AsyncConnectionPool',
    'BatchProcessor',
    'io_optimizer',
    'optimize_io',
    
    # Enhanced monitoring
    'EnhancedPerformanceMonitor',
    'RealTimeMetricsCollector',
    'PerformanceRegressionDetector',
    'PerformanceAlertSystem',
    'enhanced_monitor',
    'monitor_performance',
    
    # Integrated optimization
    'IntegratedOptimizer',
    'OptimizationConfig',
    'OptimizationStrategy',
    'PerformanceValidator',
    'integrated_optimizer',
    'optimize_system',
    'optimize_performance',
    
    # Testing
    'PerformanceTestSuite'
]

# Convenience functions
async def enable_all_optimizations(strategy: str = 'real_time'):
    """Enable all performance optimizations with specified strategy"""
    await integrated_optimizer.enable_optimizations(strategy)

async def disable_all_optimizations():
    """Disable all performance optimizations"""
    await integrated_optimizer.disable_optimizations()

async def get_performance_status():
    """Get current performance optimization status"""
    return await integrated_optimizer.get_comprehensive_stats()

async def run_performance_tests():
    """Run comprehensive performance tests"""
    test_suite = PerformanceTestSuite()
    return test_suite.run_all_tests()

# Package initialization
import logging
import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

logger.info("Performance optimization package initialized", version=__version__)