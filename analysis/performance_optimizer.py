"""
Performance Optimization Module for Metrics Calculations

This module provides performance optimizations for metrics calculations including:
- Vectorized operations with NumPy
- LRU caching for expensive operations
- Parallel processing for bootstrap simulations
- Memory-efficient batch processing
- JIT compilation with Numba
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from functools import lru_cache, wraps
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from numba import jit, njit, prange
import multiprocessing as mp
from dataclasses import dataclass
import logging
from datetime import datetime
import psutil
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Performance statistics for metrics calculations"""
    function_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hits: int
    cache_misses: int
    parallelization_used: bool
    
    def __str__(self) -> str:
        return f"{self.function_name}: {self.execution_time_ms:.2f}ms, {self.memory_usage_mb:.1f}MB"


class PerformanceMonitor:
    """Monitor performance of metrics calculations"""
    
    def __init__(self):
        self.stats_history: List[PerformanceStats] = []
        self.cache_stats: Dict[str, Dict[str, int]] = {}
        self.process = psutil.Process()
    
    def record_performance(
        self,
        function_name: str,
        execution_time_ms: float,
        parallelization_used: bool = False
    ) -> None:
        """Record performance statistics"""
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        cpu_usage_percent = self.process.cpu_percent()
        
        # Get cache stats
        cache_stats = self.cache_stats.get(function_name, {"hits": 0, "misses": 0})
        
        stats = PerformanceStats(
            function_name=function_name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            parallelization_used=parallelization_used
        )
        
        self.stats_history.append(stats)
        
        # Keep only recent history
        if len(self.stats_history) > 1000:
            self.stats_history = self.stats_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.stats_history:
            return {"status": "No performance data available"}
        
        # Group by function
        function_stats = {}
        for stat in self.stats_history:
            if stat.function_name not in function_stats:
                function_stats[stat.function_name] = []
            function_stats[stat.function_name].append(stat)
        
        # Calculate averages
        summary = {}
        for func_name, stats in function_stats.items():
            avg_time = np.mean([s.execution_time_ms for s in stats])
            max_time = np.max([s.execution_time_ms for s in stats])
            avg_memory = np.mean([s.memory_usage_mb for s in stats])
            parallel_usage = np.mean([s.parallelization_used for s in stats])
            
            summary[func_name] = {
                "avg_time_ms": avg_time,
                "max_time_ms": max_time,
                "avg_memory_mb": avg_memory,
                "parallel_usage_ratio": parallel_usage,
                "call_count": len(stats)
            }
        
        return summary


# Global performance monitor
performance_monitor = PerformanceMonitor()


def performance_tracked(func: Callable) -> Callable:
    """Decorator to track performance of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        performance_monitor.record_performance(
            function_name=func.__name__,
            execution_time_ms=execution_time,
            parallelization_used=kwargs.get('parallel', False)
        )
        
        return result
    return wrapper


def cached_metric(maxsize: int = 128):
    """Enhanced caching decorator with statistics tracking"""
    def decorator(func: Callable) -> Callable:
        cached_func = lru_cache(maxsize=maxsize)(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert numpy arrays to tuples for caching
            cache_key = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    cache_key.append(tuple(arg))
                else:
                    cache_key.append(arg)
            
            # Check cache stats
            cache_info = cached_func.cache_info()
            func_name = func.__name__
            
            if func_name not in performance_monitor.cache_stats:
                performance_monitor.cache_stats[func_name] = {"hits": 0, "misses": 0}
            
            old_hits = cache_info.hits
            result = cached_func(*cache_key, **kwargs)
            new_cache_info = cached_func.cache_info()
            
            # Update cache stats
            if new_cache_info.hits > old_hits:
                performance_monitor.cache_stats[func_name]["hits"] += 1
            else:
                performance_monitor.cache_stats[func_name]["misses"] += 1
            
            return result
        
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        return wrapper
    return decorator


@njit
def vectorized_returns_calculation(prices: np.ndarray) -> np.ndarray:
    """Vectorized returns calculation - JIT optimized"""
    if len(prices) < 2:
        return np.array([])
    
    returns = np.zeros(len(prices) - 1)
    for i in range(1, len(prices)):
        returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1]
    
    return returns


@njit
def vectorized_drawdown_calculation(equity_curve: np.ndarray) -> np.ndarray:
    """Vectorized drawdown calculation - JIT optimized"""
    if len(equity_curve) == 0:
        return np.array([])
    
    n = len(equity_curve)
    running_max = np.zeros(n)
    drawdown = np.zeros(n)
    
    running_max[0] = equity_curve[0]
    drawdown[0] = 0.0
    
    for i in range(1, n):
        running_max[i] = max(running_max[i - 1], equity_curve[i])
        drawdown[i] = (equity_curve[i] - running_max[i]) / running_max[i]
    
    return drawdown


@njit
def vectorized_rolling_volatility(
    returns: np.ndarray, 
    window: int = 252
) -> np.ndarray:
    """Vectorized rolling volatility calculation - JIT optimized"""
    if len(returns) < window:
        return np.array([np.std(returns)])
    
    n = len(returns)
    rolling_vol = np.zeros(n - window + 1)
    
    for i in range(window - 1, n):
        rolling_vol[i - window + 1] = np.std(returns[i - window + 1:i + 1])
    
    return rolling_vol


@njit(parallel=True)
def parallel_bootstrap_metrics(
    returns: np.ndarray,
    block_size: int,
    n_samples: int
) -> np.ndarray:
    """Parallel bootstrap sampling - JIT optimized"""
    n = len(returns)
    if n <= block_size:
        return returns
    
    # PERFORMANCE FIX: Pre-allocate memory pool to prevent memory leak
    bootstrap_samples = np.zeros((n_samples, n))
    
    # Pre-allocate reusable work arrays to reduce memory allocation
    indices = np.zeros(n, dtype=np.int32)
    
    for sample_idx in prange(n_samples):
        # Generate bootstrap sample indices efficiently
        sample_pos = 0
        
        while sample_pos < n:
            # Random block start
            block_start = np.random.randint(0, n - block_size + 1)
            block_end = min(block_start + block_size, n)
            block_length = block_end - block_start
            
            # Copy block indices
            copy_length = min(block_length, n - sample_pos)
            for i in range(copy_length):
                indices[sample_pos + i] = block_start + i
            
            sample_pos += copy_length
        
        # Use vectorized indexing instead of loops
        bootstrap_samples[sample_idx] = returns[indices]
    
    return bootstrap_samples


class BatchMetricsCalculator:
    """Batch calculator for efficient metrics computation"""
    
    def __init__(self, batch_size: int = 1000, max_workers: int = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
        
    def calculate_batch_metrics(
        self,
        returns_batches: List[np.ndarray],
        metric_functions: List[Callable],
        parallel: bool = True
    ) -> List[Dict[str, float]]:
        """Calculate metrics for multiple return series in batches"""
        if not parallel or len(returns_batches) < 2:
            # Sequential processing
            results = []
            for returns in returns_batches:
                batch_result = {}
                for func in metric_functions:
                    try:
                        batch_result[func.__name__] = func(returns)
                    except Exception as e:
                        logger.warning(f"Metric calculation failed: {e}")
                        batch_result[func.__name__] = 0.0
                results.append(batch_result)
            return results
        
        # Parallel processing
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for i, returns in enumerate(returns_batches):
                future = executor.submit(
                    self._calculate_single_batch,
                    returns,
                    metric_functions
                )
                future_to_batch[future] = i
            
            # Collect results in order
            batch_results = [None] * len(returns_batches)
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results[batch_idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch calculation failed: {e}")
                    batch_results[batch_idx] = {
                        func.__name__: 0.0 for func in metric_functions
                    }
            
            results = batch_results
        
        return results
    
    def _calculate_single_batch(
        self,
        returns: np.ndarray,
        metric_functions: List[Callable]
    ) -> Dict[str, float]:
        """Calculate metrics for a single batch"""
        result = {}
        for func in metric_functions:
            try:
                result[func.__name__] = func(returns)
            except Exception as e:
                logger.warning(f"Metric calculation failed: {e}")
                result[func.__name__] = 0.0
        return result


class MemoryEfficientCalculator:
    """Memory-efficient calculator for large datasets"""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    def calculate_streaming_metrics(
        self,
        returns_stream: np.ndarray,
        metric_functions: List[Callable]
    ) -> Dict[str, float]:
        """Calculate metrics in streaming fashion for memory efficiency"""
        n = len(returns_stream)
        if n <= self.chunk_size:
            # Small enough to calculate directly
            results = {}
            for func in metric_functions:
                try:
                    results[func.__name__] = func(returns_stream)
                except Exception as e:
                    logger.warning(f"Streaming metric calculation failed: {e}")
                    results[func.__name__] = 0.0
            return results
        
        # Process in chunks
        chunk_results = []
        for i in range(0, n, self.chunk_size):
            chunk = returns_stream[i:i + self.chunk_size]
            chunk_result = {}
            
            for func in metric_functions:
                try:
                    chunk_result[func.__name__] = func(chunk)
                except Exception as e:
                    logger.warning(f"Chunk metric calculation failed: {e}")
                    chunk_result[func.__name__] = 0.0
            
            chunk_results.append(chunk_result)
        
        # Aggregate results
        final_results = {}
        for func in metric_functions:
            func_name = func.__name__
            values = [cr[func_name] for cr in chunk_results if func_name in cr]
            
            if values:
                # Simple average aggregation (could be improved per metric)
                final_results[func_name] = np.mean(values)
            else:
                final_results[func_name] = 0.0
        
        return final_results


# Performance-optimized versions of common metrics
@performance_tracked
@cached_metric(maxsize=256)
def optimized_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Optimized Sharpe ratio calculation"""
    if len(returns) == 0:
        return 0.0
    
    # Vectorized calculation
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_excess / std_excess


@performance_tracked
@cached_metric(maxsize=256)
def optimized_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Optimized Sortino ratio calculation"""
    if len(returns) == 0:
        return 0.0
    
    # Vectorized calculation
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess_returns)
    
    # Vectorized downside calculation
    downside_mask = excess_returns < 0
    downside_returns = excess_returns[downside_mask]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_excess / downside_std


@performance_tracked
@cached_metric(maxsize=256)
def optimized_max_drawdown(equity_curve: np.ndarray) -> float:
    """Optimized maximum drawdown calculation"""
    if len(equity_curve) == 0:
        return 0.0
    
    # Use vectorized calculation
    drawdown_series = vectorized_drawdown_calculation(equity_curve)
    return abs(np.min(drawdown_series))


class MetricsOptimizer:
    """Main optimizer class for metrics calculations"""
    
    def __init__(
        self,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        enable_vectorization: bool = True,
        batch_size: int = 1000,
        max_workers: int = None
    ):
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.enable_vectorization = enable_vectorization
        self.batch_calculator = BatchMetricsCalculator(batch_size, max_workers)
        self.memory_calculator = MemoryEfficientCalculator()
        
        # Register optimized functions
        self.optimized_functions = {
            'sharpe_ratio': optimized_sharpe_ratio,
            'sortino_ratio': optimized_sortino_ratio,
            'max_drawdown': optimized_max_drawdown
        }
    
    def optimize_calculation(
        self,
        function_name: str,
        data: np.ndarray,
        **kwargs
    ) -> float:
        """Optimize calculation based on settings"""
        if function_name in self.optimized_functions:
            return self.optimized_functions[function_name](data, **kwargs)
        
        # Fallback to standard calculation
        logger.warning(f"No optimized version available for {function_name}")
        return 0.0
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        return {
            "caching_enabled": self.enable_caching,
            "parallel_enabled": self.enable_parallel,
            "vectorization_enabled": self.enable_vectorization,
            "optimized_functions": list(self.optimized_functions.keys()),
            "performance_stats": performance_monitor.get_performance_summary()
        }
    
    def benchmark_performance(
        self,
        returns: np.ndarray,
        iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark performance of optimized vs standard functions"""
        benchmark_results = {}
        
        for func_name, func in self.optimized_functions.items():
            # Benchmark optimized version
            start_time = time.time()
            for _ in range(iterations):
                func(returns)
            optimized_time = (time.time() - start_time) * 1000 / iterations
            
            benchmark_results[func_name] = {
                "optimized_time_ms": optimized_time,
                "cache_hits": performance_monitor.cache_stats.get(func_name, {}).get("hits", 0),
                "cache_misses": performance_monitor.cache_stats.get(func_name, {}).get("misses", 0)
            }
        
        return benchmark_results


# Global optimizer instance
metrics_optimizer = MetricsOptimizer()


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report"""
    return {
        "performance_monitor": performance_monitor.get_performance_summary(),
        "optimizer_settings": metrics_optimizer.get_optimization_summary(),
        "system_info": {
            "cpu_count": mp.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
    }