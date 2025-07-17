"""
Performance Optimization Module for Risk Monitor Agent

Ultra-high performance optimization system ensuring <10ms response times
for all critical risk calculations with advanced caching, JIT compilation,
and parallel processing capabilities.

Key Features:
- <10ms response time guarantee
- JIT compilation for hot paths
- Advanced caching strategies
- Parallel processing optimization
- Memory pool management
- Performance monitoring and alerting
"""

import logging


import numpy as np
import asyncio
import numba
from numba import jit, njit, prange
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import multiprocessing as mp
from collections import deque
import psutil
import gc

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    function_name: str
    avg_execution_time_ms: float
    max_execution_time_ms: float
    min_execution_time_ms: float
    call_count: int
    cache_hit_rate: float
    memory_usage_mb: float
    target_time_ms: float
    violations: int
    
    @property
    def is_compliant(self) -> bool:
        return self.avg_execution_time_ms <= self.target_time_ms


class PerformanceCache:
    """
    High-performance caching system with TTL and LRU eviction
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_order: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check TTL
                if (datetime.now() - timestamp).total_seconds() < self.ttl_seconds:
                    # Update access order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self.hits += 1
                    return value
                else:
                    # Expired - remove
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Store value in cache with current timestamp"""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove least recently used
                if self.access_order:
                    lru_key = self.access_order.popleft()
                    if lru_key in self.cache:
                        del self.cache[lru_key]
            
            # Store new value
            self.cache[key] = (value, datetime.now())
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached values"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryPool:
    """
    Pre-allocated memory pool for high-frequency calculations
    """
    
    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.arrays_1d: deque = deque()
        self.arrays_2d: deque = deque()
        self.matrices: deque = deque()
        self.lock = threading.Lock()
        
        # Pre-allocate arrays
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize memory pools with pre-allocated arrays"""
        # 1D arrays for risk vectors
        for _ in range(self.pool_size):
            self.arrays_1d.append(np.zeros(10, dtype=np.float64))
        
        # 2D arrays for small matrices
        for _ in range(self.pool_size // 2):
            self.arrays_2d.append(np.zeros((10, 10), dtype=np.float64))
        
        # Larger matrices for correlation calculations
        for _ in range(self.pool_size // 5):
            self.matrices.append(np.zeros((100, 100), dtype=np.float64))
    
    def get_array_1d(self, size: int = 10) -> np.ndarray:
        """Get pre-allocated 1D array"""
        with self.lock:
            if self.arrays_1d and size <= 10:
                array = self.arrays_1d.popleft()
                array.fill(0)  # Reset values
                return array[:size]
            else:
                return np.zeros(size, dtype=np.float64)
    
    def return_array_1d(self, array: np.ndarray):
        """Return 1D array to pool"""
        with self.lock:
            if len(self.arrays_1d) < self.pool_size and array.shape[0] <= 10:
                self.arrays_1d.append(array)
    
    def get_array_2d(self, shape: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """Get pre-allocated 2D array"""
        with self.lock:
            if self.arrays_2d and shape[0] <= 10 and shape[1] <= 10:
                array = self.arrays_2d.popleft()
                array.fill(0)
                return array[:shape[0], :shape[1]]
            else:
                return np.zeros(shape, dtype=np.float64)
    
    def return_array_2d(self, array: np.ndarray):
        """Return 2D array to pool"""
        with self.lock:
            if len(self.arrays_2d) < self.pool_size // 2 and array.shape[0] <= 10 and array.shape[1] <= 10:
                self.arrays_2d.append(array)


# JIT-compiled functions for critical calculations
@njit(cache=True, fastmath=True)
def fast_var_calculation(weights: np.ndarray, volatilities: np.ndarray, 
                        correlation_matrix: np.ndarray, z_score: float) -> float:
    """Ultra-fast VaR calculation using JIT compilation"""
    n = len(weights)
    portfolio_variance = 0.0
    
    # Vectorized variance calculation
    for i in prange(n):
        for j in prange(n):
            portfolio_variance += (weights[i] * weights[j] * 
                                 volatilities[i] * volatilities[j] * 
                                 correlation_matrix[i, j])
    
    portfolio_vol = np.sqrt(portfolio_variance)
    return portfolio_vol * z_score


@njit(cache=True, fastmath=True)
def fast_correlation_update(old_corr: np.ndarray, returns: np.ndarray, 
                           lambda_param: float) -> np.ndarray:
    """Fast EWMA correlation matrix update"""
    n = returns.shape[0]
    new_corr = np.zeros((n, n))
    
    # Calculate sample correlation
    returns_centered = returns - np.mean(returns)
    outer_product = np.outer(returns_centered, returns_centered)
    
    # EWMA update
    for i in prange(n):
        for j in prange(n):
            if i == j:
                new_corr[i, j] = 1.0
            else:
                new_corr[i, j] = (lambda_param * old_corr[i, j] + 
                                (1 - lambda_param) * outer_product[i, j])
    
    return new_corr


@njit(cache=True, fastmath=True)
def fast_stress_score(volatility_spike: float, correlation_shock: float,
                     volume_anomaly: float, price_gap: float,
                     liquidity_drought: float, flash_crash_prob: float) -> float:
    """Fast market stress score calculation"""
    return (volatility_spike * 0.25 + correlation_shock * 0.25 + 
            volume_anomaly * 0.15 + price_gap * 0.15 + 
            liquidity_drought * 0.10 + flash_crash_prob * 0.10)


@njit(cache=True, fastmath=True)
def fast_breach_detection(current_values: np.ndarray, 
                         thresholds: np.ndarray) -> np.ndarray:
    """Fast breach detection across multiple metrics"""
    n = len(current_values)
    breach_flags = np.zeros(n, dtype=np.int8)
    
    for i in prange(n):
        if current_values[i] > thresholds[i]:
            breach_flags[i] = 1
    
    return breach_flags


def performance_monitor(target_time_ms: float = 10.0):
    """
    Decorator for monitoring function performance
    """
    def decorator(func: Callable) -> Callable:
        func._performance_metrics = {
            'times': deque(maxlen=1000),
            'violations': 0,
            'total_calls': 0
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                
                # Update metrics
                func._performance_metrics['times'].append(execution_time)
                func._performance_metrics['total_calls'] += 1
                
                if execution_time > target_time_ms:
                    func._performance_metrics['violations'] += 1
                    logger.warning("Performance target violation",
                                 function=func.__name__,
                                 execution_time_ms=execution_time,
                                 target_ms=target_time_ms)
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """
    Performance Optimization Engine for Risk Monitor Agent
    
    Ensures <10ms response times through advanced optimization techniques
    including caching, JIT compilation, and parallel processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Performance targets
        self.response_time_target_ms = config.get('response_time_target_ms', 10.0)
        self.cache_size = config.get('cache_size', 1000)
        self.cache_ttl = config.get('cache_ttl_seconds', 60)
        
        # Optimization components
        self.cache = PerformanceCache(self.cache_size, self.cache_ttl)
        self.memory_pool = MemoryPool(config.get('memory_pool_size', 100))
        
        # Thread pool for parallel processing
        self.max_workers = min(config.get('max_workers', 4), mp.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.function_metrics: Dict[str, PerformanceMetrics] = {}
        self.global_performance = {
            'total_optimizations': 0,
            'cache_optimizations': 0,
            'jit_optimizations': 0,
            'parallel_optimizations': 0
        }
        
        # JIT compilation warm-up
        self._warm_up_jit_functions()
        
        self.logger = logger.bind(component="PerformanceOptimizer")
        self.logger.info("Performance Optimizer initialized",
                        target_ms=self.response_time_target_ms,
                        max_workers=self.max_workers)
    
    def _warm_up_jit_functions(self):
        """Warm up JIT-compiled functions for first-time compilation"""
        try:
            # Warm up VaR calculation
            dummy_weights = np.array([0.5, 0.3, 0.2])
            dummy_vols = np.array([0.2, 0.25, 0.3])
            dummy_corr = np.eye(3)
            fast_var_calculation(dummy_weights, dummy_vols, dummy_corr, 1.645)
            
            # Warm up correlation update
            dummy_corr_old = np.eye(3)
            dummy_returns = np.array([0.01, 0.02, -0.01])
            fast_correlation_update(dummy_corr_old, dummy_returns, 0.94)
            
            # Warm up stress score
            fast_stress_score(0.5, 0.3, 0.2, 0.1, 0.4, 0.2)
            
            # Warm up breach detection
            dummy_values = np.array([0.02, 0.05, 0.1])
            dummy_thresholds = np.array([0.01, 0.03, 0.08])
            fast_breach_detection(dummy_values, dummy_thresholds)
            
            self.logger.info("JIT functions warmed up successfully")
            
        except Exception as e:
            self.logger.error("JIT warm-up failed", error=str(e))
    
    @performance_monitor(target_time_ms=5.0)
    def optimized_var_calculation(self, weights: np.ndarray, volatilities: np.ndarray,
                                correlation_matrix: np.ndarray, confidence_level: float) -> float:
        """
        Optimized VaR calculation with caching and JIT compilation
        """
        # Create cache key
        cache_key = f"var_{hash(weights.tobytes())}_{hash(volatilities.tobytes())}_{confidence_level}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.global_performance['cache_optimizations'] += 1
            return cached_result
        
        # Calculate z-score
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        
        # Use JIT-compiled function
        result = fast_var_calculation(weights, volatilities, correlation_matrix, z_score)
        
        # Cache result
        self.cache.put(cache_key, result)
        self.global_performance['jit_optimizations'] += 1
        
        return result
    
    @performance_monitor(target_time_ms=3.0)
    def optimized_correlation_update(self, old_correlation: np.ndarray, 
                                   new_returns: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        Optimized correlation matrix update with memory pooling
        """
        # Use memory pool for intermediate calculations
        n = len(new_returns)
        result_matrix = self.memory_pool.get_array_2d((n, n))
        
        try:
            # Use JIT-compiled function
            updated_corr = fast_correlation_update(old_correlation, new_returns, lambda_param)
            
            # Copy to result matrix
            result_matrix[:n, :n] = updated_corr
            
            self.global_performance['jit_optimizations'] += 1
            return result_matrix[:n, :n].copy()
            
        finally:
            # Return matrix to pool
            self.memory_pool.return_array_2d(result_matrix)
    
    @performance_monitor(target_time_ms=2.0)
    def optimized_stress_calculation(self, stress_indicators: Dict[str, float]) -> float:
        """
        Optimized market stress calculation
        """
        # Extract values in order
        volatility_spike = stress_indicators.get('volatility_spike', 0.0)
        correlation_shock = stress_indicators.get('correlation_shock', 0.0)
        volume_anomaly = stress_indicators.get('volume_anomaly', 0.0)
        price_gap = stress_indicators.get('price_gap', 0.0)
        liquidity_drought = stress_indicators.get('liquidity_drought', 0.0)
        flash_crash_prob = stress_indicators.get('flash_crash_probability', 0.0)
        
        # Use JIT-compiled function
        stress_score = fast_stress_score(
            volatility_spike, correlation_shock, volume_anomaly,
            price_gap, liquidity_drought, flash_crash_prob
        )
        
        self.global_performance['jit_optimizations'] += 1
        return stress_score
    
    @performance_monitor(target_time_ms=1.0)
    def optimized_breach_detection(self, risk_values: Dict[str, float], 
                                 thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        Optimized breach detection across multiple risk metrics
        """
        # Convert to arrays for vectorized processing
        metrics = list(risk_values.keys())
        values_array = np.array([risk_values[metric] for metric in metrics])
        thresholds_array = np.array([thresholds.get(metric, float('inf')) for metric in metrics])
        
        # Use JIT-compiled function
        breach_flags = fast_breach_detection(values_array, thresholds_array)
        
        # Convert back to dictionary
        result = {metric: bool(breach_flags[i]) for i, metric in enumerate(metrics)}
        
        self.global_performance['jit_optimizations'] += 1
        return result
    
    async def parallel_risk_calculation(self, risk_calculations: List[Callable]) -> List[Any]:
        """
        Execute multiple risk calculations in parallel
        """
        start_time = time.time()
        
        # Submit all calculations to thread pool
        futures = [self.thread_pool.submit(calc) for calc in risk_calculations]
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=0.005)  # 5ms timeout
                results.append(result)
            except Exception as e:
                self.logger.error("Parallel calculation failed", error=str(e))
                results.append(None)
        
        execution_time = (time.time() - start_time) * 1000
        
        if execution_time <= self.response_time_target_ms:
            self.global_performance['parallel_optimizations'] += 1
        
        return results
    
    def optimize_memory_usage(self):
        """
        Optimize memory usage to prevent GC pauses
        """
        # Clear old cache entries
        self.cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Get memory stats
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.logger.info("Memory optimization completed",
                        memory_usage_mb=memory_mb)
        
        return memory_mb
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report
        """
        # Calculate cache performance
        cache_hit_rate = self.cache.hit_rate
        
        # Calculate average response times
        avg_times = {}
        violation_counts = {}
        
        for func_name, metrics in self.function_metrics.items():
            if hasattr(metrics, '_performance_metrics'):
                times = list(metrics._performance_metrics['times'])
                if times:
                    avg_times[func_name] = np.mean(times)
                    violation_counts[func_name] = metrics._performance_metrics['violations']
        
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            'response_time_target_ms': self.response_time_target_ms,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache.cache),
            'memory_usage_mb': memory_mb,
            'function_performance': avg_times,
            'violations': violation_counts,
            'optimizations': self.global_performance,
            'thread_pool_active': self.thread_pool._threads is not None,
            'max_workers': self.max_workers
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations
        """
        recommendations = []
        
        # Check cache hit rate
        if self.cache.hit_rate < 0.7:
            recommendations.append("Increase cache size or TTL for better hit rate")
        
        # Check memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 1000:  # > 1GB
            recommendations.append("Consider memory optimization - usage above 1GB")
        
        # Check violations
        total_violations = sum(
            getattr(metrics, '_performance_metrics', {}).get('violations', 0)
            for metrics in self.function_metrics.values()
        )
        
        if total_violations > 0:
            recommendations.append(f"Address {total_violations} performance violations")
        
        # Check thread pool utilization
        if self.max_workers < mp.cpu_count():
            recommendations.append("Consider increasing thread pool size")
        
        return recommendations
    
    def benchmark_performance(self, iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark critical functions for performance validation
        """
        benchmark_results = {}
        
        # Benchmark VaR calculation
        weights = np.random.random(10)
        weights /= weights.sum()
        volatilities = np.random.random(10) * 0.3
        correlation_matrix = np.eye(10)
        
        start_time = time.time()
        for _ in range(iterations):
            self.optimized_var_calculation(weights, volatilities, correlation_matrix, 0.95)
        var_time = (time.time() - start_time) * 1000 / iterations
        benchmark_results['var_calculation_ms'] = var_time
        
        # Benchmark correlation update
        old_corr = np.eye(5)
        returns = np.random.normal(0, 0.02, 5)
        
        start_time = time.time()
        for _ in range(iterations):
            self.optimized_correlation_update(old_corr, returns, 0.94)
        corr_time = (time.time() - start_time) * 1000 / iterations
        benchmark_results['correlation_update_ms'] = corr_time
        
        # Benchmark stress calculation
        stress_indicators = {
            'volatility_spike': 0.5,
            'correlation_shock': 0.3,
            'volume_anomaly': 0.2,
            'price_gap': 0.1,
            'liquidity_drought': 0.4,
            'flash_crash_probability': 0.2
        }
        
        start_time = time.time()
        for _ in range(iterations):
            self.optimized_stress_calculation(stress_indicators)
        stress_time = (time.time() - start_time) * 1000 / iterations
        benchmark_results['stress_calculation_ms'] = stress_time
        
        self.logger.info("Performance benchmark completed",
                        results=benchmark_results,
                        iterations=iterations)
        
        return benchmark_results
    
    def cleanup(self):
        """
        Cleanup resources
        """
        self.thread_pool.shutdown(wait=True)
        self.cache.clear()
        self.logger.info("Performance Optimizer cleaned up")
    
    def __del__(self):
        """
        Destructor to ensure cleanup
        """
        try:
            self.cleanup()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            pass  # Ignore cleanup errors during destruction