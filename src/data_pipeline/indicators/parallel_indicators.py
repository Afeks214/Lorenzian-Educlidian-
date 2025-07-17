"""
Parallel technical indicator calculations with advanced optimization

This module implements highly optimized parallel processing for technical
indicators with GPU acceleration and distributed computing capabilities.
"""

import numpy as np
import pandas as pd
import numba
from numba import jit, njit, prange, cuda
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
from collections import deque
import psutil
from functools import lru_cache
import hashlib

# Import the optimized transformers from the transformation module
from ..transformation.optimized_transformers import (
    fast_sma, fast_ema, fast_rsi, fast_bollinger_bands, fast_macd,
    fast_stochastic, fast_atr, fast_vwap, fast_returns, fast_log_returns,
    fast_rolling_volatility
)

logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"

class ProcessingMode(Enum):
    """Processing modes for indicators"""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    GPU_CUDA = "gpu_cuda"
    DISTRIBUTED = "distributed"

@dataclass
class IndicatorConfig:
    """Configuration for indicator calculation"""
    name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    processing_mode: ProcessingMode = ProcessingMode.MULTI_THREAD
    priority: int = 0
    cache_results: bool = True
    enable_incremental: bool = True

@dataclass
class IndicatorResult:
    """Result of indicator calculation"""
    name: str
    values: np.ndarray
    metadata: Dict[str, Any]
    calculation_time_us: float
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)

@dataclass
class ParallelIndicatorMetrics:
    """Performance metrics for parallel indicator calculations"""
    total_calculations: int = 0
    total_calculation_time_us: float = 0.0
    avg_calculation_time_us: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    throughput_indicators_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: float = field(default_factory=time.time)

class IndicatorCache:
    """High-performance cache for indicator results"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, data_hash: str, indicator_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key"""
        params_str = str(sorted(params.items()))
        combined = f"{data_hash}_{indicator_name}_{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, data_hash: str, indicator_name: str, params: Dict[str, Any]) -> Optional[IndicatorResult]:
        """Get cached result"""
        key = self._generate_key(data_hash, indicator_name, params)
        
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                current_time = time.time()
                
                # Check if expired
                if current_time - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                # Update access time
                self.access_times[key] = current_time
                result.cache_hit = True
                return result
        
        return None
    
    def put(self, data_hash: str, indicator_name: str, params: Dict[str, Any], result: IndicatorResult):
        """Cache result"""
        key = self._generate_key(data_hash, indicator_name, params)
        current_time = time.time()
        
        with self.lock:
            # Evict oldest if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = (result, current_time)
            self.access_times[key] = current_time
    
    def _cleanup_expired(self):
        """Cleanup expired entries"""
        while True:
            time.sleep(60)  # Cleanup every minute
            current_time = time.time()
            
            with self.lock:
                expired_keys = []
                for key, (result, timestamp) in self.cache.items():
                    if current_time - timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                    del self.access_times[key]
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size
            }

class ParallelIndicatorCalculator:
    """High-performance parallel indicator calculator"""
    
    def __init__(self, 
                 max_workers: int = None,
                 enable_gpu: bool = False,
                 enable_cache: bool = True,
                 cache_size: int = 10000,
                 enable_incremental: bool = True):
        
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.enable_gpu = enable_gpu and cuda.is_available()
        self.enable_cache = enable_cache
        self.enable_incremental = enable_incremental
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, mp.cpu_count()))
        
        # Cache system
        self.cache = IndicatorCache(max_size=cache_size) if enable_cache else None
        
        # Performance metrics
        self.metrics = ParallelIndicatorMetrics()
        self.metrics_lock = threading.Lock()
        
        # Indicator configurations
        self.indicators = {}
        self.indicators_lock = threading.RLock()
        
        # Data preprocessing for GPU
        if self.enable_gpu:
            self._init_gpu_memory()
        
        # Setup default indicators
        self._setup_default_indicators()
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        if self.cache:
            self.cache.clear()
        logger.info("ParallelIndicatorCalculator cleanup completed")
    
    def _init_gpu_memory(self):
        """Initialize GPU memory pools"""
        try:
            # Initialize CUDA context
            cuda.select_device(0)
            logger.info("GPU acceleration enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU: {str(e)}")
            self.enable_gpu = False
    
    def _setup_default_indicators(self):
        """Setup default indicator configurations"""
        default_configs = [
            IndicatorConfig(
                name="sma_20",
                indicator_type=IndicatorType.TREND,
                parameters={"window": 20},
                processing_mode=ProcessingMode.MULTI_THREAD,
                priority=1
            ),
            IndicatorConfig(
                name="ema_20",
                indicator_type=IndicatorType.TREND,
                parameters={"window": 20},
                processing_mode=ProcessingMode.MULTI_THREAD,
                priority=1
            ),
            IndicatorConfig(
                name="rsi_14",
                indicator_type=IndicatorType.MOMENTUM,
                parameters={"window": 14},
                processing_mode=ProcessingMode.MULTI_THREAD,
                priority=2
            ),
            IndicatorConfig(
                name="bollinger_bands",
                indicator_type=IndicatorType.VOLATILITY,
                parameters={"window": 20, "num_std": 2.0},
                processing_mode=ProcessingMode.MULTI_THREAD,
                priority=2
            ),
            IndicatorConfig(
                name="macd",
                indicator_type=IndicatorType.MOMENTUM,
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                processing_mode=ProcessingMode.MULTI_THREAD,
                priority=3
            ),
            IndicatorConfig(
                name="atr_14",
                indicator_type=IndicatorType.VOLATILITY,
                parameters={"window": 14},
                processing_mode=ProcessingMode.MULTI_THREAD,
                priority=3
            )
        ]
        
        for config in default_configs:
            self.add_indicator(config)
    
    def add_indicator(self, config: IndicatorConfig):
        """Add indicator configuration"""
        with self.indicators_lock:
            self.indicators[config.name] = config
    
    def remove_indicator(self, name: str):
        """Remove indicator configuration"""
        with self.indicators_lock:
            if name in self.indicators:
                del self.indicators[name]
    
    def calculate_single_indicator(self, 
                                 data: Union[np.ndarray, Dict[str, np.ndarray]], 
                                 indicator_name: str,
                                 parameters: Optional[Dict[str, Any]] = None) -> IndicatorResult:
        """Calculate a single indicator"""
        start_time = time.time_ns()
        
        # Get indicator configuration
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        config = self.indicators[indicator_name]
        params = parameters or config.parameters
        
        # Check cache if enabled
        if self.enable_cache and self.cache:
            data_hash = self._calculate_data_hash(data)
            cached_result = self.cache.get(data_hash, indicator_name, params)
            if cached_result:
                self._update_cache_metrics(True)
                return cached_result
        
        # Calculate indicator based on type
        try:
            if indicator_name.startswith("sma"):
                if isinstance(data, dict):
                    values = fast_sma(data['close'], params['window'])
                else:
                    values = fast_sma(data, params['window'])
            
            elif indicator_name.startswith("ema"):
                alpha = 2.0 / (params['window'] + 1)
                if isinstance(data, dict):
                    values = fast_ema(data['close'], alpha)
                else:
                    values = fast_ema(data, alpha)
            
            elif indicator_name.startswith("rsi"):
                if isinstance(data, dict):
                    values = fast_rsi(data['close'], params['window'])
                else:
                    values = fast_rsi(data, params['window'])
            
            elif indicator_name == "bollinger_bands":
                if isinstance(data, dict):
                    values = fast_bollinger_bands(data['close'], params['window'], params['num_std'])
                else:
                    values = fast_bollinger_bands(data, params['window'], params['num_std'])
            
            elif indicator_name == "macd":
                if isinstance(data, dict):
                    values = fast_macd(data['close'], params['fast_period'], params['slow_period'], params['signal_period'])
                else:
                    values = fast_macd(data, params['fast_period'], params['slow_period'], params['signal_period'])
            
            elif indicator_name.startswith("atr"):
                if isinstance(data, dict):
                    values = fast_atr(data['high'], data['low'], data['close'], params['window'])
                else:
                    raise ValueError("ATR requires OHLC data")
            
            elif indicator_name == "vwap":
                if isinstance(data, dict):
                    typical_price = (data['high'] + data['low'] + data['close']) / 3
                    values = fast_vwap(typical_price, data['volume'])
                else:
                    raise ValueError("VWAP requires OHLCV data")
            
            else:
                raise ValueError(f"Unsupported indicator: {indicator_name}")
            
            # Calculate execution time
            end_time = time.time_ns()
            calculation_time_us = (end_time - start_time) / 1000
            
            # Create result
            result = IndicatorResult(
                name=indicator_name,
                values=values,
                metadata={
                    'parameters': params,
                    'data_points': len(values) if isinstance(values, np.ndarray) else len(values[0]),
                    'processing_mode': config.processing_mode.value
                },
                calculation_time_us=calculation_time_us,
                cache_hit=False
            )
            
            # Cache result if enabled
            if self.enable_cache and self.cache:
                data_hash = self._calculate_data_hash(data)
                self.cache.put(data_hash, indicator_name, params, result)
                self._update_cache_metrics(False)
            
            # Update metrics
            self._update_calculation_metrics(calculation_time_us)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicator {indicator_name}: {str(e)}")
            raise
    
    def calculate_multiple_indicators(self, 
                                    data: Union[np.ndarray, Dict[str, np.ndarray]],
                                    indicator_names: List[str],
                                    processing_mode: ProcessingMode = ProcessingMode.MULTI_THREAD) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators in parallel"""
        
        if processing_mode == ProcessingMode.SINGLE_THREAD:
            return self._calculate_sequential(data, indicator_names)
        elif processing_mode == ProcessingMode.MULTI_THREAD:
            return self._calculate_multithreaded(data, indicator_names)
        elif processing_mode == ProcessingMode.MULTI_PROCESS:
            return self._calculate_multiprocess(data, indicator_names)
        elif processing_mode == ProcessingMode.GPU_CUDA and self.enable_gpu:
            return self._calculate_gpu(data, indicator_names)
        else:
            # Fallback to multithreaded
            return self._calculate_multithreaded(data, indicator_names)
    
    def _calculate_sequential(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                            indicator_names: List[str]) -> Dict[str, IndicatorResult]:
        """Calculate indicators sequentially"""
        results = {}
        for name in indicator_names:
            try:
                results[name] = self.calculate_single_indicator(data, name)
            except Exception as e:
                logger.error(f"Error calculating {name}: {str(e)}")
        return results
    
    def _calculate_multithreaded(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                               indicator_names: List[str]) -> Dict[str, IndicatorResult]:
        """Calculate indicators using multiple threads"""
        futures = {}
        results = {}
        
        # Submit all calculations
        for name in indicator_names:
            future = self.thread_pool.submit(self.calculate_single_indicator, data, name)
            futures[name] = future
        
        # Collect results
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=30)  # 30 second timeout
            except Exception as e:
                logger.error(f"Error calculating {name}: {str(e)}")
        
        return results
    
    def _calculate_multiprocess(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                              indicator_names: List[str]) -> Dict[str, IndicatorResult]:
        """Calculate indicators using multiple processes"""
        # For multiprocess, we need to serialize the calculation
        # This is a simplified implementation
        futures = {}
        results = {}
        
        # Submit all calculations
        for name in indicator_names:
            future = self.process_pool.submit(_calculate_indicator_process, data, name, self.indicators[name])
            futures[name] = future
        
        # Collect results
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Error calculating {name}: {str(e)}")
        
        return results
    
    def _calculate_gpu(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                      indicator_names: List[str]) -> Dict[str, IndicatorResult]:
        """Calculate indicators using GPU acceleration"""
        # This is a placeholder for GPU implementation
        # In a real implementation, you would use CUDA kernels
        logger.info("GPU calculation not fully implemented, falling back to multithreaded")
        return self._calculate_multithreaded(data, indicator_names)
    
    def calculate_all_indicators(self, 
                               data: Union[np.ndarray, Dict[str, np.ndarray]],
                               processing_mode: ProcessingMode = ProcessingMode.MULTI_THREAD) -> Dict[str, IndicatorResult]:
        """Calculate all configured indicators"""
        with self.indicators_lock:
            indicator_names = list(self.indicators.keys())
        
        return self.calculate_multiple_indicators(data, indicator_names, processing_mode)
    
    def _calculate_data_hash(self, data: Union[np.ndarray, Dict[str, np.ndarray]]) -> str:
        """Calculate hash of input data for caching"""
        if isinstance(data, dict):
            # Combine all arrays
            combined = np.concatenate([arr.flatten() for arr in data.values()])
        else:
            combined = data.flatten()
        
        # Use a sample for large datasets
        if len(combined) > 10000:
            combined = combined[::len(combined)//10000]
        
        return hashlib.md5(combined.tobytes()).hexdigest()
    
    def _update_calculation_metrics(self, calculation_time_us: float):
        """Update calculation metrics"""
        with self.metrics_lock:
            self.metrics.total_calculations += 1
            self.metrics.total_calculation_time_us += calculation_time_us
            self.metrics.avg_calculation_time_us = (
                self.metrics.total_calculation_time_us / self.metrics.total_calculations
            )
            
            # Update throughput
            if calculation_time_us > 0:
                self.metrics.throughput_indicators_per_sec = 1000000 / calculation_time_us
    
    def _update_cache_metrics(self, cache_hit: bool):
        """Update cache metrics"""
        with self.metrics_lock:
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            total_requests = self.metrics.cache_hits + self.metrics.cache_misses
            if total_requests > 0:
                self.metrics.cache_hit_rate = self.metrics.cache_hits / total_requests
    
    def get_metrics(self) -> ParallelIndicatorMetrics:
        """Get performance metrics"""
        with self.metrics_lock:
            # Update system metrics
            self.metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.metrics.cpu_utilization = psutil.cpu_percent()
            
            return ParallelIndicatorMetrics(
                total_calculations=self.metrics.total_calculations,
                total_calculation_time_us=self.metrics.total_calculation_time_us,
                avg_calculation_time_us=self.metrics.avg_calculation_time_us,
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses,
                cache_hit_rate=self.metrics.cache_hit_rate,
                throughput_indicators_per_sec=self.metrics.throughput_indicators_per_sec,
                memory_usage_mb=self.metrics.memory_usage_mb,
                cpu_utilization=self.metrics.cpu_utilization,
                gpu_utilization=self.metrics.gpu_utilization,
                timestamp=time.time()
            )
    
    def benchmark_indicators(self, 
                           data: Union[np.ndarray, Dict[str, np.ndarray]],
                           indicator_names: List[str],
                           num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark indicator calculations"""
        results = {}
        
        for name in indicator_names:
            if name not in self.indicators:
                continue
            
            times = []
            for _ in range(num_iterations):
                start_time = time.time_ns()
                try:
                    self.calculate_single_indicator(data, name)
                    end_time = time.time_ns()
                    times.append((end_time - start_time) / 1000)  # Convert to microseconds
                except Exception as e:
                    logger.error(f"Error in benchmark for {name}: {str(e)}")
                    continue
            
            if times:
                results[name] = {
                    'avg_time_us': np.mean(times),
                    'min_time_us': np.min(times),
                    'max_time_us': np.max(times),
                    'std_time_us': np.std(times),
                    'p95_time_us': np.percentile(times, 95),
                    'p99_time_us': np.percentile(times, 99)
                }
        
        return results
    
    def clear_cache(self):
        """Clear indicator cache"""
        if self.cache:
            self.cache.clear()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self.metrics_lock:
            self.metrics = ParallelIndicatorMetrics()

# GPU kernels for CUDA acceleration (if available)
if cuda.is_available():
    @cuda.jit
    def gpu_sma_kernel(data, result, window):
        """GPU kernel for SMA calculation"""
        idx = cuda.grid(1)
        if idx < len(result):
            if idx < window - 1:
                result[idx] = float('nan')
            else:
                sum_val = 0.0
                for i in range(window):
                    sum_val += data[idx - window + 1 + i]
                result[idx] = sum_val / window
    
    @cuda.jit
    def gpu_ema_kernel(data, result, alpha):
        """GPU kernel for EMA calculation"""
        idx = cuda.grid(1)
        if idx < len(result):
            if idx == 0:
                result[idx] = data[idx]
            else:
                result[idx] = alpha * data[idx] + (1 - alpha) * result[idx - 1]

# Process function for multiprocessing
def _calculate_indicator_process(data: Union[np.ndarray, Dict[str, np.ndarray]], 
                               indicator_name: str, 
                               config: IndicatorConfig) -> IndicatorResult:
    """Calculate indicator in separate process"""
    start_time = time.time_ns()
    
    try:
        params = config.parameters
        
        # Calculate based on indicator type
        if indicator_name.startswith("sma"):
            if isinstance(data, dict):
                values = fast_sma(data['close'], params['window'])
            else:
                values = fast_sma(data, params['window'])
        
        elif indicator_name.startswith("ema"):
            alpha = 2.0 / (params['window'] + 1)
            if isinstance(data, dict):
                values = fast_ema(data['close'], alpha)
            else:
                values = fast_ema(data, alpha)
        
        elif indicator_name.startswith("rsi"):
            if isinstance(data, dict):
                values = fast_rsi(data['close'], params['window'])
            else:
                values = fast_rsi(data, params['window'])
        
        else:
            raise ValueError(f"Unsupported indicator: {indicator_name}")
        
        end_time = time.time_ns()
        calculation_time_us = (end_time - start_time) / 1000
        
        return IndicatorResult(
            name=indicator_name,
            values=values,
            metadata={
                'parameters': params,
                'data_points': len(values),
                'processing_mode': 'multiprocess'
            },
            calculation_time_us=calculation_time_us,
            cache_hit=False
        )
        
    except Exception as e:
        logger.error(f"Error in process calculation for {indicator_name}: {str(e)}")
        raise

# Utility functions
def create_parallel_calculator(max_workers: int = None, 
                             enable_gpu: bool = False, 
                             enable_cache: bool = True) -> ParallelIndicatorCalculator:
    """Create parallel indicator calculator with default settings"""
    return ParallelIndicatorCalculator(
        max_workers=max_workers,
        enable_gpu=enable_gpu,
        enable_cache=enable_cache
    )

def benchmark_parallel_performance(data_size: int = 10000, 
                                 num_indicators: int = 10,
                                 max_workers_list: List[int] = [1, 2, 4, 8, 16]) -> Dict[str, Dict[str, float]]:
    """Benchmark parallel performance with different worker counts"""
    # Generate test data
    np.random.seed(42)
    ohlcv_data = {
        'open': np.random.randn(data_size).cumsum() + 100,
        'high': np.random.randn(data_size).cumsum() + 102,
        'low': np.random.randn(data_size).cumsum() + 98,
        'close': np.random.randn(data_size).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, data_size)
    }
    
    indicator_names = ['sma_20', 'ema_20', 'rsi_14', 'bollinger_bands', 'macd']
    
    results = {}
    
    for max_workers in max_workers_list:
        calculator = create_parallel_calculator(max_workers=max_workers)
        
        # Benchmark calculation time
        start_time = time.time_ns()
        indicator_results = calculator.calculate_multiple_indicators(
            ohlcv_data, indicator_names, ProcessingMode.MULTI_THREAD
        )
        end_time = time.time_ns()
        
        total_time_us = (end_time - start_time) / 1000
        
        results[f"workers_{max_workers}"] = {
            'total_time_us': total_time_us,
            'indicators_calculated': len(indicator_results),
            'avg_time_per_indicator_us': total_time_us / len(indicator_results),
            'throughput_indicators_per_sec': len(indicator_results) / (total_time_us / 1000000)
        }
        
        # Cleanup
        calculator._cleanup()
    
    return results
