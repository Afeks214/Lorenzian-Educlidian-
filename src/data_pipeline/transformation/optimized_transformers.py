"""
Optimized data transformation pipelines for sub-millisecond latency

This module implements highly optimized data transformation pipelines
designed for high-frequency market data processing with sub-millisecond
latency requirements.
"""

import numpy as np
import pandas as pd
import numba
from numba import jit, njit, types
from numba.typed import Dict, List as NumbaList
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
import struct
import mmap
from pathlib import Path

logger = logging.getLogger(__name__)

# Compile-time constants for maximum performance
MAX_WINDOW_SIZE = 1000
MAX_INDICATORS = 50
NAN_VALUE = np.nan

@dataclass
class TransformationMetrics:
    """Performance metrics for transformations"""
    total_transformations: int = 0
    avg_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    throughput_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

@njit(cache=True, fastmath=True)
def fast_sma(data: np.ndarray, window: int) -> np.ndarray:
    """Optimized Simple Moving Average using Numba"""
    n = len(data)
    if n < window:
        return np.full(n, np.nan)
    
    result = np.empty(n)
    result[:window-1] = np.nan
    
    # Calculate first window sum
    window_sum = 0.0
    for i in range(window):
        window_sum += data[i]
    
    result[window-1] = window_sum / window
    
    # Sliding window calculation
    for i in range(window, n):
        window_sum = window_sum - data[i-window] + data[i]
        result[i] = window_sum / window
    
    return result

@njit(cache=True, fastmath=True)
def fast_ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Optimized Exponential Moving Average using Numba"""
    n = len(data)
    if n == 0:
        return np.array([])
    
    result = np.empty(n)
    result[0] = data[0]
    
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result

@njit(cache=True, fastmath=True)
def fast_rsi(data: np.ndarray, window: int = 14) -> np.ndarray:
    """Optimized RSI calculation using Numba"""
    n = len(data)
    if n < window + 1:
        return np.full(n, np.nan)
    
    # Calculate price changes
    changes = np.empty(n-1)
    for i in range(1, n):
        changes[i-1] = data[i] - data[i-1]
    
    # Separate gains and losses
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    result = np.full(n, np.nan)
    
    # Calculate RSI
    alpha = 1.0 / window
    for i in range(window, n-1):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            result[i+1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i+1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

@njit(cache=True, fastmath=True)
def fast_bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Bollinger Bands calculation using Numba"""
    n = len(data)
    if n < window:
        empty = np.full(n, np.nan)
        return empty, empty, empty
    
    # Calculate SMA
    sma = fast_sma(data, window)
    
    # Calculate rolling standard deviation
    std = np.full(n, np.nan)
    for i in range(window-1, n):
        window_data = data[i-window+1:i+1]
        mean_val = sma[i]
        variance = 0.0
        for j in range(window):
            diff = window_data[j] - mean_val
            variance += diff * diff
        std[i] = np.sqrt(variance / window)
    
    # Calculate bands
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    
    return sma, upper_band, lower_band

@njit(cache=True, fastmath=True)
def fast_macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized MACD calculation using Numba"""
    n = len(data)
    if n < slow_period:
        empty = np.full(n, np.nan)
        return empty, empty, empty
    
    # Calculate EMAs
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)
    
    fast_ema = fast_ema(data, fast_alpha)
    slow_ema = fast_ema(data, slow_alpha)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = fast_ema(macd_line, signal_alpha)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

@njit(cache=True, fastmath=True)
def fast_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Stochastic oscillator calculation using Numba"""
    n = len(close)
    if n < k_period:
        empty = np.full(n, np.nan)
        return empty, empty
    
    k_percent = np.full(n, np.nan)
    
    # Calculate %K
    for i in range(k_period-1, n):
        window_high = np.max(high[i-k_period+1:i+1])
        window_low = np.min(low[i-k_period+1:i+1])
        
        if window_high == window_low:
            k_percent[i] = 50.0
        else:
            k_percent[i] = 100.0 * (close[i] - window_low) / (window_high - window_low)
    
    # Calculate %D (SMA of %K)
    d_percent = fast_sma(k_percent, d_period)
    
    return k_percent, d_percent

@njit(cache=True, fastmath=True)
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Optimized Average True Range calculation using Numba"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    # Calculate true range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR using RMA (Wilder's smoothing)
    atr = np.full(n, np.nan)
    if n >= window:
        atr[window-1] = np.mean(tr[:window])
        
        alpha = 1.0 / window
        for i in range(window, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    return atr

@njit(cache=True, fastmath=True)
def fast_vwap(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Optimized Volume Weighted Average Price calculation using Numba"""
    n = len(price)
    if n != len(volume):
        return np.full(n, np.nan)
    
    vwap = np.empty(n)
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    
    for i in range(n):
        cumulative_pv += price[i] * volume[i]
        cumulative_volume += volume[i]
        
        if cumulative_volume == 0:
            vwap[i] = price[i]
        else:
            vwap[i] = cumulative_pv / cumulative_volume
    
    return vwap

@njit(cache=True, fastmath=True)
def fast_returns(data: np.ndarray, periods: int = 1) -> np.ndarray:
    """Optimized returns calculation using Numba"""
    n = len(data)
    if n <= periods:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan)
    
    for i in range(periods, n):
        if data[i-periods] != 0:
            result[i] = (data[i] - data[i-periods]) / data[i-periods]
    
    return result

@njit(cache=True, fastmath=True)
def fast_log_returns(data: np.ndarray, periods: int = 1) -> np.ndarray:
    """Optimized log returns calculation using Numba"""
    n = len(data)
    if n <= periods:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan)
    
    for i in range(periods, n):
        if data[i-periods] > 0 and data[i] > 0:
            result[i] = np.log(data[i] / data[i-periods])
    
    return result

@njit(cache=True, fastmath=True)
def fast_rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Optimized rolling volatility calculation using Numba"""
    n = len(returns)
    if n < window:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan)
    
    for i in range(window-1, n):
        window_data = returns[i-window+1:i+1]
        
        # Calculate mean
        mean_val = 0.0
        valid_count = 0
        for j in range(window):
            if not np.isnan(window_data[j]):
                mean_val += window_data[j]
                valid_count += 1
        
        if valid_count > 0:
            mean_val /= valid_count
            
            # Calculate variance
            variance = 0.0
            for j in range(window):
                if not np.isnan(window_data[j]):
                    diff = window_data[j] - mean_val
                    variance += diff * diff
            
            if valid_count > 1:
                variance /= (valid_count - 1)
                result[i] = np.sqrt(variance)
    
    return result

class OptimizedTransformer:
    """High-performance data transformer with sub-millisecond latency"""
    
    def __init__(self, enable_parallel: bool = True, max_workers: int = 4):
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Performance metrics
        self.metrics = TransformationMetrics()
        self.latency_samples = deque(maxlen=10000)
        self.metrics_lock = threading.Lock()
        
        # Transformation cache
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Thread pool for parallel processing
        if enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None
        
        # Pre-compiled transformation functions
        self.compiled_functions = {
            'sma': fast_sma,
            'ema': fast_ema,
            'rsi': fast_rsi,
            'bollinger_bands': fast_bollinger_bands,
            'macd': fast_macd,
            'stochastic': fast_stochastic,
            'atr': fast_atr,
            'vwap': fast_vwap,
            'returns': fast_returns,
            'log_returns': fast_log_returns,
            'rolling_volatility': fast_rolling_volatility
        }
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("OptimizedTransformer cleanup completed")
    
    def transform_single(self, data: np.ndarray, transformation: str, **kwargs) -> np.ndarray:
        """Transform single data array"""
        start_time = time.time_ns()
        
        try:
            if transformation not in self.compiled_functions:
                raise ValueError(f"Unknown transformation: {transformation}")
            
            func = self.compiled_functions[transformation]
            result = func(data, **kwargs)
            
            # Record latency
            end_time = time.time_ns()
            latency_us = (end_time - start_time) / 1000
            self._record_latency(latency_us)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transformation {transformation}: {str(e)}")
            raise
    
    def transform_ohlcv(self, ohlcv_data: Dict[str, np.ndarray], transformations: List[str], **kwargs) -> Dict[str, np.ndarray]:
        """Transform OHLCV data with multiple indicators"""
        start_time = time.time_ns()
        results = {}
        
        try:
            # Extract arrays
            open_data = ohlcv_data.get('open', np.array([]))
            high_data = ohlcv_data.get('high', np.array([]))
            low_data = ohlcv_data.get('low', np.array([]))
            close_data = ohlcv_data.get('close', np.array([]))
            volume_data = ohlcv_data.get('volume', np.array([]))
            
            # Apply transformations
            for transformation in transformations:
                if transformation == 'sma':
                    window = kwargs.get('sma_window', 20)
                    results[f'sma_{window}'] = fast_sma(close_data, window)
                
                elif transformation == 'ema':
                    window = kwargs.get('ema_window', 20)
                    alpha = 2.0 / (window + 1)
                    results[f'ema_{window}'] = fast_ema(close_data, alpha)
                
                elif transformation == 'rsi':
                    window = kwargs.get('rsi_window', 14)
                    results[f'rsi_{window}'] = fast_rsi(close_data, window)
                
                elif transformation == 'bollinger_bands':
                    window = kwargs.get('bb_window', 20)
                    num_std = kwargs.get('bb_std', 2.0)
                    bb_middle, bb_upper, bb_lower = fast_bollinger_bands(close_data, window, num_std)
                    results[f'bb_middle_{window}'] = bb_middle
                    results[f'bb_upper_{window}'] = bb_upper
                    results[f'bb_lower_{window}'] = bb_lower
                
                elif transformation == 'macd':
                    fast_period = kwargs.get('macd_fast', 12)
                    slow_period = kwargs.get('macd_slow', 26)
                    signal_period = kwargs.get('macd_signal', 9)
                    macd_line, signal_line, histogram = fast_macd(close_data, fast_period, slow_period, signal_period)
                    results['macd_line'] = macd_line
                    results['macd_signal'] = signal_line
                    results['macd_histogram'] = histogram
                
                elif transformation == 'stochastic':
                    k_period = kwargs.get('stoch_k', 14)
                    d_period = kwargs.get('stoch_d', 3)
                    k_percent, d_percent = fast_stochastic(high_data, low_data, close_data, k_period, d_period)
                    results[f'stoch_k_{k_period}'] = k_percent
                    results[f'stoch_d_{d_period}'] = d_percent
                
                elif transformation == 'atr':
                    window = kwargs.get('atr_window', 14)
                    results[f'atr_{window}'] = fast_atr(high_data, low_data, close_data, window)
                
                elif transformation == 'vwap':
                    if len(volume_data) > 0:
                        typical_price = (high_data + low_data + close_data) / 3
                        results['vwap'] = fast_vwap(typical_price, volume_data)
                
                elif transformation == 'returns':
                    periods = kwargs.get('return_periods', 1)
                    results[f'returns_{periods}'] = fast_returns(close_data, periods)
                
                elif transformation == 'log_returns':
                    periods = kwargs.get('return_periods', 1)
                    results[f'log_returns_{periods}'] = fast_log_returns(close_data, periods)
                
                elif transformation == 'rolling_volatility':
                    window = kwargs.get('vol_window', 20)
                    returns = fast_returns(close_data, 1)
                    results[f'volatility_{window}'] = fast_rolling_volatility(returns, window)
            
            # Record latency
            end_time = time.time_ns()
            latency_us = (end_time - start_time) / 1000
            self._record_latency(latency_us)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in OHLCV transformation: {str(e)}")
            raise
    
    def transform_dataframe(self, df: pd.DataFrame, transformations: List[str], **kwargs) -> pd.DataFrame:
        """Transform DataFrame with indicators"""
        start_time = time.time_ns()
        
        try:
            # Convert to numpy arrays for performance
            ohlcv_data = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    ohlcv_data[col] = df[col].values
            
            # Apply transformations
            results = self.transform_ohlcv(ohlcv_data, transformations, **kwargs)
            
            # Convert back to DataFrame
            result_df = df.copy()
            for name, values in results.items():
                result_df[name] = values
            
            # Record latency
            end_time = time.time_ns()
            latency_us = (end_time - start_time) / 1000
            self._record_latency(latency_us)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in DataFrame transformation: {str(e)}")
            raise
    
    def transform_batch(self, batch_data: List[Dict[str, np.ndarray]], transformations: List[str], **kwargs) -> List[Dict[str, np.ndarray]]:
        """Transform batch of data with parallel processing"""
        if not self.enable_parallel or len(batch_data) < 4:
            # Sequential processing for small batches
            return [self.transform_ohlcv(data, transformations, **kwargs) for data in batch_data]
        
        # Parallel processing for large batches
        futures = [self.executor.submit(self.transform_ohlcv, data, transformations, **kwargs) for data in batch_data]
        return [future.result() for future in futures]
    
    def _record_latency(self, latency_us: float):
        """Record latency measurement"""
        with self.metrics_lock:
            self.latency_samples.append(latency_us)
            self.metrics.total_transformations += 1
    
    def get_metrics(self) -> TransformationMetrics:
        """Get performance metrics"""
        with self.metrics_lock:
            if len(self.latency_samples) > 0:
                latencies = np.array(self.latency_samples)
                self.metrics.avg_latency_us = np.mean(latencies)
                self.metrics.p95_latency_us = np.percentile(latencies, 95)
                self.metrics.p99_latency_us = np.percentile(latencies, 99)
                
                # Calculate throughput
                if len(self.latency_samples) > 1:
                    time_window = (self.latency_samples[-1] - self.latency_samples[0]) / 1000000  # Convert to seconds
                    if time_window > 0:
                        self.metrics.throughput_per_sec = len(self.latency_samples) / time_window
            
            return TransformationMetrics(
                total_transformations=self.metrics.total_transformations,
                avg_latency_us=self.metrics.avg_latency_us,
                p95_latency_us=self.metrics.p95_latency_us,
                p99_latency_us=self.metrics.p99_latency_us,
                throughput_per_sec=self.metrics.throughput_per_sec,
                memory_usage_mb=self.metrics.memory_usage_mb,
                timestamp=time.time()
            )
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self.metrics_lock:
            self.metrics = TransformationMetrics()
            self.latency_samples.clear()

class StreamingTransformer:
    """Streaming transformer for real-time data processing"""
    
    def __init__(self, buffer_size: int = 1000, enable_incremental: bool = True):
        self.buffer_size = buffer_size
        self.enable_incremental = enable_incremental
        
        # Data buffers
        self.price_buffer = deque(maxlen=buffer_size)
        self.volume_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Indicator state for incremental updates
        self.indicator_state = {}
        
        # Optimizer
        self.optimizer = OptimizedTransformer(enable_parallel=False)  # Single-threaded for streaming
        
        # Metrics
        self.update_count = 0
        self.total_latency_us = 0.0
    
    def add_tick(self, price: float, volume: int, timestamp: float) -> Dict[str, float]:
        """Add new tick data and return updated indicators"""
        start_time = time.time_ns()
        
        # Add to buffers
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        self.timestamp_buffer.append(timestamp)
        
        # Calculate indicators
        results = {}
        
        if len(self.price_buffer) >= 20:  # Minimum window for most indicators
            price_array = np.array(self.price_buffer)
            volume_array = np.array(self.volume_buffer)
            
            # Calculate common indicators
            if len(self.price_buffer) >= 20:
                sma_20 = fast_sma(price_array, 20)
                results['sma_20'] = sma_20[-1] if not np.isnan(sma_20[-1]) else None
                
                ema_20 = fast_ema(price_array, 2.0 / (20 + 1))
                results['ema_20'] = ema_20[-1] if not np.isnan(ema_20[-1]) else None
            
            if len(self.price_buffer) >= 14:
                rsi_14 = fast_rsi(price_array, 14)
                results['rsi_14'] = rsi_14[-1] if not np.isnan(rsi_14[-1]) else None
            
            # VWAP
            if len(self.volume_buffer) == len(self.price_buffer):
                vwap = fast_vwap(price_array, volume_array)
                results['vwap'] = vwap[-1] if not np.isnan(vwap[-1]) else None
        
        # Record latency
        end_time = time.time_ns()
        latency_us = (end_time - start_time) / 1000
        self.update_count += 1
        self.total_latency_us += latency_us
        
        return results
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current values of all indicators"""
        if len(self.price_buffer) == 0:
            return {}
        
        price_array = np.array(self.price_buffer)
        volume_array = np.array(self.volume_buffer)
        
        results = {}
        
        # Calculate all available indicators
        if len(self.price_buffer) >= 20:
            sma_20 = fast_sma(price_array, 20)
            results['sma_20'] = sma_20[-1] if not np.isnan(sma_20[-1]) else None
            
            ema_20 = fast_ema(price_array, 2.0 / (20 + 1))
            results['ema_20'] = ema_20[-1] if not np.isnan(ema_20[-1]) else None
            
            bb_middle, bb_upper, bb_lower = fast_bollinger_bands(price_array, 20, 2.0)
            results['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else None
            results['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else None
            results['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else None
        
        if len(self.price_buffer) >= 14:
            rsi_14 = fast_rsi(price_array, 14)
            results['rsi_14'] = rsi_14[-1] if not np.isnan(rsi_14[-1]) else None
        
        if len(self.price_buffer) >= 26:
            macd_line, signal_line, histogram = fast_macd(price_array, 12, 26, 9)
            results['macd_line'] = macd_line[-1] if not np.isnan(macd_line[-1]) else None
            results['macd_signal'] = signal_line[-1] if not np.isnan(signal_line[-1]) else None
            results['macd_histogram'] = histogram[-1] if not np.isnan(histogram[-1]) else None
        
        # VWAP
        if len(self.volume_buffer) == len(self.price_buffer):
            vwap = fast_vwap(price_array, volume_array)
            results['vwap'] = vwap[-1] if not np.isnan(vwap[-1]) else None
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        avg_latency_us = self.total_latency_us / self.update_count if self.update_count > 0 else 0
        
        return {
            'update_count': self.update_count,
            'avg_latency_us': avg_latency_us,
            'buffer_utilization': len(self.price_buffer) / self.buffer_size,
            'throughput_per_sec': 1000000 / avg_latency_us if avg_latency_us > 0 else 0
        }
    
    def reset(self):
        """Reset all buffers and state"""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.timestamp_buffer.clear()
        self.indicator_state.clear()
        self.update_count = 0
        self.total_latency_us = 0.0

# Utility functions
def create_optimized_transformer(enable_parallel: bool = True, max_workers: int = 4) -> OptimizedTransformer:
    """Create optimized transformer with default settings"""
    return OptimizedTransformer(enable_parallel=enable_parallel, max_workers=max_workers)

def create_streaming_transformer(buffer_size: int = 1000) -> StreamingTransformer:
    """Create streaming transformer with default settings"""
    return StreamingTransformer(buffer_size=buffer_size)

def benchmark_transformations(data_size: int = 10000, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark transformation performance"""
    # Generate test data
    np.random.seed(42)
    test_data = np.random.randn(data_size).cumsum() + 100
    
    transformer = OptimizedTransformer(enable_parallel=False)
    results = {}
    
    # Benchmark each transformation
    transformations = ['sma', 'ema', 'rsi', 'bollinger_bands', 'macd']
    
    for transform_name in transformations:
        start_time = time.time_ns()
        
        for _ in range(num_iterations):
            if transform_name == 'sma':
                fast_sma(test_data, 20)
            elif transform_name == 'ema':
                fast_ema(test_data, 0.1)
            elif transform_name == 'rsi':
                fast_rsi(test_data, 14)
            elif transform_name == 'bollinger_bands':
                fast_bollinger_bands(test_data, 20, 2.0)
            elif transform_name == 'macd':
                fast_macd(test_data, 12, 26, 9)
        
        end_time = time.time_ns()
        avg_latency_us = (end_time - start_time) / (num_iterations * 1000)
        results[transform_name] = avg_latency_us
    
    return results
