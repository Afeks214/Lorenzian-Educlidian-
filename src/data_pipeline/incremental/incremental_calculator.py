"""
Incremental calculation engine for real-time data processing

This module implements incremental calculation capabilities for technical indicators
and data transformations, enabling efficient real-time updates without full recalculation.
"""

import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import numba
from numba import jit, njit
import weakref
from abc import ABC, abstractmethod
import pickle
import hashlib

logger = logging.getLogger(__name__)

class UpdateType(Enum):
    """Types of data updates"""
    APPEND = "append"  # New data point added
    UPDATE = "update"  # Existing data point modified
    DELETE = "delete"  # Data point removed
    BULK_UPDATE = "bulk_update"  # Multiple points updated

@dataclass
class IncrementalState:
    """State for incremental calculations"""
    window_size: int
    current_values: deque
    accumulated_value: float = 0.0
    count: int = 0
    last_result: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class UpdateEvent:
    """Data update event"""
    update_type: UpdateType
    new_value: Optional[float] = None
    old_value: Optional[float] = None
    index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class IncrementalMetrics:
    """Performance metrics for incremental calculations"""
    total_updates: int = 0
    total_calculation_time_us: float = 0.0
    avg_calculation_time_us: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    throughput_updates_per_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)

class IncrementalCalculator(ABC):
    """Abstract base class for incremental calculators"""
    
    def __init__(self, window_size: int, initial_capacity: int = 1000):
        self.window_size = window_size
        self.initial_capacity = initial_capacity
        self.state = IncrementalState(window_size, deque(maxlen=window_size))
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = IncrementalMetrics()
        self.metrics_lock = threading.Lock()
        
        # Calculation history for debugging
        self.calculation_history = deque(maxlen=100)
    
    @abstractmethod
    def update(self, event: UpdateEvent) -> float:
        """Update calculation with new data"""
        pass
    
    @abstractmethod
    def get_current_value(self) -> Optional[float]:
        """Get current calculated value"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset calculator state"""
        pass
    
    def _record_calculation(self, calculation_time_us: float, result: float):
        """Record calculation performance"""
        with self.metrics_lock:
            self.metrics.total_updates += 1
            self.metrics.total_calculation_time_us += calculation_time_us
            self.metrics.avg_calculation_time_us = (
                self.metrics.total_calculation_time_us / self.metrics.total_updates
            )
            
            if calculation_time_us > 0:
                self.metrics.throughput_updates_per_sec = 1000000 / calculation_time_us
        
        # Record in history
        self.calculation_history.append({
            'timestamp': time.time(),
            'calculation_time_us': calculation_time_us,
            'result': result,
            'window_size': len(self.state.current_values)
        })
    
    def get_metrics(self) -> IncrementalMetrics:
        """Get performance metrics"""
        with self.metrics_lock:
            return IncrementalMetrics(
                total_updates=self.metrics.total_updates,
                total_calculation_time_us=self.metrics.total_calculation_time_us,
                avg_calculation_time_us=self.metrics.avg_calculation_time_us,
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses,
                memory_usage_mb=self.metrics.memory_usage_mb,
                throughput_updates_per_sec=self.metrics.throughput_updates_per_sec,
                timestamp=time.time()
            )

class IncrementalSMA(IncrementalCalculator):
    """Incremental Simple Moving Average calculator"""
    
    def __init__(self, window_size: int):
        super().__init__(window_size)
        self.sum_value = 0.0
    
    def update(self, event: UpdateEvent) -> float:
        start_time = time.time_ns()
        
        with self.lock:
            if event.update_type == UpdateType.APPEND:
                # Add new value
                new_value = event.new_value
                
                # If window is full, remove oldest value
                if len(self.state.current_values) == self.window_size:
                    old_value = self.state.current_values[0]
                    self.sum_value -= old_value
                
                # Add new value
                self.state.current_values.append(new_value)
                self.sum_value += new_value
                
                # Calculate new SMA
                if len(self.state.current_values) > 0:
                    result = self.sum_value / len(self.state.current_values)
                    self.state.last_result = result
                else:
                    result = 0.0
                
                # Record performance
                end_time = time.time_ns()
                calculation_time_us = (end_time - start_time) / 1000
                self._record_calculation(calculation_time_us, result)
                
                return result
            
            elif event.update_type == UpdateType.UPDATE:
                # Update existing value
                if event.index is not None and 0 <= event.index < len(self.state.current_values):
                    old_value = self.state.current_values[event.index]
                    new_value = event.new_value
                    
                    # Update sum
                    self.sum_value = self.sum_value - old_value + new_value
                    
                    # Update value in deque
                    self.state.current_values[event.index] = new_value
                    
                    # Calculate new SMA
                    result = self.sum_value / len(self.state.current_values)
                    self.state.last_result = result
                    
                    # Record performance
                    end_time = time.time_ns()
                    calculation_time_us = (end_time - start_time) / 1000
                    self._record_calculation(calculation_time_us, result)
                    
                    return result
            
            return self.state.last_result or 0.0
    
    def get_current_value(self) -> Optional[float]:
        with self.lock:
            return self.state.last_result
    
    def reset(self):
        with self.lock:
            self.state.current_values.clear()
            self.sum_value = 0.0
            self.state.last_result = None
            self.state.count = 0

class IncrementalEMA(IncrementalCalculator):
    """Incremental Exponential Moving Average calculator"""
    
    def __init__(self, window_size: int, alpha: Optional[float] = None):
        super().__init__(window_size)
        self.alpha = alpha or (2.0 / (window_size + 1))
        self.is_initialized = False
    
    def update(self, event: UpdateEvent) -> float:
        start_time = time.time_ns()
        
        with self.lock:
            if event.update_type == UpdateType.APPEND:
                new_value = event.new_value
                
                if not self.is_initialized:
                    # First value
                    self.state.last_result = new_value
                    self.is_initialized = True
                else:
                    # EMA formula: EMA = α * new_value + (1 - α) * previous_EMA
                    self.state.last_result = (
                        self.alpha * new_value + (1 - self.alpha) * self.state.last_result
                    )
                
                # Update deque for reference
                self.state.current_values.append(new_value)
                
                result = self.state.last_result
                
                # Record performance
                end_time = time.time_ns()
                calculation_time_us = (end_time - start_time) / 1000
                self._record_calculation(calculation_time_us, result)
                
                return result
            
            return self.state.last_result or 0.0
    
    def get_current_value(self) -> Optional[float]:
        with self.lock:
            return self.state.last_result
    
    def reset(self):
        with self.lock:
            self.state.current_values.clear()
            self.state.last_result = None
            self.is_initialized = False

class IncrementalRSI(IncrementalCalculator):
    """Incremental Relative Strength Index calculator"""
    
    def __init__(self, window_size: int = 14):
        super().__init__(window_size)
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.previous_close = None
        self.is_initialized = False
        self.alpha = 1.0 / window_size
    
    def update(self, event: UpdateEvent) -> float:
        start_time = time.time_ns()
        
        with self.lock:
            if event.update_type == UpdateType.APPEND:
                new_close = event.new_value
                
                if self.previous_close is None:
                    self.previous_close = new_close
                    return 50.0  # Neutral RSI
                
                # Calculate price change
                change = new_close - self.previous_close
                gain = max(change, 0)
                loss = max(-change, 0)
                
                if not self.is_initialized:
                    # Initialize with first 'window_size' values
                    self.state.current_values.append((gain, loss))
                    
                    if len(self.state.current_values) >= self.window_size:
                        # Calculate initial averages
                        gains = [g for g, l in self.state.current_values]
                        losses = [l for g, l in self.state.current_values]
                        
                        self.avg_gain = sum(gains) / len(gains)
                        self.avg_loss = sum(losses) / len(losses)
                        self.is_initialized = True
                    else:
                        self.previous_close = new_close
                        return 50.0  # Not enough data
                else:
                    # Update averages using Wilder's smoothing
                    self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
                    self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss
                
                # Calculate RSI
                if self.avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = self.avg_gain / self.avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                self.state.last_result = rsi
                self.previous_close = new_close
                
                # Record performance
                end_time = time.time_ns()
                calculation_time_us = (end_time - start_time) / 1000
                self._record_calculation(calculation_time_us, rsi)
                
                return rsi
            
            return self.state.last_result or 50.0
    
    def get_current_value(self) -> Optional[float]:
        with self.lock:
            return self.state.last_result
    
    def reset(self):
        with self.lock:
            self.state.current_values.clear()
            self.avg_gain = 0.0
            self.avg_loss = 0.0
            self.previous_close = None
            self.is_initialized = False
            self.state.last_result = None

class IncrementalStdDev(IncrementalCalculator):
    """Incremental Standard Deviation calculator"""
    
    def __init__(self, window_size: int):
        super().__init__(window_size)
        self.sum_value = 0.0
        self.sum_squared = 0.0
        self.count = 0
    
    def update(self, event: UpdateEvent) -> float:
        start_time = time.time_ns()
        
        with self.lock:
            if event.update_type == UpdateType.APPEND:
                new_value = event.new_value
                
                # If window is full, remove oldest value
                if len(self.state.current_values) == self.window_size:
                    old_value = self.state.current_values[0]
                    self.sum_value -= old_value
                    self.sum_squared -= old_value * old_value
                else:
                    self.count += 1
                
                # Add new value
                self.state.current_values.append(new_value)
                self.sum_value += new_value
                self.sum_squared += new_value * new_value
                
                # Calculate standard deviation
                n = len(self.state.current_values)
                if n > 1:
                    mean = self.sum_value / n
                    variance = (self.sum_squared / n) - (mean * mean)
                    std_dev = np.sqrt(max(0, variance))  # Ensure non-negative
                    self.state.last_result = std_dev
                else:
                    self.state.last_result = 0.0
                
                result = self.state.last_result
                
                # Record performance
                end_time = time.time_ns()
                calculation_time_us = (end_time - start_time) / 1000
                self._record_calculation(calculation_time_us, result)
                
                return result
            
            return self.state.last_result or 0.0
    
    def get_current_value(self) -> Optional[float]:
        with self.lock:
            return self.state.last_result
    
    def reset(self):
        with self.lock:
            self.state.current_values.clear()
            self.sum_value = 0.0
            self.sum_squared = 0.0
            self.count = 0
            self.state.last_result = None

class IncrementalBollingerBands(IncrementalCalculator):
    """Incremental Bollinger Bands calculator"""
    
    def __init__(self, window_size: int = 20, num_std: float = 2.0):
        super().__init__(window_size)
        self.num_std = num_std
        self.sma_calculator = IncrementalSMA(window_size)
        self.std_calculator = IncrementalStdDev(window_size)
        self.current_bands = {'middle': 0.0, 'upper': 0.0, 'lower': 0.0}
    
    def update(self, event: UpdateEvent) -> Dict[str, float]:
        start_time = time.time_ns()
        
        with self.lock:
            if event.update_type == UpdateType.APPEND:
                # Update both SMA and StdDev
                sma = self.sma_calculator.update(event)
                std_dev = self.std_calculator.update(event)
                
                # Calculate Bollinger Bands
                middle = sma
                upper = sma + (self.num_std * std_dev)
                lower = sma - (self.num_std * std_dev)
                
                self.current_bands = {
                    'middle': middle,
                    'upper': upper,
                    'lower': lower
                }
                
                # Record performance
                end_time = time.time_ns()
                calculation_time_us = (end_time - start_time) / 1000
                self._record_calculation(calculation_time_us, middle)
                
                return self.current_bands
            
            return self.current_bands
    
    def get_current_value(self) -> Optional[Dict[str, float]]:
        with self.lock:
            return self.current_bands
    
    def reset(self):
        with self.lock:
            self.sma_calculator.reset()
            self.std_calculator.reset()
            self.current_bands = {'middle': 0.0, 'upper': 0.0, 'lower': 0.0}

class IncrementalVWAP(IncrementalCalculator):
    """Incremental Volume Weighted Average Price calculator"""
    
    def __init__(self, window_size: int = 1000):
        super().__init__(window_size)
        self.cumulative_pv = 0.0  # Price * Volume
        self.cumulative_volume = 0.0
        self.pv_values = deque(maxlen=window_size)
        self.volume_values = deque(maxlen=window_size)
    
    def update(self, event: UpdateEvent) -> float:
        start_time = time.time_ns()
        
        with self.lock:
            if event.update_type == UpdateType.APPEND:
                # event.new_value should be (price, volume) tuple
                if isinstance(event.new_value, (list, tuple)) and len(event.new_value) == 2:
                    price, volume = event.new_value
                    pv = price * volume
                    
                    # If window is full, remove oldest values
                    if len(self.pv_values) == self.window_size:
                        old_pv = self.pv_values[0]
                        old_volume = self.volume_values[0]
                        self.cumulative_pv -= old_pv
                        self.cumulative_volume -= old_volume
                    
                    # Add new values
                    self.pv_values.append(pv)
                    self.volume_values.append(volume)
                    self.cumulative_pv += pv
                    self.cumulative_volume += volume
                    
                    # Calculate VWAP
                    if self.cumulative_volume > 0:
                        vwap = self.cumulative_pv / self.cumulative_volume
                        self.state.last_result = vwap
                    else:
                        self.state.last_result = price
                    
                    result = self.state.last_result
                    
                    # Record performance
                    end_time = time.time_ns()
                    calculation_time_us = (end_time - start_time) / 1000
                    self._record_calculation(calculation_time_us, result)
                    
                    return result
            
            return self.state.last_result or 0.0
    
    def get_current_value(self) -> Optional[float]:
        with self.lock:
            return self.state.last_result
    
    def reset(self):
        with self.lock:
            self.pv_values.clear()
            self.volume_values.clear()
            self.cumulative_pv = 0.0
            self.cumulative_volume = 0.0
            self.state.last_result = None

class IncrementalIndicatorManager:
    """Manager for multiple incremental indicators"""
    
    def __init__(self):
        self.indicators = {}
        self.lock = threading.RLock()
        
        # Performance metrics
        self.total_updates = 0
        self.total_calculation_time_us = 0.0
        self.update_history = deque(maxlen=10000)
    
    def add_indicator(self, name: str, indicator: IncrementalCalculator):
        """Add an indicator to the manager"""
        with self.lock:
            self.indicators[name] = indicator
    
    def remove_indicator(self, name: str):
        """Remove an indicator from the manager"""
        with self.lock:
            if name in self.indicators:
                del self.indicators[name]
    
    def update_all(self, event: UpdateEvent) -> Dict[str, Any]:
        """Update all indicators with the same event"""
        start_time = time.time_ns()
        results = {}
        
        with self.lock:
            for name, indicator in self.indicators.items():
                try:
                    result = indicator.update(event)
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error updating indicator {name}: {str(e)}")
                    results[name] = None
        
        # Record performance
        end_time = time.time_ns()
        calculation_time_us = (end_time - start_time) / 1000
        self.total_updates += 1
        self.total_calculation_time_us += calculation_time_us
        
        self.update_history.append({
            'timestamp': time.time(),
            'calculation_time_us': calculation_time_us,
            'indicators_updated': len(results),
            'event_type': event.update_type.value
        })
        
        return results
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current values from all indicators"""
        results = {}
        
        with self.lock:
            for name, indicator in self.indicators.items():
                try:
                    results[name] = indicator.get_current_value()
                except Exception as e:
                    logger.error(f"Error getting value from indicator {name}: {str(e)}")
                    results[name] = None
        
        return results
    
    def reset_all(self):
        """Reset all indicators"""
        with self.lock:
            for indicator in self.indicators.values():
                indicator.reset()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all indicators"""
        summary = {
            'total_updates': self.total_updates,
            'total_calculation_time_us': self.total_calculation_time_us,
            'avg_calculation_time_us': self.total_calculation_time_us / self.total_updates if self.total_updates > 0 else 0,
            'indicators': {}
        }
        
        with self.lock:
            for name, indicator in self.indicators.items():
                summary['indicators'][name] = indicator.get_metrics()
        
        return summary

class StreamingDataProcessor:
    """High-level streaming data processor with incremental calculations"""
    
    def __init__(self, indicators_config: Dict[str, Dict[str, Any]]):
        self.indicator_manager = IncrementalIndicatorManager()
        self.data_buffer = deque(maxlen=10000)
        self.lock = threading.RLock()
        
        # Setup indicators based on configuration
        self._setup_indicators(indicators_config)
        
        # Performance tracking
        self.process_count = 0
        self.total_processing_time_us = 0.0
    
    def _setup_indicators(self, config: Dict[str, Dict[str, Any]]):
        """Setup indicators based on configuration"""
        for name, params in config.items():
            indicator_type = params.get('type', 'sma')
            window_size = params.get('window_size', 20)
            
            if indicator_type == 'sma':
                indicator = IncrementalSMA(window_size)
            elif indicator_type == 'ema':
                alpha = params.get('alpha')
                indicator = IncrementalEMA(window_size, alpha)
            elif indicator_type == 'rsi':
                indicator = IncrementalRSI(window_size)
            elif indicator_type == 'std':
                indicator = IncrementalStdDev(window_size)
            elif indicator_type == 'bollinger':
                num_std = params.get('num_std', 2.0)
                indicator = IncrementalBollingerBands(window_size, num_std)
            elif indicator_type == 'vwap':
                indicator = IncrementalVWAP(window_size)
            else:
                logger.warning(f"Unknown indicator type: {indicator_type}")
                continue
            
            self.indicator_manager.add_indicator(name, indicator)
    
    def process_tick(self, price: float, volume: int = 0, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process a single tick of data"""
        start_time = time.time_ns()
        
        # Create update event
        if volume > 0:
            # For VWAP, pass price and volume
            event = UpdateEvent(
                update_type=UpdateType.APPEND,
                new_value=(price, volume),
                timestamp=timestamp or time.time()
            )
        else:
            # For other indicators, just pass price
            event = UpdateEvent(
                update_type=UpdateType.APPEND,
                new_value=price,
                timestamp=timestamp or time.time()
            )
        
        # Update all indicators
        results = self.indicator_manager.update_all(event)
        
        # Store in buffer
        with self.lock:
            self.data_buffer.append({
                'timestamp': timestamp or time.time(),
                'price': price,
                'volume': volume,
                'indicators': results
            })
        
        # Update performance metrics
        end_time = time.time_ns()
        processing_time_us = (end_time - start_time) / 1000
        self.process_count += 1
        self.total_processing_time_us += processing_time_us
        
        return results
    
    def process_ohlcv(self, ohlcv: Dict[str, float], timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process OHLCV data"""
        # Use close price for most indicators
        close_price = ohlcv.get('close', 0.0)
        volume = ohlcv.get('volume', 0)
        
        return self.process_tick(close_price, volume, timestamp)
    
    def get_latest_values(self) -> Dict[str, Any]:
        """Get latest indicator values"""
        return self.indicator_manager.get_current_values()
    
    def get_historical_data(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get historical data points"""
        with self.lock:
            return list(self.data_buffer)[-n:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_processing_time = self.total_processing_time_us / self.process_count if self.process_count > 0 else 0
        throughput = 1000000 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            'process_count': self.process_count,
            'total_processing_time_us': self.total_processing_time_us,
            'avg_processing_time_us': avg_processing_time,
            'throughput_ticks_per_sec': throughput,
            'buffer_size': len(self.data_buffer),
            'indicator_performance': self.indicator_manager.get_performance_summary()
        }
    
    def reset(self):
        """Reset all indicators and clear buffer"""
        self.indicator_manager.reset_all()
        
        with self.lock:
            self.data_buffer.clear()
        
        self.process_count = 0
        self.total_processing_time_us = 0.0

# Utility functions
def create_streaming_processor(indicators: List[str] = None) -> StreamingDataProcessor:
    """Create streaming processor with default indicators"""
    if indicators is None:
        indicators = ['sma_20', 'ema_20', 'rsi_14', 'bollinger_20', 'vwap']
    
    config = {}
    for indicator in indicators:
        if indicator.startswith('sma'):
            window = int(indicator.split('_')[1])
            config[indicator] = {'type': 'sma', 'window_size': window}
        elif indicator.startswith('ema'):
            window = int(indicator.split('_')[1])
            config[indicator] = {'type': 'ema', 'window_size': window}
        elif indicator.startswith('rsi'):
            window = int(indicator.split('_')[1])
            config[indicator] = {'type': 'rsi', 'window_size': window}
        elif indicator.startswith('bollinger'):
            window = int(indicator.split('_')[1])
            config[indicator] = {'type': 'bollinger', 'window_size': window}
        elif indicator == 'vwap':
            config[indicator] = {'type': 'vwap', 'window_size': 1000}
    
    return StreamingDataProcessor(config)

def benchmark_incremental_performance(data_size: int = 10000, 
                                    indicators: List[str] = None) -> Dict[str, Any]:
    """Benchmark incremental calculation performance"""
    processor = create_streaming_processor(indicators)
    
    # Generate test data
    np.random.seed(42)
    prices = np.random.randn(data_size).cumsum() + 100
    volumes = np.random.randint(100, 1000, data_size)
    
    # Benchmark processing
    start_time = time.time_ns()
    
    for i in range(data_size):
        processor.process_tick(prices[i], volumes[i])
    
    end_time = time.time_ns()
    total_time_us = (end_time - start_time) / 1000
    
    return {
        'total_time_us': total_time_us,
        'data_points': data_size,
        'avg_time_per_point_us': total_time_us / data_size,
        'throughput_points_per_sec': data_size / (total_time_us / 1000000),
        'processor_stats': processor.get_performance_stats()
    }
