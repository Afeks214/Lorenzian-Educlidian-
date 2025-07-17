"""Bar generator component for AlgoSpace trading system.

This module aggregates tick data into OHLCV bars for multiple timeframes.
It handles time gaps in data by generating synthetic bars to ensure continuous
time series for technical indicators.

Features:
- Memory pooling for efficient object allocation
- Circular buffer management for historical data
- Comprehensive input validation
- Performance monitoring and profiling
- Memory leak detection
- Batch processing capabilities
- Timezone-aware timestamp handling
- Intelligent gap detection and filling
- Market hours awareness
- Data quality metrics and monitoring

Enhanced by Agent 7 for production-ready timestamp alignment and gap handling.
"""

import gc
import logging
import psutil
import statistics
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from queue import Queue, Empty
from contextlib import contextmanager
from enum import Enum
import cProfile
import pstats
from io import StringIO

# Timezone handling imports
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    
try:
    from zoneinfo import ZoneInfo
    HAS_ZONEINFO = True
except ImportError:
    HAS_ZONEINFO = False

# Import existing utilities if available
try:
    from src.utils.time_utils import TimeUtils
    HAS_TIME_UTILS = True
except ImportError:
    HAS_TIME_UTILS = False

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class MemoryError(Exception):
    """Custom exception for memory-related errors."""
    pass


class TimestampError(Exception):
    """Custom exception for timestamp-related errors."""
    pass


class GapFillStrategy(Enum):
    """Strategies for filling gaps in bar data."""
    FORWARD_FILL = "forward_fill"  # Use last known price
    ZERO_VOLUME = "zero_volume"    # Forward fill with zero volume
    INTERPOLATE = "interpolate"    # Linear interpolation
    SKIP = "skip"                  # Skip gap periods
    SMART_FILL = "smart_fill"      # Context-aware gap filling


class MarketSession(Enum):
    """Market session types."""
    REGULAR = "regular"
    EXTENDED = "extended"
    OVERNIGHT = "overnight"
    CLOSED = "closed"


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"  # > 0.95
    GOOD = "good"           # 0.8 - 0.95
    FAIR = "fair"           # 0.6 - 0.8
    POOR = "poor"           # < 0.6


class PerformanceMonitor:
    """Monitor performance metrics and memory usage."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.tick_times = deque(maxlen=1000)
        self.memory_samples = deque(maxlen=100)
        self.profiler = None
        self.profile_enabled = False
        
    def start_profiling(self):
        """Start performance profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profile_enabled = True
        
    def stop_profiling(self) -> str:
        """Stop profiling and return stats."""
        if not self.profile_enabled or not self.profiler:
            return "Profiling not active"
            
        self.profiler.disable()
        self.profile_enabled = False
        
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        return s.getvalue()
        
    def record_tick_time(self, duration: float):
        """Record tick processing time."""
        self.tick_times.append(duration)
        
    def sample_memory(self):
        """Sample current memory usage."""
        try:
            memory_info = self.process.memory_info()
            self.memory_samples.append({
                'timestamp': time.time(),
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': self.process.memory_percent()
            })
        except Exception as e:
            logger.warning(f"Failed to sample memory: {e}")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        metrics = {
            'uptime_seconds': uptime,
            'tick_count': len(self.tick_times),
            'memory_samples': len(self.memory_samples)
        }
        
        if self.tick_times:
            metrics.update({
                'avg_tick_time_ms': sum(self.tick_times) / len(self.tick_times) * 1000,
                'max_tick_time_ms': max(self.tick_times) * 1000,
                'min_tick_time_ms': min(self.tick_times) * 1000,
                'ticks_per_second': len(self.tick_times) / uptime if uptime > 0 else 0
            })
            
        if self.memory_samples:
            latest_memory = self.memory_samples[-1]
            metrics.update({
                'current_memory_mb': latest_memory['rss'] / (1024 * 1024),
                'memory_percent': latest_memory['percent'],
                'memory_trend': self._calculate_memory_trend()
            })
            
        return metrics
        
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        if len(self.memory_samples) < 2:
            return "insufficient_data"
            
        recent_avg = sum(s['rss'] for s in list(self.memory_samples)[-10:]) / min(10, len(self.memory_samples))
        older_avg = sum(s['rss'] for s in list(self.memory_samples)[:10]) / min(10, len(self.memory_samples))
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"


class MemoryPool:
    """Memory pool for BarData objects to reduce allocation overhead."""
    
    def __init__(self, initial_size: int = 100, max_size: int = 1000):
        self.pool = Queue(maxsize=max_size)
        self.max_size = max_size
        self.created_count = 0
        self.reused_count = 0
        self.lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(initial_size):
            self.pool.put(self._create_new_bar_data())
            
    def _create_new_bar_data(self) -> 'BarData':
        """Create a new BarData instance."""
        self.created_count += 1
        return BarData(
            timestamp=datetime.now(),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            timeframe=0
        )
        
    def get(self) -> 'BarData':
        """Get a BarData instance from the pool."""
        try:
            bar_data = self.pool.get_nowait()
            self.reused_count += 1
            return bar_data
        except Empty:
            return self._create_new_bar_data()
            
    def return_to_pool(self, bar_data: 'BarData'):
        """Return a BarData instance to the pool."""
        if self.pool.qsize() < self.max_size:
            try:
                self.pool.put_nowait(bar_data)
            except:
                pass  # Pool is full, let GC handle it
                
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            'pool_size': self.pool.qsize(),
            'created_count': self.created_count,
            'reused_count': self.reused_count,
            'reuse_ratio': self.reused_count / max(1, self.created_count + self.reused_count)
        }


class CircularBuffer:
    """Circular buffer for historical bar data with memory management."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.total_added = 0
        self.lock = threading.RLock()
        
    def add(self, item: Any):
        """Add item to buffer."""
        with self.lock:
            self.buffer.append(item)
            self.total_added += 1
            
    def get_recent(self, count: int) -> List[Any]:
        """Get most recent items."""
        with self.lock:
            return list(self.buffer)[-count:] if count <= len(self.buffer) else list(self.buffer)
            
    def get_all(self) -> List[Any]:
        """Get all items in buffer."""
        with self.lock:
            return list(self.buffer)
            
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
            
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'total_added': self.total_added,
                'utilization': len(self.buffer) / self.max_size
            }


class InputValidator:
    """Comprehensive input validation for tick data and configuration."""
    
    @staticmethod
    def validate_tick_data(tick_data: Union[Dict, Any]) -> Dict[str, Any]:
        """Validate tick data structure and values."""
        if tick_data is None:
            raise ValidationError("Tick data cannot be None")
            
        # Handle both dict and dataclass formats
        if hasattr(tick_data, 'timestamp'):
            timestamp = tick_data.timestamp
            price = tick_data.price
            volume = tick_data.volume
        else:
            if not isinstance(tick_data, dict):
                raise ValidationError(f"Invalid tick data type: {type(tick_data)}")
                
            required_fields = ['timestamp', 'price', 'volume']
            for field in required_fields:
                if field not in tick_data:
                    raise ValidationError(f"Missing required field: {field}")
                    
            timestamp = tick_data['timestamp']
            price = tick_data['price']
            volume = tick_data['volume']
            
        # Validate timestamp
        if not isinstance(timestamp, datetime):
            raise ValidationError(f"Invalid timestamp type: {type(timestamp)}")
            
        # Validate price
        if not isinstance(price, (int, float, Decimal)):
            raise ValidationError(f"Invalid price type: {type(price)}")
            
        price = float(price)
        if price <= 0:
            raise ValidationError(f"Price must be positive: {price}")
            
        if price > 1000000:  # Reasonable upper bound
            raise ValidationError(f"Price seems unrealistic: {price}")
            
        # Validate volume
        if not isinstance(volume, (int, float)):
            raise ValidationError(f"Invalid volume type: {type(volume)}")
            
        volume = int(volume)
        if volume < 0:
            raise ValidationError(f"Volume cannot be negative: {volume}")
            
        return {
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        }
        
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ValidationError(f"Config must be a dictionary, got {type(config)}")
            
        # Set defaults
        validated_config = {
            'memory_pool_size': 100,
            'memory_pool_max_size': 1000,
            'circular_buffer_size': 1000,
            'max_memory_mb': 500,
            'enable_profiling': False,
            'batch_size': 50,
            'validation_enabled': True,
            'memory_monitoring_interval': 60
        }
        
        # Override with provided values
        for key, value in config.items():
            if key in validated_config:
                validated_config[key] = value
                
        # Validate specific values
        if validated_config['memory_pool_size'] < 1:
            raise ValidationError("Memory pool size must be at least 1")
            
        if validated_config['circular_buffer_size'] < 10:
            raise ValidationError("Circular buffer size must be at least 10")
            
        if validated_config['max_memory_mb'] < 50:
            raise ValidationError("Max memory must be at least 50MB")
            
        return validated_config
        
    @staticmethod
    def validate_bar_data(bar_data: Dict[str, Any]) -> bool:
        """Validate OHLCV bar data integrity."""
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for field in required_fields:
            if field not in bar_data:
                raise ValidationError(f"Missing required bar field: {field}")
                
        # Validate OHLC relationships
        o, h, l, c = bar_data['open'], bar_data['high'], bar_data['low'], bar_data['close']
        
        if h < max(o, c) or h < min(o, c):
            raise ValidationError(f"High price {h} is less than open {o} or close {c}")
            
        if l > min(o, c) or l > max(o, c):
            raise ValidationError(f"Low price {l} is greater than open {o} or close {c}")
            
        if bar_data['volume'] < 0:
            raise ValidationError(f"Volume cannot be negative: {bar_data['volume']}")
            
        return True


class TimestampManager:
    """Manages timezone-aware timestamps and validation."""
    
    def __init__(self, timezone_str: str = "America/New_York"):
        self.timezone_str = timezone_str
        self.timezone = self._get_timezone(timezone_str)
        self.utc = timezone.utc
        self.last_timestamp = None
        self.tick_latencies = deque(maxlen=100)
        
    def _get_timezone(self, timezone_str: str):
        """Get timezone object from string."""
        if HAS_ZONEINFO:
            try:
                return ZoneInfo(timezone_str)
            except Exception:
                pass
        
        if HAS_PYTZ:
            try:
                return pytz.timezone(timezone_str)
            except Exception:
                pass
        
        # Fallback to UTC
        logger.warning(f"Could not create timezone {timezone_str}, using UTC")
        return timezone.utc
    
    def normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to timezone-aware format."""
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone info
            timestamp = timestamp.replace(tzinfo=self.utc)
        
        # Convert to target timezone
        return timestamp.astimezone(self.timezone)
    
    def validate_timestamp(self, timestamp: datetime) -> Tuple[bool, str]:
        """Validate timestamp and return (is_valid, error_message)."""
        if timestamp is None:
            return False, "Timestamp is None"
        
        if not isinstance(timestamp, datetime):
            return False, f"Invalid timestamp type: {type(timestamp)}"
        
        # Check for reasonable timestamp (not too far in past or future)
        now = datetime.now(self.timezone)
        if timestamp < now - timedelta(days=365):
            return False, "Timestamp is more than 1 year in the past"
        
        if timestamp > now + timedelta(hours=1):
            return False, "Timestamp is more than 1 hour in the future"
        
        # Check for out-of-order ticks
        if self.last_timestamp and timestamp < self.last_timestamp:
            seconds_diff = (self.last_timestamp - timestamp).total_seconds()
            if seconds_diff > 10:  # Allow 10 seconds tolerance
                return False, f"Timestamp is {seconds_diff}s behind last tick"
        
        return True, ""
    
    def record_tick_latency(self, timestamp: datetime):
        """Record tick processing latency."""
        now = datetime.now(self.timezone)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.timezone)
        
        latency = (now - timestamp).total_seconds()
        self.tick_latencies.append(latency)
        self.last_timestamp = timestamp
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.tick_latencies:
            return {}
        
        latencies = list(self.tick_latencies)
        return {
            'avg_latency_ms': statistics.mean(latencies) * 1000,
            'max_latency_ms': max(latencies) * 1000,
            'min_latency_ms': min(latencies) * 1000,
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[18] * 1000,
            'p99_latency_ms': statistics.quantiles(latencies, n=100)[98] * 1000
        }


class MarketHoursManager:
    """Manages market hours and session information."""
    
    def __init__(self, timezone_str: str = "America/New_York"):
        self.timezone_str = timezone_str
        self.timezone = self._get_timezone(timezone_str)
        
        # Market sessions (times are in market timezone)
        self.sessions = {
            MarketSession.REGULAR: (
                datetime.strptime('09:30', '%H:%M').time(),
                datetime.strptime('16:00', '%H:%M').time()
            ),
            MarketSession.EXTENDED: (
                datetime.strptime('04:00', '%H:%M').time(),
                datetime.strptime('20:00', '%H:%M').time()
            ),
            MarketSession.OVERNIGHT: (
                datetime.strptime('18:00', '%H:%M').time(),
                datetime.strptime('08:00', '%H:%M').time()
            )
        }
        
    def _get_timezone(self, timezone_str: str):
        """Get timezone object from string."""
        if HAS_ZONEINFO:
            try:
                return ZoneInfo(timezone_str)
            except Exception:
                pass
        
        if HAS_PYTZ:
            try:
                return pytz.timezone(timezone_str)
            except Exception:
                pass
        
        return timezone.utc
    
    def get_market_session(self, timestamp: datetime) -> MarketSession:
        """Determine market session for given timestamp."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.timezone)
        else:
            timestamp = timestamp.astimezone(self.timezone)
        
        time_of_day = timestamp.time()
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend is closed
        if day_of_week >= 5:  # Saturday or Sunday
            return MarketSession.CLOSED
        
        # Check regular hours
        regular_start, regular_end = self.sessions[MarketSession.REGULAR]
        if regular_start <= time_of_day <= regular_end:
            return MarketSession.REGULAR
        
        # Check extended hours
        extended_start, extended_end = self.sessions[MarketSession.EXTENDED]
        if extended_start <= time_of_day <= extended_end:
            return MarketSession.EXTENDED
        
        # Check overnight session
        overnight_start, overnight_end = self.sessions[MarketSession.OVERNIGHT]
        if time_of_day >= overnight_start or time_of_day <= overnight_end:
            return MarketSession.OVERNIGHT
        
        return MarketSession.CLOSED
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp."""
        session = self.get_market_session(timestamp)
        return session != MarketSession.CLOSED
    
    def get_next_market_open(self, timestamp: datetime) -> datetime:
        """Get next market open time."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.timezone)
        else:
            timestamp = timestamp.astimezone(self.timezone)
        
        # Simple implementation - next business day at 4 AM
        next_day = timestamp.replace(hour=4, minute=0, second=0, microsecond=0)
        
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        
        return next_day


@dataclass
class BarGeneratorConfig:
    """Configuration for bar generator."""
    timezone: str = "America/New_York"
    gap_fill_strategy: GapFillStrategy = GapFillStrategy.SMART_FILL
    max_gap_minutes: int = 120  # Maximum gap before considering data missing
    enable_market_hours: bool = True
    validate_timestamps: bool = True
    enable_data_quality_checks: bool = True
    performance_monitoring: bool = True
    duplicate_detection: bool = True
    max_out_of_order_seconds: int = 10
    synthetic_bar_volume_threshold: float = 0.1  # Min volume for synthetic bars
    memory_pool_size: int = 100
    memory_pool_max_size: int = 1000
    circular_buffer_size: int = 1000
    max_memory_mb: int = 500
    enable_profiling: bool = False
    batch_size: int = 50
    memory_monitoring_interval: int = 60


@dataclass
class BarGeneratorMetrics:
    """Metrics for bar generator performance."""
    tick_count: int = 0
    bars_emitted_5min: int = 0
    bars_emitted_30min: int = 0
    gaps_filled_5min: int = 0
    gaps_filled_30min: int = 0
    synthetic_bars_5min: int = 0
    synthetic_bars_30min: int = 0
    validation_errors: int = 0
    timestamp_corrections: int = 0
    duplicate_ticks: int = 0
    out_of_order_ticks: int = 0
    average_tick_latency_ns: float = 0.0
    last_reset_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def reset(self):
        """Reset metrics counters."""
        for field_name in self.__dataclass_fields__:
            if field_name != "last_reset_time":
                setattr(self, field_name, 0)
        self.last_reset_time = datetime.now(timezone.utc)


@dataclass
class GapInfo:
    """Information about detected gaps."""
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    timeframe: int
    gap_type: str  # "market_closed", "data_missing", "weekend", "holiday"
    bars_filled: int = 0
    fill_strategy: GapFillStrategy = GapFillStrategy.FORWARD_FILL


@dataclass
class BarData:
    """Standardized bar data structure.
    
    Attributes:
        timestamp: The bar start timestamp (timezone-aware)
        open: Opening price
        high: Highest price during the bar
        low: Lowest price during the bar
        close: Closing price
        volume: Total volume during the bar
        timeframe: Timeframe in minutes (5 or 30)
        is_synthetic: True if bar was generated to fill gaps
        data_quality: Quality score (0.0 to 1.0)
        market_session: Market session type
        gap_info: Information about gap if this is a synthetic bar
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: int
    is_synthetic: bool = False
    data_quality: float = 1.0
    market_session: MarketSession = MarketSession.REGULAR
    gap_info: Optional[GapInfo] = None
    
    def __post_init__(self):
        """Validate and normalize bar data."""
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Validate OHLC relationships
        if not (self.low <= self.open <= self.high and 
                self.low <= self.close <= self.high):
            logger.warning(f"Invalid OHLC relationship in bar: {self}")
            self.data_quality *= 0.7
            
        # Validate volume
        if self.volume < 0:
            logger.warning(f"Negative volume in bar: {self.volume}")
            self.data_quality *= 0.5
            
        # Determine data quality level
        if self.data_quality > 0.95:
            self.quality_level = DataQualityLevel.EXCELLENT
        elif self.data_quality > 0.8:
            self.quality_level = DataQualityLevel.GOOD
        elif self.data_quality > 0.6:
            self.quality_level = DataQualityLevel.FAIR
        else:
            self.quality_level = DataQualityLevel.POOR


class BarGenerator:
    """Aggregates tick data into OHLCV bars for multiple timeframes.
    
    Enhanced with production-ready features:
    - Timezone-aware timestamp handling
    - Intelligent gap detection and filling
    - Market hours awareness
    - Data validation and integrity checks
    - Performance monitoring
    - Duplicate detection
    - Data quality metrics
    """
    
    def __init__(self, config: Union[Dict[str, Any], BarGeneratorConfig], event_bus: Any) -> None:
        """Initialize the bar generator.
        
        Args:
            config: Configuration dictionary or BarGeneratorConfig instance
            event_bus: Event bus instance for publishing bar events
        """
        # Handle different config types
        if isinstance(config, dict):
            # Extract only BarGeneratorConfig fields from the full config
            bar_config_fields = set(BarGeneratorConfig.__dataclass_fields__.keys())
            filtered_config = {
                k: v for k, v in config.items() 
                if k in bar_config_fields
            }
            self.config = BarGeneratorConfig(**filtered_config)
        else:
            self.config = config
        
        self.event_bus = event_bus
        
        # Initialize enhanced components
        self.timestamp_manager = TimestampManager(self.config.timezone)
        self.market_hours_manager = MarketHoursManager(self.config.timezone)
        self.metrics = BarGeneratorMetrics()
        
        # Initialize memory management
        self.memory_pool = MemoryPool(
            initial_size=self.config.memory_pool_size,
            max_size=self.config.memory_pool_max_size
        )
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize bars for both timeframes
        self.bars_5min: Optional[Dict[str, Any]] = None
        self.bars_30min: Optional[Dict[str, Any]] = None
        
        # Track last completed bar timestamps for gap detection
        self.last_bar_time_5min: Optional[datetime] = None
        self.last_bar_time_30min: Optional[datetime] = None
        
        # Track last known price for gap filling
        self.last_close_price: Optional[float] = None
        
        # Duplicate detection
        self.recent_ticks = deque(maxlen=100)  # Track recent ticks for duplicate detection
        
        # Gap tracking
        self.detected_gaps_5min: List[GapInfo] = []
        self.detected_gaps_30min: List[GapInfo] = []
        
        # Start performance monitoring if enabled
        if self.config.performance_monitoring:
            self.performance_monitor.start_profiling()
        
        # Initialize circular buffers for historical data
        self.bars_5min_history = CircularBuffer(self.config.circular_buffer_size)
        self.bars_30min_history = CircularBuffer(self.config.circular_buffer_size)
        
        # Memory monitoring
        self.memory_monitor_timer = None
        self.start_memory_monitoring()
        
        # Batch processing
        self.tick_batch = []
        self.batch_size = self.config.batch_size
        
        # Caching for frequently accessed data
        self.price_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Memory leak prevention
        self.cleanup_refs = weakref.WeakSet()
        
        logger.info(f"Enhanced BarGenerator initialized with timezone: {self.config.timezone}")
    
    def start_memory_monitoring(self):
        """Start memory monitoring timer."""
        def monitor_memory():
            try:
                self.performance_monitor.sample_memory()
                
                # Check memory usage
                metrics = self.performance_monitor.get_metrics()
                if 'current_memory_mb' in metrics:
                    if metrics['current_memory_mb'] > self.config.max_memory_mb:
                        logger.warning(f"Memory usage {metrics['current_memory_mb']:.1f}MB exceeds limit {self.config.max_memory_mb}MB")
                        self._perform_memory_cleanup()
                        
                # Schedule next monitoring
                self.memory_monitor_timer = threading.Timer(
                    self.config.memory_monitoring_interval,
                    monitor_memory
                )
                self.memory_monitor_timer.start()
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
        monitor_memory()
        
    def _perform_memory_cleanup(self):
        """Perform memory cleanup when usage is high."""
        logger.info("Performing memory cleanup")
        
        # Clear caches
        self.price_cache.clear()
        
        # Trigger garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clean up weak references
        self.cleanup_refs.clear()
        
        # Reduce buffer sizes if needed
        if len(self.bars_5min_history.buffer) > self.config.circular_buffer_size // 2:
            self.bars_5min_history.buffer = deque(
                list(self.bars_5min_history.buffer)[-self.config.circular_buffer_size//2:],
                maxlen=self.config.circular_buffer_size
            )
        if len(self.bars_30min_history.buffer) > self.config.circular_buffer_size // 2:
            self.bars_30min_history.buffer = deque(
                list(self.bars_30min_history.buffer)[-self.config.circular_buffer_size//2:],
                maxlen=self.config.circular_buffer_size
            )
    
    @contextmanager
    def performance_context(self):
        """Context manager for performance timing."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_monitor.record_tick_time(duration)
            
    def cleanup(self):
        """Clean up resources."""
        if self.memory_monitor_timer:
            self.memory_monitor_timer.cancel()
            
        if self.performance_monitor.profile_enabled:
            profile_stats = self.performance_monitor.stop_profiling()
            logger.info(f"Performance profile:\\n{profile_stats}")
            
        self.price_cache.clear()
        self.bars_5min_history.clear()
        self.bars_30min_history.clear()
        
        logger.info("BarGenerator cleanup completed")
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
    
    def on_new_tick(self, tick_data: Dict[str, Any]) -> None:
        """Process incoming tick data and update bars.
        
        Enhanced with comprehensive validation and processing:
        - Timestamp validation and normalization
        - Duplicate detection
        - Data quality checks
        - Performance monitoring
        - Error handling and recovery
        
        Args:
            tick_data: Dictionary containing tick information with keys:
                      timestamp (datetime), price (float), volume (int)
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Validate input data
            if self.config.enable_data_quality_checks:
                validated_data = InputValidator.validate_tick_data(tick_data)
                timestamp = validated_data['timestamp']
                price = validated_data['price']
                volume = validated_data['volume']
            else:
                # Fast path - extract tick data directly
                if hasattr(tick_data, 'timestamp'):
                    # It's a dataclass
                    timestamp = tick_data.timestamp
                    price = tick_data.price
                    volume = tick_data.volume
                else:
                    # It's a dict
                    timestamp = tick_data['timestamp']
                    price = tick_data['price']
                    volume = tick_data['volume']
            
            # Normalize timestamp
            timestamp = self.timestamp_manager.normalize_timestamp(timestamp)
            
            # Validate timestamp
            if self.config.validate_timestamps:
                is_valid, error_msg = self.timestamp_manager.validate_timestamp(timestamp)
                if not is_valid:
                    logger.warning(f"Invalid timestamp: {error_msg}")
                    self.metrics.validation_errors += 1
                    return
            
            # Check for duplicates
            if self.config.duplicate_detection:
                if self._is_duplicate_tick(timestamp, price, volume):
                    self.metrics.duplicate_ticks += 1
                    logger.debug(f"Duplicate tick detected: {timestamp}")
                    return
            
            # Record tick for duplicate detection
            self.recent_ticks.append((timestamp, price, volume))
            
            # Update metrics
            self.metrics.tick_count += 1
            self.timestamp_manager.record_tick_latency(timestamp)
            
            # Update both timeframes
            self._update_bar_5min(timestamp, price, volume)
            self._update_bar_30min(timestamp, price, volume)
            
            # Update last known price for gap filling
            self.last_close_price = price
            
            # Performance monitoring
            if self.config.performance_monitoring:
                end_time = time.perf_counter_ns()
                duration = (end_time - start_time) / 1_000_000  # Convert to ms
                self.performance_monitor.record_tick_time(duration / 1000)  # Convert to seconds
                
                # Sample memory periodically
                if self.metrics.tick_count % 100 == 0:
                    self.performance_monitor.sample_memory()
            
        except ValidationError as e:
            logger.error(f"Validation error processing tick: {e}")
            self.metrics.validation_errors += 1
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.metrics.validation_errors += 1
    
    def _is_duplicate_tick(self, timestamp: datetime, price: float, volume: int) -> bool:
        """Check if tick is a duplicate of recent ticks."""
        for recent_timestamp, recent_price, recent_volume in self.recent_ticks:
            if (abs((timestamp - recent_timestamp).total_seconds()) < 0.001 and
                abs(price - recent_price) < 0.0001 and
                volume == recent_volume):
                return True
        return False
    
    def _update_bar_5min(self, timestamp: datetime, price: float, volume: int) -> None:
        """Update or create 5-minute bar.
        
        Args:
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        bar_time = self._get_bar_time(timestamp, 5)
        
        # Check if we need to start a new bar
        if self.bars_5min is None or self.bars_5min['timestamp'] != bar_time:
            # Emit previous bar if it exists
            if self.bars_5min is not None:
                self._emit_bar(self.bars_5min, 5)
                
            # Check for gaps
            self._handle_gaps_5min(bar_time)
            
            # Start new bar
            self.bars_5min = {
                'timestamp': bar_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update existing bar
            self.bars_5min['high'] = max(self.bars_5min['high'], price)
            self.bars_5min['low'] = min(self.bars_5min['low'], price)
            self.bars_5min['close'] = price
            self.bars_5min['volume'] += volume
    
    def _update_bar_30min(self, timestamp: datetime, price: float, volume: int) -> None:
        """Update or create 30-minute bar.
        
        Args:
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        bar_time = self._get_bar_time(timestamp, 30)
        
        # Check if we need to start a new bar
        if self.bars_30min is None or self.bars_30min['timestamp'] != bar_time:
            # Emit previous bar if it exists
            if self.bars_30min is not None:
                self._emit_bar(self.bars_30min, 30)
                
            # Check for gaps
            self._handle_gaps_30min(bar_time)
            
            # Start new bar
            self.bars_30min = {
                'timestamp': bar_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update existing bar
            self.bars_30min['high'] = max(self.bars_30min['high'], price)
            self.bars_30min['low'] = min(self.bars_30min['low'], price)
            self.bars_30min['close'] = price
            self.bars_30min['volume'] += volume
    
    def _get_bar_time(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Calculate the bar bucket timestamp for a given tick timestamp.
        
        Enhanced with timezone awareness and precision handling.
        
        Args:
            timestamp: Original tick timestamp (timezone-aware)
            timeframe_minutes: Timeframe in minutes (5 or 30)
            
        Returns:
            Bar bucket timestamp (timezone-aware)
        """
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Convert to market timezone for calculations
        market_timestamp = timestamp.astimezone(self.timestamp_manager.timezone)
        
        # Remove seconds and microseconds for clean boundaries
        market_timestamp = market_timestamp.replace(second=0, microsecond=0)
        
        # Calculate minutes since midnight in market timezone
        minutes_since_midnight = market_timestamp.hour * 60 + market_timestamp.minute
        
        # Round down to timeframe boundary
        bar_minutes = (minutes_since_midnight // timeframe_minutes) * timeframe_minutes
        
        # Create new timestamp
        bar_hour = bar_minutes // 60
        bar_minute = bar_minutes % 60
        
        # Create bar timestamp in market timezone
        bar_timestamp = market_timestamp.replace(hour=bar_hour, minute=bar_minute)
        
        return bar_timestamp
    
    def _handle_gaps_5min(self, new_bar_time: datetime) -> None:
        """Handle gaps in 5-minute data with intelligent strategies.
        
        Enhanced with:
        - Market hours awareness
        - Multiple gap filling strategies
        - Gap classification and reporting
        - Data quality tracking
        
        Args:
            new_bar_time: Timestamp of the new bar being created
        """
        if self.last_bar_time_5min is None or self.last_close_price is None:
            self.last_bar_time_5min = new_bar_time
            return
        
        # Calculate expected next bar time
        expected_time = self.last_bar_time_5min + timedelta(minutes=5)
        
        # Check if there's actually a gap
        if expected_time >= new_bar_time:
            self.last_bar_time_5min = new_bar_time
            return
        
        # Calculate gap duration
        gap_duration = (new_bar_time - expected_time).total_seconds() / 60
        
        # Skip gap handling if it's too large (likely market closure)
        if gap_duration > self.config.max_gap_minutes:
            logger.info(f"Skipping large gap of {gap_duration:.1f} minutes (likely market closure)")
            self.last_bar_time_5min = new_bar_time
            return
        
        # Determine gap type and strategy
        gap_type = self._classify_gap(expected_time, new_bar_time)
        strategy = self._select_gap_strategy(gap_type, gap_duration)
        
        # Create gap info
        gap_info = GapInfo(
            start_time=expected_time,
            end_time=new_bar_time,
            duration_minutes=int(gap_duration),
            timeframe=5,
            gap_type=gap_type,
            fill_strategy=strategy
        )
        
        # Generate synthetic bars based on strategy
        bars_created = 0
        current_time = expected_time
        
        while current_time < new_bar_time:
            # Check if we should fill this period
            if self._should_fill_gap_period(current_time, gap_type):
                synthetic_bar = self._create_synthetic_bar(current_time, 5, strategy)
                self._emit_bar(synthetic_bar, 5, is_synthetic=True, gap_info=gap_info)
                bars_created += 1
                self.metrics.synthetic_bars_5min += 1
            
            current_time += timedelta(minutes=5)
        
        # Update gap info and metrics
        gap_info.bars_filled = bars_created
        self.detected_gaps_5min.append(gap_info)
        self.metrics.gaps_filled_5min += bars_created
        
        logger.info(f"Filled {bars_created} synthetic 5-min bars for {gap_type} gap "
                   f"({gap_duration:.1f} min) using {strategy.value} strategy")
        
        self.last_bar_time_5min = new_bar_time
    
    def _handle_gaps_30min(self, new_bar_time: datetime) -> None:
        """Handle gaps in 30-minute data with intelligent strategies.
        
        Enhanced with same features as 5-minute gap handling.
        
        Args:
            new_bar_time: Timestamp of the new bar being created
        """
        if self.last_bar_time_30min is None or self.last_close_price is None:
            self.last_bar_time_30min = new_bar_time
            return
        
        # Calculate expected next bar time
        expected_time = self.last_bar_time_30min + timedelta(minutes=30)
        
        # Check if there's actually a gap
        if expected_time >= new_bar_time:
            self.last_bar_time_30min = new_bar_time
            return
        
        # Calculate gap duration
        gap_duration = (new_bar_time - expected_time).total_seconds() / 60
        
        # Skip gap handling if it's too large (likely market closure)
        if gap_duration > self.config.max_gap_minutes:
            logger.info(f"Skipping large 30-min gap of {gap_duration:.1f} minutes (likely market closure)")
            self.last_bar_time_30min = new_bar_time
            return
        
        # Determine gap type and strategy
        gap_type = self._classify_gap(expected_time, new_bar_time)
        strategy = self._select_gap_strategy(gap_type, gap_duration)
        
        # Create gap info
        gap_info = GapInfo(
            start_time=expected_time,
            end_time=new_bar_time,
            duration_minutes=int(gap_duration),
            timeframe=30,
            gap_type=gap_type,
            fill_strategy=strategy
        )
        
        # Generate synthetic bars based on strategy
        bars_created = 0
        current_time = expected_time
        
        while current_time < new_bar_time:
            # Check if we should fill this period
            if self._should_fill_gap_period(current_time, gap_type):
                synthetic_bar = self._create_synthetic_bar(current_time, 30, strategy)
                self._emit_bar(synthetic_bar, 30, is_synthetic=True, gap_info=gap_info)
                bars_created += 1
                self.metrics.synthetic_bars_30min += 1
            
            current_time += timedelta(minutes=30)
        
        # Update gap info and metrics
        gap_info.bars_filled = bars_created
        self.detected_gaps_30min.append(gap_info)
        self.metrics.gaps_filled_30min += bars_created
        
        logger.info(f"Filled {bars_created} synthetic 30-min bars for {gap_type} gap "
                   f"({gap_duration:.1f} min) using {strategy.value} strategy")
        
        self.last_bar_time_30min = new_bar_time
    
    def _classify_gap(self, start_time: datetime, end_time: datetime) -> str:
        """Classify the type of gap based on timing and market hours."""
        # Check if gap spans weekend
        if start_time.weekday() >= 4 and end_time.weekday() <= 1:  # Friday to Monday
            return "weekend"
        
        # Check if gap is during market hours
        if self.config.enable_market_hours:
            start_session = self.market_hours_manager.get_market_session(start_time)
            end_session = self.market_hours_manager.get_market_session(end_time)
            
            if start_session == MarketSession.CLOSED or end_session == MarketSession.CLOSED:
                return "market_closed"
        
        # Check gap duration
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        if duration_minutes > 240:  # 4 hours
            return "holiday"
        elif duration_minutes > 60:  # 1 hour
            return "extended_break"
        else:
            return "data_missing"
    
    def _select_gap_strategy(self, gap_type: str, duration_minutes: float) -> GapFillStrategy:
        """Select appropriate gap filling strategy based on gap characteristics."""
        if self.config.gap_fill_strategy == GapFillStrategy.SMART_FILL:
            # Smart selection based on gap type
            if gap_type == "weekend":
                return GapFillStrategy.SKIP
            elif gap_type == "market_closed":
                return GapFillStrategy.SKIP
            elif gap_type == "holiday":
                return GapFillStrategy.SKIP
            elif gap_type == "extended_break":
                return GapFillStrategy.ZERO_VOLUME
            else:  # data_missing
                return GapFillStrategy.FORWARD_FILL
        else:
            # Use configured strategy
            return self.config.gap_fill_strategy
    
    def _should_fill_gap_period(self, timestamp: datetime, gap_type: str) -> bool:
        """Determine if a specific gap period should be filled."""
        if gap_type in ["weekend", "market_closed", "holiday"]:
            return False
        
        # Check if within market hours
        if self.config.enable_market_hours:
            return self.market_hours_manager.is_market_open(timestamp)
        
        return True
    
    def _create_synthetic_bar(self, timestamp: datetime, timeframe: int, strategy: GapFillStrategy) -> Dict[str, Any]:
        """Create a synthetic bar using the specified strategy."""
        if strategy == GapFillStrategy.FORWARD_FILL:
            # Use last known price
            price = self.last_close_price
            volume = 0
        elif strategy == GapFillStrategy.ZERO_VOLUME:
            # Forward fill price with zero volume
            price = self.last_close_price
            volume = 0
        elif strategy == GapFillStrategy.INTERPOLATE:
            # Simple interpolation (could be enhanced)
            price = self.last_close_price
            volume = 0
        else:
            # Default to forward fill
            price = self.last_close_price
            volume = 0
        
        # Add small noise to volume if configured
        if volume == 0 and self.config.synthetic_bar_volume_threshold > 0:
            volume = int(self.config.synthetic_bar_volume_threshold * 1000)
        
        return {
            'timestamp': timestamp,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume
        }
    
    def _emit_bar(self, bar_dict: Dict[str, Any], timeframe: int, 
                  is_synthetic: bool = False, gap_info: Optional[GapInfo] = None) -> None:
        """Emit a completed bar through the event bus.
        
        Enhanced with synthetic bar handling and market session detection.
        
        Args:
            bar_dict: Dictionary containing bar OHLCV data
            timeframe: Timeframe in minutes (5 or 30)
            is_synthetic: Whether this is a synthetic bar
            gap_info: Gap information if synthetic bar
        """
        # Determine market session
        market_session = self.market_hours_manager.get_market_session(bar_dict['timestamp'])
        
        # Create BarData object
        bar_data = BarData(
            timestamp=bar_dict['timestamp'],
            open=bar_dict['open'],
            high=bar_dict['high'],
            low=bar_dict['low'],
            close=bar_dict['close'],
            volume=bar_dict['volume'],
            timeframe=timeframe,
            is_synthetic=is_synthetic,
            market_session=market_session,
            gap_info=gap_info
        )
        
        # Determine event type based on timeframe
        if timeframe == 5:
            event_type = 'NEW_5MIN_BAR'
            self.metrics.bars_emitted_5min += 1
        elif timeframe == 30:
            event_type = 'NEW_30MIN_BAR'
            self.metrics.bars_emitted_30min += 1
        else:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return
        
        # Publish bar event
        self.event_bus.publish(event_type, bar_data)
        
        # Log bar emission with enhanced information
        log_level = logging.DEBUG if not is_synthetic else logging.INFO
        synthetic_label = " (synthetic)" if is_synthetic else ""
        logger.log(log_level, 
                  f"Emitted {timeframe}-min bar{synthetic_label}: {bar_data.timestamp} "
                  f"OHLC={bar_data.open:.2f}/{bar_data.high:.2f}/{bar_data.low:.2f}/{bar_data.close:.2f} "
                  f"V={bar_data.volume} Session={market_session.value} "
                  f"Quality={bar_data.data_quality:.2f}")
        
        # Return to memory pool if configured
        if hasattr(self, 'memory_pool'):
            self.memory_pool.return_to_pool(bar_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bar generation statistics.
        
        Returns:
            Dictionary containing detailed statistics about bar generation
        """
        base_stats = {
            'tick_count': self.metrics.tick_count,
            'bars_emitted_5min': self.metrics.bars_emitted_5min,
            'bars_emitted_30min': self.metrics.bars_emitted_30min,
            'gaps_filled_5min': self.metrics.gaps_filled_5min,
            'gaps_filled_30min': self.metrics.gaps_filled_30min,
            'validation_errors': self.metrics.validation_errors,
            'duplicate_ticks': self.metrics.duplicate_ticks,
            'out_of_order_ticks': self.metrics.out_of_order_ticks,
            'batch_size': len(self.tick_batch) if hasattr(self, 'tick_batch') else 0
        }
        
        # Add performance metrics
        if hasattr(self, 'performance_monitor'):
            performance_metrics = self.performance_monitor.get_metrics()
            base_stats.update(performance_metrics)
        
        # Add memory pool statistics
        if hasattr(self, 'memory_pool'):
            pool_stats = self.memory_pool.get_stats()
            base_stats.update({f'pool_{k}': v for k, v in pool_stats.items()})
        
        # Add buffer statistics
        if hasattr(self, 'bars_5min_history'):
            buffer_5min_stats = self.bars_5min_history.get_stats()
            buffer_30min_stats = self.bars_30min_history.get_stats()
            base_stats.update({f'buffer_5min_{k}': v for k, v in buffer_5min_stats.items()})
            base_stats.update({f'buffer_30min_{k}': v for k, v in buffer_30min_stats.items()})
        
        # Add cache statistics
        if hasattr(self, 'price_cache'):
            base_stats.update({
                'cache_size': len(self.price_cache),
                'cache_hit_count': self.cache_hit_count,
                'cache_miss_count': self.cache_miss_count,
                'cache_hit_ratio': self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count)
            })
        
        return base_stats
        
    def get_historical_bars(self, timeframe: int, count: int = 100) -> List[Dict[str, Any]]:
        """Get historical bars for a specific timeframe.
        
        Args:
            timeframe: Timeframe in minutes (5 or 30)
            count: Number of recent bars to return
            
        Returns:
            List of bar dictionaries
        """
        if not hasattr(self, 'bars_5min_history'):
            return []
            
        if timeframe == 5:
            return self.bars_5min_history.get_recent(count)
        elif timeframe == 30:
            return self.bars_30min_history.get_recent(count)
        else:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return []
            
    def get_current_bars(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get current incomplete bars.
        
        Returns:
            Dictionary with current 5min and 30min bars
        """
        return {
            '5min': self.bars_5min,
            '30min': self.bars_30min
        }
        
    def get_performance_profile(self) -> str:
        """Get performance profiling results.
        
        Returns:
            String containing performance profile
        """
        if hasattr(self, 'performance_monitor') and self.performance_monitor.profile_enabled:
            return self.performance_monitor.stop_profiling()
        else:
            return "Profiling not enabled"
            
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.metrics.reset()
        
        # Reset cache counters
        if hasattr(self, 'cache_hit_count'):
            self.cache_hit_count = 0
            self.cache_miss_count = 0
        
        # Reset performance monitor
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor = PerformanceMonitor()
            if self.config.performance_monitoring:
                self.performance_monitor.start_profiling()
                
        logger.info("Statistics reset")
        
    def process_batch_tick_data(self, tick_list: List[Dict[str, Any]]):
        """Process a batch of tick data efficiently.
        
        Args:
            tick_list: List of tick data dictionaries
        """
        if not tick_list:
            return
            
        logger.info(f"Processing batch of {len(tick_list)} ticks")
        
        # Sort by timestamp to ensure correct order
        sorted_ticks = sorted(tick_list, key=lambda x: x['timestamp'])
        
        # Process each tick
        for tick_data in sorted_ticks:
            self.on_new_tick(tick_data)
            
        # Force process any remaining batch
        if hasattr(self, 'tick_batch') and self.tick_batch:
            self._process_tick_batch()
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        if not hasattr(self, 'performance_monitor'):
            return {}
            
        metrics = self.performance_monitor.get_metrics()
        return {
            'current_memory_mb': metrics.get('current_memory_mb', 0),
            'memory_percent': metrics.get('memory_percent', 0),
            'memory_trend': metrics.get('memory_trend', 'unknown'),
            'memory_limit_mb': self.config.max_memory_mb
        }
        
    def optimize_memory(self):
        """Manually trigger memory optimization."""
        if hasattr(self, '_perform_memory_cleanup'):
            self._perform_memory_cleanup()
        else:
            logger.warning("Memory optimization not available")
            
    def enable_profiling(self):
        """Enable performance profiling."""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.start_profiling()
            logger.info("Performance profiling enabled")
            
    def disable_profiling(self) -> str:
        """Disable performance profiling and return results."""
        if hasattr(self, 'performance_monitor'):
            result = self.performance_monitor.stop_profiling()
            logger.info("Performance profiling disabled")
            return result
        return "Profiling not available"
    
    def get_gap_analysis(self) -> Dict[str, Any]:
        """Get detailed gap analysis report."""
        gap_analysis = {
            '5min_gaps': [],
            '30min_gaps': [],
            'gap_statistics': {
                'total_gaps': len(self.detected_gaps_5min) + len(self.detected_gaps_30min),
                'avg_gap_duration_5min': 0,
                'avg_gap_duration_30min': 0,
                'gap_types': {},
                'fill_strategies': {}
            }
        }
        
        # Analyze 5-minute gaps
        if self.detected_gaps_5min:
            gap_analysis['5min_gaps'] = [
                {
                    'start_time': gap.start_time.isoformat(),
                    'end_time': gap.end_time.isoformat(),
                    'duration_minutes': gap.duration_minutes,
                    'gap_type': gap.gap_type,
                    'bars_filled': gap.bars_filled,
                    'fill_strategy': gap.fill_strategy.value
                } for gap in self.detected_gaps_5min[-10:]  # Last 10 gaps
            ]
            
            gap_analysis['gap_statistics']['avg_gap_duration_5min'] = (
                sum(gap.duration_minutes for gap in self.detected_gaps_5min) / 
                len(self.detected_gaps_5min)
            )
        
        # Analyze 30-minute gaps
        if self.detected_gaps_30min:
            gap_analysis['30min_gaps'] = [
                {
                    'start_time': gap.start_time.isoformat(),
                    'end_time': gap.end_time.isoformat(),
                    'duration_minutes': gap.duration_minutes,
                    'gap_type': gap.gap_type,
                    'bars_filled': gap.bars_filled,
                    'fill_strategy': gap.fill_strategy.value
                } for gap in self.detected_gaps_30min[-10:]  # Last 10 gaps
            ]
            
            gap_analysis['gap_statistics']['avg_gap_duration_30min'] = (
                sum(gap.duration_minutes for gap in self.detected_gaps_30min) / 
                len(self.detected_gaps_30min)
            )
        
        # Analyze gap types and strategies
        all_gaps = self.detected_gaps_5min + self.detected_gaps_30min
        for gap in all_gaps:
            gap_analysis['gap_statistics']['gap_types'][gap.gap_type] = (
                gap_analysis['gap_statistics']['gap_types'].get(gap.gap_type, 0) + 1
            )
            gap_analysis['gap_statistics']['fill_strategies'][gap.fill_strategy.value] = (
                gap_analysis['gap_statistics']['fill_strategies'].get(gap.fill_strategy.value, 0) + 1
            )
        
        return gap_analysis
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        total_bars = self.metrics.bars_emitted_5min + self.metrics.bars_emitted_30min
        total_synthetic = self.metrics.synthetic_bars_5min + self.metrics.synthetic_bars_30min
        
        quality_report = {
            'overall_quality': {
                'data_completeness': 1.0 - (total_synthetic / max(1, total_bars)),
                'timestamp_accuracy': 1.0 - (self.metrics.timestamp_corrections / max(1, self.metrics.tick_count)),
                'data_integrity': 1.0 - (self.metrics.validation_errors / max(1, self.metrics.tick_count)),
                'duplicate_cleanliness': 1.0 - (self.metrics.duplicate_ticks / max(1, self.metrics.tick_count)),
            },
            'quality_levels': {
                'data_completeness': self._get_quality_level(1.0 - (total_synthetic / max(1, total_bars))),
                'timestamp_accuracy': self._get_quality_level(1.0 - (self.metrics.timestamp_corrections / max(1, self.metrics.tick_count))),
                'data_integrity': self._get_quality_level(1.0 - (self.metrics.validation_errors / max(1, self.metrics.tick_count))),
                'duplicate_cleanliness': self._get_quality_level(1.0 - (self.metrics.duplicate_ticks / max(1, self.metrics.tick_count))),
            },
            'recommendations': []
        }
        
        # Generate quality recommendations
        if quality_report['overall_quality']['data_completeness'] < 0.9:
            quality_report['recommendations'].append("High synthetic bar ratio - check data source quality")
        
        if quality_report['overall_quality']['timestamp_accuracy'] < 0.95:
            quality_report['recommendations'].append("Timestamp accuracy issues - verify time synchronization")
        
        if quality_report['overall_quality']['data_integrity'] < 0.95:
            quality_report['recommendations'].append("Data integrity concerns - review validation settings")
        
        if quality_report['overall_quality']['duplicate_cleanliness'] < 0.98:
            quality_report['recommendations'].append("High duplicate rate - check data deduplication")
        
        if not quality_report['recommendations']:
            quality_report['recommendations'].append("Data quality is excellent")
        
        return quality_report
    
    def _get_quality_level(self, score: float) -> str:
        """Convert quality score to quality level."""
        if score > 0.95:
            return DataQualityLevel.EXCELLENT.value
        elif score > 0.8:
            return DataQualityLevel.GOOD.value
        elif score > 0.6:
            return DataQualityLevel.FAIR.value
        else:
            return DataQualityLevel.POOR.value
    
    def reset_gap_history(self):
        """Reset gap detection history."""
        self.detected_gaps_5min.clear()
        self.detected_gaps_30min.clear()
        logger.info("Gap history reset")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary."""
        return {
            'timezone': self.config.timezone,
            'gap_fill_strategy': self.config.gap_fill_strategy.value,
            'max_gap_minutes': self.config.max_gap_minutes,
            'enable_market_hours': self.config.enable_market_hours,
            'validate_timestamps': self.config.validate_timestamps,
            'enable_data_quality_checks': self.config.enable_data_quality_checks,
            'performance_monitoring': self.config.performance_monitoring,
            'duplicate_detection': self.config.duplicate_detection,
            'max_out_of_order_seconds': self.config.max_out_of_order_seconds,
            'synthetic_bar_volume_threshold': self.config.synthetic_bar_volume_threshold,
            'memory_pool_size': self.config.memory_pool_size,
            'memory_pool_max_size': self.config.memory_pool_max_size,
            'circular_buffer_size': self.config.circular_buffer_size,
            'max_memory_mb': self.config.max_memory_mb,
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health report."""
        stats = self.get_statistics()
        quality = self.get_data_quality_report()
        memory = self.get_memory_usage()
        
        # Calculate health score
        health_score = (
            quality['overall_quality']['data_completeness'] * 0.3 +
            quality['overall_quality']['timestamp_accuracy'] * 0.2 +
            quality['overall_quality']['data_integrity'] * 0.3 +
            quality['overall_quality']['duplicate_cleanliness'] * 0.2
        )
        
        # Determine system status
        if health_score > 0.95:
            status = "EXCELLENT"
        elif health_score > 0.85:
            status = "GOOD"
        elif health_score > 0.70:
            status = "FAIR"
        else:
            status = "POOR"
        
        return {
            'overall_health_score': health_score,
            'system_status': status,
            'uptime_seconds': stats.get('uptime_seconds', 0),
            'processed_ticks': stats.get('tick_count', 0),
            'memory_usage_mb': memory.get('current_memory_mb', 0),
            'memory_trend': memory.get('memory_trend', 'unknown'),
            'performance_summary': {
                'avg_tick_time_ms': stats.get('avg_tick_time_ms', 0),
                'ticks_per_second': stats.get('ticks_per_second', 0),
                'error_rate': stats.get('error_rate', 0),
            },
            'alerts': self._generate_health_alerts(stats, quality, memory)
        }
    
    def _generate_health_alerts(self, stats: Dict, quality: Dict, memory: Dict) -> List[str]:
        """Generate health alerts based on system metrics."""
        alerts = []
        
        # Memory alerts
        if memory.get('memory_percent', 0) > 80:
            alerts.append(f"High memory usage: {memory.get('memory_percent', 0):.1f}%")
        
        if memory.get('memory_trend') == 'increasing':
            alerts.append("Memory usage trending upward")
        
        # Performance alerts
        if stats.get('avg_tick_time_ms', 0) > 1.0:
            alerts.append(f"Slow tick processing: {stats.get('avg_tick_time_ms', 0):.1f}ms")
        
        # Quality alerts
        if quality['overall_quality']['data_completeness'] < 0.9:
            alerts.append("Data completeness below 90%")
        
        if quality['overall_quality']['data_integrity'] < 0.95:
            alerts.append("Data integrity issues detected")
        
        # Error rate alerts
        if stats.get('error_rate', 0) > 0.01:
            alerts.append(f"High error rate: {stats.get('error_rate', 0):.2%}")
        
        return alerts