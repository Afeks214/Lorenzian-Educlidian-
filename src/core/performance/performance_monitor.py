"""
Performance Monitor - Comprehensive performance monitoring and profiling dashboard.
Provides real-time metrics, profiling, and benchmarking for ultra-low latency systems.
"""

import time
import threading
import psutil
import gc
import sys
import tracemalloc
import cProfile
import pstats
import io
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import numpy as np
import torch
import structlog

try:
    import py_spy
    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class TensorPool:
    """
    Tensor memory pool for efficient memory reuse.
    Reduces memory allocations by reusing tensors of the same shape.
    """
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self.pools: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        self._lock = threading.RLock()
        self._usage_stats = defaultdict(int)
        
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
        """Get a tensor from the pool or create a new one."""
        pool_key = (shape, dtype, device)
        
        with self._lock:
            if self.pools[pool_key]:
                tensor = self.pools[pool_key].pop()
                tensor.zero_()  # Clear the tensor
                self._usage_stats[pool_key] += 1
                return tensor
            else:
                # Create new tensor
                if device == 'cpu':
                    tensor = torch.zeros(shape, dtype=dtype)
                else:
                    tensor = torch.zeros(shape, dtype=dtype, device=device)
                self._usage_stats[pool_key] += 1
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool."""
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = str(tensor.device)
        pool_key = (shape, dtype, device)
        
        with self._lock:
            if len(self.pools[pool_key]) < self.max_pool_size:
                self.pools[pool_key].append(tensor)
            else:
                # Pool is full, let tensor be garbage collected
                del tensor
    
    def clear_pools(self):
        """Clear all tensor pools."""
        with self._lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()
            gc.collect()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get tensor pool statistics."""
        with self._lock:
            stats = {
                'total_pools': len(self.pools),
                'total_cached_tensors': sum(len(pool) for pool in self.pools.values()),
                'usage_stats': dict(self._usage_stats),
                'memory_saved_mb': sum(len(pool) * torch.zeros(pool_key[0]).numel() * 4 / 1024 / 1024 
                                     for pool_key, pool in self.pools.items() if pool)
            }
            return stats


class MemoryManager:
    """
    Advanced memory manager with aggressive cleanup strategies.
    Implements sophisticated garbage collection and memory optimization.
    """
    
    def __init__(self, target_memory_mb: int = 768):
        self.target_memory_mb = target_memory_mb
        self.tensor_pool = TensorPool()
        self._cleanup_counter = 0
        self._aggressive_cleanup_threshold = 50
        self._memory_pressure_threshold = 0.8  # 80% of target
        
    def check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure."""
        current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return current_memory_mb > (self.target_memory_mb * self._memory_pressure_threshold)
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        # Clear tensor pools
        self.tensor_pool.clear_pools()
        
        # Force garbage collection multiple times
        for i in range(3):
            gc.collect()
            
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear numpy cache
        try:
            import numpy as np
            # Force numpy to release memory
            np.seterr(all='raise')
            np.seterr(all='warn')
        except:
            pass
    
    def periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        self._cleanup_counter += 1
        
        if self._cleanup_counter >= self._aggressive_cleanup_threshold:
            if self.check_memory_pressure():
                self.aggressive_cleanup()
            else:
                gc.collect()
            self._cleanup_counter = 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'current_memory_mb': memory_info.rss / 1024 / 1024,
            'target_memory_mb': self.target_memory_mb,
            'memory_pressure': self.check_memory_pressure(),
            'tensor_pool_stats': self.tensor_pool.get_pool_stats(),
            'gc_counts': gc.get_count(),
            'gc_stats': gc.get_stats()
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'cached_mb': torch.cuda.memory_cached() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
            }
        
        return stats


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


class ProfilerType(Enum):
    """Types of profilers."""
    CPROFILE = "cprofile"
    LINE_PROFILER = "line_profiler"
    MEMORY_PROFILER = "memory_profiler"
    PY_SPY = "py_spy"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    NETWORK = "network"
    DISK = "disk"
    CUSTOM = "custom"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyMetrics:
    """Latency metrics collection."""
    min_ns: int = 0
    max_ns: int = 0
    avg_ns: float = 0.0
    p50_ns: int = 0
    p90_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0
    p999_ns: int = 0
    samples: int = 0
    total_time_ns: int = 0


@dataclass
class ThroughputMetrics:
    """Throughput metrics collection."""
    operations_per_second: float = 0.0
    bytes_per_second: float = 0.0
    total_operations: int = 0
    total_bytes: int = 0
    window_size_seconds: float = 1.0
    last_update: float = 0.0


@dataclass
class MemoryMetrics:
    """Memory metrics collection."""
    allocated_bytes: int = 0
    peak_allocated_bytes: int = 0
    cached_bytes: int = 0
    free_bytes: int = 0
    total_bytes: int = 0
    gpu_allocated_bytes: int = 0
    gpu_cached_bytes: int = 0
    gpu_total_bytes: int = 0
    gc_count: int = 0
    memory_leaks: int = 0


@dataclass
class CPUMetrics:
    """CPU metrics collection."""
    usage_percent: float = 0.0
    user_time_percent: float = 0.0
    system_time_percent: float = 0.0
    idle_time_percent: float = 0.0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    context_switches: int = 0
    interrupts: int = 0
    threads: int = 0


@dataclass
class GPUMetrics:
    """GPU metrics collection."""
    usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    fan_speed_percent: float = 0.0
    utilization_compute: float = 0.0
    utilization_memory: float = 0.0


class LatencyTracker:
    """
    High-precision latency tracker with histogram support.
    Tracks nanosecond-level latency with percentile calculations.
    Optimized with bounded collections and periodic cleanup.
    """
    
    def __init__(self, name: str, max_samples: int = 1000):  # Reduced default from 10000 to 1000
        self.name = name
        self.max_samples = max_samples
        self._samples = deque(maxlen=max_samples)  # Fixed-size deque prevents unbounded growth
        self._lock = threading.RLock()
        self._metrics = LatencyMetrics()
        self._gc_counter = 0
        self._gc_threshold = 100  # Trigger GC every 100 records
        
        logger.debug(f"Latency tracker created", name=name, max_samples=max_samples)
    
    def record(self, latency_ns: int):
        """Record a latency measurement with periodic cleanup."""
        with self._lock:
            self._samples.append(latency_ns)
            self._update_metrics()
            
            # Periodic garbage collection to prevent memory leaks
            self._gc_counter += 1
            if self._gc_counter >= self._gc_threshold:
                gc.collect()
                self._gc_counter = 0
    
    def record_operation(self, operation_name: str = None):
        """Context manager for recording operation latency."""
        return LatencyContext(self, operation_name)
    
    def _update_metrics(self):
        """Update latency metrics."""
        if not self._samples:
            return
        
        samples = sorted(self._samples)
        count = len(samples)
        
        self._metrics.min_ns = samples[0]
        self._metrics.max_ns = samples[-1]
        self._metrics.avg_ns = sum(samples) / count
        self._metrics.samples = count
        self._metrics.total_time_ns = sum(samples)
        
        # Calculate percentiles
        self._metrics.p50_ns = samples[int(count * 0.50)]
        self._metrics.p90_ns = samples[int(count * 0.90)]
        self._metrics.p95_ns = samples[int(count * 0.95)]
        self._metrics.p99_ns = samples[int(count * 0.99)]
        self._metrics.p999_ns = samples[int(count * 0.999)]
    
    def get_metrics(self) -> LatencyMetrics:
        """Get current latency metrics."""
        with self._lock:
            return self._metrics
    
    def reset(self):
        """Reset latency tracker."""
        with self._lock:
            self._samples.clear()
            self._metrics = LatencyMetrics()
    
    def get_histogram(self, bins: int = 50) -> Tuple[List[int], List[float]]:
        """Get latency histogram."""
        with self._lock:
            if not self._samples:
                return [], []
            
            samples = list(self._samples)
            counts, bin_edges = np.histogram(samples, bins=bins)
            return counts.tolist(), bin_edges.tolist()


class LatencyContext:
    """Context manager for latency measurement."""
    
    def __init__(self, tracker: LatencyTracker, operation_name: Optional[str] = None):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.time_ns()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time_ns()
        latency = end_time - self.start_time
        
        self.tracker.record(latency)
        
        if self.operation_name:
            logger.debug(f"Operation completed", 
                        operation=self.operation_name,
                        latency_ns=latency,
                        latency_us=latency / 1000)


class ThroughputMonitor:
    """
    Throughput monitor with sliding window calculations.
    Tracks operations per second and bytes per second.
    Memory optimized with bounded collections and aggressive cleanup.
    """
    
    def __init__(self, name: str, window_size_seconds: float = 1.0, max_entries: int = 10000):
        self.name = name
        self.window_size_seconds = window_size_seconds
        self.max_entries = max_entries
        self._operations = deque(maxlen=max_entries)  # Bounded deque prevents unbounded growth
        self._bytes = deque(maxlen=max_entries)  # Bounded deque prevents unbounded growth
        self._lock = threading.RLock()
        self._metrics = ThroughputMetrics(window_size_seconds=window_size_seconds)
        self._cleanup_counter = 0
        self._cleanup_threshold = 1000  # Clean up every 1000 operations
        
        logger.debug(f"Throughput monitor created", name=name, window_size=window_size_seconds)
    
    def record_operation(self, byte_count: int = 0):
        """Record an operation with memory optimization."""
        current_time = time.time()
        
        with self._lock:
            self._operations.append(current_time)
            if byte_count > 0:
                self._bytes.append((current_time, byte_count))
            
            # Aggressive cleanup to prevent memory growth
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_threshold:
                self._cleanup_old_entries(current_time)
                gc.collect()  # Force garbage collection
                self._cleanup_counter = 0
            else:
                self._cleanup_old_entries(current_time)
                
            self._update_metrics(current_time)
    
    def record_bytes(self, byte_count: int):
        """Record bytes processed with memory optimization."""
        current_time = time.time()
        
        with self._lock:
            self._bytes.append((current_time, byte_count))
            
            # Periodic cleanup to prevent memory growth
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_threshold:
                self._cleanup_old_entries(current_time)
                gc.collect()  # Force garbage collection
                self._cleanup_counter = 0
            else:
                self._cleanup_old_entries(current_time)
                
            self._update_metrics(current_time)
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries outside the window."""
        cutoff_time = current_time - self.window_size_seconds
        
        # Remove old operations
        while self._operations and self._operations[0] < cutoff_time:
            self._operations.popleft()
        
        # Remove old bytes
        while self._bytes and self._bytes[0][0] < cutoff_time:
            self._bytes.popleft()
    
    def _update_metrics(self, current_time: float):
        """Update throughput metrics."""
        self._metrics.operations_per_second = len(self._operations) / self.window_size_seconds
        self._metrics.total_operations = len(self._operations)
        
        if self._bytes:
            total_bytes = sum(byte_count for _, byte_count in self._bytes)
            self._metrics.bytes_per_second = total_bytes / self.window_size_seconds
            self._metrics.total_bytes = total_bytes
        
        self._metrics.last_update = current_time
    
    def get_metrics(self) -> ThroughputMetrics:
        """Get current throughput metrics."""
        with self._lock:
            current_time = time.time()
            self._cleanup_old_entries(current_time)
            self._update_metrics(current_time)
            return self._metrics
    
    def reset(self):
        """Reset throughput monitor."""
        with self._lock:
            self._operations.clear()
            self._bytes.clear()
            self._metrics = ThroughputMetrics(window_size_seconds=self.window_size_seconds)


class MemoryProfiler:
    """
    Memory profiler with leak detection and allocation tracking.
    Provides detailed memory usage analysis.
    """
    
    def __init__(self, name: str, enable_tracemalloc: bool = True):
        self.name = name
        self.enable_tracemalloc = enable_tracemalloc
        self._metrics = MemoryMetrics()
        self._lock = threading.RLock()
        self._baseline_memory = None
        self._snapshots = []
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
        
        logger.debug(f"Memory profiler created", name=name, tracemalloc=enable_tracemalloc)
    
    def update_metrics(self):
        """Update memory metrics."""
        with self._lock:
            # System memory
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self._metrics.allocated_bytes = memory_info.rss
            if self._baseline_memory is None:
                self._baseline_memory = memory_info.rss
            
            # System memory
            system_memory = psutil.virtual_memory()
            self._metrics.total_bytes = system_memory.total
            self._metrics.free_bytes = system_memory.available
            
            # GPU memory
            if torch.cuda.is_available():
                self._metrics.gpu_allocated_bytes = torch.cuda.memory_allocated()
                self._metrics.gpu_cached_bytes = torch.cuda.memory_cached()
                self._metrics.gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
            
            # Garbage collection
            gc_stats = gc.get_stats()
            self._metrics.gc_count = sum(stat['collections'] for stat in gc_stats)
            
            # Memory leaks (simple heuristic)
            if self._baseline_memory is not None:
                growth = memory_info.rss - self._baseline_memory
                if growth > 100 * 1024 * 1024:  # 100MB growth threshold
                    self._metrics.memory_leaks += 1
    
    def take_snapshot(self) -> Optional[Any]:
        """Take memory snapshot."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            self._snapshots.append(snapshot)
            return snapshot
        return None
    
    def compare_snapshots(self, snapshot1: Any, snapshot2: Any) -> List[Any]:
        """Compare two memory snapshots."""
        if snapshot1 is None or snapshot2 is None:
            return []
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        return top_stats[:10]  # Top 10 differences
    
    def get_top_allocations(self, limit: int = 10) -> List[Any]:
        """Get top memory allocations."""
        if not self._snapshots:
            return []
        
        snapshot = self._snapshots[-1]
        top_stats = snapshot.statistics('lineno')
        return top_stats[:limit]
    
    def get_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        with self._lock:
            self.update_metrics()
            return self._metrics
    
    def reset(self):
        """Reset memory profiler."""
        with self._lock:
            self._metrics = MemoryMetrics()
            self._baseline_memory = None
            self._snapshots.clear()


class CPUProfiler:
    """
    CPU profiler with detailed system metrics.
    Tracks CPU usage, load average, and system statistics.
    """
    
    def __init__(self, name: str, update_interval: float = 1.0):
        self.name = name
        self.update_interval = update_interval
        self._metrics = CPUMetrics()
        self._lock = threading.RLock()
        self._last_update = 0.0
        
        logger.debug(f"CPU profiler created", name=name, update_interval=update_interval)
    
    def update_metrics(self):
        """Update CPU metrics."""
        current_time = time.time()
        
        with self._lock:
            if current_time - self._last_update < self.update_interval:
                return
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_times = psutil.cpu_times_percent()
            
            self._metrics.usage_percent = cpu_percent
            self._metrics.user_time_percent = cpu_times.user
            self._metrics.system_time_percent = cpu_times.system
            self._metrics.idle_time_percent = cpu_times.idle
            
            # Load average
            if hasattr(psutil, 'getloadavg'):
                self._metrics.load_average = psutil.getloadavg()
            
            # System statistics
            process = psutil.Process()
            self._metrics.context_switches = process.num_ctx_switches().voluntary
            self._metrics.threads = process.num_threads()
            
            self._last_update = current_time
    
    def get_metrics(self) -> CPUMetrics:
        """Get current CPU metrics."""
        with self._lock:
            self.update_metrics()
            return self._metrics
    
    def reset(self):
        """Reset CPU profiler."""
        with self._lock:
            self._metrics = CPUMetrics()
            self._last_update = 0.0


class GPUProfiler:
    """
    GPU profiler with CUDA metrics.
    Tracks GPU utilization, memory, and thermal metrics.
    """
    
    def __init__(self, name: str, device_id: int = 0):
        self.name = name
        self.device_id = device_id
        self._metrics = GPUMetrics()
        self._lock = threading.RLock()
        self._gpu_available = torch.cuda.is_available()
        
        logger.debug(f"GPU profiler created", name=name, device_id=device_id, available=self._gpu_available)
    
    def update_metrics(self):
        """Update GPU metrics."""
        if not self._gpu_available:
            return
        
        with self._lock:
            # Basic CUDA metrics
            self._metrics.usage_percent = torch.cuda.utilization(self.device_id)
            
            # Memory metrics
            memory_allocated = torch.cuda.memory_allocated(self.device_id)
            memory_cached = torch.cuda.memory_cached(self.device_id)
            memory_total = torch.cuda.get_device_properties(self.device_id).total_memory
            
            self._metrics.memory_usage_percent = (memory_allocated / memory_total) * 100
            
            # Try to get additional metrics (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self._metrics.temperature_celsius = temp
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                self._metrics.power_watts = power
                
                # Fan speed
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    self._metrics.fan_speed_percent = fan_speed
                except (AttributeError, OSError, RuntimeError) as e:
                    # Fan speed not available on this GPU
                    pass
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._metrics.utilization_compute = util.gpu
                self._metrics.utilization_memory = util.memory
                
            except ImportError:
                # pynvml not available, use basic metrics
                pass
            except Exception as e:
                logger.debug(f"Failed to get additional GPU metrics", error=str(e))
    
    def get_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        with self._lock:
            self.update_metrics()
            return self._metrics
    
    def reset(self):
        """Reset GPU profiler."""
        with self._lock:
            self._metrics = GPUMetrics()


class PerformanceProfiler:
    """
    Comprehensive performance profiler with multiple profiling backends.
    Provides unified interface for different profiling tools.
    """
    
    def __init__(self, name: str, profiler_type: ProfilerType = ProfilerType.CPROFILE):
        self.name = name
        self.profiler_type = profiler_type
        self._profiler = None
        self._active = False
        self._results = None
        self._lock = threading.RLock()
        
        logger.debug(f"Performance profiler created", name=name, type=profiler_type.value)
    
    def start(self):
        """Start profiling."""
        with self._lock:
            if self._active:
                return
            
            if self.profiler_type == ProfilerType.CPROFILE:
                self._profiler = cProfile.Profile()
                self._profiler.enable()
            
            elif self.profiler_type == ProfilerType.LINE_PROFILER and LINE_PROFILER_AVAILABLE:
                self._profiler = line_profiler.LineProfiler()
                self._profiler.enable()
            
            elif self.profiler_type == ProfilerType.MEMORY_PROFILER and MEMORY_PROFILER_AVAILABLE:
                # Memory profiler doesn't have a start/stop mechanism
                pass
            
            self._active = True
            logger.debug(f"Profiler started", name=self.name)
    
    def stop(self):
        """Stop profiling."""
        with self._lock:
            if not self._active:
                return
            
            if self.profiler_type == ProfilerType.CPROFILE and self._profiler:
                self._profiler.disable()
            
            elif self.profiler_type == ProfilerType.LINE_PROFILER and self._profiler:
                self._profiler.disable()
            
            self._active = False
            logger.debug(f"Profiler stopped", name=self.name)
    
    def get_results(self) -> Optional[str]:
        """Get profiling results."""
        with self._lock:
            if not self._profiler:
                return None
            
            if self.profiler_type == ProfilerType.CPROFILE:
                s = io.StringIO()
                ps = pstats.Stats(self._profiler, stream=s).sort_stats('cumulative')
                ps.print_stats()
                return s.getvalue()
            
            elif self.profiler_type == ProfilerType.LINE_PROFILER:
                s = io.StringIO()
                self._profiler.print_stats(stream=s)
                return s.getvalue()
            
            return None
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """Profile a function call."""
        self.start()
        try:
            result = func(*args, **kwargs)
            return result, self.get_results()
        finally:
            self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    Aggregates all performance metrics and provides dashboard functionality.
    """
    
    def __init__(self, name: str = "performance_monitor"):
        self.name = name
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        self._throughput_monitors: Dict[str, ThroughputMonitor] = {}
        self._memory_profiler = MemoryProfiler(f"{name}_memory")
        self._cpu_profiler = CPUProfiler(f"{name}_cpu")
        self._gpu_profiler = GPUProfiler(f"{name}_gpu")
        self._custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))  # Reduced from 1000 to 500
        
        # Memory optimization components
        self._memory_manager = get_memory_manager()
        self._tensor_pool = self._memory_manager.tensor_pool
        
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread = None
        self._monitor_interval = 1.0
        self._cleanup_counter = 0
        self._cleanup_threshold = 100  # Clean up every 100 metric records
        
        logger.info(f"Performance monitor created", name=name)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring."""
        with self._lock:
            if self._monitoring_active:
                return
            
            self._monitor_interval = interval
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info(f"Performance monitoring started", interval=interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        with self._lock:
            if not self._monitoring_active:
                return
            
            self._monitoring_active = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop with memory optimization."""
        while self._monitoring_active:
            try:
                # Update system metrics
                self._memory_profiler.update_metrics()
                self._cpu_profiler.update_metrics()
                self._gpu_profiler.update_metrics()
                
                # Perform memory management
                self._memory_manager.periodic_cleanup()
                
                # Check for memory pressure and alert
                if self._memory_manager.check_memory_pressure():
                    logger.warning("Memory pressure detected, performing aggressive cleanup")
                    self._memory_manager.aggressive_cleanup()
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop", error=str(e))
                time.sleep(1.0)
    
    def get_latency_tracker(self, name: str) -> LatencyTracker:
        """Get or create latency tracker."""
        with self._lock:
            if name not in self._latency_trackers:
                self._latency_trackers[name] = LatencyTracker(name)
            return self._latency_trackers[name]
    
    def get_throughput_monitor(self, name: str) -> ThroughputMonitor:
        """Get or create throughput monitor."""
        with self._lock:
            if name not in self._throughput_monitors:
                self._throughput_monitors[name] = ThroughputMonitor(name)
            return self._throughput_monitors[name]
    
    def record_custom_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record custom metric with memory cleanup."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self._custom_metrics[name].append(metric)
            
            # Periodic cleanup to prevent memory growth
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_threshold:
                self._cleanup_old_custom_metrics()
                gc.collect()  # Force garbage collection
                self._cleanup_counter = 0
    
    def _cleanup_old_custom_metrics(self):
        """Clean up old custom metrics to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep only last hour of metrics
        
        for metric_name, metrics in self._custom_metrics.items():
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data with memory optimization stats."""
        with self._lock:
            dashboard = {
                'timestamp': time.time(),
                'system': {
                    'memory': self._memory_profiler.get_metrics().__dict__,
                    'cpu': self._cpu_profiler.get_metrics().__dict__,
                    'gpu': self._gpu_profiler.get_metrics().__dict__
                },
                'memory_optimization': self._memory_manager.get_memory_stats(),
                'latency': {
                    name: tracker.get_metrics().__dict__ 
                    for name, tracker in self._latency_trackers.items()
                },
                'throughput': {
                    name: monitor.get_metrics().__dict__ 
                    for name, monitor in self._throughput_monitors.items()
                },
                'custom_metrics': {
                    name: [metric.__dict__ for metric in metrics]
                    for name, metrics in self._custom_metrics.items()
                }
            }
            
            return dashboard
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        dashboard_data = self.get_dashboard_data()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported", filepath=filepath)
            
        except Exception as e:
            logger.error(f"Failed to export metrics", error=str(e))
    
    def reset_all_metrics(self):
        """Reset all metrics."""
        with self._lock:
            for tracker in self._latency_trackers.values():
                tracker.reset()
            
            for monitor in self._throughput_monitors.values():
                monitor.reset()
            
            self._memory_profiler.reset()
            self._cpu_profiler.reset()
            self._gpu_profiler.reset()
            
            self._custom_metrics.clear()
            
            logger.info("All metrics reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        dashboard = self.get_dashboard_data()
        
        summary = {
            'timestamp': dashboard['timestamp'],
            'system_health': {
                'memory_usage_percent': (
                    dashboard['system']['memory']['allocated_bytes'] / 
                    dashboard['system']['memory']['total_bytes'] * 100
                    if dashboard['system']['memory']['total_bytes'] > 0 else 0
                ),
                'cpu_usage_percent': dashboard['system']['cpu']['usage_percent'],
                'gpu_usage_percent': dashboard['system']['gpu']['usage_percent']
            },
            'latency_summary': {
                name: {
                    'avg_us': metrics['avg_ns'] / 1000,
                    'p95_us': metrics['p95_ns'] / 1000,
                    'p99_us': metrics['p99_ns'] / 1000
                }
                for name, metrics in dashboard['latency'].items()
            },
            'throughput_summary': {
                name: {
                    'ops_per_sec': metrics['operations_per_second'],
                    'bytes_per_sec': metrics['bytes_per_second']
                }
                for name, metrics in dashboard['throughput'].items()
            }
        }
        
        return summary
    
    def check_memory_alerts(self) -> List[Dict[str, Any]]:
        """Check for memory-related alerts."""
        alerts = []
        memory_stats = self._memory_manager.get_memory_stats()
        
        # Check memory pressure
        if memory_stats['memory_pressure']:
            alerts.append({
                'type': 'memory_pressure',
                'severity': 'high',
                'message': f"Memory usage at {memory_stats['current_memory_mb']:.1f}MB exceeds target of {memory_stats['target_memory_mb']}MB",
                'timestamp': time.time()
            })
        
        # Check for memory leaks
        if memory_stats['current_memory_mb'] > memory_stats['target_memory_mb'] * 1.2:
            alerts.append({
                'type': 'potential_memory_leak',
                'severity': 'critical',
                'message': f"Memory usage at {memory_stats['current_memory_mb']:.1f}MB is 20% above target, possible memory leak",
                'timestamp': time.time()
            })
        
        # Check tensor pool efficiency
        tensor_stats = memory_stats['tensor_pool_stats']
        if tensor_stats['total_cached_tensors'] > 1000:
            alerts.append({
                'type': 'tensor_pool_bloat',
                'severity': 'medium',
                'message': f"Tensor pool has {tensor_stats['total_cached_tensors']} cached tensors, consider reducing pool size",
                'timestamp': time.time()
            })
        
        return alerts
    
    def get_memory_optimization_report(self) -> Dict[str, Any]:
        """Get detailed memory optimization report."""
        memory_stats = self._memory_manager.get_memory_stats()
        
        report = {
            'timestamp': time.time(),
            'memory_usage': {
                'current_mb': memory_stats['current_memory_mb'],
                'target_mb': memory_stats['target_memory_mb'],
                'utilization_percent': (memory_stats['current_memory_mb'] / memory_stats['target_memory_mb']) * 100,
                'under_target': memory_stats['current_memory_mb'] < memory_stats['target_memory_mb']
            },
            'optimization_stats': {
                'tensor_pool_memory_saved_mb': memory_stats['tensor_pool_stats']['memory_saved_mb'],
                'gc_collections': sum(memory_stats['gc_counts']),
                'tensor_reuse_rate': (
                    memory_stats['tensor_pool_stats']['total_cached_tensors'] / 
                    max(1, sum(memory_stats['tensor_pool_stats']['usage_stats'].values()))
                )
            },
            'alerts': self.check_memory_alerts()
        }
        
        return report


# Global performance monitor
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def set_performance_monitor(monitor: PerformanceMonitor):
    """Set the global performance monitor."""
    global _global_performance_monitor
    if _global_performance_monitor is not None:
        _global_performance_monitor.stop_monitoring()
    _global_performance_monitor = monitor


# Convenience functions
def profile_function(func: Callable, *args, **kwargs) -> Tuple[Any, str]:
    """Profile a function call."""
    profiler = PerformanceProfiler(f"profile_{func.__name__}")
    return profiler.profile_function(func, *args, **kwargs)


def profile_context(name: str, profiler_type: ProfilerType = ProfilerType.CPROFILE):
    """Context manager for profiling."""
    return PerformanceProfiler(name, profiler_type)


def benchmark_function(func: Callable, *args, iterations: int = 1000, **kwargs) -> Dict[str, Any]:
    """Benchmark a function."""
    latency_tracker = LatencyTracker(f"benchmark_{func.__name__}")
    
    results = []
    for _ in range(iterations):
        start_time = time.time_ns()
        try:
            result = func(*args, **kwargs)
            end_time = time.time_ns()
            latency = end_time - start_time
            latency_tracker.record(latency)
            results.append(result)
        except Exception as e:
            logger.error(f"Benchmark iteration failed", error=str(e))
            continue
    
    metrics = latency_tracker.get_metrics()
    
    return {
        'function_name': func.__name__,
        'iterations': iterations,
        'successful_runs': len(results),
        'latency_metrics': metrics.__dict__,
        'throughput_ops_per_sec': 1_000_000_000 / metrics.avg_ns if metrics.avg_ns > 0 else 0,
        'results': results[:10]  # Sample results
    }


def start_performance_monitoring(interval: float = 1.0):
    """Start global performance monitoring."""
    get_performance_monitor().start_monitoring(interval)


def stop_performance_monitoring():
    """Stop global performance monitoring."""
    get_performance_monitor().stop_monitoring()


def get_performance_dashboard() -> Dict[str, Any]:
    """Get global performance dashboard."""
    return get_performance_monitor().get_dashboard_data()


def check_memory_health() -> Dict[str, Any]:
    """Check memory health and return optimization report."""
    return get_performance_monitor().get_memory_optimization_report()


def force_memory_cleanup():
    """Force aggressive memory cleanup."""
    get_memory_manager().aggressive_cleanup()


def get_tensor_pool() -> TensorPool:
    """Get the global tensor pool."""
    return get_memory_manager().tensor_pool


def export_performance_metrics(filepath: str):
    """Export performance metrics to file."""
    get_performance_monitor().export_metrics(filepath)