"""
Nanosecond-Precision Timing Framework
====================================

Ultra-low latency timing framework for high-frequency trading systems.
Provides nanosecond-precision measurements with hardware-aware optimizations.
"""

import logging


import time
import threading
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import os
import psutil
import gc
from contextlib import contextmanager


@dataclass
class TimingResult:
    """Nanosecond-precision timing result"""
    operation: str
    duration_ns: int
    timestamp_ns: int
    cpu_id: int
    thread_id: int
    process_id: int
    memory_usage_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingStats:
    """Statistical analysis of timing measurements"""
    operation: str
    count: int
    min_ns: int
    max_ns: int
    mean_ns: float
    median_ns: float
    std_ns: float
    p95_ns: float
    p99_ns: float
    p999_ns: float
    total_duration_ns: int


class NanosecondTimer:
    """
    Ultra-low latency nanosecond-precision timer
    
    Features:
    - Nanosecond precision using time.perf_counter_ns()
    - Hardware-aware CPU affinity
    - Memory usage tracking
    - Statistical analysis
    - Thread-safe operations
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.results: Dict[str, deque] = {}
        self.lock = threading.RLock()
        self.process = psutil.Process()
        
        # Hardware optimization
        self._optimize_for_latency()
        
    def _optimize_for_latency(self):
        """Optimize system for ultra-low latency"""
        try:
            # Disable garbage collection during timing
            gc.disable()
            
            # Set high priority for current process
            self.process.nice(-10)  # Higher priority
            
            # CPU affinity for consistent timing
            if hasattr(os, 'sched_getaffinity'):
                available_cpus = list(os.sched_getaffinity(0))
                if available_cpus:
                    # Pin to first CPU for consistent timing
                    os.sched_setaffinity(0, [available_cpus[0]])
                    
        except (OSError, AttributeError):
            # Best effort optimization
            pass
    
    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for nanosecond-precision timing"""
        start_time = time.perf_counter_ns()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.perf_counter_ns()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            duration_ns = end_time - start_time
            
            result = TimingResult(
                operation=operation,
                duration_ns=duration_ns,
                timestamp_ns=start_time,
                cpu_id=os.getcpu()[0] if hasattr(os, 'getcpu') else 0,
                thread_id=threading.get_ident(),
                process_id=os.getpid(),
                memory_usage_mb=end_memory - start_memory,
                metadata=metadata or {}
            )
            
            self._store_result(result)
    
    def time_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Time a function call with nanosecond precision"""
        operation = f"{func.__name__}"
        
        with self.measure(operation):
            result = func(*args, **kwargs)
            
        return result, self.get_last_result(operation)
    
    def _store_result(self, result: TimingResult):
        """Store timing result in thread-safe manner"""
        with self.lock:
            if result.operation not in self.results:
                self.results[result.operation] = deque(maxlen=self.buffer_size)
            
            self.results[result.operation].append(result)
    
    def get_last_result(self, operation: str) -> Optional[TimingResult]:
        """Get the most recent timing result"""
        with self.lock:
            if operation in self.results and self.results[operation]:
                return self.results[operation][-1]
        return None
    
    def get_statistics(self, operation: str) -> Optional[TimingStats]:
        """Get comprehensive statistics for an operation"""
        with self.lock:
            if operation not in self.results or not self.results[operation]:
                return None
            
            durations = [r.duration_ns for r in self.results[operation]]
            
            if not durations:
                return None
            
            sorted_durations = sorted(durations)
            count = len(sorted_durations)
            
            return TimingStats(
                operation=operation,
                count=count,
                min_ns=min(sorted_durations),
                max_ns=max(sorted_durations),
                mean_ns=statistics.mean(sorted_durations),
                median_ns=statistics.median(sorted_durations),
                std_ns=statistics.stdev(sorted_durations) if count > 1 else 0,
                p95_ns=sorted_durations[int(0.95 * count)],
                p99_ns=sorted_durations[int(0.99 * count)],
                p999_ns=sorted_durations[int(0.999 * count)],
                total_duration_ns=sum(sorted_durations)
            )
    
    def get_all_statistics(self) -> Dict[str, TimingStats]:
        """Get statistics for all operations"""
        stats = {}
        with self.lock:
            for operation in self.results:
                stat = self.get_statistics(operation)
                if stat:
                    stats[operation] = stat
        return stats
    
    def clear_results(self, operation: Optional[str] = None):
        """Clear timing results"""
        with self.lock:
            if operation:
                if operation in self.results:
                    self.results[operation].clear()
            else:
                self.results.clear()
    
    def export_results(self, operation: str) -> List[Dict[str, Any]]:
        """Export timing results as dictionaries"""
        with self.lock:
            if operation not in self.results:
                return []
            
            return [
                {
                    'operation': r.operation,
                    'duration_ns': r.duration_ns,
                    'timestamp_ns': r.timestamp_ns,
                    'cpu_id': r.cpu_id,
                    'thread_id': r.thread_id,
                    'process_id': r.process_id,
                    'memory_usage_mb': r.memory_usage_mb,
                    'metadata': r.metadata
                }
                for r in self.results[operation]
            ]
    
    def benchmark_operation(self, operation: str, func: Callable, 
                          iterations: int = 1000) -> TimingStats:
        """Benchmark an operation multiple times"""
        for _ in range(iterations):
            with self.measure(operation):
                func()
        
        return self.get_statistics(operation)
    
    def validate_latency_requirements(self, operation: str, 
                                    max_latency_ns: int) -> Dict[str, Any]:
        """Validate that operation meets latency requirements"""
        stats = self.get_statistics(operation)
        if not stats:
            return {"valid": False, "reason": "No timing data available"}
        
        violations = []
        
        if stats.mean_ns > max_latency_ns:
            violations.append(f"Mean latency {stats.mean_ns}ns exceeds {max_latency_ns}ns")
        
        if stats.p95_ns > max_latency_ns:
            violations.append(f"P95 latency {stats.p95_ns}ns exceeds {max_latency_ns}ns")
        
        if stats.p99_ns > max_latency_ns * 2:  # Allow 2x for p99
            violations.append(f"P99 latency {stats.p99_ns}ns exceeds {max_latency_ns * 2}ns")
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "statistics": stats
        }
    
    def __del__(self):
        """Cleanup: re-enable garbage collection"""
        try:
            gc.enable()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')


class BatchTimer:
    """Batch timing for high-throughput operations"""
    
    def __init__(self, timer: NanosecondTimer):
        self.timer = timer
        self.batch_results: List[TimingResult] = []
        self.batch_lock = threading.Lock()
    
    def start_batch(self, operation: str):
        """Start timing a batch of operations"""
        self.current_operation = operation
        self.batch_start = time.perf_counter_ns()
    
    def record_operation(self, sub_operation: str = ""):
        """Record a single operation within the batch"""
        timestamp = time.perf_counter_ns()
        
        with self.batch_lock:
            result = TimingResult(
                operation=f"{self.current_operation}_{sub_operation}",
                duration_ns=timestamp - self.batch_start,
                timestamp_ns=timestamp,
                cpu_id=os.getcpu()[0] if hasattr(os, 'getcpu') else 0,
                thread_id=threading.get_ident(),
                process_id=os.getpid(),
                memory_usage_mb=0,  # Skip memory tracking for batch
                metadata={"batch_operation": True}
            )
            
            self.batch_results.append(result)
    
    def end_batch(self) -> List[TimingResult]:
        """End batch timing and return results"""
        with self.batch_lock:
            results = self.batch_results.copy()
            self.batch_results.clear()
            
            # Store in main timer
            for result in results:
                self.timer._store_result(result)
            
            return results