"""
Concurrency Performance Monitoring and Benchmarking
===================================================

This module provides comprehensive monitoring and benchmarking tools
for concurrency performance analysis.

Features:
- Lock contention monitoring
- Throughput analysis
- Performance benchmarking
- Concurrency metrics collection
- Real-time performance alerts

Author: Agent Beta - Race Condition Elimination Specialist
"""

import threading
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import structlog

from .atomic_operations import AtomicCounter, AtomicReference

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: float
    value: float
    metric_name: str
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentionEvent:
    """Lock contention event"""
    lock_id: str
    thread_id: str
    wait_time: float
    acquisition_time: float
    contention_level: int
    timestamp: float


@dataclass
class ThroughputMeasurement:
    """Throughput measurement"""
    operations_per_second: float
    total_operations: int
    measurement_duration: float
    timestamp: float
    operation_type: str


class LockContentionMonitor:
    """
    Monitor lock contention across the system
    """
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: deque[ContentionEvent] = deque(maxlen=max_events)
        self._lock_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_acquisitions': AtomicCounter(0),
            'total_wait_time': 0.0,
            'max_wait_time': 0.0,
            'contentions': AtomicCounter(0),
            'current_holders': set(),
            'waiting_threads': set()
        })
        self._global_lock = threading.RLock()
        
    def record_lock_acquisition(self, lock_id: str, thread_id: str, 
                              wait_time: float, acquisition_time: float):
        """Record a lock acquisition event"""
        with self._global_lock:
            stats = self._lock_stats[lock_id]
            stats['total_acquisitions'].increment()
            stats['total_wait_time'] += wait_time
            stats['max_wait_time'] = max(stats['max_wait_time'], wait_time)
            
            # Check for contention
            contention_level = len(stats['waiting_threads'])
            if contention_level > 0:
                stats['contentions'].increment()
                
                event = ContentionEvent(
                    lock_id=lock_id,
                    thread_id=thread_id,
                    wait_time=wait_time,
                    acquisition_time=acquisition_time,
                    contention_level=contention_level,
                    timestamp=time.time()
                )
                
                self._events.append(event)
                
                logger.debug("Lock contention detected", 
                           lock_id=lock_id,
                           thread_id=thread_id,
                           wait_time=wait_time,
                           contention_level=contention_level)
                           
            stats['current_holders'].add(thread_id)
            stats['waiting_threads'].discard(thread_id)
            
    def record_lock_wait(self, lock_id: str, thread_id: str):
        """Record a thread waiting for lock"""
        with self._global_lock:
            stats = self._lock_stats[lock_id]
            stats['waiting_threads'].add(thread_id)
            
    def record_lock_release(self, lock_id: str, thread_id: str):
        """Record a lock release"""
        with self._global_lock:
            stats = self._lock_stats[lock_id]
            stats['current_holders'].discard(thread_id)
            
    def get_contention_statistics(self) -> Dict[str, Any]:
        """Get comprehensive contention statistics"""
        with self._global_lock:
            stats = {}
            
            for lock_id, lock_stats in self._lock_stats.items():
                acquisitions = lock_stats['total_acquisitions'].get()
                contentions = lock_stats['contentions'].get()
                
                avg_wait_time = (lock_stats['total_wait_time'] / 
                               max(1, acquisitions))
                
                contention_rate = contentions / max(1, acquisitions)
                
                stats[lock_id] = {
                    'acquisitions': acquisitions,
                    'contentions': contentions,
                    'contention_rate': contention_rate,
                    'avg_wait_time': avg_wait_time,
                    'max_wait_time': lock_stats['max_wait_time'],
                    'current_holders': len(lock_stats['current_holders']),
                    'waiting_threads': len(lock_stats['waiting_threads'])
                }
                
            return stats
            
    def get_recent_contention_events(self, limit: int = 100) -> List[ContentionEvent]:
        """Get recent contention events"""
        with self._global_lock:
            return list(self._events)[-limit:]
            
    def get_top_contended_locks(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get most contended locks by contention rate"""
        stats = self.get_contention_statistics()
        
        contended_locks = [
            (lock_id, data['contention_rate'])
            for lock_id, data in stats.items()
            if data['contention_rate'] > 0
        ]
        
        return sorted(contended_locks, key=lambda x: x[1], reverse=True)[:limit]


class ThroughputAnalyzer:
    """
    Analyze system throughput and performance
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._measurements: Dict[str, deque[ThroughputMeasurement]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._operation_counts: Dict[str, AtomicCounter] = defaultdict(AtomicCounter)
        self._start_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def start_measurement(self, operation_type: str):
        """Start measuring throughput for operation type"""
        with self._lock:
            self._start_times[operation_type] = time.time()
            self._operation_counts[operation_type].set(0)
            
    def record_operation(self, operation_type: str):
        """Record completion of an operation"""
        self._operation_counts[operation_type].increment()
        
    def end_measurement(self, operation_type: str) -> Optional[ThroughputMeasurement]:
        """End measurement and calculate throughput"""
        with self._lock:
            if operation_type not in self._start_times:
                return None
                
            end_time = time.time()
            start_time = self._start_times[operation_type]
            duration = end_time - start_time
            
            if duration <= 0:
                return None
                
            total_ops = self._operation_counts[operation_type].get()
            ops_per_second = total_ops / duration
            
            measurement = ThroughputMeasurement(
                operations_per_second=ops_per_second,
                total_operations=total_ops,
                measurement_duration=duration,
                timestamp=end_time,
                operation_type=operation_type
            )
            
            self._measurements[operation_type].append(measurement)
            
            logger.info("Throughput measurement completed",
                       operation_type=operation_type,
                       ops_per_second=ops_per_second,
                       total_operations=total_ops,
                       duration=duration)
                       
            return measurement
            
    def get_throughput_statistics(self, operation_type: str) -> Dict[str, Any]:
        """Get throughput statistics for operation type"""
        with self._lock:
            measurements = self._measurements[operation_type]
            
            if not measurements:
                return {
                    'operation_type': operation_type,
                    'measurement_count': 0,
                    'avg_throughput': 0.0,
                    'max_throughput': 0.0,
                    'min_throughput': 0.0,
                    'std_throughput': 0.0
                }
                
            throughputs = [m.operations_per_second for m in measurements]
            
            return {
                'operation_type': operation_type,
                'measurement_count': len(measurements),
                'avg_throughput': statistics.mean(throughputs),
                'max_throughput': max(throughputs),
                'min_throughput': min(throughputs),
                'std_throughput': statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
                'latest_throughput': throughputs[-1] if throughputs else 0.0
            }
            
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operation types"""
        with self._lock:
            return {
                op_type: self.get_throughput_statistics(op_type)
                for op_type in self._measurements.keys()
            }


class ConcurrencyMetrics:
    """
    Collect and analyze concurrency metrics
    """
    
    def __init__(self, max_metrics: int = 100000):
        self.max_metrics = max_metrics
        self._metrics: deque[PerformanceMetric] = deque(maxlen=max_metrics)
        self._metric_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': AtomicCounter(0),
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'last_value': 0.0,
            'last_timestamp': 0.0
        })
        self._lock = threading.RLock()
        
        # Common metrics
        self._thread_count = AtomicCounter(0)
        self._active_locks = AtomicCounter(0)
        self._waiting_threads = AtomicCounter(0)
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        timestamp = time.time()
        tags = tags or {}
        
        metric = PerformanceMetric(
            timestamp=timestamp,
            value=value,
            metric_name=name,
            tags=tags
        )
        
        with self._lock:
            self._metrics.append(metric)
            
            stats = self._metric_stats[name]
            stats['count'].increment()
            stats['sum'] += value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['last_value'] = value
            stats['last_timestamp'] = timestamp
            
    def get_metric_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific metric"""
        with self._lock:
            stats = self._metric_stats[name]
            count = stats['count'].get()
            
            if count == 0:
                return {
                    'name': name,
                    'count': 0,
                    'average': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'last_value': 0.0,
                    'last_timestamp': 0.0
                }
                
            return {
                'name': name,
                'count': count,
                'average': stats['sum'] / count,
                'min': stats['min'],
                'max': stats['max'],
                'last_value': stats['last_value'],
                'last_timestamp': stats['last_timestamp']
            }
            
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics"""
        with self._lock:
            return {
                name: self.get_metric_statistics(name)
                for name in self._metric_stats.keys()
            }
            
    def get_recent_metrics(self, name: str, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics for a specific name"""
        with self._lock:
            return [
                metric for metric in list(self._metrics)[-limit:]
                if metric.metric_name == name
            ]
            
    def update_system_metrics(self):
        """Update system-level concurrency metrics"""
        # Update thread count
        thread_count = threading.active_count()
        self._thread_count.set(thread_count)
        self.record_metric('system.thread_count', thread_count)
        
        # Record other system metrics
        self.record_metric('system.active_locks', self._active_locks.get())
        self.record_metric('system.waiting_threads', self._waiting_threads.get())
        
    def increment_active_locks(self):
        """Increment active lock count"""
        self._active_locks.increment()
        
    def decrement_active_locks(self):
        """Decrement active lock count"""
        self._active_locks.decrement()
        
    def increment_waiting_threads(self):
        """Increment waiting thread count"""
        self._waiting_threads.increment()
        
    def decrement_waiting_threads(self):
        """Decrement waiting thread count"""
        self._waiting_threads.decrement()


class PerformanceBenchmarker:
    """
    Benchmark concurrency performance
    """
    
    def __init__(self):
        self.contention_monitor = LockContentionMonitor()
        self.throughput_analyzer = ThroughputAnalyzer()
        self.metrics = ConcurrencyMetrics()
        
    def benchmark_lock_performance(self, lock_factory: Callable, 
                                  num_threads: int = 10, 
                                  operations_per_thread: int = 1000) -> Dict[str, Any]:
        """Benchmark lock performance"""
        lock = lock_factory()
        results = []
        threads = []
        
        def worker(thread_id: int):
            thread_results = []
            
            for i in range(operations_per_thread):
                start_time = time.time()
                
                # Acquire lock
                with lock:
                    # Simulate work
                    time.sleep(0.001)
                    
                end_time = time.time()
                thread_results.append(end_time - start_time)
                
            results.extend(thread_results)
            
        # Start threads
        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        
        # Calculate statistics
        total_operations = num_threads * operations_per_thread
        total_time = end_time - start_time
        throughput = total_operations / total_time
        
        return {
            'lock_type': type(lock).__name__,
            'num_threads': num_threads,
            'operations_per_thread': operations_per_thread,
            'total_operations': total_operations,
            'total_time': total_time,
            'throughput': throughput,
            'avg_operation_time': statistics.mean(results),
            'min_operation_time': min(results),
            'max_operation_time': max(results),
            'std_operation_time': statistics.stdev(results) if len(results) > 1 else 0.0
        }
        
    def benchmark_queue_performance(self, queue_factory: Callable,
                                   num_producers: int = 5,
                                   num_consumers: int = 5,
                                   items_per_producer: int = 1000) -> Dict[str, Any]:
        """Benchmark queue performance"""
        queue = queue_factory()
        producer_times = []
        consumer_times = []
        threads = []
        
        def producer(producer_id: int):
            start_time = time.time()
            
            for i in range(items_per_producer):
                queue.put(f"item_{producer_id}_{i}")
                
            end_time = time.time()
            producer_times.append(end_time - start_time)
            
        def consumer(consumer_id: int):
            start_time = time.time()
            consumed = 0
            
            while consumed < items_per_producer:
                item = queue.get(timeout=1.0)
                if item is not None:
                    consumed += 1
                    
            end_time = time.time()
            consumer_times.append(end_time - start_time)
            
        # Start producers
        start_time = time.time()
        for i in range(num_producers):
            thread = threading.Thread(target=producer, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Start consumers
        for i in range(num_consumers):
            thread = threading.Thread(target=consumer, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        
        # Calculate statistics
        total_items = num_producers * items_per_producer
        total_time = end_time - start_time
        throughput = total_items / total_time
        
        return {
            'queue_type': type(queue).__name__,
            'num_producers': num_producers,
            'num_consumers': num_consumers,
            'items_per_producer': items_per_producer,
            'total_items': total_items,
            'total_time': total_time,
            'throughput': throughput,
            'avg_producer_time': statistics.mean(producer_times),
            'avg_consumer_time': statistics.mean(consumer_times),
            'max_producer_time': max(producer_times),
            'max_consumer_time': max(consumer_times)
        }
        
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive concurrency benchmark"""
        results = {
            'timestamp': time.time(),
            'system_info': {
                'thread_count': threading.active_count(),
                'processor_count': threading.cpu_count() if hasattr(threading, 'cpu_count') else 1
            },
            'lock_benchmarks': {},
            'queue_benchmarks': {},
            'contention_statistics': self.contention_monitor.get_contention_statistics(),
            'throughput_statistics': self.throughput_analyzer.get_all_statistics(),
            'metrics': self.metrics.get_all_metrics()
        }
        
        # Test different lock types
        lock_types = [
            ('RLock', threading.RLock),
            ('Lock', threading.Lock)
        ]
        
        for lock_name, lock_factory in lock_types:
            try:
                results['lock_benchmarks'][lock_name] = self.benchmark_lock_performance(
                    lock_factory, num_threads=5, operations_per_thread=100
                )
            except Exception as e:
                logger.error(f"Error benchmarking {lock_name}", error=str(e))
                
        return results
        
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        benchmark_results = self.run_comprehensive_benchmark()
        
        report = []
        report.append("=" * 80)
        report.append("CONCURRENCY PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {time.ctime(benchmark_results['timestamp'])}")
        report.append(f"System threads: {benchmark_results['system_info']['thread_count']}")
        report.append(f"Processor count: {benchmark_results['system_info']['processor_count']}")
        report.append("")
        
        # Lock benchmarks
        if benchmark_results['lock_benchmarks']:
            report.append("LOCK PERFORMANCE BENCHMARKS")
            report.append("-" * 40)
            
            for lock_name, stats in benchmark_results['lock_benchmarks'].items():
                report.append(f"{lock_name}:")
                report.append(f"  Throughput: {stats['throughput']:.2f} ops/sec")
                report.append(f"  Avg operation time: {stats['avg_operation_time']*1000:.2f}ms")
                report.append(f"  Max operation time: {stats['max_operation_time']*1000:.2f}ms")
                report.append("")
                
        # Contention statistics
        if benchmark_results['contention_statistics']:
            report.append("LOCK CONTENTION STATISTICS")
            report.append("-" * 40)
            
            for lock_id, stats in benchmark_results['contention_statistics'].items():
                report.append(f"{lock_id}:")
                report.append(f"  Acquisitions: {stats['acquisitions']}")
                report.append(f"  Contentions: {stats['contentions']}")
                report.append(f"  Contention rate: {stats['contention_rate']:.2%}")
                report.append(f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms")
                report.append("")
                
        report.append("=" * 80)
        
        return "\n".join(report)