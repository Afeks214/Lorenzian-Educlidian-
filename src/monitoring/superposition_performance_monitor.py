"""
Real-time Performance Monitor for Universal Superposition System.

This module provides comprehensive real-time monitoring for the superposition
framework, ensuring that all components maintain performance targets (<5ms)
and system health requirements. It tracks:

- Superposition computation performance
- Agent coordination latency
- Memory usage and optimization
- Cascade integrity performance
- Real-time alerting and adaptive optimization

The monitor ensures continuous system health and performance compliance.
"""

import time
import threading
import queue
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class PerformanceLevel(Enum):
    """Performance alert levels."""
    OPTIMAL = "optimal"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringMode(Enum):
    """Monitoring operation modes."""
    REALTIME = "realtime"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    DIAGNOSTIC = "diagnostic"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str
    level: PerformanceLevel
    threshold_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'level': self.level.value,
            'threshold_ms': self.threshold_ms,
            'metadata': self.metadata
        }


@dataclass
class SystemHealthSnapshot:
    """Snapshot of system health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    active_threads: int
    queue_sizes: Dict[str, int]
    performance_metrics: List[PerformanceMetric]
    alerts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory_usage': self.gpu_memory_usage,
            'active_threads': self.active_threads,
            'queue_sizes': self.queue_sizes,
            'performance_metrics': [m.to_dict() for m in self.performance_metrics],
            'alerts': self.alerts
        }


class SuperpositionPerformanceMonitor:
    """
    Real-time performance monitor for the universal superposition system.
    
    This monitor ensures that all superposition operations maintain performance
    targets (<5ms) and provides real-time alerting, optimization recommendations,
    and system health monitoring.
    """
    
    def __init__(
        self,
        target_latency_ms: float = 5.0,
        monitoring_mode: MonitoringMode = MonitoringMode.REALTIME,
        alert_threshold_ms: float = 4.0,
        critical_threshold_ms: float = 8.0,
        max_history_size: int = 10000,
        sampling_interval_ms: float = 100.0
    ):
        """
        Initialize the superposition performance monitor.
        
        Args:
            target_latency_ms: Target latency for superposition operations
            monitoring_mode: Monitoring operation mode
            alert_threshold_ms: Threshold for performance alerts
            critical_threshold_ms: Threshold for critical alerts
            max_history_size: Maximum size of metric history
            sampling_interval_ms: Sampling interval for continuous monitoring
        """
        self.target_latency_ms = target_latency_ms
        self.monitoring_mode = monitoring_mode
        self.alert_threshold_ms = alert_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.max_history_size = max_history_size
        self.sampling_interval_ms = sampling_interval_ms
        
        # Setup logging
        self.logger = logging.getLogger('superposition_performance_monitor')
        self.logger.setLevel(logging.INFO)
        
        # Metric storage
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: deque = deque(maxlen=1000)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.metrics_queue: queue.Queue = queue.Queue()
        
        # Performance optimization
        self.performance_suggestions: List[str] = []
        self.optimization_callbacks: List[Callable] = []
        
        # System resource monitoring
        self.system_metrics: Dict[str, deque] = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_usage': deque(maxlen=100),
            'gpu_memory_usage': deque(maxlen=100)
        }
        
        # Thread safety
        self._lock = threading.Lock()
        self._alert_lock = threading.Lock()
        
        # Component registration
        self.registered_components: Dict[str, Dict[str, Any]] = {}
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, float] = {}
        
        self.logger.info(f"Superposition performance monitor initialized with "
                        f"target={target_latency_ms}ms, mode={monitoring_mode.value}")
    
    def register_component(
        self,
        component_name: str,
        expected_latency_ms: float,
        critical_path: bool = False,
        optimization_hints: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a component for monitoring.
        
        Args:
            component_name: Name of the component
            expected_latency_ms: Expected latency for this component
            critical_path: Whether this component is on the critical path
            optimization_hints: Hints for optimization
        """
        with self._lock:
            self.registered_components[component_name] = {
                'expected_latency_ms': expected_latency_ms,
                'critical_path': critical_path,
                'optimization_hints': optimization_hints or {},
                'registration_time': datetime.now(),
                'total_calls': 0,
                'total_time_ms': 0.0,
                'violations': 0
            }
            
            # Set adaptive threshold
            self.adaptive_thresholds[component_name] = expected_latency_ms
            
            self.logger.info(f"Registered component {component_name} with "
                           f"expected latency {expected_latency_ms}ms")
    
    @contextmanager
    def measure_performance(self, component_name: str, operation_name: str = "default"):
        """
        Context manager for measuring performance of operations.
        
        Args:
            component_name: Name of the component being measured
            operation_name: Name of the specific operation
            
        Yields:
            Performance measurement context
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            latency_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            self.record_metric(
                component_name=component_name,
                operation_name=operation_name,
                latency_ms=latency_ms,
                memory_delta_mb=memory_delta,
                timestamp=datetime.now()
            )
    
    def record_metric(
        self,
        component_name: str,
        operation_name: str,
        latency_ms: float,
        memory_delta_mb: float = 0.0,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a performance metric.
        
        Args:
            component_name: Name of the component
            operation_name: Name of the operation
            latency_ms: Latency in milliseconds
            memory_delta_mb: Memory usage change in MB
            timestamp: Timestamp of the measurement
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Determine performance level
        level = self._determine_performance_level(component_name, latency_ms)
        
        # Create metric
        metric = PerformanceMetric(
            name=f"{component_name}_{operation_name}",
            value=latency_ms,
            unit="ms",
            timestamp=timestamp,
            component=component_name,
            level=level,
            threshold_ms=self.adaptive_thresholds.get(component_name, self.target_latency_ms),
            metadata={
                'operation': operation_name,
                'memory_delta_mb': memory_delta_mb,
                **(metadata or {})
            }
        )
        
        # Store metric
        with self._lock:
            self.metrics_history.append(metric)
            self.component_metrics[component_name].append(metric)
            
            # Update component statistics
            if component_name in self.registered_components:
                comp_stats = self.registered_components[component_name]
                comp_stats['total_calls'] += 1
                comp_stats['total_time_ms'] += latency_ms
                
                if level in [PerformanceLevel.CRITICAL, PerformanceLevel.EMERGENCY]:
                    comp_stats['violations'] += 1
        
        # Add to monitoring queue for real-time processing
        if not self.metrics_queue.full():
            try:
                self.metrics_queue.put_nowait(metric)
            except queue.Full:
                pass  # Skip if queue is full
        
        # Check for alerts
        self._check_performance_alerts(metric)
        
        # Update adaptive thresholds
        if self.monitoring_mode == MonitoringMode.ADAPTIVE:
            self._update_adaptive_thresholds(component_name, latency_ms)
    
    def _determine_performance_level(self, component_name: str, latency_ms: float) -> PerformanceLevel:
        """Determine performance level based on latency."""
        threshold = self.adaptive_thresholds.get(component_name, self.target_latency_ms)
        
        if latency_ms <= threshold * 0.5:
            return PerformanceLevel.OPTIMAL
        elif latency_ms <= threshold * 0.8:
            return PerformanceLevel.GOOD
        elif latency_ms <= threshold:
            return PerformanceLevel.WARNING
        elif latency_ms <= threshold * 2:
            return PerformanceLevel.CRITICAL
        else:
            return PerformanceLevel.EMERGENCY
    
    def _check_performance_alerts(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any performance alerts."""
        alerts = []
        
        # Latency alerts
        if metric.value > self.critical_threshold_ms:
            alerts.append(f"CRITICAL: {metric.name} latency {metric.value:.2f}ms > {self.critical_threshold_ms}ms")
        elif metric.value > self.alert_threshold_ms:
            alerts.append(f"WARNING: {metric.name} latency {metric.value:.2f}ms > {self.alert_threshold_ms}ms")
        
        # Component-specific alerts
        if metric.component in self.registered_components:
            comp_info = self.registered_components[metric.component]
            expected_latency = comp_info['expected_latency_ms']
            
            if metric.value > expected_latency * 2:
                alerts.append(f"COMPONENT: {metric.component} severely underperforming "
                             f"({metric.value:.2f}ms vs expected {expected_latency:.2f}ms)")
            elif metric.value > expected_latency * 1.5:
                alerts.append(f"COMPONENT: {metric.component} underperforming "
                             f"({metric.value:.2f}ms vs expected {expected_latency:.2f}ms)")
        
        # Store alerts
        with self._alert_lock:
            for alert in alerts:
                self.alert_history.append({
                    'timestamp': metric.timestamp,
                    'level': metric.level.value,
                    'component': metric.component,
                    'message': alert
                })
                self.logger.warning(alert)
        
        # Trigger optimization if needed
        if metric.level in [PerformanceLevel.CRITICAL, PerformanceLevel.EMERGENCY]:
            self._trigger_optimization(metric)
    
    def _trigger_optimization(self, metric: PerformanceMetric) -> None:
        """Trigger optimization procedures for performance issues."""
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(metric)
        
        with self._lock:
            self.performance_suggestions.extend(suggestions)
            
            # Keep only recent suggestions
            if len(self.performance_suggestions) > 100:
                self.performance_suggestions = self.performance_suggestions[-50:]
        
        # Call optimization callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(metric, suggestions)
            except Exception as e:
                self.logger.error(f"Optimization callback failed: {e}")
    
    def _generate_optimization_suggestions(self, metric: PerformanceMetric) -> List[str]:
        """Generate optimization suggestions based on performance metric."""
        suggestions = []
        
        # General suggestions based on latency
        if metric.value > self.target_latency_ms * 3:
            suggestions.append(f"Consider algorithm optimization for {metric.component}")
            suggestions.append(f"Evaluate caching opportunities for {metric.component}")
        
        if metric.value > self.target_latency_ms * 2:
            suggestions.append(f"Review computational complexity in {metric.component}")
            suggestions.append(f"Consider parallel processing for {metric.component}")
        
        # Component-specific suggestions
        if metric.component in self.registered_components:
            comp_info = self.registered_components[metric.component]
            hints = comp_info.get('optimization_hints', {})
            
            if hints.get('parallelizable', False):
                suggestions.append(f"Enable parallel processing for {metric.component}")
            
            if hints.get('cacheable', False):
                suggestions.append(f"Implement caching for {metric.component}")
            
            if hints.get('gpu_accelerated', False) and TORCH_AVAILABLE:
                suggestions.append(f"Consider GPU acceleration for {metric.component}")
        
        # Memory-based suggestions
        memory_delta = metric.metadata.get('memory_delta_mb', 0)
        if memory_delta > 100:  # MB
            suggestions.append(f"High memory usage detected in {metric.component} - consider memory optimization")
        
        return suggestions
    
    def _update_adaptive_thresholds(self, component_name: str, latency_ms: float) -> None:
        """Update adaptive thresholds based on recent performance."""
        if component_name not in self.component_metrics:
            return
        
        # Get recent metrics for this component
        recent_metrics = list(self.component_metrics[component_name])[-50:]  # Last 50 measurements
        
        if len(recent_metrics) < 10:
            return  # Not enough data
        
        # Calculate adaptive threshold
        recent_latencies = [m.value for m in recent_metrics]
        p95_latency = np.percentile(recent_latencies, 95)
        mean_latency = np.mean(recent_latencies)
        
        # Adaptive threshold is between mean and p95
        adaptive_threshold = mean_latency + 0.5 * (p95_latency - mean_latency)
        
        # Don't let threshold go below target or above 2x target
        adaptive_threshold = max(self.target_latency_ms, 
                               min(adaptive_threshold, self.target_latency_ms * 2))
        
        self.adaptive_thresholds[component_name] = adaptive_threshold
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started real-time performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        self.logger.info("Stopped real-time performance monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Process metrics from queue
                self._process_metrics_queue()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Check for system-wide issues
                self._check_system_health()
                
                # Sleep for sampling interval
                time.sleep(self.sampling_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight loop on error
    
    def _process_metrics_queue(self) -> None:
        """Process metrics from the real-time queue."""
        processed = 0
        max_process = 100  # Limit processing per iteration
        
        while not self.metrics_queue.empty() and processed < max_process:
            try:
                metric = self.metrics_queue.get_nowait()
                # Additional real-time processing could go here
                processed += 1
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing metric: {e}")
    
    def _update_system_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            self.system_metrics['cpu_usage'].append(cpu_usage)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            self.system_metrics['memory_usage'].append(memory_usage)
            
            # GPU metrics (if available)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        self.system_metrics['gpu_usage'].append(gpu.load * 100)
                        self.system_metrics['gpu_memory_usage'].append(gpu.memoryUtil * 100)
                    else:
                        self.system_metrics['gpu_usage'].append(0)
                        self.system_metrics['gpu_memory_usage'].append(0)
                except Exception:
                    self.system_metrics['gpu_usage'].append(0)
                    self.system_metrics['gpu_memory_usage'].append(0)
            else:
                self.system_metrics['gpu_usage'].append(0)
                self.system_metrics['gpu_memory_usage'].append(0)
                
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    def _check_system_health(self) -> None:
        """Check overall system health."""
        # Check CPU usage
        if self.system_metrics['cpu_usage']:
            recent_cpu = list(self.system_metrics['cpu_usage'])[-10:]  # Last 10 measurements
            avg_cpu = np.mean(recent_cpu)
            
            if avg_cpu > 90:
                self._add_system_alert("CRITICAL: High CPU usage detected", "cpu_high")
            elif avg_cpu > 80:
                self._add_system_alert("WARNING: Elevated CPU usage", "cpu_elevated")
        
        # Check memory usage
        if self.system_metrics['memory_usage']:
            recent_memory = list(self.system_metrics['memory_usage'])[-10:]
            avg_memory = np.mean(recent_memory)
            
            if avg_memory > 95:
                self._add_system_alert("CRITICAL: High memory usage detected", "memory_high")
            elif avg_memory > 85:
                self._add_system_alert("WARNING: Elevated memory usage", "memory_elevated")
        
        # Check for performance degradation trends
        self._check_performance_trends()
    
    def _check_performance_trends(self) -> None:
        """Check for performance degradation trends."""
        for component_name, metrics in self.component_metrics.items():
            if len(metrics) < 20:  # Need sufficient data
                continue
            
            recent_metrics = list(metrics)[-20:]
            recent_latencies = [m.value for m in recent_metrics]
            
            # Check for upward trend
            if len(recent_latencies) >= 10:
                first_half = recent_latencies[:10]
                second_half = recent_latencies[10:]
                
                first_avg = np.mean(first_half)
                second_avg = np.mean(second_half)
                
                if second_avg > first_avg * 1.5:  # 50% increase
                    self._add_system_alert(
                        f"TREND: Performance degradation detected in {component_name}",
                        f"trend_{component_name}"
                    )
    
    def _add_system_alert(self, message: str, alert_type: str) -> None:
        """Add a system-wide alert."""
        with self._alert_lock:
            # Check if we already have this alert recently
            recent_alerts = [a for a in self.alert_history 
                           if a['timestamp'] > datetime.now() - timedelta(minutes=5)]
            
            # Don't duplicate recent alerts of same type
            if any(alert_type in a['message'] for a in recent_alerts):
                return
            
            self.alert_history.append({
                'timestamp': datetime.now(),
                'level': 'system',
                'component': 'system',
                'message': message,
                'alert_type': alert_type
            })
            
            self.logger.warning(message)
    
    def get_system_snapshot(self) -> SystemHealthSnapshot:
        """Get current system health snapshot."""
        with self._lock:
            # Get current system metrics
            cpu_usage = self.system_metrics['cpu_usage'][-1] if self.system_metrics['cpu_usage'] else 0
            memory_usage = self.system_metrics['memory_usage'][-1] if self.system_metrics['memory_usage'] else 0
            gpu_usage = self.system_metrics['gpu_usage'][-1] if self.system_metrics['gpu_usage'] else 0
            gpu_memory_usage = self.system_metrics['gpu_memory_usage'][-1] if self.system_metrics['gpu_memory_usage'] else 0
            
            # Get recent metrics
            recent_metrics = list(self.metrics_history)[-100:]
            
            # Get recent alerts
            recent_alerts = [a['message'] for a in self.alert_history 
                           if a['timestamp'] > datetime.now() - timedelta(minutes=5)]
            
            return SystemHealthSnapshot(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                active_threads=threading.active_count(),
                queue_sizes={'metrics': self.metrics_queue.qsize()},
                performance_metrics=recent_metrics,
                alerts=recent_alerts
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                'monitoring_active': self.monitoring_active,
                'monitoring_mode': self.monitoring_mode.value,
                'target_latency_ms': self.target_latency_ms,
                'total_metrics': len(self.metrics_history),
                'registered_components': len(self.registered_components),
                'recent_violations': 0,
                'component_statistics': {},
                'system_health': {},
                'optimization_suggestions': self.performance_suggestions[-10:],
                'alert_summary': {}
            }
            
            # Calculate component statistics
            for comp_name, comp_info in self.registered_components.items():
                if comp_name in self.component_metrics:
                    recent_metrics = list(self.component_metrics[comp_name])[-100:]
                    if recent_metrics:
                        latencies = [m.value for m in recent_metrics]
                        summary['component_statistics'][comp_name] = {
                            'total_calls': comp_info['total_calls'],
                            'violations': comp_info['violations'],
                            'violation_rate': comp_info['violations'] / max(1, comp_info['total_calls']),
                            'avg_latency_ms': np.mean(latencies),
                            'p95_latency_ms': np.percentile(latencies, 95),
                            'p99_latency_ms': np.percentile(latencies, 99),
                            'expected_latency_ms': comp_info['expected_latency_ms'],
                            'adaptive_threshold_ms': self.adaptive_thresholds.get(comp_name, 0),
                            'critical_path': comp_info['critical_path']
                        }
            
            # System health summary
            if self.system_metrics['cpu_usage']:
                summary['system_health'] = {
                    'avg_cpu_usage': np.mean(list(self.system_metrics['cpu_usage'])[-50:]),
                    'avg_memory_usage': np.mean(list(self.system_metrics['memory_usage'])[-50:]),
                    'avg_gpu_usage': np.mean(list(self.system_metrics['gpu_usage'])[-50:]),
                    'avg_gpu_memory_usage': np.mean(list(self.system_metrics['gpu_memory_usage'])[-50:])
                }
            
            # Alert summary
            recent_alerts = [a for a in self.alert_history 
                           if a['timestamp'] > datetime.now() - timedelta(hours=1)]
            summary['alert_summary'] = {
                'total_alerts_last_hour': len(recent_alerts),
                'critical_alerts_last_hour': len([a for a in recent_alerts if 'CRITICAL' in a['message']]),
                'warning_alerts_last_hour': len([a for a in recent_alerts if 'WARNING' in a['message']])
            }
            
            return summary
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """Add a callback for optimization triggers."""
        self.optimization_callbacks.append(callback)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        with self._lock:
            data = {
                'summary': self.get_performance_summary(),
                'system_snapshot': self.get_system_snapshot().to_dict(),
                'recent_metrics': [m.to_dict() for m in list(self.metrics_history)[-1000:]],
                'recent_alerts': [a for a in self.alert_history 
                                if a['timestamp'] > datetime.now() - timedelta(hours=24)]
            }
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics and statistics."""
        with self._lock:
            self.metrics_history.clear()
            self.component_metrics.clear()
            self.alert_history.clear()
            self.performance_suggestions.clear()
            
            # Reset component statistics
            for comp_info in self.registered_components.values():
                comp_info['total_calls'] = 0
                comp_info['total_time_ms'] = 0.0
                comp_info['violations'] = 0
            
            # Clear system metrics
            for metrics in self.system_metrics.values():
                metrics.clear()
            
            self.logger.info("Reset all performance metrics and statistics")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()