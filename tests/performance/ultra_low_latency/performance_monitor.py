"""
Real-time Performance Monitor
===========================

Provides real-time performance monitoring dashboard for ultra-low latency
systems with alerting and automated response capabilities.
"""

import time
import threading
import queue
import json
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import logging
from datetime import datetime, timedelta
from .nanosecond_timer import NanosecondTimer, TimingStats


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert structure"""
    alert_id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison_operator: str  # 'gt', 'lt', 'eq'
    window_size: int = 100
    enabled: bool = True


@dataclass
class MetricSnapshot:
    """Real-time metric snapshot"""
    metric_name: str
    value: float
    timestamp: datetime
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Real-time performance monitoring system
    
    Features:
    - Real-time metric collection
    - Configurable alerting thresholds
    - Dashboard data streaming
    - Automated response triggers
    - Historical data retention
    """
    
    def __init__(self, timer: NanosecondTimer, update_interval_ms: int = 100):
        self.timer = timer
        self.update_interval_ms = update_interval_ms
        self.update_interval_s = update_interval_ms / 1000.0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.current_metrics: Dict[str, MetricSnapshot] = {}
        
        # Alerting system
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Dashboard subscribers
        self.dashboard_subscribers: List[queue.Queue] = []
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Configure default thresholds
        self._configure_default_thresholds()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _configure_default_thresholds(self):
        """Configure default performance thresholds"""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="latency_mean_ns",
                warning_threshold=1000000,    # 1ms
                error_threshold=5000000,      # 5ms
                critical_threshold=10000000,  # 10ms
                comparison_operator='gt'
            ),
            PerformanceThreshold(
                metric_name="latency_p95_ns",
                warning_threshold=2000000,    # 2ms
                error_threshold=10000000,     # 10ms
                critical_threshold=20000000,  # 20ms
                comparison_operator='gt'
            ),
            PerformanceThreshold(
                metric_name="latency_p99_ns",
                warning_threshold=5000000,    # 5ms
                error_threshold=15000000,     # 15ms
                critical_threshold=30000000,  # 30ms
                comparison_operator='gt'
            ),
            PerformanceThreshold(
                metric_name="error_rate",
                warning_threshold=0.01,       # 1%
                error_threshold=0.05,         # 5%
                critical_threshold=0.1,       # 10%
                comparison_operator='gt'
            ),
            PerformanceThreshold(
                metric_name="throughput_ops_per_sec",
                warning_threshold=1000,       # Below 1000 ops/sec
                error_threshold=500,          # Below 500 ops/sec
                critical_threshold=100,       # Below 100 ops/sec
                comparison_operator='lt'
            )
        ]
        
        for threshold in default_thresholds:
            self.thresholds[threshold.metric_name] = threshold
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                
                # Update metrics history
                self._update_metrics_history(current_metrics)
                
                # Check thresholds and generate alerts
                self._check_thresholds(current_metrics)
                
                # Send updates to dashboard subscribers
                self._send_dashboard_updates(current_metrics)
                
                # Sleep until next update
                time.sleep(self.update_interval_s)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval_s)
    
    def _collect_metrics(self) -> Dict[str, MetricSnapshot]:
        """Collect current performance metrics"""
        metrics = {}
        timestamp = datetime.now()
        
        # Collect timing statistics from timer
        all_stats = self.timer.get_all_statistics()
        
        for operation, stats in all_stats.items():
            # Latency metrics
            metrics[f"{operation}_mean_ns"] = MetricSnapshot(
                metric_name=f"{operation}_mean_ns",
                value=stats.mean_ns,
                timestamp=timestamp,
                unit="nanoseconds",
                tags={"operation": operation}
            )
            
            metrics[f"{operation}_p95_ns"] = MetricSnapshot(
                metric_name=f"{operation}_p95_ns",
                value=stats.p95_ns,
                timestamp=timestamp,
                unit="nanoseconds",
                tags={"operation": operation}
            )
            
            metrics[f"{operation}_p99_ns"] = MetricSnapshot(
                metric_name=f"{operation}_p99_ns",
                value=stats.p99_ns,
                timestamp=timestamp,
                unit="nanoseconds",
                tags={"operation": operation}
            )
            
            # Throughput metrics
            if stats.total_duration_ns > 0:
                throughput = stats.count / (stats.total_duration_ns / 1e9)
                metrics[f"{operation}_throughput_ops_per_sec"] = MetricSnapshot(
                    metric_name=f"{operation}_throughput_ops_per_sec",
                    value=throughput,
                    timestamp=timestamp,
                    unit="operations_per_second",
                    tags={"operation": operation}
                )
            
            # Count metrics
            metrics[f"{operation}_count"] = MetricSnapshot(
                metric_name=f"{operation}_count",
                value=stats.count,
                timestamp=timestamp,
                unit="count",
                tags={"operation": operation}
            )
        
        # System metrics
        metrics["timestamp"] = MetricSnapshot(
            metric_name="timestamp",
            value=time.time(),
            timestamp=timestamp,
            unit="seconds",
            tags={"system": "monitor"}
        )
        
        return metrics
    
    def _update_metrics_history(self, metrics: Dict[str, MetricSnapshot]):
        """Update metrics history"""
        with self.lock:
            for metric_name, snapshot in metrics.items():
                self.metrics_history[metric_name].append(snapshot)
                self.current_metrics[metric_name] = snapshot
    
    def _check_thresholds(self, metrics: Dict[str, MetricSnapshot]):
        """Check performance thresholds and generate alerts"""
        for metric_name, snapshot in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                if not threshold.enabled:
                    continue
                
                # Determine alert level
                alert_level = self._determine_alert_level(snapshot.value, threshold)
                
                if alert_level:
                    # Create alert
                    alert = PerformanceAlert(
                        alert_id=f"{metric_name}_{int(time.time())}",
                        level=alert_level,
                        message=f"{metric_name} threshold exceeded",
                        metric_name=metric_name,
                        threshold_value=self._get_threshold_value(alert_level, threshold),
                        actual_value=snapshot.value,
                        timestamp=snapshot.timestamp,
                        metadata={
                            "operation": snapshot.tags.get("operation", "unknown"),
                            "unit": snapshot.unit
                        }
                    )
                    
                    # Process alert
                    self._process_alert(alert)
    
    def _determine_alert_level(self, value: float, threshold: PerformanceThreshold) -> Optional[AlertLevel]:
        """Determine alert level based on value and threshold"""
        if threshold.comparison_operator == 'gt':
            if value >= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif value >= threshold.error_threshold:
                return AlertLevel.ERROR
            elif value >= threshold.warning_threshold:
                return AlertLevel.WARNING
        elif threshold.comparison_operator == 'lt':
            if value <= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif value <= threshold.error_threshold:
                return AlertLevel.ERROR
            elif value <= threshold.warning_threshold:
                return AlertLevel.WARNING
        
        return None
    
    def _get_threshold_value(self, level: AlertLevel, threshold: PerformanceThreshold) -> float:
        """Get threshold value for alert level"""
        if level == AlertLevel.CRITICAL:
            return threshold.critical_threshold
        elif level == AlertLevel.ERROR:
            return threshold.error_threshold
        elif level == AlertLevel.WARNING:
            return threshold.warning_threshold
        return 0.0
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process performance alert"""
        with self.lock:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Log alert
            self.logger.log(
                self._get_log_level(alert.level),
                f"Performance Alert: {alert.message} - "
                f"Metric: {alert.metric_name}, "
                f"Threshold: {alert.threshold_value}, "
                f"Actual: {alert.actual_value}"
            )
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def _get_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level"""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)
    
    def _send_dashboard_updates(self, metrics: Dict[str, MetricSnapshot]):
        """Send updates to dashboard subscribers"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {name: asdict(snapshot) for name, snapshot in metrics.items()},
            'alerts': [asdict(alert) for alert in self.active_alerts.values()],
            'system_status': 'healthy' if not self.active_alerts else 'degraded'
        }
        
        # Send to all subscribers
        for subscriber_queue in self.dashboard_subscribers[:]:  # Copy list to avoid modification during iteration
            try:
                subscriber_queue.put_nowait(dashboard_data)
            except queue.Full:
                # Remove full queues
                self.dashboard_subscribers.remove(subscriber_queue)
            except Exception as e:
                self.logger.error(f"Error sending dashboard update: {e}")
    
    def subscribe_to_dashboard(self) -> queue.Queue:
        """Subscribe to dashboard updates"""
        subscriber_queue = queue.Queue(maxsize=100)
        self.dashboard_subscribers.append(subscriber_queue)
        return subscriber_queue
    
    def unsubscribe_from_dashboard(self, subscriber_queue: queue.Queue):
        """Unsubscribe from dashboard updates"""
        if subscriber_queue in self.dashboard_subscribers:
            self.dashboard_subscribers.remove(subscriber_queue)
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold"""
        with self.lock:
            self.thresholds[threshold.metric_name] = threshold
    
    def remove_threshold(self, metric_name: str):
        """Remove performance threshold"""
        with self.lock:
            if metric_name in self.thresholds:
                del self.thresholds[metric_name]
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Remove alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def resolve_alert(self, alert_id: str):
        """Resolve active alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                del self.active_alerts[alert_id]
    
    def get_current_metrics(self) -> Dict[str, MetricSnapshot]:
        """Get current metrics snapshot"""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_metric_history(self, metric_name: str, 
                          last_n_points: Optional[int] = None) -> List[MetricSnapshot]:
        """Get metric history"""
        with self.lock:
            if metric_name not in self.metrics_history:
                return []
            
            history = list(self.metrics_history[metric_name])
            
            if last_n_points:
                return history[-last_n_points:]
            
            return history
    
    def get_active_alerts(self) -> Dict[str, PerformanceAlert]:
        """Get active alerts"""
        with self.lock:
            return self.active_alerts.copy()
    
    def get_alert_history(self, last_n_alerts: Optional[int] = None) -> List[PerformanceAlert]:
        """Get alert history"""
        with self.lock:
            history = list(self.alert_history)
            
            if last_n_alerts:
                return history[-last_n_alerts:]
            
            return history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            current_metrics = self.get_current_metrics()
            active_alerts = self.get_active_alerts()
            
            # Calculate aggregated metrics
            latency_metrics = [m for name, m in current_metrics.items() if 'latency' in name]
            throughput_metrics = [m for name, m in current_metrics.items() if 'throughput' in name]
            
            avg_latency = statistics.mean([m.value for m in latency_metrics]) if latency_metrics else 0
            total_throughput = sum([m.value for m in throughput_metrics])
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'healthy' if not active_alerts else 'degraded',
                'metrics': {
                    'average_latency_ns': avg_latency,
                    'total_throughput_ops_per_sec': total_throughput,
                    'active_alerts_count': len(active_alerts),
                    'total_operations': sum([
                        m.value for name, m in current_metrics.items() 
                        if name.endswith('_count')
                    ])
                },
                'alerts': {
                    'active': len(active_alerts),
                    'by_level': {
                        level.value: len([a for a in active_alerts.values() if a.level == level])
                        for level in AlertLevel
                    }
                },
                'thresholds': {
                    'configured': len(self.thresholds),
                    'enabled': len([t for t in self.thresholds.values() if t.enabled])
                }
            }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        summary = self.get_performance_summary()
        
        if format.lower() == 'json':
            return json.dumps(summary, indent=2, default=str)
        elif format.lower() == 'csv':
            # Basic CSV export
            lines = ["metric_name,value,timestamp,unit"]
            for name, metric in self.current_metrics.items():
                lines.append(f"{name},{metric.value},{metric.timestamp},{metric.unit}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_dashboard_snapshot(self) -> Dict[str, Any]:
        """Create dashboard snapshot for external visualization"""
        return {
            'performance_summary': self.get_performance_summary(),
            'current_metrics': {
                name: asdict(metric) for name, metric in self.current_metrics.items()
            },
            'active_alerts': {
                alert_id: asdict(alert) for alert_id, alert in self.active_alerts.items()
            },
            'thresholds': {
                name: asdict(threshold) for name, threshold in self.thresholds.items()
            }
        }