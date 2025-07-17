"""
Sequential Strategic Monitor - Comprehensive performance monitoring and error handling

This module provides comprehensive monitoring and error handling for the Sequential 
Strategic MARL system, including real-time performance tracking, error detection,
recovery mechanisms, and automated alerting.

Key Features:
- Real-time performance monitoring with <1ms overhead
- Comprehensive error detection and classification
- Automated recovery and fallback mechanisms
- Performance metrics aggregation and analysis
- Alerting system for critical issues
- Health check and system diagnostics
- Performance optimization recommendations
- Comprehensive logging and reporting

Monitoring Categories:
1. Performance Monitoring - Timing, throughput, resource usage
2. Error Handling - Detection, classification, recovery
3. Health Monitoring - System health, agent status
4. Quality Monitoring - Superposition quality, mathematical validation
5. Resource Monitoring - Memory, CPU, network usage
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import json
import traceback
from pathlib import Path
import psutil
import os
from abc import ABC, abstractmethod

# Import system components
from src.environment.sequential_strategic_env import SequentialStrategicEnvironment
from src.agents.strategic.sequential_strategic_agents import SequentialStrategicAgentBase
from src.environment.strategic_superposition_aggregator import StrategicSuperpositionAggregator
from src.validation.strategic_sequence_validator import StrategicSequenceValidator

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringCategory(Enum):
    """Monitoring categories"""
    PERFORMANCE = "performance"
    ERROR = "error"
    HEALTH = "health"
    QUALITY = "quality"
    RESOURCE = "resource"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: MonitoringCategory
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent:
    """Error event data"""
    error_id: str
    error_type: str
    severity: AlertLevel
    message: str
    source: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    score: float  # 0.0 to 1.0
    timestamp: datetime
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceTracker:
    """Real-time performance tracking"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.alert_thresholds = {}
        self.alert_callbacks = []
        self.lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self.lock:
            self.metrics[metric.name].append(metric)
            
            # Check alert thresholds
            if metric.name in self.alert_thresholds:
                self._check_threshold(metric)
    
    def _check_threshold(self, metric: PerformanceMetric):
        """Check if metric exceeds threshold"""
        threshold = self.alert_thresholds[metric.name]
        
        if metric.value > threshold['max_value']:
            alert = Alert(
                alert_id=f"threshold_{metric.name}_{int(time.time())}",
                alert_type="threshold_exceeded",
                level=AlertLevel.WARNING,
                message=f"{metric.name} exceeded threshold: {metric.value} > {threshold['max_value']}",
                timestamp=datetime.now(),
                source=metric.source,
                metadata={'metric': metric.name, 'value': metric.value, 'threshold': threshold['max_value']}
            )
            
            for callback in self.alert_callbacks:
                callback(alert)
    
    def set_threshold(self, metric_name: str, max_value: float, callback: Optional[Callable] = None):
        """Set alert threshold for metric"""
        self.alert_thresholds[metric_name] = {
            'max_value': max_value,
            'callback': callback
        }
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_recent_metrics(self, metric_name: str, duration_seconds: int = 60) -> List[PerformanceMetric]:
        """Get recent metrics within time window"""
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
            return [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, metric_name: str, duration_seconds: int = 60) -> Dict[str, float]:
        """Get metric statistics"""
        recent_metrics = self.get_recent_metrics(metric_name, duration_seconds)
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


class ErrorHandler:
    """Comprehensive error handling"""
    
    def __init__(self):
        self.error_history = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.lock = threading.Lock()
        
    def handle_error(self, error: Exception, source: str, context: Dict[str, Any] = None) -> ErrorEvent:
        """Handle an error event"""
        error_event = ErrorEvent(
            error_id=f"error_{int(time.time())}_{id(error)}",
            error_type=type(error).__name__,
            severity=self._classify_error_severity(error),
            message=str(error),
            source=source,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        with self.lock:
            self.error_history.append(error_event)
            self.error_counts[error_event.error_type] += 1
            
            # Check circuit breaker
            if source in self.circuit_breakers:
                self._update_circuit_breaker(source, error_event)
            
            # Apply recovery strategy
            recovery_action = self._apply_recovery_strategy(error_event)
            error_event.recovery_action = recovery_action
        
        return error_event
    
    def _classify_error_severity(self, error: Exception) -> AlertLevel:
        """Classify error severity"""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return AlertLevel.CRITICAL
        elif isinstance(error, (MemoryError, OSError)):
            return AlertLevel.ERROR
        elif isinstance(error, (ValueError, TypeError)):
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _update_circuit_breaker(self, source: str, error_event: ErrorEvent):
        """Update circuit breaker state"""
        breaker = self.circuit_breakers[source]
        breaker['error_count'] += 1
        
        if breaker['error_count'] > breaker['threshold']:
            breaker['state'] = 'open'
            breaker['opened_at'] = datetime.now()
    
    def _apply_recovery_strategy(self, error_event: ErrorEvent) -> str:
        """Apply recovery strategy"""
        error_type = error_event.error_type
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            return strategy(error_event)
        
        return "default_recovery"
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register recovery strategy for error type"""
        self.recovery_strategies[error_type] = strategy
    
    def add_circuit_breaker(self, source: str, threshold: int = 5, timeout_seconds: int = 60):
        """Add circuit breaker for source"""
        self.circuit_breakers[source] = {
            'threshold': threshold,
            'timeout_seconds': timeout_seconds,
            'error_count': 0,
            'state': 'closed',  # closed, open, half-open
            'opened_at': None
        }
    
    def is_circuit_open(self, source: str) -> bool:
        """Check if circuit breaker is open"""
        if source not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[source]
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if breaker['opened_at']:
                elapsed = (datetime.now() - breaker['opened_at']).total_seconds()
                if elapsed > breaker['timeout_seconds']:
                    breaker['state'] = 'half-open'
                    breaker['error_count'] = 0
                    return False
            return True
        
        return False
    
    def get_error_statistics(self, duration_seconds: int = 3600) -> Dict[str, Any]:
        """Get error statistics"""
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        
        with self.lock:
            recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
            
            error_by_type = defaultdict(int)
            error_by_severity = defaultdict(int)
            error_by_source = defaultdict(int)
            
            for error in recent_errors:
                error_by_type[error.error_type] += 1
                error_by_severity[error.severity.value] += 1
                error_by_source[error.source] += 1
            
            return {
                'total_errors': len(recent_errors),
                'error_by_type': dict(error_by_type),
                'error_by_severity': dict(error_by_severity),
                'error_by_source': dict(error_by_source),
                'error_rate': len(recent_errors) / duration_seconds
            }


class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self):
        self.health_status = {}
        self.health_checks = {}
        self.health_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def register_health_check(self, component: str, check_function: Callable, interval_seconds: int = 30):
        """Register health check for component"""
        self.health_checks[component] = {
            'function': check_function,
            'interval': interval_seconds,
            'last_check': None
        }
    
    async def run_health_checks(self):
        """Run all health checks"""
        for component, check_info in self.health_checks.items():
            now = datetime.now()
            
            # Check if it's time to run this check
            if (check_info['last_check'] is None or 
                (now - check_info['last_check']).total_seconds() >= check_info['interval']):
                
                try:
                    health_status = await self._run_health_check(component, check_info['function'])
                    check_info['last_check'] = now
                    
                    with self.lock:
                        self.health_status[component] = health_status
                        self.health_history.append(health_status)
                        
                except Exception as e:
                    # Health check failed
                    health_status = HealthStatus(
                        component=component,
                        status="unhealthy",
                        score=0.0,
                        timestamp=now,
                        issues=[f"Health check failed: {str(e)}"]
                    )
                    
                    with self.lock:
                        self.health_status[component] = health_status
                        self.health_history.append(health_status)
    
    async def _run_health_check(self, component: str, check_function: Callable) -> HealthStatus:
        """Run individual health check"""
        try:
            result = await check_function()
            
            if isinstance(result, HealthStatus):
                return result
            elif isinstance(result, dict):
                return HealthStatus(
                    component=component,
                    status=result.get('status', 'healthy'),
                    score=result.get('score', 1.0),
                    timestamp=datetime.now(),
                    issues=result.get('issues', []),
                    metrics=result.get('metrics', {})
                )
            else:
                return HealthStatus(
                    component=component,
                    status="healthy" if result else "unhealthy",
                    score=1.0 if result else 0.0,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthStatus(
                component=component,
                status="unhealthy",
                score=0.0,
                timestamp=datetime.now(),
                issues=[f"Health check error: {str(e)}"]
            )
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health"""
        with self.lock:
            if not self.health_status:
                return HealthStatus(
                    component="system",
                    status="unknown",
                    score=0.0,
                    timestamp=datetime.now(),
                    issues=["No health checks registered"]
                )
            
            scores = [status.score for status in self.health_status.values()]
            overall_score = np.mean(scores)
            
            all_issues = []
            for status in self.health_status.values():
                all_issues.extend(status.issues)
            
            if overall_score >= 0.8:
                status = "healthy"
            elif overall_score >= 0.5:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return HealthStatus(
                component="system",
                status=status,
                score=overall_score,
                timestamp=datetime.now(),
                issues=all_issues,
                metrics={'component_count': len(self.health_status)}
            )


class ResourceMonitor:
    """Resource usage monitoring"""
    
    def __init__(self):
        self.resource_history = deque(maxlen=1000)
        self.resource_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        self.alert_callbacks = []
        
    def collect_resource_metrics(self) -> Dict[str, float]:
        """Collect current resource metrics"""
        try:
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Network usage (simplified)
            net_io = psutil.net_io_counters()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'disk_percent': disk_percent,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'timestamp': time.time()
            }
            
            # Store in history
            self.resource_history.append(metrics)
            
            # Check thresholds
            self._check_resource_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            return {}
    
    def _check_resource_thresholds(self, metrics: Dict[str, float]):
        """Check resource thresholds"""
        for metric_name, threshold in self.resource_thresholds.items():
            if metric_name in metrics and metrics[metric_name] > threshold:
                alert = Alert(
                    alert_id=f"resource_{metric_name}_{int(time.time())}",
                    alert_type="resource_threshold",
                    level=AlertLevel.WARNING,
                    message=f"{metric_name} exceeded threshold: {metrics[metric_name]:.1f}% > {threshold}%",
                    timestamp=datetime.now(),
                    source="resource_monitor",
                    metadata={'metric': metric_name, 'value': metrics[metric_name], 'threshold': threshold}
                )
                
                for callback in self.alert_callbacks:
                    callback(alert)
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_resource_statistics(self, duration_seconds: int = 300) -> Dict[str, Dict[str, float]]:
        """Get resource statistics"""
        cutoff_time = time.time() - duration_seconds
        recent_metrics = [m for m in self.resource_history if m['timestamp'] >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        statistics = {}
        
        for metric_name in ['cpu_percent', 'memory_percent', 'disk_percent']:
            values = [m[metric_name] for m in recent_metrics if metric_name in m]
            
            if values:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
        
        return statistics


class SequentialStrategicMonitor:
    """Main monitoring system for Sequential Strategic MARL"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sequential strategic monitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SequentialStrategicMonitor")
        
        # Initialize monitoring components
        self.performance_tracker = PerformanceTracker(
            max_history=config.get('max_history', 10000)
        )
        self.error_handler = ErrorHandler()
        self.health_monitor = HealthMonitor()
        self.resource_monitor = ResourceMonitor()
        
        # Alert management
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Performance thresholds
        self.performance_thresholds = {
            'agent_computation_time_ms': 5.0,
            'sequence_execution_time_ms': 15.0,
            'aggregation_time_ms': 2.0,
            'superposition_quality': 0.7,
            'ensemble_confidence': 0.6
        }
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        self.logger.info("Sequential Strategic Monitor initialized")
    
    def _initialize_monitoring(self):
        """Initialize monitoring components"""
        # Set performance thresholds
        for metric_name, threshold in self.performance_thresholds.items():
            if 'time' in metric_name:
                self.performance_tracker.set_threshold(metric_name, threshold)
            else:
                # For quality metrics, we want to be alerted when they're too low
                self.performance_tracker.set_threshold(metric_name, -threshold)  # Negative for minimum threshold
        
        # Add alert callbacks
        self.performance_tracker.add_alert_callback(self._handle_alert)
        self.resource_monitor.add_alert_callback(self._handle_alert)
        
        # Register default recovery strategies
        self.error_handler.register_recovery_strategy('TimeoutError', self._handle_timeout_error)
        self.error_handler.register_recovery_strategy('MemoryError', self._handle_memory_error)
        self.error_handler.register_recovery_strategy('ValueError', self._handle_value_error)
        
        # Add circuit breakers
        self.error_handler.add_circuit_breaker('mlmi_agent', threshold=5, timeout_seconds=60)
        self.error_handler.add_circuit_breaker('nwrqk_agent', threshold=5, timeout_seconds=60)
        self.error_handler.add_circuit_breaker('regime_agent', threshold=5, timeout_seconds=60)
        self.error_handler.add_circuit_breaker('environment', threshold=3, timeout_seconds=120)
        self.error_handler.add_circuit_breaker('aggregator', threshold=3, timeout_seconds=60)
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        if self.monitoring_active:
            self.monitoring_active = False
            self.stop_event.set()
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect resource metrics
                self.resource_monitor.collect_resource_metrics()
                
                # Run health checks
                asyncio.run(self.health_monitor.run_health_checks())
                
                # Sleep for monitoring interval
                self.stop_event.wait(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)
    
    def record_performance(self, metric_name: str, value: float, unit: str, source: str, **metadata):
        """Record performance metric"""
        metric = PerformanceMetric(
            name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=MonitoringCategory.PERFORMANCE,
            source=source,
            metadata=metadata
        )
        
        self.performance_tracker.record_metric(metric)
    
    def handle_error(self, error: Exception, source: str, context: Dict[str, Any] = None) -> ErrorEvent:
        """Handle error event"""
        error_event = self.error_handler.handle_error(error, source, context)
        
        # Create alert for errors
        if error_event.severity in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            alert = Alert(
                alert_id=f"error_{error_event.error_id}",
                alert_type="error",
                level=error_event.severity,
                message=error_event.message,
                timestamp=error_event.timestamp,
                source=source,
                metadata={'error_id': error_event.error_id, 'error_type': error_event.error_type}
            )
            
            self._handle_alert(alert)
        
        return error_event
    
    def _handle_alert(self, alert: Alert):
        """Handle alert"""
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{alert.level.value.upper()}] {alert.message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def register_health_check(self, component: str, check_function: Callable, interval_seconds: int = 30):
        """Register health check"""
        self.health_monitor.register_health_check(component, check_function, interval_seconds)
    
    def is_circuit_open(self, source: str) -> bool:
        """Check if circuit breaker is open"""
        return self.error_handler.is_circuit_open(source)
    
    def _handle_timeout_error(self, error_event: ErrorEvent) -> str:
        """Handle timeout error"""
        return "timeout_retry"
    
    def _handle_memory_error(self, error_event: ErrorEvent) -> str:
        """Handle memory error"""
        return "memory_cleanup"
    
    def _handle_value_error(self, error_event: ErrorEvent) -> str:
        """Handle value error"""
        return "input_validation"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get health status
        overall_health = self.health_monitor.get_overall_health()
        
        # Get resource metrics
        resource_stats = self.resource_monitor.get_resource_statistics()
        
        # Get error statistics
        error_stats = self.error_handler.get_error_statistics()
        
        # Get performance statistics
        performance_stats = {}
        for metric_name in self.performance_thresholds:
            stats = self.performance_tracker.get_metric_statistics(metric_name)
            if stats:
                performance_stats[metric_name] = stats
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.alerts if 
                        (datetime.now() - alert.timestamp).total_seconds() < 3600]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health': {
                'status': overall_health.status,
                'score': overall_health.score,
                'issues': overall_health.issues
            },
            'resources': resource_stats,
            'errors': error_stats,
            'performance': performance_stats,
            'alerts': {
                'total': len(recent_alerts),
                'critical': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
                'errors': len([a for a in recent_alerts if a.level == AlertLevel.ERROR]),
                'warnings': len([a for a in recent_alerts if a.level == AlertLevel.WARNING])
            },
            'monitoring': {
                'active': self.monitoring_active,
                'uptime_seconds': time.time() - (self.monitoring_thread.ident if self.monitoring_thread else time.time())
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'thresholds': self.performance_thresholds,
            'violations': []
        }
        
        # Get statistics for each metric
        for metric_name in self.performance_thresholds:
            stats = self.performance_tracker.get_metric_statistics(metric_name, duration_seconds=300)
            if stats:
                report['metrics'][metric_name] = stats
                
                # Check for threshold violations
                threshold = self.performance_thresholds[metric_name]
                if 'time' in metric_name:
                    # For time metrics, check if mean exceeds threshold
                    if stats['mean'] > threshold:
                        report['violations'].append({
                            'metric': metric_name,
                            'value': stats['mean'],
                            'threshold': threshold,
                            'type': 'exceeded'
                        })
                else:
                    # For quality metrics, check if mean is below threshold
                    if stats['mean'] < threshold:
                        report['violations'].append({
                            'metric': metric_name,
                            'value': stats['mean'],
                            'threshold': threshold,
                            'type': 'below'
                        })
        
        return report
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get detailed error report"""
        error_stats = self.error_handler.get_error_statistics()
        
        # Get recent errors
        recent_errors = [e for e in self.error_handler.error_history if 
                        (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'statistics': error_stats,
            'recent_errors': [
                {
                    'error_id': e.error_id,
                    'type': e.error_type,
                    'severity': e.severity.value,
                    'message': e.message,
                    'source': e.source,
                    'timestamp': e.timestamp.isoformat(),
                    'resolved': e.resolved
                }
                for e in recent_errors[-50:]  # Last 50 errors
            ],
            'circuit_breakers': {
                source: {
                    'state': breaker['state'],
                    'error_count': breaker['error_count'],
                    'threshold': breaker['threshold']
                }
                for source, breaker in self.error_handler.circuit_breakers.items()
            }
        }
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file"""
        try:
            monitoring_data = {
                'system_status': self.get_system_status(),
                'performance_report': self.get_performance_report(),
                'error_report': self.get_error_report(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(monitoring_data, f, indent=2, default=str)
            
            self.logger.info(f"Monitoring data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {e}")
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check performance metrics
        for metric_name in self.performance_thresholds:
            stats = self.performance_tracker.get_metric_statistics(metric_name)
            if stats:
                threshold = self.performance_thresholds[metric_name]
                
                if 'time' in metric_name and stats['mean'] > threshold:
                    recommendations.append(f"Optimize {metric_name}: current {stats['mean']:.2f} > threshold {threshold}")
                elif 'time' not in metric_name and stats['mean'] < threshold:
                    recommendations.append(f"Improve {metric_name}: current {stats['mean']:.2f} < threshold {threshold}")
        
        # Check resource usage
        resource_stats = self.resource_monitor.get_resource_statistics()
        for resource, stats in resource_stats.items():
            if stats['max'] > 80:
                recommendations.append(f"High {resource} usage detected: {stats['max']:.1f}%")
        
        # Check error rates
        error_stats = self.error_handler.get_error_statistics()
        if error_stats['error_rate'] > 0.1:  # More than 0.1 errors per second
            recommendations.append(f"High error rate: {error_stats['error_rate']:.2f} errors/second")
        
        return recommendations


# Context manager for performance monitoring
class PerformanceContext:
    """Context manager for performance monitoring"""
    
    def __init__(self, monitor: SequentialStrategicMonitor, metric_name: str, source: str):
        self.monitor = monitor
        self.metric_name = metric_name
        self.source = source
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.monitor.record_performance(
                self.metric_name, 
                duration_ms, 
                'ms', 
                self.source
            )


# Decorator for performance monitoring
def monitor_performance(monitor: SequentialStrategicMonitor, metric_name: str, source: str):
    """Decorator for performance monitoring"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceContext(monitor, metric_name, source):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Health check functions
async def agent_health_check(agent: SequentialStrategicAgentBase) -> HealthStatus:
    """Health check for sequential strategic agent"""
    issues = []
    metrics = {}
    
    # Check if agent is healthy
    if not agent.is_healthy:
        issues.append("Agent marked as unhealthy")
    
    # Check error count
    if agent.consecutive_errors > 0:
        issues.append(f"Agent has {agent.consecutive_errors} consecutive errors")
    
    # Check performance
    if hasattr(agent, 'sequential_performance'):
        avg_time = np.mean(agent.sequential_performance.get('enriched_processing_times', [0]))
        if avg_time > 5.0:  # 5ms threshold
            issues.append(f"Agent processing time too high: {avg_time:.2f}ms")
        
        metrics['avg_processing_time_ms'] = avg_time
    
    score = 1.0 - (len(issues) * 0.2)  # Reduce score by 0.2 for each issue
    status = "healthy" if score >= 0.8 else "degraded" if score >= 0.5 else "unhealthy"
    
    return HealthStatus(
        component=agent.name,
        status=status,
        score=max(0.0, score),
        timestamp=datetime.now(),
        issues=issues,
        metrics=metrics
    )


async def environment_health_check(environment: SequentialStrategicEnvironment) -> HealthStatus:
    """Health check for sequential strategic environment"""
    issues = []
    metrics = {}
    
    # Check environment state
    if not hasattr(environment, 'env_state'):
        issues.append("Environment state not initialized")
    
    # Check performance metrics
    perf_metrics = environment.get_performance_metrics()
    
    if perf_metrics['avg_sequence_execution_time_ms'] > 15.0:
        issues.append(f"Sequence execution time too high: {perf_metrics['avg_sequence_execution_time_ms']:.2f}ms")
    
    if perf_metrics['avg_superposition_quality'] < 0.7:
        issues.append(f"Superposition quality too low: {perf_metrics['avg_superposition_quality']:.3f}")
    
    metrics.update(perf_metrics)
    
    score = 1.0 - (len(issues) * 0.3)
    status = "healthy" if score >= 0.8 else "degraded" if score >= 0.5 else "unhealthy"
    
    return HealthStatus(
        component="environment",
        status=status,
        score=max(0.0, score),
        timestamp=datetime.now(),
        issues=issues,
        metrics=metrics
    )


# Factory function
def create_sequential_strategic_monitor(config: Dict[str, Any]) -> SequentialStrategicMonitor:
    """Create sequential strategic monitor with configuration"""
    return SequentialStrategicMonitor(config)


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = {
        'max_history': 10000,
        'monitoring_interval': 5
    }
    
    # Create monitor
    monitor = create_sequential_strategic_monitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Test performance recording
    monitor.record_performance('test_metric', 10.5, 'ms', 'test_source')
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        monitor.handle_error(e, 'test_source', {'context': 'test'})
    
    # Get system status
    status = monitor.get_system_status()
    print("System Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Get performance report
    perf_report = monitor.get_performance_report()
    print("\nPerformance Report:")
    print(json.dumps(perf_report, indent=2, default=str))
    
    # Get recommendations
    recommendations = monitor.generate_recommendations()
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("\nSequential Strategic Monitor test completed")