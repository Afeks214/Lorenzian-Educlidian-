"""
Enhanced Performance Monitoring System for GrandModel

This module provides comprehensive real-time performance monitoring with:
- Real-time metrics collection and analysis
- Performance regression detection
- Automated performance alerts
- Interactive performance dashboards
- Historical performance tracking
- Predictive performance analysis

Key Features:
- Sub-second metric collection
- Machine learning-based anomaly detection
- Automated performance regression alerts
- Real-time dashboard updates
- Performance trend analysis
- Resource usage optimization recommendations
"""

import asyncio
import threading
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import structlog
import psutil
import torch
import json
import sqlite3
from contextlib import asynccontextmanager
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Real-time performance metric"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert with detailed context"""
    timestamp: datetime
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class RealTimeMetricsCollector:
    """
    High-frequency metrics collection system.
    Collects performance metrics with minimal overhead.
    """
    
    def __init__(self, collection_interval: float = 0.1):
        self.collection_interval = collection_interval
        self.metrics_buffer = deque(maxlen=10000)
        self.metric_definitions = {}
        self.running = False
        self.collector_thread = None
        self.custom_collectors = {}
        
        # Performance tracking
        self.collection_stats = {
            'total_collections': 0,
            'collection_time_ms': deque(maxlen=100),
            'buffer_size': 0,
            'dropped_metrics': 0
        }
        
        logger.info("RealTimeMetricsCollector initialized",
                   interval=collection_interval)
    
    def register_metric(self, name: str, unit: str, 
                       description: str, tags: Dict[str, str] = None):
        """Register a new metric for collection"""
        self.metric_definitions[name] = {
            'unit': unit,
            'description': description,
            'tags': tags or {},
            'registered_at': datetime.now()
        }
        
        logger.info("Metric registered", name=name, unit=unit)
    
    def register_custom_collector(self, name: str, collector_func: Callable):
        """Register custom metric collector function"""
        self.custom_collectors[name] = collector_func
        logger.info("Custom collector registered", name=name)
    
    def start_collection(self):
        """Start real-time metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(
            target=self._collection_loop, 
            daemon=True
        )
        self.collector_thread.start()
        
        logger.info("Real-time metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        
        logger.info("Real-time metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect custom metrics
                self._collect_custom_metrics()
                
                # Update collection stats
                collection_time = (time.time() - start_time) * 1000
                self.collection_stats['collection_time_ms'].append(collection_time)
                self.collection_stats['total_collections'] += 1
                self.collection_stats['buffer_size'] = len(self.metrics_buffer)
                
                # Sleep for next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric('cpu_usage_percent', cpu_percent, 'percent', timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric('memory_usage_percent', memory.percent, 'percent', timestamp)
        self._add_metric('memory_available_mb', memory.available / 1024 / 1024, 'MB', timestamp)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self._add_metric('disk_read_mb_per_sec', disk_io.read_bytes / 1024 / 1024, 'MB/s', timestamp)
            self._add_metric('disk_write_mb_per_sec', disk_io.write_bytes / 1024 / 1024, 'MB/s', timestamp)
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            self._add_metric('network_recv_mb_per_sec', network_io.bytes_recv / 1024 / 1024, 'MB/s', timestamp)
            self._add_metric('network_sent_mb_per_sec', network_io.bytes_sent / 1024 / 1024, 'MB/s', timestamp)
        
        # PyTorch metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            self._add_metric('gpu_memory_mb', gpu_memory, 'MB', timestamp)
            
            gpu_cached = torch.cuda.memory_cached() / 1024 / 1024
            self._add_metric('gpu_cached_mb', gpu_cached, 'MB', timestamp)
    
    def _collect_custom_metrics(self):
        """Collect custom metrics from registered collectors"""
        timestamp = datetime.now()
        
        for collector_name, collector_func in self.custom_collectors.items():
            try:
                metrics = collector_func()
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        self._add_metric(f"{collector_name}_{metric_name}", value, 'custom', timestamp)
                elif isinstance(metrics, (int, float)):
                    self._add_metric(collector_name, metrics, 'custom', timestamp)
                    
            except Exception as e:
                logger.error("Custom collector error", 
                           collector=collector_name, error=str(e))
    
    def _add_metric(self, name: str, value: float, unit: str, timestamp: datetime):
        """Add metric to buffer"""
        try:
            metric = PerformanceMetric(
                timestamp=timestamp,
                metric_name=name,
                value=value,
                unit=unit,
                tags=self.metric_definitions.get(name, {}).get('tags', {}),
                context={}
            )
            
            self.metrics_buffer.append(metric)
            
        except Exception as e:
            self.collection_stats['dropped_metrics'] += 1
            logger.warning("Failed to add metric", name=name, error=str(e))
    
    def get_recent_metrics(self, metric_name: str = None, 
                          duration_seconds: int = 60) -> List[PerformanceMetric]:
        """Get recent metrics"""
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        
        metrics = [
            metric for metric in self.metrics_buffer
            if metric.timestamp >= cutoff_time
        ]
        
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        
        return metrics
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        avg_collection_time = (
            np.mean(self.collection_stats['collection_time_ms'])
            if self.collection_stats['collection_time_ms'] else 0
        )
        
        return {
            'running': self.running,
            'total_collections': self.collection_stats['total_collections'],
            'avg_collection_time_ms': avg_collection_time,
            'buffer_size': len(self.metrics_buffer),
            'dropped_metrics': self.collection_stats['dropped_metrics'],
            'registered_metrics': len(self.metric_definitions),
            'custom_collectors': len(self.custom_collectors)
        }


class PerformanceRegressionDetector:
    """
    ML-based performance regression detection system.
    Detects anomalies and performance degradation patterns.
    """
    
    def __init__(self, sensitivity: float = 0.1, min_samples: int = 100):
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        self.detection_stats = defaultdict(int)
        
        logger.info("PerformanceRegressionDetector initialized",
                   sensitivity=sensitivity, min_samples=min_samples)
    
    def train_baseline(self, metric_name: str, values: List[float]) -> bool:
        """Train baseline model for metric"""
        if len(values) < self.min_samples:
            logger.warning("Insufficient samples for baseline training",
                          metric=metric_name, samples=len(values))
            return False
        
        # Prepare data
        X = np.array(values).reshape(-1, 1)
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train isolation forest
        model = IsolationForest(
            contamination=self.sensitivity,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        
        # Store baseline statistics
        self.baseline_stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'trained_at': datetime.now(),
            'sample_count': len(values)
        }
        
        logger.info("Baseline model trained", 
                   metric=metric_name, samples=len(values))
        return True
    
    def detect_anomaly(self, metric_name: str, value: float) -> Dict:
        """Detect anomaly for metric value"""
        if metric_name not in self.models:
            return {
                'is_anomaly': False,
                'reason': 'No baseline model available',
                'confidence': 0.0
            }
        
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]
        baseline = self.baseline_stats[metric_name]
        
        # Scale value
        X = np.array([[value]])
        X_scaled = scaler.transform(X)
        
        # Predict anomaly
        prediction = model.predict(X_scaled)[0]
        anomaly_score = model.decision_function(X_scaled)[0]
        
        is_anomaly = prediction == -1
        confidence = abs(anomaly_score)
        
        # Statistical analysis
        z_score = (value - baseline['mean']) / baseline['std']
        deviation_from_median = abs(value - baseline['median']) / baseline['median']
        
        # Update detection stats
        self.detection_stats[f"{metric_name}_total"] += 1
        if is_anomaly:
            self.detection_stats[f"{metric_name}_anomalies"] += 1
        
        result = {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'z_score': z_score,
            'deviation_from_median': deviation_from_median,
            'baseline_mean': baseline['mean'],
            'baseline_std': baseline['std'],
            'current_value': value,
            'detection_time': datetime.now()
        }
        
        if is_anomaly:
            logger.warning("Performance anomaly detected",
                          metric=metric_name,
                          value=value,
                          confidence=confidence,
                          z_score=z_score)
        
        return result
    
    def detect_regression(self, metric_name: str, 
                         recent_values: List[float],
                         window_size: int = 20) -> Dict:
        """Detect performance regression trend"""
        if len(recent_values) < window_size:
            return {
                'regression_detected': False,
                'reason': 'Insufficient data for regression analysis'
            }
        
        # Calculate trends
        recent_window = recent_values[-window_size:]
        older_window = recent_values[-window_size*2:-window_size] if len(recent_values) >= window_size*2 else recent_values[:-window_size]
        
        if not older_window:
            return {
                'regression_detected': False,
                'reason': 'Insufficient historical data'
            }
        
        recent_mean = np.mean(recent_window)
        older_mean = np.mean(older_window)
        
        # Calculate regression indicators
        relative_change = (recent_mean - older_mean) / older_mean
        trend_slope = np.polyfit(range(len(recent_window)), recent_window, 1)[0]
        
        # Determine if regression occurred
        regression_threshold = 0.1  # 10% increase indicates regression
        slope_threshold = 0.01  # Positive slope indicates worsening
        
        regression_detected = (
            relative_change > regression_threshold or
            trend_slope > slope_threshold
        )
        
        severity = 'LOW'
        if relative_change > 0.2 or trend_slope > 0.05:
            severity = 'MEDIUM'
        if relative_change > 0.5 or trend_slope > 0.1:
            severity = 'HIGH'
        
        return {
            'regression_detected': regression_detected,
            'severity': severity,
            'relative_change': relative_change,
            'trend_slope': trend_slope,
            'recent_mean': recent_mean,
            'older_mean': older_mean,
            'window_size': window_size,
            'analysis_time': datetime.now()
        }
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        stats = {}
        
        for metric_name in self.models.keys():
            total_key = f"{metric_name}_total"
            anomaly_key = f"{metric_name}_anomalies"
            
            total = self.detection_stats[total_key]
            anomalies = self.detection_stats[anomaly_key]
            
            stats[metric_name] = {
                'total_predictions': total,
                'anomalies_detected': anomalies,
                'anomaly_rate': anomalies / max(total, 1),
                'baseline_stats': self.baseline_stats.get(metric_name, {})
            }
        
        return stats


class PerformanceAlertSystem:
    """
    Real-time performance alert system.
    Generates and manages performance alerts with severity levels.
    """
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.alert_rules = {}
        self.alert_stats = defaultdict(int)
        self.notification_callbacks = []
        
        logger.info("PerformanceAlertSystem initialized")
    
    def register_alert_rule(self, rule_name: str, metric_name: str,
                           threshold: float, comparison: str = 'greater',
                           severity: str = 'MEDIUM'):
        """Register performance alert rule"""
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'comparison': comparison,  # 'greater', 'less', 'equal'
            'severity': severity,
            'created_at': datetime.now(),
            'triggered_count': 0
        }
        
        logger.info("Alert rule registered",
                   rule=rule_name, metric=metric_name, threshold=threshold)
    
    def register_notification_callback(self, callback: Callable):
        """Register callback for alert notifications"""
        self.notification_callbacks.append(callback)
        logger.info("Notification callback registered")
    
    def check_alerts(self, metric: PerformanceMetric) -> List[PerformanceAlert]:
        """Check metric against alert rules"""
        alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if rule['metric_name'] != metric.metric_name:
                continue
            
            # Check threshold
            triggered = False
            if rule['comparison'] == 'greater':
                triggered = metric.value > rule['threshold']
            elif rule['comparison'] == 'less':
                triggered = metric.value < rule['threshold']
            elif rule['comparison'] == 'equal':
                triggered = abs(metric.value - rule['threshold']) < 0.01
            
            if triggered:
                alert = PerformanceAlert(
                    timestamp=metric.timestamp,
                    alert_type=rule_name,
                    severity=rule['severity'],
                    metric_name=metric.metric_name,
                    current_value=metric.value,
                    expected_value=rule['threshold'],
                    deviation=abs(metric.value - rule['threshold']),
                    message=f"Performance alert: {metric.metric_name} {rule['comparison']} {rule['threshold']} (current: {metric.value})",
                    context={
                        'rule_name': rule_name,
                        'metric_tags': metric.tags,
                        'metric_context': metric.context
                    }
                )
                
                alerts.append(alert)
                self.alerts.append(alert)
                
                # Update statistics
                self.alert_stats[rule_name] += 1
                self.alert_stats[f"{rule['severity']}_alerts"] += 1
                rule['triggered_count'] += 1
                
                # Send notifications
                self._send_notifications(alert)
                
                logger.warning("Performance alert triggered",
                              rule=rule_name,
                              metric=metric.metric_name,
                              value=metric.value,
                              threshold=rule['threshold'])
        
        return alerts
    
    def _send_notifications(self, alert: PerformanceAlert):
        """Send alert notifications"""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Notification callback failed", error=str(e))
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        # In a real implementation, you'd have alert IDs
        # For now, we'll mark the most recent matching alert as resolved
        for alert in reversed(self.alerts):
            if not alert.resolved and alert.alert_type == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                return True
        return False
    
    def get_active_alerts(self, severity: str = None) -> List[PerformanceAlert]:
        """Get active alerts"""
        active = [alert for alert in self.alerts if not alert.resolved]
        
        if severity:
            active = [alert for alert in active if alert.severity == severity]
        
        return active
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(self.get_active_alerts()),
            'alert_rules': len(self.alert_rules),
            'alert_stats': dict(self.alert_stats),
            'notification_callbacks': len(self.notification_callbacks)
        }


class EnhancedPerformanceMonitor:
    """
    Main enhanced performance monitoring system.
    Integrates all monitoring components for comprehensive performance tracking.
    """
    
    def __init__(self):
        self.metrics_collector = RealTimeMetricsCollector()
        self.regression_detector = PerformanceRegressionDetector()
        self.alert_system = PerformanceAlertSystem()
        
        self.monitoring_enabled = False
        self.analysis_thread = None
        self.analysis_running = False
        
        # Dashboard data
        self.dashboard_data = {
            'metrics': {},
            'alerts': [],
            'trends': {},
            'recommendations': []
        }
        
        logger.info("EnhancedPerformanceMonitor initialized")
    
    def enable_monitoring(self):
        """Enable enhanced performance monitoring"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start analysis thread
        self.analysis_running = True
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop, 
            daemon=True
        )
        self.analysis_thread.start()
        
        # Register default alert rules
        self._register_default_alerts()
        
        logger.info("Enhanced performance monitoring enabled")
    
    def disable_monitoring(self):
        """Disable performance monitoring"""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        
        # Stop analysis
        self.analysis_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5.0)
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        logger.info("Enhanced performance monitoring disabled")
    
    def _register_default_alerts(self):
        """Register default performance alert rules"""
        self.alert_system.register_alert_rule(
            'high_cpu_usage', 'cpu_usage_percent', 80.0, 'greater', 'HIGH'
        )
        self.alert_system.register_alert_rule(
            'high_memory_usage', 'memory_usage_percent', 85.0, 'greater', 'HIGH'
        )
        self.alert_system.register_alert_rule(
            'low_memory_available', 'memory_available_mb', 1000.0, 'less', 'MEDIUM'
        )
        
        if torch.cuda.is_available():
            self.alert_system.register_alert_rule(
                'high_gpu_memory', 'gpu_memory_mb', 8000.0, 'greater', 'MEDIUM'
            )
    
    def _analysis_loop(self):
        """Main analysis loop for regression detection"""
        while self.analysis_running:
            try:
                # Get recent metrics
                recent_metrics = self.metrics_collector.get_recent_metrics(
                    duration_seconds=300  # 5 minutes
                )
                
                # Group metrics by name
                metrics_by_name = defaultdict(list)
                for metric in recent_metrics:
                    metrics_by_name[metric.metric_name].append(metric)
                
                # Check for regressions and alerts
                for metric_name, metric_list in metrics_by_name.items():
                    if len(metric_list) < 10:
                        continue
                    
                    values = [m.value for m in metric_list]
                    
                    # Train baseline if not exists
                    if metric_name not in self.regression_detector.models:
                        if len(values) >= self.regression_detector.min_samples:
                            self.regression_detector.train_baseline(metric_name, values)
                    
                    # Check latest value for anomalies
                    if metric_list:
                        latest_metric = metric_list[-1]
                        
                        # Check alerts
                        alerts = self.alert_system.check_alerts(latest_metric)
                        
                        # Check anomalies
                        anomaly_result = self.regression_detector.detect_anomaly(
                            metric_name, latest_metric.value
                        )
                        
                        if anomaly_result['is_anomaly']:
                            logger.warning("Performance anomaly detected",
                                          metric=metric_name,
                                          value=latest_metric.value,
                                          confidence=anomaly_result['confidence'])
                        
                        # Check regression
                        regression_result = self.regression_detector.detect_regression(
                            metric_name, values
                        )
                        
                        if regression_result['regression_detected']:
                            logger.warning("Performance regression detected",
                                          metric=metric_name,
                                          severity=regression_result['severity'],
                                          change=regression_result['relative_change'])
                
                # Update dashboard data
                self._update_dashboard()
                
                # Sleep before next analysis
                time.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error("Analysis loop error", error=str(e))
                time.sleep(30)
    
    def _update_dashboard(self):
        """Update dashboard data"""
        # Get latest metrics
        recent_metrics = self.metrics_collector.get_recent_metrics(
            duration_seconds=60
        )
        
        # Aggregate metrics
        metrics_summary = {}
        for metric in recent_metrics:
            if metric.metric_name not in metrics_summary:
                metrics_summary[metric.metric_name] = {
                    'values': [],
                    'timestamps': [],
                    'unit': metric.unit
                }
            
            metrics_summary[metric.metric_name]['values'].append(metric.value)
            metrics_summary[metric.metric_name]['timestamps'].append(metric.timestamp)
        
        # Calculate statistics
        for metric_name, data in metrics_summary.items():
            if data['values']:
                data['current'] = data['values'][-1]
                data['average'] = np.mean(data['values'])
                data['min'] = np.min(data['values'])
                data['max'] = np.max(data['values'])
        
        self.dashboard_data['metrics'] = metrics_summary
        self.dashboard_data['alerts'] = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'resolved': alert.resolved
            }
            for alert in self.alert_system.get_active_alerts()
        ]
    
    def get_dashboard_data(self) -> Dict:
        """Get dashboard data"""
        return self.dashboard_data.copy()
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive monitoring statistics"""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'metrics_collector': self.metrics_collector.get_collection_stats(),
            'regression_detector': self.regression_detector.get_detection_stats(),
            'alert_system': self.alert_system.get_alert_stats(),
            'dashboard_metrics': len(self.dashboard_data['metrics']),
            'active_alerts': len(self.dashboard_data['alerts'])
        }
    
    def register_custom_metric(self, name: str, collector_func: Callable):
        """Register custom metric collector"""
        self.metrics_collector.register_custom_collector(name, collector_func)
    
    def create_performance_report(self, hours: int = 24) -> Dict:
        """Create comprehensive performance report"""
        # Get metrics for the specified duration
        recent_metrics = self.metrics_collector.get_recent_metrics(
            duration_seconds=hours * 3600
        )
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.metric_name].append(metric.value)
        
        # Generate report
        report = {
            'report_period_hours': hours,
            'generated_at': datetime.now().isoformat(),
            'metrics_summary': {},
            'performance_trends': {},
            'anomalies_detected': 0,
            'alerts_triggered': len(self.alert_system.alerts),
            'recommendations': []
        }
        
        # Analyze each metric
        for metric_name, values in metrics_by_name.items():
            if len(values) < 10:
                continue
            
            # Basic statistics
            report['metrics_summary'][metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
            
            # Trend analysis
            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
            report['performance_trends'][metric_name] = {
                'trend_slope': trend_slope,
                'trend_direction': 'improving' if trend_slope < 0 else 'degrading',
                'trend_strength': abs(trend_slope)
            }
            
            # Check for anomalies
            if metric_name in self.regression_detector.models:
                anomaly_count = 0
                for value in values[-100:]:  # Check last 100 values
                    result = self.regression_detector.detect_anomaly(metric_name, value)
                    if result['is_anomaly']:
                        anomaly_count += 1
                
                report['anomalies_detected'] += anomaly_count
        
        # Generate recommendations
        if report['anomalies_detected'] > 10:
            report['recommendations'].append(
                "High number of anomalies detected. Consider investigating system stability."
            )
        
        for metric_name, trend in report['performance_trends'].items():
            if trend['trend_direction'] == 'degrading' and trend['trend_strength'] > 0.1:
                report['recommendations'].append(
                    f"Performance degradation detected in {metric_name}. Consider optimization."
                )
        
        return report


# Global enhanced performance monitor
enhanced_monitor = EnhancedPerformanceMonitor()


def monitor_performance():
    """Decorator to enable performance monitoring for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            enhanced_monitor.enable_monitoring()
            
            # Custom metric for function execution
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Add custom metric
            def execution_time_collector():
                return {'execution_time_ms': execution_time * 1000}
            
            enhanced_monitor.register_custom_metric(
                f"function_{func.__name__}",
                execution_time_collector
            )
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    """Demo enhanced performance monitoring"""
    
    print("📊 Enhanced Performance Monitoring Demo")
    print("=" * 50)
    
    # Enable monitoring
    enhanced_monitor.enable_monitoring()
    
    # Register custom metric
    def custom_metric_collector():
        return {
            'custom_value': np.random.normal(50, 10),
            'processing_queue_size': np.random.randint(0, 100)
        }
    
    enhanced_monitor.register_custom_metric('custom_metrics', custom_metric_collector)
    
    # Let it run for a bit to collect data
    print("\n🔄 Collecting performance data...")
    time.sleep(10)
    
    # Get comprehensive stats
    stats = enhanced_monitor.get_comprehensive_stats()
    
    print("\n📈 Monitoring Statistics:")
    print(f"  Monitoring enabled: {stats['monitoring_enabled']}")
    print(f"  Total collections: {stats['metrics_collector']['total_collections']}")
    print(f"  Buffer size: {stats['metrics_collector']['buffer_size']}")
    print(f"  Registered metrics: {stats['metrics_collector']['registered_metrics']}")
    print(f"  Active alerts: {stats['alert_system']['active_alerts']}")
    
    # Generate performance report
    report = enhanced_monitor.create_performance_report(hours=1)
    
    print("\n📋 Performance Report:")
    print(f"  Metrics analyzed: {len(report['metrics_summary'])}")
    print(f"  Anomalies detected: {report['anomalies_detected']}")
    print(f"  Alerts triggered: {report['alerts_triggered']}")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Get dashboard data
    dashboard = enhanced_monitor.get_dashboard_data()
    
    print(f"\n📊 Dashboard Metrics: {len(dashboard['metrics'])}")
    print(f"🚨 Active Alerts: {len(dashboard['alerts'])}")
    
    print("\n✅ Enhanced monitoring demo completed!")
    
    # Disable monitoring
    enhanced_monitor.disable_monitoring()