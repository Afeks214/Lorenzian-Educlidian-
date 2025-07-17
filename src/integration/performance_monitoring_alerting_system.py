#!/usr/bin/env python3
"""
Agent Omega: Performance Monitoring & Alerting System
=====================================================

Mission: Implement comprehensive performance monitoring and alerting system
to ensure all agent implementations maintain world-class performance standards.

This system provides:
- Real-time performance monitoring
- Automated alerting on performance degradation
- Performance trend analysis
- System health monitoring
- Comprehensive reporting dashboard
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import logging
import threading
from collections import defaultdict, deque
import statistics


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being monitored"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    AVAILABILITY = "availability"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str
    metric_type: MetricType
    metadata: Dict[str, Any]


@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    metric_name: str
    component: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    comparison_operator: str  # 'gt', 'lt', 'eq'
    time_window_seconds: int
    consecutive_violations: int


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    message: str
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool
    resolution_time: Optional[datetime]
    metadata: Dict[str, Any]


class PerformanceMonitoringAlertingSystem:
    """
    Comprehensive performance monitoring and alerting system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance data storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Monitoring configuration
        self.monitoring_interval = config.get('monitoring_interval_seconds', 5)
        self.alert_cooldown_seconds = config.get('alert_cooldown_seconds', 300)
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Performance thresholds for all agents
        self.performance_thresholds = self._initialize_performance_thresholds()
        
        # System health tracking
        self.system_health_score = 10.0
        self.world_class_threshold = 9.5
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def _initialize_performance_thresholds(self) -> Dict[str, AlertThreshold]:
        """Initialize performance thresholds for all components"""
        
        thresholds = {}
        
        # Agent Alpha: Security Framework
        thresholds['security_auth_latency'] = AlertThreshold(
            metric_name='security_auth_latency',
            component='security_framework',
            warning_threshold=50.0,
            critical_threshold=100.0,
            emergency_threshold=500.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        thresholds['security_rate_limit_latency'] = AlertThreshold(
            metric_name='security_rate_limit_latency',
            component='security_framework',
            warning_threshold=5.0,
            critical_threshold=10.0,
            emergency_threshold=50.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        # Agent Beta: Event Bus & Real-time Infrastructure
        thresholds['event_bus_latency'] = AlertThreshold(
            metric_name='event_bus_latency',
            component='event_bus',
            warning_threshold=1.0,
            critical_threshold=5.0,
            emergency_threshold=10.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        thresholds['websocket_latency'] = AlertThreshold(
            metric_name='websocket_latency',
            component='xai_system',
            warning_threshold=50.0,
            critical_threshold=100.0,
            emergency_threshold=500.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        # Agent Gamma: Algorithm Optimization
        thresholds['algorithm_inference_latency'] = AlertThreshold(
            metric_name='algorithm_inference_latency',
            component='algorithm_optimization',
            warning_threshold=100.0,
            critical_threshold=200.0,
            emergency_threshold=500.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        # Agent Delta: Data Pipeline
        thresholds['data_pipeline_latency'] = AlertThreshold(
            metric_name='data_pipeline_latency',
            component='data_pipeline',
            warning_threshold=100.0,
            critical_threshold=200.0,
            emergency_threshold=500.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        # Agent Epsilon: XAI System
        thresholds['xai_explanation_latency'] = AlertThreshold(
            metric_name='xai_explanation_latency',
            component='xai_system',
            warning_threshold=500.0,
            critical_threshold=1000.0,
            emergency_threshold=2000.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        # Agent 2 (VaR): Correlation Specialist
        thresholds['var_calculation_latency'] = AlertThreshold(
            metric_name='var_calculation_latency',
            component='var_system',
            warning_threshold=5.0,
            critical_threshold=10.0,
            emergency_threshold=20.0,
            comparison_operator='gt',
            time_window_seconds=60,
            consecutive_violations=3
        )
        
        return thresholds
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        
        metric_key = f"{metric.component}_{metric.name}"
        self.metrics[metric_key].append(metric)
        
        # Check for threshold violations
        self._check_threshold_violations(metric)
        
        # Log metric
        self.logger.debug(
            f"Recorded metric: {metric.name} = {metric.value} {metric.unit}",
            component=metric.component,
            metric_type=metric.metric_type.value
        )
    
    def _check_threshold_violations(self, metric: PerformanceMetric):
        """Check if metric violates any thresholds"""
        
        threshold_key = f"{metric.component}_{metric.name}"
        
        if threshold_key not in self.performance_thresholds:
            return
            
        threshold = self.performance_thresholds[threshold_key]
        
        # Check if metric violates thresholds
        alert_level = self._evaluate_threshold_violation(metric.value, threshold)
        
        if alert_level:
            self._trigger_alert(metric, threshold, alert_level)
    
    def _evaluate_threshold_violation(self, value: float, threshold: AlertThreshold) -> Optional[AlertLevel]:
        """Evaluate if value violates threshold"""
        
        if threshold.comparison_operator == 'gt':
            if value >= threshold.emergency_threshold:
                return AlertLevel.EMERGENCY
            elif value >= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif value >= threshold.warning_threshold:
                return AlertLevel.WARNING
        elif threshold.comparison_operator == 'lt':
            if value <= threshold.emergency_threshold:
                return AlertLevel.EMERGENCY
            elif value <= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif value <= threshold.warning_threshold:
                return AlertLevel.WARNING
        
        return None
    
    def _trigger_alert(self, metric: PerformanceMetric, threshold: AlertThreshold, level: AlertLevel):
        """Trigger an alert"""
        
        alert_key = f"{metric.component}_{metric.name}_{level.value}"
        
        # Check alert cooldown
        if alert_key in self.last_alert_time:
            time_since_last = (datetime.now() - self.last_alert_time[alert_key]).total_seconds()
            if time_since_last < self.alert_cooldown_seconds:
                return
        
        # Create alert
        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{hash(alert_key)}",
            level=level,
            message=f"{metric.component} {metric.name} {level.value}: {metric.value} {metric.unit}",
            component=metric.component,
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=self._get_threshold_value(threshold, level),
            timestamp=datetime.now(),
            resolved=False,
            resolution_time=None,
            metadata={
                'metric_type': metric.metric_type.value,
                'threshold_config': asdict(threshold)
            }
        )
        
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = datetime.now()
        
        # Log alert
        self.logger.error(
            f"ALERT {level.value.upper()}: {alert.message}",
            alert_id=alert.alert_id,
            component=alert.component
        )
        
        # Handle critical/emergency alerts
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._handle_critical_alert(alert)
    
    def _get_threshold_value(self, threshold: AlertThreshold, level: AlertLevel) -> float:
        """Get threshold value for alert level"""
        
        if level == AlertLevel.WARNING:
            return threshold.warning_threshold
        elif level == AlertLevel.CRITICAL:
            return threshold.critical_threshold
        elif level == AlertLevel.EMERGENCY:
            return threshold.emergency_threshold
        
        return 0.0
    
    def _handle_critical_alert(self, alert: Alert):
        """Handle critical/emergency alerts"""
        
        # For demonstration, we'll just log the critical alert
        # In production, this would trigger automated responses
        
        self.logger.critical(
            f"CRITICAL ALERT TRIGGERED: {alert.message}",
            alert_id=alert.alert_id,
            component=alert.component,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value
        )
        
        # Update system health score
        self._update_system_health_score()
    
    def _update_system_health_score(self):
        """Update system health score based on alerts"""
        
        # Calculate health score based on active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        # Base score
        health_score = 10.0
        
        # Deduct points for alerts
        for alert in active_alerts:
            if alert.level == AlertLevel.WARNING:
                health_score -= 0.5
            elif alert.level == AlertLevel.CRITICAL:
                health_score -= 1.0
            elif alert.level == AlertLevel.EMERGENCY:
                health_score -= 2.0
        
        # Ensure minimum score
        self.system_health_score = max(0.0, health_score)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            self.logger.info(
                f"Alert resolved: {alert.message}",
                alert_id=alert_id,
                resolution_time=alert.resolution_time
            )
            
            # Update system health score
            self._update_system_health_score()
    
    def get_metric_statistics(self, component: str, metric_name: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for a specific metric"""
        
        metric_key = f"{component}_{metric_name}"
        
        if metric_key not in self.metrics:
            return {}
        
        # Filter metrics by time window
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [
            m for m in self.metrics[metric_key]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0],
            'p99': statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0],
            'unit': recent_metrics[0].unit
        }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        
        # Update system health score
        self._update_system_health_score()
        
        # Get active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        # Get component health
        component_health = {}
        components = ['security_framework', 'event_bus', 'algorithm_optimization', 
                     'data_pipeline', 'xai_system', 'var_system']
        
        for component in components:
            component_alerts = [alert for alert in active_alerts if alert.component == component]
            component_health[component] = {
                'status': 'HEALTHY' if not component_alerts else 'DEGRADED',
                'alert_count': len(component_alerts),
                'highest_alert_level': max([alert.level.value for alert in component_alerts]) if component_alerts else 'none'
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health_score': self.system_health_score,
            'world_class_threshold': self.world_class_threshold,
            'world_class_status': self.system_health_score >= self.world_class_threshold,
            'overall_status': 'HEALTHY' if self.system_health_score >= self.world_class_threshold else 'DEGRADED',
            'active_alerts': len(active_alerts),
            'alert_breakdown': {
                'warning': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                'emergency': len([a for a in active_alerts if a.level == AlertLevel.EMERGENCY])
            },
            'component_health': component_health,
            'total_alerts_today': len([
                alert for alert in self.alert_history
                if alert.timestamp >= datetime.now() - timedelta(days=1)
            ]),
            'monitoring_active': self.monitoring_active
        }
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health_report(),
            'performance_metrics': {},
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'level': alert.level.value,
                    'message': alert.message,
                    'component': alert.component,
                    'timestamp': alert.timestamp.isoformat(),
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value
                }
                for alert in self.alerts.values() if not alert.resolved
            ],
            'performance_trends': {}
        }
        
        # Get performance metrics for all components
        components = ['security_framework', 'event_bus', 'algorithm_optimization', 
                     'data_pipeline', 'xai_system', 'var_system']
        
        for component in components:
            # Get all metrics for this component
            component_metrics = {}
            for metric_key, metric_data in self.metrics.items():
                if metric_key.startswith(component):
                    metric_name = metric_key[len(component)+1:]
                    stats = self.get_metric_statistics(component, metric_name)
                    if stats:
                        component_metrics[metric_name] = stats
            
            dashboard['performance_metrics'][component] = component_metrics
        
        return dashboard
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Simulate collecting metrics (in production this would collect real metrics)
                self._collect_simulated_metrics()
                
                # Update system health
                self._update_system_health_score()
                
                # Sleep until next monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_simulated_metrics(self):
        """Collect simulated metrics for demonstration"""
        
        import random
        
        # Simulate metrics for all components
        components_metrics = {
            'security_framework': [
                ('security_auth_latency', 'ms', lambda: random.uniform(20, 60)),
                ('security_rate_limit_latency', 'ms', lambda: random.uniform(1, 8))
            ],
            'event_bus': [
                ('event_bus_latency', 'ms', lambda: random.uniform(0.5, 2.0))
            ],
            'algorithm_optimization': [
                ('algorithm_inference_latency', 'ms', lambda: random.uniform(70, 120))
            ],
            'data_pipeline': [
                ('data_pipeline_latency', 'ms', lambda: random.uniform(40, 80))
            ],
            'xai_system': [
                ('xai_explanation_latency', 'ms', lambda: random.uniform(120, 300)),
                ('websocket_latency', 'ms', lambda: random.uniform(15, 40))
            ],
            'var_system': [
                ('var_calculation_latency', 'ms', lambda: random.uniform(2, 6))
            ]
        }
        
        # Collect metrics
        for component, metrics in components_metrics.items():
            for metric_name, unit, value_func in metrics:
                metric = PerformanceMetric(
                    name=metric_name,
                    value=value_func(),
                    unit=unit,
                    timestamp=datetime.now(),
                    component=component,
                    metric_type=MetricType.LATENCY,
                    metadata={}
                )
                
                self.record_metric(metric)


def main():
    """Main function for performance monitoring demo"""
    
    print("=" * 80)
    print("🎯 AGENT OMEGA: PERFORMANCE MONITORING & ALERTING SYSTEM")
    print("=" * 80)
    print()
    
    # Configuration
    config = {
        'monitoring_interval_seconds': 2,
        'alert_cooldown_seconds': 10
    }
    
    # Initialize monitoring system
    monitoring_system = PerformanceMonitoringAlertingSystem(config)
    
    print("📊 Starting performance monitoring...")
    monitoring_system.start_monitoring()
    
    try:
        # Monitor for 30 seconds
        for i in range(15):
            time.sleep(2)
            
            # Get system health report
            health_report = monitoring_system.get_system_health_report()
            
            print(f"\n📈 System Health Report (Update {i+1})")
            print(f"🏆 Health Score: {health_report['system_health_score']:.2f}/10.0")
            print(f"✅ World-Class Status: {'YES' if health_report['world_class_status'] else 'NO'}")
            print(f"⚠️ Active Alerts: {health_report['active_alerts']}")
            
            # Show component health
            print("🔍 Component Health:")
            for component, health in health_report['component_health'].items():
                status_emoji = "✅" if health['status'] == 'HEALTHY' else "⚠️"
                print(f"  {status_emoji} {component}: {health['status']}")
            
            # Show any active alerts
            if health_report['active_alerts'] > 0:
                dashboard = monitoring_system.get_performance_dashboard()
                print("\n⚠️ Active Alerts:")
                for alert in dashboard['active_alerts']:
                    level_emoji = {"warning": "⚠️", "critical": "🚨", "emergency": "🆘"}
                    emoji = level_emoji.get(alert['level'], "⚠️")
                    print(f"  {emoji} {alert['message']}")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring interrupted by user")
    
    finally:
        monitoring_system.stop_monitoring()
        print("\n📊 Performance monitoring stopped")
    
    # Final dashboard
    print("\n📊 FINAL PERFORMANCE DASHBOARD")
    print("=" * 80)
    
    dashboard = monitoring_system.get_performance_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))
    
    print("=" * 80)
    print("🎯 PERFORMANCE MONITORING DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()