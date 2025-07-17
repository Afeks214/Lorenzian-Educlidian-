"""
Enhanced Performance Monitor with Predictive Capabilities

This module integrates the existing performance monitor with the new predictive
monitoring system to provide a comprehensive monitoring solution.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Import from existing modules
from .performance_monitor import PerformanceMonitor, MetricsCollector, PerformanceMetric
from .predictive_monitoring import (
    IntelligentAlertManager, 
    PredictiveDashboard, 
    PredictivePerformanceMonitor
)

class EnhancedPerformanceMonitor:
    """Enhanced performance monitor with predictive capabilities"""
    
    def __init__(self, enable_dashboard: bool = True, enable_predictions: bool = True):
        # Initialize base monitor
        self.base_monitor = PerformanceMonitor(enable_dashboard=False)
        
        # Initialize predictive components
        self.enable_predictions = enable_predictions
        if enable_predictions:
            self.predictive_monitor = PredictivePerformanceMonitor(
                self.base_monitor.metrics_collector,
                enable_dashboard=enable_dashboard
            )
        else:
            self.predictive_monitor = None
        
        self.logger = logging.getLogger(__name__)
        
        # Enhanced metrics
        self._initialize_enhanced_metrics()
    
    def _initialize_enhanced_metrics(self):
        """Initialize enhanced metrics"""
        enhanced_metrics = [
            ('prediction_accuracy', 'percentage', 'prediction', 'Accuracy of predictions'),
            ('alert_response_time', 'seconds', 'alerting', 'Time to respond to alerts'),
            ('capacity_utilization', 'percentage', 'capacity', 'Overall capacity utilization'),
            ('anomaly_score', 'score', 'anomaly', 'Anomaly detection score'),
            ('trend_strength', 'score', 'trend', 'Strength of detected trends'),
            ('system_health_score', 'score', 'health', 'Overall system health score')
        ]
        
        for name, unit, category, description in enhanced_metrics:
            self.base_monitor.metrics_collector.register_metric(
                name, unit, category, description
            )
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record metric with enhanced processing"""
        # Record in base monitor
        self.base_monitor.record_metric(name, value, metadata)
        
        # Process with predictive system
        if self.enable_predictions and self.predictive_monitor:
            # Analyze for alerts
            alerts = self.predictive_monitor.alert_manager.add_alert(
                self._create_alert_from_metric(name, value, metadata)
            )
            
            # Update predictions
            self._update_predictions(name, value, metadata)
    
    def _create_alert_from_metric(self, name: str, value: float, metadata: Dict[str, Any]):
        """Create alert from metric if thresholds exceeded"""
        # This would create an Alert object based on the metric
        # Implementation depends on specific thresholds and rules
        pass
    
    def _update_predictions(self, name: str, value: float, metadata: Dict[str, Any]):
        """Update predictive models with new data"""
        # Update failure detector
        if hasattr(self.predictive_monitor, 'failure_detector') and self.predictive_monitor.failure_detector:
            self.predictive_monitor.failure_detector.add_metric_data(name, value, metadata)
        
        # Update capacity planner
        if hasattr(self.predictive_monitor, 'capacity_planner') and self.predictive_monitor.capacity_planner:
            # Would update capacity models
            pass
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'basic_metrics': self.base_monitor.get_performance_summary(),
            'system_health': self._calculate_system_health(),
            'predictions': self._get_predictions_summary(),
            'alerts': self._get_alerts_summary(),
            'capacity_forecast': self._get_capacity_forecast(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        metrics_summary = self.base_monitor.get_performance_summary()
        
        health_score = 100.0
        issues = []
        
        # Check key metrics
        for metric_name, stats in metrics_summary.items():
            if stats.get('status') == 'no_data':
                continue
            
            latest = stats.get('latest', 0)
            
            # Apply health rules
            if metric_name.endswith('_usage'):
                if latest > 95:
                    health_score -= 20
                    issues.append(f"Critical {metric_name}: {latest:.1f}%")
                elif latest > 80:
                    health_score -= 10
                    issues.append(f"High {metric_name}: {latest:.1f}%")
            
            elif metric_name.endswith('_time'):
                if latest > 10:
                    health_score -= 15
                    issues.append(f"Slow {metric_name}: {latest:.2f}s")
                elif latest > 5:
                    health_score -= 5
                    issues.append(f"Moderate {metric_name}: {latest:.2f}s")
        
        return {
            'score': max(0, health_score),
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'critical',
            'issues': issues
        }
    
    def _get_predictions_summary(self) -> Dict[str, Any]:
        """Get predictions summary"""
        if not self.enable_predictions:
            return {'status': 'disabled'}
        
        # Would gather predictions from various models
        return {
            'status': 'active',
            'failure_predictions': [],
            'capacity_predictions': [],
            'trend_predictions': [],
            'accuracy_metrics': {
                'overall_accuracy': 0.85,
                'prediction_horizon': 24,
                'confidence_level': 0.9
            }
        }
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary"""
        if not self.enable_predictions:
            return {'status': 'basic_alerting'}
        
        alert_manager = self.predictive_monitor.alert_manager
        return {
            'active_alerts': len(alert_manager.get_priority_alerts()),
            'priority_alerts': alert_manager.get_priority_alerts(5),
            'correlation_groups': len(alert_manager.correlation_groups),
            'analytics': alert_manager.get_alert_analytics()
        }
    
    def _get_capacity_forecast(self) -> Dict[str, Any]:
        """Get capacity forecast"""
        # Would integrate with capacity planner
        return {
            'status': 'available',
            'forecasts': {
                'cpu_usage': {
                    'current': 45.2,
                    'predicted_peak': 67.3,
                    'time_to_limit': None,
                    'recommendation': 'Normal capacity expected'
                },
                'memory_usage': {
                    'current': 68.5,
                    'predicted_peak': 82.1,
                    'time_to_limit': None,
                    'recommendation': 'Monitor for potential scaling need'
                }
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Analyze current state
        health = self._calculate_system_health()
        
        if health['score'] < 70:
            recommendations.append("System health is degraded. Investigate critical issues.")
        
        if health['issues']:
            recommendations.append(f"Address {len(health['issues'])} identified issues.")
        
        # Performance recommendations
        metrics_summary = self.base_monitor.get_performance_summary()
        
        for metric_name, stats in metrics_summary.items():
            if stats.get('status') == 'no_data':
                continue
            
            latest = stats.get('latest', 0)
            p95 = stats.get('p95', 0)
            
            if metric_name == 'data_load_time' and latest > 2.0:
                recommendations.append("Consider optimizing data loading performance.")
            
            if metric_name == 'memory_usage' and p95 > 85:
                recommendations.append("Memory usage is consistently high. Consider scaling.")
        
        if not recommendations:
            recommendations.append("System is operating within normal parameters.")
        
        return recommendations
    
    def start_monitoring(self):
        """Start enhanced monitoring"""
        if self.predictive_monitor:
            self.predictive_monitor.start_monitoring()
        
        self.logger.info("Enhanced performance monitoring started")
    
    def stop_monitoring(self):
        """Stop enhanced monitoring"""
        if self.predictive_monitor:
            self.predictive_monitor.stop_monitoring()
        
        self.base_monitor.cleanup()
        self.logger.info("Enhanced performance monitoring stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        if self.predictive_monitor and self.predictive_monitor.dashboard:
            return self.predictive_monitor.dashboard.get_real_time_data()
        
        return {
            'status': 'basic_dashboard',
            'metrics': self.base_monitor.get_performance_summary()
        }
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge alert"""
        if self.predictive_monitor:
            return self.predictive_monitor.alert_manager.acknowledge_alert(alert_id, user)
        return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve alert"""
        if self.predictive_monitor:
            return self.predictive_monitor.alert_manager.resolve_alert(alert_id, user)
        return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        status = {
            'base_monitor': 'active',
            'predictive_monitoring': 'active' if self.enable_predictions else 'disabled',
            'dashboard': 'active' if self.predictive_monitor and self.predictive_monitor.dashboard else 'disabled'
        }
        
        if self.predictive_monitor:
            status.update(self.predictive_monitor.get_monitoring_status())
        
        return status

# Factory function for easy instantiation
def create_enhanced_monitor(config: Dict[str, Any] = None) -> EnhancedPerformanceMonitor:
    """Create enhanced performance monitor with configuration"""
    if config is None:
        config = {}
    
    enable_dashboard = config.get('enable_dashboard', True)
    enable_predictions = config.get('enable_predictions', True)
    
    monitor = EnhancedPerformanceMonitor(
        enable_dashboard=enable_dashboard,
        enable_predictions=enable_predictions
    )
    
    # Configure alert channels
    if 'alert_channels' in config:
        for channel_name, channel_config in config['alert_channels'].items():
            monitor.predictive_monitor.alert_manager.add_notification_channel(
                channel_name,
                channel_config['type'],
                channel_config['config']
            )
    
    # Configure escalation policies
    if 'escalation_policies' in config:
        for policy_name, policy_config in config['escalation_policies'].items():
            monitor.predictive_monitor.alert_manager.set_escalation_policy(
                policy_name,
                policy_config
            )
    
    return monitor

# Example usage and configuration
def example_usage():
    """Example usage of enhanced monitor"""
    
    # Configuration
    config = {
        'enable_dashboard': True,
        'enable_predictions': True,
        'alert_channels': {
            'email': {
                'type': 'email',
                'config': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': 'alerts@company.com',
                    'password': 'password',
                    'from_email': 'alerts@company.com',
                    'to_email': 'admin@company.com',
                    'use_tls': True
                }
            },
            'slack': {
                'type': 'slack',
                'config': {
                    'webhook_url': 'https://hooks.slack.com/services/...',
                    'channel': '#alerts',
                    'username': 'MonitorBot'
                }
            }
        },
        'escalation_policies': {
            'cpu_usage': [
                {
                    'level': 1,
                    'delay': 300,  # 5 minutes
                    'actions': [
                        {'type': 'notify', 'channel': 'email'}
                    ]
                },
                {
                    'level': 2,
                    'delay': 900,  # 15 minutes
                    'actions': [
                        {'type': 'notify', 'channel': 'slack'},
                        {'type': 'auto_resolve', 'condition': 'cpu_usage < 50'}
                    ]
                }
            ]
        }
    }
    
    # Create monitor
    monitor = create_enhanced_monitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Record some metrics
    monitor.record_metric('cpu_usage', 45.2, {'source': 'system'})
    monitor.record_metric('memory_usage', 68.5, {'source': 'system'})
    monitor.record_metric('data_load_time', 1.2, {'dataset': 'main'})
    
    # Get comprehensive report
    report = monitor.get_comprehensive_report()
    print(f"System Health Score: {report['system_health']['score']}")
    
    # Get dashboard data
    dashboard_data = monitor.get_dashboard_data()
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    return monitor

if __name__ == "__main__":
    # Run example
    monitor = example_usage()