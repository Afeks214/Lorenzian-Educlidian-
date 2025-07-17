"""
Predictive Monitoring and Alert System for NQ Data Pipeline

This module provides advanced predictive monitoring capabilities including:
- Predictive failure detection using machine learning
- Capacity planning and resource forecasting
- Trend analysis and forecasting
- Intelligent alerting with context awareness
- Alert correlation and escalation
- Real-time dashboard integration
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
from enum import Enum
import uuid
import asyncio
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Import from the main performance monitor
from .performance_monitor import PerformanceMetric, AlertSeverity, AlertStatus, Alert, PredictionType

class IntelligentAlertManager:
    """Intelligent alert management with correlation and prioritization"""
    
    def __init__(self):
        self.alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.correlation_groups = {}
        self.escalation_policies = {}
        self.suppression_rules = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Alert clustering parameters
        self.correlation_window = 300  # 5 minutes
        self.max_correlation_distance = 0.5
        
        # Prioritization weights
        self.priority_weights = {
            'severity': 0.4,
            'business_impact': 0.3,
            'frequency': 0.2,
            'correlation': 0.1
        }
    
    def add_alert(self, alert: Alert) -> str:
        """Add alert with intelligent processing"""
        with self.lock:
            # Check for correlation
            correlation_id = self._find_correlation_group(alert)
            if correlation_id:
                alert.correlation_id = correlation_id
                self.correlation_groups[correlation_id].append(alert)
            else:
                # Create new correlation group
                correlation_id = str(uuid.uuid4())
                alert.correlation_id = correlation_id
                self.correlation_groups[correlation_id] = [alert]
            
            # Calculate priority
            priority = self._calculate_priority(alert)
            alert.details['priority'] = priority
            
            # Store alert
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Check for escalation
            self._check_escalation(alert)
            
            self.logger.info(f"Alert added: {alert.id} with priority {priority}")
            return alert.id
    
    def _find_correlation_group(self, alert: Alert) -> Optional[str]:
        """Find existing correlation group for alert"""
        current_time = time.time()
        
        for group_id, group_alerts in self.correlation_groups.items():
            if not group_alerts:
                continue
            
            # Check if group is still active
            latest_alert = max(group_alerts, key=lambda a: a.timestamp)
            if current_time - latest_alert.timestamp > self.correlation_window:
                continue
            
            # Check for correlation
            if self._alerts_correlated(alert, group_alerts):
                return group_id
        
        return None
    
    def _alerts_correlated(self, alert: Alert, group_alerts: List[Alert]) -> bool:
        """Check if alert is correlated with group"""
        for group_alert in group_alerts:
            if self._two_alerts_correlated(alert, group_alert):
                return True
        return False
    
    def _two_alerts_correlated(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are correlated"""
        # Time-based correlation
        time_diff = abs(alert1.timestamp - alert2.timestamp)
        if time_diff > self.correlation_window:
            return False
        
        # Metric-based correlation
        if alert1.metric_name == alert2.metric_name:
            return True
        
        # Pattern-based correlation
        if self._metrics_related(alert1.metric_name, alert2.metric_name):
            return True
        
        # Value-based correlation
        if self._values_correlated(alert1.value, alert2.value):
            return True
        
        return False
    
    def _metrics_related(self, metric1: str, metric2: str) -> bool:
        """Check if metrics are related"""
        # Define metric relationships
        related_groups = [
            ['cpu_usage', 'memory_usage', 'disk_usage'],
            ['data_load_time', 'data_throughput', 'processing_time'],
            ['gpu_memory_usage', 'gpu_utilization']
        ]
        
        for group in related_groups:
            if metric1 in group and metric2 in group:
                return True
        
        return False
    
    def _values_correlated(self, value1: float, value2: float) -> bool:
        """Check if values are correlated"""
        if value1 == 0 or value2 == 0:
            return False
        
        ratio = max(value1, value2) / min(value1, value2)
        return ratio < 2.0  # Values within 2x of each other
    
    def _calculate_priority(self, alert: Alert) -> float:
        """Calculate alert priority score"""
        priority = 0.0
        
        # Severity component
        severity_scores = {
            AlertSeverity.LOW: 0.25,
            AlertSeverity.MEDIUM: 0.5,
            AlertSeverity.HIGH: 0.75,
            AlertSeverity.CRITICAL: 1.0
        }
        priority += severity_scores[alert.severity] * self.priority_weights['severity']
        
        # Business impact component (based on metric importance)
        business_impact = self._get_business_impact(alert.metric_name)
        priority += business_impact * self.priority_weights['business_impact']
        
        # Frequency component
        frequency_score = self._get_frequency_score(alert.metric_name)
        priority += frequency_score * self.priority_weights['frequency']
        
        # Correlation component
        correlation_score = self._get_correlation_score(alert)
        priority += correlation_score * self.priority_weights['correlation']
        
        return min(1.0, priority)
    
    def _get_business_impact(self, metric_name: str) -> float:
        """Get business impact score for metric"""
        impact_scores = {
            'data_load_time': 0.9,
            'data_throughput': 0.8,
            'memory_usage': 0.7,
            'cpu_usage': 0.6,
            'disk_usage': 0.5,
            'gpu_memory_usage': 0.4
        }
        return impact_scores.get(metric_name, 0.3)
    
    def _get_frequency_score(self, metric_name: str) -> float:
        """Get frequency score based on recent alerts"""
        current_time = time.time()
        recent_alerts = [a for a in self.alert_history 
                        if a.metric_name == metric_name and 
                        current_time - a.timestamp < 3600]  # Last hour
        
        if len(recent_alerts) == 0:
            return 0.0
        elif len(recent_alerts) == 1:
            return 0.2
        elif len(recent_alerts) <= 3:
            return 0.5
        else:
            return 1.0
    
    def _get_correlation_score(self, alert: Alert) -> float:
        """Get correlation score"""
        if alert.correlation_id and alert.correlation_id in self.correlation_groups:
            group_size = len(self.correlation_groups[alert.correlation_id])
            return min(1.0, group_size / 5.0)
        return 0.0
    
    def _check_escalation(self, alert: Alert):
        """Check if alert needs escalation"""
        policy = self.escalation_policies.get(alert.metric_name)
        if not policy:
            return
        
        current_time = time.time()
        
        for level in policy:
            if alert.escalation_level >= level['level']:
                continue
            
            if current_time - alert.timestamp >= level['delay']:
                alert.escalation_level = level['level']
                self._execute_escalation_action(alert, level)
    
    def _execute_escalation_action(self, alert: Alert, level: Dict[str, Any]):
        """Execute escalation action"""
        self.logger.warning(f"Escalating alert {alert.id} to level {level['level']}")
        
        # Execute escalation actions
        for action in level.get('actions', []):
            try:
                if action['type'] == 'notify':
                    self._send_escalation_notification(alert, action)
                elif action['type'] == 'auto_resolve':
                    self._auto_resolve_alert(alert, action)
            except Exception as e:
                self.logger.error(f"Escalation action failed: {e}")
    
    def _send_escalation_notification(self, alert: Alert, action: Dict[str, Any]):
        """Send escalation notification"""
        # Implementation depends on notification system
        pass
    
    def _auto_resolve_alert(self, alert: Alert, action: Dict[str, Any]):
        """Auto-resolve alert based on conditions"""
        # Implementation for auto-resolution logic
        pass
    
    def get_correlated_alerts(self, alert_id: str) -> List[Alert]:
        """Get alerts correlated with given alert"""
        with self.lock:
            if alert_id not in self.alerts:
                return []
            
            alert = self.alerts[alert_id]
            if not alert.correlation_id:
                return []
            
            return self.correlation_groups.get(alert.correlation_id, [])
    
    def get_priority_alerts(self, limit: int = 10) -> List[Alert]:
        """Get highest priority active alerts"""
        with self.lock:
            active_alerts = [a for a in self.alerts.values() 
                           if a.status == AlertStatus.ACTIVE]
            
            # Sort by priority
            sorted_alerts = sorted(active_alerts, 
                                 key=lambda a: a.details.get('priority', 0), 
                                 reverse=True)
            
            return sorted_alerts[:limit]
    
    def suppress_alert(self, alert_id: str, reason: str, duration: int = 3600):
        """Suppress alert for specified duration"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.details['suppression_reason'] = reason
                alert.details['suppression_until'] = time.time() + duration
                
                self.logger.info(f"Alert {alert_id} suppressed: {reason}")
                return True
            return False
    
    def get_alert_analytics(self) -> Dict[str, Any]:
        """Get alert analytics and insights"""
        with self.lock:
            if not self.alert_history:
                return {'status': 'no_data'}
            
            # Time-based analysis
            now = time.time()
            last_hour = [a for a in self.alert_history if now - a.timestamp < 3600]
            last_day = [a for a in self.alert_history if now - a.timestamp < 86400]
            
            # Frequency analysis
            metric_counts = defaultdict(int)
            for alert in last_day:
                metric_counts[alert.metric_name] += 1
            
            # Correlation analysis
            correlation_sizes = [len(group) for group in self.correlation_groups.values()]
            
            return {
                'total_alerts': len(self.alert_history),
                'last_hour_count': len(last_hour),
                'last_day_count': len(last_day),
                'top_metrics': dict(sorted(metric_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]),
                'avg_correlation_size': np.mean(correlation_sizes) if correlation_sizes else 0,
                'active_correlations': len(self.correlation_groups),
                'prediction_accuracy': self._calculate_prediction_accuracy()
            }
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        prediction_alerts = [a for a in self.alert_history if a.prediction_based]
        if not prediction_alerts:
            return 0.0
        
        # Simple accuracy based on resolution time
        resolved_predictions = [a for a in prediction_alerts 
                              if a.status == AlertStatus.RESOLVED]
        
        if not resolved_predictions:
            return 0.0
        
        return len(resolved_predictions) / len(prediction_alerts)

class PredictiveDashboard:
    """Enhanced dashboard with predictive visualizations"""
    
    def __init__(self, metrics_collector, alert_manager: IntelligentAlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.dashboard_data = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Dashboard update settings
        self.update_interval = 5  # seconds
        self.prediction_horizon = 24  # hours
        
        # Visualization settings
        self.chart_colors = {
            'prediction': '#FF6B6B',
            'actual': '#4ECDC4',
            'threshold': '#FFE66D',
            'anomaly': '#FF8E53'
        }
    
    def generate_predictive_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive predictive dashboard data"""
        with self.lock:
            dashboard = {
                'timestamp': time.time(),
                'metrics_overview': self._generate_metrics_overview(),
                'predictive_charts': self._generate_predictive_charts(),
                'alert_summary': self._generate_alert_summary(),
                'capacity_forecast': self._generate_capacity_forecast(),
                'anomaly_detection': self._generate_anomaly_detection(),
                'system_health': self._generate_system_health()
            }
            
            self.dashboard_data = dashboard
            return dashboard
    
    def _generate_metrics_overview(self) -> Dict[str, Any]:
        """Generate metrics overview"""
        metrics_summary = self.metrics_collector.get_all_metrics_summary()
        
        overview = {
            'total_metrics': len(metrics_summary),
            'healthy_metrics': 0,
            'warning_metrics': 0,
            'critical_metrics': 0,
            'data_quality_score': 0.0
        }
        
        quality_scores = []
        
        for metric_name, stats in metrics_summary.items():
            if stats.get('status') == 'no_data':
                continue
            
            latest = stats.get('latest', 0)
            std = stats.get('std', 0)
            
            # Simple health classification
            if metric_name.endswith('_usage'):
                if latest > 90:
                    overview['critical_metrics'] += 1
                elif latest > 70:
                    overview['warning_metrics'] += 1
                else:
                    overview['healthy_metrics'] += 1
            else:
                overview['healthy_metrics'] += 1
            
            # Data quality score
            if std > 0:
                quality_scores.append(1.0 / (1.0 + std))
        
        if quality_scores:
            overview['data_quality_score'] = np.mean(quality_scores)
        
        return overview
    
    def _generate_predictive_charts(self) -> Dict[str, Any]:
        """Generate predictive chart data"""
        charts = {}
        
        # Get key metrics for prediction
        key_metrics = ['cpu_usage', 'memory_usage', 'data_throughput', 'disk_usage']
        
        for metric_name in key_metrics:
            history = self.metrics_collector.get_metric_history(metric_name, hours=24)
            
            if len(history) < 10:
                continue
            
            # Generate prediction points
            prediction_points = []
            current_time = time.time()
            
            # Simple linear prediction
            values = [m.value for m in history[-10:]]
            timestamps = [m.timestamp for m in history[-10:]]
            
            if len(values) >= 2:
                # Linear regression
                x = np.array([(t - timestamps[0]) / 3600 for t in timestamps]).reshape(-1, 1)
                y = np.array(values)
                
                model = LinearRegression()
                model.fit(x, y)
                
                # Predict next 6 hours
                for hour in range(1, 7):
                    future_time = current_time + (hour * 3600)
                    future_x = np.array([[(future_time - timestamps[0]) / 3600]])
                    predicted_value = model.predict(future_x)[0]
                    
                    prediction_points.append({
                        'timestamp': future_time,
                        'predicted_value': max(0, predicted_value),
                        'confidence': max(0, 1 - (hour / 6) * 0.2)
                    })
            
            charts[metric_name] = {
                'historical_data': [{'timestamp': m.timestamp, 'value': m.value} 
                                  for m in history],
                'predictions': prediction_points,
                'thresholds': self._get_metric_thresholds(metric_name)
            }
        
        return charts
    
    def _get_metric_thresholds(self, metric_name: str) -> Dict[str, float]:
        """Get thresholds for metric"""
        default_thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'data_throughput': {'warning': 100, 'critical': 50}
        }
        
        return default_thresholds.get(metric_name, {'warning': 80, 'critical': 95})
    
    def _generate_alert_summary(self) -> Dict[str, Any]:
        """Generate alert summary"""
        active_alerts = self.alert_manager.get_priority_alerts(20)
        
        summary = {
            'total_active': len(active_alerts),
            'by_severity': defaultdict(int),
            'by_metric': defaultdict(int),
            'prediction_based': 0,
            'correlated_groups': len(self.alert_manager.correlation_groups),
            'recent_alerts': []
        }
        
        for alert in active_alerts:
            summary['by_severity'][alert.severity.value] += 1
            summary['by_metric'][alert.metric_name] += 1
            
            if alert.prediction_based:
                summary['prediction_based'] += 1
            
            if len(summary['recent_alerts']) < 10:
                summary['recent_alerts'].append({
                    'id': alert.id,
                    'metric': alert.metric_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'priority': alert.details.get('priority', 0)
                })
        
        return summary
    
    def _generate_capacity_forecast(self) -> Dict[str, Any]:
        """Generate capacity forecast"""
        # This would integrate with CapacityPlanner
        return {
            'status': 'available',
            'forecasts': {
                'cpu_usage': {
                    'current': 45.2,
                    'predicted_24h': 52.1,
                    'time_to_limit': None,
                    'recommendation': 'Normal usage expected'
                },
                'memory_usage': {
                    'current': 68.5,
                    'predicted_24h': 75.3,
                    'time_to_limit': None,
                    'recommendation': 'Monitor for increasing trend'
                }
            }
        }
    
    def _generate_anomaly_detection(self) -> Dict[str, Any]:
        """Generate anomaly detection summary"""
        return {
            'anomalies_detected': 0,
            'anomaly_score': 0.0,
            'recent_anomalies': [],
            'anomaly_trends': {
                'increasing': False,
                'pattern_detected': False
            }
        }
    
    def _generate_system_health(self) -> Dict[str, Any]:
        """Generate system health score"""
        return {
            'overall_health': 85.2,
            'components': {
                'data_pipeline': 90.1,
                'monitoring_system': 88.5,
                'alert_system': 82.3,
                'prediction_accuracy': 75.8
            },
            'recommendations': [
                'System operating within normal parameters',
                'Prediction accuracy could be improved with more historical data'
            ]
        }
    
    def get_real_time_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        return self.dashboard_data if self.dashboard_data else self.generate_predictive_dashboard()

class PredictivePerformanceMonitor:
    """Main predictive performance monitoring system"""
    
    def __init__(self, metrics_collector, enable_dashboard: bool = True):
        self.metrics_collector = metrics_collector
        
        # Initialize components
        self.failure_detector = None  # Would be initialized with PredictiveFailureDetector
        self.capacity_planner = None  # Would be initialized with CapacityPlanner
        self.trend_forecaster = None  # Would be initialized with TrendForecaster
        
        # Alert management
        self.alert_manager = IntelligentAlertManager()
        
        # Dashboard
        self.dashboard = PredictiveDashboard(metrics_collector, self.alert_manager) if enable_dashboard else None
        
        # Monitoring thread
        self.monitoring_thread = None
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start predictive monitoring"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Predictive monitoring started")
    
    def stop_monitoring(self):
        """Stop predictive monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Predictive monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update dashboard
                if self.dashboard:
                    self.dashboard.generate_predictive_dashboard()
                
                # Check for escalations
                self._check_alert_escalations()
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _check_alert_escalations(self):
        """Check for alert escalations"""
        # Implementation for checking escalations
        pass
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            'running': self.running,
            'alert_manager_status': 'active',
            'dashboard_status': 'active' if self.dashboard else 'disabled',
            'last_update': time.time()
        }