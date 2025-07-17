"""
Monitoring and observability components for GrandModel Strategic MARL System.

This module provides comprehensive monitoring capabilities including:
- Basic system health monitoring
- RTO (Recovery Time Objective) monitoring and validation
- Real-time dashboards and alerting
- Historical trend analysis
- Automated validation testing
"""

# Import with graceful fallback for missing dependencies
try:
    from .metrics_exporter import MetricsExporter, metrics_registry
except ImportError:
    MetricsExporter = None
    metrics_registry = None

try:
    from .health_monitor import HealthMonitor, HealthStatus
except ImportError:
    HealthMonitor = None
    HealthStatus = None

try:
    from .rto_monitor import RTOMonitoringSystem, RTOStatus, RTOTarget
    from .rto_dashboard import RTODashboard
    from .rto_alerting import RTOAlertingSystem, AlertSeverity
    from .rto_analytics import RTOAnalyticsSystem, TrendDirection
    from .rto_validation import RTOValidationFramework, TestType
    from .rto_system import RTOSystem, RTOSystemConfig
except ImportError:
    # RTO components not available
    RTOMonitoringSystem = None
    RTOStatus = None
    RTOTarget = None
    RTODashboard = None
    RTOAlertingSystem = None
    AlertSeverity = None
    RTOAnalyticsSystem = None
    TrendDirection = None
    RTOValidationFramework = None
    TestType = None
    RTOSystem = None
    RTOSystemConfig = None

__all__ = [
    # Basic monitoring
    'MetricsExporter',
    'metrics_registry',
    'HealthMonitor',
    'HealthStatus',
    
    # RTO monitoring system
    'RTOMonitoringSystem',
    'RTOStatus',
    'RTOTarget',
    'RTODashboard',
    'RTOAlertingSystem',
    'AlertSeverity',
    'RTOAnalyticsSystem',
    'TrendDirection',
    'RTOValidationFramework',
    'TestType',
    'RTOSystem',
    'RTOSystemConfig',
]