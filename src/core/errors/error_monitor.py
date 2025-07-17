"""
Error Monitoring and Alerting System

Provides real-time error monitoring, threshold-based alerting, and trend analysis
to proactively identify and respond to error patterns.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .base_exceptions import BaseGrandModelError, ErrorSeverity, ErrorCategory
from .error_logger import ErrorReport, ErrorMetrics

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ERROR_RATE_HIGH = "error_rate_high"
    CRITICAL_ERROR = "critical_error"
    REPEATED_ERROR = "repeated_error"
    SYSTEM_HEALTH = "system_health"
    TREND_ANOMALY = "trend_anomaly"


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    name: str
    alert_type: AlertType
    threshold: float
    window_size: int  # seconds
    min_occurrences: int = 1
    severity_filter: Optional[Set[ErrorSeverity]] = None
    category_filter: Optional[Set[ErrorCategory]] = None
    enabled: bool = True
    cooldown_period: int = 300  # 5 minutes
    last_triggered: float = 0.0


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    alert_type: AlertType
    level: AlertLevel
    message: str
    details: Dict[str, Any]
    timestamp: float
    correlation_id: str
    resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_notes: Optional[str] = None


class ErrorThresholdMonitor:
    """Monitors error thresholds and triggers alerts."""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.error_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule."""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
    
    def check_thresholds(self, error_report: ErrorReport) -> List[Alert]:
        """Check error against all rules and return triggered alerts."""
        alerts = []
        current_time = time.time()
        
        with self._lock:
            # Add to history
            self.error_history.append(error_report)
            
            # Check each rule
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if current_time - rule.last_triggered < rule.cooldown_period:
                    continue
                
                # Check filters
                if rule.severity_filter and error_report.severity not in [s.value for s in rule.severity_filter]:
                    continue
                
                if rule.category_filter and error_report.category not in [c.value for c in rule.category_filter]:
                    continue
                
                # Check threshold
                if self._check_rule_threshold(rule, current_time):
                    alert = self._create_alert(rule, error_report)
                    alerts.append(alert)
                    self.alerts.append(alert)
                    rule.last_triggered = current_time
        
        return alerts
    
    def _check_rule_threshold(self, rule: AlertRule, current_time: float) -> bool:
        """Check if rule threshold is exceeded."""
        window_start = current_time - rule.window_size
        
        # Count relevant errors in window
        relevant_errors = [
            error for error in self.error_history
            if error.timestamp >= window_start
            and (not rule.severity_filter or error.severity in [s.value for s in rule.severity_filter])
            and (not rule.category_filter or error.category in [c.value for c in rule.category_filter])
        ]
        
        error_count = len(relevant_errors)
        
        if rule.alert_type == AlertType.THRESHOLD_EXCEEDED:
            return error_count >= rule.threshold
        elif rule.alert_type == AlertType.ERROR_RATE_HIGH:
            error_rate = error_count / (rule.window_size / 60)  # per minute
            return error_rate >= rule.threshold
        elif rule.alert_type == AlertType.REPEATED_ERROR:
            # Check for repeated identical errors
            error_signatures = defaultdict(int)
            for error in relevant_errors:
                signature = f"{error.error_type}:{error.error_code}"
                error_signatures[signature] += 1
            
            return any(count >= rule.threshold for count in error_signatures.values())
        
        return False
    
    def _create_alert(self, rule: AlertRule, error_report: ErrorReport) -> Alert:
        """Create alert from rule and error report."""
        import uuid
        
        alert_level = AlertLevel.WARNING
        if rule.alert_type == AlertType.CRITICAL_ERROR:
            alert_level = AlertLevel.CRITICAL
        elif error_report.severity == "critical":
            alert_level = AlertLevel.CRITICAL
        elif error_report.severity == "high":
            alert_level = AlertLevel.ERROR
        
        return Alert(
            id=str(uuid.uuid4()),
            rule_name=rule.name,
            alert_type=rule.alert_type,
            level=alert_level,
            message=f"Alert rule '{rule.name}' triggered",
            details={
                "rule": rule.name,
                "threshold": rule.threshold,
                "window_size": rule.window_size,
                "triggering_error": error_report.error_message,
                "error_type": error_report.error_type,
                "correlation_id": error_report.correlation_id
            },
            timestamp=time.time(),
            correlation_id=error_report.correlation_id
        )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = time.time()
                    alert.resolution_notes = resolution_notes
                    break


class ErrorTrendAnalyzer:
    """Analyzes error trends and detects anomalies."""
    
    def __init__(self, window_size: int = 3600):
        self.window_size = window_size
        self.trend_data: Dict[str, deque] = defaultdict(lambda: deque())
        self.baseline_rates: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def add_error(self, error_report: ErrorReport):
        """Add error to trend analysis."""
        with self._lock:
            current_time = time.time()
            
            # Add to overall trend
            self.trend_data['overall'].append(current_time)
            
            # Add to category trend
            self.trend_data[f"category_{error_report.category}"].append(current_time)
            
            # Add to severity trend
            self.trend_data[f"severity_{error_report.severity}"].append(current_time)
            
            # Clean old data
            self._clean_old_data(current_time)
    
    def _clean_old_data(self, current_time: float):
        """Clean old trend data outside window."""
        cutoff_time = current_time - self.window_size
        
        for key, data in self.trend_data.items():
            while data and data[0] < cutoff_time:
                data.popleft()
    
    def get_error_rate(self, trend_key: str = 'overall') -> float:
        """Get current error rate for trend."""
        with self._lock:
            if trend_key not in self.trend_data:
                return 0.0
            
            count = len(self.trend_data[trend_key])
            return count / (self.window_size / 60)  # per minute
    
    def detect_anomalies(self, threshold_multiplier: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in error trends."""
        anomalies = []
        
        with self._lock:
            for key, data in self.trend_data.items():
                if len(data) < 10:  # Need minimum data points
                    continue
                
                current_rate = self.get_error_rate(key)
                baseline_rate = self.baseline_rates.get(key, 0.0)
                
                # Update baseline (exponential moving average)
                alpha = 0.1
                self.baseline_rates[key] = (
                    alpha * current_rate + (1 - alpha) * baseline_rate
                )
                
                # Check for anomaly
                if current_rate > baseline_rate * threshold_multiplier:
                    anomalies.append({
                        'trend_key': key,
                        'current_rate': current_rate,
                        'baseline_rate': baseline_rate,
                        'threshold': baseline_rate * threshold_multiplier,
                        'anomaly_factor': current_rate / baseline_rate if baseline_rate > 0 else float('inf')
                    })
        
        return anomalies
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of all trends."""
        with self._lock:
            summary = {}
            
            for key, data in self.trend_data.items():
                summary[key] = {
                    'count': len(data),
                    'rate': self.get_error_rate(key),
                    'baseline_rate': self.baseline_rates.get(key, 0.0)
                }
            
            return summary


class ErrorNotifier:
    """Handles error notifications via various channels."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.notification_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def send_alert(self, alert: Alert):
        """Send alert notification."""
        with self._lock:
            notification = {
                'timestamp': time.time(),
                'alert_id': alert.id,
                'alert_type': alert.alert_type.value,
                'level': alert.level.value,
                'message': alert.message,
                'details': alert.details
            }
            
            self.notification_history.append(notification)
            
            # Send via configured channels
            if self.config.get('email', {}).get('enabled', False):
                self._send_email_alert(alert)
            
            if self.config.get('webhook', {}).get('enabled', False):
                self._send_webhook_alert(alert)
            
            if self.config.get('log', {}).get('enabled', True):
                self._log_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.message}"
            
            body = f"""
            Alert Details:
            - ID: {alert.id}
            - Type: {alert.alert_type.value}
            - Level: {alert.level.value}
            - Message: {alert.message}
            - Timestamp: {datetime.fromtimestamp(alert.timestamp)}
            - Correlation ID: {alert.correlation_id}
            
            Details:
            {json.dumps(alert.details, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert."""
        try:
            import requests
            
            webhook_config = self.config['webhook']
            
            payload = {
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'alert_type': alert.alert_type.value,
                'level': alert.level.value,
                'message': alert.message,
                'details': alert.details,
                'timestamp': alert.timestamp,
                'correlation_id': alert.correlation_id
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to logger."""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.WARNING)
        
        logger.log(
            log_level,
            f"ALERT: {alert.message}",
            extra={
                'alert_id': alert.id,
                'alert_type': alert.alert_type.value,
                'level': alert.level.value,
                'details': alert.details,
                'correlation_id': alert.correlation_id
            }
        )


class AlertManager:
    """Manages alert rules and notifications."""
    
    def __init__(self, notifier_config: Optional[Dict[str, Any]] = None):
        self.threshold_monitor = ErrorThresholdMonitor()
        self.trend_analyzer = ErrorTrendAnalyzer()
        self.notifier = ErrorNotifier(notifier_config)
        self.running = False
        self.monitor_thread = None
        self._lock = threading.Lock()
    
    def add_default_rules(self):
        """Add default alert rules."""
        default_rules = [
            AlertRule(
                name="critical_errors",
                alert_type=AlertType.CRITICAL_ERROR,
                threshold=1,
                window_size=60,
                severity_filter={ErrorSeverity.CRITICAL}
            ),
            AlertRule(
                name="high_error_rate",
                alert_type=AlertType.ERROR_RATE_HIGH,
                threshold=10,  # 10 errors per minute
                window_size=300
            ),
            AlertRule(
                name="repeated_errors",
                alert_type=AlertType.REPEATED_ERROR,
                threshold=5,
                window_size=300
            ),
            AlertRule(
                name="security_alerts",
                alert_type=AlertType.THRESHOLD_EXCEEDED,
                threshold=1,
                window_size=60,
                category_filter={ErrorCategory.SECURITY}
            )
        ]
        
        for rule in default_rules:
            self.threshold_monitor.add_rule(rule)
    
    def process_error(self, error_report: ErrorReport):
        """Process error report through monitoring system."""
        # Check thresholds
        alerts = self.threshold_monitor.check_thresholds(error_report)
        
        # Add to trend analysis
        self.trend_analyzer.add_error(error_report)
        
        # Send alerts
        for alert in alerts:
            self.notifier.send_alert(alert)
    
    def start_monitoring(self):
        """Start background monitoring."""
        with self._lock:
            if not self.running:
                self.running = True
                self.monitor_thread = threading.Thread(target=self._monitor_loop)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        with self._lock:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Check for trend anomalies
                anomalies = self.trend_analyzer.detect_anomalies()
                
                for anomaly in anomalies:
                    # Create trend anomaly alert
                    alert = Alert(
                        id=str(time.time()),
                        rule_name="trend_anomaly",
                        alert_type=AlertType.TREND_ANOMALY,
                        level=AlertLevel.WARNING,
                        message=f"Trend anomaly detected in {anomaly['trend_key']}",
                        details=anomaly,
                        timestamp=time.time(),
                        correlation_id=f"trend_{anomaly['trend_key']}"
                    )
                    
                    self.notifier.send_alert(alert)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            'active_alerts': self.threshold_monitor.get_active_alerts(),
            'trend_summary': self.trend_analyzer.get_trend_summary(),
            'notification_history': self.notifier.notification_history[-50:],  # Last 50
            'alert_rules': list(self.threshold_monitor.rules.keys())
        }


class ErrorMonitor:
    """Main error monitoring interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_manager = AlertManager(self.config.get('notifications'))
        
        # Add default rules
        self.alert_manager.add_default_rules()
        
        # Start monitoring
        if self.config.get('auto_start', True):
            self.alert_manager.start_monitoring()
    
    def monitor_error(self, error_report: ErrorReport):
        """Monitor error report."""
        self.alert_manager.process_error(error_report)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule."""
        self.alert_manager.threshold_monitor.add_rule(rule)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        return self.alert_manager.get_dashboard_data()
    
    def start(self):
        """Start monitoring."""
        self.alert_manager.start_monitoring()
    
    def stop(self):
        """Stop monitoring."""
        self.alert_manager.stop_monitoring()


# Global error monitor instance
_global_error_monitor = None
_monitor_lock = threading.Lock()


def get_global_error_monitor() -> ErrorMonitor:
    """Get global error monitor instance."""
    global _global_error_monitor
    
    if _global_error_monitor is None:
        with _monitor_lock:
            if _global_error_monitor is None:
                _global_error_monitor = ErrorMonitor()
    
    return _global_error_monitor


def monitor_error(error_report: ErrorReport):
    """Monitor error using global monitor."""
    get_global_error_monitor().monitor_error(error_report)