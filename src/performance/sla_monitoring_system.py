"""
SLA Monitoring and Alerting System

This module provides comprehensive SLA monitoring with real-time alerting,
escalation policies, and automated response mechanisms.

Features:
- Real-time SLA monitoring
- Multi-level alerting system
- Escalation policies
- Automated response mechanisms
- SLA breach tracking
- Performance trend analysis
- Customizable notification channels
- Historical SLA reporting

Author: Performance Validation Agent
"""

import asyncio
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import structlog
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
from collections import defaultdict, deque
import statistics
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class SLAStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"

class ResponseAction(Enum):
    NOTIFY = "notify"
    SCALE_UP = "scale_up"
    RESTART_SERVICE = "restart_service"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAILOVER = "failover"

@dataclass
class SLADefinition:
    """SLA definition with thresholds and targets"""
    name: str
    description: str
    metric_name: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    breach_threshold: float
    unit: str
    measurement_window_minutes: int = 5
    evaluation_frequency_seconds: int = 60
    breach_duration_minutes: int = 5
    enabled: bool = True

@dataclass
class SLAMetric:
    """SLA metric measurement"""
    sla_name: str
    timestamp: datetime
    value: float
    status: SLAStatus
    threshold_exceeded: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SLAAlert:
    """SLA alert"""
    id: str
    sla_name: str
    severity: AlertSeverity
    status: SLAStatus
    timestamp: datetime
    message: str
    current_value: float
    threshold_value: float
    escalation_level: int = 0
    acknowledged: bool = False
    resolved: bool = False
    response_actions: List[ResponseAction] = field(default_factory=list)

@dataclass
class EscalationPolicy:
    """Alert escalation policy"""
    sla_name: str
    escalation_levels: List[Dict[str, Any]]
    max_escalation_level: int
    escalation_interval_minutes: int
    auto_resolution_enabled: bool = True
    auto_resolution_threshold: float = 0.9

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    channel_type: AlertChannel
    configuration: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)

class SLAMonitoringSystem:
    """
    Comprehensive SLA monitoring system with real-time alerting and response
    """

    def __init__(self, db_path: str = "sla_monitoring.db"):
        self.db_path = db_path
        self.sla_definitions = {}
        self.escalation_policies = {}
        self.notification_channels = {}
        self.active_alerts = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history = deque(maxlen=10000)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.evaluation_thread = None
        
        # Initialize system
        self._init_database()
        self._load_default_slas()
        self._load_default_escalation_policies()
        self._load_default_notification_channels()
        
        logger.info("SLA monitoring system initialized",
                   slas=len(self.sla_definitions),
                   policies=len(self.escalation_policies),
                   channels=len(self.notification_channels))

    def _init_database(self):
        """Initialize SLA monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SLA definitions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_definitions (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                target_value REAL NOT NULL,
                warning_threshold REAL NOT NULL,
                critical_threshold REAL NOT NULL,
                breach_threshold REAL NOT NULL,
                unit TEXT NOT NULL,
                measurement_window_minutes INTEGER NOT NULL,
                evaluation_frequency_seconds INTEGER NOT NULL,
                breach_duration_minutes INTEGER NOT NULL,
                enabled BOOLEAN NOT NULL
            )
        """)
        
        # SLA metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sla_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                status TEXT NOT NULL,
                threshold_exceeded TEXT,
                context TEXT
            )
        """)
        
        # SLA alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_alerts (
                id TEXT PRIMARY KEY,
                sla_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                escalation_level INTEGER NOT NULL,
                acknowledged BOOLEAN NOT NULL,
                resolved BOOLEAN NOT NULL,
                response_actions TEXT
            )
        """)
        
        # SLA breaches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_breaches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sla_name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_minutes REAL,
                severity TEXT NOT NULL,
                impact_score REAL NOT NULL,
                root_cause TEXT,
                resolution_actions TEXT,
                lessons_learned TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def _load_default_slas(self):
        """Load default SLA definitions"""
        default_slas = [
            SLADefinition(
                name="strategic_inference_latency",
                description="Strategic inference latency must be under 50ms",
                metric_name="strategic_inference_latency_ms",
                target_value=50.0,
                warning_threshold=40.0,
                critical_threshold=45.0,
                breach_threshold=50.0,
                unit="ms",
                measurement_window_minutes=5,
                evaluation_frequency_seconds=30
            ),
            SLADefinition(
                name="tactical_inference_latency",
                description="Tactical inference latency must be under 20ms",
                metric_name="tactical_inference_latency_ms",
                target_value=20.0,
                warning_threshold=15.0,
                critical_threshold=18.0,
                breach_threshold=20.0,
                unit="ms",
                measurement_window_minutes=5,
                evaluation_frequency_seconds=30
            ),
            SLADefinition(
                name="database_rto",
                description="Database recovery time objective must be under 30 seconds",
                metric_name="database_rto_seconds",
                target_value=30.0,
                warning_threshold=20.0,
                critical_threshold=25.0,
                breach_threshold=30.0,
                unit="seconds",
                measurement_window_minutes=1,
                evaluation_frequency_seconds=60
            ),
            SLADefinition(
                name="trading_engine_rto",
                description="Trading engine recovery time objective must be under 5 seconds",
                metric_name="trading_engine_rto_seconds",
                target_value=5.0,
                warning_threshold=3.0,
                critical_threshold=4.0,
                breach_threshold=5.0,
                unit="seconds",
                measurement_window_minutes=1,
                evaluation_frequency_seconds=60
            ),
            SLADefinition(
                name="system_availability",
                description="System availability must be above 99.9%",
                metric_name="system_availability_percent",
                target_value=99.9,
                warning_threshold=99.5,
                critical_threshold=99.0,
                breach_threshold=99.9,
                unit="percent",
                measurement_window_minutes=60,
                evaluation_frequency_seconds=300
            ),
            SLADefinition(
                name="end_to_end_latency",
                description="End-to-end pipeline latency must be under 100ms",
                metric_name="end_to_end_latency_ms",
                target_value=100.0,
                warning_threshold=80.0,
                critical_threshold=90.0,
                breach_threshold=100.0,
                unit="ms",
                measurement_window_minutes=5,
                evaluation_frequency_seconds=60
            )
        ]
        
        for sla in default_slas:
            self.sla_definitions[sla.name] = sla

    def _load_default_escalation_policies(self):
        """Load default escalation policies"""
        default_policies = [
            EscalationPolicy(
                sla_name="strategic_inference_latency",
                escalation_levels=[
                    {"level": 1, "channels": ["email"], "delay_minutes": 0},
                    {"level": 2, "channels": ["email", "slack"], "delay_minutes": 5},
                    {"level": 3, "channels": ["email", "slack", "pagerduty"], "delay_minutes": 15}
                ],
                max_escalation_level=3,
                escalation_interval_minutes=10
            ),
            EscalationPolicy(
                sla_name="database_rto",
                escalation_levels=[
                    {"level": 1, "channels": ["email", "slack"], "delay_minutes": 0},
                    {"level": 2, "channels": ["email", "slack", "pagerduty"], "delay_minutes": 2},
                    {"level": 3, "channels": ["email", "slack", "pagerduty", "sms"], "delay_minutes": 5}
                ],
                max_escalation_level=3,
                escalation_interval_minutes=5
            ),
            EscalationPolicy(
                sla_name="trading_engine_rto",
                escalation_levels=[
                    {"level": 1, "channels": ["email", "slack"], "delay_minutes": 0},
                    {"level": 2, "channels": ["email", "slack", "pagerduty"], "delay_minutes": 1},
                    {"level": 3, "channels": ["email", "slack", "pagerduty", "sms"], "delay_minutes": 3}
                ],
                max_escalation_level=3,
                escalation_interval_minutes=3
            )
        ]
        
        for policy in default_policies:
            self.escalation_policies[policy.sla_name] = policy

    def _load_default_notification_channels(self):
        """Load default notification channels"""
        default_channels = [
            NotificationChannel(
                name="primary_email",
                channel_type=AlertChannel.EMAIL,
                configuration={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "alerts@grandmodel.com",
                    "sender_password": "your_app_password",
                    "recipients": ["team@grandmodel.com"]
                },
                severity_filter=[AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ),
            NotificationChannel(
                name="slack_alerts",
                channel_type=AlertChannel.SLACK,
                configuration={
                    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                    "channel": "#alerts"
                },
                severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ),
            NotificationChannel(
                name="pagerduty_critical",
                channel_type=AlertChannel.PAGERDUTY,
                configuration={
                    "integration_key": "your_pagerduty_integration_key",
                    "service_key": "your_service_key"
                },
                severity_filter=[AlertSeverity.CRITICAL]
            )
        ]
        
        for channel in default_channels:
            self.notification_channels[channel.name] = channel

    def start_monitoring(self):
        """Start SLA monitoring"""
        if self.monitoring_active:
            logger.warning("SLA monitoring is already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start evaluation thread
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop)
        self.evaluation_thread.daemon = True
        self.evaluation_thread.start()
        
        logger.info("SLA monitoring started",
                   active_slas=len([s for s in self.sla_definitions.values() if s.enabled]))

    def stop_monitoring(self):
        """Stop SLA monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5)
        
        logger.info("SLA monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics for all SLAs
                for sla_name, sla_def in self.sla_definitions.items():
                    if sla_def.enabled:
                        self._collect_sla_metric(sla_def)
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(30)

    def _evaluation_loop(self):
        """SLA evaluation loop"""
        while self.monitoring_active:
            try:
                # Evaluate all SLAs
                for sla_name, sla_def in self.sla_definitions.items():
                    if sla_def.enabled:
                        self._evaluate_sla(sla_def)
                
                # Process escalations
                self._process_escalations()
                
                time.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error("Error in evaluation loop", error=str(e))
                time.sleep(60)

    def _collect_sla_metric(self, sla_def: SLADefinition):
        """Collect metric for a specific SLA"""
        try:
            # Get current metric value
            current_value = self._get_current_metric_value(sla_def.metric_name)
            
            if current_value is None:
                return
            
            # Determine status
            status = self._determine_sla_status(sla_def, current_value)
            
            # Create metric record
            metric = SLAMetric(
                sla_name=sla_def.name,
                timestamp=datetime.now(),
                value=current_value,
                status=status,
                threshold_exceeded=self._get_threshold_exceeded(sla_def, current_value),
                context=self._get_metric_context(sla_def.metric_name)
            )
            
            # Store metric
            self._store_sla_metric(metric)
            
            # Add to history
            self.metric_history[sla_def.name].append(metric)
            
        except Exception as e:
            logger.error("Error collecting SLA metric", sla_name=sla_def.name, error=str(e))

    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value from the system"""
        # This would integrate with the performance validation framework
        # For now, we'll simulate some values
        
        metric_simulations = {
            "strategic_inference_latency_ms": lambda: np.random.uniform(30, 60),
            "tactical_inference_latency_ms": lambda: np.random.uniform(10, 25),
            "database_rto_seconds": lambda: np.random.uniform(15, 35),
            "trading_engine_rto_seconds": lambda: np.random.uniform(2, 8),
            "system_availability_percent": lambda: np.random.uniform(99.0, 100.0),
            "end_to_end_latency_ms": lambda: np.random.uniform(70, 120)
        }
        
        simulator = metric_simulations.get(metric_name)
        if simulator:
            return simulator()
        
        return None

    def _determine_sla_status(self, sla_def: SLADefinition, current_value: float) -> SLAStatus:
        """Determine SLA status based on current value"""
        if current_value >= sla_def.breach_threshold:
            return SLAStatus.BREACH
        elif current_value >= sla_def.critical_threshold:
            return SLAStatus.CRITICAL
        elif current_value >= sla_def.warning_threshold:
            return SLAStatus.WARNING
        else:
            return SLAStatus.HEALTHY

    def _get_threshold_exceeded(self, sla_def: SLADefinition, current_value: float) -> Optional[str]:
        """Get which threshold was exceeded"""
        if current_value >= sla_def.breach_threshold:
            return "breach"
        elif current_value >= sla_def.critical_threshold:
            return "critical"
        elif current_value >= sla_def.warning_threshold:
            return "warning"
        return None

    def _get_metric_context(self, metric_name: str) -> Dict[str, Any]:
        """Get additional context for the metric"""
        import psutil
        
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }

    def _evaluate_sla(self, sla_def: SLADefinition):
        """Evaluate SLA and trigger alerts if necessary"""
        try:
            # Get recent metrics
            recent_metrics = self._get_recent_metrics(sla_def.name, sla_def.measurement_window_minutes)
            
            if not recent_metrics:
                return
            
            # Calculate aggregate metrics
            current_status = self._calculate_aggregate_status(recent_metrics)
            avg_value = statistics.mean([m.value for m in recent_metrics])
            
            # Check if alert should be triggered
            if current_status in [SLAStatus.WARNING, SLAStatus.CRITICAL, SLAStatus.BREACH]:
                self._trigger_alert_if_needed(sla_def, current_status, avg_value, recent_metrics)
            else:
                # Check if existing alert should be resolved
                self._resolve_alert_if_needed(sla_def.name, current_status, avg_value)
                
        except Exception as e:
            logger.error("Error evaluating SLA", sla_name=sla_def.name, error=str(e))

    def _get_recent_metrics(self, sla_name: str, window_minutes: int) -> List[SLAMetric]:
        """Get recent metrics within the specified window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_metrics = []
        for metric in self.metric_history[sla_name]:
            if metric.timestamp >= cutoff_time:
                recent_metrics.append(metric)
        
        return recent_metrics

    def _calculate_aggregate_status(self, metrics: List[SLAMetric]) -> SLAStatus:
        """Calculate aggregate status from multiple metrics"""
        if not metrics:
            return SLAStatus.HEALTHY
        
        # Count status occurrences
        status_counts = defaultdict(int)
        for metric in metrics:
            status_counts[metric.status] += 1
        
        # Determine overall status
        if status_counts[SLAStatus.BREACH] > 0:
            return SLAStatus.BREACH
        elif status_counts[SLAStatus.CRITICAL] > len(metrics) * 0.3:  # 30% critical
            return SLAStatus.CRITICAL
        elif status_counts[SLAStatus.WARNING] > len(metrics) * 0.5:  # 50% warning
            return SLAStatus.WARNING
        else:
            return SLAStatus.HEALTHY

    def _trigger_alert_if_needed(self, sla_def: SLADefinition, status: SLAStatus, 
                                avg_value: float, recent_metrics: List[SLAMetric]):
        """Trigger alert if conditions are met"""
        
        # Check if alert already exists
        existing_alert = self._get_active_alert(sla_def.name)
        
        if existing_alert:
            # Update existing alert if status worsened
            if self._status_severity_level(status) > self._status_severity_level(existing_alert.status):
                self._update_alert_status(existing_alert, status, avg_value)
            return
        
        # Create new alert
        alert_id = f"{sla_def.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        severity = self._determine_alert_severity(status)
        
        alert = SLAAlert(
            id=alert_id,
            sla_name=sla_def.name,
            severity=severity,
            status=status,
            timestamp=datetime.now(),
            message=self._generate_alert_message(sla_def, status, avg_value),
            current_value=avg_value,
            threshold_value=self._get_threshold_for_status(sla_def, status),
            response_actions=self._determine_response_actions(sla_def, status)
        )
        
        # Store alert
        self._store_alert(alert)
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        # Execute response actions
        self._execute_response_actions(alert)
        
        logger.warning("SLA alert triggered",
                      sla_name=sla_def.name,
                      severity=severity.value,
                      status=status.value,
                      current_value=avg_value)

    def _resolve_alert_if_needed(self, sla_name: str, status: SLAStatus, avg_value: float):
        """Resolve alert if conditions are met"""
        active_alert = self._get_active_alert(sla_name)
        
        if active_alert and status == SLAStatus.HEALTHY:
            # Resolve the alert
            active_alert.resolved = True
            active_alert.status = status
            
            # Update in database
            self._update_alert_in_database(active_alert)
            
            # Remove from active alerts
            if active_alert.id in self.active_alerts:
                del self.active_alerts[active_alert.id]
            
            # Send resolution notification
            self._send_resolution_notification(active_alert)
            
            logger.info("SLA alert resolved",
                       sla_name=sla_name,
                       alert_id=active_alert.id,
                       current_value=avg_value)

    def _get_active_alert(self, sla_name: str) -> Optional[SLAAlert]:
        """Get active alert for SLA"""
        for alert in self.active_alerts.values():
            if alert.sla_name == sla_name and not alert.resolved:
                return alert
        return None

    def _status_severity_level(self, status: SLAStatus) -> int:
        """Get numerical severity level for status"""
        levels = {
            SLAStatus.HEALTHY: 0,
            SLAStatus.WARNING: 1,
            SLAStatus.CRITICAL: 2,
            SLAStatus.BREACH: 3
        }
        return levels.get(status, 0)

    def _determine_alert_severity(self, status: SLAStatus) -> AlertSeverity:
        """Determine alert severity from SLA status"""
        severity_map = {
            SLAStatus.WARNING: AlertSeverity.MEDIUM,
            SLAStatus.CRITICAL: AlertSeverity.HIGH,
            SLAStatus.BREACH: AlertSeverity.CRITICAL
        }
        return severity_map.get(status, AlertSeverity.LOW)

    def _generate_alert_message(self, sla_def: SLADefinition, status: SLAStatus, current_value: float) -> str:
        """Generate alert message"""
        return f"SLA '{sla_def.name}' is in {status.value} state. " \
               f"Current value: {current_value:.2f} {sla_def.unit}, " \
               f"Target: {sla_def.target_value:.2f} {sla_def.unit}"

    def _get_threshold_for_status(self, sla_def: SLADefinition, status: SLAStatus) -> float:
        """Get threshold value for the given status"""
        threshold_map = {
            SLAStatus.WARNING: sla_def.warning_threshold,
            SLAStatus.CRITICAL: sla_def.critical_threshold,
            SLAStatus.BREACH: sla_def.breach_threshold
        }
        return threshold_map.get(status, sla_def.target_value)

    def _determine_response_actions(self, sla_def: SLADefinition, status: SLAStatus) -> List[ResponseAction]:
        """Determine response actions based on SLA and status"""
        actions = [ResponseAction.NOTIFY]
        
        if status == SLAStatus.CRITICAL:
            actions.append(ResponseAction.SCALE_UP)
        elif status == SLAStatus.BREACH:
            actions.extend([ResponseAction.SCALE_UP, ResponseAction.CIRCUIT_BREAKER])
        
        return actions

    def _send_alert_notifications(self, alert: SLAAlert):
        """Send alert notifications through configured channels"""
        policy = self.escalation_policies.get(alert.sla_name)
        if not policy:
            return
        
        # Get channels for current escalation level
        escalation_level = policy.escalation_levels[alert.escalation_level]
        channels = escalation_level.get("channels", [])
        
        for channel_name in channels:
            channel = self.notification_channels.get(channel_name)
            if channel and channel.enabled:
                # Check severity filter
                if not channel.severity_filter or alert.severity in channel.severity_filter:
                    self._send_notification(channel, alert)

    def _send_notification(self, channel: NotificationChannel, alert: SLAAlert):
        """Send notification through specific channel"""
        try:
            if channel.channel_type == AlertChannel.EMAIL:
                self._send_email_notification(channel, alert)
            elif channel.channel_type == AlertChannel.SLACK:
                self._send_slack_notification(channel, alert)
            elif channel.channel_type == AlertChannel.WEBHOOK:
                self._send_webhook_notification(channel, alert)
            elif channel.channel_type == AlertChannel.PAGERDUTY:
                self._send_pagerduty_notification(channel, alert)
            
        except Exception as e:
            logger.error("Failed to send notification",
                        channel=channel.name,
                        alert_id=alert.id,
                        error=str(e))

    def _send_email_notification(self, channel: NotificationChannel, alert: SLAAlert):
        """Send email notification"""
        config = channel.configuration
        
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"SLA Alert: {alert.sla_name} - {alert.severity.value.upper()}"
        
        body = f"""
SLA Alert Notification

SLA: {alert.sla_name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Current Value: {alert.current_value:.2f}
Threshold: {alert.threshold_value:.2f}
Timestamp: {alert.timestamp.isoformat()}

Message: {alert.message}

Alert ID: {alert.id}
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['sender_email'], config['sender_password'])
        server.send_message(msg)
        server.quit()

    def _send_slack_notification(self, channel: NotificationChannel, alert: SLAAlert):
        """Send Slack notification"""
        config = channel.configuration
        
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "channel": config.get("channel", "#alerts"),
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": f"SLA Alert: {alert.sla_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Status", "value": alert.status.value.upper(), "short": True},
                    {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True}
                ],
                "footer": "GrandModel SLA Monitoring",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        response.raise_for_status()

    def _send_webhook_notification(self, channel: NotificationChannel, alert: SLAAlert):
        """Send webhook notification"""
        config = channel.configuration
        
        payload = {
            "alert_id": alert.id,
            "sla_name": alert.sla_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "timestamp": alert.timestamp.isoformat()
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        response.raise_for_status()

    def _send_pagerduty_notification(self, channel: NotificationChannel, alert: SLAAlert):
        """Send PagerDuty notification"""
        config = channel.configuration
        
        payload = {
            "routing_key": config['integration_key'],
            "event_action": "trigger",
            "dedup_key": alert.id,
            "payload": {
                "summary": f"SLA Alert: {alert.sla_name} - {alert.severity.value}",
                "source": "GrandModel SLA Monitoring",
                "severity": alert.severity.value,
                "custom_details": {
                    "sla_name": alert.sla_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "message": alert.message
                }
            }
        }
        
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload
        )
        response.raise_for_status()

    def _send_resolution_notification(self, alert: SLAAlert):
        """Send alert resolution notification"""
        # Similar to alert notification but for resolution
        pass

    def _execute_response_actions(self, alert: SLAAlert):
        """Execute automated response actions"""
        for action in alert.response_actions:
            try:
                if action == ResponseAction.SCALE_UP:
                    self._scale_up_resources(alert)
                elif action == ResponseAction.RESTART_SERVICE:
                    self._restart_service(alert)
                elif action == ResponseAction.CIRCUIT_BREAKER:
                    self._activate_circuit_breaker(alert)
                elif action == ResponseAction.FAILOVER:
                    self._initiate_failover(alert)
                    
            except Exception as e:
                logger.error("Failed to execute response action",
                           action=action.value,
                           alert_id=alert.id,
                           error=str(e))

    def _scale_up_resources(self, alert: SLAAlert):
        """Scale up system resources"""
        logger.info("Scaling up resources", sla_name=alert.sla_name, alert_id=alert.id)
        # Implementation would depend on infrastructure

    def _restart_service(self, alert: SLAAlert):
        """Restart service"""
        logger.info("Restarting service", sla_name=alert.sla_name, alert_id=alert.id)
        # Implementation would depend on service architecture

    def _activate_circuit_breaker(self, alert: SLAAlert):
        """Activate circuit breaker"""
        logger.info("Activating circuit breaker", sla_name=alert.sla_name, alert_id=alert.id)
        # Implementation would depend on circuit breaker system

    def _initiate_failover(self, alert: SLAAlert):
        """Initiate failover"""
        logger.info("Initiating failover", sla_name=alert.sla_name, alert_id=alert.id)
        # Implementation would depend on failover system

    def _process_escalations(self):
        """Process alert escalations"""
        for alert in self.active_alerts.values():
            if alert.resolved or alert.acknowledged:
                continue
            
            policy = self.escalation_policies.get(alert.sla_name)
            if not policy:
                continue
            
            # Check if escalation is needed
            time_since_alert = (datetime.now() - alert.timestamp).total_seconds() / 60
            escalation_interval = policy.escalation_interval_minutes
            
            if time_since_alert >= escalation_interval * (alert.escalation_level + 1):
                # Escalate
                if alert.escalation_level < policy.max_escalation_level - 1:
                    alert.escalation_level += 1
                    self._send_alert_notifications(alert)
                    self._update_alert_in_database(alert)
                    
                    logger.warning("Alert escalated",
                                 alert_id=alert.id,
                                 sla_name=alert.sla_name,
                                 escalation_level=alert.escalation_level)

    def _store_sla_metric(self, metric: SLAMetric):
        """Store SLA metric in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sla_metrics 
            (sla_name, timestamp, value, status, threshold_exceeded, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metric.sla_name,
            metric.timestamp.isoformat(),
            metric.value,
            metric.status.value,
            metric.threshold_exceeded,
            json.dumps(metric.context)
        ))
        
        conn.commit()
        conn.close()

    def _store_alert(self, alert: SLAAlert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO sla_alerts 
            (id, sla_name, severity, status, timestamp, message, current_value, threshold_value,
             escalation_level, acknowledged, resolved, response_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id,
            alert.sla_name,
            alert.severity.value,
            alert.status.value,
            alert.timestamp.isoformat(),
            alert.message,
            alert.current_value,
            alert.threshold_value,
            alert.escalation_level,
            alert.acknowledged,
            alert.resolved,
            json.dumps([a.value for a in alert.response_actions])
        ))
        
        conn.commit()
        conn.close()

    def _update_alert_in_database(self, alert: SLAAlert):
        """Update alert in database"""
        self._store_alert(alert)

    def _update_alert_status(self, alert: SLAAlert, new_status: SLAStatus, current_value: float):
        """Update alert status"""
        alert.status = new_status
        alert.current_value = current_value
        alert.severity = self._determine_alert_severity(new_status)
        
        self._update_alert_in_database(alert)

    def get_sla_status_summary(self) -> Dict[str, Any]:
        """Get SLA status summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_slas": len(self.sla_definitions),
            "active_slas": len([s for s in self.sla_definitions.values() if s.enabled]),
            "active_alerts": len(self.active_alerts),
            "sla_statuses": {},
            "alert_summary": defaultdict(int)
        }
        
        # SLA statuses
        for sla_name, sla_def in self.sla_definitions.items():
            if sla_def.enabled:
                recent_metrics = self._get_recent_metrics(sla_name, 5)
                if recent_metrics:
                    current_status = self._calculate_aggregate_status(recent_metrics)
                    avg_value = statistics.mean([m.value for m in recent_metrics])
                    
                    summary["sla_statuses"][sla_name] = {
                        "status": current_status.value,
                        "current_value": avg_value,
                        "target_value": sla_def.target_value,
                        "unit": sla_def.unit
                    }
        
        # Alert summary
        for alert in self.active_alerts.values():
            summary["alert_summary"][alert.severity.value] += 1
        
        return summary

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            self._update_alert_in_database(alert)
            
            logger.info("Alert acknowledged",
                       alert_id=alert_id,
                       sla_name=alert.sla_name,
                       acknowledged_by=acknowledged_by)

    def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            self._update_alert_in_database(alert)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info("Alert resolved",
                       alert_id=alert_id,
                       sla_name=alert.sla_name,
                       resolved_by=resolved_by)

    def get_sla_history(self, sla_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get SLA history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, value, status, threshold_exceeded, context
            FROM sla_metrics 
            WHERE sla_name = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (sla_name, cutoff_time.isoformat()))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": result[0],
                "value": result[1],
                "status": result[2],
                "threshold_exceeded": result[3],
                "context": json.loads(result[4]) if result[4] else {}
            }
            for result in results
        ]


# Global instance
sla_monitoring_system = SLAMonitoringSystem()