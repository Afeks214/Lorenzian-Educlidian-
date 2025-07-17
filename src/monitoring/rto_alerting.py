"""
Comprehensive RTO Alerting System with multiple notification channels and escalation policies.

This module provides:
- Multi-channel alerting (email, Slack, webhook, SMS)
- Escalation policies for critical breaches
- Alert aggregation and deduplication
- Historical alert tracking
- Alert resolution workflows
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import httpx
import threading
from pathlib import Path

from src.monitoring.rto_monitor import RTOMetric, RTOStatus, RTOEvent
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    component: str
    condition: str  # 'breach', 'warning', 'critical'
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10
    escalation_minutes: int = 30
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "component": self.component,
            "condition": self.condition,
            "severity": self.severity.value,
            "channels": [c.value for c in self.channels],
            "cooldown_minutes": self.cooldown_minutes,
            "max_alerts_per_hour": self.max_alerts_per_hour,
            "escalation_minutes": self.escalation_minutes,
            "enabled": self.enabled
        }

@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    component: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "component": self.component,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }

class AlertDatabase:
    """SQLite database for alert storage."""
    
    def __init__(self, db_path: str = "rto_alerts.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    user_id TEXT,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_component_timestamp 
                ON alerts(component, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status 
                ON alerts(status)
            """)
    
    def store_alert(self, alert: Alert):
        """Store alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts (
                    id, rule_name, component, severity, title, message, timestamp,
                    status, acknowledged_by, acknowledged_at, resolved_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.rule_name,
                alert.component,
                alert.severity.value,
                alert.title,
                alert.message,
                alert.timestamp.isoformat(),
                alert.status.value,
                alert.acknowledged_by,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                json.dumps(alert.metadata)
            ))
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM alerts WHERE status = 'active'
                ORDER BY timestamp DESC
            """)
            
            alerts = []
            for row in cursor.fetchall():
                alert = Alert(
                    id=row['id'],
                    rule_name=row['rule_name'],
                    component=row['component'],
                    severity=AlertSeverity(row['severity']),
                    title=row['title'],
                    message=row['message'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    status=AlertStatus(row['status']),
                    acknowledged_by=row['acknowledged_by'],
                    acknowledged_at=datetime.fromisoformat(row['acknowledged_at']) if row['acknowledged_at'] else None,
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                alerts.append(alert)
            
            return alerts
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return Alert(
                id=row['id'],
                rule_name=row['rule_name'],
                component=row['component'],
                severity=AlertSeverity(row['severity']),
                title=row['title'],
                message=row['message'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                status=AlertStatus(row['status']),
                acknowledged_by=row['acknowledged_by'],
                acknowledged_at=datetime.fromisoformat(row['acknowledged_at']) if row['acknowledged_at'] else None,
                resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
    
    def get_recent_alerts(self, component: str, hours: int = 24) -> List[Alert]:
        """Get recent alerts for a component."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM alerts 
                WHERE component = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (component, cutoff.isoformat()))
            
            alerts = []
            for row in cursor.fetchall():
                alert = Alert(
                    id=row['id'],
                    rule_name=row['rule_name'],
                    component=row['component'],
                    severity=AlertSeverity(row['severity']),
                    title=row['title'],
                    message=row['message'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    status=AlertStatus(row['status']),
                    acknowledged_by=row['acknowledged_by'],
                    acknowledged_at=datetime.fromisoformat(row['acknowledged_at']) if row['acknowledged_at'] else None,
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                alerts.append(alert)
            
            return alerts
    
    def log_alert_action(self, alert_id: str, action: str, user_id: str = None, details: Dict[str, Any] = None):
        """Log alert action to history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alert_history (alert_id, action, user_id, timestamp, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert_id,
                action,
                user_id,
                datetime.utcnow().isoformat(),
                json.dumps(details) if details else None
            ))

class EmailNotifier:
    """Email notification handler."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.enabled = smtp_config.get('enabled', False)
    
    async def send_notification(self, alert: Alert, recipients: List[str]):
        """Send email notification."""
        if not self.enabled or not recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"RTO Alert: {alert.title}"
            
            body = f"""
RTO Alert Notification

Component: {alert.component}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Alert ID: {alert.id}
Rule: {alert.rule_name}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.smtp_config['from'], recipients, text)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

class SlackNotifier:
    """Slack notification handler."""
    
    def __init__(self, slack_config: Dict[str, Any]):
        self.slack_config = slack_config
        self.enabled = slack_config.get('enabled', False)
        self.webhook_url = slack_config.get('webhook_url')
    
    async def send_notification(self, alert: Alert, channels: List[str]):
        """Send Slack notification."""
        if not self.enabled or not self.webhook_url:
            return
        
        try:
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "danger")
            
            payload = {
                "text": f"RTO Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {"title": "Component", "value": alert.component, "short": True},
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Status", "value": alert.status.value.upper(), "short": True},
                            {"title": "Timestamp", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Alert ID", "value": alert.id, "short": True},
                            {"title": "Rule", "value": alert.rule_name, "short": True}
                        ]
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

class WebhookNotifier:
    """Webhook notification handler."""
    
    def __init__(self, webhook_config: Dict[str, Any]):
        self.webhook_config = webhook_config
        self.enabled = webhook_config.get('enabled', False)
    
    async def send_notification(self, alert: Alert, endpoints: List[str]):
        """Send webhook notification."""
        if not self.enabled or not endpoints:
            return
        
        try:
            payload = {
                "alert_type": "rto_alert",
                "alert": alert.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with httpx.AsyncClient() as client:
                for endpoint in endpoints:
                    try:
                        response = await client.post(endpoint, json=payload, timeout=10.0)
                        response.raise_for_status()
                        logger.info(f"Webhook notification sent to {endpoint} for alert {alert.id}")
                    except Exception as e:
                        logger.error(f"Failed to send webhook notification to {endpoint}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notifications: {e}")

class ConsoleNotifier:
    """Console notification handler."""
    
    async def send_notification(self, alert: Alert):
        """Send console notification."""
        severity_symbols = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "💥"
        }
        
        symbol = severity_symbols.get(alert.severity, "🔔")
        
        print(f"\n{symbol} RTO ALERT {symbol}")
        print(f"Component: {alert.component}")
        print(f"Severity: {alert.severity.value.upper()}")
        print(f"Title: {alert.title}")
        print(f"Message: {alert.message}")
        print(f"Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Alert ID: {alert.id}")
        print("-" * 50)

class RTOAlertingSystem:
    """Comprehensive RTO alerting system."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 event_bus: Optional[EventBus] = None):
        self.config = config
        self.event_bus = event_bus or EventBus()
        
        # Initialize components
        self.database = AlertDatabase()
        self._init_notifiers()
        self._init_rules()
        
        # State management
        self._alert_counters: Dict[str, int] = {}
        self._last_alerts: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        # Subscribe to events
        self.event_bus.subscribe("rto_breach_alert", self._handle_rto_breach)
        self.event_bus.subscribe("rto_recovery_alert", self._handle_rto_recovery)
        
        # Start background tasks
        self._escalation_task = asyncio.create_task(self._escalation_worker())
    
    def _init_notifiers(self):
        """Initialize notification handlers."""
        self.notifiers = {}
        
        # Email notifier
        if self.config.get('email', {}).get('enabled'):
            self.notifiers[AlertChannel.EMAIL] = EmailNotifier(self.config['email'])
        
        # Slack notifier
        if self.config.get('slack', {}).get('enabled'):
            self.notifiers[AlertChannel.SLACK] = SlackNotifier(self.config['slack'])
        
        # Webhook notifier
        if self.config.get('webhook', {}).get('enabled'):
            self.notifiers[AlertChannel.WEBHOOK] = WebhookNotifier(self.config['webhook'])
        
        # Console notifier (always enabled)
        self.notifiers[AlertChannel.CONSOLE] = ConsoleNotifier()
    
    def _init_rules(self):
        """Initialize alert rules."""
        self.rules = {}
        
        # Default rules
        default_rules = [
            AlertRule(
                name="database_breach",
                component="database",
                condition="breach",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.CONSOLE],
                cooldown_minutes=5,
                escalation_minutes=15
            ),
            AlertRule(
                name="database_critical",
                component="database",
                condition="critical",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK, AlertChannel.CONSOLE],
                cooldown_minutes=1,
                escalation_minutes=5
            ),
            AlertRule(
                name="trading_engine_breach",
                component="trading_engine",
                condition="breach",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.CONSOLE],
                cooldown_minutes=2,
                escalation_minutes=10
            ),
            AlertRule(
                name="trading_engine_critical",
                component="trading_engine",
                condition="critical",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK, AlertChannel.CONSOLE],
                cooldown_minutes=1,
                escalation_minutes=5
            )
        ]
        
        # Load custom rules from config
        custom_rules = self.config.get('rules', [])
        for rule_config in custom_rules:
            rule = AlertRule(
                name=rule_config['name'],
                component=rule_config['component'],
                condition=rule_config['condition'],
                severity=AlertSeverity(rule_config['severity']),
                channels=[AlertChannel(c) for c in rule_config['channels']],
                cooldown_minutes=rule_config.get('cooldown_minutes', 5),
                max_alerts_per_hour=rule_config.get('max_alerts_per_hour', 10),
                escalation_minutes=rule_config.get('escalation_minutes', 30),
                enabled=rule_config.get('enabled', True)
            )
            default_rules.append(rule)
        
        for rule in default_rules:
            self.rules[rule.name] = rule
    
    async def _handle_rto_breach(self, alert_data: Dict[str, Any]):
        """Handle RTO breach event."""
        component = alert_data.get('component')
        breach_percentage = alert_data.get('breach_percentage', 0)
        
        # Determine severity and rule
        if breach_percentage > 200:  # >200% breach
            rule_name = f"{component}_critical"
        else:
            rule_name = f"{component}_breach"
        
        rule = self.rules.get(rule_name)
        if not rule or not rule.enabled:
            return
        
        # Check rate limiting
        if not self._should_alert(rule):
            return
        
        # Create alert
        alert = Alert(
            id=f"rto_{component}_{int(datetime.utcnow().timestamp())}",
            rule_name=rule_name,
            component=component,
            severity=rule.severity,
            title=f"RTO Breach: {component}",
            message=f"Component {component} has breached RTO target by {breach_percentage:.1f}%",
            timestamp=datetime.utcnow(),
            metadata=alert_data
        )
        
        # Store alert
        self.database.store_alert(alert)
        
        # Send notifications
        await self._send_notifications(alert, rule)
        
        # Update counters
        with self._lock:
            self._alert_counters[rule_name] = self._alert_counters.get(rule_name, 0) + 1
            self._last_alerts[rule_name] = datetime.utcnow()
    
    async def _handle_rto_recovery(self, alert_data: Dict[str, Any]):
        """Handle RTO recovery event."""
        component = alert_data.get('component')
        
        # Resolve active alerts for this component
        active_alerts = self.database.get_active_alerts()
        for alert in active_alerts:
            if alert.component == component:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                self.database.store_alert(alert)
                self.database.log_alert_action(alert.id, "auto_resolved", details=alert_data)
                
                logger.info(f"Auto-resolved alert {alert.id} due to recovery")
    
    def _should_alert(self, rule: AlertRule) -> bool:
        """Check if alert should be sent based on rate limiting."""
        with self._lock:
            now = datetime.utcnow()
            
            # Check cooldown
            last_alert = self._last_alerts.get(rule.name)
            if last_alert and (now - last_alert).total_seconds() < rule.cooldown_minutes * 60:
                return False
            
            # Check hourly rate limit
            hour_ago = now - timedelta(hours=1)
            recent_alerts = self.database.get_recent_alerts(rule.component, 1)
            if len(recent_alerts) >= rule.max_alerts_per_hour:
                return False
            
            return True
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert."""
        notification_tasks = []
        
        for channel in rule.channels:
            notifier = self.notifiers.get(channel)
            if not notifier:
                continue
            
            if channel == AlertChannel.EMAIL:
                recipients = self.config.get('email', {}).get('recipients', [])
                if recipients:
                    notification_tasks.append(notifier.send_notification(alert, recipients))
            
            elif channel == AlertChannel.SLACK:
                channels = self.config.get('slack', {}).get('channels', [])
                if channels:
                    notification_tasks.append(notifier.send_notification(alert, channels))
            
            elif channel == AlertChannel.WEBHOOK:
                endpoints = self.config.get('webhook', {}).get('endpoints', [])
                if endpoints:
                    notification_tasks.append(notifier.send_notification(alert, endpoints))
            
            elif channel == AlertChannel.CONSOLE:
                notification_tasks.append(notifier.send_notification(alert))
        
        # Send all notifications concurrently
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)
    
    async def _escalation_worker(self):
        """Background worker for alert escalation."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                active_alerts = self.database.get_active_alerts()
                now = datetime.utcnow()
                
                for alert in active_alerts:
                    rule = self.rules.get(alert.rule_name)
                    if not rule:
                        continue
                    
                    # Check if alert should be escalated
                    alert_age = (now - alert.timestamp).total_seconds() / 60
                    if (alert_age >= rule.escalation_minutes and 
                        alert.status == AlertStatus.ACTIVE):
                        
                        # Escalate alert
                        await self._escalate_alert(alert, rule)
                        
            except Exception as e:
                logger.error(f"Error in escalation worker: {e}")
    
    async def _escalate_alert(self, alert: Alert, rule: AlertRule):
        """Escalate alert to higher severity."""
        # Increase severity
        if alert.severity == AlertSeverity.LOW:
            alert.severity = AlertSeverity.MEDIUM
        elif alert.severity == AlertSeverity.MEDIUM:
            alert.severity = AlertSeverity.HIGH
        elif alert.severity == AlertSeverity.HIGH:
            alert.severity = AlertSeverity.CRITICAL
        
        # Update alert
        alert.metadata['escalated'] = True
        alert.metadata['escalation_time'] = datetime.utcnow().isoformat()
        self.database.store_alert(alert)
        self.database.log_alert_action(alert.id, "escalated")
        
        # Send escalation notifications
        await self._send_notifications(alert, rule)
        
        logger.warning(f"Escalated alert {alert.id} to {alert.severity.value}")
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        alert = self.database.get_alert_by_id(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user_id
        alert.acknowledged_at = datetime.utcnow()
        
        self.database.store_alert(alert)
        self.database.log_alert_action(alert_id, "acknowledged", user_id)
        
        logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        return True
    
    async def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Resolve an alert."""
        alert = self.database.get_alert_by_id(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        self.database.store_alert(alert)
        self.database.log_alert_action(alert_id, "resolved", user_id)
        
        logger.info(f"Alert {alert_id} resolved by {user_id}")
        return True
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        summary = {
            "total_alerts": 0,
            "active_alerts": 0,
            "by_severity": {s.value: 0 for s in AlertSeverity},
            "by_component": {},
            "by_status": {s.value: 0 for s in AlertStatus}
        }
        
        # Get all alerts in timeframe
        with sqlite3.connect(self.database.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM alerts WHERE timestamp >= ?
            """, (cutoff.isoformat(),))
            
            for row in cursor.fetchall():
                summary["total_alerts"] += 1
                
                if row['status'] == 'active':
                    summary["active_alerts"] += 1
                
                summary["by_severity"][row['severity']] += 1
                summary["by_status"][row['status']] += 1
                
                component = row['component']
                if component not in summary["by_component"]:
                    summary["by_component"][component] = 0
                summary["by_component"][component] += 1
        
        return summary

# Default configuration
DEFAULT_ALERTING_CONFIG = {
    "email": {
        "enabled": False,
        "host": "smtp.gmail.com",
        "port": 587,
        "use_tls": True,
        "username": "",
        "password": "",
        "from": "rto-alerts@trading-system.com",
        "recipients": ["ops@trading-system.com"]
    },
    "slack": {
        "enabled": False,
        "webhook_url": "",
        "channels": ["#alerts"]
    },
    "webhook": {
        "enabled": False,
        "endpoints": ["https://your-webhook-endpoint.com/alerts"]
    },
    "rules": []
}

# Global alerting system instance
alerting_system = None

def initialize_alerting_system(config: Dict[str, Any] = None) -> RTOAlertingSystem:
    """Initialize global alerting system."""
    global alerting_system
    if config is None:
        config = DEFAULT_ALERTING_CONFIG
    alerting_system = RTOAlertingSystem(config)
    return alerting_system

def get_alerting_system() -> Optional[RTOAlertingSystem]:
    """Get global alerting system instance."""
    return alerting_system