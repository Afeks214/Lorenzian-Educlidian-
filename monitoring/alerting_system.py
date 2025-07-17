#!/usr/bin/env python3
"""
Comprehensive Alerting System for GrandModel MARL Trading System
Real-time alerting with multiple notification channels and escalation
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import redis
from prometheus_client import Counter, Histogram, Gauge
import yaml

# Alert metrics
ALERTS_SENT = Counter('alerts_sent_total', 'Total alerts sent', ['alert_type', 'severity', 'channel'])
ALERT_RESPONSE_TIME = Histogram('alert_response_time_seconds', 'Alert response time', ['alert_type'])
ALERT_ESCALATION_COUNT = Counter('alert_escalations_total', 'Alert escalations', ['alert_type', 'escalation_level'])
ACTIVE_ALERTS = Gauge('active_alerts', 'Number of active alerts', ['alert_type', 'severity'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts."""
    SUPERPOSITION_ANOMALY = "superposition_anomaly"
    SYSTEM_PERFORMANCE = "system_performance"
    MARL_COORDINATION = "marl_coordination"
    RISK_MANAGEMENT = "risk_management"
    TRADING_PERFORMANCE = "trading_performance"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"

class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    status: AlertStatus
    metadata: Dict[str, Any]
    tags: List[str]
    escalation_level: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'status': self.status.value,
            'metadata': self.metadata,
            'tags': self.tags,
            'escalation_level': self.escalation_level,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_by': self.resolved_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    alert_type: AlertType
    condition: str
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    escalation_time: int  # seconds
    cooldown_time: int  # seconds
    enabled: bool = True
    description: str = ""
    
class EmailNotifier:
    """Email notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.use_tls = config.get('use_tls', True)
        
    async def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        color_map = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }
        
        color = color_map.get(alert.severity, '#6c757d')
        
        html = f"""
        <html>
        <body>
            <h2 style="color: {color};">{alert.title}</h2>
            <p><strong>Severity:</strong> <span style="color: {color};">{alert.severity.value.upper()}</span></p>
            <p><strong>Type:</strong> {alert.alert_type.value}</p>
            <p><strong>Source:</strong> {alert.source}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p><strong>Description:</strong></p>
            <p>{alert.description}</p>
            
            <h3>Details:</h3>
            <ul>
        """
        
        for key, value in alert.metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        
        html += """
            </ul>
            
            <p><strong>Tags:</strong> {}</p>
            <p><em>This is an automated alert from the GrandModel monitoring system.</em></p>
        </body>
        </html>
        """.format(', '.join(alert.tags))
        
        return html

class SlackNotifier:
    """Slack notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'GrandModel Alerts')
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via Slack."""
        try:
            # Create Slack message
            message = self._create_slack_message(alert)
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=message, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _create_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack message payload."""
        color_map = {
            AlertSeverity.LOW: 'good',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.HIGH: 'danger',
            AlertSeverity.CRITICAL: 'danger'
        }
        
        color = color_map.get(alert.severity, 'warning')
        
        fields = [
            {
                'title': 'Severity',
                'value': alert.severity.value.upper(),
                'short': True
            },
            {
                'title': 'Type',
                'value': alert.alert_type.value,
                'short': True
            },
            {
                'title': 'Source',
                'value': alert.source,
                'short': True
            },
            {
                'title': 'Time',
                'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'short': True
            }
        ]
        
        # Add metadata fields
        for key, value in list(alert.metadata.items())[:5]:  # Limit to first 5 items
            fields.append({
                'title': key,
                'value': str(value),
                'short': True
            })
        
        attachment = {
            'color': color,
            'title': alert.title,
            'text': alert.description,
            'fields': fields,
            'footer': 'GrandModel Monitoring',
            'ts': int(alert.timestamp.timestamp())
        }
        
        return {
            'channel': self.channel,
            'username': self.username,
            'attachments': [attachment]
        }

class PagerDutyNotifier:
    """PagerDuty notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integration_key = config.get('integration_key')
        self.api_url = config.get('api_url', 'https://events.pagerduty.com/v2/enqueue')
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via PagerDuty."""
        try:
            # Create PagerDuty event
            event = self._create_pagerduty_event(alert)
            
            # Send to PagerDuty
            response = requests.post(self.api_url, json=event, timeout=10)
            
            if response.status_code == 202:
                logger.info(f"PagerDuty alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"PagerDuty notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    def _create_pagerduty_event(self, alert: Alert) -> Dict[str, Any]:
        """Create PagerDuty event payload."""
        severity_map = {
            AlertSeverity.LOW: 'info',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.HIGH: 'error',
            AlertSeverity.CRITICAL: 'critical'
        }
        
        severity = severity_map.get(alert.severity, 'error')
        
        return {
            'routing_key': self.integration_key,
            'event_action': 'trigger',
            'dedup_key': alert.alert_id,
            'payload': {
                'summary': alert.title,
                'severity': severity,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'component': alert.alert_type.value,
                'group': 'GrandModel',
                'class': alert.alert_type.value,
                'custom_details': alert.metadata
            }
        }

class WebhookNotifier:
    """Webhook notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config.get('url')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 10)
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            # Create webhook payload
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'grandmodel-monitoring'
            }
            
            # Send webhook
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code < 400:
                logger.info(f"Webhook alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(**config.get('redis', {}))
        self.active_alerts = {}
        self.alert_rules = {}
        self.notifiers = {}
        
        # Initialize notifiers
        self._initialize_notifiers()
        
        # Load alert rules
        self._load_alert_rules()
        
        # Alert management
        self.alert_history = []
        self.escalation_timers = {}
        
    def _initialize_notifiers(self):
        """Initialize notification handlers."""
        notifier_config = self.config.get('notifiers', {})
        
        if 'email' in notifier_config:
            self.notifiers[AlertChannel.EMAIL] = EmailNotifier(notifier_config['email'])
        
        if 'slack' in notifier_config:
            self.notifiers[AlertChannel.SLACK] = SlackNotifier(notifier_config['slack'])
        
        if 'pagerduty' in notifier_config:
            self.notifiers[AlertChannel.PAGERDUTY] = PagerDutyNotifier(notifier_config['pagerduty'])
        
        if 'webhook' in notifier_config:
            self.notifiers[AlertChannel.WEBHOOK] = WebhookNotifier(notifier_config['webhook'])
    
    def _load_alert_rules(self):
        """Load alert rules from configuration."""
        rules_config = self.config.get('alert_rules', [])
        
        for rule_config in rules_config:
            rule = AlertRule(
                rule_id=rule_config['rule_id'],
                alert_type=AlertType(rule_config['alert_type']),
                condition=rule_config['condition'],
                threshold=rule_config['threshold'],
                severity=AlertSeverity(rule_config['severity']),
                channels=[AlertChannel(ch) for ch in rule_config['channels']],
                escalation_time=rule_config.get('escalation_time', 300),
                cooldown_time=rule_config.get('cooldown_time', 900),
                enabled=rule_config.get('enabled', True),
                description=rule_config.get('description', '')
            )
            
            self.alert_rules[rule.rule_id] = rule
    
    async def create_alert(self, alert: Alert) -> str:
        """Create and process a new alert."""
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Update metrics
        ACTIVE_ALERTS.labels(
            alert_type=alert.alert_type.value,
            severity=alert.severity.value
        ).inc()
        
        # Store in Redis
        await self._store_alert_in_redis(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Start escalation timer
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self._start_escalation_timer(alert)
        
        logger.info(f"Alert created: {alert.alert_id}")
        return alert.alert_id
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        
        # Cancel escalation timer
        if alert_id in self.escalation_timers:
            self.escalation_timers[alert_id].cancel()
            del self.escalation_timers[alert_id]
        
        # Update in Redis
        await self._store_alert_in_redis(alert)
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.utcnow()
        
        # Update metrics
        ACTIVE_ALERTS.labels(
            alert_type=alert.alert_type.value,
            severity=alert.severity.value
        ).dec()
        
        # Cancel escalation timer
        if alert_id in self.escalation_timers:
            self.escalation_timers[alert_id].cancel()
            del self.escalation_timers[alert_id]
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Update in Redis
        await self._store_alert_in_redis(alert)
        
        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        # Find matching rules
        matching_rules = self._find_matching_rules(alert)
        
        for rule in matching_rules:
            for channel in rule.channels:
                if channel in self.notifiers:
                    try:
                        start_time = time.time()
                        
                        # Send notification
                        success = await self._send_notification(alert, channel)
                        
                        # Update metrics
                        ALERTS_SENT.labels(
                            alert_type=alert.alert_type.value,
                            severity=alert.severity.value,
                            channel=channel.value
                        ).inc()
                        
                        ALERT_RESPONSE_TIME.labels(
                            alert_type=alert.alert_type.value
                        ).observe(time.time() - start_time)
                        
                        if success:
                            logger.info(f"Notification sent via {channel.value} for {alert.alert_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    async def _send_notification(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send notification via specific channel."""
        notifier = self.notifiers.get(channel)
        if not notifier:
            return False
        
        if channel == AlertChannel.EMAIL:
            recipients = self.config.get('email_recipients', [])
            return await notifier.send_alert(alert, recipients)
        else:
            return await notifier.send_alert(alert)
    
    def _find_matching_rules(self, alert: Alert) -> List[AlertRule]:
        """Find rules that match the alert."""
        matching_rules = []
        
        for rule in self.alert_rules.values():
            if rule.enabled and rule.alert_type == alert.alert_type:
                # Check if alert meets rule conditions
                if self._evaluate_rule_condition(alert, rule):
                    matching_rules.append(rule)
        
        return matching_rules
    
    def _evaluate_rule_condition(self, alert: Alert, rule: AlertRule) -> bool:
        """Evaluate if alert meets rule conditions."""
        try:
            # This is a simplified condition evaluation
            # In a real system, this would parse and evaluate complex conditions
            
            # For severity-based rules
            if rule.condition == 'severity':
                severity_levels = {
                    AlertSeverity.LOW: 1,
                    AlertSeverity.MEDIUM: 2,
                    AlertSeverity.HIGH: 3,
                    AlertSeverity.CRITICAL: 4
                }
                return severity_levels.get(alert.severity, 0) >= rule.threshold
            
            # For metric-based rules
            elif rule.condition in alert.metadata:
                return float(alert.metadata[rule.condition]) >= rule.threshold
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    def _start_escalation_timer(self, alert: Alert):
        """Start escalation timer for an alert."""
        def escalate():
            asyncio.create_task(self._escalate_alert(alert))
        
        rule = self._find_matching_rules(alert)[0] if self._find_matching_rules(alert) else None
        escalation_time = rule.escalation_time if rule else 300
        
        timer = threading.Timer(escalation_time, escalate)
        timer.start()
        self.escalation_timers[alert.alert_id] = timer
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert."""
        if alert.status == AlertStatus.ACKNOWLEDGED:
            return
        
        alert.escalation_level += 1
        
        # Update metrics
        ALERT_ESCALATION_COUNT.labels(
            alert_type=alert.alert_type.value,
            escalation_level=alert.escalation_level
        ).inc()
        
        # Create escalated alert
        escalated_alert = Alert(
            alert_id=f"{alert.alert_id}_escalated_{alert.escalation_level}",
            alert_type=alert.alert_type,
            severity=AlertSeverity.CRITICAL,  # Escalate to critical
            title=f"[ESCALATED] {alert.title}",
            description=f"Alert escalated to level {alert.escalation_level}. Original: {alert.description}",
            timestamp=datetime.utcnow(),
            source=alert.source,
            status=AlertStatus.ACTIVE,
            metadata=alert.metadata,
            tags=alert.tags + ['escalated'],
            escalation_level=alert.escalation_level
        )
        
        # Send escalated notifications
        await self._send_notifications(escalated_alert)
        
        logger.warning(f"Alert escalated: {alert.alert_id} to level {alert.escalation_level}")
        
        # Schedule next escalation
        if alert.escalation_level < 3:  # Max 3 escalation levels
            self._start_escalation_timer(alert)
    
    async def _store_alert_in_redis(self, alert: Alert):
        """Store alert in Redis."""
        try:
            alert_key = f"alert:{alert.alert_id}"
            alert_data = json.dumps(alert.to_dict())
            await self.redis_client.setex(alert_key, 86400, alert_data)  # 24 hours TTL
            
        except Exception as e:
            logger.error(f"Failed to store alert in Redis: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        stats = {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': {},
            'alerts_by_type': {},
            'escalation_count': sum(1 for a in self.alert_history if a.escalation_level > 0)
        }
        
        # Count by severity
        for severity in AlertSeverity:
            stats['alerts_by_severity'][severity.value] = sum(
                1 for a in self.active_alerts.values() if a.severity == severity
            )
        
        # Count by type
        for alert_type in AlertType:
            stats['alerts_by_type'][alert_type.value] = sum(
                1 for a in self.active_alerts.values() if a.alert_type == alert_type
            )
        
        return stats

# Factory function
def create_alert_manager(config: Dict[str, Any]) -> AlertManager:
    """Create alert manager instance."""
    return AlertManager(config)

# Example configuration
EXAMPLE_CONFIG = {
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    },
    'notifiers': {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'alerts@grandmodel.com',
            'password': 'password',
            'from_email': 'alerts@grandmodel.com',
            'use_tls': True
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            'channel': '#alerts',
            'username': 'GrandModel Alerts'
        },
        'pagerduty': {
            'integration_key': 'YOUR_INTEGRATION_KEY'
        },
        'webhook': {
            'url': 'https://your-webhook-endpoint.com/alerts',
            'headers': {
                'Authorization': 'Bearer YOUR_TOKEN',
                'Content-Type': 'application/json'
            }
        }
    },
    'email_recipients': ['ops@grandmodel.com', 'dev@grandmodel.com'],
    'alert_rules': [
        {
            'rule_id': 'critical_superposition_anomaly',
            'alert_type': 'superposition_anomaly',
            'condition': 'severity',
            'threshold': 3,
            'severity': 'critical',
            'channels': ['email', 'slack', 'pagerduty'],
            'escalation_time': 300,
            'cooldown_time': 900,
            'description': 'Critical superposition anomaly detected'
        },
        {
            'rule_id': 'high_system_load',
            'alert_type': 'system_performance',
            'condition': 'cpu_usage',
            'threshold': 80,
            'severity': 'high',
            'channels': ['email', 'slack'],
            'escalation_time': 600,
            'cooldown_time': 1200,
            'description': 'High system load detected'
        }
    ]
}

# Example usage
async def main():
    """Example usage of alerting system."""
    config = EXAMPLE_CONFIG
    alert_manager = create_alert_manager(config)
    
    # Create sample alert
    alert = Alert(
        alert_id='test_alert_001',
        alert_type=AlertType.SUPERPOSITION_ANOMALY,
        severity=AlertSeverity.HIGH,
        title='Superposition Coherence Drop',
        description='Sudden drop in quantum coherence detected in strategic agent',
        timestamp=datetime.utcnow(),
        source='superposition_monitor',
        status=AlertStatus.ACTIVE,
        metadata={'coherence': 0.3, 'agent_id': 'strategic_agent'},
        tags=['quantum', 'coherence', 'strategic']
    )
    
    # Create alert
    alert_id = await alert_manager.create_alert(alert)
    print(f"Created alert: {alert_id}")
    
    # Get statistics
    stats = alert_manager.get_alert_statistics()
    print(f"Alert statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(main())