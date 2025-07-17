"""
Alert Manager for Operations System

This module provides comprehensive alerting capabilities including
alert processing, routing, notification channels, and escalation.
"""

import asyncio
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
import uuid
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from collections import defaultdict, deque

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class AlertStatus(Enum):
    """Alert status"""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChannelType(Enum):
    """Alert channel types"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    title: str
    description: str
    severity: str
    priority: AlertPriority
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.OPEN
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalation_level: int = 0
    notification_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "priority": self.priority.value,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalated": self.escalated,
            "escalation_level": self.escalation_level,
            "notification_count": self.notification_count
        }


@dataclass
class AlertChannel:
    """Alert notification channel"""
    channel_id: str
    channel_name: str
    channel_type: ChannelType
    configuration: Dict[str, Any]
    enabled: bool = True
    rate_limit: int = 60  # seconds between notifications
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    failure_count: int = 0
    
    def can_send_notification(self) -> bool:
        """Check if channel can send notification"""
        if not self.enabled:
            return False
        
        if self.last_notification:
            elapsed = (datetime.now() - self.last_notification).total_seconds()
            if elapsed < self.rate_limit:
                return False
        
        return True


@dataclass
class EscalationRule:
    """Alert escalation rule"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    escalation_delay: int = 300  # seconds
    max_escalation_level: int = 3
    escalation_channels: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AlertRule:
    """Alert routing rule"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    channels: List[str]
    priority_override: Optional[AlertPriority] = None
    template: Optional[str] = None
    enabled: bool = True


class AlertManager:
    """Comprehensive alert management system"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.is_running = False
        self.processing_task = None
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=10000)
        
        # Channels and rules
        self.channels: Dict[str, AlertChannel] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.escalation_rules: Dict[str, EscalationRule] = {}
        
        # Processing queue
        self.alert_queue = asyncio.Queue()
        
        # Statistics
        self.total_alerts = 0
        self.alerts_processed = 0
        self.notifications_sent = 0
        self.notifications_failed = 0
        
        # Suppression and grouping
        self.suppressed_alerts: Dict[str, datetime] = {}
        self.alert_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize default channels
        self._initialize_default_channels()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Alert Manager initialized")
    
    def _initialize_default_channels(self):
        """Initialize default alert channels"""
        # Email channel
        self.channels["email_default"] = AlertChannel(
            channel_id="email_default",
            channel_name="Default Email",
            channel_type=ChannelType.EMAIL,
            configuration={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "alerts@grandmodel.com",
                "sender_password": "app_password_here",
                "recipients": ["admin@grandmodel.com"]
            }
        )
        
        # Slack channel
        self.channels["slack_default"] = AlertChannel(
            channel_id="slack_default",
            channel_name="Default Slack",
            channel_type=ChannelType.SLACK,
            configuration={
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                "channel": "#alerts",
                "username": "AlertBot"
            }
        )
        
        # Webhook channel
        self.channels["webhook_default"] = AlertChannel(
            channel_id="webhook_default",
            channel_name="Default Webhook",
            channel_type=ChannelType.WEBHOOK,
            configuration={
                "url": "https://your-webhook-endpoint.com/alerts",
                "method": "POST",
                "headers": {"Content-Type": "application/json"}
            }
        )
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions"""
        self.event_bus.subscribe(EventType.ALERT, self._handle_alert_event)
        self.event_bus.subscribe(EventType.HEALTH_CHECK, self._handle_health_check_event)
    
    async def start_processing(self):
        """Start alert processing"""
        if self.is_running:
            logger.warning("Alert processing already running")
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        logger.info("Alert processing started")
    
    async def stop_processing(self):
        """Stop alert processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert processing stopped")
    
    async def _processing_loop(self):
        """Main alert processing loop"""
        while self.is_running:
            try:
                # Process alerts from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                await self._process_alert(alert)
                
            except asyncio.TimeoutError:
                # Check for escalations
                await self._check_escalations()
                
            except Exception as e:
                logger.error("Error in alert processing loop", error=str(e))
    
    async def _handle_alert_event(self, event: Event):
        """Handle incoming alert events"""
        try:
            # Create alert from event
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                title=event.payload.get("rule_name", "System Alert"),
                description=event.payload.get("message", ""),
                severity=event.payload.get("severity", "info"),
                priority=AlertPriority(event.payload.get("priority", "medium")),
                source=event.payload.get("source", "system"),
                tags=event.payload.get("tags", {}),
                metadata=event.payload
            )
            
            # Add to processing queue
            await self.alert_queue.put(alert)
            
        except Exception as e:
            logger.error("Error handling alert event", error=str(e))
    
    async def _handle_health_check_event(self, event: Event):
        """Handle health check events"""
        try:
            if event.payload.get("status") == "failed":
                # Create alert for failed health check
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    title=f"Health Check Failed: {event.payload.get('check_name')}",
                    description=event.payload.get("error", "Health check failed"),
                    severity="warning",
                    priority=AlertPriority.HIGH,
                    source="health_check",
                    tags={"check_id": event.payload.get("check_id", "unknown")},
                    metadata=event.payload
                )
                
                # Add to processing queue
                await self.alert_queue.put(alert)
                
        except Exception as e:
            logger.error("Error handling health check event", error=str(e))
    
    async def _process_alert(self, alert: Alert):
        """Process a single alert"""
        try:
            self.total_alerts += 1
            
            # Check if alert should be suppressed
            if self._should_suppress_alert(alert):
                logger.info("Alert suppressed", alert_id=alert.alert_id)
                return
            
            # Store alert
            self.alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Find matching routing rules
            matching_rules = self._find_matching_rules(alert)
            
            # Process each matching rule
            for rule in matching_rules:
                await self._process_alert_rule(alert, rule)
            
            # Default processing if no rules matched
            if not matching_rules:
                await self._process_default_alert(alert)
            
            self.alerts_processed += 1
            
            logger.info(
                "Alert processed",
                alert_id=alert.alert_id,
                title=alert.title,
                severity=alert.severity,
                priority=alert.priority.value
            )
            
        except Exception as e:
            logger.error("Error processing alert", alert_id=alert.alert_id, error=str(e))
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        # Check if alert source is suppressed
        if alert.source in self.suppressed_alerts:
            suppression_time = self.suppressed_alerts[alert.source]
            if datetime.now() < suppression_time:
                return True
        
        # Check for duplicate alerts (same title and source)
        duplicate_key = f"{alert.source}:{alert.title}"
        for existing_alert in self.alerts.values():
            if (existing_alert.source == alert.source and 
                existing_alert.title == alert.title and
                existing_alert.status == AlertStatus.OPEN):
                return True
        
        return False
    
    def _find_matching_rules(self, alert: Alert) -> List[AlertRule]:
        """Find alert rules that match the alert"""
        matching_rules = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            if self._rule_matches_alert(rule, alert):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _rule_matches_alert(self, rule: AlertRule, alert: Alert) -> bool:
        """Check if rule matches alert"""
        conditions = rule.conditions
        
        # Check severity
        if "severity" in conditions:
            if alert.severity not in conditions["severity"]:
                return False
        
        # Check source
        if "source" in conditions:
            if alert.source not in conditions["source"]:
                return False
        
        # Check tags
        if "tags" in conditions:
            for tag_key, tag_values in conditions["tags"].items():
                if tag_key not in alert.tags:
                    return False
                if alert.tags[tag_key] not in tag_values:
                    return False
        
        return True
    
    async def _process_alert_rule(self, alert: Alert, rule: AlertRule):
        """Process alert using specific rule"""
        try:
            # Override priority if specified
            if rule.priority_override:
                alert.priority = rule.priority_override
            
            # Send notifications to rule channels
            for channel_id in rule.channels:
                if channel_id in self.channels:
                    await self._send_notification(alert, self.channels[channel_id], rule.template)
            
        except Exception as e:
            logger.error("Error processing alert rule", rule_id=rule.rule_id, error=str(e))
    
    async def _process_default_alert(self, alert: Alert):
        """Process alert with default settings"""
        try:
            # Send to default channels based on priority
            if alert.priority == AlertPriority.CRITICAL:
                channels = ["email_default", "slack_default"]
            elif alert.priority == AlertPriority.HIGH:
                channels = ["slack_default"]
            else:
                channels = ["email_default"]
            
            for channel_id in channels:
                if channel_id in self.channels:
                    await self._send_notification(alert, self.channels[channel_id])
            
        except Exception as e:
            logger.error("Error processing default alert", alert_id=alert.alert_id, error=str(e))
    
    async def _send_notification(self, alert: Alert, channel: AlertChannel, template: Optional[str] = None):
        """Send notification through channel"""
        try:
            # Check if channel can send notification
            if not channel.can_send_notification():
                logger.debug("Channel rate limited", channel_id=channel.channel_id)
                return
            
            # Send notification based on channel type
            if channel.channel_type == ChannelType.EMAIL:
                await self._send_email_notification(alert, channel, template)
            elif channel.channel_type == ChannelType.SLACK:
                await self._send_slack_notification(alert, channel, template)
            elif channel.channel_type == ChannelType.WEBHOOK:
                await self._send_webhook_notification(alert, channel, template)
            else:
                logger.warning("Unsupported channel type", channel_type=channel.channel_type)
                return
            
            # Update channel statistics
            channel.last_notification = datetime.now()
            channel.notification_count += 1
            alert.notification_count += 1
            self.notifications_sent += 1
            
            logger.info(
                "Notification sent",
                alert_id=alert.alert_id,
                channel_id=channel.channel_id,
                channel_type=channel.channel_type.value
            )
            
        except Exception as e:
            channel.failure_count += 1
            self.notifications_failed += 1
            logger.error(
                "Failed to send notification",
                alert_id=alert.alert_id,
                channel_id=channel.channel_id,
                error=str(e)
            )
    
    async def _send_email_notification(self, alert: Alert, channel: AlertChannel, template: Optional[str] = None):
        """Send email notification"""
        config = channel.configuration
        
        # Create email message
        msg = MimeMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
        
        # Create email body
        body = self._format_alert_message(alert, template)
        msg.attach(MimeText(body, 'plain'))
        
        # Send email (mock implementation)
        await asyncio.sleep(0.1)  # Simulate email sending delay
        
        # In production, you would use actual SMTP:
        # server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        # server.starttls()
        # server.login(config['sender_email'], config['sender_password'])
        # text = msg.as_string()
        # server.sendmail(config['sender_email'], config['recipients'], text)
        # server.quit()
    
    async def _send_slack_notification(self, alert: Alert, channel: AlertChannel, template: Optional[str] = None):
        """Send Slack notification"""
        config = channel.configuration
        
        # Create Slack message
        message = {
            "channel": config['channel'],
            "username": config['username'],
            "text": f"[{alert.severity.upper()}] {alert.title}",
            "attachments": [
                {
                    "color": self._get_alert_color(alert.severity),
                    "fields": [
                        {"title": "Description", "value": alert.description, "short": False},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Priority", "value": alert.priority.value, "short": True},
                        {"title": "Created", "value": alert.created_at.isoformat(), "short": True}
                    ]
                }
            ]
        }
        
        # Send Slack message (mock implementation)
        await asyncio.sleep(0.1)  # Simulate Slack API delay
        
        # In production, you would use actual Slack API:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(config['webhook_url'], json=message) as response:
        #         if response.status != 200:
        #             raise Exception(f"Slack API error: {response.status}")
    
    async def _send_webhook_notification(self, alert: Alert, channel: AlertChannel, template: Optional[str] = None):
        """Send webhook notification"""
        config = channel.configuration
        
        # Create webhook payload
        payload = {
            "alert": alert.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "channel": channel.channel_id
        }
        
        # Send webhook (mock implementation)
        await asyncio.sleep(0.1)  # Simulate webhook delay
        
        # In production, you would use actual HTTP request:
        # async with aiohttp.ClientSession() as session:
        #     async with session.request(
        #         method=config['method'],
        #         url=config['url'],
        #         json=payload,
        #         headers=config.get('headers', {})
        #     ) as response:
        #         if response.status >= 400:
        #             raise Exception(f"Webhook error: {response.status}")
    
    def _format_alert_message(self, alert: Alert, template: Optional[str] = None) -> str:
        """Format alert message"""
        if template:
            # Use custom template (simplified implementation)
            return template.format(
                title=alert.title,
                description=alert.description,
                severity=alert.severity,
                priority=alert.priority.value,
                source=alert.source,
                created_at=alert.created_at.isoformat()
            )
        else:
            # Default format
            return f"""
Alert: {alert.title}
Description: {alert.description}
Severity: {alert.severity}
Priority: {alert.priority.value}
Source: {alert.source}
Created: {alert.created_at.isoformat()}

Tags: {json.dumps(alert.tags, indent=2)}
"""
    
    def _get_alert_color(self, severity: str) -> str:
        """Get color for alert severity"""
        color_map = {
            "critical": "danger",
            "error": "danger",
            "warning": "warning",
            "info": "good"
        }
        return color_map.get(severity.lower(), "good")
    
    async def _check_escalations(self):
        """Check for alerts that need escalation"""
        for alert in self.alerts.values():
            if alert.status != AlertStatus.OPEN:
                continue
            
            # Check escalation rules
            for rule in self.escalation_rules.values():
                if not rule.enabled:
                    continue
                
                if self._should_escalate_alert(alert, rule):
                    await self._escalate_alert(alert, rule)
    
    def _should_escalate_alert(self, alert: Alert, rule: EscalationRule) -> bool:
        """Check if alert should be escalated"""
        # Check if alert matches escalation conditions
        if not self._rule_matches_alert_escalation(rule, alert):
            return False
        
        # Check if already escalated to max level
        if alert.escalation_level >= rule.max_escalation_level:
            return False
        
        # Check escalation delay
        time_since_created = (datetime.now() - alert.created_at).total_seconds()
        required_delay = rule.escalation_delay * (alert.escalation_level + 1)
        
        return time_since_created >= required_delay
    
    def _rule_matches_alert_escalation(self, rule: EscalationRule, alert: Alert) -> bool:
        """Check if escalation rule matches alert"""
        conditions = rule.conditions
        
        # Check priority
        if "priority" in conditions:
            if alert.priority.value not in conditions["priority"]:
                return False
        
        # Check if alert is acknowledged
        if "acknowledged" in conditions:
            if alert.acknowledged_by is not None:
                return False
        
        return True
    
    async def _escalate_alert(self, alert: Alert, rule: EscalationRule):
        """Escalate an alert"""
        try:
            alert.escalation_level += 1
            alert.escalated = True
            alert.updated_at = datetime.now()
            
            # Send notifications to escalation channels
            for channel_id in rule.escalation_channels:
                if channel_id in self.channels:
                    await self._send_escalation_notification(alert, self.channels[channel_id], rule)
            
            logger.info(
                "Alert escalated",
                alert_id=alert.alert_id,
                escalation_level=alert.escalation_level,
                rule_id=rule.rule_id
            )
            
        except Exception as e:
            logger.error("Error escalating alert", alert_id=alert.alert_id, error=str(e))
    
    async def _send_escalation_notification(self, alert: Alert, channel: AlertChannel, rule: EscalationRule):
        """Send escalation notification"""
        escalation_template = f"""
[ESCALATION - Level {alert.escalation_level}] {alert.title}

This alert has been escalated due to lack of acknowledgment.

Original Description: {alert.description}
Severity: {alert.severity}
Priority: {alert.priority.value}
Source: {alert.source}
Created: {alert.created_at.isoformat()}
Escalation Rule: {rule.name}

Please acknowledge this alert immediately.
"""
        
        await self._send_notification(alert, channel, escalation_template)
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user_id
        alert.acknowledged_at = datetime.now()
        alert.updated_at = datetime.now()
        
        logger.info("Alert acknowledged", alert_id=alert_id, user_id=user_id)
        return True
    
    async def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()
        
        logger.info("Alert resolved", alert_id=alert_id, user_id=user_id)
        return True
    
    def suppress_alerts(self, source: str, duration_minutes: int):
        """Suppress alerts from a source"""
        suppression_time = datetime.now() + timedelta(minutes=duration_minutes)
        self.suppressed_alerts[source] = suppression_time
        
        logger.info("Alerts suppressed", source=source, duration_minutes=duration_minutes)
    
    def add_alert_channel(self, channel: AlertChannel) -> bool:
        """Add an alert channel"""
        try:
            self.channels[channel.channel_id] = channel
            logger.info("Alert channel added", channel_id=channel.channel_id)
            return True
        except Exception as e:
            logger.error("Failed to add alert channel", channel_id=channel.channel_id, error=str(e))
            return False
    
    def remove_alert_channel(self, channel_id: str) -> bool:
        """Remove an alert channel"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            logger.info("Alert channel removed", channel_id=channel_id)
            return True
        return False
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add an alert rule"""
        try:
            self.alert_rules[rule.rule_id] = rule
            logger.info("Alert rule added", rule_id=rule.rule_id)
            return True
        except Exception as e:
            logger.error("Failed to add alert rule", rule_id=rule.rule_id, error=str(e))
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info("Alert rule removed", rule_id=rule_id)
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return [alert for alert in self.alerts.values() if alert.status == AlertStatus.OPEN]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.created_at >= cutoff_time]
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get alert manager status"""
        return {
            "is_running": self.is_running,
            "total_alerts": self.total_alerts,
            "alerts_processed": self.alerts_processed,
            "notifications_sent": self.notifications_sent,
            "notifications_failed": self.notifications_failed,
            "active_alerts": len(self.get_active_alerts()),
            "channels_count": len(self.channels),
            "alert_rules_count": len(self.alert_rules),
            "escalation_rules_count": len(self.escalation_rules),
            "queue_size": self.alert_queue.qsize(),
            "suppressed_sources": len(self.suppressed_alerts)
        }