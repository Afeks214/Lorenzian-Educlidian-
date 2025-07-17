"""
Performance Alerting System - Agent 3

This module provides comprehensive performance alerting with multiple channels,
severity levels, and automated escalation procedures.
"""

import os
import json
import smtplib
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import structlog
from pathlib import Path

from .performance_regression_system import RegressionResult, performance_detector

logger = structlog.get_logger()

@dataclass
class AlertChannel:
    """Alert delivery channel configuration"""
    name: str
    enabled: bool
    config: Dict[str, Any] = field(default_factory=dict)
    min_severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # "regression", "threshold", "trend"
    parameters: Dict[str, Any] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)
    cooldown_minutes: int = 60
    enabled: bool = True

@dataclass
class Alert:
    """Performance alert"""
    id: str
    test_name: str
    severity: str
    message: str
    timestamp: datetime
    channels_notified: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceAlertingSystem:
    """
    Comprehensive performance alerting system with multiple channels
    and automated escalation procedures
    """
    
    def __init__(self):
        self.channels = {}
        self.rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.cooldown_tracker = {}
        
        # Initialize default channels
        self._setup_default_channels()
        self._setup_default_rules()
        
        logger.info("PerformanceAlertingSystem initialized")
    
    def _setup_default_channels(self):
        """Setup default alert channels"""
        # Console/Log channel
        self.channels['console'] = AlertChannel(
            name='console',
            enabled=True,
            min_severity='LOW'
        )
        
        # Email channel
        self.channels['email'] = AlertChannel(
            name='email',
            enabled=bool(os.getenv('SMTP_HOST')),
            config={
                'smtp_host': os.getenv('SMTP_HOST', 'localhost'),
                'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                'username': os.getenv('SMTP_USERNAME'),
                'password': os.getenv('SMTP_PASSWORD'),
                'recipients': os.getenv('ALERT_RECIPIENTS', '').split(','),
                'sender': os.getenv('ALERT_SENDER', 'performance-alerts@grandmodel.com')
            },
            min_severity='MEDIUM'
        )
        
        # Slack channel
        self.channels['slack'] = AlertChannel(
            name='slack',
            enabled=bool(os.getenv('SLACK_WEBHOOK_URL')),
            config={
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
                'channel': os.getenv('SLACK_CHANNEL', '#performance-alerts')
            },
            min_severity='HIGH'
        )
        
        # Prometheus/Grafana alerts
        self.channels['prometheus'] = AlertChannel(
            name='prometheus',
            enabled=bool(os.getenv('PROMETHEUS_PUSHGATEWAY_URL')),
            config={
                'pushgateway_url': os.getenv('PROMETHEUS_PUSHGATEWAY_URL'),
                'job_name': 'performance-regression-alerts'
            },
            min_severity='MEDIUM'
        )
        
        # PagerDuty for critical alerts
        self.channels['pagerduty'] = AlertChannel(
            name='pagerduty',
            enabled=bool(os.getenv('PAGERDUTY_INTEGRATION_KEY')),
            config={
                'integration_key': os.getenv('PAGERDUTY_INTEGRATION_KEY'),
                'service_name': 'GrandModel Performance'
            },
            min_severity='CRITICAL'
        )
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        # Regression detection rule
        self.rules['performance_regression'] = AlertRule(
            name='performance_regression',
            condition='regression',
            parameters={
                'min_severity': 'MEDIUM',
                'max_regression_percent': 25.0
            },
            channels=['console', 'email', 'slack'],
            cooldown_minutes=30
        )
        
        # Critical performance threshold rule
        self.rules['critical_threshold'] = AlertRule(
            name='critical_threshold',
            condition='threshold',
            parameters={
                'threshold_ms': 1000.0,  # 1 second
                'consecutive_failures': 3
            },
            channels=['console', 'email', 'slack', 'pagerduty'],
            cooldown_minutes=15
        )
        
        # Performance trend degradation rule
        self.rules['trend_degradation'] = AlertRule(
            name='trend_degradation',
            condition='trend',
            parameters={
                'trend_direction': 'DEGRADING',
                'min_data_points': 10,
                'trend_threshold': 0.1  # 10% degradation trend
            },
            channels=['console', 'email'],
            cooldown_minutes=120
        )
    
    def add_channel(self, channel: AlertChannel):
        """Add a new alert channel"""
        self.channels[channel.name] = channel
        logger.info("Alert channel added", channel=channel.name)
    
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.rules[rule.name] = rule
        logger.info("Alert rule added", rule=rule.name)
    
    async def process_regression_result(self, result: RegressionResult):
        """Process a regression result and generate alerts if needed"""
        if not result.regression_detected:
            return
        
        # Check if we should generate an alert
        alert_generated = False
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            if rule.condition == 'regression':
                if self._should_trigger_regression_alert(result, rule):
                    alert = self._create_regression_alert(result, rule)
                    await self._send_alert(alert, rule.channels)
                    alert_generated = True
        
        if alert_generated:
            logger.info("Regression alert generated", test_name=result.test_name)
    
    def _should_trigger_regression_alert(self, result: RegressionResult, rule: AlertRule) -> bool:
        """Check if regression alert should be triggered"""
        # Check severity threshold
        severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        min_severity = rule.parameters.get('min_severity', 'MEDIUM')
        
        if severity_levels[result.regression_severity] < severity_levels[min_severity]:
            return False
        
        # Check cooldown
        cooldown_key = f"{result.test_name}_{rule.name}"
        if cooldown_key in self.cooldown_tracker:
            last_alert = self.cooldown_tracker[cooldown_key]
            if datetime.now() - last_alert < timedelta(minutes=rule.cooldown_minutes):
                return False
        
        return True
    
    def _create_regression_alert(self, result: RegressionResult, rule: AlertRule) -> Alert:
        """Create regression alert"""
        alert_id = f"regression_{result.test_name}_{int(result.timestamp.timestamp())}"
        
        # Calculate regression percentage
        regression_percent = ((result.current_performance - result.baseline_performance) / 
                            result.baseline_performance) * 100
        
        message = (
            f"Performance regression detected in {result.test_name}\n"
            f"Severity: {result.regression_severity}\n"
            f"Current Performance: {result.current_performance:.4f}s\n"
            f"Baseline Performance: {result.baseline_performance:.4f}s\n"
            f"Regression: {regression_percent:.1f}%\n"
            f"Statistical Significance: {result.statistical_significance:.4f}\n"
            f"Trend: {result.trend_direction}\n"
            f"Recommendation: {result.recommendation}"
        )
        
        alert = Alert(
            id=alert_id,
            test_name=result.test_name,
            severity=result.regression_severity,
            message=message,
            timestamp=result.timestamp,
            metadata={
                'regression_percent': regression_percent,
                'statistical_significance': result.statistical_significance,
                'trend_direction': result.trend_direction,
                'baseline_performance': result.baseline_performance,
                'current_performance': result.current_performance
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update cooldown tracker
        cooldown_key = f"{result.test_name}_{rule.name}"
        self.cooldown_tracker[cooldown_key] = datetime.now()
        
        return alert
    
    async def _send_alert(self, alert: Alert, channels: List[str]):
        """Send alert to specified channels"""
        for channel_name in channels:
            if channel_name not in self.channels:
                continue
            
            channel = self.channels[channel_name]
            if not channel.enabled:
                continue
            
            # Check severity threshold
            severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
            if severity_levels[alert.severity] < severity_levels[channel.min_severity]:
                continue
            
            try:
                await self._send_to_channel(alert, channel)
                alert.channels_notified.append(channel_name)
                logger.info("Alert sent", 
                           alert_id=alert.id,
                           channel=channel_name,
                           severity=alert.severity)
            except Exception as e:
                logger.error("Failed to send alert",
                           alert_id=alert.id,
                           channel=channel_name,
                           error=str(e))
    
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        if channel.name == 'console':
            await self._send_console_alert(alert)
        elif channel.name == 'email':
            await self._send_email_alert(alert, channel)
        elif channel.name == 'slack':
            await self._send_slack_alert(alert, channel)
        elif channel.name == 'prometheus':
            await self._send_prometheus_alert(alert, channel)
        elif channel.name == 'pagerduty':
            await self._send_pagerduty_alert(alert, channel)
    
    async def _send_console_alert(self, alert: Alert):
        """Send alert to console/log"""
        logger.error("PERFORMANCE ALERT",
                    alert_id=alert.id,
                    test_name=alert.test_name,
                    severity=alert.severity,
                    message=alert.message)
    
    async def _send_email_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert via email"""
        config = channel.config
        
        if not config.get('recipients'):
            return
        
        msg = MIMEMultipart()
        msg['From'] = config['sender']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"[{alert.severity}] Performance Alert: {alert.test_name}"
        
        body = f"""
Performance Alert - {alert.severity}

Test: {alert.test_name}
Timestamp: {alert.timestamp}
Alert ID: {alert.id}

{alert.message}

This alert was generated by the GrandModel Performance Regression Detection System.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
            server.starttls()
            if config.get('username'):
                server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))
            raise
    
    async def _send_slack_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert to Slack"""
        webhook_url = channel.config.get('webhook_url')
        if not webhook_url:
            return
        
        # Color coding based on severity
        color_map = {
            'LOW': '#36a64f',      # Green
            'MEDIUM': '#ff9900',   # Orange
            'HIGH': '#ff0000',     # Red
            'CRITICAL': '#8B0000'  # Dark Red
        }
        
        payload = {
            'channel': channel.config.get('channel', '#performance-alerts'),
            'username': 'Performance Monitor',
            'icon_emoji': ':warning:',
            'attachments': [
                {
                    'color': color_map.get(alert.severity, '#ff9900'),
                    'title': f"Performance Alert - {alert.severity}",
                    'text': f"Test: {alert.test_name}",
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': alert.severity,
                            'short': True
                        },
                        {
                            'title': 'Test Name',
                            'value': alert.test_name,
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        },
                        {
                            'title': 'Alert ID',
                            'value': alert.id,
                            'short': True
                        }
                    ],
                    'footer': 'GrandModel Performance Monitor',
                    'ts': int(alert.timestamp.timestamp())
                }
            ]
        }
        
        # Add regression details if available
        if 'regression_percent' in alert.metadata:
            payload['attachments'][0]['fields'].extend([
                {
                    'title': 'Regression',
                    'value': f"{alert.metadata['regression_percent']:.1f}%",
                    'short': True
                },
                {
                    'title': 'Trend',
                    'value': alert.metadata['trend_direction'],
                    'short': True
                }
            ])
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack webhook returned status {response.status}")
    
    async def _send_prometheus_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert to Prometheus Pushgateway"""
        pushgateway_url = channel.config.get('pushgateway_url')
        if not pushgateway_url:
            return
        
        job_name = channel.config.get('job_name', 'performance-alerts')
        
        # Convert severity to numeric value
        severity_value = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}[alert.severity]
        
        # Create metrics
        metrics = [
            f'performance_regression_alert{{test_name="{alert.test_name}",severity="{alert.severity}"}} {severity_value}',
            f'performance_regression_alert_timestamp{{test_name="{alert.test_name}"}} {int(alert.timestamp.timestamp())}'
        ]
        
        if 'regression_percent' in alert.metadata:
            metrics.append(
                f'performance_regression_percent{{test_name="{alert.test_name}"}} {alert.metadata["regression_percent"]}'
            )
        
        data = '\n'.join(metrics)
        
        url = f"{pushgateway_url}/metrics/job/{job_name}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status not in [200, 202]:
                    raise Exception(f"Prometheus pushgateway returned status {response.status}")
    
    async def _send_pagerduty_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert to PagerDuty"""
        integration_key = channel.config.get('integration_key')
        if not integration_key:
            return
        
        payload = {
            'routing_key': integration_key,
            'event_action': 'trigger',
            'dedup_key': f"performance_regression_{alert.test_name}",
            'payload': {
                'summary': f"Performance regression in {alert.test_name}",
                'severity': alert.severity.lower(),
                'source': 'GrandModel Performance Monitor',
                'component': alert.test_name,
                'group': 'performance',
                'class': 'regression',
                'custom_details': {
                    'test_name': alert.test_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'metadata': alert.metadata
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload
            ) as response:
                if response.status != 202:
                    raise Exception(f"PagerDuty API returned status {response.status}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info("Alert acknowledged", 
                       alert_id=alert_id,
                       acknowledged_by=acknowledged_by)
    
    def resolve_alert(self, alert_id: str, resolved_by: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info("Alert resolved", 
                       alert_id=alert_id,
                       resolved_by=resolved_by)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alerting_report(self) -> Dict:
        """Generate alerting system report"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(24)
        
        # Alert statistics
        severity_counts = {}
        for alert in recent_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Channel statistics
        channel_stats = {}
        for alert in recent_alerts:
            for channel in alert.channels_notified:
                channel_stats[channel] = channel_stats.get(channel, 0) + 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_alerts': len(active_alerts),
            'alerts_last_24h': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'channel_usage': channel_stats,
            'configured_channels': len(self.channels),
            'enabled_channels': len([c for c in self.channels.values() if c.enabled]),
            'configured_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled])
        }

# Global instance
alerting_system = PerformanceAlertingSystem()