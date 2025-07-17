#!/usr/bin/env python3
"""
Agent 6: Production Alerting System
Critical risk and system health alerts with <1-minute response time
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge
import psutil
import requests

# Alert metrics
ALERTS_SENT = Counter('alerts_sent_total', 'Total alerts sent', ['alert_type', 'severity', 'channel'])
ALERT_LATENCY = Histogram('alert_latency_seconds', 'Alert processing latency')
ACTIVE_ALERTS = Gauge('active_alerts', 'Currently active alerts', ['alert_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"

@dataclass
class Alert:
    """Alert structure."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    title: str
    message: str
    source: str
    metrics: Dict[str, Any]
    threshold_breached: Optional[Dict[str, Any]] = None
    remediation_steps: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'alert_type': self.alert_type,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metrics': self.metrics,
            'threshold_breached': self.threshold_breached,
            'remediation_steps': self.remediation_steps
        }

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str  # Python expression
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    remediation_steps: Optional[List[str]] = None

class RiskMonitor:
    """Risk-specific monitoring and alerting."""
    
    def __init__(self):
        self.risk_thresholds = {
            'var_95_threshold': 0.03,  # 3% VaR threshold
            'correlation_shock_threshold': 0.5,  # 50% correlation spike
            'kelly_max_threshold': 0.25,  # 25% max Kelly fraction
            'margin_usage_threshold': 0.8,  # 80% margin usage
            'drawdown_threshold': 0.15,  # 15% max drawdown
            'sharpe_ratio_min': 1.0,  # Minimum Sharpe ratio
            'position_concentration_max': 0.2  # 20% max single position
        }
        
    async def check_risk_alerts(self, risk_metrics: Dict[str, float]) -> List[Alert]:
        """Check risk metrics against thresholds."""
        alerts = []
        
        # VaR threshold breach
        if risk_metrics.get('var_95', 0) > self.risk_thresholds['var_95_threshold']:
            alerts.append(Alert(
                id=f"risk_var_breach_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=AlertSeverity.CRITICAL,
                alert_type="risk_var_breach",
                title="VaR Threshold Breached",
                message=f"95% VaR ({risk_metrics['var_95']:.3f}) exceeds threshold ({self.risk_thresholds['var_95_threshold']:.3f})",
                source="risk_monitor",
                metrics=risk_metrics,
                threshold_breached={'metric': 'var_95', 'value': risk_metrics['var_95'], 'threshold': self.risk_thresholds['var_95_threshold']},
                remediation_steps=[
                    "Reduce position sizes immediately",
                    "Review correlation matrix for regime change",
                    "Consider hedge positions",
                    "Alert risk management team"
                ]
            ))
            
        # Correlation shock
        if risk_metrics.get('correlation_shock_level', 0) > self.risk_thresholds['correlation_shock_threshold']:
            alerts.append(Alert(
                id=f"correlation_shock_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=AlertSeverity.EMERGENCY,
                alert_type="correlation_shock",
                title="Market Correlation Shock Detected",
                message=f"Correlation shock level ({risk_metrics['correlation_shock_level']:.2f}) indicates potential market stress",
                source="risk_monitor",
                metrics=risk_metrics,
                threshold_breached={'metric': 'correlation_shock_level', 'value': risk_metrics['correlation_shock_level'], 'threshold': self.risk_thresholds['correlation_shock_threshold']},
                remediation_steps=[
                    "IMMEDIATE: Reduce leverage by 50%",
                    "Activate crisis management protocols",
                    "Review all open positions",
                    "Prepare for potential liquidations"
                ]
            ))
            
        # Margin usage
        if risk_metrics.get('margin_usage', 0) > self.risk_thresholds['margin_usage_threshold']:
            severity = AlertSeverity.CRITICAL if risk_metrics['margin_usage'] > 0.9 else AlertSeverity.WARNING
            alerts.append(Alert(
                id=f"margin_usage_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=severity,
                alert_type="high_margin_usage",
                title="High Margin Usage",
                message=f"Margin usage ({risk_metrics['margin_usage']:.1%}) approaching limits",
                source="risk_monitor",
                metrics=risk_metrics,
                threshold_breached={'metric': 'margin_usage', 'value': risk_metrics['margin_usage'], 'threshold': self.risk_thresholds['margin_usage_threshold']},
                remediation_steps=[
                    "Reduce position sizes",
                    "Close least profitable positions",
                    "Increase available margin",
                    "Review risk limits"
                ]
            ))
            
        # Max drawdown
        if risk_metrics.get('max_drawdown', 0) > self.risk_thresholds['drawdown_threshold']:
            alerts.append(Alert(
                id=f"max_drawdown_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=AlertSeverity.CRITICAL,
                alert_type="max_drawdown_breach",
                title="Maximum Drawdown Exceeded",
                message=f"Portfolio drawdown ({risk_metrics['max_drawdown']:.1%}) exceeds risk tolerance",
                source="risk_monitor",
                metrics=risk_metrics,
                threshold_breached={'metric': 'max_drawdown', 'value': risk_metrics['max_drawdown'], 'threshold': self.risk_thresholds['drawdown_threshold']},
                remediation_steps=[
                    "Halt new position opening",
                    "Review strategy performance",
                    "Consider portfolio rebalancing",
                    "Investigate cause of drawdown"
                ]
            ))
            
        return alerts

class PerformanceMonitor:
    """System performance monitoring and alerting."""
    
    def __init__(self):
        self.performance_thresholds = {
            'cpu_usage_threshold': 80.0,  # 80% CPU usage
            'memory_usage_threshold': 85.0,  # 85% memory usage
            'disk_usage_threshold': 90.0,  # 90% disk usage
            'latency_threshold_ms': 10.0,  # 10ms latency
            'error_rate_threshold': 0.05,  # 5% error rate
            'throughput_min_ops_sec': 100  # Min 100 ops/sec
        }
        
    async def check_performance_alerts(self, performance_metrics: Dict[str, float]) -> List[Alert]:
        """Check performance metrics against thresholds."""
        alerts = []
        
        # CPU usage
        cpu_usage = performance_metrics.get('cpu_usage', 0)
        if cpu_usage > self.performance_thresholds['cpu_usage_threshold']:
            severity = AlertSeverity.CRITICAL if cpu_usage > 95 else AlertSeverity.WARNING
            alerts.append(Alert(
                id=f"high_cpu_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=severity,
                alert_type="high_cpu_usage",
                title="High CPU Usage Detected",
                message=f"CPU usage ({cpu_usage:.1f}%) exceeds threshold",
                source="performance_monitor",
                metrics=performance_metrics,
                threshold_breached={'metric': 'cpu_usage', 'value': cpu_usage, 'threshold': self.performance_thresholds['cpu_usage_threshold']},
                remediation_steps=[
                    "Identify CPU-intensive processes",
                    "Scale horizontally if possible", 
                    "Optimize algorithms for efficiency",
                    "Consider load balancing"
                ]
            ))
            
        # Memory usage
        memory_usage = performance_metrics.get('memory_usage', 0)
        if memory_usage > self.performance_thresholds['memory_usage_threshold']:
            severity = AlertSeverity.CRITICAL if memory_usage > 95 else AlertSeverity.WARNING
            alerts.append(Alert(
                id=f"high_memory_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=severity,
                alert_type="high_memory_usage",
                title="High Memory Usage Detected",
                message=f"Memory usage ({memory_usage:.1f}%) approaching limits",
                source="performance_monitor",
                metrics=performance_metrics,
                threshold_breached={'metric': 'memory_usage', 'value': memory_usage, 'threshold': self.performance_thresholds['memory_usage_threshold']},
                remediation_steps=[
                    "Clear unnecessary cached data",
                    "Restart memory-intensive services",
                    "Investigate memory leaks",
                    "Scale up memory resources"
                ]
            ))
            
        # Latency breach
        avg_latency = performance_metrics.get('avg_latency_ms', 0)
        if avg_latency > self.performance_thresholds['latency_threshold_ms']:
            severity = AlertSeverity.CRITICAL if avg_latency > 20 else AlertSeverity.WARNING
            alerts.append(Alert(
                id=f"high_latency_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=severity,
                alert_type="high_latency",
                title="High Latency Detected",
                message=f"Average latency ({avg_latency:.2f}ms) exceeds {self.performance_thresholds['latency_threshold_ms']}ms target",
                source="performance_monitor",
                metrics=performance_metrics,
                threshold_breached={'metric': 'avg_latency_ms', 'value': avg_latency, 'threshold': self.performance_thresholds['latency_threshold_ms']},
                remediation_steps=[
                    "Optimize critical code paths",
                    "Check database query performance",
                    "Review caching strategies",
                    "Consider JIT compilation optimizations"
                ]
            ))
            
        return alerts

class AgentHealthMonitor:
    """MARL agent health monitoring."""
    
    def __init__(self):
        self.agent_thresholds = {
            'inference_latency_ms': 8.0,  # 8ms inference latency
            'throughput_ops_sec': 50,  # 50 ops/sec minimum
            'error_rate': 0.02,  # 2% error rate
            'model_accuracy': 0.8  # 80% minimum accuracy
        }
        
    async def check_agent_health(self, agent_metrics: Dict[str, Dict[str, float]]) -> List[Alert]:
        """Check agent health metrics."""
        alerts = []
        
        for agent_type, metrics in agent_metrics.items():
            # Inference latency
            inference_latency = metrics.get('inference_latency_ms', 0)
            if inference_latency > self.agent_thresholds['inference_latency_ms']:
                alerts.append(Alert(
                    id=f"agent_latency_{agent_type}_{int(time.time())}",
                    timestamp=datetime.utcnow(),
                    severity=AlertSeverity.WARNING,
                    alert_type="agent_high_latency",
                    title=f"{agent_type.title()} Agent High Latency",
                    message=f"{agent_type} agent inference latency ({inference_latency:.2f}ms) exceeds threshold",
                    source="agent_monitor",
                    metrics=metrics,
                    threshold_breached={'metric': 'inference_latency_ms', 'value': inference_latency, 'threshold': self.agent_thresholds['inference_latency_ms']},
                    remediation_steps=[
                        f"Optimize {agent_type} agent model",
                        "Check GPU/CPU utilization",
                        "Review batch processing efficiency",
                        "Consider model quantization"
                    ]
                ))
                
            # Low throughput
            throughput = metrics.get('throughput_ops_per_sec', 0)
            if throughput < self.agent_thresholds['throughput_ops_sec']:
                alerts.append(Alert(
                    id=f"agent_throughput_{agent_type}_{int(time.time())}",
                    timestamp=datetime.utcnow(),
                    severity=AlertSeverity.WARNING,
                    alert_type="agent_low_throughput",
                    title=f"{agent_type.title()} Agent Low Throughput",
                    message=f"{agent_type} agent throughput ({throughput:.1f} ops/sec) below minimum",
                    source="agent_monitor",
                    metrics=metrics,
                    threshold_breached={'metric': 'throughput_ops_per_sec', 'value': throughput, 'threshold': self.agent_thresholds['throughput_ops_sec']},
                    remediation_steps=[
                        f"Scale {agent_type} agent instances",
                        "Optimize request batching",
                        "Check for bottlenecks",
                        "Review resource allocation"
                    ]
                ))
                
            # High error rate
            error_rate = metrics.get('error_rate', 0)
            if error_rate > self.agent_thresholds['error_rate']:
                severity = AlertSeverity.CRITICAL if error_rate > 0.1 else AlertSeverity.WARNING
                alerts.append(Alert(
                    id=f"agent_errors_{agent_type}_{int(time.time())}",
                    timestamp=datetime.utcnow(),
                    severity=severity,
                    alert_type="agent_high_error_rate",
                    title=f"{agent_type.title()} Agent High Error Rate",
                    message=f"{agent_type} agent error rate ({error_rate:.1%}) exceeds threshold",
                    source="agent_monitor",
                    metrics=metrics,
                    threshold_breached={'metric': 'error_rate', 'value': error_rate, 'threshold': self.agent_thresholds['error_rate']},
                    remediation_steps=[
                        f"Investigate {agent_type} agent errors",
                        "Check input data quality",
                        "Review model validation",
                        "Consider model retraining"
                    ]
                ))
                
        return alerts

class AlertDelivery:
    """Alert delivery system with multiple channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cooldown_cache = {}  # Alert cooldown tracking
        
    async def send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert through specified channels."""
        start_time = time.time()
        
        try:
            # Check cooldown
            if self._is_in_cooldown(alert):
                logger.info(f"Alert {alert.id} in cooldown, skipping")
                return
                
            # Send through each channel
            for channel in channels:
                try:
                    await self._send_to_channel(alert, channel)
                    ALERTS_SENT.labels(
                        alert_type=alert.alert_type,
                        severity=alert.severity.value,
                        channel=channel.value
                    ).inc()
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel.value}: {e}")
                    
            # Update cooldown cache
            self._update_cooldown(alert)
            
        finally:
            ALERT_LATENCY.observe(time.time() - start_time)
            
    def _is_in_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period."""
        cooldown_key = f"{alert.alert_type}_{alert.source}"
        last_sent = self.cooldown_cache.get(cooldown_key)
        
        if last_sent:
            cooldown_period = timedelta(minutes=15)  # Default 15 minute cooldown
            if datetime.utcnow() - last_sent < cooldown_period:
                return True
                
        return False
        
    def _update_cooldown(self, alert: Alert):
        """Update cooldown cache."""
        cooldown_key = f"{alert.alert_type}_{alert.source}"
        self.cooldown_cache[cooldown_key] = datetime.utcnow()
        
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel."""
        if channel == AlertChannel.EMAIL:
            await self._send_email(alert)
        elif channel == AlertChannel.SLACK:
            await self._send_slack(alert)
        elif channel == AlertChannel.WEBHOOK:
            await self._send_webhook(alert)
        elif channel == AlertChannel.CONSOLE:
            self._send_console(alert)
            
    async def _send_email(self, alert: Alert):
        """Send email alert."""
        if 'email' not in self.config:
            return
            
        email_config = self.config['email']
        
        msg = MimeMultipart()
        msg['From'] = email_config['from']
        msg['To'] = ', '.join(email_config['to'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
Alert Details:
- Severity: {alert.severity.value.upper()}
- Type: {alert.alert_type}
- Source: {alert.source}
- Time: {alert.timestamp.isoformat()}

Message:
{alert.message}

Metrics:
{json.dumps(alert.metrics, indent=2)}

Remediation Steps:
"""
        if alert.remediation_steps:
            for i, step in enumerate(alert.remediation_steps, 1):
                body += f"{i}. {step}\n"
                
        msg.attach(MimeText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            
    async def _send_slack(self, alert: Alert):
        """Send Slack alert."""
        if 'slack' not in self.config:
            return
            
        slack_config = self.config['slack']
        
        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.EMERGENCY: "danger"
        }.get(alert.severity, "warning")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Type", "value": alert.alert_type, "short": True},
                    {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                ],
                "footer": "GrandModel Alert System",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        if alert.remediation_steps:
            payload["attachments"][0]["fields"].append({
                "title": "Remediation Steps",
                "value": "\n".join(f"• {step}" for step in alert.remediation_steps),
                "short": False
            })
            
        async with aiohttp.ClientSession() as session:
            async with session.post(slack_config['webhook_url'], json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack webhook failed: {response.status}")
                    
    async def _send_webhook(self, alert: Alert):
        """Send webhook alert."""
        if 'webhook' not in self.config:
            return
            
        webhook_config = self.config['webhook']
        
        payload = {
            "alert": alert.to_dict(),
            "timestamp": alert.timestamp.isoformat(),
            "source": "grandmodel_alerting"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_config['url'], json=payload) as response:
                if response.status not in [200, 201, 202]:
                    logger.error(f"Webhook failed: {response.status}")
                    
    def _send_console(self, alert: Alert):
        """Send console alert."""
        severity_icons = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔥"
        }
        
        icon = severity_icons.get(alert.severity, "⚠️")
        print(f"\n{icon} ALERT [{alert.severity.value.upper()}] {alert.title}")
        print(f"Source: {alert.source}")
        print(f"Time: {alert.timestamp.isoformat()}")
        print(f"Message: {alert.message}")
        if alert.remediation_steps:
            print("Remediation Steps:")
            for i, step in enumerate(alert.remediation_steps, 1):
                print(f"  {i}. {step}")
        print("-" * 50)

class AlertingSystem:
    """Main alerting system coordinator."""
    
    def __init__(self, config: Dict[str, Any], redis_client: redis.Redis):
        self.config = config
        self.redis_client = redis_client
        
        # Initialize monitors
        self.risk_monitor = RiskMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.agent_monitor = AgentHealthMonitor()
        
        # Initialize delivery
        self.alert_delivery = AlertDelivery(config)
        
        # Alert rules
        self.alert_rules = self._load_alert_rules()
        
        # Active alerts tracking
        self.active_alerts = {}
        
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules configuration."""
        return [
            AlertRule(
                name="critical_risk_breach",
                condition="severity == 'critical' and alert_type.startswith('risk_')",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                cooldown_minutes=5
            ),
            AlertRule(
                name="emergency_correlation_shock",
                condition="alert_type == 'correlation_shock'",
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
                cooldown_minutes=1
            ),
            AlertRule(
                name="performance_degradation", 
                condition="alert_type.startswith('high_') or alert_type.startswith('agent_')",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
                cooldown_minutes=10
            )
        ]
        
    async def process_metrics_and_alert(self, metrics: Dict[str, Any]):
        """Process metrics and generate alerts."""
        all_alerts = []
        
        # Risk monitoring
        if 'risk_metrics' in metrics:
            risk_alerts = await self.risk_monitor.check_risk_alerts(metrics['risk_metrics'])
            all_alerts.extend(risk_alerts)
            
        # Performance monitoring
        if 'performance_metrics' in metrics:
            perf_alerts = await self.performance_monitor.check_performance_alerts(metrics['performance_metrics'])
            all_alerts.extend(perf_alerts)
            
        # Agent monitoring
        if 'agent_metrics' in metrics:
            agent_alerts = await self.agent_monitor.check_agent_health(metrics['agent_metrics'])
            all_alerts.extend(agent_alerts)
            
        # Process and send alerts
        for alert in all_alerts:
            await self._process_alert(alert)
            
        return all_alerts
        
    async def _process_alert(self, alert: Alert):
        """Process individual alert."""
        # Update active alerts
        ACTIVE_ALERTS.labels(alert_type=alert.alert_type).inc()
        self.active_alerts[alert.id] = alert
        
        # Store in Redis
        await self.redis_client.setex(
            f"alert:{alert.id}",
            3600,  # 1 hour TTL
            json.dumps(alert.to_dict())
        )
        
        # Find matching rules and send alerts
        for rule in self.alert_rules:
            if self._rule_matches(rule, alert):
                await self.alert_delivery.send_alert(alert, rule.channels)
                break
                
    def _rule_matches(self, rule: AlertRule, alert: Alert) -> bool:
        """Check if alert matches rule condition."""
        try:
            # Create evaluation context
            context = {
                'severity': alert.severity.value,
                'alert_type': alert.alert_type,
                'source': alert.source
            }
            
            # Evaluate condition safely - SECURITY FIX
            # Replace eval() with safe expression evaluator
            import ast
            import operator
            
            # Safe operators for expression evaluation
            safe_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.And: operator.and_,
                ast.Or: operator.or_,
            }
            
            def safe_eval(node, context):
                """Safe expression evaluator that replaces eval()"""
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Name):
                    if node.id in context:
                        return context[node.id]
                    else:
                        raise ValueError(f"Undefined variable: {node.id}")
                elif isinstance(node, ast.BinOp):
                    left = safe_eval(node.left, context)
                    right = safe_eval(node.right, context)
                    if type(node.op) in safe_operators:
                        return safe_operators[type(node.op)](left, right)
                    else:
                        raise ValueError(f"Unsupported operator: {type(node.op)}")
                elif isinstance(node, ast.Compare):
                    left = safe_eval(node.left, context)
                    for op, comparator in zip(node.ops, node.comparators):
                        right = safe_eval(comparator, context)
                        if type(op) in safe_operators:
                            result = safe_operators[type(op)](left, right)
                            if not result:
                                return False
                            left = right
                        else:
                            raise ValueError(f"Unsupported comparison: {type(op)}")
                    return True
                elif isinstance(node, ast.BoolOp):
                    if isinstance(node.op, ast.And):
                        return all(safe_eval(value, context) for value in node.values)
                    elif isinstance(node.op, ast.Or):
                        return any(safe_eval(value, context) for value in node.values)
                else:
                    raise ValueError(f"Unsupported node type: {type(node)}")
            
            # For immediate safety, validate condition format
            allowed_vars = {'severity', 'alert_type', 'source'}
            
            # Basic validation - only allow simple comparisons
            if any(dangerous in rule.condition for dangerous in ['import', 'exec', 'eval', '__', 'open', 'file']):
                logger.error(f"Dangerous condition detected: {rule.condition}")
                return False
            
            # Parse and validate AST
            try:
                tree = ast.parse(rule.condition, mode='eval')
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id not in allowed_vars:
                        logger.error(f"Unauthorized variable in condition: {node.id}")
                        return False
                    elif isinstance(node, ast.Call):
                        logger.error(f"Function calls not allowed in conditions: {rule.condition}")
                        return False
                
                # SECURITY FIX: Replace eval() with safe expression evaluator
                return safe_eval(tree.body, context)
            except (SyntaxError, ValueError) as e:
                logger.error(f"Invalid condition syntax: {rule.condition} - {e}")
                return False
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return False
            
    async def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status."""
        try:
            # Get recent alerts from Redis
            alert_keys = await self.redis_client.keys("alert:*")
            recent_alerts = []
            
            for key in alert_keys[-10:]:  # Last 10 alerts
                alert_data = await self.redis_client.get(key)
                if alert_data:
                    recent_alerts.append(json.loads(alert_data))
                    
            # Count by severity
            severity_counts = {}
            for alert in recent_alerts:
                severity = alert['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_active_alerts": len(self.active_alerts),
                "recent_alerts_count": len(recent_alerts),
                "severity_breakdown": severity_counts,
                "recent_alerts": recent_alerts
            }
            
        except Exception as e:
            logger.error(f"Error getting alert status: {e}")
            return {"status": "error", "message": str(e)}

# Factory function
def create_alerting_system(config: Dict[str, Any], redis_client: redis.Redis) -> AlertingSystem:
    """Create alerting system instance."""
    return AlertingSystem(config, redis_client)

# Example configuration
EXAMPLE_CONFIG = {
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
        "from": "alerts@grandmodel.ai",
        "to": ["risk@grandmodel.ai", "ops@grandmodel.ai"],
        "username": "alerts@grandmodel.ai",
        "password": "app_password"
    },
    "slack": {
        "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    },
    "webhook": {
        "url": "https://your-monitoring-system.com/alerts"
    }
}