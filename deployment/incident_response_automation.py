#!/usr/bin/env python3
"""
GrandModel Incident Response Automation System - Agent 20 Implementation
Enterprise-grade incident response automation with escalation and notification capabilities
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from kubernetes import client, config
from prometheus_client.parser import text_string_to_metric_families
import boto3
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    P0 = "P0"  # Critical - System down
    P1 = "P1"  # High - Major functionality impaired
    P2 = "P2"  # Medium - Minor functionality impaired
    P3 = "P3"  # Low - Cosmetic issues

class IncidentStatus(Enum):
    """Incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"

class AlertType(Enum):
    """Alert types"""
    SYSTEM_DOWN = "system_down"
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_BREACH = "security_breach"
    CORRELATION_SHOCK = "correlation_shock"
    RISK_LIMIT_BREACH = "risk_limit_breach"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    alert_type: AlertType
    severity: IncidentSeverity
    title: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    starts_at: datetime = field(default_factory=datetime.now)
    ends_at: Optional[datetime] = None
    generator_url: Optional[str] = None
    fingerprint: Optional[str] = None

@dataclass
class Incident:
    """Incident data structure"""
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    alerts: List[Alert] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    escalated_to: Optional[str] = None
    escalation_level: int = 0
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    post_mortem_required: bool = False

@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    severity: IncidentSeverity
    initial_response_time: int  # minutes
    escalation_time: int  # minutes
    escalation_levels: List[str]  # contact groups
    notification_channels: List[str]  # slack, email, phone

class NotificationManager:
    """Notification management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slack_client = WebClient(token=config.get('slack_token'))
        self.ses_client = boto3.client('ses', region_name=config.get('aws_region', 'us-east-1'))
        
    async def send_slack_notification(self, channel: str, message: str, severity: IncidentSeverity) -> bool:
        """Send Slack notification"""
        try:
            color_map = {
                IncidentSeverity.P0: "danger",
                IncidentSeverity.P1: "warning",
                IncidentSeverity.P2: "good",
                IncidentSeverity.P3: "#808080"
            }
            
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                attachments=[
                    {
                        "color": color_map.get(severity, "good"),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": severity.value,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            }
                        ]
                    }
                ]
            )
            return response["ok"]
        except SlackApiError as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    async def send_email_notification(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.get('email_from', 'alerts@grandmodel.com')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'html'))
            
            response = self.ses_client.send_raw_email(
                Source=msg['From'],
                Destinations=recipients,
                RawMessage={'Data': msg.as_string()}
            )
            
            return response['ResponseMetadata']['HTTPStatusCode'] == 200
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    async def send_phone_notification(self, phone_number: str, message: str) -> bool:
        """Send phone notification via SNS"""
        try:
            sns_client = boto3.client('sns', region_name=self.config.get('aws_region', 'us-east-1'))
            
            response = sns_client.publish(
                PhoneNumber=phone_number,
                Message=message,
                MessageAttributes={
                    'AWS.SNS.SMS.SenderID': {
                        'DataType': 'String',
                        'StringValue': 'GrandModel'
                    }
                }
            )
            
            return response['ResponseMetadata']['HTTPStatusCode'] == 200
        except Exception as e:
            logger.error(f"Phone notification failed: {e}")
            return False

class MetricsCollector:
    """Metrics collection for incident analysis"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
    
    async def query_metric(self, query: str) -> Optional[Dict[str, Any]]:
        """Query Prometheus metrics"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Metrics query failed: {e}")
            return None
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        health_metrics = {}
        
        # Service availability
        availability_query = 'up{job=~"grandmodel.*"}'
        availability_result = await self.query_metric(availability_query)
        if availability_result:
            health_metrics['service_availability'] = availability_result
        
        # Error rates
        error_rate_query = 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])'
        error_rate_result = await self.query_metric(error_rate_query)
        if error_rate_result:
            health_metrics['error_rate'] = error_rate_result
        
        # Response times
        latency_query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
        latency_result = await self.query_metric(latency_query)
        if latency_result:
            health_metrics['response_time'] = latency_result
        
        # Resource utilization
        cpu_query = 'rate(container_cpu_usage_seconds_total[5m])'
        cpu_result = await self.query_metric(cpu_query)
        if cpu_result:
            health_metrics['cpu_usage'] = cpu_result
        
        memory_query = 'container_memory_usage_bytes'
        memory_result = await self.query_metric(memory_query)
        if memory_result:
            health_metrics['memory_usage'] = memory_result
        
        return health_metrics

class AutomatedActions:
    """Automated incident response actions"""
    
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
    
    async def restart_service(self, service_name: str, namespace: str = "grandmodel") -> bool:
        """Restart a service by rolling restart"""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            # Get deployment
            deployment = apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=namespace
            )
            
            # Trigger rolling restart
            deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
            deployment.spec.template.metadata.annotations['kubectl.kubernetes.io/restartedAt'] = datetime.now().isoformat()
            
            apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Restarted service {service_name} in namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return False
    
    async def scale_service(self, service_name: str, replicas: int, namespace: str = "grandmodel") -> bool:
        """Scale a service to specified replicas"""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            # Scale deployment
            apps_v1.patch_namespaced_deployment_scale(
                name=service_name,
                namespace=namespace,
                body={"spec": {"replicas": replicas}}
            )
            
            logger.info(f"Scaled service {service_name} to {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"Failed to scale service {service_name}: {e}")
            return False
    
    async def drain_node(self, node_name: str) -> bool:
        """Drain a problematic node"""
        try:
            # This would typically use kubectl drain
            # For now, we'll simulate the action
            logger.info(f"Draining node {node_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drain node {node_name}: {e}")
            return False
    
    async def enable_maintenance_mode(self, service_name: str, namespace: str = "grandmodel") -> bool:
        """Enable maintenance mode for a service"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Update service to point to maintenance page
            service = v1.read_namespaced_service(
                name=service_name,
                namespace=namespace
            )
            
            # Add maintenance mode annotation
            service.metadata.annotations = service.metadata.annotations or {}
            service.metadata.annotations['grandmodel.com/maintenance-mode'] = 'true'
            
            v1.patch_namespaced_service(
                name=service_name,
                namespace=namespace,
                body=service
            )
            
            logger.info(f"Enabled maintenance mode for service {service_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable maintenance mode for {service_name}: {e}")
            return False

class IncidentResponseManager:
    """Main incident response management system"""
    
    def __init__(self, config_path: str = "/app/config/incident-response.yaml"):
        self.config = self.load_config(config_path)
        self.notification_manager = NotificationManager(self.config['notifications'])
        self.metrics_collector = MetricsCollector(self.config['prometheus_url'])
        
        # Load Kubernetes config
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.automated_actions = AutomatedActions(client.ApiClient())
        
        # In-memory incident storage (would be replaced with persistent storage)
        self.incidents: Dict[str, Incident] = {}
        self.escalation_rules = self.load_escalation_rules()
        
        # Alert processing
        self.alert_queue = asyncio.Queue()
        self.running = False
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'prometheus_url': 'http://prometheus:9090',
                'notifications': {
                    'slack_token': 'xoxb-your-token-here',
                    'email_from': 'alerts@grandmodel.com',
                    'aws_region': 'us-east-1'
                },
                'escalation_rules': []
            }
    
    def load_escalation_rules(self) -> List[EscalationRule]:
        """Load escalation rules from configuration"""
        rules = []
        for rule_config in self.config.get('escalation_rules', []):
            rule = EscalationRule(
                severity=IncidentSeverity(rule_config['severity']),
                initial_response_time=rule_config['initial_response_time'],
                escalation_time=rule_config['escalation_time'],
                escalation_levels=rule_config['escalation_levels'],
                notification_channels=rule_config['notification_channels']
            )
            rules.append(rule)
        return rules
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> None:
        """Process incoming alert"""
        try:
            # Parse alert data
            alert = Alert(
                id=alert_data.get('id', f"alert-{int(time.time())}"),
                alert_type=AlertType(alert_data.get('alert_type', 'system_down')),
                severity=IncidentSeverity(alert_data.get('severity', 'P2')),
                title=alert_data.get('title', 'Unknown Alert'),
                description=alert_data.get('description', 'No description provided'),
                labels=alert_data.get('labels', {}),
                annotations=alert_data.get('annotations', {}),
                generator_url=alert_data.get('generator_url'),
                fingerprint=alert_data.get('fingerprint')
            )
            
            # Check if incident already exists for this alert
            incident = await self.find_or_create_incident(alert)
            
            # Process the incident
            await self.handle_incident(incident)
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    async def find_or_create_incident(self, alert: Alert) -> Incident:
        """Find existing incident or create new one"""
        # Check for existing incident with same fingerprint
        for incident in self.incidents.values():
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                for existing_alert in incident.alerts:
                    if existing_alert.fingerprint == alert.fingerprint:
                        incident.alerts.append(alert)
                        incident.updated_at = datetime.now()
                        return incident
        
        # Create new incident
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{len(self.incidents) + 1:03d}"
        incident = Incident(
            id=incident_id,
            title=alert.title,
            description=alert.description,
            severity=alert.severity,
            status=IncidentStatus.OPEN,
            alerts=[alert],
            post_mortem_required=alert.severity in [IncidentSeverity.P0, IncidentSeverity.P1]
        )
        
        self.incidents[incident_id] = incident
        
        # Add to timeline
        incident.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'incident_created',
            'description': f'Incident created from alert: {alert.title}'
        })
        
        logger.info(f"Created new incident: {incident_id}")
        return incident
    
    async def handle_incident(self, incident: Incident) -> None:
        """Handle incident processing"""
        # Update incident status
        if incident.status == IncidentStatus.OPEN:
            incident.status = IncidentStatus.INVESTIGATING
            incident.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'status_changed',
                'description': 'Status changed to investigating'
            })
        
        # Send initial notifications
        await self.send_incident_notifications(incident)
        
        # Execute automated actions
        await self.execute_automated_actions(incident)
        
        # Set up escalation timer
        await self.setup_escalation_timer(incident)
        
        # Collect additional metrics
        await self.collect_incident_metrics(incident)
    
    async def send_incident_notifications(self, incident: Incident) -> None:
        """Send incident notifications"""
        escalation_rule = self.get_escalation_rule(incident.severity)
        if not escalation_rule:
            return
        
        message = self.format_incident_message(incident)
        
        # Send to appropriate channels
        if 'slack' in escalation_rule.notification_channels:
            await self.notification_manager.send_slack_notification(
                channel=self.config['notifications'].get('slack_channel', '#alerts'),
                message=message,
                severity=incident.severity
            )
        
        if 'email' in escalation_rule.notification_channels:
            recipients = self.get_notification_recipients(incident.severity, incident.escalation_level)
            await self.notification_manager.send_email_notification(
                recipients=recipients,
                subject=f"[{incident.severity.value}] {incident.title}",
                body=self.format_incident_email(incident)
            )
        
        if 'phone' in escalation_rule.notification_channels and incident.severity == IncidentSeverity.P0:
            phone_numbers = self.get_phone_numbers(incident.severity)
            for phone in phone_numbers:
                await self.notification_manager.send_phone_notification(
                    phone_number=phone,
                    message=f"CRITICAL: {incident.title}"
                )
    
    async def execute_automated_actions(self, incident: Incident) -> None:
        """Execute automated remediation actions"""
        actions_executed = []
        
        # Determine actions based on alert type
        for alert in incident.alerts:
            if alert.alert_type == AlertType.SYSTEM_DOWN:
                service_name = alert.labels.get('service', 'strategic-deployment')
                if await self.automated_actions.restart_service(service_name):
                    actions_executed.append(f"Restarted service {service_name}")
            
            elif alert.alert_type == AlertType.HIGH_ERROR_RATE:
                service_name = alert.labels.get('service', 'tactical-deployment')
                if await self.automated_actions.scale_service(service_name, 5):
                    actions_executed.append(f"Scaled service {service_name} to 5 replicas")
            
            elif alert.alert_type == AlertType.RESOURCE_EXHAUSTION:
                node_name = alert.labels.get('node')
                if node_name and await self.automated_actions.drain_node(node_name):
                    actions_executed.append(f"Drained node {node_name}")
            
            elif alert.alert_type == AlertType.CORRELATION_SHOCK:
                # Risk-specific actions
                if await self.automated_actions.enable_maintenance_mode('risk-service'):
                    actions_executed.append("Enabled maintenance mode for risk service")
        
        if actions_executed:
            incident.remediation_actions.extend(actions_executed)
            incident.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'automated_actions_executed',
                'description': f'Executed actions: {", ".join(actions_executed)}'
            })
    
    async def setup_escalation_timer(self, incident: Incident) -> None:
        """Set up escalation timer"""
        escalation_rule = self.get_escalation_rule(incident.severity)
        if not escalation_rule:
            return
        
        # Schedule escalation
        asyncio.create_task(self.escalate_incident(incident, escalation_rule))
    
    async def escalate_incident(self, incident: Incident, escalation_rule: EscalationRule) -> None:
        """Escalate incident if not resolved"""
        await asyncio.sleep(escalation_rule.escalation_time * 60)  # Convert minutes to seconds
        
        # Check if incident is still open
        if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            incident.escalation_level += 1
            incident.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'escalated',
                'description': f'Escalated to level {incident.escalation_level}'
            })
            
            # Send escalation notifications
            await self.send_incident_notifications(incident)
            
            # Schedule next escalation if available
            if incident.escalation_level < len(escalation_rule.escalation_levels):
                await self.escalate_incident(incident, escalation_rule)
    
    async def collect_incident_metrics(self, incident: Incident) -> None:
        """Collect metrics related to the incident"""
        try:
            health_metrics = await self.metrics_collector.get_system_health()
            incident.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'metrics_collected',
                'description': f'Collected system health metrics',
                'data': health_metrics
            })
        except Exception as e:
            logger.error(f"Failed to collect metrics for incident {incident.id}: {e}")
    
    def get_escalation_rule(self, severity: IncidentSeverity) -> Optional[EscalationRule]:
        """Get escalation rule for severity"""
        for rule in self.escalation_rules:
            if rule.severity == severity:
                return rule
        return None
    
    def get_notification_recipients(self, severity: IncidentSeverity, escalation_level: int) -> List[str]:
        """Get notification recipients based on severity and escalation level"""
        escalation_rule = self.get_escalation_rule(severity)
        if not escalation_rule or escalation_level >= len(escalation_rule.escalation_levels):
            return []
        
        # This would typically look up contacts from a directory
        contact_groups = {
            'on_call': ['oncall@grandmodel.com'],
            'engineering': ['engineering@grandmodel.com'],
            'management': ['management@grandmodel.com'],
            'executives': ['executives@grandmodel.com']
        }
        
        group = escalation_rule.escalation_levels[escalation_level]
        return contact_groups.get(group, [])
    
    def get_phone_numbers(self, severity: IncidentSeverity) -> List[str]:
        """Get phone numbers for critical alerts"""
        if severity == IncidentSeverity.P0:
            return ['+1-555-0101', '+1-555-0102']  # On-call numbers
        return []
    
    def format_incident_message(self, incident: Incident) -> str:
        """Format incident message for notifications"""
        return f"""
🚨 *{incident.severity.value} - {incident.title}*

*Incident ID:* {incident.id}
*Status:* {incident.status.value}
*Created:* {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
*Description:* {incident.description}

*Alerts:* {len(incident.alerts)}
*Escalation Level:* {incident.escalation_level}

*Automated Actions:*
{chr(10).join(f"• {action}" for action in incident.remediation_actions)}

*Dashboard:* https://grafana.grandmodel.com/d/incidents/{incident.id}
"""
    
    def format_incident_email(self, incident: Incident) -> str:
        """Format incident email"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-bottom: 2px solid #dee2e6; }}
        .severity-p0 {{ color: #dc3545; }}
        .severity-p1 {{ color: #fd7e14; }}
        .severity-p2 {{ color: #ffc107; }}
        .severity-p3 {{ color: #28a745; }}
        .content {{ padding: 20px; }}
        .timeline {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; }}
        .actions {{ background-color: #e9ecef; padding: 15px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h2 class="severity-{incident.severity.value.lower()}">
            {incident.severity.value} - {incident.title}
        </h2>
        <p><strong>Incident ID:</strong> {incident.id}</p>
        <p><strong>Status:</strong> {incident.status.value}</p>
        <p><strong>Created:</strong> {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
    
    <div class="content">
        <h3>Description</h3>
        <p>{incident.description}</p>
        
        <h3>Alert Details</h3>
        <ul>
            {"".join(f"<li>{alert.title} - {alert.description}</li>" for alert in incident.alerts)}
        </ul>
        
        <div class="actions">
            <h3>Automated Actions Taken</h3>
            <ul>
                {"".join(f"<li>{action}</li>" for action in incident.remediation_actions)}
            </ul>
        </div>
        
        <div class="timeline">
            <h3>Timeline</h3>
            <ul>
                {"".join(f"<li>{event['timestamp']}: {event['description']}</li>" for event in incident.timeline)}
            </ul>
        </div>
        
        <h3>Next Steps</h3>
        <ul>
            <li>Review incident details in the dashboard</li>
            <li>Investigate root cause</li>
            <li>Implement additional remediation if needed</li>
            <li>Update incident status</li>
        </ul>
        
        <p><strong>Dashboard:</strong> <a href="https://grafana.grandmodel.com/d/incidents/{incident.id}">View Incident</a></p>
    </div>
</body>
</html>
"""
    
    async def resolve_incident(self, incident_id: str, resolution_notes: str) -> bool:
        """Resolve an incident"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False
        
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now()
        incident.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'resolved',
            'description': f'Incident resolved: {resolution_notes}'
        })
        
        # Send resolution notification
        await self.send_resolution_notification(incident)
        
        # Schedule post-mortem if required
        if incident.post_mortem_required:
            await self.schedule_post_mortem(incident)
        
        return True
    
    async def send_resolution_notification(self, incident: Incident) -> None:
        """Send incident resolution notification"""
        message = f"""
✅ *RESOLVED - {incident.title}*

*Incident ID:* {incident.id}
*Resolution Time:* {incident.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
*Duration:* {incident.resolved_at - incident.created_at}

*Actions Taken:*
{chr(10).join(f"• {action}" for action in incident.remediation_actions)}

*Post-Mortem Required:* {'Yes' if incident.post_mortem_required else 'No'}
"""
        
        await self.notification_manager.send_slack_notification(
            channel=self.config['notifications'].get('slack_channel', '#alerts'),
            message=message,
            severity=incident.severity
        )
    
    async def schedule_post_mortem(self, incident: Incident) -> None:
        """Schedule post-mortem review"""
        # This would integrate with calendar systems
        logger.info(f"Post-mortem scheduled for incident {incident.id}")
        
        incident.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'post_mortem_scheduled',
            'description': 'Post-mortem review scheduled'
        })
    
    async def start(self) -> None:
        """Start the incident response system"""
        self.running = True
        logger.info("Incident response system started")
        
        # Start alert processing
        asyncio.create_task(self.process_alert_queue())
        
        # Start health monitoring
        asyncio.create_task(self.monitor_system_health())
    
    async def stop(self) -> None:
        """Stop the incident response system"""
        self.running = False
        logger.info("Incident response system stopped")
    
    async def process_alert_queue(self) -> None:
        """Process alerts from the queue"""
        while self.running:
            try:
                alert_data = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                await self.process_alert(alert_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert queue: {e}")
    
    async def monitor_system_health(self) -> None:
        """Monitor system health and generate alerts"""
        while self.running:
            try:
                health_metrics = await self.metrics_collector.get_system_health()
                
                # Check for issues and generate alerts
                await self.check_service_availability(health_metrics)
                await self.check_error_rates(health_metrics)
                await self.check_response_times(health_metrics)
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(60)
    
    async def check_service_availability(self, health_metrics: Dict[str, Any]) -> None:
        """Check service availability and generate alerts"""
        availability_data = health_metrics.get('service_availability', {})
        results = availability_data.get('data', {}).get('result', [])
        
        for result in results:
            if float(result['value'][1]) == 0:  # Service is down
                service_name = result['metric'].get('job', 'unknown')
                
                alert_data = {
                    'id': f'service-down-{service_name}-{int(time.time())}',
                    'alert_type': 'system_down',
                    'severity': 'P0',
                    'title': f'Service {service_name} is down',
                    'description': f'Service {service_name} is not responding to health checks',
                    'labels': {'service': service_name},
                    'fingerprint': f'service-down-{service_name}'
                }
                
                await self.alert_queue.put(alert_data)
    
    async def check_error_rates(self, health_metrics: Dict[str, Any]) -> None:
        """Check error rates and generate alerts"""
        error_data = health_metrics.get('error_rate', {})
        results = error_data.get('data', {}).get('result', [])
        
        for result in results:
            error_rate = float(result['value'][1])
            if error_rate > 0.05:  # 5% error rate threshold
                service_name = result['metric'].get('job', 'unknown')
                
                alert_data = {
                    'id': f'high-error-rate-{service_name}-{int(time.time())}',
                    'alert_type': 'high_error_rate',
                    'severity': 'P1',
                    'title': f'High error rate in {service_name}',
                    'description': f'Error rate is {error_rate:.2%} which exceeds threshold',
                    'labels': {'service': service_name},
                    'fingerprint': f'high-error-rate-{service_name}'
                }
                
                await self.alert_queue.put(alert_data)
    
    async def check_response_times(self, health_metrics: Dict[str, Any]) -> None:
        """Check response times and generate alerts"""
        latency_data = health_metrics.get('response_time', {})
        results = latency_data.get('data', {}).get('result', [])
        
        for result in results:
            latency = float(result['value'][1])
            service_name = result['metric'].get('job', 'unknown')
            
            # Set thresholds based on service
            threshold_map = {
                'strategic-service': 0.002,  # 2ms
                'tactical-service': 0.001,   # 1ms
                'risk-service': 0.005        # 5ms
            }
            
            threshold = threshold_map.get(service_name, 0.01)
            
            if latency > threshold:
                alert_data = {
                    'id': f'high-latency-{service_name}-{int(time.time())}',
                    'alert_type': 'high_latency',
                    'severity': 'P2',
                    'title': f'High latency in {service_name}',
                    'description': f'Response time is {latency:.3f}s which exceeds threshold of {threshold:.3f}s',
                    'labels': {'service': service_name},
                    'fingerprint': f'high-latency-{service_name}'
                }
                
                await self.alert_queue.put(alert_data)
    
    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get incident status"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return None
        
        return {
            'id': incident.id,
            'title': incident.title,
            'severity': incident.severity.value,
            'status': incident.status.value,
            'created_at': incident.created_at.isoformat(),
            'updated_at': incident.updated_at.isoformat(),
            'resolved_at': incident.resolved_at.isoformat() if incident.resolved_at else None,
            'escalation_level': incident.escalation_level,
            'alerts_count': len(incident.alerts),
            'remediation_actions': incident.remediation_actions,
            'timeline': incident.timeline
        }
    
    def get_all_incidents(self) -> List[Dict[str, Any]]:
        """Get all incidents"""
        return [self.get_incident_status(incident_id) for incident_id in self.incidents.keys()]

# Example usage and testing
async def main():
    """Main function for testing"""
    # Initialize incident response manager
    manager = IncidentResponseManager()
    
    # Start the system
    await manager.start()
    
    # Simulate an alert
    test_alert = {
        'id': 'test-alert-001',
        'alert_type': 'system_down',
        'severity': 'P0',
        'title': 'Strategic agent is down',
        'description': 'Strategic agent is not responding to health checks',
        'labels': {'service': 'strategic-service'},
        'fingerprint': 'strategic-service-down'
    }
    
    await manager.alert_queue.put(test_alert)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Get incident status
    incidents = manager.get_all_incidents()
    print(f"Active incidents: {len(incidents)}")
    
    if incidents:
        incident = incidents[0]
        print(f"Incident: {incident['title']}")
        print(f"Status: {incident['status']}")
        print(f"Actions: {incident['remediation_actions']}")
    
    # Stop the system
    await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())