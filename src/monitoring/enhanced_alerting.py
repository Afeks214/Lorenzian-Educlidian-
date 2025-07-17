#!/usr/bin/env python3
"""
AGENT 13: Enhanced Alerting System with Intelligent Correlation and Escalation
Multi-channel alerting with intelligent alert correlation, suppression, and escalation policies
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import hashlib
import re
from statistics import mean, stdev
import aiohttp
import smtplib
import redis
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced alerting metrics
ALERT_CORRELATIONS = Counter(
    'alert_correlations_total',
    'Total number of alert correlations detected',
    ['correlation_type', 'severity']
)

ALERT_SUPPRESSIONS = Counter(
    'alert_suppressions_total',
    'Total number of alert suppressions',
    ['suppression_type', 'reason']
)

ALERT_ESCALATIONS = Counter(
    'alert_escalations_total',
    'Total number of alert escalations',
    ['escalation_level', 'reason']
)

ALERT_RESPONSE_TIME = Histogram(
    'alert_response_time_seconds',
    'Alert response time in seconds',
    ['channel', 'priority'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, float('inf')]
)

ALERT_DELIVERY_SUCCESS = Counter(
    'alert_delivery_success_total',
    'Total successful alert deliveries',
    ['channel', 'priority']
)

ALERT_DELIVERY_FAILURES = Counter(
    'alert_delivery_failures_total',
    'Total failed alert deliveries',
    ['channel', 'error_type']
)

class AlertPriority(Enum):
    """Alert priority levels for intelligent routing."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class AlertStatus(Enum):
    """Alert status tracking."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"

class CorrelationType(Enum):
    """Types of alert correlation."""
    DUPLICATE = "duplicate"
    RELATED = "related"
    CASCADE = "cascade"
    STORM = "storm"

class EscalationLevel(Enum):
    """Escalation levels."""
    LEVEL_1 = "level_1"  # Team leads
    LEVEL_2 = "level_2"  # Managers
    LEVEL_3 = "level_3"  # Directors
    LEVEL_4 = "level_4"  # Executives

@dataclass
class EnhancedAlert:
    """Enhanced alert with correlation and escalation info."""
    id: str
    timestamp: datetime
    priority: AlertPriority
    status: AlertStatus
    source: str
    alert_type: str
    title: str
    message: str
    metrics: Dict[str, Any]
    tags: Set[str] = field(default_factory=set)
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    related_alerts: List[str] = field(default_factory=list)
    suppression_rules: List[str] = field(default_factory=list)
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    acknowledgment_info: Optional[Dict[str, Any]] = None
    resolution_info: Optional[Dict[str, Any]] = None
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()
            
    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for alert deduplication."""
        content = f"{self.source}:{self.alert_type}:{self.title}"
        # Include relevant metrics for fingerprinting
        if 'threshold_breached' in self.metrics:
            content += f":{self.metrics['threshold_breached'].get('metric', '')}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'status': self.status.value,
            'source': self.source,
            'alert_type': self.alert_type,
            'title': self.title,
            'message': self.message,
            'metrics': self.metrics,
            'tags': list(self.tags),
            'correlation_id': self.correlation_id,
            'parent_alert_id': self.parent_alert_id,
            'related_alerts': self.related_alerts,
            'suppression_rules': self.suppression_rules,
            'escalation_history': self.escalation_history,
            'acknowledgment_info': self.acknowledgment_info,
            'resolution_info': self.resolution_info,
            'fingerprint': self.fingerprint
        }

@dataclass
class AlertCorrelationRule:
    """Rule for correlating related alerts."""
    name: str
    correlation_type: CorrelationType
    time_window: timedelta
    conditions: List[str]  # Python expressions
    action: str  # suppress, group, escalate
    threshold: Optional[int] = None  # For storm detection
    
@dataclass
class AlertSuppressionRule:
    """Rule for suppressing alerts."""
    name: str
    conditions: List[str]  # Python expressions
    duration: timedelta
    reason: str
    exceptions: List[str] = field(default_factory=list)
    
@dataclass
class EscalationPolicy:
    """Escalation policy configuration."""
    name: str
    triggers: List[str]  # Conditions that trigger escalation
    levels: List[Dict[str, Any]]  # Escalation levels with contacts and delays
    repeat_interval: timedelta = timedelta(hours=1)
    max_escalations: int = 3

class AlertCorrelationEngine:
    """Engine for correlating related alerts."""
    
    def __init__(self):
        self.correlation_rules = []
        self.recent_alerts = deque(maxlen=1000)
        self.correlation_groups = {}
        self.storm_detection_window = timedelta(minutes=5)
        
    def add_correlation_rule(self, rule: AlertCorrelationRule):
        """Add correlation rule."""
        self.correlation_rules.append(rule)
        logger.info(f"Added correlation rule: {rule.name}")
        
    def correlate_alert(self, alert: EnhancedAlert) -> List[str]:
        """Correlate alert with existing alerts."""
        correlations = []
        
        # Check for duplicate alerts
        duplicates = self._find_duplicates(alert)
        if duplicates:
            correlations.extend(duplicates)
            ALERT_CORRELATIONS.labels(
                correlation_type=CorrelationType.DUPLICATE.value,
                severity=alert.priority.name.lower()
            ).inc()
            
        # Check for related alerts
        related = self._find_related_alerts(alert)
        if related:
            correlations.extend(related)
            ALERT_CORRELATIONS.labels(
                correlation_type=CorrelationType.RELATED.value,
                severity=alert.priority.name.lower()
            ).inc()
            
        # Check for cascade alerts
        cascade = self._find_cascade_alerts(alert)
        if cascade:
            correlations.extend(cascade)
            ALERT_CORRELATIONS.labels(
                correlation_type=CorrelationType.CASCADE.value,
                severity=alert.priority.name.lower()
            ).inc()
            
        # Check for alert storms
        if self._detect_alert_storm(alert):
            ALERT_CORRELATIONS.labels(
                correlation_type=CorrelationType.STORM.value,
                severity=alert.priority.name.lower()
            ).inc()
            
        # Store alert for future correlation
        self.recent_alerts.append(alert)
        
        return correlations
        
    def _find_duplicates(self, alert: EnhancedAlert) -> List[str]:
        """Find duplicate alerts based on fingerprint."""
        duplicates = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        
        for existing_alert in self.recent_alerts:
            if (existing_alert.fingerprint == alert.fingerprint and
                existing_alert.timestamp > cutoff_time and
                existing_alert.status == AlertStatus.ACTIVE):
                duplicates.append(existing_alert.id)
                
        return duplicates
        
    def _find_related_alerts(self, alert: EnhancedAlert) -> List[str]:
        """Find related alerts based on correlation rules."""
        related = []
        
        for rule in self.correlation_rules:
            if rule.correlation_type != CorrelationType.RELATED:
                continue
                
            cutoff_time = datetime.utcnow() - rule.time_window
            
            for existing_alert in self.recent_alerts:
                if (existing_alert.timestamp > cutoff_time and
                    existing_alert.status == AlertStatus.ACTIVE and
                    self._evaluate_correlation_conditions(rule, alert, existing_alert)):
                    related.append(existing_alert.id)
                    
        return related
        
    def _find_cascade_alerts(self, alert: EnhancedAlert) -> List[str]:
        """Find cascade alerts (alerts that typically follow others)."""
        cascade = []
        
        # Common cascade patterns
        cascade_patterns = {
            'high_latency': ['high_cpu', 'high_memory'],
            'correlation_shock': ['var_breach', 'margin_usage'],
            'system_down': ['network_error', 'database_error']
        }
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        
        for parent_type, child_types in cascade_patterns.items():
            if alert.alert_type in child_types:
                for existing_alert in self.recent_alerts:
                    if (existing_alert.alert_type == parent_type and
                        existing_alert.timestamp > cutoff_time and
                        existing_alert.status == AlertStatus.ACTIVE):
                        cascade.append(existing_alert.id)
                        
        return cascade
        
    def _detect_alert_storm(self, alert: EnhancedAlert) -> bool:
        """Detect if we're in an alert storm."""
        cutoff_time = datetime.utcnow() - self.storm_detection_window
        
        # Count recent alerts from same source
        recent_count = sum(1 for a in self.recent_alerts 
                          if a.source == alert.source and a.timestamp > cutoff_time)
        
        # Storm threshold: more than 10 alerts from same source in 5 minutes
        return recent_count > 10
        
    def _evaluate_correlation_conditions(self, rule: AlertCorrelationRule, 
                                       alert1: EnhancedAlert, alert2: EnhancedAlert) -> bool:
        """Evaluate correlation rule conditions."""
        try:
            context = {
                'alert1': alert1,
                'alert2': alert2,
                'alert1_type': alert1.alert_type,
                'alert2_type': alert2.alert_type,
                'alert1_source': alert1.source,
                'alert2_source': alert2.source,
                'alert1_priority': alert1.priority.value,
                'alert2_priority': alert2.priority.value
            }
            
            for condition in rule.conditions:
                # Simple condition evaluation (in production, use a safer evaluator)
                if not self._safe_eval(condition, context):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating correlation condition: {e}")
            return False
            
    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safe evaluation of expressions."""
        # This is a simplified version - in production, use a proper expression evaluator
        # For now, just check some basic patterns
        if 'alert1_source == alert2_source' in expression:
            return context['alert1_source'] == context['alert2_source']
        elif 'alert1_type' in expression and 'alert2_type' in expression:
            # Handle type comparisons
            return True  # Simplified for demo
        return False

class AlertSuppressionEngine:
    """Engine for suppressing alerts based on rules."""
    
    def __init__(self):
        self.suppression_rules = []
        self.active_suppressions = {}
        
    def add_suppression_rule(self, rule: AlertSuppressionRule):
        """Add suppression rule."""
        self.suppression_rules.append(rule)
        logger.info(f"Added suppression rule: {rule.name}")
        
    def should_suppress_alert(self, alert: EnhancedAlert) -> Tuple[bool, Optional[str]]:
        """Check if alert should be suppressed."""
        for rule in self.suppression_rules:
            if self._evaluate_suppression_conditions(rule, alert):
                # Check if suppression is still active
                suppression_key = f"{rule.name}:{alert.fingerprint}"
                if suppression_key in self.active_suppressions:
                    suppression_end = self.active_suppressions[suppression_key]
                    if datetime.utcnow() < suppression_end:
                        ALERT_SUPPRESSIONS.labels(
                            suppression_type=rule.name,
                            reason=rule.reason
                        ).inc()
                        return True, rule.reason
                    else:
                        # Suppression expired
                        del self.active_suppressions[suppression_key]
                        
                # Start new suppression
                self.active_suppressions[suppression_key] = datetime.utcnow() + rule.duration
                ALERT_SUPPRESSIONS.labels(
                    suppression_type=rule.name,
                    reason=rule.reason
                ).inc()
                return True, rule.reason
                
        return False, None
        
    def _evaluate_suppression_conditions(self, rule: AlertSuppressionRule, alert: EnhancedAlert) -> bool:
        """Evaluate suppression rule conditions."""
        try:
            context = {
                'alert': alert,
                'alert_type': alert.alert_type,
                'source': alert.source,
                'priority': alert.priority.value,
                'tags': alert.tags
            }
            
            for condition in rule.conditions:
                if not self._safe_eval_suppression(condition, context):
                    return False
                    
            # Check exceptions
            for exception in rule.exceptions:
                if self._safe_eval_suppression(exception, context):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating suppression condition: {e}")
            return False
            
    def _safe_eval_suppression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safe evaluation of suppression expressions."""
        # Simplified evaluation - in production, use a proper expression evaluator
        if 'alert_type ==' in expression:
            target_type = expression.split('==')[1].strip().strip('"\'')
            return context['alert_type'] == target_type
        elif 'source ==' in expression:
            target_source = expression.split('==')[1].strip().strip('"\'')
            return context['source'] == target_source
        elif 'priority <' in expression:
            target_priority = int(expression.split('<')[1].strip())
            return context['priority'] < target_priority
        return False

class EscalationEngine:
    """Engine for escalating alerts based on policies."""
    
    def __init__(self):
        self.escalation_policies = []
        self.active_escalations = {}
        
    def add_escalation_policy(self, policy: EscalationPolicy):
        """Add escalation policy."""
        self.escalation_policies.append(policy)
        logger.info(f"Added escalation policy: {policy.name}")
        
    async def check_escalation(self, alert: EnhancedAlert) -> bool:
        """Check if alert should be escalated."""
        for policy in self.escalation_policies:
            if self._should_escalate(policy, alert):
                await self._escalate_alert(policy, alert)
                return True
        return False
        
    def _should_escalate(self, policy: EscalationPolicy, alert: EnhancedAlert) -> bool:
        """Check if alert meets escalation criteria."""
        # Check if already escalated recently
        escalation_key = f"{policy.name}:{alert.id}"
        if escalation_key in self.active_escalations:
            last_escalation = self.active_escalations[escalation_key]
            if datetime.utcnow() - last_escalation < policy.repeat_interval:
                return False
                
        # Check escalation triggers
        for trigger in policy.triggers:
            if self._evaluate_escalation_trigger(trigger, alert):
                return True
                
        return False
        
    def _evaluate_escalation_trigger(self, trigger: str, alert: EnhancedAlert) -> bool:
        """Evaluate escalation trigger."""
        # Common escalation triggers
        if trigger == 'unacknowledged_critical':
            return (alert.priority == AlertPriority.CRITICAL and
                   alert.status == AlertStatus.ACTIVE and
                   alert.acknowledgment_info is None)
        elif trigger == 'emergency_priority':
            return alert.priority == AlertPriority.EMERGENCY
        elif trigger == 'prolonged_unresolved':
            return (datetime.utcnow() - alert.timestamp > timedelta(hours=1) and
                   alert.status == AlertStatus.ACTIVE)
        return False
        
    async def _escalate_alert(self, policy: EscalationPolicy, alert: EnhancedAlert):
        """Escalate alert through policy levels."""
        escalation_key = f"{policy.name}:{alert.id}"
        
        # Determine current escalation level
        current_level = len(alert.escalation_history)
        if current_level >= len(policy.levels):
            current_level = len(policy.levels) - 1
            
        level_config = policy.levels[current_level]
        
        # Record escalation
        escalation_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': current_level + 1,
            'policy': policy.name,
            'contacts': level_config.get('contacts', []),
            'reason': 'Automatic escalation due to policy triggers'
        }
        
        alert.escalation_history.append(escalation_record)
        self.active_escalations[escalation_key] = datetime.utcnow()
        
        # Record metrics
        ALERT_ESCALATIONS.labels(
            escalation_level=f"level_{current_level + 1}",
            reason="policy_trigger"
        ).inc()
        
        logger.warning(f"Escalated alert {alert.id} to level {current_level + 1}")

class EnhancedAlertingSystem:
    """Main enhanced alerting system."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.correlation_engine = AlertCorrelationEngine()
        self.suppression_engine = AlertSuppressionEngine()
        self.escalation_engine = EscalationEngine()
        self.active_alerts = {}
        self.delivery_channels = {}
        
        # Initialize default rules
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default correlation, suppression, and escalation rules."""
        # Default correlation rules
        self.correlation_engine.add_correlation_rule(AlertCorrelationRule(
            name="same_source_related",
            correlation_type=CorrelationType.RELATED,
            time_window=timedelta(minutes=10),
            conditions=["alert1_source == alert2_source"],
            action="group"
        ))
        
        # Default suppression rules
        self.suppression_engine.add_suppression_rule(AlertSuppressionRule(
            name="low_priority_during_maintenance",
            conditions=["priority < 3"],
            duration=timedelta(hours=1),
            reason="Maintenance window active"
        ))
        
        # Default escalation policy
        self.escalation_engine.add_escalation_policy(EscalationPolicy(
            name="critical_alert_escalation",
            triggers=["unacknowledged_critical", "emergency_priority"],
            levels=[
                {
                    "contacts": ["team-lead@company.com"],
                    "delay": timedelta(minutes=5)
                },
                {
                    "contacts": ["manager@company.com"],
                    "delay": timedelta(minutes=15)
                },
                {
                    "contacts": ["director@company.com"],
                    "delay": timedelta(minutes=30)
                }
            ]
        ))
        
    def add_delivery_channel(self, name: str, channel: Any):
        """Add alert delivery channel."""
        self.delivery_channels[name] = channel
        
    async def process_alert(self, alert: EnhancedAlert) -> bool:
        """Process alert through correlation, suppression, and escalation."""
        start_time = time.time()
        
        try:
            # Check for suppression
            should_suppress, suppression_reason = self.suppression_engine.should_suppress_alert(alert)
            if should_suppress:
                alert.status = AlertStatus.SUPPRESSED
                alert.suppression_rules.append(suppression_reason)
                logger.info(f"Alert {alert.id} suppressed: {suppression_reason}")
                return False
                
            # Correlate with existing alerts
            correlations = self.correlation_engine.correlate_alert(alert)
            if correlations:
                alert.related_alerts = correlations
                # Generate correlation ID
                alert.correlation_id = hashlib.md5(
                    f"{alert.id}:{''.join(sorted(correlations))}".encode()
                ).hexdigest()[:8]
                
            # Store active alert
            self.active_alerts[alert.id] = alert
            
            # Store in Redis
            await self.redis_client.setex(
                f"alert:{alert.id}",
                3600,  # 1 hour TTL
                json.dumps(alert.to_dict())
            )
            
            # Check for escalation
            await self.escalation_engine.check_escalation(alert)
            
            # Send alert through appropriate channels
            await self._deliver_alert(alert)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.id}: {e}")
            return False
        finally:
            # Record processing time
            processing_time = time.time() - start_time
            ALERT_RESPONSE_TIME.labels(
                channel="system",
                priority=alert.priority.name.lower()
            ).observe(processing_time)
            
    async def _deliver_alert(self, alert: EnhancedAlert):
        """Deliver alert through appropriate channels based on priority."""
        channels = self._get_delivery_channels(alert)
        
        for channel_name in channels:
            if channel_name in self.delivery_channels:
                try:
                    await self.delivery_channels[channel_name].send_alert(alert)
                    ALERT_DELIVERY_SUCCESS.labels(
                        channel=channel_name,
                        priority=alert.priority.name.lower()
                    ).inc()
                except Exception as e:
                    ALERT_DELIVERY_FAILURES.labels(
                        channel=channel_name,
                        error_type=type(e).__name__
                    ).inc()
                    logger.error(f"Failed to deliver alert via {channel_name}: {e}")
                    
    def _get_delivery_channels(self, alert: EnhancedAlert) -> List[str]:
        """Get appropriate delivery channels based on alert priority."""
        if alert.priority == AlertPriority.EMERGENCY:
            return ["email", "slack", "sms", "webhook"]
        elif alert.priority == AlertPriority.CRITICAL:
            return ["email", "slack", "webhook"]
        elif alert.priority == AlertPriority.HIGH:
            return ["slack", "webhook"]
        elif alert.priority == AlertPriority.MEDIUM:
            return ["slack"]
        else:
            return ["webhook"]
            
    async def acknowledge_alert(self, alert_id: str, user: str, comment: str = ""):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledgment_info = {
                'user': user,
                'timestamp': datetime.utcnow().isoformat(),
                'comment': comment
            }
            
            # Update in Redis
            await self.redis_client.setex(
                f"alert:{alert_id}",
                3600,
                json.dumps(alert.to_dict())
            )
            
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            
    async def resolve_alert(self, alert_id: str, user: str, resolution: str = ""):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolution_info = {
                'user': user,
                'timestamp': datetime.utcnow().isoformat(),
                'resolution': resolution
            }
            
            # Update in Redis
            await self.redis_client.setex(
                f"alert:{alert_id}",
                3600,
                json.dumps(alert.to_dict())
            )
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved by {user}")
            
    async def get_alert_status(self) -> Dict[str, Any]:
        """Get current alerting system status."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_alerts': len(self.active_alerts),
            'alert_distribution': {
                priority.name: sum(1 for a in self.active_alerts.values() if a.priority == priority)
                for priority in AlertPriority
            },
            'active_suppressions': len(self.suppression_engine.active_suppressions),
            'active_escalations': len(self.escalation_engine.active_escalations),
            'correlation_groups': len(self.correlation_engine.correlation_groups)
        }

# Example usage and configuration
if __name__ == "__main__":
    # Example of enhanced alerting system
    import asyncio
    
    async def main():
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        alerting_system = EnhancedAlertingSystem(redis_client)
        
        # Create test alert
        alert = EnhancedAlert(
            id="test_alert_001",
            timestamp=datetime.utcnow(),
            priority=AlertPriority.CRITICAL,
            status=AlertStatus.ACTIVE,
            source="trading_engine",
            alert_type="high_latency",
            title="Trading Engine High Latency",
            message="Trading engine response time exceeds 20ms",
            metrics={
                'response_time': 25.5,
                'threshold': 20.0,
                'service': 'trading_engine'
            },
            tags={"service", "performance", "critical"}
        )
        
        # Process alert
        success = await alerting_system.process_alert(alert)
        print(f"Alert processed: {success}")
        
        # Get status
        status = await alerting_system.get_alert_status()
        print(f"Alerting system status: {json.dumps(status, indent=2)}")
        
        # Acknowledge alert
        await alerting_system.acknowledge_alert("test_alert_001", "john.doe", "Investigating latency issue")
        
        # Resolve alert
        await alerting_system.resolve_alert("test_alert_001", "john.doe", "Fixed database connection pool")
        
    asyncio.run(main())
