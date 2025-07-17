#!/usr/bin/env python3
"""
PagerDuty Integration for GrandModel MARL Trading System
High-performance critical alerting with <30 second response times
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from prometheus_client import Counter, Histogram, Gauge
import hashlib
import uuid

# PagerDuty metrics
PAGERDUTY_EVENTS_SENT = Counter('pagerduty_events_sent_total', 'Total PagerDuty events sent', ['event_type', 'severity'])
PAGERDUTY_LATENCY = Histogram('pagerduty_event_latency_seconds', 'PagerDuty event processing latency')
PAGERDUTY_INCIDENTS = Gauge('pagerduty_active_incidents', 'Currently active PagerDuty incidents', ['service'])
PAGERDUTY_ERRORS = Counter('pagerduty_errors_total', 'PagerDuty integration errors', ['error_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PagerDutyEventType(Enum):
    """PagerDuty event types."""
    TRIGGER = "trigger"
    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"

class PagerDutySeverity(Enum):
    """PagerDuty severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PagerDutyEvent:
    """PagerDuty event structure."""
    routing_key: str
    event_action: PagerDutyEventType
    dedup_key: str
    payload: Dict[str, Any]
    client: Optional[str] = "GrandModel-MARL"
    client_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to PagerDuty API format."""
        return {
            "routing_key": self.routing_key,
            "event_action": self.event_action.value,
            "dedup_key": self.dedup_key,
            "payload": self.payload,
            "client": self.client,
            "client_url": self.client_url
        }

@dataclass
class PagerDutyService:
    """PagerDuty service configuration."""
    name: str
    routing_key: str
    escalation_policy: str
    auto_resolve_timeout: int = 14400  # 4 hours
    acknowledgement_timeout: int = 1800  # 30 minutes

class PagerDutyClient:
    """High-performance PagerDuty client."""
    
    def __init__(self, api_token: str, default_routing_key: str):
        self.api_token = api_token
        self.default_routing_key = default_routing_key
        self.events_api_url = "https://events.pagerduty.com/v2/enqueue"
        self.rest_api_url = "https://api.pagerduty.com"
        self.session = None
        self.dedup_cache = {}  # Track dedup keys for incident management
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                "Authorization": f"Token token={self.api_token}",
                "Content-Type": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def send_event(self, event: PagerDutyEvent) -> Dict[str, Any]:
        """Send event to PagerDuty with high performance."""
        start_time = time.time()
        
        try:
            payload = event.to_dict()
            
            async with self.session.post(
                self.events_api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 202:
                    result = await response.json()
                    
                    # Update metrics
                    PAGERDUTY_EVENTS_SENT.labels(
                        event_type=event.event_action.value,
                        severity=event.payload.get('severity', 'unknown')
                    ).inc()
                    
                    # Cache dedup key for incident management
                    if event.event_action == PagerDutyEventType.TRIGGER:
                        self.dedup_cache[event.dedup_key] = {
                            'timestamp': datetime.utcnow(),
                            'service': event.payload.get('source', 'unknown'),
                            'incident_key': result.get('incident_key')
                        }
                    elif event.event_action == PagerDutyEventType.RESOLVE:
                        self.dedup_cache.pop(event.dedup_key, None)
                    
                    logger.info(f"PagerDuty event sent successfully: {event.dedup_key}")
                    return result
                    
                else:
                    error_text = await response.text()
                    logger.error(f"PagerDuty API error {response.status}: {error_text}")
                    PAGERDUTY_ERRORS.labels(error_type=f"api_error_{response.status}").inc()
                    return {"error": f"API error {response.status}"}
                    
        except asyncio.TimeoutError:
            logger.error(f"PagerDuty API timeout for event: {event.dedup_key}")
            PAGERDUTY_ERRORS.labels(error_type="timeout").inc()
            return {"error": "timeout"}
            
        except Exception as e:
            logger.error(f"PagerDuty event failed: {e}")
            PAGERDUTY_ERRORS.labels(error_type="exception").inc()
            return {"error": str(e)}
            
        finally:
            PAGERDUTY_LATENCY.observe(time.time() - start_time)
            
    async def get_incidents(self, service_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Get active incidents from PagerDuty."""
        try:
            params = {
                "statuses[]": ["triggered", "acknowledged"],
                "limit": 100
            }
            
            if service_ids:
                params["service_ids[]"] = service_ids
            
            async with self.session.get(
                f"{self.rest_api_url}/incidents",
                params=params
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("incidents", [])
                else:
                    logger.error(f"Failed to get incidents: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting incidents: {e}")
            return []

class GrandModelPagerDutyIntegration:
    """Comprehensive PagerDuty integration for GrandModel MARL system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = PagerDutyClient(
            api_token=config["api_token"],
            default_routing_key=config["default_routing_key"]
        )
        
        # Service configurations
        self.services = {
            "trading_engine": PagerDutyService(
                name="Trading Engine",
                routing_key=config["services"]["trading_engine"]["routing_key"],
                escalation_policy="trading_critical",
                auto_resolve_timeout=7200,  # 2 hours
                acknowledgement_timeout=300  # 5 minutes
            ),
            "strategic_agent": PagerDutyService(
                name="Strategic MARL Agent",
                routing_key=config["services"]["strategic_agent"]["routing_key"],
                escalation_policy="marl_agents",
                auto_resolve_timeout=3600,  # 1 hour
                acknowledgement_timeout=600  # 10 minutes
            ),
            "tactical_agent": PagerDutyService(
                name="Tactical MARL Agent",
                routing_key=config["services"]["tactical_agent"]["routing_key"],
                escalation_policy="marl_agents",
                auto_resolve_timeout=3600,  # 1 hour
                acknowledgement_timeout=600  # 10 minutes
            ),
            "risk_agent": PagerDutyService(
                name="Risk Management Agent",
                routing_key=config["services"]["risk_agent"]["routing_key"],
                escalation_policy="risk_critical",
                auto_resolve_timeout=1800,  # 30 minutes
                acknowledgement_timeout=180  # 3 minutes
            ),
            "data_pipeline": PagerDutyService(
                name="Data Pipeline",
                routing_key=config["services"]["data_pipeline"]["routing_key"],
                escalation_policy="infrastructure",
                auto_resolve_timeout=7200,  # 2 hours
                acknowledgement_timeout=900  # 15 minutes
            ),
            "system_health": PagerDutyService(
                name="System Health",
                routing_key=config["services"]["system_health"]["routing_key"],
                escalation_policy="infrastructure",
                auto_resolve_timeout=14400,  # 4 hours
                acknowledgement_timeout=1800  # 30 minutes
            )
        }
        
        # Alert routing rules
        self.alert_routing_rules = self._create_alert_routing_rules()
        
        # Incident management
        self.incident_cache = {}
        self.escalation_timers = {}
        
    def _create_alert_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Create alert routing rules for different alert types."""
        return {
            # Trading System Alerts
            "trading_engine_down": {
                "service": "trading_engine",
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "trading_critical",
                "auto_escalate_seconds": 60
            },
            "execution_failure_rate": {
                "service": "trading_engine",
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "trading_critical",
                "auto_escalate_seconds": 120
            },
            "high_slippage": {
                "service": "trading_engine",
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "trading_normal",
                "auto_escalate_seconds": 900
            },
            
            # MARL Agent Alerts
            "agent_high_latency": {
                "service": "strategic_agent",  # Will be mapped based on agent_type
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "marl_agents",
                "auto_escalate_seconds": 300
            },
            "agent_critical_latency": {
                "service": "strategic_agent",  # Will be mapped based on agent_type
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "marl_agents",
                "auto_escalate_seconds": 120
            },
            "agent_low_accuracy": {
                "service": "strategic_agent",  # Will be mapped based on agent_type
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "marl_agents",
                "auto_escalate_seconds": 600
            },
            "agent_critical_accuracy": {
                "service": "strategic_agent",  # Will be mapped based on agent_type
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "marl_agents",
                "auto_escalate_seconds": 180
            },
            
            # Risk Management Alerts
            "risk_var_breach": {
                "service": "risk_agent",
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "risk_critical",
                "auto_escalate_seconds": 60
            },
            "correlation_shock": {
                "service": "risk_agent",
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "risk_critical",
                "auto_escalate_seconds": 30
            },
            "high_margin_usage": {
                "service": "risk_agent",
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "risk_normal",
                "auto_escalate_seconds": 300
            },
            
            # System Health Alerts
            "high_cpu_usage": {
                "service": "system_health",
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "infrastructure",
                "auto_escalate_seconds": 600
            },
            "high_memory_usage": {
                "service": "system_health",
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "infrastructure",
                "auto_escalate_seconds": 300
            },
            "system_availability": {
                "service": "system_health",
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "infrastructure",
                "auto_escalate_seconds": 120
            },
            
            # Data Pipeline Alerts
            "data_pipeline_latency": {
                "service": "data_pipeline",
                "severity": PagerDutySeverity.WARNING,
                "escalation_policy": "infrastructure",
                "auto_escalate_seconds": 300
            },
            "data_pipeline_throughput": {
                "service": "data_pipeline",
                "severity": PagerDutySeverity.CRITICAL,
                "escalation_policy": "infrastructure",
                "auto_escalate_seconds": 180
            }
        }
    
    def _generate_dedup_key(self, alert_type: str, source: str, labels: Dict[str, str] = None) -> str:
        """Generate consistent deduplication key for alerts."""
        key_parts = [alert_type, source]
        
        if labels:
            # Add relevant labels for deduplication
            relevant_labels = ['agent_type', 'instance', 'service']
            for label in relevant_labels:
                if label in labels:
                    key_parts.append(f"{label}={labels[label]}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _map_service_from_alert(self, alert_type: str, labels: Dict[str, str] = None) -> str:
        """Map alert to appropriate service based on type and labels."""
        # Agent-specific mapping
        if alert_type.startswith("agent_") and labels and "agent_type" in labels:
            agent_type = labels["agent_type"]
            if agent_type in ["strategic", "tactical", "risk"]:
                return f"{agent_type}_agent"
        
        # Use routing rules
        rule = self.alert_routing_rules.get(alert_type, {})
        return rule.get("service", "system_health")
    
    async def send_alert(self, alert_type: str, title: str, message: str, 
                        severity: PagerDutySeverity, source: str, 
                        labels: Dict[str, str] = None, 
                        custom_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send alert to PagerDuty with intelligent routing."""
        
        # Determine service
        service_name = self._map_service_from_alert(alert_type, labels)
        service = self.services.get(service_name, self.services["system_health"])
        
        # Generate dedup key
        dedup_key = self._generate_dedup_key(alert_type, source, labels)
        
        # Get routing rule
        routing_rule = self.alert_routing_rules.get(alert_type, {})
        
        # Create payload
        payload = {
            "summary": title,
            "severity": severity.value,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "component": service_name,
            "group": "grandmodel_marl",
            "class": alert_type,
            "custom_details": {
                "alert_type": alert_type,
                "labels": labels or {},
                "message": message,
                "escalation_policy": routing_rule.get("escalation_policy", "default"),
                "auto_escalate_seconds": routing_rule.get("auto_escalate_seconds", 300),
                **(custom_details or {})
            }
        }
        
        # Create PagerDuty event
        event = PagerDutyEvent(
            routing_key=service.routing_key,
            event_action=PagerDutyEventType.TRIGGER,
            dedup_key=dedup_key,
            payload=payload,
            client="GrandModel-MARL",
            client_url=f"https://monitoring.grandmodel.ai/alerts/{dedup_key}"
        )
        
        # Send event
        async with self.client as client:
            result = await client.send_event(event)
            
            # Schedule auto-escalation if configured
            if routing_rule.get("auto_escalate_seconds"):
                self._schedule_auto_escalation(
                    dedup_key, 
                    routing_rule["auto_escalate_seconds"],
                    service_name
                )
            
            return result
    
    async def resolve_alert(self, alert_type: str, source: str, 
                           labels: Dict[str, str] = None, 
                           resolution_note: str = None) -> Dict[str, Any]:
        """Resolve alert in PagerDuty."""
        
        # Determine service
        service_name = self._map_service_from_alert(alert_type, labels)
        service = self.services.get(service_name, self.services["system_health"])
        
        # Generate dedup key
        dedup_key = self._generate_dedup_key(alert_type, source, labels)
        
        # Create resolve payload
        payload = {
            "summary": f"Resolved: {alert_type}",
            "severity": "info",
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "custom_details": {
                "resolution_note": resolution_note or "Alert condition resolved",
                "resolved_by": "GrandModel Monitoring System"
            }
        }
        
        # Create PagerDuty event
        event = PagerDutyEvent(
            routing_key=service.routing_key,
            event_action=PagerDutyEventType.RESOLVE,
            dedup_key=dedup_key,
            payload=payload
        )
        
        # Send event
        async with self.client as client:
            result = await client.send_event(event)
            
            # Cancel auto-escalation timer
            if dedup_key in self.escalation_timers:
                self.escalation_timers[dedup_key].cancel()
                del self.escalation_timers[dedup_key]
            
            return result
    
    def _schedule_auto_escalation(self, dedup_key: str, delay_seconds: int, service_name: str):
        """Schedule automatic escalation for unacknowledged alerts."""
        async def escalate():
            await asyncio.sleep(delay_seconds)
            
            # Check if incident is still active and unacknowledged
            # This would typically check with PagerDuty API
            logger.warning(f"Auto-escalating alert {dedup_key} for service {service_name}")
            
            # Send escalation notification
            await self._send_escalation_notification(dedup_key, service_name)
        
        # Schedule the escalation
        task = asyncio.create_task(escalate())
        self.escalation_timers[dedup_key] = task
    
    async def _send_escalation_notification(self, dedup_key: str, service_name: str):
        """Send escalation notification for unacknowledged alerts."""
        # This would typically send additional notifications
        # or trigger higher-level escalation policies
        logger.warning(f"Escalating unacknowledged alert {dedup_key} for {service_name}")
    
    async def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get active incidents from PagerDuty."""
        async with self.client as client:
            return await client.get_incidents()
    
    async def acknowledge_alert(self, alert_type: str, source: str, 
                              labels: Dict[str, str] = None,
                              acknowledgement_note: str = None) -> Dict[str, Any]:
        """Acknowledge alert in PagerDuty."""
        
        # Determine service
        service_name = self._map_service_from_alert(alert_type, labels)
        service = self.services.get(service_name, self.services["system_health"])
        
        # Generate dedup key
        dedup_key = self._generate_dedup_key(alert_type, source, labels)
        
        # Create acknowledge payload
        payload = {
            "summary": f"Acknowledged: {alert_type}",
            "severity": "info",
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "custom_details": {
                "acknowledgement_note": acknowledgement_note or "Alert acknowledged by operator",
                "acknowledged_by": "GrandModel Operations Team"
            }
        }
        
        # Create PagerDuty event
        event = PagerDutyEvent(
            routing_key=service.routing_key,
            event_action=PagerDutyEventType.ACKNOWLEDGE,
            dedup_key=dedup_key,
            payload=payload
        )
        
        # Send event
        async with self.client as client:
            return await client.send_event(event)

# Factory function for creating PagerDuty integration
def create_pagerduty_integration(config: Dict[str, Any]) -> GrandModelPagerDutyIntegration:
    """Create PagerDuty integration instance."""
    return GrandModelPagerDutyIntegration(config)

# Example configuration
EXAMPLE_CONFIG = {
    "api_token": "your-pagerduty-api-token",
    "default_routing_key": "your-default-routing-key",
    "services": {
        "trading_engine": {
            "routing_key": "trading-engine-routing-key"
        },
        "strategic_agent": {
            "routing_key": "strategic-agent-routing-key"
        },
        "tactical_agent": {
            "routing_key": "tactical-agent-routing-key"
        },
        "risk_agent": {
            "routing_key": "risk-agent-routing-key"
        },
        "data_pipeline": {
            "routing_key": "data-pipeline-routing-key"
        },
        "system_health": {
            "routing_key": "system-health-routing-key"
        }
    }
}

# Example usage
async def main():
    """Example usage of PagerDuty integration."""
    config = EXAMPLE_CONFIG
    pd_integration = create_pagerduty_integration(config)
    
    # Send critical alert
    await pd_integration.send_alert(
        alert_type="trading_engine_down",
        title="Trading Engine Critical Failure",
        message="Trading engine has been down for 60 seconds",
        severity=PagerDutySeverity.CRITICAL,
        source="trading_monitoring",
        labels={"service": "execution_engine"},
        custom_details={"downtime_seconds": 60}
    )
    
    # Resolve alert
    await pd_integration.resolve_alert(
        alert_type="trading_engine_down",
        source="trading_monitoring",
        labels={"service": "execution_engine"},
        resolution_note="Trading engine restarted and operational"
    )

if __name__ == "__main__":
    asyncio.run(main())