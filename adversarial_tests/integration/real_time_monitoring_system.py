"""
Real-time Monitoring and Feedback System for Adversarial-VaR Integration
========================================================================

This module implements comprehensive real-time monitoring and feedback loops
between the VaR system, attack detection, and Byzantine fault tolerance systems.

Key Features:
- Real-time performance monitoring and alerting
- Automated feedback loops between systems
- Adaptive threshold adjustment based on attack patterns
- Performance degradation detection and mitigation
- Comprehensive logging and metrics collection
- WebSocket-based real-time dashboard support

Author: Agent Beta Mission - Real-time Integration
Version: 1.0.0
Classification: CRITICAL SYSTEM INTEGRATION
"""

import asyncio
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import psutil
import websockets
import queue
import structlog

# Import integration components
from adversarial_tests.integration.adversarial_var_integration import AdversarialVaRIntegration
from adversarial_tests.integration.enhanced_byzantine_detection import EnhancedByzantineDetector
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationShock
from src.risk.core.var_calculator import VaRCalculator, VaRResult
from src.security.attack_detection import TacticalMARLAttackDetector
from src.core.events import EventBus, EventType, Event

logger = structlog.get_logger()


class MonitoringLevel(Enum):
    """Monitoring alert levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FeedbackAction(Enum):
    """Types of feedback actions"""
    ADJUST_THRESHOLD = "ADJUST_THRESHOLD"
    INCREASE_MONITORING = "INCREASE_MONITORING"
    TRIGGER_DEFENSIVE_MODE = "TRIGGER_DEFENSIVE_MODE"
    QUARANTINE_NODES = "QUARANTINE_NODES"
    RECALCULATE_VAR = "RECALCULATE_VAR"
    RESET_CORRELATIONS = "RESET_CORRELATIONS"
    ESCALATE_SECURITY = "ESCALATE_SECURITY"


@dataclass
class MonitoringAlert:
    """Monitoring alert structure"""
    alert_id: str
    timestamp: datetime
    level: MonitoringLevel
    source: str
    message: str
    metrics: Dict[str, Any]
    suggested_actions: List[str]
    auto_resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class FeedbackEvent:
    """Feedback event structure"""
    event_id: str
    timestamp: datetime
    action: FeedbackAction
    source_system: str
    target_system: str
    parameters: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    execution_time: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    var_calculation_time: float
    attack_detection_time: float
    byzantine_consensus_time: float
    correlation_update_time: float
    active_alerts: int
    threat_level: str
    system_health: str


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    time_window_seconds: int
    breach_count_threshold: int


class RealTimeMonitoringSystem:
    """
    Comprehensive real-time monitoring system for adversarial-VaR integration.
    """
    
    def __init__(
        self,
        adversarial_integration: AdversarialVaRIntegration,
        byzantine_detector: EnhancedByzantineDetector,
        event_bus: EventBus,
        websocket_port: int = 8765,
        monitoring_interval: float = 0.1,
        alert_history_size: int = 1000
    ):
        self.adversarial_integration = adversarial_integration
        self.byzantine_detector = byzantine_detector
        self.event_bus = event_bus
        self.websocket_port = websocket_port
        self.monitoring_interval = monitoring_interval
        self.alert_history_size = alert_history_size
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.websocket_server = None
        self.connected_clients = set()
        
        # Metrics collection
        self.metrics_history: deque = deque(maxlen=10000)
        self.alerts_history: deque = deque(maxlen=alert_history_size)
        self.feedback_events: deque = deque(maxlen=1000)
        
        # Performance thresholds
        self.performance_thresholds = {
            'var_calculation_time': PerformanceThreshold(
                metric_name='var_calculation_time',
                warning_threshold=5.0,  # 5ms
                critical_threshold=10.0,  # 10ms
                time_window_seconds=60,
                breach_count_threshold=5
            ),
            'attack_detection_time': PerformanceThreshold(
                metric_name='attack_detection_time',
                warning_threshold=100.0,  # 100ms
                critical_threshold=500.0,  # 500ms
                time_window_seconds=60,
                breach_count_threshold=3
            ),
            'memory_usage': PerformanceThreshold(
                metric_name='memory_usage',
                warning_threshold=80.0,  # 80%
                critical_threshold=90.0,  # 90%
                time_window_seconds=30,
                breach_count_threshold=2
            ),
            'cpu_usage': PerformanceThreshold(
                metric_name='cpu_usage',
                warning_threshold=75.0,  # 75%
                critical_threshold=85.0,  # 85%
                time_window_seconds=30,
                breach_count_threshold=3
            )
        }
        
        # Adaptive parameters
        self.adaptive_parameters = {
            'correlation_shock_threshold': 0.5,
            'var_confidence_level': 0.95,
            'byzantine_trust_threshold': 0.3,
            'ml_detection_threshold': 0.75,
            'monitoring_frequency': 0.1
        }
        
        # Feedback loops
        self.feedback_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.auto_mitigation_enabled = True
        
        # Real-time state
        self.current_threat_level = "LOW"
        self.system_health = "HEALTHY"
        self.defensive_mode = False
        
        # Initialize event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("RealTimeMonitoringSystem initialized",
                   websocket_port=websocket_port,
                   monitoring_interval=monitoring_interval)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for monitoring"""
        # VaR system events
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
        
        # System events
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._handle_system_error)
        
        # Custom monitoring events
        try:
            self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_update)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
    
    async def start_monitoring(self):
        """Start comprehensive real-time monitoring"""
        logger.info("🚀 Starting Real-time Monitoring System")
        
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        # Start subsystem monitoring
        self.adversarial_integration.start_real_time_monitoring()
        self.byzantine_detector.start_real_time_monitoring()
        
        logger.info("✅ Real-time monitoring system started")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        logger.info("🛑 Stopping Real-time Monitoring System")
        
        self.monitoring_active = False
        
        # Stop monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Stop subsystem monitoring
        self.adversarial_integration.stop_real_time_monitoring()
        self.byzantine_detector.stop_real_time_monitoring()
        
        logger.info("✅ Real-time monitoring system stopped")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time dashboard"""
        async def handle_client(websocket, path):
            """Handle WebSocket client connections"""
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            self.connected_clients.add(websocket)
            
            try:
                # Send initial system state
                await self._send_initial_state(websocket)
                
                # Handle client messages
                async for message in websocket:
                    await self._handle_client_message(websocket, message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
            except Exception as e:
                logger.error(f"Error handling WebSocket client: {e}")
            finally:
                self.connected_clients.discard(websocket)
        
        # Start WebSocket server
        self.websocket_server = await websockets.serve(
            handle_client, 
            "localhost", 
            self.websocket_port
        )
        
        logger.info(f"WebSocket server started on port {self.websocket_port}")
    
    async def _send_initial_state(self, websocket):
        """Send initial system state to client"""
        initial_state = {
            "type": "initial_state",
            "timestamp": datetime.now().isoformat(),
            "system_health": self.system_health,
            "threat_level": self.current_threat_level,
            "defensive_mode": self.defensive_mode,
            "active_alerts": len([alert for alert in self.alerts_history if not alert.auto_resolved]),
            "performance_thresholds": {
                name: asdict(threshold) for name, threshold in self.performance_thresholds.items()
            },
            "adaptive_parameters": self.adaptive_parameters.copy()
        }
        
        await websocket.send(json.dumps(initial_state))
    
    async def _handle_client_message(self, websocket, message):
        """Handle client WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "get_metrics":
                await self._send_metrics_update(websocket)
            elif message_type == "get_alerts":
                await self._send_alerts_update(websocket)
            elif message_type == "adjust_threshold":
                await self._handle_threshold_adjustment(data)
            elif message_type == "trigger_action":
                await self._handle_manual_action(data)
            
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received from WebSocket client")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check performance thresholds
                self._check_performance_thresholds(metrics)
                
                # Update threat level
                self._update_threat_level(metrics)
                
                # Execute feedback loops
                self._execute_feedback_loops(metrics)
                
                # Broadcast updates to clients
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_metrics_update(metrics),
                    asyncio.get_event_loop()
                )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        current_time = datetime.now()
        
        # System resource metrics
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        
        # VaR system metrics
        var_performance = self.adversarial_integration.var_calculator.get_performance_stats()
        var_calc_time = var_performance.get('avg_calc_time_ms', 0.0)
        
        # Correlation tracker metrics
        correlation_performance = self.adversarial_integration.correlation_tracker.get_performance_stats()
        correlation_time = correlation_performance.get('avg_calc_time_ms', 0.0)
        
        # Attack detection metrics
        attack_detection_time = 0.0
        if self.adversarial_integration.performance_metrics['attack_detection_times']:
            recent_times = list(self.adversarial_integration.performance_metrics['attack_detection_times'])
            attack_detection_time = np.mean(recent_times[-10:]) if recent_times else 0.0
        
        # Byzantine consensus metrics
        byzantine_time = 0.0
        if self.adversarial_integration.performance_metrics['consensus_times']:
            recent_times = list(self.adversarial_integration.performance_metrics['consensus_times'])
            byzantine_time = np.mean(recent_times[-10:]) if recent_times else 0.0
        
        # Alert metrics
        active_alerts = len([alert for alert in self.alerts_history if not alert.auto_resolved])
        
        # Create metrics object
        metrics = SystemMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            var_calculation_time=var_calc_time,
            attack_detection_time=attack_detection_time,
            byzantine_consensus_time=byzantine_time,
            correlation_update_time=correlation_time,
            active_alerts=active_alerts,
            threat_level=self.current_threat_level,
            system_health=self.system_health
        )
        
        return metrics
    
    def _check_performance_thresholds(self, metrics: SystemMetrics):
        """Check performance thresholds and generate alerts"""
        current_time = datetime.now()
        
        # Check each threshold
        for threshold_name, threshold in self.performance_thresholds.items():
            # Get metric value
            metric_value = getattr(metrics, threshold_name, 0.0)
            
            # Check for breaches
            if metric_value > threshold.critical_threshold:
                self._generate_alert(
                    level=MonitoringLevel.CRITICAL,
                    source="PerformanceMonitor",
                    message=f"CRITICAL: {threshold_name} exceeded critical threshold",
                    metrics={
                        "metric_name": threshold_name,
                        "current_value": metric_value,
                        "threshold": threshold.critical_threshold,
                        "breach_severity": "CRITICAL"
                    },
                    suggested_actions=[
                        f"Investigate {threshold_name} performance degradation",
                        "Consider scaling resources",
                        "Activate defensive mode",
                        "Reduce system load"
                    ]
                )
            elif metric_value > threshold.warning_threshold:
                self._generate_alert(
                    level=MonitoringLevel.WARNING,
                    source="PerformanceMonitor",
                    message=f"WARNING: {threshold_name} exceeded warning threshold",
                    metrics={
                        "metric_name": threshold_name,
                        "current_value": metric_value,
                        "threshold": threshold.warning_threshold,
                        "breach_severity": "WARNING"
                    },
                    suggested_actions=[
                        f"Monitor {threshold_name} closely",
                        "Review recent system changes",
                        "Consider optimization"
                    ]
                )
    
    def _update_threat_level(self, metrics: SystemMetrics):
        """Update system threat level based on metrics"""
        threat_score = 0
        
        # Factor in active alerts
        critical_alerts = len([alert for alert in self.alerts_history 
                             if not alert.auto_resolved and alert.level == MonitoringLevel.CRITICAL])
        warning_alerts = len([alert for alert in self.alerts_history 
                            if not alert.auto_resolved and alert.level == MonitoringLevel.WARNING])
        
        threat_score += critical_alerts * 10
        threat_score += warning_alerts * 3
        
        # Factor in performance degradation
        if metrics.var_calculation_time > 10.0:
            threat_score += 5
        if metrics.memory_usage > 85.0:
            threat_score += 4
        if metrics.cpu_usage > 80.0:
            threat_score += 3
        
        # Factor in attack detections
        recent_attacks = len(self.adversarial_integration.attack_detector.discovered_vulnerabilities)
        threat_score += recent_attacks * 2
        
        # Determine threat level
        if threat_score >= 20:
            new_threat_level = "CRITICAL"
        elif threat_score >= 10:
            new_threat_level = "HIGH"
        elif threat_score >= 5:
            new_threat_level = "MEDIUM"
        else:
            new_threat_level = "LOW"
        
        # Update threat level if changed
        if new_threat_level != self.current_threat_level:
            old_level = self.current_threat_level
            self.current_threat_level = new_threat_level
            
            self._generate_alert(
                level=MonitoringLevel.INFO,
                source="ThreatAnalyzer",
                message=f"Threat level changed from {old_level} to {new_threat_level}",
                metrics={
                    "old_threat_level": old_level,
                    "new_threat_level": new_threat_level,
                    "threat_score": threat_score
                },
                suggested_actions=[
                    "Review threat assessment",
                    "Adjust monitoring sensitivity",
                    "Update security posture"
                ]
            )
    
    def _execute_feedback_loops(self, metrics: SystemMetrics):
        """Execute automated feedback loops"""
        if not self.auto_mitigation_enabled:
            return
        
        # VaR performance feedback
        if metrics.var_calculation_time > 8.0:
            self._execute_feedback_action(
                action=FeedbackAction.ADJUST_THRESHOLD,
                source_system="MonitoringSystem",
                target_system="VaRCalculator",
                parameters={
                    "action": "reduce_calculation_complexity",
                    "current_time": metrics.var_calculation_time,
                    "target_time": 5.0
                }
            )
        
        # Attack detection feedback
        if len(self.adversarial_integration.attack_detector.discovered_vulnerabilities) > 5:
            self._execute_feedback_action(
                action=FeedbackAction.INCREASE_MONITORING,
                source_system="MonitoringSystem",
                target_system="AttackDetector",
                parameters={
                    "monitoring_frequency": self.monitoring_interval * 0.5,
                    "sensitivity_increase": 0.1
                }
            )
        
        # Byzantine consensus feedback
        byzantine_report = self.byzantine_detector.get_detection_report()
        malicious_nodes = byzantine_report.get("detection_summary", {}).get("malicious_nodes_detected", 0)
        
        if malicious_nodes > len(self.byzantine_detector.nodes) * 0.4:
            self._execute_feedback_action(
                action=FeedbackAction.TRIGGER_DEFENSIVE_MODE,
                source_system="MonitoringSystem",
                target_system="ByzantineDetector",
                parameters={
                    "malicious_ratio": malicious_nodes / len(self.byzantine_detector.nodes),
                    "action": "increase_consensus_threshold"
                }
            )
        
        # Correlation shock feedback
        correlation_shocks = len(self.adversarial_integration.correlation_tracker.shock_alerts)
        if correlation_shocks > 3:
            self._execute_feedback_action(
                action=FeedbackAction.RECALCULATE_VAR,
                source_system="MonitoringSystem",
                target_system="CorrelationTracker",
                parameters={
                    "recalculation_reason": "multiple_correlation_shocks",
                    "shock_count": correlation_shocks
                }
            )
    
    def _execute_feedback_action(
        self,
        action: FeedbackAction,
        source_system: str,
        target_system: str,
        parameters: Dict[str, Any]
    ):
        """Execute a feedback action"""
        event_id = f"feedback_{int(time.time() * 1000)}"
        
        feedback_event = FeedbackEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            action=action,
            source_system=source_system,
            target_system=target_system,
            parameters=parameters
        )
        
        try:
            # Execute action based on type
            if action == FeedbackAction.ADJUST_THRESHOLD:
                self._adjust_system_threshold(parameters)
            elif action == FeedbackAction.INCREASE_MONITORING:
                self._increase_monitoring_frequency(parameters)
            elif action == FeedbackAction.TRIGGER_DEFENSIVE_MODE:
                self._trigger_defensive_mode(parameters)
            elif action == FeedbackAction.RECALCULATE_VAR:
                self._trigger_var_recalculation(parameters)
            
            feedback_event.execution_result = {"status": "success"}
            feedback_event.execution_time = datetime.now()
            
            logger.info(f"Feedback action executed: {action.value}",
                       source=source_system,
                       target=target_system,
                       parameters=parameters)
            
        except Exception as e:
            feedback_event.execution_result = {"status": "error", "error": str(e)}
            feedback_event.execution_time = datetime.now()
            
            logger.error(f"Failed to execute feedback action: {action.value}",
                        error=str(e))
        
        # Store feedback event
        self.feedback_events.append(feedback_event)
    
    def _adjust_system_threshold(self, parameters: Dict[str, Any]):
        """Adjust system thresholds based on feedback"""
        if "action" in parameters:
            action = parameters["action"]
            
            if action == "reduce_calculation_complexity":
                # Reduce VaR calculation complexity
                self.adaptive_parameters['var_confidence_level'] = max(
                    0.90, self.adaptive_parameters['var_confidence_level'] - 0.01
                )
    
    def _increase_monitoring_frequency(self, parameters: Dict[str, Any]):
        """Increase monitoring frequency"""
        if "monitoring_frequency" in parameters:
            self.monitoring_interval = max(0.05, parameters["monitoring_frequency"])
    
    def _trigger_defensive_mode(self, parameters: Dict[str, Any]):
        """Trigger defensive mode"""
        self.defensive_mode = True
        
        # Adjust parameters for defensive mode
        self.adaptive_parameters['byzantine_trust_threshold'] = 0.5
        self.adaptive_parameters['ml_detection_threshold'] = 0.6
        self.adaptive_parameters['correlation_shock_threshold'] = 0.3
        
        logger.warning("Defensive mode activated", parameters=parameters)
    
    def _trigger_var_recalculation(self, parameters: Dict[str, Any]):
        """Trigger VaR recalculation"""
        # This would trigger a VaR recalculation through the event system
        event = Event(
            event_type=EventType.VAR_UPDATE,
            timestamp=datetime.now(),
            payload={"recalculation_trigger": "correlation_shock_feedback"},
            source="MonitoringSystem"
        )
        self.event_bus.publish(event)
    
    def _generate_alert(
        self,
        level: MonitoringLevel,
        source: str,
        message: str,
        metrics: Dict[str, Any],
        suggested_actions: List[str]
    ):
        """Generate monitoring alert"""
        alert = MonitoringAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            level=level,
            source=source,
            message=message,
            metrics=metrics,
            suggested_actions=suggested_actions
        )
        
        self.alerts_history.append(alert)
        
        # Log alert
        logger.log(
            level.value.lower(),
            f"Alert generated: {message}",
            source=source,
            metrics=metrics
        )
        
        # Broadcast to WebSocket clients
        asyncio.run_coroutine_threadsafe(
            self._broadcast_alert(alert),
            asyncio.get_event_loop()
        )
    
    async def _broadcast_metrics_update(self, metrics: SystemMetrics):
        """Broadcast metrics update to WebSocket clients"""
        if not self.connected_clients:
            return
        
        update_data = {
            "type": "metrics_update",
            "data": asdict(metrics)
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(update_data, default=str))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending metrics to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def _broadcast_alert(self, alert: MonitoringAlert):
        """Broadcast alert to WebSocket clients"""
        if not self.connected_clients:
            return
        
        alert_data = {
            "type": "alert",
            "data": asdict(alert)
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(alert_data, default=str))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending alert to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def _send_metrics_update(self, websocket):
        """Send metrics update to specific client"""
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            update_data = {
                "type": "metrics_response",
                "data": asdict(latest_metrics)
            }
            await websocket.send(json.dumps(update_data, default=str))
    
    async def _send_alerts_update(self, websocket):
        """Send alerts update to specific client"""
        recent_alerts = list(self.alerts_history)[-20:]  # Last 20 alerts
        update_data = {
            "type": "alerts_response",
            "data": [asdict(alert) for alert in recent_alerts]
        }
        await websocket.send(json.dumps(update_data, default=str))
    
    async def _handle_threshold_adjustment(self, data: Dict[str, Any]):
        """Handle threshold adjustment from client"""
        threshold_name = data.get("threshold_name")
        new_value = data.get("new_value")
        
        if threshold_name in self.performance_thresholds:
            threshold = self.performance_thresholds[threshold_name]
            
            if "warning" in data:
                threshold.warning_threshold = new_value
            elif "critical" in data:
                threshold.critical_threshold = new_value
            
            logger.info(f"Threshold adjusted: {threshold_name} = {new_value}")
    
    async def _handle_manual_action(self, data: Dict[str, Any]):
        """Handle manual action from client"""
        action_type = data.get("action_type")
        parameters = data.get("parameters", {})
        
        if action_type == "reset_defensive_mode":
            self.defensive_mode = False
            logger.info("Defensive mode reset by user")
        elif action_type == "clear_alerts":
            for alert in self.alerts_history:
                alert.auto_resolved = True
                alert.resolution_time = datetime.now()
            logger.info("Alerts cleared by user")
    
    # Event handlers
    def _handle_var_update(self, event: Event):
        """Handle VaR update events"""
        if hasattr(event.payload, 'performance_ms'):
            performance_ms = event.payload.performance_ms
            self.adversarial_integration.performance_metrics['var_calc_times'].append(performance_ms)
    
    def _handle_risk_breach(self, event: Event):
        """Handle risk breach events"""
        breach_data = event.payload
        
        self._generate_alert(
            level=MonitoringLevel.CRITICAL,
            source="RiskManager",
            message=f"Risk breach detected: {breach_data.get('type', 'Unknown')}",
            metrics=breach_data,
            suggested_actions=[
                "Review risk parameters",
                "Implement immediate risk reduction",
                "Investigate breach cause"
            ]
        )
    
    def _handle_system_error(self, event: Event):
        """Handle system error events"""
        error_data = event.payload
        
        self._generate_alert(
            level=MonitoringLevel.ERROR,
            source="SystemMonitor",
            message=f"System error: {error_data}",
            metrics={"error_details": str(error_data)},
            suggested_actions=[
                "Review system logs",
                "Check system components",
                "Implement error recovery"
            ]
        )
    
    def _handle_risk_update(self, event: Event):
        """Handle risk update events"""
        # Process risk updates for monitoring
        pass
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        current_time = datetime.now()
        
        # Recent metrics
        recent_metrics = list(self.metrics_history)[-100:] if self.metrics_history else []
        
        # Alert statistics
        alert_stats = {
            "total_alerts": len(self.alerts_history),
            "active_alerts": len([alert for alert in self.alerts_history if not alert.auto_resolved]),
            "critical_alerts": len([alert for alert in self.alerts_history 
                                  if alert.level == MonitoringLevel.CRITICAL and not alert.auto_resolved]),
            "warning_alerts": len([alert for alert in self.alerts_history 
                                 if alert.level == MonitoringLevel.WARNING and not alert.auto_resolved])
        }
        
        # Feedback statistics
        feedback_stats = {
            "total_feedback_events": len(self.feedback_events),
            "successful_actions": len([event for event in self.feedback_events 
                                     if event.execution_result and event.execution_result.get("status") == "success"]),
            "failed_actions": len([event for event in self.feedback_events 
                                 if event.execution_result and event.execution_result.get("status") == "error"])
        }
        
        return {
            "monitoring_status": {
                "active": self.monitoring_active,
                "threat_level": self.current_threat_level,
                "system_health": self.system_health,
                "defensive_mode": self.defensive_mode,
                "connected_clients": len(self.connected_clients),
                "monitoring_interval": self.monitoring_interval
            },
            "alert_statistics": alert_stats,
            "feedback_statistics": feedback_stats,
            "performance_metrics": {
                "metrics_collected": len(self.metrics_history),
                "avg_var_calc_time": np.mean([m.var_calculation_time for m in recent_metrics]) if recent_metrics else 0,
                "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]) if recent_metrics else 0,
                "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]) if recent_metrics else 0
            },
            "adaptive_parameters": self.adaptive_parameters.copy(),
            "performance_thresholds": {
                name: asdict(threshold) for name, threshold in self.performance_thresholds.items()
            }
        }


# Example usage
async def main():
    """Example usage of Real-time Monitoring System"""
    # This would be integrated with the full system
    pass


if __name__ == "__main__":
    asyncio.run(main())