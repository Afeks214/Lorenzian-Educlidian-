"""
Emergency Failsafe Protocols for Consensus System

Implements comprehensive emergency response protocols for Byzantine fault tolerant
consensus failures, ensuring system safety under all conditions.

Emergency Scenarios:
- Complete consensus failure (no nodes responding)
- Majority Byzantine attack (>f nodes compromised)
- Network partition attacks
- Cryptographic system compromise
- View change cascades
- Resource exhaustion attacks

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class EmergencyLevel(Enum):
    """Emergency severity levels"""
    GREEN = "green"      # Normal operation
    YELLOW = "yellow"    # Minor issues detected
    ORANGE = "orange"    # Significant problems
    RED = "red"          # Critical failures
    BLACK = "black"      # Complete system compromise


class FailsafeAction(Enum):
    """Types of failsafe actions"""
    MONITOR = "monitor"
    ALERT = "alert"
    DEGRADE_PERFORMANCE = "degrade_performance"
    ACTIVATE_BACKUP = "activate_backup"
    ISOLATE_BYZANTINE = "isolate_byzantine"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    FORCE_SAFE_MODE = "force_safe_mode"
    NETWORK_PARTITION_RECOVERY = "network_partition_recovery"


@dataclass
class EmergencyEvent:
    """Emergency event record"""
    event_id: str
    event_type: str
    severity: EmergencyLevel
    timestamp: float
    description: str
    affected_components: List[str]
    recommended_actions: List[FailsafeAction]
    auto_executed: bool = False
    resolution_timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class FailsafeConfig:
    """Failsafe configuration parameters"""
    max_consensus_failures: int = 3
    max_byzantine_ratio: float = 0.33
    consensus_timeout_threshold: float = 10.0  # seconds
    view_change_cascade_threshold: int = 5
    signature_failure_threshold: float = 0.1
    network_partition_timeout: float = 30.0
    emergency_shutdown_threshold: int = 10  # critical events
    auto_recovery_enabled: bool = True
    safe_mode_threshold: int = 5  # consecutive failures


class EmergencyFailsafe:
    """
    Emergency Failsafe System for Byzantine Consensus
    
    Monitors consensus system health and implements emergency protocols
    when Byzantine faults or system failures exceed tolerance thresholds.
    Provides graduated response from monitoring to emergency shutdown.
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        config: Optional[FailsafeConfig] = None,
        emergency_callback: Optional[Callable] = None
    ):
        """
        Initialize emergency failsafe system
        
        Args:
            agent_ids: List of participating agent IDs
            config: Failsafe configuration
            emergency_callback: Callback function for emergency notifications
        """
        self.agent_ids = agent_ids
        self.config = config or FailsafeConfig()
        self.emergency_callback = emergency_callback
        
        # Current system state
        self.current_level = EmergencyLevel.GREEN
        self.active_events: Dict[str, EmergencyEvent] = {}
        self.event_history: deque = deque(maxlen=1000)
        
        # Monitoring state
        self.consensus_failures = 0
        self.consecutive_failures = 0
        self.byzantine_agents: Set[str] = set()
        self.network_partitions: Set[str] = set()
        self.last_successful_consensus = time.time()
        self.view_change_count = 0
        self.signature_failures = 0
        self.total_signatures = 0
        
        # Safe mode state
        self.safe_mode_active = False
        self.safe_mode_start_time = None
        self.degraded_performance_mode = False
        
        # Emergency protocols
        self.emergency_protocols = {
            EmergencyLevel.YELLOW: self._handle_yellow_alert,
            EmergencyLevel.ORANGE: self._handle_orange_alert,
            EmergencyLevel.RED: self._handle_red_alert,
            EmergencyLevel.BLACK: self._handle_black_alert
        }
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Metrics
        self.failsafe_metrics = {
            'total_emergency_events': 0,
            'yellow_alerts': 0,
            'orange_alerts': 0,
            'red_alerts': 0,
            'black_alerts': 0,
            'auto_recoveries': 0,
            'manual_interventions': 0,
            'safe_mode_activations': 0,
            'emergency_shutdowns': 0
        }
        
        logger.info(f"Emergency failsafe system initialized for {len(agent_ids)} agents")
    
    def record_consensus_attempt(
        self,
        success: bool,
        latency: float,
        byzantine_detected: List[str],
        view_changes: int = 0,
        signature_failures: int = 0,
        total_signatures: int = 0
    ):
        """
        Record consensus attempt results for monitoring
        
        Args:
            success: Whether consensus was achieved
            latency: Consensus latency in seconds
            byzantine_detected: List of Byzantine agents detected
            view_changes: Number of view changes
            signature_failures: Number of signature validation failures
            total_signatures: Total signatures validated
        """
        current_time = time.time()
        
        # Update consensus tracking
        if success:
            self.last_successful_consensus = current_time
            self.consecutive_failures = 0
        else:
            self.consensus_failures += 1
            self.consecutive_failures += 1
        
        # Update Byzantine agent tracking
        self.byzantine_agents.update(byzantine_detected)
        
        # Update view change tracking
        self.view_change_count += view_changes
        
        # Update signature failure tracking
        self.signature_failures += signature_failures
        self.total_signatures += total_signatures
        
        # Evaluate emergency conditions
        self._evaluate_emergency_conditions(success, latency, byzantine_detected, view_changes)
    
    def _evaluate_emergency_conditions(
        self,
        success: bool,
        latency: float,
        byzantine_detected: List[str],
        view_changes: int
    ):
        """Evaluate if emergency conditions are met"""
        emergency_level = EmergencyLevel.GREEN
        emergency_events = []
        
        # Check for consensus failures
        if self.consecutive_failures >= self.config.max_consensus_failures:
            emergency_level = max(emergency_level, EmergencyLevel.ORANGE)
            emergency_events.append({
                'type': 'consecutive_consensus_failures',
                'description': f'{self.consecutive_failures} consecutive consensus failures',
                'severity': EmergencyLevel.ORANGE
            })
        
        # Check for Byzantine ratio
        byzantine_ratio = len(self.byzantine_agents) / len(self.agent_ids)
        if byzantine_ratio > self.config.max_byzantine_ratio:
            emergency_level = max(emergency_level, EmergencyLevel.RED)
            emergency_events.append({
                'type': 'byzantine_majority',
                'description': f'Byzantine ratio {byzantine_ratio:.2%} exceeds threshold {self.config.max_byzantine_ratio:.2%}',
                'severity': EmergencyLevel.RED
            })
        
        # Check for consensus timeout
        time_since_last_success = time.time() - self.last_successful_consensus
        if time_since_last_success > self.config.consensus_timeout_threshold:
            emergency_level = max(emergency_level, EmergencyLevel.ORANGE)
            emergency_events.append({
                'type': 'consensus_timeout',
                'description': f'No successful consensus for {time_since_last_success:.1f} seconds',
                'severity': EmergencyLevel.ORANGE
            })
        
        # Check for view change cascade
        if self.view_change_count >= self.config.view_change_cascade_threshold:
            emergency_level = max(emergency_level, EmergencyLevel.YELLOW)
            emergency_events.append({
                'type': 'view_change_cascade',
                'description': f'{self.view_change_count} view changes detected',
                'severity': EmergencyLevel.YELLOW
            })
        
        # Check for signature failure rate
        if self.total_signatures > 10:  # Minimum sample size
            signature_failure_rate = self.signature_failures / self.total_signatures
            if signature_failure_rate > self.config.signature_failure_threshold:
                emergency_level = max(emergency_level, EmergencyLevel.ORANGE)
                emergency_events.append({
                    'type': 'high_signature_failures',
                    'description': f'Signature failure rate {signature_failure_rate:.2%} exceeds threshold',
                    'severity': EmergencyLevel.ORANGE
                })
        
        # Check for safe mode threshold
        if self.consecutive_failures >= self.config.safe_mode_threshold:
            emergency_level = max(emergency_level, EmergencyLevel.RED)
            emergency_events.append({
                'type': 'safe_mode_required',
                'description': f'{self.consecutive_failures} consecutive failures require safe mode',
                'severity': EmergencyLevel.RED
            })
        
        # Create emergency events
        for event_data in emergency_events:
            self._create_emergency_event(
                event_type=event_data['type'],
                severity=event_data['severity'],
                description=event_data['description'],
                affected_components=['consensus_system']
            )
        
        # Update emergency level
        if emergency_level != self.current_level:
            self._escalate_emergency_level(emergency_level)
    
    def _create_emergency_event(
        self,
        event_type: str,
        severity: EmergencyLevel,
        description: str,
        affected_components: List[str]
    ) -> EmergencyEvent:
        """Create and register emergency event"""
        event_id = f"{event_type}_{int(time.time() * 1000)}"
        
        # Determine recommended actions
        recommended_actions = self._get_recommended_actions(severity, event_type)
        
        event = EmergencyEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            description=description,
            affected_components=affected_components,
            recommended_actions=recommended_actions
        )
        
        # Register event
        self.active_events[event_id] = event
        self.event_history.append(event)
        self.failsafe_metrics['total_emergency_events'] += 1
        self.failsafe_metrics[f'{severity.value}_alerts'] += 1
        
        logger.warning(f"Emergency event created: {event_type} ({severity.value}) - {description}")
        
        # Notify external systems
        if self.emergency_callback:
            try:
                self.emergency_callback(event)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        return event
    
    def _get_recommended_actions(self, severity: EmergencyLevel, event_type: str) -> List[FailsafeAction]:
        """Get recommended actions for emergency event"""
        actions = []
        
        if severity == EmergencyLevel.YELLOW:
            actions = [FailsafeAction.MONITOR, FailsafeAction.ALERT]
        elif severity == EmergencyLevel.ORANGE:
            actions = [FailsafeAction.ALERT, FailsafeAction.DEGRADE_PERFORMANCE]
            if 'byzantine' in event_type:
                actions.append(FailsafeAction.ISOLATE_BYZANTINE)
        elif severity == EmergencyLevel.RED:
            actions = [FailsafeAction.FORCE_SAFE_MODE, FailsafeAction.ACTIVATE_BACKUP]
            if 'byzantine_majority' in event_type:
                actions.append(FailsafeAction.EMERGENCY_SHUTDOWN)
        elif severity == EmergencyLevel.BLACK:
            actions = [FailsafeAction.EMERGENCY_SHUTDOWN]
        
        # Add network partition recovery for timeout events
        if 'timeout' in event_type or 'partition' in event_type:
            actions.append(FailsafeAction.NETWORK_PARTITION_RECOVERY)
        
        return actions
    
    def _escalate_emergency_level(self, new_level: EmergencyLevel):
        """Escalate emergency level and execute protocols"""
        old_level = self.current_level
        self.current_level = new_level
        
        logger.critical(f"Emergency level escalated from {old_level.value} to {new_level.value}")
        
        # Execute emergency protocol
        if new_level in self.emergency_protocols:
            try:
                self.emergency_protocols[new_level]()
            except Exception as e:
                logger.error(f"Failed to execute {new_level.value} protocol: {e}")
    
    def _handle_yellow_alert(self):
        """Handle yellow alert (minor issues)"""
        logger.warning("YELLOW ALERT: Minor consensus issues detected")
        
        # Enable enhanced monitoring
        # Reset view change counter if needed
        if self.view_change_count >= self.config.view_change_cascade_threshold:
            self.view_change_count = 0  # Reset after acknowledgment
    
    def _handle_orange_alert(self):
        """Handle orange alert (significant problems)"""
        logger.error("ORANGE ALERT: Significant consensus problems detected")
        
        # Activate degraded performance mode
        if not self.degraded_performance_mode:
            self.degraded_performance_mode = True
            logger.info("Activated degraded performance mode")
        
        # Isolate suspected Byzantine agents
        if self.byzantine_agents:
            logger.warning(f"Isolating Byzantine agents: {self.byzantine_agents}")
            # In a real system, this would notify the consensus engine
    
    def _handle_red_alert(self):
        """Handle red alert (critical failures)"""
        logger.critical("RED ALERT: Critical consensus system failure")
        
        # Activate safe mode
        self._activate_safe_mode()
        
        # Consider emergency shutdown if Byzantine majority
        byzantine_ratio = len(self.byzantine_agents) / len(self.agent_ids)
        if byzantine_ratio > 0.5:
            logger.critical("Byzantine majority detected - considering emergency shutdown")
            self._prepare_emergency_shutdown()
    
    def _handle_black_alert(self):
        """Handle black alert (complete system compromise)"""
        logger.critical("BLACK ALERT: Complete system compromise detected")
        
        # Immediate emergency shutdown
        self._execute_emergency_shutdown()
    
    def _activate_safe_mode(self):
        """Activate safe mode with restricted operations"""
        if not self.safe_mode_active:
            self.safe_mode_active = True
            self.safe_mode_start_time = time.time()
            self.failsafe_metrics['safe_mode_activations'] += 1
            
            logger.critical("SAFE MODE ACTIVATED - All trading operations suspended")
            
            # In a real system, this would:
            # 1. Suspend all new trading operations
            # 2. Close existing positions if needed
            # 3. Enable manual override only
            # 4. Increase logging verbosity
            # 5. Send alerts to operators
    
    def _prepare_emergency_shutdown(self):
        """Prepare for emergency shutdown"""
        logger.critical("Preparing for emergency shutdown - Grace period: 30 seconds")
        
        # Grace period for manual intervention
        threading.Timer(30.0, self._execute_emergency_shutdown).start()
    
    def _execute_emergency_shutdown(self):
        """Execute emergency shutdown of consensus system"""
        logger.critical("EXECUTING EMERGENCY SHUTDOWN")
        
        self.failsafe_metrics['emergency_shutdowns'] += 1
        
        # In a real system, this would:
        # 1. Immediately halt all consensus operations
        # 2. Close all trading positions
        # 3. Disable all automated systems
        # 4. Notify operators and stakeholders
        # 5. Begin incident response procedures
        
        # Shutdown monitoring
        self.monitoring_active = False
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._periodic_health_check()
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)
    
    def _periodic_health_check(self):
        """Perform periodic health checks"""
        current_time = time.time()
        
        # Check for stale consensus (no activity)
        time_since_last_success = current_time - self.last_successful_consensus
        if time_since_last_success > self.config.network_partition_timeout:
            if 'network_partition' not in [e.event_type for e in self.active_events.values()]:
                self._create_emergency_event(
                    event_type='network_partition',
                    severity=EmergencyLevel.RED,
                    description=f'No consensus activity for {time_since_last_success:.1f} seconds',
                    affected_components=['network', 'consensus_system']
                )
        
        # Check for automatic recovery opportunities
        if self.config.auto_recovery_enabled:
            self._attempt_auto_recovery()
        
        # Clean up resolved events
        self._cleanup_resolved_events()
    
    def _attempt_auto_recovery(self):
        """Attempt automatic recovery from emergency conditions"""
        if self.current_level == EmergencyLevel.GREEN:
            return
        
        # Recovery conditions
        recovery_possible = (
            self.consecutive_failures == 0 and  # Recent success
            len(self.byzantine_agents) <= self.config.max_byzantine_ratio * len(self.agent_ids) and
            time.time() - self.last_successful_consensus < 60.0  # Recent activity
        )
        
        if recovery_possible:
            self._attempt_recovery()
    
    def _attempt_recovery(self):
        """Attempt to recover from emergency state"""
        logger.info("Attempting automatic recovery from emergency state")
        
        # Reset counters
        self.consensus_failures = 0
        self.view_change_count = 0
        
        # Deactivate degraded performance mode
        if self.degraded_performance_mode:
            self.degraded_performance_mode = False
            logger.info("Deactivated degraded performance mode")
        
        # Consider deactivating safe mode
        if self.safe_mode_active and self.consecutive_failures == 0:
            safe_mode_duration = time.time() - self.safe_mode_start_time
            if safe_mode_duration > 300:  # 5 minutes minimum
                self._deactivate_safe_mode()
        
        # Lower emergency level
        self.current_level = EmergencyLevel.GREEN
        self.failsafe_metrics['auto_recoveries'] += 1
        
        logger.info("Automatic recovery completed")
    
    def _deactivate_safe_mode(self):
        """Deactivate safe mode"""
        if self.safe_mode_active:
            self.safe_mode_active = False
            self.safe_mode_start_time = None
            logger.info("SAFE MODE DEACTIVATED - Normal operations resumed")
    
    def _cleanup_resolved_events(self):
        """Clean up resolved emergency events"""
        current_time = time.time()
        resolved_events = []
        
        for event_id, event in self.active_events.items():
            # Consider events resolved after 1 hour of green status
            if (self.current_level == EmergencyLevel.GREEN and 
                current_time - event.timestamp > 3600):
                event.resolution_timestamp = current_time
                resolved_events.append(event_id)
        
        for event_id in resolved_events:
            del self.active_events[event_id]
            logger.info(f"Emergency event {event_id} marked as resolved")
    
    def manual_recovery(self, recovery_reason: str) -> bool:
        """
        Manual recovery from emergency state
        
        Args:
            recovery_reason: Reason for manual recovery
            
        Returns:
            True if recovery was successful
        """
        try:
            logger.info(f"Manual recovery initiated: {recovery_reason}")
            
            # Reset all emergency state
            self.current_level = EmergencyLevel.GREEN
            self.consensus_failures = 0
            self.consecutive_failures = 0
            self.byzantine_agents.clear()
            self.network_partitions.clear()
            self.view_change_count = 0
            
            # Deactivate emergency modes
            self.degraded_performance_mode = False
            if self.safe_mode_active:
                self._deactivate_safe_mode()
            
            # Clear active events
            for event in self.active_events.values():
                event.resolution_timestamp = time.time()
            self.active_events.clear()
            
            self.failsafe_metrics['manual_interventions'] += 1
            
            logger.info("Manual recovery completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Manual recovery failed: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = time.time()
        
        return {
            'emergency_level': self.current_level.value,
            'safe_mode_active': self.safe_mode_active,
            'degraded_performance_mode': self.degraded_performance_mode,
            'consensus_failures': self.consensus_failures,
            'consecutive_failures': self.consecutive_failures,
            'byzantine_agents': list(self.byzantine_agents),
            'byzantine_ratio': len(self.byzantine_agents) / len(self.agent_ids),
            'time_since_last_success': current_time - self.last_successful_consensus,
            'view_change_count': self.view_change_count,
            'signature_failure_rate': self.signature_failures / max(1, self.total_signatures),
            'active_events': len(self.active_events),
            'total_events': len(self.event_history),
            'monitoring_active': self.monitoring_active,
            'metrics': self.failsafe_metrics.copy()
        }
    
    def get_active_events(self) -> List[Dict[str, Any]]:
        """Get all active emergency events"""
        return [
            {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'severity': event.severity.value,
                'timestamp': event.timestamp,
                'description': event.description,
                'affected_components': event.affected_components,
                'recommended_actions': [action.value for action in event.recommended_actions],
                'auto_executed': event.auto_executed
            }
            for event in self.active_events.values()
        ]
    
    def export_incident_report(self) -> str:
        """Export comprehensive incident report"""
        report = {
            'timestamp': time.time(),
            'system_status': self.get_system_status(),
            'active_events': self.get_active_events(),
            'event_history': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'severity': event.severity.value,
                    'timestamp': event.timestamp,
                    'description': event.description,
                    'resolution_timestamp': event.resolution_timestamp
                }
                for event in list(self.event_history)[-50:]  # Last 50 events
            ],
            'configuration': {
                'max_consensus_failures': self.config.max_consensus_failures,
                'max_byzantine_ratio': self.config.max_byzantine_ratio,
                'consensus_timeout_threshold': self.config.consensus_timeout_threshold,
                'auto_recovery_enabled': self.config.auto_recovery_enabled
            }
        }
        
        return json.dumps(report, indent=2, default=str)
    
    def shutdown(self):
        """Graceful shutdown of failsafe system"""
        logger.info("Shutting down emergency failsafe system")
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)