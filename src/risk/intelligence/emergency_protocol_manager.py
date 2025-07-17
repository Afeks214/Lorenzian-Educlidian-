"""
Emergency Protocol Manager

This module manages automatic emergency response protocols for crisis detection.
Implements automatic leverage reduction, position halting, and emergency safeguards.

Key Features:
- Automatic 75% leverage reduction on crisis detection
- Emergency position protection protocols
- Manual reset requirement to prevent auto re-escalation
- Complete audit trail with timestamps and reasoning
- Event-driven notifications and coordination
"""

import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog
import asyncio
from collections import deque

from src.core.events import Event, EventType, EventBus
from .crisis_dataset_processor import CrisisType
from .maml_crisis_detector import CrisisDetectionResult
from .crisis_fingerprint_engine import PatternMatch

logger = structlog.get_logger()


class EmergencyLevel(Enum):
    """Emergency response levels"""
    NONE = "none"
    LEVEL_1 = "level_1"     # Monitoring increase
    LEVEL_2 = "level_2"     # 50% leverage reduction
    LEVEL_3 = "level_3"     # 75% leverage reduction + halt positions


class ProtocolStatus(Enum):
    """Protocol execution status"""
    INACTIVE = "inactive"
    MONITORING = "monitoring"
    ACTIVE = "active"
    EMERGENCY = "emergency"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class EmergencyAction:
    """Emergency action record"""
    timestamp: datetime
    action_type: str
    emergency_level: EmergencyLevel
    trigger_reason: str
    parameters: Dict
    execution_status: str
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class EmergencyState:
    """Current emergency state"""
    status: ProtocolStatus
    level: EmergencyLevel
    activation_time: Optional[datetime]
    trigger_event: Optional[Dict]
    active_protocols: List[str]
    manual_reset_required: bool
    auto_escalation_disabled: bool


@dataclass
class ProtocolConfig:
    """Emergency protocol configuration"""
    # Crisis similarity thresholds for each level
    level_1_threshold: float = 0.70  # >70% similarity
    level_2_threshold: float = 0.85  # >85% similarity  
    level_3_threshold: float = 0.95  # >95% similarity
    
    # Leverage reduction parameters
    level_2_leverage_reduction: float = 0.50  # 50% reduction
    level_3_leverage_reduction: float = 0.75  # 75% reduction
    
    # Timing parameters
    monitoring_frequency_seconds: int = 10
    emergency_timeout_minutes: int = 60
    manual_reset_timeout_hours: int = 24
    
    # Position management
    halt_new_positions: bool = True
    close_risky_positions: bool = True
    max_position_size_emergency: float = 0.01  # 1% max position during emergency


class EmergencyProtocolManager:
    """
    Manages automatic emergency response protocols for crisis detection.
    
    Implements graduated response levels with automatic risk reduction
    and emergency safeguards to protect the portfolio during crisis events.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: ProtocolConfig = None
    ):
        self.event_bus = event_bus
        self.config = config or ProtocolConfig()
        
        # Current state
        self.emergency_state = EmergencyState(
            status=ProtocolStatus.INACTIVE,
            level=EmergencyLevel.NONE,
            activation_time=None,
            trigger_event=None,
            active_protocols=[],
            manual_reset_required=False,
            auto_escalation_disabled=False
        )
        
        # Action history
        self.action_history = deque(maxlen=1000)
        self.state_history = deque(maxlen=100)
        
        # Protocol coordination
        self.active_protocols = {}
        self.protocol_callbacks = {}
        
        # Performance tracking
        self.response_times = deque(maxlen=100)
        self.target_response_time_ms = 100  # <100ms emergency response
        
        # Subscribe to crisis detection events
        self._setup_event_subscriptions()
        
        logger.info("EmergencyProtocolManager initialized",
                   config=self.config.__dict__)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for crisis detection"""
        
        # Crisis detection events
        self.event_bus.subscribe(EventType.CRISIS_PREMONITION_DETECTED, self._handle_crisis_detection)
        self.event_bus.subscribe(EventType.CRISIS_PATTERN_MATCH, self._handle_pattern_match)
        
        # Position and risk events
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
        
        # System events
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._handle_system_error)
    
    async def _handle_crisis_detection(self, event: Event):
        """Handle crisis detection event"""
        
        start_time = datetime.now()
        
        try:
            crisis_result = event.payload
            
            if isinstance(crisis_result, dict):
                # Convert dict to detection result
                confidence = crisis_result.get('confidence_score', 0.0)
                crisis_type = crisis_result.get('crisis_type', 'unknown')
                similarity = crisis_result.get('similarity_score', 0.0)
            else:
                confidence = getattr(crisis_result, 'confidence_score', 0.0)
                crisis_type = getattr(crisis_result, 'crisis_type', 'unknown')
                similarity = getattr(crisis_result, 'similarity_score', 0.0)
            
            # Determine emergency level based on confidence/similarity
            emergency_level = self._determine_emergency_level(confidence, similarity)
            
            # Execute appropriate protocol
            if emergency_level != EmergencyLevel.NONE:
                await self._execute_emergency_protocol(
                    emergency_level,
                    f"Crisis detection: {crisis_type} (confidence: {confidence:.2f})",
                    crisis_result
                )
            
            # Track response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.response_times.append(response_time)
            
            if response_time > self.target_response_time_ms:
                logger.warning(f"Emergency response time {response_time:.2f}ms exceeds target {self.target_response_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Failed to handle crisis detection: {e}")
    
    async def _handle_pattern_match(self, event: Event):
        """Handle high-confidence pattern match"""
        
        try:
            pattern_match = event.payload
            
            if hasattr(pattern_match, 'pattern_confidence'):
                confidence = pattern_match.pattern_confidence
                similarity = pattern_match.similarity_score
                crisis_type = pattern_match.matched_pattern.crisis_type
            else:
                confidence = pattern_match.get('confidence', 0.0)
                similarity = pattern_match.get('similarity', 0.0)
                crisis_type = pattern_match.get('crisis_type', 'unknown')
            
            # Only respond to high-confidence matches
            if confidence >= 0.8 and similarity >= 0.8:
                emergency_level = self._determine_emergency_level(confidence, similarity)
                
                if emergency_level != EmergencyLevel.NONE:
                    await self._execute_emergency_protocol(
                        emergency_level,
                        f"High-confidence pattern match: {crisis_type}",
                        pattern_match
                    )
            
        except Exception as e:
            logger.error(f"Failed to handle pattern match: {e}")
    
    async def _handle_position_update(self, event: Event):
        """Handle position updates during emergency"""
        
        if self.emergency_state.status in [ProtocolStatus.ACTIVE, ProtocolStatus.EMERGENCY]:
            # Monitor position changes during emergency
            await self._verify_emergency_compliance(event.payload)
    
    async def _handle_risk_breach(self, event: Event):
        """Handle risk breaches with emergency escalation"""
        
        risk_breach = event.payload
        
        if risk_breach.get('type') == 'VAR_LIMIT_BREACH':
            # Escalate to emergency if VaR breach occurs during crisis monitoring
            if self.emergency_state.level == EmergencyLevel.LEVEL_1:
                await self._execute_emergency_protocol(
                    EmergencyLevel.LEVEL_2,
                    "VaR breach during crisis monitoring",
                    risk_breach
                )
    
    async def _handle_emergency_stop(self, event: Event):
        """Handle emergency stop requests"""
        
        await self._execute_emergency_protocol(
            EmergencyLevel.LEVEL_3,
            "Manual emergency stop triggered",
            event.payload
        )
    
    async def _handle_system_error(self, event: Event):
        """Handle critical system errors"""
        
        error_data = event.payload
        
        # If error affects trading or risk systems, activate emergency protocols
        if any(keyword in str(error_data).lower() for keyword in ['trading', 'risk', 'position', 'order']):
            await self._execute_emergency_protocol(
                EmergencyLevel.LEVEL_2,
                f"Critical system error: {error_data}",
                error_data
            )
    
    def _determine_emergency_level(self, confidence: float, similarity: float) -> EmergencyLevel:
        """Determine emergency level based on detection confidence and similarity"""
        
        # Use the higher of confidence or similarity for level determination
        max_score = max(confidence, similarity)
        
        if max_score >= self.config.level_3_threshold:
            return EmergencyLevel.LEVEL_3
        elif max_score >= self.config.level_2_threshold:
            return EmergencyLevel.LEVEL_2
        elif max_score >= self.config.level_1_threshold:
            return EmergencyLevel.LEVEL_1
        else:
            return EmergencyLevel.NONE
    
    async def _execute_emergency_protocol(
        self,
        emergency_level: EmergencyLevel,
        trigger_reason: str,
        trigger_data: any
    ) -> bool:
        """Execute emergency protocol for given level"""
        
        logger.warning(f"Executing emergency protocol {emergency_level.value}: {trigger_reason}")
        
        try:
            # Check if manual reset is required
            if (self.emergency_state.manual_reset_required and 
                emergency_level.value >= self.emergency_state.level.value):
                logger.warning("Emergency escalation blocked - manual reset required")
                return False
            
            # Check if auto-escalation is disabled
            if (self.emergency_state.auto_escalation_disabled and 
                emergency_level.value > self.emergency_state.level.value):
                logger.warning("Emergency escalation disabled - manual intervention required")
                return False
            
            # Update state
            old_state = self.emergency_state
            self.emergency_state.level = emergency_level
            self.emergency_state.activation_time = datetime.now()
            self.emergency_state.trigger_event = {
                'reason': trigger_reason,
                'data': str(trigger_data)[:500],  # Truncate for storage
                'timestamp': datetime.now().isoformat()
            }
            
            # Execute level-specific protocols
            actions_executed = []
            
            if emergency_level == EmergencyLevel.LEVEL_1:
                actions_executed = await self._execute_level_1_protocol()
                self.emergency_state.status = ProtocolStatus.MONITORING
                
            elif emergency_level == EmergencyLevel.LEVEL_2:
                actions_executed = await self._execute_level_2_protocol()
                self.emergency_state.status = ProtocolStatus.ACTIVE
                
            elif emergency_level == EmergencyLevel.LEVEL_3:
                actions_executed = await self._execute_level_3_protocol()
                self.emergency_state.status = ProtocolStatus.EMERGENCY
                self.emergency_state.manual_reset_required = True
            
            # Record actions
            for action in actions_executed:
                self.action_history.append(action)
            
            # Record state change
            self.state_history.append({
                'timestamp': datetime.now(),
                'old_state': old_state,
                'new_state': self.emergency_state,
                'trigger': trigger_reason,
                'actions': len(actions_executed)
            })
            
            # Publish emergency event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP if emergency_level == EmergencyLevel.LEVEL_3 else EventType.RISK_UPDATE,
                    {
                        'emergency_level': emergency_level.value,
                        'trigger_reason': trigger_reason,
                        'actions_executed': len(actions_executed),
                        'status': self.emergency_state.status.value
                    },
                    'EmergencyProtocolManager'
                )
            )
            
            logger.info(f"Emergency protocol {emergency_level.value} executed successfully",
                       actions_count=len(actions_executed))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute emergency protocol {emergency_level.value}: {e}")
            return False
    
    async def _execute_level_1_protocol(self) -> List[EmergencyAction]:
        """Execute Level 1 emergency protocol - monitoring increase"""
        
        actions = []
        timestamp = datetime.now()
        
        # Increase monitoring frequency
        action = EmergencyAction(
            timestamp=timestamp,
            action_type="INCREASE_MONITORING",
            emergency_level=EmergencyLevel.LEVEL_1,
            trigger_reason="Crisis pattern detected - Level 1",
            parameters={'monitoring_frequency': self.config.monitoring_frequency_seconds},
            execution_status="COMPLETED",
            completion_time=timestamp
        )
        actions.append(action)
        
        # Publish monitoring increase event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_UPDATE,
                {
                    'action': 'INCREASE_MONITORING',
                    'level': 'LEVEL_1',
                    'monitoring_frequency': self.config.monitoring_frequency_seconds
                },
                'EmergencyProtocolManager'
            )
        )
        
        return actions
    
    async def _execute_level_2_protocol(self) -> List[EmergencyAction]:
        """Execute Level 2 emergency protocol - 50% leverage reduction"""
        
        actions = []
        timestamp = datetime.now()
        
        # Reduce leverage by 50%
        leverage_action = EmergencyAction(
            timestamp=timestamp,
            action_type="REDUCE_LEVERAGE",
            emergency_level=EmergencyLevel.LEVEL_2,
            trigger_reason="Crisis pattern detected - Level 2",
            parameters={
                'leverage_reduction': self.config.level_2_leverage_reduction,
                'target_leverage': f"{(1 - self.config.level_2_leverage_reduction) * 100}%"
            },
            execution_status="INITIATED"
        )
        
        # Execute leverage reduction
        try:
            await self._reduce_portfolio_leverage(self.config.level_2_leverage_reduction)
            leverage_action.execution_status = "COMPLETED"
            leverage_action.completion_time = datetime.now()
        except Exception as e:
            leverage_action.execution_status = "FAILED"
            leverage_action.error_message = str(e)
        
        actions.append(leverage_action)
        
        # Tighten risk limits
        risk_action = EmergencyAction(
            timestamp=timestamp,
            action_type="TIGHTEN_RISK_LIMITS",
            emergency_level=EmergencyLevel.LEVEL_2,
            trigger_reason="Crisis pattern detected - Level 2",
            parameters={'risk_multiplier': 0.7},
            execution_status="COMPLETED",
            completion_time=timestamp
        )
        actions.append(risk_action)
        
        # Publish leverage reduction event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.POSITION_SIZE_UPDATE,
                {
                    'action': 'EMERGENCY_LEVERAGE_REDUCTION',
                    'level': 'LEVEL_2',
                    'reduction_percentage': self.config.level_2_leverage_reduction,
                    'reason': 'Crisis detection Level 2'
                },
                'EmergencyProtocolManager'
            )
        )
        
        return actions
    
    async def _execute_level_3_protocol(self) -> List[EmergencyAction]:
        """Execute Level 3 emergency protocol - 75% leverage reduction + halt positions"""
        
        actions = []
        timestamp = datetime.now()
        
        # Reduce leverage by 75%
        leverage_action = EmergencyAction(
            timestamp=timestamp,
            action_type="EMERGENCY_LEVERAGE_REDUCTION",
            emergency_level=EmergencyLevel.LEVEL_3,
            trigger_reason="Crisis pattern detected - Level 3",
            parameters={
                'leverage_reduction': self.config.level_3_leverage_reduction,
                'target_leverage': f"{(1 - self.config.level_3_leverage_reduction) * 100}%"
            },
            execution_status="INITIATED"
        )
        
        try:
            await self._reduce_portfolio_leverage(self.config.level_3_leverage_reduction)
            leverage_action.execution_status = "COMPLETED"
            leverage_action.completion_time = datetime.now()
        except Exception as e:
            leverage_action.execution_status = "FAILED"
            leverage_action.error_message = str(e)
        
        actions.append(leverage_action)
        
        # Halt new positions
        if self.config.halt_new_positions:
            halt_action = EmergencyAction(
                timestamp=timestamp,
                action_type="HALT_NEW_POSITIONS",
                emergency_level=EmergencyLevel.LEVEL_3,
                trigger_reason="Crisis pattern detected - Level 3",
                parameters={'halt_duration': 'indefinite'},
                execution_status="COMPLETED",
                completion_time=timestamp
            )
            actions.append(halt_action)
            
            # Publish position halt event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP,
                    {
                        'action': 'HALT_NEW_POSITIONS',
                        'level': 'LEVEL_3',
                        'reason': 'Crisis detection Level 3'
                    },
                    'EmergencyProtocolManager'
                )
            )
        
        # Close risky positions
        if self.config.close_risky_positions:
            close_action = EmergencyAction(
                timestamp=timestamp,
                action_type="CLOSE_RISKY_POSITIONS",
                emergency_level=EmergencyLevel.LEVEL_3,
                trigger_reason="Crisis pattern detected - Level 3",
                parameters={'risk_threshold': 0.8},
                execution_status="INITIATED"
            )
            
            try:
                await self._close_risky_positions()
                close_action.execution_status = "COMPLETED"
                close_action.completion_time = datetime.now()
            except Exception as e:
                close_action.execution_status = "FAILED"
                close_action.error_message = str(e)
            
            actions.append(close_action)
        
        # Set emergency position limits
        limit_action = EmergencyAction(
            timestamp=timestamp,
            action_type="SET_EMERGENCY_LIMITS",
            emergency_level=EmergencyLevel.LEVEL_3,
            trigger_reason="Crisis pattern detected - Level 3",
            parameters={
                'max_position_size': self.config.max_position_size_emergency,
                'max_total_exposure': 0.25
            },
            execution_status="COMPLETED",
            completion_time=timestamp
        )
        actions.append(limit_action)
        
        return actions
    
    async def _reduce_portfolio_leverage(self, reduction_percentage: float) -> bool:
        """Reduce portfolio leverage by specified percentage"""
        
        try:
            # Publish leverage reduction event for execution system
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.KELLY_SIZING,
                    {
                        'action': 'EMERGENCY_LEVERAGE_REDUCTION',
                        'reduction_percentage': reduction_percentage,
                        'emergency_level': self.emergency_state.level.value,
                        'timestamp': datetime.now().isoformat(),
                        'reason': 'Crisis detection emergency protocol'
                    },
                    'EmergencyProtocolManager'
                )
            )
            
            logger.info(f"Initiated {reduction_percentage:.0%} leverage reduction")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reduce leverage: {e}")
            return False
    
    async def _close_risky_positions(self) -> bool:
        """Close positions that exceed risk thresholds"""
        
        try:
            # Publish position closure event for execution system
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP,
                    {
                        'action': 'CLOSE_RISKY_POSITIONS',
                        'risk_threshold': 0.8,
                        'emergency_level': self.emergency_state.level.value,
                        'timestamp': datetime.now().isoformat()
                    },
                    'EmergencyProtocolManager'
                )
            )
            
            logger.info("Initiated closure of risky positions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close risky positions: {e}")
            return False
    
    async def _verify_emergency_compliance(self, position_data: any) -> bool:
        """Verify position updates comply with emergency protocols"""
        
        if self.emergency_state.status == ProtocolStatus.EMERGENCY:
            # During Level 3 emergency, verify no new positions are opened
            if hasattr(position_data, 'new_positions') and position_data.new_positions:
                logger.warning("New positions detected during emergency halt")
                
                # Publish compliance violation
                self.event_bus.publish(
                    self.event_bus.create_event(
                        EventType.SYSTEM_ERROR,
                        {
                            'type': 'EMERGENCY_COMPLIANCE_VIOLATION',
                            'message': 'New positions opened during emergency halt',
                            'emergency_level': self.emergency_state.level.value
                        },
                        'EmergencyProtocolManager'
                    )
                )
                
                return False
        
        return True
    
    async def manual_reset_emergency(self, reason: str, authorized_by: str) -> bool:
        """Manually reset emergency state (requires human authorization)"""
        
        if not self.emergency_state.manual_reset_required:
            logger.warning("Manual reset attempted but not required")
            return False
        
        try:
            # Record manual reset
            reset_action = EmergencyAction(
                timestamp=datetime.now(),
                action_type="MANUAL_RESET",
                emergency_level=self.emergency_state.level,
                trigger_reason=f"Manual reset: {reason}",
                parameters={
                    'authorized_by': authorized_by,
                    'reset_reason': reason
                },
                execution_status="COMPLETED",
                completion_time=datetime.now()
            )
            
            self.action_history.append(reset_action)
            
            # Reset emergency state
            old_state = self.emergency_state
            self.emergency_state = EmergencyState(
                status=ProtocolStatus.INACTIVE,
                level=EmergencyLevel.NONE,
                activation_time=None,
                trigger_event=None,
                active_protocols=[],
                manual_reset_required=False,
                auto_escalation_disabled=False
            )
            
            # Record state change
            self.state_history.append({
                'timestamp': datetime.now(),
                'old_state': old_state,
                'new_state': self.emergency_state,
                'trigger': f"Manual reset by {authorized_by}",
                'actions': 1
            })
            
            # Publish reset event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_UPDATE,
                    {
                        'action': 'EMERGENCY_RESET',
                        'authorized_by': authorized_by,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat()
                    },
                    'EmergencyProtocolManager'
                )
            )
            
            logger.info(f"Emergency state manually reset by {authorized_by}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset emergency state: {e}")
            return False
    
    def get_emergency_status(self) -> Dict:
        """Get current emergency status"""
        
        return {
            'status': self.emergency_state.status.value,
            'level': self.emergency_state.level.value,
            'activation_time': self.emergency_state.activation_time.isoformat() if self.emergency_state.activation_time else None,
            'trigger_event': self.emergency_state.trigger_event,
            'active_protocols': self.emergency_state.active_protocols,
            'manual_reset_required': self.emergency_state.manual_reset_required,
            'auto_escalation_disabled': self.emergency_state.auto_escalation_disabled,
            'actions_executed': len(self.action_history),
            'state_changes': len(self.state_history)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get emergency response performance statistics"""
        
        if not self.response_times:
            return {"no_responses": True}
        
        times = list(self.response_times)
        
        return {
            'avg_response_time_ms': np.mean(times),
            'max_response_time_ms': np.max(times),
            'min_response_time_ms': np.min(times),
            'target_response_time_ms': self.target_response_time_ms,
            'target_met_percentage': sum(1 for t in times if t <= self.target_response_time_ms) / len(times) * 100,
            'total_responses': len(times),
            'emergency_activations': len([s for s in self.state_history if s['new_state'].status != ProtocolStatus.INACTIVE])
        }
    
    def get_action_audit_trail(self, hours: int = 24) -> List[Dict]:
        """Get audit trail of emergency actions"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_actions = [
            {
                'timestamp': action.timestamp.isoformat(),
                'action_type': action.action_type,
                'emergency_level': action.emergency_level.value,
                'trigger_reason': action.trigger_reason,
                'parameters': action.parameters,
                'execution_status': action.execution_status,
                'completion_time': action.completion_time.isoformat() if action.completion_time else None,
                'error_message': action.error_message
            }
            for action in self.action_history
            if action.timestamp >= cutoff_time
        ]
        
        return recent_actions
    
    async def export_emergency_report(self, output_path: str) -> bool:
        """Export comprehensive emergency response report"""
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'emergency_status': self.get_emergency_status(),
                'performance_stats': self.get_performance_stats(),
                'configuration': {
                    'level_1_threshold': self.config.level_1_threshold,
                    'level_2_threshold': self.config.level_2_threshold,
                    'level_3_threshold': self.config.level_3_threshold,
                    'level_2_leverage_reduction': self.config.level_2_leverage_reduction,
                    'level_3_leverage_reduction': self.config.level_3_leverage_reduction,
                    'target_response_time_ms': self.target_response_time_ms
                },
                'action_audit_trail': self.get_action_audit_trail(168),  # Last 7 days
                'state_history': [
                    {
                        'timestamp': state['timestamp'].isoformat(),
                        'trigger': state['trigger'],
                        'old_status': state['old_state'].status.value,
                        'new_status': state['new_state'].status.value,
                        'old_level': state['old_state'].level.value,
                        'new_level': state['new_state'].level.value,
                        'actions': state['actions']
                    }
                    for state in list(self.state_history)
                ]
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Emergency report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export emergency report: {e}")
            return False