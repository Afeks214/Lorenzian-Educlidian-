"""
Agent Communication Protocol for MARL Coordination
==================================================

This module implements a standardized communication protocol for agent coordination
to fix the issues where 51 agent decisions don't convert to trades properly.

Key Features:
- Standardized message format for all agents
- Strategy signal propagation
- Conflict resolution mechanisms
- Decision synchronization
- Trade execution coordination
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the agent communication protocol"""
    STRATEGY_SIGNAL = "strategy_signal"
    AGENT_DECISION = "agent_decision"
    COORDINATION_REQUEST = "coordination_request"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TRADE_EXECUTION = "trade_execution"
    PERFORMANCE_UPDATE = "performance_update"


class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Standardized message format for agent communication"""
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast
    priority: Priority = Priority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_id: int = 0
    
    # Message content
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Strategy context
    strategy_signal: Optional[Dict[str, Any]] = None
    
    # Coordination metadata
    requires_response: bool = False
    response_timeout_ms: int = 5000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'sequence_id': self.sequence_id,
            'content': self.content,
            'strategy_signal': self.strategy_signal,
            'requires_response': self.requires_response,
            'response_timeout_ms': self.response_timeout_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            message_type=MessageType(data['message_type']),
            sender_id=data['sender_id'],
            recipient_id=data.get('recipient_id'),
            priority=Priority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            sequence_id=data['sequence_id'],
            content=data['content'],
            strategy_signal=data.get('strategy_signal'),
            requires_response=data['requires_response'],
            response_timeout_ms=data['response_timeout_ms']
        )


@dataclass
class StrategySignal:
    """Standardized strategy signal format"""
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    strength: float    # Signal strength
    source: str        # Source of signal (e.g., 'synergy_detector')
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Strategy context
    pattern_detected: Optional[str] = None
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    # Execution parameters
    urgency: float = 0.5  # 0.0 to 1.0
    time_horizon: str = "immediate"  # "immediate", "short", "medium", "long"
    risk_level: float = 0.5  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'strength': self.strength,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'pattern_detected': self.pattern_detected,
            'indicators': self.indicators,
            'urgency': self.urgency,
            'time_horizon': self.time_horizon,
            'risk_level': self.risk_level
        }


class AgentCommunicationHub:
    """Central hub for agent communication and coordination"""
    
    def __init__(self):
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.registered_agents: Dict[str, Any] = {}
        self.message_history: List[AgentMessage] = []
        self.sequence_counter = 0
        self.running = False
        
        # Strategy signal tracking
        self.current_strategy_signal: Optional[StrategySignal] = None
        self.strategy_signal_history: List[StrategySignal] = []
        
        # Coordination state
        self.pending_decisions: Dict[str, AgentMessage] = {}
        self.conflict_resolution_active = False
        
        # Performance tracking
        self.message_counts = {msg_type: 0 for msg_type in MessageType}
        self.response_times = []
        
        logger.info("AgentCommunicationHub initialized")
    
    async def start(self):
        """Start the communication hub"""
        self.running = True
        await self._message_processing_loop()
    
    async def stop(self):
        """Stop the communication hub"""
        self.running = False
        logger.info("AgentCommunicationHub stopped")
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """Register an agent with the communication hub"""
        self.registered_agents[agent_id] = agent_instance
        logger.info(f"Agent {agent_id} registered with communication hub")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the communication hub"""
        try:
            message.sequence_id = self.sequence_counter
            self.sequence_counter += 1
            
            await self.message_queue.put(message)
            self.message_history.append(message)
            self.message_counts[message.message_type] += 1
            
            # Keep history bounded
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]
            
            logger.debug(f"Message queued: {message.message_type.value} from {message.sender_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def broadcast_strategy_signal(self, strategy_signal: StrategySignal):
        """Broadcast a strategy signal to all registered agents"""
        try:
            self.current_strategy_signal = strategy_signal
            self.strategy_signal_history.append(strategy_signal)
            
            # Keep history bounded
            if len(self.strategy_signal_history) > 100:
                self.strategy_signal_history = self.strategy_signal_history[-100:]
            
            # Create broadcast message
            message = AgentMessage(
                message_type=MessageType.STRATEGY_SIGNAL,
                sender_id="strategy_coordinator",
                recipient_id=None,  # Broadcast
                priority=Priority.HIGH,
                content=strategy_signal.to_dict(),
                strategy_signal=strategy_signal.to_dict(),
                requires_response=True,
                response_timeout_ms=3000
            )
            
            await self.send_message(message)
            logger.info(f"Strategy signal broadcasted: {strategy_signal.signal_type} (confidence: {strategy_signal.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error broadcasting strategy signal: {e}")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
    
    async def _process_message(self, message: AgentMessage):
        """Process a single message"""
        try:
            start_time = datetime.now()
            
            if message.message_type == MessageType.STRATEGY_SIGNAL:
                await self._handle_strategy_signal(message)
            elif message.message_type == MessageType.AGENT_DECISION:
                await self._handle_agent_decision(message)
            elif message.message_type == MessageType.COORDINATION_REQUEST:
                await self._handle_coordination_request(message)
            elif message.message_type == MessageType.CONFLICT_RESOLUTION:
                await self._handle_conflict_resolution(message)
            elif message.message_type == MessageType.TRADE_EXECUTION:
                await self._handle_trade_execution(message)
            elif message.message_type == MessageType.PERFORMANCE_UPDATE:
                await self._handle_performance_update(message)
            
            # Track response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_type.value}: {e}")
    
    async def _handle_strategy_signal(self, message: AgentMessage):
        """Handle strategy signal messages"""
        try:
            strategy_signal = StrategySignal(**message.content)
            
            # Notify all registered agents
            for agent_id, agent in self.registered_agents.items():
                if hasattr(agent, 'handle_strategy_signal'):
                    try:
                        await agent.handle_strategy_signal(strategy_signal)
                    except Exception as e:
                        logger.error(f"Error notifying agent {agent_id} of strategy signal: {e}")
            
            logger.info(f"Strategy signal processed: {strategy_signal.signal_type}")
            
        except Exception as e:
            logger.error(f"Error handling strategy signal: {e}")
    
    async def _handle_agent_decision(self, message: AgentMessage):
        """Handle agent decision messages"""
        try:
            agent_id = message.sender_id
            decision_data = message.content
            
            # Store pending decision
            self.pending_decisions[agent_id] = message
            
            # Check if we have all required decisions
            required_agents = ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer', 'routing']
            if all(agent in self.pending_decisions for agent in required_agents):
                await self._coordinate_decisions()
            
            logger.debug(f"Agent decision received from {agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling agent decision: {e}")
    
    async def _handle_coordination_request(self, message: AgentMessage):
        """Handle coordination request messages"""
        try:
            request_type = message.content.get('request_type')
            
            if request_type == 'conflict_resolution':
                await self._start_conflict_resolution(message)
            elif request_type == 'decision_sync':
                await self._sync_decisions(message)
            
            logger.debug(f"Coordination request handled: {request_type}")
            
        except Exception as e:
            logger.error(f"Error handling coordination request: {e}")
    
    async def _handle_conflict_resolution(self, message: AgentMessage):
        """Handle conflict resolution messages"""
        try:
            resolution_data = message.content
            
            # Apply conflict resolution
            if resolution_data.get('resolution_type') == 'strategy_priority':
                await self._apply_strategy_priority_resolution(resolution_data)
            elif resolution_data.get('resolution_type') == 'weighted_average':
                await self._apply_weighted_average_resolution(resolution_data)
            
            logger.info("Conflict resolution applied")
            
        except Exception as e:
            logger.error(f"Error handling conflict resolution: {e}")
    
    async def _handle_trade_execution(self, message: AgentMessage):
        """Handle trade execution messages"""
        try:
            execution_data = message.content
            
            # Validate trade execution
            if self._validate_trade_execution(execution_data):
                await self._execute_trade(execution_data)
            else:
                logger.warning("Trade execution validation failed")
            
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
    
    async def _handle_performance_update(self, message: AgentMessage):
        """Handle performance update messages"""
        try:
            performance_data = message.content
            agent_id = message.sender_id
            
            # Update performance tracking
            logger.debug(f"Performance update from {agent_id}: {performance_data}")
            
        except Exception as e:
            logger.error(f"Error handling performance update: {e}")
    
    async def _coordinate_decisions(self):
        """Coordinate decisions from all agents"""
        try:
            # Check for conflicts
            conflicts = self._detect_conflicts()
            
            if conflicts:
                await self._resolve_conflicts(conflicts)
            else:
                # No conflicts, proceed with execution
                await self._execute_coordinated_decision()
            
        except Exception as e:
            logger.error(f"Error coordinating decisions: {e}")
    
    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between agent decisions"""
        conflicts = []
        
        try:
            # Check for conflicting signals
            decisions = list(self.pending_decisions.values())
            
            # Extract action preferences
            actions = []
            for decision in decisions:
                content = decision.content
                if 'action' in content:
                    actions.append(content['action'])
                elif 'signal_type' in content:
                    actions.append(content['signal_type'])
            
            # Simple conflict detection: if actions are opposite
            if 'buy' in actions and 'sell' in actions:
                conflicts.append({
                    'type': 'opposite_actions',
                    'actions': actions,
                    'severity': 'high'
                })
            
            # Check confidence conflicts
            confidences = []
            for decision in decisions:
                content = decision.content
                if 'confidence' in content:
                    confidences.append(content['confidence'])
            
            if confidences and max(confidences) - min(confidences) > 0.5:
                conflicts.append({
                    'type': 'confidence_variance',
                    'confidence_range': [min(confidences), max(confidences)],
                    'severity': 'medium'
                })
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
        
        return conflicts
    
    async def _resolve_conflicts(self, conflicts: List[Dict[str, Any]]):
        """Resolve conflicts between agent decisions"""
        try:
            for conflict in conflicts:
                if conflict['type'] == 'opposite_actions':
                    # Use strategy signal to resolve
                    if self.current_strategy_signal:
                        await self._apply_strategy_priority_resolution(conflict)
                    else:
                        await self._apply_weighted_average_resolution(conflict)
                elif conflict['type'] == 'confidence_variance':
                    # Use confidence weighting
                    await self._apply_confidence_weighting(conflict)
            
            logger.info(f"Resolved {len(conflicts)} conflicts")
            
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
    
    async def _apply_strategy_priority_resolution(self, conflict: Dict[str, Any]):
        """Apply strategy priority conflict resolution"""
        try:
            if self.current_strategy_signal:
                # Override agent decisions with strategy signal
                strategy_action = self.current_strategy_signal.signal_type
                
                # Update all pending decisions to align with strategy
                for agent_id, decision in self.pending_decisions.items():
                    decision.content['action'] = strategy_action
                    decision.content['strategy_override'] = True
                    decision.content['original_action'] = decision.content.get('action')
                
                logger.info(f"Applied strategy priority resolution: {strategy_action}")
            
        except Exception as e:
            logger.error(f"Error applying strategy priority resolution: {e}")
    
    async def _apply_weighted_average_resolution(self, conflict: Dict[str, Any]):
        """Apply weighted average conflict resolution"""
        try:
            # Calculate weighted average of decisions
            total_weight = 0
            weighted_sum = 0
            
            for agent_id, decision in self.pending_decisions.items():
                content = decision.content
                confidence = content.get('confidence', 0.5)
                action_value = self._action_to_value(content.get('action', 'hold'))
                
                weighted_sum += action_value * confidence
                total_weight += confidence
            
            if total_weight > 0:
                avg_action_value = weighted_sum / total_weight
                final_action = self._value_to_action(avg_action_value)
                
                # Update all decisions with resolved action
                for agent_id, decision in self.pending_decisions.items():
                    decision.content['action'] = final_action
                    decision.content['conflict_resolved'] = True
                
                logger.info(f"Applied weighted average resolution: {final_action}")
            
        except Exception as e:
            logger.error(f"Error applying weighted average resolution: {e}")
    
    async def _apply_confidence_weighting(self, conflict: Dict[str, Any]):
        """Apply confidence-based weighting"""
        try:
            # Weight decisions by confidence
            for agent_id, decision in self.pending_decisions.items():
                content = decision.content
                confidence = content.get('confidence', 0.5)
                
                # Adjust action strength based on confidence
                if 'action_strength' not in content:
                    content['action_strength'] = 1.0
                
                content['action_strength'] *= confidence
            
            logger.info("Applied confidence weighting")
            
        except Exception as e:
            logger.error(f"Error applying confidence weighting: {e}")
    
    def _action_to_value(self, action: str) -> float:
        """Convert action to numeric value"""
        action_map = {
            'strong_sell': -1.0,
            'sell': -0.5,
            'hold': 0.0,
            'buy': 0.5,
            'strong_buy': 1.0
        }
        return action_map.get(action.lower(), 0.0)
    
    def _value_to_action(self, value: float) -> str:
        """Convert numeric value to action"""
        if value <= -0.75:
            return 'strong_sell'
        elif value <= -0.25:
            return 'sell'
        elif value <= 0.25:
            return 'hold'
        elif value <= 0.75:
            return 'buy'
        else:
            return 'strong_buy'
    
    async def _execute_coordinated_decision(self):
        """Execute the coordinated decision"""
        try:
            # Create final execution message
            execution_data = {
                'decisions': {agent_id: msg.content for agent_id, msg in self.pending_decisions.items()},
                'timestamp': datetime.now().isoformat(),
                'coordination_successful': True
            }
            
            # Send execution message
            execution_message = AgentMessage(
                message_type=MessageType.TRADE_EXECUTION,
                sender_id="coordination_hub",
                priority=Priority.HIGH,
                content=execution_data
            )
            
            await self.send_message(execution_message)
            
            # Clear pending decisions
            self.pending_decisions.clear()
            
            logger.info("Coordinated decision executed")
            
        except Exception as e:
            logger.error(f"Error executing coordinated decision: {e}")
    
    def _validate_trade_execution(self, execution_data: Dict[str, Any]) -> bool:
        """Validate trade execution data"""
        try:
            # Check required fields
            if 'decisions' not in execution_data:
                return False
            
            # Check that all required agents provided decisions
            required_agents = ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer', 'routing']
            decisions = execution_data['decisions']
            
            for agent in required_agents:
                if agent not in decisions:
                    logger.warning(f"Missing decision from agent: {agent}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade execution: {e}")
            return False
    
    async def _execute_trade(self, execution_data: Dict[str, Any]):
        """Execute the trade"""
        try:
            # This would integrate with the actual trading system
            logger.info("Trade execution initiated")
            
            # For now, just log the execution
            decisions = execution_data['decisions']
            for agent_id, decision in decisions.items():
                logger.info(f"Agent {agent_id} decision: {decision}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication hub status"""
        return {
            'running': self.running,
            'registered_agents': list(self.registered_agents.keys()),
            'message_queue_size': self.message_queue.qsize(),
            'pending_decisions': list(self.pending_decisions.keys()),
            'current_strategy_signal': self.current_strategy_signal.to_dict() if self.current_strategy_signal else None,
            'message_counts': {msg_type.value: count for msg_type, count in self.message_counts.items()},
            'avg_response_time_ms': sum(self.response_times) / len(self.response_times) if self.response_times else 0
        }


# Global communication hub instance
communication_hub = AgentCommunicationHub()