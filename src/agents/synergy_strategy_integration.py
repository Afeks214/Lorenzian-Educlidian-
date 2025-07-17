"""
Synergy Strategy Integration for MARL Agents
============================================

This module ensures that MARL agents properly support synergy strategy decisions
instead of overriding them, fixing the issue where agent decisions don't convert
to trades properly.

Key Features:
- Strategy signal interpretation
- Agent decision modification to support strategy
- Conflict resolution between strategy and agent decisions
- Trade execution coordination
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .agent_communication_protocol import (
    AgentCommunicationHub, StrategySignal, MessageType, Priority, AgentMessage,
    communication_hub
)

logger = logging.getLogger(__name__)


@dataclass
class SynergyPattern:
    """Represents a detected synergy pattern"""
    pattern_type: str
    confidence: float
    strength: float
    indicators: Dict[str, Any]
    timestamp: datetime
    
    def to_strategy_signal(self) -> StrategySignal:
        """Convert synergy pattern to strategy signal"""
        # Map pattern types to trading signals
        signal_mapping = {
            'synergy_bullish': 'buy',
            'synergy_bearish': 'sell',
            'synergy_neutral': 'hold',
            'synergy_strong_bullish': 'strong_buy',
            'synergy_strong_bearish': 'strong_sell'
        }
        
        signal_type = signal_mapping.get(self.pattern_type, 'hold')
        
        return StrategySignal(
            signal_type=signal_type,
            confidence=self.confidence,
            strength=self.strength,
            source='synergy_detector',
            timestamp=self.timestamp,
            pattern_detected=self.pattern_type,
            indicators=self.indicators,
            urgency=min(1.0, self.strength * 1.5),
            time_horizon='immediate' if self.strength > 0.8 else 'short',
            risk_level=self.strength * 0.5
        )


class SynergyStrategyCoordinator:
    """Coordinates synergy strategy with MARL agents"""
    
    def __init__(self, communication_hub: AgentCommunicationHub):
        self.communication_hub = communication_hub
        self.current_synergy_pattern: Optional[SynergyPattern] = None
        self.strategy_active = False
        self.agent_overrides: Dict[str, bool] = {}
        
        # Strategy parameters
        self.min_confidence_threshold = 0.6
        self.override_threshold = 0.8
        self.strategy_timeout_seconds = 300  # 5 minutes
        
        # Performance tracking
        self.strategy_signals_sent = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        logger.info("SynergyStrategyCoordinator initialized")
    
    async def process_synergy_detection(self, synergy_data: Dict[str, Any]):
        """Process synergy detection and coordinate with agents"""
        try:
            # Extract synergy pattern
            pattern = self._extract_synergy_pattern(synergy_data)
            
            if pattern and pattern.confidence >= self.min_confidence_threshold:
                await self._activate_strategy(pattern)
            else:
                logger.debug(f"Synergy pattern below threshold: {pattern.confidence if pattern else 'None'}")
                
        except Exception as e:
            logger.error(f"Error processing synergy detection: {e}")
    
    def _extract_synergy_pattern(self, synergy_data: Dict[str, Any]) -> Optional[SynergyPattern]:
        """Extract synergy pattern from detection data"""
        try:
            # This would be customized based on the actual synergy detection format
            pattern_type = synergy_data.get('pattern_type', 'synergy_neutral')
            confidence = synergy_data.get('confidence', 0.5)
            strength = synergy_data.get('strength', 0.5)
            indicators = synergy_data.get('indicators', {})
            
            return SynergyPattern(
                pattern_type=pattern_type,
                confidence=confidence,
                strength=strength,
                indicators=indicators,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error extracting synergy pattern: {e}")
            return None
    
    async def _activate_strategy(self, pattern: SynergyPattern):
        """Activate strategy based on synergy pattern"""
        try:
            self.current_synergy_pattern = pattern
            self.strategy_active = True
            
            # Convert to strategy signal
            strategy_signal = pattern.to_strategy_signal()
            
            # Broadcast to all agents
            await self.communication_hub.broadcast_strategy_signal(strategy_signal)
            
            self.strategy_signals_sent += 1
            
            logger.info(f"Strategy activated: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
            
            # Set timeout for strategy
            asyncio.create_task(self._strategy_timeout_handler())
            
        except Exception as e:
            logger.error(f"Error activating strategy: {e}")
    
    async def _strategy_timeout_handler(self):
        """Handle strategy timeout"""
        try:
            await asyncio.sleep(self.strategy_timeout_seconds)
            
            if self.strategy_active:
                self.strategy_active = False
                self.current_synergy_pattern = None
                logger.info("Strategy timed out")
                
        except Exception as e:
            logger.error(f"Error in strategy timeout handler: {e}")
    
    async def modify_agent_decision(self, agent_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Modify agent decision to support strategy"""
        try:
            if not self.strategy_active or not self.current_synergy_pattern:
                return decision
            
            strategy_signal = self.current_synergy_pattern.to_strategy_signal()
            
            # Modify decision based on strategy
            modified_decision = decision.copy()
            
            # Override action if strategy confidence is high
            if strategy_signal.confidence >= self.override_threshold:
                original_action = modified_decision.get('action', 'hold')
                modified_decision['action'] = strategy_signal.signal_type
                modified_decision['strategy_override'] = True
                modified_decision['original_action'] = original_action
                
                # Boost confidence for strategy support
                original_confidence = modified_decision.get('confidence', 0.5)
                modified_decision['confidence'] = min(1.0, original_confidence + strategy_signal.confidence * 0.3)
                
                logger.info(f"Agent {agent_id} action overridden: {original_action} -> {strategy_signal.signal_type}")
            
            else:
                # Modify probabilities to support strategy
                if 'action_probabilities' in modified_decision:
                    probs = np.array(modified_decision['action_probabilities'])
                    
                    # Boost probability for strategy action
                    action_index = self._get_action_index(strategy_signal.signal_type)
                    if action_index is not None:
                        boost_factor = strategy_signal.confidence * 0.5
                        probs[action_index] += boost_factor
                        
                        # Normalize probabilities
                        probs = probs / np.sum(probs)
                        modified_decision['action_probabilities'] = probs.tolist()
                        
                        logger.debug(f"Agent {agent_id} probabilities modified to support strategy")
            
            # Add strategy context
            modified_decision['strategy_support'] = True
            modified_decision['strategy_signal'] = strategy_signal.to_dict()
            modified_decision['strategy_confidence'] = strategy_signal.confidence
            
            return modified_decision
            
        except Exception as e:
            logger.error(f"Error modifying agent decision for {agent_id}: {e}")
            return decision
    
    def _get_action_index(self, action: str) -> Optional[int]:
        """Get action index for probability modification"""
        action_map = {
            'buy': 0,
            'hold': 1,
            'sell': 2,
            'strong_buy': 0,
            'strong_sell': 2
        }
        return action_map.get(action.lower())
    
    async def handle_trade_execution(self, execution_data: Dict[str, Any]) -> bool:
        """Handle trade execution with strategy coordination"""
        try:
            if not self.strategy_active:
                return True  # No strategy active, proceed normally
            
            # Validate execution aligns with strategy
            if self._validate_execution_alignment(execution_data):
                self.successful_executions += 1
                logger.info("Trade execution aligned with strategy")
                return True
            else:
                self.failed_executions += 1
                logger.warning("Trade execution NOT aligned with strategy")
                return False
                
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
            return False
    
    def _validate_execution_alignment(self, execution_data: Dict[str, Any]) -> bool:
        """Validate that execution aligns with strategy"""
        try:
            if not self.current_synergy_pattern:
                return True
            
            strategy_signal = self.current_synergy_pattern.to_strategy_signal()
            decisions = execution_data.get('decisions', {})
            
            # Check if majority of agents support strategy
            supporting_agents = 0
            total_agents = len(decisions)
            
            for agent_id, decision in decisions.items():
                if decision.get('strategy_support', False):
                    supporting_agents += 1
                elif decision.get('action') == strategy_signal.signal_type:
                    supporting_agents += 1
            
            support_ratio = supporting_agents / max(1, total_agents)
            required_support = 0.6  # 60% of agents should support strategy
            
            return support_ratio >= required_support
            
        except Exception as e:
            logger.error(f"Error validating execution alignment: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy coordinator status"""
        return {
            'strategy_active': self.strategy_active,
            'current_pattern': self.current_synergy_pattern.pattern_type if self.current_synergy_pattern else None,
            'pattern_confidence': self.current_synergy_pattern.confidence if self.current_synergy_pattern else 0.0,
            'signals_sent': self.strategy_signals_sent,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.successful_executions / max(1, self.successful_executions + self.failed_executions),
            'agent_overrides': self.agent_overrides.copy()
        }


class StrategyAwareAgent:
    """Base class for strategy-aware agents"""
    
    def __init__(self, agent_id: str, communication_hub: AgentCommunicationHub):
        self.agent_id = agent_id
        self.communication_hub = communication_hub
        self.current_strategy_signal: Optional[StrategySignal] = None
        self.strategy_support_enabled = True
        
        # Register with communication hub
        self.communication_hub.register_agent(agent_id, self)
        
        logger.info(f"StrategyAwareAgent {agent_id} initialized")
    
    async def handle_strategy_signal(self, strategy_signal: StrategySignal):
        """Handle incoming strategy signal"""
        try:
            self.current_strategy_signal = strategy_signal
            
            if self.strategy_support_enabled:
                logger.info(f"Agent {self.agent_id} received strategy signal: {strategy_signal.signal_type}")
            
        except Exception as e:
            logger.error(f"Error handling strategy signal in {self.agent_id}: {e}")
    
    async def make_strategy_aware_decision(self, base_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision that considers strategy signal"""
        try:
            if not self.strategy_support_enabled or not self.current_strategy_signal:
                return base_decision
            
            # Get strategy coordinator
            coordinator = SynergyStrategyCoordinator(self.communication_hub)
            
            # Modify decision to support strategy
            modified_decision = await coordinator.modify_agent_decision(
                self.agent_id, base_decision
            )
            
            return modified_decision
            
        except Exception as e:
            logger.error(f"Error making strategy-aware decision in {self.agent_id}: {e}")
            return base_decision
    
    async def send_decision(self, decision: Dict[str, Any]):
        """Send decision through communication hub"""
        try:
            message = AgentMessage(
                message_type=MessageType.AGENT_DECISION,
                sender_id=self.agent_id,
                priority=Priority.HIGH,
                content=decision,
                strategy_signal=self.current_strategy_signal.to_dict() if self.current_strategy_signal else None
            )
            
            await self.communication_hub.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending decision from {self.agent_id}: {e}")


# Global strategy coordinator instance
strategy_coordinator = SynergyStrategyCoordinator(communication_hub)


async def initialize_strategy_integration():
    """Initialize strategy integration system"""
    try:
        # Start communication hub
        asyncio.create_task(communication_hub.start())
        
        logger.info("Strategy integration system initialized")
        
    except Exception as e:
        logger.error(f"Error initializing strategy integration: {e}")


def create_strategy_aware_agent(agent_id: str) -> StrategyAwareAgent:
    """Create a strategy-aware agent"""
    return StrategyAwareAgent(agent_id, communication_hub)