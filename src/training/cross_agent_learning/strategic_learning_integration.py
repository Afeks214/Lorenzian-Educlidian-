"""
Strategic Agent Learning Integration System
==========================================

This module implements learning integration for strategic agents, enabling them
to adapt their behavior based on execution outcomes and performance feedback
from tactical and execution layers.

Key Features:
- Long-term performance learning
- Regime-specific adaptation
- Cross-timeframe signal integration
- Strategic pattern recognition
- Hierarchical reward integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import structlog

from .outcome_propagation_system import LearningSignal, AgentLevel, OutcomeType

logger = structlog.get_logger(__name__)


class StrategicLearningMode(Enum):
    """Strategic learning modes"""
    REGIME_ADAPTATION = "regime_adaptation"
    PATTERN_REINFORCEMENT = "pattern_reinforcement"
    LONG_TERM_OPTIMIZATION = "long_term_optimization"
    CROSS_TIMEFRAME_INTEGRATION = "cross_timeframe_integration"
    RISK_ADJUSTED_LEARNING = "risk_adjusted_learning"


@dataclass
class StrategicLearningState:
    """State of strategic learning for an agent"""
    agent_id: str
    last_update: datetime
    
    # Performance metrics
    cumulative_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.5
    
    # Learning adaptations
    regime_weights: Dict[str, float] = field(default_factory=dict)
    pattern_strengths: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Adaptation parameters
    confidence_adjustment: float = 0.0
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    
    # Memory components
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failed_patterns: List[Dict[str, Any]] = field(default_factory=list)
    regime_performance: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'last_update': self.last_update.isoformat(),
            'cumulative_pnl': self.cumulative_pnl,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'regime_weights': self.regime_weights,
            'pattern_strengths': self.pattern_strengths,
            'feature_importance': self.feature_importance,
            'confidence_adjustment': self.confidence_adjustment,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'successful_patterns': self.successful_patterns,
            'failed_patterns': self.failed_patterns,
            'regime_performance': dict(self.regime_performance)
        }


class StrategicPatternLearner:
    """Learns and recognizes strategic patterns from execution outcomes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_memory_size = config.get('pattern_memory_size', 100)
        self.pattern_similarity_threshold = config.get('pattern_similarity_threshold', 0.7)
        self.pattern_success_threshold = config.get('pattern_success_threshold', 0.6)
        
        # Pattern storage
        self.pattern_library = {}
        self.pattern_performance = defaultdict(list)
        
        logger.info("StrategicPatternLearner initialized")
    
    def learn_pattern(self, pattern_data: Dict[str, Any], outcome_success: float):
        """
        Learn a new strategic pattern from execution outcomes
        
        Args:
            pattern_data: Pattern characteristics and context
            outcome_success: Success score of the outcome
        """
        try:
            pattern_id = self._generate_pattern_id(pattern_data)
            
            # Store pattern if new or update existing
            if pattern_id not in self.pattern_library:
                self.pattern_library[pattern_id] = {
                    'pattern_data': pattern_data,
                    'first_seen': datetime.now(),
                    'occurrence_count': 0,
                    'success_scores': []
                }
            
            # Update pattern performance
            pattern_info = self.pattern_library[pattern_id]
            pattern_info['occurrence_count'] += 1
            pattern_info['success_scores'].append(outcome_success)
            
            # Keep success scores bounded
            if len(pattern_info['success_scores']) > self.pattern_memory_size:
                pattern_info['success_scores'].pop(0)
            
            # Update pattern performance tracking
            self.pattern_performance[pattern_id].append(outcome_success)
            
            logger.debug("Pattern learned",
                        pattern_id=pattern_id,
                        occurrence_count=pattern_info['occurrence_count'],
                        avg_success=np.mean(pattern_info['success_scores']))
            
        except Exception as e:
            logger.error("Error learning pattern", error=str(e))
    
    def recognize_pattern(self, current_data: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """
        Recognize if current data matches known patterns
        
        Args:
            current_data: Current market/agent state data
            
        Returns:
            Tuple of (pattern_id, similarity_score) if match found
        """
        try:
            best_match = None
            best_similarity = 0.0
            
            for pattern_id, pattern_info in self.pattern_library.items():
                similarity = self._calculate_pattern_similarity(
                    current_data, pattern_info['pattern_data']
                )
                
                if similarity > best_similarity and similarity > self.pattern_similarity_threshold:
                    best_similarity = similarity
                    best_match = pattern_id
            
            if best_match:
                logger.debug("Pattern recognized",
                            pattern_id=best_match,
                            similarity=best_similarity)
                return best_match, best_similarity
            
            return None
            
        except Exception as e:
            logger.error("Error recognizing pattern", error=str(e))
            return None
    
    def get_pattern_performance(self, pattern_id: str) -> Dict[str, Any]:
        """Get performance statistics for a pattern"""
        if pattern_id not in self.pattern_library:
            return {}
        
        pattern_info = self.pattern_library[pattern_id]
        success_scores = pattern_info['success_scores']
        
        return {
            'pattern_id': pattern_id,
            'occurrence_count': pattern_info['occurrence_count'],
            'average_success': np.mean(success_scores) if success_scores else 0.0,
            'success_std': np.std(success_scores) if success_scores else 0.0,
            'last_success': success_scores[-1] if success_scores else 0.0,
            'is_reliable': len(success_scores) >= 5 and np.mean(success_scores) > self.pattern_success_threshold
        }
    
    def _generate_pattern_id(self, pattern_data: Dict[str, Any]) -> str:
        """Generate unique pattern ID from pattern data"""
        # Simple hash-based ID generation
        key_features = []
        for key in sorted(pattern_data.keys()):
            if isinstance(pattern_data[key], (int, float)):
                # Discretize continuous values
                key_features.append(f"{key}_{int(pattern_data[key] * 10)}")
            else:
                key_features.append(f"{key}_{pattern_data[key]}")
        
        return "_".join(key_features)[:50]  # Limit length
    
    def _calculate_pattern_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two pattern data sets"""
        common_keys = set(data1.keys()) & set(data2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = data1[key], data2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                if val1 == val2 == 0:
                    sim = 1.0
                else:
                    sim = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-8)
            elif val1 == val2:
                sim = 1.0
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0


class StrategicLearningIntegrator:
    """Integrates learning signals into strategic agent behavior"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.01)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.memory_decay = config.get('memory_decay', 0.95)
        
        # Learning state for each agent
        self.agent_states = {}
        
        # Pattern learning
        self.pattern_learner = StrategicPatternLearner(config.get('pattern_learning', {}))
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.adaptation_history = defaultdict(list)
        
        logger.info("StrategicLearningIntegrator initialized")
    
    def get_or_create_agent_state(self, agent_id: str) -> StrategicLearningState:
        """Get or create learning state for an agent"""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = StrategicLearningState(
                agent_id=agent_id,
                last_update=datetime.now()
            )
        return self.agent_states[agent_id]
    
    def process_learning_signal(self, signal: LearningSignal, agent_id: str) -> Dict[str, Any]:
        """
        Process a learning signal and update agent behavior
        
        Args:
            signal: Learning signal from execution outcomes
            agent_id: Target agent ID
            
        Returns:
            Dictionary of behavioral adaptations
        """
        try:
            state = self.get_or_create_agent_state(agent_id)
            adaptations = {}
            
            # Process different signal types
            if signal.signal_type == 'performance_feedback':
                adaptations.update(self._process_performance_feedback(signal, state))
            elif signal.signal_type == 'regime_adaptation':
                adaptations.update(self._process_regime_adaptation(signal, state))
            elif signal.signal_type == 'pattern_feedback':
                adaptations.update(self._process_pattern_feedback(signal, state))
            elif signal.signal_type == 'coordination_feedback':
                adaptations.update(self._process_coordination_feedback(signal, state))
            
            # Update agent state
            state.last_update = datetime.now()
            state.confidence_adjustment += signal.confidence_adjustment * self.adaptation_rate
            
            # Apply memory decay
            self._apply_memory_decay(state)
            
            # Record adaptation
            self.adaptation_history[agent_id].append({
                'timestamp': datetime.now(),
                'signal_type': signal.signal_type,
                'adaptations': adaptations,
                'signal_priority': signal.priority
            })
            
            logger.debug("Learning signal processed",
                        agent_id=agent_id,
                        signal_type=signal.signal_type,
                        adaptations=list(adaptations.keys()))
            
            return adaptations
            
        except Exception as e:
            logger.error("Error processing learning signal",
                        agent_id=agent_id,
                        signal_id=signal.signal_id,
                        error=str(e))
            return {}
    
    def _process_performance_feedback(self, signal: LearningSignal, state: StrategicLearningState) -> Dict[str, Any]:
        """Process performance feedback signals"""
        adaptations = {}
        
        try:
            # Update performance metrics
            reward = signal.reward_adjustment
            state.cumulative_pnl += reward
            
            # Update feature importance
            for feature, importance in signal.feature_importance.items():
                current_importance = state.feature_importance.get(feature, 0.0)
                state.feature_importance[feature] = (
                    current_importance * (1 - self.learning_rate) + 
                    importance * self.learning_rate
                )
            
            # Update pattern strengths
            for pattern, strength in signal.pattern_feedback.items():
                current_strength = state.pattern_strengths.get(pattern, 0.0)
                state.pattern_strengths[pattern] = (
                    current_strength * (1 - self.learning_rate) +
                    strength * self.learning_rate
                )
            
            # Generate adaptations
            if reward > 0:
                adaptations['reinforce_features'] = signal.feature_importance
                adaptations['increase_confidence'] = min(abs(reward) * 0.1, 0.1)
            else:
                adaptations['reduce_feature_weights'] = signal.feature_importance
                adaptations['decrease_confidence'] = min(abs(reward) * 0.1, 0.1)
            
            # Pattern-based adaptations
            if signal.pattern_feedback:
                adaptations['pattern_adjustments'] = signal.pattern_feedback
            
        except Exception as e:
            logger.error("Error processing performance feedback", error=str(e))
        
        return adaptations
    
    def _process_regime_adaptation(self, signal: LearningSignal, state: StrategicLearningState) -> Dict[str, Any]:
        """Process regime adaptation signals"""
        adaptations = {}
        
        try:
            # Update regime weights based on behavior adjustment
            behavior_adj = signal.behavior_adjustment
            
            if 'market_regime_weight' in behavior_adj:
                # Determine current regime (simplified)
                current_regime = 'default'  # Would be determined from market data
                
                current_weight = state.regime_weights.get(current_regime, 1.0)
                new_weight = current_weight * (1 - self.adaptation_rate) + behavior_adj['market_regime_weight'] * self.adaptation_rate
                state.regime_weights[current_regime] = new_weight
                
                adaptations['regime_weights'] = state.regime_weights
            
            if behavior_adj.get('reduce_position_size'):
                adaptations['position_size_multiplier'] = 0.8
            
            if behavior_adj.get('increase_regime_sensitivity'):
                adaptations['regime_sensitivity_multiplier'] = 1.2
            
            # Confidence adjustment for regime uncertainty
            if signal.confidence_adjustment < 0:
                adaptations['regime_confidence_penalty'] = abs(signal.confidence_adjustment)
            
        except Exception as e:
            logger.error("Error processing regime adaptation", error=str(e))
        
        return adaptations
    
    def _process_pattern_feedback(self, signal: LearningSignal, state: StrategicLearningState) -> Dict[str, Any]:
        """Process pattern feedback signals"""
        adaptations = {}
        
        try:
            # Update pattern memory
            for pattern, feedback in signal.pattern_feedback.items():
                if feedback > 0.5:  # Successful pattern
                    pattern_data = {'pattern_type': pattern, 'success': True}
                    if len(state.successful_patterns) < 50:
                        state.successful_patterns.append(pattern_data)
                    else:
                        state.successful_patterns.pop(0)
                        state.successful_patterns.append(pattern_data)
                else:  # Failed pattern
                    pattern_data = {'pattern_type': pattern, 'success': False}
                    if len(state.failed_patterns) < 50:
                        state.failed_patterns.append(pattern_data)
                    else:
                        state.failed_patterns.pop(0)
                        state.failed_patterns.append(pattern_data)
                
                # Update pattern strengths
                current_strength = state.pattern_strengths.get(pattern, 0.0)
                state.pattern_strengths[pattern] = (
                    current_strength * (1 - self.learning_rate) +
                    feedback * self.learning_rate
                )
            
            # Generate pattern-based adaptations
            adaptations['pattern_memory_update'] = True
            adaptations['pattern_strengths'] = state.pattern_strengths
            
        except Exception as e:
            logger.error("Error processing pattern feedback", error=str(e))
        
        return adaptations
    
    def _process_coordination_feedback(self, signal: LearningSignal, state: StrategicLearningState) -> Dict[str, Any]:
        """Process coordination feedback signals"""
        adaptations = {}
        
        try:
            behavior_adj = signal.behavior_adjustment
            
            if behavior_adj.get('improve_coordination'):
                adaptations['coordination_weight'] = 1.0 + behavior_adj.get('coordination_weight', 0.1)
            
            if behavior_adj.get('increase_communication'):
                adaptations['communication_frequency'] = 1.5
            
            # Adjust exploration based on coordination success
            if signal.reward_adjustment < 0:
                # Poor coordination - reduce exploration
                state.exploration_rate *= 0.9
                adaptations['exploration_rate'] = state.exploration_rate
            
        except Exception as e:
            logger.error("Error processing coordination feedback", error=str(e))
        
        return adaptations
    
    def _apply_memory_decay(self, state: StrategicLearningState):
        """Apply memory decay to prevent overfitting to recent events"""
        try:
            # Decay feature importance
            for feature in state.feature_importance:
                state.feature_importance[feature] *= self.memory_decay
            
            # Decay pattern strengths
            for pattern in state.pattern_strengths:
                state.pattern_strengths[pattern] *= self.memory_decay
            
            # Decay confidence adjustment
            state.confidence_adjustment *= self.memory_decay
            
        except Exception as e:
            logger.error("Error applying memory decay", error=str(e))
    
    def get_agent_adaptations(self, agent_id: str) -> Dict[str, Any]:
        """Get current adaptations for an agent"""
        if agent_id not in self.agent_states:
            return {}
        
        state = self.agent_states[agent_id]
        return {
            'feature_importance': state.feature_importance,
            'pattern_strengths': state.pattern_strengths,
            'regime_weights': state.regime_weights,
            'confidence_adjustment': state.confidence_adjustment,
            'exploration_rate': state.exploration_rate,
            'successful_patterns': len(state.successful_patterns),
            'failed_patterns': len(state.failed_patterns)
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        return {
            'total_agents': len(self.agent_states),
            'adaptation_history_size': {
                agent_id: len(history) 
                for agent_id, history in self.adaptation_history.items()
            },
            'average_confidence_adjustment': np.mean([
                state.confidence_adjustment 
                for state in self.agent_states.values()
            ]) if self.agent_states else 0.0,
            'pattern_library_size': len(self.pattern_learner.pattern_library)
        }


class StrategicLearningManager:
    """Main manager for strategic agent learning integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integrator = StrategicLearningIntegrator(config.get('integrator', {}))
        
        # Agent registration
        self.registered_agents = {}
        
        # Learning modes
        self.learning_modes = {
            agent_id: StrategicLearningMode.LONG_TERM_OPTIMIZATION
            for agent_id in config.get('default_agents', [])
        }
        
        # Performance tracking
        self.performance_tracker = defaultdict(list)
        
        logger.info("StrategicLearningManager initialized")
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """Register a strategic agent for learning"""
        self.registered_agents[agent_id] = agent_instance
        self.learning_modes[agent_id] = StrategicLearningMode.LONG_TERM_OPTIMIZATION
        
        logger.info("Strategic agent registered for learning", agent_id=agent_id)
    
    def unregister_agent(self, agent_id: str):
        """Unregister a strategic agent"""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
        if agent_id in self.learning_modes:
            del self.learning_modes[agent_id]
        
        logger.info("Strategic agent unregistered", agent_id=agent_id)
    
    async def process_learning_signal(self, signal: LearningSignal):
        """Process a learning signal for strategic agents"""
        if signal.target_agent_level != AgentLevel.STRATEGIC:
            return
        
        try:
            # Process for specific agents or all strategic agents
            target_agents = (
                signal.target_agents 
                if signal.target_agents != ['all'] 
                else list(self.registered_agents.keys())
            )
            
            for agent_id in target_agents:
                if agent_id in self.registered_agents:
                    adaptations = self.integrator.process_learning_signal(signal, agent_id)
                    
                    # Apply adaptations to agent
                    await self._apply_adaptations(agent_id, adaptations)
                    
                    # Track performance
                    self.performance_tracker[agent_id].append({
                        'timestamp': datetime.now(),
                        'signal_type': signal.signal_type,
                        'adaptations': list(adaptations.keys()),
                        'reward_adjustment': signal.reward_adjustment
                    })
            
        except Exception as e:
            logger.error("Error processing strategic learning signal",
                        signal_id=signal.signal_id,
                        error=str(e))
    
    async def _apply_adaptations(self, agent_id: str, adaptations: Dict[str, Any]):
        """Apply learning adaptations to an agent"""
        try:
            agent = self.registered_agents[agent_id]
            
            # Apply feature importance updates
            if 'feature_importance' in adaptations and hasattr(agent, 'update_feature_importance'):
                agent.update_feature_importance(adaptations['feature_importance'])
            
            # Apply pattern strength updates
            if 'pattern_strengths' in adaptations and hasattr(agent, 'update_pattern_strengths'):
                agent.update_pattern_strengths(adaptations['pattern_strengths'])
            
            # Apply regime weight updates
            if 'regime_weights' in adaptations and hasattr(agent, 'update_regime_weights'):
                agent.update_regime_weights(adaptations['regime_weights'])
            
            # Apply confidence adjustments
            if 'confidence_adjustment' in adaptations and hasattr(agent, 'adjust_confidence'):
                agent.adjust_confidence(adaptations['confidence_adjustment'])
            
            # Apply exploration rate updates
            if 'exploration_rate' in adaptations and hasattr(agent, 'set_exploration_rate'):
                agent.set_exploration_rate(adaptations['exploration_rate'])
            
            logger.debug("Adaptations applied to strategic agent",
                        agent_id=agent_id,
                        adaptations=list(adaptations.keys()))
            
        except Exception as e:
            logger.error("Error applying adaptations to strategic agent",
                        agent_id=agent_id,
                        error=str(e))
    
    def get_agent_learning_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get learning state for an agent"""
        if agent_id in self.integrator.agent_states:
            return self.integrator.agent_states[agent_id].to_dict()
        return None
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'registered_agents': list(self.registered_agents.keys()),
            'learning_modes': {
                agent_id: mode.value 
                for agent_id, mode in self.learning_modes.items()
            },
            'integrator_stats': self.integrator.get_learning_statistics(),
            'performance_tracking': {
                agent_id: len(history) 
                for agent_id, history in self.performance_tracker.items()
            }
        }


# Global instance
strategic_learning_manager = StrategicLearningManager({
    'integrator': {
        'learning_rate': 0.01,
        'adaptation_rate': 0.1,
        'memory_decay': 0.95,
        'pattern_learning': {
            'pattern_memory_size': 100,
            'pattern_similarity_threshold': 0.7,
            'pattern_success_threshold': 0.6
        }
    },
    'default_agents': ['structure_analyzer', 'mlmi_strategic_agent', 'nwrqk_strategic_agent']
})