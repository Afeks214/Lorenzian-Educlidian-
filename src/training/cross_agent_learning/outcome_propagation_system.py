"""
Cross-Agent Learning Outcome Propagation System
==============================================

This module implements a comprehensive system for propagating execution outcomes
to all agent levels (strategic, tactical, and risk) to enable cross-layer learning
and adaptation based on real execution performance.

Key Features:
- Execution outcome collection and analysis
- Multi-level outcome propagation
- Performance impact assessment
- Learning signal generation
- Cross-agent feedback distribution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import json
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)


class OutcomeType(Enum):
    """Types of execution outcomes to propagate"""
    TRADE_EXECUTION = "trade_execution"
    POSITION_CHANGE = "position_change"
    RISK_EVENT = "risk_event"
    PERFORMANCE_METRIC = "performance_metric"
    MARKET_IMPACT = "market_impact"
    COORDINATION_RESULT = "coordination_result"
    LEARNING_SIGNAL = "learning_signal"


class AgentLevel(Enum):
    """Agent hierarchy levels"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    EXECUTION = "execution"
    RISK = "risk"


@dataclass
class ExecutionOutcome:
    """Structured representation of execution outcomes"""
    outcome_id: str
    outcome_type: OutcomeType
    timestamp: datetime
    source_agent: str
    
    # Execution details
    execution_data: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    pnl: float = 0.0
    slippage: float = 0.0
    execution_time: float = 0.0
    fill_quality: float = 1.0
    
    # Impact assessment
    market_impact: float = 0.0
    risk_impact: float = 0.0
    portfolio_impact: float = 0.0
    
    # Learning signals
    success_score: float = 0.5
    confidence_validation: float = 0.5
    prediction_accuracy: float = 0.5
    
    # Contextual information
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    coordination_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'outcome_id': self.outcome_id,
            'outcome_type': self.outcome_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source_agent': self.source_agent,
            'execution_data': self.execution_data,
            'pnl': self.pnl,
            'slippage': self.slippage,
            'execution_time': self.execution_time,
            'fill_quality': self.fill_quality,
            'market_impact': self.market_impact,
            'risk_impact': self.risk_impact,
            'portfolio_impact': self.portfolio_impact,
            'success_score': self.success_score,
            'confidence_validation': self.confidence_validation,
            'prediction_accuracy': self.prediction_accuracy,
            'market_conditions': self.market_conditions,
            'agent_states': self.agent_states,
            'coordination_context': self.coordination_context
        }


@dataclass
class LearningSignal:
    """Learning signal to be propagated to agents"""
    signal_id: str
    target_agent_level: AgentLevel
    target_agents: List[str]
    signal_type: str
    
    # Learning content
    reward_adjustment: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    pattern_feedback: Dict[str, Any] = field(default_factory=dict)
    
    # Adaptation guidance
    behavior_adjustment: Dict[str, Any] = field(default_factory=dict)
    confidence_adjustment: float = 0.0
    exploration_guidance: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    priority: float = 0.5
    persistence: float = 1.0  # How long signal should influence learning
    source_outcome: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'signal_id': self.signal_id,
            'target_agent_level': self.target_agent_level.value,
            'target_agents': self.target_agents,
            'signal_type': self.signal_type,
            'reward_adjustment': self.reward_adjustment,
            'feature_importance': self.feature_importance,
            'pattern_feedback': self.pattern_feedback,
            'behavior_adjustment': self.behavior_adjustment,
            'confidence_adjustment': self.confidence_adjustment,
            'exploration_guidance': self.exploration_guidance,
            'priority': self.priority,
            'persistence': self.persistence,
            'source_outcome': self.source_outcome
        }


class OutcomeAnalyzer:
    """Analyzes execution outcomes to extract learning signals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.success_threshold = config.get('success_threshold', 0.0)
        self.slippage_threshold = config.get('slippage_threshold', 0.001)
        self.impact_threshold = config.get('impact_threshold', 0.01)
        
        # Pattern recognition for common success/failure modes
        self.success_patterns = config.get('success_patterns', {})
        self.failure_patterns = config.get('failure_patterns', {})
        
        # Learning signal generation weights
        self.signal_weights = config.get('signal_weights', {
            'pnl_weight': 0.4,
            'execution_quality_weight': 0.3,
            'prediction_accuracy_weight': 0.3
        })
        
        logger.info("OutcomeAnalyzer initialized", 
                   success_threshold=self.success_threshold,
                   slippage_threshold=self.slippage_threshold)
    
    def analyze_outcome(self, outcome: ExecutionOutcome) -> List[LearningSignal]:
        """
        Analyze execution outcome and generate learning signals
        
        Args:
            outcome: Execution outcome to analyze
            
        Returns:
            List of learning signals for different agent levels
        """
        signals = []
        
        try:
            # Calculate overall success score
            success_score = self._calculate_success_score(outcome)
            outcome.success_score = success_score
            
            # Generate strategic learning signals
            strategic_signals = self._generate_strategic_signals(outcome)
            signals.extend(strategic_signals)
            
            # Generate tactical learning signals
            tactical_signals = self._generate_tactical_signals(outcome)
            signals.extend(tactical_signals)
            
            # Generate execution learning signals
            execution_signals = self._generate_execution_signals(outcome)
            signals.extend(execution_signals)
            
            # Generate risk learning signals
            risk_signals = self._generate_risk_signals(outcome)
            signals.extend(risk_signals)
            
            # Generate coordination learning signals
            coordination_signals = self._generate_coordination_signals(outcome)
            signals.extend(coordination_signals)
            
            logger.debug("Outcome analysis complete",
                        outcome_id=outcome.outcome_id,
                        success_score=success_score,
                        signals_generated=len(signals))
            
        except Exception as e:
            logger.error("Error analyzing outcome", 
                        outcome_id=outcome.outcome_id,
                        error=str(e))
        
        return signals
    
    def _calculate_success_score(self, outcome: ExecutionOutcome) -> float:
        """Calculate overall success score for the outcome"""
        try:
            # P&L component
            pnl_score = np.tanh(outcome.pnl / 0.01)  # Normalize around 1% moves
            
            # Execution quality component
            execution_quality = (
                outcome.fill_quality * 0.4 +
                (1 - min(abs(outcome.slippage) / self.slippage_threshold, 1)) * 0.3 +
                (1 - min(outcome.execution_time / 5.0, 1)) * 0.3  # 5s target
            )
            
            # Prediction accuracy component
            prediction_score = outcome.prediction_accuracy
            
            # Weighted combination
            success_score = (
                self.signal_weights['pnl_weight'] * pnl_score +
                self.signal_weights['execution_quality_weight'] * execution_quality +
                self.signal_weights['prediction_accuracy_weight'] * prediction_score
            )
            
            return np.clip(success_score, -1.0, 1.0)
            
        except Exception as e:
            logger.error("Error calculating success score", error=str(e))
            return 0.0
    
    def _generate_strategic_signals(self, outcome: ExecutionOutcome) -> List[LearningSignal]:
        """Generate learning signals for strategic agents"""
        signals = []
        
        try:
            # Long-term performance signal
            if outcome.outcome_type == OutcomeType.TRADE_EXECUTION:
                reward_adjustment = outcome.pnl * 0.1  # Scale for strategic learning
                
                # Feature importance based on success
                feature_importance = {}
                if outcome.success_score > 0.5:
                    # Positive outcome - reinforce features that led to success
                    for feature, value in outcome.agent_states.get('strategic_features', {}).items():
                        if abs(value) > 0.5:  # Strong feature signals
                            feature_importance[feature] = outcome.success_score * 0.5
                
                # Pattern feedback for regime detection
                pattern_feedback = {}
                if 'market_regime' in outcome.market_conditions:
                    regime = outcome.market_conditions['market_regime']
                    pattern_feedback[f'regime_{regime}'] = outcome.success_score
                
                signal = LearningSignal(
                    signal_id=f"strategic_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.STRATEGIC,
                    target_agents=['structure_analyzer', 'mlmi_strategic_agent', 'nwrqk_strategic_agent'],
                    signal_type='performance_feedback',
                    reward_adjustment=reward_adjustment,
                    feature_importance=feature_importance,
                    pattern_feedback=pattern_feedback,
                    priority=0.7,
                    persistence=10.0,  # Long-term learning
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
            
            # Market regime adaptation signal
            if outcome.market_impact > self.impact_threshold:
                behavior_adjustment = {
                    'reduce_position_size': True,
                    'increase_regime_sensitivity': True,
                    'market_regime_weight': min(outcome.market_impact * 2, 1.0)
                }
                
                signal = LearningSignal(
                    signal_id=f"strategic_regime_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.STRATEGIC,
                    target_agents=['structure_analyzer'],
                    signal_type='regime_adaptation',
                    behavior_adjustment=behavior_adjustment,
                    confidence_adjustment=-outcome.market_impact * 0.5,
                    priority=0.8,
                    persistence=5.0,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error("Error generating strategic signals", error=str(e))
        
        return signals
    
    def _generate_tactical_signals(self, outcome: ExecutionOutcome) -> List[LearningSignal]:
        """Generate learning signals for tactical agents"""
        signals = []
        
        try:
            # Execution timing signal
            if outcome.outcome_type == OutcomeType.TRADE_EXECUTION:
                timing_score = 1.0 - min(outcome.execution_time / 2.0, 1.0)  # 2s target
                
                # Timing-based reward adjustment
                reward_adjustment = timing_score * 0.05
                
                # Feature importance for tactical features
                feature_importance = {}
                for feature, value in outcome.agent_states.get('tactical_features', {}).items():
                    if 'timing' in feature.lower() or 'momentum' in feature.lower():
                        feature_importance[feature] = timing_score * 0.3
                
                signal = LearningSignal(
                    signal_id=f"tactical_timing_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.TACTICAL,
                    target_agents=['short_term_tactician', 'mid_freq_arbitrageur'],
                    signal_type='timing_feedback',
                    reward_adjustment=reward_adjustment,
                    feature_importance=feature_importance,
                    priority=0.6,
                    persistence=3.0,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
            
            # Slippage optimization signal
            if abs(outcome.slippage) > self.slippage_threshold:
                behavior_adjustment = {
                    'reduce_order_size': True,
                    'increase_patience': True,
                    'slippage_sensitivity': min(abs(outcome.slippage) * 100, 1.0)
                }
                
                signal = LearningSignal(
                    signal_id=f"tactical_slippage_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.TACTICAL,
                    target_agents=['short_term_tactician'],
                    signal_type='slippage_optimization',
                    behavior_adjustment=behavior_adjustment,
                    confidence_adjustment=-abs(outcome.slippage) * 10,
                    priority=0.7,
                    persistence=2.0,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error("Error generating tactical signals", error=str(e))
        
        return signals
    
    def _generate_execution_signals(self, outcome: ExecutionOutcome) -> List[LearningSignal]:
        """Generate learning signals for execution agents"""
        signals = []
        
        try:
            # Fill quality feedback
            if outcome.fill_quality < 0.9:
                behavior_adjustment = {
                    'improve_venue_selection': True,
                    'adjust_order_timing': True,
                    'fill_quality_weight': 1.0 - outcome.fill_quality
                }
                
                signal = LearningSignal(
                    signal_id=f"execution_quality_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.EXECUTION,
                    target_agents=['routing_agent', 'execution_timing_agent'],
                    signal_type='quality_feedback',
                    behavior_adjustment=behavior_adjustment,
                    reward_adjustment=-(1.0 - outcome.fill_quality) * 0.1,
                    priority=0.8,
                    persistence=1.0,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
            
            # Market impact feedback
            if outcome.market_impact > 0.005:  # 50bps threshold
                exploration_guidance = {
                    'explore_smaller_sizes': True,
                    'explore_different_venues': True,
                    'market_impact_threshold': outcome.market_impact
                }
                
                signal = LearningSignal(
                    signal_id=f"execution_impact_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.EXECUTION,
                    target_agents=['routing_agent', 'position_sizing_agent'],
                    signal_type='impact_feedback',
                    exploration_guidance=exploration_guidance,
                    reward_adjustment=-outcome.market_impact * 2,
                    priority=0.9,
                    persistence=1.5,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error("Error generating execution signals", error=str(e))
        
        return signals
    
    def _generate_risk_signals(self, outcome: ExecutionOutcome) -> List[LearningSignal]:
        """Generate learning signals for risk agents"""
        signals = []
        
        try:
            # Risk impact assessment
            if outcome.risk_impact > 0.01:  # 1% risk threshold
                behavior_adjustment = {
                    'tighten_risk_controls': True,
                    'reduce_position_limits': True,
                    'risk_sensitivity': outcome.risk_impact * 10
                }
                
                signal = LearningSignal(
                    signal_id=f"risk_impact_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.RISK,
                    target_agents=['risk_monitor_agent', 'position_sizing_agent'],
                    signal_type='risk_feedback',
                    behavior_adjustment=behavior_adjustment,
                    reward_adjustment=-outcome.risk_impact * 5,
                    priority=1.0,
                    persistence=5.0,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
            
            # Portfolio impact signal
            if abs(outcome.portfolio_impact) > 0.005:
                pattern_feedback = {
                    'portfolio_concentration': outcome.portfolio_impact,
                    'diversification_need': abs(outcome.portfolio_impact) > 0.01
                }
                
                signal = LearningSignal(
                    signal_id=f"portfolio_impact_{outcome.outcome_id}",
                    target_agent_level=AgentLevel.RISK,
                    target_agents=['portfolio_optimizer_agent'],
                    signal_type='portfolio_feedback',
                    pattern_feedback=pattern_feedback,
                    confidence_adjustment=-abs(outcome.portfolio_impact) * 2,
                    priority=0.8,
                    persistence=3.0,
                    source_outcome=outcome.outcome_id
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error("Error generating risk signals", error=str(e))
        
        return signals
    
    def _generate_coordination_signals(self, outcome: ExecutionOutcome) -> List[LearningSignal]:
        """Generate learning signals for agent coordination"""
        signals = []
        
        try:
            coordination_success = outcome.coordination_context.get('coordination_success', 0.5)
            
            if coordination_success < 0.7:  # Poor coordination
                behavior_adjustment = {
                    'improve_coordination': True,
                    'increase_communication': True,
                    'coordination_weight': 1.0 - coordination_success
                }
                
                # Signal to all agent levels
                for level in [AgentLevel.STRATEGIC, AgentLevel.TACTICAL, AgentLevel.EXECUTION]:
                    signal = LearningSignal(
                        signal_id=f"coordination_{level.value}_{outcome.outcome_id}",
                        target_agent_level=level,
                        target_agents=['all'],  # All agents at this level
                        signal_type='coordination_feedback',
                        behavior_adjustment=behavior_adjustment,
                        reward_adjustment=-(1.0 - coordination_success) * 0.05,
                        priority=0.6,
                        persistence=2.0,
                        source_outcome=outcome.outcome_id
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error("Error generating coordination signals", error=str(e))
        
        return signals


class OutcomePropagationSystem:
    """Main system for propagating execution outcomes to all agent levels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer = OutcomeAnalyzer(config.get('analyzer', {}))
        
        # Outcome storage
        self.outcome_history = deque(maxlen=config.get('max_history', 1000))
        self.learning_signals = deque(maxlen=config.get('max_signals', 5000))
        
        # Agent subscriptions
        self.agent_subscriptions = defaultdict(list)
        self.level_subscriptions = defaultdict(list)
        
        # Performance tracking
        self.propagation_stats = {
            'outcomes_processed': 0,
            'signals_generated': 0,
            'signals_delivered': 0,
            'delivery_failures': 0
        }
        
        # Async components
        self.signal_queue = asyncio.Queue()
        self.running = False
        
        logger.info("OutcomePropagationSystem initialized", 
                   config=config)
    
    async def start(self):
        """Start the propagation system"""
        self.running = True
        
        # Start signal processing loop
        asyncio.create_task(self._signal_processing_loop())
        
        logger.info("OutcomePropagationSystem started")
    
    async def stop(self):
        """Stop the propagation system"""
        self.running = False
        logger.info("OutcomePropagationSystem stopped")
    
    def subscribe_agent(self, agent_id: str, agent_level: AgentLevel, 
                       callback: Callable[[LearningSignal], None]):
        """
        Subscribe an agent to receive learning signals
        
        Args:
            agent_id: Unique agent identifier
            agent_level: Agent's hierarchical level
            callback: Function to call with learning signals
        """
        self.agent_subscriptions[agent_id].append(callback)
        self.level_subscriptions[agent_level].append((agent_id, callback))
        
        logger.info("Agent subscribed to learning signals",
                   agent_id=agent_id,
                   agent_level=agent_level.value)
    
    def unsubscribe_agent(self, agent_id: str, agent_level: AgentLevel):
        """Unsubscribe an agent from learning signals"""
        if agent_id in self.agent_subscriptions:
            del self.agent_subscriptions[agent_id]
        
        # Remove from level subscriptions
        self.level_subscriptions[agent_level] = [
            (aid, callback) for aid, callback in self.level_subscriptions[agent_level]
            if aid != agent_id
        ]
        
        logger.info("Agent unsubscribed from learning signals",
                   agent_id=agent_id,
                   agent_level=agent_level.value)
    
    async def propagate_outcome(self, outcome: ExecutionOutcome):
        """
        Propagate an execution outcome to generate learning signals
        
        Args:
            outcome: Execution outcome to propagate
        """
        try:
            # Store outcome
            self.outcome_history.append(outcome)
            self.propagation_stats['outcomes_processed'] += 1
            
            # Analyze outcome to generate learning signals
            learning_signals = self.analyzer.analyze_outcome(outcome)
            
            # Queue signals for delivery
            for signal in learning_signals:
                await self.signal_queue.put(signal)
                self.learning_signals.append(signal)
                self.propagation_stats['signals_generated'] += 1
            
            logger.info("Outcome propagated",
                       outcome_id=outcome.outcome_id,
                       signals_generated=len(learning_signals))
            
        except Exception as e:
            logger.error("Error propagating outcome",
                        outcome_id=outcome.outcome_id,
                        error=str(e))
    
    async def _signal_processing_loop(self):
        """Main loop for processing and delivering learning signals"""
        while self.running:
            try:
                # Get signal with timeout
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                
                await self._deliver_signal(signal)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error in signal processing loop", error=str(e))
    
    async def _deliver_signal(self, signal: LearningSignal):
        """Deliver a learning signal to target agents"""
        try:
            delivered = 0
            
            # Deliver to specific agents
            if signal.target_agents != ['all']:
                for agent_id in signal.target_agents:
                    if agent_id in self.agent_subscriptions:
                        for callback in self.agent_subscriptions[agent_id]:
                            try:
                                await self._safe_callback(callback, signal)
                                delivered += 1
                            except Exception as e:
                                logger.error("Error delivering signal to agent",
                                           agent_id=agent_id,
                                           signal_id=signal.signal_id,
                                           error=str(e))
                                self.propagation_stats['delivery_failures'] += 1
            else:
                # Deliver to all agents at target level
                for agent_id, callback in self.level_subscriptions[signal.target_agent_level]:
                    try:
                        await self._safe_callback(callback, signal)
                        delivered += 1
                    except Exception as e:
                        logger.error("Error delivering signal to level",
                                   agent_id=agent_id,
                                   level=signal.target_agent_level.value,
                                   signal_id=signal.signal_id,
                                   error=str(e))
                        self.propagation_stats['delivery_failures'] += 1
            
            self.propagation_stats['signals_delivered'] += delivered
            
            logger.debug("Signal delivered",
                        signal_id=signal.signal_id,
                        target_level=signal.target_agent_level.value,
                        delivered_count=delivered)
            
        except Exception as e:
            logger.error("Error delivering signal",
                        signal_id=signal.signal_id,
                        error=str(e))
    
    async def _safe_callback(self, callback: Callable, signal: LearningSignal):
        """Safely execute a callback with error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(signal)
            else:
                callback(signal)
        except Exception as e:
            logger.error("Callback execution failed", error=str(e))
            raise
    
    def get_recent_outcomes(self, count: int = 10) -> List[ExecutionOutcome]:
        """Get recent execution outcomes"""
        return list(self.outcome_history)[-count:]
    
    def get_recent_signals(self, count: int = 20) -> List[LearningSignal]:
        """Get recent learning signals"""
        return list(self.learning_signals)[-count:]
    
    def get_propagation_stats(self) -> Dict[str, Any]:
        """Get propagation system statistics"""
        return {
            'propagation_stats': self.propagation_stats.copy(),
            'outcome_history_size': len(self.outcome_history),
            'learning_signals_size': len(self.learning_signals),
            'agent_subscriptions': len(self.agent_subscriptions),
            'level_subscriptions': {
                level.value: len(subscriptions) 
                for level, subscriptions in self.level_subscriptions.items()
            },
            'signal_queue_size': self.signal_queue.qsize()
        }
    
    def create_execution_outcome(self, **kwargs) -> ExecutionOutcome:
        """
        Helper method to create ExecutionOutcome with defaults
        
        Args:
            **kwargs: Outcome parameters
            
        Returns:
            ExecutionOutcome instance
        """
        return ExecutionOutcome(
            outcome_id=kwargs.get('outcome_id', f"outcome_{datetime.now().timestamp()}"),
            outcome_type=kwargs.get('outcome_type', OutcomeType.TRADE_EXECUTION),
            timestamp=kwargs.get('timestamp', datetime.now()),
            source_agent=kwargs.get('source_agent', 'unknown'),
            **{k: v for k, v in kwargs.items() if k not in ['outcome_id', 'outcome_type', 'timestamp', 'source_agent']}
        )


# Global instance
outcome_propagation_system = OutcomePropagationSystem({
    'analyzer': {
        'success_threshold': 0.0,
        'slippage_threshold': 0.001,
        'impact_threshold': 0.01,
        'signal_weights': {
            'pnl_weight': 0.4,
            'execution_quality_weight': 0.3,
            'prediction_accuracy_weight': 0.3
        }
    },
    'max_history': 1000,
    'max_signals': 5000
})