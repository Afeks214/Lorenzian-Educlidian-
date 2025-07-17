"""
Tactical Agent Adaptation System
===============================

This module implements adaptation mechanisms for tactical agents based on
execution outcomes and performance feedback. Tactical agents focus on
short-term execution optimization and market microstructure learning.

Key Features:
- Execution quality optimization
- Market microstructure learning
- Timing precision improvement
- Slippage minimization
- Cross-venue adaptation
- Real-time feedback integration
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


class TacticalAdaptationMode(Enum):
    """Tactical adaptation modes"""
    EXECUTION_OPTIMIZATION = "execution_optimization"
    TIMING_PRECISION = "timing_precision"
    SLIPPAGE_MINIMIZATION = "slippage_minimization"
    VENUE_SELECTION = "venue_selection"
    MARKET_IMPACT_REDUCTION = "market_impact_reduction"
    LIQUIDITY_OPTIMIZATION = "liquidity_optimization"


@dataclass
class TacticalAdaptationState:
    """State of tactical adaptation for an agent"""
    agent_id: str
    last_update: datetime
    adaptation_mode: TacticalAdaptationMode
    
    # Execution metrics
    average_slippage: float = 0.0
    average_execution_time: float = 0.0
    average_fill_quality: float = 1.0
    market_impact_score: float = 0.0
    
    # Learning parameters
    timing_sensitivity: float = 1.0
    slippage_tolerance: float = 0.001
    execution_urgency: float = 0.5
    venue_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Adaptation history
    successful_executions: List[Dict[str, Any]] = field(default_factory=list)
    failed_executions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    execution_count: int = 0
    success_rate: float = 0.5
    improvement_trend: float = 0.0
    
    # Microstructure learning
    liquidity_patterns: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    timing_patterns: Dict[str, float] = field(default_factory=dict)
    venue_performance: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'last_update': self.last_update.isoformat(),
            'adaptation_mode': self.adaptation_mode.value,
            'average_slippage': self.average_slippage,
            'average_execution_time': self.average_execution_time,
            'average_fill_quality': self.average_fill_quality,
            'market_impact_score': self.market_impact_score,
            'timing_sensitivity': self.timing_sensitivity,
            'slippage_tolerance': self.slippage_tolerance,
            'execution_urgency': self.execution_urgency,
            'venue_preferences': self.venue_preferences,
            'successful_executions': len(self.successful_executions),
            'failed_executions': len(self.failed_executions),
            'execution_count': self.execution_count,
            'success_rate': self.success_rate,
            'improvement_trend': self.improvement_trend,
            'liquidity_patterns': dict(self.liquidity_patterns),
            'timing_patterns': self.timing_patterns,
            'venue_performance': dict(self.venue_performance)
        }


class ExecutionPatternAnalyzer:
    """Analyzes execution patterns to identify optimization opportunities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_window = config.get('pattern_window', 20)
        self.success_threshold = config.get('success_threshold', 0.7)
        self.impact_threshold = config.get('impact_threshold', 0.005)
        
        # Pattern storage
        self.execution_patterns = defaultdict(list)
        self.timing_patterns = {}
        self.venue_patterns = defaultdict(dict)
        
        logger.info("ExecutionPatternAnalyzer initialized")
    
    def analyze_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze execution data to identify patterns and opportunities
        
        Args:
            execution_data: Execution outcome data
            
        Returns:
            Dictionary of analysis results
        """
        try:
            analysis = {
                'timing_analysis': self._analyze_timing_patterns(execution_data),
                'venue_analysis': self._analyze_venue_performance(execution_data),
                'liquidity_analysis': self._analyze_liquidity_patterns(execution_data),
                'impact_analysis': self._analyze_market_impact(execution_data)
            }
            
            # Store execution for future pattern analysis
            self._store_execution_pattern(execution_data)
            
            return analysis
            
        except Exception as e:
            logger.error("Error analyzing execution", error=str(e))
            return {}
    
    def _analyze_timing_patterns(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timing patterns in execution"""
        try:
            execution_time = execution_data.get('execution_time', 0.0)
            fill_quality = execution_data.get('fill_quality', 1.0)
            slippage = execution_data.get('slippage', 0.0)
            
            # Time-of-day patterns
            timestamp = execution_data.get('timestamp', datetime.now())
            hour = timestamp.hour
            minute = timestamp.minute
            
            time_key = f"{hour}_{minute//15}"  # 15-minute buckets
            
            if time_key not in self.timing_patterns:
                self.timing_patterns[time_key] = {
                    'count': 0,
                    'avg_execution_time': 0.0,
                    'avg_fill_quality': 0.0,
                    'avg_slippage': 0.0
                }
            
            pattern = self.timing_patterns[time_key]
            pattern['count'] += 1
            pattern['avg_execution_time'] = (
                pattern['avg_execution_time'] * (pattern['count'] - 1) + execution_time
            ) / pattern['count']
            pattern['avg_fill_quality'] = (
                pattern['avg_fill_quality'] * (pattern['count'] - 1) + fill_quality
            ) / pattern['count']
            pattern['avg_slippage'] = (
                pattern['avg_slippage'] * (pattern['count'] - 1) + abs(slippage)
            ) / pattern['count']
            
            return {
                'optimal_time_bucket': time_key,
                'timing_score': fill_quality / max(execution_time, 0.1),
                'time_pattern_reliability': min(pattern['count'] / 10, 1.0)
            }
            
        except Exception as e:
            logger.error("Error analyzing timing patterns", error=str(e))
            return {}
    
    def _analyze_venue_performance(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze venue performance patterns"""
        try:
            venue = execution_data.get('venue', 'unknown')
            fill_quality = execution_data.get('fill_quality', 1.0)
            slippage = execution_data.get('slippage', 0.0)
            execution_time = execution_data.get('execution_time', 0.0)
            
            if venue not in self.venue_patterns:
                self.venue_patterns[venue] = {
                    'count': 0,
                    'avg_fill_quality': 0.0,
                    'avg_slippage': 0.0,
                    'avg_execution_time': 0.0,
                    'reliability_score': 0.0
                }
            
            pattern = self.venue_patterns[venue]
            pattern['count'] += 1
            pattern['avg_fill_quality'] = (
                pattern['avg_fill_quality'] * (pattern['count'] - 1) + fill_quality
            ) / pattern['count']
            pattern['avg_slippage'] = (
                pattern['avg_slippage'] * (pattern['count'] - 1) + abs(slippage)
            ) / pattern['count']
            pattern['avg_execution_time'] = (
                pattern['avg_execution_time'] * (pattern['count'] - 1) + execution_time
            ) / pattern['count']
            
            # Calculate reliability score
            pattern['reliability_score'] = (
                pattern['avg_fill_quality'] * 0.4 +
                (1 - min(pattern['avg_slippage'] / 0.001, 1)) * 0.3 +
                (1 - min(pattern['avg_execution_time'] / 2.0, 1)) * 0.3
            )
            
            return {
                'venue_performance': pattern,
                'venue_rank': self._rank_venues(),
                'venue_recommendation': self._recommend_venue(execution_data)
            }
            
        except Exception as e:
            logger.error("Error analyzing venue performance", error=str(e))
            return {}
    
    def _analyze_liquidity_patterns(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity patterns"""
        try:
            market_conditions = execution_data.get('market_conditions', {})
            bid_ask_spread = market_conditions.get('bid_ask_spread', 0.0)
            volume = market_conditions.get('volume', 0.0)
            volatility = market_conditions.get('volatility', 0.0)
            
            # Liquidity quality score
            liquidity_score = 1.0 / (1.0 + bid_ask_spread * 1000 + volatility * 10)
            
            # Store liquidity pattern
            hour = execution_data.get('timestamp', datetime.now()).hour
            self.execution_patterns[f'liquidity_{hour}'].append(liquidity_score)
            
            # Keep patterns bounded
            if len(self.execution_patterns[f'liquidity_{hour}']) > self.pattern_window:
                self.execution_patterns[f'liquidity_{hour}'].pop(0)
            
            return {
                'liquidity_score': liquidity_score,
                'optimal_liquidity_time': self._find_optimal_liquidity_time(),
                'liquidity_trend': self._calculate_liquidity_trend()
            }
            
        except Exception as e:
            logger.error("Error analyzing liquidity patterns", error=str(e))
            return {}
    
    def _analyze_market_impact(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market impact patterns"""
        try:
            market_impact = execution_data.get('market_impact', 0.0)
            order_size = execution_data.get('order_size', 0.0)
            
            # Calculate impact per unit size
            impact_per_unit = market_impact / max(order_size, 1.0)
            
            # Store impact pattern
            size_bucket = int(order_size / 1000)  # 1000 unit buckets
            self.execution_patterns[f'impact_{size_bucket}'].append(impact_per_unit)
            
            return {
                'market_impact': market_impact,
                'impact_per_unit': impact_per_unit,
                'impact_efficiency': 1.0 / (1.0 + impact_per_unit * 100),
                'optimal_size_recommendation': self._recommend_optimal_size()
            }
            
        except Exception as e:
            logger.error("Error analyzing market impact", error=str(e))
            return {}
    
    def _store_execution_pattern(self, execution_data: Dict[str, Any]):
        """Store execution data for pattern analysis"""
        try:
            pattern_key = f"execution_{datetime.now().strftime('%Y%m%d_%H')}"
            
            pattern_data = {
                'timestamp': execution_data.get('timestamp', datetime.now()),
                'execution_time': execution_data.get('execution_time', 0.0),
                'fill_quality': execution_data.get('fill_quality', 1.0),
                'slippage': execution_data.get('slippage', 0.0),
                'market_impact': execution_data.get('market_impact', 0.0),
                'venue': execution_data.get('venue', 'unknown')
            }
            
            self.execution_patterns[pattern_key].append(pattern_data)
            
            # Keep patterns bounded
            if len(self.execution_patterns[pattern_key]) > self.pattern_window:
                self.execution_patterns[pattern_key].pop(0)
                
        except Exception as e:
            logger.error("Error storing execution pattern", error=str(e))
    
    def _rank_venues(self) -> List[Tuple[str, float]]:
        """Rank venues by performance"""
        try:
            venue_scores = [
                (venue, data['reliability_score'])
                for venue, data in self.venue_patterns.items()
                if data['count'] >= 5  # Minimum executions for reliable ranking
            ]
            return sorted(venue_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error("Error ranking venues", error=str(e))
            return []
    
    def _recommend_venue(self, execution_data: Dict[str, Any]) -> str:
        """Recommend optimal venue for execution"""
        try:
            venue_rankings = self._rank_venues()
            if venue_rankings:
                return venue_rankings[0][0]  # Top ranked venue
            return 'default'
        except Exception as e:
            logger.error("Error recommending venue", error=str(e))
            return 'default'
    
    def _find_optimal_liquidity_time(self) -> int:
        """Find optimal time for liquidity"""
        try:
            best_hour = 0
            best_score = 0.0
            
            for hour in range(24):
                liquidity_data = self.execution_patterns.get(f'liquidity_{hour}', [])
                if liquidity_data:
                    avg_score = np.mean(liquidity_data)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_hour = hour
            
            return best_hour
        except Exception as e:
            logger.error("Error finding optimal liquidity time", error=str(e))
            return 12  # Default to noon
    
    def _calculate_liquidity_trend(self) -> float:
        """Calculate liquidity trend"""
        try:
            recent_scores = []
            for hour in range(24):
                liquidity_data = self.execution_patterns.get(f'liquidity_{hour}', [])
                if liquidity_data:
                    recent_scores.extend(liquidity_data[-5:])  # Last 5 observations
            
            if len(recent_scores) >= 2:
                return np.mean(recent_scores[-5:]) - np.mean(recent_scores[-10:-5])
            return 0.0
        except Exception as e:
            logger.error("Error calculating liquidity trend", error=str(e))
            return 0.0
    
    def _recommend_optimal_size(self) -> float:
        """Recommend optimal order size"""
        try:
            best_size = 1000  # Default size
            best_efficiency = 0.0
            
            for size_bucket in range(10):  # Check different size buckets
                impact_data = self.execution_patterns.get(f'impact_{size_bucket}', [])
                if impact_data:
                    avg_impact = np.mean(impact_data)
                    efficiency = 1.0 / (1.0 + avg_impact * 100)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_size = size_bucket * 1000
            
            return best_size
        except Exception as e:
            logger.error("Error recommending optimal size", error=str(e))
            return 1000


class TacticalAdaptationEngine:
    """Engine for adapting tactical agent behavior based on execution outcomes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_rate = config.get('adaptation_rate', 0.15)
        self.learning_rate = config.get('learning_rate', 0.02)
        self.memory_decay = config.get('memory_decay', 0.9)
        
        # Agent adaptation states
        self.agent_states = {}
        
        # Pattern analyzer
        self.pattern_analyzer = ExecutionPatternAnalyzer(config.get('pattern_analyzer', {}))
        
        # Adaptation history
        self.adaptation_history = defaultdict(list)
        
        logger.info("TacticalAdaptationEngine initialized")
    
    def get_or_create_agent_state(self, agent_id: str) -> TacticalAdaptationState:
        """Get or create adaptation state for an agent"""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = TacticalAdaptationState(
                agent_id=agent_id,
                last_update=datetime.now(),
                adaptation_mode=TacticalAdaptationMode.EXECUTION_OPTIMIZATION
            )
        return self.agent_states[agent_id]
    
    def adapt_from_outcome(self, outcome_data: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """
        Adapt agent behavior based on execution outcome
        
        Args:
            outcome_data: Execution outcome data
            agent_id: Target agent ID
            
        Returns:
            Dictionary of behavioral adaptations
        """
        try:
            state = self.get_or_create_agent_state(agent_id)
            
            # Analyze execution patterns
            analysis = self.pattern_analyzer.analyze_execution(outcome_data)
            
            # Update agent metrics
            self._update_agent_metrics(state, outcome_data)
            
            # Generate adaptations
            adaptations = self._generate_adaptations(state, outcome_data, analysis)
            
            # Apply adaptations to state
            self._apply_adaptations_to_state(state, adaptations)
            
            # Record adaptation
            self.adaptation_history[agent_id].append({
                'timestamp': datetime.now(),
                'outcome_data': outcome_data,
                'analysis': analysis,
                'adaptations': adaptations
            })
            
            # Keep history bounded
            if len(self.adaptation_history[agent_id]) > 100:
                self.adaptation_history[agent_id].pop(0)
            
            state.last_update = datetime.now()
            
            logger.debug("Tactical adaptation completed",
                        agent_id=agent_id,
                        adaptations=list(adaptations.keys()))
            
            return adaptations
            
        except Exception as e:
            logger.error("Error adapting from outcome",
                        agent_id=agent_id,
                        error=str(e))
            return {}
    
    def _update_agent_metrics(self, state: TacticalAdaptationState, outcome_data: Dict[str, Any]):
        """Update agent performance metrics"""
        try:
            # Update execution metrics
            slippage = abs(outcome_data.get('slippage', 0.0))
            execution_time = outcome_data.get('execution_time', 0.0)
            fill_quality = outcome_data.get('fill_quality', 1.0)
            market_impact = outcome_data.get('market_impact', 0.0)
            
            # Exponential moving average updates
            alpha = self.learning_rate
            state.average_slippage = (1 - alpha) * state.average_slippage + alpha * slippage
            state.average_execution_time = (1 - alpha) * state.average_execution_time + alpha * execution_time
            state.average_fill_quality = (1 - alpha) * state.average_fill_quality + alpha * fill_quality
            state.market_impact_score = (1 - alpha) * state.market_impact_score + alpha * market_impact
            
            # Update execution count and success rate
            state.execution_count += 1
            success = (slippage < state.slippage_tolerance and 
                      execution_time < 3.0 and 
                      fill_quality > 0.8)
            
            # Update success rate
            state.success_rate = (
                (state.success_rate * (state.execution_count - 1) + (1.0 if success else 0.0)) / 
                state.execution_count
            )
            
            # Store execution record
            execution_record = {
                'timestamp': datetime.now(),
                'slippage': slippage,
                'execution_time': execution_time,
                'fill_quality': fill_quality,
                'market_impact': market_impact,
                'success': success
            }
            
            if success:
                state.successful_executions.append(execution_record)
                if len(state.successful_executions) > 50:
                    state.successful_executions.pop(0)
            else:
                state.failed_executions.append(execution_record)
                if len(state.failed_executions) > 50:
                    state.failed_executions.pop(0)
            
        except Exception as e:
            logger.error("Error updating agent metrics", error=str(e))
    
    def _generate_adaptations(self, state: TacticalAdaptationState, outcome_data: Dict[str, Any], 
                             analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate behavioral adaptations based on outcome and analysis"""
        adaptations = {}
        
        try:
            # Timing adaptations
            timing_analysis = analysis.get('timing_analysis', {})
            if timing_analysis:
                timing_score = timing_analysis.get('timing_score', 0.5)
                if timing_score < 0.7:
                    adaptations['timing_sensitivity'] = state.timing_sensitivity * 1.1
                    adaptations['execution_urgency'] = min(state.execution_urgency * 0.9, 1.0)
                else:
                    adaptations['timing_sensitivity'] = state.timing_sensitivity * 0.95
                    adaptations['execution_urgency'] = min(state.execution_urgency * 1.05, 1.0)
            
            # Slippage adaptations
            current_slippage = abs(outcome_data.get('slippage', 0.0))
            if current_slippage > state.slippage_tolerance:
                adaptations['slippage_tolerance'] = state.slippage_tolerance * 1.1
                adaptations['order_aggressiveness'] = 0.8  # Reduce aggressiveness
                adaptations['patience_factor'] = 1.2  # Increase patience
            
            # Venue adaptations
            venue_analysis = analysis.get('venue_analysis', {})
            if venue_analysis:
                venue_recommendation = venue_analysis.get('venue_recommendation', 'default')
                venue_performance = venue_analysis.get('venue_performance', {})
                
                if venue_performance:
                    reliability_score = venue_performance.get('reliability_score', 0.5)
                    current_venue = outcome_data.get('venue', 'unknown')
                    
                    # Update venue preferences
                    if current_venue not in state.venue_preferences:
                        state.venue_preferences[current_venue] = 0.5
                    
                    # Adjust venue preference based on performance
                    state.venue_preferences[current_venue] = (
                        state.venue_preferences[current_venue] * (1 - self.adaptation_rate) +
                        reliability_score * self.adaptation_rate
                    )
                    
                    adaptations['venue_preferences'] = state.venue_preferences
                    adaptations['recommended_venue'] = venue_recommendation
            
            # Market impact adaptations
            impact_analysis = analysis.get('impact_analysis', {})
            if impact_analysis:
                impact_efficiency = impact_analysis.get('impact_efficiency', 0.5)
                optimal_size = impact_analysis.get('optimal_size_recommendation', 1000)
                
                if impact_efficiency < 0.7:
                    adaptations['order_size_multiplier'] = 0.8
                    adaptations['fragmentation_enabled'] = True
                else:
                    adaptations['order_size_multiplier'] = 1.0
                
                adaptations['optimal_order_size'] = optimal_size
            
            # Liquidity adaptations
            liquidity_analysis = analysis.get('liquidity_analysis', {})
            if liquidity_analysis:
                liquidity_score = liquidity_analysis.get('liquidity_score', 0.5)
                optimal_time = liquidity_analysis.get('optimal_liquidity_time', 12)
                
                if liquidity_score < 0.6:
                    adaptations['liquidity_sensitivity'] = 1.2
                    adaptations['wait_for_liquidity'] = True
                
                adaptations['optimal_execution_time'] = optimal_time
            
            # Exploration adaptations
            if state.success_rate < 0.6:
                adaptations['exploration_rate'] = min(0.2, state.success_rate * 0.3)
            else:
                adaptations['exploration_rate'] = max(0.05, state.success_rate * 0.1)
            
            # Confidence adaptations
            if state.success_rate > 0.8:
                adaptations['confidence_multiplier'] = 1.1
            elif state.success_rate < 0.4:
                adaptations['confidence_multiplier'] = 0.8
            
        except Exception as e:
            logger.error("Error generating adaptations", error=str(e))
        
        return adaptations
    
    def _apply_adaptations_to_state(self, state: TacticalAdaptationState, adaptations: Dict[str, Any]):
        """Apply adaptations to agent state"""
        try:
            # Apply timing adaptations
            if 'timing_sensitivity' in adaptations:
                state.timing_sensitivity = adaptations['timing_sensitivity']
            
            if 'execution_urgency' in adaptations:
                state.execution_urgency = adaptations['execution_urgency']
            
            # Apply slippage adaptations
            if 'slippage_tolerance' in adaptations:
                state.slippage_tolerance = adaptations['slippage_tolerance']
            
            # Apply venue adaptations
            if 'venue_preferences' in adaptations:
                state.venue_preferences.update(adaptations['venue_preferences'])
            
            # Calculate improvement trend
            if len(state.successful_executions) >= 10:
                recent_success = len([e for e in state.successful_executions[-10:] 
                                    if e['timestamp'] > datetime.now() - timedelta(hours=1)])
                old_success = len([e for e in state.successful_executions[-20:-10] 
                                 if e['timestamp'] > datetime.now() - timedelta(hours=2)])
                state.improvement_trend = (recent_success - old_success) / 10.0
            
            # Apply memory decay
            for venue in state.venue_preferences:
                state.venue_preferences[venue] *= self.memory_decay
            
        except Exception as e:
            logger.error("Error applying adaptations to state", error=str(e))
    
    def get_agent_adaptations(self, agent_id: str) -> Dict[str, Any]:
        """Get current adaptations for an agent"""
        if agent_id not in self.agent_states:
            return {}
        
        state = self.agent_states[agent_id]
        return {
            'timing_sensitivity': state.timing_sensitivity,
            'slippage_tolerance': state.slippage_tolerance,
            'execution_urgency': state.execution_urgency,
            'venue_preferences': state.venue_preferences,
            'success_rate': state.success_rate,
            'improvement_trend': state.improvement_trend,
            'adaptation_mode': state.adaptation_mode.value
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_agents': len(self.agent_states),
            'average_success_rate': np.mean([
                state.success_rate for state in self.agent_states.values()
            ]) if self.agent_states else 0.0,
            'average_slippage': np.mean([
                state.average_slippage for state in self.agent_states.values()
            ]) if self.agent_states else 0.0,
            'average_execution_time': np.mean([
                state.average_execution_time for state in self.agent_states.values()
            ]) if self.agent_states else 0.0,
            'adaptation_history_size': {
                agent_id: len(history) 
                for agent_id, history in self.adaptation_history.items()
            }
        }


class TacticalAdaptationManager:
    """Main manager for tactical agent adaptation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_engine = TacticalAdaptationEngine(config.get('adaptation_engine', {}))
        
        # Agent registration
        self.registered_agents = {}
        
        # Performance tracking
        self.performance_tracker = defaultdict(list)
        
        logger.info("TacticalAdaptationManager initialized")
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """Register a tactical agent for adaptation"""
        self.registered_agents[agent_id] = agent_instance
        logger.info("Tactical agent registered for adaptation", agent_id=agent_id)
    
    def unregister_agent(self, agent_id: str):
        """Unregister a tactical agent"""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
        logger.info("Tactical agent unregistered", agent_id=agent_id)
    
    async def process_learning_signal(self, signal: LearningSignal):
        """Process a learning signal for tactical agents"""
        if signal.target_agent_level != AgentLevel.TACTICAL:
            return
        
        try:
            # Process for specific agents or all tactical agents
            target_agents = (
                signal.target_agents 
                if signal.target_agents != ['all'] 
                else list(self.registered_agents.keys())
            )
            
            for agent_id in target_agents:
                if agent_id in self.registered_agents:
                    # Convert learning signal to outcome data
                    outcome_data = self._signal_to_outcome_data(signal)
                    
                    # Adapt agent behavior
                    adaptations = self.adaptation_engine.adapt_from_outcome(outcome_data, agent_id)
                    
                    # Apply adaptations to agent
                    await self._apply_adaptations_to_agent(agent_id, adaptations)
                    
                    # Track performance
                    self.performance_tracker[agent_id].append({
                        'timestamp': datetime.now(),
                        'signal_type': signal.signal_type,
                        'adaptations': list(adaptations.keys()),
                        'reward_adjustment': signal.reward_adjustment
                    })
            
        except Exception as e:
            logger.error("Error processing tactical learning signal",
                        signal_id=signal.signal_id,
                        error=str(e))
    
    def _signal_to_outcome_data(self, signal: LearningSignal) -> Dict[str, Any]:
        """Convert learning signal to outcome data format"""
        return {
            'timestamp': datetime.now(),
            'slippage': signal.behavior_adjustment.get('slippage_penalty', 0.0),
            'execution_time': signal.behavior_adjustment.get('execution_time', 1.0),
            'fill_quality': 1.0 - signal.behavior_adjustment.get('quality_penalty', 0.0),
            'market_impact': signal.behavior_adjustment.get('market_impact', 0.0),
            'venue': signal.behavior_adjustment.get('venue', 'default'),
            'market_conditions': signal.behavior_adjustment.get('market_conditions', {})
        }
    
    async def _apply_adaptations_to_agent(self, agent_id: str, adaptations: Dict[str, Any]):
        """Apply adaptations to a tactical agent"""
        try:
            agent = self.registered_agents[agent_id]
            
            # Apply timing adaptations
            if 'timing_sensitivity' in adaptations and hasattr(agent, 'set_timing_sensitivity'):
                agent.set_timing_sensitivity(adaptations['timing_sensitivity'])
            
            # Apply slippage adaptations
            if 'slippage_tolerance' in adaptations and hasattr(agent, 'set_slippage_tolerance'):
                agent.set_slippage_tolerance(adaptations['slippage_tolerance'])
            
            # Apply venue adaptations
            if 'venue_preferences' in adaptations and hasattr(agent, 'update_venue_preferences'):
                agent.update_venue_preferences(adaptations['venue_preferences'])
            
            # Apply execution urgency
            if 'execution_urgency' in adaptations and hasattr(agent, 'set_execution_urgency'):
                agent.set_execution_urgency(adaptations['execution_urgency'])
            
            # Apply exploration rate
            if 'exploration_rate' in adaptations and hasattr(agent, 'set_exploration_rate'):
                agent.set_exploration_rate(adaptations['exploration_rate'])
            
            logger.debug("Adaptations applied to tactical agent",
                        agent_id=agent_id,
                        adaptations=list(adaptations.keys()))
            
        except Exception as e:
            logger.error("Error applying adaptations to tactical agent",
                        agent_id=agent_id,
                        error=str(e))
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get adaptation state for an agent"""
        if agent_id in self.adaptation_engine.agent_states:
            return self.adaptation_engine.agent_states[agent_id].to_dict()
        return None
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'registered_agents': list(self.registered_agents.keys()),
            'adaptation_engine_stats': self.adaptation_engine.get_system_statistics(),
            'performance_tracking': {
                agent_id: len(history) 
                for agent_id, history in self.performance_tracker.items()
            }
        }


# Global instance
tactical_adaptation_manager = TacticalAdaptationManager({
    'adaptation_engine': {
        'adaptation_rate': 0.15,
        'learning_rate': 0.02,
        'memory_decay': 0.9,
        'pattern_analyzer': {
            'pattern_window': 20,
            'success_threshold': 0.7,
            'impact_threshold': 0.005
        }
    }
})