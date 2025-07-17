"""
PettingZoo Reward System for MARL Training

This module implements a sophisticated reward system designed specifically for
PettingZoo environments, handling the turn-based nature of agent interactions
while providing multi-objective rewards and advanced reward shaping.

Key Features:
- Turn-based reward calculation compatible with PettingZoo AEC environments
- Multi-objective reward optimization
- Advanced reward shaping and normalization
- Agent-specific reward components
- Cooperative and competitive reward structures
- Reward history tracking and analysis
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
import threading
from abc import ABC, abstractmethod

# PettingZoo imports
from pettingzoo import AECEnv

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of rewards in the system"""
    INDIVIDUAL = "individual"
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    SYSTEM = "system"


class RewardComponent(Enum):
    """Individual reward components"""
    PERFORMANCE = "performance"
    RISK_ADJUSTED = "risk_adjusted"
    TIMING = "timing"
    COOPERATION = "cooperation"
    EXPLORATION = "exploration"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"


@dataclass
class RewardConfig:
    """Configuration for reward system"""
    # Component weights
    component_weights: Dict[RewardComponent, float] = field(default_factory=lambda: {
        RewardComponent.PERFORMANCE: 0.4,
        RewardComponent.RISK_ADJUSTED: 0.2,
        RewardComponent.TIMING: 0.15,
        RewardComponent.COOPERATION: 0.1,
        RewardComponent.EXPLORATION: 0.05,
        RewardComponent.CONSISTENCY: 0.05,
        RewardComponent.EFFICIENCY: 0.05
    })
    
    # Reward scaling
    reward_scale: float = 1.0
    normalize_rewards: bool = True
    clip_rewards: bool = True
    reward_clip_range: Tuple[float, float] = (-10.0, 10.0)
    
    # Multi-objective settings
    enable_multi_objective: bool = True
    pareto_optimization: bool = False
    
    # Cooperative/competitive balance
    cooperation_weight: float = 0.3
    competition_weight: float = 0.7
    
    # Reward shaping
    enable_reward_shaping: bool = True
    potential_function: Optional[Callable] = None
    discount_factor: float = 0.99
    
    # Exploration incentives
    exploration_bonus: float = 0.1
    novelty_threshold: float = 0.1
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 0.9,
        'good': 0.7,
        'acceptable': 0.5,
        'poor': 0.3
    })
    
    # Adaptive reward settings
    adaptive_weights: bool = True
    adaptation_rate: float = 0.01
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    # History tracking
    track_reward_history: bool = True
    history_length: int = 10000


class RewardCalculator(ABC):
    """Abstract base class for reward calculators"""
    
    @abstractmethod
    def calculate_reward(self, agent: str, state: Dict[str, Any], 
                        action: Any, next_state: Dict[str, Any],
                        info: Dict[str, Any]) -> float:
        """Calculate reward for agent action"""
        pass
    
    @abstractmethod
    def get_reward_components(self, agent: str, state: Dict[str, Any],
                            action: Any, next_state: Dict[str, Any],
                            info: Dict[str, Any]) -> Dict[RewardComponent, float]:
        """Get individual reward components"""
        pass


class TradingRewardCalculator(RewardCalculator):
    """Reward calculator for trading agents"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.agent_histories = defaultdict(lambda: deque(maxlen=config.history_length))
        
    def calculate_reward(self, agent: str, state: Dict[str, Any], 
                        action: Any, next_state: Dict[str, Any],
                        info: Dict[str, Any]) -> float:
        """Calculate trading reward"""
        components = self.get_reward_components(agent, state, action, next_state, info)
        
        # Weighted sum of components
        total_reward = sum(
            self.config.component_weights.get(comp, 0.0) * value
            for comp, value in components.items()
        )
        
        # Apply reward scaling
        total_reward *= self.config.reward_scale
        
        # Apply reward shaping if enabled
        if self.config.enable_reward_shaping:
            total_reward = self._apply_reward_shaping(
                agent, total_reward, state, next_state
            )
        
        # Normalize if enabled
        if self.config.normalize_rewards:
            total_reward = self._normalize_reward(agent, total_reward)
        
        # Clip if enabled
        if self.config.clip_rewards:
            total_reward = np.clip(
                total_reward, 
                self.config.reward_clip_range[0], 
                self.config.reward_clip_range[1]
            )
        
        # Track history
        if self.config.track_reward_history:
            self.agent_histories[agent].append({
                'reward': total_reward,
                'components': components,
                'timestamp': datetime.now().isoformat()
            })
        
        return float(total_reward)
    
    def get_reward_components(self, agent: str, state: Dict[str, Any],
                            action: Any, next_state: Dict[str, Any],
                            info: Dict[str, Any]) -> Dict[RewardComponent, float]:
        """Get individual reward components for trading"""
        components = {}
        
        # Performance component
        components[RewardComponent.PERFORMANCE] = self._calculate_performance_reward(
            agent, state, action, next_state, info
        )
        
        # Risk-adjusted component
        components[RewardComponent.RISK_ADJUSTED] = self._calculate_risk_adjusted_reward(
            agent, state, action, next_state, info
        )
        
        # Timing component
        components[RewardComponent.TIMING] = self._calculate_timing_reward(
            agent, state, action, next_state, info
        )
        
        # Cooperation component
        components[RewardComponent.COOPERATION] = self._calculate_cooperation_reward(
            agent, state, action, next_state, info
        )
        
        # Exploration component
        components[RewardComponent.EXPLORATION] = self._calculate_exploration_reward(
            agent, state, action, next_state, info
        )
        
        # Consistency component
        components[RewardComponent.CONSISTENCY] = self._calculate_consistency_reward(
            agent, state, action, next_state, info
        )
        
        # Efficiency component
        components[RewardComponent.EFFICIENCY] = self._calculate_efficiency_reward(
            agent, state, action, next_state, info
        )
        
        return components
    
    def _calculate_performance_reward(self, agent: str, state: Dict[str, Any],
                                    action: Any, next_state: Dict[str, Any],
                                    info: Dict[str, Any]) -> float:
        """Calculate performance-based reward"""
        # Base performance from market outcome
        base_reward = info.get('market_reward', 0.0)
        
        # Confidence scaling
        confidence = info.get('confidence', 0.5)
        confidence_multiplier = 1.0 + (confidence - 0.5)
        
        # Action quality
        action_quality = info.get('action_quality', 0.5)
        
        performance_reward = base_reward * confidence_multiplier * action_quality
        
        return performance_reward
    
    def _calculate_risk_adjusted_reward(self, agent: str, state: Dict[str, Any],
                                      action: Any, next_state: Dict[str, Any],
                                      info: Dict[str, Any]) -> float:
        """Calculate risk-adjusted reward"""
        raw_return = info.get('raw_return', 0.0)
        volatility = info.get('volatility', 0.1)
        
        # Sharpe ratio-like calculation
        if volatility > 0:
            risk_adjusted = raw_return / volatility
        else:
            risk_adjusted = raw_return
        
        # Penalize high risk
        risk_penalty = info.get('risk_penalty', 0.0)
        risk_adjusted -= risk_penalty
        
        return np.tanh(risk_adjusted)  # Bounded between -1 and 1
    
    def _calculate_timing_reward(self, agent: str, state: Dict[str, Any],
                               action: Any, next_state: Dict[str, Any],
                               info: Dict[str, Any]) -> float:
        """Calculate timing-based reward"""
        # Reward good timing
        entry_timing = info.get('entry_timing_score', 0.0)
        exit_timing = info.get('exit_timing_score', 0.0)
        
        # Market regime alignment
        regime_alignment = info.get('regime_alignment', 0.0)
        
        timing_reward = (entry_timing + exit_timing) * 0.5 + regime_alignment * 0.2
        
        return timing_reward
    
    def _calculate_cooperation_reward(self, agent: str, state: Dict[str, Any],
                                    action: Any, next_state: Dict[str, Any],
                                    info: Dict[str, Any]) -> float:
        """Calculate cooperation reward"""
        # Reward alignment with other agents
        agent_alignment = info.get('agent_alignment', 0.0)
        
        # Consensus participation
        consensus_participation = info.get('consensus_participation', 0.0)
        
        # Information sharing
        information_sharing = info.get('information_sharing', 0.0)
        
        cooperation_reward = (
            agent_alignment * 0.5 +
            consensus_participation * 0.3 +
            information_sharing * 0.2
        )
        
        return cooperation_reward
    
    def _calculate_exploration_reward(self, agent: str, state: Dict[str, Any],
                                    action: Any, next_state: Dict[str, Any],
                                    info: Dict[str, Any]) -> float:
        """Calculate exploration reward"""
        # Reward novel actions
        action_novelty = info.get('action_novelty', 0.0)
        
        # State space exploration
        state_novelty = info.get('state_novelty', 0.0)
        
        # Exploration vs exploitation balance
        exploration_value = info.get('exploration_value', 0.0)
        
        exploration_reward = (
            action_novelty * 0.4 +
            state_novelty * 0.3 +
            exploration_value * 0.3
        )
        
        return exploration_reward * self.config.exploration_bonus
    
    def _calculate_consistency_reward(self, agent: str, state: Dict[str, Any],
                                    action: Any, next_state: Dict[str, Any],
                                    info: Dict[str, Any]) -> float:
        """Calculate consistency reward"""
        # Reward consistent behavior
        behavior_consistency = info.get('behavior_consistency', 0.0)
        
        # Strategy adherence
        strategy_adherence = info.get('strategy_adherence', 0.0)
        
        # Performance stability
        performance_stability = info.get('performance_stability', 0.0)
        
        consistency_reward = (
            behavior_consistency * 0.4 +
            strategy_adherence * 0.3 +
            performance_stability * 0.3
        )
        
        return consistency_reward
    
    def _calculate_efficiency_reward(self, agent: str, state: Dict[str, Any],
                                   action: Any, next_state: Dict[str, Any],
                                   info: Dict[str, Any]) -> float:
        """Calculate efficiency reward"""
        # Computational efficiency
        computation_time = info.get('computation_time', 0.1)
        time_efficiency = max(0, 1 - computation_time / 0.1)  # Normalize to 0.1s
        
        # Resource utilization
        resource_efficiency = info.get('resource_efficiency', 0.5)
        
        # Action efficiency
        action_efficiency = info.get('action_efficiency', 0.5)
        
        efficiency_reward = (
            time_efficiency * 0.4 +
            resource_efficiency * 0.3 +
            action_efficiency * 0.3
        )
        
        return efficiency_reward
    
    def _apply_reward_shaping(self, agent: str, reward: float, 
                            state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """Apply reward shaping using potential function"""
        if self.config.potential_function is None:
            return reward
        
        # Calculate potential difference
        phi_next = self.config.potential_function(next_state)
        phi_current = self.config.potential_function(state)
        
        # Shaped reward = original + γ * Φ(s') - Φ(s)
        shaped_reward = reward + self.config.discount_factor * (phi_next - phi_current)
        
        return shaped_reward
    
    def _normalize_reward(self, agent: str, reward: float) -> float:
        """Normalize reward using running statistics"""
        if len(self.agent_histories[agent]) < 2:
            return reward
        
        # Calculate running mean and std
        recent_rewards = [entry['reward'] for entry in self.agent_histories[agent]]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        if std_reward > 0:
            normalized = (reward - mean_reward) / std_reward
        else:
            normalized = reward
        
        return normalized


class MultiObjectiveRewardCalculator(RewardCalculator):
    """Multi-objective reward calculator"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.calculators = {
            'trading': TradingRewardCalculator(config)
        }
        self.objective_weights = {
            'profit': 0.4,
            'risk': 0.3,
            'efficiency': 0.2,
            'cooperation': 0.1
        }
        
    def calculate_reward(self, agent: str, state: Dict[str, Any], 
                        action: Any, next_state: Dict[str, Any],
                        info: Dict[str, Any]) -> Union[float, np.ndarray]:
        """Calculate multi-objective reward"""
        # Get objectives
        objectives = self._calculate_objectives(agent, state, action, next_state, info)
        
        if self.config.pareto_optimization:
            # Return multi-dimensional reward for Pareto optimization
            return np.array(list(objectives.values()))
        else:
            # Return scalar reward as weighted sum
            return sum(
                self.objective_weights.get(obj, 0.0) * value
                for obj, value in objectives.items()
            )
    
    def get_reward_components(self, agent: str, state: Dict[str, Any],
                            action: Any, next_state: Dict[str, Any],
                            info: Dict[str, Any]) -> Dict[RewardComponent, float]:
        """Get reward components from base calculator"""
        return self.calculators['trading'].get_reward_components(
            agent, state, action, next_state, info
        )
    
    def _calculate_objectives(self, agent: str, state: Dict[str, Any],
                            action: Any, next_state: Dict[str, Any],
                            info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual objectives"""
        objectives = {}
        
        # Profit objective
        objectives['profit'] = info.get('profit', 0.0)
        
        # Risk objective (negative risk is good)
        objectives['risk'] = -info.get('risk', 0.0)
        
        # Efficiency objective
        objectives['efficiency'] = info.get('efficiency', 0.0)
        
        # Cooperation objective
        objectives['cooperation'] = info.get('cooperation', 0.0)
        
        return objectives


class PettingZooRewardSystem:
    """Main reward system for PettingZoo environments"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.calculators = {}
        self.reward_history = defaultdict(lambda: deque(maxlen=config.history_length))
        self.adaptive_weights = config.component_weights.copy()
        
        # Initialize calculators
        self._initialize_calculators()
        
        # Performance tracking
        self.performance_metrics = {
            'total_rewards': defaultdict(float),
            'reward_variance': defaultdict(float),
            'component_contributions': defaultdict(lambda: defaultdict(float))
        }
        
        # Threading for adaptive weights
        self.weight_lock = threading.Lock()
        
    def _initialize_calculators(self):
        """Initialize reward calculators"""
        if self.config.enable_multi_objective:
            self.calculators['default'] = MultiObjectiveRewardCalculator(self.config)
        else:
            self.calculators['default'] = TradingRewardCalculator(self.config)
    
    def calculate_agent_reward(self, agent: str, env: AECEnv, 
                             action: Any, pre_step_state: Dict[str, Any],
                             post_step_info: Dict[str, Any]) -> float:
        """Calculate reward for agent in PettingZoo environment"""
        # Get environment reward
        env_reward = env.rewards.get(agent, 0.0)
        
        # Prepare state information
        state = {
            'agent': agent,
            'observation': pre_step_state.get('observation'),
            'env_reward': env_reward,
            'done': env.dones.get(agent, False),
            'truncated': env.truncations.get(agent, False)
        }
        
        # Get next state if available
        next_state = {}
        if agent in env.agents:
            next_state = {
                'agent': agent,
                'observation': env.observe(agent),
                'agents_remaining': len(env.agents)
            }
        
        # Enhance info with environment context
        enhanced_info = post_step_info.copy()
        enhanced_info.update({
            'env_reward': env_reward,
            'agent_count': len(env.agents),
            'episode_step': enhanced_info.get('episode_step', 0),
            'market_reward': env_reward  # Use env reward as base market reward
        })
        
        # Calculate reward using appropriate calculator
        calculator = self.calculators.get(agent, self.calculators['default'])
        calculated_reward = calculator.calculate_reward(
            agent, state, action, next_state, enhanced_info
        )
        
        # Apply cooperative/competitive adjustments
        adjusted_reward = self._apply_multi_agent_adjustments(
            agent, calculated_reward, env, enhanced_info
        )
        
        # Update adaptive weights if enabled
        if self.config.adaptive_weights:
            self._update_adaptive_weights(agent, calculator, enhanced_info)
        
        # Track performance metrics
        self._update_performance_metrics(agent, adjusted_reward, calculator, enhanced_info)
        
        return adjusted_reward
    
    def _apply_multi_agent_adjustments(self, agent: str, reward: float,
                                     env: AECEnv, info: Dict[str, Any]) -> float:
        """Apply multi-agent cooperative/competitive adjustments"""
        # Get other agents' recent performance
        other_agents = [a for a in env.possible_agents if a != agent]
        
        if not other_agents:
            return reward
        
        # Calculate cooperation bonus
        cooperation_bonus = 0.0
        if self.config.cooperation_weight > 0:
            # Reward alignment with other agents
            agent_alignment = info.get('agent_alignment', 0.0)
            cooperation_bonus = self.config.cooperation_weight * agent_alignment
        
        # Calculate competition component
        competition_component = 0.0
        if self.config.competition_weight > 0:
            # Reward outperforming other agents
            relative_performance = info.get('relative_performance', 0.0)
            competition_component = self.config.competition_weight * relative_performance
        
        # Combined adjustment
        adjusted_reward = reward + cooperation_bonus + competition_component
        
        return adjusted_reward
    
    def _update_adaptive_weights(self, agent: str, calculator: RewardCalculator,
                               info: Dict[str, Any]):
        """Update adaptive weights based on performance"""
        if not hasattr(calculator, 'get_reward_components'):
            return
        
        # Get current components
        components = calculator.get_reward_components(agent, {}, None, {}, info)
        
        # Performance-based weight adaptation
        performance = info.get('performance', 0.0)
        adaptation_factor = self.config.adaptation_rate * performance
        
        with self.weight_lock:
            for component, value in components.items():
                current_weight = self.adaptive_weights.get(component, 0.0)
                
                # Increase weight for positive contributions
                if value > 0:
                    new_weight = current_weight + adaptation_factor
                else:
                    new_weight = current_weight - adaptation_factor
                
                # Clamp weights
                new_weight = np.clip(
                    new_weight, 
                    self.config.min_weight, 
                    self.config.max_weight
                )
                
                self.adaptive_weights[component] = new_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(self.adaptive_weights.values())
            if total_weight > 0:
                for component in self.adaptive_weights:
                    self.adaptive_weights[component] /= total_weight
    
    def _update_performance_metrics(self, agent: str, reward: float,
                                  calculator: RewardCalculator, info: Dict[str, Any]):
        """Update performance metrics"""
        # Update total rewards
        self.performance_metrics['total_rewards'][agent] += reward
        
        # Update reward variance
        agent_history = self.reward_history[agent]
        if len(agent_history) > 1:
            recent_rewards = [entry['reward'] for entry in agent_history]
            variance = np.var(recent_rewards)
            self.performance_metrics['reward_variance'][agent] = variance
        
        # Update component contributions
        if hasattr(calculator, 'get_reward_components'):
            components = calculator.get_reward_components(agent, {}, None, {}, info)
            for component, value in components.items():
                self.performance_metrics['component_contributions'][agent][component] += value
        
        # Store in history
        self.reward_history[agent].append({
            'reward': reward,
            'timestamp': datetime.now().isoformat(),
            'info': info
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'total_rewards': dict(self.performance_metrics['total_rewards']),
            'reward_variance': dict(self.performance_metrics['reward_variance']),
            'component_contributions': {
                agent: dict(components) 
                for agent, components in self.performance_metrics['component_contributions'].items()
            },
            'adaptive_weights': dict(self.adaptive_weights),
            'config': self.config
        }
        
        return summary
    
    def reset_agent_history(self, agent: str):
        """Reset history for specific agent"""
        self.reward_history[agent].clear()
        self.performance_metrics['total_rewards'][agent] = 0.0
        self.performance_metrics['reward_variance'][agent] = 0.0
        self.performance_metrics['component_contributions'][agent].clear()
    
    def save_reward_history(self, filepath: str):
        """Save reward history to file"""
        data = {
            'reward_history': {
                agent: list(history) 
                for agent, history in self.reward_history.items()
            },
            'performance_metrics': self.performance_metrics,
            'adaptive_weights': self.adaptive_weights,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Reward history saved to {filepath}")


def create_reward_config(**kwargs) -> RewardConfig:
    """Create reward configuration with defaults"""
    return RewardConfig(**kwargs)


def create_trading_reward_system(config: RewardConfig = None) -> PettingZooRewardSystem:
    """Create reward system for trading environments"""
    if config is None:
        config = create_reward_config()
    
    return PettingZooRewardSystem(config)


# Example usage
if __name__ == "__main__":
    # Create reward system
    reward_config = create_reward_config(
        component_weights={
            RewardComponent.PERFORMANCE: 0.5,
            RewardComponent.RISK_ADJUSTED: 0.3,
            RewardComponent.COOPERATION: 0.2
        },
        adaptive_weights=True,
        enable_multi_objective=True
    )
    
    reward_system = create_trading_reward_system(reward_config)
    
    # Example usage with mock environment
    class MockEnv:
        def __init__(self):
            self.rewards = {'agent1': 0.5}
            self.dones = {'agent1': False}
            self.truncations = {'agent1': False}
            self.agents = ['agent1']
            self.possible_agents = ['agent1']
        
        def observe(self, agent):
            return np.random.randn(10)
    
    env = MockEnv()
    
    # Calculate reward
    reward = reward_system.calculate_agent_reward(
        agent='agent1',
        env=env,
        action=1,
        pre_step_state={'observation': np.random.randn(10)},
        post_step_info={
            'performance': 0.8,
            'risk': 0.2,
            'cooperation': 0.6
        }
    )
    
    print(f"Calculated reward: {reward:.3f}")
    
    # Get performance summary
    summary = reward_system.get_performance_summary()
    print(f"Performance summary: {summary}")