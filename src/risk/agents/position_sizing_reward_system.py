"""
Position Sizing Reward System - Advanced reward function for MARL position sizing optimization

This module implements sophisticated reward functions for training the Position Sizing Agent,
focusing on Kelly Criterion alignment, risk-adjusted returns, and position sizing accuracy.

Key Features:
- Kelly Criterion alignment rewards
- Risk-adjusted return calculation
- Position sizing accuracy bonuses
- Drawdown penalties
- Multi-objective optimization support

Author: Agent 2 - Position Sizing Specialist
Date: 2025-07-13
Mission: Create intelligent reward system for optimal position sizing
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import structlog

logger = structlog.get_logger()


@dataclass
class RewardComponents:
    """Individual components of the position sizing reward"""
    kelly_alignment_reward: float      # Reward for Kelly Criterion alignment
    risk_adjusted_return_reward: float # Risk-adjusted return component
    accuracy_bonus: float              # Position sizing accuracy bonus
    drawdown_penalty: float           # Penalty for excessive drawdowns
    consistency_bonus: float          # Bonus for consistent performance
    efficiency_reward: float          # Reward for computational efficiency
    total_reward: float              # Combined total reward


@dataclass
class TradingOutcome:
    """Trading outcome for reward calculation"""
    contracts_used: int               # Actual contracts used
    entry_price: float               # Entry price
    exit_price: float                # Exit price  
    pnl: float                       # Profit/Loss
    hold_time_minutes: int           # How long position was held
    kelly_suggested_contracts: int   # What Kelly suggested
    agent_suggested_contracts: int   # What agent suggested
    market_conditions: Dict[str, float]  # Market state during trade
    timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for reward calculation"""
    total_trades: int
    winning_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    kelly_accuracy: float
    avg_response_time_ms: float
    consistency_score: float


class PositionSizingRewardSystem:
    """
    Advanced reward system for position sizing agent training.
    
    This system calculates multi-objective rewards that encourage:
    1. Kelly Criterion alignment for mathematical optimality
    2. Risk-adjusted return maximization
    3. Consistent position sizing accuracy
    4. Drawdown minimization
    5. Computational efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward system
        
        Args:
            config: Reward system configuration
        """
        self.config = config
        self.reward_config = config.get('reward_system', {})
        
        # Reward weights
        self.kelly_alignment_weight = self.reward_config.get('kelly_alignment_weight', 0.3)
        self.risk_adjusted_return_weight = self.reward_config.get('risk_adjusted_return_weight', 0.4)
        self.accuracy_bonus_weight = self.reward_config.get('accuracy_bonus_weight', 0.15)
        self.drawdown_penalty_weight = self.reward_config.get('drawdown_penalty_weight', 0.1)
        self.consistency_bonus_weight = self.reward_config.get('consistency_bonus_weight', 0.05)
        
        # Performance tracking
        self.trading_outcomes: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        self.performance_window = self.reward_config.get('performance_window', 100)
        
        # Normalization factors
        self.max_expected_return = self.reward_config.get('max_expected_return', 0.1)  # 10% max expected return
        self.target_sharpe_ratio = self.reward_config.get('target_sharpe_ratio', 2.0)
        self.max_acceptable_drawdown = self.reward_config.get('max_acceptable_drawdown', 0.15)  # 15%
        
        # Kelly accuracy tracking
        self.kelly_accuracy_history: deque = deque(maxlen=self.performance_window)
        self.response_time_history: deque = deque(maxlen=self.performance_window)
        
        logger.info("Position Sizing Reward System initialized",
                   kelly_weight=self.kelly_alignment_weight,
                   return_weight=self.risk_adjusted_return_weight,
                   accuracy_weight=self.accuracy_bonus_weight)
    
    def calculate_reward(self, outcome: TradingOutcome, agent_metrics: Dict[str, Any]) -> RewardComponents:
        """
        Calculate comprehensive reward for a trading outcome
        
        Args:
            outcome: Trading outcome data
            agent_metrics: Current agent performance metrics
            
        Returns:
            RewardComponents with detailed reward breakdown
        """
        try:
            # 1. Kelly Alignment Reward
            kelly_alignment_reward = self._calculate_kelly_alignment_reward(outcome)
            
            # 2. Risk-Adjusted Return Reward
            risk_adjusted_return_reward = self._calculate_risk_adjusted_return_reward(outcome)
            
            # 3. Position Sizing Accuracy Bonus
            accuracy_bonus = self._calculate_accuracy_bonus(outcome, agent_metrics)
            
            # 4. Drawdown Penalty
            drawdown_penalty = self._calculate_drawdown_penalty(agent_metrics)
            
            # 5. Consistency Bonus
            consistency_bonus = self._calculate_consistency_bonus(agent_metrics)
            
            # 6. Efficiency Reward
            efficiency_reward = self._calculate_efficiency_reward(agent_metrics)
            
            # 7. Combine all components
            total_reward = (
                kelly_alignment_reward * self.kelly_alignment_weight +
                risk_adjusted_return_reward * self.risk_adjusted_return_weight +
                accuracy_bonus * self.accuracy_bonus_weight -
                drawdown_penalty * self.drawdown_penalty_weight +
                consistency_bonus * self.consistency_bonus_weight +
                efficiency_reward * 0.05  # Small efficiency bonus
            )
            
            # Create reward components
            reward_components = RewardComponents(
                kelly_alignment_reward=kelly_alignment_reward,
                risk_adjusted_return_reward=risk_adjusted_return_reward,
                accuracy_bonus=accuracy_bonus,
                drawdown_penalty=drawdown_penalty,
                consistency_bonus=consistency_bonus,
                efficiency_reward=efficiency_reward,
                total_reward=total_reward
            )
            
            # Track outcomes and rewards
            self.trading_outcomes.append(outcome)
            self.reward_history.append(reward_components)
            
            return reward_components
            
        except Exception as e:
            logger.error("Error calculating reward", error=str(e))
            return self._get_fallback_reward()
    
    def _calculate_kelly_alignment_reward(self, outcome: TradingOutcome) -> float:
        """
        Calculate reward for Kelly Criterion alignment
        
        Perfect alignment (agent matches Kelly suggestion) = +1.0
        Complete misalignment = -1.0
        """
        try:
            kelly_contracts = outcome.kelly_suggested_contracts
            agent_contracts = outcome.agent_suggested_contracts
            
            # Calculate alignment score
            max_deviation = 4  # Maximum possible deviation (5-1)
            actual_deviation = abs(kelly_contracts - agent_contracts)
            
            # Normalize to [-1, 1] range
            alignment_score = 1.0 - (actual_deviation / max_deviation)
            
            # Bonus for exact match
            if kelly_contracts == agent_contracts:
                alignment_score += 0.2  # Extra bonus for perfect alignment
            
            # Penalty for large deviations
            if actual_deviation >= 3:
                alignment_score -= 0.5  # Large penalty for major deviations
            
            return max(-1.0, min(1.0, alignment_score))
            
        except Exception as e:
            logger.error("Error calculating Kelly alignment reward", error=str(e))
            return 0.0
    
    def _calculate_risk_adjusted_return_reward(self, outcome: TradingOutcome) -> float:
        """
        Calculate risk-adjusted return reward using Sharpe-like metric
        
        High risk-adjusted returns = +1.0
        Poor risk-adjusted returns = -1.0
        """
        try:
            # Calculate return percentage
            if outcome.entry_price <= 0:
                return 0.0
            
            return_pct = (outcome.exit_price - outcome.entry_price) / outcome.entry_price
            
            # Estimate risk from market conditions
            volatility = outcome.market_conditions.get('volatility_regime', 0.2)
            market_stress = outcome.market_conditions.get('market_stress_level', 0.3)
            
            # Combine volatility and stress for risk estimate
            risk_estimate = max(0.01, (volatility + market_stress) / 2)
            
            # Calculate risk-adjusted return (Sharpe-like)
            risk_adjusted_return = return_pct / risk_estimate
            
            # Normalize to [-1, 1] range
            normalized_return = np.tanh(risk_adjusted_return * 10)  # Scale and bound
            
            # Bonus for positive returns
            if return_pct > 0:
                normalized_return += 0.1
            
            return max(-1.0, min(1.0, normalized_return))
            
        except Exception as e:
            logger.error("Error calculating risk-adjusted return reward", error=str(e))
            return 0.0
    
    def _calculate_accuracy_bonus(self, outcome: TradingOutcome, agent_metrics: Dict[str, Any]) -> float:
        """
        Calculate bonus for consistent position sizing accuracy
        
        High accuracy over time = +1.0
        Poor accuracy = 0.0
        """
        try:
            # Get recent Kelly accuracy
            kelly_accuracy = agent_metrics.get('kelly_accuracy_avg', 0.0)
            
            # Normalize accuracy to [0, 1] range
            accuracy_bonus = min(1.0, kelly_accuracy)
            
            # Extra bonus for exceeding target accuracy (95%)
            if kelly_accuracy >= 0.95:
                accuracy_bonus += 0.2
            
            # Bonus for consistency (low variance in accuracy)
            if len(self.kelly_accuracy_history) >= 10:
                accuracy_variance = np.var(self.kelly_accuracy_history)
                consistency_bonus = max(0, 0.2 - accuracy_variance * 10)
                accuracy_bonus += consistency_bonus
            
            return max(0.0, min(1.0, accuracy_bonus))
            
        except Exception as e:
            logger.error("Error calculating accuracy bonus", error=str(e))
            return 0.0
    
    def _calculate_drawdown_penalty(self, agent_metrics: Dict[str, Any]) -> float:
        """
        Calculate penalty for excessive drawdowns
        
        Low drawdown = 0.0 penalty
        High drawdown = +1.0 penalty (subtracted from total)
        """
        try:
            current_drawdown = agent_metrics.get('current_drawdown_pct', 0.0)
            max_drawdown = agent_metrics.get('max_drawdown', 0.0)
            
            # Use the worse of current or max recent drawdown
            worst_drawdown = max(current_drawdown, max_drawdown)
            
            # Calculate penalty based on acceptable threshold
            if worst_drawdown <= 0.05:  # 5% or less is good
                return 0.0
            elif worst_drawdown <= self.max_acceptable_drawdown:  # Up to 15% is acceptable
                return (worst_drawdown - 0.05) / (self.max_acceptable_drawdown - 0.05) * 0.5
            else:  # Above 15% is heavily penalized
                return 0.5 + min(0.5, (worst_drawdown - self.max_acceptable_drawdown) * 2)
            
        except Exception as e:
            logger.error("Error calculating drawdown penalty", error=str(e))
            return 0.0
    
    def _calculate_consistency_bonus(self, agent_metrics: Dict[str, Any]) -> float:
        """
        Calculate bonus for consistent performance over time
        
        Highly consistent = +1.0
        Inconsistent = 0.0
        """
        try:
            if len(self.trading_outcomes) < 20:  # Need minimum history
                return 0.0
            
            # Calculate consistency metrics
            recent_outcomes = list(self.trading_outcomes)[-20:]  # Last 20 trades
            
            # PnL consistency
            pnls = [outcome.pnl for outcome in recent_outcomes]
            pnl_mean = np.mean(pnls)
            pnl_std = np.std(pnls)
            
            if pnl_std == 0:
                pnl_consistency = 1.0
            else:
                pnl_consistency = max(0, 1.0 - (pnl_std / max(abs(pnl_mean), 0.01)))
            
            # Position sizing consistency
            contract_variations = [abs(outcome.agent_suggested_contracts - outcome.kelly_suggested_contracts) 
                                 for outcome in recent_outcomes]
            avg_variation = np.mean(contract_variations)
            sizing_consistency = max(0, 1.0 - avg_variation / 4.0)  # Normalize by max possible variation
            
            # Combine consistency metrics
            overall_consistency = (pnl_consistency + sizing_consistency) / 2
            
            return max(0.0, min(1.0, overall_consistency))
            
        except Exception as e:
            logger.error("Error calculating consistency bonus", error=str(e))
            return 0.0
    
    def _calculate_efficiency_reward(self, agent_metrics: Dict[str, Any]) -> float:
        """
        Calculate reward for computational efficiency
        
        Fast response time = +1.0
        Slow response time = 0.0
        """
        try:
            avg_response_time = agent_metrics.get('avg_response_time_ms', 10.0)
            target_response_time = 10.0  # 10ms target
            
            if avg_response_time <= target_response_time:
                return 1.0
            else:
                # Linear decay for slower response times
                efficiency = max(0.0, 1.0 - (avg_response_time - target_response_time) / target_response_time)
                return efficiency
            
        except Exception as e:
            logger.error("Error calculating efficiency reward", error=str(e))
            return 0.0
    
    def _get_fallback_reward(self) -> RewardComponents:
        """Get fallback reward components for error cases"""
        return RewardComponents(
            kelly_alignment_reward=0.0,
            risk_adjusted_return_reward=0.0,
            accuracy_bonus=0.0,
            drawdown_penalty=0.0,
            consistency_bonus=0.0,
            efficiency_reward=0.0,
            total_reward=0.0
        )
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics"""
        if not self.reward_history:
            return {'status': 'no_rewards_calculated'}
        
        recent_rewards = list(self.reward_history)[-self.performance_window:]
        
        return {
            'total_rewards_calculated': len(self.reward_history),
            'recent_performance': {
                'avg_total_reward': np.mean([r.total_reward for r in recent_rewards]),
                'avg_kelly_alignment': np.mean([r.kelly_alignment_reward for r in recent_rewards]),
                'avg_risk_adjusted_return': np.mean([r.risk_adjusted_return_reward for r in recent_rewards]),
                'avg_accuracy_bonus': np.mean([r.accuracy_bonus for r in recent_rewards]),
                'avg_drawdown_penalty': np.mean([r.drawdown_penalty for r in recent_rewards]),
                'avg_consistency_bonus': np.mean([r.consistency_bonus for r in recent_rewards])
            },
            'reward_distribution': {
                'positive_rewards': sum(1 for r in recent_rewards if r.total_reward > 0),
                'negative_rewards': sum(1 for r in recent_rewards if r.total_reward < 0),
                'neutral_rewards': sum(1 for r in recent_rewards if r.total_reward == 0)
            },
            'performance_trends': {
                'improving_kelly_alignment': self._calculate_trend([r.kelly_alignment_reward for r in recent_rewards]),
                'improving_returns': self._calculate_trend([r.risk_adjusted_return_reward for r in recent_rewards]),
                'improving_accuracy': self._calculate_trend([r.accuracy_bonus for r in recent_rewards])
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Simple linear trend
        recent_half = values[-len(values)//2:]
        earlier_half = values[:len(values)//2]
        
        recent_avg = np.mean(recent_half)
        earlier_avg = np.mean(earlier_half)
        
        if recent_avg > earlier_avg * 1.05:
            return 'improving'
        elif recent_avg < earlier_avg * 0.95:
            return 'declining'
        else:
            return 'stable'
    
    def create_training_sample(self, outcome: TradingOutcome, reward_components: RewardComponents) -> Dict[str, Any]:
        """
        Create training sample for MARL learning
        
        Args:
            outcome: Trading outcome
            reward_components: Calculated reward components
            
        Returns:
            Training sample dictionary
        """
        return {
            'state_features': {
                'market_conditions': outcome.market_conditions,
                'kelly_fraction': outcome.kelly_suggested_contracts / 5.0,  # Normalize
                'contracts_used': outcome.contracts_used,
                'hold_time_minutes': outcome.hold_time_minutes
            },
            'action': outcome.agent_suggested_contracts - 1,  # Convert to 0-4 range
            'reward': reward_components.total_reward,
            'reward_breakdown': {
                'kelly_alignment': reward_components.kelly_alignment_reward,
                'risk_adjusted_return': reward_components.risk_adjusted_return_reward,
                'accuracy_bonus': reward_components.accuracy_bonus,
                'drawdown_penalty': reward_components.drawdown_penalty,
                'consistency_bonus': reward_components.consistency_bonus,
                'efficiency_reward': reward_components.efficiency_reward
            },
            'outcome_metrics': {
                'pnl': outcome.pnl,
                'return_pct': (outcome.exit_price - outcome.entry_price) / outcome.entry_price if outcome.entry_price > 0 else 0,
                'kelly_accuracy': 1.0 - abs(outcome.kelly_suggested_contracts - outcome.agent_suggested_contracts) / 4.0
            },
            'timestamp': outcome.timestamp
        }
    
    def update_performance_tracking(self, kelly_accuracy: float, response_time_ms: float):
        """Update performance tracking metrics"""
        self.kelly_accuracy_history.append(kelly_accuracy)
        self.response_time_history.append(response_time_ms)


def create_position_sizing_reward_system(config: Dict[str, Any]) -> PositionSizingRewardSystem:
    """
    Factory function to create position sizing reward system
    
    Args:
        config: Reward system configuration
        
    Returns:
        Configured PositionSizingRewardSystem instance
    """
    default_config = {
        'reward_system': {
            'kelly_alignment_weight': 0.3,
            'risk_adjusted_return_weight': 0.4,
            'accuracy_bonus_weight': 0.15,
            'drawdown_penalty_weight': 0.1,
            'consistency_bonus_weight': 0.05,
            'performance_window': 100,
            'max_expected_return': 0.1,
            'target_sharpe_ratio': 2.0,
            'max_acceptable_drawdown': 0.15
        }
    }
    
    merged_config = {**default_config, **config}
    return PositionSizingRewardSystem(merged_config)