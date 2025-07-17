"""
Multi-Objective Reward System for Strategic MARL

Implements the reward function with granular components as specified in the PRD:
R_total = α·R_pnl + β·R_synergy + γ·R_risk + δ·R_exploration
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import yaml
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class RewardComponents:
    """Container for granular reward components"""
    pnl: float
    synergy: float
    risk: float
    exploration: float
    total: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'pnl': self.pnl,
            'synergy': self.synergy,
            'risk': self.risk,
            'exploration': self.exploration,
            'total': self.total
        }


class RewardSystem:
    """
    Multi-Objective Reward System
    
    Calculates rewards based on trading performance, synergy alignment,
    risk management, and exploration incentives. Returns granular components
    for detailed analysis and monitoring.
    
    Key Features:
    - PnL reward with tanh normalization
    - Synergy alignment bonus using cosine similarity
    - Risk penalty based on drawdown and position sizing
    - Exploration bonus from action entropy
    - Configuration-driven weights
    - Running statistics for normalization
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Reward System
        
        Args:
            config_path: Path to configuration YAML file
            config: Direct configuration dictionary (overrides config_path)
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Extract reward weights
        self.alpha = self.config['rewards']['alpha']  # PnL weight
        self.beta = self.config['rewards']['beta']    # Synergy weight
        self.gamma = self.config['rewards']['gamma']  # Risk weight (negative)
        self.delta = self.config['rewards']['delta']  # Exploration weight
        
        # Risk parameters
        self.max_drawdown_threshold = self.config['rewards'].get('max_drawdown', 0.15)
        self.position_size_limit = self.config['rewards'].get('position_limit', 1.0)
        
        # Normalization parameters
        self.pnl_normalizer = self.config['rewards'].get('pnl_normalizer', 100.0)
        self.use_running_stats = self.config['rewards'].get('use_running_stats', True)
        
        # Running statistics for adaptive normalization
        self.running_pnl_mean = 0.0
        self.running_pnl_std = 1.0
        self.stats_update_alpha = 0.99  # EMA decay
        self.stats_count = 0
        
        # Performance tracking
        self.reward_history = []
        
        logger.info(f"RewardSystem initialized with weights: α={self.alpha}, β={self.beta}, γ={self.gamma}, δ={self.delta}")
    
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> RewardComponents:
        """
        Calculate multi-objective reward with granular components
        
        Args:
            state: Current environment state
            action: Agent action (probability distribution)
            next_state: Next environment state
            info: Additional information (synergy, market context, etc.)
            
        Returns:
            RewardComponents with individual and total rewards
        """
        # Calculate individual reward components
        r_pnl = self._calculate_pnl_reward(state, next_state, info)
        r_synergy = self._calculate_synergy_reward(action, info)
        r_risk = self._calculate_risk_penalty(state, action, info)
        r_exploration = self._calculate_exploration_bonus(action)
        
        # Calculate total reward
        r_total = (
            self.alpha * r_pnl +
            self.beta * r_synergy +
            self.gamma * r_risk +
            self.delta * r_exploration
        )
        
        # Create reward components object
        rewards = RewardComponents(
            pnl=r_pnl,
            synergy=r_synergy,
            risk=r_risk,
            exploration=r_exploration,
            total=r_total
        )
        
        # Update statistics
        self._update_statistics(rewards, info)
        
        # Store in history
        self.reward_history.append({
            'rewards': rewards.to_dict(),
            'action': action.tolist(),
            'info': info
        })
        
        return rewards
    
    def _calculate_pnl_reward(
        self, 
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate P&L-based reward component
        
        Uses tanh normalization to bound rewards and handle outliers
        """
        # Extract PnL from state transition
        # In production, this would come from actual trading results
        pnl = info.get('pnl', 0.0)
        
        # Update running statistics if enabled
        if self.use_running_stats and self.stats_count > 10:
            # Normalize using running statistics
            normalized_pnl = (pnl - self.running_pnl_mean) / (self.running_pnl_std + 1e-8)
        else:
            # Use fixed normalizer
            normalized_pnl = pnl / self.pnl_normalizer
        
        # Apply tanh to bound reward in [-1, 1]
        r_pnl = np.tanh(normalized_pnl)
        
        # Update running statistics
        if self.use_running_stats:
            self._update_pnl_stats(pnl)
        
        return float(r_pnl)
    
    def _calculate_synergy_reward(
        self,
        action: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate synergy alignment reward
        
        Rewards actions that align with detected synergy patterns
        """
        synergy_info = info.get('synergy', None)
        
        if synergy_info is None:
            return 0.0
        
        # Extract synergy parameters
        synergy_type = synergy_info.get('type', 'None')
        synergy_direction = synergy_info.get('direction', 0)  # -1, 0, or 1
        synergy_confidence = synergy_info.get('confidence', 0.5)
        
        if synergy_type == 'None' or synergy_direction == 0:
            return 0.0
        
        # Convert action to direction vector
        # action = [p_bearish, p_neutral, p_bullish]
        action_direction = action[2] - action[0]  # Bullish - Bearish
        
        # Calculate alignment using cosine similarity principle
        # Perfect alignment: same sign and high magnitude
        alignment = synergy_direction * action_direction
        
        # Weight by synergy confidence
        r_synergy = synergy_confidence * alignment
        
        # Clip to reasonable range
        r_synergy = np.clip(r_synergy, -1.0, 1.0)
        
        return float(r_synergy)
    
    def _calculate_risk_penalty(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate risk management penalty
        
        Penalizes excessive drawdown and large position sizes
        """
        # Extract risk metrics
        current_drawdown = info.get('drawdown', 0.0)
        position_size = info.get('position_size', 0.0)
        volatility = info.get('volatility_30', 1.0)
        
        # Drawdown penalty (quadratic for severe drawdowns)
        drawdown_penalty = 0.0
        if current_drawdown > self.max_drawdown_threshold:
            excess_drawdown = current_drawdown - self.max_drawdown_threshold
            drawdown_penalty = (excess_drawdown / self.max_drawdown_threshold) ** 2
        
        # Position size penalty (linear beyond limit)
        position_penalty = 0.0
        if abs(position_size) > self.position_size_limit:
            excess_position = abs(position_size) - self.position_size_limit
            position_penalty = excess_position / self.position_size_limit
        
        # Volatility-adjusted penalty
        # Higher volatility increases the penalty
        volatility_multiplier = min(volatility / 1.0, 2.0)  # Cap at 2x
        
        # Combine penalties
        r_risk = -(drawdown_penalty + position_penalty) * volatility_multiplier
        
        # Additional penalty for extreme actions in high volatility
        if volatility > 1.5:
            action_extremity = max(action[0], action[2])  # Max of bearish/bullish
            if action_extremity > 0.8:
                r_risk -= 0.2 * (action_extremity - 0.8)
        
        # Clip to reasonable range
        r_risk = np.clip(r_risk, -2.0, 0.0)
        
        return float(r_risk)
    
    def _calculate_exploration_bonus(self, action: np.ndarray) -> float:
        """
        Calculate exploration bonus based on action entropy
        
        Encourages diverse actions during training
        """
        # Calculate entropy of action distribution
        # H = -Σ p_i * log(p_i)
        entropy = -np.sum(action * np.log(action + 1e-8))
        
        # Normalize by maximum possible entropy
        max_entropy = -np.log(1.0 / len(action))
        normalized_entropy = entropy / max_entropy
        
        # Scale exploration bonus
        r_exploration = normalized_entropy
        
        return float(r_exploration)
    
    def _update_statistics(self, rewards: RewardComponents, info: Dict[str, Any]):
        """Update internal statistics and performance metrics"""
        # Track reward component distributions
        if len(self.reward_history) > 1000:
            # Keep only recent history
            self.reward_history = self.reward_history[-1000:]
    
    def _update_pnl_stats(self, pnl: float):
        """Update running PnL statistics using exponential moving average"""
        self.stats_count += 1
        
        if self.stats_count == 1:
            self.running_pnl_mean = pnl
            self.running_pnl_std = abs(pnl) * 0.1  # Initial estimate
        else:
            # Update mean
            self.running_pnl_mean = (
                self.stats_update_alpha * self.running_pnl_mean +
                (1 - self.stats_update_alpha) * pnl
            )
            
            # Update standard deviation
            variance = (pnl - self.running_pnl_mean) ** 2
            self.running_pnl_std = np.sqrt(
                self.stats_update_alpha * self.running_pnl_std ** 2 +
                (1 - self.stats_update_alpha) * variance
            )
            
            # Ensure minimum std
            self.running_pnl_std = max(self.running_pnl_std, 0.1)
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward distribution"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        
        # Extract component arrays
        pnl_rewards = [r['rewards']['pnl'] for r in recent_rewards]
        synergy_rewards = [r['rewards']['synergy'] for r in recent_rewards]
        risk_penalties = [r['rewards']['risk'] for r in recent_rewards]
        exploration_bonuses = [r['rewards']['exploration'] for r in recent_rewards]
        total_rewards = [r['rewards']['total'] for r in recent_rewards]
        
        stats = {
            'pnl': {
                'mean': np.mean(pnl_rewards),
                'std': np.std(pnl_rewards),
                'min': np.min(pnl_rewards),
                'max': np.max(pnl_rewards)
            },
            'synergy': {
                'mean': np.mean(synergy_rewards),
                'positive_rate': sum(r > 0 for r in synergy_rewards) / len(synergy_rewards)
            },
            'risk': {
                'mean': np.mean(risk_penalties),
                'penalty_rate': sum(r < 0 for r in risk_penalties) / len(risk_penalties)
            },
            'exploration': {
                'mean': np.mean(exploration_bonuses),
                'std': np.std(exploration_bonuses)
            },
            'total': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'positive_rate': sum(r > 0 for r in total_rewards) / len(total_rewards)
            },
            'running_stats': {
                'pnl_mean': self.running_pnl_mean,
                'pnl_std': self.running_pnl_std,
                'stats_count': self.stats_count
            }
        }
        
        return stats
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'rewards': {
                'alpha': 1.0,      # PnL weight
                'beta': 0.2,       # Synergy weight  
                'gamma': -0.3,     # Risk weight
                'delta': 0.1,      # Exploration weight
                'max_drawdown': 0.15,
                'position_limit': 1.0,
                'pnl_normalizer': 100.0,
                'use_running_stats': True
            }
        }


# Convenience function for direct usage
def calculate_reward(
    state: Dict[str, Any],
    action: np.ndarray,
    next_state: Dict[str, Any],
    info: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calculate reward components without instantiating RewardSystem
    
    Args:
        state: Current environment state
        action: Agent action (probability distribution)
        next_state: Next environment state  
        info: Additional information
        config: Optional configuration override
        
    Returns:
        Dictionary of reward components
    """
    reward_system = RewardSystem(config=config)
    rewards = reward_system.calculate_reward(state, action, next_state, info)
    return rewards.to_dict()