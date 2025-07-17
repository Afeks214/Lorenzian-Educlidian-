"""
Comprehensive tests for Reward System

Tests the multi-objective reward calculation including:
- PnL reward with normalization
- Synergy alignment bonus
- Risk management penalties
- Exploration incentives
- Granular component tracking
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.reward_system import RewardSystem, RewardComponents, calculate_reward


class TestRewardSystem:
    """Test suite for multi-objective reward system"""
    
    @pytest.fixture
    def reward_system(self):
        """Create reward system instance for testing"""
        config = {
            'rewards': {
                'alpha': 1.0,      # PnL weight
                'beta': 0.2,       # Synergy weight
                'gamma': -0.3,     # Risk weight
                'delta': 0.1,      # Exploration weight
                'max_drawdown': 0.15,
                'position_limit': 1.0,
                'pnl_normalizer': 100.0,
                'use_running_stats': False  # Disable for predictable tests
            }
        }
        return RewardSystem(config=config)
    
    @pytest.fixture
    def base_state(self):
        """Base state for testing"""
        return {
            'portfolio_value': 10000,
            'position_size': 0.5,
            'drawdown': 0.05
        }
    
    @pytest.fixture
    def base_action(self):
        """Base action (probability distribution)"""
        return np.array([0.2, 0.3, 0.5])  # Slightly bullish
    
    def test_reward_system_initialization(self, reward_system):
        """Test reward system initializes correctly"""
        assert reward_system.alpha == 1.0
        assert reward_system.beta == 0.2
        assert reward_system.gamma == -0.3
        assert reward_system.delta == 0.1
        assert reward_system.max_drawdown_threshold == 0.15
    
    def test_positive_pnl_reward(self, reward_system, base_state, base_action):
        """Test profitable trade generates positive reward"""
        next_state = {
            'portfolio_value': 10100,  # +100 profit
            'position_size': 0.5,
            'drawdown': 0.04
        }
        
        info = {
            'pnl': 100,  # Profit
            'synergy': None,
            'drawdown': 0.04,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, base_action, next_state, info)
        
        # PnL component should be positive
        assert rewards.pnl > 0
        # With alpha=1.0, total should be dominated by positive PnL
        assert rewards.total > 0
    
    def test_negative_pnl_reward(self, reward_system, base_state, base_action):
        """Test losing trade generates negative reward"""
        next_state = {
            'portfolio_value': 9900,  # -100 loss
            'position_size': 0.5,
            'drawdown': 0.06
        }
        
        info = {
            'pnl': -100,  # Loss
            'synergy': None,
            'drawdown': 0.06,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, base_action, next_state, info)
        
        # PnL component should be negative
        assert rewards.pnl < 0
    
    def test_synergy_bonus_applied(self, reward_system, base_state):
        """Test synergy alignment bonus when action aligns with synergy"""
        # Bullish action
        bullish_action = np.array([0.1, 0.1, 0.8])
        
        info = {
            'pnl': 0,
            'synergy': {
                'type': 'TYPE_1',
                'direction': 1,  # Bullish synergy
                'confidence': 0.9
            },
            'drawdown': 0.05,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, bullish_action, base_state, info)
        
        # Synergy component should be positive (aligned)
        assert rewards.synergy > 0
        # Should be weighted by confidence
        assert rewards.synergy <= 0.9
    
    def test_synergy_penalty_misaligned(self, reward_system, base_state):
        """Test synergy penalty when action opposes synergy"""
        # Bearish action
        bearish_action = np.array([0.8, 0.1, 0.1])
        
        info = {
            'pnl': 0,
            'synergy': {
                'type': 'TYPE_1',
                'direction': 1,  # Bullish synergy
                'confidence': 0.9
            },
            'drawdown': 0.05,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, bearish_action, base_state, info)
        
        # Synergy component should be negative (misaligned)
        assert rewards.synergy < 0
    
    def test_risk_penalty_high_drawdown(self, reward_system, base_state, base_action):
        """Test risk penalty applied for excessive drawdown"""
        info = {
            'pnl': 0,
            'synergy': None,
            'drawdown': 0.25,  # 25% drawdown exceeds 15% threshold
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, base_action, base_state, info)
        
        # Risk component should be negative
        assert rewards.risk < 0
        # Total reward should be penalized
        assert rewards.total < 0
    
    def test_risk_penalty_large_position(self, reward_system, base_state, base_action):
        """Test risk penalty for oversized positions"""
        info = {
            'pnl': 0,
            'synergy': None,
            'drawdown': 0.05,
            'position_size': 1.5,  # Exceeds 1.0 limit
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, base_action, base_state, info)
        
        # Risk component should be negative
        assert rewards.risk < 0
    
    def test_risk_penalty_high_volatility(self, reward_system, base_state):
        """Test increased risk penalty in high volatility"""
        # Extreme action in high volatility
        extreme_action = np.array([0.9, 0.05, 0.05])
        
        info = {
            'pnl': 0,
            'synergy': None,
            'drawdown': 0.10,
            'position_size': 0.8,
            'volatility_30': 2.0  # High volatility
        }
        
        rewards = reward_system.calculate_reward(base_state, extreme_action, base_state, info)
        
        # Risk penalty should be amplified
        assert rewards.risk < -0.1
    
    def test_exploration_bonus_high_entropy(self, reward_system, base_state):
        """Test exploration bonus for high entropy actions"""
        # High entropy action (nearly uniform)
        high_entropy_action = np.array([0.33, 0.34, 0.33])
        
        info = {
            'pnl': 0,
            'synergy': None,
            'drawdown': 0.05,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, high_entropy_action, base_state, info)
        
        # Exploration component should be high
        assert rewards.exploration > 0.9
    
    def test_exploration_bonus_low_entropy(self, reward_system, base_state):
        """Test minimal exploration bonus for deterministic actions"""
        # Low entropy action (nearly deterministic)
        low_entropy_action = np.array([0.98, 0.01, 0.01])
        
        info = {
            'pnl': 0,
            'synergy': None,
            'drawdown': 0.05,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(base_state, low_entropy_action, base_state, info)
        
        # Exploration component should be low
        assert rewards.exploration < 0.2
    
    def test_reward_components_structure(self, reward_system, base_state, base_action):
        """Test that all reward components are returned"""
        info = {
            'pnl': 50,
            'synergy': {'type': 'TYPE_1', 'direction': 1, 'confidence': 0.8},
            'drawdown': 0.10,
            'position_size': 0.7,
            'volatility_30': 1.2
        }
        
        rewards = reward_system.calculate_reward(base_state, base_action, base_state, info)
        
        # Check all components exist
        assert hasattr(rewards, 'pnl')
        assert hasattr(rewards, 'synergy')
        assert hasattr(rewards, 'risk')
        assert hasattr(rewards, 'exploration')
        assert hasattr(rewards, 'total')
        
        # Check total is weighted sum
        expected_total = (
            reward_system.alpha * rewards.pnl +
            reward_system.beta * rewards.synergy +
            reward_system.gamma * rewards.risk +
            reward_system.delta * rewards.exploration
        )
        assert np.isclose(rewards.total, expected_total)
    
    def test_running_statistics_update(self):
        """Test running PnL statistics update"""
        reward_system = RewardSystem(config={
            'rewards': {
                'alpha': 1.0, 'beta': 0.2, 'gamma': -0.3, 'delta': 0.1,
                'use_running_stats': True
            }
        })
        
        base_state = {'portfolio_value': 10000}
        base_action = np.array([0.33, 0.34, 0.33])
        
        # Generate multiple rewards with different PnLs
        pnls = [100, -50, 200, -100, 150]
        
        for pnl in pnls:
            info = {
                'pnl': pnl,
                'synergy': None,
                'drawdown': 0.05,
                'position_size': 0.5,
                'volatility_30': 1.0
            }
            reward_system.calculate_reward(base_state, base_action, base_state, info)
        
        # Check statistics were updated
        assert reward_system.stats_count == 5
        assert reward_system.running_pnl_mean != 0
        assert reward_system.running_pnl_std > 0
    
    def test_reward_statistics_tracking(self, reward_system, base_state, base_action):
        """Test reward statistics calculation"""
        # Generate some rewards
        for i in range(10):
            info = {
                'pnl': np.random.normal(0, 100),
                'synergy': {'type': 'TYPE_1', 'direction': 1, 'confidence': 0.8} if i % 2 == 0 else None,
                'drawdown': 0.05 + i * 0.01,
                'position_size': 0.5,
                'volatility_30': 1.0
            }
            reward_system.calculate_reward(base_state, base_action, base_state, info)
        
        # Get statistics
        stats = reward_system.get_reward_statistics()
        
        # Check statistics structure
        assert 'pnl' in stats
        assert 'synergy' in stats
        assert 'risk' in stats
        assert 'exploration' in stats
        assert 'total' in stats
        
        # Check PnL stats
        assert 'mean' in stats['pnl']
        assert 'std' in stats['pnl']
        assert 'min' in stats['pnl']
        assert 'max' in stats['pnl']
        
        # Check synergy stats
        assert 'positive_rate' in stats['synergy']
        assert 0 <= stats['synergy']['positive_rate'] <= 1
    
    def test_calculate_reward_function(self, base_state, base_action):
        """Test standalone calculate_reward function"""
        info = {
            'pnl': 100,
            'synergy': None,
            'drawdown': 0.05,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards_dict = calculate_reward(base_state, base_action, base_state, info)
        
        # Check returns dictionary
        assert isinstance(rewards_dict, dict)
        assert 'pnl' in rewards_dict
        assert 'synergy' in rewards_dict
        assert 'risk' in rewards_dict
        assert 'exploration' in rewards_dict
        assert 'total' in rewards_dict
    
    def test_extreme_values_handling(self, reward_system, base_state):
        """Test handling of extreme input values"""
        extreme_action = np.array([1.0, 0.0, 0.0])  # Fully bearish
        
        info = {
            'pnl': 10000,  # Extreme profit
            'synergy': {'type': 'TYPE_1', 'direction': 1, 'confidence': 1.0},
            'drawdown': 0.5,  # 50% drawdown
            'position_size': 3.0,  # 3x leverage
            'volatility_30': 5.0  # Extreme volatility
        }
        
        rewards = reward_system.calculate_reward(base_state, extreme_action, base_state, info)
        
        # Check rewards are bounded reasonably
        assert -10 < rewards.total < 10  # Should be bounded by config or logic
        assert rewards.risk < 0  # Should have significant risk penalty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])