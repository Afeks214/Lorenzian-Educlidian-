"""
Reward System Unit Tests with Scenario Validation

Comprehensive test suite for TacticalRewardSystem with:
- PnL reward calculation validation
- Synergy bonus scenarios
- Risk penalty edge cases
- Agent-specific reward shaping
- Granular component verification
- Performance attribution analysis

Author: Quantitative Engineer
Version: 1.0
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List
import copy
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.tactical_reward_system import TacticalRewardSystem, TacticalRewardComponents
from environment.tactical_env import MarketState


class TestTacticalRewardCalculation:
    """Test suite for tactical reward calculation"""
    
    @pytest.fixture
    def reward_system(self):
        """Create reward system with test configuration"""
        config = {
            'pnl_weight': 1.0,
            'synergy_weight': 0.2,
            'risk_weight': -0.5,
            'execution_weight': 0.1,
            'max_drawdown_threshold': 0.02,
            'position_size_limit': 1.0,
            'pnl_normalizer': 100.0,
            'agent_configs': {
                'fvg_agent': {
                    'proximity_bonus_weight': 0.1,
                    'mitigation_bonus_weight': 0.15,
                    'proximity_threshold': 5.0
                },
                'momentum_agent': {
                    'alignment_bonus_weight': 0.1,
                    'counter_trend_penalty_weight': -0.1,
                    'momentum_threshold': 0.5
                },
                'entry_opt_agent': {
                    'timing_bonus_weight': 0.1,
                    'execution_quality_weight': 0.05,
                    'slippage_threshold': 0.05
                }
            }
        }
        return TacticalRewardSystem(config)
    
    @pytest.fixture
    def mock_market_state(self):
        """Create mock market state"""
        features = {
            'current_price': 100.0,
            'current_volume': 1000.0,
            'price_momentum_5': 0.5,
            'volume_ratio': 1.2,
            'fvg_bullish_active': 1.0,
            'fvg_bearish_active': 0.0,
            'fvg_nearest_level': 99.5,
            'fvg_mitigation_signal': 0.0
        }
        
        return MarketState(
            matrix=np.random.randn(60, 7).astype(np.float32),
            price=100.0,
            volume=1000.0,
            timestamp=0,
            features=features
        )
    
    @pytest.fixture
    def mock_decision_result(self):
        """Create mock decision result"""
        return {
            'execute': True,
            'action': 2,  # Bullish
            'confidence': 0.8,
            'synergy_alignment': 0.7,
            'execution_command': {
                'side': 'BUY',
                'quantity': 1.0,
                'stop_loss': 98.0,
                'take_profit': 103.0
            }
        }
    
    @pytest.fixture
    def mock_agent_outputs(self):
        """Create mock agent outputs"""
        return {
            'fvg_agent': Mock(probabilities=np.array([0.1, 0.2, 0.7]), action=2, confidence=0.8),
            'momentum_agent': Mock(probabilities=np.array([0.2, 0.3, 0.5]), action=2, confidence=0.6),
            'entry_opt_agent': Mock(probabilities=np.array([0.3, 0.4, 0.3]), action=1, confidence=0.7)
        }
    
    def test_pnl_reward_positive_trade(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test PnL reward for profitable trade"""
        # Create profitable trade result
        trade_result = {
            'pnl': 50.0,  # $50 profit
            'drawdown': 0.01,
            'slippage': 0.02
        }
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs,
            trade_result=trade_result
        )
        
        # PnL reward should be positive
        assert reward_components.pnl_reward > 0
        # Should be approximately tanh(50/100) = tanh(0.5) ≈ 0.462
        expected_pnl = np.tanh(0.5)
        assert abs(reward_components.pnl_reward - expected_pnl) < 0.01
    
    def test_pnl_reward_negative_trade(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test PnL reward for losing trade"""
        # Create losing trade result
        trade_result = {
            'pnl': -30.0,  # $30 loss
            'drawdown': 0.015,
            'slippage': 0.03
        }
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs,
            trade_result=trade_result
        )
        
        # PnL reward should be negative
        assert reward_components.pnl_reward < 0
        # Should be approximately tanh(-30/100) = tanh(-0.3) ≈ -0.291
        expected_pnl = np.tanh(-0.3)
        assert abs(reward_components.pnl_reward - expected_pnl) < 0.01
    
    def test_synergy_bonus_aligned_trade(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test synergy bonus for aligned trade"""
        # Strong synergy alignment
        mock_decision_result['synergy_alignment'] = 0.8
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs
        )
        
        # Synergy bonus should be positive
        assert reward_components.synergy_bonus > 0
        # Should be 0.2 * 0.8 = 0.16
        expected_bonus = 0.2 * 0.8
        assert abs(reward_components.synergy_bonus - expected_bonus) < 0.01
    
    def test_synergy_bonus_misaligned_trade(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test synergy bonus for misaligned trade"""
        # Negative synergy alignment
        mock_decision_result['synergy_alignment'] = -0.5
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs
        )
        
        # Synergy bonus should be zero for negative alignment
        assert reward_components.synergy_bonus == 0.0
    
    def test_risk_penalty_excessive_position(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test risk penalty for excessive position size"""
        # Large position size
        mock_decision_result['execution_command']['quantity'] = 2.0  # Exceeds limit of 1.0
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs
        )
        
        # Risk penalty should be negative
        assert reward_components.risk_penalty < 0
    
    def test_risk_penalty_excessive_drawdown(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test risk penalty for excessive drawdown"""
        # High drawdown trade result
        trade_result = {
            'pnl': -10.0,
            'drawdown': 0.05,  # 5% drawdown (exceeds 2% threshold)
            'slippage': 0.02
        }
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs,
            trade_result=trade_result
        )
        
        # Risk penalty should be significantly negative
        assert reward_components.risk_penalty < -0.1
    
    def test_execution_bonus_high_confidence(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test execution bonus for high confidence trades"""
        # High confidence decision
        mock_decision_result['confidence'] = 0.9
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs
        )
        
        # Execution bonus should be positive
        assert reward_components.execution_bonus > 0
    
    def test_execution_bonus_low_slippage(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test execution bonus for low slippage"""
        # Low slippage trade result
        trade_result = {
            'pnl': 20.0,
            'drawdown': 0.01,
            'slippage': 0.01  # 1bp slippage (below 5bp threshold)
        }
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs,
            trade_result=trade_result
        )
        
        # Execution bonus should be positive
        assert reward_components.execution_bonus > 0
    
    def test_total_reward_calculation(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test total reward calculation"""
        trade_result = {
            'pnl': 50.0,
            'drawdown': 0.01,
            'slippage': 0.02
        }
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs,
            trade_result=trade_result
        )
        
        # Total reward should equal weighted sum of components
        expected_total = (
            1.0 * reward_components.pnl_reward +
            0.2 * reward_components.synergy_bonus +
            -0.5 * reward_components.risk_penalty +
            0.1 * reward_components.execution_bonus
        )
        
        assert abs(reward_components.total_reward - expected_total) < 0.01
    
    def test_reward_bounds(self, reward_system, mock_market_state, mock_decision_result, mock_agent_outputs):
        """Test reward bounds are enforced"""
        # Extreme trade result
        trade_result = {
            'pnl': 10000.0,  # Extreme profit
            'drawdown': 0.5,  # Extreme drawdown
            'slippage': 0.1   # High slippage
        }
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=mock_decision_result,
            market_state=mock_market_state,
            agent_outputs=mock_agent_outputs,
            trade_result=trade_result
        )
        
        # Total reward should be clipped to [-2.0, 2.0]
        assert reward_components.total_reward >= -2.0
        assert reward_components.total_reward <= 2.0


class TestAgentSpecificRewards:
    """Test suite for agent-specific reward shaping"""
    
    @pytest.fixture
    def reward_system(self):
        """Create reward system with test configuration"""
        config = {
            'pnl_weight': 1.0,
            'synergy_weight': 0.2,
            'risk_weight': -0.5,
            'execution_weight': 0.1,
            'agent_configs': {
                'fvg_agent': {
                    'proximity_bonus_weight': 0.1,
                    'mitigation_bonus_weight': 0.15,
                    'proximity_threshold': 5.0
                },
                'momentum_agent': {
                    'alignment_bonus_weight': 0.1,
                    'counter_trend_penalty_weight': -0.1,
                    'momentum_threshold': 0.5
                },
                'entry_opt_agent': {
                    'timing_bonus_weight': 0.1,
                    'execution_quality_weight': 0.05,
                    'slippage_threshold': 0.05
                }
            }
        }
        return TacticalRewardSystem(config)
    
    def test_fvg_agent_proximity_bonus(self, reward_system):
        """Test FVG agent proximity bonus"""
        market_state = Mock()
        market_state.features = {
            'current_price': 100.0,
            'fvg_nearest_level': 99.0  # Within 5 point threshold
        }
        
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8}
        agent_outputs = {'fvg_agent': Mock()}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # FVG agent should get proximity bonus
        assert 'fvg_agent' in reward_components.agent_specific
        assert 'proximity_bonus' in reward_components.agent_specific['fvg_agent']
        assert reward_components.agent_specific['fvg_agent']['proximity_bonus'] > 0
    
    def test_fvg_agent_mitigation_bonus(self, reward_system):
        """Test FVG agent mitigation bonus"""
        market_state = Mock()
        market_state.features = {
            'current_price': 100.0,
            'fvg_mitigation_signal': 1.0  # FVG just mitigated
        }
        
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8}
        agent_outputs = {'fvg_agent': Mock()}
        trade_result = {'pnl': 20.0}  # Profitable trade
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs,
            trade_result=trade_result
        )
        
        # FVG agent should get mitigation bonus
        assert 'fvg_agent' in reward_components.agent_specific
        assert 'mitigation_bonus' in reward_components.agent_specific['fvg_agent']
        assert reward_components.agent_specific['fvg_agent']['mitigation_bonus'] > 0
    
    def test_momentum_agent_alignment_bonus(self, reward_system):
        """Test momentum agent alignment bonus"""
        market_state = Mock()
        market_state.features = {
            'current_price': 100.0,
            'price_momentum_5': 0.8  # Strong positive momentum
        }
        
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8}  # Bullish action
        agent_outputs = {'momentum_agent': Mock()}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # Momentum agent should get alignment bonus
        assert 'momentum_agent' in reward_components.agent_specific
        assert 'alignment_bonus' in reward_components.agent_specific['momentum_agent']
        assert reward_components.agent_specific['momentum_agent']['alignment_bonus'] > 0
    
    def test_momentum_agent_counter_trend_penalty(self, reward_system):
        """Test momentum agent counter-trend penalty"""
        market_state = Mock()
        market_state.features = {
            'current_price': 100.0,
            'price_momentum_5': 0.8  # Strong positive momentum
        }
        
        decision_result = {'execute': True, 'action': 0, 'confidence': 0.8}  # Bearish action (counter-trend)
        agent_outputs = {'momentum_agent': Mock()}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # Momentum agent should get counter-trend penalty
        assert 'momentum_agent' in reward_components.agent_specific
        assert 'counter_trend_penalty' in reward_components.agent_specific['momentum_agent']
        assert reward_components.agent_specific['momentum_agent']['counter_trend_penalty'] < 0
    
    def test_entry_opt_agent_timing_bonus(self, reward_system):
        """Test entry optimization agent timing bonus"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.9}  # High confidence
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        agent_outputs = {'entry_opt_agent': Mock()}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # Entry opt agent should get timing bonus
        assert 'entry_opt_agent' in reward_components.agent_specific
        assert 'timing_bonus' in reward_components.agent_specific['entry_opt_agent']
        assert reward_components.agent_specific['entry_opt_agent']['timing_bonus'] > 0
    
    def test_entry_opt_agent_execution_quality(self, reward_system):
        """Test entry optimization agent execution quality bonus"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8}
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        agent_outputs = {'entry_opt_agent': Mock()}
        trade_result = {'slippage': 0.01}  # Low slippage
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs,
            trade_result=trade_result
        )
        
        # Entry opt agent should get execution quality bonus
        assert 'entry_opt_agent' in reward_components.agent_specific
        assert 'execution_quality' in reward_components.agent_specific['entry_opt_agent']
        assert reward_components.agent_specific['entry_opt_agent']['execution_quality'] > 0


class TestGranularComponentVerification:
    """Test suite for granular component verification"""
    
    @pytest.fixture
    def reward_system(self):
        """Create reward system with test configuration"""
        return TacticalRewardSystem()
    
    def test_reward_components_structure(self, reward_system):
        """Test reward components structure"""
        # Create minimal inputs
        decision_result = {'execute': False, 'action': 1, 'confidence': 0.5, 'synergy_alignment': 0.0}
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        market_state.timestamp = 0
        agent_outputs = {}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # Check component structure
        assert hasattr(reward_components, 'pnl_reward')
        assert hasattr(reward_components, 'synergy_bonus')
        assert hasattr(reward_components, 'risk_penalty')
        assert hasattr(reward_components, 'execution_bonus')
        assert hasattr(reward_components, 'total_reward')
        assert hasattr(reward_components, 'agent_specific')
        assert hasattr(reward_components, 'timestamp')
        assert hasattr(reward_components, 'decision_confidence')
        assert hasattr(reward_components, 'market_context')
    
    def test_reward_components_dictionary_conversion(self, reward_system):
        """Test reward components dictionary conversion"""
        decision_result = {'execute': False, 'action': 1, 'confidence': 0.5, 'synergy_alignment': 0.0}
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        market_state.timestamp = 0
        agent_outputs = {}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        # Convert to dictionary
        reward_dict = reward_components.to_dict()
        
        # Check dictionary structure
        assert 'pnl_reward' in reward_dict
        assert 'synergy_bonus' in reward_dict
        assert 'risk_penalty' in reward_dict
        assert 'execution_bonus' in reward_dict
        assert 'total_reward' in reward_dict
        assert 'agent_specific' in reward_dict
        assert 'timestamp' in reward_dict
        assert 'decision_confidence' in reward_dict
        assert 'market_context' in reward_dict
    
    def test_component_sum_verification(self, reward_system):
        """Test that component sum equals total reward"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        market_state.timestamp = 0
        agent_outputs = {}
        trade_result = {'pnl': 25.0}
        
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs,
            trade_result=trade_result
        )
        
        # Calculate expected total
        expected_total = (
            1.0 * reward_components.pnl_reward +
            0.2 * reward_components.synergy_bonus +
            -0.5 * reward_components.risk_penalty +
            0.1 * reward_components.execution_bonus
        )
        
        # Account for clipping
        expected_total = np.clip(expected_total, -2.0, 2.0)
        
        assert abs(reward_components.total_reward - expected_total) < 0.01


class TestPerformanceAttributionAnalysis:
    """Test suite for performance attribution analysis"""
    
    @pytest.fixture
    def reward_system(self):
        """Create reward system with test configuration"""
        return TacticalRewardSystem()
    
    def test_performance_metrics_tracking(self, reward_system):
        """Test performance metrics tracking"""
        # Generate some rewards
        for i in range(10):
            decision_result = {'execute': True, 'action': i % 3, 'confidence': 0.7, 'synergy_alignment': 0.3}
            market_state = Mock()
            market_state.features = {'current_price': 100.0}
            market_state.timestamp = i
            agent_outputs = {}
            trade_result = {'pnl': np.random.normal(0, 10)}
            
            reward_system.calculate_tactical_reward(
                decision_result=decision_result,
                market_state=market_state,
                agent_outputs=agent_outputs,
                trade_result=trade_result
            )
        
        # Get performance metrics
        metrics = reward_system.get_performance_metrics()
        
        # Check metrics structure
        assert 'total_rewards' in metrics
        assert 'positive_rate' in metrics
        assert 'negative_rate' in metrics
        assert metrics['total_rewards'] == 10
    
    def test_reward_history_tracking(self, reward_system):
        """Test reward history tracking"""
        # Generate some rewards
        for i in range(5):
            decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
            market_state = Mock()
            market_state.features = {'current_price': 100.0}
            market_state.timestamp = i
            agent_outputs = {}
            
            reward_system.calculate_tactical_reward(
                decision_result=decision_result,
                market_state=market_state,
                agent_outputs=agent_outputs
            )
        
        # Get reward history
        history = reward_system.get_reward_history()
        
        # Check history
        assert len(history) == 5
        assert all(isinstance(reward, TacticalRewardComponents) for reward in history)
    
    def test_component_statistics(self, reward_system):
        """Test component statistics calculation"""
        # Generate rewards with known patterns
        for i in range(20):
            decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
            market_state = Mock()
            market_state.features = {'current_price': 100.0}
            market_state.timestamp = i
            agent_outputs = {}
            trade_result = {'pnl': 50.0 if i % 2 == 0 else -25.0}  # Alternating wins/losses
            
            reward_system.calculate_tactical_reward(
                decision_result=decision_result,
                market_state=market_state,
                agent_outputs=agent_outputs,
                trade_result=trade_result
            )
        
        # Get performance metrics
        metrics = reward_system.get_performance_metrics()
        
        # Check component statistics
        assert 'pnl_reward_avg' in metrics
        assert 'synergy_bonus_avg' in metrics
        assert 'risk_penalty_avg' in metrics
        assert 'execution_bonus_avg' in metrics


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases"""
    
    @pytest.fixture
    def reward_system(self):
        """Create reward system with test configuration"""
        return TacticalRewardSystem()
    
    def test_missing_trade_result(self, reward_system):
        """Test reward calculation without trade result"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        market_state.timestamp = 0
        agent_outputs = {}
        
        # Should not crash without trade_result
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        assert isinstance(reward_components, TacticalRewardComponents)
        assert reward_components.total_reward is not None
    
    def test_missing_market_features(self, reward_system):
        """Test reward calculation with missing market features"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
        market_state = Mock()
        market_state.features = {}  # Empty features
        market_state.timestamp = 0
        agent_outputs = {}
        
        # Should not crash with missing features
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs
        )
        
        assert isinstance(reward_components, TacticalRewardComponents)
    
    def test_zero_division_protection(self, reward_system):
        """Test protection against zero division"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
        market_state = Mock()
        market_state.features = {'current_price': 0.0}  # Zero price
        market_state.timestamp = 0
        agent_outputs = {}
        trade_result = {'pnl': 0.0}
        
        # Should not crash with zero values
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs,
            trade_result=trade_result
        )
        
        assert isinstance(reward_components, TacticalRewardComponents)
        assert np.isfinite(reward_components.total_reward)
    
    def test_extreme_values_handling(self, reward_system):
        """Test handling of extreme values"""
        decision_result = {'execute': True, 'action': 2, 'confidence': 0.8, 'synergy_alignment': 0.5}
        market_state = Mock()
        market_state.features = {'current_price': 1e10}  # Extreme price
        market_state.timestamp = 0
        agent_outputs = {}
        trade_result = {'pnl': 1e6}  # Extreme profit
        
        # Should handle extreme values gracefully
        reward_components = reward_system.calculate_tactical_reward(
            decision_result=decision_result,
            market_state=market_state,
            agent_outputs=agent_outputs,
            trade_result=trade_result
        )
        
        assert isinstance(reward_components, TacticalRewardComponents)
        assert -2.0 <= reward_components.total_reward <= 2.0  # Should be clipped


if __name__ == '__main__':
    pytest.main([__file__, '-v'])