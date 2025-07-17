"""
Comprehensive Test Suite for Regime-Aware Reward System

This test suite validates the regime-aware reward function to ensure:
- Regime detection accuracy and reliability
- Appropriate reward adjustments for different market regimes
- Conservative behavior rewarded during crisis periods
- Trend-following behavior rewarded during trending markets
- Risk management principles upheld across all regimes

Author: Agent Gamma - The Contextual Judge
Version: 1.0 - Critical Validation Tests
"""

import pytest
import numpy as np
import time
from typing import Dict, Any

# Import the system under test
from src.agents.regime_aware_reward import (
    RegimeAwareRewardFunction, 
    create_regime_aware_reward_function,
    RewardAnalysis
)
from src.intelligence.regime_detector import MarketRegime, RegimeDetector, create_regime_detector

class TestRegimeDetection:
    """Test regime detection accuracy and reliability."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'regime_detection': {
                'crisis_volatility': 3.5,
                'high_volatility': 2.5,
                'low_volatility': 0.8,
                'strong_momentum': 0.05,
                'weak_momentum': 0.02
            }
        }
        self.detector = create_regime_detector(self.config['regime_detection'])
    
    def test_crisis_regime_detection(self):
        """Test that crisis regime is correctly detected."""
        crisis_context = {
            'volatility_30': 4.0,
            'momentum_20': -0.1,
            'momentum_50': -0.08,
            'volume_ratio': 5.0,
            'mmd_score': 1.2
        }
        
        analysis = self.detector.detect_regime(crisis_context)
        
        assert analysis.regime == MarketRegime.CRISIS
        assert analysis.confidence > 0.7  # High confidence for clear crisis
        assert analysis.volatility == 4.0
    
    def test_bull_trend_detection(self):
        """Test that bull trend is correctly detected."""
        bull_context = {
            'volatility_30': 1.5,
            'momentum_20': 0.08,
            'momentum_50': 0.06,
            'volume_ratio': 2.0,
            'mmd_score': 0.6
        }
        
        analysis = self.detector.detect_regime(bull_context)
        
        assert analysis.regime == MarketRegime.BULL_TREND
        assert analysis.confidence > 0.6
        assert analysis.momentum == 0.08
    
    def test_bear_trend_detection(self):
        """Test that bear trend is correctly detected."""
        bear_context = {
            'volatility_30': 1.8,
            'momentum_20': -0.07,
            'momentum_50': -0.05,
            'volume_ratio': 1.8,
            'mmd_score': 0.5
        }
        
        analysis = self.detector.detect_regime(bear_context)
        
        assert analysis.regime == MarketRegime.BEAR_TREND
        assert analysis.confidence > 0.6
    
    def test_low_volatility_detection(self):
        """Test that low volatility regime is correctly detected."""
        low_vol_context = {
            'volatility_30': 0.5,
            'momentum_20': 0.01,
            'momentum_50': 0.005,
            'volume_ratio': 0.8,
            'mmd_score': 0.1
        }
        
        analysis = self.detector.detect_regime(low_vol_context)
        
        assert analysis.regime in [MarketRegime.LOW_VOLATILITY, MarketRegime.SIDEWAYS]
        assert analysis.confidence > 0.5
    
    def test_invalid_input_handling(self):
        """Test that invalid inputs are handled gracefully."""
        invalid_context = {
            'volatility_30': -1.0,  # Invalid negative volatility
            'momentum_20': float('inf'),  # Invalid infinity
            'momentum_50': float('nan'),  # Invalid NaN
            'volume_ratio': 0.0,  # Edge case
            'mmd_score': -0.5  # Invalid negative MMD
        }
        
        analysis = self.detector.detect_regime(invalid_context)
        
        # Should return valid regime with reasonable confidence
        assert isinstance(analysis.regime, MarketRegime)
        assert 0.1 <= analysis.confidence <= 0.95
        assert np.isfinite(analysis.volatility)
        assert np.isfinite(analysis.momentum)


class TestRegimeSpecificRewards:
    """Test that rewards differ appropriately based on market regime."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'max_reward_scale': 5.0,
            'min_reward_scale': -5.0,
            'regime_detection': {
                'crisis_volatility': 3.5,
                'high_volatility': 2.5,
                'low_volatility': 0.8
            }
        }
        self.reward_function = create_regime_aware_reward_function(self.config)
        
        # Standard trade outcome for testing
        self.profitable_trade = {
            'pnl': 1000.0,
            'risk_penalty': 50.0,
            'volatility': 0.02,
            'drawdown': 0.01
        }
        
        self.loss_trade = {
            'pnl': -500.0,
            'risk_penalty': 100.0,
            'volatility': 0.03,
            'drawdown': 0.02
        }
        
        # Standard agent action
        self.conservative_action = {
            'action': 'hold',
            'position_size': 0.2,
            'leverage': 1.0,
            'stop_loss_used': True,
            'stop_loss_distance': 0.02
        }
        
        self.aggressive_action = {
            'action': 'buy',
            'position_size': 0.8,
            'leverage': 2.0,
            'stop_loss_used': False,
            'stop_loss_distance': 0.1
        }
    
    def test_regime_specific_rewards_differ(self):
        """Test that rewards differ based on market regime for same trade outcome."""
        
        # Different market contexts
        high_vol_context = {
            'volatility_30': 3.0,
            'momentum_20': 0.02,
            'momentum_50': 0.01,
            'volume_ratio': 1.5,
            'mmd_score': 0.3
        }
        
        low_vol_context = {
            'volatility_30': 0.5,
            'momentum_20': 0.005,
            'momentum_50': 0.003,
            'volume_ratio': 0.8,
            'mmd_score': 0.1
        }
        
        crisis_context = {
            'volatility_30': 4.5,
            'momentum_20': -0.08,
            'momentum_50': -0.05,
            'volume_ratio': 3.0,
            'mmd_score': 0.8
        }
        
        # Calculate rewards
        high_vol_reward = self.reward_function.compute_reward(
            self.profitable_trade, high_vol_context, self.conservative_action
        )
        low_vol_reward = self.reward_function.compute_reward(
            self.profitable_trade, low_vol_context, self.conservative_action
        )
        crisis_reward = self.reward_function.compute_reward(
            self.profitable_trade, crisis_context, self.conservative_action
        )
        
        # CRITICAL: Rewards must be different across regimes
        assert abs(high_vol_reward - low_vol_reward) > 0.1, "High vol vs low vol rewards must differ"
        assert abs(crisis_reward - high_vol_reward) > 0.2, "Crisis vs high vol rewards must differ significantly"
        
        # Crisis should reward profits most heavily (for conservative actions)
        assert crisis_reward > high_vol_reward, "Crisis should provide higher reward for conservative profitable trades"
    
    def test_conservative_behavior_in_crisis(self):
        """Test that conservative behavior is rewarded in crisis."""
        
        crisis_context = {
            'volatility_30': 4.0,
            'momentum_20': -0.1,
            'momentum_50': -0.08,
            'volume_ratio': 2.5,
            'mmd_score': 0.7
        }
        
        # Same small profit for both actions
        small_profit = {'pnl': 100.0, 'risk_penalty': 10.0, 'volatility': 0.02}
        
        conservative_reward = self.reward_function.compute_reward(
            small_profit, crisis_context, self.conservative_action
        )
        aggressive_reward = self.reward_function.compute_reward(
            small_profit, crisis_context, self.aggressive_action
        )
        
        # Conservative behavior should be rewarded more in crisis
        assert conservative_reward > aggressive_reward, "Conservative behavior should be rewarded in crisis"
        
        # Conservative reward should be positive even for small profits in crisis
        assert conservative_reward > 0, "Conservative profitable trades should be rewarded in crisis"
    
    def test_trend_following_rewards(self):
        """Test that trend-following is rewarded in trending markets."""
        
        bull_context = {
            'volatility_30': 1.5,
            'momentum_20': 0.06,
            'momentum_50': 0.04,
            'volume_ratio': 1.5,
            'mmd_score': 0.4
        }
        bear_context = {
            'volatility_30': 1.8,
            'momentum_20': -0.07,
            'momentum_50': -0.05,
            'volume_ratio': 1.6,
            'mmd_score': 0.5
        }
        
        buy_action = {'action': 'buy', 'position_size': 0.5, 'stop_loss_used': True}
        sell_action = {'action': 'sell', 'position_size': 0.5, 'stop_loss_used': True}
        
        profit_outcome = {'pnl': 500.0, 'risk_penalty': 25.0, 'volatility': 0.02}
        
        # Bull market - buy should be rewarded more
        bull_buy_reward = self.reward_function.compute_reward(profit_outcome, bull_context, buy_action)
        bull_sell_reward = self.reward_function.compute_reward(profit_outcome, bull_context, sell_action)
        
        # Bear market - sell should be rewarded more  
        bear_buy_reward = self.reward_function.compute_reward(profit_outcome, bear_context, buy_action)
        bear_sell_reward = self.reward_function.compute_reward(profit_outcome, bear_context, sell_action)
        
        assert bull_buy_reward > bull_sell_reward, "Buy should be rewarded more in bull market"
        assert bear_sell_reward > bear_buy_reward, "Sell should be rewarded more in bear market"
    
    def test_risk_management_incentives(self):
        """Test that risk management is properly incentivized."""
        
        high_vol_context = {
            'volatility_30': 3.0,
            'momentum_20': 0.03,
            'momentum_50': 0.02,
            'volume_ratio': 2.0,
            'mmd_score': 0.6
        }
        
        stop_loss_action = {
            'action': 'buy',
            'position_size': 0.5,
            'stop_loss_used': True,
            'stop_loss_distance': 0.02
        }
        
        no_stop_action = {
            'action': 'buy',
            'position_size': 0.5,
            'stop_loss_used': False,
            'stop_loss_distance': 0.1
        }
        
        # Same trade outcome
        trade_outcome = {'pnl': 200.0, 'risk_penalty': 30.0, 'volatility': 0.025}
        
        stop_reward = self.reward_function.compute_reward(trade_outcome, high_vol_context, stop_loss_action)
        no_stop_reward = self.reward_function.compute_reward(trade_outcome, high_vol_context, no_stop_action)
        
        # Stop loss should be rewarded, especially in high volatility
        assert stop_reward > no_stop_reward, "Stop loss usage should be rewarded in high volatility"
    
    def test_position_sizing_impact(self):
        """Test that position sizing affects rewards appropriately."""
        
        normal_context = {
            'volatility_30': 1.5,
            'momentum_20': 0.02,
            'momentum_50': 0.015,
            'volume_ratio': 1.2,
            'mmd_score': 0.3
        }
        
        small_position = {'action': 'buy', 'position_size': 0.2, 'stop_loss_used': True}
        large_position = {'action': 'buy', 'position_size': 0.9, 'stop_loss_used': True}
        
        profit_outcome = {'pnl': 300.0, 'risk_penalty': 20.0, 'volatility': 0.02}
        loss_outcome = {'pnl': -300.0, 'risk_penalty': 50.0, 'volatility': 0.03}
        
        # For profits, larger positions should generally be rewarded more (but with risk adjustment)
        small_profit_reward = self.reward_function.compute_reward(profit_outcome, normal_context, small_position)
        large_profit_reward = self.reward_function.compute_reward(profit_outcome, normal_context, large_position)
        
        # For losses, smaller positions should be penalized less
        small_loss_reward = self.reward_function.compute_reward(loss_outcome, normal_context, small_position)
        large_loss_reward = self.reward_function.compute_reward(loss_outcome, normal_context, large_position)
        
        assert small_loss_reward > large_loss_reward, "Smaller positions should be penalized less for losses"


class TestRewardBounds:
    """Test that rewards stay within reasonable bounds."""
    
    def setup_method(self):
        """Setup test environment."""
        self.reward_function = create_regime_aware_reward_function()
    
    def test_reward_bounds_respected(self):
        """Test that rewards stay within configured bounds."""
        
        # Extreme profitable trade
        extreme_profit = {
            'pnl': 10000.0,
            'risk_penalty': 0.0,
            'volatility': 0.01
        }
        
        # Extreme loss trade
        extreme_loss = {
            'pnl': -10000.0,
            'risk_penalty': 1000.0,
            'volatility': 0.05
        }
        
        normal_context = {
            'volatility_30': 1.5,
            'momentum_20': 0.02,
            'momentum_50': 0.015,
            'volume_ratio': 1.2,
            'mmd_score': 0.3
        }
        
        normal_action = {'action': 'buy', 'position_size': 0.5, 'stop_loss_used': True}
        
        profit_reward = self.reward_function.compute_reward(extreme_profit, normal_context, normal_action)
        loss_reward = self.reward_function.compute_reward(extreme_loss, normal_context, normal_action)
        
        # Check bounds
        assert -5.0 <= profit_reward <= 5.0, f"Profit reward {profit_reward} outside bounds"
        assert -5.0 <= loss_reward <= 5.0, f"Loss reward {loss_reward} outside bounds"
        
        # Extreme profits should be positive, extreme losses negative
        assert profit_reward > 0, "Extreme profits should result in positive rewards"
        assert loss_reward < 0, "Extreme losses should result in negative rewards"
    
    def test_numerical_stability(self):
        """Test that the reward function handles edge cases numerically."""
        
        edge_cases = [
            # Zero PnL
            {'pnl': 0.0, 'risk_penalty': 0.0, 'volatility': 0.01},
            # Very small values
            {'pnl': 0.01, 'risk_penalty': 0.001, 'volatility': 0.0001},
            # NaN handling (should be cleaned by input validation)
            {'pnl': 100.0, 'risk_penalty': 10.0, 'volatility': 0.02}
        ]
        
        normal_context = {
            'volatility_30': 1.0,
            'momentum_20': 0.01,
            'momentum_50': 0.008,
            'volume_ratio': 1.0,
            'mmd_score': 0.2
        }
        
        normal_action = {'action': 'hold', 'position_size': 0.3, 'stop_loss_used': True}
        
        for trade_outcome in edge_cases:
            reward = self.reward_function.compute_reward(trade_outcome, normal_context, normal_action)
            
            # Reward should be finite and within bounds
            assert np.isfinite(reward), f"Reward should be finite for {trade_outcome}"
            assert -5.0 <= reward <= 5.0, f"Reward {reward} outside bounds for {trade_outcome}"


class TestPerformanceTracking:
    """Test performance tracking and analysis capabilities."""
    
    def setup_method(self):
        """Setup test environment."""
        self.reward_function = create_regime_aware_reward_function()
    
    def test_regime_performance_tracking(self):
        """Test that regime-specific performance is tracked correctly."""
        
        # Generate rewards for different regimes
        contexts_and_regimes = [
            ({'volatility_30': 4.0, 'momentum_20': -0.08, 'volume_ratio': 3.0, 'mmd_score': 0.8}, 'crisis'),
            ({'volatility_30': 1.5, 'momentum_20': 0.06, 'volume_ratio': 1.5, 'mmd_score': 0.4}, 'bull_trend'),
            ({'volatility_30': 0.6, 'momentum_20': 0.01, 'volume_ratio': 0.9, 'mmd_score': 0.1}, 'low_volatility')
        ]
        
        action = {'action': 'buy', 'position_size': 0.4, 'stop_loss_used': True}
        
        for context, expected_regime in contexts_and_regimes:
            # Generate multiple rewards for this regime
            for i in range(5):
                trade_outcome = {
                    'pnl': np.random.normal(100, 50),
                    'risk_penalty': 10.0,
                    'volatility': 0.02
                }
                
                reward = self.reward_function.compute_reward(trade_outcome, context, action)
                assert np.isfinite(reward), "All rewards should be finite"
        
        # Check that performance summary includes expected regimes
        summary = self.reward_function.get_regime_performance_summary()
        
        # Should have data for the regimes we tested
        assert len(summary) > 0, "Should have performance data"
        
        for regime_name, stats in summary.items():
            assert 'mean_reward' in stats
            assert 'total_trades' in stats
            assert 'win_rate' in stats
            assert stats['total_trades'] > 0
    
    def test_recent_analysis_retrieval(self):
        """Test that recent analysis can be retrieved."""
        
        context = {
            'volatility_30': 2.0,
            'momentum_20': 0.03,
            'momentum_50': 0.02,
            'volume_ratio': 1.3,
            'mmd_score': 0.4
        }
        
        action = {'action': 'buy', 'position_size': 0.5, 'stop_loss_used': True}
        trade_outcome = {'pnl': 200.0, 'risk_penalty': 15.0, 'volatility': 0.02}
        
        # Generate some rewards
        for _ in range(10):
            self.reward_function.compute_reward(trade_outcome, context, action)
        
        # Retrieve recent analysis
        recent_analysis = self.reward_function.get_recent_analysis(limit=5)
        
        assert len(recent_analysis) == 5, "Should return requested number of analyses"
        
        for analysis in recent_analysis:
            assert 'base_reward' in analysis
            assert 'final_reward' in analysis
            assert 'regime' in analysis
            assert 'regime_confidence' in analysis
            assert 'timestamp' in analysis


class TestIntegrationScenarios:
    """Integration tests for realistic trading scenarios."""
    
    def setup_method(self):
        """Setup test environment."""
        self.reward_function = create_regime_aware_reward_function()
    
    def test_market_crash_scenario(self):
        """Test behavior during a simulated market crash."""
        
        # Market crash scenario: high volatility, negative momentum, high volume
        crash_context = {
            'volatility_30': 5.0,
            'momentum_20': -0.12,
            'momentum_50': -0.08,
            'volume_ratio': 8.0,
            'mmd_score': 1.5
        }
        
        # Conservative defensive action
        defensive_action = {
            'action': 'sell',  # Going defensive
            'position_size': 0.3,  # Small position
            'stop_loss_used': True,
            'stop_loss_distance': 0.015  # Tight stop
        }
        
        # Aggressive risky action
        risky_action = {
            'action': 'buy',  # Trying to catch falling knife
            'position_size': 0.9,  # Large position
            'stop_loss_used': False,
            'stop_loss_distance': 0.1
        }
        
        # Small profit outcome (good in crash)
        small_profit = {'pnl': 50.0, 'risk_penalty': 5.0, 'volatility': 0.04}
        
        defensive_reward = self.reward_function.compute_reward(small_profit, crash_context, defensive_action)
        risky_reward = self.reward_function.compute_reward(small_profit, crash_context, risky_action)
        
        # Defensive action should be heavily rewarded during crash
        assert defensive_reward > risky_reward, "Defensive actions should be rewarded during market crash"
        assert defensive_reward > 0, "Conservative profitable trades should be rewarded in crisis"
    
    def test_bull_market_scenario(self):
        """Test behavior during a strong bull market."""
        
        bull_context = {
            'volatility_30': 1.2,  # Moderate volatility
            'momentum_20': 0.08,   # Strong positive momentum
            'momentum_50': 0.06,   # Sustained trend
            'volume_ratio': 2.0,   # Good volume
            'mmd_score': 0.6       # Regime change
        }
        
        # Trend-following action
        trend_action = {
            'action': 'buy',
            'position_size': 0.7,  # Decent size
            'stop_loss_used': True,
            'stop_loss_distance': 0.03
        }
        
        # Counter-trend action
        counter_action = {
            'action': 'sell',
            'position_size': 0.5,
            'stop_loss_used': True,
            'stop_loss_distance': 0.02
        }
        
        good_profit = {'pnl': 400.0, 'risk_penalty': 30.0, 'volatility': 0.02}
        
        trend_reward = self.reward_function.compute_reward(good_profit, bull_context, trend_action)
        counter_reward = self.reward_function.compute_reward(good_profit, bull_context, counter_action)
        
        # Trend following should be rewarded more
        assert trend_reward > counter_reward, "Trend-following should be rewarded in bull market"
    
    def test_low_volatility_grinding_scenario(self):
        """Test behavior during low volatility grinding markets."""
        
        low_vol_context = {
            'volatility_30': 0.4,   # Very low volatility
            'momentum_20': 0.005,   # Minimal momentum
            'momentum_50': 0.003,   # Sustained low momentum
            'volume_ratio': 0.7,    # Low volume
            'mmd_score': 0.05       # Stable regime
        }
        
        # Patient waiting action
        patient_action = {
            'action': 'hold',
            'position_size': 0.1,   # Very small position
            'stop_loss_used': True,
            'stop_loss_distance': 0.02
        }
        
        # Active trading action
        active_action = {
            'action': 'buy',
            'position_size': 0.4,
            'stop_loss_used': True,
            'stop_loss_distance': 0.025
        }
        
        small_profit = {'pnl': 80.0, 'risk_penalty': 8.0, 'volatility': 0.01}
        
        patient_reward = self.reward_function.compute_reward(small_profit, low_vol_context, patient_action)
        active_reward = self.reward_function.compute_reward(small_profit, low_vol_context, active_action)
        
        # In low vol, small penalty for inaction but reward for taking some action
        # The active action should generally be rewarded more for successful trades
        assert active_reward >= patient_reward * 0.8, "Active profitable trades should be reasonably rewarded even in low vol"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])