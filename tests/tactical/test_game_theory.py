"""
Game Theory Test Suite for Tactical MARL System

Tests for reward gaming vulnerabilities and consensus override exploits.
Validates that the system is mathematically resistant to gaming strategies
and enforces strategic alignment through hard constraints.

Author: Agent 3 - Game Theorist & Reward Architect
Mission: Aegis - Eliminate Gaming Vulnerabilities
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import logging

# Import the systems under test
import sys
import os
sys.path.insert(0, os.path.abspath('/home/QuantNova/GrandModel'))

# Direct imports to avoid module loading issues
from training.tactical_reward_system import TacticalRewardSystem
from src.tactical.aggregator import TacticalDecisionAggregator

logger = logging.getLogger(__name__)


class TestRewardGameResistance:
    """Test suite for reward function gaming resistance"""
    
    def setup_method(self):
        """Setup test environment"""
        self.reward_system = TacticalRewardSystem()
        self.aggregator = TacticalDecisionAggregator()
        
        # Mock market state
        self.mock_market_state = Mock()
        self.mock_market_state.features = {
            'current_price': 100.0,
            'current_volume': 1000.0,
            'price_momentum_5': 0.5,
            'volume_ratio': 1.5,
            'fvg_nearest_level': 101.0,
            'fvg_bullish_active': 1.0,
            'fvg_bearish_active': 0.0,
            'fvg_mitigation_signal': 1.0
        }
        self.mock_market_state.timestamp = 1234567890
        
        # Mock synergy event
        self.mock_synergy_event = Mock()
        self.mock_synergy_event.direction = 1  # Bullish
        self.mock_synergy_event.synergy_type = 'TYPE_1'
    
    def test_linear_combination_gaming_prevented(self):
        """
        Test that linear combination gaming is prevented by product formulation
        
        Old system: agents could game by minimizing risk_penalty term
        New system: requires optimization of ALL components simultaneously
        """
        # Gaming strategy: High risk, zero synergy (old system exploit)
        gaming_decision = {
            'execute': True,
            'confidence': 0.8,
            'synergy_alignment': 0.0,  # No strategic alignment
            'action': 1
        }
        
        gaming_trade_result = {
            'pnl': 100.0,  # High PnL
            'drawdown': 0.1,  # High risk (should kill reward in new system)
            'slippage': 0.01
        }
        
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        # Calculate reward with gaming strategy
        reward = self.reward_system.calculate_tactical_reward(
            gaming_decision, self.mock_market_state, agent_outputs, gaming_trade_result
        )
        
        # In the new system, poor strategic alignment should severely penalize reward
        # Even with high PnL, the strategic gate should reduce the reward significantly
        assert reward.total_reward < 0.2, f"Gaming strategy should yield low reward, got {reward.total_reward}"
        assert reward.synergy_bonus <= 0.05, "No synergy alignment should yield minimal bonus"
        
        logger.info(f"Gaming prevention test passed. Gaming reward: {reward.total_reward}")
    
    def test_strategic_alignment_dominance(self):
        """
        Test that strategic alignment is the dominant factor and cannot be bypassed
        """
        # Test 1: High PnL but poor strategic alignment
        poor_alignment_decision = {
            'execute': True,
            'confidence': 0.9,
            'synergy_alignment': 0.01,  # Very poor alignment
            'action': 1
        }
        
        excellent_trade_result = {
            'pnl': 200.0,  # Excellent PnL
            'drawdown': 0.01,  # Low risk
            'slippage': 0.005  # Excellent execution
        }
        
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        poor_reward = self.reward_system.calculate_tactical_reward(
            poor_alignment_decision, self.mock_market_state, agent_outputs, excellent_trade_result
        )
        
        # Test 2: Moderate PnL but excellent strategic alignment
        good_alignment_decision = {
            'execute': True,
            'confidence': 0.8,
            'synergy_alignment': 0.8,  # Excellent alignment
            'action': 1
        }
        
        moderate_trade_result = {
            'pnl': 50.0,  # Moderate PnL
            'drawdown': 0.02,  # Moderate risk
            'slippage': 0.01  # Moderate execution
        }
        
        good_reward = self.reward_system.calculate_tactical_reward(
            good_alignment_decision, self.mock_market_state, agent_outputs, moderate_trade_result
        )
        
        # Strategic alignment should dominate - good alignment with moderate performance
        # should beat poor alignment with excellent performance
        assert good_reward.total_reward > poor_reward.total_reward, \
            f"Strategic alignment should dominate: good={good_reward.total_reward}, poor={poor_reward.total_reward}"
        
        logger.info(f"Strategic dominance test passed. Good alignment: {good_reward.total_reward}, "
                   f"Poor alignment: {poor_reward.total_reward}")
    
    def test_product_formulation_prevents_zero_gaming(self):
        """
        Test that product formulation prevents agents from zeroing out negative components
        """
        # Strategy: Try to zero out risk by taking no position (but this conflicts with PnL goal)
        zero_risk_decision = {
            'execute': False,  # No execution = no risk but also no PnL
            'confidence': 0.9,
            'synergy_alignment': 0.5,
            'action': 0  # Hold
        }
        
        no_trade_result = None  # No trade executed
        
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        zero_risk_reward = self.reward_system.calculate_tactical_reward(
            zero_risk_decision, self.mock_market_state, agent_outputs, no_trade_result
        )
        
        # Balanced execution with reasonable risk
        balanced_decision = {
            'execute': True,
            'confidence': 0.8,
            'synergy_alignment': 0.5,
            'action': 1
        }
        
        balanced_trade_result = {
            'pnl': 25.0,  # Modest PnL
            'drawdown': 0.015,  # Reasonable risk
            'slippage': 0.008
        }
        
        balanced_reward = self.reward_system.calculate_tactical_reward(
            balanced_decision, self.mock_market_state, agent_outputs, balanced_trade_result
        )
        
        # Balanced approach should outperform zero-risk gaming
        assert balanced_reward.total_reward > zero_risk_reward.total_reward, \
            f"Balanced approach should beat zero-risk gaming: balanced={balanced_reward.total_reward}, " \
            f"zero_risk={zero_risk_reward.total_reward}"
        
        logger.info(f"Zero-gaming prevention test passed. Balanced: {balanced_reward.total_reward}, "
                   f"Zero-risk: {zero_risk_reward.total_reward}")
    
    def test_nash_equilibrium_validation(self):
        """
        Test that the optimal Nash equilibrium favors genuine strategy optimization
        """
        # Define multiple competing strategies
        strategies = [
            # Strategy 1: Gaming by risk avoidance
            {
                'name': 'risk_avoider',
                'decision': {'execute': False, 'confidence': 0.9, 'synergy_alignment': 0.3, 'action': 0},
                'trade_result': None
            },
            # Strategy 2: Gaming by synergy neglect
            {
                'name': 'synergy_neglect',
                'decision': {'execute': True, 'confidence': 0.9, 'synergy_alignment': 0.0, 'action': 1},
                'trade_result': {'pnl': 100.0, 'drawdown': 0.05, 'slippage': 0.01}
            },
            # Strategy 3: Balanced optimization (true optimal)
            {
                'name': 'balanced_optimal',
                'decision': {'execute': True, 'confidence': 0.85, 'synergy_alignment': 0.7, 'action': 1},
                'trade_result': {'pnl': 60.0, 'drawdown': 0.02, 'slippage': 0.008}
            },
            # Strategy 4: Conservative but aligned
            {
                'name': 'conservative_aligned',
                'decision': {'execute': True, 'confidence': 0.7, 'synergy_alignment': 0.6, 'action': 1},
                'trade_result': {'pnl': 30.0, 'drawdown': 0.01, 'slippage': 0.005}
            }
        ]
        
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        # Calculate rewards for all strategies
        strategy_rewards = {}
        for strategy in strategies:
            reward = self.reward_system.calculate_tactical_reward(
                strategy['decision'], self.mock_market_state, agent_outputs, strategy['trade_result']
            )
            strategy_rewards[strategy['name']] = reward.total_reward
        
        # The balanced optimal strategy should achieve the highest reward
        optimal_strategy = max(strategy_rewards, key=strategy_rewards.get)
        assert optimal_strategy == 'balanced_optimal', \
            f"Nash equilibrium should favor balanced optimization, but {optimal_strategy} won: {strategy_rewards}"
        
        # Gaming strategies should be suboptimal
        assert strategy_rewards['risk_avoider'] < strategy_rewards['balanced_optimal']
        assert strategy_rewards['synergy_neglect'] < strategy_rewards['balanced_optimal']
        
        logger.info(f"Nash equilibrium validation passed. Strategy rewards: {strategy_rewards}")


class TestConsensusOverrideResistance:
    """Test suite for consensus override gaming resistance"""
    
    def setup_method(self):
        """Setup test environment"""
        self.aggregator = TacticalDecisionAggregator()
        
        # Mock synergy event - bullish direction
        self.bullish_synergy = Mock()
        self.bullish_synergy.direction = 1
        self.bullish_synergy.synergy_type = 'TYPE_1'
        
        # Mock synergy event - bearish direction  
        self.bearish_synergy = Mock()
        self.bearish_synergy.direction = -1
        self.bearish_synergy.synergy_type = 'TYPE_1'
    
    @pytest.mark.asyncio
    async def test_hard_gate_blocks_counter_synergy(self):
        """
        Test that the hard gate blocks counter-synergy trades below 95% confidence
        """
        # Counter-synergy decision: agents want to short but synergy is bullish
        counter_synergy_decisions = [
            {'action': -1, 'confidence': 0.8},  # Short
            {'action': -1, 'confidence': 0.85}, # Short  
            {'action': 0, 'confidence': 0.7}    # Hold
        ]
        
        result = await self.aggregator.aggregate_decisions(
            counter_synergy_decisions, self.bullish_synergy
        )
        
        # Should be blocked by hard gate since confidence < 0.95
        assert not result['should_execute'], "Counter-synergy trade should be blocked"
        assert result['confidence'] == 0.0, "Confidence should be zeroed by hard gate"
        assert result['strategic_gate_enforced'], "Strategic gate enforcement should be flagged"
        
        logger.info(f"Hard gate blocking test passed: {result}")
    
    @pytest.mark.asyncio  
    async def test_ultra_high_confidence_override(self):
        """
        Test that ultra-high confidence (>95%) can override strategic direction
        """
        # Ultra-high confidence counter-synergy decision
        ultra_high_confidence_decisions = [
            {'action': -1, 'confidence': 0.98},  # Short with very high confidence
            {'action': -1, 'confidence': 0.96},  # Short with high confidence
            {'action': -1, 'confidence': 0.95}   # Short at threshold
        ]
        
        result = await self.aggregator.aggregate_decisions(
            ultra_high_confidence_decisions, self.bullish_synergy
        )
        
        # Should be allowed but heavily logged
        weighted_score = 0.98 * 0.5 + 0.96 * 0.3 + 0.95 * 0.2  # TYPE_1 weights
        expected_above_95 = weighted_score >= 0.95
        
        if expected_above_95:
            assert result['should_execute'], "Ultra-high confidence should override"
            assert result['confidence'] > 0.95, "Confidence should remain high"
        else:
            assert not result['should_execute'], "Should still be blocked if weighted confidence < 95%"
        
        logger.info(f"Ultra-high confidence test: weighted={weighted_score:.3f}, result={result}")
    
    @pytest.mark.asyncio
    async def test_aligned_trades_get_bonus(self):
        """
        Test that strategically aligned trades receive confidence bonus
        """
        # Aligned decision: agents want to go long and synergy is bullish
        aligned_decisions = [
            {'action': 1, 'confidence': 0.7},   # Long
            {'action': 1, 'confidence': 0.75},  # Long
            {'action': 0, 'confidence': 0.6}    # Hold
        ]
        
        result = await self.aggregator.aggregate_decisions(
            aligned_decisions, self.bullish_synergy
        )
        
        # Calculate expected weighted score
        base_weighted_score = 0.7 * 0.5 + 0.75 * 0.3 + 0.6 * 0.2  # For action 1
        
        # Should receive strategic bonus and execute
        assert result['should_execute'], "Aligned trade should execute"
        assert result['confidence'] >= base_weighted_score, "Should receive strategic bonus"
        assert result['synergy_alignment'], "Should be marked as aligned"
        
        logger.info(f"Aligned trade bonus test passed. Base: {base_weighted_score:.3f}, "
                   f"Final: {result['confidence']:.3f}")
    
    @pytest.mark.asyncio
    async def test_gaming_resistance_across_confidence_levels(self):
        """
        Test that gaming attempts across different confidence levels are blocked
        """
        gaming_attempts = [
            # Attempt 1: High confidence but not ultra-high
            ([{'action': -1, 'confidence': 0.9}, {'action': -1, 'confidence': 0.85}, 
              {'action': -1, 'confidence': 0.8}], False),
            
            # Attempt 2: Mixed signals trying to game threshold
            ([{'action': -1, 'confidence': 0.94}, {'action': 0, 'confidence': 0.6}, 
              {'action': 1, 'confidence': 0.5}], False),
            
            # Attempt 3: Just below ultra-high threshold
            ([{'action': -1, 'confidence': 0.949}, {'action': -1, 'confidence': 0.948}, 
              {'action': -1, 'confidence': 0.947}], False),
        ]
        
        for decisions, should_pass in gaming_attempts:
            result = await self.aggregator.aggregate_decisions(decisions, self.bullish_synergy)
            
            if should_pass:
                assert result['should_execute'], f"This gaming attempt should have passed: {decisions}"
            else:
                assert not result['should_execute'], f"Gaming attempt should be blocked: {decisions}"
        
        logger.info("Gaming resistance test passed across multiple confidence levels")
    
    @pytest.mark.asyncio
    async def test_strategic_override_logging(self):
        """
        Test that strategic overrides are properly logged for audit trail
        """
        with patch('src.tactical.aggregator.logger') as mock_logger:
            # Ultra-high confidence counter-synergy
            ultra_high_decisions = [
                {'action': -1, 'confidence': 0.99},
                {'action': -1, 'confidence': 0.98}, 
                {'action': -1, 'confidence': 0.97}
            ]
            
            result = await self.aggregator.aggregate_decisions(
                ultra_high_decisions, self.bullish_synergy
            )
            
            # Should have logged the strategic override
            if result['should_execute']:
                mock_logger.critical.assert_called()
                logged_message = mock_logger.critical.call_args[0][0]
                assert "STRATEGIC OVERRIDE" in logged_message
                assert "counter-synergy" in logged_message.lower()
        
        logger.info("Strategic override logging test passed")


class TestMathematicalGameTheoryProperties:
    """Test mathematical properties and game theory foundations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.reward_system = TacticalRewardSystem()
        
        self.mock_market_state = Mock()
        self.mock_market_state.features = {
            'current_price': 100.0,
            'price_momentum_5': 0.3,
            'volume_ratio': 1.2
        }
        self.mock_market_state.timestamp = 1234567890
    
    def test_reward_monotonicity_properties(self):
        """
        Test that reward function has proper monotonicity properties
        """
        # Test PnL monotonicity: higher PnL should yield higher rewards (all else equal)
        base_decision = {'execute': True, 'confidence': 0.8, 'synergy_alignment': 0.6, 'action': 1}
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        pnl_values = [10.0, 50.0, 100.0, 200.0]
        rewards = []
        
        for pnl in pnl_values:
            trade_result = {'pnl': pnl, 'drawdown': 0.02, 'slippage': 0.01}
            reward = self.reward_system.calculate_tactical_reward(
                base_decision, self.mock_market_state, agent_outputs, trade_result
            )
            rewards.append(reward.total_reward)
        
        # Rewards should be monotonically increasing with PnL
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i-1], \
                f"Reward should increase with PnL: {rewards[i]} >= {rewards[i-1]}"
        
        logger.info(f"PnL monotonicity test passed: {list(zip(pnl_values, rewards))}")
    
    def test_strategic_alignment_criticality(self):
        """
        Test that strategic alignment is mathematically critical (cannot be bypassed)
        """
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        excellent_trade = {'pnl': 500.0, 'drawdown': 0.005, 'slippage': 0.001}
        
        # Test across synergy alignment spectrum
        alignments = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
        rewards = []
        
        for alignment in alignments:
            decision = {'execute': True, 'confidence': 0.9, 'synergy_alignment': alignment, 'action': 1}
            reward = self.reward_system.calculate_tactical_reward(
                decision, self.mock_market_state, agent_outputs, excellent_trade
            )
            rewards.append(reward.total_reward)
        
        # There should be a clear threshold effect around 0.05 due to strategic gate
        low_alignment_rewards = [r for r, a in zip(rewards, alignments) if a <= 0.05]
        high_alignment_rewards = [r for r, a in zip(rewards, alignments) if a > 0.05]
        
        if high_alignment_rewards and low_alignment_rewards:
            avg_low = np.mean(low_alignment_rewards)
            avg_high = np.mean(high_alignment_rewards)
            
            # Strategic gate should create significant reward difference
            assert avg_high > avg_low * 1.5, \
                f"Strategic gate should create significant reward difference: {avg_high} > {avg_low * 1.5}"
        
        logger.info(f"Strategic alignment criticality test passed: {list(zip(alignments, rewards))}")
    
    def test_product_vs_linear_gaming_resistance(self):
        """
        Mathematical test proving product formulation is more game-resistant than linear
        """
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        # Test scenario: Agent tries to maximize reward through component manipulation
        test_cases = [
            # Case 1: High PnL, zero synergy, moderate risk
            {'pnl': 100.0, 'synergy': 0.0, 'risk': -0.3, 'execution': 0.1},
            # Case 2: Moderate PnL, high synergy, zero risk  
            {'pnl': 30.0, 'synergy': 0.8, 'risk': 0.0, 'execution': 0.1},
            # Case 3: Balanced optimization
            {'pnl': 60.0, 'synergy': 0.6, 'risk': -0.1, 'execution': 0.08},
        ]
        
        new_system_rewards = []
        simulated_linear_rewards = []
        
        for case in test_cases:
            # New system (product-based)
            decision = {'execute': True, 'confidence': 0.8, 'synergy_alignment': case['synergy'], 'action': 1}
            trade_result = {'pnl': case['pnl'], 'drawdown': abs(case['risk']) * 0.1, 'slippage': 0.01}
            
            reward = self.reward_system.calculate_tactical_reward(
                decision, self.mock_market_state, agent_outputs, trade_result
            )
            new_system_rewards.append(reward.total_reward)
            
            # Simulate old linear system for comparison
            linear_reward = (
                1.0 * np.tanh(case['pnl'] / 100.0) +  # PnL component
                0.2 * case['synergy'] +               # Synergy component  
                -0.5 * abs(case['risk']) +            # Risk component
                0.1 * case['execution']               # Execution component
            )
            simulated_linear_rewards.append(linear_reward)
        
        # In the linear system, case 0 (high PnL, zero synergy) might win
        # In the new system, case 2 (balanced) should win due to strategic gate
        linear_best = np.argmax(simulated_linear_rewards)
        product_best = np.argmax(new_system_rewards)
        
        logger.info(f"Linear system best case: {linear_best}, rewards: {simulated_linear_rewards}")
        logger.info(f"Product system best case: {product_best}, rewards: {new_system_rewards}")
        
        # The product system should favor the balanced case over pure PnL gaming
        assert product_best != 0 or new_system_rewards[2] > new_system_rewards[0] * 0.8, \
            "Product system should not favor pure PnL gaming over strategic alignment"
    
    def test_mathematical_bounds_and_stability(self):
        """
        Test mathematical bounds and numerical stability of the reward function
        """
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        # Test extreme values
        extreme_cases = [
            # Extreme positive
            {'decision': {'execute': True, 'confidence': 1.0, 'synergy_alignment': 1.0, 'action': 1},
             'trade': {'pnl': 1000.0, 'drawdown': 0.0, 'slippage': 0.0}},
            # Extreme negative  
            {'decision': {'execute': True, 'confidence': 0.0, 'synergy_alignment': 0.0, 'action': 1},
             'trade': {'pnl': -1000.0, 'drawdown': 0.5, 'slippage': 0.1}},
            # Mixed extremes
            {'decision': {'execute': True, 'confidence': 1.0, 'synergy_alignment': 0.0, 'action': 1},
             'trade': {'pnl': 1000.0, 'drawdown': 0.5, 'slippage': 0.1}},
        ]
        
        for case in extreme_cases:
            reward = self.reward_system.calculate_tactical_reward(
                case['decision'], self.mock_market_state, agent_outputs, case['trade']
            )
            
            # Ensure rewards are bounded
            assert -2.0 <= reward.total_reward <= 2.0, \
                f"Reward should be bounded in [-2, 2]: {reward.total_reward}"
            
            # Ensure no NaN or inf values
            assert np.isfinite(reward.total_reward), f"Reward should be finite: {reward.total_reward}"
            assert not np.isnan(reward.total_reward), f"Reward should not be NaN: {reward.total_reward}"
        
        logger.info("Mathematical bounds and stability test passed")


# Integration test for full system
class TestIntegratedGameTheoryResistance:
    """Integration tests for complete game-theory resistance"""
    
    def setup_method(self):
        """Setup integrated test environment"""
        self.reward_system = TacticalRewardSystem()
        self.aggregator = TacticalDecisionAggregator()
        
        self.mock_market_state = Mock()
        self.mock_market_state.features = {
            'current_price': 100.0,
            'price_momentum_5': 0.4,
            'volume_ratio': 1.3
        }
        self.mock_market_state.timestamp = 1234567890
        
        self.bullish_synergy = Mock()
        self.bullish_synergy.direction = 1
        self.bullish_synergy.synergy_type = 'TYPE_1'
    
    @pytest.mark.asyncio
    async def test_end_to_end_gaming_resistance(self):
        """
        End-to-end test of complete gaming resistance across both systems
        """
        # Gaming attempt: High confidence counter-synergy with reward optimization attempt
        gaming_agent_decisions = [
            {'action': -1, 'confidence': 0.88},  # Short (counter to bullish synergy)
            {'action': -1, 'confidence': 0.85},  # Short
            {'action': 0, 'confidence': 0.7}     # Hold
        ]
        
        # First, test consensus system
        consensus_result = await self.aggregator.aggregate_decisions(
            gaming_agent_decisions, self.bullish_synergy
        )
        
        # Should be blocked by hard gate
        assert not consensus_result['should_execute'], \
            "Gaming attempt should be blocked by consensus system"
        
        # Now test what would happen if it somehow got through (for reward system testing)
        hypothetical_decision = {
            'execute': True,
            'confidence': 0.85,
            'synergy_alignment': 0.02,  # Poor alignment (gaming attempt)
            'action': -1
        }
        
        hypothetical_trade = {
            'pnl': 80.0,  # Decent PnL
            'drawdown': 0.04,  # Moderate risk
            'slippage': 0.008
        }
        
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        reward_result = self.reward_system.calculate_tactical_reward(
            hypothetical_decision, self.mock_market_state, agent_outputs, hypothetical_trade
        )
        
        # Even if it got through consensus, reward system should penalize it
        assert reward_result.total_reward < 0.3, \
            f"Reward system should penalize gaming even with good PnL: {reward_result.total_reward}"
        
        logger.info(f"End-to-end gaming resistance test passed. "
                   f"Consensus blocked: {not consensus_result['should_execute']}, "
                   f"Reward penalty: {reward_result.total_reward}")
    
    @pytest.mark.asyncio
    async def test_legitimate_strategy_optimization(self):
        """
        Test that legitimate strategy optimization is properly rewarded
        """
        # Legitimate strategy: Aligned with synergy, balanced risk-reward
        legitimate_decisions = [
            {'action': 1, 'confidence': 0.78},   # Long (aligned with bullish synergy)
            {'action': 1, 'confidence': 0.82},   # Long
            {'action': 1, 'confidence': 0.75}    # Long
        ]
        
        # Should pass consensus system
        consensus_result = await self.aggregator.aggregate_decisions(
            legitimate_decisions, self.bullish_synergy
        )
        
        assert consensus_result['should_execute'], "Legitimate strategy should execute"
        assert consensus_result['synergy_alignment'], "Should be marked as aligned"
        
        # Should be well-rewarded by reward system
        legitimate_decision = {
            'execute': True,
            'confidence': consensus_result['confidence'],
            'synergy_alignment': 0.7,  # Good alignment
            'action': 1
        }
        
        legitimate_trade = {
            'pnl': 45.0,  # Reasonable PnL
            'drawdown': 0.018,  # Reasonable risk
            'slippage': 0.007
        }
        
        agent_outputs = {'fvg_agent': Mock(), 'momentum_agent': Mock(), 'entry_opt_agent': Mock()}
        
        reward_result = self.reward_system.calculate_tactical_reward(
            legitimate_decision, self.mock_market_state, agent_outputs, legitimate_trade
        )
        
        # Should receive positive reward
        assert reward_result.total_reward > 0.2, \
            f"Legitimate strategy should be well-rewarded: {reward_result.total_reward}"
        
        logger.info(f"Legitimate strategy optimization test passed. "
                   f"Consensus confidence: {consensus_result['confidence']:.3f}, "
                   f"Reward: {reward_result.total_reward:.3f}")


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    pytest.main([__file__, '-v'])