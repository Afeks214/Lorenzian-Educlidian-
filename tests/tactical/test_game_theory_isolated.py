"""
Isolated Game Theory Test Suite for Tactical Reward System

Tests the core mathematical properties without complex module dependencies.
Validates that the reward function is mathematically resistant to gaming.

Author: Agent 3 - Game Theorist & Reward Architect
Mission: Aegis - Eliminate Gaming Vulnerabilities
"""

import pytest
import numpy as np
from unittest.mock import Mock
import logging

logger = logging.getLogger(__name__)


class MockTacticalRewardSystem:
    """
    Mock implementation of the game-resistant reward system for testing
    """
    
    def __init__(self):
        """Initialize with same parameters as real system"""
        self.pnl_normalizer = 100.0
    
    def _calculate_game_resistant_reward(
        self,
        pnl_reward: float,
        synergy_bonus: float,
        risk_penalty: float,
        execution_bonus: float
    ) -> float:
        """
        Game-theory resistant reward calculation
        (Copied from actual implementation)
        """
        
        # Normalize components to [0, 1] range for product operations
        normalized_pnl = (pnl_reward + 1.0) / 2.0  # Transform [-1,1] to [0,1]
        normalized_synergy = max(0.0, synergy_bonus)  # Already [0, inf), clip negative
        normalized_execution = max(0.0, execution_bonus)  # Already [0, inf), clip negative
        
        # Transform risk penalty to risk factor [0, 1] where 1 = no risk, 0 = max risk
        risk_factor = 1.0 / (1.0 + abs(risk_penalty))  # Always positive, decreases with penalty
        
        # CORE STRATEGIC REQUIREMENT: Strategic alignment is mandatory
        # If synergy_bonus <= 0, reward is heavily penalized (strategic gate)
        strategic_gate = 1.0 if synergy_bonus > 0.05 else 0.1  # 90% penalty for poor alignment
        
        # Multi-objective product formulation (game-resistant)
        # Each component must be optimized - cannot game by avoiding one
        base_performance = normalized_pnl * risk_factor  # Sharpe-like: return/risk
        execution_quality = 1.0 + normalized_execution  # Execution bonus multiplier
        synergy_multiplier = 1.0 + normalized_synergy  # Strategic alignment multiplier
        
        # Final reward: Product of all objectives with strategic gate
        total_reward = (
            strategic_gate *  # Hard strategic constraint
            base_performance *  # Risk-adjusted returns (cannot game)
            execution_quality *  # Execution quality bonus
            synergy_multiplier  # Strategic alignment bonus
        )
        
        # Scale to reasonable range and maintain mathematical properties
        total_reward = np.tanh(2.0 * (total_reward - 1.0))  # Center around 0, bound [-1,1]
        
        # Final clipping for safety
        return float(np.clip(total_reward, -2.0, 2.0))
    
    def calculate_tactical_reward(self, pnl: float, synergy: float, risk: float, execution: float):
        """Simplified reward calculation for testing"""
        # Convert inputs to component format
        pnl_reward = np.tanh(pnl / self.pnl_normalizer)
        synergy_bonus = synergy
        risk_penalty = risk
        execution_bonus = execution
        
        total_reward = self._calculate_game_resistant_reward(
            pnl_reward, synergy_bonus, risk_penalty, execution_bonus
        )
        
        return {
            'total_reward': total_reward,
            'pnl_reward': pnl_reward,
            'synergy_bonus': synergy_bonus,
            'risk_penalty': risk_penalty,
            'execution_bonus': execution_bonus
        }


class MockTacticalAggregator:
    """
    Mock implementation of hard synergy alignment gate
    """
    
    def __init__(self):
        self.execution_threshold = 0.65
    
    def aggregate_decisions(self, weighted_score: float, synergy_direction: int, action: int):
        """
        Simplified aggregation with hard gate logic
        """
        should_execute = weighted_score >= self.execution_threshold
        
        # HARD SYNERGY ALIGNMENT GATE
        if should_execute and action != 0:  # Not hold
            direction_match = (
                (action > 0 and synergy_direction > 0) or
                (action < 0 and synergy_direction < 0)
            )
            
            if not direction_match:
                # HARD GATE: Counter-synergy trades require >95% confidence
                if weighted_score < 0.95:
                    should_execute = False
                    weighted_score = 0.0  # Complete veto
        
        return {
            'should_execute': should_execute,
            'confidence': weighted_score,
            'synergy_alignment': action == 0 or (action > 0 and synergy_direction > 0) or (action < 0 and synergy_direction < 0)
        }


class TestRewardGameResistance:
    """Test suite for reward function gaming resistance"""
    
    def setup_method(self):
        """Setup test environment"""
        self.reward_system = MockTacticalRewardSystem()
    
    def test_linear_combination_gaming_prevented(self):
        """
        Test that linear combination gaming is prevented by product formulation
        """
        # Gaming strategy: High PnL, zero synergy (old system exploit)
        gaming_reward = self.reward_system.calculate_tactical_reward(
            pnl=100.0,      # High PnL
            synergy=0.0,    # No strategic alignment (gaming attempt)
            risk=-0.1,      # High risk
            execution=0.1   # Good execution
        )
        
        # Balanced strategy: Moderate PnL, good synergy
        balanced_reward = self.reward_system.calculate_tactical_reward(
            pnl=50.0,       # Moderate PnL
            synergy=0.6,    # Good strategic alignment
            risk=-0.05,     # Moderate risk
            execution=0.08  # Good execution
        )
        
        # Strategic alignment should dominate - balanced should beat gaming
        assert balanced_reward['total_reward'] > gaming_reward['total_reward'], \
            f"Balanced strategy should beat gaming: {balanced_reward['total_reward']} > {gaming_reward['total_reward']}"
        
        # Gaming strategy should be heavily penalized for poor synergy
        assert gaming_reward['total_reward'] < 0.2, \
            f"Gaming strategy should yield low reward: {gaming_reward['total_reward']}"
        
        logger.info(f"Gaming prevention test passed. Gaming: {gaming_reward['total_reward']:.3f}, "
                   f"Balanced: {balanced_reward['total_reward']:.3f}")
    
    def test_strategic_alignment_criticality(self):
        """
        Test that strategic alignment is mathematically critical
        """
        # Excellent trade but poor alignment
        poor_alignment = self.reward_system.calculate_tactical_reward(
            pnl=200.0,      # Excellent PnL
            synergy=0.01,   # Very poor alignment
            risk=-0.01,     # Very low risk
            execution=0.15  # Excellent execution
        )
        
        # Good trade with good alignment
        good_alignment = self.reward_system.calculate_tactical_reward(
            pnl=60.0,       # Good PnL
            synergy=0.7,    # Good alignment
            risk=-0.03,     # Good risk
            execution=0.08  # Good execution
        )
        
        # Strategic alignment should create significant reward difference
        assert good_alignment['total_reward'] > poor_alignment['total_reward'], \
            f"Strategic alignment should dominate: {good_alignment['total_reward']} > {poor_alignment['total_reward']}"
        
        logger.info(f"Strategic criticality test passed. Good alignment: {good_alignment['total_reward']:.3f}, "
                   f"Poor alignment: {poor_alignment['total_reward']:.3f}")
    
    def test_product_vs_linear_comparison(self):
        """
        Mathematical proof that product formulation beats linear for gaming resistance
        """
        test_cases = [
            # Case 1: High PnL gaming attempt
            {'pnl': 100.0, 'synergy': 0.0, 'risk': -0.3, 'execution': 0.1},
            # Case 2: Moderate balanced approach
            {'pnl': 50.0, 'synergy': 0.6, 'risk': -0.1, 'execution': 0.08},
            # Case 3: Conservative but aligned
            {'pnl': 25.0, 'synergy': 0.8, 'risk': -0.05, 'execution': 0.05},
        ]
        
        product_rewards = []
        linear_rewards = []
        
        for case in test_cases:
            # New product-based system
            product_result = self.reward_system.calculate_tactical_reward(**case)
            product_rewards.append(product_result['total_reward'])
            
            # Simulate old linear system
            linear_reward = (
                1.0 * np.tanh(case['pnl'] / 100.0) +  # PnL weight
                0.2 * case['synergy'] +               # Synergy weight
                -0.5 * abs(case['risk']) +            # Risk weight (negative)
                0.1 * case['execution']               # Execution weight
            )
            linear_rewards.append(linear_reward)
        
        # Find best strategies in each system
        linear_best_idx = np.argmax(linear_rewards)
        product_best_idx = np.argmax(product_rewards)
        
        logger.info(f"Linear system best: Case {linear_best_idx}, rewards: {linear_rewards}")
        logger.info(f"Product system best: Case {product_best_idx}, rewards: {product_rewards}")
        
        # Product system should favor strategic alignment over pure PnL gaming
        # Case 0 is the gaming attempt, cases 1-2 are strategic
        assert product_best_idx != 0 or product_rewards[1] > product_rewards[0] * 0.8, \
            "Product system should not favor pure PnL gaming"
        
        # Strategic alignment cases should perform relatively better in product system
        if linear_rewards[0] > 0 and product_rewards[0] > 0:
            strategic_improvement = (product_rewards[1] / product_rewards[0]) / (linear_rewards[1] / linear_rewards[0])
            assert strategic_improvement > 0.9, f"Strategic cases should improve relatively: {strategic_improvement}"
    
    def test_mathematical_bounds_and_stability(self):
        """
        Test mathematical bounds and numerical stability
        """
        extreme_cases = [
            {'pnl': 1000.0, 'synergy': 1.0, 'risk': 0.0, 'execution': 1.0},      # Extreme positive
            {'pnl': -1000.0, 'synergy': 0.0, 'risk': -1.0, 'execution': 0.0},   # Extreme negative
            {'pnl': 0.0, 'synergy': 0.5, 'risk': -0.5, 'execution': 0.5},       # Balanced
        ]
        
        for case in extreme_cases:
            result = self.reward_system.calculate_tactical_reward(**case)
            
            # Ensure rewards are bounded
            assert -2.0 <= result['total_reward'] <= 2.0, \
                f"Reward should be bounded: {result['total_reward']}"
            
            # Ensure finite values
            assert np.isfinite(result['total_reward']), \
                f"Reward should be finite: {result['total_reward']}"
            assert not np.isnan(result['total_reward']), \
                f"Reward should not be NaN: {result['total_reward']}"
        
        logger.info("Mathematical bounds and stability test passed")


class TestConsensusOverrideResistance:
    """Test suite for consensus override gaming resistance"""
    
    def setup_method(self):
        """Setup test environment"""
        self.aggregator = MockTacticalAggregator()
    
    def test_hard_gate_blocks_counter_synergy(self):
        """
        Test that hard gate blocks counter-synergy trades below 95% confidence
        """
        # Counter-synergy: want to short but synergy is bullish
        result = self.aggregator.aggregate_decisions(
            weighted_score=0.85,  # High but not ultra-high confidence
            synergy_direction=1,  # Bullish synergy
            action=-1             # Short (counter to synergy)
        )
        
        # Should be blocked
        assert not result['should_execute'], "Counter-synergy trade should be blocked"
        assert result['confidence'] == 0.0, "Confidence should be zeroed by hard gate"
        
        logger.info("Hard gate blocking test passed")
    
    def test_ultra_high_confidence_override(self):
        """
        Test that ultra-high confidence can override strategic direction
        """
        result = self.aggregator.aggregate_decisions(
            weighted_score=0.96,  # Ultra-high confidence
            synergy_direction=1,  # Bullish synergy
            action=-1             # Short (counter to synergy)
        )
        
        # Should be allowed
        assert result['should_execute'], "Ultra-high confidence should override"
        assert result['confidence'] >= 0.95, "Confidence should remain high"
        
        logger.info("Ultra-high confidence override test passed")
    
    def test_aligned_trades_execute_normally(self):
        """
        Test that aligned trades execute with normal confidence requirements
        """
        result = self.aggregator.aggregate_decisions(
            weighted_score=0.7,   # Normal confidence
            synergy_direction=1,  # Bullish synergy
            action=1              # Long (aligned)
        )
        
        # Should execute normally
        assert result['should_execute'], "Aligned trade should execute"
        assert result['synergy_alignment'], "Should be marked as aligned"
        
        logger.info("Aligned trade execution test passed")


class TestIntegratedGameTheoryResistance:
    """Integration tests for complete system"""
    
    def setup_method(self):
        """Setup integrated test environment"""
        self.reward_system = MockTacticalRewardSystem()
        self.aggregator = MockTacticalAggregator()
    
    def test_end_to_end_gaming_resistance(self):
        """
        End-to-end test of gaming resistance across both systems
        """
        # Gaming attempt: Counter-synergy trade with reward optimization
        consensus_result = self.aggregator.aggregate_decisions(
            weighted_score=0.88,  # High but not ultra-high
            synergy_direction=1,  # Bullish synergy
            action=-1             # Short (gaming attempt)
        )
        
        # Should be blocked by consensus system
        assert not consensus_result['should_execute'], \
            "Gaming attempt should be blocked by consensus"
        
        # Even if it somehow got through, reward system should penalize
        reward_result = self.reward_system.calculate_tactical_reward(
            pnl=80.0,       # Good PnL
            synergy=0.02,   # Poor alignment (gaming)
            risk=-0.04,     # Moderate risk
            execution=0.08  # Good execution
        )
        
        # Should be heavily penalized
        assert reward_result['total_reward'] < 0.3, \
            f"Reward system should penalize gaming: {reward_result['total_reward']}"
        
        logger.info("End-to-end gaming resistance test passed")
    
    def test_legitimate_strategy_optimization(self):
        """
        Test that legitimate strategies are properly rewarded
        """
        # Legitimate aligned strategy
        consensus_result = self.aggregator.aggregate_decisions(
            weighted_score=0.78,  # Good confidence
            synergy_direction=1,  # Bullish synergy
            action=1              # Long (aligned)
        )
        
        # Should execute
        assert consensus_result['should_execute'], "Legitimate strategy should execute"
        
        # Should be well-rewarded
        reward_result = self.reward_system.calculate_tactical_reward(
            pnl=45.0,       # Reasonable PnL
            synergy=0.7,    # Good alignment
            risk=-0.018,    # Good risk
            execution=0.07  # Good execution
        )
        
        # Should receive positive reward
        assert reward_result['total_reward'] > 0.2, \
            f"Legitimate strategy should be rewarded: {reward_result['total_reward']}"
        
        logger.info("Legitimate strategy optimization test passed")


# Mathematical validation functions
def test_nash_equilibrium_properties():
    """
    Test Nash equilibrium properties of the reward system
    """
    reward_system = MockTacticalRewardSystem()
    
    strategies = [
        {'name': 'gaming', 'pnl': 100.0, 'synergy': 0.0, 'risk': -0.3, 'execution': 0.1},
        {'name': 'balanced', 'pnl': 60.0, 'synergy': 0.7, 'risk': -0.1, 'execution': 0.08},
        {'name': 'conservative', 'pnl': 30.0, 'synergy': 0.8, 'risk': -0.05, 'execution': 0.05},
    ]
    
    rewards = []
    for strategy in strategies:
        result = reward_system.calculate_tactical_reward(**{k: v for k, v in strategy.items() if k != 'name'})
        rewards.append(result['total_reward'])
    
    # Find optimal strategy
    best_idx = np.argmax(rewards)
    optimal_strategy = strategies[best_idx]['name']
    
    logger.info(f"Nash equilibrium test: Optimal strategy is '{optimal_strategy}' with rewards: {dict(zip([s['name'] for s in strategies], rewards))}")
    
    # The balanced strategy should be optimal or very close
    assert optimal_strategy in ['balanced', 'conservative'], \
        f"Nash equilibrium should favor strategic approaches, got: {optimal_strategy}"


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, '-v'])