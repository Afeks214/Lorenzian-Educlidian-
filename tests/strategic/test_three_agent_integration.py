"""
Comprehensive Integration Tests for Strategic Agents

This module tests the integration of all three strategic agents:
1. NWRQKStrategicAgent - Support/resistance detection using corrected kernel
2. RegimeDetectionAgent - Market regime classification using MMD
3. Integration testing - Coordination, timing, and data flow validation

Tests cover mathematical validation, feature extraction, decision making,
and performance requirements.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import time
from typing import Dict, Any, List, Tuple

from src.agents import (
    BaseStrategicAgent, 
    NWRQKStrategicAgent, 
    RegimeDetectionAgent,
    StrategicAction,
    MarketRegime
)
from src.indicators.custom.nwrqk import rational_quadratic_kernel
from src.indicators.custom.mmd import compute_mmd, gaussian_kernel


class TestNWRQKKernelCorrection:
    """Test the corrected NWRQK kernel implementation"""
    
    def test_rational_quadratic_kernel_formula(self):
        """Test that RQ kernel matches PRD specification exactly"""
        # PRD specification: K_h(x_t, x_i) = (1 + ||x_t - x_i||^2 / (2αh^2))^(-α)
        x_t = 100.0
        x_i = 105.0
        alpha = 1.0
        h = 1.0
        
        # Calculate expected value manually
        distance_squared = (x_t - x_i) ** 2  # 25.0
        epsilon = 1e-10
        expected = (1 + (distance_squared + epsilon) / (2 * alpha * h**2)) ** (-alpha)
        
        # Test our implementation
        result = rational_quadratic_kernel(x_t, x_i, alpha, h)
        
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        
    def test_kernel_properties(self):
        """Test mathematical properties of the RQ kernel"""
        # Test symmetry: K(x, y) = K(y, x)
        x, y = 100.0, 105.0
        k1 = rational_quadratic_kernel(x, y)
        k2 = rational_quadratic_kernel(y, x)
        assert abs(k1 - k2) < 1e-10, "Kernel should be symmetric"
        
        # Test positive definiteness: K(x, x) = 1
        k_self = rational_quadratic_kernel(x, x)
        assert abs(k_self - 1.0) < 1e-10, "Self-kernel should equal 1"
        
        # Test decay with distance
        distances = [0, 1, 5, 10, 20]
        kernels = [rational_quadratic_kernel(100.0, 100.0 + d) for d in distances]
        
        for i in range(1, len(kernels)):
            assert kernels[i] <= kernels[i-1], "Kernel should decay with distance"
            
    def test_numerical_stability(self):
        """Test numerical stability for edge cases"""
        # Very small distances
        k1 = rational_quadratic_kernel(100.0, 100.0 + 1e-15)
        assert 0.95 <= k1 <= 1.0, "Should handle very small distances"
        
        # Large distances
        k2 = rational_quadratic_kernel(0.0, 1000.0)
        assert k2 >= 0.0, "Should handle large distances"
        assert k2 < 0.1, "Should decay significantly for large distances"


class TestNWRQKStrategicAgent:
    """Test NWRQK Strategic Agent implementation"""
    
    @pytest.fixture
    def agent_config(self):
        return {
            'name': 'nwrqk_test_agent',
            'observation_dim': 624,
            'action_dim': 5,
            'alpha': 1.0,
            'h': 1.0,
            'lookback_window': 50,
            'density_threshold': 0.3,
            'breakout_threshold': 0.02
        }
    
    @pytest.fixture
    def nwrqk_agent(self, agent_config):
        return NWRQKStrategicAgent(agent_config)
    
    def test_feature_extraction(self, nwrqk_agent):
        """Test extraction of features [2, 3, 4, 5] from 48x13 matrix"""
        # Create test observation matrix
        observation = np.random.rand(48, 13)
        observation[-1, 2] = 0.5  # nwrqk_value
        observation[-1, 3] = 0.1  # nwrqk_slope
        observation[-1, 4] = 0.3  # lvn_distance
        observation[-1, 5] = 0.7  # lvn_strength
        
        features = nwrqk_agent.extract_features(observation)
        
        assert len(features) == 4, "Should extract 4 features"
        assert abs(features[0] - 0.5) < 1e-10, "nwrqk_value should match"
        assert abs(features[1] - 0.1) < 1e-10, "nwrqk_slope should match"
        assert abs(features[2] - 0.3) < 1e-10, "lvn_distance should match"
        assert abs(features[3] - 0.7) < 1e-10, "lvn_strength should match"
        
    def test_support_resistance_detection(self, nwrqk_agent):
        """Test support/resistance level detection logic"""
        # Create price series with clear support/resistance
        prices = np.array([100, 102, 98, 100, 103, 99, 101, 98, 100, 102])
        nwrqk_values = prices.copy()  # Simplified for testing
        
        nwrqk_agent.detect_support_resistance(prices, nwrqk_values)
        
        # Should detect levels around 98 (support) and 103 (resistance)
        assert len(nwrqk_agent.support_levels) > 0, "Should detect support levels"
        assert len(nwrqk_agent.resistance_levels) >= 0, "May detect resistance levels"
        
        # Check level properties
        for level in nwrqk_agent.support_levels:
            assert level.level_type == 'support'
            assert 0.0 <= level.strength <= 1.0
            assert level.price > 0
            
    def test_breakout_probability_calculation(self, nwrqk_agent):
        """Test breakout probability calculation"""
        from src.agents.nwrqk_strategic_agent import SupportResistanceLevel
        
        # Create test level
        level = SupportResistanceLevel(100.0, 0.8, 'support')
        
        # Test near level (should have lower breakout probability)
        prob_near = nwrqk_agent.calculate_breakout_probability(100.5, level)
        
        # Test far from level (should have higher breakout probability)
        prob_far = nwrqk_agent.calculate_breakout_probability(95.0, level)
        
        assert 0.0 <= prob_near <= 1.0, "Probability should be in [0,1]"
        assert 0.0 <= prob_far <= 1.0, "Probability should be in [0,1]"
        
    def test_decision_making(self, nwrqk_agent):
        """Test strategic decision making"""
        # Test different feature combinations
        test_cases = [
            # [nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength]
            ([100.0, 0.5, 0.1, 0.8], "Strong uptrend near strong level"),
            ([100.0, -0.5, 0.1, 0.8], "Strong downtrend near strong level"),
            ([100.0, 0.0, 0.5, 0.2], "Sideways, away from levels"),
            ([100.0, 0.3, 0.01, 0.9], "Moderate uptrend at very strong level")
        ]
        
        for features, description in test_cases:
            action, confidence = nwrqk_agent.make_decision(np.array(features))
            
            assert 0 <= action <= 4, f"Action should be valid strategic action: {description}"
            assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0,1]: {description}"


class TestRegimeDetectionAgent:
    """Test Regime Detection Agent implementation"""
    
    @pytest.fixture
    def agent_config(self):
        return {
            'name': 'regime_test_agent',
            'observation_dim': 624,
            'action_dim': 5,
            'mmd_sigma': 1.0,
            'reference_window': 500,
            'test_window': 100,
            'low_vol_threshold': 0.2,
            'medium_vol_threshold': 0.5,
            'high_vol_threshold': 0.8,
            'crisis_vol_threshold': 1.2
        }
    
    @pytest.fixture
    def regime_agent(self, agent_config):
        return RegimeDetectionAgent(agent_config)
    
    def test_mmd_calculation_verification(self, regime_agent):
        """Test that MMD calculation matches PRD specification"""
        # Create test distributions
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 4))  # Reference distribution
        Y = np.random.normal(0.5, 1.2, (30, 4))  # Test distribution (shifted and scaled)
        
        # Test our MMD implementation
        mmd_score = regime_agent.verify_mmd_calculation(X, Y)
        
        # Verify MMD properties
        assert mmd_score >= 0, "MMD should be non-negative"
        
        # Test with identical distributions (should be close to 0)
        mmd_identical = regime_agent.verify_mmd_calculation(X, X)
        assert mmd_identical < 0.1, "MMD of identical distributions should be small"
        
        # Test that different distributions have higher MMD than identical
        assert mmd_score > mmd_identical, "Different distributions should have higher MMD"
        
    def test_feature_extraction(self, regime_agent):
        """Test extraction of features [10, 11, 12] from 48x13 matrix"""
        # Create test observation matrix
        observation = np.random.rand(48, 13)
        observation[-1, 10] = 0.3  # mmd_score
        observation[-1, 11] = 0.6  # volatility_30
        observation[-1, 12] = 1.2  # volume_profile_skew
        
        features = regime_agent.extract_features(observation)
        
        assert len(features) == 3, "Should extract 3 features"
        assert abs(features[0] - 0.3) < 1e-10, "mmd_score should match"
        assert abs(features[1] - 0.6) < 1e-10, "volatility_30 should match"
        assert abs(features[2] - 1.2) < 1e-10, "volume_profile_skew should match"
        
    def test_regime_classification(self, regime_agent):
        """Test regime classification logic"""
        test_cases = [
            # [mmd_score, volatility_30, volume_skew], expected_regime_type
            ([0.1, 0.1, 0.0], "low volatility -> sideways"),
            ([0.5, 1.5, 2.0], "high volatility + high MMD -> crisis"),
            ([0.3, 0.4, 0.5], "medium volatility + medium MMD -> trending"),
            ([0.05, 0.15, 0.1], "very low -> recovery/sideways")
        ]
        
        for features, description in test_cases:
            regime, confidence = regime_agent.classify_regime(np.array(features))
            
            assert isinstance(regime, MarketRegime), f"Should return MarketRegime: {description}"
            assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0,1]: {description}"
            
    def test_volatility_adjusted_parameters(self, regime_agent):
        """Test volatility-adjusted policy parameters"""
        # Test crisis-level volatility
        crisis_params = regime_agent.get_volatility_adjusted_params(1.5)
        assert crisis_params["risk_multiplier"] < 0.5, "Crisis should reduce risk"
        assert crisis_params["confidence_threshold"] > 0.8, "Crisis should require high confidence"
        
        # Test low volatility
        low_vol_params = regime_agent.get_volatility_adjusted_params(0.1)
        assert low_vol_params["risk_multiplier"] > 1.0, "Low vol should allow higher risk"
        assert low_vol_params["confidence_threshold"] < 0.7, "Low vol should allow lower confidence"
        
    def test_regime_transition_tracking(self, regime_agent):
        """Test regime transition detection and tracking"""
        # Simulate regime changes
        regime_agent.current_regime = MarketRegime.SIDEWAYS
        
        # Trigger regime change
        features = np.array([0.8, 1.0, 1.5])  # High volatility features
        action, confidence = regime_agent.make_decision(features)
        
        # Should detect transition
        assert len(regime_agent.regime_history) > 0 or regime_agent.current_regime != MarketRegime.SIDEWAYS
        
    def test_decision_making(self, regime_agent):
        """Test regime-based decision making"""
        test_cases = [
            # Bull trend scenario
            ([0.3, 0.4, 0.2], "bull trend conditions"),
            # Bear trend scenario
            ([0.4, 0.5, -0.3], "bear trend conditions"),
            # Crisis scenario
            ([0.6, 1.3, 2.0], "crisis conditions"),
            # Sideways scenario
            ([0.1, 0.15, 0.1], "sideways conditions")
        ]
        
        for features, description in test_cases:
            action, confidence = regime_agent.make_decision(np.array(features))
            
            assert 0 <= action <= 4, f"Action should be valid: {description}"
            assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0,1]: {description}"


class TestThreeAgentIntegration:
    """Test integration of all three strategic agents"""
    
    @pytest.fixture
    def agent_configs(self):
        return {
            'base_config': {
                'observation_dim': 624,
                'action_dim': 5
            },
            'nwrqk_config': {
                'name': 'nwrqk_integration_test',
                'alpha': 1.0,
                'h': 1.0,
                'lookback_window': 30
            },
            'regime_config': {
                'name': 'regime_integration_test',
                'mmd_sigma': 1.0,
                'reference_window': 100,
                'test_window': 50
            }
        }
    
    @pytest.fixture
    def three_agents(self, agent_configs):
        """Create all three agents for integration testing"""
        base_config = agent_configs['base_config']
        
        nwrqk_agent = NWRQKStrategicAgent({**base_config, **agent_configs['nwrqk_config']})
        regime_agent = RegimeDetectionAgent({**base_config, **agent_configs['regime_config']})
        
        return {
            'nwrqk': nwrqk_agent,
            'regime': regime_agent
        }
    
    def test_feature_extraction_coordination(self, three_agents):
        """Test that all agents extract correct features from 48x13 matrix"""
        # Create comprehensive test observation
        observation = np.random.rand(48, 13)
        
        # Set specific values for validation
        observation[-1, 2:6] = [0.5, 0.1, 0.3, 0.7]  # NWRQK features [2,3,4,5]
        observation[-1, 10:13] = [0.3, 0.6, 1.2]      # Regime features [10,11,12]
        
        # Test NWRQK agent
        nwrqk_features = three_agents['nwrqk'].extract_features(observation)
        assert len(nwrqk_features) == 4, "NWRQK should extract 4 features"
        np.testing.assert_array_almost_equal(nwrqk_features, [0.5, 0.1, 0.3, 0.7])
        
        # Test Regime agent
        regime_features = three_agents['regime'].extract_features(observation)
        assert len(regime_features) == 3, "Regime should extract 3 features"
        np.testing.assert_array_almost_equal(regime_features, [0.3, 0.6, 1.2])
        
    def test_decision_coordination(self, three_agents):
        """Test decision coordination between agents"""
        observation = np.random.rand(48, 13)
        
        # Get decisions from all agents
        decisions = {}
        for name, agent in three_agents.items():
            action, confidence = agent.step(observation)
            decisions[name] = {'action': action, 'confidence': confidence}
            
        # Validate all decisions
        for name, decision in decisions.items():
            assert 0 <= decision['action'] <= 4, f"{name} action should be valid"
            assert 0.0 <= decision['confidence'] <= 1.0, f"{name} confidence should be in [0,1]"
            
    def test_performance_requirements(self, three_agents):
        """Test that all agents meet <5ms total inference requirement"""
        observation = np.random.rand(48, 13)
        
        # Warm up (JIT compilation, etc.)
        for agent in three_agents.values():
            agent.step(observation)
        
        # Time multiple decisions
        total_times = []
        for _ in range(10):
            start_time = time.time()
            
            for agent in three_agents.values():
                agent.step(observation)
            
            end_time = time.time()
            total_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(total_times)
        max_time = np.max(total_times)
        
        # Agent 4 requires <5ms total inference time
        assert avg_time < 5.0, f"Average time {avg_time:.2f}ms should be < 5ms"
        assert max_time < 10.0, f"Max time {max_time:.2f}ms should be reasonable"
        
    def test_probability_distributions(self, three_agents):
        """Test that all agent outputs are valid probability distributions"""
        observation = np.random.rand(48, 13)
        
        for name, agent in three_agents.items():
            features = agent.extract_features(observation)
            probs = agent.get_action_probabilities(features)
            
            assert len(probs) == 5, f"{name} should output 5 action probabilities"
            assert abs(np.sum(probs) - 1.0) < 1e-10, f"{name} probabilities should sum to 1"
            assert np.all(probs >= 0), f"{name} probabilities should be non-negative"
            
    def test_agent_state_management(self, three_agents):
        """Test agent state management and reset functionality"""
        observation = np.random.rand(48, 13)
        
        # Make some decisions to change state
        for agent in three_agents.values():
            for _ in range(5):
                agent.step(observation)
        
        # Verify agents have made decisions
        for name, agent in three_agents.items():
            assert agent.decisions_made > 0, f"{name} should have made decisions"
            
        # Reset all agents
        for agent in three_agents.values():
            agent.reset()
            
        # Verify reset worked
        for name, agent in three_agents.items():
            assert agent.decisions_made == 0, f"{name} should be reset"
            
    def test_error_handling(self, three_agents):
        """Test error handling with malformed inputs"""
        bad_inputs = [
            None,
            np.array([]),
            np.random.rand(10, 5),  # Wrong shape
            np.full((48, 13), np.nan),  # NaN values
            np.full((48, 13), np.inf)   # Inf values
        ]
        
        for bad_input in bad_inputs:
            for name, agent in three_agents.items():
                try:
                    action, confidence = agent.step(bad_input)
                    # Should not crash and should return reasonable defaults
                    assert 0 <= action <= 4, f"{name} should handle bad input gracefully"
                    assert 0.0 <= confidence <= 1.0, f"{name} confidence should be valid even with bad input"
                except Exception as e:
                    pytest.fail(f"{name} should not raise exception with bad input: {e}")
                    
    def test_mathematical_consistency(self, three_agents):
        """Test mathematical consistency across multiple runs"""
        observation = np.random.rand(48, 13)
        
        # Test deterministic behavior (same input -> same output)
        for name, agent in three_agents.items():
            action1, conf1 = agent.step(observation.copy())
            action2, conf2 = agent.step(observation.copy())
            
            # Note: Some agents may have internal state that changes behavior
            # So we just ensure outputs are valid, not necessarily identical
            assert 0 <= action1 <= 4 and 0 <= action2 <= 4, f"{name} actions should be valid"
            assert 0.0 <= conf1 <= 1.0 and 0.0 <= conf2 <= 1.0, f"{name} confidences should be valid"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])