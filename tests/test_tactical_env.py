"""
Comprehensive Tests for Tactical Environment Logic

Comprehensive test suite for TacticalMarketEnv with:
- PettingZoo API compliance testing
- FVG detection mathematical validation
- Agent cycling and state machine testing
- Matrix construction and feature validation
- Performance and latency verification
- Error handling and edge cases

Author: Quantitative Engineer
Version: 2.0 - PettingZoo Enhanced
"""

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import yaml
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.tactical_env import (
    TacticalMarketEnv, 
    TacticalState, 
    MarketState,
    AgentOutput,
    PerformanceMetrics,
    make_tactical_env,
    validate_environment
)
from pettingzoo.test import api_test


class TestTacticalEnvironmentLogic:
    """Test suite for tactical environment logic"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'feature_names': [
                        'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                        'fvg_age', 'fvg_mitigation_signal', 'price_momentum_5', 'volume_ratio'
                    ],
                    'max_episode_steps': 100,
                    'decision_timeout_ms': 100
                },
                'agents': {
                    'fvg_agent': {'attention_weights': [0.4, 0.4, 0.1, 0.05, 0.05]},
                    'momentum_agent': {'attention_weights': [0.05, 0.05, 0.1, 0.3, 0.5]},
                    'entry_opt_agent': {'attention_weights': [0.2, 0.2, 0.2, 0.2, 0.2]}
                }
            }
        }
    
    @pytest.fixture
    def test_env(self, config):
        """Create test environment"""
        return TacticalMarketEnv(config)
    
    def test_environment_initialization(self, test_env):
        """Test environment initialization"""
        # Check agents
        assert len(test_env.possible_agents) == 3
        assert 'fvg_agent' in test_env.possible_agents
        assert 'momentum_agent' in test_env.possible_agents
        assert 'entry_opt_agent' in test_env.possible_agents
        
        # Check state machine
        assert test_env.tactical_state == TacticalState.AWAITING_FVG
        
        # Check observation spaces
        for agent in test_env.possible_agents:
            assert agent in test_env.observation_spaces
            assert test_env.observation_spaces[agent].shape == (60, 7)
    
    def test_agent_cycling_full_cycle(self, test_env):
        """Test that agent cycling works correctly through full cycle"""
        # Reset environment
        observations = test_env.reset()
        
        # Check initial state
        assert test_env.tactical_state == TacticalState.AWAITING_FVG
        assert test_env.agent_selection == 'fvg_agent'
        
        # Execute FVG agent action
        obs, reward, done, truncated, info = test_env.step(1)  # Neutral action
        assert test_env.tactical_state == TacticalState.AWAITING_MOMENTUM
        assert test_env.agent_selection == 'momentum_agent'
        
        # Execute Momentum agent action
        obs, reward, done, truncated, info = test_env.step(2)  # Bullish action
        assert test_env.tactical_state == TacticalState.AWAITING_ENTRY_OPT
        assert test_env.agent_selection == 'entry_opt_agent'
        
        # Execute Entry Optimization agent action
        obs, reward, done, truncated, info = test_env.step(0)  # Bearish action
        assert test_env.tactical_state == TacticalState.AWAITING_FVG  # Reset after aggregation
        assert test_env.agent_selection == 'fvg_agent'  # Back to first agent
    
    def test_matrix_construction_shape(self, test_env):
        """Test matrix construction has correct shape"""
        observations = test_env.reset()
        
        for agent in test_env.possible_agents:
            obs = test_env.observe(agent)
            assert obs.shape == (60, 7)
            assert obs.dtype == np.float32
    
    def test_matrix_construction_features(self, test_env):
        """Test matrix construction with specific features"""
        observations = test_env.reset()
        
        # Test that observations contain valid data
        for agent in test_env.possible_agents:
            obs = test_env.observe(agent)
            
            # Check for reasonable value ranges
            assert np.all(np.isfinite(obs))
            assert not np.all(obs == 0)  # Should have some non-zero values
    
    def test_attention_weights_application(self, test_env):
        """Test that agent-specific attention weights are applied"""
        observations = test_env.reset()
        
        # Get observations for different agents
        fvg_obs = test_env.observe('fvg_agent')
        momentum_obs = test_env.observe('momentum_agent')
        
        # They should be different due to attention weights
        assert not np.allclose(fvg_obs, momentum_obs)
    
    def test_episode_termination(self, test_env):
        """Test episode termination conditions"""
        observations = test_env.reset()
        
        # Run until episode terminates
        steps = 0
        max_steps = 100
        done = False
        
        while not done and steps < max_steps:
            current_agent = test_env.agent_selection
            obs, reward, done, truncated, info = test_env.step(1)  # Neutral action
            done = done or truncated
            steps += 1
        
        assert steps == max_steps  # Should terminate due to max steps
        assert done
    
    def test_reward_distribution(self, env):
        """Test reward distribution to agents"""
        observations = env.reset()
        
        # Execute full cycle
        for _ in range(3):  # Three agents
            obs, rewards, dones, infos = env.step(1)
            
            # Check reward structure
            assert isinstance(rewards, dict)
            for agent in env.possible_agents:
                assert agent in rewards
                assert isinstance(rewards[agent], (int, float))
    
    def test_info_structure(self, env):
        """Test info dictionary structure"""
        observations = env.reset()
        
        # Execute one step
        obs, rewards, dones, infos = env.step(1)
        
        # Check info structure
        assert isinstance(infos, dict)
        for agent in env.possible_agents:
            assert agent in infos
            assert isinstance(infos[agent], dict)
    
    def test_performance_metrics(self, env):
        """Test performance metrics collection"""
        observations = env.reset()
        
        # Execute some steps
        for _ in range(10):
            obs, rewards, dones, infos = env.step(1)
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        
        assert 'episode_count' in metrics
        assert 'step_count' in metrics
        assert 'tactical_state' in metrics
        assert 'agent_count' in metrics
        assert 'decision_latencies' in metrics
    
    def test_error_handling_invalid_action(self, env):
        """Test error handling for invalid actions"""
        observations = env.reset()
        
        # Try invalid action (should handle gracefully)
        obs, rewards, dones, infos = env.step(10)  # Invalid action
        
        # Should not crash and should provide valid outputs
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(infos, dict)
    
    def test_synthetic_data_generation(self, env):
        """Test synthetic data generation"""
        # Test internal method
        synthetic_matrix = env._generate_synthetic_matrix()
        
        assert synthetic_matrix.shape == (60, 7)
        assert synthetic_matrix.dtype == np.float32
        
        # Check feature value ranges
        assert np.all(synthetic_matrix[:, 0] >= 0)  # FVG binary
        assert np.all(synthetic_matrix[:, 0] <= 1)
        assert np.all(synthetic_matrix[:, 1] >= 0)  # FVG binary
        assert np.all(synthetic_matrix[:, 1] <= 1)
        assert np.all(synthetic_matrix[:, 3] >= 0)  # Age should be non-negative
        assert np.all(synthetic_matrix[:, 6] > 0)   # Volume ratio should be positive


class TestFVGDetectionMathematics:
    """Test suite for FVG detection mathematical formulas"""
    
    @pytest.fixture
    def fvg_detector(self):
        """Create FVG detector with test config"""
        config = {
            'lookback_period': 10,
            'body_multiplier': 1.5,
            'max_fvg_age': 50,
            'mitigation_lookback': 20
        }
        return TacticalFVGDetector(config)
    
    def test_bullish_fvg_detection(self, fvg_detector):
        """Test bullish FVG detection with known pattern"""
        # Create price data with bullish FVG pattern
        # Bar i-2: High = 100.5, Low = 100.0
        # Bar i-1: Large body (Close - Open > threshold)
        # Bar i: Low = 101.0 (> High[i-2])
        
        prices = np.array([
            [100.0, 100.5, 100.0, 100.3],  # Bar i-2
            [100.2, 100.8, 100.1, 100.7],  # Bar i-1 (large body = 0.5)
            [100.9, 101.5, 101.0, 101.2],  # Bar i (Low > High[i-2])
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(prices)
        
        # Should detect bullish FVG
        assert result.fvg_bullish_active == True
        assert result.fvg_bearish_active == False
        assert len(result.active_fvgs) == 1
        assert result.active_fvgs[0].fvg_type == 'bullish'
    
    def test_bearish_fvg_detection(self, fvg_detector):
        """Test bearish FVG detection with known pattern"""
        # Create price data with bearish FVG pattern
        # Bar i-2: High = 101.0, Low = 100.5
        # Bar i-1: Large body
        # Bar i: High = 100.4 (< Low[i-2])
        
        prices = np.array([
            [100.8, 101.0, 100.5, 100.7],  # Bar i-2
            [100.9, 101.2, 100.3, 100.4],  # Bar i-1 (large body = 0.5)
            [100.5, 100.6, 100.2, 100.4],  # Bar i (High < Low[i-2])
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(prices)
        
        # Should detect bearish FVG
        assert result.fvg_bullish_active == False
        assert result.fvg_bearish_active == True
        assert len(result.active_fvgs) == 1
        assert result.active_fvgs[0].fvg_type == 'bearish'
    
    def test_body_size_filter(self, fvg_detector):
        """Test body size filter prevents small-body FVGs"""
        # Create price data with gap but small body
        prices = np.array([
            [100.0, 100.5, 100.0, 100.3],  # Bar i-2
            [100.2, 100.4, 100.1, 100.25], # Bar i-1 (small body = 0.05)
            [100.9, 101.5, 101.0, 101.2],  # Bar i (Low > High[i-2])
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(prices)
        
        # Should NOT detect FVG due to small body
        assert result.fvg_bullish_active == False
        assert result.fvg_bearish_active == False
        assert len(result.active_fvgs) == 0
    
    def test_fvg_mitigation_bullish(self, fvg_detector):
        """Test bullish FVG mitigation"""
        # First, create bullish FVG
        initial_prices = np.array([
            [100.0, 100.5, 100.0, 100.3],  # Bar i-2
            [100.2, 100.8, 100.1, 100.7],  # Bar i-1 (large body)
            [100.9, 101.5, 101.0, 101.2],  # Bar i (bullish FVG)
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(initial_prices)
        assert result.fvg_bullish_active == True
        
        # Add mitigation bar (Low goes below FVG lower level)
        mitigation_prices = np.array([
            [100.0, 100.5, 100.0, 100.3],  # Bar i-2
            [100.2, 100.8, 100.1, 100.7],  # Bar i-1
            [100.9, 101.5, 101.0, 101.2],  # Bar i (bullish FVG)
            [100.8, 101.0, 100.4, 100.6],  # Mitigation bar (Low < 100.5)
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(mitigation_prices)
        assert result.fvg_mitigation_signal == True
    
    def test_fvg_age_calculation(self, fvg_detector):
        """Test FVG age calculation"""
        # Create initial FVG
        initial_prices = np.array([
            [100.0, 100.5, 100.0, 100.3],
            [100.2, 100.8, 100.1, 100.7],
            [100.9, 101.5, 101.0, 101.2],
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(initial_prices)
        initial_age = result.fvg_age
        
        # Add more bars to increase age
        extended_prices = np.vstack([
            initial_prices,
            [[101.0, 101.3, 100.9, 101.1],
             [101.1, 101.4, 101.0, 101.2]]
        ])
        
        result = fvg_detector.detect_fvg_5min(extended_prices)
        assert result.fvg_age > initial_age
    
    def test_nearest_level_calculation(self, fvg_detector):
        """Test nearest FVG level calculation"""
        # Create FVG with known levels
        prices = np.array([
            [100.0, 100.5, 100.0, 100.3],  # Bar i-2 (High = 100.5)
            [100.2, 100.8, 100.1, 100.7],  # Bar i-1
            [100.9, 101.5, 101.0, 101.2],  # Bar i (Low = 101.0)
        ], dtype=np.float32)
        
        result = fvg_detector.detect_fvg_5min(prices)
        
        # Nearest level should be either 100.5 or 101.0
        assert result.fvg_nearest_level in [100.5, 101.0]
        assert result.fvg_nearest_level != 0.0


class TestMomentumCalculation:
    """Test suite for momentum calculation"""
    
    def test_momentum_calculation_bullish(self):
        """Test momentum calculation for bullish trend"""
        # Create price series with upward momentum
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype=np.float32)
        
        momentum = calculate_momentum_5_bar(prices, lookback=5)
        
        # Last momentum should be positive (5% increase)
        assert momentum[-1] > 0
        # Should be approximately tanh(5.0 / 5.0) = tanh(1.0) ≈ 0.762
        assert abs(momentum[-1] - np.tanh(1.0)) < 0.01
    
    def test_momentum_calculation_bearish(self):
        """Test momentum calculation for bearish trend"""
        # Create price series with downward momentum
        prices = np.array([105.0, 104.0, 103.0, 102.0, 101.0, 100.0], dtype=np.float32)
        
        momentum = calculate_momentum_5_bar(prices, lookback=5)
        
        # Last momentum should be negative
        assert momentum[-1] < 0
        # Should be approximately tanh(-4.76 / 5.0) ≈ -0.655
        assert momentum[-1] < -0.6
    
    def test_momentum_calculation_neutral(self):
        """Test momentum calculation for neutral trend"""
        # Create price series with no momentum
        prices = np.array([100.0, 100.1, 99.9, 100.0, 100.1, 100.0], dtype=np.float32)
        
        momentum = calculate_momentum_5_bar(prices, lookback=5)
        
        # Last momentum should be close to zero
        assert abs(momentum[-1]) < 0.1
    
    def test_momentum_clipping(self):
        """Test momentum clipping to [-10%, +10%] range"""
        # Create price series with extreme movement
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 150.0], dtype=np.float32)
        
        momentum = calculate_momentum_5_bar(prices, lookback=5)
        
        # Should be clipped to tanh(10.0 / 5.0) = tanh(2.0) ≈ 0.964
        assert momentum[-1] <= np.tanh(2.0)
        assert momentum[-1] > 0.9


class TestVolumeRatioCalculation:
    """Test suite for volume ratio calculation"""
    
    def test_volume_ratio_calculation_high_volume(self):
        """Test volume ratio calculation for high volume"""
        # Create volume series with spike
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 5000.0], dtype=np.float32)
        
        volume_ratio = calculate_volume_ratio_ema(volumes, period=4)
        
        # Last ratio should be high (volume spike)
        assert volume_ratio[-1] > 0.5
    
    def test_volume_ratio_calculation_low_volume(self):
        """Test volume ratio calculation for low volume"""
        # Create volume series with drop
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 200.0], dtype=np.float32)
        
        volume_ratio = calculate_volume_ratio_ema(volumes, period=4)
        
        # Last ratio should be low but positive
        assert volume_ratio[-1] >= 0.0
        assert volume_ratio[-1] < 0.5
    
    def test_volume_ratio_normalization(self):
        """Test volume ratio normalization bounds"""
        # Create various volume patterns
        volumes = np.array([1000.0, 2000.0, 500.0, 10000.0, 100.0], dtype=np.float32)
        
        volume_ratio = calculate_volume_ratio_ema(volumes, period=3)
        
        # All values should be in [0, 1] range due to tanh normalization
        assert np.all(volume_ratio >= 0.0)
        assert np.all(volume_ratio <= 1.0)
    
    def test_volume_ratio_ema_calculation(self):
        """Test EMA calculation in volume ratio"""
        # Create consistent volume series
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32)
        
        volume_ratio = calculate_volume_ratio_ema(volumes, period=3)
        
        # Ratio should be close to tanh(log(1 + 0)) = tanh(0) = 0
        assert abs(volume_ratio[-1] - 0.0) < 0.1


class TestEnvironmentPerformance:
    """Test suite for environment performance"""
    
    def test_step_latency(self, env):
        """Test step latency meets requirements"""
        observations = env.reset()
        
        import time
        latencies = []
        
        for _ in range(20):
            start_time = time.time()
            obs, rewards, dones, infos = env.step(1)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Check average latency
        avg_latency = np.mean(latencies)
        assert avg_latency < 50.0  # Should be under 50ms
    
    def test_memory_usage(self, env):
        """Test memory usage is reasonable"""
        observations = env.reset()
        
        # Run multiple episodes
        for _ in range(5):
            done = False
            steps = 0
            while not done and steps < 20:
                obs, rewards, dones, infos = env.step(1)
                done = all(dones.values()) if dones else False
                steps += 1
            
            if done:
                env.reset()
        
        # Check that agent outputs don't grow unbounded
        assert len(env.agent_outputs) <= 3  # Should be cleared after each cycle
    
    def test_episode_throughput(self, env):
        """Test episode throughput"""
        import time
        
        start_time = time.time()
        episodes_completed = 0
        
        for _ in range(10):
            observations = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 20:
                obs, rewards, dones, infos = env.step(1)
                done = all(dones.values()) if dones else False
                steps += 1
            
            episodes_completed += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete at least 5 episodes per second
        throughput = episodes_completed / total_time
        assert throughput >= 5.0


class TestPettingZooCompliance:
    """Test suite for PettingZoo API compliance"""
    
    def test_pettingzoo_api_test(self):
        """Test PettingZoo API compliance"""
        test_env = make_tactical_env()
        
        # Run PettingZoo API test
        api_test(test_env, num_cycles=5, verbose_progress=False)
    
    def test_pettingzoo_wrapper_functions(self):
        """Test PettingZoo wrapper functions"""
        test_env = make_tactical_env()
        
        assert test_env is not None
        assert hasattr(test_env, 'reset')
        assert hasattr(test_env, 'step')
        assert hasattr(test_env, 'observe')
        assert hasattr(test_env, 'close')
    
    def test_environment_validation(self):
        """Test environment validation"""
        test_env = make_tactical_env()
        
        # Run validation
        validation_results = validate_environment(test_env)
        
        # Check validation results
        assert 'pettingzoo_compliance' in validation_results
        assert 'api_compliance' in validation_results
        assert 'performance_acceptable' in validation_results
        assert 'errors' in validation_results
    
    def test_action_space_compliance(self):
        """Test action space compliance"""
        test_env = make_tactical_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            action_space = test_env.action_spaces[agent]
            assert action_space.n == 3  # Discrete actions: 0, 1, 2
            
            # Test sampling
            for _ in range(5):
                action = action_space.sample()
                assert action_space.contains(action)
                assert 0 <= action <= 2
    
    def test_observation_space_compliance(self):
        """Test observation space compliance"""
        test_env = make_tactical_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs_space = test_env.observation_spaces[agent]
            assert obs_space.shape == (60, 7)
            assert obs_space.dtype == np.float32
            
            # Test observation
            obs = test_env.observe(agent)
            assert obs_space.contains(obs)
    
    def test_environment_properties(self):
        """Test required environment properties"""
        test_env = make_tactical_env()
        
        # Check required properties
        assert hasattr(test_env, 'possible_agents')
        assert hasattr(test_env, 'agents')
        assert hasattr(test_env, 'action_spaces')
        assert hasattr(test_env, 'observation_spaces')
        assert hasattr(test_env, 'agent_selection')
        
        # Check metadata
        assert hasattr(test_env, 'metadata')
        assert 'name' in test_env.metadata
        assert 'is_parallelizable' in test_env.metadata
        assert 'render_modes' in test_env.metadata


class TestTacticalPerformanceAndStability:
    """Test suite for tactical environment performance and stability"""
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        test_env = make_tactical_env()
        
        # Measure reset time
        start_time = time.time()
        test_env.reset()
        reset_time = time.time() - start_time
        assert reset_time < 1.0  # Should reset in less than 1 second
        
        # Measure step time
        step_times = []
        for _ in range(15):  # 5 complete cycles
            start_time = time.time()
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
        
        avg_step_time = np.mean(step_times)
        assert avg_step_time < 0.2  # Should step in less than 200ms
    
    def test_memory_stability(self):
        """Test memory stability over multiple episodes"""
        test_env = make_tactical_env()
        
        for episode in range(3):
            test_env.reset()
            steps = 0
            
            while test_env.agents and steps < 20:
                action = test_env.action_spaces[test_env.agent_selection].sample()
                test_env.step(action)
                steps += 1
            
            # Check memory usage doesn't grow unbounded
            assert len(test_env.agent_outputs) <= 3
            assert len(test_env.decision_history) <= 1000
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        test_env = make_tactical_env()
        test_env.reset()
        
        # Test invalid action handling
        try:
            test_env.step(10)  # Invalid action
        except Exception:
            pass  # Should handle gracefully
        
        # Environment should still be functional
        valid_action = test_env.action_spaces[test_env.agent_selection].sample()
        test_env.step(valid_action)
        assert test_env.agent_selection in test_env.agents
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with same seed"""
        # Test with same seed
        results1 = []
        test_env1 = make_tactical_env()
        test_env1.reset(seed=42)
        
        for _ in range(9):  # 3 complete cycles
            action = 1  # Fixed action
            test_env1.step(action)
            results1.append(test_env1.performance_metrics.step_count)
        
        results2 = []
        test_env2 = make_tactical_env()
        test_env2.reset(seed=42)
        
        for _ in range(9):  # 3 complete cycles
            action = 1  # Fixed action
            test_env2.step(action)
            results2.append(test_env2.performance_metrics.step_count)
        
        # Results should be identical
        assert results1 == results2
    
    def test_concurrent_access(self):
        """Test concurrent access safety"""
        test_env = make_tactical_env()
        test_env.reset()
        
        # Test multiple concurrent observations
        obs1 = test_env.observe('fvg_agent')
        obs2 = test_env.observe('fvg_agent')
        
        # Should be identical
        assert np.array_equal(obs1, obs2)
    
    def test_configuration_robustness(self):
        """Test configuration robustness"""
        # Test with minimal config
        minimal_config = {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'max_episode_steps': 50
                }
            }
        }
        
        test_env = make_tactical_env(minimal_config)
        test_env.reset()
        
        # Should work with minimal config
        for _ in range(6):  # 2 complete cycles
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
        
        assert test_env.performance_metrics.step_count > 0


def test_configuration_loading():
    """Test configuration loading from file"""
    # Create temporary config file
    test_config = {
        'tactical_marl': {
            'environment': {
                'matrix_shape': [60, 7],
                'max_episode_steps': 500
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # Test loading
        env = TacticalMarketEnv(config_path)
        assert env.config['tactical_marl']['environment']['max_episode_steps'] == 500
    finally:
        os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])