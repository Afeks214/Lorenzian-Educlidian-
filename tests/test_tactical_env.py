"""
Quantitative Unit Tests for Tactical Environment Logic

Comprehensive test suite for TacticalMarketEnv with:
- FVG detection mathematical validation
- Agent cycling and state machine testing
- Matrix construction and feature validation
- Performance and latency verification
- Error handling and edge cases

Author: Quantitative Engineer
Version: 1.0
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import yaml
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from environment.tactical_env import TacticalMarketEnv, TacticalState, MarketState
from src.indicators.custom.tactical_fvg import TacticalFVGDetector, calculate_momentum_5_bar, calculate_volume_ratio_ema


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
    def env(self, config):
        """Create test environment"""
        return TacticalMarketEnv(config)
    
    def test_environment_initialization(self, env):
        """Test environment initialization"""
        # Check agents
        assert len(env.possible_agents) == 3
        assert 'fvg_agent' in env.possible_agents
        assert 'momentum_agent' in env.possible_agents
        assert 'entry_opt_agent' in env.possible_agents
        
        # Check state machine
        assert env.tactical_state == TacticalState.AWAITING_FVG
        
        # Check observation spaces
        for agent in env.possible_agents:
            assert agent in env.observation_spaces
            assert env.observation_spaces[agent].shape == (60, 7)
    
    def test_agent_cycling_full_cycle(self, env):
        """Test that agent cycling works correctly through full cycle"""
        # Reset environment
        observations = env.reset()
        
        # Check initial state
        assert env.tactical_state == TacticalState.AWAITING_FVG
        assert env.agent_selection == 'fvg_agent'
        
        # Execute FVG agent action
        obs, rewards, dones, infos = env.step(1)  # Neutral action
        assert env.tactical_state == TacticalState.AWAITING_MOMENTUM
        assert env.agent_selection == 'momentum_agent'
        
        # Execute Momentum agent action
        obs, rewards, dones, infos = env.step(2)  # Bullish action
        assert env.tactical_state == TacticalState.AWAITING_ENTRY_OPT
        assert env.agent_selection == 'entry_opt_agent'
        
        # Execute Entry Optimization agent action
        obs, rewards, dones, infos = env.step(0)  # Bearish action
        assert env.tactical_state == TacticalState.AWAITING_FVG  # Reset after aggregation
        assert env.agent_selection == 'fvg_agent'  # Back to first agent
    
    def test_matrix_construction_shape(self, env):
        """Test matrix construction has correct shape"""
        observations = env.reset()
        
        for agent in env.possible_agents:
            assert agent in observations
            obs = observations[agent]
            assert obs.shape == (60, 7)
            assert obs.dtype == np.float32
    
    def test_matrix_construction_features(self, env):
        """Test matrix construction with specific features"""
        observations = env.reset()
        
        # Test that observations contain valid data
        for agent in env.possible_agents:
            obs = observations[agent]
            
            # Check for reasonable value ranges
            assert np.all(np.isfinite(obs))
            assert not np.all(obs == 0)  # Should have some non-zero values
    
    def test_attention_weights_application(self, env):
        """Test that agent-specific attention weights are applied"""
        observations = env.reset()
        
        # Get observations for different agents
        fvg_obs = observations['fvg_agent']
        momentum_obs = observations['momentum_agent']
        
        # They should be different due to attention weights
        assert not np.allclose(fvg_obs, momentum_obs)
    
    def test_episode_termination(self, env):
        """Test episode termination conditions"""
        observations = env.reset()
        
        # Run until episode terminates
        steps = 0
        max_steps = 100
        done = False
        
        while not done and steps < max_steps:
            current_agent = env.agent_selection
            obs, rewards, dones, infos = env.step(1)  # Neutral action
            done = all(dones.values()) if dones else False
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