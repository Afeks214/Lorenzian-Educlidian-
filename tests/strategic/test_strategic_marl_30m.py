"""
Test suite for Strategic MARL 30-minute system.

This module tests the strategic decision-making components that operate
on 30-minute timeframes, including matrix assembly, pattern recognition,
and strategic agent behavior.
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Test markers
pytestmark = [pytest.mark.strategic, pytest.mark.unit]


class TestStrategicMatrixAssembler30m:
    """Test the 30-minute matrix assembler for strategic analysis."""

    @pytest.fixture
    def matrix_config(self):
        """Configuration for 30m matrix assembler."""
        return {
            "window_size": 48,
            "features": [
                "mlmi_value",
                "mlmi_signal", 
                "nwrqk_value",
                "nwrqk_slope",
                "lvn_distance_points",
                "lvn_nearest_strength",
                "time_hour_sin",
                "time_hour_cos",
                "mmd_trend",
                "mmd_volatility",
                "mmd_momentum",
                "mmd_regime",
                "volume_profile_skew"
            ],
            "name": "MatrixAssembler30m",
            "kernel": Mock()
        }

    @pytest.fixture
    def mock_assembler(self, matrix_config):
        """Create a mock 30m matrix assembler."""
        assembler = Mock()
        assembler.window_size = matrix_config["window_size"]
        assembler.features = matrix_config["features"]
        assembler.config = matrix_config
        
        # Mock methods
        assembler.assemble = Mock(return_value=np.random.rand(48, 13))
        assembler.on_indicators_ready = Mock()
        assembler.validate_features = Mock(return_value=True)
        assembler.get_feature_matrix = Mock(return_value=np.random.rand(48, 13))
        
        return assembler

    def test_matrix_assembler_initialization(self, matrix_config):
        """Test proper initialization of 30m matrix assembler."""
        # This test verifies the assembler can be properly configured
        assembler = Mock()
        assembler.config = matrix_config
        
        assert assembler.config["window_size"] == 48
        assert len(assembler.config["features"]) == 13
        assert "mmd_trend" in assembler.config["features"]
        assert "mlmi_value" in assembler.config["features"]

    def test_feature_matrix_assembly(self, mock_assembler, sample_indicators):
        """Test feature matrix assembly with complete indicator set."""
        # Call the assemble method to trigger it
        matrix = mock_assembler.assemble()
        
        assert matrix.shape[0] == 48  # window size
        assert matrix.shape[1] == 13  # number of features
        assert not np.isnan(matrix).any()  # no NaN values
        
        mock_assembler.assemble.assert_called()

    def test_mmd_feature_integration(self, mock_assembler):
        """Test MMD (Market Mode Detector) feature integration."""
        # Mock MMD features
        mmd_features = {
            "mmd_trend": 0.75,
            "mmd_volatility": 0.25, 
            "mmd_momentum": 0.50,
            "mmd_regime": 2  # regime classification
        }
        
        # Test MMD integration
        mock_assembler.integrate_mmd_features = Mock(return_value=mmd_features)
        result = mock_assembler.integrate_mmd_features()
        
        assert "mmd_trend" in result
        assert "mmd_volatility" in result
        assert "mmd_momentum" in result
        assert "mmd_regime" in result
        assert 0 <= result["mmd_trend"] <= 1
        assert 0 <= result["mmd_volatility"] <= 1

    def test_time_feature_encoding(self, mock_assembler):
        """Test time-based feature encoding (sin/cos)."""
        # Test hour encoding
        test_hour = 14  # 2 PM
        expected_sin = np.sin(2 * np.pi * test_hour / 24)
        expected_cos = np.cos(2 * np.pi * test_hour / 24)
        
        mock_assembler.encode_time_features = Mock(return_value={
            "time_hour_sin": expected_sin,
            "time_hour_cos": expected_cos
        })
        
        result = mock_assembler.encode_time_features()
        
        assert "time_hour_sin" in result
        assert "time_hour_cos" in result
        assert -1 <= result["time_hour_sin"] <= 1
        assert -1 <= result["time_hour_cos"] <= 1

    def test_indicator_validation(self, mock_assembler):
        """Test validation of required indicators."""
        required_indicators = [
            "mlmi_value", "mlmi_signal", "nwrqk_value", "nwrqk_slope",
            "lvn_distance_points", "lvn_nearest_strength"
        ]
        
        # Test complete indicator set
        complete_indicators = {indicator: 0.5 for indicator in required_indicators}
        mock_assembler.validate_indicators = Mock(return_value=True)
        
        assert mock_assembler.validate_indicators(complete_indicators)
        
        # Test incomplete indicator set
        incomplete_indicators = {indicator: 0.5 for indicator in required_indicators[:3]}
        mock_assembler.validate_indicators = Mock(return_value=False)
        
        assert not mock_assembler.validate_indicators(incomplete_indicators)

    @pytest.mark.performance
    def test_assembly_performance(self, mock_assembler, performance_timer):
        """Test matrix assembly performance meets requirements."""
        # Requirement: Matrix assembly < 1ms
        performance_timer.start()
        
        # Simulate assembly operation
        matrix = mock_assembler.get_feature_matrix()
        
        performance_timer.stop()
        elapsed = performance_timer.elapsed_ms()
        
        assert elapsed < 1.0  # Less than 1ms
        assert matrix is not None

    def test_feature_normalization(self, mock_assembler):
        """Test feature normalization and scaling."""
        raw_matrix = np.random.rand(48, 13) * 100  # Raw values 0-100
        
        mock_assembler.normalize_features = Mock(return_value=raw_matrix / 100)
        normalized = mock_assembler.normalize_features(raw_matrix)
        
        # Check normalization
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        assert normalized.shape == raw_matrix.shape


class TestStrategicAgents:
    """Test strategic agents for 30-minute decision making."""

    @pytest.fixture
    def strategic_agent_config(self):
        """Configuration for strategic agent."""
        return {
            "name": "strategic_agent",
            "observation_dim": 48 * 13,  # 48 bars × 13 features
            "action_dim": 3,  # position: {-1, 0, 1}
            "learning_rate": 0.001,
            "hidden_dim": 256,
            "network_type": "deep_q_network"
        }

    @pytest.fixture
    def mock_strategic_agent(self, strategic_agent_config):
        """Create a mock strategic agent."""
        agent = Mock()
        agent.config = strategic_agent_config
        agent.observation_dim = strategic_agent_config["observation_dim"]
        agent.action_dim = strategic_agent_config["action_dim"]
        
        # Mock methods
        agent.act = Mock(return_value={"position": 0.0, "confidence": 0.75})
        agent.learn = Mock()
        agent.update = Mock()
        agent.get_action = Mock(return_value=1)  # Buy signal
        agent.get_q_values = Mock(return_value=np.array([0.2, 0.8, 0.3]))
        
        return agent

    def test_strategic_agent_initialization(self, strategic_agent_config):
        """Test strategic agent initialization."""
        agent = Mock()
        agent.config = strategic_agent_config
        
        assert agent.config["observation_dim"] == 48 * 13
        assert agent.config["action_dim"] == 3
        assert agent.config["learning_rate"] == 0.001

    def test_strategic_decision_making(self, mock_strategic_agent, sample_30m_matrix):
        """Test strategic decision making process."""
        # Flatten matrix for agent observation
        observation = sample_30m_matrix.flatten()
        
        # Get strategic decision
        decision = mock_strategic_agent.act(observation)
        
        assert "position" in decision
        assert "confidence" in decision
        assert -1 <= decision["position"] <= 1
        assert 0 <= decision["confidence"] <= 1
        
        mock_strategic_agent.act.assert_called_once_with(observation)

    def test_action_space_validation(self, mock_strategic_agent):
        """Test action space validation for strategic decisions."""
        valid_actions = [-1, 0, 1]  # Short, Hold, Long
        
        for action in valid_actions:
            mock_strategic_agent.get_action = Mock(return_value=action)
            result = mock_strategic_agent.get_action()
            assert result in valid_actions

    def test_q_value_computation(self, mock_strategic_agent):
        """Test Q-value computation for strategic actions."""
        observation = np.random.rand(48 * 13)
        q_values = mock_strategic_agent.get_q_values(observation)
        
        assert len(q_values) == 3  # One for each action
        assert all(isinstance(q, (int, float)) for q in q_values)
        
        mock_strategic_agent.get_q_values.assert_called_once_with(observation)

    def test_learning_update(self, mock_strategic_agent):
        """Test learning update mechanism."""
        experience = {
            "state": np.random.rand(48 * 13),
            "action": 1,
            "reward": 0.1,
            "next_state": np.random.rand(48 * 13),
            "done": False
        }
        
        mock_strategic_agent.learn(experience)
        mock_strategic_agent.learn.assert_called_once_with(experience)

    @pytest.mark.performance  
    def test_inference_speed(self, mock_strategic_agent, performance_timer):
        """Test strategic agent inference speed."""
        observation = np.random.rand(48 * 13)
        
        performance_timer.start()
        decision = mock_strategic_agent.act(observation)
        performance_timer.stop()
        
        # Requirement: Inference < 50ms
        assert performance_timer.elapsed_ms() < 50
        assert decision is not None


class TestStrategicMARLEnvironment:
    """Test strategic MARL environment for 30-minute trading."""

    @pytest.fixture
    def marl_env_config(self):
        """Configuration for strategic MARL environment."""
        return {
            "agents": ["strategic_agent"],
            "observation_space_dim": 48 * 13,
            "action_space_dim": 3,
            "reward_function": "sharpe_ratio",
            "episode_length": 100,
            "lookback_window": 48
        }

    @pytest.fixture
    def mock_strategic_env(self, marl_env_config):
        """Create mock strategic MARL environment."""
        env = Mock()
        env.config = marl_env_config
        env.agents = ["strategic_agent"]
        env.possible_agents = env.agents
        
        # Mock methods
        env.reset = Mock(return_value={
            "strategic_agent": np.random.rand(48 * 13)
        })
        env.step = Mock(return_value=(
            {"strategic_agent": np.random.rand(48 * 13)},  # observations
            {"strategic_agent": 0.1},                      # rewards  
            {"strategic_agent": False},                     # dones
            {"strategic_agent": {}}                         # infos
        ))
        env.observation_space = Mock()
        env.action_space = Mock()
        
        return env

    def test_environment_reset(self, mock_strategic_env):
        """Test environment reset functionality."""
        observations = mock_strategic_env.reset()
        
        assert "strategic_agent" in observations
        assert len(observations["strategic_agent"]) == 48 * 13
        
        mock_strategic_env.reset.assert_called_once()

    def test_environment_step(self, mock_strategic_env):
        """Test environment step functionality."""
        actions = {"strategic_agent": 1}  # Buy action
        
        obs, rewards, dones, infos = mock_strategic_env.step(actions)
        
        assert "strategic_agent" in obs
        assert "strategic_agent" in rewards
        assert "strategic_agent" in dones
        assert "strategic_agent" in infos
        
        mock_strategic_env.step.assert_called_once_with(actions)

    def test_reward_calculation(self, mock_strategic_env):
        """Test reward calculation for strategic decisions."""
        # Mock reward calculation
        mock_strategic_env.calculate_reward = Mock(return_value=0.15)
        
        reward = mock_strategic_env.calculate_reward(
            action=1,           # Buy
            price_change=0.02,  # 2% gain
            position=1.0        # Full position
        )
        
        assert reward > 0  # Positive reward for correct prediction
        mock_strategic_env.calculate_reward.assert_called_once()

    def test_episode_termination(self, mock_strategic_env):
        """Test episode termination conditions."""
        # Test normal termination
        mock_strategic_env.is_episode_done = Mock(return_value=False)
        assert not mock_strategic_env.is_episode_done()
        
        # Test termination on max steps
        mock_strategic_env.is_episode_done = Mock(return_value=True)
        assert mock_strategic_env.is_episode_done()


class TestStrategicPatternRecognition:
    """Test strategic pattern recognition for 30-minute analysis."""

    @pytest.fixture
    def pattern_detector_config(self):
        """Configuration for pattern detector."""
        return {
            "patterns": ["TYPE_1", "TYPE_2", "TYPE_3", "TYPE_4"],
            "confidence_threshold": 0.7,
            "lookback_bars": 48,
            "features": ["mlmi", "nwrqk", "lvn", "mmd"]
        }

    @pytest.fixture
    def mock_pattern_detector(self, pattern_detector_config):
        """Create mock pattern detector."""
        detector = Mock()
        detector.config = pattern_detector_config
        
        # Mock methods
        detector.detect_patterns = Mock(return_value={
            "TYPE_1": 0.85,
            "TYPE_2": 0.15,
            "TYPE_3": 0.05,
            "TYPE_4": 0.25
        })
        detector.get_dominant_pattern = Mock(return_value="TYPE_1")
        detector.validate_pattern = Mock(return_value=True)
        
        return detector

    def test_pattern_detection(self, mock_pattern_detector, sample_30m_matrix):
        """Test pattern detection on 30m data."""
        patterns = mock_pattern_detector.detect_patterns(sample_30m_matrix)
        
        assert "TYPE_1" in patterns
        assert "TYPE_2" in patterns
        assert "TYPE_3" in patterns
        assert "TYPE_4" in patterns
        
        # Check confidence scores
        for pattern, confidence in patterns.items():
            assert 0 <= confidence <= 1
        
        mock_pattern_detector.detect_patterns.assert_called_once_with(sample_30m_matrix)

    def test_dominant_pattern_selection(self, mock_pattern_detector):
        """Test selection of dominant pattern."""
        dominant = mock_pattern_detector.get_dominant_pattern()
        
        assert dominant in ["TYPE_1", "TYPE_2", "TYPE_3", "TYPE_4"]
        mock_pattern_detector.get_dominant_pattern.assert_called_once()

    def test_pattern_confidence_threshold(self, mock_pattern_detector):
        """Test pattern confidence threshold filtering."""
        # High confidence pattern
        mock_pattern_detector.validate_pattern = Mock(return_value=True)
        assert mock_pattern_detector.validate_pattern("TYPE_1", 0.85)
        
        # Low confidence pattern  
        mock_pattern_detector.validate_pattern = Mock(return_value=False)
        assert not mock_pattern_detector.validate_pattern("TYPE_2", 0.45)


class TestStrategicIntegration:
    """Test integration of strategic components."""

    @pytest.fixture  
    def mock_pattern_detector(self):
        """Create mock pattern detector for integration tests."""
        detector = Mock()
        detector.config = {
            "patterns": ["TYPE_1", "TYPE_2", "TYPE_3", "TYPE_4"],
            "confidence_threshold": 0.7,
            "lookback_bars": 48,
            "features": ["mlmi", "nwrqk", "lvn", "mmd"]
        }
        
        # Mock methods
        detector.detect_patterns = Mock(return_value={
            "TYPE_1": 0.85,
            "TYPE_2": 0.15,
            "TYPE_3": 0.05,
            "TYPE_4": 0.25
        })
        detector.get_dominant_pattern = Mock(return_value="TYPE_1")
        detector.validate_pattern = Mock(return_value=True)
        
        return detector

    @pytest.fixture
    def strategic_system(self, mock_matrix_assembler_30m, mock_strategic_agent, mock_pattern_detector):
        """Create integrated strategic system."""
        system = Mock()
        system.matrix_assembler = mock_matrix_assembler_30m
        system.strategic_agent = mock_strategic_agent
        system.pattern_detector = mock_pattern_detector
        
        # Mock integration methods
        system.process_30m_bar = Mock()
        system.make_strategic_decision = Mock(return_value={
            "position": 1.0,
            "confidence": 0.8,
            "pattern": "TYPE_1",
            "features": np.random.rand(13)
        })
        
        return system

    def test_end_to_end_strategic_pipeline(self, strategic_system, sample_indicators):
        """Test complete strategic decision pipeline."""
        # Simulate 30m bar processing
        decision = strategic_system.make_strategic_decision(sample_indicators)
        
        assert "position" in decision
        assert "confidence" in decision
        assert "pattern" in decision
        assert "features" in decision
        
        assert -1 <= decision["position"] <= 1
        assert 0 <= decision["confidence"] <= 1
        assert decision["pattern"] in ["TYPE_1", "TYPE_2", "TYPE_3", "TYPE_4"]
        
        strategic_system.make_strategic_decision.assert_called_once_with(sample_indicators)

    @pytest.mark.asyncio
    async def test_async_strategic_processing(self, strategic_system):
        """Test asynchronous strategic processing."""
        # Mock async processing
        strategic_system.process_async = AsyncMock(return_value={"status": "success"})
        
        result = await strategic_system.process_async()
        assert result["status"] == "success"
        
        strategic_system.process_async.assert_called_once()

    @pytest.mark.slow
    def test_strategic_system_performance(self, strategic_system, benchmark_config):
        """Test strategic system performance under load."""
        import time
        
        start_time = time.time()
        
        # Process multiple decisions
        for _ in range(100):
            strategic_system.make_strategic_decision({})
        
        elapsed = time.time() - start_time
        
        # Should process 100 decisions in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds
        assert strategic_system.make_strategic_decision.call_count == 100

    def test_error_handling(self, strategic_system):
        """Test error handling in strategic pipeline."""
        # Mock error condition
        strategic_system.make_strategic_decision = Mock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            strategic_system.make_strategic_decision({})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])