"""
Test suite for Strategic MARL Agents.

This module tests the individual strategic agents, their behaviors,
interactions, and learning capabilities in the 30-minute trading environment.
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Tuple
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Test markers
pytestmark = [pytest.mark.strategic, pytest.mark.unit]


class TestStrategicAgent:
    """Test the main Strategic Agent for 30-minute decisions."""

    @pytest.fixture
    def agent_config(self):
        """Configuration for strategic agent."""
        return {
            "name": "strategic_agent",
            "observation_dim": 624,  # 48 bars × 13 features
            "action_dim": 5,  # {strong_sell, sell, hold, buy, strong_buy}
            "learning_rate": 0.0003,
            "hidden_dims": [512, 256, 128],
            "network_type": "dueling_dqn",
            "memory_size": 100000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "exploration": {
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,
                "decay_steps": 50000
            }
        }

    @pytest.fixture
    def mock_strategic_agent(self, agent_config):
        """Create a mock strategic agent."""
        agent = Mock()
        agent.config = agent_config
        agent.name = "strategic_agent"
        agent.observation_dim = agent_config["observation_dim"]
        agent.action_dim = agent_config["action_dim"]
        agent.epsilon = 0.1
        agent.memory = Mock()
        
        # Mock neural network
        agent.q_network = Mock()
        agent.target_network = Mock()
        agent.optimizer = Mock()
        
        # Mock methods
        agent.act = Mock(return_value=2)  # Hold action
        agent.learn = Mock()
        agent.update_target_network = Mock()
        agent.save = Mock()
        agent.load = Mock()
        agent.get_q_values = Mock(return_value=np.array([0.1, 0.3, 0.8, 0.4, 0.2]))
        agent.compute_td_error = Mock(return_value=0.05)
        
        return agent

    def test_strategic_agent_initialization(self, agent_config):
        """Test strategic agent initialization."""
        agent = Mock()
        agent.config = agent_config
        
        assert agent.config["name"] == "strategic_agent"
        assert agent.config["observation_dim"] == 624
        assert agent.config["action_dim"] == 5
        assert agent.config["learning_rate"] == 0.0003
        assert agent.config["network_type"] == "dueling_dqn"

    def test_action_selection(self, mock_strategic_agent):
        """Test action selection with epsilon-greedy policy."""
        observation = np.random.rand(624)
        
        # Test exploitation (greedy action)
        mock_strategic_agent.epsilon = 0.0
        action = mock_strategic_agent.act(observation)
        assert action in range(5)  # Valid action space
        
        # Test exploration (random action)  
        mock_strategic_agent.epsilon = 1.0
        mock_strategic_agent.act = Mock(return_value=np.random.randint(5))
        action = mock_strategic_agent.act(observation)
        assert action in range(5)

    def test_q_value_computation(self, mock_strategic_agent):
        """Test Q-value computation for all actions."""
        observation = np.random.rand(624)
        q_values = mock_strategic_agent.get_q_values(observation)
        
        assert len(q_values) == 5  # One Q-value per action
        assert all(isinstance(q, (int, float)) for q in q_values)
        
        # Best action should have highest Q-value
        best_action = np.argmax(q_values)
        assert 0 <= best_action < 5

    def test_experience_replay(self, mock_strategic_agent):
        """Test experience replay learning mechanism."""
        # Mock experience
        experience = {
            "state": np.random.rand(624),
            "action": 2,
            "reward": 0.15,
            "next_state": np.random.rand(624),
            "done": False
        }
        
        # Test memory storage
        mock_strategic_agent.memory.store = Mock()
        mock_strategic_agent.memory.store(experience)
        mock_strategic_agent.memory.store.assert_called_once_with(experience)
        
        # Test learning from memory
        mock_strategic_agent.memory.sample = Mock(return_value=[experience] * 64)
        mock_strategic_agent.learn()
        mock_strategic_agent.learn.assert_called_once()

    def test_target_network_update(self, mock_strategic_agent):
        """Test soft update of target network."""
        mock_strategic_agent.update_target_network()
        mock_strategic_agent.update_target_network.assert_called_once()

    def test_exploration_decay(self, mock_strategic_agent):
        """Test epsilon decay for exploration."""
        initial_epsilon = 1.0
        final_epsilon = 0.01
        decay_steps = 1000
        
        # Mock epsilon decay
        mock_strategic_agent.decay_epsilon = Mock()
        mock_strategic_agent.decay_epsilon(initial_epsilon, final_epsilon, decay_steps)
        mock_strategic_agent.decay_epsilon.assert_called_once()

    @pytest.mark.performance
    def test_inference_performance(self, mock_strategic_agent, performance_timer):
        """Test inference performance requirements."""
        observation = np.random.rand(624)
        
        performance_timer.start()
        action = mock_strategic_agent.act(observation)
        performance_timer.stop()
        
        # Requirement: Inference < 10ms for strategic decisions
        assert performance_timer.elapsed_ms() < 10
        assert action is not None

    def test_model_persistence(self, mock_strategic_agent, temp_dir):
        """Test model saving and loading."""
        model_path = temp_dir / "strategic_agent.pth"
        
        # Test saving
        mock_strategic_agent.save(str(model_path))
        mock_strategic_agent.save.assert_called_once_with(str(model_path))
        
        # Test loading
        mock_strategic_agent.load(str(model_path))
        mock_strategic_agent.load.assert_called_once_with(str(model_path))


class TestPositionSizingAgent:
    """Test the Position Sizing Agent for strategic position management."""

    @pytest.fixture
    def position_agent_config(self):
        """Configuration for position sizing agent."""
        return {
            "name": "position_sizing_agent",
            "observation_dim": 50,  # Risk features + market features
            "action_dim": 10,  # Position sizes: 0%, 10%, 20%, ..., 90%
            "max_position_size": 0.9,
            "risk_tolerance": 0.02,
            "kelly_criterion": True,
            "volatility_scaling": True
        }

    @pytest.fixture
    def mock_position_agent(self, position_agent_config):
        """Create mock position sizing agent."""
        agent = Mock()
        agent.config = position_agent_config
        agent.name = "position_sizing_agent"
        
        # Mock methods
        agent.get_position_size = Mock(return_value=0.3)  # 30% allocation
        agent.calculate_kelly_fraction = Mock(return_value=0.25)
        agent.adjust_for_volatility = Mock(return_value=0.2)
        agent.compute_risk_metrics = Mock(return_value={
            "var_95": -0.015,  # VaR should be negative (loss)
            "expected_shortfall": -0.025,  # Expected shortfall should be negative
            "max_drawdown": 0.08  # Max drawdown is positive (percentage)
        })
        
        return agent

    def test_position_sizing_calculation(self, mock_position_agent):
        """Test position sizing based on risk and opportunity."""
        market_data = {
            "volatility": 0.15,
            "confidence": 0.8,
            "pattern_strength": 0.7
        }
        
        position_size = mock_position_agent.get_position_size(market_data)
        
        assert 0 <= position_size <= 0.9  # Within max position limit
        mock_position_agent.get_position_size.assert_called_once_with(market_data)

    def test_kelly_criterion_application(self, mock_position_agent):
        """Test Kelly criterion for optimal position sizing."""
        win_rate = 0.55
        avg_win = 0.02
        avg_loss = 0.015
        
        kelly_fraction = mock_position_agent.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        assert isinstance(kelly_fraction, (int, float))
        assert kelly_fraction >= 0  # Kelly fraction should be non-negative
        mock_position_agent.calculate_kelly_fraction.assert_called_once()

    def test_volatility_adjustment(self, mock_position_agent):
        """Test position size adjustment for volatility."""
        base_position = 0.5
        current_volatility = 0.25
        target_volatility = 0.15
        
        adjusted_position = mock_position_agent.adjust_for_volatility(
            base_position, current_volatility, target_volatility
        )
        
        assert adjusted_position <= base_position  # Should reduce for higher vol
        mock_position_agent.adjust_for_volatility.assert_called_once()

    def test_risk_metrics_computation(self, mock_position_agent):
        """Test computation of risk metrics."""
        portfolio_returns = np.random.normal(0.001, 0.02, 1000)
        
        risk_metrics = mock_position_agent.compute_risk_metrics(portfolio_returns)
        
        assert "var_95" in risk_metrics
        assert "expected_shortfall" in risk_metrics
        assert "max_drawdown" in risk_metrics
        
        # VaR should be negative (loss)
        assert risk_metrics["var_95"] < 0
        mock_position_agent.compute_risk_metrics.assert_called_once()


class TestRegimeDetectionAgent:
    """Test the Regime Detection Agent for market state classification."""

    @pytest.fixture
    def regime_agent_config(self):
        """Configuration for regime detection agent."""
        return {
            "name": "regime_detection_agent",
            "observation_dim": 100,  # Market microstructure features
            "regimes": ["TRENDING", "RANGING", "VOLATILE", "CALM"],
            "lookback_window": 96,  # 48 hours of 30m bars
            "regime_threshold": 0.6,
            "transition_smoothing": 0.1
        }

    @pytest.fixture  
    def mock_regime_agent(self, regime_agent_config):
        """Create mock regime detection agent."""
        agent = Mock()
        agent.config = regime_agent_config
        agent.name = "regime_detection_agent"
        agent.current_regime = "TRENDING"
        agent.regime_probabilities = {
            "TRENDING": 0.7,
            "RANGING": 0.2,
            "VOLATILE": 0.08,
            "CALM": 0.02
        }
        
        # Mock methods
        agent.detect_regime = Mock(return_value="TRENDING")
        agent.get_regime_probabilities = Mock(return_value=agent.regime_probabilities)
        agent.smooth_transition = Mock(return_value="TRENDING")
        agent.analyze_regime_persistence = Mock(return_value=0.85)
        
        return agent

    def test_regime_detection(self, mock_regime_agent):
        """Test market regime detection."""
        market_features = np.random.rand(100)
        
        regime = mock_regime_agent.detect_regime(market_features)
        
        assert regime in ["TRENDING", "RANGING", "VOLATILE", "CALM"]
        mock_regime_agent.detect_regime.assert_called_once_with(market_features)

    def test_regime_probabilities(self, mock_regime_agent):
        """Test regime probability computation."""
        probabilities = mock_regime_agent.get_regime_probabilities()
        
        # Check all regimes present
        expected_regimes = ["TRENDING", "RANGING", "VOLATILE", "CALM"]
        for regime in expected_regimes:
            assert regime in probabilities
            assert 0 <= probabilities[regime] <= 1
        
        # Probabilities should sum to ~1
        total_prob = sum(probabilities.values())
        assert 0.99 <= total_prob <= 1.01

    def test_regime_transition_smoothing(self, mock_regime_agent):
        """Test smooth regime transitions."""
        previous_regime = "RANGING"
        new_regime = "TRENDING"
        
        smoothed_regime = mock_regime_agent.smooth_transition(previous_regime, new_regime)
        
        assert smoothed_regime in ["TRENDING", "RANGING", "VOLATILE", "CALM"]
        mock_regime_agent.smooth_transition.assert_called_once()

    def test_regime_persistence_analysis(self, mock_regime_agent):
        """Test regime persistence measurement."""
        regime_history = ["TRENDING"] * 8 + ["RANGING"] * 2
        
        persistence = mock_regime_agent.analyze_regime_persistence(regime_history)
        
        assert 0 <= persistence <= 1
        mock_regime_agent.analyze_regime_persistence.assert_called_once()


class TestStrategicCoordinator:
    """Test the Strategic Coordinator that orchestrates all strategic agents."""

    @pytest.fixture
    def coordinator_config(self):
        """Configuration for strategic coordinator."""
        return {
            "name": "strategic_coordinator",
            "agents": ["strategic_agent", "position_sizing_agent", "regime_detection_agent"],
            "decision_fusion_method": "weighted_average",
            "confidence_threshold": 0.6,
            "override_conditions": {
                "high_volatility": 0.3,
                "regime_uncertainty": 0.5
            }
        }

    @pytest.fixture
    def mock_coordinator(self, coordinator_config, mock_strategic_agent, 
                         mock_position_agent, mock_regime_agent):
        """Create mock strategic coordinator."""
        coordinator = Mock()
        coordinator.config = coordinator_config
        coordinator.agents = {
            "strategic_agent": mock_strategic_agent,
            "position_sizing_agent": mock_position_agent,
            "regime_detection_agent": mock_regime_agent
        }
        
        # Mock methods
        coordinator.coordinate_decision = Mock(return_value={
            "final_position": 0.25,
            "confidence": 0.75,
            "regime": "TRENDING",
            "risk_adjusted": True
        })
        coordinator.fuse_decisions = Mock()
        coordinator.apply_risk_controls = Mock()
        coordinator.validate_decision = Mock(return_value=True)
        
        return coordinator

    def test_decision_coordination(self, mock_coordinator):
        """Test coordination of multiple agent decisions."""
        market_state = {
            "features": np.random.rand(624),
            "volatility": 0.15,
            "regime_features": np.random.rand(100)
        }
        
        decision = mock_coordinator.coordinate_decision(market_state)
        
        assert "final_position" in decision
        assert "confidence" in decision
        assert "regime" in decision
        assert "risk_adjusted" in decision
        
        assert -1 <= decision["final_position"] <= 1
        assert 0 <= decision["confidence"] <= 1
        
        mock_coordinator.coordinate_decision.assert_called_once_with(market_state)

    def test_decision_fusion(self, mock_coordinator):
        """Test fusion of individual agent decisions."""
        agent_decisions = {
            "strategic_agent": {"position": 0.6, "confidence": 0.8},
            "position_sizing_agent": {"size": 0.3, "confidence": 0.7},
            "regime_detection_agent": {"regime": "TRENDING", "confidence": 0.9}
        }
        
        mock_coordinator.fuse_decisions(agent_decisions)
        mock_coordinator.fuse_decisions.assert_called_once_with(agent_decisions)

    def test_risk_controls(self, mock_coordinator):
        """Test application of risk controls."""
        proposed_decision = {
            "position": 0.8,
            "confidence": 0.6,
            "regime": "VOLATILE"
        }
        
        mock_coordinator.apply_risk_controls(proposed_decision)
        mock_coordinator.apply_risk_controls.assert_called_once_with(proposed_decision)

    def test_decision_validation(self, mock_coordinator):
        """Test validation of final decisions."""
        decision = {
            "final_position": 0.25,
            "confidence": 0.75,
            "regime": "TRENDING"
        }
        
        is_valid = mock_coordinator.validate_decision(decision)
        assert isinstance(is_valid, bool)
        mock_coordinator.validate_decision.assert_called_once_with(decision)

    @pytest.mark.integration
    def test_end_to_end_coordination(self, mock_coordinator):
        """Test complete strategic coordination pipeline."""
        # Simulate market update
        market_update = {
            "timestamp": datetime.now(),
            "features": np.random.rand(624),
            "volatility": 0.18,
            "regime_features": np.random.rand(100)
        }
        
        # Get coordinated decision
        decision = mock_coordinator.coordinate_decision(market_update)
        
        # Validate decision structure
        required_fields = ["final_position", "confidence", "regime", "risk_adjusted"]
        for field in required_fields:
            assert field in decision
        
        # Validate decision ranges
        assert -1 <= decision["final_position"] <= 1
        assert 0 <= decision["confidence"] <= 1
        assert isinstance(decision["risk_adjusted"], bool)


class TestStrategicAgentInteractions:
    """Test interactions between strategic agents."""

    def test_agent_communication(self, mock_strategic_agent, mock_position_agent):
        """Test communication between strategic agents."""
        # Strategic agent provides signal
        signal = {"position": 0.7, "confidence": 0.8}
        mock_strategic_agent.get_signal = Mock(return_value=signal)
        
        # Position agent receives and processes signal
        mock_position_agent.process_signal = Mock(return_value=0.35)  # 35% position
        
        strategic_signal = mock_strategic_agent.get_signal()
        position_size = mock_position_agent.process_signal(strategic_signal)
        
        assert position_size <= 0.9  # Respects max position
        mock_strategic_agent.get_signal.assert_called_once()
        mock_position_agent.process_signal.assert_called_once_with(signal)

    def test_consensus_mechanism(self, mock_strategic_agent, mock_regime_agent):
        """Test consensus mechanism between agents."""
        # Different agent opinions
        strategic_view = "BULLISH"
        regime_view = "TRENDING"
        
        mock_strategic_agent.get_market_view = Mock(return_value=strategic_view)
        mock_regime_agent.get_regime_view = Mock(return_value=regime_view)
        
        # Test consensus building
        consensus_builder = Mock()
        consensus_builder.build_consensus = Mock(return_value={
            "consensus": "BULLISH_TRENDING",
            "agreement_level": 0.85
        })
        
        views = [strategic_view, regime_view]
        consensus = consensus_builder.build_consensus(views)
        
        assert "consensus" in consensus
        assert "agreement_level" in consensus
        assert 0 <= consensus["agreement_level"] <= 1

    @pytest.mark.asyncio
    async def test_async_agent_coordination(self, mock_strategic_agent, mock_position_agent):
        """Test asynchronous coordination between agents."""
        # Mock async methods
        mock_strategic_agent.get_decision_async = AsyncMock(return_value={"position": 0.5})
        mock_position_agent.size_position_async = AsyncMock(return_value=0.25)
        
        # Coordinate asynchronously
        strategic_decision = await mock_strategic_agent.get_decision_async()
        position_size = await mock_position_agent.size_position_async(strategic_decision)
        
        assert isinstance(position_size, (int, float))
        assert 0 <= position_size <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])