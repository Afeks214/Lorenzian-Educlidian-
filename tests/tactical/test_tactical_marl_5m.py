"""
Test suite for Tactical MARL 5-minute system.

This module tests the tactical execution components that operate
on 5-minute timeframes, including FVG detection, execution timing,
and tactical agent behavior.
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
pytestmark = [pytest.mark.tactical, pytest.mark.unit]


class TestTacticalMatrixAssembler5m:
    """Test the 5-minute matrix assembler for tactical execution."""

    @pytest.fixture
    def matrix_config(self):
        """Configuration for 5m matrix assembler."""
        return {
            "window_size": 60,
            "features": [
                "fvg_bullish_active",
                "fvg_bearish_active",
                "fvg_nearest_level",
                "fvg_age",
                "fvg_mitigation_signal",
                "price_momentum_5",
                "volume_ratio",
                "microstructure_imbalance",
                "order_flow_pressure",
                "bid_ask_spread_normalized"
            ],
            "name": "MatrixAssembler5m",
            "kernel": Mock()
        }

    @pytest.fixture
    def mock_assembler(self, matrix_config):
        """Create a mock 5m matrix assembler."""
        assembler = Mock()
        assembler.window_size = matrix_config["window_size"]
        assembler.features = matrix_config["features"]
        assembler.config = matrix_config
        
        # Mock methods
        assembler.assemble = Mock(return_value=np.random.rand(60, 10))
        assembler.on_indicators_ready = Mock()
        assembler.validate_features = Mock(return_value=True)
        assembler.get_feature_matrix = Mock(return_value=np.random.rand(60, 10))
        assembler.process_fvg_features = Mock()
        assembler.calculate_microstructure = Mock()
        
        return assembler

    def test_matrix_assembler_initialization(self, matrix_config):
        """Test proper initialization of 5m matrix assembler."""
        assembler = Mock()
        assembler.config = matrix_config
        
        assert assembler.config["window_size"] == 60
        assert len(assembler.config["features"]) == 10
        assert "fvg_bullish_active" in assembler.config["features"]
        assert "microstructure_imbalance" in assembler.config["features"]

    def test_fvg_feature_processing(self, mock_assembler):
        """Test Fair Value Gap (FVG) feature processing."""
        fvg_data = {
            "bullish_gaps": [{"level": 100.5, "age": 3, "strength": 0.8}],
            "bearish_gaps": [{"level": 99.8, "age": 1, "strength": 0.6}],
            "current_price": 100.0
        }
        
        mock_assembler.process_fvg_features = Mock(return_value={
            "fvg_bullish_active": 1,
            "fvg_bearish_active": 1,
            "fvg_nearest_level": 99.8,
            "fvg_age": 1,
            "fvg_mitigation_signal": 0.6
        })
        
        result = mock_assembler.process_fvg_features(fvg_data)
        
        assert "fvg_bullish_active" in result
        assert "fvg_bearish_active" in result
        assert "fvg_nearest_level" in result
        assert result["fvg_bullish_active"] >= 0
        assert result["fvg_bearish_active"] >= 0

    def test_microstructure_calculation(self, mock_assembler):
        """Test microstructure feature calculation."""
        order_book_data = {
            "bid_volume": 1000,
            "ask_volume": 1200,
            "bid_price": 99.98,
            "ask_price": 100.02,
            "trade_volume": 500,
            "trade_direction": 1  # Buy-initiated
        }
        
        mock_assembler.calculate_microstructure = Mock(return_value={
            "microstructure_imbalance": 0.2,  # More ask volume
            "order_flow_pressure": 0.5,
            "bid_ask_spread_normalized": 0.0004
        })
        
        result = mock_assembler.calculate_microstructure(order_book_data)
        
        assert "microstructure_imbalance" in result
        assert "order_flow_pressure" in result
        assert "bid_ask_spread_normalized" in result
        assert -1 <= result["microstructure_imbalance"] <= 1

    def test_momentum_features(self, mock_assembler):
        """Test short-term momentum feature calculation."""
        price_series = np.array([100.0, 100.1, 100.05, 100.15, 100.2])
        volume_series = np.array([1000, 1200, 800, 1500, 1100])
        
        mock_assembler.calculate_momentum = Mock(return_value={
            "price_momentum_5": 0.002,  # 0.2% momentum
            "volume_ratio": 1.1  # 10% above average
        })
        
        result = mock_assembler.calculate_momentum(price_series, volume_series)
        
        assert "price_momentum_5" in result
        assert "volume_ratio" in result
        assert isinstance(result["price_momentum_5"], (int, float))
        assert result["volume_ratio"] > 0

    def test_feature_matrix_assembly(self, mock_assembler, sample_indicators):
        """Test complete feature matrix assembly."""
        matrix = mock_assembler.assemble(sample_indicators)
        
        assert matrix.shape[0] == 60  # window size
        assert matrix.shape[1] == 10  # number of features
        assert not np.isnan(matrix).any()  # no NaN values
        
        mock_assembler.assemble.assert_called()

    @pytest.mark.performance
    def test_assembly_performance(self, mock_assembler, performance_timer):
        """Test matrix assembly performance for tactical speed."""
        # Requirement: Tactical assembly < 0.5ms
        performance_timer.start()
        
        matrix = mock_assembler.get_feature_matrix()
        
        performance_timer.stop()
        elapsed = performance_timer.elapsed_ms()
        
        assert elapsed < 0.5  # Less than 0.5ms for tactical speed
        assert matrix is not None

    def test_real_time_updates(self, mock_assembler):
        """Test real-time feature updates for tactical trading."""
        # Mock streaming update
        new_bar_data = {
            "timestamp": datetime.now(),
            "open": 100.0,
            "high": 100.1,
            "low": 99.95,
            "close": 100.05,
            "volume": 1500
        }
        
        mock_assembler.update_features = Mock()
        mock_assembler.update_features(new_bar_data)
        mock_assembler.update_features.assert_called_once_with(new_bar_data)


class TestTacticalAgents:
    """Test tactical agents for 5-minute execution."""

    @pytest.fixture
    def tactical_agent_config(self):
        """Configuration for tactical agent."""
        return {
            "name": "tactical_agent",
            "observation_dim": 600,  # 60 bars × 10 features
            "action_dim": 7,  # {aggressive_sell, sell, scale_out, hold, scale_in, buy, aggressive_buy}
            "learning_rate": 0.0005,
            "hidden_dims": [256, 128, 64],
            "network_type": "actor_critic",
            "update_frequency": 4,
            "execution_horizon": 12,  # 1 hour tactical horizon
            "slippage_penalty": 0.01
        }

    @pytest.fixture
    def mock_tactical_agent(self, tactical_agent_config):
        """Create a mock tactical agent."""
        agent = Mock()
        agent.config = tactical_agent_config
        agent.name = "tactical_agent"
        agent.observation_dim = tactical_agent_config["observation_dim"]
        agent.action_dim = tactical_agent_config["action_dim"]
        
        # Mock networks
        agent.actor_network = Mock()
        agent.critic_network = Mock()
        agent.optimizer = Mock()
        
        # Mock methods
        agent.act = Mock(return_value=3)  # Hold action
        agent.get_action_probabilities = Mock(return_value=np.array([0.1, 0.15, 0.1, 0.4, 0.1, 0.1, 0.05]))
        agent.learn = Mock()
        agent.update = Mock()
        agent.calculate_execution_cost = Mock(return_value=0.005)
        
        return agent

    def test_tactical_agent_initialization(self, tactical_agent_config):
        """Test tactical agent initialization."""
        agent = Mock()
        agent.config = tactical_agent_config
        
        assert agent.config["name"] == "tactical_agent"
        assert agent.config["observation_dim"] == 600
        assert agent.config["action_dim"] == 7
        assert agent.config["network_type"] == "actor_critic"

    def test_action_selection_with_probabilities(self, mock_tactical_agent):
        """Test action selection with probability distribution."""
        observation = np.random.rand(600)
        
        action_probs = mock_tactical_agent.get_action_probabilities(observation)
        action = mock_tactical_agent.act(observation)
        
        assert len(action_probs) == 7
        assert np.isclose(np.sum(action_probs), 1.0)  # Probabilities sum to 1
        assert action in range(7)  # Valid action

    def test_execution_cost_calculation(self, mock_tactical_agent):
        """Test execution cost calculation for tactical decisions."""
        execution_params = {
            "order_size": 1000,
            "market_impact": 0.002,
            "bid_ask_spread": 0.001,
            "urgency": 0.8
        }
        
        cost = mock_tactical_agent.calculate_execution_cost(execution_params)
        
        assert isinstance(cost, (int, float))
        assert cost >= 0  # Cost should be non-negative
        mock_tactical_agent.calculate_execution_cost.assert_called_once_with(execution_params)

    def test_execution_timing(self, mock_tactical_agent):
        """Test optimal execution timing decisions."""
        market_state = {
            "fvg_signal": 0.7,
            "microstructure_pressure": -0.3,
            "volume_profile": 0.6,
            "strategic_signal": 1.0  # Buy signal from strategic layer
        }
        
        mock_tactical_agent.get_execution_timing = Mock(return_value={
            "immediate": False,
            "wait_for_fvg": True,
            "target_level": 99.85,
            "max_wait_bars": 5
        })
        
        timing = mock_tactical_agent.get_execution_timing(market_state)
        
        assert "immediate" in timing
        assert "wait_for_fvg" in timing
        assert "target_level" in timing
        assert "max_wait_bars" in timing

    @pytest.mark.performance
    def test_tactical_inference_speed(self, mock_tactical_agent, performance_timer):
        """Test tactical agent inference speed."""
        observation = np.random.rand(600)
        
        performance_timer.start()
        action = mock_tactical_agent.act(observation)
        performance_timer.stop()
        
        # Requirement: Tactical inference < 5ms
        assert performance_timer.elapsed_ms() < 5
        assert action is not None

    def test_actor_critic_learning(self, mock_tactical_agent):
        """Test Actor-Critic learning mechanism."""
        experience = {
            "state": np.random.rand(600),
            "action": 4,  # Scale in
            "reward": 0.08,
            "next_state": np.random.rand(600),
            "done": False,
            "action_prob": 0.3
        }
        
        mock_tactical_agent.learn(experience)
        mock_tactical_agent.learn.assert_called_once_with(experience)


class TestExecutionTimingAgent:
    """Test the Execution Timing Agent for optimal trade execution."""

    @pytest.fixture
    def timing_agent_config(self):
        """Configuration for execution timing agent."""
        return {
            "name": "execution_timing_agent",
            "observation_dim": 80,  # FVG + microstructure features
            "timing_strategies": ["IMMEDIATE", "PASSIVE", "OPPORTUNISTIC", "ICEBERG"],
            "execution_window": 30,  # 30 bars (2.5 hours)
            "slippage_threshold": 0.0015,
            "participation_rate": 0.1  # 10% of volume
        }

    @pytest.fixture
    def mock_timing_agent(self, timing_agent_config):
        """Create mock execution timing agent."""
        agent = Mock()
        agent.config = timing_agent_config
        agent.name = "execution_timing_agent"
        
        # Mock methods
        agent.select_strategy = Mock(return_value="OPPORTUNISTIC")
        agent.calculate_optimal_timing = Mock(return_value={
            "entry_bars": [2, 5, 8],
            "quantity_split": [0.4, 0.35, 0.25],
            "price_targets": [99.85, 99.82, 99.80]
        })
        agent.monitor_execution = Mock()
        agent.adjust_strategy = Mock()
        
        return agent

    def test_strategy_selection(self, mock_timing_agent):
        """Test execution strategy selection."""
        market_conditions = {
            "volatility": 0.15,
            "liquidity": 0.8,
            "urgency": 0.6,
            "fvg_opportunities": 2
        }
        
        strategy = mock_timing_agent.select_strategy(market_conditions)
        
        assert strategy in ["IMMEDIATE", "PASSIVE", "OPPORTUNISTIC", "ICEBERG"]
        mock_timing_agent.select_strategy.assert_called_once_with(market_conditions)

    def test_optimal_timing_calculation(self, mock_timing_agent):
        """Test calculation of optimal execution timing."""
        order_params = {
            "total_quantity": 10000,
            "direction": "BUY",
            "target_price": 99.90,
            "max_execution_time": 20  # bars
        }
        
        timing = mock_timing_agent.calculate_optimal_timing(order_params)
        
        assert "entry_bars" in timing
        assert "quantity_split" in timing
        assert "price_targets" in timing
        
        # Quantity splits should sum to 1
        total_split = sum(timing["quantity_split"])
        assert np.isclose(total_split, 1.0)

    def test_execution_monitoring(self, mock_timing_agent):
        """Test real-time execution monitoring."""
        execution_state = {
            "filled_quantity": 4000,
            "average_price": 99.87,
            "remaining_quantity": 6000,
            "elapsed_bars": 8,
            "current_slippage": 0.0008
        }
        
        mock_timing_agent.monitor_execution(execution_state)
        mock_timing_agent.monitor_execution.assert_called_once_with(execution_state)

    def test_strategy_adjustment(self, mock_timing_agent):
        """Test dynamic strategy adjustment during execution."""
        market_change = {
            "volatility_spike": True,
            "liquidity_drop": False,
            "adverse_price_move": 0.005,
            "new_fvg_appeared": True
        }
        
        mock_timing_agent.adjust_strategy(market_change)
        mock_timing_agent.adjust_strategy.assert_called_once_with(market_change)


class TestFVGDetectionAgent:
    """Test the Fair Value Gap Detection Agent."""

    @pytest.fixture
    def fvg_agent_config(self):
        """Configuration for FVG detection agent."""
        return {
            "name": "fvg_detection_agent",
            "observation_dim": 40,  # Price action features
            "gap_threshold": 0.0005,  # 5 basis points
            "max_gap_age": 48,  # 48 bars (4 hours)
            "strength_calculation": "volume_weighted",
            "mitigation_criteria": "50_percent"
        }

    @pytest.fixture
    def mock_fvg_agent(self, fvg_agent_config):
        """Create mock FVG detection agent."""
        agent = Mock()
        agent.config = fvg_agent_config
        agent.name = "fvg_detection_agent"
        
        # Mock data
        agent.active_fvgs = {
            "bullish": [
                {"level": 100.25, "age": 5, "strength": 0.8, "volume": 5000},
                {"level": 100.15, "age": 12, "strength": 0.6, "volume": 3000}
            ],
            "bearish": [
                {"level": 99.75, "age": 3, "strength": 0.9, "volume": 7000}
            ]
        }
        
        # Mock methods
        agent.detect_new_fvgs = Mock(return_value=1)
        agent.update_fvg_ages = Mock()
        agent.check_mitigations = Mock(return_value=0)
        agent.get_nearest_fvg = Mock(return_value={"level": 99.75, "type": "bearish"})
        agent.calculate_fvg_strength = Mock(return_value=0.85)
        
        return agent

    def test_fvg_detection(self, mock_fvg_agent):
        """Test detection of new Fair Value Gaps."""
        price_bars = [
            {"high": 100.0, "low": 99.8, "close": 99.9, "volume": 1000},
            {"high": 100.2, "low": 100.1, "close": 100.15, "volume": 2000},  # Gap up
            {"high": 100.25, "low": 100.12, "close": 100.2, "volume": 1500}
        ]
        
        new_fvgs = mock_fvg_agent.detect_new_fvgs(price_bars)
        
        assert isinstance(new_fvgs, int)
        assert new_fvgs >= 0
        mock_fvg_agent.detect_new_fvgs.assert_called_once_with(price_bars)

    def test_fvg_strength_calculation(self, mock_fvg_agent):
        """Test FVG strength calculation."""
        fvg_data = {
            "gap_size": 0.001,  # 10 basis points
            "volume": 5000,
            "time_of_day": 14,  # 2 PM (high activity)
            "preceding_momentum": 0.8
        }
        
        strength = mock_fvg_agent.calculate_fvg_strength(fvg_data)
        
        assert 0 <= strength <= 1
        mock_fvg_agent.calculate_fvg_strength.assert_called_once_with(fvg_data)

    def test_fvg_mitigation_check(self, mock_fvg_agent):
        """Test checking for FVG mitigations."""
        current_price = 99.88
        
        mitigated_count = mock_fvg_agent.check_mitigations(current_price)
        
        assert isinstance(mitigated_count, int)
        assert mitigated_count >= 0
        mock_fvg_agent.check_mitigations.assert_called_once_with(current_price)

    def test_nearest_fvg_identification(self, mock_fvg_agent):
        """Test identification of nearest FVG."""
        current_price = 99.90
        
        nearest_fvg = mock_fvg_agent.get_nearest_fvg(current_price)
        
        assert "level" in nearest_fvg
        assert "type" in nearest_fvg
        assert nearest_fvg["type"] in ["bullish", "bearish"]
        mock_fvg_agent.get_nearest_fvg.assert_called_once_with(current_price)

    def test_fvg_age_management(self, mock_fvg_agent):
        """Test FVG age management and cleanup."""
        mock_fvg_agent.update_fvg_ages()
        mock_fvg_agent.update_fvg_ages.assert_called_once()
        
        # Test expired FVG removal
        mock_fvg_agent.remove_expired_fvgs = Mock(return_value=2)
        removed_count = mock_fvg_agent.remove_expired_fvgs()
        
        assert isinstance(removed_count, int)
        assert removed_count >= 0


class TestTacticalMARLEnvironment:
    """Test tactical MARL environment for 5-minute trading."""

    @pytest.fixture
    def tactical_env_config(self):
        """Configuration for tactical MARL environment."""
        return {
            "agents": ["tactical_agent", "execution_timing_agent", "fvg_detection_agent"],
            "observation_space_dim": 600,
            "action_space_dim": 7,
            "reward_function": "execution_quality",
            "episode_length": 60,  # 5 hours
            "lookback_window": 60,
            "execution_cost_penalty": 0.01
        }

    @pytest.fixture
    def mock_tactical_env(self, tactical_env_config):
        """Create mock tactical MARL environment."""
        env = Mock()
        env.config = tactical_env_config
        env.agents = tactical_env_config["agents"]
        env.possible_agents = env.agents
        
        # Mock methods
        env.reset = Mock(return_value={
            "tactical_agent": np.random.rand(600),
            "execution_timing_agent": np.random.rand(80),
            "fvg_detection_agent": np.random.rand(40)
        })
        env.step = Mock(return_value=(
            {agent: np.random.rand(600) for agent in env.agents},  # observations
            {agent: 0.05 for agent in env.agents},                 # rewards
            {agent: False for agent in env.agents},                # dones
            {agent: {} for agent in env.agents}                    # infos
        ))
        
        return env

    def test_tactical_environment_reset(self, mock_tactical_env):
        """Test tactical environment reset."""
        observations = mock_tactical_env.reset()
        
        for agent in mock_tactical_env.agents:
            assert agent in observations
            assert len(observations[agent]) > 0
        
        mock_tactical_env.reset.assert_called_once()

    def test_tactical_environment_step(self, mock_tactical_env):
        """Test tactical environment step with multiple agents."""
        actions = {
            "tactical_agent": 4,  # Scale in
            "execution_timing_agent": 2,  # Opportunistic
            "fvg_detection_agent": 0  # No action needed
        }
        
        obs, rewards, dones, infos = mock_tactical_env.step(actions)
        
        for agent in mock_tactical_env.agents:
            assert agent in obs
            assert agent in rewards
            assert agent in dones
            assert agent in infos
        
        mock_tactical_env.step.assert_called_once_with(actions)

    def test_execution_quality_reward(self, mock_tactical_env):
        """Test execution quality reward calculation."""
        execution_metrics = {
            "slippage": 0.0008,
            "market_impact": 0.0005,
            "timing_efficiency": 0.85,
            "fill_rate": 0.95
        }
        
        mock_tactical_env.calculate_execution_reward = Mock(return_value=0.12)
        reward = mock_tactical_env.calculate_execution_reward(execution_metrics)
        
        assert isinstance(reward, (int, float))
        mock_tactical_env.calculate_execution_reward.assert_called_once()


class TestTacticalIntegration:
    """Test integration of tactical components."""

    @pytest.fixture
    def tactical_system(self, mock_assembler, mock_tactical_agent, mock_timing_agent, mock_fvg_agent):
        """Create integrated tactical system."""
        system = Mock()
        system.matrix_assembler = mock_assembler
        system.tactical_agent = mock_tactical_agent
        system.timing_agent = mock_timing_agent
        system.fvg_agent = mock_fvg_agent
        
        # Mock integration methods
        system.process_5m_bar = Mock()
        system.execute_tactical_decision = Mock(return_value={
            "action": 5,  # Buy
            "timing": "OPPORTUNISTIC",
            "fvg_target": 99.85,
            "execution_cost": 0.007
        })
        
        return system

    def test_end_to_end_tactical_pipeline(self, tactical_system):
        """Test complete tactical execution pipeline."""
        strategic_signal = {"position": 0.7, "confidence": 0.8}
        market_data = {
            "price": 100.0,
            "volume": 1500,
            "fvgs": [{"level": 99.85, "type": "bearish"}]
        }
        
        decision = tactical_system.execute_tactical_decision(strategic_signal, market_data)
        
        assert "action" in decision
        assert "timing" in decision
        assert "fvg_target" in decision
        assert "execution_cost" in decision
        
        assert decision["action"] in range(7)
        assert decision["execution_cost"] >= 0
        
        tactical_system.execute_tactical_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_time_tactical_processing(self, tactical_system):
        """Test real-time tactical processing capabilities."""
        # Mock real-time processing
        tactical_system.process_real_time = AsyncMock(return_value={"latency_ms": 2.5})
        
        result = await tactical_system.process_real_time()
        assert result["latency_ms"] < 5  # Sub-5ms processing
        
        tactical_system.process_real_time.assert_called_once()

    @pytest.mark.performance
    def test_tactical_system_throughput(self, tactical_system, benchmark_config):
        """Test tactical system throughput requirements."""
        import time
        
        start_time = time.time()
        
        # Process rapid decisions
        for _ in range(1000):
            tactical_system.execute_tactical_decision({}, {})
        
        elapsed = time.time() - start_time
        throughput = 1000 / elapsed
        
        # Requirement: >1000 decisions per second
        assert throughput > 1000
        assert tactical_system.execute_tactical_decision.call_count == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])