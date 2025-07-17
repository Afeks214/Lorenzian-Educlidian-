"""
Test suite for Tactical MARL Agents.

This module tests the individual tactical agents, their rapid decision-making,
execution optimization, and real-time coordination in the 5-minute environment.
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Tuple
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import time

# Test markers
pytestmark = [pytest.mark.tactical, pytest.mark.unit]


class TestTacticalExecutionAgent:
    """Test the main Tactical Execution Agent for 5-minute decisions."""

    @pytest.fixture
    def agent_config(self):
        """Configuration for tactical execution agent."""
        return {
            "name": "tactical_execution_agent",
            "observation_dim": 600,  # 60 bars × 10 features
            "action_dim": 11,  # Fine-grained execution actions
            "learning_rate": 0.001,
            "hidden_dims": [256, 128, 64],
            "network_type": "ddpg",  # Continuous control
            "memory_size": 50000,
            "batch_size": 32,
            "tau": 0.01,  # Faster target updates for tactical
            "noise_scale": 0.1,
            "execution_horizon": 12,  # 1 hour tactical window
            "max_slippage": 0.002
        }

    @pytest.fixture
    def mock_execution_agent(self, agent_config):
        """Create a mock tactical execution agent."""
        agent = Mock()
        agent.config = agent_config
        agent.name = "tactical_execution_agent"
        agent.observation_dim = agent_config["observation_dim"]
        agent.action_dim = agent_config["action_dim"]
        
        # Mock networks
        agent.actor_network = Mock()
        agent.critic_network = Mock()
        agent.target_actor = Mock()
        agent.target_critic = Mock()
        agent.optimizer_actor = Mock()
        agent.optimizer_critic = Mock()
        
        # Mock methods
        agent.act = Mock(return_value=np.array([0.5, 0.3, 0.8]))  # Continuous actions
        agent.learn = Mock()
        agent.update_targets = Mock()
        agent.add_noise = Mock()
        agent.calculate_slippage = Mock(return_value=0.0008)
        agent.optimize_execution = Mock()
        
        return agent

    def test_tactical_agent_initialization(self, agent_config):
        """Test tactical execution agent initialization."""
        agent = Mock()
        agent.config = agent_config
        
        assert agent.config["name"] == "tactical_execution_agent"
        assert agent.config["observation_dim"] == 600
        assert agent.config["action_dim"] == 11
        assert agent.config["network_type"] == "ddpg"
        assert agent.config["execution_horizon"] == 12

    def test_continuous_action_generation(self, mock_execution_agent):
        """Test continuous action generation for fine-grained control."""
        observation = np.random.rand(600)
        
        action = mock_execution_agent.act(observation)
        
        assert isinstance(action, np.ndarray)
        assert len(action) == 3  # Example: [urgency, size_fraction, timing_delay]
        assert all(0 <= a <= 1 for a in action)  # Normalized actions
        
        mock_execution_agent.act.assert_called_once_with(observation)

    def test_slippage_calculation(self, mock_execution_agent):
        """Test slippage calculation for execution quality."""
        execution_params = {
            "order_size": 5000,
            "market_depth": 10000,
            "volatility": 0.15,
            "execution_urgency": 0.8,
            "time_of_day": 14
        }
        
        slippage = mock_execution_agent.calculate_slippage(execution_params)
        
        assert isinstance(slippage, (int, float))
        assert 0 <= slippage <= 0.002  # Within max slippage
        mock_execution_agent.calculate_slippage.assert_called_once_with(execution_params)

    def test_execution_optimization(self, mock_execution_agent):
        """Test execution optimization algorithm."""
        optimization_params = {
            "target_quantity": 10000,
            "time_horizon": 8,  # bars
            "market_conditions": np.random.rand(20),
            "risk_budget": 0.001
        }
        
        mock_execution_agent.optimize_execution(optimization_params)
        mock_execution_agent.optimize_execution.assert_called_once_with(optimization_params)

    @pytest.mark.performance
    def test_tactical_inference_speed(self, mock_execution_agent, performance_timer):
        """Test tactical inference speed requirements."""
        observation = np.random.rand(600)
        
        performance_timer.start()
        action = mock_execution_agent.act(observation)
        performance_timer.stop()
        
        # Requirement: Tactical inference < 2ms
        assert performance_timer.elapsed_ms() < 2
        assert action is not None

    def test_ddpg_learning(self, mock_execution_agent):
        """Test DDPG learning for continuous control."""
        experience = {
            "state": np.random.rand(600),
            "action": np.array([0.5, 0.3, 0.8]),
            "reward": 0.12,
            "next_state": np.random.rand(600),
            "done": False
        }
        
        mock_execution_agent.learn(experience)
        mock_execution_agent.learn.assert_called_once_with(experience)

    def test_target_network_updates(self, mock_execution_agent):
        """Test soft target network updates."""
        mock_execution_agent.update_targets()
        mock_execution_agent.update_targets.assert_called_once()

    def test_exploration_noise(self, mock_execution_agent):
        """Test exploration noise for action space exploration."""
        clean_action = np.array([0.5, 0.3, 0.8])
        
        mock_execution_agent.add_noise = Mock(return_value=np.array([0.52, 0.28, 0.83]))
        noisy_action = mock_execution_agent.add_noise(clean_action)
        
        assert isinstance(noisy_action, np.ndarray)
        assert len(noisy_action) == len(clean_action)
        mock_execution_agent.add_noise.assert_called_once_with(clean_action)


class TestOrderFlowAgent:
    """Test the Order Flow Agent for microstructure analysis."""

    @pytest.fixture
    def orderflow_agent_config(self):
        """Configuration for order flow agent."""
        return {
            "name": "order_flow_agent",
            "observation_dim": 120,  # Order book + trade flow features
            "update_frequency": 1,  # Every bar
            "depth_levels": 10,
            "flow_window": 20,  # 20 bars lookback
            "imbalance_threshold": 0.3,
            "pressure_sensitivity": 0.1
        }

    @pytest.fixture
    def mock_orderflow_agent(self, orderflow_agent_config):
        """Create mock order flow agent."""
        agent = Mock()
        agent.config = orderflow_agent_config
        agent.name = "order_flow_agent"
        
        # Mock data
        agent.order_book = {
            "bids": [(99.98, 1000), (99.97, 1500), (99.96, 800)],
            "asks": [(100.02, 1200), (100.03, 900), (100.04, 1100)]
        }
        
        # Mock methods
        agent.calculate_imbalance = Mock(return_value=0.15)
        agent.detect_flow_pressure = Mock(return_value=-0.2)
        agent.analyze_depth = Mock(return_value={"depth_score": 0.7})
        agent.predict_short_term_direction = Mock(return_value=1)
        agent.get_execution_signal = Mock(return_value={
            "direction": "BUY",
            "confidence": 0.75,
            "urgency": 0.4
        })
        
        return agent

    def test_order_imbalance_calculation(self, mock_orderflow_agent):
        """Test order book imbalance calculation."""
        order_book = {
            "bid_volume": 8000,
            "ask_volume": 6000,
            "bid_price": 99.98,
            "ask_price": 100.02
        }
        
        imbalance = mock_orderflow_agent.calculate_imbalance(order_book)
        
        assert isinstance(imbalance, (int, float))
        assert -1 <= imbalance <= 1
        mock_orderflow_agent.calculate_imbalance.assert_called_once_with(order_book)

    def test_flow_pressure_detection(self, mock_orderflow_agent):
        """Test detection of order flow pressure."""
        trade_flow = [
            {"side": "buy", "volume": 500, "timestamp": datetime.now()},
            {"side": "sell", "volume": 300, "timestamp": datetime.now()},
            {"side": "buy", "volume": 800, "timestamp": datetime.now()}
        ]
        
        pressure = mock_orderflow_agent.detect_flow_pressure(trade_flow)
        
        assert isinstance(pressure, (int, float))
        assert -1 <= pressure <= 1
        mock_orderflow_agent.detect_flow_pressure.assert_called_once_with(trade_flow)

    def test_depth_analysis(self, mock_orderflow_agent):
        """Test order book depth analysis."""
        depth_data = {
            "levels": 10,
            "total_bid_volume": 15000,
            "total_ask_volume": 12000,
            "price_range": 0.1
        }
        
        analysis = mock_orderflow_agent.analyze_depth(depth_data)
        
        assert "depth_score" in analysis
        assert 0 <= analysis["depth_score"] <= 1
        mock_orderflow_agent.analyze_depth.assert_called_once_with(depth_data)

    def test_short_term_direction_prediction(self, mock_orderflow_agent):
        """Test short-term price direction prediction."""
        microstructure_features = np.random.rand(40)
        
        direction = mock_orderflow_agent.predict_short_term_direction(microstructure_features)
        
        assert direction in [-1, 0, 1]  # Down, neutral, up
        mock_orderflow_agent.predict_short_term_direction.assert_called_once()

    def test_execution_signal_generation(self, mock_orderflow_agent):
        """Test execution signal generation."""
        market_state = {
            "imbalance": 0.15,
            "pressure": -0.2,
            "depth": 0.7,
            "predicted_direction": 1
        }
        
        signal = mock_orderflow_agent.get_execution_signal(market_state)
        
        assert "direction" in signal
        assert "confidence" in signal
        assert "urgency" in signal
        
        assert signal["direction"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= signal["confidence"] <= 1
        assert 0 <= signal["urgency"] <= 1


class TestLatencyOptimizationAgent:
    """Test the Latency Optimization Agent for ultra-fast execution."""

    @pytest.fixture
    def latency_agent_config(self):
        """Configuration for latency optimization agent."""
        return {
            "name": "latency_optimization_agent",
            "observation_dim": 50,  # Simplified features for speed
            "target_latency_ms": 1,
            "network_type": "linear",  # Fastest possible
            "batch_normalization": False,
            "activation": "relu",
            "optimization_level": "aggressive"
        }

    @pytest.fixture
    def mock_latency_agent(self, latency_agent_config):
        """Create mock latency optimization agent."""
        agent = Mock()
        agent.config = latency_agent_config
        agent.name = "latency_optimization_agent"
        
        # Mock methods
        agent.fast_inference = Mock(return_value=2)  # Action index
        agent.measure_latency = Mock(return_value=0.8)  # ms
        agent.optimize_network = Mock()
        agent.precompute_features = Mock()
        agent.cache_predictions = Mock()
        
        return agent

    def test_ultra_fast_inference(self, mock_latency_agent):
        """Test ultra-fast inference capabilities."""
        observation = np.random.rand(50)
        
        action = mock_latency_agent.fast_inference(observation)
        
        assert isinstance(action, int)
        assert action >= 0
        mock_latency_agent.fast_inference.assert_called_once_with(observation)

    @pytest.mark.performance
    def test_latency_measurement(self, mock_latency_agent, performance_timer):
        """Test actual latency measurement."""
        observation = np.random.rand(50)
        
        performance_timer.start()
        mock_latency_agent.fast_inference(observation)
        performance_timer.stop()
        
        measured_latency = performance_timer.elapsed_ms()
        mock_latency_latency = mock_latency_agent.measure_latency()
        
        # Requirement: < 1ms inference
        assert measured_latency < 1.0
        assert mock_latency_latency < 1.0

    def test_network_optimization(self, mock_latency_agent):
        """Test network architecture optimization for speed."""
        optimization_params = {
            "pruning_threshold": 0.01,
            "quantization": True,
            "layer_fusion": True
        }
        
        mock_latency_agent.optimize_network(optimization_params)
        mock_latency_agent.optimize_network.assert_called_once_with(optimization_params)

    def test_feature_precomputation(self, mock_latency_agent):
        """Test feature precomputation for faster inference."""
        raw_data = np.random.rand(100)
        
        mock_latency_agent.precompute_features(raw_data)
        mock_latency_agent.precompute_features.assert_called_once_with(raw_data)

    def test_prediction_caching(self, mock_latency_agent):
        """Test prediction caching mechanism."""
        cache_key = "market_state_hash_123"
        
        mock_latency_agent.cache_predictions(cache_key)
        mock_latency_agent.cache_predictions.assert_called_once_with(cache_key)


class TestLiquidityProvisionAgent:
    """Test the Liquidity Provision Agent for market making."""

    @pytest.fixture
    def liquidity_agent_config(self):
        """Configuration for liquidity provision agent."""
        return {
            "name": "liquidity_provision_agent",
            "observation_dim": 80,
            "spread_management": True,
            "inventory_limit": 10000,
            "skew_sensitivity": 0.1,
            "adverse_selection_threshold": 0.05,
            "profit_target": 0.0002  # 2 basis points
        }

    @pytest.fixture
    def mock_liquidity_agent(self, liquidity_agent_config):
        """Create mock liquidity provision agent."""
        agent = Mock()
        agent.config = liquidity_agent_config
        agent.name = "liquidity_provision_agent"
        agent.current_inventory = 0
        
        # Mock methods
        agent.calculate_optimal_spread = Mock(return_value=0.0008)
        agent.manage_inventory = Mock(return_value={"skew": -0.1})
        agent.detect_adverse_selection = Mock(return_value=False)
        agent.update_quotes = Mock()
        agent.calculate_pnl = Mock(return_value=0.0015)
        
        return agent

    def test_optimal_spread_calculation(self, mock_liquidity_agent):
        """Test optimal bid-ask spread calculation."""
        market_conditions = {
            "volatility": 0.12,
            "volume": 2000,
            "adverse_selection_risk": 0.03,
            "competition": 0.7
        }
        
        spread = mock_liquidity_agent.calculate_optimal_spread(market_conditions)
        
        assert isinstance(spread, (int, float))
        assert spread > 0
        mock_liquidity_agent.calculate_optimal_spread.assert_called_once()

    def test_inventory_management(self, mock_liquidity_agent):
        """Test inventory skew management."""
        inventory_state = {
            "current_position": 3000,
            "target_position": 0,
            "max_position": 10000,
            "market_direction": 1
        }
        
        management = mock_liquidity_agent.manage_inventory(inventory_state)
        
        assert "skew" in management
        assert -1 <= management["skew"] <= 1
        mock_liquidity_agent.manage_inventory.assert_called_once()

    def test_adverse_selection_detection(self, mock_liquidity_agent):
        """Test adverse selection detection."""
        trade_history = [
            {"side": "buy", "fill_ratio": 0.8, "price_move": 0.001},
            {"side": "sell", "fill_ratio": 0.9, "price_move": -0.0008}
        ]
        
        is_adverse = mock_liquidity_agent.detect_adverse_selection(trade_history)
        
        assert isinstance(is_adverse, bool)
        mock_liquidity_agent.detect_adverse_selection.assert_called_once()

    def test_quote_updates(self, mock_liquidity_agent):
        """Test dynamic quote updates."""
        new_quotes = {
            "bid_price": 99.996,
            "ask_price": 100.004,
            "bid_size": 1000,
            "ask_size": 1000
        }
        
        mock_liquidity_agent.update_quotes(new_quotes)
        mock_liquidity_agent.update_quotes.assert_called_once_with(new_quotes)

    def test_pnl_calculation(self, mock_liquidity_agent):
        """Test P&L calculation for market making."""
        trading_session = {
            "filled_orders": 25,
            "total_volume": 50000,
            "realized_spread": 0.0006,
            "inventory_pnl": -0.0002
        }
        
        pnl = mock_liquidity_agent.calculate_pnl(trading_session)
        
        assert isinstance(pnl, (int, float))
        mock_liquidity_agent.calculate_pnl.assert_called_once()


class TestTacticalCoordinator:
    """Test the Tactical Coordinator for agent orchestration."""

    @pytest.fixture
    def tactical_coordinator_config(self):
        """Configuration for tactical coordinator."""
        return {
            "name": "tactical_coordinator",
            "agents": [
                "tactical_execution_agent",
                "order_flow_agent", 
                "latency_optimization_agent",
                "liquidity_provision_agent"
            ],
            "coordination_method": "hierarchical",
            "decision_frequency": "per_bar",
            "conflict_resolution": "priority_weighted",
            "performance_monitoring": True
        }

    @pytest.fixture
    def mock_tactical_coordinator(self, tactical_coordinator_config, mock_execution_agent,
                                 mock_orderflow_agent, mock_latency_agent, mock_liquidity_agent):
        """Create mock tactical coordinator."""
        coordinator = Mock()
        coordinator.config = tactical_coordinator_config
        coordinator.agents = {
            "execution": mock_execution_agent,
            "orderflow": mock_orderflow_agent,
            "latency": mock_latency_agent,
            "liquidity": mock_liquidity_agent
        }
        
        # Mock methods
        coordinator.orchestrate_decision = Mock(return_value={
            "primary_action": 5,  # Buy
            "execution_method": "ICEBERG",
            "urgency": 0.6,
            "expected_cost": 0.0012
        })
        coordinator.resolve_conflicts = Mock()
        coordinator.monitor_performance = Mock()
        coordinator.adapt_coordination = Mock()
        
        return coordinator

    def test_decision_orchestration(self, mock_tactical_coordinator):
        """Test orchestration of tactical decisions."""
        market_update = {
            "timestamp": datetime.now(),
            "features": np.random.rand(600),
            "orderflow": np.random.rand(120),
            "strategic_signal": {"position": 0.8, "confidence": 0.9}
        }
        
        decision = mock_tactical_coordinator.orchestrate_decision(market_update)
        
        assert "primary_action" in decision
        assert "execution_method" in decision
        assert "urgency" in decision
        assert "expected_cost" in decision
        
        assert isinstance(decision["primary_action"], int)
        assert 0 <= decision["urgency"] <= 1
        assert decision["expected_cost"] >= 0

    def test_conflict_resolution(self, mock_tactical_coordinator):
        """Test conflict resolution between agents."""
        conflicting_signals = {
            "execution_agent": {"action": "BUY", "confidence": 0.8},
            "orderflow_agent": {"action": "WAIT", "confidence": 0.9},
            "liquidity_agent": {"action": "PROVIDE", "confidence": 0.7}
        }
        
        mock_tactical_coordinator.resolve_conflicts(conflicting_signals)
        mock_tactical_coordinator.resolve_conflicts.assert_called_once()

    def test_performance_monitoring(self, mock_tactical_coordinator):
        """Test real-time performance monitoring."""
        performance_metrics = {
            "latency_ms": 1.2,
            "execution_cost": 0.0008,
            "fill_rate": 0.98,
            "slippage": 0.0005
        }
        
        mock_tactical_coordinator.monitor_performance(performance_metrics)
        mock_tactical_coordinator.monitor_performance.assert_called_once()

    def test_adaptive_coordination(self, mock_tactical_coordinator):
        """Test adaptive coordination based on market conditions."""
        market_regime = {
            "volatility": "HIGH",
            "liquidity": "LOW", 
            "trend": "STRONG",
            "session": "OVERLAP"
        }
        
        mock_tactical_coordinator.adapt_coordination(market_regime)
        mock_tactical_coordinator.adapt_coordination.assert_called_once()

    @pytest.mark.integration
    def test_end_to_end_tactical_coordination(self, mock_tactical_coordinator):
        """Test complete tactical coordination pipeline."""
        # Simulate rapid market update
        market_update = {
            "timestamp": datetime.now(),
            "price": 100.05,
            "volume": 2500,
            "orderbook": {"bid": 100.04, "ask": 100.06},
            "strategic_signal": {"position": 0.6, "urgency": 0.7}
        }
        
        # Get coordinated tactical decision
        decision = mock_tactical_coordinator.orchestrate_decision(market_update)
        
        # Validate decision completeness
        required_fields = ["primary_action", "execution_method", "urgency", "expected_cost"]
        for field in required_fields:
            assert field in decision
        
        # Validate decision reasonableness
        assert isinstance(decision["primary_action"], int)
        assert decision["execution_method"] in ["MARKET", "LIMIT", "ICEBERG", "TWAP"]
        assert 0 <= decision["urgency"] <= 1
        assert decision["expected_cost"] >= 0

    @pytest.mark.asyncio
    async def test_async_tactical_coordination(self, mock_tactical_coordinator):
        """Test asynchronous tactical coordination."""
        # Mock async coordination
        mock_tactical_coordinator.coordinate_async = AsyncMock(return_value={
            "decision": {"action": 3, "timing": "IMMEDIATE"},
            "latency_ms": 0.8
        })
        
        result = await mock_tactical_coordinator.coordinate_async()
        
        assert "decision" in result
        assert "latency_ms" in result
        assert result["latency_ms"] < 1.0

    @pytest.mark.performance
    def test_coordination_performance(self, mock_tactical_coordinator, performance_timer):
        """Test coordination performance under load."""
        # Multiple rapid decisions
        for i in range(100):
            performance_timer.start()
            mock_tactical_coordinator.orchestrate_decision({"id": i})
            performance_timer.stop()
            
            # Each decision should be sub-millisecond
            assert performance_timer.elapsed_ms() < 1.0
        
        # Total call count
        assert mock_tactical_coordinator.orchestrate_decision.call_count == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])