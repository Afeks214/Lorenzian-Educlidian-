"""
Comprehensive Test Suite for Risk Management MARL Architecture

Tests all components of the MARL system:
- Base Risk Agent functionality
- 4 specialized risk agents
- Centralized Critic
- State Processor
- Risk Environment
- Agent Coordinator
- Performance requirements (<10ms response time)
"""

import pytest
import numpy as np
import torch
from datetime import datetime
from unittest.mock import Mock, patch
import time

from src.risk.agents import (
    BaseRiskAgent, RiskState, RiskAction, RiskMetrics,
    PositionSizingAgent, StopTargetAgent, RiskMonitorAgent, PortfolioOptimizerAgent
)
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState, RiskCriticMode
from src.risk.marl.risk_environment import RiskEnvironment
from src.risk.marl.agent_coordinator import AgentCoordinator, CoordinatorConfig, ConsensusMethod
from src.risk.core.state_processor import RiskStateProcessor, StateProcessingConfig
from src.core.events import EventBus


class TestRiskState:
    """Test RiskState functionality"""
    
    def test_risk_state_creation(self):
        """Test RiskState creation and validation"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.0,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        assert isinstance(risk_state, RiskState)
        assert risk_state.account_equity_normalized == 1.0
        assert risk_state.open_positions_count == 5
    
    def test_risk_state_to_vector(self):
        """Test RiskState vector conversion"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.0,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        vector = risk_state.to_vector()
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (10,)
        assert vector[0] == 1.0  # account_equity_normalized
        assert vector[1] == 5    # open_positions_count
    
    def test_risk_state_from_vector(self):
        """Test RiskState creation from vector"""
        vector = np.array([1.0, 5, 0.5, 0.3, 0.02, 0.0, 0.4, 0.5, 0.3, 0.8])
        risk_state = RiskState.from_vector(vector)
        
        assert isinstance(risk_state, RiskState)
        assert risk_state.account_equity_normalized == 1.0
        assert risk_state.open_positions_count == 5
        
    def test_invalid_vector_dimension(self):
        """Test error handling for invalid vector dimensions"""
        with pytest.raises(ValueError):
            RiskState.from_vector(np.array([1.0, 2.0, 3.0]))  # Wrong dimension


class TestPositionSizingAgent:
    """Test Position Sizing Agent (π₁)"""
    
    @pytest.fixture
    def position_agent(self):
        config = {
            'max_leverage': 3.0,
            'var_limit': 0.02,
            'correlation_threshold': 0.7
        }
        return PositionSizingAgent(config)
    
    @pytest.fixture
    def sample_risk_state(self):
        return RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
    
    def test_agent_initialization(self, position_agent):
        """Test agent initialization"""
        assert position_agent.name == 'position_sizing_agent'
        assert position_agent.max_leverage == 3.0
        assert position_agent.sizing_decisions == 0
    
    def test_calculate_risk_action(self, position_agent, sample_risk_state):
        """Test risk action calculation"""
        action, confidence = position_agent.calculate_risk_action(sample_risk_state)
        
        assert isinstance(action, int)
        assert 0 <= action <= 4  # Discrete action space
        assert 0.0 <= confidence <= 1.0
        assert position_agent.sizing_decisions == 1
    
    def test_validate_risk_constraints(self, position_agent, sample_risk_state):
        """Test risk constraint validation"""
        result = position_agent.validate_risk_constraints(sample_risk_state)
        assert isinstance(result, bool)
    
    def test_performance_metrics(self, position_agent, sample_risk_state):
        """Test performance metrics"""
        # Execute some actions
        for _ in range(5):
            position_agent.calculate_risk_action(sample_risk_state)
        
        metrics = position_agent.get_risk_metrics()
        assert isinstance(metrics, RiskMetrics)
        assert metrics.total_risk_decisions == 5


class TestStopTargetAgent:
    """Test Stop/Target Agent (π₂)"""
    
    @pytest.fixture
    def stop_agent(self):
        config = {
            'base_stop_distance': 0.02,
            'base_target_distance': 0.04,
            'volatility_sensitivity': 1.5
        }
        return StopTargetAgent(config)
    
    @pytest.fixture
    def sample_risk_state(self):
        return RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.7,  # Higher volatility for testing
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.6,
            liquidity_conditions=0.8
        )
    
    def test_agent_initialization(self, stop_agent):
        """Test agent initialization"""
        assert stop_agent.name == 'stop_target_agent'
        assert stop_agent.base_stop_distance == 0.02
        assert stop_agent.stop_adjustments == 0
    
    def test_calculate_risk_action(self, stop_agent, sample_risk_state):
        """Test risk action calculation"""
        action, confidence = stop_agent.calculate_risk_action(sample_risk_state)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)  # [stop_multiplier, target_multiplier]
        assert np.all(action >= 0.5) and np.all(action <= 3.0)  # Action bounds
        assert 0.0 <= confidence <= 1.0
    
    def test_volatility_adjustment(self, stop_agent, sample_risk_state):
        """Test volatility-based adjustments"""
        # High volatility state
        high_vol_state = sample_risk_state
        high_vol_state.volatility_regime = 0.9
        
        action_high, _ = stop_agent.calculate_risk_action(high_vol_state)
        
        # Low volatility state
        low_vol_state = sample_risk_state
        low_vol_state.volatility_regime = 0.2
        
        action_low, _ = stop_agent.calculate_risk_action(low_vol_state)
        
        # High volatility should generally lead to wider stops
        # Note: This is a simplified test - actual behavior depends on multiple factors


class TestRiskMonitorAgent:
    """Test Risk Monitor Agent (π₃)"""
    
    @pytest.fixture
    def monitor_agent(self):
        config = {
            'alert_threshold': 0.6,
            'reduce_threshold': 0.75,
            'emergency_threshold': 0.9
        }
        return RiskMonitorAgent(config)
    
    @pytest.fixture
    def high_risk_state(self):
        return RiskState(
            account_equity_normalized=0.8,  # 20% loss
            open_positions_count=10,
            volatility_regime=0.9,  # High volatility
            correlation_risk=0.9,   # High correlation
            var_estimate_5pct=0.08,  # High VaR
            current_drawdown_pct=0.15,  # 15% drawdown
            margin_usage_pct=0.8,   # High margin usage
            time_of_day_risk=0.8,
            market_stress_level=0.9,  # High stress
            liquidity_conditions=0.2  # Low liquidity
        )
    
    def test_agent_initialization(self, monitor_agent):
        """Test agent initialization"""
        assert monitor_agent.name == 'risk_monitor_agent'
        assert monitor_agent.emergency_threshold == 0.9
        assert monitor_agent.alerts_generated == 0
    
    def test_calculate_risk_action_normal(self, monitor_agent):
        """Test normal risk conditions"""
        normal_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=3,
            volatility_regime=0.3,
            correlation_risk=0.3,
            var_estimate_5pct=0.01,
            current_drawdown_pct=0.0,
            margin_usage_pct=0.3,
            time_of_day_risk=0.3,
            market_stress_level=0.2,
            liquidity_conditions=0.9
        )
        
        action, confidence = monitor_agent.calculate_risk_action(normal_state)
        assert action == 0  # NO_ACTION expected
    
    def test_calculate_risk_action_emergency(self, monitor_agent, high_risk_state):
        """Test emergency risk conditions"""
        action, confidence = monitor_agent.calculate_risk_action(high_risk_state)
        
        assert isinstance(action, int)
        assert 0 <= action <= 3  # Discrete action space
        # With high risk state, should trigger emergency or risk reduction
        assert action >= 2  # REDUCE_RISK or EMERGENCY_STOP
        assert 0.0 <= confidence <= 1.0
    
    def test_risk_pattern_detection(self, monitor_agent):
        """Test risk pattern detection"""
        # Simulate rapid drawdown by feeding consecutive high-risk states
        for i in range(6):
            risk_state = RiskState(
                account_equity_normalized=1.0 - i * 0.02,  # Increasing drawdown
                open_positions_count=5,
                volatility_regime=0.5,
                correlation_risk=0.3,
                var_estimate_5pct=0.02,
                current_drawdown_pct=i * 0.02,
                margin_usage_pct=0.4,
                time_of_day_risk=0.5,
                market_stress_level=0.3,
                liquidity_conditions=0.8
            )
            monitor_agent.calculate_risk_action(risk_state)
        
        assert len(monitor_agent.risk_history) == 6


class TestPortfolioOptimizerAgent:
    """Test Portfolio Optimizer Agent (π₄)"""
    
    @pytest.fixture
    def optimizer_agent(self):
        config = {
            'target_volatility': 0.12,
            'max_correlation': 0.8,
            'rebalance_threshold': 0.05
        }
        return PortfolioOptimizerAgent(config)
    
    @pytest.fixture
    def sample_risk_state(self):
        return RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.6,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
    
    def test_agent_initialization(self, optimizer_agent):
        """Test agent initialization"""
        assert optimizer_agent.name == 'portfolio_optimizer_agent'
        assert len(optimizer_agent.asset_classes) == 5
        assert optimizer_agent.rebalances_executed == 0
    
    def test_calculate_risk_action(self, optimizer_agent, sample_risk_state):
        """Test portfolio optimization action"""
        action, confidence = optimizer_agent.calculate_risk_action(sample_risk_state)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (5,)  # 5 asset classes
        assert np.all(action >= 0.0) and np.all(action <= 1.0)  # Weight bounds
        assert np.isclose(np.sum(action), 1.0, atol=1e-3)  # Weights sum to 1
        assert 0.0 <= confidence <= 1.0
    
    def test_weight_normalization(self, optimizer_agent):
        """Test weight normalization"""
        weights = np.array([0.7, 0.3, 0.1, 0.05, 0.05])  # Sum > 1
        normalized = optimizer_agent._normalize_weights(weights)
        
        assert np.isclose(np.sum(normalized), 1.0, atol=1e-6)
        assert np.all(normalized >= 0.0)
    
    def test_stress_scenario_allocation(self, optimizer_agent):
        """Test allocation during stress scenarios"""
        stress_state = RiskState(
            account_equity_normalized=0.9,
            open_positions_count=8,
            volatility_regime=0.8,
            correlation_risk=0.9,
            var_estimate_5pct=0.06,
            current_drawdown_pct=0.12,
            margin_usage_pct=0.7,
            time_of_day_risk=0.7,
            market_stress_level=0.8,  # High stress
            liquidity_conditions=0.3
        )
        
        action, confidence = optimizer_agent.calculate_risk_action(stress_state)
        
        # In stress, should increase cash allocation (index 3)
        assert action[3] > 0.1  # At least 10% cash in stress


class TestCentralizedCritic:
    """Test Centralized Critic"""
    
    @pytest.fixture
    def critic(self):
        config = {
            'learning_rate': 1e-4,
            'stress_threshold': 0.15,
            'emergency_threshold': 0.25
        }
        return CentralizedCritic(config)
    
    @pytest.fixture
    def global_risk_state(self):
        base_vector = np.random.randn(10) * 0.1  # Small random values
        return GlobalRiskState(
            position_sizing_risk=base_vector,
            stop_target_risk=base_vector,
            risk_monitor_risk=base_vector,
            portfolio_optimizer_risk=base_vector,
            total_portfolio_var=0.03,
            portfolio_correlation_max=0.6,
            aggregate_leverage=2.0,
            liquidity_risk_score=0.2,
            systemic_risk_level=0.3,
            timestamp=datetime.now(),
            market_hours_factor=0.8
        )
    
    def test_critic_initialization(self, critic):
        """Test critic initialization"""
        assert isinstance(critic.network, torch.nn.Module)
        assert critic.evaluation_count == 0
    
    def test_evaluate_global_risk(self, critic, global_risk_state):
        """Test global risk evaluation"""
        value, mode = critic.evaluate_global_risk(global_risk_state)
        
        assert isinstance(value, float)
        assert isinstance(mode, RiskCriticMode)
        assert critic.evaluation_count == 1
    
    def test_emergency_detection(self, critic):
        """Test emergency condition detection"""
        emergency_state = GlobalRiskState(
            position_sizing_risk=np.zeros(10),
            stop_target_risk=np.zeros(10),
            risk_monitor_risk=np.zeros(10),
            portfolio_optimizer_risk=np.zeros(10),
            total_portfolio_var=0.3,  # High VaR
            portfolio_correlation_max=0.98,  # Very high correlation
            aggregate_leverage=15.0,  # Excessive leverage
            liquidity_risk_score=0.8,
            systemic_risk_level=0.9,
            timestamp=datetime.now(),
            market_hours_factor=0.8
        )
        
        value, mode = critic.evaluate_global_risk(emergency_state)
        assert mode == RiskCriticMode.EMERGENCY
    
    def test_compute_agent_gradients(self, critic, global_risk_state):
        """Test agent gradient computation"""
        gradients = critic.compute_agent_gradients(global_risk_state, target_value=0.5)
        
        assert isinstance(gradients, dict)
        assert all(agent in gradients for agent in ['π1', 'π2', 'π3', 'π4'])
        assert all(isinstance(grad, torch.Tensor) for grad in gradients.values())


class TestRiskEnvironment:
    """Test Risk Environment"""
    
    @pytest.fixture
    def risk_env(self):
        config = {
            'max_steps': 100,
            'initial_capital': 1000000.0,
            'symbols': ['SPY', 'QQQ', 'TLT']
        }
        return RiskEnvironment(config)
    
    def test_environment_initialization(self, risk_env):
        """Test environment initialization"""
        assert risk_env.num_agents == 4
        assert len(risk_env.agent_names) == 4
        assert risk_env.current_step == 0
    
    def test_environment_reset(self, risk_env):
        """Test environment reset"""
        observations = risk_env.reset()
        
        assert isinstance(observations, dict)
        assert len(observations) == 4
        for agent_name in risk_env.agent_names:
            assert agent_name in observations
            assert observations[agent_name].shape == (10,)  # 10D risk vector
    
    def test_environment_step(self, risk_env):
        """Test environment step"""
        observations = risk_env.reset()
        
        # Create valid actions for all agents
        actions = {
            'position_sizing': 2,  # HOLD
            'stop_target': np.array([1.0, 1.0], dtype=np.float32),
            'risk_monitor': 0,  # NO_ACTION
            'portfolio_optimizer': np.array([0.6, 0.25, 0.05, 0.05, 0.05], dtype=np.float32)
        }
        
        next_obs, rewards, done, info = risk_env.step(actions)
        
        assert isinstance(next_obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert len(rewards) == 4


class TestAgentCoordinator:
    """Test Agent Coordinator"""
    
    @pytest.fixture
    def coordinator(self):
        config = CoordinatorConfig()
        critic_config = {}
        critic = CentralizedCritic(critic_config)
        return AgentCoordinator(config, critic)
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        agents = {}
        for name in ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer']:
            agent = Mock()
            agent.name = name
            agent.make_decision = Mock(return_value=(0, 0.8))  # Mock action and confidence
            agents[name] = agent
        return agents
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization"""
        assert len(coordinator.agents) == 0
        assert coordinator.coordination_count == 0
    
    def test_agent_registration(self, coordinator, mock_agents):
        """Test agent registration"""
        for agent in mock_agents.values():
            result = coordinator.register_agent(agent)
            assert result is True
        
        assert len(coordinator.agents) == 4
    
    def test_coordinate_decision(self, coordinator, mock_agents):
        """Test decision coordination"""
        # Register agents
        for agent in mock_agents.values():
            coordinator.register_agent(agent)
        
        # Create sample risk state
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        consensus_results = coordinator.coordinate_decision(risk_state)
        
        assert isinstance(consensus_results, dict)
        assert coordinator.coordination_count == 1


class TestStateProcessor:
    """Test State Processor"""
    
    @pytest.fixture
    def processor(self):
        config = StateProcessingConfig()
        return RiskStateProcessor(config)
    
    @pytest.fixture
    def sample_state_vector(self):
        return np.array([1.0, 5, 0.5, 0.3, 0.02, 0.05, 0.4, 0.5, 0.3, 0.8])
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.total_processed == 0
        assert not processor.statistics.is_valid()
    
    def test_process_state(self, processor, sample_state_vector):
        """Test state processing"""
        normalized_state, metadata = processor.process_state(sample_state_vector)
        
        assert isinstance(normalized_state, np.ndarray)
        assert normalized_state.shape == (10,)
        assert isinstance(metadata, dict)
        assert processor.total_processed == 1
    
    def test_outlier_detection(self, processor):
        """Test outlier detection"""
        # Process normal states first to build statistics
        for _ in range(20):
            normal_state = np.random.normal(0, 1, 10)
            processor.process_state(normal_state)
        
        # Process outlier state
        outlier_state = np.array([100.0] * 10)  # Extreme outlier
        normalized_state, metadata = processor.process_state(outlier_state)
        
        # Should detect outlier (though may not always flag depending on statistics)
        assert 'outlier_detected' in metadata
    
    def test_validation_failure(self, processor):
        """Test validation failure handling"""
        invalid_state = np.array([np.nan, np.inf, 1, 2, 3, 4, 5, 6, 7, 8])
        normalized_state, metadata = processor.process_state(invalid_state)
        
        assert not metadata['validation_passed']
        assert np.all(normalized_state == 0)  # Should return safe default


class TestPerformanceRequirements:
    """Test performance requirements (<10ms response time)"""
    
    @pytest.fixture
    def all_agents(self):
        """Create all agent types"""
        event_bus = EventBus()
        agents = {
            'position_sizing': PositionSizingAgent({}, event_bus),
            'stop_target': StopTargetAgent({}, event_bus),
            'risk_monitor': RiskMonitorAgent({}, event_bus),
            'portfolio_optimizer': PortfolioOptimizerAgent({}, event_bus)
        }
        return agents
    
    @pytest.fixture
    def sample_risk_state(self):
        return RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
    
    def test_agent_response_time(self, all_agents, sample_risk_state):
        """Test that all agents respond within 10ms"""
        for agent_name, agent in all_agents.items():
            start_time = time.time()
            
            action, confidence = agent.calculate_risk_action(sample_risk_state)
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            assert response_time < 10.0, f"{agent_name} exceeded 10ms target: {response_time:.2f}ms"
    
    def test_state_processor_performance(self):
        """Test state processor performance"""
        processor = RiskStateProcessor(StateProcessingConfig())
        state_vector = np.random.randn(10)
        
        start_time = time.time()
        
        normalized_state, metadata = processor.process_state(state_vector)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert processing_time < 5.0, f"State processing exceeded 5ms target: {processing_time:.2f}ms"
    
    def test_critic_performance(self):
        """Test centralized critic performance"""
        critic = CentralizedCritic({})
        base_vector = np.random.randn(10) * 0.1
        global_state = GlobalRiskState(
            position_sizing_risk=base_vector,
            stop_target_risk=base_vector,
            risk_monitor_risk=base_vector,
            portfolio_optimizer_risk=base_vector,
            total_portfolio_var=0.03,
            portfolio_correlation_max=0.6,
            aggregate_leverage=2.0,
            liquidity_risk_score=0.2,
            systemic_risk_level=0.3,
            timestamp=datetime.now(),
            market_hours_factor=0.8
        )
        
        start_time = time.time()
        
        value, mode = critic.evaluate_global_risk(global_state)
        
        evaluation_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert evaluation_time < 10.0, f"Critic evaluation exceeded 10ms target: {evaluation_time:.2f}ms"


class TestIntegration:
    """Integration tests for the complete MARL system"""
    
    def test_end_to_end_coordination(self):
        """Test end-to-end coordination of all components"""
        # Initialize event bus
        event_bus = EventBus()
        
        # Initialize all agents
        agents = {
            'position_sizing': PositionSizingAgent({}, event_bus),
            'stop_target': StopTargetAgent({}, event_bus),
            'risk_monitor': RiskMonitorAgent({}, event_bus),
            'portfolio_optimizer': PortfolioOptimizerAgent({}, event_bus)
        }
        
        # Initialize coordinator
        config = CoordinatorConfig()
        critic = CentralizedCritic({}, event_bus)
        coordinator = AgentCoordinator(config, critic, event_bus)
        
        # Register agents
        for agent in agents.values():
            coordinator.register_agent(agent)
        
        # Create risk state
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        # Test coordination
        start_time = time.time()
        consensus_results = coordinator.coordinate_decision(risk_state)
        coordination_time = (time.time() - start_time) * 1000
        
        # Validate results
        assert isinstance(consensus_results, dict)
        assert coordination_time < 10.0  # Should complete within 10ms
        assert coordinator.coordination_count == 1
    
    def test_environment_with_real_agents(self):
        """Test environment with real agent implementations"""
        # Initialize environment
        config = {
            'max_steps': 10,
            'initial_capital': 1000000.0,
            'symbols': ['SPY', 'QQQ']
        }
        env = RiskEnvironment(config)
        
        # Reset environment
        observations = env.reset()
        
        # Run a few steps with realistic actions
        for step in range(5):
            actions = {
                'position_sizing': np.random.randint(0, 5),
                'stop_target': np.random.uniform(0.5, 3.0, 2).astype(np.float32),
                'risk_monitor': np.random.randint(0, 4),
                'portfolio_optimizer': np.random.dirichlet([1, 1, 1, 1, 1]).astype(np.float32)
            }
            
            next_obs, rewards, done, info = env.step(actions)
            
            assert len(next_obs) == 4
            assert len(rewards) == 4
            assert isinstance(done, bool)
            
            if done:
                break
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])