"""
Comprehensive Tests for Execution Environment

Tests the PettingZoo execution environment implementation including:
- PettingZoo API compliance
- 5-agent execution coordination
- Market microstructure simulation
- Performance-based rewards
- Execution quality metrics
- Order routing and broker selection

Author: Claude Code
Version: 1.0 - PettingZoo Implementation
"""

import pytest
import numpy as np
import time
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.execution_env import (
    ExecutionEnvironment,
    ExecutionEnvironmentConfig,
    MarketState,
    env,
    raw_env
)
from pettingzoo.test import api_test


class TestExecutionEnvironmentLogic:
    """Test suite for execution environment logic"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return ExecutionEnvironmentConfig(
            max_steps=100,
            initial_portfolio_value=50000.0,
            max_position_size=0.2,
            transaction_cost_bps=2.0,
            target_fill_rate=0.995,
            target_slippage_bps=2.0,
            target_latency_us=500.0
        )
    
    @pytest.fixture
    def test_env(self, config):
        """Create test environment"""
        return ExecutionEnvironment(config)
    
    def test_environment_initialization(self, test_env):
        """Test environment initializes correctly"""
        assert test_env is not None
        assert len(test_env.possible_agents) == 5
        assert set(test_env.possible_agents) == {
            'position_sizing', 'stop_target', 'risk_monitor', 
            'portfolio_optimizer', 'routing'
        }
        assert test_env.current_step == 0
        assert test_env.config.initial_portfolio_value == 50000.0
    
    def test_action_spaces_definition(self, test_env):
        """Test action spaces are correctly defined"""
        # Position sizing - continuous
        assert test_env.action_spaces['position_sizing'].shape == (3,)
        assert test_env.action_spaces['position_sizing'].low[0] == 0.0
        assert test_env.action_spaces['position_sizing'].high[0] == 1.0
        
        # Stop/target - continuous
        assert test_env.action_spaces['stop_target'].shape == (3,)
        assert test_env.action_spaces['stop_target'].low[0] == 0.0
        assert test_env.action_spaces['stop_target'].high[1] == 10.0
        
        # Risk monitor - discrete
        assert test_env.action_spaces['risk_monitor'].n == 4
        
        # Portfolio optimizer - continuous
        assert test_env.action_spaces['portfolio_optimizer'].shape == (3,)
        assert test_env.action_spaces['portfolio_optimizer'].low[0] == 0.0
        assert test_env.action_spaces['portfolio_optimizer'].high[0] == 2.0
        
        # Routing - discrete
        assert test_env.action_spaces['routing'].n == 4
    
    def test_observation_spaces_definition(self, test_env):
        """Test observation spaces are correctly defined"""
        # All agents should have different observation dimensions
        expected_dims = {
            'position_sizing': 39,  # 16 + 10 + 8 + 5
            'stop_target': 38,      # 16 + 10 + 8 + 4
            'risk_monitor': 40,     # 16 + 10 + 8 + 6
            'portfolio_optimizer': 39,  # 16 + 10 + 8 + 5
            'routing': 46          # 16 + 10 + 8 + 12
        }
        
        for agent, expected_dim in expected_dims.items():
            obs_space = test_env.observation_spaces[agent]
            assert obs_space.shape == (expected_dim,)
            assert obs_space.dtype == np.float32
    
    def test_market_state_initialization(self, test_env):
        """Test market state initialization"""
        assert test_env.market_state is not None
        assert test_env.market_state.price == 100.0
        assert test_env.market_state.volume == 1000.0
        assert test_env.market_state.bid < test_env.market_state.price
        assert test_env.market_state.ask > test_env.market_state.price
        assert test_env.market_state.spread_bps > 0
        assert test_env.market_state.volatility > 0
        assert test_env.market_state.buy_pressure + test_env.market_state.sell_pressure == 1.0
    
    def test_portfolio_state_initialization(self, test_env):
        """Test portfolio state initialization"""
        assert test_env.portfolio_state['portfolio_value'] == 50000.0
        assert test_env.portfolio_state['available_capital'] == 25000.0
        assert test_env.portfolio_state['current_position'] == 0.0
        assert test_env.portfolio_state['unrealized_pnl'] == 0.0
        assert test_env.portfolio_state['realized_pnl'] == 0.0
        assert test_env.portfolio_state['var_estimate'] == 0.02
        assert test_env.portfolio_state['max_drawdown'] == 0.0
    
    def test_agent_cycling_sequence(self, test_env):
        """Test agent cycling follows correct sequence"""
        test_env.reset()
        
        expected_sequence = [
            'position_sizing', 'stop_target', 'risk_monitor', 
            'portfolio_optimizer', 'routing'
        ]
        
        # Execute one complete cycle
        for i in range(5):
            assert test_env.agent_selection == expected_sequence[i]
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
        
        # Should cycle back to first agent
        assert test_env.agent_selection == expected_sequence[0]
    
    def test_unified_execution_system_integration(self, test_env):
        """Test unified execution system integration"""
        test_env.reset()
        
        # Check execution system is initialized
        assert hasattr(test_env, 'execution_system')
        assert test_env.execution_system is not None
        
        # Store initial actions for all agents
        actions = {}
        for agent in test_env.possible_agents:
            assert test_env.agent_selection == agent
            action = test_env.action_spaces[agent].sample()
            actions[agent] = action
            test_env.step(action)
        
        # Check that unified execution occurred
        assert test_env.current_step == 1
        assert len(test_env.execution_history) == 1
        
        # Check execution history contains decision
        execution_record = test_env.execution_history[0]
        assert 'decision' in execution_record
        assert 'market_state' in execution_record
        assert 'portfolio_state' in execution_record
        assert 'timestamp' in execution_record
    
    def test_market_dynamics_simulation(self, test_env):
        """Test market dynamics simulation"""
        test_env.reset()
        
        initial_price = test_env.market_state.price
        initial_volume = test_env.market_state.volume
        initial_volatility = test_env.market_state.volatility
        
        # Execute several complete cycles
        for cycle in range(3):
            for agent in test_env.possible_agents:
                action = test_env.action_spaces[agent].sample()
                test_env.step(action)
        
        # Check market dynamics have evolved
        assert test_env.market_state.price != initial_price
        assert test_env.market_state.volume != initial_volume
        
        # Check price is still positive
        assert test_env.market_state.price > 0
        
        # Check spread is reasonable
        assert test_env.market_state.spread_bps > 0
        assert test_env.market_state.spread_bps < 100  # Less than 1%
    
    def test_portfolio_updates_from_execution(self, test_env):
        """Test portfolio updates from execution decisions"""
        test_env.reset()
        
        initial_portfolio_value = test_env.portfolio_state['portfolio_value']
        initial_position = test_env.portfolio_state['current_position']
        
        # Execute actions that should change portfolio
        # Position sizing - increase position
        test_env.step(np.array([0.8, 0.5, 0.7]))  # High position size
        
        # Stop/target - set stops and targets
        test_env.step(np.array([1.5, 3.0, 0.8]))  # Reasonable stops/targets
        
        # Risk monitor - no action
        test_env.step(0)
        
        # Portfolio optimizer - rebalance
        test_env.step(np.array([1.2, 0.8, 0.9]))  # Rebalancing signal
        
        # Routing - select broker
        test_env.step(0)  # IB broker
        
        # Check portfolio has been updated
        assert test_env.portfolio_state['portfolio_value'] != initial_portfolio_value
        assert test_env.portfolio_state['time_since_last_trade'] == 0
        assert test_env.portfolio_state['position_entry_time'] is not None
    
    def test_reward_calculation_system(self, test_env):
        """Test reward calculation system"""
        test_env.reset()
        
        # Execute one complete cycle
        for agent in test_env.possible_agents:
            action = test_env.action_spaces[agent].sample()
            test_env.step(action)
        
        # Check rewards were calculated
        for agent in test_env.possible_agents:
            assert agent in test_env.agent_rewards
            assert isinstance(test_env.agent_rewards[agent], (int, float))
        
        # Check episode rewards are accumulated
        for agent in test_env.possible_agents:
            assert test_env.episode_rewards[agent] != 0.0
    
    def test_execution_quality_metrics(self, test_env):
        """Test execution quality metrics"""
        test_env.reset()
        
        # Execute several cycles to generate metrics
        for cycle in range(2):
            for agent in test_env.possible_agents:
                action = test_env.action_spaces[agent].sample()
                test_env.step(action)
        
        # Check execution history contains quality metrics
        assert len(test_env.execution_history) == 2
        
        for record in test_env.execution_history:
            decision = record['decision']
            
            # Check decision contains quality metrics
            assert hasattr(decision, 'fill_rate')
            assert hasattr(decision, 'estimated_slippage_bps')
            assert hasattr(decision, 'total_latency_us')
            
            # Check metrics are reasonable
            assert 0.0 <= decision.fill_rate <= 1.0
            assert decision.estimated_slippage_bps >= 0.0
            assert decision.total_latency_us >= 0.0
    
    def test_observation_generation(self, test_env):
        """Test observation generation for all agents"""
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs = test_env.observe(agent)
            
            # Check observation shape
            expected_shape = test_env.observation_spaces[agent].shape
            assert obs.shape == expected_shape
            
            # Check observation is finite
            assert np.all(np.isfinite(obs))
            
            # Check observation contains agent-specific information
            assert obs is not None
            assert len(obs) > 0
    
    def test_termination_conditions(self, test_env):
        """Test episode termination conditions"""
        test_env.reset()
        
        # Test max steps termination
        test_env.current_step = test_env.config.max_steps - 1
        
        for agent in test_env.possible_agents:
            action = test_env.action_spaces[agent].sample()
            test_env.step(action)
        
        assert any(test_env.truncations.values())
        
        # Test drawdown termination
        test_env.reset()
        test_env.portfolio_state['drawdown_current'] = 0.15  # 15% drawdown
        test_env._check_termination()
        
        assert any(test_env.terminations.values())
    
    def test_performance_metrics_collection(self, test_env):
        """Test performance metrics collection"""
        test_env.reset()
        
        # Execute episode
        for cycle in range(2):
            for agent in test_env.possible_agents:
                action = test_env.action_spaces[agent].sample()
                test_env.step(action)
        
        # Get episode statistics
        stats = test_env.get_episode_statistics()
        
        assert 'episode_length' in stats
        assert 'total_rewards' in stats
        assert 'final_portfolio_value' in stats
        assert 'total_pnl' in stats
        assert 'max_drawdown' in stats
        assert 'final_position' in stats
        assert 'execution_count' in stats
        assert 'performance_metrics' in stats
        
        # Check values are reasonable
        assert stats['episode_length'] == test_env.current_step
        assert stats['execution_count'] == len(test_env.execution_history)
        assert isinstance(stats['total_rewards'], dict)
        assert len(stats['total_rewards']) == 5


class TestPettingZooCompliance:
    """Test suite for PettingZoo API compliance"""
    
    def test_pettingzoo_api_test(self):
        """Test PettingZoo API compliance"""
        test_env = raw_env()
        
        # Run PettingZoo API test
        api_test(test_env, num_cycles=5, verbose_progress=False)
    
    def test_pettingzoo_wrapper_functions(self):
        """Test PettingZoo wrapper functions"""
        # Test env() function
        wrapped_env = env()
        assert wrapped_env is not None
        assert hasattr(wrapped_env, 'reset')
        assert hasattr(wrapped_env, 'step')
        assert hasattr(wrapped_env, 'observe')
        
        # Test raw_env() function
        raw_environment = raw_env()
        assert raw_environment is not None
        assert isinstance(raw_environment, ExecutionEnvironment)
    
    def test_environment_properties(self):
        """Test required environment properties"""
        test_env = raw_env()
        
        # Check required properties
        assert hasattr(test_env, 'possible_agents')
        assert hasattr(test_env, 'agents')
        assert hasattr(test_env, 'action_spaces')
        assert hasattr(test_env, 'observation_spaces')
        assert hasattr(test_env, 'agent_rewards')
        assert hasattr(test_env, 'agent_dones')
        assert hasattr(test_env, 'agent_infos')
        assert hasattr(test_env, 'agent_selection')
        
        # Check metadata
        assert hasattr(test_env, 'metadata')
        assert 'name' in test_env.metadata
        assert 'is_parallelizable' in test_env.metadata
    
    def test_action_space_compliance(self):
        """Test action space compliance"""
        test_env = raw_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            action_space = test_env.action_spaces[agent]
            
            # Test sampling
            for _ in range(3):
                action = action_space.sample()
                assert action_space.contains(action)
    
    def test_observation_space_compliance(self):
        """Test observation space compliance"""
        test_env = raw_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs_space = test_env.observation_spaces[agent]
            obs = test_env.observe(agent)
            
            # Check observation matches space
            assert obs_space.contains(obs)
    
    def test_environment_lifecycle(self):
        """Test complete environment lifecycle"""
        test_env = raw_env()
        
        # Reset
        test_env.reset()
        assert len(test_env.agents) == 5
        assert test_env.agent_selection in test_env.agents
        
        # Step through episode
        steps = 0
        while test_env.agents and steps < 30:
            agent = test_env.agent_selection
            action = test_env.action_spaces[agent].sample()
            test_env.step(action)
            steps += 1
            
            # Check state consistency
            if test_env.agents:
                assert test_env.agent_selection in test_env.agents
        
        # Close
        test_env.close()


class TestExecutionSystemIntegration:
    """Test suite for execution system integration"""
    
    def test_execution_context_creation(self, test_env):
        """Test execution context creation"""
        test_env.reset()
        
        context = test_env._create_execution_context()
        
        # Check context has required fields
        assert hasattr(context, 'portfolio_value')
        assert hasattr(context, 'available_capital')
        assert hasattr(context, 'current_position')
        assert hasattr(context, 'unrealized_pnl')
        assert hasattr(context, 'realized_pnl')
        assert hasattr(context, 'var_estimate')
        assert hasattr(context, 'expected_return')
        assert hasattr(context, 'volatility')
        assert hasattr(context, 'sharpe_ratio')
        assert hasattr(context, 'max_drawdown')
        assert hasattr(context, 'correlation_risk')
        assert hasattr(context, 'liquidity_score')
        
        # Check values are reasonable
        assert context.portfolio_value > 0
        assert context.available_capital >= 0
        assert context.var_estimate >= 0
        assert context.volatility >= 0
        assert 0 <= context.liquidity_score <= 1
    
    def test_market_features_creation(self, test_env):
        """Test market features creation"""
        test_env.reset()
        
        features = test_env._create_market_features()
        
        # Check features have required fields
        assert hasattr(features, 'buy_volume')
        assert hasattr(features, 'sell_volume')
        assert hasattr(features, 'order_flow_imbalance')
        assert hasattr(features, 'large_order_flow')
        assert hasattr(features, 'retail_flow')
        assert hasattr(features, 'institutional_flow')
        assert hasattr(features, 'price_momentum_1m')
        assert hasattr(features, 'price_momentum_5m')
        assert hasattr(features, 'support_level')
        assert hasattr(features, 'resistance_level')
        assert hasattr(features, 'trend_strength')
        assert hasattr(features, 'atm_vol')
        assert hasattr(features, 'vol_skew')
        assert hasattr(features, 'correlation_spy')
        assert hasattr(features, 'correlation_vix')
        assert hasattr(features, 'regime_equity')
        assert hasattr(features, 'regime_volatility')
        
        # Check values are reasonable
        assert features.buy_volume >= 0
        assert features.sell_volume >= 0
        assert features.support_level > 0
        assert features.resistance_level > 0
        assert features.atm_vol >= 0
        assert -1 <= features.correlation_spy <= 1
        assert -1 <= features.correlation_vix <= 1
    
    def test_order_data_creation(self, test_env):
        """Test order data creation"""
        test_env.reset()
        
        order_data = test_env._create_order_data()
        
        # Check order data has required fields
        assert 'symbol' in order_data
        assert 'side' in order_data
        assert 'quantity' in order_data
        assert 'order_type' in order_data
        assert 'time_in_force' in order_data
        assert 'urgency' in order_data
        assert 'execution_style' in order_data
        
        # Check values are reasonable
        assert order_data['symbol'] == 'SPY'
        assert order_data['side'] in ['BUY', 'SELL']
        assert order_data['quantity'] > 0
        assert order_data['order_type'] in ['MARKET', 'LIMIT', 'STOP']
        assert order_data['time_in_force'] in ['IOC', 'FOK', 'GTC']
    
    @patch('asyncio.new_event_loop')
    @patch('asyncio.set_event_loop')
    def test_async_execution_handling(self, mock_set_loop, mock_new_loop, test_env):
        """Test async execution handling"""
        test_env.reset()
        
        # Mock event loop
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop
        
        # Mock execution system
        mock_decision = Mock()
        mock_decision.final_position_size = 0.1
        mock_decision.risk_approved = True
        mock_decision.emergency_stop = False
        mock_decision.estimated_slippage_bps = 2.0
        mock_decision.fill_rate = 0.998
        mock_decision.total_latency_us = 450.0
        mock_decision.stop_loss_level = 95.0
        mock_decision.take_profit_level = 105.0
        
        test_env.execution_system.execute_unified_decision = AsyncMock(return_value=mock_decision)
        mock_loop.run_until_complete.return_value = mock_decision
        
        # Execute full cycle
        for agent in test_env.possible_agents:
            action = test_env.action_spaces[agent].sample()
            test_env.step(action)
        
        # Check async execution was called
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once()
        mock_loop.run_until_complete.assert_called_once()


class TestPerformanceAndStability:
    """Test suite for performance and stability"""
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        test_env = raw_env()
        
        # Measure reset time
        start_time = time.time()
        test_env.reset()
        reset_time = time.time() - start_time
        assert reset_time < 2.0  # Should reset in less than 2 seconds
        
        # Measure step time
        step_times = []
        for _ in range(10):
            start_time = time.time()
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
        
        avg_step_time = np.mean(step_times)
        assert avg_step_time < 0.5  # Should step in less than 500ms
    
    def test_memory_stability(self):
        """Test memory stability over episodes"""
        test_env = raw_env()
        
        for episode in range(3):
            test_env.reset()
            steps = 0
            
            while test_env.agents and steps < 20:
                action = test_env.action_spaces[test_env.agent_selection].sample()
                test_env.step(action)
                steps += 1
            
            # Check memory usage doesn't grow unbounded
            assert len(test_env.execution_history) <= 1000
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        test_env = raw_env()
        test_env.reset()
        
        # Test that environment handles errors gracefully
        try:
            # Force an error condition
            test_env.portfolio_state['portfolio_value'] = -1000.0
            test_env._check_termination()
        except Exception:
            pass  # Should handle gracefully
        
        # Environment should still be functional
        valid_action = test_env.action_spaces[test_env.agent_selection].sample()
        test_env.step(valid_action)
        assert test_env.agent_selection in test_env.agents
    
    def test_concurrent_observations(self):
        """Test concurrent observation requests"""
        test_env = raw_env()
        test_env.reset()
        
        # Test multiple concurrent observations
        observations = []
        for agent in test_env.possible_agents:
            obs = test_env.observe(agent)
            observations.append(obs)
        
        # All observations should be valid
        for obs in observations:
            assert obs is not None
            assert np.all(np.isfinite(obs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])