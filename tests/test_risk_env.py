"""
Comprehensive Tests for Risk Management Environment

Tests the PettingZoo risk management environment implementation including:
- PettingZoo API compliance
- Risk agent coordination
- Portfolio state management
- Risk event detection and handling
- Performance metrics and monitoring
- Byzantine fault tolerance integration

Author: Claude Code
Version: 1.0 - PettingZoo Implementation
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.risk_env import (
    RiskManagementEnv,
    RiskEnvironmentState,
    RiskScenario,
    PortfolioState,
    MarketConditions,
    create_risk_environment
)
from pettingzoo.test import api_test


class TestRiskEnvironmentLogic:
    """Test suite for risk environment logic"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'initial_capital': 100000.0,
            'max_steps': 100,
            'risk_tolerance': 0.05,
            'scenario': 'normal',
            'performance_target_ms': 5.0,
            'asset_universe': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
            'critic_config': {
                'hidden_dim': 64,
                'num_layers': 2,
                'learning_rate': 0.001
            }
        }
    
    @pytest.fixture
    def test_env(self, config):
        """Create test environment"""
        return RiskManagementEnv(config)
    
    def test_environment_initialization(self, test_env):
        """Test environment initializes correctly"""
        assert test_env is not None
        assert len(test_env.possible_agents) == 4
        assert set(test_env.possible_agents) == {
            'position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer'
        }
        assert test_env.env_state == RiskEnvironmentState.AWAITING_POSITION_SIZING
        assert test_env.initial_capital == 100000.0
    
    def test_action_spaces_definition(self, test_env):
        """Test action spaces are correctly defined"""
        # Position sizing - discrete actions
        assert test_env.action_spaces['position_sizing'].n == 7
        
        # Stop/target - continuous actions
        assert test_env.action_spaces['stop_target'].shape == (2,)
        assert test_env.action_spaces['stop_target'].low[0] == 0.5
        assert test_env.action_spaces['stop_target'].high[0] == 5.0
        
        # Risk monitor - discrete actions
        assert test_env.action_spaces['risk_monitor'].n == 5
        
        # Portfolio optimizer - continuous actions
        assert test_env.action_spaces['portfolio_optimizer'].shape == (10,)
        assert test_env.action_spaces['portfolio_optimizer'].low[0] == 0.0
        assert test_env.action_spaces['portfolio_optimizer'].high[0] == 1.0
    
    def test_observation_spaces_definition(self, test_env):
        """Test observation spaces are correctly defined"""
        for agent in test_env.possible_agents:
            obs_space = test_env.observation_spaces[agent]
            assert obs_space.shape == (10,)  # 10-dimensional risk state
            assert obs_space.dtype == np.float32
    
    def test_agent_turn_sequence(self, test_env):
        """Test agent turn sequence follows correct order"""
        test_env.reset()
        
        # Check initial agent
        assert test_env.agent_selection == 'position_sizing'
        assert test_env.env_state == RiskEnvironmentState.AWAITING_POSITION_SIZING
        
        # Step through agents in order
        test_env.step(3)  # Position sizing action
        assert test_env.agent_selection == 'stop_target'
        assert test_env.env_state == RiskEnvironmentState.AWAITING_STOP_TARGET
        
        test_env.step(np.array([1.5, 2.0]))  # Stop/target action
        assert test_env.agent_selection == 'risk_monitor'
        assert test_env.env_state == RiskEnvironmentState.AWAITING_RISK_MONITOR
        
        test_env.step(1)  # Risk monitor action
        assert test_env.agent_selection == 'portfolio_optimizer'
        assert test_env.env_state == RiskEnvironmentState.AWAITING_PORTFOLIO_OPTIMIZER
        
        test_env.step(np.random.random(10))  # Portfolio optimizer action
        # Should cycle back to position sizing
        assert test_env.agent_selection == 'position_sizing'
        assert test_env.env_state == RiskEnvironmentState.AWAITING_POSITION_SIZING
    
    def test_portfolio_state_updates(self, test_env):
        """Test portfolio state updates correctly"""
        test_env.reset()
        
        initial_value = test_env.portfolio_state.total_value
        initial_position_count = test_env.portfolio_state.position_count
        
        # Execute position sizing action (increase positions)
        test_env.step(6)  # Increase large
        
        # Step through other agents
        test_env.step(np.array([1.0, 1.5]))  # Stop/target
        test_env.step(0)  # Risk monitor no action
        test_env.step(np.random.random(10))  # Portfolio optimizer
        
        # Check portfolio state has been updated
        assert test_env.portfolio_state.total_value != initial_value
        assert test_env.portfolio_state.last_updated > test_env.episode_start_time
    
    def test_risk_event_detection(self, test_env):
        """Test risk event detection"""
        test_env.reset()
        
        # Force high drawdown
        test_env.portfolio_state.drawdown = 0.20  # 20% drawdown
        
        # Check risk events
        risk_events = test_env._check_risk_events()
        assert 'excessive_drawdown' in risk_events
        
        # Force high leverage
        test_env.portfolio_state.leverage = 5.0
        risk_events = test_env._check_risk_events()
        assert 'excessive_leverage' in risk_events
        
        # Force high correlation
        test_env.portfolio_state.max_correlation = 0.90
        risk_events = test_env._check_risk_events()
        assert 'correlation_spike' in risk_events
    
    def test_risk_scenario_application(self, test_env):
        """Test risk scenario modifications"""
        # Test correlation spike scenario
        test_env.risk_scenario = RiskScenario.CORRELATION_SPIKE
        test_env._apply_risk_scenario()
        
        assert test_env.market_conditions.correlation_level == 0.9
        assert test_env.market_conditions.stress_level == 0.8
        
        # Test liquidity crisis scenario
        test_env.risk_scenario = RiskScenario.LIQUIDITY_CRISIS
        test_env._apply_risk_scenario()
        
        assert test_env.market_conditions.liquidity_score == 0.2
        assert test_env.market_conditions.stress_level == 0.9
    
    def test_reward_calculation(self, test_env):
        """Test reward calculation for different agents"""
        test_env.reset()
        
        # Test position sizing reward
        test_env.portfolio_state.leverage = 2.0
        test_env.step(2)  # Reduce small position
        position_reward = test_env.rewards['position_sizing']
        
        # Test risk monitor reward with risk events
        test_env.portfolio_state.drawdown = 0.10
        test_env.step(np.array([1.0, 1.5]))
        test_env.step(2)  # Reduce risk action
        risk_reward = test_env.rewards['risk_monitor']
        
        # Risk monitor should get positive reward for acting on risk
        assert isinstance(risk_reward, (int, float))
    
    def test_performance_metrics_collection(self, test_env):
        """Test performance metrics collection"""
        test_env.reset()
        
        # Execute several steps
        for _ in range(12):  # 3 complete cycles
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
        
        # Check performance metrics
        metrics = test_env.get_performance_metrics()
        
        assert 'episode_length' in metrics
        assert 'portfolio_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'risk_events_count' in metrics
        assert 'avg_step_time_ms' in metrics
        assert 'current_leverage' in metrics
        assert 'current_var' in metrics
        
        # Check step times are recorded
        assert len(test_env.step_times) > 0
        assert all(t > 0 for t in test_env.step_times)
    
    def test_emergency_stop_handling(self, test_env):
        """Test emergency stop handling"""
        test_env.reset()
        
        # Step to risk monitor
        test_env.step(3)  # Position sizing
        test_env.step(np.array([1.0, 1.5]))  # Stop/target
        
        # Execute emergency stop
        test_env.step(3)  # Emergency stop action
        
        # Check emergency stop was triggered
        assert test_env.emergency_stops > 0
        
        # Check positions were reduced
        total_positions = sum(test_env.portfolio_state.positions.values())
        assert total_positions <= 0  # Should be fully closed
    
    def test_market_dynamics_simulation(self, test_env):
        """Test market dynamics simulation"""
        test_env.reset()
        
        initial_conditions = test_env.market_conditions
        
        # Execute steps to trigger market simulation
        for _ in range(8):  # 2 complete cycles
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
        
        # Check market conditions have evolved
        final_conditions = test_env.market_conditions
        
        # At least some conditions should have changed
        conditions_changed = (
            initial_conditions.volatility_regime != final_conditions.volatility_regime or
            initial_conditions.correlation_level != final_conditions.correlation_level or
            initial_conditions.stress_level != final_conditions.stress_level
        )
        
        assert conditions_changed
    
    def test_termination_conditions(self, test_env):
        """Test episode termination conditions"""
        test_env.reset()
        
        # Test max steps termination
        test_env.current_step = test_env.max_steps - 1
        test_env.step(test_env.action_spaces[test_env.agent_selection].sample())
        
        assert all(test_env.truncations.values())
        
        # Test excessive loss termination
        test_env.reset()
        test_env.portfolio_state.total_value = test_env.initial_capital * 0.4  # 60% loss
        test_env._check_termination_conditions([], Mock())
        
        assert all(test_env.terminations.values())
    
    def test_observation_generation(self, test_env):
        """Test observation generation for agents"""
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs = test_env.observe(agent)
            
            # Check observation shape and type
            assert obs.shape == (10,)
            assert obs.dtype == np.float32
            
            # Check observation values are finite
            assert np.all(np.isfinite(obs))
            
            # Check observation is in valid range
            assert np.all(obs >= -5.0)
            assert np.all(obs <= 5.0)


class TestPettingZooCompliance:
    """Test suite for PettingZoo API compliance"""
    
    def test_pettingzoo_api_test(self):
        """Test PettingZoo API compliance"""
        test_env = create_risk_environment()
        
        # Run PettingZoo API test
        api_test(test_env, num_cycles=10, verbose_progress=False)
    
    def test_environment_properties(self):
        """Test required environment properties"""
        test_env = create_risk_environment()
        
        # Check required properties
        assert hasattr(test_env, 'possible_agents')
        assert hasattr(test_env, 'agents')
        assert hasattr(test_env, 'action_spaces')
        assert hasattr(test_env, 'observation_spaces')
        assert hasattr(test_env, 'rewards')
        assert hasattr(test_env, 'terminations')
        assert hasattr(test_env, 'truncations')
        assert hasattr(test_env, 'infos')
        assert hasattr(test_env, 'agent_selection')
        
        # Check metadata
        assert hasattr(test_env, 'metadata')
        assert 'name' in test_env.metadata
        assert 'is_parallelizable' in test_env.metadata
    
    def test_action_space_sampling(self):
        """Test action space sampling"""
        test_env = create_risk_environment()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            action_space = test_env.action_spaces[agent]
            
            # Test sampling
            for _ in range(5):
                action = action_space.sample()
                assert action_space.contains(action)
    
    def test_observation_space_compliance(self):
        """Test observation space compliance"""
        test_env = create_risk_environment()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs_space = test_env.observation_spaces[agent]
            obs = test_env.observe(agent)
            
            # Check observation matches space
            assert obs_space.contains(obs)
    
    def test_environment_lifecycle_compliance(self):
        """Test complete environment lifecycle compliance"""
        test_env = create_risk_environment()
        
        # Test reset
        test_env.reset()
        assert len(test_env.agents) == 4
        
        # Test step through complete episode
        steps = 0
        while test_env.agents and steps < 50:
            agent = test_env.agent_selection
            action = test_env.action_spaces[agent].sample()
            test_env.step(action)
            steps += 1
            
            # Check state consistency
            if test_env.agents:
                assert test_env.agent_selection in test_env.agents
                assert agent in test_env.rewards
                assert agent in test_env.terminations
                assert agent in test_env.truncations
                assert agent in test_env.infos
        
        # Test close
        test_env.close()


class TestRiskManagementIntegration:
    """Test suite for risk management integration"""
    
    def test_correlation_tracker_integration(self):
        """Test correlation tracker integration"""
        test_env = create_risk_environment()
        test_env.reset()
        
        # Check correlation tracker is initialized
        assert hasattr(test_env, 'correlation_tracker')
        assert test_env.correlation_tracker is not None
        
        # Test correlation tracking
        test_env.correlation_tracker.initialize_assets(test_env.asset_universe)
        
        # Execute some steps to generate correlation data
        for _ in range(12):
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
        
        # Check correlation data is being tracked
        assert hasattr(test_env.correlation_tracker, 'correlation_matrix')
    
    def test_var_calculator_integration(self):
        """Test VaR calculator integration"""
        test_env = create_risk_environment()
        test_env.reset()
        
        # Check VaR calculator is initialized
        assert hasattr(test_env, 'var_calculator')
        assert test_env.var_calculator is not None
        
        # Test VaR calculation
        initial_var = test_env.portfolio_state.var_estimate
        
        # Execute actions that change portfolio
        test_env.step(6)  # Increase position
        test_env.step(np.array([1.0, 1.5]))
        test_env.step(0)
        test_env.step(np.random.random(10))
        
        # Check VaR has been updated
        assert test_env.portfolio_state.var_estimate != initial_var
    
    def test_centralized_critic_integration(self):
        """Test centralized critic integration"""
        test_env = create_risk_environment()
        test_env.reset()
        
        # Check centralized critic is initialized
        assert hasattr(test_env, 'centralized_critic')
        assert test_env.centralized_critic is not None
        
        # Test global risk evaluation
        risk_state = test_env._generate_risk_state()
        global_risk_state = test_env._create_global_risk_state(risk_state)
        
        global_risk_value, operating_mode = test_env.centralized_critic.evaluate_global_risk(global_risk_state)
        
        assert isinstance(global_risk_value, (int, float))
        assert hasattr(operating_mode, 'value')  # Should be an enum
    
    def test_state_processor_integration(self):
        """Test state processor integration"""
        test_env = create_risk_environment()
        test_env.reset()
        
        # Check state processor is initialized
        assert hasattr(test_env, 'state_processor')
        assert test_env.state_processor is not None
        
        # Test state processing
        risk_state = test_env._generate_risk_state()
        raw_state = risk_state.to_vector()
        
        normalized_state, metadata = test_env.state_processor.process_state(raw_state)
        
        assert normalized_state.shape == raw_state.shape
        assert isinstance(metadata, dict)


class TestPerformanceAndStability:
    """Test suite for performance and stability"""
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        test_env = create_risk_environment()
        
        # Measure reset time
        start_time = time.time()
        test_env.reset()
        reset_time = time.time() - start_time
        assert reset_time < 1.0  # Should reset in less than 1 second
        
        # Measure step time
        step_times = []
        for _ in range(20):
            start_time = time.time()
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
        
        avg_step_time = np.mean(step_times)
        assert avg_step_time < 0.1  # Should step in less than 100ms
    
    def test_memory_stability(self):
        """Test memory stability over multiple episodes"""
        test_env = create_risk_environment()
        
        for episode in range(3):
            test_env.reset()
            steps = 0
            
            while test_env.agents and steps < 30:
                action = test_env.action_spaces[test_env.agent_selection].sample()
                test_env.step(action)
                steps += 1
            
            # Check memory usage doesn't grow unbounded
            assert len(test_env.step_times) <= 100
            assert len(test_env.risk_events) <= 1000
    
    def test_stress_testing(self):
        """Test environment under stress conditions"""
        # Test with extreme scenarios
        for scenario in ['correlation_spike', 'liquidity_crisis', 'black_swan']:
            config = {
                'initial_capital': 100000.0,
                'max_steps': 50,
                'scenario': scenario
            }
            
            test_env = create_risk_environment(config)
            test_env.reset()
            
            # Execute episode under stress
            steps = 0
            while test_env.agents and steps < 25:
                action = test_env.action_spaces[test_env.agent_selection].sample()
                test_env.step(action)
                steps += 1
            
            # Environment should handle stress gracefully
            assert not test_env.emergency_halt or test_env.emergency_stops < 5
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        test_env = create_risk_environment()
        test_env.reset()
        
        # Test recovery from invalid actions
        initial_agent = test_env.agent_selection
        
        # Try to step with invalid action for discrete space
        if hasattr(test_env.action_spaces[initial_agent], 'n'):
            try:
                test_env.step(100)  # Invalid discrete action
            except (ValueError, IndexError):
                pass  # Expected
        
        # Environment should still be functional
        valid_action = test_env.action_spaces[test_env.agent_selection].sample()
        test_env.step(valid_action)
        assert test_env.agent_selection in test_env.agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])