"""
Comprehensive tests for Strategic Market Environment

Tests the PettingZoo environment implementation including:
- PettingZoo API compliance
- Observation space correctness
- State machine logic
- Agent turn sequencing
- Synergy integration
- Performance and error handling
"""

import pytest
import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.strategic_env import (
    StrategicMARLEnvironment, 
    EnvironmentPhase, 
    EnvironmentState,
    env, 
    raw_env
)
from pettingzoo.test import api_test, parallel_test


class TestStrategicEnvironment:
    """Test suite for Strategic Market Environment"""
    
    @pytest.fixture
    def test_env(self):
        """Create environment instance for testing"""
        config = {
            "matrix_shape": [48, 13],
            "max_episode_steps": 100,
            "total_episodes": 10,
            "feature_indices": {
                "mlmi_expert": [0, 1, 9, 10],
                "nwrqk_expert": [2, 3, 4, 5],
                "regime_expert": [10, 11, 12],
            }
        }
        return StrategicMARLEnvironment(config)
    
    def test_environment_initialization(self, test_env):
        """Test environment initializes correctly"""
        assert test_env is not None
        assert len(test_env.possible_agents) == 3
        assert set(test_env.possible_agents) == {"mlmi_expert", "nwrqk_expert", "regime_expert"}
        assert test_env.env_state.phase == EnvironmentPhase.SETUP
    
    def test_observation_space(self, test_env):
        """Verify correct feature extraction for each agent"""
        test_env.reset()
        
        # Test MLMI expert observation
        obs_mlmi = test_env.observe('mlmi_expert')
        assert obs_mlmi['agent_features'].shape == (4,)  # 4 features for MLMI
        assert obs_mlmi['shared_context'].shape == (6,)
        assert obs_mlmi['market_matrix'].shape == (48, 13)
        assert 'episode_info' in obs_mlmi
        
        # Test NWRQK expert observation
        obs_nwrqk = test_env.observe('nwrqk_expert')
        assert obs_nwrqk['agent_features'].shape == (4,)  # 4 features for NWRQK
        
        # Test Regime expert observation
        obs_regime = test_env.observe('regime_expert')
        assert obs_regime['agent_features'].shape == (3,)  # 3 features for Regime
    
    def test_feature_extraction_correctness(self, test_env):
        """Test that correct features are extracted for each agent"""
        test_env.reset()
        
        # Create known matrix
        test_matrix = np.arange(48 * 13).reshape(48, 13).astype(np.float32)
        test_env.env_state.matrix_data = test_matrix
        
        # Expected features (averaged across time dimension)
        matrix_avg = np.mean(test_matrix, axis=0)
        
        # Test MLMI features
        obs_mlmi = test_env.observe('mlmi_expert')
        expected_mlmi = matrix_avg[[0, 1, 9, 10]]
        np.testing.assert_array_almost_equal(obs_mlmi['agent_features'], expected_mlmi)
        
        # Test NWRQK features  
        obs_nwrqk = test_env.observe('nwrqk_expert')
        expected_nwrqk = matrix_avg[[2, 3, 4, 5]]
        np.testing.assert_array_almost_equal(obs_nwrqk['agent_features'], expected_nwrqk)
        
        # Test Regime features
        obs_regime = test_env.observe('regime_expert')
        expected_regime = matrix_avg[[10, 11, 12]]
        np.testing.assert_array_almost_equal(obs_regime['agent_features'], expected_regime)
    
    def test_state_machine_transitions(self, test_env):
        """Test state machine advances correctly"""
        test_env.reset()
        
        # Initial state
        assert test_env.env_state.phase == EnvironmentPhase.AGENT_DECISION
        
        # Test agent selection progression
        assert test_env.agent_selection == 'mlmi_expert'
        
        # Step with MLMI action
        test_env.step(np.array([0.3, 0.4, 0.3]))
        assert test_env.agent_selection == 'nwrqk_expert'
        
        # Step with NWRQK action
        test_env.step(np.array([0.2, 0.5, 0.3]))
        assert test_env.agent_selection == 'regime_expert'
        
        # Step with Regime action
        test_env.step(np.array([0.1, 0.2, 0.7]))
        # After all agents act, should cycle back to first agent
        assert test_env.agent_selection == 'mlmi_expert'
    
    def test_agent_turn_sequence(self, test_env):
        """Test complete agent turn sequence"""
        test_env.reset()
        
        agents_acted = []
        action_count = 0
        
        # Run for exactly 3 steps (one complete cycle)
        while action_count < 3:
            agent = test_env.agent_selection
            agents_acted.append(agent)
            test_env.step(np.array([0.33, 0.34, 0.33]))
            action_count += 1
        
        # Verify correct sequence
        assert agents_acted == ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
    
    def test_agent_decisions_storage(self, test_env):
        """Test that agent decisions are stored correctly"""
        test_env.reset()
        
        # Define test actions
        test_actions = {
            'mlmi_expert': np.array([0.7, 0.2, 0.1]),
            'nwrqk_expert': np.array([0.1, 0.8, 0.1]),
            'regime_expert': np.array([0.3, 0.3, 0.4])
        }
        
        decisions_count = 0
        
        # Step through agents
        for agent_name, action in test_actions.items():
            # Ensure we're on the correct agent
            assert test_env.agent_selection == agent_name
            
            test_env.step(action)
            decisions_count += 1
            
            # Check decisions are stored (before aggregation)
            if decisions_count < 3:  # Not yet aggregated
                assert agent_name in test_env.env_state.agent_decisions
            
        # After aggregation, decisions should be cleared for next step
        assert len(test_env.env_state.agent_decisions) == 0
    
    def test_action_normalization(self, test_env):
        """Test that actions are normalized to valid probabilities"""
        test_env.reset()
        
        # Test with unnormalized action
        unnormalized_action = np.array([1.0, 2.0, 3.0])
        test_env.step(unnormalized_action)
        
        # Check normalized action is stored
        agent_name = test_env.possible_agents[0]
        if agent_name in test_env.env_state.agent_decisions:
            stored_prediction = test_env.env_state.agent_decisions[agent_name]
            assert np.allclose(stored_prediction.action_probabilities.sum(), 1.0)
            assert np.all(stored_prediction.action_probabilities >= 0)
            assert np.all(stored_prediction.action_probabilities <= 1)
    
    def test_shared_context_integration(self, test_env):
        """Test shared context integration"""
        test_env.reset()
        
        # Check observation includes shared context
        obs = test_env.observe('mlmi_expert')
        assert 'shared_context' in obs
        assert obs['shared_context'].shape == (6,)
        
        # Check shared context has market metrics
        shared_context = obs['shared_context']
        assert len(shared_context) == 6
        assert all(isinstance(val, (int, float, np.floating)) for val in shared_context)
    
    def test_invalid_action_handling(self, test_env):
        """Test handling of invalid actions"""
        test_env.reset()
        
        # Test wrong shape
        with pytest.raises(ValueError, match="Action must be shape"):
            test_env.step(np.array([0.5, 0.5]))  # Wrong shape
        
        # Test action space validation
        current_agent = test_env.agent_selection
        action_space = test_env.action_spaces[current_agent]
        
        # Valid action should work
        valid_action = np.array([0.3, 0.4, 0.3])
        assert action_space.contains(valid_action)
        
        # Invalid action should be rejected
        invalid_action = np.array([1.5, -0.3, 0.8])  # Values outside [0,1]
        assert not action_space.contains(invalid_action)
    
    def test_episode_termination(self, test_env):
        """Test episode termination conditions"""
        test_env.reset()
        test_env.env_state.episode_step = test_env.max_episode_steps - 1
        
        # Run through one complete cycle
        for i in range(3):
            test_env.step(np.array([0.33, 0.34, 0.33]))
        
        # Check terminations are set
        assert all(test_env.terminations.values())
    
    def test_performance_tracking(self, test_env):
        """Test performance metrics tracking"""
        test_env.reset()
        
        # Perform some steps
        for i in range(6):  # Two complete cycles
            test_env.step(np.array([0.33, 0.34, 0.33]))
        
        # Check performance metrics are tracked
        metrics = test_env.get_performance_metrics()
        assert 'avg_decision_time_ms' in metrics
        assert 'avg_ensemble_confidence' in metrics
        assert metrics['avg_decision_time_ms'] >= 0
        assert len(test_env.performance_metrics['agent_decision_times']) > 0
    
    def test_ensemble_decision_processing(self, test_env):
        """Test that ensemble decisions are processed after all agents act"""
        test_env.reset()
        
        # Step through complete cycle
        for i in range(3):
            test_env.step(np.array([0.33, 0.34, 0.33]))
        
        # Check that ensemble decision was processed
        assert test_env.env_state.shared_context is not None
        if 'ensemble_decision' in test_env.env_state.shared_context:
            ensemble_decision = test_env.env_state.shared_context['ensemble_decision']
            assert 'probabilities' in ensemble_decision
            assert 'confidence' in ensemble_decision
            assert 'action' in ensemble_decision
    
    def test_market_context_features(self, test_env):
        """Test market context in shared features"""
        test_env.reset()
        
        # Update market state
        test_env._update_market_state()
        
        obs = test_env.observe('mlmi_expert')
        shared_context = obs['shared_context']
        
        # Check shared context contains market features
        assert len(shared_context) == 6
        assert all(isinstance(val, (int, float, np.floating)) for val in shared_context)
        
        # Check market state exists
        assert test_env.env_state.matrix_data is not None
        assert test_env.env_state.shared_context is not None
    
    def test_empty_observation_handling(self, test_env):
        """Test handling when matrix is None"""
        test_env.reset()
        test_env.env_state.matrix_data = None
        
        # Should return empty observation
        obs = test_env.observe('mlmi_expert')
        assert obs['agent_features'] is not None
        assert obs['shared_context'] is not None
        assert obs['market_matrix'] is not None
        assert obs['episode_info'] is not None


class TestPettingZooCompliance:
    """Test suite for PettingZoo API compliance"""
    
    def test_pettingzoo_api_test(self):
        """Test PettingZoo API compliance"""
        # Create environment with default config
        test_env = raw_env()
        
        # Run PettingZoo API test
        api_test(test_env, num_cycles=10, verbose_progress=False)
    
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
        assert isinstance(raw_environment, StrategicMARLEnvironment)
    
    def test_action_space_compliance(self):
        """Test action space compliance"""
        test_env = raw_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            action_space = test_env.action_spaces[agent]
            assert action_space.shape == (3,)
            assert action_space.dtype == np.float32
            
            # Test sample
            sample_action = action_space.sample()
            assert action_space.contains(sample_action)
    
    def test_observation_space_compliance(self):
        """Test observation space compliance"""
        test_env = raw_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs_space = test_env.observation_spaces[agent]
            assert isinstance(obs_space, dict)
            
            # Test observation
            obs = test_env.observe(agent)
            assert obs_space.contains(obs)
    
    def test_environment_properties(self):
        """Test required environment properties"""
        test_env = raw_env()
        
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
    
    def test_environment_lifecycle(self):
        """Test complete environment lifecycle"""
        test_env = raw_env()
        
        # Initial state
        assert test_env.agents == []
        
        # Reset
        test_env.reset()
        assert len(test_env.agents) == 3
        assert test_env.agent_selection in test_env.agents
        
        # Step through episode
        steps = 0
        while test_env.agents and steps < 20:
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
        
        # Close
        test_env.close()


class TestPerformanceAndStability:
    """Test suite for performance and stability"""
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        test_env = raw_env()
        
        # Measure reset time
        start_time = time.time()
        test_env.reset()
        reset_time = time.time() - start_time
        assert reset_time < 1.0  # Should reset in less than 1 second
        
        # Measure step time
        step_times = []
        for _ in range(10):
            start_time = time.time()
            action = test_env.action_spaces[test_env.agent_selection].sample()
            test_env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
        
        avg_step_time = np.mean(step_times)
        assert avg_step_time < 0.1  # Should step in less than 100ms
    
    def test_memory_stability(self):
        """Test memory stability over multiple episodes"""
        test_env = raw_env()
        
        for episode in range(5):
            test_env.reset()
            steps = 0
            
            while test_env.agents and steps < 50:
                action = test_env.action_spaces[test_env.agent_selection].sample()
                test_env.step(action)
                steps += 1
            
            # Check memory usage doesn't grow unbounded
            assert len(test_env.performance_metrics['agent_decision_times']) <= 1000
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with same seed"""
        config = {"seed": 42}
        
        # Run environment twice with same seed
        results1 = []
        test_env1 = raw_env(config)
        test_env1.reset(seed=42)
        
        for _ in range(6):  # Two complete cycles
            action = np.array([0.5, 0.3, 0.2])
            test_env1.step(action)
            results1.append(test_env1.rewards.copy())
        
        results2 = []
        test_env2 = raw_env(config)
        test_env2.reset(seed=42)
        
        for _ in range(6):  # Two complete cycles
            action = np.array([0.5, 0.3, 0.2])
            test_env2.step(action)
            results2.append(test_env2.rewards.copy())
        
        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            for agent in r1:
                assert np.isclose(r1[agent], r2[agent], atol=1e-6)
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        test_env = raw_env()
        test_env.reset()
        
        # Test recovery from invalid actions
        try:
            test_env.step(np.array([100, -50, 25]))  # Invalid action
        except ValueError:
            pass  # Expected
        
        # Environment should still be functional
        valid_action = np.array([0.33, 0.34, 0.33])
        test_env.step(valid_action)
        assert test_env.agent_selection in test_env.agents
    
    def test_concurrent_access(self):
        """Test thread safety (basic)"""
        test_env = raw_env()
        test_env.reset()
        
        # Test that multiple observations don't interfere
        obs1 = test_env.observe('mlmi_expert')
        obs2 = test_env.observe('mlmi_expert')
        
        # Should be identical
        assert np.array_equal(obs1['agent_features'], obs2['agent_features'])
        assert np.array_equal(obs1['shared_context'], obs2['shared_context'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])