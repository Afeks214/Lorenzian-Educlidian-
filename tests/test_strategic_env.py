"""
Comprehensive tests for Strategic Market Environment

Tests the PettingZoo environment implementation including:
- Observation space correctness
- State machine logic
- Agent turn sequencing
- Synergy integration
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.strategic_env import StrategicMarketEnv, EnvironmentState


class TestStrategicEnvironment:
    """Test suite for Strategic Market Environment"""
    
    @pytest.fixture
    def env(self):
        """Create environment instance for testing"""
        config = {
            "matrix_shape": [48, 13],
            "max_timesteps": 100,
            "feature_indices": {
                "mlmi_expert": [0, 1, 9, 10],
                "nwrqk_expert": [2, 3, 4, 5],
                "regime_expert": [10, 11, 12],
            }
        }
        return StrategicMarketEnv(config)
    
    def test_environment_initialization(self, env):
        """Test environment initializes correctly"""
        assert env is not None
        assert len(env.possible_agents) == 3
        assert set(env.possible_agents) == {"mlmi_expert", "nwrqk_expert", "regime_expert"}
        assert env.state == EnvironmentState.AWAITING_MLMI
    
    def test_observation_space(self, env):
        """Verify correct feature extraction for each agent"""
        env.reset()
        
        # Test MLMI expert observation
        obs_mlmi = env.observe('mlmi_expert')
        assert obs_mlmi['features'].shape == (4,)  # 4 features for MLMI
        assert obs_mlmi['shared_context'].shape == (6,)
        assert obs_mlmi['synergy_active'] in [0, 1]
        assert 0 <= obs_mlmi['synergy_type'] <= 4
        
        # Test NWRQK expert observation
        obs_nwrqk = env.observe('nwrqk_expert')
        assert obs_nwrqk['features'].shape == (4,)  # 4 features for NWRQK
        
        # Test Regime expert observation
        obs_regime = env.observe('regime_expert')
        assert obs_regime['features'].shape == (3,)  # 3 features for Regime
    
    def test_feature_extraction_correctness(self, env):
        """Test that correct features are extracted for each agent"""
        env.reset()
        
        # Create known matrix
        test_matrix = np.arange(48 * 13).reshape(48, 13).astype(np.float32)
        env.current_matrix = test_matrix
        
        # Expected features (averaged across time dimension)
        matrix_avg = np.mean(test_matrix, axis=0)
        
        # Test MLMI features
        obs_mlmi = env.observe('mlmi_expert')
        expected_mlmi = matrix_avg[[0, 1, 9, 10]]
        np.testing.assert_array_almost_equal(obs_mlmi['features'], expected_mlmi)
        
        # Test NWRQK features  
        obs_nwrqk = env.observe('nwrqk_expert')
        expected_nwrqk = matrix_avg[[2, 3, 4, 5]]
        np.testing.assert_array_almost_equal(obs_nwrqk['features'], expected_nwrqk)
        
        # Test Regime features
        obs_regime = env.observe('regime_expert')
        expected_regime = matrix_avg[[10, 11, 12]]
        np.testing.assert_array_almost_equal(obs_regime['features'], expected_regime)
    
    def test_state_machine_transitions(self, env):
        """Test state machine advances correctly"""
        env.reset()
        
        # Initial state
        assert env.state == EnvironmentState.AWAITING_MLMI
        
        # Step with MLMI action
        env.step(np.array([0.3, 0.4, 0.3]))
        assert env.state == EnvironmentState.AWAITING_NWRQK
        
        # Step with NWRQK action
        env.step(np.array([0.2, 0.5, 0.3]))
        assert env.state == EnvironmentState.AWAITING_REGIME
        
        # Step with Regime action
        env.step(np.array([0.1, 0.2, 0.7]))
        assert env.state == EnvironmentState.AWAITING_MLMI  # Reset after aggregation
    
    def test_agent_turn_sequence(self, env):
        """Test complete agent turn sequence"""
        env.reset()
        
        agents_acted = []
        action_count = 0
        
        # Run for exactly 3 steps (one complete cycle)
        for agent in env.agent_iter():
            agents_acted.append(agent)
            env.step(np.array([0.33, 0.34, 0.33]))
            action_count += 1
            if action_count >= 3:
                break
        
        # Verify correct sequence
        assert agents_acted == ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
    
    def test_superposition_storage(self, env):
        """Test that agent outputs are stored correctly"""
        env.reset()
        
        # Define test actions
        test_actions = {
            'mlmi_expert': np.array([0.7, 0.2, 0.1]),
            'nwrqk_expert': np.array([0.1, 0.8, 0.1]),
            'regime_expert': np.array([0.3, 0.3, 0.4])
        }
        
        # Step through agents
        for i, agent in enumerate(env.agent_iter()):
            if i >= 3:
                break
            action = test_actions[agent]
            env.step(action)
            
            # Check action is stored (before aggregation)
            if i < 2:  # Not yet aggregated
                assert agent in env.agent_outputs
                np.testing.assert_array_almost_equal(env.agent_outputs[agent], action)
        
        # After aggregation, outputs should be cleared
        assert len(env.agent_outputs) == 0
    
    def test_action_normalization(self, env):
        """Test that actions are normalized to valid probabilities"""
        env.reset()
        
        # Test with unnormalized action
        unnormalized_action = np.array([1.0, 2.0, 3.0])
        env.step(unnormalized_action)
        
        # Check stored action is normalized
        stored_action = env.agent_outputs[env.possible_agents[0]]
        assert np.allclose(stored_action.sum(), 1.0)
        assert np.all(stored_action >= 0)
        assert np.all(stored_action <= 1)
    
    def test_synergy_state_integration(self, env):
        """Test synergy detection integration"""
        env.reset()
        
        # Mock synergy info
        env.synergy_info = {
            "type": "TYPE_2",
            "confidence": 0.85,
            "direction": 1
        }
        
        # Check observation includes synergy
        obs = env.observe('mlmi_expert')
        assert obs['synergy_active'] == 1
        assert obs['synergy_type'] == 2  # TYPE_2 -> 2
        
        # Check shared context includes synergy confidence
        assert obs['shared_context'][0] == 0.85  # Confidence
        assert obs['shared_context'][1] == 1.0   # Direction
    
    def test_invalid_action_handling(self, env):
        """Test handling of invalid actions"""
        env.reset()
        
        # Test wrong shape
        with pytest.raises(ValueError, match="must be numpy array of shape"):
            env.step(np.array([0.5, 0.5]))  # Wrong shape
        
        # Test wrong type
        with pytest.raises(ValueError, match="must be numpy array of shape"):
            env.step([0.3, 0.3, 0.4])  # List instead of array
    
    def test_episode_termination(self, env):
        """Test episode termination conditions"""
        env.reset()
        env.timestep = env.config["max_timesteps"] - 1
        
        # Run through one complete cycle
        for i, agent in enumerate(env.agent_iter()):
            if i >= 3:
                break
            env.step(np.array([0.33, 0.34, 0.33]))
        
        # Check terminations are set
        assert all(env.terminations.values())
    
    def test_performance_tracking(self, env):
        """Test inference time tracking"""
        env.reset()
        
        # Perform some steps
        for i in range(6):  # Two complete cycles
            agent = env.agents[i % 3]
            env.agent_selection = agent
            env.step(np.array([0.33, 0.34, 0.33]))
        
        # Check inference times are tracked
        assert len(env.inference_times) == 6
        assert all(t > 0 for t in env.inference_times)
        assert all(t < 1000 for t in env.inference_times)  # Less than 1 second
    
    def test_aggregation_trigger(self, env):
        """Test that aggregation is triggered after all agents act"""
        env.reset()
        
        # Track whether aggregation occurred
        aggregation_triggered = False
        original_perform_aggregation = env._perform_aggregation
        
        def mock_aggregation():
            nonlocal aggregation_triggered
            aggregation_triggered = True
            original_perform_aggregation()
        
        env._perform_aggregation = mock_aggregation
        
        # Step through complete cycle
        for i, agent in enumerate(env.agent_iter()):
            if i >= 3:
                break
            env.step(np.array([0.33, 0.34, 0.33]))
        
        # Verify aggregation was triggered
        assert aggregation_triggered
    
    def test_market_context_features(self, env):
        """Test market context in shared features"""
        env.reset()
        
        # Set known market context
        env.market_context = {
            "volatility_30": 1.5,
            "volume_profile_skew": 0.3
        }
        
        obs = env.observe('mlmi_expert')
        shared_context = obs['shared_context']
        
        # Check volatility is log-normalized
        expected_vol = np.log(1.5)
        assert np.isclose(shared_context[2], expected_vol)
    
    def test_empty_observation_handling(self, env):
        """Test handling when matrix is None"""
        env.reset()
        env.current_matrix = None
        
        # Should return empty observation
        obs = env.observe('mlmi_expert')
        assert np.all(obs['features'] == 0)
        assert np.all(obs['shared_context'] == 0)
        assert obs['synergy_active'] == 0
        assert obs['synergy_type'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])