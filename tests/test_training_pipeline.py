"""
Integration tests for the complete training pipeline

Tests end-to-end functionality with all components working together.
"""

import pytest
import numpy as np
import sys
import tempfile
import shutil
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.strategic_env import StrategicMarketEnv
from components.decision_aggregator import DecisionAggregator
from training.reward_system import RewardSystem
from train_strategic_marl import StrategicMARLTrainer, TrainingMetrics


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline"""
    
    @pytest.fixture
    def test_config(self):
        """Create minimal test configuration"""
        return {
            'seed': 42,
            'environment': {
                'matrix_shape': [48, 13],
                'max_timesteps': 10,  # Short episodes for testing
                'feature_indices': {
                    'mlmi_expert': [0, 1, 9, 10],
                    'nwrqk_expert': [2, 3, 4, 5],
                    'regime_expert': [10, 11, 12],
                }
            },
            'agents': {
                'mlmi_expert': {
                    'features': [0, 1, 9, 10],
                    'hidden_dims': [32, 16],  # Smaller for testing
                    'learning_rate': 1e-3
                },
                'nwrqk_expert': {
                    'features': [2, 3, 4, 5],
                    'hidden_dims': [32, 16],
                    'learning_rate': 1e-3
                },
                'regime_expert': {
                    'features': [10, 11, 12],
                    'hidden_dims': [32, 16],
                    'learning_rate': 1e-3
                }
            },
            'ensemble': {
                'weights': [0.4, 0.35, 0.25],
                'confidence_threshold': 0.65,
                'learning_rate': 1e-3
            },
            'rewards': {
                'alpha': 1.0,
                'beta': 0.2,
                'gamma': -0.3,
                'delta': 0.1,
                'max_drawdown': 0.15,
                'position_limit': 1.0,
                'use_running_stats': False
            },
            'training': {
                'episodes': 2,  # Very short for testing
                'batch_size': 4,
                'min_buffer_size': 10,
                'checkpoint_freq': 1,
                'checkpoint_dir': 'test_checkpoints',
                'keep_last': 2
            },
            'monitoring': {
                'metrics': {
                    'log_frequency': 1
                }
            }
        }
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_smoke_training_pipeline(self, test_config, temp_dir):
        """End-to-end smoke test with minimal configuration"""
        # Update checkpoint directory
        test_config['training']['checkpoint_dir'] = str(Path(temp_dir) / 'checkpoints')
        
        # Save config to file
        config_path = Path(temp_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create trainer
        trainer = StrategicMARLTrainer(str(config_path))
        
        # Run training for minimal episodes
        trainer.train(num_episodes=2)
        
        # Verify training completed
        assert trainer.current_episode >= 1
        assert trainer.total_steps > 0
        assert len(trainer.metrics.episode_rewards) >= 1
        
        # Verify checkpoint was saved
        checkpoint_dir = Path(test_config['training']['checkpoint_dir'])
        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0
    
    def test_component_integration(self):
        """Test that all components work together correctly"""
        # Create components
        env = StrategicMarketEnv()
        aggregator = DecisionAggregator()
        reward_system = RewardSystem()
        
        # Reset environment
        env.reset()
        
        # Collect agent outputs
        agent_outputs = {}
        for i, agent in enumerate(env.agent_iter()):
            if i >= 3:  # One complete cycle
                break
            
            # Get observation
            obs = env.observe(agent)
            
            # Generate action
            action = np.random.dirichlet(np.ones(3))
            
            # Step environment
            env.step(action)
            
            # Store for aggregation
            agent_outputs[agent] = action
        
        # Test aggregation
        decision = aggregator.aggregate(agent_outputs)
        assert decision.ensemble_probabilities.shape == (3,)
        assert 0 <= decision.confidence <= 1
        assert 0 <= decision.uncertainty <= 1
        
        # Test reward calculation
        state = {'portfolio_value': 10000}
        next_state = {'portfolio_value': 10050}
        info = {
            'pnl': 50,
            'synergy': None,
            'drawdown': 0.05,
            'position_size': 0.5,
            'volatility_30': 1.0
        }
        
        rewards = reward_system.calculate_reward(
            state, 
            decision.ensemble_probabilities,
            next_state,
            info
        )
        
        assert hasattr(rewards, 'total')
        assert isinstance(rewards.total, float)
    
    def test_environment_reset_cycle(self):
        """Test multiple reset cycles work correctly"""
        env = StrategicMarketEnv()
        
        for cycle in range(3):
            env.reset()
            
            # Run one complete agent cycle
            agents_seen = []
            for i, agent in enumerate(env.agent_iter()):
                if i >= 3:
                    break
                agents_seen.append(agent)
                env.step(np.array([0.33, 0.34, 0.33]))
            
            # Verify we saw all agents
            assert len(agents_seen) == 3
            assert set(agents_seen) == set(env.possible_agents)
    
    def test_aggregation_with_synergy(self):
        """Test aggregation when synergy is present"""
        aggregator = DecisionAggregator()
        
        agent_outputs = {
            'mlmi_expert': np.array([0.1, 0.2, 0.7]),   # Bullish
            'nwrqk_expert': np.array([0.2, 0.3, 0.5]),  # Slightly bullish
            'regime_expert': np.array([0.15, 0.25, 0.6]) # Bullish
        }
        
        synergy_info = {
            'type': 'TYPE_1',
            'direction': 1,  # Bullish
            'confidence': 0.85
        }
        
        decision = aggregator.aggregate(agent_outputs, synergy_info=synergy_info)
        
        # With bullish synergy and mostly bullish agents, should be confident
        assert decision.confidence > 0.5
        assert decision.reasoning['synergy_alignment'] == 'aligned'
    
    def test_reward_components_consistency(self):
        """Test reward components add up correctly"""
        reward_system = RewardSystem()
        
        state = {'portfolio_value': 10000}
        action = np.array([0.3, 0.4, 0.3])
        next_state = {'portfolio_value': 10100}
        
        info = {
            'pnl': 100,
            'synergy': {'type': 'TYPE_1', 'direction': 1, 'confidence': 0.8},
            'drawdown': 0.08,
            'position_size': 0.6,
            'volatility_30': 1.1
        }
        
        rewards = reward_system.calculate_reward(state, action, next_state, info)
        
        # Manually calculate expected total
        expected_total = (
            reward_system.alpha * rewards.pnl +
            reward_system.beta * rewards.synergy +
            reward_system.gamma * rewards.risk +
            reward_system.delta * rewards.exploration
        )
        
        assert np.isclose(rewards.total, expected_total, rtol=1e-6)
    
    def test_performance_requirements(self):
        """Test that performance requirements are met"""
        env = StrategicMarketEnv()
        env.reset()
        
        # Test inference latency
        import time
        inference_times = []
        
        for i in range(30):  # 10 complete cycles
            agent = env.agents[i % 3]
            env.agent_selection = agent
            
            start = time.time()
            obs = env.observe(agent)
            action = np.random.dirichlet(np.ones(3))
            env.step(action)
            elapsed = (time.time() - start) * 1000  # ms
            
            inference_times.append(elapsed)
        
        # Check average inference time
        avg_inference = np.mean(inference_times)
        assert avg_inference < 5.0, f"Inference time {avg_inference:.2f}ms exceeds 5ms requirement"
    
    def test_configuration_loading(self, temp_dir):
        """Test configuration loading and validation"""
        config = {
            'environment': {
                'matrix_shape': [48, 13],
                'max_timesteps': 100
            },
            'rewards': {
                'alpha': 2.0,  # Different from default
                'beta': 0.5
            }
        }
        
        config_path = Path(temp_dir) / 'custom_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Test DecisionAggregator config loading
        aggregator = DecisionAggregator(config_path=str(config_path))
        # Should use defaults when not specified
        assert aggregator.confidence_threshold == 0.65
        
        # Test RewardSystem config loading
        reward_system = RewardSystem(config_path=str(config_path))
        assert reward_system.alpha == 2.0
        assert reward_system.beta == 0.5
    
    def test_error_handling(self):
        """Test graceful error handling"""
        env = StrategicMarketEnv()
        env.reset()
        
        # Test invalid action shape
        with pytest.raises(ValueError):
            env.step(np.array([0.5, 0.5]))  # Wrong shape
        
        # Test missing agent in aggregator
        aggregator = DecisionAggregator()
        with pytest.raises(ValueError):
            aggregator.aggregate({'mlmi_expert': np.array([0.3, 0.3, 0.4])})  # Missing agents
    
    def test_metrics_tracking(self, test_config, temp_dir):
        """Test that metrics are properly tracked during training"""
        test_config['training']['checkpoint_dir'] = str(Path(temp_dir) / 'checkpoints')
        test_config['training']['episodes'] = 5
        
        config_path = Path(temp_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = StrategicMARLTrainer(str(config_path))
        trainer.train()
        
        # Verify metrics were collected
        assert len(trainer.metrics.episode_rewards) == 5
        assert len(trainer.metrics.episode_lengths) == 5
        assert all(r > 0 for r in trainer.metrics.episode_lengths)
        
        # Verify statistics are available
        agg_stats = trainer.aggregator.get_decision_statistics()
        assert 'avg_confidence' in agg_stats
        assert 'agent_agreement_rate' in agg_stats
        
        reward_stats = trainer.reward_system.get_reward_statistics()
        assert 'pnl' in reward_stats
        assert 'total' in reward_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])