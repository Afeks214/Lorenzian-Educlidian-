"""
Test and Validation Module for PettingZoo Training Integration

This module provides comprehensive testing for the PettingZoo training system,
including unit tests, integration tests, and validation of the complete
training pipeline.

Key Features:
- Unit tests for individual components
- Integration tests for complete training pipeline
- Performance benchmarking
- API compliance validation
- Error handling and edge case testing
"""

import unittest
import numpy as np
import torch
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import tempfile
import shutil
from pathlib import Path
import logging
import time
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

# PettingZoo imports
from pettingzoo.utils import agent_selector
from pettingzoo.test import api_test

# Internal imports
from .pettingzoo_mappo_trainer import PettingZooMAPPOTrainer, TrainingConfig
from .pettingzoo_environment_manager import (
    EnvironmentFactory, EnvironmentConfig, EnvironmentType,
    EnvironmentValidator, create_tactical_environment
)
from .pettingzoo_training_loops import (
    PettingZooTrainingLoop, TrainingLoopConfig, 
    ExperienceCollector, CurriculumManager
)
from .pettingzoo_reward_system import (
    PettingZooRewardSystem, RewardConfig, create_reward_config
)
from .unified_pettingzoo_trainer import (
    UnifiedPettingZooTrainer, UnifiedTrainingConfig, TrainingMode
)
from .pettingzoo_parallel_trainer import (
    PettingZooParallelTrainer, ParallelTrainingConfig
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class TestResults:
    """Container for test results"""
    passed: int = 0
    failed: int = 0
    errors: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class MockEnvironment:
    """Mock PettingZoo environment for testing"""
    
    def __init__(self):
        self.agents = ['agent1', 'agent2']
        self.possible_agents = ['agent1', 'agent2']
        self.agent_selection = 'agent1'
        self.agent_selector = agent_selector(self.agents)
        
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.step_count = 0
        self.max_steps = 100
        
        # Observation and action spaces
        from gymnasium import spaces
        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.agents
        }
    
    def reset(self):
        """Reset environment"""
        self.agents = self.possible_agents.copy()
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()
        self.step_count = 0
        
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
    
    def step(self, action):
        """Step environment"""
        if not self.agents:
            return
        
        current_agent = self.agent_selection
        
        # Generate reward
        reward = np.random.uniform(-1, 1)
        self.rewards[current_agent] = reward
        
        # Check termination
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.dones[current_agent] = True
            self.truncations[current_agent] = True
        
        # Move to next agent
        self.agent_selection = self.agent_selector.next()
        
        # Remove done agents
        if self.dones[current_agent]:
            self.agents.remove(current_agent)
    
    def observe(self, agent):
        """Get observation for agent"""
        return np.random.randn(10).astype(np.float32)
    
    def observation_space(self, agent):
        """Get observation space for agent"""
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """Get action space for agent"""
        return self.action_spaces[agent]
    
    def close(self):
        """Close environment"""
        pass


class TestEnvironmentManager(unittest.TestCase):
    """Test environment management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.factory = EnvironmentFactory()
        self.validator = EnvironmentValidator()
    
    def test_environment_factory_creation(self):
        """Test environment factory creation"""
        # Test that we can create factory
        self.assertIsNotNone(self.factory)
        
        # Test registered environments
        self.assertIn(EnvironmentType.TACTICAL, self.factory.registered_environments)
        self.assertIn(EnvironmentType.STRATEGIC, self.factory.registered_environments)
    
    def test_mock_environment_validation(self):
        """Test validation of mock environment"""
        mock_env = MockEnvironment()
        
        # Test basic validation
        results = self.validator.validate_environment(mock_env)
        
        # Check that validation passes
        self.assertTrue(results['is_valid'])
        self.assertEqual(len(results['errors']), 0)
        
        # Check performance metrics
        self.assertIn('performance_metrics', results)
        self.assertIn('reset_time', results['performance_metrics'])
    
    def test_environment_config_creation(self):
        """Test environment configuration creation"""
        config = EnvironmentConfig(
            env_type=EnvironmentType.TACTICAL,
            max_cycles=500,
            render_mode="human"
        )
        
        self.assertEqual(config.env_type, EnvironmentType.TACTICAL)
        self.assertEqual(config.max_cycles, 500)
        self.assertEqual(config.render_mode, "human")


class TestMAPPOTrainer(unittest.TestCase):
    """Test MAPPO trainer functionality"""
    
    def setUp(self):
        """Set up test trainer"""
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = TrainingConfig(
            env_factory=lambda: MockEnvironment(),
            num_episodes=10,
            max_episode_steps=50,
            batch_size=32,
            learning_rate=1e-3,
            log_dir=self.temp_dir,
            device="cpu"
        )
        
        self.trainer = PettingZooMAPPOTrainer(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        self.trainer._cleanup()
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer.env)
        self.assertIsNotNone(self.trainer.networks)
        self.assertIsNotNone(self.trainer.optimizers)
        self.assertIsNotNone(self.trainer.replay_buffer)
        
        # Check agents
        self.assertEqual(len(self.trainer.agents), 2)
        self.assertIn('agent1', self.trainer.agents)
        self.assertIn('agent2', self.trainer.agents)
    
    def test_action_and_value_generation(self):
        """Test action and value generation"""
        observation = np.random.randn(10).astype(np.float32)
        
        action, log_prob, value = self.trainer.get_action_and_value(observation, 'agent1')
        
        # Check types and ranges
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
        self.assertIn(action, [0, 1, 2])
    
    def test_training_step(self):
        """Test single training step"""
        initial_metrics = len(self.trainer.metrics['episode_rewards'])
        
        # Run single episode
        result = self.trainer._run_episode()
        
        # Check result structure
        self.assertIn('episode_rewards', result)
        self.assertIn('total_reward', result)
        self.assertIn('episode_length', result)
        
        # Check metrics update
        self.assertEqual(len(self.trainer.metrics['episode_rewards']), initial_metrics + 1)
    
    def test_network_update(self):
        """Test network parameter updates"""
        # Fill replay buffer with some data
        for _ in range(100):
            obs = np.random.randn(10).astype(np.float32)
            action = np.random.randint(0, 3)
            reward = np.random.uniform(-1, 1)
            value = np.random.uniform(-1, 1)
            log_prob = np.random.uniform(-2, 0)
            
            self.trainer.replay_buffer.add(
                obs=obs, action=action, reward=reward, value=value,
                log_prob=log_prob, done=False, agent_id=0, step_id=0
            )
        
        # Test network update
        initial_loss = len(self.trainer.metrics['policy_losses'])
        self.trainer._update_networks()
        
        # Check that losses were recorded
        self.assertGreater(len(self.trainer.metrics['policy_losses']), initial_loss)


class TestRewardSystem(unittest.TestCase):
    """Test reward system functionality"""
    
    def setUp(self):
        """Set up test reward system"""
        self.reward_config = create_reward_config(
            normalize_rewards=True,
            clip_rewards=True,
            adaptive_weights=True
        )
        
        self.reward_system = PettingZooRewardSystem(self.reward_config)
        self.mock_env = MockEnvironment()
    
    def test_reward_calculation(self):
        """Test basic reward calculation"""
        reward = self.reward_system.calculate_agent_reward(
            agent='agent1',
            env=self.mock_env,
            action=1,
            pre_step_state={'observation': np.random.randn(10)},
            post_step_info={'performance': 0.8, 'risk': 0.2}
        )
        
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, self.reward_config.reward_clip_range[0])
        self.assertLessEqual(reward, self.reward_config.reward_clip_range[1])
    
    def test_reward_components(self):
        """Test individual reward components"""
        from .pettingzoo_reward_system import TradingRewardCalculator
        
        calculator = TradingRewardCalculator(self.reward_config)
        
        components = calculator.get_reward_components(
            agent='agent1',
            state={'observation': np.random.randn(10)},
            action=1,
            next_state={'observation': np.random.randn(10)},
            info={'performance': 0.8, 'risk': 0.2, 'cooperation': 0.6}
        )
        
        # Check component types
        from .pettingzoo_reward_system import RewardComponent
        self.assertIn(RewardComponent.PERFORMANCE, components)
        self.assertIn(RewardComponent.RISK_ADJUSTED, components)
        self.assertIn(RewardComponent.COOPERATION, components)
    
    def test_performance_tracking(self):
        """Test reward performance tracking"""
        # Generate multiple rewards
        for _ in range(10):
            self.reward_system.calculate_agent_reward(
                agent='agent1',
                env=self.mock_env,
                action=np.random.randint(0, 3),
                pre_step_state={'observation': np.random.randn(10)},
                post_step_info={'performance': np.random.uniform(0, 1)}
            )
        
        # Check performance summary
        summary = self.reward_system.get_performance_summary()
        
        self.assertIn('total_rewards', summary)
        self.assertIn('agent1', summary['total_rewards'])
        self.assertIn('adaptive_weights', summary)


class TestTrainingLoop(unittest.TestCase):
    """Test training loop functionality"""
    
    def setUp(self):
        """Set up test training loop"""
        self.temp_dir = tempfile.mkdtemp()
        
        trainer_config = TrainingConfig(
            env_factory=lambda: MockEnvironment(),
            num_episodes=20,
            max_episode_steps=50,
            batch_size=32,
            log_dir=self.temp_dir,
            device="cpu"
        )
        
        self.trainer = PettingZooMAPPOTrainer(trainer_config)
        
        self.loop_config = TrainingLoopConfig(
            max_episodes=20,
            max_steps_per_episode=50,
            batch_size=32,
            parallel_envs=1,
            enable_curriculum=False,
            log_frequency=5
        )
        
        self.training_loop = PettingZooTrainingLoop(self.trainer, self.loop_config)
    
    def tearDown(self):
        """Clean up test environment"""
        self.training_loop._cleanup()
        shutil.rmtree(self.temp_dir)
    
    def test_experience_collection(self):
        """Test experience collection"""
        collector = ExperienceCollector(self.loop_config)
        
        # Add some experiences
        for i in range(10):
            collector.collect_step(
                agent='agent1',
                observation=np.random.randn(10),
                action=np.random.randint(0, 3),
                reward=np.random.uniform(-1, 1),
                value=np.random.uniform(-1, 1),
                log_prob=np.random.uniform(-2, 0),
                done=False
            )
        
        # Test batch retrieval
        batch = collector.get_batch(5)
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch['observations']), 5)
        
        # Test agent-specific experiences
        agent_exp = collector.get_agent_experiences('agent1')
        self.assertEqual(len(agent_exp), 10)
    
    def test_curriculum_management(self):
        """Test curriculum learning management"""
        curriculum_config = TrainingLoopConfig(
            max_episodes=100,
            enable_curriculum=True,
            curriculum_stages=[
                {'name': 'easy', 'episodes': 30, 'learning_rate': 3e-4},
                {'name': 'medium', 'episodes': 40, 'learning_rate': 1e-4},
                {'name': 'hard', 'episodes': 30, 'learning_rate': 5e-5}
            ]
        )
        
        curriculum = CurriculumManager(curriculum_config)
        
        # Test stage progression
        self.assertEqual(curriculum.current_stage, 0)
        self.assertEqual(curriculum.get_current_stage()['name'], 'easy')
        
        # Test stage transition
        self.assertTrue(curriculum.should_transition(30))
        curriculum.transition_stage()
        self.assertEqual(curriculum.current_stage, 1)
        self.assertEqual(curriculum.get_current_stage()['name'], 'medium')
    
    def test_training_loop_execution(self):
        """Test complete training loop execution"""
        # Reduce episodes for faster testing
        self.loop_config.max_episodes = 5
        
        start_time = time.time()
        results = self.training_loop.run_training()
        end_time = time.time()
        
        # Check results structure
        self.assertIn('training_time', results)
        self.assertIn('total_episodes', results)
        self.assertIn('best_reward', results)
        self.assertIn('metrics', results)
        
        # Check training progress
        self.assertGreater(results['total_episodes'], 0)
        self.assertGreater(results['training_time'], 0)
        
        # Check reasonable training time
        self.assertLess(end_time - start_time, 30)  # Should complete within 30 seconds


class TestUnifiedTrainer(unittest.TestCase):
    """Test unified trainer functionality"""
    
    def setUp(self):
        """Set up test unified trainer"""
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = UnifiedTrainingConfig(
            training_mode=TrainingMode.TACTICAL,
            total_episodes=10,
            max_episode_steps=50,
            batch_size=32,
            log_directory=self.temp_dir,
            enable_hyperparameter_optimization=False,
            enable_tensorboard=False,
            enable_wandb=False,
            parallel_environments=1
        )
        
        # Mock environment configs
        self.config.environment_configs = {
            'tactical': EnvironmentConfig(
                env_type=EnvironmentType.TACTICAL,
                env_params={'max_episode_steps': 50}
            )
        }
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.training.unified_pettingzoo_trainer.EnvironmentFactory')
    def test_unified_trainer_initialization(self, mock_factory):
        """Test unified trainer initialization"""
        # Mock factory to return MockEnvironment
        mock_factory.return_value.create_environment.return_value = MockEnvironment()
        
        trainer = UnifiedPettingZooTrainer(self.config)
        
        self.assertIsNotNone(trainer.environment_factory)
        self.assertIsNotNone(trainer.reward_systems)
        self.assertIsNotNone(trainer.trainers)
    
    @patch('src.training.unified_pettingzoo_trainer.EnvironmentFactory')
    def test_unified_trainer_single_mode(self, mock_factory):
        """Test unified trainer single mode training"""
        # Mock factory to return MockEnvironment
        mock_factory.return_value.create_environment.return_value = MockEnvironment()
        
        trainer = UnifiedPettingZooTrainer(self.config)
        
        # Mock the training components
        with patch.object(trainer, '_run_single_mode_training') as mock_train:
            mock_train.return_value = {
                'best_reward': 0.5,
                'metrics': {'episode_rewards': [0.1, 0.2, 0.5]},
                'convergence_achieved': False
            }
            
            with patch.object(trainer, '_run_comprehensive_evaluation') as mock_eval:
                mock_eval.return_value = {
                    'tactical': {'mean_reward': 0.4, 'std_reward': 0.1}
                }
                
                results = trainer.train()
                
                # Check results
                self.assertIn('training_results', results)
                self.assertIn('evaluation_results', results)
                self.assertIn('total_training_time', results)


class TestParallelTrainer(unittest.TestCase):
    """Test parallel trainer functionality"""
    
    def setUp(self):
        """Set up test parallel trainer"""
        self.config = ParallelTrainingConfig(
            num_workers=2,
            episodes_per_worker=5,
            steps_per_episode=20,
            use_multiprocessing=False,  # Use threading for testing
            worker_timeout=10.0,
            enable_fault_tolerance=True
        )
        
        self.trainer_config = TrainingConfig(
            num_episodes=10,
            max_episode_steps=20,
            batch_size=16,
            device="cpu"
        )
    
    def test_parallel_config_creation(self):
        """Test parallel configuration creation"""
        from .pettingzoo_parallel_trainer import create_parallel_config
        
        config = create_parallel_config(
            num_workers=4,
            episodes_per_worker=100,
            use_multiprocessing=True
        )
        
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.episodes_per_worker, 100)
        self.assertTrue(config.use_multiprocessing)
    
    def test_experience_aggregator(self):
        """Test experience aggregation"""
        from .pettingzoo_parallel_trainer import ExperienceAggregator
        
        aggregator = ExperienceAggregator(self.config)
        
        # Add sample results
        sample_result = {
            'worker_id': 0,
            'episode_reward': 0.5,
            'episode_length': 20,
            'agent_experiences': {
                'agent1': [
                    {
                        'observation': np.random.randn(10),
                        'action': 1,
                        'reward': 0.1,
                        'value': 0.05,
                        'log_prob': -0.5,
                        'done': False,
                        'agent': 'agent1'
                    }
                ]
            }
        }
        
        aggregator.add_episode_result(sample_result)
        
        # Test batch retrieval
        batch = aggregator.get_batch(1)
        self.assertIsNotNone(batch)
        self.assertIn('observations', batch)
        self.assertIn('actions', batch)
        
        # Test statistics
        stats = aggregator.get_statistics()
        self.assertEqual(stats['episode_count'], 1)
        self.assertEqual(stats['total_experiences'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete training pipeline"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training(self):
        """Test complete end-to-end training pipeline"""
        # Create minimal configuration
        config = TrainingConfig(
            env_factory=lambda: MockEnvironment(),
            num_episodes=5,
            max_episode_steps=20,
            batch_size=16,
            log_dir=self.temp_dir,
            device="cpu"
        )
        
        # Create trainer
        trainer = PettingZooMAPPOTrainer(config)
        
        # Run training
        results = trainer.train()
        
        # Verify results
        self.assertIn('training_time', results)
        self.assertIn('total_episodes', results)
        self.assertIn('best_reward', results)
        
        # Check that training actually occurred
        self.assertGreater(results['total_episodes'], 0)
        self.assertGreater(results['training_time'], 0)
        
        # Cleanup
        trainer._cleanup()
    
    def test_reward_system_integration(self):
        """Test reward system integration with training"""
        # Create reward system
        reward_config = create_reward_config(
            normalize_rewards=True,
            adaptive_weights=True
        )
        reward_system = PettingZooRewardSystem(reward_config)
        
        # Create mock environment
        mock_env = MockEnvironment()
        
        # Test reward calculation during training steps
        for episode in range(3):
            mock_env.reset()
            
            while mock_env.agents:
                current_agent = mock_env.agent_selection
                observation = mock_env.observe(current_agent)
                action = np.random.randint(0, 3)
                
                # Calculate reward
                reward = reward_system.calculate_agent_reward(
                    agent=current_agent,
                    env=mock_env,
                    action=action,
                    pre_step_state={'observation': observation},
                    post_step_info={'episode_step': mock_env.step_count}
                )
                
                # Verify reward
                self.assertIsInstance(reward, float)
                self.assertGreaterEqual(reward, reward_config.reward_clip_range[0])
                self.assertLessEqual(reward, reward_config.reward_clip_range[1])
                
                # Step environment
                mock_env.step(action)
        
        # Check performance summary
        summary = reward_system.get_performance_summary()
        self.assertIn('total_rewards', summary)
        self.assertGreater(len(summary['total_rewards']), 0)


class PerformanceBenchmark:
    """Performance benchmarking for training components"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_trainer_performance(self, num_episodes: int = 50) -> Dict[str, float]:
        """Benchmark trainer performance"""
        config = TrainingConfig(
            env_factory=lambda: MockEnvironment(),
            num_episodes=num_episodes,
            max_episode_steps=50,
            batch_size=32,
            device="cpu"
        )
        
        trainer = PettingZooMAPPOTrainer(config)
        
        # Benchmark training
        start_time = time.time()
        results = trainer.train()
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        episodes_per_second = num_episodes / total_time
        steps_per_second = results['total_steps'] / total_time
        
        trainer._cleanup()
        
        performance_metrics = {
            'total_time': total_time,
            'episodes_per_second': episodes_per_second,
            'steps_per_second': steps_per_second,
            'final_reward': results['final_reward']
        }
        
        self.results['trainer_performance'] = performance_metrics
        return performance_metrics
    
    def benchmark_reward_system_performance(self, num_calculations: int = 1000) -> Dict[str, float]:
        """Benchmark reward system performance"""
        reward_config = create_reward_config()
        reward_system = PettingZooRewardSystem(reward_config)
        mock_env = MockEnvironment()
        
        # Benchmark reward calculations
        start_time = time.time()
        
        for _ in range(num_calculations):
            reward_system.calculate_agent_reward(
                agent='agent1',
                env=mock_env,
                action=np.random.randint(0, 3),
                pre_step_state={'observation': np.random.randn(10)},
                post_step_info={'performance': np.random.uniform(0, 1)}
            )
        
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        calculations_per_second = num_calculations / total_time
        
        performance_metrics = {
            'total_time': total_time,
            'calculations_per_second': calculations_per_second,
            'avg_time_per_calculation': total_time / num_calculations
        }
        
        self.results['reward_system_performance'] = performance_metrics
        return performance_metrics
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get benchmark summary"""
        return {
            'results': self.results,
            'timestamp': time.time()
        }


def run_all_tests() -> TestResults:
    """Run all tests and return results"""
    test_results = TestResults()
    
    # Test suites
    test_suites = [
        TestEnvironmentManager,
        TestMAPPOTrainer,
        TestRewardSystem,
        TestTrainingLoop,
        TestUnifiedTrainer,
        TestParallelTrainer,
        TestIntegration
    ]
    
    for test_suite in test_suites:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
        result = runner.run(suite)
        
        test_results.passed += result.testsRun - len(result.failures) - len(result.errors)
        test_results.failed += len(result.failures) + len(result.errors)
        
        # Collect error messages
        for failure in result.failures:
            test_results.errors.append(f"FAILURE: {failure[0]} - {failure[1]}")
        
        for error in result.errors:
            test_results.errors.append(f"ERROR: {error[0]} - {error[1]}")
    
    return test_results


def run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    
    print("Running performance benchmarks...")
    
    # Benchmark trainer
    print("Benchmarking trainer performance...")
    trainer_perf = benchmark.benchmark_trainer_performance(num_episodes=20)
    print(f"Trainer: {trainer_perf['episodes_per_second']:.2f} episodes/sec")
    
    # Benchmark reward system
    print("Benchmarking reward system performance...")
    reward_perf = benchmark.benchmark_reward_system_performance(num_calculations=500)
    print(f"Reward system: {reward_perf['calculations_per_second']:.2f} calculations/sec")
    
    return benchmark.get_benchmark_summary()


def validate_pettingzoo_compliance() -> Dict[str, Any]:
    """Validate PettingZoo API compliance"""
    validation_results = {
        'environments_tested': 0,
        'compliance_passed': 0,
        'compliance_failed': 0,
        'errors': []
    }
    
    # Test mock environment
    try:
        mock_env = MockEnvironment()
        api_test(mock_env, num_cycles=10, verbose_progress=False)
        validation_results['environments_tested'] += 1
        validation_results['compliance_passed'] += 1
    except Exception as e:
        validation_results['environments_tested'] += 1
        validation_results['compliance_failed'] += 1
        validation_results['errors'].append(f"MockEnvironment: {str(e)}")
    
    return validation_results


if __name__ == "__main__":
    print("Running PettingZoo Training Integration Tests...")
    print("=" * 50)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    test_results = run_all_tests()
    
    print(f"Tests Passed: {test_results.passed}")
    print(f"Tests Failed: {test_results.failed}")
    
    if test_results.errors:
        print("\nErrors:")
        for error in test_results.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Run performance benchmarks
    print("\n2. Running Performance Benchmarks...")
    benchmark_results = run_performance_benchmarks()
    
    # Validate PettingZoo compliance
    print("\n3. Validating PettingZoo API Compliance...")
    compliance_results = validate_pettingzoo_compliance()
    
    print(f"Environments tested: {compliance_results['environments_tested']}")
    print(f"Compliance passed: {compliance_results['compliance_passed']}")
    print(f"Compliance failed: {compliance_results['compliance_failed']}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    total_tests = test_results.passed + test_results.failed
    pass_rate = (test_results.passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Overall Pass Rate: {pass_rate:.1f}%")
    print(f"PettingZoo Compliance: {compliance_results['compliance_passed']}/{compliance_results['environments_tested']} environments")
    
    if pass_rate >= 95 and compliance_results['compliance_passed'] == compliance_results['environments_tested']:
        print("✅ PettingZoo integration is READY for production!")
    else:
        print("❌ PettingZoo integration needs additional work")
    
    # Save results
    results_file = Path("pettingzoo_integration_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'test_results': {
                'passed': test_results.passed,
                'failed': test_results.failed,
                'errors': test_results.errors
            },
            'benchmark_results': benchmark_results,
            'compliance_results': compliance_results,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")