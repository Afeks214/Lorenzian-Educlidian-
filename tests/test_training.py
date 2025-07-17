"""
Unit tests for MAPPO training components
Tests trainer, losses, schedulers, and training pipeline
"""

import pytest
import torch
import numpy as np
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mappo_trainer import MAPPOTrainer
from training.losses import PPOLoss, GAEAdvantageEstimator, MultiAgentPPOLoss
from training.schedulers import LinearDecayScheduler, AdaptiveScheduler, SchedulerManager
from utils.metrics import MetricsTracker


class TestPPOLoss:
    """Test PPO loss computation."""
    
    @pytest.fixture
    def ppo_loss(self):
        return PPOLoss(
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5
        )
    
    def test_initialization(self, ppo_loss):
        """Test loss initialization."""
        assert ppo_loss.clip_epsilon == 0.2
        assert ppo_loss.value_clip_epsilon == 0.2
        assert ppo_loss.entropy_coef == 0.01
        assert ppo_loss.value_loss_coef == 0.5
    
    def test_policy_loss_clipping(self, ppo_loss):
        """Test PPO clipping mechanism."""
        batch_size = 32
        
        # Create inputs
        log_probs = torch.randn(batch_size)
        old_log_probs = log_probs.detach() + torch.randn(batch_size) * 0.1
        advantages = torch.randn(batch_size)
        values = torch.randn(batch_size)
        old_values = values.detach() + torch.randn(batch_size) * 0.1
        returns = values + torch.randn(batch_size) * 0.1
        entropy = torch.rand(batch_size) * 2
        
        # Compute loss
        loss_dict = ppo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            old_values=old_values,
            returns=returns,
            entropy=entropy
        )
        
        # Check outputs
        assert 'total_loss' in loss_dict
        assert 'policy_loss' in loss_dict
        assert 'value_loss' in loss_dict
        assert 'entropy_loss' in loss_dict
        assert 'approx_kl' in loss_dict
        assert 'clip_fraction' in loss_dict
        
        # Check that losses are finite
        for key, value in loss_dict.items():
            assert torch.isfinite(value)
    
    def test_importance_sampling(self, ppo_loss):
        """Test loss with importance sampling weights."""
        batch_size = 16
        
        # Create inputs
        log_probs = torch.randn(batch_size)
        old_log_probs = log_probs.detach()
        advantages = torch.ones(batch_size)
        values = torch.zeros(batch_size)
        old_values = torch.zeros(batch_size)
        returns = torch.ones(batch_size)
        entropy = torch.ones(batch_size)
        importance_weights = torch.rand(batch_size) + 0.5
        
        # Compute loss with IS weights
        loss_dict = ppo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            old_values=old_values,
            returns=returns,
            entropy=entropy,
            importance_weights=importance_weights
        )
        
        # Loss should be affected by weights
        assert torch.isfinite(loss_dict['total_loss'])


class TestGAEAdvantageEstimator:
    """Test GAE computation."""
    
    @pytest.fixture
    def gae_estimator(self):
        return GAEAdvantageEstimator(gamma=0.99, lam=0.95)
    
    def test_gae_computation(self, gae_estimator):
        """Test GAE advantage calculation."""
        batch_size = 4
        seq_len = 10
        
        # Create synthetic data
        rewards = torch.rand(batch_size, seq_len)
        values = torch.rand(batch_size, seq_len)
        next_values = torch.rand(batch_size, seq_len)
        dones = torch.zeros(batch_size, seq_len)
        dones[:, -1] = 1  # Episode ends
        
        # Compute advantages
        advantages, returns = gae_estimator.compute_advantages(
            rewards, values, next_values, dones
        )
        
        # Check shapes
        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)
        
        # Check that returns = advantages + values
        assert torch.allclose(returns, advantages + values, atol=1e-5)
    
    def test_td_error_computation(self, gae_estimator):
        """Test TD error calculation."""
        batch_size = 8
        
        rewards = torch.rand(batch_size)
        values = torch.rand(batch_size)
        next_values = torch.rand(batch_size)
        dones = torch.zeros(batch_size)
        
        td_errors = gae_estimator.compute_td_errors(
            rewards, values, next_values, dones
        )
        
        # Manual calculation
        expected_td = rewards + gae_estimator.gamma * next_values - values
        
        assert torch.allclose(td_errors, expected_td)


class TestMultiAgentPPOLoss:
    """Test multi-agent PPO loss."""
    
    @pytest.fixture
    def ma_loss(self):
        return MultiAgentPPOLoss(
            n_agents=3,
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5
        )
    
    def test_multi_agent_loss(self, ma_loss):
        """Test loss computation for multiple agents."""
        batch_size = 16
        
        # Create per-agent data
        agent_names = ['mlmi', 'nwrqk', 'mmd']
        agent_log_probs = {name: torch.randn(batch_size) for name in agent_names}
        agent_old_log_probs = {name: torch.randn(batch_size) for name in agent_names}
        agent_advantages = {name: torch.randn(batch_size) for name in agent_names}
        agent_entropies = {name: torch.rand(batch_size) for name in agent_names}
        
        # Centralized values
        centralized_values = torch.randn(batch_size)
        centralized_old_values = torch.randn(batch_size)
        centralized_returns = torch.randn(batch_size)
        
        # Compute loss
        loss_dict = ma_loss(
            agent_log_probs=agent_log_probs,
            agent_old_log_probs=agent_old_log_probs,
            agent_advantages=agent_advantages,
            agent_entropies=agent_entropies,
            centralized_values=centralized_values,
            centralized_old_values=centralized_old_values,
            centralized_returns=centralized_returns
        )
        
        # Check outputs
        assert 'total_loss' in loss_dict
        assert 'policy_loss' in loss_dict
        assert 'value_loss' in loss_dict
        
        # Check per-agent metrics
        for name in agent_names:
            assert f'{name}_policy_loss' in loss_dict
            assert f'{name}_entropy' in loss_dict
            assert f'{name}_kl' in loss_dict


class TestSchedulers:
    """Test learning rate schedulers."""
    
    def test_linear_decay_scheduler(self):
        """Test linear decay scheduler."""
        optimizer = torch.optim.Adam([torch.randn(10, 10)], lr=1e-3)
        scheduler = LinearDecayScheduler(
            optimizer=optimizer,
            start_lr=1e-3,
            end_lr=1e-4,
            decay_steps=100
        )
        
        # Initial LR
        assert optimizer.param_groups[0]['lr'] == 1e-3
        
        # Step halfway
        for _ in range(50):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        expected_lr = 1e-3 + (1e-4 - 1e-3) * 0.5
        assert abs(current_lr - expected_lr) < 1e-6
        
        # Step to end
        for _ in range(50):
            scheduler.step()
        
        assert optimizer.param_groups[0]['lr'] == 1e-4
    
    def test_adaptive_scheduler(self):
        """Test adaptive scheduler."""
        optimizer = torch.optim.Adam([torch.randn(10, 10)], lr=1e-3)
        scheduler = AdaptiveScheduler(
            optimizer=optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate improving metric
        for i in range(10):
            scheduler.step(metric=i)
        
        # LR should not change with improving metric
        assert optimizer.param_groups[0]['lr'] == initial_lr
        
        # Simulate stagnant metric
        for _ in range(10):
            scheduler.step(metric=5)
        
        # LR should be reduced after patience
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_scheduler_manager(self):
        """Test scheduler manager."""
        optimizers = {
            'actor': torch.optim.Adam([torch.randn(10, 10)], lr=1e-3),
            'critic': torch.optim.Adam([torch.randn(10, 10)], lr=5e-4)
        }
        
        manager = SchedulerManager()
        
        # Add schedulers
        for name, opt in optimizers.items():
            scheduler = LinearDecayScheduler(
                optimizer=opt,
                start_lr=opt.param_groups[0]['lr'],
                end_lr=opt.param_groups[0]['lr'] * 0.1,
                decay_steps=100
            )
            manager.add_scheduler(name, scheduler)
        
        # Get initial LRs
        initial_lrs = manager.get_lrs()
        assert 'actor' in initial_lrs
        assert 'critic' in initial_lrs
        
        # Step all
        for _ in range(50):
            manager.step_all()
        
        # Check LRs decreased
        current_lrs = manager.get_lrs()
        assert current_lrs['actor'] < initial_lrs['actor']
        assert current_lrs['critic'] < initial_lrs['critic']


class TestMAPPOTrainer:
    """Test MAPPO trainer."""
    
    @pytest.fixture
    def config(self):
        return {
            'device': 'cpu',
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'learning_rate': 3e-4,
            'batch_size': 32,
            'ppo_epochs': 4,
            'agent_configs': {
                'mlmi': {'input_dim': 4},
                'nwrqk': {'input_dim': 6},
                'mmd': {'input_dim': 3}
            }
        }
    
    def test_trainer_initialization(self, config):
        """Test trainer initialization."""
        trainer = MAPPOTrainer(config)
        
        # Check components initialized
        assert hasattr(trainer, 'agents')
        assert hasattr(trainer, 'critic')
        assert hasattr(trainer, 'optimizers')
        assert hasattr(trainer, 'experience_buffer')
        assert hasattr(trainer, 'checkpoint_manager')
        
        # Check agents
        assert len(trainer.agents) == 3
        assert 'mlmi' in trainer.agents
        assert 'nwrqk' in trainer.agents
        assert 'mmd' in trainer.agents
    
    def test_experience_processing(self, config):
        """Test experience processing."""
        trainer = MAPPOTrainer(config)
        
        # Create mock experiences
        experiences = []
        for i in range(10):
            exp = {
                'states': {
                    'mlmi': torch.randn(1, 4),
                    'nwrqk': torch.randn(1, 6),
                    'mmd': torch.randn(1, 3)
                },
                'actions': {'mlmi': 0, 'nwrqk': 1, 'mmd': 2},
                'rewards': {'mlmi': 0.1, 'nwrqk': 0.2, 'mmd': 0.3},
                'next_states': {
                    'mlmi': torch.randn(1, 4),
                    'nwrqk': torch.randn(1, 6),
                    'mmd': torch.randn(1, 3)
                },
                'dones': {'mlmi': False, 'nwrqk': False, 'mmd': False},
                'log_probs': {'mlmi': -1.0, 'nwrqk': -1.5, 'mmd': -2.0},
                'centralized_value': 0.5
            }
            experiences.append(exp)
        
        # Process experiences
        batch = trainer._process_experiences(experiences)
        
        # Check batch structure
        assert hasattr(batch, 'states')
        assert hasattr(batch, 'actions')
        assert hasattr(batch, 'advantages')
        assert hasattr(batch, 'returns')
        
        # Check shapes
        assert batch.states['mlmi'].shape == (10, 4)
        assert batch.actions['mlmi'].shape == (10,)


class TestMetricsTracker:
    """Test metrics tracking."""
    
    def test_scalar_tracking(self):
        """Test scalar metric tracking."""
        tracker = MetricsTracker(window_size=5)
        
        # Add metrics
        for i in range(10):
            tracker.add_scalar('loss', 1.0 - i * 0.1)
            tracker.add_scalar('reward', i * 0.2)
        
        # Check current metrics
        current = tracker.get_current_metrics()
        assert 'loss' in current
        assert 'reward' in current
        
        # Window average should be over last 5 values
        assert abs(current['loss'] - 0.3) < 0.01  # Average of 0.1, 0.2, 0.3, 0.4, 0.5
    
    def test_statistics(self):
        """Test metric statistics."""
        tracker = MetricsTracker()
        
        values = [1, 2, 3, 4, 5]
        for v in values:
            tracker.add_scalar('test_metric', v)
        
        stats = tracker.get_statistics('test_metric')
        
        assert stats['min'] == 1
        assert stats['max'] == 5
        assert stats['mean'] == 3
        assert stats['count'] == 5
    
    def test_histogram_tracking(self):
        """Test histogram data tracking."""
        tracker = MetricsTracker()
        
        # Add histogram
        values = np.random.randn(100)
        tracker.add_histogram('weights', values)
        
        # Check stored histogram data
        hist_data = tracker.metrics['weights_histogram']
        assert len(hist_data) == 1
        assert 'mean' in hist_data[0]
        assert 'std' in hist_data[0]
        assert 'min' in hist_data[0]
        assert 'max' in hist_data[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])