"""
Unit tests for experience replay buffers
Tests prioritized and uniform replay buffers
"""

import pytest
import torch
import numpy as np
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.replay_buffer import (
    PrioritizedExperienceBuffer, UniformExperienceBuffer, 
    Experience, BatchedExperience, SumTree
)


class TestSumTree:
    """Test sum tree data structure."""
    
    def test_initialization(self):
        """Test sum tree initialization."""
        capacity = 8
        tree = SumTree(capacity)
        
        assert tree.capacity == capacity
        assert len(tree.tree) == 2 * capacity - 1
        assert tree.n_entries == 0
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving from sum tree."""
        tree = SumTree(capacity=4)
        
        # Add elements with priorities
        priorities = [1.0, 2.0, 3.0, 4.0]
        data = ['a', 'b', 'c', 'd']
        
        for p, d in zip(priorities, data):
            tree.add(p, d)
        
        # Total should be sum of priorities
        assert tree.total() == sum(priorities)
        
        # Test retrieval
        # With cumsum [1, 3, 6, 10], s=5 should return 'c'
        idx, priority, retrieved_data = tree.get(5.0)
        assert retrieved_data == 'c'
        assert priority == 3.0
    
    def test_update_priority(self):
        """Test updating priorities."""
        tree = SumTree(capacity=4)
        
        # Add initial elements
        for i in range(4):
            tree.add(1.0, f'data_{i}')
        
        # Update priority
        # Tree indices for data are capacity-1 to 2*capacity-2
        data_idx = tree.capacity - 1  # First data element
        tree.update(data_idx, 5.0)
        
        # Total should reflect update
        assert tree.total() == 8.0  # 5 + 1 + 1 + 1
    
    def test_circular_buffer_behavior(self):
        """Test that sum tree acts as circular buffer."""
        tree = SumTree(capacity=2)
        
        # Add more than capacity
        tree.add(1.0, 'a')
        tree.add(2.0, 'b')
        tree.add(3.0, 'c')  # Should overwrite 'a'
        
        # Check that old data is overwritten
        assert tree.n_entries == 2
        assert tree.total() == 5.0  # 2 + 3


class TestPrioritizedExperienceBuffer:
    """Test prioritized experience replay buffer."""
    
    @pytest.fixture
    def buffer(self):
        return PrioritizedExperienceBuffer(
            capacity=100,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=1000
        )
    
    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert buffer.beta_start == 0.4
        assert len(buffer) == 0
    
    def test_add_experience(self, buffer):
        """Test adding experiences."""
        # Create experience
        states = {'agent1': torch.randn(4), 'agent2': torch.randn(6)}
        actions = {'agent1': 0, 'agent2': 1}
        rewards = {'agent1': 1.0, 'agent2': 0.5}
        next_states = {'agent1': torch.randn(4), 'agent2': torch.randn(6)}
        dones = {'agent1': False, 'agent2': False}
        log_probs = {'agent1': -1.0, 'agent2': -1.5}
        
        # Add experience
        buffer.add(states, actions, rewards, next_states, dones, log_probs)
        
        assert len(buffer) == 1
    
    def test_sampling(self, buffer):
        """Test experience sampling."""
        # Add multiple experiences
        for i in range(50):
            states = {'agent1': torch.randn(4)}
            actions = {'agent1': i % 3}
            rewards = {'agent1': float(i)}
            next_states = {'agent1': torch.randn(4)}
            dones = {'agent1': i % 10 == 0}
            log_probs = {'agent1': -float(i) / 10}
            td_errors = {'agent1': abs(i - 25) / 25}  # Higher error near edges
            
            buffer.add(states, actions, rewards, next_states, dones, log_probs, td_errors=td_errors)
        
        # Sample batch
        batch_size = 16
        batch, indices = buffer.sample(batch_size)
        
        # Check batch structure
        assert isinstance(batch, BatchedExperience)
        assert 'agent1' in batch.states
        assert batch.states['agent1'].shape[0] == batch_size
        assert batch.weights.shape == (batch_size,)
        assert len(indices) == batch_size
        
        # Check importance sampling weights
        assert (batch.weights > 0).all()
        assert (batch.weights <= 1).all()
    
    def test_priority_update(self, buffer):
        """Test updating priorities."""
        # Add experiences
        for i in range(10):
            buffer.add(
                states={'agent1': torch.randn(4)},
                actions={'agent1': 0},
                rewards={'agent1': 1.0},
                next_states={'agent1': torch.randn(4)},
                dones={'agent1': False},
                log_probs={'agent1': -1.0}
            )
        
        # Sample
        batch, indices = buffer.sample(5)
        
        # Update priorities with new TD errors
        new_td_errors = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        buffer.update_priorities(indices, new_td_errors)
        
        # Max priority should be updated
        assert buffer.max_priority >= 0.8
    
    def test_beta_annealing(self, buffer):
        """Test beta annealing for importance sampling."""
        initial_beta = buffer.beta()
        
        # Step many times
        for _ in range(500):
            if len(buffer) > 10:
                buffer.sample(5)
        
        # Beta should increase
        assert buffer.beta() > initial_beta
        
        # Step to completion
        buffer.frame = buffer.beta_frames + 1
        assert buffer.beta() == 1.0


class TestUniformExperienceBuffer:
    """Test uniform experience replay buffer."""
    
    @pytest.fixture
    def buffer(self):
        return UniformExperienceBuffer(capacity=100)
    
    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 100
        assert len(buffer) == 0
    
    def test_add_and_sample(self, buffer):
        """Test adding and sampling uniformly."""
        # Add experiences
        for i in range(50):
            buffer.add(
                states={'agent1': torch.randn(4)},
                actions={'agent1': i % 3},
                rewards={'agent1': float(i)},
                next_states={'agent1': torch.randn(4)},
                dones={'agent1': False},
                log_probs={'agent1': -1.0}
            )
        
        # Sample
        batch, _ = buffer.sample(10)
        
        # Check batch
        assert batch.states['agent1'].shape[0] == 10
        assert torch.allclose(batch.weights, torch.ones(10))
    
    def test_capacity_limit(self, buffer):
        """Test that buffer respects capacity."""
        # Add more than capacity
        for i in range(150):
            buffer.add(
                states={'agent1': torch.randn(4)},
                actions={'agent1': 0},
                rewards={'agent1': 1.0},
                next_states={'agent1': torch.randn(4)},
                dones={'agent1': False},
                log_probs={'agent1': -1.0}
            )
        
        # Should only keep last 100
        assert len(buffer) == 100


class TestBatchedExperience:
    """Test batched experience structure."""
    
    def test_batched_experience_creation(self):
        """Test creating batched experience."""
        batch_size = 8
        
        batch = BatchedExperience(
            states={'agent1': torch.randn(batch_size, 4)},
            actions={'agent1': torch.randint(0, 3, (batch_size,))},
            rewards={'agent1': torch.randn(batch_size)},
            next_states={'agent1': torch.randn(batch_size, 4)},
            dones={'agent1': torch.zeros(batch_size)},
            log_probs={'agent1': torch.randn(batch_size)},
            values={'agent1': torch.randn(batch_size)},
            advantages={'agent1': torch.randn(batch_size)},
            returns={'agent1': torch.randn(batch_size)},
            weights=torch.ones(batch_size)
        )
        
        # Check all fields accessible
        assert batch.states['agent1'].shape[0] == batch_size
        assert batch.actions['agent1'].shape[0] == batch_size
        assert batch.weights.shape[0] == batch_size


class TestBufferIntegration:
    """Test buffer integration with training."""
    
    def test_buffer_with_multiple_agents(self):
        """Test buffer with multiple agents."""
        buffer = PrioritizedExperienceBuffer(capacity=100)
        
        # Add multi-agent experiences
        for i in range(20):
            states = {
                'agent1': torch.randn(4),
                'agent2': torch.randn(6),
                'agent3': torch.randn(3)
            }
            actions = {
                'agent1': i % 3,
                'agent2': (i + 1) % 3,
                'agent3': (i + 2) % 3
            }
            rewards = {
                'agent1': 0.1,
                'agent2': 0.2,
                'agent3': 0.3
            }
            next_states = {
                'agent1': torch.randn(4),
                'agent2': torch.randn(6),
                'agent3': torch.randn(3)
            }
            dones = {
                'agent1': False,
                'agent2': False,
                'agent3': i == 19
            }
            log_probs = {
                'agent1': -1.0,
                'agent2': -1.5,
                'agent3': -2.0
            }
            
            buffer.add(states, actions, rewards, next_states, dones, log_probs)
        
        # Sample batch
        batch, indices = buffer.sample(8)
        
        # Verify all agents present
        assert all(agent in batch.states for agent in ['agent1', 'agent2', 'agent3'])
        assert all(agent in batch.actions for agent in ['agent1', 'agent2', 'agent3'])
        
        # Verify batch consistency
        for agent in ['agent1', 'agent2', 'agent3']:
            assert batch.states[agent].shape[0] == 8
            assert batch.actions[agent].shape[0] == 8
            assert batch.rewards[agent].shape[0] == 8
    
    def test_buffer_memory_efficiency(self):
        """Test memory efficiency with large buffer."""
        buffer = PrioritizedExperienceBuffer(capacity=10000)
        
        # Add many experiences
        for i in range(5000):
            buffer.add(
                states={'agent1': torch.randn(100)},  # Large state
                actions={'agent1': 0},
                rewards={'agent1': 1.0},
                next_states={'agent1': torch.randn(100)},
                dones={'agent1': False},
                log_probs={'agent1': -1.0}
            )
        
        # Should handle large buffer efficiently
        assert len(buffer) == 5000
        
        # Sampling should be fast
        import time
        start = time.time()
        batch, _ = buffer.sample(256)
        duration = time.time() - start
        
        assert duration < 0.1  # Should be fast
        assert batch.states['agent1'].shape == (256, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])