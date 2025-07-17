"""
Unit tests for neural network architectures
Tests all actor and critic implementations
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architectures import MLMIActor, NWRQKActor, MMDActor, CentralizedCritic
from models.base import BaseStrategicActor


class TestBaseStrategicActor:
    """Test the base strategic actor abstract class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseStrategicActor(input_dim=4)
    
    def test_actor_interface(self):
        """Test that all actors implement required interface."""
        actors = [MLMIActor(), NWRQKActor(), MMDActor()]
        
        for actor in actors:
            # Check required methods exist
            assert hasattr(actor, 'forward')
            assert hasattr(actor, 'get_action')
            assert hasattr(actor, 'evaluate_actions')
            assert hasattr(actor, '_extract_features')
            assert hasattr(actor, '_temporal_modeling')


class TestMLMIActor:
    """Test MLMI strategic actor."""
    
    @pytest.fixture
    def actor(self):
        return MLMIActor(
            input_dim=4,
            hidden_dims=[256, 128, 64],
            action_dim=3,
            dropout_rate=0.1
        )
    
    def test_initialization(self, actor):
        """Test actor initialization."""
        assert actor.input_dim == 4
        assert actor.action_dim == 3
        assert len(actor.conv_layers) == 2  # Two conv layers with kernel sizes [3, 5]
        assert hasattr(actor, 'temporal')
        assert hasattr(actor, 'attention')
        assert hasattr(actor, 'output_head')
        assert hasattr(actor, 'value_head')
    
    def test_forward_pass(self, actor):
        """Test forward pass through network."""
        batch_size = 32
        sequence_length = 48
        
        # Create input
        state = torch.randn(batch_size, actor.input_dim, sequence_length)
        
        # Forward pass
        output = actor(state)
        
        # Check output structure
        assert 'action' in output
        assert 'action_probs' in output
        assert 'log_prob' in output
        assert 'entropy' in output
        assert 'logits' in output
        assert 'value' in output
        
        # Check shapes
        assert output['action'].shape == (batch_size,)
        assert output['action_probs'].shape == (batch_size, actor.action_dim)
        assert output['log_prob'].shape == (batch_size,)
        assert output['entropy'].shape == (batch_size,)
        assert output['value'].shape == (batch_size,)
        
        # Check probability constraints
        assert torch.allclose(output['action_probs'].sum(dim=-1), torch.ones(batch_size))
        assert (output['action_probs'] >= 0).all()
        assert (output['action_probs'] <= 1).all()
    
    def test_deterministic_mode(self, actor):
        """Test deterministic action selection."""
        state = torch.randn(1, actor.input_dim, 48)
        
        # Get deterministic action
        output_det = actor(state, deterministic=True)
        
        # Action should be argmax of probabilities
        expected_action = output_det['action_probs'].argmax(dim=-1)
        assert torch.equal(output_det['action'], expected_action)
    
    def test_gradient_flow(self, actor):
        """Test gradient flow through network."""
        state = torch.randn(1, actor.input_dim, 48, requires_grad=True)
        output = actor(state)
        # Use both value and policy loss to ensure gradients flow through all parts
        loss = output['value'].sum() + output['log_prob'].sum() + output['entropy'].sum()
        loss.backward()
        
        # Check gradients exist
        assert state.grad is not None
        # Check that most parameters have gradients
        params_with_grad = sum(1 for p in actor.parameters() if p.grad is not None)
        total_params = sum(1 for p in actor.parameters())
        assert params_with_grad / total_params > 0.95  # At least 95% of params should have gradients


class TestNWRQKActor:
    """Test NWRQK strategic actor."""
    
    @pytest.fixture
    def actor(self):
        return NWRQKActor(
            input_dim=6,
            hidden_dims=[256, 128, 64],
            action_dim=3,
            dropout_rate=0.1
        )
    
    def test_initialization(self, actor):
        """Test actor initialization."""
        assert actor.input_dim == 6
        assert actor.action_dim == 3
        assert len(actor.conv_layers) == 3  # Three conv layers
        assert hasattr(actor, 'temporal')
        assert actor.temporal.bidirectional  # Should be bidirectional
    
    def test_forward_pass(self, actor):
        """Test forward pass with 6D input."""
        batch_size = 16
        state = torch.randn(batch_size, actor.input_dim, 48)
        
        output = actor(state)
        
        # Verify output
        assert output['action_probs'].shape == (batch_size, 3)
        assert output['action'].max() < 3
        assert output['action'].min() >= 0


class TestMMDActor:
    """Test MMD (Regime) strategic actor."""
    
    @pytest.fixture
    def actor(self):
        return MMDActor(
            input_dim=3,
            hidden_dims=[128, 64, 32],
            action_dim=3,
            dropout_rate=0.1
        )
    
    def test_initialization(self, actor):
        """Test actor initialization."""
        assert actor.input_dim == 3
        assert actor.action_dim == 3
        assert actor.temporal.rnn_type == 'gru'  # Should use GRU
        assert hasattr(actor, 'volatility_adjust')
    
    def test_volatility_adjustment(self, actor):
        """Test volatility adjustment mechanism."""
        batch_size = 8
        # Create state with volatility as second feature
        state = torch.randn(batch_size, actor.input_dim, 48)
        state[:, 1, :] = torch.rand(batch_size, 48) * 2  # Volatility values
        
        output = actor(state)
        
        # Check that output is valid
        assert 'logits' in output
        assert output['action_probs'].shape == (batch_size, 3)


class TestCentralizedCritic:
    """Test centralized critic network."""
    
    @pytest.fixture
    def critic(self):
        return CentralizedCritic(
            state_dim=13,  # 4 + 6 + 3
            n_agents=3,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.1
        )
    
    def test_initialization(self, critic):
        """Test critic initialization."""
        assert critic.state_dim == 13
        assert critic.n_agents == 3
        assert hasattr(critic, 'feature_extractor')
        assert hasattr(critic, 'value_head')
        assert hasattr(critic, 'agent_value_heads')
        assert len(critic.agent_value_heads) == 3
    
    def test_forward_pass(self, critic):
        """Test forward pass through critic."""
        batch_size = 32
        combined_state = torch.randn(batch_size, critic.state_dim)
        
        output = critic(combined_state)
        
        # Check output
        assert 'value' in output
        assert output['value'].shape == (batch_size,)
        assert output['value'].dim() == 1
    
    def test_with_agent_states(self, critic):
        """Test critic with individual agent states."""
        batch_size = 16
        combined_state = torch.randn(batch_size, critic.state_dim)
        
        agent_states = {
            'mlmi': torch.randn(batch_size, 4),
            'nwrqk': torch.randn(batch_size, 6),
            'mmd': torch.randn(batch_size, 3)
        }
        
        output = critic(combined_state, agent_states)
        
        # Check outputs
        assert 'value' in output
        assert 'agent_values' in output
        assert len(output['agent_values']) == 3
        
        for agent_name, agent_value in output['agent_values'].items():
            assert agent_value.shape == (batch_size,)
    
    def test_gradient_flow(self, critic):
        """Test gradient flow through critic."""
        state = torch.randn(1, critic.state_dim, requires_grad=True)
        value = critic(state)['value']
        loss = value.sum()
        loss.backward()
        
        assert state.grad is not None
        # Check that most parameters have gradients
        # Note: agent_value_heads won't have gradients unless we compute loss on them
        params_with_grad = sum(1 for p in critic.parameters() if p.grad is not None)
        total_params = sum(1 for p in critic.parameters())
        assert params_with_grad / total_params > 0.7  # At least 70% of params should have gradients


class TestArchitectureIntegration:
    """Test integration between actors and critic."""
    
    def test_actor_critic_compatibility(self):
        """Test that actor outputs are compatible with critic inputs."""
        # Create actors
        actors = {
            'mlmi': MLMIActor(input_dim=4),
            'nwrqk': NWRQKActor(input_dim=6),
            'mmd': MMDActor(input_dim=3)
        }
        
        # Create critic
        total_dim = sum(actor.input_dim for actor in actors.values())
        critic = CentralizedCritic(state_dim=total_dim, n_agents=3)
        
        # Create inputs
        batch_size = 8
        states = {
            'mlmi': torch.randn(batch_size, 4, 48),
            'nwrqk': torch.randn(batch_size, 6, 48),
            'mmd': torch.randn(batch_size, 3, 48)
        }
        
        # Get actor outputs
        actor_outputs = {}
        for name, actor in actors.items():
            actor_outputs[name] = actor(states[name])
        
        # Combine states for critic (simplified - just flatten)
        combined_states = []
        for name in ['mlmi', 'nwrqk', 'mmd']:
            # Take mean across sequence dimension
            state_features = states[name].mean(dim=-1)
            combined_states.append(state_features)
        
        combined_state = torch.cat(combined_states, dim=-1)
        
        # Get critic value
        critic_output = critic(combined_state)
        
        # Verify shapes match
        assert critic_output['value'].shape[0] == batch_size
        for name, output in actor_outputs.items():
            assert output['action'].shape[0] == batch_size
    
    def test_memory_efficiency(self):
        """Test memory efficiency of networks."""
        import torch.cuda
        
        # Skip if no GPU
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device('cuda')
        
        # Create networks
        actors = {
            'mlmi': MLMIActor().to(device),
            'nwrqk': NWRQKActor().to(device),
            'mmd': MMDActor().to(device)
        }
        critic = CentralizedCritic(state_dim=13).to(device)
        
        # Large batch
        batch_size = 256
        states = {
            'mlmi': torch.randn(batch_size, 4, 48).to(device),
            'nwrqk': torch.randn(batch_size, 6, 48).to(device),
            'mmd': torch.randn(batch_size, 3, 48).to(device)
        }
        
        # Forward pass should not OOM
        try:
            for name, actor in actors.items():
                _ = actor(states[name])
            
            combined_state = torch.randn(batch_size, 13).to(device)
            _ = critic(combined_state)
            
        except torch.cuda.OutOfMemoryError:
            pytest.fail("Out of memory with reasonable batch size")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])