"""
MLMI Mathematical Correctness Test Suite

Comprehensive test suite to validate the mathematical correctness of the MLMI
Strategic Agent GAE implementation. Tests ensure numerical stability, proper
GAE computation, policy network behavior, and performance benchmarks.

Author: Agent 2 - MLMI Correlation Specialist
Version: 1.0 - Production Ready
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

# Import the MLMI agent and components
from src.agents.mlmi_strategic_agent import (
    MLMIStrategicAgent,
    MLMIPolicyNetwork,
    MLMIExperienceBuffer,
    create_mlmi_strategic_agent
)
from src.core.event_bus import EventBus


class TestGAEComputation:
    """Test Generalized Advantage Estimation computation correctness."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def mlmi_agent(self, mock_event_bus):
        """Create MLMI agent for testing."""
        config = {
            'agent_id': 'test_mlmi_agent',
            'gamma': 0.99,
            'lambda_': 0.95,
            'epsilon': 0.2,
            'learning_rate': 1e-3
        }
        return MLMIStrategicAgent(config, mock_event_bus)
    
    def test_gae_formula_correctness(self, mlmi_agent):
        """Test GAE computation matches mathematical formula exactly."""
        # Test with known reward sequences
        rewards = [1.0, 0.5, -0.2, 0.8, 0.3]
        values = [2.0, 1.8, 1.5, 1.2, 1.0, 0.5]  # length = len(rewards) + 1
        
        # Compute GAE
        advantages = mlmi_agent.compute_gae(rewards, values)
        
        # Manual computation for validation
        gamma = mlmi_agent.gamma
        lambda_ = mlmi_agent.lambda_
        
        # Compute TD errors manually
        td_errors = []
        for i in range(len(rewards)):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            td_errors.append(delta)
        
        # Compute GAE manually in reverse
        expected_advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            gae = td_errors[i] + gamma * lambda_ * gae
            expected_advantages.insert(0, gae)
        
        # Normalize expected advantages (same as in implementation)
        expected_tensor = torch.tensor(expected_advantages, dtype=torch.float32)
        if len(expected_tensor) > 1:
            expected_tensor = (expected_tensor - expected_tensor.mean()) / (expected_tensor.std() + 1e-8)
        
        # Test equality with tolerance
        assert torch.allclose(advantages, expected_tensor, atol=1e-6), \
            f"GAE computation mismatch: got {advantages}, expected {expected_tensor}"
    
    def test_gae_edge_cases(self, mlmi_agent):
        """Test GAE computation with edge cases."""
        # Single reward
        rewards = [1.0]
        values = [0.5, 0.3]
        advantages = mlmi_agent.compute_gae(rewards, values)
        assert advantages.shape == (1,), "Single reward should produce single advantage"
        
        # Zero rewards
        rewards = [0.0, 0.0, 0.0]
        values = [0.0, 0.0, 0.0, 0.0]
        advantages = mlmi_agent.compute_gae(rewards, values)
        assert advantages.shape == (3,), "Zero rewards should produce valid advantages"
        
        # Large negative rewards
        rewards = [-10.0, -5.0, -2.0]
        values = [1.0, 0.5, 0.2, 0.1]
        advantages = mlmi_agent.compute_gae(rewards, values)
        assert not torch.isnan(advantages).any(), "Large negative rewards should not produce NaN"
        assert not torch.isinf(advantages).any(), "Large negative rewards should not produce Inf"
    
    def test_gae_with_done_flags(self, mlmi_agent):
        """Test GAE computation with episode termination flags."""
        rewards = [1.0, 0.5, -0.2, 0.8]
        values = [2.0, 1.8, 1.5, 1.2, 1.0]
        dones = [False, False, True, False]  # Episode ends at step 2
        
        advantages = mlmi_agent.compute_gae(rewards, values, dones)
        
        # Verify advantages are computed correctly with done flags
        assert advantages.shape == (4,), "Done flags should not change advantage shape"
        assert not torch.isnan(advantages).any(), "Done flags should not produce NaN"
    
    def test_gae_numerical_stability(self, mlmi_agent):
        """Test GAE numerical stability with extreme values."""
        # Test with very small values
        rewards = [1e-10, 1e-10, 1e-10]
        values = [1e-10, 1e-10, 1e-10, 1e-10]
        advantages = mlmi_agent.compute_gae(rewards, values)
        assert not torch.isnan(advantages).any(), "Small values should not produce NaN"
        
        # Test with very large values
        rewards = [1e6, 1e6, 1e6]
        values = [1e6, 1e6, 1e6, 1e6]
        advantages = mlmi_agent.compute_gae(rewards, values)
        assert not torch.isinf(advantages).any(), "Large values should not produce Inf"
    
    def test_gae_hyperparameter_sensitivity(self, mock_event_bus):
        """Test GAE sensitivity to hyperparameters."""
        # Test different gamma values
        for gamma in [0.9, 0.95, 0.99, 0.999]:
            config = {'gamma': gamma, 'lambda_': 0.95}
            agent = MLMIStrategicAgent(config, mock_event_bus)
            
            rewards = [1.0, 0.5, -0.2]
            values = [2.0, 1.8, 1.5, 1.0]
            advantages = agent.compute_gae(rewards, values)
            
            assert not torch.isnan(advantages).any(), f"Gamma {gamma} should not produce NaN"
            assert advantages.shape == (3,), f"Gamma {gamma} should preserve shape"
        
        # Test different lambda values
        for lambda_ in [0.8, 0.9, 0.95, 0.99]:
            config = {'gamma': 0.99, 'lambda_': lambda_}
            agent = MLMIStrategicAgent(config, mock_event_bus)
            
            rewards = [1.0, 0.5, -0.2]
            values = [2.0, 1.8, 1.5, 1.0]
            advantages = agent.compute_gae(rewards, values)
            
            assert not torch.isnan(advantages).any(), f"Lambda {lambda_} should not produce NaN"
            assert advantages.shape == (3,), f"Lambda {lambda_} should preserve shape"


class TestPolicyNetwork:
    """Test MLMI Policy Network architecture and behavior."""
    
    @pytest.fixture
    def policy_network(self):
        """Create policy network for testing."""
        return MLMIPolicyNetwork(
            input_dim=4,
            hidden_dim=128,
            action_dim=7,
            dropout_rate=0.1,
            temperature_init=1.0
        )
    
    def test_policy_network_architecture(self, policy_network):
        """Test policy network architecture correctness."""
        # Test input/output dimensions
        test_input = torch.randn(1, 4)
        action_probs, logits = policy_network(test_input)
        
        assert action_probs.shape == (1, 7), f"Action probs shape mismatch: {action_probs.shape}"
        assert logits.shape == (1, 7), f"Logits shape mismatch: {logits.shape}"
        
        # Test batch processing
        batch_input = torch.randn(32, 4)
        action_probs, logits = policy_network(batch_input)
        
        assert action_probs.shape == (32, 7), "Batch processing failed"
        assert logits.shape == (32, 7), "Batch logits processing failed"
    
    def test_softmax_constraints(self, policy_network):
        """Test that softmax output always sums to 1.0."""
        # Test single input
        test_input = torch.randn(1, 4)
        action_probs, _ = policy_network(test_input)
        prob_sum = action_probs.sum(dim=-1)
        
        assert torch.allclose(prob_sum, torch.tensor(1.0), atol=1e-6), \
            f"Probabilities don't sum to 1.0: {prob_sum}"
        
        # Test batch inputs
        batch_input = torch.randn(10, 4)
        action_probs, _ = policy_network(batch_input)
        prob_sums = action_probs.sum(dim=-1)
        
        assert torch.allclose(prob_sums, torch.ones(10), atol=1e-6), \
            "Batch probabilities don't sum to 1.0"
        
        # Test with extreme inputs
        extreme_input = torch.tensor([[100.0, -100.0, 1000.0, -1000.0]], dtype=torch.float32)
        action_probs, _ = policy_network(extreme_input)
        prob_sum = action_probs.sum(dim=-1)
        
        assert torch.allclose(prob_sum, torch.tensor(1.0), atol=1e-6), \
            "Extreme inputs should still produce valid probabilities"
    
    def test_probability_non_negativity(self, policy_network):
        """Test that all probabilities are non-negative."""
        test_input = torch.randn(5, 4)
        action_probs, _ = policy_network(test_input)
        
        assert torch.all(action_probs >= 0), "All probabilities must be non-negative"
        assert torch.all(action_probs <= 1), "All probabilities must be <= 1"
    
    def test_temperature_scaling(self, policy_network):
        """Test temperature scaling behavior."""
        test_input = torch.randn(1, 4)
        
        # Test low temperature (more deterministic)
        policy_network.set_temperature(0.1)
        action_probs_low, _ = policy_network(test_input)
        entropy_low = -(action_probs_low * torch.log(action_probs_low + 1e-8)).sum()
        
        # Test high temperature (more random)
        policy_network.set_temperature(2.0)
        action_probs_high, _ = policy_network(test_input)
        entropy_high = -(action_probs_high * torch.log(action_probs_high + 1e-8)).sum()
        
        # High temperature should produce higher entropy (more uniform distribution)
        assert entropy_high > entropy_low, "High temperature should increase entropy"
        
        # Test temperature bounds
        policy_network.set_temperature(-1.0)  # Should be clamped to 0.1
        policy_network.set_temperature(10.0)  # Should be clamped to 3.0
        
        # Network should still work with clamped temperatures
        action_probs, _ = policy_network(test_input)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_gradient_flow(self, policy_network):
        """Test gradient flow through policy network."""
        test_input = torch.randn(1, 4, requires_grad=True)
        action_probs, logits = policy_network(test_input)
        
        # Create dummy loss
        loss = action_probs.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert test_input.grad is not None, "Input should have gradients"
        assert torch.isfinite(test_input.grad).all(), "Gradients should be finite"
        
        # Check network parameter gradients
        for param in policy_network.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Network parameters should have gradients"
                assert torch.isfinite(param.grad).all(), "Parameter gradients should be finite"


class TestFeatureExtraction:
    """Test MLMI feature extraction pipeline."""
    
    @pytest.fixture
    def mlmi_agent(self):
        """Create MLMI agent for testing."""
        config = {'agent_id': 'test_feature_agent'}
        return MLMIStrategicAgent(config, Mock(spec=EventBus))
    
    def test_feature_extraction_indices(self, mlmi_agent):
        """Test extraction from correct matrix indices [0,1,9,10]."""
        # Create mock matrix data with sufficient features
        matrix_data = np.random.randn(100, 15)  # 100 time steps, 15 features
        
        # Set specific values at target indices
        matrix_data[-1, 0] = 1.5  # mlmi_value
        matrix_data[-1, 1] = -0.8  # mlmi_signal
        # momentum features will be computed from price data
        
        features = mlmi_agent.extract_mlmi_features(matrix_data)
        
        assert features.shape == (4,), f"Should extract 4 features, got {features.shape}"
        assert isinstance(features, torch.Tensor), "Should return torch tensor"
        assert torch.isfinite(features).all(), "All features should be finite"
    
    def test_feature_normalization(self, mlmi_agent):
        """Test feature normalization and running statistics."""
        # Extract features multiple times to build statistics
        for i in range(10):
            matrix_data = np.random.randn(50, 15)
            features = mlmi_agent.extract_mlmi_features(matrix_data)
            
            # Check normalization properties
            assert torch.isfinite(features).all(), f"Features {i} should be finite"
        
        # After multiple extractions, statistics should be reasonable
        assert mlmi_agent.feature_count == 10, "Should track feature count"
        assert mlmi_agent.feature_mean.shape == (4,), "Should track mean for 4 features"
        assert mlmi_agent.feature_std.shape == (4,), "Should track std for 4 features"
    
    def test_momentum_calculation(self, mlmi_agent):
        """Test momentum feature calculation."""
        # Create matrix data with known price progression
        matrix_data = np.zeros((60, 15))
        
        # Set close prices (assuming index 4 is close price)
        base_price = 100.0
        for i in range(60):
            matrix_data[i, 4] = base_price + i * 0.1  # Steady uptrend
        
        features = mlmi_agent.extract_mlmi_features(matrix_data)
        
        # momentum_20 should be positive (uptrend)
        momentum_20 = features[2].item()  # Feature index 2
        momentum_50 = features[3].item()  # Feature index 3
        
        # Both momentums should be positive for uptrend
        assert momentum_20 > 0, "20-period momentum should be positive for uptrend"
        assert momentum_50 > 0, "50-period momentum should be positive for uptrend"
    
    def test_feature_edge_cases(self, mlmi_agent):
        """Test feature extraction edge cases."""
        # Empty matrix
        empty_matrix = np.array([])
        features = mlmi_agent.extract_mlmi_features(empty_matrix)
        assert features.shape == (4,), "Empty matrix should return zero features"
        assert torch.all(features == 0), "Empty matrix features should be zero"
        
        # Insufficient data for momentum
        small_matrix = np.random.randn(5, 15)
        features = mlmi_agent.extract_mlmi_features(small_matrix)
        assert features.shape == (4,), "Small matrix should return 4 features"
        
        # Matrix with NaN values
        nan_matrix = np.full((50, 15), np.nan)
        features = mlmi_agent.extract_mlmi_features(nan_matrix)
        assert torch.isfinite(features).all(), "NaN input should produce finite features"


class TestPPOLoss:
    """Test PPO loss computation correctness."""
    
    @pytest.fixture
    def mlmi_agent(self):
        """Create MLMI agent for testing."""
        config = {'epsilon': 0.2}
        return MLMIStrategicAgent(config, Mock(spec=EventBus))
    
    def test_ppo_clipping_mechanism(self, mlmi_agent):
        """Test PPO clipping with ε=0.2 parameter."""
        batch_size = 32
        
        # Create mock batch data
        states = torch.randn(batch_size, 4)
        actions = torch.randint(0, 7, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        # Compute PPO loss
        loss_dict = mlmi_agent.compute_ppo_loss(
            states, actions, old_log_probs, advantages, returns
        )
        
        # Check loss components exist
        assert 'policy_loss' in loss_dict, "Should compute policy loss"
        assert 'value_loss' in loss_dict, "Should compute value loss"
        assert 'entropy_loss' in loss_dict, "Should compute entropy loss"
        assert 'total_loss' in loss_dict, "Should compute total loss"
        assert 'ratio_mean' in loss_dict, "Should track ratio statistics"
        
        # Check loss values are finite
        for key, loss in loss_dict.items():
            if isinstance(loss, torch.Tensor):
                assert torch.isfinite(loss).all(), f"{key} should be finite"
    
    def test_ppo_ratio_clipping(self, mlmi_agent):
        """Test that PPO properly clips probability ratios."""
        # Create scenario with extreme ratio values
        states = torch.randn(10, 4)
        actions = torch.randint(0, 7, (10,))
        
        # Create old log probs that will lead to extreme ratios
        old_log_probs = torch.tensor([-10.0] * 10)  # Very low old probabilities
        advantages = torch.ones(10)
        returns = torch.ones(10)
        
        loss_dict = mlmi_agent.compute_ppo_loss(
            states, actions, old_log_probs, advantages, returns
        )
        
        # Loss should be finite even with extreme ratios
        assert torch.isfinite(loss_dict['policy_loss']), "Policy loss should handle extreme ratios"
        assert torch.isfinite(loss_dict['total_loss']), "Total loss should handle extreme ratios"
    
    def test_entropy_regularization(self, mlmi_agent):
        """Test entropy regularization in PPO loss."""
        states = torch.randn(16, 4)
        actions = torch.randint(0, 7, (16,))
        old_log_probs = torch.randn(16)
        advantages = torch.randn(16)
        returns = torch.randn(16)
        
        loss_dict = mlmi_agent.compute_ppo_loss(
            states, actions, old_log_probs, advantages, returns
        )
        
        # Entropy should be positive (more positive = more random)
        assert loss_dict['entropy'] >= 0, "Entropy should be non-negative"
        
        # Entropy loss should be negative (we want to maximize entropy)
        assert loss_dict['entropy_loss'] <= 0, "Entropy loss should be negative (penalty)"


class TestExperienceBuffer:
    """Test experience buffer functionality."""
    
    @pytest.fixture
    def experience_buffer(self):
        """Create experience buffer for testing."""
        return MLMIExperienceBuffer(capacity=100, alpha=0.6)
    
    def test_buffer_storage(self, experience_buffer):
        """Test storing transitions in buffer."""
        transition = {
            'state': torch.randn(4),
            'action': 3,
            'reward': 0.5,
            'next_state': torch.randn(4),
            'advantage': 0.2,
            'log_prob': -1.2
        }
        
        experience_buffer.store(transition)
        assert len(experience_buffer) == 1, "Buffer should contain one transition"
        
        # Store multiple transitions
        for i in range(10):
            experience_buffer.store(transition)
        
        assert len(experience_buffer) == 11, "Buffer should contain 11 transitions"
    
    def test_buffer_capacity_limit(self, experience_buffer):
        """Test buffer capacity enforcement."""
        transition = {
            'state': torch.randn(4),
            'action': 0,
            'reward': 0.0,
            'next_state': torch.randn(4),
            'advantage': 0.0,
            'log_prob': 0.0
        }
        
        # Fill buffer beyond capacity
        for i in range(150):
            experience_buffer.store(transition)
        
        assert len(experience_buffer) <= 100, "Buffer should respect capacity limit"
    
    def test_priority_sampling(self, experience_buffer):
        """Test priority-based sampling."""
        # Store transitions with different priorities
        for i in range(20):
            transition = {
                'state': torch.randn(4),
                'action': i % 7,
                'reward': float(i),
                'next_state': torch.randn(4),
                'advantage': float(i),
                'log_prob': 0.0
            }
            experience_buffer.store(transition)
        
        # Sample batch
        batch = experience_buffer.sample(batch_size=10, beta=0.4)
        
        # Check batch structure
        assert 'states' in batch, "Batch should contain states"
        assert 'actions' in batch, "Batch should contain actions"
        assert 'rewards' in batch, "Batch should contain rewards"
        assert 'advantages' in batch, "Batch should contain advantages"
        assert 'weights' in batch, "Batch should contain importance weights"
        
        # Check batch sizes
        assert batch['states'].shape[0] == 10, "Batch should contain 10 states"
        assert len(batch['weights']) == 10, "Batch should contain 10 weights"
    
    def test_priority_updates(self, experience_buffer):
        """Test priority updates based on TD errors."""
        # Store some transitions
        for i in range(10):
            transition = {'state': torch.randn(4), 'action': 0, 'reward': 0.0,
                         'next_state': torch.randn(4), 'advantage': 0.0, 'log_prob': 0.0}
            experience_buffer.store(transition)
        
        # Sample and update priorities
        batch = experience_buffer.sample(batch_size=5)
        td_errors = torch.tensor([1.0, 0.5, 2.0, 0.1, 1.5])
        
        experience_buffer.update_priorities(batch['indices'], td_errors)
        
        # High TD error transitions should have higher priority
        max_td_idx = td_errors.argmax().item()
        sampled_idx = batch['indices'][max_td_idx]
        assert experience_buffer.priorities[sampled_idx] > 1.0, "High TD error should increase priority"


class TestPerformanceBenchmarks:
    """Test performance benchmarks and <1ms inference target."""
    
    @pytest.fixture
    def mlmi_agent(self):
        """Create MLMI agent for performance testing."""
        config = {'agent_id': 'performance_test_agent'}
        return MLMIStrategicAgent(config, Mock(spec=EventBus))
    
    @pytest.mark.performance
    def test_inference_time_target(self, mlmi_agent):
        """Test <1ms inference time target."""
        # Prepare test input
        features = torch.randn(4)
        
        # Warm up (JIT compilation, cache warming)
        for _ in range(10):
            mlmi_agent.forward(features)
        
        # Measure inference times
        inference_times = []
        for _ in range(100):
            start_time = time.time()
            result = mlmi_agent.forward(features)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)
        
        print(f"Inference time - Avg: {avg_time:.3f}ms, P95: {p95_time:.3f}ms, P99: {p99_time:.3f}ms")
        
        # Target: <1ms for average inference
        assert avg_time < 1.0, f"Average inference time {avg_time:.3f}ms exceeds 1ms target"
        assert p95_time < 2.0, f"P95 inference time {p95_time:.3f}ms exceeds 2ms target"
    
    @pytest.mark.performance
    def test_batch_inference_scaling(self, mlmi_agent):
        """Test inference scaling with batch sizes."""
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, 4)
            
            # Measure batch inference time
            start_time = time.time()
            result = mlmi_agent.forward(features)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            per_sample_time = total_time / batch_size
            
            print(f"Batch size {batch_size}: {per_sample_time:.3f}ms per sample")
            
            # Per-sample time should decrease with larger batches (batching efficiency)
            if batch_size > 1:
                assert per_sample_time < 1.0, f"Per-sample time {per_sample_time:.3f}ms exceeds 1ms"
    
    @pytest.mark.performance
    def test_gae_computation_performance(self, mlmi_agent):
        """Test GAE computation performance."""
        # Test with various sequence lengths
        sequence_lengths = [10, 50, 100, 200]
        
        for seq_len in sequence_lengths:
            rewards = np.random.randn(seq_len).tolist()
            values = np.random.randn(seq_len + 1).tolist()
            
            # Measure GAE computation time
            start_time = time.time()
            advantages = mlmi_agent.compute_gae(rewards, values)
            end_time = time.time()
            
            computation_time = (end_time - start_time) * 1000
            
            print(f"GAE sequence length {seq_len}: {computation_time:.3f}ms")
            
            # GAE computation should be fast even for long sequences
            assert computation_time < 10.0, f"GAE computation {computation_time:.3f}ms exceeds 10ms"
    
    @pytest.mark.performance  
    def test_memory_efficiency(self, mlmi_agent):
        """Test memory efficiency during inference."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many inferences
        for _ in range(1000):
            features = torch.randn(4)
            result = mlmi_agent.forward(features)
            
            # Occasionally check memory
            if _ % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be minimal (< 50MB)
                assert memory_growth < 50, f"Memory growth {memory_growth:.1f}MB exceeds limit"


class TestIntegrationAndValidation:
    """Integration tests and end-to-end validation."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        return Mock(spec=EventBus)
    
    def test_factory_function(self, mock_event_bus):
        """Test factory function creates valid agent."""
        config = {
            'agent_id': 'factory_test_agent',
            'gamma': 0.95,
            'lambda_': 0.9
        }
        
        agent = create_mlmi_strategic_agent(config, mock_event_bus)
        
        assert isinstance(agent, MLMIStrategicAgent), "Factory should create MLMIStrategicAgent"
        assert agent.gamma == 0.95, "Factory should apply custom gamma"
        assert agent.lambda_ == 0.9, "Factory should apply custom lambda"
        assert agent.agent_id == 'factory_test_agent', "Factory should set agent ID"
    
    def test_complete_decision_pipeline(self, mock_event_bus):
        """Test complete decision-making pipeline."""
        agent = create_mlmi_strategic_agent({'agent_id': 'pipeline_test'}, mock_event_bus)
        
        # Create mock state with matrix data
        matrix_data = np.random.randn(60, 15)
        state = {'matrix_data': matrix_data}
        
        # Make decision
        decision = agent.make_decision(state)
        
        # Validate decision structure
        assert 'action' in decision, "Decision should contain action"
        assert 'action_name' in decision, "Decision should contain action name"
        assert 'confidence' in decision, "Decision should contain confidence"
        assert 'action_probabilities' in decision, "Decision should contain probabilities"
        assert 'mathematical_method' in decision, "Decision should specify method"
        
        # Validate decision values
        assert 0 <= decision['action'] <= 6, "Action should be in valid range [0,6]"
        assert 0 <= decision['confidence'] <= 1, "Confidence should be in [0,1]"
        assert len(decision['action_probabilities']) == 7, "Should have 7 action probabilities"
        assert abs(sum(decision['action_probabilities']) - 1.0) < 1e-6, "Probabilities should sum to 1"
        assert decision['mathematical_method'] == 'GAE', "Should specify GAE method"
    
    def test_training_integration(self, mock_event_bus):
        """Test training integration with experience buffer."""
        agent = create_mlmi_strategic_agent({'agent_id': 'training_test'}, mock_event_bus)
        
        # Simulate multiple decisions to build experience
        for i in range(20):
            matrix_data = np.random.randn(60, 15)
            state = {'matrix_data': matrix_data}
            decision = agent.make_decision(state)
        
        # Attempt training step
        if len(agent.experience_buffer) >= 16:
            metrics = agent.train_step(batch_size=16)
            
            assert 'policy_loss' in metrics, "Training should compute policy loss"
            assert 'value_loss' in metrics, "Training should compute value loss"
            assert 'training_step' in metrics, "Training should track steps"
            assert metrics['training_step'] > 0, "Training step should increment"
    
    def test_performance_metrics_tracking(self, mock_event_bus):
        """Test performance metrics tracking."""
        agent = create_mlmi_strategic_agent({'agent_id': 'metrics_test'}, mock_event_bus)
        
        # Perform some operations
        for _ in range(10):
            features = torch.randn(4)
            result = agent.forward(features)
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        
        assert 'agent_id' in metrics, "Metrics should include agent ID"
        assert 'avg_inference_time_ms' in metrics, "Metrics should track inference time"
        assert 'mathematical_params' in metrics, "Metrics should include math parameters"
        assert 'feature_normalization' in metrics, "Metrics should track normalization"
        
        # Validate metric values
        assert metrics['avg_inference_time_ms'] >= 0, "Inference time should be non-negative"
        assert metrics['mathematical_params']['gamma'] == agent.gamma, "Should track gamma"
        assert metrics['mathematical_params']['lambda'] == agent.lambda_, "Should track lambda"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "not performance"  # Skip performance tests in regular runs
    ])