"""
Comprehensive validation tests for the Intelligent Gating Network.

These tests validate that the gating network successfully replaces static ensemble weights
with dynamic, context-aware agent coordination. Tests ensure proper functionality,
performance requirements, and context sensitivity.

Test Coverage:
- Gating network initialization and configuration
- Context sensitivity and weight adaptation
- Performance requirements (< 0.5ms additional latency)
- Probability constraints and mathematical validity
- Training and learning capabilities
- Integration with StrategicMARLComponent
- Error handling and fallback behavior
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import the gating network and related components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from agents.gating_network import GatingNetwork, GatingNetworkTrainer
from agents.strategic_marl_component import StrategicMARLComponent
from agents.strategic_agent_base import AgentPrediction
from datetime import datetime


class TestGatingNetworkArchitecture:
    """Test the core gating network architecture and functionality."""
    
    @pytest.fixture
    def gating_network(self):
        """Create a test gating network instance."""
        return GatingNetwork(
            shared_context_dim=6,
            n_agents=3,
            hidden_dim=64
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample market context tensor."""
        return torch.tensor([
            0.015,  # volatility_30
            1.2,    # volume_ratio
            0.005,  # momentum_20
            0.003,  # momentum_50
            0.1,    # mmd_score
            0.002   # price_trend
        ], dtype=torch.float32)
    
    def test_gating_network_initialization(self, gating_network):
        """Test proper initialization of gating network."""
        assert gating_network.shared_context_dim == 6
        assert gating_network.n_agents == 3
        assert gating_network.hidden_dim == 64
        
        # Check network layers exist
        assert hasattr(gating_network, 'gating_net')
        assert hasattr(gating_network, 'context_analyzer')
        
        # Check performance tracking initialized
        assert torch.all(gating_network.agent_performance_history == 0)
        assert len(gating_network.gating_decision_history) == 0
    
    def test_forward_pass_validity(self, gating_network, sample_context):
        """Test that forward pass produces valid outputs."""
        with torch.no_grad():
            result = gating_network(sample_context)
        
        # Check required outputs exist
        assert 'gating_weights' in result
        assert 'context_features' in result
        assert 'confidence' in result
        assert 'weight_entropy' in result
        
        # Validate gating weights properties
        weights = result['gating_weights']
        assert weights.shape == (1, 3)  # batch_size=1, n_agents=3
        
        # CRITICAL: Weights must sum to 1 (probability constraint)
        assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0), atol=1e-6)
        
        # CRITICAL: All weights must be non-negative
        assert torch.all(weights >= 0)
        
        # Confidence should be in [0, 1]
        confidence = result['confidence']
        assert 0.0 <= confidence.item() <= 1.0
    
    def test_context_sensitivity(self, gating_network):
        """CRITICAL: Test that gating weights differ across market contexts."""
        
        # High volatility context (crisis scenario)
        high_vol_context = torch.tensor([3.0, 0.5, -0.2, -0.1, 0.8, -0.3])
        
        # Low volatility context (ranging market)
        low_vol_context = torch.tensor([0.3, 1.2, 0.05, 0.02, 0.1, 0.0])
        
        # Trending market context
        trending_context = torch.tensor([1.0, 2.0, 0.15, 0.12, 0.3, 0.1])
        
        with torch.no_grad():
            high_vol_result = gating_network(high_vol_context)
            low_vol_result = gating_network(low_vol_context)
            trending_result = gating_network(trending_context)
        
        high_vol_weights = high_vol_result['gating_weights']
        low_vol_weights = low_vol_result['gating_weights']
        trending_weights = trending_result['gating_weights']
        
        # CRITICAL: Weights must be significantly different across contexts
        high_low_diff = torch.abs(high_vol_weights - low_vol_weights).mean()
        high_trend_diff = torch.abs(high_vol_weights - trending_weights).mean()
        low_trend_diff = torch.abs(low_vol_weights - trending_weights).mean()
        
        assert high_low_diff > 0.1, f"High vol vs low vol weights too similar (diff: {high_low_diff})"
        assert high_trend_diff > 0.1, f"High vol vs trending weights too similar (diff: {high_trend_diff})"
        assert low_trend_diff > 0.1, f"Low vol vs trending weights too similar (diff: {low_trend_diff})"
        
        # Validate all probability constraints still hold
        for weights in [high_vol_weights, low_vol_weights, trending_weights]:
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
            assert torch.all(weights >= 0)
    
    def test_gating_performance_requirements(self, gating_network, sample_context):
        """CRITICAL: Ensure gating network doesn't impact performance."""
        
        # Warmup to account for initial overhead
        for _ in range(10):
            with torch.no_grad():
                _ = gating_network(sample_context)
        
        # Measure inference time
        inference_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            with torch.no_grad():
                result = gating_network(sample_context)
            inference_time = (time.perf_counter() - start_time) * 1000  # milliseconds
            inference_times.append(inference_time)
        
        avg_inference_time = np.mean(inference_times)
        max_inference_time = np.max(inference_times)
        p95_inference_time = np.percentile(inference_times, 95)
        
        # CRITICAL: Performance requirements
        assert avg_inference_time < 0.3, f"Average inference time {avg_inference_time}ms too slow"
        assert max_inference_time < 0.5, f"Max inference time {max_inference_time}ms too slow"
        assert p95_inference_time < 0.4, f"P95 inference time {p95_inference_time}ms too slow"
    
    def test_batch_processing(self, gating_network):
        """Test that gating network handles batch processing correctly."""
        batch_size = 8
        context_batch = torch.randn(batch_size, 6)
        
        with torch.no_grad():
            result = gating_network(context_batch)
        
        # Check batch dimensions
        assert result['gating_weights'].shape == (batch_size, 3)
        assert result['confidence'].shape == (batch_size,)
        
        # Validate each sample in batch
        for i in range(batch_size):
            weights = result['gating_weights'][i]
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
            assert torch.all(weights >= 0)
    
    def test_performance_history_tracking(self, gating_network):
        """Test agent performance history tracking."""
        # Update performance for different agents
        gating_network.update_performance_history(0, 0.8)  # MLMI
        gating_network.update_performance_history(1, 0.6)  # NWRQK
        gating_network.update_performance_history(2, 0.9)  # Regime
        
        history = gating_network.agent_performance_history
        assert len(history) == 3
        assert history[0] > 0  # MLMI performance updated
        assert history[1] > 0  # NWRQK performance updated
        assert history[2] > 0  # Regime performance updated
        
        # Test exponential moving average behavior
        initial_value = float(history[0])
        gating_network.update_performance_history(0, 0.9)
        new_value = float(history[0])
        assert new_value > initial_value  # Should increase with better performance
    
    def test_specialization_analysis(self, gating_network, sample_context):
        """Test agent specialization analysis functionality."""
        specialization = gating_network.get_agent_specialization(sample_context)
        
        # Check all agents have specialization descriptions
        assert 'MLMI' in specialization
        assert 'NWRQK' in specialization
        assert 'Regime' in specialization
        
        # Check descriptions contain relevant context info
        for agent, description in specialization.items():
            assert len(description) > 10  # Meaningful description
            assert any(keyword in description.lower() for keyword in ['vol', 'momentum', 'regime'])
    
    def test_gating_rationale_generation(self, gating_network, sample_context):
        """Test human-readable rationale generation."""
        with torch.no_grad():
            result = gating_network(sample_context)
        
        rationale = gating_network.get_gating_rationale(sample_context, result['gating_weights'])
        
        # Check rationale is meaningful
        assert len(rationale) > 50  # Substantial explanation
        assert any(agent in rationale for agent in ['MLMI', 'NWRQK', 'Regime'])
        assert 'weight' in rationale.lower()
        assert 'condition' in rationale.lower()


class TestGatingNetworkTrainer:
    """Test the gating network training capabilities."""
    
    @pytest.fixture
    def gating_network(self):
        return GatingNetwork(shared_context_dim=6, n_agents=3, hidden_dim=32)
    
    @pytest.fixture
    def trainer(self, gating_network):
        return GatingNetworkTrainer(gating_network, learning_rate=1e-3)
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert hasattr(trainer, 'gating_network')
        assert hasattr(trainer, 'optimizer')
        assert len(trainer.training_history) == 0
    
    def test_training_step(self, trainer):
        """Test single training step execution."""
        batch_size = 4
        context_batch = torch.randn(batch_size, 6)
        performance_batch = torch.rand(batch_size, 3)  # Random performance scores
        
        # Generate target weights
        target_weights = trainer.generate_training_targets(performance_batch.numpy())
        
        # Perform training step
        loss = trainer.train_step(context_batch, performance_batch, target_weights)
        
        # Check training completed
        assert isinstance(loss, float)
        assert loss < float('inf')
        assert len(trainer.training_history) == 1
        
        # Check training history contains expected fields
        history_entry = trainer.training_history[0]
        assert 'loss' in history_entry
        assert 'kl_loss' in history_entry
        assert 'l2_reg' in history_entry
        assert 'entropy_reg' in history_entry
    
    def test_target_generation(self, trainer):
        """Test target weight generation from performance scores."""
        performance_scores = np.array([0.8, 0.6, 0.9])  # Agent performance
        
        target_weights = trainer.generate_training_targets(performance_scores)
        
        # Check valid probability distribution
        assert torch.allclose(target_weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(target_weights >= 0)
        
        # Best performing agent should have highest weight
        best_agent_idx = np.argmax(performance_scores)
        assert torch.argmax(target_weights) == best_agent_idx
    
    def test_training_convergence(self, trainer):
        """Test that training can reduce loss over multiple steps."""
        batch_size = 8
        n_steps = 20
        
        # Fixed context and performance for consistent training
        context_batch = torch.randn(batch_size, 6)
        performance_batch = torch.tensor([[0.8, 0.6, 0.9]] * batch_size)
        
        losses = []
        for _ in range(n_steps):
            target_weights = trainer.generate_training_targets(performance_batch.numpy())
            loss = trainer.train_step(context_batch, performance_batch, target_weights)
            losses.append(loss)
        
        # Loss should generally decrease over training
        initial_avg = np.mean(losses[:5])
        final_avg = np.mean(losses[-5:])
        assert final_avg < initial_avg, "Training should reduce loss"


class TestStrategicMARLIntegration:
    """Test integration of gating network with StrategicMARLComponent."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Create mock kernel for testing."""
        kernel = Mock()
        kernel.config = {
            'strategic': {
                'environment': {
                    'matrix_shape': [48, 13],
                    'feature_indices': {
                        'mlmi_expert': [0, 1, 9, 10],
                        'nwrqk_expert': [2, 3, 4, 5],
                        'regime_expert': [10, 11, 12]
                    }
                },
                'ensemble': {
                    'confidence_threshold': 0.65
                },
                'performance': {
                    'max_inference_latency_ms': 5.0
                },
                'safety': {
                    'max_consecutive_failures': 5
                }
            }
        }
        return kernel
    
    @pytest.fixture
    def strategic_component(self, mock_kernel):
        """Create StrategicMARLComponent with mocked dependencies."""
        component = StrategicMARLComponent(mock_kernel)
        
        # Mock the event bus
        component.event_bus = Mock()
        component.event_bus.subscribe = Mock()
        component.event_bus.create_event = Mock()
        component.event_bus.publish = Mock()
        
        return component
    
    def test_gating_network_initialization_in_component(self, strategic_component):
        """Test that gating network is properly initialized in component."""
        assert hasattr(strategic_component, 'gating_network')
        assert hasattr(strategic_component, 'gating_trainer')
        
        # Check gating network configuration
        assert strategic_component.gating_network.shared_context_dim == 6
        assert strategic_component.gating_network.n_agents == 3
        
        # Check no static ensemble weights
        assert not hasattr(strategic_component, 'ensemble_weights')
    
    def test_shared_context_extraction(self, strategic_component):
        """Test shared context extraction for gating network."""
        # Create sample matrix data
        matrix_data = np.random.randn(48, 13)
        
        # Extract shared context
        context = strategic_component._extract_shared_context(matrix_data)
        
        # Check required context fields
        required_fields = ['volatility_30', 'volume_ratio', 'momentum_20', 
                          'momentum_50', 'mmd_score', 'price_trend']
        for field in required_fields:
            assert field in context
            assert isinstance(context[field], float)
        
        # Test tensor extraction
        context_tensor = strategic_component._extract_shared_context_tensor(context)
        assert context_tensor.shape == (6,)
        assert context_tensor.dtype == torch.float32
    
    def test_dynamic_gating_in_agent_combination(self, strategic_component):
        """Test that agent outputs are combined using dynamic gating."""
        # Create mock agent results
        agent_results = [
            {
                'agent_name': 'MLMI',
                'action_probabilities': [0.6, 0.3, 0.1],
                'confidence': 0.8
            },
            {
                'agent_name': 'NWRQK',
                'action_probabilities': [0.2, 0.5, 0.3],
                'confidence': 0.7
            },
            {
                'agent_name': 'Regime',
                'action_probabilities': [0.1, 0.2, 0.7],
                'confidence': 0.9
            }
        ]
        
        # Create shared context
        shared_context = {
            'volatility_30': 0.02,
            'volume_ratio': 1.5,
            'momentum_20': 0.01,
            'momentum_50': 0.005,
            'mmd_score': 0.2,
            'price_trend': 0.003
        }
        
        # Combine outputs using dynamic gating
        decision = strategic_component._combine_agent_outputs(agent_results, shared_context)
        
        # Check decision structure
        assert hasattr(decision, 'action')
        assert hasattr(decision, 'confidence')
        assert hasattr(decision, 'reasoning')
        assert hasattr(decision, 'agent_contributions')
        
        # Check performance metrics include gating info
        assert 'dynamic_weights' in decision.performance_metrics
        assert 'gating_confidence' in decision.performance_metrics
        
        # Validate dynamic weights used
        dynamic_weights = decision.performance_metrics['dynamic_weights']
        assert len(dynamic_weights) == 3
        assert abs(sum(dynamic_weights) - 1.0) < 1e-6  # Sum to 1
        assert all(w >= 0 for w in dynamic_weights)  # Non-negative
    
    def test_gating_training_integration(self, strategic_component):
        """Test gating network training integration."""
        batch_size = 4
        context_batch = torch.randn(batch_size, 6)
        performance_batch = torch.rand(batch_size, 3)
        
        # Test training capability
        loss = strategic_component.train_gating_network(context_batch, performance_batch)
        assert isinstance(loss, float)
        assert loss < float('inf')
        
        # Test performance update
        strategic_component.update_agent_performance(0, 0.85)
        performance_history = strategic_component.gating_network.agent_performance_history
        assert performance_history[0] > 0
    
    def test_status_reporting_with_gating(self, strategic_component):
        """Test that component status includes gating network information."""
        status = strategic_component.get_status()
        
        # Check gating network status included
        assert 'gating_network_status' in status
        assert 'intelligent_gating' in status
        
        # Check no static ensemble weights reported
        assert 'ensemble_weights' not in status
        
        # Check intelligent gating details
        gating_info = status['intelligent_gating']
        assert 'context_dim' in gating_info
        assert 'n_agents' in gating_info
        assert 'agent_performance_ema' in gating_info


class TestGatingNetworkErrorHandling:
    """Test error handling and edge cases for gating network."""
    
    @pytest.fixture
    def gating_network(self):
        return GatingNetwork(shared_context_dim=6, n_agents=3, hidden_dim=32)
    
    def test_invalid_context_handling(self, gating_network):
        """Test handling of invalid or malformed context."""
        # Test with NaN values
        nan_context = torch.tensor([float('nan'), 1.0, 0.5, 0.2, 0.1, 0.0])
        
        with torch.no_grad():
            result = gating_network(nan_context)
        
        # Should still produce valid outputs
        weights = result['gating_weights']
        assert not torch.isnan(weights).any()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_extreme_context_values(self, gating_network):
        """Test handling of extreme context values."""
        # Very large values
        extreme_context = torch.tensor([1000.0, -1000.0, 500.0, -500.0, 100.0, -100.0])
        
        with torch.no_grad():
            result = gating_network(extreme_context)
        
        # Should produce valid probability distribution
        weights = result['gating_weights']
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()
    
    def test_empty_context_fallback(self, gating_network):
        """Test fallback behavior with empty or zero context."""
        zero_context = torch.zeros(6)
        
        with torch.no_grad():
            result = gating_network(zero_context)
        
        # Should produce valid, though potentially uniform, weights
        weights = result['gating_weights']
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_checkpoint_save_load(self, gating_network, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_path = tmp_path / "gating_checkpoint.pth"
        
        # Update some performance history
        gating_network.update_performance_history(0, 0.8)
        gating_network.update_performance_history(1, 0.6)
        
        # Save checkpoint
        gating_network.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # Create new network and load checkpoint
        new_network = GatingNetwork(shared_context_dim=6, n_agents=3, hidden_dim=32)
        success = new_network.load_checkpoint(str(checkpoint_path))
        
        assert success
        # Performance history should be preserved
        assert torch.allclose(
            new_network.agent_performance_history,
            gating_network.agent_performance_history
        )


def test_overall_gating_system_integration():
    """Integration test for the complete gating system."""
    # Create a complete test scenario
    gating_network = GatingNetwork(shared_context_dim=6, n_agents=3, hidden_dim=64)
    trainer = GatingNetworkTrainer(gating_network, learning_rate=1e-4)
    
    # Simulate market conditions over time
    contexts = [
        torch.tensor([0.01, 1.0, 0.005, 0.003, 0.1, 0.002]),  # Normal market
        torch.tensor([0.05, 0.5, -0.02, -0.01, 0.8, -0.01]),  # Crisis
        torch.tensor([0.02, 2.0, 0.015, 0.012, 0.3, 0.008]),  # Bull market
    ]
    
    # Test adaptation across different market regimes
    decisions = []
    for context in contexts:
        with torch.no_grad():
            result = gating_network(context)
            decisions.append(result['gating_weights'])
    
    # Verify that gating adapts to different contexts
    for i in range(len(decisions)):
        for j in range(i + 1, len(decisions)):
            weight_diff = torch.abs(decisions[i] - decisions[j]).mean()
            assert weight_diff > 0.05, f"Insufficient adaptation between contexts {i} and {j}"
    
    # Test that system maintains performance requirements
    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = gating_network(contexts[0])
    avg_time = (time.perf_counter() - start_time) * 1000 / 100
    
    assert avg_time < 0.3, f"Average gating time {avg_time}ms exceeds requirement"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])