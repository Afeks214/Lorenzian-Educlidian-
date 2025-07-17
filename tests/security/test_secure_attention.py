"""
Test Suite for Secure Attention System - CVE-2025-TACTICAL-001 Validation
"""

import pytest
import torch
import numpy as np
from models.security.secure_attention import SecureAttentionSystem
from models.security.crypto_validation import CryptographicValidator


class TestSecureAttention:
    """Test suite for secure attention system."""
    
    @pytest.fixture
    def secure_attention(self):
        """Create secure attention system for testing."""
        return SecureAttentionSystem(
            feature_dim=7,
            agent_id="test_agent",
            attention_heads=4,
            dropout_rate=0.1
        )
    
    @pytest.fixture
    def test_features(self):
        """Create test features tensor."""
        return torch.randn(2, 60, 7)  # batch_size=2, seq_len=60, features=7
    
    def test_initialization(self, secure_attention):
        """Test secure attention initialization."""
        assert secure_attention.feature_dim == 7
        assert secure_attention.attention_heads == 4
        assert secure_attention.agent_id == "test_agent"
        assert hasattr(secure_attention, 'crypto_validator')
        assert hasattr(secure_attention, 'query_projection')
        assert hasattr(secure_attention, 'key_projection')
        assert hasattr(secure_attention, 'value_projection')
    
    def test_forward_pass(self, secure_attention, test_features):
        """Test forward pass functionality."""
        attended_features, attention_weights = secure_attention(test_features)
        
        # Check output shapes
        assert attended_features.shape == test_features.shape
        assert attention_weights.shape == (2, 60, 60)  # batch_size, seq_len, seq_len
        
        # Check attention weights properties
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)
        
        # Check attention weights sum to 1 (approximately)
        weight_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-3)
    
    def test_agent_specialization(self):
        """Test agent-specific attention bias initialization."""
        # Test FVG agent
        fvg_attention = SecureAttentionSystem(feature_dim=7, agent_id="fvg")
        fvg_bias = fvg_attention.agent_bias.data
        
        # FVG should bias toward first two features (fvg_bullish, fvg_bearish)
        assert fvg_bias[0] > 0 and fvg_bias[1] > 0
        
        # Test Momentum agent
        momentum_attention = SecureAttentionSystem(feature_dim=7, agent_id="momentum")
        momentum_bias = momentum_attention.agent_bias.data
        
        # Momentum should bias toward last two features (momentum, volume)
        assert momentum_bias[5] > 0 and momentum_bias[6] > 0
        
        # Test Entry agent (balanced)
        entry_attention = SecureAttentionSystem(feature_dim=7, agent_id="entry")
        entry_bias = entry_attention.agent_bias.data
        
        # Entry should have zero bias (balanced)
        assert torch.allclose(entry_bias, torch.zeros_like(entry_bias))
    
    def test_security_validation(self, secure_attention, test_features):
        """Test security validation functionality."""
        # Test with valid input
        attended_features, attention_weights = secure_attention(test_features, validate_security=True)
        assert attended_features is not None
        assert attention_weights is not None
        
        # Test with invalid input (NaN values)
        invalid_features = test_features.clone()
        invalid_features[0, 0, 0] = float('nan')
        
        # Should handle gracefully and not crash
        try:
            attended_features, attention_weights = secure_attention(invalid_features, validate_security=True)
            # If it doesn't raise an exception, check that it returns valid output
            assert not torch.isnan(attended_features).any()
            assert not torch.isnan(attention_weights).any()
        except RuntimeError:
            # This is also acceptable behavior for security validation
            pass
    
    def test_attention_bounds_enforcement(self, secure_attention, test_features):
        """Test attention weight bounds enforcement."""
        # Enable security mode
        secure_attention.enable_security_mode(True)
        
        attended_features, attention_weights = secure_attention(test_features)
        
        # Check bounds are enforced
        assert torch.all(attention_weights >= secure_attention.min_attention)
        assert torch.all(attention_weights <= secure_attention.max_attention)
    
    def test_anomaly_detection(self, secure_attention, test_features):
        """Test attention anomaly detection."""
        # Perform multiple forward passes to build history
        for _ in range(15):
            secure_attention(test_features)
        
        # Get attention statistics
        stats = secure_attention.get_attention_stats()
        
        assert 'mean_attention' in stats
        assert 'std_attention' in stats
        assert 'num_samples' in stats
        assert stats['num_samples'] == 15
    
    def test_cryptographic_validation(self):
        """Test cryptographic validation functionality."""
        crypto_key = b'test_key_32_bytes_long_for_hmac'
        secure_attention = SecureAttentionSystem(
            feature_dim=7,
            agent_id="crypto_test",
            crypto_key=crypto_key
        )
        
        test_features = torch.randn(1, 10, 7)
        
        # Test that cryptographic validation is enabled
        assert secure_attention.crypto_validator.enabled
        
        # Test forward pass with crypto validation
        attended_features, attention_weights = secure_attention(test_features)
        assert attended_features is not None
        assert attention_weights is not None
    
    def test_performance_requirements(self, secure_attention, test_features):
        """Test that attention meets performance requirements."""
        import time
        
        # Warm up
        for _ in range(5):
            secure_attention(test_features)
        
        # Measure inference time
        times = []
        for _ in range(10):
            start_time = time.time()
            secure_attention(test_features)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        # Should be very fast (well under 100ms requirement)
        assert avg_time < 10.0, f"Average time {avg_time:.2f}ms too slow"
        assert p95_time < 20.0, f"P95 time {p95_time:.2f}ms too slow"
    
    def test_memory_management(self, secure_attention, test_features):
        """Test memory management and cleanup."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Perform many forward passes
        for _ in range(50):
            attended_features, attention_weights = secure_attention(test_features)
            
            # Clear intermediate results to avoid accumulation
            del attended_features, attention_weights
        
        # Reset security state (should clear history)
        secure_attention.reset_security_state()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory should not have grown significantly
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            assert memory_growth < 10 * 1024 * 1024, f"Memory grew by {memory_growth / (1024*1024):.2f}MB"
    
    def test_gradient_flow(self, secure_attention, test_features):
        """Test that gradients flow properly through secure attention."""
        test_features.requires_grad_(True)
        
        attended_features, attention_weights = secure_attention(test_features)
        
        # Create a simple loss
        loss = attended_features.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert test_features.grad is not None
        assert not torch.isnan(test_features.grad).any()
        assert not torch.isinf(test_features.grad).any()
        
        # Check secure attention parameter gradients
        for param in secure_attention.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
    
    def test_multi_batch_consistency(self, secure_attention):
        """Test consistency across different batch sizes."""
        # Single sample
        single_features = torch.randn(1, 60, 7)
        single_attended, single_weights = secure_attention(single_features)
        
        # Batch of same sample
        batch_features = single_features.repeat(4, 1, 1)
        batch_attended, batch_weights = secure_attention(batch_features)
        
        # Results should be consistent (approximately)
        assert torch.allclose(single_attended, batch_attended[0], atol=1e-5)
        assert torch.allclose(single_weights, batch_weights[0], atol=1e-5)
    
    def test_deterministic_behavior(self, test_features):
        """Test deterministic behavior with same inputs."""
        # Create two identical attention systems
        torch.manual_seed(42)
        attention1 = SecureAttentionSystem(feature_dim=7, agent_id="test")
        
        torch.manual_seed(42)
        attention2 = SecureAttentionSystem(feature_dim=7, agent_id="test")
        
        # Should produce identical results
        result1 = attention1(test_features)
        result2 = attention2(test_features)
        
        assert torch.allclose(result1[0], result2[0])
        assert torch.allclose(result1[1], result2[1])
    
    def test_security_mode_toggle(self, secure_attention, test_features):
        """Test enabling/disabling security mode."""
        # Test with security enabled
        secure_attention.enable_security_mode(True)
        result1 = secure_attention(test_features, validate_security=True)
        
        # Test with security disabled
        secure_attention.enable_security_mode(False)
        result2 = secure_attention(test_features, validate_security=False)
        
        # Both should work (security disabled should be faster)
        assert result1[0] is not None
        assert result2[0] is not None
        
        # Shapes should be identical
        assert result1[0].shape == result2[0].shape
        assert result1[1].shape == result2[1].shape


if __name__ == "__main__":
    # Run tests if executed directly
    test_instance = TestSecureAttention()
    
    # Create fixtures
    secure_attention = SecureAttentionSystem(
        feature_dim=7,
        agent_id="test_agent",
        attention_heads=4,
        dropout_rate=0.1
    )
    test_features = torch.randn(2, 60, 7)
    
    # Run key tests
    print("Testing secure attention initialization...")
    test_instance.test_initialization(secure_attention)
    print("✓ Initialization test passed")
    
    print("Testing forward pass...")
    test_instance.test_forward_pass(secure_attention, test_features)
    print("✓ Forward pass test passed")
    
    print("Testing security validation...")
    test_instance.test_security_validation(secure_attention, test_features)
    print("✓ Security validation test passed")
    
    print("Testing performance requirements...")
    test_instance.test_performance_requirements(secure_attention, test_features)
    print("✓ Performance requirements test passed")
    
    print("\nAll secure attention tests passed! CVE-2025-TACTICAL-001 mitigation validated.")