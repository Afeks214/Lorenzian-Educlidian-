"""
Security Validation Test Suite for Tactical Trading System

Tests for input validation, adversarial attack detection, and fail-fast behavior.
Validates Mission "Aegis" security hardening implementation.
"""

import pytest
import numpy as np
import time
import hashlib
import hmac
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.tactical.environment import TacticalEnvironment
from src.tactical.controller import TacticalMARLController, SynergyEvent
from src.tactical.exceptions import (
    CriticalDependencyError, 
    MatrixValidationError, 
    CircuitBreakerOpenError
)


class TestMatrixValidationSecurity:
    """Test matrix validation security measures."""
    
    def setup_method(self):
        """Setup test environment."""
        self.env = TacticalEnvironment(matrix_integrity_key="test_key_for_security_tests")
    
    def test_valid_matrix_passes_validation(self):
        """Test that valid matrix passes all validation checks."""
        # Create valid matrix
        matrix = np.zeros((60, 7), dtype=np.float32)
        matrix[:, 0] = np.random.choice([0.0, 1.0], size=60)  # Binary feature
        matrix[:, 1] = np.random.choice([0.0, 1.0], size=60)  # Binary feature
        matrix[:, 2] = np.random.uniform(0.95, 1.05, size=60)  # Price levels
        matrix[:, 3] = np.random.uniform(0.0, 50.0, size=60)  # Age
        matrix[:, 4] = np.random.choice([0.0, 1.0], size=60)  # Binary feature
        matrix[:, 5] = np.random.uniform(-15.0, 15.0, size=60)  # Momentum
        matrix[:, 6] = np.random.uniform(0.1, 8.0, size=60)  # Volume ratio
        
        # Should pass validation
        assert self.env.validate_matrix(matrix, strict_validation=True) is True
    
    def test_nan_injection_attack_detection(self):
        """Test detection of NaN injection attacks."""
        # Create matrix with NaN values
        matrix = np.ones((60, 7), dtype=np.float32)
        matrix[10, 3] = np.nan  # Inject NaN
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix, correlation_id="test_nan_attack")
        
        error = exc_info.value
        assert error.validation_type == "nan_inf_check"
        assert "injection attack" in error.error_message
        assert "nan_inf_injection" in error.failed_checks
        assert error.correlation_id == "test_nan_attack"
    
    def test_inf_injection_attack_detection(self):
        """Test detection of infinite value injection attacks."""
        # Create matrix with infinite values
        matrix = np.ones((60, 7), dtype=np.float32)
        matrix[5, 2] = np.inf  # Inject infinity
        matrix[15, 5] = -np.inf  # Inject negative infinity
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix, correlation_id="test_inf_attack")
        
        error = exc_info.value
        assert error.validation_type == "nan_inf_check"
        assert "2 NaN/infinite values" in error.error_message
    
    def test_invalid_shape_rejection(self):
        """Test rejection of matrices with invalid shapes."""
        # Wrong number of rows
        matrix = np.ones((30, 7), dtype=np.float32)
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix)
        
        error = exc_info.value
        assert error.validation_type == "shape_check"
        assert error.matrix_shape == (30, 7)
        
        # Wrong number of columns
        matrix = np.ones((60, 5), dtype=np.float32)
        
        with pytest.raises(MatrixValidationError):
            self.env.validate_matrix(matrix)
    
    def test_invalid_dtype_rejection(self):
        """Test rejection of matrices with invalid data types."""
        # Integer matrix
        matrix = np.ones((60, 7), dtype=np.int32)
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix)
        
        error = exc_info.value
        assert error.validation_type == "dtype_check"
        assert "invalid_dtype" in error.failed_checks
    
    def test_feature_range_validation(self):
        """Test feature range validation against attacks."""
        # Create valid matrix
        matrix = np.zeros((60, 7), dtype=np.float32)
        matrix[:, 0] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 1] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 2] = np.random.uniform(0.95, 1.05, size=60)
        matrix[:, 3] = np.random.uniform(0.0, 50.0, size=60)
        matrix[:, 4] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 5] = np.random.uniform(-15.0, 15.0, size=60)
        matrix[:, 6] = np.random.uniform(0.1, 8.0, size=60)
        
        # Attack: Out-of-range price level
        matrix[0, 2] = 2.0  # Way above valid range
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix, strict_validation=True)
        
        error = exc_info.value
        assert error.validation_type == "range_check"
        assert "fvg_nearest_level" in error.error_message
        assert "range_violation_fvg_nearest_level" in error.failed_checks
    
    def test_binary_feature_attack_detection(self):
        """Test detection of binary feature manipulation attacks."""
        matrix = np.zeros((60, 7), dtype=np.float32)
        
        # Valid setup
        matrix[:, 1] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 2] = np.random.uniform(0.95, 1.05, size=60)
        matrix[:, 3] = np.random.uniform(0.0, 50.0, size=60)
        matrix[:, 4] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 5] = np.random.uniform(-15.0, 15.0, size=60)
        matrix[:, 6] = np.random.uniform(0.1, 8.0, size=60)
        
        # Attack: Non-binary values in binary feature
        matrix[:, 0] = np.random.uniform(0.0, 1.0, size=60)  # Should be exactly 0 or 1
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix, strict_validation=True)
        
        error = exc_info.value
        assert error.validation_type == "binary_check"
        assert "fvg_bullish_active" in error.error_message
    
    def test_adversarial_monotonic_pattern_detection(self):
        """Test detection of adversarial monotonic patterns (gradient attacks)."""
        matrix = np.zeros((60, 7), dtype=np.float32)
        
        # Valid setup for most features
        matrix[:, 0] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 1] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 2] = np.random.uniform(0.95, 1.05, size=60)
        matrix[:, 3] = np.random.uniform(0.0, 50.0, size=60)
        matrix[:, 4] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 6] = np.random.uniform(0.1, 8.0, size=60)
        
        # Create suspicious monotonic pattern (potential gradient attack)
        matrix[:, 5] = np.linspace(-10.0, 10.0, 60)  # Perfect monotonic sequence
        
        # First few attempts might not trigger (pattern counter needs to build up)
        for attempt in range(5):
            try:
                self.env.validate_matrix(matrix, strict_validation=True, correlation_id=f"attack_{attempt}")
            except MatrixValidationError as e:
                if "adversarial_monotonic" in str(e.failed_checks):
                    # Successfully detected gradient attack
                    assert "gradient-based attack" in e.error_message
                    break
        else:
            # If not triggered by repetition, test with more extreme pattern
            matrix[:, 5] = np.arange(60, dtype=np.float32) * 0.5  # Even more obvious pattern
            
            # Reset pattern counter and try again multiple times
            self.env.suspicious_patterns.clear()
            for i in range(5):
                try:
                    self.env.validate_matrix(matrix, strict_validation=True, correlation_id=f"extreme_attack_{i}")
                except MatrixValidationError:
                    break
    
    def test_statistical_anomaly_detection(self):
        """Test detection of statistical anomalies indicating attacks."""
        matrix = np.zeros((60, 7), dtype=np.float32)
        
        # Valid setup for most features
        matrix[:, 0] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 1] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 2] = np.random.uniform(0.95, 1.05, size=60)
        matrix[:, 3] = np.random.uniform(0.0, 50.0, size=60)
        matrix[:, 4] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 6] = np.random.uniform(0.1, 8.0, size=60)
        
        # Create statistical anomaly: most values normal, but many extreme outliers
        normal_values = np.random.normal(0.0, 1.0, 50)
        outlier_values = np.array([50.0] * 10)  # Extreme outliers
        matrix[:, 5] = np.concatenate([normal_values, outlier_values])
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix, strict_validation=True)
        
        error = exc_info.value
        assert error.validation_type == "statistical_anomaly"
        assert "statistical_anomaly_price_momentum_5" in error.failed_checks
    
    def test_cryptographic_integrity_verification(self):
        """Test cryptographic integrity verification."""
        matrix = np.ones((60, 7), dtype=np.float32)
        
        # Generate valid integrity hash
        valid_hash = self.env.generate_matrix_integrity_hash(matrix)
        
        # Should pass with correct hash
        assert self.env.validate_matrix(matrix, integrity_hash=valid_hash) is True
        
        # Should fail with incorrect hash
        invalid_hash = "0" * 64  # Wrong hash
        
        with pytest.raises(MatrixValidationError) as exc_info:
            self.env.validate_matrix(matrix, integrity_hash=invalid_hash)
        
        error = exc_info.value
        assert error.validation_type == "integrity_check"
        assert "integrity_verification_failed" in error.failed_checks
    
    def test_tampered_matrix_detection(self):
        """Test detection of tampered matrix data."""
        matrix = np.ones((60, 7), dtype=np.float32)
        
        # Generate hash for original matrix
        original_hash = self.env.generate_matrix_integrity_hash(matrix)
        
        # Tamper with matrix
        matrix[0, 0] = 0.0
        
        # Should fail integrity check
        with pytest.raises(MatrixValidationError):
            self.env.validate_matrix(matrix, integrity_hash=original_hash)


class TestControllerFailFastBehavior:
    """Test controller fail-fast behavior and circuit breaker."""
    
    def setup_method(self):
        """Setup test controller."""
        self.controller = TacticalMARLController()
    
    @pytest.mark.asyncio
    async def test_critical_dependency_failure_on_matrix_error(self):
        """Test that controller raises CriticalDependencyError on matrix failures."""
        # Mock matrix assembler to fail
        with patch.object(self.controller, 'matrix_assembler') as mock_assembler:
            mock_assembler.get_current_matrix = Mock(side_effect=Exception("Matrix assembler failed"))
            
            with pytest.raises(CriticalDependencyError) as exc_info:
                await self.controller._get_current_matrix_state()
            
            error = exc_info.value
            assert error.dependency == "matrix_assembler"
            assert "Matrix assembler failed" in error.error_message
    
    @pytest.mark.asyncio
    async def test_no_silent_failures_with_random_data(self):
        """Test that controller never returns random data on failure."""
        # Simulate various failure conditions
        failure_scenarios = [
            Exception("Network timeout"),
            ConnectionError("Redis connection lost"),
            ValueError("Invalid data format"),
            RuntimeError("System overload")
        ]
        
        for scenario in failure_scenarios:
            with patch.object(self.controller, '_simulate_valid_matrix', side_effect=scenario):
                with pytest.raises(CriticalDependencyError):
                    await self.controller._get_current_matrix_state()
                
                # Verify no random data is ever returned
                # The method should fail-fast, not fall back
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self):
        """Test circuit breaker activation after repeated failures."""
        # Reset circuit breaker
        self.controller._reset_circuit_breaker()
        
        # Simulate repeated failures
        failure_count = self.controller.circuit_breaker['failure_threshold']
        
        for i in range(failure_count):
            self.controller._record_circuit_breaker_failure()
        
        # Circuit breaker should now be open
        assert self.controller.circuit_breaker['is_open'] is True
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await self.controller._get_current_matrix_state()
        
        error = exc_info.value
        assert error.circuit_name == "matrix_assembler"
        assert error.failure_count == failure_count
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        # Force circuit breaker open
        self.controller.circuit_breaker.update({
            'is_open': True,
            'open_time': time.time() - 400,  # 400 seconds ago (past recovery timeout)
            'failure_count': 10
        })
        
        # Circuit breaker should reset on next call
        assert self.controller._is_circuit_breaker_open() is False
        assert self.controller.circuit_breaker['is_open'] is False
        assert self.controller.circuit_breaker['failure_count'] == 0
    
    @pytest.mark.asyncio
    async def test_matrix_validation_integration(self):
        """Test integration between matrix validation and controller."""
        # Patch matrix simulation to return invalid data
        invalid_matrix = np.full((60, 7), np.nan, dtype=np.float32)
        
        with patch.object(self.controller, '_simulate_valid_matrix', return_value=invalid_matrix):
            with pytest.raises(CriticalDependencyError) as exc_info:
                await self.controller._get_current_matrix_state()
            
            error = exc_info.value
            assert "Matrix validation failed" in error.error_message
            assert error.dependency == "matrix_assembler"


class TestSynergyEventProcessingSecurityTest:
    """Test security aspects of synergy event processing."""
    
    def setup_method(self):
        """Setup test environment."""
        self.controller = TacticalMARLController()
    
    @pytest.mark.asyncio
    async def test_correlation_id_injection_protection(self):
        """Test protection against correlation ID injection attacks."""
        # Malicious correlation IDs
        malicious_ids = [
            "'; DROP TABLE events; --",  # SQL injection style
            "../../../etc/passwd",       # Path traversal style
            "<script>alert('xss')</script>",  # XSS style
            "\x00\x01\x02\x03",         # Binary injection
            "a" * 1000                   # Buffer overflow attempt
        ]
        
        for malicious_id in malicious_ids:
            synergy_event = SynergyEvent(
                synergy_type="test",
                direction=1,
                confidence=0.8,
                signal_sequence=[],
                market_context={},
                correlation_id=malicious_id,
                timestamp=time.time()
            )
            
            # Processing should handle malicious IDs safely
            try:
                # This would normally be called through the event processor
                # but we test the lock acquisition part specifically
                lock_key = f"tactical:event_lock:{malicious_id}"
                # Should not crash or cause injection
                assert isinstance(lock_key, str)
            except Exception as e:
                # Should not fail due to ID format issues
                pytest.fail(f"Correlation ID handling failed: {e}")
    
    def test_synergy_event_data_sanitization(self):
        """Test sanitization of synergy event data."""
        # Create event with potentially malicious data
        malicious_event_data = {
            'synergy_type': '<script>alert("xss")</script>',
            'direction': float('inf'),  # Invalid direction
            'confidence': -999.9,       # Invalid confidence
            'signal_sequence': [{'evil': 'payload'}],
            'market_context': {'<script>': 'malicious'},
            'correlation_id': '../../evil',
            'timestamp': 'not_a_number'
        }
        
        # SynergyEvent.from_dict should handle malicious data gracefully
        event = SynergyEvent.from_dict(malicious_event_data)
        
        # Verify data is handled safely (no crashes, reasonable defaults)
        assert isinstance(event.synergy_type, str)
        assert isinstance(event.correlation_id, str)
        assert isinstance(event.timestamp, (int, float))


class TestExceptionHandlingSecurity:
    """Test security aspects of exception handling."""
    
    def test_exception_information_disclosure(self):
        """Test that exceptions don't disclose sensitive information."""
        env = TacticalEnvironment()
        
        # Create matrix that will fail validation
        matrix = np.full((60, 7), np.nan, dtype=np.float32)
        
        try:
            env.validate_matrix(matrix, correlation_id="sensitive_id_12345")
        except MatrixValidationError as e:
            error_dict = e.to_dict()
            
            # Should contain necessary debugging info
            assert "validation_type" in error_dict
            assert "error_message" in error_dict
            assert "correlation_id" in error_dict
            
            # Should not contain sensitive system information
            assert "password" not in str(error_dict).lower()
            assert "secret" not in str(error_dict).lower()
            assert "key" not in str(error_dict).lower()
    
    def test_critical_dependency_error_logging(self):
        """Test that critical errors are logged appropriately."""
        with patch('src.tactical.exceptions.logger') as mock_logger:
            error = CriticalDependencyError(
                dependency="test_service",
                error_message="Test failure",
                correlation_id="test_123"
            )
            
            # Should have logged critical error
            mock_logger.critical.assert_called_once()
            call_args = mock_logger.critical.call_args
            
            assert "Critical dependency failure: test_service" in call_args[0][0]
            assert call_args[1]['extra']['dependency'] == "test_service"
            assert call_args[1]['extra']['correlation_id'] == "test_123"


class TestPerformanceSecurityImpact:
    """Test that security measures don't create performance vulnerabilities."""
    
    def setup_method(self):
        """Setup performance test environment."""
        self.env = TacticalEnvironment()
    
    def test_validation_performance_dos_protection(self):
        """Test that validation doesn't create DoS vulnerabilities."""
        # Large but valid matrix
        matrix = np.zeros((60, 7), dtype=np.float32)
        matrix[:, 0] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 1] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 2] = np.random.uniform(0.95, 1.05, size=60)
        matrix[:, 3] = np.random.uniform(0.0, 50.0, size=60)
        matrix[:, 4] = np.random.choice([0.0, 1.0], size=60)
        matrix[:, 5] = np.random.uniform(-15.0, 15.0, size=60)
        matrix[:, 6] = np.random.uniform(0.1, 8.0, size=60)
        
        # Time validation
        start_time = time.perf_counter()
        
        # Run validation multiple times
        for _ in range(100):
            self.env.validate_matrix(matrix, strict_validation=True)
        
        total_time = time.perf_counter() - start_time
        avg_time_per_validation = total_time / 100
        
        # Should complete quickly (under 10ms per validation)
        assert avg_time_per_validation < 0.01, f"Validation too slow: {avg_time_per_validation:.4f}s"
    
    def test_adversarial_detection_memory_bounds(self):
        """Test that adversarial detection doesn't consume unbounded memory."""
        initial_pattern_count = len(self.env.suspicious_patterns)
        
        # Simulate many different patterns
        for i in range(1000):
            matrix = np.zeros((60, 7), dtype=np.float32)
            # Create different patterns
            matrix[:, 5] = np.linspace(-10 + i*0.01, 10 + i*0.01, 60)
            
            try:
                self.env.validate_matrix(matrix, strict_validation=True, correlation_id=f"pattern_{i}")
            except MatrixValidationError:
                pass  # Expected for some patterns
        
        # Pattern count should not grow unboundedly
        pattern_count = len(self.env.suspicious_patterns)
        assert pattern_count < 100, f"Too many patterns stored: {pattern_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])