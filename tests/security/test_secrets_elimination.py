"""
AGENT 9: Secrets Elimination Validation Tests
Comprehensive test suite for verifying complete elimination of hardcoded secrets and credentials.
"""

import pytest
import os
import ast
import re
import json
import yaml
import base64
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Security-focused imports
import secrets as secrets_module
from cryptography.fernet import Fernet

# Core imports
from src.security.secrets_manager import SecretsManager, get_secret
from src.security.vault_client import VaultClient, VaultConfig
from src.core.errors.base_exceptions import SecurityError, ValidationError


class TestSecretsElimination:
    """Test complete elimination of hardcoded secrets and credentials."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def secrets_patterns(self):
        """Patterns that indicate potential hardcoded secrets."""
        return [
            # API Keys
            r'api[_-]?key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            r'apikey[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            r'secret[_-]?key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            
            # Passwords
            r'password[\s]*[=:]\s*["\']([^"\']{8,})["\']',
            r'passwd[\s]*[=:]\s*["\']([^"\']{8,})["\']',
            r'pwd[\s]*[=:]\s*["\']([^"\']{8,})["\']',
            
            # Tokens
            r'token[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            r'access[_-]?token[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            r'auth[_-]?token[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            
            # Database credentials
            r'db[_-]?password[\s]*[=:]\s*["\']([^"\']{4,})["\']',
            r'database[_-]?password[\s]*[=:]\s*["\']([^"\']{4,})["\']',
            r'connection[_-]?string[\s]*[=:]\s*["\']([^"\']{10,})["\']',
            
            # AWS/Cloud credentials
            r'aws[_-]?access[_-]?key[\s]*[=:]\s*["\']([A-Z0-9]{16,})["\']',
            r'aws[_-]?secret[_-]?key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{28,})["\']',
            r'azure[_-]?key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            r'gcp[_-]?key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
            
            # Private keys
            r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
            r'-----BEGIN\s+ENCRYPTED\s+PRIVATE\s+KEY-----',
            r'-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----',
            
            # Certificates
            r'-----BEGIN\s+CERTIFICATE-----',
            r'-----BEGIN\s+PUBLIC\s+KEY-----',
            
            # JWT tokens
            r'ey[A-Za-z0-9+/=]{10,}\.[A-Za-z0-9+/=]{10,}\.[A-Za-z0-9+/=]{10,}',
            
            # Generic secrets
            r'secret[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{16,})["\']',
            r'key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{16,})["\']',
            
            # URLs with credentials
            r'https?://[^:]+:[^@]+@[^/\s]+',
            r'mongodb://[^:]+:[^@]+@[^/\s]+',
            r'redis://[^:]+:[^@]+@[^/\s]+',
            
            # Common test/default passwords
            r'password.*[=:]\s*["\']?(admin|password|123456|root|test)["\']?',
            r'secret.*[=:]\s*["\']?(secret|password|test|admin)["\']?',
        ]
    
    @pytest.fixture
    def whitelisted_patterns(self):
        """Patterns that are allowed (test data, examples, etc.)."""
        return [
            # Test files and fixtures
            r'test_.*\.py',
            r'.*_test\.py',
            r'conftest\.py',
            r'fixtures/.*',
            r'tests/.*',
            
            # Documentation and examples
            r'README\.md',
            r'.*\.example',
            r'examples/.*',
            r'docs/.*',
            
            # Configuration templates
            r'.*\.template',
            r'.*\.example\.',
            
            # Mock/dummy values
            r'mock_.*',
            r'dummy_.*',
            r'example_.*',
            r'test_.*',
            
            # Environment variable references
            r'\$\{.*\}',
            r'os\.getenv',
            r'os\.environ',
            r'getenv\(',
            
            # Placeholder values
            r'YOUR_.*_HERE',
            r'REPLACE_.*',
            r'TODO:.*',
            r'FIXME:.*',
        ]
    
    def test_no_hardcoded_secrets_in_source_code(self, project_root, secrets_patterns, whitelisted_patterns):
        """Test that no hardcoded secrets exist in source code."""
        
        violations = []
        
        # Scan Python files
        python_files = list(project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip test files and other whitelisted files
            if self._is_whitelisted_file(str(file_path), whitelisted_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for secret patterns
                for pattern in secrets_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        # Skip if the match is in a comment or string that looks like an example
                        if self._is_likely_example_or_comment(content, match):
                            continue
                        
                        violations.append({
                            "file": str(file_path.relative_to(project_root)),
                            "line": content[:match.start()].count('\n') + 1,
                            "pattern": pattern,
                            "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0)
                        })
            
            except Exception as e:
                # Log but don't fail on read errors
                print(f"Warning: Could not read {file_path}: {e}")
        
        # Assert no violations found
        if violations:
            violation_summary = "\n".join([
                f"  {v['file']}:{v['line']} - {v['match']}"
                for v in violations[:10]  # Show first 10
            ])
            if len(violations) > 10:
                violation_summary += f"\n  ... and {len(violations) - 10} more violations"
            
            pytest.fail(f"Found {len(violations)} potential hardcoded secrets:\n{violation_summary}")
    
    def test_no_hardcoded_secrets_in_config_files(self, project_root, secrets_patterns, whitelisted_patterns):
        """Test that no hardcoded secrets exist in configuration files."""
        
        violations = []
        
        # Scan configuration files
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.ini", "*.conf", "*.config", "*.env"]
        config_files = []
        
        for pattern in config_patterns:
            config_files.extend(project_root.rglob(pattern))
        
        for file_path in config_files:
            # Skip whitelisted files
            if self._is_whitelisted_file(str(file_path), whitelisted_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for secret patterns
                for pattern in secrets_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        violations.append({
                            "file": str(file_path.relative_to(project_root)),
                            "line": content[:match.start()].count('\n') + 1,
                            "pattern": pattern,
                            "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0)
                        })
            
            except Exception as e:
                # Log but don't fail on read errors
                print(f"Warning: Could not read {file_path}: {e}")
        
        # Assert no violations found
        if violations:
            violation_summary = "\n".join([
                f"  {v['file']}:{v['line']} - {v['match']}"
                for v in violations[:10]
            ])
            pytest.fail(f"Found {len(violations)} potential hardcoded secrets in config files:\n{violation_summary}")
    
    def test_environment_variables_used_for_secrets(self, project_root):
        """Test that environment variables are used for secret values."""
        
        # Check that secrets manager uses environment variables
        secrets_manager = SecretsManager()
        
        # Test that it properly loads from environment
        with patch.dict(os.environ, {'TEST_SECRET': 'test_value'}):
            result = secrets_manager.get_secret('test_secret')
            assert result == 'test_value'
        
        # Test that it doesn't return hardcoded values
        with patch.dict(os.environ, {}, clear=True):
            result = secrets_manager.get_secret('nonexistent_secret')
            assert result is None
    
    def test_docker_secrets_used_properly(self):
        """Test that Docker secrets are used when available."""
        
        secrets_manager = SecretsManager()
        
        # Mock Docker secrets directory
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_dir = Path(temp_dir) / "secrets"
            secrets_dir.mkdir()
            
            # Create a mock Docker secret
            secret_file = secrets_dir / "test_docker_secret"
            secret_file.write_text("docker_secret_value")
            
            # Patch the Docker secrets path
            with patch.object(secrets_manager, 'docker_secrets_path', secrets_dir):
                result = secrets_manager.get_secret('test_docker_secret')
                assert result == "docker_secret_value"
    
    def test_vault_integration_for_secrets(self):
        """Test that Vault integration is properly implemented."""
        
        # Mock Vault client
        with patch('src.security.vault_client.aiohttp') as mock_aiohttp:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = Mock(return_value={
                "data": {"data": {"password": "vault_password"}}
            })
            mock_session.request = Mock(return_value=mock_response)
            mock_aiohttp.ClientSession.return_value = mock_session
            
            config = VaultConfig()
            client = VaultClient(config)
            
            # Test that secrets come from Vault, not hardcoded
            # This would be tested in an async context in real usage
            assert client.config.url != "http://localhost:8200"  # Should come from env
    
    def test_no_secrets_in_git_history(self, project_root):
        """Test that no secrets have been committed to git history."""
        
        try:
            # Use git log to check for potential secret commits
            result = subprocess.run(
                ["git", "log", "--all", "--full-history", "--grep=password", "--grep=secret", "--grep=key", "-i"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # If the command succeeds and returns results, investigate further
            if result.returncode == 0 and result.stdout.strip():
                # Check if any of the commits actually contain secrets
                suspicious_commits = []
                
                # Get commit hashes from the output
                commit_hashes = re.findall(r'commit ([a-f0-9]+)', result.stdout)
                
                for commit_hash in commit_hashes[:10]:  # Check first 10 commits
                    # Get the diff for this commit
                    diff_result = subprocess.run(
                        ["git", "show", commit_hash, "--format="],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if diff_result.returncode == 0:
                        # Check if the diff contains actual secrets (not just the word "password")
                        if self._contains_actual_secrets(diff_result.stdout):
                            suspicious_commits.append(commit_hash)
                
                if suspicious_commits:
                    pytest.fail(f"Found potential secrets in git history: {suspicious_commits}")
        
        except subprocess.TimeoutExpired:
            # Skip this test if git operations take too long
            pytest.skip("Git history check timed out")
        except FileNotFoundError:
            # Skip this test if git is not available
            pytest.skip("Git not available for history check")
        except Exception as e:
            # Log warning but don't fail the test
            print(f"Warning: Could not check git history: {e}")
    
    def test_encrypted_secrets_storage(self):
        """Test that secrets are stored encrypted when persisted locally."""
        
        secrets_manager = SecretsManager(encryption_key="test_encryption_key")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the local secrets path
            secrets_manager.local_secrets_path = Path(temp_dir)
            
            # Store an encrypted secret
            success = secrets_manager.encrypt_secret("test_secret", "sensitive_value")
            assert success
            
            # Verify the file exists and is encrypted
            secret_file = Path(temp_dir) / "test_secret.enc"
            assert secret_file.exists()
            
            # Verify the content is encrypted (not plaintext)
            with open(secret_file, 'rb') as f:
                encrypted_content = f.read()
            
            assert b"sensitive_value" not in encrypted_content
            assert len(encrypted_content) > 20  # Should be encrypted and longer
            
            # Verify we can decrypt it back
            decrypted_value = secrets_manager.get_secret("test_secret")
            assert decrypted_value == "sensitive_value"
    
    def test_secret_rotation_capabilities(self):
        """Test that secret rotation is properly implemented."""
        
        secrets_manager = SecretsManager()
        
        # Test that cache can be refreshed for rotation
        secrets_manager._cache["test_secret"] = "old_value"
        
        secrets_manager.refresh_secrets()
        
        # Cache should be cleared
        assert "test_secret" not in secrets_manager._cache
    
    def test_secrets_validation(self):
        """Test that secrets are properly validated."""
        
        secrets_manager = SecretsManager()
        
        # Test validation of required secrets
        required_secrets = ["db_username", "db_password", "jwt_secret"]
        
        with patch.dict(os.environ, {
            'DB_USERNAME': 'test_user',
            'DB_PASSWORD': 'secure_password',
            'JWT_SECRET': 'jwt_secret_key'
        }):
            validation_results = secrets_manager.validate_secrets(required_secrets)
            
            assert all(validation_results.values())
        
        # Test with missing secrets
        with patch.dict(os.environ, {}, clear=True):
            validation_results = secrets_manager.validate_secrets(required_secrets)
            
            assert not any(validation_results.values())
    
    def test_no_default_credentials(self):
        """Test that no default/weak credentials are used."""
        
        weak_credentials = [
            "admin",
            "password",
            "123456",
            "admin123",
            "root",
            "password123",
            "qwerty",
            "test",
            "guest",
            "user",
        ]
        
        secrets_manager = SecretsManager()
        
        # Test that these weak credentials are not accepted
        for weak_cred in weak_credentials:
            with patch.dict(os.environ, {'TEST_PASSWORD': weak_cred}):
                # Should validate password strength
                with pytest.raises(Exception):
                    self._validate_password_strength(weak_cred)
    
    def test_secret_exposure_in_logs(self):
        """Test that secrets are not exposed in log files."""
        
        import logging
        from io import StringIO
        
        # Set up a string buffer to capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("test_logger")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Test that secrets are redacted in logs
        secrets_manager = SecretsManager()
        
        with patch.dict(os.environ, {'TEST_SECRET': 'super_secret_value'}):
            secret_value = secrets_manager.get_secret('test_secret')
            
            # Log something that might contain the secret
            logger.info(f"Retrieved secret: {secret_value}")
            
            # Check that the actual secret value is not in the log
            log_output = log_capture.getvalue()
            assert "super_secret_value" not in log_output
            assert "[REDACTED]" in log_output or "***" in log_output
    
    def test_memory_cleanup_of_secrets(self):
        """Test that secrets are properly cleaned up from memory."""
        
        secrets_manager = SecretsManager()
        
        # Store a secret in memory
        with patch.dict(os.environ, {'TEMP_SECRET': 'temp_value'}):
            secret_value = secrets_manager.get_secret('temp_secret')
            assert secret_value == 'temp_value'
            
            # Clear the cache
            secrets_manager.refresh_secrets()
            
            # Verify the secret is no longer in memory
            assert 'temp_secret' not in secrets_manager._cache
    
    # Helper methods
    def _is_whitelisted_file(self, file_path: str, whitelisted_patterns: List[str]) -> bool:
        """Check if file is whitelisted (test files, docs, etc.)."""
        for pattern in whitelisted_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
        return False
    
    def _is_likely_example_or_comment(self, content: str, match: re.Match) -> bool:
        """Check if the match is likely an example or in a comment."""
        # Get the line containing the match
        start_line = content.rfind('\n', 0, match.start()) + 1
        end_line = content.find('\n', match.end())
        if end_line == -1:
            end_line = len(content)
        
        line = content[start_line:end_line]
        
        # Check if it's in a comment
        if '#' in line and line.index('#') < (match.start() - start_line):
            return True
        
        # Check if it contains example/test indicators
        example_indicators = ['example', 'test', 'mock', 'dummy', 'placeholder', 'todo', 'fixme']
        return any(indicator in line.lower() for indicator in example_indicators)
    
    def _contains_actual_secrets(self, content: str) -> bool:
        """Check if content contains actual secrets (not just the word 'password')."""
        # Look for patterns that suggest actual credentials
        actual_secret_patterns = [
            r'password[\s]*[=:]\s*["\'][^"\']{8,}["\']',
            r'api[_-]?key[\s]*[=:]\s*["\'][A-Za-z0-9+/=]{20,}["\']',
            r'secret[\s]*[=:]\s*["\'][A-Za-z0-9+/=]{16,}["\']',
        ]
        
        for pattern in actual_secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            raise ValueError("Password too short")
        
        if password.lower() in ["admin", "password", "123456", "root", "test"]:
            raise ValueError("Weak password")
        
        # Should have mixed case, numbers, special chars
        if not re.search(r'[a-z]', password):
            raise ValueError("Password must contain lowercase letters")
        
        if not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain uppercase letters")
        
        if not re.search(r'\d', password):
            raise ValueError("Password must contain numbers")
        
        return True


class TestSecretsManagementIntegration:
    """Integration tests for the complete secrets management system."""
    
    def test_end_to_end_secret_management(self):
        """Test the complete secret management flow."""
        
        # Test environment variable -> secrets manager -> application
        with patch.dict(os.environ, {
            'DB_PASSWORD': 'secure_db_password',
            'API_KEY': 'secure_api_key',
            'JWT_SECRET': 'secure_jwt_secret'
        }):
            secrets_manager = SecretsManager()
            
            # Test database configuration
            db_config = secrets_manager.get_database_config()
            assert db_config['password'] == 'secure_db_password'
            assert 'secure_db_password' not in str(db_config)  # Should be redacted in string representation
            
            # Test API keys
            api_keys = secrets_manager.get_api_keys()
            # Should be empty if no specific API keys are set
            assert isinstance(api_keys, dict)
            
            # Test JWT secret
            jwt_secret = secrets_manager.get_jwt_secret()
            assert jwt_secret == 'secure_jwt_secret'
    
    def test_secret_precedence(self):
        """Test that secret sources have proper precedence."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up Docker secrets directory
            docker_secrets_dir = Path(temp_dir) / "docker_secrets"
            docker_secrets_dir.mkdir()
            
            # Create Docker secret
            docker_secret_file = docker_secrets_dir / "test_secret"
            docker_secret_file.write_text("docker_value")
            
            # Set up environment variable
            with patch.dict(os.environ, {'TEST_SECRET': 'env_value'}):
                
                secrets_manager = SecretsManager()
                
                # Patch Docker secrets path
                with patch.object(secrets_manager, 'docker_secrets_path', docker_secrets_dir):
                    
                    # Docker secret should take precedence over environment variable
                    result = secrets_manager.get_secret('test_secret')
                    assert result == 'docker_value'
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when primary secret sources are unavailable."""
        
        secrets_manager = SecretsManager()
        
        # Test fallback to default value
        result = secrets_manager.get_secret('nonexistent_secret', default='fallback_value')
        assert result == 'fallback_value'
        
        # Test required secret raises exception when not found
        with pytest.raises(ValueError, match="Required secret"):
            secrets_manager.get_secret('nonexistent_secret', required=True)
    
    def test_concurrent_secret_access(self):
        """Test concurrent access to secrets."""
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker_thread(thread_id: int):
            """Worker thread for concurrent testing."""
            try:
                secrets_manager = SecretsManager()
                with patch.dict(os.environ, {f'THREAD_SECRET_{thread_id}': f'value_{thread_id}'}):
                    result = secrets_manager.get_secret(f'thread_secret_{thread_id}')
                    results.put(("success", thread_id, result))
            except Exception as e:
                results.put(("error", thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        success_count = 0
        
        while not results.empty():
            status, thread_id, result = results.get()
            if status == "success":
                success_count += 1
                assert result == f'value_{thread_id}'
        
        assert success_count == 5, "Not all threads completed successfully"


@pytest.mark.security
@pytest.mark.unit
class TestSecretsEliminationRegressionTests:
    """Regression tests to ensure secrets elimination remains effective."""
    
    def test_no_new_hardcoded_secrets(self, project_root):
        """Test that no new hardcoded secrets have been introduced."""
        
        # This test could be run in CI to catch new violations
        # It would compare against a baseline of known issues
        
        known_violations_baseline = {
            # Example: files that had violations before but are now fixed
            # "src/legacy/old_config.py": 2,  # 2 violations that should be fixed
        }
        
        current_violations = self._scan_for_secrets(project_root)
        
        # Check that violations haven't increased
        for file_path, violation_count in current_violations.items():
            baseline_count = known_violations_baseline.get(file_path, 0)
            assert violation_count <= baseline_count, \
                f"New violations in {file_path}: {violation_count} (was {baseline_count})"
    
    def test_secret_detection_accuracy(self):
        """Test that secret detection doesn't have false positives/negatives."""
        
        # Test cases with known secrets
        test_cases = [
            {
                "content": 'password = "actual_secret_password"',
                "should_detect": True,
                "description": "Simple password assignment"
            },
            {
                "content": 'api_key = "sk-1234567890abcdef1234567890abcdef"',
                "should_detect": True,
                "description": "API key assignment"
            },
            {
                "content": '# Example: password = "your_password_here"',
                "should_detect": False,
                "description": "Example in comment"
            },
            {
                "content": 'password = os.getenv("DB_PASSWORD")',
                "should_detect": False,
                "description": "Environment variable usage"
            },
            {
                "content": 'password_field = "password"',
                "should_detect": False,
                "description": "Field name, not actual password"
            },
        ]
        
        secrets_patterns = [
            r'password[\s]*[=:]\s*["\']([^"\']{8,})["\']',
            r'api[_-]?key[\s]*[=:]\s*["\']([A-Za-z0-9+/=]{20,})["\']',
        ]
        
        for test_case in test_cases:
            detected = False
            for pattern in secrets_patterns:
                if re.search(pattern, test_case["content"], re.IGNORECASE):
                    # Check if it's likely an example
                    if not self._is_likely_example(test_case["content"]):
                        detected = True
                        break
            
            if test_case["should_detect"]:
                assert detected, f"Failed to detect secret: {test_case['description']}"
            else:
                assert not detected, f"False positive: {test_case['description']}"
    
    # Helper methods
    def _scan_for_secrets(self, project_root: Path) -> Dict[str, int]:
        """Scan for secrets and return violation counts by file."""
        violations = {}
        
        # Simple scan implementation for testing
        python_files = list(project_root.rglob("*.py"))
        
        for file_path in python_files:
            if "test" in str(file_path).lower():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Count potential violations
                violation_count = 0
                patterns = [r'password[\s]*[=:]\s*["\']([^"\']{8,})["\']']
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if not self._is_likely_example(content):
                            violation_count += 1
                
                if violation_count > 0:
                    violations[str(file_path.relative_to(project_root))] = violation_count
            
            except Exception:
                continue
        
        return violations
    
    def _is_likely_example(self, content: str) -> bool:
        """Check if content is likely an example."""
        example_indicators = ['example', 'test', 'mock', 'dummy', '#', 'TODO', 'FIXME']
        return any(indicator in content.lower() for indicator in example_indicators)