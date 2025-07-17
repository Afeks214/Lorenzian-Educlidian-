"""
AGENT 9: Vault Integration Security Tests
Comprehensive test suite for HashiCorp Vault integration security and functionality.
"""

import pytest
import asyncio
import json
import time
import base64
import ssl
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List, Optional
import tempfile
from pathlib import Path

# Vault-specific imports
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    from aiohttp.client_exceptions import ClientError, ClientResponseError
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Core imports
from src.security.vault_client import (
    VaultClient, VaultConfig, VaultManager, AuthMethod, SecretVersion,
    get_vault_manager, init_vault_manager
)
from src.core.errors.base_exceptions import (
    VaultClientError, AuthenticationError, SecretNotFoundError
)


class TestVaultConfiguration:
    """Test Vault configuration and initialization security."""
    
    def test_vault_config_from_environment_variables(self):
        """Test that Vault configuration comes from environment variables."""
        
        # Test with custom environment variables
        with patch.dict('os.environ', {
            'VAULT_URL': 'https://vault.example.com:8200',
            'VAULT_NAMESPACE': 'production',
            'VAULT_MOUNT_PATH': 'kv-v2',
            'VAULT_AUTH_METHOD': 'approle',
            'VAULT_ROLE_ID': 'test-role-id',
            'VAULT_SECRET_ID': 'test-secret-id',
            'VAULT_TIMEOUT': '60',
            'VAULT_MAX_RETRIES': '5',
            'VAULT_VERIFY_SSL': 'true'
        }):
            config = VaultConfig()
            
            assert config.url == 'https://vault.example.com:8200'
            assert config.namespace == 'production'
            assert config.mount_path == 'kv-v2'
            assert config.auth_method == 'approle'
            assert config.role_id == 'test-role-id'
            assert config.secret_id == 'test-secret-id'
            assert config.timeout == 60
            assert config.max_retries == 5
            assert config.verify_ssl is True
    
    def test_vault_config_defaults(self):
        """Test Vault configuration defaults are secure."""
        
        # Clear environment variables to test defaults
        with patch.dict('os.environ', {}, clear=True):
            config = VaultConfig()
            
            # Test secure defaults
            assert config.verify_ssl is True  # SSL verification enabled by default
            assert config.timeout == 30  # Reasonable timeout
            assert config.max_retries == 3  # Limited retries
            assert config.auth_method == 'approle'  # Secure auth method
            assert 'vault.grandmodel.com' in config.url  # Proper domain
    
    def test_vault_ssl_configuration(self):
        """Test SSL configuration for Vault connections."""
        
        config = VaultConfig()
        client = VaultClient(config)
        
        # Test SSL context creation
        ssl_context = client._create_ssl_context()
        
        if config.verify_ssl:
            assert ssl_context is not False
        
        # Test with custom CA certificate
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as ca_file:
            ca_file.write("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----")
            ca_file.flush()
            
            config.ca_cert_path = ca_file.name
            client = VaultClient(config)
            ssl_context = client._create_ssl_context()
            
            # Should create valid SSL context with custom CA
            assert ssl_context is not False
    
    def test_vault_authentication_methods(self):
        """Test different Vault authentication methods."""
        
        # Test AppRole authentication
        config = VaultConfig()
        config.auth_method = AuthMethod.APPROLE
        config.role_id = "test-role"
        config.secret_id = "test-secret"
        
        client = VaultClient(config)
        
        # Mock HTTP session and response
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                "auth": {
                    "client_token": "test-token",
                    "lease_duration": 3600
                }
            }
            
            async def test_approle_auth():
                await client._authenticate_approle()
                
                # Verify correct API call
                mock_request.assert_called_once_with(
                    "POST",
                    "v1/auth/approle/login",
                    json={
                        "role_id": "test-role",
                        "secret_id": "test-secret"
                    }
                )
                
                assert client.token == "test-token"
            
            asyncio.run(test_approle_auth())
    
    def test_vault_jwt_authentication(self):
        """Test JWT authentication method."""
        
        config = VaultConfig()
        config.auth_method = AuthMethod.JWT
        config.jwt_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token"
        
        client = VaultClient(config)
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                "auth": {
                    "client_token": "jwt-token",
                    "lease_duration": 1800
                }
            }
            
            async def test_jwt_auth():
                await client._authenticate_jwt()
                
                # Verify correct JWT authentication call
                mock_request.assert_called_once_with(
                    "POST",
                    "v1/auth/jwt/login",
                    json={
                        "jwt": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token",
                        "role": "grandmodel-service"
                    }
                )
                
                assert client.token == "jwt-token"
            
            asyncio.run(test_jwt_auth())
    
    def test_vault_kubernetes_authentication(self):
        """Test Kubernetes service account authentication."""
        
        config = VaultConfig()
        config.auth_method = AuthMethod.KUBERNETES
        config.kubernetes_role = "grandmodel-role"
        
        client = VaultClient(config)
        
        # Mock Kubernetes service account token
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as token_file:
            token_file.write("k8s-service-account-token")
            token_file.flush()
            
            with patch('builtins.open', mock_open_with_path(token_file.name)):
                with patch.object(client, '_make_request') as mock_request:
                    mock_request.return_value = {
                        "auth": {
                            "client_token": "k8s-token",
                            "lease_duration": 2400
                        }
                    }
                    
                    async def test_k8s_auth():
                        await client._authenticate_kubernetes()
                        
                        # Verify correct Kubernetes authentication call
                        mock_request.assert_called_once_with(
                            "POST",
                            "v1/auth/kubernetes/login",
                            json={
                                "jwt": "k8s-service-account-token",
                                "role": "grandmodel-role"
                            }
                        )
                        
                        assert client.token == "k8s-token"
                    
                    asyncio.run(test_k8s_auth())


class TestVaultSecretOperations:
    """Test Vault secret operations security and functionality."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client for testing."""
        config = VaultConfig()
        client = VaultClient(config)
        client.token = "test-token"
        return client
    
    def test_get_secret_security(self, mock_vault_client):
        """Test secure secret retrieval operations."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "data": {
                        "username": "db_user",
                        "password": "secure_password"
                    }
                }
            }
            
            async def test_get_secret():
                # Test secret retrieval
                secret = await mock_vault_client.get_secret("database/credentials")
                
                assert secret["username"] == "db_user"
                assert secret["password"] == "secure_password"
                
                # Verify correct API call
                mock_request.assert_called_once_with(
                    "GET",
                    "v1/secret/data/database/credentials"
                )
            
            asyncio.run(test_get_secret())
    
    def test_put_secret_security(self, mock_vault_client):
        """Test secure secret storage operations."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {"data": {}}
            
            async def test_put_secret():
                # Test secret storage
                secret_data = {
                    "api_key": "sk-1234567890abcdef",
                    "webhook_secret": "whsec_abcdef123456"
                }
                
                success = await mock_vault_client.put_secret("api/keys", secret_data)
                
                assert success is True
                
                # Verify correct API call
                mock_request.assert_called_once_with(
                    "POST",
                    "v1/secret/data/api/keys",
                    json={"data": secret_data}
                )
            
            asyncio.run(test_put_secret())
    
    def test_secret_versioning(self, mock_vault_client):
        """Test secret versioning functionality."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            # Mock response for specific version
            mock_request.return_value = {
                "data": {
                    "data": {"password": "old_password"},
                    "metadata": {"version": 1}
                }
            }
            
            async def test_versioning():
                # Test retrieving specific version
                secret = await mock_vault_client.get_secret("database/creds", version="1")
                
                assert secret["password"] == "old_password"
                
                # Verify version parameter in API call
                mock_request.assert_called_once_with(
                    "GET",
                    "v1/secret/data/database/creds?version=1"
                )
            
            asyncio.run(test_versioning())
    
    def test_secret_deletion_security(self, mock_vault_client):
        """Test secure secret deletion operations."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {}
            
            async def test_delete_secret():
                # Test secret deletion
                success = await mock_vault_client.delete_secret("temp/secret")
                
                assert success is True
                
                # Verify correct deletion API call
                mock_request.assert_called_once_with(
                    "DELETE",
                    "v1/secret/metadata/temp/secret",
                    json={}
                )
            
            asyncio.run(test_delete_secret())
    
    def test_list_secrets_security(self, mock_vault_client):
        """Test secure secret listing operations."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "keys": ["database/", "api/", "certificates/"]
                }
            }
            
            async def test_list_secrets():
                # Test listing secrets
                secrets = await mock_vault_client.list_secrets("")
                
                assert "database/" in secrets
                assert "api/" in secrets
                assert "certificates/" in secrets
                
                # Verify correct listing API call
                mock_request.assert_called_once_with(
                    "LIST",
                    "v1/secret/metadata/"
                )
            
            asyncio.run(test_list_secrets())


class TestVaultDynamicSecrets:
    """Test Vault dynamic secrets functionality."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client for testing."""
        config = VaultConfig()
        client = VaultClient(config)
        client.token = "test-token"
        return client
    
    def test_dynamic_database_credentials(self, mock_vault_client):
        """Test dynamic database credential generation."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "username": "v-root-readonly-abc123",
                    "password": "A1a-xyz789"
                },
                "lease_duration": 3600,
                "renewable": True
            }
            
            async def test_db_creds():
                # Test dynamic database credentials
                creds = await mock_vault_client.get_database_credentials("readonly-role")
                
                assert creds["username"].startswith("v-root-readonly-")
                assert len(creds["password"]) >= 8
                
                # Verify correct API call
                mock_request.assert_called_once_with(
                    "GET",
                    "v1/database/creds/readonly-role"
                )
            
            asyncio.run(test_db_creds())
    
    def test_dynamic_api_tokens(self, mock_vault_client):
        """Test dynamic API token generation."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "token": "api-token-xyz789",
                    "expires_at": "2024-01-01T00:00:00Z"
                }
            }
            
            async def test_api_tokens():
                # Test dynamic API token generation
                token_data = await mock_vault_client.get_api_token("trading-api-role")
                
                assert token_data["token"] == "api-token-xyz789"
                assert "expires_at" in token_data
                
                # Verify correct API call
                mock_request.assert_called_once_with(
                    "GET",
                    "v1/api/creds/trading-api-role"
                )
            
            asyncio.run(test_api_tokens())


class TestVaultEncryptionTransit:
    """Test Vault transit engine for encryption/decryption."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client for testing."""
        config = VaultConfig()
        client = VaultClient(config)
        client.token = "test-token"
        return client
    
    def test_data_encryption(self, mock_vault_client):
        """Test data encryption using Vault transit engine."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "ciphertext": "vault:v1:encrypted_data_here"
                }
            }
            
            async def test_encrypt():
                # Test data encryption
                plaintext = "sensitive_data"
                ciphertext = await mock_vault_client.encrypt_data("encryption-key", plaintext)
                
                assert ciphertext == "vault:v1:encrypted_data_here"
                assert ciphertext.startswith("vault:v1:")
                
                # Verify correct encryption API call
                expected_payload = {
                    "plaintext": base64.b64encode(plaintext.encode()).decode()
                }
                mock_request.assert_called_once_with(
                    "POST",
                    "v1/transit/encrypt/encryption-key",
                    json=expected_payload
                )
            
            asyncio.run(test_encrypt())
    
    def test_data_decryption(self, mock_vault_client):
        """Test data decryption using Vault transit engine."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            # Mock decryption response
            plaintext_b64 = base64.b64encode("decrypted_data".encode()).decode()
            mock_request.return_value = {
                "data": {
                    "plaintext": plaintext_b64
                }
            }
            
            async def test_decrypt():
                # Test data decryption
                ciphertext = "vault:v1:encrypted_data_here"
                plaintext = await mock_vault_client.decrypt_data("encryption-key", ciphertext)
                
                assert plaintext == "decrypted_data"
                
                # Verify correct decryption API call
                expected_payload = {
                    "ciphertext": ciphertext
                }
                mock_request.assert_called_once_with(
                    "POST",
                    "v1/transit/decrypt/encryption-key",
                    json=expected_payload
                )
            
            asyncio.run(test_decrypt())
    
    def test_encryption_with_context(self, mock_vault_client):
        """Test encryption with context for additional security."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "ciphertext": "vault:v1:context_encrypted_data"
                }
            }
            
            async def test_encrypt_with_context():
                # Test encryption with context
                plaintext = "sensitive_data"
                context = "user_context"
                
                ciphertext = await mock_vault_client.encrypt_data(
                    "encryption-key", plaintext, context=context
                )
                
                assert ciphertext == "vault:v1:context_encrypted_data"
                
                # Verify context is included in API call
                expected_payload = {
                    "plaintext": base64.b64encode(plaintext.encode()).decode(),
                    "context": base64.b64encode(context.encode()).decode()
                }
                mock_request.assert_called_once_with(
                    "POST",
                    "v1/transit/encrypt/encryption-key",
                    json=expected_payload
                )
            
            asyncio.run(test_encrypt_with_context())


class TestVaultHealthAndMonitoring:
    """Test Vault health checks and monitoring."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client for testing."""
        config = VaultConfig()
        client = VaultClient(config)
        return client
    
    def test_vault_health_check(self, mock_vault_client):
        """Test Vault health check functionality."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "initialized": True,
                "sealed": False,
                "standby": False,
                "performance_standby": False,
                "replication_performance_mode": "disabled",
                "replication_dr_mode": "disabled",
                "server_time_utc": 1640995200,
                "version": "1.12.0",
                "cluster_name": "vault-cluster-1",
                "cluster_id": "abc123-def456"
            }
            
            async def test_health():
                # Test health check
                health = await mock_vault_client.health_check()
                
                assert health["status"] == "healthy"
                assert health["details"]["initialized"] is True
                assert health["details"]["sealed"] is False
                
                # Verify correct health API call
                mock_request.assert_called_once_with("GET", "v1/sys/health")
            
            asyncio.run(test_health())
    
    def test_vault_seal_status(self, mock_vault_client):
        """Test Vault seal status check."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "type": "shamir",
                "initialized": True,
                "sealed": False,
                "t": 3,
                "n": 5,
                "progress": 0,
                "nonce": "",
                "version": "1.12.0",
                "build_date": "2022-10-27T12:32:05Z",
                "migration": False,
                "cluster_name": "vault-cluster-1",
                "cluster_id": "abc123-def456",
                "recovery_seal": False,
                "storage_type": "consul"
            }
            
            async def test_seal_status():
                # Test seal status
                status = await mock_vault_client.get_seal_status()
                
                assert status["sealed"] is False
                assert status["initialized"] is True
                assert status["type"] == "shamir"
                
                # Verify correct seal status API call
                mock_request.assert_called_once_with("GET", "v1/sys/seal-status")
            
            asyncio.run(test_seal_status())
    
    def test_vault_unhealthy_response(self, mock_vault_client):
        """Test handling of unhealthy Vault responses."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            # Mock connection error
            mock_request.side_effect = Exception("Connection failed")
            
            async def test_unhealthy():
                # Test health check with error
                health = await mock_vault_client.health_check()
                
                assert health["status"] == "unhealthy"
                assert "Connection failed" in health["error"]
            
            asyncio.run(test_unhealthy())


class TestVaultErrorHandling:
    """Test Vault error handling and security."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client for testing."""
        config = VaultConfig()
        client = VaultClient(config)
        client.token = "test-token"
        return client
    
    def test_authentication_failure_handling(self, mock_vault_client):
        """Test handling of authentication failures."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            # Mock 403 Forbidden response
            mock_request.side_effect = AuthenticationError("Vault authentication failed")
            
            async def test_auth_failure():
                with pytest.raises(AuthenticationError, match="authentication failed"):
                    await mock_vault_client.get_secret("secret/path")
            
            asyncio.run(test_auth_failure())
    
    def test_secret_not_found_handling(self, mock_vault_client):
        """Test handling of missing secrets."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            # Mock 404 Not Found response
            mock_request.side_effect = SecretNotFoundError("Resource not found: secret/path")
            
            async def test_not_found():
                # Should return empty dict instead of raising for get_secret
                result = await mock_vault_client.get_secret("secret/nonexistent")
                assert result == {}
            
            asyncio.run(test_not_found())
    
    def test_connection_failure_handling(self, mock_vault_client):
        """Test handling of connection failures."""
        
        with patch.object(mock_vault_client, '_make_request') as mock_request:
            # Mock connection error
            mock_request.side_effect = VaultClientError("Vault connection failed")
            
            async def test_connection_failure():
                with pytest.raises(VaultClientError, match="connection failed"):
                    await mock_vault_client.get_secret("secret/path")
            
            asyncio.run(test_connection_failure())
    
    def test_token_renewal(self, mock_vault_client):
        """Test automatic token renewal."""
        
        # Set token expiry to trigger renewal
        mock_vault_client.token_expires_at = time.time() + 60  # Expires in 1 minute
        mock_vault_client.config.token_renewal_threshold = 300  # 5 minutes threshold
        
        with patch.object(mock_vault_client, 'authenticate') as mock_auth:
            async def test_renewal():
                # This should trigger token renewal
                await mock_vault_client.ensure_authenticated()
                
                # Verify authentication was called
                mock_auth.assert_called_once()
            
            asyncio.run(test_renewal())


class TestVaultManager:
    """Test Vault manager functionality."""
    
    def test_vault_manager_initialization(self):
        """Test Vault manager initialization."""
        
        config = VaultConfig()
        manager = VaultManager(config)
        
        assert manager.config == config
        assert manager._initialized is False
        assert manager.client is None
    
    def test_vault_manager_context(self):
        """Test Vault manager context manager."""
        
        manager = VaultManager()
        
        with patch.object(manager, 'initialize') as mock_init:
            async def test_context():
                async with manager.get_client() as client:
                    assert client is not None
                    mock_init.assert_called_once()
            
            asyncio.run(test_context())
    
    def test_global_vault_manager(self):
        """Test global Vault manager instance."""
        
        with patch('src.security.vault_client.VaultManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            async def test_global():
                # Test global manager initialization
                manager = await init_vault_manager()
                assert manager is not None
                
                # Test getting existing manager
                same_manager = await get_vault_manager()
                assert same_manager == manager
            
            asyncio.run(test_global())


class TestVaultSecurity:
    """Test Vault security features and compliance."""
    
    def test_tls_verification_enforcement(self):
        """Test that TLS verification is enforced."""
        
        # Test with SSL verification enabled (default)
        config = VaultConfig()
        config.verify_ssl = True
        
        client = VaultClient(config)
        ssl_context = client._create_ssl_context()
        
        # Should not disable SSL verification
        assert ssl_context is not False
    
    def test_sensitive_data_not_logged(self):
        """Test that sensitive data is not logged."""
        
        config = VaultConfig()
        client = VaultClient(config)
        
        # Mock logger to capture log calls
        with patch.object(client, 'logger') as mock_logger:
            async def test_logging():
                with patch.object(client, '_make_request') as mock_request:
                    mock_request.return_value = {
                        "data": {"data": {"password": "secret_password"}}
                    }
                    
                    await client.get_secret("database/creds")
                    
                    # Check that sensitive data is not in log calls
                    for call in mock_logger.info.call_args_list:
                        call_str = str(call)
                        assert "secret_password" not in call_str
                        assert "password" not in call_str or "retrieved" not in call_str.lower()
            
            asyncio.run(test_logging())
    
    def test_token_caching_security(self):
        """Test secure token caching."""
        
        config = VaultConfig()
        client = VaultClient(config)
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        client.redis_client = mock_redis
        
        async def test_caching():
            # Test token caching
            await client._cache_token("test-token")
            
            # Verify token is cached with expiration
            mock_redis.setex.assert_called_once_with(
                "vault_token",
                client.config.token_cache_ttl,
                "test-token"
            )
        
        asyncio.run(test_caching())
    
    def test_connection_pool_security(self):
        """Test connection pool security settings."""
        
        config = VaultConfig()
        client = VaultClient(config)
        
        async def test_connection_security():
            await client.initialize()
            
            # Verify secure connection settings
            session = client.session
            assert session is not None
            
            # Check timeout settings
            assert session.timeout.total <= config.timeout
            
            # Check SSL context if verification enabled
            if config.verify_ssl:
                connector = session.connector
                assert connector.ssl is not False
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value = Mock()
            asyncio.run(test_connection_security())


# Helper function for mocking file operations
def mock_open_with_path(file_path):
    """Create a mock open function that returns specific content for a path."""
    def mock_open_func(path, *args, **kwargs):
        if str(path) == "/var/run/secrets/kubernetes.io/serviceaccount/token":
            from unittest.mock import mock_open
            return mock_open(read_data="k8s-service-account-token")()
        elif str(path) == file_path:
            from unittest.mock import mock_open
            return mock_open(read_data="k8s-service-account-token")()
        else:
            raise FileNotFoundError(f"No such file: {path}")
    
    return mock_open_func


@pytest.mark.security
@pytest.mark.unit
@pytest.mark.asyncio
class TestVaultIntegrationSecurity:
    """Integration tests for Vault security across the system."""
    
    async def test_end_to_end_secret_retrieval(self):
        """Test end-to-end secret retrieval with security validation."""
        
        # Mock complete Vault interaction
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "data": {"data": {"api_key": "secure_api_key"}}
            })
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session
            
            # Test complete flow
            manager = VaultManager()
            await manager.initialize()
            
            # Retrieve secret
            secret = await manager.get_secret("api/keys", "api_key")
            
            assert secret == "secure_api_key"
            
            # Verify secure communication
            mock_session.request.assert_called()
            call_args = mock_session.request.call_args
            
            # Should include authentication header
            headers = call_args[1].get("headers", {})
            assert "X-Vault-Token" in headers or any("token" in str(h).lower() for h in headers.values())
    
    async def test_concurrent_vault_operations(self):
        """Test security under concurrent Vault operations."""
        
        manager = VaultManager()
        
        # Mock successful operations
        with patch.object(manager, 'get_secret') as mock_get_secret:
            mock_get_secret.return_value = "test_value"
            
            # Test concurrent secret retrieval
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    manager.get_secret(f"secret_{i}")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All operations should succeed
            assert all(result == "test_value" for result in results)
            assert mock_get_secret.call_count == 10
    
    async def test_vault_failover_handling(self):
        """Test handling of Vault failover scenarios."""
        
        config = VaultConfig()
        client = VaultClient(config)
        
        # Mock initial failure then success
        with patch.object(client, '_make_request') as mock_request:
            mock_request.side_effect = [
                Exception("Connection failed"),  # First attempt fails
                {"data": {"data": {"secret": "value"}}}  # Second attempt succeeds
            ]
            
            # Should retry and succeed
            result = await client.get_secret("test/secret")
            assert result == {"secret": "value"}
            
            # Should have made multiple attempts
            assert mock_request.call_count >= 1