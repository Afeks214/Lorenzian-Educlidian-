"""
Secrets management for secure handling of sensitive configuration.
Supports Docker secrets, environment variables, and cloud secret managers.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)


class SecretsManager:
    """
    Manages secrets from various sources with encryption support.
    
    Supports:
    - Docker secrets (files in /run/secrets/)
    - Environment variables
    - Encrypted local files
    - Future: AWS Secrets Manager, HashiCorp Vault
    """
    
    DOCKER_SECRETS_PATH = "/run/secrets"
    LOCAL_SECRETS_PATH = "/app/secrets"
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize secrets manager.
        
        Args:
            encryption_key: Master key for local encryption (from env var)
        """
        self.docker_secrets_path = Path(self.DOCKER_SECRETS_PATH)
        self.local_secrets_path = Path(self.LOCAL_SECRETS_PATH)
        self._cache: Dict[str, Any] = {}
        
        # Initialize encryption if key provided
        self.cipher_suite = None
        if encryption_key:
            self._init_encryption(encryption_key)
        elif os.getenv("SECRETS_ENCRYPTION_KEY"):
            self._init_encryption(os.getenv("SECRETS_ENCRYPTION_KEY"))
            
        logger.info("Secrets manager initialized")
    
    def _init_encryption(self, master_key: str):
        """Initialize Fernet encryption with master key."""
        try:
            # Derive encryption key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'grandmodel-salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
            self.cipher_suite = Fernet(key)
            logger.info("Encryption initialized for secrets")
        except Exception as e:
            logger.error("Failed to initialize encryption", error=str(e))
            raise
    
    def get_secret(self, 
                   secret_name: str,
                   default: Optional[str] = None,
                   required: bool = False) -> Optional[str]:
        """
        Get a secret value from available sources.
        
        Priority order:
        1. Docker secrets
        2. Environment variables
        3. Encrypted local file
        4. Default value
        
        Args:
            secret_name: Name of the secret
            default: Default value if not found
            required: Raise exception if not found
            
        Returns:
            Secret value or None
            
        Raises:
            ValueError: If required secret not found
        """
        # Check cache first
        if secret_name in self._cache:
            return self._cache[secret_name]
        
        # Try Docker secrets
        value = self._get_docker_secret(secret_name)
        
        # Try environment variable
        if value is None:
            env_name = secret_name.upper().replace("-", "_")
            value = os.getenv(env_name)
        
        # Try encrypted local file
        if value is None and self.cipher_suite:
            value = self._get_encrypted_secret(secret_name)
        
        # Use default
        if value is None:
            value = default
        
        # Validate required
        if required and value is None:
            logger.error("Required secret not found", secret_name=secret_name)
            raise ValueError(f"Required secret '{secret_name}' not found")
        
        # Cache the value
        if value is not None:
            self._cache[secret_name] = value
            # Don't log the actual secret value
            logger.info("Secret loaded", secret_name=secret_name, source="[REDACTED]")
        
        return value
    
    def _get_docker_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from Docker secrets directory."""
        secret_file = self.docker_secrets_path / secret_name
        
        if secret_file.exists() and secret_file.is_file():
            try:
                with open(secret_file, 'r') as f:
                    value = f.read().strip()
                    if value:
                        logger.info("Secret loaded from Docker", secret_name=secret_name)
                        return value
            except Exception as e:
                logger.error(
                    "Failed to read Docker secret",
                    secret_name=secret_name,
                    error=str(e)
                )
        
        return None
    
    def _get_encrypted_secret(self, secret_name: str) -> Optional[str]:
        """Get and decrypt secret from local encrypted file."""
        secret_file = self.local_secrets_path / f"{secret_name}.enc"
        
        if secret_file.exists() and secret_file.is_file():
            try:
                with open(secret_file, 'rb') as f:
                    encrypted_data = f.read()
                    decrypted_value = self.cipher_suite.decrypt(encrypted_data)
                    value = decrypted_value.decode('utf-8').strip()
                    if value:
                        logger.info("Secret loaded from encrypted file", secret_name=secret_name)
                        return value
            except Exception as e:
                logger.error(
                    "Failed to decrypt secret",
                    secret_name=secret_name,
                    error=str(e)
                )
        
        return None
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration with secrets.
        
        Returns:
            Database configuration dict
        """
        return {
            "host": self.get_secret("db_host", default="localhost"),
            "port": int(self.get_secret("db_port", default="5432")),
            "database": self.get_secret("db_name", default="grandmodel"),
            "username": self.get_secret("db_username", required=True),
            "password": self.get_secret("db_password", required=True),
            "ssl_mode": self.get_secret("db_ssl_mode", default="require")
        }
    
    def get_api_keys(self) -> Dict[str, str]:
        """
        Get API keys for external services.
        
        Returns:
            Dict of service name to API key
        """
        api_keys = {}
        
        # Load API keys for known services
        services = ["binance", "coinbase", "kraken", "openai", "anthropic"]
        
        for service in services:
            key = self.get_secret(f"{service}_api_key")
            if key:
                api_keys[service] = key
        
        # Load generic API keys from JSON file
        api_keys_json = self.get_secret("api_keys_json")
        if api_keys_json:
            try:
                additional_keys = json.loads(api_keys_json)
                api_keys.update(additional_keys)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in api_keys_json secret")
        
        return api_keys
    
    def get_jwt_secret(self) -> str:
        """
        Get JWT secret key.
        
        Returns:
            JWT secret key
            
        Raises:
            ValueError: If JWT secret not found
        """
        return self.get_secret("jwt_secret", required=True)
    
    def get_redis_password(self) -> Optional[str]:
        """
        Get Redis password.
        
        Returns:
            Redis password or None
        """
        return self.get_secret("redis_password")
    
    def refresh_secrets(self):
        """
        Refresh secrets by clearing cache.
        Useful for secret rotation.
        """
        self._cache.clear()
        logger.info("Secrets cache cleared")
    
    def encrypt_secret(self, secret_name: str, secret_value: str) -> bool:
        """
        Encrypt and store a secret locally.
        
        Args:
            secret_name: Name of the secret
            secret_value: Secret value to encrypt
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cipher_suite:
            logger.error("Encryption not initialized")
            return False
        
        try:
            # Ensure secrets directory exists
            self.local_secrets_path.mkdir(parents=True, exist_ok=True)
            
            # Encrypt the value
            encrypted_data = self.cipher_suite.encrypt(secret_value.encode())
            
            # Write to file
            secret_file = self.local_secrets_path / f"{secret_name}.enc"
            with open(secret_file, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info("Secret encrypted and stored", secret_name=secret_name)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to encrypt secret",
                secret_name=secret_name,
                error=str(e)
            )
            return False
    
    def validate_secrets(self, required_secrets: list[str]) -> Dict[str, bool]:
        """
        Validate that required secrets are available.
        
        Args:
            required_secrets: List of required secret names
            
        Returns:
            Dict of secret name to availability status
        """
        validation_results = {}
        
        for secret_name in required_secrets:
            try:
                value = self.get_secret(secret_name, required=True)
                validation_results[secret_name] = value is not None
            except ValueError:
                validation_results[secret_name] = False
        
        # Log validation summary
        available = sum(1 for v in validation_results.values() if v)
        total = len(required_secrets)
        
        if available == total:
            logger.info(
                "All required secrets available",
                available=available,
                total=total
            )
        else:
            missing = [k for k, v in validation_results.items() if not v]
            logger.error(
                "Missing required secrets",
                missing=missing,
                available=available,
                total=total
            )
        
        return validation_results


# Global secrets manager instance
secrets_manager = SecretsManager()


def get_secret(secret_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Convenience function to get a secret.
    
    Args:
        secret_name: Name of the secret
        default: Default value if not found
        required: Raise exception if not found
        
    Returns:
        Secret value or None
    """
    return secrets_manager.get_secret(secret_name, default, required)