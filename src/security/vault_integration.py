"""
HashiCorp Vault Integration
Centralized secrets management with automatic rotation
"""

import os
import json
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import base64
import hashlib
import hmac
import ssl
from urllib.parse import urljoin

from src.monitoring.logger_config import get_logger
from src.security.tls_manager import create_secure_ssl_context

logger = get_logger(__name__)

class VaultAuthMethod(Enum):
    """Vault authentication methods"""
    TOKEN = "token"
    APPROLE = "approle"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    JWT = "jwt"
    USERPASS = "userpass"
    LDAP = "ldap"

class SecretEngine(Enum):
    """Vault secret engines"""
    KV_V2 = "kv-v2"
    KV_V1 = "kv-v1"
    DATABASE = "database"
    PKI = "pki"
    TRANSIT = "transit"
    TOTP = "totp"
    SSH = "ssh"
    AWS_SECRETS = "aws"
    AZURE_SECRETS = "azure"

@dataclass
class VaultConfig:
    """Vault configuration"""
    url: str
    auth_method: VaultAuthMethod
    mount_point: str = "secret"
    namespace: Optional[str] = None
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30
    retries: int = 3
    retry_delay: float = 1.0
    token_renewal_threshold: int = 300  # Renew token if TTL < 5 minutes
    
@dataclass
class VaultSecret:
    """Vault secret metadata"""
    path: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: Optional[int] = None
    created_time: Optional[datetime] = None
    deletion_time: Optional[datetime] = None
    destroyed: bool = False
    
@dataclass
class VaultToken:
    """Vault token information"""
    token: str
    accessor: str
    ttl: int
    renewable: bool
    policies: List[str]
    entity_id: Optional[str] = None
    expire_time: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        if self.expire_time:
            return datetime.utcnow() > self.expire_time
        return False
    
    def needs_renewal(self, threshold: int = 300) -> bool:
        """Check if token needs renewal"""
        if self.expire_time:
            return (self.expire_time - datetime.utcnow()).total_seconds() < threshold
        return False

class VaultClient:
    """Production-ready HashiCorp Vault client"""
    
    def __init__(self, config: VaultConfig):
        self.config = config
        self.token: Optional[VaultToken] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._auth_lock = asyncio.Lock()
        self._token_renewal_task: Optional[asyncio.Task] = None
        
        # SSL context
        self.ssl_context = None
        if config.verify_ssl:
            self.ssl_context = create_secure_ssl_context()
            if config.ca_cert_path:
                self.ssl_context.load_verify_locations(config.ca_cert_path)
            if config.client_cert_path and config.client_key_path:
                self.ssl_context.load_cert_chain(config.client_cert_path, config.client_key_path)
        
        logger.info("VaultClient initialized", url=config.url, auth_method=config.auth_method.value)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize Vault client"""
        # Create HTTP session
        connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"X-Vault-Namespace": self.config.namespace} if self.config.namespace else {}
        )
        
        # Authenticate
        await self.authenticate()
        
        # Start token renewal task
        if self.token and self.token.renewable:
            self._token_renewal_task = asyncio.create_task(self._token_renewal_loop())
        
        logger.info("VaultClient initialized successfully")
    
    async def close(self):
        """Close Vault client"""
        if self._token_renewal_task:
            self._token_renewal_task.cancel()
            try:
                await self._token_renewal_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("VaultClient closed")
    
    async def authenticate(self):
        """Authenticate with Vault"""
        async with self._auth_lock:
            if self.config.auth_method == VaultAuthMethod.TOKEN:
                await self._authenticate_token()
            elif self.config.auth_method == VaultAuthMethod.APPROLE:
                await self._authenticate_approle()
            elif self.config.auth_method == VaultAuthMethod.KUBERNETES:
                await self._authenticate_kubernetes()
            elif self.config.auth_method == VaultAuthMethod.AWS:
                await self._authenticate_aws()
            elif self.config.auth_method == VaultAuthMethod.AZURE:
                await self._authenticate_azure()
            elif self.config.auth_method == VaultAuthMethod.JWT:
                await self._authenticate_jwt()
            elif self.config.auth_method == VaultAuthMethod.USERPASS:
                await self._authenticate_userpass()
            elif self.config.auth_method == VaultAuthMethod.LDAP:
                await self._authenticate_ldap()
            else:
                raise ValueError(f"Unsupported auth method: {self.config.auth_method}")
    
    async def _authenticate_token(self):
        """Authenticate using token"""
        token = os.getenv("VAULT_TOKEN")
        if not token:
            token_file = Path("~/.vault-token").expanduser()
            if token_file.exists():
                token = token_file.read_text().strip()
        
        if not token:
            raise ValueError("No Vault token found")
        
        # Verify token
        url = urljoin(self.config.url, "v1/auth/token/lookup-self")
        headers = {"X-Vault-Token": token}
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                token_data = data["data"]
                
                self.token = VaultToken(
                    token=token,
                    accessor=token_data.get("accessor"),
                    ttl=token_data.get("ttl", 0),
                    renewable=token_data.get("renewable", False),
                    policies=token_data.get("policies", []),
                    entity_id=token_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=token_data.get("ttl", 0))
                )
                
                logger.info("Token authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"Token authentication failed: {response.status}")
    
    async def _authenticate_approle(self):
        """Authenticate using AppRole"""
        role_id = os.getenv("VAULT_ROLE_ID")
        secret_id = os.getenv("VAULT_SECRET_ID")
        
        if not role_id or not secret_id:
            raise ValueError("AppRole credentials not found")
        
        url = urljoin(self.config.url, "v1/auth/approle/login")
        payload = {
            "role_id": role_id,
            "secret_id": secret_id
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("AppRole authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"AppRole authentication failed: {response.status}")
    
    async def _authenticate_kubernetes(self):
        """Authenticate using Kubernetes service account"""
        jwt_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        if not jwt_path.exists():
            raise ValueError("Kubernetes service account token not found")
        
        jwt_token = jwt_path.read_text()
        role = os.getenv("VAULT_K8S_ROLE", "default")
        
        url = urljoin(self.config.url, "v1/auth/kubernetes/login")
        payload = {
            "jwt": jwt_token,
            "role": role
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("Kubernetes authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"Kubernetes authentication failed: {response.status}")
    
    async def _authenticate_aws(self):
        """Authenticate using AWS IAM"""
        import boto3
        
        # Get AWS credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if not credentials:
            raise ValueError("AWS credentials not found")
        
        # Create AWS signature
        region = os.getenv("AWS_REGION", "us-east-1")
        role = os.getenv("VAULT_AWS_ROLE", "default")
        
        url = urljoin(self.config.url, "v1/auth/aws/login")
        payload = {
            "role": role,
            "iam_http_request_method": "POST",
            "iam_request_url": base64.b64encode(url.encode()).decode(),
            "iam_request_body": base64.b64encode(json.dumps({"role": role}).encode()).decode(),
            "iam_request_headers": base64.b64encode(json.dumps({
                "Authorization": f"AWS4-HMAC-SHA256 Credential={credentials.access_key}",
                "X-Amz-Date": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            }).encode()).decode()
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("AWS authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"AWS authentication failed: {response.status}")
    
    async def _authenticate_azure(self):
        """Authenticate using Azure managed identity"""
        # Get Azure access token
        metadata_url = "http://169.254.169.254/metadata/identity/oauth2/token"
        params = {
            "api-version": "2018-02-01",
            "resource": "https://management.azure.com/"
        }
        headers = {"Metadata": "true"}
        
        async with self.session.get(metadata_url, params=params, headers=headers) as response:
            if response.status == 200:
                token_data = await response.json()
                azure_token = token_data["access_token"]
            else:
                raise ValueError("Failed to get Azure access token")
        
        # Authenticate with Vault
        role = os.getenv("VAULT_AZURE_ROLE", "default")
        url = urljoin(self.config.url, "v1/auth/azure/login")
        payload = {
            "role": role,
            "jwt": azure_token
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("Azure authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"Azure authentication failed: {response.status}")
    
    async def _authenticate_jwt(self):
        """Authenticate using JWT"""
        jwt_token = os.getenv("VAULT_JWT_TOKEN")
        role = os.getenv("VAULT_JWT_ROLE", "default")
        
        if not jwt_token:
            raise ValueError("JWT token not found")
        
        url = urljoin(self.config.url, "v1/auth/jwt/login")
        payload = {
            "jwt": jwt_token,
            "role": role
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("JWT authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"JWT authentication failed: {response.status}")
    
    async def _authenticate_userpass(self):
        """Authenticate using username/password"""
        username = os.getenv("VAULT_USERNAME")
        password = os.getenv("VAULT_PASSWORD")
        
        if not username or not password:
            raise ValueError("Username/password not found")
        
        url = urljoin(self.config.url, f"v1/auth/userpass/login/{username}")
        payload = {"password": password}
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("Userpass authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"Userpass authentication failed: {response.status}")
    
    async def _authenticate_ldap(self):
        """Authenticate using LDAP"""
        username = os.getenv("VAULT_LDAP_USERNAME")
        password = os.getenv("VAULT_LDAP_PASSWORD")
        
        if not username or not password:
            raise ValueError("LDAP username/password not found")
        
        url = urljoin(self.config.url, f"v1/auth/ldap/login/{username}")
        payload = {"password": password}
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token = VaultToken(
                    token=auth_data["client_token"],
                    accessor=auth_data["accessor"],
                    ttl=auth_data.get("lease_duration", 0),
                    renewable=auth_data.get("renewable", False),
                    policies=auth_data.get("policies", []),
                    entity_id=auth_data.get("entity_id"),
                    expire_time=datetime.utcnow() + timedelta(seconds=auth_data.get("lease_duration", 0))
                )
                
                logger.info("LDAP authentication successful", policies=self.token.policies)
            else:
                raise ValueError(f"LDAP authentication failed: {response.status}")
    
    async def _token_renewal_loop(self):
        """Background task for token renewal"""
        while True:
            try:
                if self.token and self.token.needs_renewal(self.config.token_renewal_threshold):
                    await self.renew_token()
                
                # Check every minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token renewal failed: {e}")
                await asyncio.sleep(60)
    
    async def renew_token(self):
        """Renew the current token"""
        if not self.token or not self.token.renewable:
            logger.warning("Token is not renewable")
            return
        
        url = urljoin(self.config.url, "v1/auth/token/renew-self")
        headers = {"X-Vault-Token": self.token.token}
        
        async with self.session.post(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                auth_data = data["auth"]
                
                self.token.ttl = auth_data.get("lease_duration", 0)
                self.token.expire_time = datetime.utcnow() + timedelta(seconds=self.token.ttl)
                
                logger.info("Token renewed successfully", new_ttl=self.token.ttl)
            else:
                logger.error(f"Token renewal failed: {response.status}")
    
    async def read_secret(self, path: str, version: Optional[int] = None) -> Optional[VaultSecret]:
        """Read secret from Vault"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        # Handle KV v2 versioning
        if version is not None:
            url = urljoin(self.config.url, f"v1/{self.config.mount_point}/data/{path}")
            params = {"version": version}
        else:
            url = urljoin(self.config.url, f"v1/{self.config.mount_point}/data/{path}")
            params = {}
        
        headers = {"X-Vault-Token": self.token.token}
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                # Handle KV v2 format
                if "data" in data and "data" in data["data"]:
                    secret_data = data["data"]["data"]
                    metadata = data["data"]["metadata"]
                    
                    return VaultSecret(
                        path=path,
                        data=secret_data,
                        metadata=metadata,
                        version=metadata.get("version"),
                        created_time=datetime.fromisoformat(metadata.get("created_time", "").replace("Z", "+00:00")) if metadata.get("created_time") else None,
                        deletion_time=datetime.fromisoformat(metadata.get("deletion_time", "").replace("Z", "+00:00")) if metadata.get("deletion_time") else None,
                        destroyed=metadata.get("destroyed", False)
                    )
                else:
                    # KV v1 format
                    return VaultSecret(
                        path=path,
                        data=data["data"],
                        metadata={}
                    )
            elif response.status == 404:
                return None
            else:
                raise ValueError(f"Failed to read secret: {response.status}")
    
    async def write_secret(self, path: str, data: Dict[str, Any], cas: Optional[int] = None) -> bool:
        """Write secret to Vault"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        url = urljoin(self.config.url, f"v1/{self.config.mount_point}/data/{path}")
        headers = {"X-Vault-Token": self.token.token}
        
        # Handle KV v2 format
        payload = {
            "data": data
        }
        
        if cas is not None:
            payload["options"] = {"cas": cas}
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status in [200, 204]:
                logger.info(f"Secret written successfully: {path}")
                return True
            else:
                logger.error(f"Failed to write secret: {response.status}")
                return False
    
    async def delete_secret(self, path: str, versions: Optional[List[int]] = None) -> bool:
        """Delete secret from Vault"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        if versions:
            # Delete specific versions
            url = urljoin(self.config.url, f"v1/{self.config.mount_point}/delete/{path}")
            payload = {"versions": versions}
        else:
            # Delete latest version
            url = urljoin(self.config.url, f"v1/{self.config.mount_point}/data/{path}")
            payload = {}
        
        headers = {"X-Vault-Token": self.token.token}
        
        async with self.session.delete(url, headers=headers, json=payload) as response:
            if response.status in [200, 204]:
                logger.info(f"Secret deleted successfully: {path}")
                return True
            else:
                logger.error(f"Failed to delete secret: {response.status}")
                return False
    
    async def list_secrets(self, path: str = "") -> List[str]:
        """List secrets in Vault"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        url = urljoin(self.config.url, f"v1/{self.config.mount_point}/metadata/{path}")
        headers = {"X-Vault-Token": self.token.token}
        params = {"list": "true"}
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data["data"]["keys"]
            elif response.status == 404:
                return []
            else:
                raise ValueError(f"Failed to list secrets: {response.status}")
    
    async def create_database_credentials(self, role: str) -> Dict[str, Any]:
        """Create dynamic database credentials"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        url = urljoin(self.config.url, f"v1/database/creds/{role}")
        headers = {"X-Vault-Token": self.token.token}
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "username": data["data"]["username"],
                    "password": data["data"]["password"],
                    "lease_id": data["lease_id"],
                    "lease_duration": data["lease_duration"],
                    "renewable": data["renewable"]
                }
            else:
                raise ValueError(f"Failed to create database credentials: {response.status}")
    
    async def encrypt_data(self, key_name: str, plaintext: str) -> str:
        """Encrypt data using Transit engine"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        url = urljoin(self.config.url, f"v1/transit/encrypt/{key_name}")
        headers = {"X-Vault-Token": self.token.token}
        payload = {
            "plaintext": base64.b64encode(plaintext.encode()).decode()
        }
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["data"]["ciphertext"]
            else:
                raise ValueError(f"Failed to encrypt data: {response.status}")
    
    async def decrypt_data(self, key_name: str, ciphertext: str) -> str:
        """Decrypt data using Transit engine"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        url = urljoin(self.config.url, f"v1/transit/decrypt/{key_name}")
        headers = {"X-Vault-Token": self.token.token}
        payload = {
            "ciphertext": ciphertext
        }
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return base64.b64decode(data["data"]["plaintext"]).decode()
            else:
                raise ValueError(f"Failed to decrypt data: {response.status}")
    
    async def generate_certificate(self, role: str, common_name: str, 
                                 alt_names: Optional[List[str]] = None,
                                 ttl: str = "24h") -> Dict[str, Any]:
        """Generate certificate using PKI engine"""
        if not self.token:
            raise ValueError("Not authenticated")
        
        url = urljoin(self.config.url, f"v1/pki/issue/{role}")
        headers = {"X-Vault-Token": self.token.token}
        payload = {
            "common_name": common_name,
            "ttl": ttl
        }
        
        if alt_names:
            payload["alt_names"] = ",".join(alt_names)
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "certificate": data["data"]["certificate"],
                    "private_key": data["data"]["private_key"],
                    "ca_chain": data["data"]["ca_chain"],
                    "issuing_ca": data["data"]["issuing_ca"],
                    "serial_number": data["data"]["serial_number"],
                    "expiration": data["data"]["expiration"]
                }
            else:
                raise ValueError(f"Failed to generate certificate: {response.status}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Vault health status"""
        url = urljoin(self.config.url, "v1/sys/health")
        
        async with self.session.get(url) as response:
            data = await response.json()
            return {
                "initialized": data.get("initialized", False),
                "sealed": data.get("sealed", True),
                "standby": data.get("standby", False),
                "performance_standby": data.get("performance_standby", False),
                "replication_performance_mode": data.get("replication_performance_mode"),
                "replication_dr_mode": data.get("replication_dr_mode"),
                "server_time_utc": data.get("server_time_utc"),
                "version": data.get("version"),
                "cluster_name": data.get("cluster_name"),
                "cluster_id": data.get("cluster_id")
            }

class VaultSecretsManager:
    """High-level secrets manager using Vault"""
    
    def __init__(self, vault_config: VaultConfig):
        self.vault_config = vault_config
        self.vault_client: Optional[VaultClient] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize Vault secrets manager"""
        if self._initialized:
            return
        
        self.vault_client = VaultClient(self.vault_config)
        await self.vault_client.initialize()
        self._initialized = True
        
        logger.info("VaultSecretsManager initialized")
    
    async def close(self):
        """Close Vault secrets manager"""
        if self.vault_client:
            await self.vault_client.close()
        self._initialized = False
    
    async def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret value"""
        if not self._initialized:
            await self.initialize()
        
        try:
            secret = await self.vault_client.read_secret(key)
            if secret and secret.data:
                # If secret has single value, return it directly
                if len(secret.data) == 1:
                    return list(secret.data.values())[0]
                else:
                    return secret.data
            return default
        except Exception as e:
            logger.error(f"Failed to get secret {key}: {e}")
            return default
    
    async def set_secret(self, key: str, value: Union[str, Dict[str, Any]]) -> bool:
        """Set secret value"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if isinstance(value, str):
                data = {"value": value}
            else:
                data = value
            
            return await self.vault_client.write_secret(key, data)
        except Exception as e:
            logger.error(f"Failed to set secret {key}: {e}")
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete secret"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.delete_secret(key)
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False
    
    async def list_secrets(self, path: str = "") -> List[str]:
        """List secrets"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.list_secrets(path)
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def rotate_secret(self, key: str, generator_func: callable) -> bool:
        """Rotate secret using generator function"""
        if not self._initialized:
            await self.initialize()
        
        try:
            new_value = generator_func()
            return await self.set_secret(key, new_value)
        except Exception as e:
            logger.error(f"Failed to rotate secret {key}: {e}")
            return False
    
    async def get_database_credentials(self, role: str) -> Optional[Dict[str, Any]]:
        """Get dynamic database credentials"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.create_database_credentials(role)
        except Exception as e:
            logger.error(f"Failed to get database credentials: {e}")
            return None
    
    async def encrypt_sensitive_data(self, key_name: str, data: str) -> Optional[str]:
        """Encrypt sensitive data using Transit engine"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.encrypt_data(key_name, data)
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            return None
    
    async def decrypt_sensitive_data(self, key_name: str, ciphertext: str) -> Optional[str]:
        """Decrypt sensitive data using Transit engine"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.decrypt_data(key_name, ciphertext)
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return None
    
    async def generate_tls_certificate(self, role: str, common_name: str,
                                     alt_names: Optional[List[str]] = None,
                                     ttl: str = "24h") -> Optional[Dict[str, Any]]:
        """Generate TLS certificate using PKI engine"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.generate_certificate(role, common_name, alt_names, ttl)
        except Exception as e:
            logger.error(f"Failed to generate certificate: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Vault health"""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self.vault_client.get_health_status()
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"error": str(e)}

# Factory function
def create_vault_secrets_manager(vault_url: str = None, 
                               auth_method: VaultAuthMethod = VaultAuthMethod.TOKEN,
                               mount_point: str = "secret") -> VaultSecretsManager:
    """Create Vault secrets manager"""
    config = VaultConfig(
        url=vault_url or os.getenv("VAULT_ADDR", "https://vault.grandmodel.com"),
        auth_method=auth_method,
        mount_point=mount_point,
        namespace=os.getenv("VAULT_NAMESPACE"),
        verify_ssl=os.getenv("VAULT_SKIP_VERIFY", "false").lower() != "true"
    )
    
    return VaultSecretsManager(config)

# Global instance
vault_secrets_manager = create_vault_secrets_manager()

# Utility functions
async def get_vault_secret(key: str, default: Any = None) -> Any:
    """Get secret from Vault"""
    return await vault_secrets_manager.get_secret(key, default)

async def set_vault_secret(key: str, value: Union[str, Dict[str, Any]]) -> bool:
    """Set secret in Vault"""
    return await vault_secrets_manager.set_secret(key, value)

async def rotate_vault_secret(key: str, generator_func: callable) -> bool:
    """Rotate secret in Vault"""
    return await vault_secrets_manager.rotate_secret(key, generator_func)

async def get_database_credentials(role: str) -> Optional[Dict[str, Any]]:
    """Get dynamic database credentials"""
    return await vault_secrets_manager.get_database_credentials(role)

async def encrypt_with_vault(key_name: str, data: str) -> Optional[str]:
    """Encrypt data using Vault Transit engine"""
    return await vault_secrets_manager.encrypt_sensitive_data(key_name, data)

async def decrypt_with_vault(key_name: str, ciphertext: str) -> Optional[str]:
    """Decrypt data using Vault Transit engine"""
    return await vault_secrets_manager.decrypt_sensitive_data(key_name, ciphertext)
