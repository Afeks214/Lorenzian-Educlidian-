"""
Vault Dynamic Secrets Management
Provides dynamic secret generation for databases, APIs, and other services.
"""

import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from src.monitoring.logger_config import get_logger
from src.security.vault_client import VaultClient, VaultSecret, VaultException, get_vault_client

logger = get_logger(__name__)

class DatabaseEngine(str, Enum):
    """Supported database engines"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    CASSANDRA = "cassandra"
    ELASTICSEARCH = "elasticsearch"
    MSSQL = "mssql"
    ORACLE = "oracle"
    INFLUXDB = "influxdb"

class CloudProvider(str, Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"

class SecretType(str, Enum):
    """Types of dynamic secrets"""
    DATABASE = "database"
    AWS_IAM = "aws_iam"
    AZURE_AD = "azure_ad"
    GCP_IAM = "gcp_iam"
    SSH = "ssh"
    CONSUL = "consul"
    NOMAD = "nomad"
    RABBITMQ = "rabbitmq"
    TOTP = "totp"
    PKI = "pki"

@dataclass
class DatabaseConnection:
    """Database connection configuration"""
    engine: DatabaseEngine
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"
    connection_url: Optional[str] = None
    max_connections: int = 100
    max_idle_connections: int = 10
    max_connection_lifetime: str = "1h"
    
    def __post_init__(self):
        if not self.connection_url:
            if self.engine == DatabaseEngine.POSTGRESQL:
                self.connection_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"
            elif self.engine == DatabaseEngine.MYSQL:
                self.connection_url = f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            elif self.engine == DatabaseEngine.MONGODB:
                self.connection_url = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            elif self.engine == DatabaseEngine.REDIS:
                self.connection_url = f"redis://{self.username}:{self.password}@{self.host}:{self.port}"

@dataclass
class DynamicCredentials:
    """Dynamic credentials with lease information"""
    username: str
    password: str
    lease_id: str
    lease_duration: int
    renewable: bool
    created_at: datetime
    expires_at: datetime
    connection_info: Optional[Dict[str, Any]] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        return datetime.utcnow() > self.expires_at
    
    def expires_in(self) -> int:
        """Get seconds until expiration"""
        return max(0, int((self.expires_at - datetime.utcnow()).total_seconds()))
    
    def should_renew(self, buffer_seconds: int = 300) -> bool:
        """Check if credentials should be renewed"""
        return self.expires_in() <= buffer_seconds

@dataclass
class RoleConfiguration:
    """Role configuration for dynamic secrets"""
    name: str
    db_name: str
    creation_statements: List[str]
    default_ttl: str = "1h"
    max_ttl: str = "24h"
    revocation_statements: Optional[List[str]] = None
    rollback_statements: Optional[List[str]] = None
    renew_statements: Optional[List[str]] = None

class VaultDynamicSecrets:
    """
    Vault Dynamic Secrets Management
    
    Provides dynamic secret generation for:
    - Database credentials
    - Cloud provider credentials
    - SSH keys
    - API keys
    - Certificates
    """
    
    def __init__(self, 
                 vault_client: Optional[VaultClient] = None,
                 renewal_buffer: int = 300,
                 auto_renewal: bool = True):
        """
        Initialize dynamic secrets manager
        
        Args:
            vault_client: Vault client instance
            renewal_buffer: Buffer time in seconds before renewal
            auto_renewal: Whether to automatically renew credentials
        """
        self.vault_client = vault_client
        self.renewal_buffer = renewal_buffer
        self.auto_renewal = auto_renewal
        
        # Credential cache
        self.credentials_cache: Dict[str, DynamicCredentials] = {}
        self.cache_lock = threading.Lock()
        
        # Background renewal
        self.renewal_executor = ThreadPoolExecutor(max_workers=5)
        self.renewal_tasks: Dict[str, asyncio.Task] = {}
        
        # Configuration cache
        self.database_configs: Dict[str, DatabaseConnection] = {}
        self.role_configs: Dict[str, RoleConfiguration] = {}
        
        # Callbacks
        self.on_credentials_generated: Optional[Callable] = None
        self.on_credentials_renewed: Optional[Callable] = None
        self.on_credentials_revoked: Optional[Callable] = None
        
        logger.info("Dynamic secrets manager initialized", auto_renewal=auto_renewal)
    
    async def _get_vault_client(self) -> VaultClient:
        """Get Vault client instance"""
        if self.vault_client:
            return self.vault_client
        return await get_vault_client()
    
    async def configure_database(self, 
                               config_name: str,
                               connection: DatabaseConnection,
                               mount_point: str = "database") -> bool:
        """
        Configure database connection for dynamic secrets
        
        Args:
            config_name: Configuration name
            connection: Database connection details
            mount_point: Database engine mount point
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Configure database connection
            client.client.secrets.database.configure(
                name=config_name,
                plugin_name=f"{connection.engine.value}-database-plugin",
                connection_url=connection.connection_url,
                allowed_roles=["*"],
                username=connection.username,
                password=connection.password,
                max_open_connections=connection.max_connections,
                max_idle_connections=connection.max_idle_connections,
                max_connection_lifetime=connection.max_connection_lifetime,
                mount_point=mount_point
            )
            
            # Cache configuration
            self.database_configs[config_name] = connection
            
            logger.info(f"Database configured: {config_name}", engine=connection.engine.value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure database: {e}", config_name=config_name)
            raise VaultException(f"Failed to configure database {config_name}: {e}")
    
    async def configure_role(self, 
                           config_name: str,
                           role_config: RoleConfiguration,
                           mount_point: str = "database") -> bool:
        """
        Configure database role for dynamic secrets
        
        Args:
            config_name: Database configuration name
            role_config: Role configuration
            mount_point: Database engine mount point
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Configure role
            client.client.secrets.database.create_role(
                name=role_config.name,
                db_name=role_config.db_name,
                creation_statements=role_config.creation_statements,
                default_ttl=role_config.default_ttl,
                max_ttl=role_config.max_ttl,
                revocation_statements=role_config.revocation_statements,
                rollback_statements=role_config.rollback_statements,
                renew_statements=role_config.renew_statements,
                mount_point=mount_point
            )
            
            # Cache role configuration
            self.role_configs[role_config.name] = role_config
            
            logger.info(f"Database role configured: {role_config.name}", db_name=role_config.db_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure role: {e}", role_name=role_config.name)
            raise VaultException(f"Failed to configure role {role_config.name}: {e}")
    
    async def generate_database_credentials(self, 
                                          role_name: str,
                                          mount_point: str = "database") -> DynamicCredentials:
        """
        Generate dynamic database credentials
        
        Args:
            role_name: Database role name
            mount_point: Database engine mount point
            
        Returns:
            DynamicCredentials object
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Generate credentials
            response = client.client.secrets.database.generate_credentials(
                name=role_name,
                mount_point=mount_point
            )
            
            # Create credentials object
            created_at = datetime.utcnow()
            expires_at = created_at + timedelta(seconds=response['lease_duration'])
            
            credentials = DynamicCredentials(
                username=response['data']['username'],
                password=response['data']['password'],
                lease_id=response['lease_id'],
                lease_duration=response['lease_duration'],
                renewable=response['renewable'],
                created_at=created_at,
                expires_at=expires_at,
                connection_info={'role': role_name, 'mount_point': mount_point}
            )
            
            # Cache credentials
            with self.cache_lock:
                self.credentials_cache[role_name] = credentials
            
            # Start auto-renewal if enabled
            if self.auto_renewal and credentials.renewable:
                await self._start_auto_renewal(role_name, credentials)
            
            logger.info(f"Database credentials generated: {role_name}", 
                       username=credentials.username, 
                       lease_duration=credentials.lease_duration)
            
            if self.on_credentials_generated:
                await self.on_credentials_generated(role_name, credentials)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to generate database credentials: {e}", role_name=role_name)
            raise VaultException(f"Failed to generate database credentials for {role_name}: {e}")
    
    async def renew_credentials(self, 
                              role_name: str,
                              increment: Optional[int] = None) -> DynamicCredentials:
        """
        Renew dynamic credentials
        
        Args:
            role_name: Role name
            increment: Lease increment in seconds
            
        Returns:
            Updated DynamicCredentials object
        """
        try:
            with self.cache_lock:
                credentials = self.credentials_cache.get(role_name)
            
            if not credentials:
                raise VaultException(f"No credentials found for role {role_name}")
            
            if not credentials.renewable:
                raise VaultException(f"Credentials for role {role_name} are not renewable")
            
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Renew lease
            response = client.client.sys.renew_lease(
                lease_id=credentials.lease_id,
                increment=increment
            )
            
            # Update credentials
            credentials.lease_duration = response['lease_duration']
            credentials.expires_at = datetime.utcnow() + timedelta(seconds=response['lease_duration'])
            
            # Update cache
            with self.cache_lock:
                self.credentials_cache[role_name] = credentials
            
            logger.info(f"Credentials renewed: {role_name}", 
                       lease_duration=credentials.lease_duration)
            
            if self.on_credentials_renewed:
                await self.on_credentials_renewed(role_name, credentials)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to renew credentials: {e}", role_name=role_name)
            raise VaultException(f"Failed to renew credentials for {role_name}: {e}")
    
    async def revoke_credentials(self, role_name: str) -> bool:
        """
        Revoke dynamic credentials
        
        Args:
            role_name: Role name
            
        Returns:
            True if successful
        """
        try:
            with self.cache_lock:
                credentials = self.credentials_cache.get(role_name)
            
            if not credentials:
                logger.warning(f"No credentials found for role {role_name}")
                return True
            
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Revoke lease
            client.client.sys.revoke_lease(lease_id=credentials.lease_id)
            
            # Remove from cache
            with self.cache_lock:
                if role_name in self.credentials_cache:
                    del self.credentials_cache[role_name]
            
            # Stop auto-renewal
            if role_name in self.renewal_tasks:
                self.renewal_tasks[role_name].cancel()
                del self.renewal_tasks[role_name]
            
            logger.info(f"Credentials revoked: {role_name}")
            
            if self.on_credentials_revoked:
                await self.on_credentials_revoked(role_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke credentials: {e}", role_name=role_name)
            raise VaultException(f"Failed to revoke credentials for {role_name}: {e}")
    
    async def get_credentials(self, role_name: str) -> Optional[DynamicCredentials]:
        """
        Get cached credentials
        
        Args:
            role_name: Role name
            
        Returns:
            DynamicCredentials object or None
        """
        with self.cache_lock:
            credentials = self.credentials_cache.get(role_name)
        
        if credentials and credentials.is_expired():
            logger.warning(f"Credentials expired for role {role_name}")
            # Remove expired credentials
            with self.cache_lock:
                if role_name in self.credentials_cache:
                    del self.credentials_cache[role_name]
            return None
        
        return credentials
    
    async def _start_auto_renewal(self, role_name: str, credentials: DynamicCredentials):
        """Start auto-renewal task for credentials"""
        if role_name in self.renewal_tasks:
            self.renewal_tasks[role_name].cancel()
        
        self.renewal_tasks[role_name] = asyncio.create_task(
            self._auto_renewal_loop(role_name, credentials)
        )
    
    async def _auto_renewal_loop(self, role_name: str, credentials: DynamicCredentials):
        """Auto-renewal loop for credentials"""
        try:
            while True:
                # Wait until renewal is needed
                sleep_time = credentials.expires_in() - self.renewal_buffer
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                # Check if credentials still exist
                with self.cache_lock:
                    current_credentials = self.credentials_cache.get(role_name)
                
                if not current_credentials or current_credentials.lease_id != credentials.lease_id:
                    # Credentials have been replaced or removed
                    break
                
                # Renew credentials
                try:
                    credentials = await self.renew_credentials(role_name)
                    logger.info(f"Auto-renewed credentials: {role_name}")
                except Exception as e:
                    logger.error(f"Auto-renewal failed: {e}", role_name=role_name)
                    break
                
        except asyncio.CancelledError:
            logger.info(f"Auto-renewal cancelled: {role_name}")
        except Exception as e:
            logger.error(f"Auto-renewal error: {e}", role_name=role_name)
    
    async def generate_aws_credentials(self, 
                                     role_name: str,
                                     mount_point: str = "aws") -> Dict[str, Any]:
        """
        Generate AWS dynamic credentials
        
        Args:
            role_name: AWS role name
            mount_point: AWS engine mount point
            
        Returns:
            AWS credentials dictionary
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Generate AWS credentials
            response = client.client.secrets.aws.generate_credentials(
                name=role_name,
                mount_point=mount_point
            )
            
            logger.info(f"AWS credentials generated: {role_name}")
            
            return {
                'access_key': response['data']['access_key'],
                'secret_key': response['data']['secret_key'],
                'security_token': response['data'].get('security_token'),
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration'],
                'renewable': response['renewable']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate AWS credentials: {e}", role_name=role_name)
            raise VaultException(f"Failed to generate AWS credentials for {role_name}: {e}")
    
    async def generate_ssh_key(self, 
                              role_name: str,
                              ip: str,
                              username: str,
                              mount_point: str = "ssh") -> Dict[str, Any]:
        """
        Generate SSH key pair
        
        Args:
            role_name: SSH role name
            ip: Target IP address
            username: SSH username
            mount_point: SSH engine mount point
            
        Returns:
            SSH key information
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Generate SSH key
            response = client.client.secrets.ssh.generate_credentials(
                name=role_name,
                ip=ip,
                username=username,
                mount_point=mount_point
            )
            
            logger.info(f"SSH key generated: {role_name}", ip=ip, username=username)
            
            return {
                'key': response['data']['key'],
                'key_type': response['data']['key_type'],
                'ip': ip,
                'username': username,
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate SSH key: {e}", role_name=role_name)
            raise VaultException(f"Failed to generate SSH key for {role_name}: {e}")
    
    async def generate_certificate(self, 
                                 role_name: str,
                                 common_name: str,
                                 alt_names: Optional[List[str]] = None,
                                 ip_sans: Optional[List[str]] = None,
                                 ttl: Optional[str] = None,
                                 mount_point: str = "pki") -> Dict[str, Any]:
        """
        Generate TLS certificate
        
        Args:
            role_name: PKI role name
            common_name: Certificate common name
            alt_names: Alternative names
            ip_sans: IP SANs
            ttl: Certificate TTL
            mount_point: PKI engine mount point
            
        Returns:
            Certificate information
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Generate certificate
            response = client.client.secrets.pki.generate_certificate(
                name=role_name,
                common_name=common_name,
                alt_names=alt_names,
                ip_sans=ip_sans,
                ttl=ttl,
                mount_point=mount_point
            )
            
            logger.info(f"Certificate generated: {role_name}", common_name=common_name)
            
            return {
                'certificate': response['data']['certificate'],
                'private_key': response['data']['private_key'],
                'ca_chain': response['data']['ca_chain'],
                'issuing_ca': response['data']['issuing_ca'],
                'serial_number': response['data']['serial_number'],
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate certificate: {e}", role_name=role_name)
            raise VaultException(f"Failed to generate certificate for {role_name}: {e}")
    
    async def cleanup_expired_credentials(self):
        """Clean up expired credentials from cache"""
        expired_roles = []
        
        with self.cache_lock:
            for role_name, credentials in self.credentials_cache.items():
                if credentials.is_expired():
                    expired_roles.append(role_name)
        
        for role_name in expired_roles:
            try:
                await self.revoke_credentials(role_name)
                logger.info(f"Cleaned up expired credentials: {role_name}")
            except Exception as e:
                logger.error(f"Failed to cleanup credentials: {e}", role_name=role_name)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get dynamic secrets status"""
        with self.cache_lock:
            credentials_count = len(self.credentials_cache)
            active_renewals = len(self.renewal_tasks)
            
            credentials_status = {}
            for role_name, credentials in self.credentials_cache.items():
                credentials_status[role_name] = {
                    'username': credentials.username,
                    'expires_in': credentials.expires_in(),
                    'renewable': credentials.renewable,
                    'should_renew': credentials.should_renew(self.renewal_buffer)
                }
        
        return {
            'auto_renewal_enabled': self.auto_renewal,
            'renewal_buffer_seconds': self.renewal_buffer,
            'total_credentials': credentials_count,
            'active_renewals': active_renewals,
            'database_configs': len(self.database_configs),
            'role_configs': len(self.role_configs),
            'credentials_status': credentials_status
        }
    
    async def close(self):
        """Close dynamic secrets manager"""
        # Cancel all renewal tasks
        for task in self.renewal_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.renewal_tasks:
            await asyncio.gather(*self.renewal_tasks.values(), return_exceptions=True)
        
        # Shutdown executor
        self.renewal_executor.shutdown(wait=True)
        
        logger.info("Dynamic secrets manager closed")

# Global dynamic secrets instance
dynamic_secrets: Optional[VaultDynamicSecrets] = None

async def get_dynamic_secrets() -> VaultDynamicSecrets:
    """Get global dynamic secrets instance"""
    global dynamic_secrets
    
    if not dynamic_secrets:
        dynamic_secrets = VaultDynamicSecrets()
    
    return dynamic_secrets

# Convenience functions
async def get_database_credentials(role_name: str) -> Optional[DynamicCredentials]:
    """Get database credentials"""
    manager = await get_dynamic_secrets()
    return await manager.get_credentials(role_name)

async def generate_db_credentials(role_name: str) -> DynamicCredentials:
    """Generate database credentials"""
    manager = await get_dynamic_secrets()
    return await manager.generate_database_credentials(role_name)

async def renew_db_credentials(role_name: str) -> DynamicCredentials:
    """Renew database credentials"""
    manager = await get_dynamic_secrets()
    return await manager.renew_credentials(role_name)

async def revoke_db_credentials(role_name: str) -> bool:
    """Revoke database credentials"""
    manager = await get_dynamic_secrets()
    return await manager.revoke_credentials(role_name)