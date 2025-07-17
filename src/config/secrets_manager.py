"""
Secrets Management System
Secure storage, rotation, and access control for sensitive configuration data.
"""

import os
import json
import base64
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import threading


class SecretType(Enum):
    """Types of secrets"""
    API_KEY = "api_key"
    PASSWORD = "password"
    DATABASE_URL = "database_url"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"


@dataclass
class SecretMetadata:
    """Metadata for a secret"""
    name: str
    secret_type: SecretType
    created_at: datetime
    expires_at: Optional[datetime]
    last_rotated: Optional[datetime]
    rotation_interval: Optional[timedelta]
    description: str
    tags: List[str]
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class SecretAccess:
    """Secret access record"""
    secret_name: str
    user: str
    timestamp: datetime
    access_type: str  # 'read', 'write', 'rotate'
    success: bool
    ip_address: Optional[str] = None


class SecretsManager:
    """
    Secure secrets management with:
    - Encrypted storage
    - Access control
    - Secret rotation
    - Audit logging
    - Compliance features
    """

    def __init__(self, vault_path: Optional[Path] = None,
                 master_key: Optional[str] = None):
        self.vault_path = vault_path or Path(__file__).parent.parent.parent / "vault"
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize encryption
        self._init_encryption(master_key)
        
        # Storage
        self.secrets_path = self.vault_path / "secrets"
        self.metadata_path = self.vault_path / "metadata"
        self.audit_path = self.vault_path / "audit"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load metadata
        self.metadata: Dict[str, SecretMetadata] = {}
        self.access_log: List[SecretAccess] = []
        
        self._load_metadata()
        self._load_audit_log()
        
        # Access control
        self.access_policies: Dict[str, List[str]] = {}
        self._load_access_policies()
        
        self.logger.info("SecretsManager initialized")

    def _init_encryption(self, master_key: Optional[str] = None):
        """Initialize encryption system"""
        if master_key:
            # Use provided master key
            key = master_key.encode()
        else:
            # Get from environment or generate
            key_env = os.getenv('VAULT_MASTER_KEY')
            if key_env:
                key = base64.b64decode(key_env)
            else:
                # Generate new key for development
                key = Fernet.generate_key()
                self.logger.warning("Generated new master key - store securely!")
                self.logger.warning(f"VAULT_MASTER_KEY={base64.b64encode(key).decode()}")
        
        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'grandmodel_vault_salt',
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key))
        self.cipher = Fernet(derived_key)

    def _ensure_directories(self):
        """Ensure all vault directories exist"""
        for path in [self.vault_path, self.secrets_path, self.metadata_path, self.audit_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """Load secrets metadata"""
        metadata_file = self.metadata_path / "secrets.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                for name, meta_dict in data.items():
                    # Convert datetime strings back to datetime objects
                    if meta_dict.get('created_at'):
                        meta_dict['created_at'] = datetime.fromisoformat(meta_dict['created_at'])
                    if meta_dict.get('expires_at'):
                        meta_dict['expires_at'] = datetime.fromisoformat(meta_dict['expires_at'])
                    if meta_dict.get('last_rotated'):
                        meta_dict['last_rotated'] = datetime.fromisoformat(meta_dict['last_rotated'])
                    if meta_dict.get('last_accessed'):
                        meta_dict['last_accessed'] = datetime.fromisoformat(meta_dict['last_accessed'])
                    if meta_dict.get('rotation_interval'):
                        meta_dict['rotation_interval'] = timedelta(seconds=meta_dict['rotation_interval'])
                    
                    # Convert string enum back to enum
                    meta_dict['secret_type'] = SecretType(meta_dict['secret_type'])
                    
                    self.metadata[name] = SecretMetadata(**meta_dict)
            
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")

    def _save_metadata(self):
        """Save secrets metadata"""
        metadata_file = self.metadata_path / "secrets.json"
        
        # Convert to serializable format
        data = {}
        for name, metadata in self.metadata.items():
            meta_dict = asdict(metadata)
            
            # Convert datetime objects to ISO strings
            if meta_dict.get('created_at'):
                meta_dict['created_at'] = meta_dict['created_at'].isoformat()
            if meta_dict.get('expires_at'):
                meta_dict['expires_at'] = meta_dict['expires_at'].isoformat()
            if meta_dict.get('last_rotated'):
                meta_dict['last_rotated'] = meta_dict['last_rotated'].isoformat()
            if meta_dict.get('last_accessed'):
                meta_dict['last_accessed'] = meta_dict['last_accessed'].isoformat()
            if meta_dict.get('rotation_interval'):
                meta_dict['rotation_interval'] = meta_dict['rotation_interval'].total_seconds()
            
            # Convert enum to string
            meta_dict['secret_type'] = meta_dict['secret_type'].value
            
            data[name] = meta_dict
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_audit_log(self):
        """Load audit log"""
        audit_file = self.audit_path / "access.json"
        
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    data = json.load(f)
                
                for record in data:
                    # Convert timestamp string back to datetime
                    record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    self.access_log.append(SecretAccess(**record))
                    
            except Exception as e:
                self.logger.error(f"Failed to load audit log: {e}")

    def _save_audit_log(self):
        """Save audit log"""
        audit_file = self.audit_path / "access.json"
        
        # Convert to serializable format
        data = []
        for record in self.access_log[-1000:]:  # Keep last 1000 records
            record_dict = asdict(record)
            record_dict['timestamp'] = record_dict['timestamp'].isoformat()
            data.append(record_dict)
        
        with open(audit_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_access_policies(self):
        """Load access control policies"""
        policies_file = self.vault_path / "access_policies.json"
        
        if policies_file.exists():
            try:
                with open(policies_file, 'r') as f:
                    self.access_policies = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load access policies: {e}")

    def _save_access_policies(self):
        """Save access control policies"""
        policies_file = self.vault_path / "access_policies.json"
        
        with open(policies_file, 'w') as f:
            json.dump(self.access_policies, f, indent=2)

    def _log_access(self, secret_name: str, user: str, access_type: str, 
                   success: bool, ip_address: Optional[str] = None):
        """Log secret access"""
        record = SecretAccess(
            secret_name=secret_name,
            user=user,
            timestamp=datetime.now(),
            access_type=access_type,
            success=success,
            ip_address=ip_address
        )
        
        self.access_log.append(record)
        
        # Save periodically
        if len(self.access_log) % 100 == 0:
            self._save_audit_log()

    def _check_access(self, secret_name: str, user: str, access_type: str) -> bool:
        """Check if user has access to secret"""
        # System user has full access
        if user == "system":
            return True
        
        # Check if user has specific access to this secret
        if secret_name in self.access_policies:
            allowed_users = self.access_policies[secret_name]
            if user not in allowed_users:
                return False
        
        # Check if user has wildcard access
        if "*" in self.access_policies:
            allowed_users = self.access_policies["*"]
            if user not in allowed_users:
                return False
        
        return True

    def create_secret(self, name: str, value: str, secret_type: SecretType,
                     description: str = "", tags: List[str] = None,
                     expires_at: Optional[datetime] = None,
                     rotation_interval: Optional[timedelta] = None,
                     user: str = "system") -> bool:
        """
        Create a new secret
        
        Args:
            name: Secret name
            value: Secret value
            secret_type: Type of secret
            description: Description
            tags: Tags for categorization
            expires_at: Expiration date
            rotation_interval: Automatic rotation interval
            user: User creating the secret
            
        Returns:
            True if successful
        """
        with self._lock:
            if not self._check_access(name, user, "write"):
                self._log_access(name, user, "write", False)
                raise PermissionError(f"User '{user}' does not have write access to secret '{name}'")
            
            if name in self.metadata:
                self._log_access(name, user, "write", False)
                raise ValueError(f"Secret '{name}' already exists")
            
            try:
                # Encrypt the secret value
                encrypted_value = self.cipher.encrypt(value.encode())
                
                # Save encrypted secret
                secret_file = self.secrets_path / f"{name}.enc"
                with open(secret_file, 'wb') as f:
                    f.write(encrypted_value)
                
                # Create metadata
                metadata = SecretMetadata(
                    name=name,
                    secret_type=secret_type,
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    last_rotated=None,
                    rotation_interval=rotation_interval,
                    description=description,
                    tags=tags or []
                )
                
                self.metadata[name] = metadata
                self._save_metadata()
                
                self._log_access(name, user, "write", True)
                self.logger.info(f"Secret '{name}' created by {user}")
                
                return True
                
            except Exception as e:
                self._log_access(name, user, "write", False)
                self.logger.error(f"Failed to create secret '{name}': {e}")
                return False

    def get_secret(self, name: str, user: str = "system") -> Optional[str]:
        """
        Get secret value
        
        Args:
            name: Secret name
            user: User requesting the secret
            
        Returns:
            Secret value if found and accessible
        """
        with self._lock:
            if not self._check_access(name, user, "read"):
                self._log_access(name, user, "read", False)
                raise PermissionError(f"User '{user}' does not have read access to secret '{name}'")
            
            if name not in self.metadata:
                self._log_access(name, user, "read", False)
                return None
            
            try:
                # Check if secret is expired
                metadata = self.metadata[name]
                if metadata.expires_at and datetime.now() > metadata.expires_at:
                    self._log_access(name, user, "read", False)
                    self.logger.warning(f"Secret '{name}' has expired")
                    return None
                
                # Read encrypted secret
                secret_file = self.secrets_path / f"{name}.enc"
                if not secret_file.exists():
                    self._log_access(name, user, "read", False)
                    return None
                
                with open(secret_file, 'rb') as f:
                    encrypted_value = f.read()
                
                # Decrypt secret
                decrypted_value = self.cipher.decrypt(encrypted_value).decode()
                
                # Update access metadata
                metadata.access_count += 1
                metadata.last_accessed = datetime.now()
                self._save_metadata()
                
                self._log_access(name, user, "read", True)
                
                return decrypted_value
                
            except Exception as e:
                self._log_access(name, user, "read", False)
                self.logger.error(f"Failed to get secret '{name}': {e}")
                return None

    def update_secret(self, name: str, value: str, user: str = "system") -> bool:
        """
        Update secret value
        
        Args:
            name: Secret name
            value: New secret value
            user: User updating the secret
            
        Returns:
            True if successful
        """
        with self._lock:
            if not self._check_access(name, user, "write"):
                self._log_access(name, user, "write", False)
                raise PermissionError(f"User '{user}' does not have write access to secret '{name}'")
            
            if name not in self.metadata:
                self._log_access(name, user, "write", False)
                return False
            
            try:
                # Encrypt the new value
                encrypted_value = self.cipher.encrypt(value.encode())
                
                # Save encrypted secret
                secret_file = self.secrets_path / f"{name}.enc"
                with open(secret_file, 'wb') as f:
                    f.write(encrypted_value)
                
                # Update metadata
                metadata = self.metadata[name]
                metadata.last_rotated = datetime.now()
                self._save_metadata()
                
                self._log_access(name, user, "write", True)
                self.logger.info(f"Secret '{name}' updated by {user}")
                
                return True
                
            except Exception as e:
                self._log_access(name, user, "write", False)
                self.logger.error(f"Failed to update secret '{name}': {e}")
                return False

    def delete_secret(self, name: str, user: str = "system") -> bool:
        """
        Delete a secret
        
        Args:
            name: Secret name
            user: User deleting the secret
            
        Returns:
            True if successful
        """
        with self._lock:
            if not self._check_access(name, user, "write"):
                self._log_access(name, user, "write", False)
                raise PermissionError(f"User '{user}' does not have write access to secret '{name}'")
            
            if name not in self.metadata:
                self._log_access(name, user, "write", False)
                return False
            
            try:
                # Delete encrypted file
                secret_file = self.secrets_path / f"{name}.enc"
                if secret_file.exists():
                    secret_file.unlink()
                
                # Remove metadata
                del self.metadata[name]
                self._save_metadata()
                
                self._log_access(name, user, "write", True)
                self.logger.info(f"Secret '{name}' deleted by {user}")
                
                return True
                
            except Exception as e:
                self._log_access(name, user, "write", False)
                self.logger.error(f"Failed to delete secret '{name}': {e}")
                return False

    def rotate_secret(self, name: str, new_value: Optional[str] = None, 
                     user: str = "system") -> bool:
        """
        Rotate a secret (update with new value)
        
        Args:
            name: Secret name
            new_value: New secret value (auto-generated if None)
            user: User rotating the secret
            
        Returns:
            True if successful
        """
        with self._lock:
            if not self._check_access(name, user, "rotate"):
                self._log_access(name, user, "rotate", False)
                raise PermissionError(f"User '{user}' does not have rotate access to secret '{name}'")
            
            if name not in self.metadata:
                self._log_access(name, user, "rotate", False)
                return False
            
            try:
                # Generate new value if not provided
                if new_value is None:
                    metadata = self.metadata[name]
                    new_value = self._generate_secret_value(metadata.secret_type)
                
                # Update the secret
                success = self.update_secret(name, new_value, user)
                
                if success:
                    self._log_access(name, user, "rotate", True)
                    self.logger.info(f"Secret '{name}' rotated by {user}")
                else:
                    self._log_access(name, user, "rotate", False)
                
                return success
                
            except Exception as e:
                self._log_access(name, user, "rotate", False)
                self.logger.error(f"Failed to rotate secret '{name}': {e}")
                return False

    def _generate_secret_value(self, secret_type: SecretType) -> str:
        """Generate a new secret value based on type"""
        if secret_type == SecretType.PASSWORD:
            # Generate secure password
            return secrets.token_urlsafe(32)
        elif secret_type == SecretType.API_KEY:
            # Generate API key
            return secrets.token_hex(32)
        elif secret_type == SecretType.TOKEN:
            # Generate JWT-like token
            return secrets.token_urlsafe(64)
        else:
            # Default to secure random string
            return secrets.token_urlsafe(32)

    def list_secrets(self, user: str = "system") -> List[str]:
        """
        List all secret names the user has access to
        
        Args:
            user: User requesting the list
            
        Returns:
            List of secret names
        """
        accessible_secrets = []
        
        for name in self.metadata.keys():
            if self._check_access(name, user, "read"):
                accessible_secrets.append(name)
        
        return accessible_secrets

    def get_secret_metadata(self, name: str, user: str = "system") -> Optional[SecretMetadata]:
        """
        Get secret metadata
        
        Args:
            name: Secret name
            user: User requesting metadata
            
        Returns:
            Secret metadata if accessible
        """
        if not self._check_access(name, user, "read"):
            return None
        
        return self.metadata.get(name)

    def check_expiring_secrets(self, days_ahead: int = 30) -> List[str]:
        """
        Check for secrets expiring within specified days
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of secret names expiring soon
        """
        expiring_secrets = []
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        for name, metadata in self.metadata.items():
            if metadata.expires_at and metadata.expires_at <= cutoff_date:
                expiring_secrets.append(name)
        
        return expiring_secrets

    def check_rotation_due(self) -> List[str]:
        """
        Check for secrets due for rotation
        
        Returns:
            List of secret names due for rotation
        """
        rotation_due = []
        now = datetime.now()
        
        for name, metadata in self.metadata.items():
            if (metadata.rotation_interval and 
                metadata.last_rotated and
                (now - metadata.last_rotated) >= metadata.rotation_interval):
                rotation_due.append(name)
        
        return rotation_due

    def apply_secrets(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply secrets to configuration data
        
        Args:
            config_data: Configuration data with secret placeholders
            
        Returns:
            Configuration data with secrets applied
        """
        def replace_secrets(obj):
            if isinstance(obj, dict):
                return {k: replace_secrets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_secrets(item) for item in obj]
            elif isinstance(obj, str):
                # Replace ${SECRET:name} patterns
                if obj.startswith("${SECRET:") and obj.endswith("}"):
                    secret_name = obj[9:-1]  # Remove ${SECRET: and }
                    secret_value = self.get_secret(secret_name)
                    return secret_value if secret_value else obj
            return obj
        
        return replace_secrets(config_data)

    def set_access_policy(self, secret_name: str, allowed_users: List[str]):
        """
        Set access policy for a secret
        
        Args:
            secret_name: Secret name (or "*" for all secrets)
            allowed_users: List of users allowed to access
        """
        self.access_policies[secret_name] = allowed_users
        self._save_access_policies()

    def get_access_policy(self, secret_name: str) -> List[str]:
        """
        Get access policy for a secret
        
        Args:
            secret_name: Secret name
            
        Returns:
            List of users allowed to access
        """
        return self.access_policies.get(secret_name, [])

    def get_audit_log(self, secret_name: Optional[str] = None,
                     user: Optional[str] = None,
                     hours_back: int = 24) -> List[SecretAccess]:
        """
        Get audit log entries
        
        Args:
            secret_name: Filter by secret name
            user: Filter by user
            hours_back: Number of hours to look back
            
        Returns:
            List of audit log entries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_log = []
        for record in self.access_log:
            if record.timestamp < cutoff_time:
                continue
            
            if secret_name and record.secret_name != secret_name:
                continue
                
            if user and record.user != user:
                continue
            
            filtered_log.append(record)
        
        return sorted(filtered_log, key=lambda x: x.timestamp, reverse=True)

    def get_secrets_status(self) -> Dict[str, Any]:
        """Get overall secrets status"""
        now = datetime.now()
        
        expired_count = 0
        expiring_soon_count = 0
        rotation_due_count = 0
        
        for metadata in self.metadata.values():
            if metadata.expires_at and metadata.expires_at <= now:
                expired_count += 1
            elif metadata.expires_at and metadata.expires_at <= now + timedelta(days=30):
                expiring_soon_count += 1
            
            if (metadata.rotation_interval and 
                metadata.last_rotated and
                (now - metadata.last_rotated) >= metadata.rotation_interval):
                rotation_due_count += 1
        
        return {
            'total_secrets': len(self.metadata),
            'expired_secrets': expired_count,
            'expiring_soon': expiring_soon_count,
            'rotation_due': rotation_due_count,
            'access_policies': len(self.access_policies),
            'audit_records': len(self.access_log),
            'last_access': max([r.timestamp for r in self.access_log], default=None)
        }